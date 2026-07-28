"""Stage22 low-rank residual adapters and checkpoint merge utilities."""

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairedResidualLoRALinear(nn.Module):
    """Frozen linear layer plus a zero-start low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Stage22 LoRA base must be nn.Linear")
        if int(rank) <= 0 or not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError("Stage22 LoRA rank and alpha must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=base.weight.device,
                        dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=base.weight.device,
                        dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        residual = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return base_output + residual * self.scale

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(
            self.base.weight
            + self.scale * torch.matmul(self.lora_B, self.lora_A)
        )
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        merged.requires_grad_(False)
        return merged


def _stage22_regression_branches(decoder: nn.Module) -> Iterable[Tuple[int, nn.Sequential]]:
    layers = getattr(decoder, "layers", None)
    if layers is None or len(layers) != 2:
        raise RuntimeError("Stage22 requires exactly two diffusion decoder layers")
    for index, layer in enumerate(layers):
        branch = layer.task_decoder.plan_reg_branch
        if not isinstance(branch, nn.Sequential) or len(branch) == 0:
            raise RuntimeError("Stage22 regression branch is not a non-empty Sequential")
        yield index, branch


def attach_stage22_lora(
    decoder: nn.Module, rank: int = 8, alpha: float = 8.0
) -> Tuple[str, ...]:
    """Attach adapters to both final regression linears and return trainable names."""
    decoder.requires_grad_(False)
    for _, branch in _stage22_regression_branches(decoder):
        if isinstance(branch[-1], PairedResidualLoRALinear):
            raise RuntimeError("Stage22 LoRA adapters are already attached")
        if not isinstance(branch[-1], nn.Linear):
            raise RuntimeError("Stage22 final regression module must be nn.Linear")
        branch[-1] = PairedResidualLoRALinear(branch[-1], rank=rank, alpha=alpha)
    names = tuple(name for name, parameter in decoder.named_parameters()
                  if parameter.requires_grad)
    expected = tuple(
        f"layers.{layer}.task_decoder.plan_reg_branch.4.lora_{part}"
        for layer in range(2) for part in ("A", "B")
    )
    if names != expected:
        raise RuntimeError(
            f"Stage22 trainable adapter set drifted: {names!r} != {expected!r}"
        )
    return names


def merge_stage22_lora_inplace(decoder: nn.Module) -> None:
    """Replace Stage22 wrappers with ordinary merged nn.Linear modules."""
    for _, branch in _stage22_regression_branches(decoder):
        adapter = branch[-1]
        if not isinstance(adapter, PairedResidualLoRALinear):
            raise RuntimeError("Stage22 merge requires attached LoRA adapters")
        branch[-1] = adapter.merged_linear()


def merge_stage22_checkpoint_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Tuple[str, ...]]:
    """Return a plain-generator state dict with all Stage22 adapters merged."""
    merged = dict(state_dict)
    suffix = ".task_decoder.plan_reg_branch.4.lora_A"
    prefixes = sorted(key[:-len("lora_A")] for key in merged if key.endswith(suffix))
    if len(prefixes) != 2:
        raise RuntimeError(
            f"Stage22 checkpoint must contain two adapter pairs, found {len(prefixes)}"
        )
    for prefix in prefixes:
        required = (
            prefix + "base.weight",
            prefix + "lora_A",
            prefix + "lora_B",
        )
        missing = [key for key in required if key not in merged]
        if missing:
            raise RuntimeError(f"Stage22 checkpoint adapter is incomplete: {missing}")
        base_weight = merged[prefix + "base.weight"]
        lora_a = merged[prefix + "lora_A"]
        lora_b = merged[prefix + "lora_B"]
        if lora_a.shape[0] != 8 or lora_b.shape[1] != 8:
            raise RuntimeError("Stage22 checkpoint LoRA rank is not 8")
        merged[prefix + "weight"] = (
            base_weight + torch.matmul(lora_b, lora_a)
        )
        base_bias_key = prefix + "base.bias"
        if base_bias_key in merged:
            merged[prefix + "bias"] = merged[base_bias_key]
        for key in (
            prefix + "base.weight",
            base_bias_key,
            prefix + "lora_A",
            prefix + "lora_B",
        ):
            merged.pop(key, None)
    return merged, tuple(prefixes)
