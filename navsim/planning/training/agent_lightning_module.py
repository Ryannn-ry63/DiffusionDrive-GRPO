import pytorch_lightning as pl
import torch

from pathlib import Path
from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self._decoder_weight_snapshot = None
        self._latest_generation_kl = None
        self._adaptive_kl_stop_pending = False
        self._adaptive_kl_stop_checkpoint_saved = False

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if logging_prefix == "train":
            self._latest_generation_kl = loss_dict.get("generation_kl_loss")
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))
        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def on_fit_start(self) -> None:
        """Revalidate the frozen base after Lightning restores a checkpoint."""
        validator = getattr(
            self.agent, "validate_reference_policy_immutability", None
        )
        if validator is not None:
            validator()

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Sync PPO's old policy by Lightning's optimizer step, not forwards.

        This hook is reached for training batches only; validation forwards no
        longer advance the behavior-policy snapshot cadence.
        """
        model = getattr(self.agent, "_transfuser_model", None)
        head = getattr(model, "_trajectory_head", None)
        if head is not None and hasattr(head, "maybe_sync_old_policy"):
            head.maybe_sync_old_policy(self.global_step)

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the trainable planning decoder gradient norm for GRPO diagnostics."""
        model = getattr(self.agent, "_transfuser_model", None)
        trajectory_head = getattr(model, "_trajectory_head", None)
        decoder = getattr(trajectory_head, "diff_decoder", None)
        if decoder is None:
            return
        squared_norms = {
            "shared": torch.zeros((), device=self.device),
            "regression": torch.zeros((), device=self.device),
            "classification": torch.zeros((), device=self.device),
        }
        for name, parameter in decoder.named_parameters():
            if parameter.grad is None:
                continue
            if "plan_reg_branch" in name:
                group = "regression"
            elif "plan_cls_branch" in name:
                group = "classification"
            else:
                group = "shared"
            squared_norms[group] = (
                squared_norms[group] + parameter.grad.detach().float().square().sum()
            )

        # Per-refinement diagnostics make a missing first-layer gradient
        # immediately visible in smoke/audit runs.
        for layer_index, layer in enumerate(getattr(decoder, "layers", ())):
            layer_groups = {
                "shared_attention": torch.zeros((), device=self.device),
                "ffn": torch.zeros((), device=self.device),
                "time_modulation": torch.zeros((), device=self.device),
                "regression": torch.zeros((), device=self.device),
                "classification": torch.zeros((), device=self.device),
            }
            for name, parameter in layer.named_parameters():
                if parameter.grad is None:
                    continue
                if "plan_reg_branch" in name:
                    group = "regression"
                elif "plan_cls_branch" in name:
                    group = "classification"
                elif "ffn" in name:
                    group = "ffn"
                elif "time_modulation" in name:
                    group = "time_modulation"
                else:
                    group = "shared_attention"
                layer_groups[group] += parameter.grad.detach().float().square().sum()
            for group, squared in layer_groups.items():
                self.log(
                    f"train/decoder_layer_{layer_index}_{group}_grad_norm",
                    squared.sqrt(), on_step=True, on_epoch=True,
                    prog_bar=False, sync_dist=True,
                )

        # Difference from the previous optimizer boundary (detached snapshot).
        current_snapshot = {
            name: parameter.detach().float().clone()
            for name, parameter in decoder.named_parameters()
        }
        if self._decoder_weight_snapshot is not None:
            for layer_index in range(len(getattr(decoder, "layers", ()) )):
                for group in ("shared_attention", "ffn", "time_modulation", "regression", "classification"):
                    deltas = []
                    for name, value in current_snapshot.items():
                        if not name.startswith(f"layers.{layer_index}."):
                            continue
                        if group == "regression" and "plan_reg_branch" not in name:
                            continue
                        if group == "classification" and "plan_cls_branch" not in name:
                            continue
                        if group == "ffn" and "ffn" not in name:
                            continue
                        if group == "time_modulation" and "time_modulation" not in name:
                            continue
                        if group == "shared_attention" and any(
                            token in name for token in ("plan_reg_branch", "plan_cls_branch", "ffn", "time_modulation")
                        ):
                            continue
                        deltas.append((value - self._decoder_weight_snapshot[name]).square().sum())
                    change = torch.stack(deltas).sum().sqrt() if deltas else torch.zeros((), device=self.device)
                    self.log(
                        f"train/decoder_layer_{layer_index}_{group}_weight_change",
                        change, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
                    )
        self._decoder_weight_snapshot = current_snapshot

        grad_norms = {name: value.sqrt() for name, value in squared_norms.items()}
        total_grad_norm = sum(squared_norms.values()).sqrt()
        perception_squared_norm = torch.zeros((), device=self.device)
        backbone = getattr(model, "_backbone", None)
        if backbone is not None:
            for parameter in backbone.parameters():
                if parameter.grad is not None:
                    perception_squared_norm = (
                        perception_squared_norm
                        + parameter.grad.detach().float().square().sum()
                    )
        perception_grad_norm = perception_squared_norm.sqrt()
        training_mode = getattr(
            getattr(self.agent, "_config", None), "grpo_training_mode", None
        )
        if training_mode in {
            "generation", "generation_group", "generation_group_adaptive"
        } and (
            grad_norms["classification"] != 0 or perception_grad_norm != 0
        ):
            raise RuntimeError(
                "Generation-only GRPO produced classification/perception gradients"
            )
        if not all(torch.isfinite(value) for value in (
            *grad_norms.values(), total_grad_norm, perception_grad_norm
        )):
            raise FloatingPointError("Non-finite diff_decoder gradient norm")
        self.log(
            "train/diff_decoder_grad_norm", total_grad_norm,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        self.log(
            "train/perception_grad_norm", perception_grad_norm,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
        )
        for group, grad_norm in grad_norms.items():
            self.log(
                f"train/{group}_grad_norm", grad_norm,
                on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
            )

        controller_update = getattr(
            self.agent, "update_generation_kl_controller", None
        )
        if controller_update is not None and self._latest_generation_kl is not None:
            controller_metrics = controller_update(self._latest_generation_kl)
            self._latest_generation_kl = None
            for name, value in controller_metrics.items():
                self.log(
                    f"train/{name}", value, on_step=True, on_epoch=False,
                    prog_bar=name in {
                        "generation_kl_coefficient",
                        "generation_kl_rolling_mean",
                    },
                    sync_dist=True,
                )
            should_stop = controller_metrics.get("generation_kl_should_stop")
            if should_stop is not None and bool(should_stop.item()):
                self._adaptive_kl_stop_pending = True

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self._adaptive_kl_stop_pending:
            return
        if not self._adaptive_kl_stop_checkpoint_saved:
            log_dir = getattr(self.logger, "log_dir", self.trainer.default_root_dir)
            checkpoint_dir = Path(log_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"kl-stop-step-{self.global_step}.ckpt"
            self.trainer.save_checkpoint(str(checkpoint_path))
            self._adaptive_kl_stop_checkpoint_saved = True
        self.trainer.should_stop = True

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
