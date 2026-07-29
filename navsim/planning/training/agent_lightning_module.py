import pytorch_lightning as pl
import torch

from pathlib import Path
from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.diffusiondrive.stage37_bistate_projected import (
    project_reward_gradient,
)


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
        self._stage37_preservation_gradients = None
        self._stage37_frontier_delta_sum = None
        self._stage37_microbatch_count = 0

    def _stage37_trainable_parameters(self):
        model = getattr(self.agent, "_transfuser_model", None)
        head = getattr(model, "_trajectory_head", None)
        decoder = getattr(head, "diff_decoder", None)
        if decoder is None:
            raise RuntimeError("Stage37 cannot find the diffusion decoder")
        named = [
            (name, parameter)
            for name, parameter in decoder.named_parameters()
            if parameter.requires_grad
        ]
        if len(named) != 64:
            raise RuntimeError(
                "Stage37 requires exactly 64 trainable decoder tensors; "
                f"got {len(named)}"
            )
        if any("plan_cls_branch" in name for name, _ in named):
            raise RuntimeError("Stage37 classification tensors must remain frozen")
        return named

    def _capture_stage37_preservation_gradient(
        self, loss_dict: Dict[str, Tensor]
    ) -> None:
        preservation = loss_dict.get("stage37_preservation_loss")
        reward = loss_dict.get("stage37_reward_loss")
        frontier_delta = loss_dict.get("stage37_public_frontier_delta_mean")
        if preservation is None or reward is None or frontier_delta is None:
            raise RuntimeError("Stage37 split loss/provenance is incomplete")
        if loss_dict.get("loss") is not reward:
            raise RuntimeError("Stage37 automatic backward must use reward loss only")
        named = self._stage37_trainable_parameters()
        parameters = [parameter for _, parameter in named]
        gradients = torch.autograd.grad(
            preservation,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        detached = [
            (
                torch.zeros_like(parameter)
                if gradient is None
                else gradient.detach().to(parameter)
            )
            for parameter, gradient in zip(parameters, gradients)
        ]
        if not all(torch.isfinite(value).all() for value in detached):
            raise FloatingPointError(
                "Stage37 preservation gradient is non-finite"
            )
        if self._stage37_preservation_gradients is None:
            self._stage37_preservation_gradients = [
                value.clone() for value in detached
            ]
            self._stage37_frontier_delta_sum = (
                frontier_delta.detach().float().clone()
            )
        else:
            if len(self._stage37_preservation_gradients) != len(detached):
                raise RuntimeError("Stage37 gradient accumulator shape drifted")
            for accumulator, value in zip(
                self._stage37_preservation_gradients, detached
            ):
                accumulator.add_(value)
            self._stage37_frontier_delta_sum.add_(
                frontier_delta.detach().float()
            )
        self._stage37_microbatch_count += 1

    def _apply_stage37_gradient_projection(self) -> None:
        named = self._stage37_trainable_parameters()
        if (
            self._stage37_preservation_gradients is None
            or self._stage37_frontier_delta_sum is None
            or self._stage37_microbatch_count <= 0
        ):
            raise RuntimeError("Stage37 optimizer step lacks captured constraints")
        expected = int(getattr(
            getattr(self.agent, "_config", None),
            "stage37_gradient_accumulation",
            8,
        ))
        if self._stage37_microbatch_count != expected:
            raise RuntimeError(
                "Stage37 optimizer boundary has "
                f"{self._stage37_microbatch_count} microbatches, expected {expected}"
            )
        count = float(self._stage37_microbatch_count)
        preserve = [
            value.div(count) for value in self._stage37_preservation_gradients
        ]
        frontier_delta = self._stage37_frontier_delta_sum.div(count)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = float(torch.distributed.get_world_size())
            for value in preserve:
                torch.distributed.all_reduce(
                    value, op=torch.distributed.ReduceOp.SUM
                )
                value.div_(world)
            torch.distributed.all_reduce(
                frontier_delta, op=torch.distributed.ReduceOp.SUM
            )
            frontier_delta.div_(world)
        reward = []
        for name, parameter in named:
            if parameter.grad is None:
                raise RuntimeError(
                    f"Stage37 reward gradient missing for {name}"
                )
            reward.append(parameter.grad.detach().clone())
        projected, diagnostics = project_reward_gradient(
            reward,
            preserve,
            frontier_delta=frontier_delta,
            regression_tolerance=float(getattr(
                self.agent._config,
                "stage37_frontier_regression_tolerance",
                1e-4,
            )),
            recovery_coefficient=float(getattr(
                self.agent._config,
                "stage37_projection_recovery_coefficient",
                0.25,
            )),
            epsilon=float(getattr(
                self.agent._config, "stage37_projection_epsilon", 1e-12
            )),
        )
        for (_, parameter), gradient in zip(named, projected):
            parameter.grad.copy_(gradient)
        for name, value in diagnostics.items():
            self.log(
                f"train/stage37_projection_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=name in {
                    "conflict", "recovery_active", "frontier_delta"
                },
                sync_dist=False,
            )
        self.log(
            "train/stage37_projection_microbatch_count",
            frontier_delta.new_tensor(float(self._stage37_microbatch_count)),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        self._stage37_preservation_gradients = None
        self._stage37_frontier_delta_sum = None
        self._stage37_microbatch_count = 0

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
            if str(getattr(
                getattr(self.agent, "_config", None),
                "grpo_training_mode",
                "",
            )) == "stage37_bistate_projected_deployment_grpo":
                self._capture_stage37_preservation_gradient(loss_dict)
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
        training_mode = getattr(
            getattr(self.agent, "_config", None), "grpo_training_mode", None
        )
        if (
            training_mode not in {
                "diffgrpo_full_chain", "diffgrpo_selected_anchor",
                "diffgrpo_paired_residual",
                "paired_tail_risk_selector",
            }
            and head is not None
            and hasattr(head, "maybe_sync_old_policy")
        ):
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
        training_mode = str(getattr(
            getattr(self.agent, "_config", None), "grpo_training_mode", ""
        ))
        if training_mode == "stage37_bistate_projected_deployment_grpo":
            self._apply_stage37_gradient_projection()
        model = getattr(self.agent, "_transfuser_model", None)
        trajectory_head = getattr(model, "_trajectory_head", None)
        stage39_modes = {
            "stage39_challenger_bc",
            "stage39_challenger_standard_grpo",
            "stage39_challenger_set_grpo",
        }
        decoder = (
            getattr(trajectory_head, "stage39_challenger_decoder", None)
            if training_mode in stage39_modes
            else getattr(trajectory_head, "diff_decoder", None)
        )
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
        value_squared_norm = torch.zeros((), device=self.device)
        value_selector = getattr(trajectory_head, "value_selector", None)
        if value_selector is not None:
            for parameter in value_selector.parameters():
                if parameter.grad is not None:
                    value_squared_norm = (
                        value_squared_norm
                        + parameter.grad.detach().float().square().sum()
                    )
        value_selector_grad_norm = value_squared_norm.sqrt()
        stage23_squared_norm = torch.zeros((), device=self.device)
        stage23_selector = getattr(trajectory_head, "stage23_selector", None)
        if stage23_selector is not None:
            for parameter in stage23_selector.parameters():
                if parameter.grad is not None:
                    stage23_squared_norm = (
                        stage23_squared_norm
                        + parameter.grad.detach().float().square().sum()
                    )
        stage23_selector_grad_norm = stage23_squared_norm.sqrt()
        stage24_squared_norm = torch.zeros((), device=self.device)
        stage24_selector = getattr(trajectory_head, "stage24_selector", None)
        if stage24_selector is not None:
            for parameter in stage24_selector.parameters():
                if parameter.grad is not None:
                    stage24_squared_norm = (
                        stage24_squared_norm
                        + parameter.grad.detach().float().square().sum()
                    )
        stage24_selector_grad_norm = stage24_squared_norm.sqrt()
        stage25_squared_norm = torch.zeros((), device=self.device)
        stage25_selector = getattr(trajectory_head, "stage25_selector", None)
        if stage25_selector is not None:
            for parameter in stage25_selector.parameters():
                if parameter.grad is not None:
                    stage25_squared_norm = (
                        stage25_squared_norm
                        + parameter.grad.detach().float().square().sum()
                    )
        stage25_selector_grad_norm = stage25_squared_norm.sqrt()
        stage37_jfi_squared_norm = torch.zeros((), device=self.device)
        stage37_jfi_selector = getattr(
            trajectory_head, "stage37_jfi_selector", None
        )
        if stage37_jfi_selector is not None:
            for parameter in stage37_jfi_selector.parameters():
                if parameter.grad is not None:
                    stage37_jfi_squared_norm = (
                        stage37_jfi_squared_norm
                        + parameter.grad.detach().float().square().sum()
                    )
        stage37_jfi_grad_norm = stage37_jfi_squared_norm.sqrt()
        paired_squared_norm = torch.zeros((), device=self.device)
        paired_risk = getattr(trajectory_head, "paired_risk_head", None)
        if paired_risk is not None:
            for parameter in paired_risk.parameters():
                if parameter.grad is not None:
                    paired_squared_norm = (
                        paired_squared_norm
                        + parameter.grad.detach().float().square().sum()
                    )
        paired_risk_grad_norm = paired_squared_norm.sqrt()
        training_mode = getattr(
            getattr(self.agent, "_config", None), "grpo_training_mode", None
        )
        if training_mode in {
            "generation", "generation_group", "generation_group_adaptive",
            "diffgrpo_full_chain", "diffgrpo_selected_anchor",
            "diffgrpo_selected_anchor_base_preserve",
            "diffgrpo_paired_residual",
            "diffgrpo_selected_set",
            "stage27_public_diffgrpo_selected_set",
            "stage28_public_paired_uplift_multi",
            "stage28_public_paired_uplift_explore",
            "stage29_public_headroom_hybrid",
            "stage29_public_headroom_conditional",
            "stage30_public_mode_coverage",
            "stage30_public_mode_coverage_constrained",
            "stage31_public_deployed_pair",
            "stage31_public_deployed_frontier",
            "stage32_public_deployed_extended",
            "stage32_selector_aware_frontier",
            "stage33_cdc_grpo",
            "stage34_mode_aligned_frontier_grpo",
            "stage35_nested_counterfactual_deployment_grpo",
            "stage36_reference_gated_tail_ncd_grpo",
            "stage37_bistate_projected_deployment_grpo",
            "stage38_elite_set_counterfactual_repair_grpo",
            "stage39_challenger_bc",
            "stage39_challenger_standard_grpo",
            "stage39_challenger_set_grpo",
        } and (
            grad_norms["classification"] != 0 or perception_grad_norm != 0
        ):
            raise RuntimeError(
                "Generation-only GRPO produced classification/perception gradients"
            )
        if training_mode in {
            "diffgrpo_selected_set",
            "stage27_public_diffgrpo_selected_set",
            "stage28_public_paired_uplift_multi",
            "stage28_public_paired_uplift_explore",
            "stage29_public_headroom_hybrid",
            "stage29_public_headroom_conditional",
            "stage30_public_mode_coverage",
            "stage30_public_mode_coverage_constrained",
            "stage31_public_deployed_pair",
            "stage31_public_deployed_frontier",
            "stage32_public_deployed_extended",
            "stage32_selector_aware_frontier",
            "stage33_cdc_grpo",
            "stage34_mode_aligned_frontier_grpo",
            "stage35_nested_counterfactual_deployment_grpo",
            "stage36_reference_gated_tail_ncd_grpo",
            "stage37_bistate_projected_deployment_grpo",
            "stage38_elite_set_counterfactual_repair_grpo",
            "stage39_challenger_bc",
            "stage39_challenger_standard_grpo",
            "stage39_challenger_set_grpo",
        } and (
            total_grad_norm <= 0
            or value_selector_grad_norm != 0
            or stage23_selector_grad_norm != 0
            or stage24_selector_grad_norm != 0
            or stage25_selector_grad_norm != 0
            or stage37_jfi_grad_norm != 0
            or paired_risk_grad_norm != 0
        ):
            raise RuntimeError(
                "Selected-set GRPO violated its frozen selector boundary or has "
                "zero decoder gradient"
            )
        if training_mode == "value_selector" and (
            total_grad_norm != 0
            or perception_grad_norm != 0
            or value_selector_grad_norm <= 0
            or stage23_selector_grad_norm != 0
            or stage24_selector_grad_norm != 0
            or stage25_selector_grad_norm != 0
        ):
            raise RuntimeError(
                "Value-selector training violated its frozen boundary or has zero gradient"
            )
        if training_mode == "stage23_selector" and (
            total_grad_norm != 0
            or perception_grad_norm != 0
            or value_selector_grad_norm != 0
            or stage23_selector_grad_norm <= 0
            or stage24_selector_grad_norm != 0
            or stage25_selector_grad_norm != 0
            or paired_risk_grad_norm != 0
        ):
            raise RuntimeError(
                "Stage23 selector violated its frozen boundary or has zero gradient"
            )
        if training_mode == "stage24_selector" and (
            total_grad_norm != 0
            or perception_grad_norm != 0
            or value_selector_grad_norm != 0
            or stage23_selector_grad_norm != 0
            or stage24_selector_grad_norm <= 0
            or stage25_selector_grad_norm != 0
            or paired_risk_grad_norm != 0
        ):
            raise RuntimeError(
                "Stage24 selector violated its frozen boundary or has zero gradient"
            )
        if training_mode == "stage25_relative_harm_selector" and (
            total_grad_norm != 0
            or perception_grad_norm != 0
            or value_selector_grad_norm != 0
            or stage23_selector_grad_norm != 0
            or stage24_selector_grad_norm != 0
            or stage25_selector_grad_norm <= 0
            or stage37_jfi_grad_norm != 0
            or paired_risk_grad_norm != 0
        ):
            raise RuntimeError(
                "Stage25 selector violated its frozen boundary or has zero gradient"
            )
        if training_mode == "stage37_jfi_selector" and (
            total_grad_norm != 0
            or perception_grad_norm != 0
            or value_selector_grad_norm != 0
            or stage23_selector_grad_norm != 0
            or stage24_selector_grad_norm != 0
            or stage25_selector_grad_norm != 0
            or stage37_jfi_grad_norm <= 0
            or paired_risk_grad_norm != 0
        ):
            raise RuntimeError(
                "Stage37 JFI violated its frozen encoder boundary or has zero gradient"
            )
        if training_mode == "paired_tail_risk_selector" and (
            total_grad_norm != 0
            or perception_grad_norm != 0
            or value_selector_grad_norm != 0
            or stage23_selector_grad_norm != 0
            or stage24_selector_grad_norm != 0
            or stage25_selector_grad_norm != 0
            or paired_risk_grad_norm <= 0
        ):
            raise RuntimeError(
                "Paired-risk training violated its frozen boundary or has zero gradient"
            )
        if not all(torch.isfinite(value) for value in (
            *grad_norms.values(), total_grad_norm, perception_grad_norm,
            value_selector_grad_norm, stage23_selector_grad_norm,
            stage24_selector_grad_norm, paired_risk_grad_norm,
            stage25_selector_grad_norm, stage37_jfi_grad_norm,
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
        self.log(
            "train/value_selector_grad_norm", value_selector_grad_norm,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        self.log(
            "train/stage23_selector_grad_norm", stage23_selector_grad_norm,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
        )
        self.log(
            "train/stage24_selector_grad_norm", stage24_selector_grad_norm,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        self.log(
            "train/stage25_selector_grad_norm", stage25_selector_grad_norm,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        self.log(
            "train/stage37_jfi_grad_norm", stage37_jfi_grad_norm,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        self.log(
            "train/paired_risk_grad_norm", paired_risk_grad_norm,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
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
