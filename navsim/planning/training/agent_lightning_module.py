import pytorch_lightning as pl
import torch

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
        squared_norm = torch.zeros((), device=self.device)
        for parameter in decoder.parameters():
            if parameter.grad is not None:
                squared_norm = squared_norm + parameter.grad.detach().float().square().sum()
        grad_norm = squared_norm.sqrt()
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite diff_decoder gradient norm")
        self.log(
            "train/diff_decoder_grad_norm", grad_norm,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
        )

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
