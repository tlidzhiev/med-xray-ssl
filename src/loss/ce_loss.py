import numpy as np
import torch
from torch import nn

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, weight: list[float] | np.ndarray | torch.Tensor | None = None):
        super().__init__()
        if weight is not None:
            if isinstance(weight, list):
                weight = torch.tensor(weight, dtype=torch.float32)
            elif isinstance(weight, np.ndarray):
                weight = torch.from_numpy(weight).float()
            elif isinstance(weight, torch.Tensor):
                weight = weight.float()
            else:
                weight = torch.tensor(weight, dtype=torch.float32)

        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        self.loss_names: list[str] = ['loss']

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return {'loss': self.loss_fn(logits, labels)}
