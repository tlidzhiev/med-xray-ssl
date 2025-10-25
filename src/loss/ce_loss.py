import numpy as np
import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
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
        self.loss_names = ['loss']

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        **kwargs,
    ) -> dict[str, torch.FloatTensor]:
        return {'loss': self.loss_fn(logits, labels)}
