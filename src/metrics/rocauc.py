import torch
from torchmetrics import AUROC

from .base import BaseMetric


class ROCAUCMetric(BaseMetric):
    def __init__(
        self,
        device: str,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.auroc = AUROC(
            task='binary',
            **kwargs,
        ).to(device)

    def update(self, logits: torch.FloatTensor, labels: torch.FloatTensor, **kwargs):
        with torch.no_grad():
            logits = logits.detach()
            preds = torch.softmax(logits, dim=1)[:, 1]
            self.auroc.update(preds, labels)

    def __call__(self) -> float:
        value = self.auroc.compute().item()
        self.auroc.reset()
        return value

    def __repr__(self) -> str:
        return f'{type(self).__name__}(task=binary)'
