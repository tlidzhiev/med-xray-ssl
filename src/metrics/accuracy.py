import torch
from torchmetrics import Accuracy

from .base import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(
        self,
        device: str,
        name: str | None = None,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(name=name)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.acc = Accuracy(
            task='binary',
            threshold=threshold,
            **kwargs,
        ).to(device)

    def update(self, logits: torch.FloatTensor, labels: torch.FloatTensor):
        with torch.no_grad():
            logits = logits.detach()
            preds = torch.argmax(logits, dim=1)
            self.acc.update(preds, labels)

    def __call__(self) -> float:
        value = self.acc.compute().item()
        self.acc.reset()
        return value

    def __repr__(self) -> str:
        return f'{type(self).__name__}(task=binary)'
