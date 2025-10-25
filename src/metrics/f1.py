import torch
from torchmetrics import F1Score

from .base import BaseMetric


class F1Metric(BaseMetric):
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
        self.f1 = F1Score(
            task='binary',
            threshold=threshold,
            **kwargs,
        ).to(device)

    def update(self, logits: torch.FloatTensor, labels: torch.FloatTensor):
        with torch.no_grad():
            logits = logits.detach()
            preds = torch.argmax(logits, dim=1)
            self.f1.update(preds, labels)

    def __call__(self) -> float:
        value = self.f1.compute().item()
        self.f1.reset()
        return value

    def __repr__(self) -> str:
        return f'{type(self).__name__}(task=binary)'
