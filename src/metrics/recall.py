import torch
from torchmetrics import Recall

from .base import BaseMetric


class RecallMetric(BaseMetric):
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
        self.recall = Recall(
            task='binary',
            threshold=threshold,
            **kwargs,
        ).to(device)

    def update(self, logits: torch.FloatTensor, labels: torch.FloatTensor):
        with torch.no_grad():
            logits = logits.detach()
            preds = torch.argmax(logits, dim=1)
            self.recall.update(preds, labels)

    def __call__(self) -> float:
        value = self.recall.compute().item()
        self.recall.reset()
        return value

    def __repr__(self) -> str:
        return f'{type(self).__name__}(task=binary)'
