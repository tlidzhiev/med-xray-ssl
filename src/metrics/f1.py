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
    ) -> None:
        """
        Initialize F1 Score metric.

        Parameters
        ----------
        device : str
            Device to run metric computation on ('cuda', 'cpu', or 'auto').
        name : str, optional
            Name of the metric, by default None.
        threshold : float, optional
            Threshold for binary classification, by default 0.5.
        **kwargs
            Additional keyword arguments passed to torchmetrics.F1Score.
        """
        super().__init__(name=name)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold
        self.f1 = F1Score(
            task='binary',
            threshold=threshold,
            **kwargs,
        ).to(device)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> None:
        """
        Update F1 score metric with predictions and labels.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits.
        labels : torch.Tensor
            Ground truth labels.
        **kwargs
            Additional keyword arguments (unused).

        Notes
        -----
        Predictions are computed using argmax over logits before updating the metric.
        """
        with torch.no_grad():
            logits = logits.detach()
            preds = torch.argmax(logits, dim=1)
            self.f1.update(preds, labels)  # ty: ignore[invalid-argument-type]

    def __call__(
        self,
        logits: torch.FloatTensor | None = None,
        labels: torch.FloatTensor | None = None,
        **kwargs,
    ) -> float:
        """
        Compute F1 score and reset internal state.

        Parameters
        ----------
        logits : torch.FloatTensor or None, optional
            Model output logits, by default None.
        labels : torch.FloatTensor or None, optional
            Ground truth labels, by default None.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Computed F1 score.

        Notes
        -----
        This method computes the final F1 score and automatically resets
        the metric state for the next epoch.
        """
        if logits is not None and labels is not None:
            self.update(logits=logits, labels=labels)
        value = self.f1.compute().item()  # ty: ignore[missing-argument, possibly-missing-attribute]
        self.f1.reset()
        return value

    def __repr__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation with task type and threshold.
        """
        return f'{type(self).__name__}(task=binary, threshold={self.threshold})'
