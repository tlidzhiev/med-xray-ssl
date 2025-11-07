import torch
from torchmetrics import AUROC

from .base import BaseMetric


class ROCAUCMetric(BaseMetric):
    def __init__(
        self,
        device: str,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize ROC AUC (Area Under the ROC Curve) metric.

        Parameters
        ----------
        device : str
            Device to run metric computation on ('cuda', 'cpu', or 'auto').
        name : str, optional
            Name of the metric, by default None.
        **kwargs
            Additional keyword arguments passed to torchmetrics.AUROC.
        """
        super().__init__(name=name)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.auroc = AUROC(
            task='binary',
            **kwargs,
        ).to(device)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> None:
        """
        Update ROC AUC metric with predictions and labels.

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
        Predictions are computed using softmax over logits, taking the positive
        class probability (index 1) before updating the metric.
        """
        with torch.no_grad():
            logits = logits.detach()
            preds = torch.softmax(logits, dim=1)[:, 1]
            self.auroc.update(preds, labels)  # ty: ignore[invalid-argument-type]

    def __call__(
        self, logits: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs
    ) -> float:
        """
        Compute ROC AUC score and reset internal state.

        Parameters
        ----------
        logits : torch.Tensor or None, optional
            Model output logits, by default None.
        labels : torch.Tensor or None, optional
            Ground truth labels, by default None.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Computed ROC AUC score.

        Notes
        -----
        This method computes the final ROC AUC score and automatically resets
        the metric state for the next epoch.
        """
        if logits is not None and labels is not None:
            self.update(logits=logits, labels=labels)
        value = self.auroc.compute().item()  # ty: ignore[missing-argument, possibly-missing-attribute]
        self.auroc.reset()
        return value

    def __repr__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation with task type.
        """
        return f'{type(self).__name__}(task=binary)'
