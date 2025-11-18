from typing import Any, Literal

import torch

from src.metrics.tracker import MetricTracker

from .base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class.

    Defines the logic of batch logging and processing for models.
    """

    def process_batch(
        self,
        batch: dict[str, Any],
        metric_tracker: MetricTracker,
        part: Literal['train', 'val', 'test'] | str = 'train',
    ) -> dict[str, Any]:
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.
        metric_tracker : MetricTracker
            MetricTracker object that computes and aggregates the metrics.
            The metrics depend on the type of the partition (train or inference).
        part : {'train', 'val', 'test'}, optional
            Partition type, by default 'train'.

        Returns
        -------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader
            (possibly transformed via batch transform), model outputs, and losses.
        """
        batch = self._to_device(batch)
        batch = self._transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics['train' if part == 'train' else 'inference']
        if part == 'train':
            self.optimizer.zero_grad()

        output = self.model(**batch)
        batch.update(output)
        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if part == 'train':
            batch['loss'].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.criterion.loss_names:
            metric_tracker.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metric_tracker.update(met.name, met(**batch))

        return batch

    @torch.no_grad()
    def _log_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        epoch: int,
    ) -> None:
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch.
        batch : dict[str, Any]
            Dict-based batch after going through the 'process_batch' function.
        epoch : int
            Current epoch number.
        """
        pass
