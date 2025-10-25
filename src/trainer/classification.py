from typing import Any, Literal

import torch

from src.metrics.tracker import MetricTracker

from .base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def process_batch(
        self,
        batch: dict[str, Any],
        metrics: MetricTracker,
        part: Literal['train', 'val', 'test'] = 'train',
    ) -> dict[str, Any]:
        batch = self._to_device(batch)
        batch = self._transform_batch(batch)  # transform batch on device -- faster
        print(batch['imgs'].shape)

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
            metrics.update(loss_name, batch[loss_name].item())

        for metric in metric_funcs:
            metric.update(**batch)

        return batch

    @torch.no_grad()
    def _log_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        epoch: int,
    ):
        pass
