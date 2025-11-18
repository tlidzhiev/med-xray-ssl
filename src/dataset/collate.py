from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate and pad fields in the dataset items.

    Converts individual items into a batch.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        List of objects from dataset.__getitem__.

    Returns
    -------
    dict[str, torch.Tensor]
        Dict containing batch-version of the tensors.
    """
    images = [item['img'] for item in batch]
    labels = [item['label'] for item in batch]
    return {'imgs': torch.stack(images), 'labels': torch.tensor(labels, dtype=torch.long)}
