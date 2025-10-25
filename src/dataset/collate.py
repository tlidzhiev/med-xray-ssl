from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    images = [item['img'] for item in batch]
    labels = [item['label'] for item in batch]
    return {'imgs': torch.stack(images), 'labels': torch.tensor(labels, dtype=torch.long)}
