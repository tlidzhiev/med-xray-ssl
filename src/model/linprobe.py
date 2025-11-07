from typing import Literal

import torch
import torch.nn as nn

from .base import BaseModel


class LinearProbe(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        embedding_dim: int,
        freeze_encoder: bool = True,
        pooling: Literal['cls', 'mean'] = 'cls',
    ):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()

        self.pooling: Literal['cls', 'mean'] = pooling
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, imgs: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        outputs = self.encoder(imgs)
        if self.pooling == 'cls':
            features = outputs.pooler_output
        elif self.pooling == 'mean':
            features = outputs.last_hidden_state.mean(dim=1)

        logits = self.head(features)
        return {'logits': logits}

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad_(True)
