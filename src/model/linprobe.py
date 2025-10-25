from typing import Literal

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        freeze_encoder: bool = True,
        pooling: Literal['cls', 'mean'] = 'cls',
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.embedding_dim = self.encoder.config.hidden_size
        self.pooling = pooling
        self.head = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, imgs: torch.FloatTensor, **kwargs) -> dict[str, torch.FloatTensor]:
        outputs = self.encoder(imgs)
        if self.pooling == 'cls':
            features = outputs.pooler_output
        elif self.pooling == 'mean':
            features = outputs.last_hidden_state.mean(dim=1)

        logits = self.head(features)
        return {'logits': logits}

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(True)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(False)
