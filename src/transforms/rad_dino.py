import torch
import torch.nn as nn
from transformers import AutoImageProcessor


class RadDinoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            'microsoft/rad-dino',
            use_fast=True,
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        processed = self.processor(img, do_rescale=False, return_tensors='pt')['pixel_values']
        if img.ndim == 3:
            return processed[0]
        elif img.ndim == 4:
            return processed
        else:
            raise ValueError(
                f'Invalid image dimension. Supported dimensions are 3 or 4, got {img.shape}'
            )
