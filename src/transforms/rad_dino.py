import torch
from transformers import AutoImageProcessor


class RadDinoTransformer:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(
            'microsoft/rad-dino',
            use_fast=True,
        )

    def __call__(self, img: torch.FloatTensor) -> torch.FloatTensor:
        return self.processor(img, do_rescale=False, return_tensors='pt')['pixel_values'][0]
