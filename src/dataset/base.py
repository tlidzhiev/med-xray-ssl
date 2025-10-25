import random
from typing import Any, Callable

import safetensors.torch
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    NO_FINDING_CLASS = 14

    def __init__(
        self,
        index: list[dict[str, Any]],
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
    ) -> None:
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: list[dict[str, Any]] = index

        self.instance_transforms = instance_transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        data_dict = self._index[index]
        data_path = data_dict['path']
        data_object = self.load_object(data_path)

        data_label = data_dict['label']
        instance_data = {'img': data_object, 'label': data_label}
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self) -> int:
        return len(self._index)

    def load_object(self, path: str) -> torch.Tensor:
        img = safetensors.torch.load_file(path)['tensor']
        return img

    def preprocess_data(self, instance_data: dict[str, Any]) -> dict[str, Any]:
        if self.instance_transforms is not None:
            for name, transform in self.instance_transforms.items():
                if name in instance_data:
                    instance_data[name] = transform(instance_data[name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]]) -> None:
        for entry in index:
            assert 'path' in entry, (
                "Each dataset item should include field 'path' - path to image file."
            )
            assert 'label' in entry, (
                "Each dataset item should include field 'label' - "
                'object ground-truth label (required when use_condition=True).'
            )

    @staticmethod
    def _sort_index(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pass

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[dict[str, Any]],
        limit: int | None,
        shuffle_index: bool,
    ) -> list[dict[str, Any]]:
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
