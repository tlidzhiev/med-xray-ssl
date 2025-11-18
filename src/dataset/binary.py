import shutil
from pathlib import Path
from typing import Any, Callable, Literal

import kagglehub
import numpy as np
import pandas as pd
import safetensors.torch
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import v2
from tqdm.auto import tqdm

from src.dataset.base import BaseDataset
from src.utils.io import get_root, read_json, write_json


class BinaryLabelDataset(BaseDataset):
    def __init__(
        self,
        root: Path | str | None = None,
        split: Literal['train', 'val'] = 'train',
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
        force_reindex: bool = False,
        dim: int = 256,
        val_size: float = 0.2,
    ):
        if root is None:
            root = get_root() / 'data' / f'binary-{dim}' / split
        else:
            root = get_root() / root

        index_path = root / 'index.json'
        if index_path.exists() and not force_reindex:
            index: list[dict[str, Any]] = read_json(str(index_path)) # ty: ignore[invalid-assignment]
        else:
            index: list[dict[str, Any]] = self._create_index(
                split=split,
                data_path=root,
                dim=dim,
                val_size=val_size,
            )

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _create_index(
        self,
        split: Literal['train', 'val'],
        data_path: Path,
        dim: int,
        val_size: float,
    ) -> list[dict[str, Any]]:
        index: list[dict[str, Any]] = []
        data_path.mkdir(exist_ok=True, parents=True)

        kaggle_dataset_path = f'awsaf49/vinbigdata-{dim}-image-dataset'
        kaggle_path = kagglehub.dataset_download(kaggle_dataset_path)
        kaggle_path = Path(kaggle_path) / 'vinbigdata'
        print(f'Path to dataset files: {kaggle_path}')

        train_df_path = kaggle_path / 'train.csv'
        train_data_path = kaggle_path / 'train'

        train_df = pd.read_csv(train_df_path)
        train_df = self._make_binary_labels(train_df, self.NO_FINDING_CLASS)

        train, val = train_test_split(
            train_df,
            test_size=val_size,
            shuffle=True,
            stratify=train_df['label'],
            random_state=42,
        )

        df = train if split == 'train' else val
        print(f'{split.capitalize()} label statistics:\n{df["label"].value_counts(normalize=True)}')

        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        print(f'Processing {split} dataset images...')
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Creating {split} index'):
            image_id = row['image_id']
            label = int(row['label'])

            image_path = train_data_path / f'{image_id}.png'
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).contiguous()

            save_dict = {'tensor': img_tensor}
            save_path = data_path / f'{image_id}.safetensors'
            safetensors.torch.save_file(save_dict, save_path)

            index.append({'path': str(save_path), 'label': label})

        kaggle_path = kaggle_path.parent.parent.parent
        if kaggle_path.exists():
            shutil.rmtree(kaggle_path)
            print(f'Cleaned up original Kaggle data at {kaggle_path}')

        write_json(index, str(data_path / 'index.json'))
        print(f'Successfully processed {len(index)} images for {split} split.')
        return index

    @staticmethod
    def _make_binary_labels(df: pd.DataFrame, no_finding_class: int) -> pd.DataFrame:
        df = df.groupby('image_id')['class_id'].agg(list).reset_index()
        class_id_array = df['class_id'].apply(np.array)
        binary_label = np.array(
            [int(not np.all(arr == no_finding_class)) for arr in class_id_array]
        )
        df['label'] = binary_label
        return df[['image_id', 'label']]

    def compute_weights(self) -> torch.Tensor:
        labels = [item['label'] for item in self._index]
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels,
        ).astype(np.float32)
        weights = torch.from_numpy(weights)
        return weights
