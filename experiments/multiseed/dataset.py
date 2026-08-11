from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    """Dataset loader used by the published multi-seed robustness experiment.

    The CSV is expected to contain a ``filename`` column followed by one binary
    column per class. Generated GDI samples are identified from the filename
    suffix ``_<ClassName>.jpg``. When only a fraction of original data is used,
    generated samples are eligible only when they are derived from an original
    image selected into that subset.
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        transform=None,
        original_percentages: Mapping[str, float] | None = None,
        inpainted_percentages: Mapping[str, float] | None = None,
        random_seed: int = 42,
    ) -> None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.data = pd.read_csv(csv_path).sort_values("filename").reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.original_percentages = dict(original_percentages or {})
        self.inpainted_percentages = dict(inpainted_percentages or {})

        self.classes = [column for column in self.data.columns if column != "filename"]
        self.class_names = self.classes.copy()

        for column in self.classes:
            self.data[column] = (
                pd.to_numeric(self.data[column], errors="coerce")
                .fillna(0)
                .astype(np.float32)
            )

        self.original_samples: list[int] = []
        self.inpainted_samples: dict[str, list[int]] = {
            class_name: [] for class_name in self.class_names
        }
        self.original_to_inpainted_map: dict[int, list[int]] = {}
        self.original_samples_by_class: dict[str, list[int]] = {
            class_name: [] for class_name in self.class_names
        }

        filename_to_index = {
            str(row["filename"]): idx for idx, row in self.data.iterrows()
        }

        for idx, row in self.data.iterrows():
            filename = str(row["filename"])
            is_inpainted = False

            for class_name in self.class_names:
                suffix = f"_{class_name}.jpg"
                if filename.endswith(suffix):
                    self.inpainted_samples[class_name].append(idx)
                    is_inpainted = True
                    original_filename = filename[: -len(suffix)] + ".jpg"
                    original_idx = filename_to_index.get(original_filename)
                    if original_idx is not None:
                        self.original_to_inpainted_map.setdefault(original_idx, []).append(idx)
                    break

            if not is_inpainted:
                self.original_samples.append(idx)
                for class_name in self.class_names:
                    if row[class_name] > 0:
                        self.original_samples_by_class[class_name].append(idx)

        self.indices = self._select_samples()
        subset_data = self.data.iloc[self.indices]
        self.class_counts = {
            column: int(subset_data[column].sum()) for column in self.classes
        }

    def _select_samples(self) -> list[int]:
        selected_indices: list[int] = []
        selected_original_indices: set[int] = set()

        if self.original_percentages:
            for class_name, percentage in sorted(self.original_percentages.items()):
                if class_name not in self.original_samples_by_class or percentage <= 0:
                    continue
                available_samples = sorted(self.original_samples_by_class[class_name])
                num_to_include = max(1, int(len(available_samples) * percentage / 100))
                num_to_include = min(num_to_include, len(available_samples))
                selected_original_indices.update(available_samples[:num_to_include])
        else:
            selected_original_indices = set(sorted(self.original_samples))

        selected_indices.extend(sorted(selected_original_indices))

        eligible_inpainted_samples = {
            class_name: [] for class_name in self.class_names
        }
        for original_idx in sorted(selected_original_indices):
            for inpainted_idx in sorted(self.original_to_inpainted_map.get(original_idx, [])):
                filename = str(self.data.iloc[inpainted_idx]["filename"])
                for class_name in sorted(self.class_names):
                    if filename.endswith(f"_{class_name}.jpg"):
                        eligible_inpainted_samples[class_name].append(inpainted_idx)
                        break

        for class_name, percentage in sorted(self.inpainted_percentages.items()):
            if percentage <= 0:
                continue
            available_samples = sorted(eligible_inpainted_samples.get(class_name, []))
            if not available_samples:
                continue
            num_to_include = max(1, int(len(available_samples) * percentage / 100))
            num_to_include = min(num_to_include, len(available_samples))
            selected_indices.extend(available_samples[:num_to_include])

        return sorted(selected_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        data_idx = self.indices[idx]
        row = self.data.iloc[data_idx]
        image = Image.open(self.img_dir / str(row["filename"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.tensor(
            row[self.classes].values.astype(np.float32), dtype=torch.float32
        )
        return image, labels

    def summary(self) -> dict:
        original_set = set(self.original_samples)
        original_used = [idx for idx in self.indices if idx in original_set]
        generated_used = [idx for idx in self.indices if idx not in original_set]
        return {
            "total": len(self.indices),
            "original": len(original_used),
            "generated": len(generated_used),
            "class_counts": self.class_counts,
        }
