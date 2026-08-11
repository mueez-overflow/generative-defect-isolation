#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MultiLabelDataset
from models import MultiLabelClassifier, SUPPORTED_MODELS


MEAN = [0.3709965, 0.3709965, 0.3709965]
STD = [0.27227294, 0.27227294, 0.27227294]
THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a published multi-seed checkpoint at fixed threshold 0.5."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS)
    parser.add_argument("--condition", choices=("baseline", "gdi"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_name = args.model or checkpoint.get("model_name")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model name must be supplied or stored in the checkpoint.")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_dataset = MultiLabelDataset(args.test_csv, args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MultiLabelClassifier(
        num_classes=len(test_dataset.classes),
        model_name=model_name,
        dropout_rate=float(checkpoint.get("dropout", 0.7)),
        stochastic_depth_prob=float(checkpoint.get("stochastic_depth", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            all_probs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    predictions = (probs >= THRESHOLD).astype(int)

    class_f1 = {
        class_name: float(
            f1_score(labels[:, idx], predictions[:, idx], zero_division=0)
        )
        for idx, class_name in enumerate(test_dataset.class_names)
    }
    result = {
        "model": model_name,
        "condition": args.condition,
        "seed": args.seed if args.seed is not None else checkpoint.get("seed"),
        "threshold": THRESHOLD,
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "num_test_samples": int(labels.shape[0]),
        "zero_one_accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(list(class_f1.values()))),
        "classwise_f1": class_f1,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
