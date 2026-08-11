#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import MultiLabelDataset
from models import FocalLoss, MultiLabelClassifier, SUPPORTED_MODELS


MEAN = [0.3709965, 0.3709965, 0.3709965]
STD = [0.27227294, 0.27227294, 0.27227294]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
            ),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def build_training_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def train_epoch(model, loader, criterion, optimizer, scheduler, device, gradient_clip):
    model.train()
    total_loss = 0.0
    total_hamming = 0.0
    total_zero_one_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.cpu().numpy()
            pred = (probs >= 0.5).astype(int)
            total_hamming += hamming_loss(labels_np, pred)
            total_zero_one_loss += 1 - accuracy_score(labels_np, pred)
            all_probs.append(probs)
            all_labels.append(labels_np)

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    return {
        "loss": total_loss / len(loader),
        "hamming_loss": total_hamming / len(loader),
        "zero_one_loss": total_zero_one_loss / len(loader),
        "accuracy": float(accuracy_score(labels, (probs >= 0.5).astype(int))),
        "macro_f1": float(f1_score(labels, (probs >= 0.5).astype(int), average="macro", zero_division=0)),
    }


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_hamming = 0.0
    total_zero_one_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels_np = labels.cpu().numpy()
            pred = (probs >= 0.5).astype(int)

            total_loss += loss.item()
            total_hamming += hamming_loss(labels_np, pred)
            total_zero_one_loss += 1 - accuracy_score(labels_np, pred)
            all_probs.append(probs)
            all_labels.append(labels_np)

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    pred = (probs >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader),
        "hamming_loss": total_hamming / len(loader),
        "zero_one_loss": total_zero_one_loss / len(loader),
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
    }


def build_optimizer(model, model_name: str, lr: float, weight_decay: float, layer_decay: float):
    momentum = 0.9

    if "vit" in model_name and hasattr(model.backbone, "blocks"):
        num_layers = len(model.backbone.blocks)
        layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
        parameters = [
            {"params": model.backbone.patch_embed.parameters(), "lr": lr * layer_scales[-1]}
        ]
        for i, block in enumerate(model.backbone.blocks):
            parameters.append({"params": block.parameters(), "lr": lr * layer_scales[i]})
        parameters.append({"params": model.classifier.parameters(), "lr": lr})
        return torch.optim.AdamW(
            parameters, weight_decay=weight_decay, betas=(momentum, 0.999)
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(momentum, 0.999),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one run from the 20% multi-seed robustness experiment."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--original-pct", type=float, default=20.0)
    parser.add_argument("--inpainted-pct", type=float, choices=(0.0, 100.0), required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--stochastic-depth", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd_read_classes(args.train_csv)
    class_names = raw
    original_percentages = {name: args.original_pct for name in class_names}
    inpainted_percentages = {name: args.inpainted_pct for name in class_names}
    test_percentages = {name: 100 for name in class_names}

    train_dataset = MultiLabelDataset(
        args.train_csv,
        args.train_dir,
        transform=build_train_transform(),
        original_percentages=original_percentages,
        inpainted_percentages=inpainted_percentages,
    )
    test_dataset = MultiLabelDataset(
        args.test_csv,
        args.test_dir,
        transform=build_training_eval_transform(),
        original_percentages=test_percentages,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = MultiLabelClassifier(
        num_classes=len(train_dataset.classes),
        model_name=args.model,
        dropout_rate=args.dropout,
        stochastic_depth_prob=args.stochastic_depth,
    ).to(device)

    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = build_optimizer(
        model, args.model, args.lr, args.weight_decay, args.layer_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["train_dataset"] = train_dataset.summary()
    config["test_dataset"] = test_dataset.summary()
    config["class_names"] = train_dataset.class_names
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_accuracy = 0.0
    best_f1 = 0.0
    early_stop_counter = 0
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args.gradient_clip,
        )
        test_metrics = evaluate_epoch(model, test_loader, criterion, device)

        is_best_accuracy = test_metrics["accuracy"] > best_accuracy
        is_best_f1 = test_metrics["macro_f1"] > best_f1

        if is_best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(
                checkpoint_payload(model, optimizer, scheduler, args, epoch, best_accuracy, best_f1),
                output_dir / "best_accuracy_model.pth",
            )

        if is_best_f1:
            best_f1 = test_metrics["macro_f1"]
            torch.save(
                checkpoint_payload(model, optimizer, scheduler, args, epoch, best_accuracy, best_f1),
                output_dir / "best_f1_model.pth",
            )

        if is_best_accuracy or is_best_f1:
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "test": test_metrics,
            "is_best_accuracy": is_best_accuracy,
            "is_best_f1": is_best_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train acc={train_metrics['accuracy']:.4f} f1={train_metrics['macro_f1']:.4f} | "
            f"test acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}"
        )

        if args.early_stopping > 0 and early_stop_counter >= args.early_stopping:
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    summary = {
        "best_test_accuracy_during_training": best_accuracy,
        "best_test_macro_f1_during_training": best_f1,
        "epochs_completed": len(history),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def pd_read_classes(csv_path: str) -> list[str]:
    import pandas as pd

    columns = pd.read_csv(csv_path, nrows=1).columns.tolist()
    return [column for column in columns if column != "filename"]


def checkpoint_payload(model, optimizer, scheduler, args, epoch, best_accuracy, best_f1):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_accuracy,
        "best_val_f1": best_f1,
        "seed": args.seed,
        "model_name": args.model,
        "dropout": args.dropout,
        "stochastic_depth": args.stochastic_depth,
    }


if __name__ == "__main__":
    main()
