#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "efficientnetv2_l": "EfficientNetV2-L",
    "vit_small": "ViT-S",
    "vit_large": "ViT-L",
}
MODEL_ORDER = ["efficientnetv2_l", "vit_small", "vit_large"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize fixed-threshold multi-seed runs.")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(runs_dir.rglob("fixed_threshold_results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "model": data["model"],
                "architecture": DISPLAY_NAMES.get(data["model"], data["model"]),
                "condition": data["condition"],
                "seed": int(data["seed"]),
                "accuracy": float(data["zero_one_accuracy"]),
                "macro_f1": float(data["macro_f1"]),
                "result_file": str(path),
            }
        )

    if not records:
        raise SystemExit(f"No fixed_threshold_results.json files found under {runs_dir}")

    raw = pd.DataFrame(records)
    raw["model_rank"] = raw["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    raw = raw.sort_values(["model_rank", "condition", "seed"]).drop(columns="model_rank")
    raw.to_csv(output_dir / "per_seed_results.csv", index=False)

    aggregate_rows = []
    for (model, condition), group in raw.groupby(["model", "condition"], sort=False):
        aggregate_rows.append(
            {
                "architecture": DISPLAY_NAMES.get(model, model),
                "method": "GDI" if condition == "gdi" else "Baseline",
                "n": len(group),
                "seeds": ", ".join(map(str, sorted(group["seed"].tolist()))),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["architecture_rank"] = aggregate["architecture"].map(
        {DISPLAY_NAMES[m]: i for i, m in enumerate(MODEL_ORDER)}
    )
    aggregate["method_rank"] = aggregate["method"].map({"Baseline": 0, "GDI": 1})
    aggregate = aggregate.sort_values(["architecture_rank", "method_rank"]).drop(
        columns=["architecture_rank", "method_rank"]
    )
    aggregate["Accuracy (mean ± std)"] = aggregate.apply(
        lambda r: f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}", axis=1
    )
    aggregate["Macro F1 (mean ± std)"] = aggregate.apply(
        lambda r: f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}", axis=1
    )
    aggregate.to_csv(output_dir / "aggregate_results.csv", index=False)

    paired_rows = []
    for model in MODEL_ORDER:
        base = raw[(raw["model"] == model) & (raw["condition"] == "baseline")].set_index("seed")
        gdi = raw[(raw["model"] == model) & (raw["condition"] == "gdi")].set_index("seed")
        for seed in sorted(set(base.index) & set(gdi.index)):
            base_acc = base.loc[seed, "accuracy"]
            gdi_acc = gdi.loc[seed, "accuracy"]
            base_f1 = base.loc[seed, "macro_f1"]
            gdi_f1 = gdi.loc[seed, "macro_f1"]
            paired_rows.append(
                {
                    "architecture": DISPLAY_NAMES[model],
                    "seed": seed,
                    "baseline_accuracy": base_acc,
                    "gdi_accuracy": gdi_acc,
                    "accuracy_relative_change_pct": ((gdi_acc - base_acc) / base_acc) * 100,
                    "accuracy_improved": bool(gdi_acc > base_acc),
                    "baseline_macro_f1": base_f1,
                    "gdi_macro_f1": gdi_f1,
                    "macro_f1_relative_change_pct": ((gdi_f1 - base_f1) / base_f1) * 100,
                    "macro_f1_improved": bool(gdi_f1 > base_f1),
                }
            )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_results.csv", index=False)

    display = aggregate[
        ["architecture", "method", "Accuracy (mean ± std)", "Macro F1 (mean ± std)"]
    ]
    markdown = [
        "# Multi-seed robustness summary",
        "",
        "Fixed decision threshold: `0.5`",
        "",
        display.to_markdown(index=False),
        "",
        f"Accuracy improved in {int(paired['accuracy_improved'].sum())}/{len(paired)} paired runs.",
        f"Macro F1 improved in {int(paired['macro_f1_improved'].sum())}/{len(paired)} paired runs.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
