# Multi-seed robustness experiment

This directory contains a cleaned version of the code used for the paper's **20% multi-seed robustness study**. The experiment compares the baseline and GDI training conditions for:

- EfficientNetV2-L
- ViT-S
- ViT-L

using seeds `24`, `42`, `67` and `76`.

The final reported metrics use a fixed decision threshold of `0.5`.

## Data

For direct reproduction, use the exact paper split available on Google Drive:

https://drive.google.com/drive/folders/1Grwl55IqPvrFaZK4f7tlIi9u8ymuQO53?usp=sharing

The training data contain the original training images together with their GDI-generated samples. The test data contain **original images only**.

The 20% experiment is reconstructed from the training data using the same sample-selection logic as the original implementation:

- Baseline: `20%` of original training data and `0%` of eligible generated samples.
- GDI: the same `20%` original subset and `100%` of GDI samples derived from those selected originals.

## Run all experiments

From the repository root:

```bash
python experiments/multiseed/run_multiseed.py \
  --train-csv /path/to/train.csv \
  --train-dir /path/to/train \
  --test-csv /path/to/test.csv \
  --test-dir /path/to/test \
  --output-dir runs/multiseed_20pct \
  --device cuda:0
```

This runs all `3 architectures × 2 conditions × 4 seeds = 24` training runs, evaluates each `best_f1_model.pth` checkpoint at threshold `0.5` and generates aggregate and paired summaries.

To run only a subset:

```bash
python experiments/multiseed/run_multiseed.py \
  --train-csv /path/to/train.csv \
  --train-dir /path/to/train \
  --test-csv /path/to/test.csv \
  --test-dir /path/to/test \
  --models vit_small \
  --conditions gdi \
  --seeds 24 42 \
  --device cuda:0
```

## Experimental settings

The defaults match the experiment code used for the published multi-seed study:

| Setting | Value |
|---|---:|
| Epochs | 100 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 0.05 |
| Layer-wise LR decay for ViTs | 0.75 |
| Dropout | 0.7 |
| Stochastic depth | 0.1 |
| Loss | Focal loss |
| Focal gamma | 2.0 |
| Gradient clipping | 1.0 |
| Early stopping patience | 15 |
| Final decision threshold | 0.5 |

The image normalization used by the experiment is:

```text
mean = [0.3709965, 0.3709965, 0.3709965]
std  = [0.27227294, 0.27227294, 0.27227294]
```

## Reproduction behavior

The code intentionally preserves the training and evaluation behavior of the experimental implementation rather than redesigning the protocol.

During training, the supplied test split is evaluated after every epoch and is used for best-checkpoint selection and early stopping. The post-training robustness evaluation then loads `best_f1_model.pth` and computes Zero-One Accuracy and Macro F1 at the fixed threshold `0.5`. No per-class threshold optimization is performed in this final evaluation.

The training-time evaluation transform and the final fixed-threshold inference transform are also preserved as implemented in the original experiment code.

## Files

- `dataset.py` — original/generated sample identification and deterministic 20% subset selection.
- `models.py` — ViT-S, ViT-L, EfficientNetV2-L and focal loss.
- `train.py` — one training run.
- `evaluate.py` — fixed-threshold (`0.5`) inference for one checkpoint.
- `run_multiseed.py` — runs the complete architecture/condition/seed grid.
- `summarize.py` — aggregates per-seed JSON results into mean ± standard deviation and paired comparisons.
- `published_results.md` — multi-seed tables reported in the paper.

## Published results

See [`published_results.md`](published_results.md) for the reported aggregate and paired multi-seed results.
