# Published multi-seed robustness results

The paper repeats the 20% data experiment for four random seeds (`24`, `42`, `67`, `76`) and reports final inference metrics at a fixed decision threshold of `0.5`.

| Architecture | Method | Accuracy (mean ± std) | Macro F1 (mean ± std) |
|---|---|---:|---:|
| EfficientNetV2-L | Baseline | 0.3439 ± 0.0072 | 0.5743 ± 0.0168 |
| EfficientNetV2-L | **GDI** | **0.3872 ± 0.0318** | **0.6006 ± 0.0057** |
| ViT-S | Baseline | 0.3467 ± 0.0123 | 0.5710 ± 0.0057 |
| ViT-S | **GDI** | **0.3883 ± 0.0111** | **0.5965 ± 0.0119** |
| ViT-L | Baseline | 0.3083 ± 0.0196 | 0.5430 ± 0.0199 |
| ViT-L | **GDI** | **0.3379 ± 0.0217** | **0.5572 ± 0.0140** |

In the paired seed-wise comparison, GDI improved Zero-One Accuracy in **11 of 12** runs and Macro F1 in **10 of 12** runs.

## Paired relative changes

| Architecture | Seed | Δ Accuracy | Δ Macro F1 |
|---|---:|---:|---:|
| EfficientNetV2-L | 24 | +14.5% | +4.9% |
| EfficientNetV2-L | 42 | −1.0% | +4.9% |
| EfficientNetV2-L | 67 | +25.5% | +0.4% |
| EfficientNetV2-L | 76 | +11.9% | +8.4% |
| ViT-S | 24 | +15.0% | +2.6% |
| ViT-S | 42 | +12.0% | +3.5% |
| ViT-S | 67 | +2.7% | +7.3% |
| ViT-S | 76 | +19.0% | +4.6% |
| ViT-L | 24 | +5.7% | +2.8% |
| ViT-L | 42 | +9.5% | −1.8% |
| ViT-L | 67 | +8.9% | −1.5% |
| ViT-L | 76 | +14.5% | +11.7% |
