# ToMA and RA-CFGCache: preliminary results

Two methods from the literature, tested alone and on top of the best recipe so far (`taylor-3+bf16+layer`, see [sweep.md](sweep.md)).

**Setup:** 5 prompts × 3 seeds, 512×512, M1 Pro with 16 GB, `bf16` weights. Run with `scripts/evaluate-klein.py`. Quality is measured against `full-50` (`bf16`). **LPIPS** is perceptual difference (0 = same, lower is better).

## Methods

- **ToMA** (Lu et al., 2025) merges similar image tokens before attention in the single-stream blocks (about 80% of the time, see [profile.md](../profiling/profile.md)) and splits them back after. `toma-0.75` keeps 75% of image tokens. Code: `feature_reuse/toma.py`.
- **RA-CFGCache** (Liu et al., 2026) decides when to run the full model from the estimated error of the guided output, using both the prompt and no-prompt passes. On skipped steps it reuses the last output. Its error estimates are calibrated once on 3 prompts (`scripts/calibrate-racfg.py`), so in practice it is a fixed schedule. The threshold was set to 0.04 so it uses 17 full steps, the same as `taylor-3`. Code: `feature_reuse/racfg.py`.
- `racfg-taylor` uses RA-CFGCache's choice of full steps and predicts the steps in between with Taylor, like `taylor-3`.

Full steps:

| Config | Steps run fully |
|---|---|
| `taylor-3` | every 3rd: 0, 3, 6, …, 48 |
| `racfg-0.04` | 0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 31, 35, 40, 48 |

## Results

| Config | Full steps | Time (s) | Speedup | Peak GB | Energy (J) | LPIPS ↓ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full-50` (reference) | 50 | 337 | 1.00× | 10.8 | 5,621 | 0 | 1.47 |
| `full-50+toma-0.75` | 50 | 293 | 1.15× | 10.8 | 4,902 | 0.259 | 1.27 |
| `racfg-0.04` | 17 | 130 | 2.59× | 10.8 | 2,059 | 0.106 | 1.42 |
| **`taylor-3+bf16+layer`** | 17 | 112 | 3.00× | 2.0 | 1,915 | **0.075** | 1.43 |
| `taylor-3+toma-0.75+bf16+layer` | 17 | 105 | 3.20× | 2.0 | 1,577 | 0.267 | 1.18 |
| `racfg-taylor-0.04+bf16+layer` | 17 | 118 | 2.85× | 2.0 | 1,828 | 0.102 | **1.47** |

Means over 15 images. Energy is estimated CPU+GPU energy per image. Per-run data is in [methods.csv](methods.csv).

- **ToMA isn't worth it here.** It's only 15% faster and changes images a lot (LPIPS 0.26, ImageReward 1.47 → 1.27; lower than `full-50` on 14 of 15 images). Images are softer and objects sometimes move. Stacked on `taylor-3` it adds only 7% speed and loses the same quality.
- **RA-CFGCache alone is worse than `taylor-3`.** Same 17 full steps, but further from `full-50` (LPIPS 0.106 vs 0.075; closer on only 4 of 15 images). Predicting skipped steps beats reusing the last output.
- **RA-CFGCache's schedule with Taylor is a tie.** It's further from `full-50` (LPIPS 0.102 vs 0.075) but matches its ImageReward (1.47; higher than `taylor-3` on 8 of 15 images). With 15 images that difference is noise. `taylor-3` stays the default.

## Why ToMA does poorly at 512×512

- The image is only 32×32 tokens. Each token covers a 16×16 pixel patch, so merging neighbours blurs visible detail. The paper used 1024×1024.
- Merging 25% of image tokens removes only about 17% of all tokens, because the 512 text tokens are untouched.
- Keeping 50% (`toma-0.5`) was 26% faster but clearly blurry in a quick test, so it wasn't run here.
- A merge temperature of 0.1 blurred images badly. Neighbouring tokens are 0.9–0.99 similar, so it had to drop to 0.002.

## Limitations

- `racfg-0.04` ran without `+layer`, as in the paper, so its time and memory aren't directly comparable with the streamed rows. Compare it by quality.
- RA-CFGCache was calibrated on 3 prompts, separate from the 5 test prompts.
- 5 prompts × 3 seeds, one machine. Energy is a CPU+GPU estimate including background activity.
