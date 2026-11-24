# Results

Preliminary experiments on speeding up FLUX.2 Klein Base 4B on an M1 Pro with 16 GB, at 512×512 with 50 sampling steps and guidance 3.5. Most use 5 prompts × 3 seeds.

## Best config: `taylor-3+bf16+layer`

Compared with the full model (`full-50`, `bf16` weights):

- **About 3× faster** (337 → 112 s per image)
- **2 GB peak memory** instead of 10.8 GB
- **About 2,000 J per image** instead of 5,600 J
- **Close in quality:** LPIPS 0.075, ImageReward 1.43 vs 1.47

It combines three parts:

| Part | What it does |
|---|---|
| `taylor-3` | Runs the full model every 3rd step (17 of 50) and predicts the steps in between (TaylorSeer). |
| `bf16` | Keeps full-precision weights. |
| `+layer` | Streams weights from disk one block at a time, so only one block is in memory. |

## Studies

| Topic | Note | Finding |
|---|---|---|
| Profiling | [profile.md](profiling/profile.md) | Denoising is about 99% of the time; linear layers about 85%. |
| Feature reuse | [summary.md](feature-reuse/summary.md) | Predicting skipped steps (Taylor) beats reusing them, and SpeCa's checks cost more than they save. |
| | [teacache.md](feature-reuse/teacache.md) | TeaCache acts like a fixed schedule and is worse than `taylor-3`. |
| | [methods.md](feature-reuse/methods.md) | ToMA blurs images at 512×512; RA-CFGCache doesn't beat `taylor-3`. |
| | [sweep.md](feature-reuse/sweep.md) | The Taylor interval is a speed–quality dial (2.1× to 4.9×). Energy follows time, and streaming works with little memory. |
| Memory | [streaming.md](memory/streaming.md) | Layer streaming cuts peak memory by over 80% with identical images and no slowdown. |
| Weight format | [svdquant.md](quantization/svdquant.md) | 4-bit weights change images a lot and aren't faster on M1. SVDQuant helps only a little; `q8` is close to `bf16`. |
| Combined | [combined.md](combined.md) | With streaming, `bf16` is the best format: fastest and closest to the full model. |

Each note has its per-image data in a CSV next to it.

## Caveats

- Small samples (15 images per config) on one machine.
- Quality is measured against the full model's image. Energy is a CPU+GPU estimate.
- Early notes ([summary.md](feature-reuse/summary.md), [teacache.md](feature-reuse/teacache.md), [streaming.md](memory/streaming.md)) used `q4` weights, so their numbers aren't directly comparable with the later `bf16` notes.
