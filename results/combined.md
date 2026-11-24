# Combined recipe: preliminary results

What happens when we combine the three methods on FLUX.2 Klein?

**Setup:** 5 prompts × 3 seeds, 512×512, M1 Pro with 16 GB. Run with `scripts/evaluate-klein.py`.

## The three parts

| Part | Options tested | What it does |
|---|---|---|
| Feature reuse | `taylor-3` | Runs the full model on 17 of 50 steps and predicts the rest. See [teacache.md](feature-reuse/teacache.md). |
| Streaming | `+layer` | Loads one block of weights at a time instead of the whole model. See [streaming.md](memory/streaming.md). |
| Weight format | `bf16`, `q8`, `q4`, `svdq-128` | 16-, 8- or 4-bit weights. See [svdquant.md](quantization/svdquant.md). |

## Results

Quality is measured against `full-50+bf16` (all 50 steps, full-precision weights). **LPIPS** is perceptual difference (0 = same, lower is better).

| Config | Time (s) | Speedup | Peak memory (GB) | LPIPS ↓ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|
| `full-50+bf16` (reference) | 338 | 1.00× | 10.77 | 0 | 1.47 |
| `full-50+q4` (old default) | 359 | 0.94× | 5.20 | 0.310 | 1.45 |
| `taylor-3+q4+layer` | 123 | 2.74× | 0.96 | 0.309 | 1.44 |
| `taylor-3+svdq-128+layer` | 129 | 2.62× | 1.22 | 0.270 | 1.36 |
| `taylor-3+q8+layer` | 131 | 2.59× | 1.30 | 0.104 | 1.41 |
| **`taylor-3+bf16+layer`** | **117** | **2.90×** | **1.95** | **0.075** | 1.43 |

Means over 15 images. Time is the whole generation. Configs without a weight suffix used `q4` in earlier notes; full metrics are in [combined.csv](combined.csv), where `full-50` means `full-50+bf16`.

## Compared with plain Taylor

Plain `taylor-3` is the earlier best: q4 weights, no streaming. Streaming gives pixel-identical images, so its quality is the same as `taylor-3+q4+layer`.

| | Plain `taylor-3` (q4) | `taylor-3+bf16+layer` |
|---|---:|---:|
| Time (s) | ~123 | **117** |
| Peak memory (GB) | 5.20 | **1.95** |
| LPIPS vs `full-50+bf16` ↓ | 0.309 | **0.075** |

[teacache.md](feature-reuse/teacache.md) lists plain `taylor-3` at LPIPS 0.073, but that was measured against `full-50` with q4 weights. Against `bf16`, the 4-bit weights add far more difference than the Taylor prediction does.

## Takeaways

- **Best recipe: `taylor-3+bf16+layer`.** Against the old default `full-50+q4`, it is **3.1× faster**, uses **62% less memory** and is much closer to `bf16` (closer on 15 of 15 images).
- **With streaming, full precision is the best choice.** Only one block is in memory at a time, so smaller weights barely save memory, and 4-bit weights hurt quality without being faster.
- **4-bit weights, not feature reuse, cause most of the quality loss.** `taylor-3+q4+layer` and `full-50+q4` are equally far from `bf16`.
- **If disk space is tight**, `taylor-3+q4+layer` is the smallest: 0.96 GB peak, 2.2 GB on disk.

## Limitations

- Each image reads the 7.75 GB weight file 34 times. A follow-up showed these reads came from the SSD, not the file cache, so the times above already include them, and they stayed the same with less free memory. See [sweep.md](feature-reuse/sweep.md).
- 512×512 only, 5 prompts × 3 seeds, one machine.
