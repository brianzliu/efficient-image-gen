# Feature reuse: preliminary results

This is a preliminary experiment comparing feature reuse methods on 4-bit FLUX.2 Klein.

**Setup:** 5 prompts, 3 seeds, 512×512, guidance 3.5, on an M1 Pro with 16 GB of memory.

## Configs

| Config | Description |
|---|---|
| `full-50` | Plain Klein for 50 steps. This is the quality reference. |
| `full-17` | Plain Klein for 17 steps, with no reuse. This is the speed baseline. |
| `fixed-3` | 50 steps. Runs the blocks every 3rd step and reuses the last output in between. |
| `taylor-3` | 50 steps. Runs the blocks every 3rd step and predicts the output in between (TaylorSeer, order 1). |
| `speca-0.15-0.3` | 50 steps. Predicts the output, checks it with the last block, and redoes the step in full if the error is above the threshold (SpeCa). The starting threshold is 0.15 and the decay is 0.3. |

## Metrics

| Metric | Measures | Better |
|---|---|---|
| PSNR (dB) | Pixel closeness to `full-50` | Higher |
| SSIM | Structural closeness to `full-50` | Higher |
| LPIPS (AlexNet) | Perceptual distance from `full-50` | Lower |
| CLIP score (ViT-L/14) | How well the image matches the prompt, as 100 × cosine similarity | Higher |
| ImageReward (v1.0) | Predicted human preference for the image given the prompt | Higher |

PSNR, SSIM, and LPIPS compare against the `full-50` image with the same prompt and seed, so they show closeness, not how good an image looks.

## Results

Values are means over 15 images per config.

| Config | Full steps | Denoise time (s) | Speedup | PSNR | SSIM | LPIPS | CLIP | ImageReward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full-50` | 50 | 318.4 | 1.00x | n/a | n/a | n/a | 33.1 | 1.45 |
| `full-17` | 17 | 108.0 | 2.95x | 21.91 | 0.834 | 0.132 | 33.4 | 1.29 |
| `fixed-3` | 17 | 108.7 | 2.93x | 21.60 | 0.850 | 0.127 | 33.2 | 1.32 |
| `taylor-3` | 17 | 107.9 | 2.95x | 24.84 | 0.906 | 0.073 | 33.2 | 1.44 |
| `speca-0.15-0.3` | 18.5 | 125.7 | 2.53x | 25.04 | 0.906 | 0.073 | 33.1 | 1.41 |

Peak MLX memory was about 3.4 GB for every config.

## Taylor order ablation

All Taylor configs run 17 full steps. Orders 0 and 1 are the `fixed-3` and `taylor-3` runs above. Orders 2 to 4 are `taylor-3-2`, `taylor-3-3`, and `taylor-3-4`.

| Config | Denoise time (s) | Speedup | Cache (MB) | PSNR | SSIM | LPIPS | CLIP | ImageReward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full-50` | 318.4 | 1.00x | 0 | n/a | n/a | n/a | 33.1 | 1.45 |
| `full-17` | 108.0 | 2.95x | 0 | 21.91 | 0.834 | 0.132 | 33.4 | 1.29 |
| Order 0 | 108.7 | 2.93x | 37.7 | 21.60 | 0.850 | 0.127 | 33.2 | 1.32 |
| Order 1 | 107.9 | 2.95x | 75.5 | 24.84 | 0.906 | 0.073 | 33.2 | 1.44 |
| Order 2 | 107.8 | 2.95x | 113.2 | 24.84 | 0.907 | 0.073 | 33.1 | 1.44 |
| Order 3 | 107.3 | 2.97x | 151.0 | 24.94 | 0.906 | 0.073 | 33.1 | 1.44 |
| Order 4 | 108.2 | 2.94x | 188.7 | 24.83 | 0.904 | 0.074 | 33.1 | 1.44 |

## Takeaways

- `taylor-3` was closer to `full-50` than `full-17` in 14 of 15 runs on PSNR, SSIM, and LPIPS, at about the same time.
- `fixed-3` was no better than `full-17` on PSNR.
- `speca-0.15-0.3` took 16% longer than `taylor-3` for a similar PSNR.
- Going from order 0 to order 1 made the largest difference. Orders 2 to 4 added cache but showed no clear gain over order 1.
- ImageReward was close to `full-50` for every forecasting config and lower for `full-17` and `fixed-3`.
- CLIP score barely changed across configs, so it did not separate them here.

## Limitations

- The sample is small, and runs that share a prompt are not independent.
- Only one SpeCa setting was tested, so this does not show that SpeCa in general is worse.
- Orders 2 to 4 ran in a later session than the other configs, so small time differences are not meaningful.

Per-run data is in [results.csv](results.csv) and [scores.csv](scores.csv). Scoring uses its own environment, from [requirements-metrics.txt](../requirements-metrics.txt).
