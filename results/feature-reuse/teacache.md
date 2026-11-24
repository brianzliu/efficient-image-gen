# TeaCache: preliminary results

This is a preliminary experiment testing TeaCache on 4-bit FLUX.2 Klein.

**Setup:** 5 prompts, 3 seeds, 512×512, guidance 3.5, on an M1 Pro with 16 GB of memory. Generated, scored and validated with `scripts/evaluate-klein.py`.

## Configs

| Config | Description |
|---|---|
| `full-50` | Plain Klein for 50 steps. This is the quality reference. |
| `full-17` | Plain Klein for 17 steps, with no reuse. |
| `taylor-3` | 50 steps. Runs the blocks every 3rd step and predicts the output in between (TaylorSeer). |
| `teacache-0.11` | 50 steps. Skips the blocks until the change in the first block's input adds up to 0.11, and reuses the last output while skipping (TeaCache). |
| `teacache-taylor-0.11` | Same schedule as `teacache-0.11`, but predicts the skipped outputs like `taylor-3`. |

The threshold 0.11 was chosen on 4 separate prompts (`prompts/calibration.txt`) to give 17 full steps, matching `taylor-3`. See `scripts/calibrate-teacache.py`.

## Results

Values are means over 15 images per config. Every config except `full-50` ran 17 full steps.

| Config | Time (s) | Speedup | LPIPS ↓ | SSIM ↑ | PSNR (dB) ↑ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `full-50` | 359 | 1.00x | 0 | 1 | n/a | 1.45 |
| `full-17` | 125 | 2.87x | 0.132 | 0.834 | 21.9 | 1.29 |
| `taylor-3` | 123 | 2.91x | 0.073 | 0.906 | 24.8 | 1.44 |
| `teacache-0.11` | 122 | 2.94x | 0.140 | 0.839 | 21.0 | 1.34 |
| `teacache-taylor-0.11` | 126 | 2.86x | 0.107 | 0.873 | 22.4 | 1.43 |

LPIPS, SSIM and PSNR compare each image with the `full-50` image for the same prompt and seed.

## Takeaways

- `teacache-0.11` was no closer to `full-50` than `full-17`.
- `teacache-taylor-0.11` was closer than `full-17`, but further than `taylor-3` in 14 of 15 runs (LPIPS).
- TeaCache's input change was nearly the same for every prompt, so it acts like a fixed schedule. That schedule has some 4-step gaps, which hurt the Taylor prediction.

## Limitations

- The sample is small, and runs that share a prompt are not independent.
- Only one threshold was tested.

Per-run data is in [teacache.csv](teacache.csv).
