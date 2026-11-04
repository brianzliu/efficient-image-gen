# Feature reuse: preliminary results

This is a preliminary experiment comparing feature reuse methods on 4-bit FLUX.2 Klein.

**Setup:** 5 prompts, 3 seeds, 512×512, guidance 3.5, on an M1 Pro with 16 GB of memory.

## Configs

| Config | Description |
|---|---|
| `full-50` | Plain Klein for 50 steps. This is the quality reference. |
| `full-17` | Plain Klein for 17 steps, with no reuse. This is the speed baseline. |
| `fixed-3` | 50 steps. Runs the blocks every 3rd step and reuses the last output in between. |
| `taylor-3` | 50 steps. Runs the blocks every 3rd step and predicts the output in between (TaylorSeer). |
| `speca-0.15-0.3` | 50 steps. Predicts the output, checks it with the last block, and redoes the step in full if the error is above the threshold (SpeCa). The starting threshold is 0.15 and the decay is 0.3. |

## Results

Values are means over 15 images per config.

| Config | Full steps | Denoise time (s) | Speedup | PSNR (dB) |
|---|---:|---:|---:|---:|
| `full-50` | 50 | 318.4 | 1.00x | n/a |
| `full-17` | 17 | 108.0 | 2.95x | 21.91 |
| `fixed-3` | 17 | 108.7 | 2.93x | 21.60 |
| `taylor-3` | 17 | 107.9 | 2.95x | 24.84 |
| `speca-0.15-0.3` | 18.5 | 125.7 | 2.53x | 25.04 |

PSNR measures pixel closeness to the `full-50` image with the same prompt and seed. Higher is closer. It does not measure how good an image looks.

Peak MLX memory was about 3.4 GB for every config.

## Takeaways

- `taylor-3` was closer to `full-50` than `full-17` in 14 of 15 runs, at about the same time.
- `fixed-3` was no better than `full-17`.
- `speca-0.15-0.3` took 16% longer than `taylor-3` for a similar PSNR.

## Limitations

- The sample is small, and runs that share a prompt are not independent.
- Only one SpeCa setting was tested, so this does not show that SpeCa in general is worse.

Per-run data is in [results.csv](results.csv).
