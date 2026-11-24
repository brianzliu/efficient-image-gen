# SVDQuant: preliminary results

Can a smarter 4-bit format keep FLUX.2 Klein images closer to the full-precision model?

**Setup:** 5 prompts × 3 seeds, 512×512, 50 steps, M1 Pro with 16 GB. Run with `scripts/evaluate-klein.py`.

## Weight formats

The transformer's weights can be stored with fewer bits to save memory. Weights are split into groups of 64; each group stores a scale and an offset, and each weight becomes a small integer (`weight ≈ integer × scale + offset`).

| Name | What it is | Bits per weight | Size (GB) |
|---|---|---:|---:|
| `bf16` | Full precision. The reference. | 16 | 7.75 |
| `q8` | 8-bit integers (256 levels per group). | 8 | 4.12 |
| `q4` | 4-bit integers (16 levels per group). The mflux checkpoint and our old default. | 4 | 2.18 |
| `svdq-32` | `q4` plus SVDQuant fixes, rank 32. | ~4 | 2.28 |
| `svdq-128` | `q4` plus SVDQuant fixes, rank 128. | ~4 | 2.59 |

**SVDQuant fixes** (`feature_reuse/svdquant.py`):

1. **Smoothing:** a few input channels are much larger than the rest. Their scale is moved into the weights, so the 4-bit values are spread more evenly.
2. **Low-rank branch:** the largest part of each weight matrix (its top 32 or 128 SVD directions) is kept in 16 bits, and only the rest is stored in 4 bits.

Both were tuned on 4 separate calibration prompts. Only the weights are 4-bit. The original SVDQuant also uses 4-bit activations for speed, which MLX can't do, so this tests quality only.

## Results

Quality is measured against the `bf16` image for the same prompt and seed. **LPIPS** is perceptual difference (0 = same, lower is better).

| Format | Time (s) | Peak memory (GB) | LPIPS ↓ | ImageReward ↑ | Closer to bf16 than `q4` |
|---|---:|---:|---:|---:|---:|
| `bf16` | 306 | 10.59 | 0 | 1.47 | n/a |
| `q4` | 340 | 5.06 | 0.310 | 1.45 | n/a |
| `svdq-32` | 345 | 5.23 | 0.287 | 1.34 | 9/15 |
| `svdq-128` | 358 | 5.53 | 0.264 | 1.35 | 10/15 |
| `q8` | 340 | 6.96 | 0.079 | 1.42 | 15/15 |

Means over 15 images. Time is denoising only. Full metrics (SSIM, PSNR, CLIP) are in [svdquant.csv](svdquant.csv).

## Takeaways

- **SVDQuant helped only a little.** It moved images slightly closer to `bf16` (on 9–10 of 15 images) but scored lower on ImageReward.
- **`q8` is the clear quality pick**: nearly the same as `bf16`, for 1.9 GB more than `q4`.
- **Quantization saves memory, not time.** `bf16` was 10% faster than `q4` on M1.
- `q4` can change what's in the image: on the books prompt it drew four books instead of three.

## Limitations

- 5 prompts × 3 seeds, one machine.
- 4-bit activations, the main point of SVDQuant, were not tested.
