# SVDQuant: preliminary results

This is a preliminary experiment testing SVDQuant-style 4-bit weights on FLUX.2 Klein.

**Setup:** 5 prompts, 3 seeds (42, 43, 44), 512×512, guidance 3.5, 50 steps, on an M1 Pro with 16 GB of memory. Generated, scored and validated with `scripts/evaluate-klein.py`.

**Reference:** BF16 weights, unlike the other notes, which used Q4. Quality is measured as distance from the BF16 image for the same prompt and seed.

## Method

Each linear layer in the transformer gets W·x ≈ Q(R)·x′ + up·(down·x′), where:

1. **Smoothing:** x′ = x / λ and W′ = W·λ. λ moves large activation channels into the weights. The strength α (none, 0.25, 0.5 or 0.75) is chosen per layer by output error on calibration inputs.
2. **Low rank:** up·down is the top-r SVD of W′. It is kept in 16 bits.
3. **Residual:** R = W′ − up·down, quantized to 4 bits like Q4 (affine, groups of 64).

Calibration used 4 separate prompts × 25 steps. Only the weights are 4-bit, and activations stay 16-bit (W4A16). The paper's speedup needs 4-bit activations and custom kernels, which MLX does not have, so this tests quality only.

See `feature_reuse/svdquant.py` and `scripts/build-svdquant.py`.

## Configs

| Config | Description | Size (GB) |
|---|---|---:|
| `full-50` | BF16, the reference | 7.75 |
| `+q4` | mflux Q4 (all linears, 4-bit) | 2.18 |
| `+svdq-32` | SVDQuant, rank 32 | 2.28 |
| `+svdq-128` | SVDQuant, rank 128 | 2.59 |
| `+q8` | 8-bit, same scheme as Q4 | 4.12 |

## Results

| Config | Denoise (s) | Peak memory (GB) | LPIPS ↓ | SSIM ↑ | PSNR ↑ | CLIP ↑ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full-50` | 306 | 10.59 | 0 | 1 | n/a | 35.56 | 1.47 |
| `full-50+q4` | 340 | 5.06 | 0.310 | 0.669 | 15.5 | 37.07 | 1.45 |
| `full-50+svdq-32` | 345 | 5.23 | 0.287 | 0.695 | 16.0 | 35.25 | 1.34 |
| `full-50+svdq-128` | 358 | 5.53 | 0.264 | 0.721 | 16.5 | 35.33 | 1.35 |
| `full-50+q8` | 340 | 6.96 | 0.079 | 0.903 | 23.9 | 35.70 | 1.42 |

Means over 15 images. Wins against Q4 on the same prompt and seed:

| Config | LPIPS | SSIM | PSNR | CLIP | ImageReward |
|---|---:|---:|---:|---:|---:|
| `svdq-32` | 9/15 | 9/15 | 9/15 | 1/15 | 4/15 |
| `svdq-128` | 10/15 | 10/15 | 11/15 | 4/15 | 2/15 |
| `q8` | 15/15 | 15/15 | 15/15 | 6/15 | 4/15 |

Mean output error per layer on calibration inputs: Q4 6.2%, svdq-32 5.4%, svdq-128 4.8%, Q8 0.5%. This is in-sample, because the same inputs chose α.

## Takeaways

- SVDQuant moved images a little closer to BF16 (LPIPS −7% at rank 32, −15% at rank 128), but the gain is small and inconsistent (9–11 of 15 images).
- It lowered CLIP and ImageReward compared with Q4. Q4 itself scores about the same as BF16 on both.
- Q8 is far closer to BF16 than any 4-bit variant, for 1.9 GB more memory.
- BF16 was 10% faster than Q4 at 512×512. On M1, quantization saves memory, not time. The low-rank branch adds 2–5% time.
- With `+layer` streaming the transformer weights are mostly unloaded, so their size matters much less (see [streaming.md](streaming.md)).

## Limitations

- Weights only (W4A16). Activation quantization, the main point of SVDQuant, is not tested.
- 5 prompts × 3 seeds, with a single 4-bit scheme (affine, groups of 64).
- 4-bit changes each image's composition, so pixel metrics against BF16 are noisy. On the books prompt Q4 drew four books instead of three, and svdq-128 kept three.
- `report-evaluation.py` still names the Q4 checkpoint in its setup line. The transformer actually used is in `manifest.json` (`transformer`).

Per-run data is in [svdquant.csv](svdquant.csv).
