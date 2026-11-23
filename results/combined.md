# Combined recipe: preliminary results

This is a preliminary experiment combining feature reuse, weight streaming and quantization on FLUX.2 Klein.

**Setup:** 5 prompts, 3 seeds (42, 43, 44), 512×512, guidance 3.5, on an M1 Pro with 16 GB of memory. Generated, scored and validated with `scripts/evaluate-klein.py`.

**Reference:** `full-50` with BF16 weights. Quality is measured as distance from that image for the same prompt and seed.

## Configs

| Part | Choice |
|---|---|
| Feature reuse | `taylor-3`: 17 full steps out of 50, with Taylor prediction in between. TeaCache was worse, see [teacache.md](teacache.md). |
| Streaming | `+layer`: one block loaded at a time, see [streaming.md](streaming.md). |
| Weights | `q4`, `svdq-128`, `q8` or `bf16`, see [svdquant.md](svdquant.md). |

## Results

| Config | Time (s) | Speedup | Peak memory (GB) | LPIPS ↓ | SSIM ↑ | PSNR ↑ | CLIP ↑ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full-50` (BF16) | 338 | 1.00× | 10.77 | 0 | 1 | n/a | 35.56 | 1.47 |
| `full-50+q4` | 359 | 0.94× | 5.20 | 0.310 | 0.669 | 15.5 | 37.06 | 1.45 |
| `taylor-3+q4+layer` | 123 | 2.74× | 0.96 | 0.309 | 0.670 | 15.5 | 36.30 | 1.44 |
| `taylor-3+svdq-128+layer` | 129 | 2.62× | 1.22 | 0.270 | 0.713 | 16.3 | 35.37 | 1.36 |
| `taylor-3+q8+layer` | 131 | 2.59× | 1.30 | 0.104 | 0.873 | 22.1 | 36.08 | 1.41 |
| **`taylor-3+bf16+layer`** | **117** | **2.90×** | **1.95** | **0.075** | **0.910** | **23.9** | 36.04 | 1.43 |

Times are mean total generation time (encode, denoise, decode) over 15 images. Peak memory is the highest peak MLX memory over the 15 images.

Wins of `taylor-3+bf16+layer` on the same prompt and seed:

| Against | LPIPS | SSIM | CLIP | ImageReward |
|---|---:|---:|---:|---:|
| `taylor-3+q4+layer` | 15/15 | 15/15 | 7/15 | 6/15 |
| `full-50+q4` | 15/15 | 15/15 | 5/15 | 7/15 |

## Takeaways

- **Best recipe: `taylor-3+bf16+layer`.** Compared with `full-50+q4`, the current default, it is 3.1× faster, uses 62% less peak memory (1.95 vs 5.20 GB) and is much closer to the BF16 image (LPIPS 0.075 vs 0.310).
- Streaming makes weight size matter little for memory, so the best weights are the unquantized ones. Quantization only made images further from BF16, and on M1 it was not faster.
- `taylor-3` with BF16 weights stayed close to the BF16 reference (LPIPS 0.075). Most of Q4's distance comes from quantization, not from feature reuse: `taylor-3+q4+layer` and `full-50+q4` are almost equally far from BF16.
- If disk or file-cache space is tight, `taylor-3+q4+layer` is the smallest option: 0.96 GB peak and 2.2 GB on disk.

## Limitations

- Peak MLX memory does not count the OS file cache. With `+layer`, each image reads the 7.75 GB BF16 transformer 34 times. Here it came from the file cache. On a machine with less free memory it would come from the SSD, which could be slower.
- CLIP varies by about 1.5 between seeds of the same prompt, so it does not separate the configs. ImageReward is steadier (about 0.05). Every config except `svdq-128` (1.36) is within 0.07 of BF16 (1.47).
- Only 512×512, 5 prompts × 3 seeds, one machine.

Per-run data is in [combined.csv](combined.csv).
