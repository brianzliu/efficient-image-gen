# Streaming: preliminary results

This is a preliminary experiment testing weight streaming on 4-bit FLUX.2 Klein.

**Setup:** 5 prompts, 1 seed, 512×512, guidance 3.5, on an M1 Pro with 16 GB of memory. Generated, scored and validated with `scripts/evaluate-klein.py`.

## Configs

| Suffix | Description |
|---|---|
| none | All weights stay loaded (text encoder, transformer, VAE). |
| `+stage` | Each part is loaded from disk for its stage, then freed. |
| `+layer` | Blocks are loaded one at a time; the next block is read while the current one computes. |

Each suffix was applied to `full-50` and `taylor-3`. See `feature_reuse/streaming.py`.

## Results

| Config | Median time (s) | Peak memory (GB) | Same image as no streaming |
|---|---:|---:|---:|
| `full-50` | 315 | 5.20 | n/a |
| `full-50+stage` | 318 | 3.23 | 5/5 |
| `full-50+layer` | 318 | 0.88 | 5/5 |
| `taylor-3` | 110 | 5.20 | n/a |
| `taylor-3+stage` | 110 | 3.27 | 5/5 |
| `taylor-3+layer` | 110 | 0.96 | 5/5 |

Peak memory is peak MLX memory during generation, including loaded weights. Times are medians because a few runs in one time window were slow for every config.

## Takeaways

- `+layer` cut peak memory by 82%, from 5.2 GB to under 1 GB, with no measurable slowdown.
- Every image was pixel-identical to the same config without streaming, so quality is unchanged.
- Reading the next block (69–138 MB) is hidden behind computing the current block (about 128 ms).

## Limitations

- The weight files were read from the OS file cache, not directly from the SSD. On a machine short on memory they would come from the SSD, which could be slower.
- Only 5 runs per config.

Per-run data is in [streaming.csv](streaming.csv).
