# Taylor sweep, energy and low memory: preliminary results

Three follow-ups to [combined.md](../combined.md): how often to refresh, how much energy each config uses, and whether streaming still works when memory is short.

**Setup:** 5 prompts × 3 seeds, 512×512, M1 Pro with 16 GB. Run with `scripts/evaluate-klein.py`. Quality is measured against `full-50+bf16`. **LPIPS** is perceptual difference (0 = same, lower is better).

## Refresh interval and energy

`taylor-N` runs the full model every N-th step and predicts the steps in between. All rows except the first two use `bf16+layer`.

| Config | Full steps | Time (s) | Speedup | Energy (J) | LPIPS ↓ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `full-50+bf16` (reference) | 50 | 327 | 1.00× | 5,823 | 0 | 1.47 |
| `full-50+q4` (old default) | 50 | 335 | 0.98× | 7,214 | 0.310 | 1.45 |
| `taylor-2` | 25 | 155 | 2.11× | 3,022 | 0.032 | 1.43 |
| **`taylor-3`** | 17 | 111 | 2.94× | 2,090 | 0.075 | 1.43 |
| `taylor-4` | 13 | 88 | 3.70× | 1,594 | 0.101 | 1.42 |
| `taylor-5` | 10 | 67 | 4.86× | 1,263 | 0.136 | 1.43 |
| `speca-0.15-0.3` | 18.8 | 130 | 2.51× | 2,440 | 0.081 | 1.45 |

Means over 15 images. Peak memory was 1.95 GB for every streamed config. Energy is estimated CPU+GPU energy per image, from macmon sampled every 0.5 s. Per-run data is in [sweep.csv](sweep.csv).

- **The interval is a clean speed–quality dial.** Each extra skipped step makes images faster and a little further from `bf16`, while ImageReward stays flat (1.42–1.43).
- **`taylor-3` remains a good default.** `taylor-4` is 26% faster if a bit more difference is fine.
- **SpeCa doesn't pay off.** It was slower than `taylor-3` and no closer to `bf16` (closer on 7 of 15 images).
- **Energy follows time.** `taylor-3+bf16+layer` uses 2.8× less energy than `full-50+bf16` and 3.5× less than the old default.
- **`q4` used 24% more energy than `bf16`** for the same 50 steps, likely because unpacking 4-bit weights adds work.

## Low memory

With `+layer`, weights are read again for every model pass. These runs checked whether that slows down when the weights can't stay in the OS file cache. Each config ran on 5 images twice: normally, and with 7 GB held by `scripts/hold-memory.py` and the file cache emptied before each image (`--purge`).

| Config | Normal (s) | Low memory (s) | Read from disk, normal (GB/image) | Read from disk, low memory (GB/image) |
|---|---:|---:|---:|---:|
| `taylor-3+bf16+layer` | 147 | 139 | 254 | 256 |
| `taylor-3+q8+layer` | 143 | 164 | 4 | 138 |
| `taylor-3+q4+layer` | 140 | 162 | 2 | 14 |

Disk reads are macOS page-ins over each whole run divided by 5, so they include model loading. Data is in [low-memory.csv](low-memory.csv).

- **`bf16` weights never fit in the file cache, even normally.** Every pass read the 7.75 GB file from the SSD, so all `bf16+layer` timings, including those above, already include SSD reads.
- **The SSD keeps up.** `bf16` was no slower with low memory.
- `q4` and `q8` normally stay cached. They were about 15% slower with low memory. `q4` read little from disk, so part of that is likely the memory pressure itself rather than disk reads.

## Limitations

- The low-memory session ran about 30% slower than the sweep for every config, including the normal runs, and a recheck afterwards was still slow. Compare its rows only with each other.
- Energy is a CPU+GPU estimate including background activity, not a wall-socket measurement.
- 5 prompts × 3 seeds (5 × 1 for low memory), one machine.
