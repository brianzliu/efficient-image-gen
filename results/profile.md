# Klein profile: where time and memory go

This is a preliminary profile of 4-bit FLUX.2 Klein, used to decide what to target for efficiency.

**Setup:** one prompt, seed 42, 512×512, guidance 3.5 (2 model passes per step), on an M1 Pro with 16 GB of memory. Stage times come from runs without instrumentation. The breakdowns force evaluation at each boundary, which added 2% to 4% to run time.

## Pipeline

| Stage | Time (s) | Share | Peak memory (GB) |
|---|---:|---:|---:|
| Model load | 0.9 | 0.3% | 0.0 |
| Prompt encoding | 2.6 | 0.8% | 3.0 |
| Denoising, 50 steps | 317 | 98.6% | 3.2 |
| VAE decoding | 0.9 | 0.3% | 2.8 |

The 50-step time is scaled from 10-step runs. The text encoder is freed before denoising.

## Denoising time

One step takes about 6.3 s. Every block costs about the same, around 128 ms per pass, so no single block stands out.

| Part | Share of denoising |
|---|---:|
| 20 single blocks | 79.7% |
| 5 double blocks | 19.7% |
| Everything else | 0.6% |

| Operation | Share of denoising | Multiply-accumulates per pass |
|---|---:|---:|
| Linear layers | 85.0% | 4.7 T |
| Attention dot product | 7.1% | 0.4 T |
| Rotary position embedding | 3.2% | not counted |
| Everything else | 4.7% | not counted |

Two layers in the single blocks take about two-thirds of denoising time: `to_qkv_mlp_proj` about 47% and `to_out` about 20%. These two shares are estimated from timing each layer alone.

## Memory

| Part of the denoising peak | Size (GB) | Scales with tokens |
|---|---:|---|
| Transformer weights and inputs | 2.20 | No |
| Working memory of one block | 0.48 | Yes, about 0.31 MB per token |
| Temporary memory held for GPU batches | 0.53 | Partly |
| Total | 3.21 | |

Once the VAE decoder weights are in memory, later denoising peaks are about 0.1 GB higher.

`to_qkv_mlp_proj` and `to_out` hold 63% of the transformer weights (1.38 GB). The text encoder stores another 2.26 GB, but it is not in memory during denoising.

The temporary memory can be removed without changing the image. MLX batches GPU work, and each setting below changes how much temporary data is alive at once:

| Setting | Step time (s) | Denoising peak (GB) |
|---|---:|---:|
| Default | 6.19 | 3.21 |
| Evaluate after each model pass | 6.22 | 3.42 |
| Evaluate after each block | 6.30 | 2.67 |
| Evaluate after each linear layer | 6.40 | 2.62 |
| `MLX_MAX_MB_PER_BUFFER=20` | 6.20 | 2.83 |
| `MLX_MAX_MB_PER_BUFFER=5` | 6.24 | 2.65 |

Over two 10-step runs of each, `MLX_MAX_MB_PER_BUFFER=5` took 6.26 and 6.42 s per step, against 6.31 and 6.53 s by default, and the decoded images were identical. `MLX_MAX_MB_PER_BUFFER=1` reached the same 2.65 GB. With the setting on, VAE decoding (2.76 GB) peaks above denoising. Its effect on prompt encoding was not measured.

Dropping the unconditional pass would not lower the peak, because one pass alone peaks at 3.42 GB.

## 4-bit quantization

| Layer | 4-bit call (ms) | float16 call (ms) | Slowdown |
|---|---:|---:|---:|
| `to_qkv_mlp_proj` | 78.2 | 69.0 | 13% |
| `to_out` | 33.7 | 31.9 | 6% |

The transformer weights would take 7.75 GB as float16 instead of 2.18 GB. In these layers, 4-bit calls used no more temporary memory than float16 calls.

## Scaling

Each step processes 1024 image tokens and 512 text tokens. Real prompts use only 37 to 57 of the text tokens, and the rest are padding that the model still processes.

| Text tokens kept | Step time (s) | Time saved |
|---:|---:|---:|
| 512 (current) | 6.19 | n/a |
| 256 | 5.12 | 17% |
| 128 | 4.68 | 24% |
| 64 | 4.35 | 30% |
| 32 | 4.16 | 33% |

| Resolution | Total tokens | Step time (s) | Peak memory (GB) |
|---:|---:|---:|---:|
| 256 | 768 | 3.10 | 3.24 |
| 384 | 1088 | 4.35 | 3.28 |
| 512 | 1536 | 6.33 | 3.31 |
| 768 | 2816 | 11.99 | 3.58 |

Step time grows about linearly with tokens, at about 4 ms per token. Peak memory grows slowly, because weights dominate it. These peaks include the VAE decoder weights.

## Targets

| Priority | Target | Why | Status |
|---:|---|---|---|
| 1 | Unconditional pass | It is half of all model passes | Not tested |
| 2 | Padding tokens | Keeping 64 saves 30% of step time | Image quality not tested |
| 3 | Full model passes | Feature reuse already skips these, and every block costs about the same | Tested, see [summary.md](summary.md) |
| 4 | `to_qkv_mlp_proj` and `to_out` | They take about 67% of time and 63% of weights | Not tested |
| 5 | `MLX_MAX_MB_PER_BUFFER=5` | Saves 0.55 GB of denoising peak with no measurable slowdown | Tested on one prompt, image unchanged |

Poor targets: prompt encoding, VAE decoding, and the attention dot product each take too little time. Removing 4-bit quantization would make linear layers only slightly faster and use far more memory.

## Limitations

- Time and memory were measured on one prompt and seed, on one machine.
- Most time savings above are measured or estimated without checking image quality.
- Step times varied by a few percent between identical runs, so small time differences are not meaningful.

Raw data is in [profile.json](profile.json) and [memory.json](memory.json). To reproduce it, run `scripts/profile-klein.py` and `scripts/profile-memory.py`.
