# Efficient on-device image generation

DSC 180 Capstone · Alex, Alan, Brian, and Edward

## Project overview

We want to develop an acceleration algorithm for image generation that makes more capable, computationally expensive models practical on Apple Silicon Macs and, eventually, iPhones. Our focus is inference: reducing the work needed to generate an image from a pretrained model while retaining useful image quality and staying within the device's memory budget.

Our inspiration is [DFlash](https://arxiv.org/abs/2602.06036), which uses a lightweight block diffusion drafter to propose several tokens in parallel for an autoregressive LLM to verify. We want to investigate whether a similar strategy of predicting work cheaply and checking it selectively can accelerate image generation on Apple hardware. DFlash's diffusion drafter and its autoregressive target have different roles. Using a diffusion drafter does not require the target image model to be a diffusion model.

By the end of Quarter 1, we aim to deliver a usable Mac MVP with a reproducible baseline, an accelerated generation mode, and measurements of latency, memory, and image quality. iPhone deployment is a stretch goal after we establish a benefit on Mac. Neither device support nor a speedup is demonstrated yet.

## Running Klein Base 4B on Mac

The local test uses the community [MFLUX Q4 conversion of FLUX.2 Klein Base 4B](https://huggingface.co/mflux-community/flux-2-klein-base-4b-mflux-q4), pinned to revision `f69aa0b43285d0db24695ba68faabbdb1a0ad016`. This is the undistilled Base model with approximately 4.6 GB of quantized weights, not the four-step distilled Klein model. Download size is not peak runtime memory.

Install on an Apple Silicon Mac with `uv` available:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt
.venv/bin/python scripts/download-klein.py
bash scripts/run-klein.sh
```

Edit `prompts/klein-test.txt` to change the prompt. The script defaults to 512 × 512, 50 steps, guidance 3.5, seed 42, FP16 compute, and low-RAM mode. It saves the image, generation metadata, and timing log under `outputs/`. For a quick execution check, use `bash scripts/run-klein.sh --steps 4`; a four-step Base run is not a quality benchmark. Additional CLI flags override the defaults, for example `--steps 28 --output outputs/klein-28.png`.

The environment, model downloads, caches, and generated outputs are ignored by Git. The first generation can include initialization and compilation costs. The CLI timing includes model loading and should not be reported as warm inference latency. Run from a local terminal with Metal GPU access; restricted sandboxes may report that no Metal device is available.

Initial execution check on an M1 Pro with 16 GB unified memory and macOS 27.0.1 succeeded using the defaults above and MFLUX 0.24.0. One run took 369 seconds end to end, reported 3.84 GB peak MLX memory, and produced a valid image matching the fox-and-mug prompt. macOS `time` separately reported a 5.32 GB peak process memory footprint. These memory measures have different scopes. This single run confirms execution, not a comparative performance or quality result.

## Goals and research question

### Evaluating acceleration

The evaluator runs all eight configured ablations at 512 × 512: `full-50`, `full-17`, `fixed-3`, `taylor-3`, `speca-0.1-0.5`, `speca-0.3-0.5`, `speca-0.3-1.0`, and `speca-nocheck`. The last disables verification while retaining adaptive policy warmup and the maximum reuse interval. Fixed and Taylor interval three each use 17 full steps; adaptive methods choose their full-step counts. These are approximate feature-reuse methods, not distribution-preserving speculative sampling.

```sh
uv pip install --python .venv/bin/python -r requirements-eval.txt
.venv/bin/python scripts/check-klein-equivalence.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/evaluate-klein.py --out outputs/evaluation
uv venv --python 3.11 .venv-quality
uv pip install --python .venv-quality/bin/python -r requirements-quality.txt
.venv-quality/bin/python scripts/score-evaluation.py --out outputs/evaluation --image-reward
.venv/bin/python scripts/report-evaluation.py --out outputs/evaluation
```

Generation defaults to all five prompts and seed 42, with one run per pair/configuration. Increase coverage with `--seeds 42 43 44 --repeats 3` in a new output directory. The runner rejects changed experiment identities when resuming. It warms full/prediction/verification paths before measurement and rotates configuration order across prompts. Prompt encoding is performed for each trial with resident model weights. Total latency includes encoding, denoising, and image decoding, excluding model load, PNG writing, and scoring. Full-pipeline peak MLX allocation is separate from process or whole-system memory. Full transformer passes and verification-block passes count both classifier-free guidance branches. Rejected forecasts incur both their check and a subsequent full pass.

Quality scoring uses a separate Python 3.11 environment because ImageReward's older Transformers and Diffusers APIs conflict with MFLUX's newer dependencies. It runs on CPU after generation completes. `requirements-quality.txt` locks that environment; `.venv-quality/` is excluded from Git.

Offline scoring reports LPIPS-Alex, SSIM, PSNR, CLIP alignment, and optionally ImageReward. PSNR is null for identical images, with `identical_pixels=true`, to avoid invalid infinite JSON numbers. Similarity uses the matching full-50 output as reference. CLIP uses `openai/clip-vit-base-patch32` with its revision recorded; it is not a compositional correctness test. GenEval and blinded human preference require separate evaluators/raters and are not replaced by CLIP or ImageReward. FID is intentionally omitted for this small prompt suite.

Energy is null unless independently recorded telemetry is supplied during scoring with `--power-samples file.json`. The file must contain `scope` and `samples`, where samples are ordered `[Unix timestamp seconds, power watts]` pairs. Integration requires coverage of the recorded generation windows and rejects gaps over two seconds. State the measured components; CPU/GPU estimates are not whole-device energy. No energy is inferred from latency or compute counts. Current sudo authentication prevents collecting `powermetrics` automatically in the agent session.

Our research question is: **Can an adaptive prediction-and-verification algorithm reduce image generation latency on Apple Silicon enough to justify its extra memory and verification cost?**

- Develop and evaluate an acceleration algorithm, rather than only package an existing generator.
- Reduce end-to-end generation time against the same model and runtime without our algorithm.
- Measure the quality cost explicitly, or establish the assumptions needed to preserve the target sampling distribution.
- Keep the target model, cached features, and any drafter within a measured device memory budget.
- Provide a local prompt-to-image demo and reproducible benchmark settings.

Acceleration and compression solve different problems. Skipping computation does not by itself make oversized model weights fit in memory. Quantization may be needed alongside our algorithm, and its effects must be measured separately.

## Choosing diffusion or autoregressive models

**Proposed direction: start with a compact latent diffusion model for the Mac MVP.** This is a recommendation to validate with an early feasibility experiment, not a completed team decision. Apple provides an existing [Stable Diffusion conversion and Swift deployment path](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon), which reduces deployment uncertainty. Autoregressive image generation is the stronger conceptual match if our priority becomes lossless DFlash-style token speculation.

| Consideration | Diffusion or flow-based target | Autoregressive image target |
| --- | --- | --- |
| Repeated work | Repeated denoiser or velocity-network evaluations over an image latent | Sequential predictions of image tokens, depending on the model's factorization |
| Acceleration opportunity | Predict or reuse intermediate features, skip selected computation, or speculate over sampling transitions | Propose token blocks and verify them with the target in parallel |
| Relationship to DFlash | Similar motivation, but continuous-state verification needs its own derivation | Direct connection to token drafting and verification |
| Quality guarantee | Feature caching is generally approximate; exact speculative diffusion exists under specified sampler assumptions | Correct speculative acceptance can preserve the target token distribution |
| Memory concerns | Weights, latent activations, and feature caches | Weights, attention/KV caches, and any separate drafter |
| First deployment experiment | Start from an Apple-supported Stable Diffusion path and check whether the runtime exposes the needed computation | First establish a suitable open model, tokenizer, and efficient Apple implementation |

There is no universal winner in efficiency or accuracy. Compare specific checkpoints at matched resolution, prompt set, quality requirements, and device settings. Diffusion processes many spatial locations in parallel within a step, but its timesteps remain dependent. Autoregressive models can verify multiple proposed tokens in parallel. Modern few-step diffusion models also leave less redundant work to remove than a long sampling schedule.

For this project, the deployment evidence favors diffusion as the first experiment. That does not establish that diffusion is inherently faster. If preserving the target distribution is a hard requirement, prioritize an autoregressive target with SJD/SCD baselines, or restrict diffusion experiments to the exact speculative sampler's supported setting.

## Existing research and implications

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) conditions a small parallel drafter on target-model features. It motivates a future learned image-token drafter, but transferring it to images would require training and evaluation, not just changing the output vocabulary.
- [Accelerated Diffusion Models via Speculative Sampling](https://arxiv.org/abs/2501.05370) extends speculation to continuous Markov chains and includes a training-free drafting strategy. It reports roughly halving function evaluations while sampling exactly from the target. This is evidence that exact diffusion speculation is possible. It does not establish compatibility with every deterministic scheduler or rectified-flow sampler, and fewer evaluations do not guarantee proportional wall-clock savings.
- [Speculative Jacobi Decoding, SJD](https://arxiv.org/abs/2410.01699) predicts multiple image tokens through parallel refinement and uses probabilistic acceptance. **Yes, there is a target model:** the original autoregressive image generator. The method is training-free and does not require a separately trained small drafter.
- [Speculative Coupled Decoding, SCD](https://arxiv.org/abs/2510.24211) stabilizes successive SJD drafts through coupled sampling while preserving the target distribution. It reports up to 4.2x image-generation speedup in its experiments. Reproducing this method would be a baseline; an adaptive device-aware policy would need to show a further benefit to count as our contribution.
- [SpeCa](https://arxiv.org/abs/2509.11628) forecasts diffusion-transformer features and uses lightweight verification to decide when to refresh them. It motivates a training-free first prototype, but its quality-speed tradeoff is approximate. Its transformer-specific design should not be assumed to transfer unchanged to a U-Net.
- [SVDQuant](https://arxiv.org/abs/2411.05007) combines 4-bit quantization with a high-precision low-rank branch. It reports 3.5x lower memory usage for 12B FLUX.1 and 3x speedup over a 4-bit weight-only baseline on a laptop RTX 4090. Its Nunchaku kernel optimizations are hardware-dependent; these results are not Apple Silicon measurements.

These papers use different models, hardware, baselines, and quality measures. Their reported speedups are evidence of opportunity, not a ranking or a forecast of our results.

## Proposed methodology

1. **Establish feasibility.** Choose a checkpoint that fits the available Mac, confirm its license and runtime support, and profile text encoding, denoising, image decoding, and peak memory. A compact Stable Diffusion model is a deployment starting point; use a compatible diffusion transformer if reproducing SpeCa specifically. Verify that intermediate features and refresh decisions can be accessed without expensive device transfers or recompilation.
2. **Reproduce a baseline.** Compare ordinary sampling, a competitive reduced-step schedule, and a simple fixed feature-reuse policy where supported. Keep model weights, resolution, precision, and runtime fixed when isolating the acceleration algorithm.
3. **Develop our algorithm.** Test an adaptive feature-prediction and refresh policy. Use recent feature changes, denoising stage, and measured prediction error to choose how long to reuse predictions and when to recompute. Treat this as an approximate algorithm unless we implement a proven exact acceptance rule. Existing adaptive methods make novelty an open question; our intended contribution is a policy constrained by measured Apple-device latency and memory.
4. **Evaluate the actual savings.** Include cache management, prediction, verification, fallback, and synchronization in wall-clock timing. Compare fixed and adaptive policies through ablations. Record full-network evaluations, refresh frequency, and cache memory to explain results.
5. **Package the MVP.** Provide local prompt input, baseline and accelerated modes, saved outputs, and benchmark logs. Investigate iPhone conversion only after the Mac implementation shows a useful benefit and fits a realistic mobile memory budget.

At the first feasibility checkpoint, continue with diffusion only if the baseline runs and the runtime permits cheap prediction and refresh. Otherwise, evaluate an open autoregressive image model with training-free SJD/SCD before committing to a learned DFlash-style drafter. Avoid building both full pipelines in Quarter 1.

## Evaluation and MVP acceptance

Use a fixed prompt suite covering objects, spatial relationships, text rendering, and difficult compositions, with multiple seeds per prompt. Report warm generation latency separately from model loading and compilation, include repeated-run variability, and record the device, OS, runtime, precision, resolution, sampler, and step count.

Measure end-to-end latency and peak memory alongside prompt alignment, visual defects, and blinded baseline-versus-accelerated preference checks. Paired seeds help comparisons but do not prove distribution preservation. Use FID only with an appropriate reference dataset and enough samples; a few example images cannot support a reliable FID claim. Repeated generation should also reveal thermal slowdowns. Energy measurements are optional if reliable device instrumentation is available.

The MVP should run locally on one documented Mac, expose both modes, and produce reproducible measurements. Our proposed performance goal is a repeatable latency reduction with no material quality loss under a quality tolerance chosen before tuning. We have not yet set a numerical speedup target or demonstrated an improvement. A negative result should report where verification or memory overhead removes the expected gain.

## Progress and team contributions

The team has completed an initial literature review and identified candidate acceleration methods. The repository now includes a Mac setup and test prompt for an existing quantized Klein Base 4B implementation. Our own acceleration algorithm and comparative device benchmarks remain to be implemented.

| Team member | Literature review completed |
| --- | --- |
| Alex | DDPM and the Qwen-Image-2.0 technical report |
| Alan | SVDQuant and SpeCa |
| Brian | Speculative diffusion sampling, SJD, and SCD |
| Edward | Rectified Flow and MiniMax H3 |

These are literature-review contributions from the Week 5 check-in. Implementation responsibilities remain to be assigned.

## Next steps

- First, inventory available Mac/iPhone hardware and choose a model, runtime, prompt suite, and quality tolerance.
- In the next phase, profile the Mac baseline, test access to intermediate features, and make the diffusion-versus-autoregressive decision using the feasibility checkpoint above.
- Then reproduce an existing acceleration baseline and implement the adaptive policy with latency, quality, and memory ablations.
- By the end of Quarter 1, deliver the usable Mac MVP and a report of measured results and limitations. Attempt iPhone deployment if the Mac results and remaining time support it.
