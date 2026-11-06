# Efficient on-device image generation

DSC 180 Capstone · Alex, Alan, Brian, and Edward

This repository evaluates ways to accelerate FLUX.2 Klein Base 4B image generation on Apple Silicon using feature reuse, forecasting, higher-order samplers, and guidance policies. This guide walks through reproducing our experiments. Project goals, research background, and the original planning document are in [PLAN.md](PLAN.md). AI contributions are documented in [AI_USE.md](AI_USE.md).

## Results and scope

Our [initial ablation pilot](reports/klein-ablation-pilot/README.md) contains 40 images: five prompts, seed 42, one repeat, and eight configurations. The [findings](reports/klein-ablation-pilot/findings.md), raw measurements, quality scores, telemetry, and image grids are checked in alongside the report. The pilot used an M1 Pro with 16 GB unified memory and macOS 27.0.1.

The steps below reproduce the pilot protocol with the current code. They do not guarantee identical timings or pixels on different hardware or software versions. The saved manifest records the original environment and source hash; the report was published in commit `ec3232b`. Sampler and guidance extensions were added afterward and are covered separately below.

Alan's [earlier three-seed study](results/summary.md) uses a different measurement scope. Its reproduction commands are at the end of this guide.

## Accessing and storing the data

No training dataset is needed. The experiment generates images from the five text prompts in [prompts/benchmark.txt](prompts/benchmark.txt). Cloning this repository provides those prompts and the saved pilot results; model weights are downloaded separately in step 1.

| Data | How to access it | Storage location |
| --- | --- | --- |
| Input prompts | Included in the repository | `prompts/benchmark.txt` |
| Klein Q4 weights | Run `scripts/download-klein.py` | `models/klein-base-4b-q4/` |
| LPIPS, CLIP, and ImageReward weights | Downloaded by the quality scorer as needed | Repository `.cache/huggingface/` and `.cache/torch/` by default; some libraries may also use their own caches |
| Original pilot images, measurements, and scores | Included in the repository | `reports/klein-ablation-pilot/` |
| Newly generated images and experiment identity | Created in step 4 | `outputs/evaluation/images/` and `outputs/evaluation/manifest.json` |
| New timings, memory readings, quality scores, and telemetry | Created in steps 3–5 | `outputs/evaluation/` |
| New tables, plots, and portable report | Created in step 6 | `outputs/reproduced-pilot/` |

Initial setup and quality scoring require internet access to download dependencies and model weights. The scorer records the CLIP and ImageReward revisions it resolves in `scoring.json`; unlike the generation checkpoint, these revisions are not fixed in the command line. Check them against the saved pilot metadata when comparing quality scores. Existing `HF_HOME` or `TORCH_HOME` environment variables override the default cache locations.

Keep each experiment's manifest, images, measurements, scores, and power trace together. New runs go under Git-ignored `outputs/`, so copy or archive the complete run and report directories if you need to retain or share them. Do not replace the checked-in initial pilot with a new run.

## 1. Prepare an Apple Silicon Mac

Use a local terminal with Metal GPU access, Git, and `uv` installed. Run all commands from the repository root. The pilot ran on a 16 GB Mac; this is an observed setup, not a measured minimum memory requirement. Allow disk space for approximately 4.6 GB of model weights, two Python environments, scoring-model downloads, and generated images.

### Software dependencies

| Environment or tool | Purpose | Dependency specification |
| --- | --- | --- |
| Apple Silicon macOS with Metal | Local GPU generation | Pilot used macOS 27.0.1 |
| Git and `uv` | Checkout, Python environments, and package installation | Available on your command line |
| Python 3.13.5, `.venv` | Generation, tests, reporting, and validation | `requirements-mac.txt` and `requirements-eval.txt`, including MFLUX 0.24.0 and MLX 0.32.4 |
| Python 3.11, `.venv-quality` | CPU quality scoring | `requirements-quality.txt`, including PyTorch, LPIPS, Transformers, and ImageReward |
| macmon 0.10.0, optional | CPU+GPU power telemetry | Pinned archive and checksum in `scripts/install-macmon.sh` |

The requirements files pin package versions. The commands below use `uv` environments; no conda environment or Docker image is required for this workflow.

```sh
git clone https://github.com/brianzliu/efficient-image-gen.git
cd efficient-image-gen
uv venv --python 3.13.5 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt
uv pip install --python .venv/bin/python -r requirements-eval.txt
.venv/bin/python scripts/download-klein.py
```

If you already have a checkout, start with the environment commands. The downloader pins `mflux-community/flux-2-klein-base-4b-mflux-q4` to revision `f69aa0b43285d0db24695ba68faabbdb1a0ad016` and saves it under `models/klein-base-4b-q4/`. This is the undistilled Base model. Environments, model weights, caches, and `outputs/` are ignored by Git.

Create the separate CPU quality-scoring environment before starting measurements:

```sh
uv venv --python 3.11 .venv-quality
uv pip install --python .venv-quality/bin/python -r requirements-quality.txt
```

ImageReward uses older Transformers and Diffusers APIs that conflict with the generation environment. Keep scoring in `.venv-quality` and run it after generation. Scoring may download additional pretrained models on its first run.

## 2. Check generation and correctness

A short execution check confirms that the model can run:

```sh
bash scripts/run-klein.sh --steps 4
```

This uses `prompts/klein-test.txt` and writes an image, metadata, and a timing log under `outputs/`. Four steps with the Base model are only an execution check. To generate the ordinary 50-step baseline, run `bash scripts/run-klein.sh`.

Run the tests and real-model equivalence checks before the measured suite:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check-klein-equivalence.py
```

The equivalence script checks the custom loop against MFLUX and checks that guidance settings equivalent to plain CFG produce matching outputs. The current suite includes tests added after the pilot; its test count can differ from the saved pilot logs.

## 3. Start optional energy collection

Skip this step if you only need latency, memory, and quality results. Without telemetry, energy scores remain null.

Install the pinned macmon v0.10.0 collector:

```sh
bash scripts/install-macmon.sh
```

In a second terminal, change to this same repository root and run:

```sh
.venv/bin/python scripts/collect-macmon.py \
  --out outputs/evaluation/power.json \
  --stop-file outputs/evaluation/stop-power
```

Leave that terminal running. Start collection before the generation command in step 4. On a fresh run, `outputs/evaluation/stop-power` must not exist. If reusing this directory, remove the old stop file only after confirming that the previous collector has exited. Use a fresh output directory for a separate experiment and change both collector paths to match it.

The collector records CPU+GPU power at 500 ms intervals, plus temperature and system RAM/swap usage. These are component-power estimates including background activity, not whole-device or process-isolated measurements. Our saved pilot has complete matched energy coverage for only three of its five prompts. A new collection started before generation can cover the whole suite.

## 4. Generate the pilot images

In the first terminal, run:

```sh
.venv/bin/python scripts/evaluate-klein.py \
  --out outputs/evaluation \
  --prompts prompts/benchmark.txt \
  --seeds 42 --repeats 1 \
  --configs full-50 full-17 fixed-3 taylor-3 \
    speca-0.1-0.5 speca-0.3-0.5 speca-0.3-1.0 speca-nocheck
```

The five prompts cover a fox and mug, stacked books, a shop sign, a night market, and hands tying a ribbon. Settings are 512 × 512, guidance 3.5, FP16 transformer compute, Euler sampling, and the pinned Q4 checkpoint.

| Configuration | What changes |
| --- | --- |
| `full-50` | 50 full steps, used as the image reference |
| `full-17` | 17 full steps, without feature reuse |
| `fixed-3` | 50 scheduler updates with a full computation every third step |
| `taylor-3` | Same interval, with first-order feature extrapolation |
| `speca-0.1-0.5` | Adaptive verification with starting threshold 0.1 and decay 0.5 |
| `speca-0.3-0.5` | Looser starting threshold 0.3, with decay 0.5 |
| `speca-0.3-1.0` | Starting threshold 0.3, without threshold decay |
| `speca-nocheck` | Verification disabled, retaining warmup and the reuse cap |

These are approximate feature-reuse methods. The SpeCa-named variants use a simplified final-block verification heuristic and are not validated reproductions of the published method.

The runner warms the computation paths, rotates configuration order across prompts, and saves `manifest.json`, `measurements.jsonl`, and 40 PNGs in `outputs/evaluation/`. It skips completed trials when rerun with the same experiment identity. Changes to prompts, settings, source code, or recorded environment require a new output directory.

Allow several hours for the full suite on a laptop. Keep other workloads low and record your hardware and environment when comparing results. Warm generation latency includes prompt encoding, denoising, and decoding with resident weights; it excludes model loading, PNG writing, and scoring. CLI execution-check timings include loading and use a different memory scope.

## 5. Stop collection and score the images

If you started the collector, stop it after generation finishes:

```sh
touch outputs/evaluation/stop-power
```

Wait for the collector in the second terminal to exit and finalize `power.json`. Then score with telemetry:

```sh
.venv-quality/bin/python scripts/score-evaluation.py \
  --out outputs/evaluation --image-reward \
  --power-samples outputs/evaluation/power.json
```

If you skipped energy collection, use this command instead:

```sh
.venv-quality/bin/python scripts/score-evaluation.py \
  --out outputs/evaluation --image-reward
```

Scoring runs on CPU and writes `scores.jsonl` and `scoring.json`. It computes LPIPS, SSIM, PSNR, CLIP alignment, and ImageReward. Similarity metrics compare each image with its matching `full-50` image. They measure reference similarity, and CLIP and ImageReward are automated proxies; none establishes human preference or compositional correctness. GenEval, human evaluation, and FID are not part of this pilot.

Energy integration requires coverage of generation windows and rejects gaps over two seconds. Summaries use only prompt/seed/repeat pairs covered for every configuration. If the collector was interrupted, recover its actual flushed samples before scoring:

```sh
.venv/bin/python scripts/collect-macmon.py \
  --out outputs/evaluation/power.json --recover
```

Recovery does not fill missing coverage. The manifest's initial `energy_status` message refers to powermetrics availability; `scoring.json` records the energy scope actually used.

## 6. Build and validate a new report

Use a separate report directory to preserve the checked-in pilot:

```sh
.venv/bin/python scripts/report-evaluation.py \
  --out outputs/evaluation --report outputs/reproduced-pilot
.venv/bin/python scripts/validate-evaluation.py \
  --report outputs/reproduced-pilot
```

Open `outputs/reproduced-pilot/README.md` or `overview.html` to inspect the results. The report includes `results.csv`, `summary.json`, quality/resource plots, image grids, and copies of raw measurements and scoring metadata. Validation writes `validation.json` after checking coverage, image hashes, computation counters, timing totals, speedup calculations, and energy aggregation where available.

Compare your results with the [saved pilot](reports/klein-ablation-pilot/README.md) and its [interpretation](reports/klein-ablation-pilot/findings.md). The saved pilot's `findings.md`, unit-test logs, equivalence logs, and baseline-validation file are supporting artifacts; the report generator does not recreate those files automatically. One seed and one repeat provide no estimate of cross-seed robustness or timing variation. For broader evaluation, use a new directory with `--seeds 42 43 44 --repeats 3` and additional held-out prompts.

## Sampler and guidance experiments

These extensions are separate from the initial pilot. Configuration names accept one sampler suffix (`+euler`, `+dpmpp2m`, or `+unipc`) and one guidance suffix (`+cfg`, `+interval-HIGH-LOW`, or `+ag-T`). Without suffixes, configurations use Euler and plain CFG. See [PLAN.md](PLAN.md#samplers-and-guidance) for implementation details.

For example, compare samplers while keeping the reference:

```sh
.venv/bin/python scripts/evaluate-klein.py --out outputs/exp-samplers \
  --configs full-50 full-17 full-17+dpmpp2m full-17+unipc
```

Choose guidance thresholds using plain-CFG diagnostics before quality scoring:

```sh
for p in 0 1 2 3 4; do
  for s in 42 43; do
    .venv/bin/python scripts/diagnose-guidance.py --prompt-id "$p" --seed "$s"
  done
done
.venv/bin/python scripts/summarize-guidance.py --thresholds 0.99 0.995 0.998
```

Diagnostics write logs, images, and plots under `outputs/guidance-diagnostic/`. The summary reports guided steps and passes saved, and suggests a fixed interval cutoff matched to each adaptive threshold. The existing five-prompt, two-seed Euler diagnostics give this example:

```sh
.venv/bin/python scripts/evaluate-klein.py --out outputs/exp-guidance \
  --configs full-50 full-50+ag-0.995 full-50+interval-1-0.918 \
    full-50+ag-0.998 full-50+interval-1-0.840
```

Rerun diagnostics if prompts, step count, sampler, or guidance scale change. Score either experiment using step 5 with its own `--out` path. Keep `full-50` in the suite because scoring uses it as the reference. Compare guidance costs using `full_transformer_passes` and `guided_steps` in `measurements.jsonl`. The report generator still uses pilot-specific prose and does not display guidance/pass counters, so review its text before sharing reports for these extensions.

## Earlier three-seed benchmark

To rerun the protocol behind [results/summary.md](results/summary.md), use the generation environment and downloaded checkpoint from step 1:

```sh
.venv/bin/python scripts/benchmark-klein.py --out outputs/benchmark \
  --seeds 42 43 44 \
  --configs full-50 full-17 fixed-3 taylor-3 speca-0.15-0.3
.venv/bin/python scripts/score-benchmark.py --out outputs/benchmark
```

This produces 75 images and `outputs/benchmark/results.csv`; the scorer prints per-configuration means and PSNR. Use a fresh directory for a new run. This benchmark measures denoising after unloading the text encoder. Its timing and memory values should be compared within that study, rather than pooled with the resident-weight, full-generation pilot.
