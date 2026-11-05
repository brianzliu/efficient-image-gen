# Findings and recommended next experiment

Taylor reuse is the most promising starting point for preserving the full-generation image at a lower cost in this pilot. This recommendation concerns five development prompts at seed 42 on one M1 Pro. It does not establish a generally optimal configuration or statistically equivalent image quality.

## What the ablations show

**Forecasting helps at approximately the same compute budget.** `full-17`, `fixed-3`, and `taylor-3` each use 17 full steps. Their mean warm generation times were 118.0, 119.7, and 118.4 seconds. Taylor's mean LPIPS distance to the matching full-50 image was 0.0484, versus 0.1359 for reduced-step sampling and 0.1041 for fixed reuse. Its ImageReward was 1.438, compared with 1.246 and 1.334 respectively. These are observed quality differences at similar measured latency; the small timing differences between the three should not be treated as meaningful speed advantages.

**Taylor reuse is a practical reference for the adaptive algorithm.** Against full-50, Taylor achieved a 2.68x speedup, reducing mean warm generation time from 316.6 to 118.4 seconds. Mean ImageReward was near the full-50 value of 1.453, but there was no predeclared quality tolerance or noninferiority test. Higher reference similarity and a similar learned preference score are insufficient to call this lossless.

**Stricter adaptive verification trades cost for fidelity.** `speca-0.1-0.5` used 19.6 full steps on average and took 141.4 seconds, with LPIPS 0.0347. It preserved reference features more closely than Taylor, but took about 19% longer. Its mean ImageReward of 1.425 did not exceed Taylor's 1.438. This disagreement is one reason to report perceptual similarity and preference proxies separately.

**Threshold and decay affect the operating point.** Increasing the starting threshold from 0.1 to 0.3 at decay 0.5 reduced mean full steps from 19.6 to 13.0 and time from 141.4 to 100.3 seconds, while LPIPS increased from 0.0347 to 0.1127. Removing threshold decay reduced time further to 91.1 seconds with LPIPS 0.1182. The two looser policies had nearly identical mean ImageReward. These results show a tradeoff in this implementation, not a general verdict on the published SpeCa method.

**Removing verification maximizes speed and increases reference drift.** `speca-nocheck` used eight full steps, averaged 58.7 seconds, and reached 5.39x speedup. Its LPIPS was 0.2558 and ImageReward 1.295. Visual inspection shows larger changes in composition and hand/ribbon geometry. Some outputs remain plausible, so large reference distance should not automatically be called poor image quality. Choosing this setting would require stronger task-specific evaluation and human judgment.

## Efficiency findings

The common energy subset contains the sign, market, and hands prompts: three images per configuration, 24 images total. The collector began too late to cover every books run, so all books and fox runs are excluded from energy averages, even when some individual readings are available.

Estimated CPU+GPU energy averaged 7.700 kJ/image for full-50, 2.747 kJ for Taylor, and 3.308 kJ for strict adaptive reuse. Those correspond to observed reductions of 64.3% and 57.0%. The verification-disabled variant used 1.367 kJ/image, an 82.2% reduction, alongside the largest reference drift. These are integrated component-power estimates on this active Mac, including background work. They are not whole-device or isolated model energy, and have not been calibrated against an external meter.

Peak MLX allocation across encoding, denoising, and decoding was 5.20 GB for every configuration. Feature caches added about 37.7 MB for fixed reuse and 75.5 MB for forecasting. This pilot demonstrates reductions in time and estimated component energy, with no measured reduction in the full-pipeline allocation peak. It does not establish a smaller minimum device-memory requirement.

## Validation and interpretation

All 14 unit tests passed, as did four four-step equivalence checks against MFLUX. The new full-50 fox reference was also pixel-identical to the previously generated CLI image. [Validation results](validation.json) independently check trial coverage, prompt identity, image hashes, timing sums, CFG pass counts, speedup calculations, and energy integration/aggregation. Supporting logs are included in this directory.

The collector's final JSON was recovered from its flushed JSONL after the session interruption. No readings were synthesized or extended into uncovered windows. The recovered trace retains the tool checksum and recovery flag.

All 40 images were scored with LPIPS, SSIM, PSNR, CLIP alignment, and ImageReward. GenEval, human preference, and FID were not measured. The dataset is too small for a persuasive FID result. Automated scores cannot establish correct counting, anatomy, or exact lettering. The image grids support qualitative inspection but are not a blinded human study.

One seed and one repeat provide no estimate of cross-seed robustness or repeat-to-repeat timing variation. Configuration order rotates, but the machine was not isolated and had substantial system-wide swap usage. Small latency and energy differences remain uncertain. The first prompt also overlapped evaluator dependency preparation before scoring; the fully matched energy subset was collected later. All generation completed before CPU quality scoring began.

Alan's separate [three-seed preliminary benchmark](../../results/summary.md) measures denoising after unloading the text encoder. Its timing and memory scope differs from this resident-weight, full-generation evaluator. Keep the datasets separate rather than pooling their latency or memory figures.

## Next experiment

Use Taylor-3 as the fixed-cost reference when developing the adaptive policy. Evaluate additional seeds and held-out prompts, repeat timings in a controlled session, and choose a quality tolerance before selecting thresholds. Add a proper compositional benchmark and blinded human comparisons. Measure component energy over the complete suite from startup, and validate a subset against a physical power meter if absolute energy claims matter. Compare against the distilled Klein model before deciding which approach makes the most useful on-device product.
