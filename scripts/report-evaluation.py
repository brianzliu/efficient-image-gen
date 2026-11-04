"""Build a portable Markdown report, raw results, and scientific plots."""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--report", type=Path, default=Path("reports/klein-ablation-pilot"))
    args = parser.parse_args()
    manifest = json.loads((args.out / "manifest.json").read_text())
    rows = list(map(json.loads, (args.out / "measurements.jsonl").read_text().splitlines()))
    scores = {r["key"]: r for r in map(json.loads, (args.out / "scores.jsonl").read_text().splitlines())}
    identity = manifest["identity"]
    expected = len(identity["prompts"])*len(identity["seeds"])*identity["repeats"]*len(identity["configs"])
    if len(rows) != expected or len({r["key"] for r in rows}) != expected or set(scores) != {r["key"] for r in rows}:
        raise ValueError("Incomplete or duplicated experiment")
    for row in rows:
        row.update({k:v for k,v in scores[row["key"]].items() if k != "key"})
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "images").mkdir(exist_ok=True)
    for name in ["manifest.json", "scoring.json", "measurements.jsonl", "scores.jsonl"]:
        shutil.copyfile(args.out / name, args.report / name)
    for row in rows:
        shutil.copyfile(args.out / row["image"], args.report / row["image"])
    with (args.report / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["config"]].append(row)
    names = identity["configs"]
    energy_pairs = set.intersection(*[
        {(r["repeat"],r["prompt_id"],r["seed"]) for r in groups[name] if r.get("energy_j") is not None}
        for name in names])
    def mean(name, metric):
        vals = [r[metric] for r in groups[name] if r.get(metric) is not None]
        return float(np.mean(vals)) if vals else None
    baseline = mean("full-50", "total_s")
    summaries = []
    for name in names:
        item = {"config":name, "n":len(groups[name]), "speedup":baseline/mean(name,"total_s")}
        for field in ["total_s","denoise_s","encode_s","decode_s","peak_mlx_gb","cache_mb","full_steps",
                      "checked_steps","rejected_steps","lpips","ssim","psnr_db","clip_score","image_reward","energy_j"]:
            item[field] = mean(name,field)
        item["min_s"] = min(r["total_s"] for r in groups[name])
        item["max_s"] = max(r["total_s"] for r in groups[name])
        energies = [r["energy_j"] for r in groups[name] if (r["repeat"],r["prompt_id"],r["seed"]) in energy_pairs]
        item["energy_j"] = float(np.mean(energies)) if energies else None
        item["energy_n"] = len(energies)
        summaries.append(item)
    (args.report / "summary.json").write_text(json.dumps(summaries,indent=2,allow_nan=False))
    plt.rcParams.update({"font.size":10, "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    for s in summaries:
        axes[0].scatter(s["total_s"],s["lpips"],s=40)
        axes[0].annotate(s["config"],(s["total_s"],s["lpips"]),fontsize=8,xytext=(4,4),textcoords="offset points")
        axes[1].scatter(s["total_s"],s["image_reward"] if s["image_reward"] is not None else s["clip_score"],s=40)
        axes[1].annotate(s["config"],(s["total_s"],s["image_reward"] if s["image_reward"] is not None else s["clip_score"]),fontsize=8,xytext=(4,4),textcoords="offset points")
    axes[0].set(xlabel="Mean warm generation latency (seconds/image)",ylabel="Mean LPIPS to full-50 (lower is closer)",title="Reference fidelity versus latency")
    reward_available = summaries[0]["image_reward"] is not None
    axes[1].set(xlabel="Mean warm generation latency (seconds/image)",ylabel="Mean ImageReward" if reward_available else "Mean CLIP alignment score",title="Automated quality versus latency")
    fig.suptitle(f"Klein Base 4B Q4 · {manifest['hardware']} · {identity['size']} × {identity['size']}")
    fig.text(.01,.01,"Source: accompanying raw pilot results. Points average matched prompts/seeds; no confidence intervals implied.",fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.95]); fig.savefig(args.report / "tradeoffs.png",dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    axes[0].barh(names,[s["total_s"] for s in summaries]); axes[0].set(xlabel="Mean seconds/image",title="Warm generation latency")
    axes[1].barh(names,[s["peak_mlx_gb"] for s in summaries]); axes[1].set(xlabel="Mean per-trial peak MLX allocation (GB, decimal)",title="Full generation window allocation")
    fig.tight_layout(); fig.savefig(args.report / "resources.png",dpi=160); plt.close(fig)
    if energy_pairs:
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(names,[s["energy_j"] for s in summaries])
        ax.set(xlabel="Mean estimated CPU+GPU energy (joules/image)",title=f"Matched energy subset: {len(energy_pairs)} prompt/seed pairs per configuration")
        fig.text(.01,.01,"macmon v0.10.0, 500 ms samples; background activity included; not whole-device energy.",fontsize=8)
        fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(args.report / "energy.png",dpi=160); plt.close(fig)
        shutil.copyfile(args.out / "power.json", args.report / "power.json")
    for pid in range(len(identity["prompts"])):
        fig,axes=plt.subplots(2,4,figsize=(14,8))
        for ax,name in zip(axes.flat,names):
            row=next(r for r in rows if r["prompt_id"]==pid and r["config"]==name)
            ax.imshow(Image.open(args.out / row["image"])); ax.axis("off"); ax.set_title(name)
        fig.suptitle(identity["prompts"][pid],fontsize=10,wrap=True)
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(args.report / f"prompt-{pid}.jpg",dpi=120); plt.close(fig)
    lines=["# Klein Base 4B feature-reuse ablation pilot", "",
           "This report records one local pilot, not a final quality or energy-efficiency claim.","",
           "## Setup", "",
           f"Hardware: {manifest['hardware']}; macOS {identity['macos']}. Model: {identity['checkpoint']} at `{identity['checkpoint_revision']}`. "
           f"MFLUX {identity['packages']['mflux']}, MLX {identity['packages']['mlx']}; FP16 transformer compute, {identity['size']} × {identity['size']}, guidance {identity['guidance']}, scheduler `{identity['scheduler']}`.","",
           f"Coverage: {len(rows)} images, {len(identity['prompts'])} prompts, seeds {identity['seeds']}, {identity['repeats']} repeat per prompt/seed/config. "
           "Configurations rotate order across prompts. All thresholds were specified before inspecting results; these prompts remain a development set, not a held-out generalization test.","",
           "Full-50 is the reference. Full-17 reduces sampling steps. Fixed-3 and Taylor-3 each evaluate 17 full steps while retaining 50 scheduler updates. "
           "The two threshold settings compare stricter versus looser verification. Speca-0.3-1.0 removes threshold decay; speca-nocheck removes verification while retaining warmup and the reuse cap.","",
           "## Measurements", "",
           "| Configuration | Images | Total s | Speedup | Full steps | Checks | Rejections | Peak MLX GB | Cache MB | LPIPS ↓ | SSIM ↑ | CLIP ↑ | ImageReward ↑ | CPU+GPU J/image | Energy n |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(v): return "unmeasured" if v is None else f"{v:.3f}"
    for s in summaries:
        lines.append("| "+" | ".join([s["config"],str(s["n"])]+[fmt(s[k]) for k in ["total_s","speedup","full_steps","checked_steps","rejected_steps","peak_mlx_gb","cache_mb","lpips","ssim","clip_score","image_reward","energy_j"]]+[str(s["energy_n"])])+" |")
    lines += ["", "![Quality and latency](tradeoffs.png)","", "![Resource measurements](resources.png)","",
        "All values in the table are arithmetic means over the same prompt/seed pairs. Speedup is the ratio of mean full-50 time to mean configuration time. Individual results and latency ranges are in results.csv and summary.json.","",
        "## Scope and limitations", "",
        "Total latency covers prompt encoding, denoising, and decoding with resident weights and GPU synchronization. It excludes model loading, PNG writing, and offline scoring. The runner warms the full, forecast, and verification paths. MLX peak memory covers the generation window and includes resident weights; it is not total process footprint or whole-device memory. Persistent prompt-independent weights stay loaded across trials.","",
        (f"Energy is estimated from macmon v0.10.0 CPU+GPU telemetry on the common subset of {len(energy_pairs)} prompt/seed pairs with complete coverage for every configuration. Earlier images without telemetry are excluded from ALL energy averages. Samples are 500 ms interval averages, positioned approximately at interval midpoints and integrated with linear interpolation. Gaps over two seconds are rejected. These are system component estimates including background activity, not process-isolated, wall-socket, or whole-device energy. No idle subtraction or external-meter calibration was performed. The manifest's initial powermetrics-unavailable status predates this alternative collector; scoring.json records the actual energy scope." if energy_pairs else "Energy is unmeasured; no energy savings can be inferred from latency."),"",
        "GenEval and human preference are unmeasured. CLIP is a broad alignment proxy and ImageReward, if available, is a learned preference proxy, not a human judgment or a compositional correctness guarantee.","",
        "LPIPS/SSIM/PSNR compare each output with the matching full-50 image. Different but valid images can have poor similarity; higher similarity does not prove higher quality. The full-50 reference can itself contain defects. Identical-image PSNR is represented by null with an explicit identity flag. No FID is computed from this small sample.","",
        "One seed and one repeat cannot establish cross-seed robustness or isolate thermal/run-order noise. There are no confidence intervals or p95 claims. This is an exploratory ablation, with no predeclared numerical noninferiority threshold. Do not describe approximate feature prediction as lossless speculation.","",
        "## Reproduction", "", "```sh", ".venv/bin/python scripts/evaluate-klein.py --out outputs/evaluation", ".venv-quality/bin/python scripts/score-evaluation.py --out outputs/evaluation --image-reward", ".venv/bin/python scripts/report-evaluation.py --out outputs/evaluation", "```", "",
        "Environment locks: requirements-mac.txt, requirements-eval.txt, requirements-quality.txt. Raw measurements, scoring metadata, the generation manifest, and all pilot outputs accompany this report. Generation and scoring run separately. Hardware telemetry also includes system RAM/swap usage and GPU temperature for diagnosing environmental interference.","",
        "## Visual inspection", ""]
    for pid,prompt in enumerate(identity["prompts"]):
        lines += [f"### Prompt {pid}","",prompt,"",f"![Prompt {pid} ablations](prompt-{pid}.jpg)",""]
    if energy_pairs:
        lines += ["## Estimated energy", "", "![Matched energy comparisons](energy.png)", "",
            "Collector: [macmon](https://github.com/vladkens/macmon). Power uses CPU+GPU watts only. See power.json for samples, timestamps, temperatures, and RAM/swap context.", ""]
        telemetry=json.loads((args.out / "power.json").read_text())["telemetry"]
        temperatures=[r["gpu_temp_c"] for r in telemetry if r["gpu_temp_c"] is not None]
        swaps=[r["memory"]["swap_usage"]/1e9 for r in telemetry]
        if temperatures:
            lines += [f"During recorded telemetry, GPU temperature ranged from {min(temperatures):.1f} to {max(temperatures):.1f} °C. System-wide swap in use ranged from {min(swaps):.1f} to {max(swaps):.1f} GB. "
                "Existing swap is not attributable to the model. This was an active desktop environment, not an isolated machine; background activity, memory pressure, and thermal drift limit causal interpretation of small differences.", ""]
    lines += ["## References", "",
        "- [TaylorSeer](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_From_Reusing_to_Forecasting_Accelerating_Diffusion_Models_with_TaylorSeers_ICCV_2025_paper.pdf)",
        "- [SpeCa](https://arxiv.org/abs/2509.11628)",
        "- [SVDQuant](https://arxiv.org/abs/2411.05007)",
        "- [LPIPS](https://arxiv.org/abs/1801.03924)",
        "- [ImageReward](https://arxiv.org/abs/2304.05977)",
        "- [GenEval](https://arxiv.org/abs/2310.11513)",
        "- [ML.ENERGY](https://arxiv.org/abs/2505.06371)", ""]
    (args.report / "README.md").write_text("\n".join(lines))
    print(json.dumps(summaries,indent=2))


if __name__ == "__main__":
    main()
