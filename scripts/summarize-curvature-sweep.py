"""Summarize a damped-curvature sweep from existing evaluator measurements and scores."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FIELDS = ["lpips", "ssim", "image_reward", "denoise_s", "full_transformer_passes", "full_steps"]


def weight(config: str):
    if match := re.fullmatch(r"taylor2-\d+-c([\d.]+)", config):
        return float(match[1])
    return 1.0 if re.fullmatch(r"taylor2-\d+", config) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/curvature-sweep"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.out / "measurements.jsonl").read_text().splitlines()]
    scores = {r["key"]: r for r in map(json.loads, (args.out / "scores.jsonl").read_text().splitlines())}
    groups = defaultdict(list)
    for row in rows:
        groups[row["config"]].append({**row, **scores.get(row["key"], {})})

    def mean(items, field):
        values = [r[field] for r in items if r.get(field) is not None]
        return float(np.mean(values)) if values else None

    def fmt(value, digits=4):
        return "n/a" if value is None else f"{value:.{digits}f}"

    configs = json.loads((args.out / "manifest.json").read_text())["identity"]["configs"]
    lines = [
        "| Config | Curvature weight | Images | LPIPS ↓ | SSIM ↑ | ImageReward ↑ | Warm denoise s | Denoise speedup | Full transformer passes | Full steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = []
    baseline = mean(groups.get("full-50", []), "denoise_s")
    for name in configs:
        items = groups[name]
        item = {"config": name, "curvature_weight": weight(name), "n": len(items)}
        item.update({field: mean(items, field) for field in FIELDS})
        item["denoise_speedup"] = baseline / item["denoise_s"] if baseline and item["denoise_s"] else None
        summary.append(item)
        w = item["curvature_weight"]
        lines.append(" | ".join([
            f"| {name}", "first order" if name.startswith("taylor-") else ("—" if w is None else f"{w:g}"), str(item["n"]),
            fmt(item["lpips"]), fmt(item["ssim"]), fmt(item["image_reward"]), fmt(item["denoise_s"], 1),
            "n/a" if item["denoise_speedup"] is None else f"{item['denoise_speedup']:.2f}×", fmt(item["full_transformer_passes"], 1), fmt(item["full_steps"], 1),
        ]) + " |")

    # Zero damping should reproduce first-order Taylor pixel for pixel on each prompt/seed.
    checks = []
    first = {(r["repeat"], r["prompt_id"], r["seed"]): r.get("image_sha256") for r in groups.get("taylor-3", [])}
    for name in configs:
        if weight(name) == 0.0:
            pairs = [(first.get((r["repeat"], r["prompt_id"], r["seed"])), r.get("image_sha256")) for r in groups[name]]
            identical = sum(a is not None and a == b for a, b in pairs)
            checks.append(f"{name} vs taylor-3: {identical}/{len(pairs)} images pixel-identical")
    cached = [r for r in rows if r.get("baseline_cache")]
    if cached:
        checks.append(f"{len(cached)} baseline image(s) reused from the baseline cache; their latency was measured in "
                      f"an earlier session ({', '.join(sorted({str(r['cached_from']) for r in cached}))}).")
    text = "\n".join(lines + [""] + checks) + "\n"
    (args.out / "curvature-summary.md").write_text(text)
    (args.out / "curvature-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(text, end="")


if __name__ == "__main__":
    main()
