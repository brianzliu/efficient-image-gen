"""Apply cosine switch-off thresholds to saved guidance diagnostics and derive matched fixed cutoffs.

Mirrors CosineSwitchOff exactly: guidance stays on through the first step whose cos(cond, uncond)
reaches the threshold, and every later step is conditional only. The matched fixed cutoff
interval-1-LOW guides the median number of steps on every image, so both cost about the same.
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def guided_steps(rows: list[dict], threshold: float) -> int:
    """Steps that use CFG before the switch-off, the same for every scheduled step count."""
    for row in rows:
        if row["cos_cond_uncond"] >= threshold:
            return row["step"] + 1
    return len(rows)


def cutoff(sigmas: list[float], guided: int) -> str:
    """A rounded LOW with low < sigma for the first `guided` steps and sigma <= low afterward."""
    if guided >= len(sigmas):
        return "0"
    above, below = sigmas[guided - 1], sigmas[guided]
    for digits in range(3, 8):
        value = round((above + below) / 2, digits)
        if below <= value < above:
            return f"{value:.{digits}f}"
    raise ValueError("sigmas too close to separate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "outputs/guidance-diagnostic")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.99, 0.995, 0.998])
    args = parser.parse_args()
    runs = []
    for path in sorted(args.dir.glob("*.json")):
        data = json.loads(path.read_text())
        runs.append((data["meta"], data["rows"]))
    if not runs:
        raise SystemExit(f"no diagnostics in {args.dir}")
    shapes = {(m["steps"], m["sampler"], m["guidance"]) for m, _ in runs}
    if len(shapes) != 1:
        raise SystemExit(f"diagnostics mix step counts, samplers, or guidance scales: {shapes}")
    steps = runs[0][0]["steps"]
    sigmas = [r["sigma"] for r in runs[0][1]]
    if any([r["sigma"] for r in rows] != sigmas for _, rows in runs):
        raise SystemExit("diagnostics use different sigma schedules")
    print(f"{len(runs)} diagnostics, {steps} steps; each skipped step saves 1 of {2 * steps} transformer passes\n")

    for threshold in args.thresholds:
        print(f"ag-{threshold}")
        print(f"  {'image':<8} {'guided steps':>12} {'cond-only from σ':>17} {'passes saved':>13}  later dips below γ")
        counts = []
        for meta, rows in runs:
            guided = guided_steps(rows, threshold)
            counts.append(guided)
            dips = sum(r["cos_cond_uncond"] < threshold for r in rows[guided:])
            sigma = f"{sigmas[guided]:.4f}" if guided < steps else "never"
            print(f"  p{meta['prompt_id']}-s{meta['seed']:<4} {guided:>12} {sigma:>17} {steps - guided:>13}  {dips}")
        median = round(statistics.median(counts))
        mean_saved = steps - statistics.mean(counts)
        print(f"  guided steps: min {min(counts)}, median {statistics.median(counts):g}, max {max(counts)}, "
              f"spread {max(counts) - min(counts)} steps; mean saved {mean_saved:.1f}/{2 * steps} passes")
        print(f"  matched fixed cutoff: full-{steps}+interval-1-{cutoff(sigmas, median)}  "
              f"(guides {median} steps on every image)\n")


if __name__ == "__main__":
    main()
