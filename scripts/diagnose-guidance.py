"""Log how the conditional and unconditional velocities relate at every step of plain CFG sampling.

Used to choose adaptive-guidance thresholds before any quality scoring. The run itself is
unmodified CFG; a recorder only observes the two transformer outputs of each step.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx

from feature_reuse.klein import GUIDANCE, decode, denoise, encode_prompt, load_klein
from feature_reuse.samplers import SAMPLERS


class Recorder:
    """Wraps the transformer and keeps every output, in call order: cond, uncond, cond, ..."""

    def __init__(self, transformer) -> None:
        self.transformer = transformer
        self.outputs = []

    def __call__(self, **kwargs):
        output = self.transformer(**kwargs)
        self.outputs.append(output.astype(mx.float32))
        return output


def norm(x: mx.array) -> float:
    return mx.sqrt(mx.sum(x * x)).item()


def cosine(a: mx.array, b: mx.array) -> float:
    return (mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)))).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts/benchmark.txt")
    parser.add_argument("--prompt-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", choices=sorted(SAMPLERS), default="euler")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/guidance-diagnostic")
    args = parser.parse_args()
    prompt = [p.strip() for p in args.prompts.read_text().splitlines() if p.strip()][args.prompt_id]

    model = load_klein()
    embeds = encode_prompt(model, prompt)
    recorder = Recorder(model.transformer)
    model.transformer = recorder
    try:
        result = denoise(model, embeds, args.seed, args.steps, sampler=args.sampler)
    finally:
        model.transformer = recorder.transformer
    if len(recorder.outputs) != 2 * args.steps:
        raise RuntimeError(f"expected {2 * args.steps} transformer calls, got {len(recorder.outputs)}")

    from mflux.models.common.config.config import Config
    from feature_reuse.klein import SCHEDULER, SIZE

    sigmas = Config(model_config=model.model_config, num_inference_steps=args.steps, height=SIZE, width=SIZE,
                    guidance=GUIDANCE, scheduler=SCHEDULER).scheduler.sigmas.tolist()
    rows, previous_delta = [], None
    for step in range(args.steps):
        cond, uncond = recorder.outputs[2 * step], recorder.outputs[2 * step + 1]
        delta = cond - uncond
        guided = uncond + GUIDANCE * delta
        rows.append(dict(
            step=step,
            sigma=sigmas[step],
            cos_cond_uncond=cosine(cond, uncond),
            # Size of the guidance signal relative to the conditional velocity.
            rel_delta=norm(delta) / norm(cond),
            # How far plain cond-only sampling would be from the guided velocity at this step.
            rel_guided_vs_cond=norm(guided - cond) / norm(guided),
            # How stable the guidance difference is between consecutive steps (for reusing it).
            cos_delta_prev=None if previous_delta is None else cosine(delta, previous_delta),
            rel_delta_change=None if previous_delta is None else norm(delta - previous_delta) / norm(delta),
        ))
        previous_delta = delta

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"p{args.prompt_id}-s{args.seed}-{args.sampler}-{args.steps}"
    meta = dict(prompt=prompt, prompt_id=args.prompt_id, seed=args.seed, steps=args.steps,
                sampler=args.sampler, guidance=GUIDANCE)
    (args.out / f"{tag}.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=2))
    decode(model, result.latents).save(args.out / f"{tag}.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    plots = [
        ("cos_cond_uncond", "cos(cond, uncond)"),
        ("rel_delta", "‖cond − uncond‖ / ‖cond‖"),
        ("cos_delta_prev", "cos(Δ_t, Δ_{t−1})"),
        ("rel_delta_change", "‖Δ_t − Δ_{t−1}‖ / ‖Δ_t‖"),
    ]
    for ax, (key, label) in zip(axes.flat, plots):
        points = [(r["step"], r[key]) for r in rows if r[key] is not None]
        ax.plot(*zip(*points), marker=".")
        ax.set(title=label, xlabel="step (σ falls from 1 to 0)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"CFG diagnostic, prompt {args.prompt_id}, seed {args.seed}, {args.sampler} × {args.steps}, w = {GUIDANCE}")
    fig.tight_layout()
    fig.savefig(args.out / f"{tag}.png".replace(".png", "-plot.png"), dpi=140)

    print(f"{'step':>4} {'sigma':>6} {'cos(c,u)':>9} {'|Δ|/|c|':>8} {'|g-c|/|g|':>9} {'cos(Δ,Δ-1)':>10} {'|ΔΔ|/|Δ|':>9}")
    for r in rows:
        fmt = lambda v: "     -" if v is None else f"{v:.4f}"
        print(f"{r['step']:>4} {r['sigma']:>6.3f} {r['cos_cond_uncond']:>9.4f} {r['rel_delta']:>8.4f} "
              f"{r['rel_guided_vs_cond']:>9.4f} {fmt(r['cos_delta_prev']):>10} {fmt(r['rel_delta_change']):>9}")
    print(f"Wrote {args.out / tag}.json, .png, -plot.png")


if __name__ == "__main__":
    main()
