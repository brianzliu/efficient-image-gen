import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import mlx.core as mx
import numpy as np

from feature_reuse.klein import GUIDANCE, denoise, encode_prompt, load_klein
from feature_reuse.racfg import CALIBRATION, fit_gain, risk_schedule
from feature_reuse.reuse import BlockReuse, FixedInterval
from feature_reuse.svdquant import load_transformer

STEPS = 50
VARIANTS = ROOT / "models/klein-base-4b-bf16/variants"
TARGET_FULL_STEPS = 17


@dataclass(frozen=True)
class ReuseOnce:
    """Every step full except one, which reuses the previous step's result for both branches."""

    step: int
    forecast = False
    order = 1

    def runs_blocks(self, step: int, predicted: int) -> bool:
        return step != self.step

    def checks(self, predicted: int) -> bool:
        return False


def record_residuals(model, embeds, seed: int):
    """Run every step in full; return the image-token residual of the whole block stack per branch and step,
    the guided velocity per step, and the final latent."""
    residuals = {"cond": [], "uncond": []}
    velocities = []
    store = BlockReuse.store

    def recording(self, segment, residual):
        store(self, segment, residual)
        if segment == "body":
            self._body = residual
        else:
            # body + last = output of the last block minus the input to the first block.
            total = (self._body + residual)[:, self.text_tokens :].astype(mx.float16)
            mx.eval(total)
            residuals[self.branch].append(total)

    BlockReuse.store = recording
    try:
        result = denoise(model, embeds, seed, STEPS, FixedInterval(1), observe_velocity=velocities.append)
    finally:
        BlockReuse.store = store
    return residuals, velocities, result.latents


def magnitude_ratios(residuals: list) -> list[float]:
    # MagCache: mean over tokens of |r_k| / |r_{k-1}|. Step 0 has no previous step.
    norms = [mx.linalg.norm(r.astype(mx.float32), axis=-1) for r in residuals]
    return [1.0] + [mx.mean(b / (a + 1e-6)).item() for a, b in zip(norms, norms[1:])]


def alignment(cond: list, uncond: list) -> np.ndarray:
    """cos(c_a - c_b, u_a - u_b) for every a < b, from Gram matrices of the flattened residuals."""
    n = len(cond)
    c = mx.stack([r.reshape(-1) for r in cond]).astype(mx.float32)
    u = mx.stack([r.reshape(-1) for r in uncond]).astype(mx.float32)
    cc, uu, cu = (np.array(x) for x in (c @ c.T, u @ u.T, c @ u.T))
    rho = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            dot = cu[a, a] - cu[a, b] - cu[b, a] + cu[b, b]
            norm_c = cc[a, a] - 2 * cc[a, b] + cc[b, b]
            norm_u = uu[a, a] - 2 * uu[a, b] + uu[b, b]
            rho[a, b] = dot / np.sqrt(max(norm_c * norm_u, 1e-12))
    return rho


def propagation(model, embeds, seed: int, step: int, base_velocities, base_latents) -> float:
    """Final-latent deviation per unit of local guided-velocity error, for one reuse at this step."""
    velocities = []
    latents = denoise(model, embeds, seed, STEPS, ReuseOnce(step), observe_velocity=velocities.append).latents
    local = mx.linalg.norm((velocities[step] - base_velocities[step]).astype(mx.float32)).item()
    final = mx.linalg.norm((latents - base_latents).astype(mx.float32)).item()
    return final / local if local > 1e-8 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "calibration.txt")
    parser.add_argument("--prompt-count", type=int, default=3, help="prompts used for the propagation probes")
    parser.add_argument("--probe-every", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=CALIBRATION)
    parser.add_argument("--raw", type=Path, default=ROOT / "outputs" / "racfg-calibration.json")
    args = parser.parse_args()
    prompts = [p for p in args.prompts.read_text().splitlines() if p.strip()]

    model = load_klein()
    load_transformer(model, "bf16", VARIANTS)
    ratios = {"cond": [], "uncond": []}
    rhos, probes = [], []
    probe_steps = list(range(args.probe_every, STEPS, args.probe_every))
    for prompt_id, prompt in enumerate(prompts):
        start = time.perf_counter()
        embeds = encode_prompt(model, prompt)
        residuals, velocities, latents = record_residuals(model, embeds, args.seed)
        for branch in ratios:
            ratios[branch].append(magnitude_ratios(residuals[branch]))
        rhos.append(alignment(residuals["cond"], residuals["uncond"]))
        del residuals
        print(f"p{prompt_id}: recorded in {time.perf_counter() - start:.0f}s", flush=True)
        if prompt_id < args.prompt_count:
            for step in probe_steps:
                ratio = propagation(model, embeds, args.seed, step, velocities, latents)
                probes.append(dict(prompt_id=prompt_id, step=step, ratio=ratio))
                print(f"p{prompt_id} step {step}: gain {ratio:.3f}", flush=True)
        mx.clear_cache()

    by_step = {s: [p["ratio"] for p in probes if p["step"] == s and np.isfinite(p["ratio"])] for s in probe_steps}
    mean_ratio = {s: float(np.mean(v)) for s, v in by_step.items() if v}
    calibration = dict(
        steps=STEPS,
        guidance=GUIDANCE,
        magnitude_ratio={b: np.mean(ratios[b], axis=0).tolist() for b in ratios},
        alignment=np.mean(rhos, axis=0).tolist(),
        gain=fit_gain(list(mean_ratio), list(mean_ratio.values()), STEPS),
    )
    thresholds = [round(t, 3) for t in np.geomspace(0.01, 10, 61)]
    full_steps = {t: len(risk_schedule(calibration, t, GUIDANCE)) for t in thresholds}
    closest = min(thresholds, key=lambda t: (abs(full_steps[t] - TARGET_FULL_STEPS), -t))
    calibration.update(full_steps_by_threshold=full_steps, threshold_for_target=closest,
                       schedule_for_target=risk_schedule(calibration, closest, GUIDANCE))
    args.out.write_text(json.dumps(calibration, indent=1))
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.raw.write_text(json.dumps(dict(prompts=prompts, seed=args.seed, probes=probes, mean_ratio=mean_ratio), indent=1))
    print("gain", calibration["gain"])
    print(f"threshold {closest} gives {full_steps[closest]} full steps: {calibration['schedule_for_target']}")


if __name__ == "__main__":
    main()
