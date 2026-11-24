import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.variants import Flux2Klein
from mlx.utils import tree_flatten

from feature_reuse.klein import MODEL_PATH, denoise, encode_prompt, load_klein
from feature_reuse.svdquant import (
    _convert,
    best_layer,
    build,
    load_transformer,
    output_error,
    parse_variant,
    quantize,
)

BF16_PATH = ROOT / "models" / "klein-base-4b-bf16"
VARIANTS = BF16_PATH / "variants"
CALIBRATION = VARIANTS / "calibration.safetensors"
STEPS = 25
TOKENS_PER_CALL = 6
CONFIG = ModelConfig.flux2_klein_base_4b()


def transformer_from(path: Path) -> Flux2Transformer:
    # Without compute_precision, so the saved weights are the checkpoint's own.
    return Flux2Klein(model_config=CONFIG, model_path=str(path)).transformer


def save(transformer, name: str) -> float:
    path = VARIANTS / f"{name}.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(transformer.parameters())))
    return path.stat().st_size / 1e9


def load(name: str) -> Flux2Transformer:
    kind, *_ = parse_variant(name)
    transformer = Flux2Transformer(**CONFIG.transformer_overrides)
    if kind in ("q4", "q8"):
        quantize(transformer, int(kind[1]))
    transformer.load_weights(str(VARIANTS / f"{name}.safetensors"))
    return transformer


def prepare_checkpoints(report: dict) -> None:
    # The BF16 download has only the transformer; share the Q4 text encoder, tokenizer and VAE.
    for part in ("text_encoder", "tokenizer", "vae"):
        link = BF16_PATH / part
        if not link.exists():
            link.symlink_to(MODEL_PATH / part)
    if not (VARIANTS / "bf16.safetensors").exists():
        save(transformer_from(BF16_PATH), "bf16")
    if not (VARIANTS / "q4.safetensors").exists():
        mflux_q4 = transformer_from(MODEL_PATH)
        rebuilt = load("bf16")
        quantize(rebuilt)
        pairs = zip(tree_flatten(mflux_q4.parameters()), tree_flatten(rebuilt.parameters()))
        report["q4_matches_mflux"] = all(a == b and mx.array_equal(x, y).item() for (a, x), (b, y) in pairs)
        save(mflux_q4, "q4")
    if not (VARIANTS / "q8.safetensors").exists():
        q8 = load("bf16")
        quantize(q8, 8)
        save(q8, "q8")


def calibrate(prompts: list[str], seed: int) -> None:
    """Record each linear layer's per-channel input maximum and a sample of its inputs, running BF16."""
    model = load_klein()
    load_transformer(model, "bf16", VARIANTS)
    paths = {id(m): p for p, m in model.transformer.named_modules() if isinstance(m, nn.Linear)}
    maxima, samples = {}, defaultdict(list)
    rng = np.random.default_rng(0)
    call = nn.Linear.__call__

    def recording(self, x):
        path = paths.get(id(self))
        if path is not None:
            flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            top = mx.max(mx.abs(flat), axis=0)
            maxima[path] = top if path not in maxima else mx.maximum(maxima[path], top)
            samples[path].append(flat[mx.array(rng.integers(0, flat.shape[0], TOKENS_PER_CALL))].astype(mx.float16))
            mx.eval(maxima[path], samples[path][-1])
        return call(self, x)

    nn.Linear.__call__ = recording
    try:
        for prompt in prompts:
            denoise(model, encode_prompt(model, prompt), seed, STEPS)
            print(f"calibrated on: {prompt[:40]}", flush=True)
    finally:
        nn.Linear.__call__ = call
    arrays = {f"{p}.max": m for p, m in maxima.items()}
    arrays.update({f"{p}.inputs": mx.concatenate(s) for p, s in samples.items()})
    mx.save_safetensors(str(CALIBRATION), arrays)


def errors_of(transformer, original, calibration) -> dict[str, float]:
    originals = dict(original.named_modules())
    return {
        path: output_error(module, originals[path], calibration[f"{path}.inputs"].astype(mx.float32))
        for path, module in transformer.named_modules()
        if f"{path}.inputs" in calibration and hasattr(module, "scales")
    }


def build_variant(name: str, original, calibration) -> dict:
    _, rank, smooth = parse_variant(name)
    transformer = load("bf16")
    alphas = Counter()

    def convert(path, module):
        if not isinstance(module, nn.Linear):
            return module
        if f"{path}.inputs" not in calibration:
            # Never called during generation, so there is nothing to calibrate against.
            return build(module, rank, mx.ones((module.weight.shape[1],)) if smooth else None)
        inputs = calibration[f"{path}.inputs"].astype(mx.float32)
        layer, alpha, _ = best_layer(module, rank, smooth, calibration[f"{path}.max"], inputs)
        alphas[str(alpha)] += 1
        return layer

    start = time.perf_counter()
    _convert(transformer, convert)
    build_s = time.perf_counter() - start
    gb = save(transformer, name)
    return dict(gb=gb, build_s=build_s, alphas=dict(alphas), errors=errors_of(transformer, original, calibration))


def summary(errors: dict[str, float]) -> dict:
    values = np.array(list(errors.values()))
    return dict(mean_error=float(values.mean()), median_error=float(np.median(values)), max_error=float(values.max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["smooth", "lowrank-32", "svdq-32"])
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "calibration.txt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    VARIANTS.mkdir(parents=True, exist_ok=True)
    report_path = VARIANTS / "build.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    prepare_checkpoints(report)
    if not CALIBRATION.exists():
        calibrate([p for p in args.prompts.read_text().splitlines() if p.strip()], args.seed)
        gc.collect()
        mx.clear_cache()
    calibration = mx.load(str(CALIBRATION))
    original = load("bf16")

    for name in ("q4", "q8"):
        errors = errors_of(load(name), original, calibration)
        report[name] = dict(gb=(VARIANTS / f"{name}.safetensors").stat().st_size / 1e9, **summary(errors))
        print(name, report[name], flush=True)
    for name in args.variants:
        result = build_variant(name, original, calibration)
        report[name] = dict(gb=result["gb"], build_s=result["build_s"], alphas=result["alphas"], **summary(result["errors"]))
        print(name, report[name], flush=True)
        gc.collect()
        mx.clear_cache()
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
