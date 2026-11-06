import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import mlx.core as mx
import numpy as np
from mflux.models.common.config.config import Config
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_transformer.single_transformer_block import Flux2SingleTransformerBlock
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock
from mlx import nn

import feature_reuse.klein as klein

GB = 1000**3
MB = 1000**2
SEED = 42
STEPS = 2
REPEATS = 3
PROMPT = (ROOT / "prompts" / "benchmark.txt").read_text().splitlines()[0]
EVAL_TARGETS = {
    "step": [],
    "pass": [Flux2Transformer],
    "block": [Flux2TransformerBlock, Flux2SingleTransformerBlock],
    "linear": [nn.QuantizedLinear],
}
SIZES = [256, 512, 768]
BUFFER_SETTINGS = [
    {},
    {"MLX_MAX_OPS_PER_BUFFER": "25"},
    {"MLX_MAX_OPS_PER_BUFFER": "10"},
    {"MLX_MAX_OPS_PER_BUFFER": "2"},
    {"MLX_MAX_MB_PER_BUFFER": "20"},
    {"MLX_MAX_MB_PER_BUFFER": "5"},
    {"MLX_BFS_MAX_WIDTH": "5"},
    {"MLX_BFS_MAX_WIDTH": "1"},
    {"MLX_MAX_OPS_PER_BUFFER": "10", "MLX_BFS_MAX_WIDTH": "1"},
]


def arrays(value) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in arrays(item)]
    return []


@contextmanager
def evaluated_after(owners: list):
    originals = [(owner, owner.__dict__["__call__"]) for owner in owners]
    for owner, original in originals:

        def wrapper(*args, _original=original, **kwargs):
            output = _original(*args, **kwargs)
            mx.eval(arrays(output))
            return output

        owner.__call__ = wrapper
    try:
        yield
    finally:
        for owner, original in originals:
            owner.__call__ = original


def latent_hash(latents: mx.array) -> str:
    return hashlib.sha256(np.array(latents.astype(mx.float32)).tobytes()).hexdigest()[:12]


def repeat(fn, repeats: int = REPEATS) -> dict:
    peaks, extras, seconds, outputs = [], [], [], None
    for _ in range(repeats):
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        active = mx.get_active_memory()
        start = time.perf_counter()
        outputs = fn()
        seconds.append(time.perf_counter() - start)
        peaks.append(mx.get_peak_memory())
        extras.append(mx.get_peak_memory() - active)
    return {
        "seconds": statistics.median(seconds),
        "peak_gb": [peak / GB for peak in peaks],
        "extra_mb": statistics.median(extras) / MB,
        "output": outputs,
    }


def prepare():
    model = klein.load_klein()
    embeds = klein.encode_prompt(model, PROMPT)
    model.text_encoder = None
    gc.collect()
    mx.clear_cache()
    klein.denoise(model, embeds, SEED, 1)
    return model, embeds


def denoise_summary(model, embeds, repeats: int = REPEATS) -> dict:
    result = repeat(lambda: klein.denoise(model, embeds, SEED, STEPS), repeats)
    denoised = result.pop("output")
    result["step_s"] = denoised.seconds / STEPS
    result["latents"] = latent_hash(denoised.latents)
    del result["seconds"]
    return result


def eval_modes(model, embeds) -> dict:
    results = {}
    for name, owners in EVAL_TARGETS.items():
        with evaluated_after(owners):
            results[name] = denoise_summary(model, embeds)
    return results


def model_passes(model, embeds) -> dict:
    prompt, text_ids, negative, negative_ids = embeds
    timestep = Config(
        model_config=model.model_config,
        num_inference_steps=STEPS,
        height=klein.SIZE,
        width=klein.SIZE,
        guidance=klein.GUIDANCE,
        scheduler=klein.SCHEDULER,
    ).scheduler.timesteps[0]
    latents, latent_ids, _, _ = Flux2LatentCreator.prepare_packed_latents(
        seed=SEED, height=klein.SIZE, width=klein.SIZE, batch_size=1
    )
    mx.eval(latents, latent_ids)

    def forward(text: mx.array, ids: mx.array) -> mx.array:
        return model.transformer(
            hidden_states=latents,
            encoder_hidden_states=text,
            timestep=timestep,
            img_ids=latent_ids,
            txt_ids=ids,
            guidance=None,
        )

    def one() -> None:
        mx.eval(forward(prompt, text_ids))

    def two_together() -> None:
        mx.eval(forward(prompt, text_ids), forward(negative, negative_ids))

    def two_separate() -> None:
        first = forward(prompt, text_ids)
        mx.eval(first)
        mx.eval(forward(negative, negative_ids))

    return {
        name: {key: value for key, value in repeat(fn).items() if key != "output"}
        for name, fn in [("one", one), ("two_together", two_together), ("two_separate", two_separate)]
    }


def block_chains(model, embeds) -> dict:
    captured = {}
    original = Flux2SingleTransformerBlock.__dict__["__call__"]

    def capture(self, **kwargs):
        if not captured:
            captured.update(kwargs)
            mx.eval(arrays(list(kwargs.values())))
        return original(self, **kwargs)

    Flux2SingleTransformerBlock.__call__ = capture
    try:
        klein.denoise(model, embeds, SEED, 1)
    finally:
        Flux2SingleTransformerBlock.__call__ = original

    blocks = model.transformer.single_transformer_blocks

    def chain(length: int, evaluate_each: bool) -> None:
        hidden = captured["hidden_states"]
        for block in blocks[:length]:
            hidden = block(**{**captured, "hidden_states": hidden})
            if evaluate_each:
                mx.eval(hidden)
        mx.eval(hidden)

    results = {}
    for length in [1, 2, 5, 10, 20]:
        results[length] = {
            name: {key: value for key, value in repeat(lambda: chain(length, each)).items() if key != "output"}
            for name, each in [("lazy", False), ("evaluated", True)]
        }
    results["token_shape"] = list(captured["hidden_states"].shape)
    return results


def timelines(model, embeds, out: Path) -> dict:
    results = {}
    for name, owners in [("step", []), ("block", EVAL_TARGETS["block"])]:
        samples = []
        stop = threading.Event()

        def sample() -> None:
            start = time.perf_counter()
            while not stop.is_set():
                samples.append((time.perf_counter() - start, mx.get_active_memory() / GB))
                time.sleep(0.002)

        gc.collect()
        mx.clear_cache()
        thread = threading.Thread(target=sample)
        with evaluated_after(owners):
            thread.start()
            klein.denoise(model, embeds, SEED, STEPS)
            stop.set()
            thread.join()
        (out / f"memory-timeline-{name}.json").write_text(json.dumps(samples))
        values = sorted(value for _, value in samples)
        results[name] = {
            "samples": len(samples),
            "min_gb": values[0],
            "median_gb": values[len(values) // 2],
            "p90_gb": values[len(values) * 9 // 10],
            "max_gb": values[-1],
        }
    return results


def resolutions(model, embeds) -> dict:
    results = {}
    for size in SIZES:
        klein.SIZE = size
        results[size] = {}
        for name in ["step", "block"]:
            with evaluated_after(EVAL_TARGETS[name]):
                results[size][name] = denoise_summary(model, embeds, repeats=2)
    klein.SIZE = 512
    return results


def child() -> None:
    model, embeds = prepare()
    print("RESULT " + json.dumps(denoise_summary(model, embeds)), flush=True)


def buffer_settings() -> list:
    results = []
    for setting in BUFFER_SETTINGS:
        process = subprocess.run(
            [sys.executable, __file__, "--child"], env={**os.environ, **setting}, capture_output=True, text=True, check=True
        )
        line = next(line for line in process.stdout.splitlines() if line.startswith("RESULT "))
        results.append({"setting": setting, **json.loads(line.removeprefix("RESULT "))})
        print(results[-1], flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "profile")
    args = parser.parse_args()
    if args.child:
        child()
        return

    args.out.mkdir(parents=True, exist_ok=True)
    report = {}

    def save() -> None:
        (args.out / "memory.json").write_text(json.dumps(report, indent=2))

    model, embeds = prepare()
    for name, run in [
        ("eval_modes", lambda: eval_modes(model, embeds)),
        ("passes", lambda: model_passes(model, embeds)),
        ("block_chains", lambda: block_chains(model, embeds)),
        ("timelines", lambda: timelines(model, embeds, args.out)),
        ("resolutions", lambda: resolutions(model, embeds)),
    ]:
        report[name] = run()
        save()
        print(name, "done", flush=True)
    del model, embeds
    gc.collect()
    mx.clear_cache()
    report["buffer_settings"] = buffer_settings()
    save()
    print("saved", args.out / "memory.json")


if __name__ == "__main__":
    main()
