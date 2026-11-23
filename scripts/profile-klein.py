import argparse
import gc
import json
import os
import re
import resource
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import mlx.core as mx
from mflux.models.flux2.model.flux2_transformer.single_transformer_block import Flux2SingleTransformerBlock
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock
from mlx import nn
from mlx.utils import tree_flatten

import feature_reuse.klein as klein
from feature_reuse.profiling import BLOCK_TARGETS, OP_TARGETS, arrays, profile_summary, profiled

GB = 1000**3
SEED = 42
PROFILE_STEPS = 10
SWEEP_STEPS = 3
EVAL_STEPS = 5


EVAL_TARGETS = {
    "step": [],
    "pass": [(Flux2Transformer, "__call__")],
    "block": [(Flux2TransformerBlock, "__call__"), (Flux2SingleTransformerBlock, "__call__")],
    "linear": [(nn.QuantizedLinear, "__call__")],
}


@contextmanager
def evaluated_after(targets: list):
    originals = []
    for owner, attribute in targets:
        original = owner.__dict__[attribute]

        def wrapper(*args, _original=original, **kwargs):
            output = _original(*args, **kwargs)
            mx.eval(arrays(output))
            return output

        setattr(owner, attribute, wrapper)
        originals.append((owner, attribute, original))
    try:
        yield
    finally:
        for owner, attribute, original in originals:
            setattr(owner, attribute, original)


def measure(fn) -> tuple:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    start = perf_counter()
    output = fn()
    seconds = perf_counter() - start
    stats = {
        "seconds": seconds,
        "active_before_gb": before / GB,
        "active_after_gb": mx.get_active_memory() / GB,
        "peak_gb": mx.get_peak_memory() / GB,
    }
    return output, stats


def weight_groups(module) -> dict:
    groups = defaultdict(int)
    for name, array in tree_flatten(module.parameters()):
        groups[".".join(re.sub(r"\.\d+", ".*", name).split(".")[:4])] += array.nbytes
    return dict(sorted(groups.items(), key=lambda item: -item[1]))


def weight_summary(module) -> dict:
    stored = sum(array.nbytes for _, array in tree_flatten(module.parameters()))
    expanded = 0
    quantized = 0
    for _, layer in module.named_modules():
        if isinstance(layer, nn.QuantizedLinear):
            elements = layer.weight.shape[0] * layer.weight.shape[1] * 32 // layer.bits
            expanded += elements * 2
            quantized += sum(array.nbytes for _, array in tree_flatten(layer.parameters()))
    return {
        "stored_gb": stored / GB,
        "quantized_gb": quantized / GB,
        "unquantized_gb": (stored - quantized) / GB,
        "quantized_as_float16_gb": expanded / GB,
    }


def quantized_layer_benchmark(layer: nn.QuantizedLinear) -> dict:
    inputs = mx.random.normal((1, 1536, layer.weight.shape[1] * 32 // layer.bits), dtype=mx.float16)
    mx.eval(inputs, layer.parameters())

    def timed(fn) -> tuple[float, float]:
        mx.eval(fn())
        times = []
        for _ in range(10):
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            active = mx.get_active_memory()
            start = perf_counter()
            mx.eval(fn())
            times.append(perf_counter() - start)
        return sorted(times)[len(times) // 2], (mx.get_peak_memory() - active) / 1000**2

    def expand() -> mx.array:
        biases = layer["biases"] if "biases" in layer else None
        return mx.dequantize(
            layer["weight"], layer["scales"], biases, group_size=layer.group_size, bits=layer.bits, mode=layer.mode
        ).astype(mx.float16)

    quantized_s, quantized_mb = timed(lambda: layer(inputs))
    expand_s, expand_mb = timed(expand)
    weight = expand()
    mx.eval(weight)
    plain_s, plain_mb = timed(lambda: inputs @ weight.T)
    return {
        "shape": list(weight.shape),
        "bits": layer.bits,
        "quantized_call_s": quantized_s,
        "quantized_call_transient_mb": quantized_mb,
        "dequantize_s": expand_s,
        "dequantize_transient_mb": expand_mb,
        "plain_float16_call_s": plain_s,
        "plain_float16_call_transient_mb": plain_mb,
        "float16_weight_mb": weight.nbytes / 1000**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "benchmark.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "profile")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = args.prompts.read_text().splitlines()
    report = {"stages": {}, "chip_memory_gb": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB}

    def save() -> None:
        (args.out / "profile.json").write_text(json.dumps(report, indent=2))

    model, report["stages"]["load"] = measure(klein.load_klein)
    report["weights"] = {
        name: {"summary": weight_summary(module), "groups": weight_groups(module)}
        for name, module in [("text_encoder", model.text_encoder), ("transformer", model.transformer), ("vae", model.vae)]
    }
    tokenizer = model.tokenizers["qwen3"]
    report["prompt_tokens"] = [
        int(tokenizer.tokenize(prompt=prompt, max_length=512).attention_mask.sum().item()) for prompt in [*prompts, " "]
    ]

    embeds, report["stages"]["encode_cold"] = measure(lambda: klein.encode_prompt(model, prompts[0]))
    _, report["stages"]["encode_warm"] = measure(lambda: klein.encode_prompt(model, prompts[1]))
    report["embed_shape"] = list(embeds[0].shape)
    model.text_encoder = None
    _, report["stages"]["after_freeing_text_encoder"] = measure(lambda: None)
    save()

    _, report["stages"]["denoise_cold_1_step"] = measure(lambda: klein.denoise(model, embeds, SEED, 1))
    denoise = lambda: klein.denoise(model, embeds, SEED, PROFILE_STEPS)

    result, report["stages"]["baseline_before"] = measure(denoise)
    report["baseline_before_denoise_s"] = result.seconds
    _, report["stages"]["decode_cold"] = measure(lambda: klein.decode(model, result.latents))
    _, report["stages"]["decode_warm"] = measure(lambda: klein.decode(model, result.latents))
    save()

    for level, targets in [("block", BLOCK_TARGETS), ("op", OP_TARGETS)]:
        with profiled(targets) as profiler:
            # The profiler resets the peak counter, so this run's overall peak is not meaningful.
            profiled_result, _ = measure(denoise)
        report[f"profile_{level}"] = profile_summary(profiler, profiled_result.seconds)
        save()

    result, report["stages"]["baseline_after"] = measure(denoise)
    report["baseline_after_denoise_s"] = result.seconds
    save()

    report["eval_granularity"] = {}
    for name, targets in EVAL_TARGETS.items():
        with evaluated_after(targets):
            result, stats = measure(lambda: klein.denoise(model, embeds, SEED, EVAL_STEPS))
        report["eval_granularity"][name] = {"step_s": result.seconds / EVAL_STEPS, "peak_gb": stats["peak_gb"]}
        save()

    single =model.transformer.single_transformer_blocks[0].attn
    report["quantized_layers"] = {
        "single.to_qkv_mlp_proj": quantized_layer_benchmark(single.to_qkv_mlp_proj),
        "single.to_out": quantized_layer_benchmark(single.to_out),
    }
    save()

    report["text_token_sweep"] = {}
    prompt, text_ids, negative, negative_ids = embeds
    for text_tokens in [512, 256, 128, 64, 32]:
        cut = (prompt[:, :text_tokens], text_ids[:, :text_tokens], negative[:, :text_tokens], negative_ids[:, :text_tokens])
        result, stats = measure(lambda: klein.denoise(model, cut, SEED, SWEEP_STEPS))
        report["text_token_sweep"][text_tokens] = {"step_s": result.seconds / SWEEP_STEPS, "peak_gb": stats["peak_gb"]}
        save()

    report["resolution_sweep"] = {}
    for size in [256, 384, 512, 768]:
        klein.SIZE = size
        result, stats = measure(lambda: klein.denoise(model, embeds, SEED, SWEEP_STEPS))
        report["resolution_sweep"][size] = {"step_s": result.seconds / SWEEP_STEPS, "peak_gb": stats["peak_gb"]}
        save()
    klein.SIZE = 512

    report["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB
    save()
    print(f"saved {args.out / 'profile.json'}")


if __name__ == "__main__":
    main()
