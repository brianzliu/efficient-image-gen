"""Profile FLUX.2 Klein with a fixed workload, so results from different Macs can be compared.

Run from the repository root, with the Mac plugged in, Low Power Mode off and other apps closed:

    .venv/bin/python scripts/profile-chip.py

The result is written to results/chips/<chip>.json. Every run uses the same prompt, seed, size,
guidance and step count unless options are changed, and the file records whether they were.

Measurement rules:
- Every timed configuration gets one untimed warm-up run first, so one-time setup is excluded.
- Every timing is repeated, and the median, minimum and maximum are all reported.
- The breakdowns force MLX to finish its work only at block boundaries (and around the two
  hottest layers), never around small operations. The added overhead is measured and reported.
- Memory is reported three ways: MLX's exact peak of arrays in use, MLX arrays plus its cache of
  freed buffers, and the whole process footprint as macOS counts it.
"""

import argparse
import ctypes
import gc
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import mlx.core as mx
from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
from mlx import nn
from mlx.utils import tree_flatten

import feature_reuse.klein as klein

GB = 1000**3
SEED = 42
DEFAULTS = {"steps": 3, "repeats": 3, "prompt_index": 0}
HOT_LAYERS = ("to_qkv_mlp_proj", "to_out")


# ---------------------------------------------------------------- environment


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict:
    device = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    gpu_cores = None
    displays = command_output(["system_profiler", "SPDisplaysDataType", "-json"])
    if displays:
        for gpu in json.loads(displays).get("SPDisplaysDataType", []):
            gpu_cores = gpu_cores or gpu.get("sppci_cores")
    battery = command_output(["pmset", "-g", "batt"]) or ""
    low_power = re.search(r"lowpowermode\s+(\d)", command_output(["pmset", "-g"]) or "")
    return {
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "gpu_cores": int(gpu_cores) if gpu_cores else None,
        "cpu_cores": os.cpu_count(),
        "memory_gib": device["memory_size"] / 1024**3,
        "gpu_architecture": device.get("architecture"),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mflux": importlib.metadata.version("mflux"),
        "git_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_uncommitted_changes": bool(command_output(["git", "-C", str(ROOT), "status", "--porcelain"])),
        "on_battery": "Battery Power" in battery.splitlines()[0] if battery else None,
        "low_power_mode": low_power.group(1) == "1" if low_power else None,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def chip_slug(env: dict) -> str:
    chip = re.sub(r"[^a-z0-9]+", "-", (env["chip"] or "unknown").lower()).strip("-")
    return f"{chip}-{env['gpu_cores']}gpu-{round(env['memory_gib'])}gb"


# ---------------------------------------------------------------- memory


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
            "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
            "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions",
            "ri_cycles", "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
RUSAGE_INFO_V4 = 4


def process_usage() -> RusageInfoV4:
    """The process memory macOS reports, including GPU buffers. Activity Monitor shows the same footprint."""
    info = RusageInfoV4()
    if LIBPROC.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info


@contextmanager
def memory_watch(interval_s: float = 0.005):
    """Track peak memory of one stage. MLX's in-use peak is exact; cache and footprint are polled."""
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    result = {
        "mlx_in_use_before_gb": mx.get_active_memory() / GB,
        "footprint_before_gb": process_usage().ri_phys_footprint / GB,
    }
    peaks = {"with_cache": 0, "footprint": 0}
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            peaks["with_cache"] = max(peaks["with_cache"], mx.get_active_memory() + mx.get_cache_memory())
            peaks["footprint"] = max(peaks["footprint"], process_usage().ri_phys_footprint)
            time.sleep(interval_s)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield result
    finally:
        stop.set()
        thread.join()
        result["mlx_in_use_peak_gb"] = mx.get_peak_memory() / GB
        result["mlx_in_use_plus_cache_peak_gb"] = max(peaks["with_cache"], mx.get_peak_memory()) / GB
        result["footprint_peak_gb"] = peaks["footprint"] / GB


# ---------------------------------------------------------------- timing helpers


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}


def arrays(value) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in arrays(item)]
    return []


def timed(fn) -> float:
    start = time.perf_counter()
    mx.eval(arrays(fn()))
    return time.perf_counter() - start


def benchmark(fn, repeats: int) -> list[float]:
    """Warm up once, then time each repeat."""
    timed(fn)
    return [timed(fn) for _ in range(repeats)]


class SyncTimer:
    """Times parts of a pass by making MLX finish its queued work at the end of each part only.

    The time between two consecutive sync points is credited to the part that just ended. Forcing
    work to finish at the start of a part as well was measured to add 8% to 17% to step time,
    while forcing it only at the end added 1% to 2%.
    """

    def __init__(self) -> None:
        self.seconds = defaultdict(float)
        self.macs = defaultdict(int)
        self.last = time.perf_counter()

    def sync(self, values, label: str | None = None) -> None:
        mx.eval(arrays(values))
        now = time.perf_counter()
        if label is not None:
            self.seconds[label] += now - self.last
        self.last = now

    def block(self, name: str, block, end_label: str):
        def call(**kwargs):
            # Work queued before the block (such as the embedders) is finished here and not credited.
            self.sync(kwargs)
            output = block(**kwargs)
            self.sync(output, f"{name}.{end_label}" if end_label else name)
            return output

        return call

    def layer(self, name: str, layer, before_label: str | None = None):
        def call(x):
            if before_label:
                self.sync(x, before_label)
            output = layer(x)
            self.sync(output, name)
            self.macs[name] += output.size * x.shape[-1]
            return output

        return call


@contextmanager
def timed_parts(transformer, timer: SyncTimer, hot_layers: bool):
    """Time every block, and optionally split each single block into its main parts."""
    doubles = transformer.transformer_blocks
    singles = transformer.single_transformer_blocks
    transformer.transformer_blocks = [timer.block(f"double.{i}", block, "") for i, block in enumerate(doubles)]
    transformer.single_transformer_blocks = [
        timer.block(f"single.{i}", block, "residual" if hot_layers else "") for i, block in enumerate(singles)
    ]
    replaced = []
    if hot_layers:
        for i, block in enumerate(singles):
            attn = block.attn
            replaced += [(attn, "to_qkv_mlp_proj", attn["to_qkv_mlp_proj"]), (attn, "to_out", attn["to_out"])]
            # The first part also contains the block's input norm, which is small next to the matmul.
            attn["to_qkv_mlp_proj"] = timer.layer(f"single.{i}.to_qkv_mlp_proj", attn["to_qkv_mlp_proj"])
            attn["to_out"] = timer.layer(f"single.{i}.to_out", attn["to_out"], f"single.{i}.attention_and_activation")
    try:
        yield
    finally:
        transformer.transformer_blocks = doubles
        transformer.single_transformer_blocks = singles
        for owner, name, layer in replaced:
            owner[name] = layer


@contextmanager
def counted_macs():
    """Count multiply-accumulates from tensor shapes, without forcing any evaluation."""
    counts = defaultdict(int)
    patches = []

    def patch(owner, attribute: str, key: str, count) -> None:
        original = owner.__dict__[attribute]
        is_static = isinstance(original, staticmethod)
        function = original.__func__ if is_static else original

        def counting(*args, **kwargs):
            output = function(*args, **kwargs)
            counts[key] += count(args, kwargs, output)
            return output

        setattr(owner, attribute, staticmethod(counting) if is_static else counting)
        patches.append((owner, attribute, original))

    def linear(args, kwargs, output) -> int:
        return output.size * args[1].shape[-1]

    def attention(args, kwargs, output) -> int:
        batch, heads, tokens, head_dim = kwargs["query"].shape
        return 2 * batch * heads * tokens * kwargs["key"].shape[2] * head_dim

    patch(nn.QuantizedLinear, "__call__", "linear", linear)
    patch(nn.Linear, "__call__", "linear", linear)
    patch(AttentionUtils, "compute_attention", "attention", attention)
    try:
        yield counts
    finally:
        for owner, attribute, original in patches:
            setattr(owner, attribute, original)


# ---------------------------------------------------------------- chip limits


def matmul_peak_tflops(repeats: int = 10, size: int = 4096) -> dict:
    """The float16 matrix-multiply speed this chip reaches in MLX, used as its practical compute ceiling."""
    a = mx.random.normal((size, size)).astype(mx.float16)
    b = mx.random.normal((size, size)).astype(mx.float16)
    mx.eval(a, b)
    seconds = benchmark(lambda: a @ b, repeats)
    return {"size": size, "tflops": 2 * size**3 / statistics.median(seconds) / 1e12, "seconds": spread(seconds)}


def bandwidth_gbps(repeats: int = 10, megabytes: int = 512) -> dict:
    """The memory speed this chip reaches in MLX: read one large array and write another."""
    x = mx.ones((megabytes * 1000**2 // 2,), dtype=mx.float16)
    mx.eval(x)
    seconds = benchmark(lambda: x + 1, repeats)
    return {"megabytes": megabytes, "gbps": 2 * x.nbytes / statistics.median(seconds) / GB, "seconds": spread(seconds)}


# ---------------------------------------------------------------- profile stages


def denoise_seconds_per_step(model, embeds, steps: int) -> tuple[float, klein.Denoised]:
    result = klein.denoise(model, embeds, SEED, steps)
    return result.seconds / steps, result


def time_denoise(model, embeds, steps: int, repeats: int) -> tuple[dict, klein.Denoised]:
    denoise_seconds_per_step(model, embeds, steps)
    samples, result = [], None
    for _ in range(repeats):
        seconds, result = denoise_seconds_per_step(model, embeds, steps)
        samples.append(seconds)
    return spread(samples), result


def group_name(name: str) -> str:
    parts = name.split(".")
    kind = "double_blocks" if parts[0] == "double" else "single_blocks"
    return f"{kind}.{parts[2]}" if len(parts) == 3 else kind


def breakdown(model, embeds, steps: int, repeats: int, hot_layers: bool, unforced_s: float) -> dict:
    """Step time split into parts. Shares are of the instrumented step time."""

    def run() -> tuple[float, SyncTimer]:
        timer = SyncTimer()
        with timed_parts(model.transformer, timer, hot_layers):
            seconds, _ = denoise_seconds_per_step(model, embeds, steps)
        return seconds, timer

    run()
    runs = [run() for _ in range(repeats)]
    totals = [seconds for seconds, _ in runs]
    groups = defaultdict(list)
    for seconds, timer in runs:
        per_step = defaultdict(float)
        for name, value in timer.seconds.items():
            per_step[group_name(name)] += value / steps
        per_step["outside_blocks"] = seconds - sum(per_step.values())
        for name, value in per_step.items():
            groups[name].append(value)
    instrumented = statistics.median(totals)
    result = {
        "step_s": spread(totals),
        "overhead_vs_unforced": instrumented / unforced_s - 1,
        "parts": {
            name: {"step_s": statistics.median(values), "share": statistics.median(values) / instrumented}
            for name, values in sorted(groups.items(), key=lambda item: -statistics.median(item[1]))
        },
    }
    if hot_layers:
        timer = runs[-1][1]
        result["hot_layer_macs_per_step"] = {
            layer: sum(value for name, value in timer.macs.items() if name.endswith(layer)) / steps
            for layer in HOT_LAYERS
        }
    else:
        timer = runs[-1][1]
        result["per_block_s"] = {
            kind: [timer.seconds[f"{kind}.{i}"] / steps for i in range(count)]
            for kind, count in [
                ("double", len(model.transformer.transformer_blocks)),
                ("single", len(model.transformer.single_transformer_blocks)),
            ]
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"], help="denoising steps per timed run")
    parser.add_argument("--repeats", type=int, default=DEFAULTS["repeats"], help="timed runs per measurement")
    parser.add_argument("--prompt-index", type=int, default=DEFAULTS["prompt_index"], help="line of the prompts file")
    parser.add_argument("--out", type=Path, help="output file (default: results/chips/<chip>.json)")
    args = parser.parse_args()

    env = environment()
    if env["on_battery"] or env["low_power_mode"]:
        print("warning: on battery or in Low Power Mode; results may not be comparable", file=sys.stderr)
    out = args.out or ROOT / "results" / "chips" / f"{chip_slug(env)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = (ROOT / "prompts" / "benchmark.txt").read_text().splitlines()[args.prompt_index]
    settings = {"steps": args.steps, "repeats": args.repeats, "prompt_index": args.prompt_index}
    report = {
        "environment": env,
        "workload": {
            **settings,
            "standard": settings == DEFAULTS,
            "prompt": prompt,
            "seed": SEED,
            "size": klein.SIZE,
            "guidance": klein.GUIDANCE,
            "mlx_cache_limit_gb": None,
        },
    }

    def save() -> None:
        out.write_text(json.dumps(report, indent=2))

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    stages = report["stages"] = {}

    log("loading model")
    start = time.perf_counter()
    model = klein.load_klein()
    stages["load"] = {"seconds": time.perf_counter() - start}
    cache_limit = mx.set_cache_limit(0)
    mx.set_cache_limit(cache_limit)
    report["workload"]["mlx_cache_limit_gb"] = cache_limit / GB
    report["weights_gb"] = {
        name: sum(array.nbytes for _, array in tree_flatten(module.parameters())) / GB
        for name, module in [("text_encoder", model.text_encoder), ("transformer", model.transformer), ("vae", model.vae)]
    }
    tokenizer = model.tokenizers["qwen3"]
    report["workload"]["prompt_tokens"] = int(tokenizer.tokenize(prompt=prompt, max_length=512).attention_mask.sum().item())

    log("prompt encoding")
    encode = lambda: klein.encode_prompt(model, prompt)
    stages["encode"] = {"seconds": spread(benchmark(encode, args.repeats))}
    with memory_watch() as memory:
        embeds = encode()
    stages["encode"]["memory"] = memory
    report["workload"]["tokens"] = {"text": embeds[0].shape[1], "image": (klein.SIZE // 16) ** 2}
    model.text_encoder = None
    save()

    log("counting work per step")
    with counted_macs() as counts:
        _, counted = denoise_seconds_per_step(model, embeds, 1)
    passes_per_step = counted.full_passes
    report["work_per_step"] = {
        "model_passes": passes_per_step,
        "macs": dict(counts),
        "total_macs": sum(counts.values()),
    }

    log(f"denoising, {args.repeats} runs of {args.steps} steps")
    unforced, latents_result = time_denoise(model, embeds, args.steps, args.repeats)
    stages["denoise"] = {"step_s": unforced, "estimated_50_steps_s": 50 * unforced["median"]}
    with memory_watch() as memory:
        denoise_seconds_per_step(model, embeds, args.steps)
    stages["denoise"]["memory"] = memory
    save()

    log("block breakdown")
    report["block_breakdown"] = breakdown(model, embeds, args.steps, args.repeats, False, unforced["median"])
    save()
    log("hot layer breakdown")
    report["hot_layer_breakdown"] = breakdown(model, embeds, args.steps, args.repeats, True, unforced["median"])
    save()

    log("image decoding")
    packed = latents_result.latents.reshape(1, klein.SIZE // 16, klein.SIZE // 16, -1).transpose(0, 3, 1, 2)
    decode = lambda: model.vae.decode_packed_latents(packed, tiling_config=model.tiling_config)
    stages["decode"] = {"seconds": spread(benchmark(decode, args.repeats))}
    with memory_watch() as memory:
        mx.eval(decode())
    stages["decode"]["memory"] = memory
    save()

    log("denoising again, to check for slowdown during the run")
    end, _ = time_denoise(model, embeds, args.steps, args.repeats)
    stages["denoise_end"] = {"step_s": end, "change_vs_start": end["median"] / unforced["median"] - 1}
    save()

    log("chip limits")
    peak = matmul_peak_tflops()
    bandwidth = bandwidth_gbps()
    step_s = unforced["median"]
    achieved = 2 * report["work_per_step"]["total_macs"] / step_s / 1e12
    weight_gbps = report["weights_gb"]["transformer"] * passes_per_step / step_s
    hot = report["hot_layer_breakdown"]
    report["utilization"] = {
        "matmul_peak": peak,
        "bandwidth": bandwidth,
        "achieved_tflops": achieved,
        "compute_share_of_peak": achieved / peak["tflops"],
        "weight_read_gbps": weight_gbps,
        "bandwidth_share_lower_bound": weight_gbps / bandwidth["gbps"],
        "likely_limit": "compute" if achieved / peak["tflops"] > weight_gbps / bandwidth["gbps"] else "memory",
        "hot_layers": {
            layer: {
                "tflops": 2 * hot["hot_layer_macs_per_step"][layer] / hot["parts"][f"single_blocks.{layer}"]["step_s"] / 1e12,
                "share_of_peak": 2 * hot["hot_layer_macs_per_step"][layer]
                / hot["parts"][f"single_blocks.{layer}"]["step_s"]
                / 1e12
                / peak["tflops"],
            }
            for layer in HOT_LAYERS
        },
    }
    report["lifetime_peak_footprint_gb"] = process_usage().ri_lifetime_max_phys_footprint / GB
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save()
    print_summary(report)
    log(f"saved {out}")


def print_summary(report: dict) -> None:
    env, stages, use = report["environment"], report["stages"], report["utilization"]
    print(f"\n{env['chip']}, {env['gpu_cores']}-core GPU, {env['memory_gib']:.0f} GB, MLX {env['mlx']}")
    if not report["workload"]["standard"]:
        print("non-standard settings: not comparable with standard runs")
    step = stages["denoise"]["step_s"]
    print(f"denoising step: {step['median']:.2f} s (range {step['min']:.2f} to {step['max']:.2f})")
    print(f"estimated 50 steps: {stages['denoise']['estimated_50_steps_s']:.0f} s")
    print(f"encode {stages['encode']['seconds']['median']:.2f} s, decode {stages['decode']['seconds']['median']:.2f} s")
    memory = stages["denoise"]["memory"]
    print(
        f"denoising memory: in use {memory['mlx_in_use_peak_gb']:.2f} GB, "
        f"with cache {memory['mlx_in_use_plus_cache_peak_gb']:.2f} GB, process {memory['footprint_peak_gb']:.2f} GB"
    )
    for name in ("block_breakdown", "hot_layer_breakdown"):
        parts = report[name]["parts"]
        shares = ", ".join(f"{part} {value['share']:.0%}" for part, value in parts.items())
        print(f"{name} (overhead {report[name]['overhead_vs_unforced']:+.1%}): {shares}")
    print(
        f"GPU: {use['achieved_tflops']:.2f} TFLOPS of {use['matmul_peak']['tflops']:.2f} measured peak "
        f"({use['compute_share_of_peak']:.0%}); weight reads use at least "
        f"{use['bandwidth_share_lower_bound']:.1%} of {use['bandwidth']['gbps']:.0f} GB/s; likely {use['likely_limit']}-bound"
    )
    print(f"drift: last denoising runs were {stages['denoise_end']['change_vs_start']:+.1%} vs first")


if __name__ == "__main__":
    main()
