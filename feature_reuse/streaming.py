import gc
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

MODES = ("resident", "stage", "layer")
# Per component: the layer lists that "layer" mode streams.
LAYERS = {
    "text_encoder": ("layers",),
    "transformer": ("transformer_blocks", "single_transformer_blocks"),
    "vae": (),
}
# Children that never run, so no mode loads them.
UNUSED = {"vae": ("encoder",)}
# Klein reads text encoder hidden states 9, 18 and 27, so text encoder layers from 27 on never run.
LAYER_LIMITS = {"layers": 27}
EMPTY = mx.zeros((0,))


class Weights:
    """One module's weights, kept in a safetensors file while unloaded."""

    def __init__(self, module, path: Path, skip: tuple[str, ...] = ()) -> None:
        self.module, self.path, self.skip = module, path, skip
        self.loaded = True
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.stem + ".partial.safetensors")
            mx.save_safetensors(str(partial), dict(self._flat()))
            partial.rename(path)
        self.nbytes = path.stat().st_size

    def _flat(self):
        return tree_flatten({k: v for k, v in self.module.parameters().items() if k not in self.skip})

    def load(self) -> None:
        # Lazy: the file is read when the weights are first evaluated.
        if not self.loaded:
            self.module.update(tree_unflatten(list(mx.load(str(self.path)).items())))
            self.loaded = True

    def fetch(self) -> None:
        self.load()
        mx.eval([v for _, v in self._flat()])

    def unload(self) -> None:
        if self.loaded:
            self.module.update(tree_unflatten([(k, EMPTY) for k, _ in self._flat()]))
            self.loaded = False


class StreamedLayer:
    """Runs a layer, reading the next layer's weights while this one computes."""

    def __init__(self, weights: Weights) -> None:
        self.weights = weights
        self.next = None

    def __getattr__(self, name):
        return getattr(self.weights.module, name)

    def __call__(self, *args, **kwargs):
        self.weights.fetch()
        output = self.weights.module(*args, **kwargs)
        arrays = [v for _, v in tree_flatten(output) if isinstance(v, mx.array)]
        mx.async_eval(arrays)
        self.next.fetch()
        mx.eval(arrays)
        self.weights.unload()
        return output


class Streamer:
    """Keeps weights resident, or loads them from disk per stage or per layer."""

    def __init__(self, model, cache_dir: Path) -> None:
        self.model = model
        self.parts = {}
        for name in LAYERS:
            self.bind(name, cache_dir)

    def bind(self, name: str, cache_dir: Path) -> None:
        """Track the model's current module for this part, e.g. after swapping in another transformer."""
        module = getattr(self.model, name)
        layer_keys = LAYERS[name]
        rest = Weights(module, cache_dir / f"{name}.safetensors", layer_keys + UNUSED.get(name, ()))
        layers = {
            key: [
                Weights(layer, cache_dir / f"{name}.{key}.{i}.safetensors")
                for i, layer in enumerate(getattr(module, key)[: LAYER_LIMITS.get(key)])
            ]
            for key in layer_keys
        }
        self.parts[name] = (rest, layers)

    def release(self, name: str) -> None:
        """Stop tracking a part so its module can be freed."""
        self.parts.pop(name, None)

    def _weights(self, name: str) -> list[Weights]:
        rest, layers = self.parts[name]
        return [rest] + [weights for group in layers.values() for weights in group]

    def prepare(self, mode: str) -> None:
        """Put every weight where the mode expects it before a timed run."""
        for name in self.parts:
            for weights in self._weights(name):
                weights.fetch() if mode == "resident" else weights.unload()
        gc.collect()
        mx.clear_cache()

    @contextmanager
    def stage(self, name: str, mode: str):
        if mode == "resident":
            yield
            return
        rest, layers = self.parts[name]
        module = getattr(self.model, name)
        originals = {key: getattr(module, key) for key in layers}
        rest.load()
        if mode == "stage":
            for group in layers.values():
                for weights in group:
                    weights.load()
        elif layers:
            streamed = {key: [StreamedLayer(weights) for weights in group] for key, group in layers.items()}
            ring = [layer for group in streamed.values() for layer in group]
            for layer, following in zip(ring, ring[1:] + ring[:1]):
                layer.next = following.weights
            for key, group in streamed.items():
                setattr(module, key, group)
            ring[0].weights.fetch()
        try:
            yield
        finally:
            for key, original in originals.items():
                setattr(module, key, original)
            for weights in self._weights(name):
                weights.unload()
            gc.collect()
            mx.clear_cache()


def split_mode(name: str) -> tuple[str, str]:
    """Split a "+stage" or "+layer" suffix off a config name."""
    base, *suffixes = name.split("+")
    modes = [s for s in suffixes if s in MODES]
    if len(modes) > 1:
        raise ValueError(f"repeated streaming mode in config {name!r}")
    return "+".join([base] + [s for s in suffixes if s not in MODES]), (modes or ["resident"])[0]
