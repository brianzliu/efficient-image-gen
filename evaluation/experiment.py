"""Resume validation and a content-addressed full-step baseline cache for evaluate-klein.py.

Standard library only, so it can be tested without loading the model.
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

# Everything in an experiment identity that can change a full-step image or its timing.
# Experiment-level choices (label, config list, prompt list, seeds, repeats) are excluded so a
# baseline generated for one experiment can serve another; the per-trial prompt, seed and repeat are
# added by baseline_fields.
BASELINE_IDENTITY_FIELDS = (
    "size", "guidance", "scheduler", "source_sha256", "checkpoint", "checkpoint_revision",
    "packages", "macos", "machine", "hardware",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial_key(repeat: int, prompt_id: int, seed: int, config: str) -> str:
    return f"r{repeat}-p{prompt_id}-s{seed}-{config}"


def baseline_fields(identity: dict, config: str, steps: int, prompt: str, seed: int, repeat: int) -> dict:
    """Cache key material for one full-step trial; a missing identity field is an error."""
    fields = {name: identity[name] for name in BASELINE_IDENTITY_FIELDS}
    fields.update(config=config, sampling_steps=steps, prompt=prompt, seed=seed, repeat=repeat)
    return fields


def load_completed(measurements: Path, out: Path, identity: dict) -> dict[str, dict]:
    """Return completed rows keyed by trial, each validated against its image and the identity.

    A final line without a newline is an interrupted append (rows are fsynced after the newline),
    so it is truncated away and that trial reruns. Any other inconsistency raises.
    """
    if not measurements.exists():
        return {}
    data = measurements.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        print(f"Dropping an incomplete final measurement line ({len(data) - keep} bytes) from {measurements}", flush=True)
        with measurements.open("r+b") as file:
            file.truncate(keep)
        data = data[:keep]
    rows = {}
    for number, line in enumerate(data.decode().splitlines(), start=1):
        row = json.loads(line)
        key = row["key"]
        if key in rows:
            raise ValueError(f"Duplicate trial key {key} at line {number}")
        prompt_id = row["prompt_id"]
        consistent = (
            key == trial_key(row["repeat"], prompt_id, row["seed"], row["config"])
            and row["config"] in identity["configs"]
            and row["seed"] in identity["seeds"]
            and 0 <= row["repeat"] < identity["repeats"]
            and 0 <= prompt_id < len(identity["prompts"])
            and row["prompt"] == identity["prompts"][prompt_id]
        )
        if not consistent:
            raise ValueError(f"{key}: measurement does not match the experiment identity")
        image = out / row["image"]
        if not image.is_file():
            raise ValueError(f"{key}: recorded image {row['image']} is missing")
        if row.get("image_sha256") != sha256_file(image):
            raise ValueError(f"{key}: image does not match its recorded SHA-256")
        rows[key] = row
    return rows


class BaselineCache:
    """Full-step images and their measurements, stored under the SHA-256 of their baseline fields."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def key(fields: dict) -> str:
        return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def lookup(self, fields: dict) -> tuple[str, dict, Path] | None:
        key = self.key(fields)
        entry = self.root / key
        if not (entry / "entry.json").is_file():
            return None
        record = json.loads((entry / "entry.json").read_text())
        if record["fields"] != fields:
            raise ValueError(f"Baseline cache entry {key} does not match its key; refusing to reuse it")
        image = entry / "image.png"
        if not image.is_file() or sha256_file(image) != record["image_sha256"]:
            raise ValueError(f"Baseline cache entry {key} has a missing or modified image; refusing to reuse it")
        return key, record["row"], image

    def store(self, fields: dict, row: dict, image: Path) -> str:
        key = self.key(fields)
        entry = self.root / key
        if entry.exists():
            return key
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=".staging-"))
        try:
            shutil.copyfile(image, staging / "image.png")
            record = dict(fields=fields, row=row, image_sha256=sha256_file(staging / "image.png"))
            (staging / "entry.json").write_text(json.dumps(record, indent=2, allow_nan=False))
            staging.rename(entry)  # Atomic, so a reader never sees a partial entry.
        except OSError:
            if not entry.exists():
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return key
