"""Download the pinned community MLX Q4 conversion of Klein Base 4B."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mflux-community/flux-2-klein-base-4b-mflux-q4",
    revision="f69aa0b43285d0db24695ba68faabbdb1a0ad016",
    local_dir=ROOT / "models" / "klein-base-4b-q4",
    max_workers=3,
)
