"""Download the pinned community MLX Q4 conversion of Klein Base 4B.

With --bf16-transformer, also download the BF16 transformer, used as the SVDQuant starting point and quality reference.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bf16-transformer", action="store_true")
args = parser.parse_args()

snapshot_download(
    repo_id="mflux-community/flux-2-klein-base-4b-mflux-q4",
    revision="f69aa0b43285d0db24695ba68faabbdb1a0ad016",
    local_dir=ROOT / "models" / "klein-base-4b-q4",
    max_workers=3,
)
if args.bf16_transformer:
    snapshot_download(
        repo_id="mflux-community/flux-2-klein-base-4b-mflux-bf16",
        revision="4bd950744e23b76059848d78bcaaf702b919bda2",
        local_dir=ROOT / "models" / "klein-base-4b-bf16",
        allow_patterns=["transformer/*", "manifest.json", "README.md"],
        max_workers=3,
    )
