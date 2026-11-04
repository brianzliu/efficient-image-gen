"""Score saved images on CPU, outside generation timing and memory windows."""
import argparse
import hashlib
import json
import os
import sys
import subprocess
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
import numpy as np
from PIL import Image
import torch
import lpips
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import HfApi, hf_hub_download
from evaluation.metrics import similarity, integrate_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--image-reward", action="store_true")
    parser.add_argument("--power-samples", type=Path, help='JSON: {scope: string, samples: [[unix_seconds, watts], ...]}')
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = list(map(json.loads, (args.out / "measurements.jsonl").read_text().splitlines()))
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trial keys")
    refs = {(r["repeat"], r["prompt_id"], r["seed"]): r for r in rows if r["config"] == "full-50"}
    if any((r["repeat"], r["prompt_id"], r["seed"]) not in refs for r in rows):
        raise ValueError("Missing full-50 reference")
    revision = HfApi().model_info("openai/clip-vit-base-patch32").sha
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision=revision).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", revision=revision)
    perceptual = lpips.LPIPS(net="alex").eval()
    reward = None
    reward_revision = None
    if args.image_reward:
        import ImageReward
        reward_revision = HfApi().model_info("THUDM/ImageReward").sha
        checkpoint = hf_hub_download("THUDM/ImageReward", "ImageReward.pt", revision=reward_revision)
        config = hf_hub_download("THUDM/ImageReward", "med_config.json", revision=reward_revision)
        reward = ImageReward.load(name=checkpoint, med_config=config, device="cpu")
    power = json.loads(args.power_samples.read_text()) if args.power_samples else None
    info = dict(clip_model="openai/clip-vit-base-patch32", clip_revision=revision,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        packages={p:version(p) for p in ["torch","torchvision","transformers","lpips","scikit-image","image-reward","numpy"]},
        clip_definition="100 * max(0, cosine(image,text)); CLIP truncates text at 77 tokens",
        lpips="alex, v0.1, RGB [-1,1], original resolution", image_reward="ImageReward-v1.0" if reward else None,
        image_reward_revision=reward_revision,
        energy_scope=power["scope"] if power else None, geneval=None, human_preference=None,
        measurements_sha256=hashlib.sha256((args.out / "measurements.jsonl").read_bytes()).hexdigest())
    (args.out / "scoring.json").write_text(json.dumps(info, indent=2))
    def tensor(im):
        return torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2,0,1)[None] / 127.5 - 1
    with torch.inference_mode(), (args.out / "scores.jsonl").open("w") as f:
        for r in rows:
            image = Image.open(args.out / r["image"]).convert("RGB")
            ref = refs[(r["repeat"], r["prompt_id"], r["seed"])]
            reference = Image.open(args.out / ref["image"]).convert("RGB")
            metrics = similarity(reference, image)
            metrics["image_sha256"] = hashlib.sha256((args.out / r["image"]).read_bytes()).hexdigest()
            metrics["lpips"] = float(perceptual(tensor(reference), tensor(image)).item())
            inputs = processor(text=[r["prompt"]], images=image, return_tensors="pt", padding=True, truncation=True)
            prediction = clip(**inputs)
            metrics["clip_score"] = 100 * max(0, float(torch.nn.functional.cosine_similarity(
                prediction.image_embeds, prediction.text_embeds).item()))
            metrics["image_reward"] = float(reward.score(r["prompt"], image)) if reward else None
            metrics["energy_j"] = None
            metrics["energy_status"] = "not measured"
            if power:
                try:
                    metrics["energy_j"] = integrate_energy(power["samples"], r["started_unix"], r["ended_unix"])
                    metrics["energy_status"] = "measured estimated CPU+GPU energy"
                except ValueError as error:
                    metrics["energy_status"] = str(error)
            f.write(json.dumps(dict(key=r["key"], **metrics), allow_nan=False)+"\n"); f.flush()
            print(r["key"], metrics, flush=True)


if __name__ == "__main__":
    main()
