import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import ImageReward as RM
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from transformers import CLIPModel, CLIPProcessor

REFERENCE = "full-50"
CLIP_MODEL = "openai/clip-vit-large-patch14"
METRICS = ["psnr", "ssim", "lpips", "clip", "imagereward"]


def to_tensor(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels).permute(2, 0, 1)[None].float() / 127.5 - 1


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "benchmark.txt")
    args = parser.parse_args()

    prompts = args.prompts.read_text().splitlines()
    lpips_model = lpips.LPIPS(net="alex", verbose=False)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    reward_model = RM.load("ImageReward-v1.0", device="cpu", download_root=str(ROOT / ".cache" / "ImageReward"))

    def path(row: dict, config: str) -> Path:
        return args.out / "images" / f"p{row['prompt_id']}-s{row['seed']}-{config}.png"

    def load(row: dict, config: str) -> np.ndarray:
        return np.asarray(Image.open(path(row, config)).convert("RGB"))

    def clip_score(image: np.ndarray, prompt: str) -> float:
        inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", truncation=True)
        outputs = clip_model(**inputs)
        return 100 * torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()

    with (args.out / "results.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    by_config = defaultdict(list)
    for row in rows:
        image = load(row, row["config"])
        prompt = prompts[int(row["prompt_id"])]
        row["clip"] = clip_score(image, prompt)
        row["imagereward"] = reward_model.score(prompt, [str(path(row, row["config"]))])
        if row["config"] != REFERENCE:
            reference = load(row, REFERENCE)
            mse = np.mean((image.astype(np.float64) - reference) ** 2)
            row["psnr"] = 10 * np.log10(255**2 / mse)
            row["ssim"] = structural_similarity(image, reference, channel_axis=2)
            row["lpips"] = lpips_model(to_tensor(image), to_tensor(reference)).item()
        by_config[row["config"]].append(row)

    with (args.out / "scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["prompt_id", "seed", "config", *METRICS], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'config':16} {'runs':>4} {'full':>5} {'time_s':>7} {'cache_mb':>9} " + " ".join(f"{m:>11}" for m in METRICS))
    for name, group in by_config.items():
        mean = lambda field: np.mean([float(row.get(field, np.nan)) for row in group])
        print(
            f"{name:16} {len(group):4} {mean('full_steps'):5.1f} {mean('denoise_s'):7.1f} {mean('cache_mb'):9.1f} "
            + " ".join(f"{mean(m):11.3f}" for m in METRICS)
        )


if __name__ == "__main__":
    main()
