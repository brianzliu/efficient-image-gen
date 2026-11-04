import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "full-50"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark")
    args = parser.parse_args()

    def load(row: dict, config: str) -> np.ndarray:
        path = args.out / "images" / f"p{row['prompt_id']}-s{row['seed']}-{config}.png"
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)

    with (args.out / "results.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    by_config = defaultdict(list)
    for row in rows:
        if row["config"] != REFERENCE:
            mse = np.mean((load(row, row["config"]) - load(row, REFERENCE)) ** 2)
            row["psnr_db"] = 10 * np.log10(255**2 / mse)
        by_config[row["config"]].append(row)

    print(f"{'config':16} {'runs':>4} {'full':>5} {'time_s':>7} {'peak_gb':>8} {'cache_mb':>9} {'psnr_db':>8}")
    for name, group in by_config.items():
        mean = lambda field: np.mean([float(row.get(field, np.nan)) for row in group])
        print(
            f"{name:16} {len(group):4} {mean('full_steps'):5.1f} {mean('denoise_s'):7.1f} "
            f"{mean('peak_mlx_gb'):8.2f} {mean('cache_mb'):9.1f} {mean('psnr_db'):8.2f}"
        )


if __name__ == "__main__":
    main()
