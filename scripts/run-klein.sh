#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p outputs
# Override steps for a smoke test: bash scripts/run-klein.sh --steps 4
/usr/bin/time -l .venv/bin/mflux-generate-flux2 \
  --model "$PWD/models/klein-base-4b-q4" \
  --base-model flux2-klein-base-4b \
  --prompt-file prompts/klein-test.txt \
  --width 512 --height 512 \
  --steps 50 --guidance 3.5 --seed 42 \
  --compute-precision float16 \
  --low-ram --mlx-cache-limit-gb 2 \
  --make-conf --output outputs/klein-test.png \
  "$@" 2>&1 | tee outputs/klein-test.log
