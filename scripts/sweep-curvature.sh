#!/usr/bin/env bash
# Damped second-order Taylor sweep. Every damped config uses the same fixed interval-3
# refresh schedule, so all variants run identical full steps on identical prompts and seeds
# in one model session. taylor-3 is the first-order control that taylor2-3-c0.0 should match.
#
# Usage:
#   bash scripts/sweep-curvature.sh quick [OUT_DIR] [-- evaluate args]
#       All six weights + taylor-3 + full-50 on prompt 0, seed 42.   Default OUT: outputs/curvature-quick
#   bash scripts/sweep-curvature.sh validate WEIGHT... [--out OUT_DIR] [-- evaluate args]
#       Selected weights + taylor-3 + full-50 on every prompt.        Default OUT: outputs/curvature-validate-c<w>...
#   bash scripts/sweep-curvature.sh full [OUT_DIR] [-- evaluate args]
#       All six weights on every prompt (3–4 hours on an M4).        Default OUT: outputs/curvature-sweep
#   bash scripts/sweep-curvature.sh score OUT_DIR
#       Rescore, report, and summarize existing images without generating.
#
# Each mode records its own experiment label in the identity, so quick-screen and validation results
# cannot share a directory. Rerunning a directory resumes it: completed trials are validated against
# their image hashes and skipped. full-50 images are reused from outputs/baseline-cache only when the
# prompt, seed, settings, packages, hardware, and generation source hash all match.
# Stages: generation (.venv), scoring (.venv-quality, ImageReward), report, summary.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_WEIGHTS=(0.0 0.1 0.25 0.5 0.75 1.0)
usage() { sed -n '6,15p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

MODE="${1:-}"
shift || true
OUT=""
WEIGHTS=()
EXTRA=()
case "$MODE" in
  quick|full)
    if [[ $# -gt 0 && "$1" != "--" ]]; then OUT="$1"; shift; fi
    WEIGHTS=("${ALL_WEIGHTS[@]}") ;;
  validate)
    while [[ $# -gt 0 && "$1" != "--" ]]; do
      case "$1" in
        --out) OUT="$2"; shift 2 ;;
        *) [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "Not a curvature weight: $1" >&2; exit 2; }
           WEIGHTS+=("$1"); shift ;;
      esac
    done
    [[ ${#WEIGHTS[@]} -gt 0 ]] || { echo "validate needs at least one weight, e.g. validate 0.25 0.5" >&2; exit 2; } ;;
  score)
    [[ $# -ge 1 ]] || usage
    OUT="$1"; shift ;;
  *) usage ;;
esac
if [[ $# -gt 0 && "$1" == "--" ]]; then shift; EXTRA=("$@"); fi

# Weight names match taylor2-3-c<w>; 0 and 1 keep the established 0.0 and 1.0 spelling.
name() { case "$1" in 0|0.0) echo 0.0 ;; 1|1.0) echo 1.0 ;; *) echo "$1" ;; esac; }
CONFIGS=(full-50 taylor-3)
for w in ${WEIGHTS[@]+"${WEIGHTS[@]}"}; do CONFIGS+=("taylor2-3-c$(name "$w")"); done

case "$MODE" in
  quick) OUT="${OUT:-outputs/curvature-quick}"; ARGS=(--experiment curvature-quick --prompt-ids 0 --seeds 42) ;;
  validate)
    LABEL="$(printf 'c%s_' "${CONFIGS[@]:2}" | sed 's/taylor2-3-c//g; s/_$//')"
    OUT="${OUT:-outputs/curvature-validate-$LABEL}"; ARGS=(--experiment curvature-validate) ;;
  full) OUT="${OUT:-outputs/curvature-sweep}"; ARGS=(--experiment curvature-full) ;;
esac
case "$OUT" in
  reports/*|results/*|./reports/*|./results/*) echo "Refusing to write into tracked results directory: $OUT" >&2; exit 1 ;;
esac
[[ -x .venv-quality/bin/python ]] || { echo "Missing .venv-quality; create it as described in README.md before scoring" >&2; exit 1; }

if [[ "$MODE" != score ]]; then
  .venv/bin/python scripts/evaluate-klein.py --out "$OUT" --configs "${CONFIGS[@]}" "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
else
  [[ -f "$OUT/measurements.jsonl" ]] || { echo "No measurements in $OUT" >&2; exit 1; }
fi
.venv-quality/bin/python scripts/score-evaluation.py --out "$OUT" --image-reward
.venv/bin/python scripts/report-evaluation.py --out "$OUT" --report "$OUT/report"
.venv/bin/python scripts/summarize-curvature-sweep.py --out "$OUT"
