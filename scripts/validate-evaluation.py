"""Independently validate a saved report's coverage, identities, and calculations."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.metrics import integrate_energy

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report",type=Path,default=ROOT/"reports/klein-ablation-pilot")
args=parser.parse_args()
root=args.report
identity=json.loads((root/"manifest.json").read_text())["identity"]
rows=[json.loads(line) for line in (root/"measurements.jsonl").read_text().splitlines()]
scores={r["key"]:r for r in map(json.loads,(root/"scores.jsonl").read_text().splitlines())}
expected={f"r{rep}-p{pid}-s{seed}-{name}" for rep in range(identity["repeats"])
          for pid in range(len(identity["prompts"])) for seed in identity["seeds"] for name in identity["configs"]}
assert {r["key"] for r in rows}==expected==set(scores)
assert len(rows)==len(expected)
scoring=json.loads((root/"scoring.json").read_text())
assert scoring["measurements_sha256"]==hashlib.sha256((root/"measurements.jsonl").read_bytes()).hexdigest()
power=json.loads((root/"power.json").read_text()) if (root/"power.json").exists() else None
phase_peaks=[f"{phase}_peak_mlx_gb" for phase in ("encode", "denoise", "decode")]
for r in rows:
    assert r["prompt"]==identity["prompts"][r["prompt_id"]]
    assert abs(r["total_s"]-sum(r[k] for k in ["encode_s","denoise_s","decode_s"]))<1e-6
    assert r["full_steps"]+r["accepted_reuse_steps"]==r["sampling_steps"]
    if "encode_peak_mlx_gb" in r:
        assert r["peak_mlx_gb"]==max(r[key] for key in phase_peaks)
    # Guidance policies can leave the unconditional branch out, so each step has one or two passes.
    guided = r.get("guided_steps", r["sampling_steps"])
    if guided == r["sampling_steps"]:
        assert r["full_transformer_passes"]==2*r["full_steps"]
        assert r["verification_block_passes"]==2*r["checked_steps"]
    else:
        assert r["full_steps"]<=r["full_transformer_passes"]<=2*r["full_steps"]
        assert r["checked_steps"]<=r["verification_block_passes"]<=2*r["checked_steps"]
    assert r["rejected_steps"]<=r["checked_steps"]
    s=scores[r["key"]]
    assert s["image_sha256"]==hashlib.sha256((root/r["image"]).read_bytes()).hexdigest()
    assert all(math.isfinite(s[k]) for k in ["lpips","ssim","clip_score"])
    if r["config"]=="full-50":
        assert s["identical_pixels"] and s["ssim"]==1 and abs(s["lpips"])<1e-8
    if s["energy_j"] is not None:
        assert abs(s["energy_j"]-integrate_energy(power["samples"],r["started_unix"],r["ended_unix"]))<1e-6
summary=json.loads((root/"summary.json").read_text())
baseline=next(r for r in summary if r["config"]=="full-50")["total_s"]
energy_sets=[{(r["repeat"],r["prompt_id"],r["seed"]) for r in rows
              if r["config"]==name and scores[r["key"]]["energy_j"] is not None} for name in identity["configs"]]
common=set.intersection(*energy_sets)
for s in summary:
    group=[r for r in rows if r["config"]==s["config"]]
    mean=sum(r["total_s"] for r in group)/len(group)
    assert abs(s["total_s"]-mean)<1e-9
    assert abs(s["speedup"]-baseline/mean)<1e-9
    energies=[scores[r["key"]]["energy_j"] for r in group if (r["repeat"],r["prompt_id"],r["seed"]) in common]
    assert s["energy_n"]==len(energies)
    if energies:
        assert abs(s["energy_j"]-sum(energies)/len(energies))<1e-6
    for key in phase_peaks:
        if key in group[0]:
            assert math.isclose(s[key], sum(r[key] for r in group)/len(group))
profile_checks=[]
if identity.get("profile"):
    profile=json.loads((root/"profile.json").read_text())
    assert profile["mode"]==identity["profile"]
    assert profile["prompt_id"]==0 and profile["prompt"]==identity["prompts"][0]
    assert profile["seed"]==identity["seeds"][0]
    assert set(profile["configs"])==set(identity["configs"])
    for name, diagnostic in profile["configs"].items():
        assert diagnostic["sampling_steps"]==next(r["sampling_steps"] for r in rows if r["config"]==name)
        assert 0 < diagnostic["full_steps"] <= diagnostic["sampling_steps"]
        assert diagnostic["paths"]["transformer"]["calls"] >= diagnostic["full_transformer_passes"]
    profile_checks.append("profile coverage and transformer call counts")
result=dict(generations=len(rows),matched_energy_pairs=sorted(common),matched_energy_images=len(common)*len(summary),
    status="passed",checks=["complete unique trial keys","prompt identities","phase timing sums","CFG computation counters",
        "image hashes","self-reference metrics","energy reintegration","paired energy aggregation","speedup recalculation",
        *(["phase memory peaks and averages"] if phase_peaks[0] in rows[0] else []), *profile_checks])
(root/"validation.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
