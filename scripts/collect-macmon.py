"""Collect non-root CPU+GPU telemetry; stop cleanly by creating --stop-file."""
import argparse
from datetime import datetime
import json
import hashlib
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", default=".tools/macmon/macmon")
parser.add_argument("--out", type=Path, default=Path("outputs/evaluation/power.json"))
parser.add_argument("--stop-file", type=Path, default=Path("outputs/evaluation/stop-power"))
parser.add_argument("--recover", action="store_true", help="Finalize an interrupted collection from the flushed JSONL")
args = parser.parse_args()


def finalize(telemetry, recovered=False):
    args.out.write_text(json.dumps(dict(scope="macmon v0.10.0 estimated system CPU+GPU power; background included; excludes other components; no idle subtraction",
        binary_sha256=hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
        recovered_from_flushed_jsonl=recovered, interval_ms=500,
        timestamp_convention="source timestamp minus 250 ms; linear interpolation of interval averages",
        samples=[[p["unix_s"],p["power_w"]] for p in telemetry], telemetry=telemetry)))


if args.recover:
    finalize([json.loads(line) for line in args.out.with_suffix(".jsonl").read_text().splitlines()], recovered=True)
    raise SystemExit(0)
if args.stop_file.exists():
    raise SystemExit("Remove the existing stop file before collecting")
args.out.parent.mkdir(parents=True, exist_ok=True)
proc = subprocess.Popen([args.binary, "pipe", "-i", "500"], stdout=subprocess.PIPE, text=True)
samples, telemetry = [], []
try:
    with args.out.with_suffix(".jsonl").open("w") as f:
        for line in proc.stdout:
            row = json.loads(line)
            # macmon reports interval-averaged power. Place each 500 ms average
            # at its approximate interval midpoint; retain source timestamps.
            timestamp = datetime.fromisoformat(row["timestamp"]).timestamp()
            power = row["cpu_power"] + row["gpu_power"]
            point = dict(unix_s=timestamp-.25, source_timestamp=row["timestamp"], power_w=power,
                cpu_w=row["cpu_power"], gpu_w=row["gpu_power"],
                gpu_temp_c=row["temp"]["gpu_temp_avg"], memory=row["memory"])
            samples.append([point["unix_s"],power]); telemetry.append(point)
            f.write(json.dumps(point)+"\n"); f.flush()
            if args.stop_file.exists():
                break
finally:
    proc.terminate(); proc.wait(timeout=10)
    finalize(telemetry)
