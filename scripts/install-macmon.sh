#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools/macmon
curl -fL https://github.com/vladkens/macmon/releases/download/v0.10.0/macmon-v0.10.0.tar.gz \
  -o .tools/macmon/release.tar.gz
expected=0ccff60b7bf67bcb9e09cc8f804656a1cafa7eeae7297ee83d10e460ead7d43d
actual=$(shasum -a 256 .tools/macmon/release.tar.gz | cut -d ' ' -f 1)
if [[ "$actual" != "$expected" ]]; then
  echo "macmon archive checksum mismatch" >&2
  exit 1
fi
tar -xzf .tools/macmon/release.tar.gz -C .tools/macmon macmon readme.md LICENSE
