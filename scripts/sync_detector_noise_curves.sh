#!/usr/bin/env bash
# Copy tabulated PSD files from the Python `asgwb` tree (repo directory is often
# `asgbw` on disk). Override with ASGWB_PY_REPO.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ASGWB_PY_REPO:-${HOME}/work/research/phd/asgbw}/src/asgwb/detector/noise_curves"
DST="${ROOT}/assets/detector/noise_curves"
if [[ ! -d "${SRC}" ]]; then
  echo "Noise curve directory not found: ${SRC}" >&2
  echo "Set ASGWB_PY_REPO to your asgwb/asgbw checkout root." >&2
  exit 1
fi
mkdir -p "${DST}"
shopt -s nullglob
files=("${SRC}"/*.txt)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .txt curves under ${SRC}" >&2
  exit 1
fi
cp -f "${SRC}"/*.txt "${DST}/"
echo "Copied ${#files[@]} noise curve file(s) to ${DST}"
