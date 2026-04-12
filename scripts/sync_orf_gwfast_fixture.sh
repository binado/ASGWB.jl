#!/usr/bin/env bash
# Copy the GWFast ORF NPZ reference from the Python `asgwb` package tree (repo
# directory is often named `asgbw` on disk). Override with ASGWB_PY_REPO.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ASGWB_PY_REPO:-${HOME}/work/research/phd/asgbw}/tests/fixtures/gwfast_orf_reference.npz"
DST="${ROOT}/test/fixtures/orf_gwfast_reference.npz"
if [[ ! -f "${SRC}" ]]; then
  echo "Source fixture not found: ${SRC}" >&2
  echo "Set ASGWB_PY_REPO to your asgwb/asgbw checkout (parent of src/asgwb)." >&2
  exit 1
fi
cp "${SRC}" "${DST}"
echo "Updated ${DST} from ${SRC}"
