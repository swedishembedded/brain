#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset preparation pipelines for
# multimodal detector studies for its clients. If your team needs expertise
# in RGB / thermal-IR detector training data, you can procure our services by
# sending an email to info@swedishembedded.com.

# Driver for the data stages of the rgb-to-ir study. Stages share one work
# directory ($RIR_WORK, default ${TMPDIR:-/tmp}/rgb-to-ir); nothing is written
# elsewhere.
#
#   rgb_to_ir.sh manifest --dataset ID --rgb-glob G ...  -> $RIR_WORK/manifests/ID.jsonl
#   rgb_to_ir.sh validate [--check-files]                check every manifest against the contract
#   rgb_to_ir.sh splits                                  -> $RIR_WORK/splits.json (+ leaks.json)
#   rgb_to_ir.sh arms [--limit N] [--arms a1,a2,b1,b2,b3,b4] [--datasets id,...]
#                                                        -> sensor-model.json, arms/<arm>/
#   rgb_to_ir.sh pack <arm> [--size 512] [--limit N]     -> packed/<arm>/ for `brain yolov8 fine-tune`
#   rgb_to_ir.sh test                                    unit tests of the stages (no data needed)
#
# `manifest` takes the flags of rir_readers.py (see its --help). SEED (default 1)
# seeds splits, sensor fit, frame selection and packing order.

set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WORK="${RIR_WORK:-${TMPDIR:-/tmp}/rgb-to-ir}"
SEED="${SEED:-1}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"

usage() { sed -n '10,24p' "$0"; exit "${1:-2}"; }
[ $# -ge 1 ] || usage
cmd="$1"; shift
mkdir -p "$WORK/manifests"

manifests() {
  local found=("$WORK"/manifests/*.jsonl)
  [ -e "${found[0]}" ] || { echo "no manifests in $WORK/manifests: run '$0 manifest' first" >&2; exit 1; }
  printf '%s\n' "${found[@]}"
}

case "$cmd" in
  manifest)
    # --out defaults to manifests/<dataset>.jsonl, so it is derived from --dataset here.
    ds=""
    prev=""
    for a in "$@"; do [ "$prev" = "--dataset" ] && ds="$a"; prev="$a"; done
    [ -n "$ds" ] || { echo "manifest needs --dataset ID" >&2; exit 2; }
    exec python3 "$HERE/rir_readers.py" --out "$WORK/manifests/$ds.jsonl" "$@"
    ;;
  validate)
    mapfile -t files < <(manifests)
    exec python3 "$HERE/rir_manifest.py" validate "${files[@]}" "$@"
    ;;
  splits)
    mapfile -t files < <(manifests)
    exec python3 "$HERE/rir_splits.py" --manifest "${files[@]}" --out "$WORK/splits.json" \
      --leak-audit "$WORK/leaks.json" --seed "$SEED" "$@"
    ;;
  arms)
    [ -f "$WORK/splits.json" ] || { echo "no $WORK/splits.json: run '$0 splits' first" >&2; exit 1; }
    python3 "$HERE/rir_arms.py" fit-sensor --splits "$WORK/splits.json" --out "$WORK/sensor-model.json" --seed "$SEED"
    exec python3 "$HERE/rir_arms.py" render --splits "$WORK/splits.json" \
      --sensor-model "$WORK/sensor-model.json" --out "$WORK/arms" --seed "$SEED" "$@"
    ;;
  pack)
    arm="${1:?usage: $0 pack <arm> [--size N] [--limit N]}"; shift
    [ -f "$WORK/arms/$arm/manifest.jsonl" ] || { echo "no rendered arm '$arm': run '$0 arms' first" >&2; exit 1; }
    exec python3 "$HERE/rir_pack.py" --manifest "$WORK/arms/$arm/manifest.jsonl" \
      --out "$WORK/packed/$arm" --seed "$SEED" "$@"
    ;;
  test)
    exec python3 -m unittest discover -s "$HERE/tests" "$@"
    ;;
  -h|--help) usage 0 ;;
  *) echo "unknown stage: $cmd" >&2; usage ;;
esac
