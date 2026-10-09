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
#   rgb_to_ir.sh tiles [--size 512] [--limit N]          -> lora-set/ : aligned RGB / IR tiles of split T
#                                                        + pairs.yaml for `brain flux2 finetune`
#   rgb_to_ir.sh captions [--neutral-share 0.3]          -> lora-set/captions.yaml, captions-report.json (needs `measure`)
#   rgb_to_ir.sh sheet                                   -> lora-set/sheet.png : random tiles, captions, mask overlays
#   rgb_to_ir.sh evalset [--split Test|V] [--size 512] [--limit N]
#                                                        -> packed/eval-<split>/ : the REAL IR frames of that split, packed
#                                                        for `brain yolov8 eval`, with sequences.json (image index -> sequence)
#   rgb_to_ir.sh evaluate <arm> <seed> <weights> [--split Test|V]
#                                                        -> results/<split>/<arm>/seed<seed>.jsonl (+ .score.json): the
#                                                        detector scored on the eval set, predictions dumped at --conf 0.001
#   rgb_to_ir.sh decide --config FILE [--split Test|V]   -> decision-<split>/decision.md, decision.json (see rir_decide.py)
#   rgb_to_ir.sh test                                    unit tests of the stages (no data needed)
#
# `manifest` takes the flags of rir_readers.py (see its --help). SEED (default 1)
# seeds splits, sensor fit, frame selection and packing order. BRAIN (default
# `brain`) is the binary `evaluate` runs.

set -euo pipefail

HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WORK="${RIR_WORK:-${TMPDIR:-/tmp}/rgb-to-ir}"
SEED="${SEED:-1}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"

usage() { sed -n '10,34p' "$0"; exit "${1:-2}"; }
[ $# -ge 1 ] || usage
cmd="$1"; shift
mkdir -p "$WORK/manifests"

# Splits `--split NAME` (default Test) out of the arguments: sets $split and the array $rest.
take_split() {
  split="Test"
  rest=()
  while [ $# -gt 0 ]; do
    if [ "$1" = "--split" ]; then split="${2:?--split needs V or Test}"; shift 2; else rest+=("$1"); shift; fi
  done
}

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
  tiles)
    [ -f "$WORK/splits.json" ] || { echo "no $WORK/splits.json: run '$0 splits' first" >&2; exit 1; }
    exec python3 "$HERE/rir_tiles.py" --splits "$WORK/splits.json" --out "$WORK/lora-set" --seed "$SEED" "$@"
    ;;
  captions)
    [ -d "$WORK/lora-set/regions" ] || { echo "no $WORK/lora-set/regions: run '$0 measure' first" >&2; exit 1; }
    exec python3 "$HERE/rir_captions.py" "$WORK/lora-set" --seed "$SEED" "$@"
    ;;
  sheet)
    [ -f "$WORK/lora-set/captions.yaml" ] || { echo "no $WORK/lora-set/captions.yaml: run '$0 captions' first" >&2; exit 1; }
    exec python3 "$HERE/rir_sheet.py" "$WORK/lora-set" --out "$WORK/lora-set/sheet.png" --seed "$SEED" "$@"
    ;;
  evalset)
    [ -f "$WORK/splits.json" ] || { echo "no $WORK/splits.json: run '$0 splits' first" >&2; exit 1; }
    take_split "$@"
    python3 "$HERE/rir_arms.py" render --splits "$WORK/splits.json" --out "$WORK/evalsets/$split" \
      --arms a2 --split "$split" --seed "$SEED"
    exec python3 "$HERE/rir_pack.py" --manifest "$WORK/evalsets/$split/a2/manifest.jsonl" \
      --out "$WORK/packed/eval-$split" --seed "$SEED" ${rest[@]+"${rest[@]}"}
    ;;
  evaluate)
    arm="${1:?usage: $0 evaluate <arm> <seed> <weights> [--split Test|V]}"
    seed="${2:?usage: $0 evaluate <arm> <seed> <weights> [--split Test|V]}"
    weights="${3:?usage: $0 evaluate <arm> <seed> <weights> [--split Test|V]}"
    shift 3
    take_split "$@"
    data="$WORK/packed/eval-$split"
    [ -f "$data/sequences.json" ] || { echo "no $data/sequences.json: run '$0 evalset --split $split' first" >&2; exit 1; }
    out="$WORK/results/$split/$arm"
    mkdir -p "$out"
    "${BRAIN:-brain}" yolov8 eval --weights "$weights" --data "$data" --split all --conf 0.001 \
      --dump-preds "$out/seed$seed.jsonl" ${rest[@]+"${rest[@]}"}
    python3 "$HERE/rir_eval.py" "$out/seed$seed.jsonl" --json > "$out/seed$seed.score.json"
    ;;
  decide)
    take_split "$@"
    [ -d "$WORK/results/$split" ] || { echo "no $WORK/results/$split: run '$0 evaluate' for each arm and seed first" >&2; exit 1; }
    exec python3 "$HERE/rir_decide.py" "$WORK/results/$split" --out "$WORK/decision-$split" ${rest[@]+"${rest[@]}"}
    ;;
  test)
    exec python3 -m unittest discover -s "$HERE/tests" "$@"
    ;;
  -h|--help) usage 0 ;;
  *) echo "unknown stage: $cmd" >&2; usage ;;
esac
