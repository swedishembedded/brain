#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Full-universe walk-forward validation: prep -> fine-tune (<= T0) -> sharded
# base+ft sweep -> scored report + trademiner-compatible summary.
#
# Replaces the old tools/backtest.sh orchestration (rankic_eval was INDEX-keyed
# and silently misaligns calendars at full-universe scale; the oos_skill_eval
# harness keys origins by date).
#
# Env (defaults follow the pre-registered protocol in trademiner
# docs/validation-criteria.md):
#   DB=stocks.db  OUT=out/bt  TOK=<tokenizer dir>  BASE_DEC=<decoder dir>
#   NAMES=0 (full universe)  BT_BARS=400  FT_BARS=400  OOS_BARS=260  EMBARGO=10
#   SEED=7  CTX=120  HOR=5  STEP=5  NSAMPLES=3  START=<auto: manifest first_origin>
#   SHARDS=(cores-2)  EPOCHS=2  LR=5e-4  LORA=8  FT_DEVICE=vulkan
#   SUMMARY=out/backtest_summary.json
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"

DB="${DB:-stocks.db}"
OUT="${OUT:-out/bt}"
TOK="${TOK:?set TOK=<kronos tokenizer dir>}"
BASE_DEC="${BASE_DEC:?set BASE_DEC=<kronos decoder dir>}"
NAMES="${NAMES:-0}"
BT_BARS="${BT_BARS:-400}"
FT_BARS="${FT_BARS:-400}"
OOS_BARS="${OOS_BARS:-260}"
EMBARGO="${EMBARGO:-10}"
SEED="${SEED:-7}"
CTX="${CTX:-120}"
HOR="${HOR:-5}"
STEP="${STEP:-5}"
NSAMPLES="${NSAMPLES:-3}"
SHARDS="${SHARDS:-$(( $(nproc) - 2 ))}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-4}"
LORA="${LORA:-8}"
FT_DEVICE="${FT_DEVICE:-vulkan}"
SUMMARY="${SUMMARY:-out/backtest_summary.json}"

mkdir -p "$OUT"

echo "== [1/4] prep: leak-free stratified split =="
python3 "$HERE/prep_backtest_data.py" --db "$DB" --out "$OUT" \
    --names "$NAMES" --bt-bars "$BT_BARS" --ft-bars "$FT_BARS" \
    --oos-bars "$OOS_BARS" --embargo "$EMBARGO" --seed "$SEED"

# First OOS origin comes from the prep manifest: embargo bars after T0.
T0="$(python3 -c "import json;print(json.load(open('$OUT/split_manifest.json'))['t0'])")"
START="${START:-$(python3 -c "import json;print(json.load(open('$OUT/split_manifest.json'))['first_origin'])")}"
echo "T0=$T0  first OOS origin=$START"

echo "== [2/4] fine-tune on the ft-half (data <= T0, gated) =="
"$ROOT/target/release/brain" --device "$FT_DEVICE" forecast finetune \
    --kronos-tokenizer "$TOK" --kronos-decoder "$BASE_DEC" \
    --data "$OUT/ft" --holdout-data "$OUT/holdout" \
    --context "$CTX" --horizon "$HOR" --epochs "$EPOCHS" --lr "$LR" --lora "$LORA" \
    --out "$OUT/ft.weights"

echo "== [3/4] sharded base+ft sweep ($SHARDS shards) =="
KRONOS_TOKENIZER_DIR="$TOK" KRONOS_DECODER_DIR="$BASE_DEC" \
python3 "$HERE/oos_shard.py" --data "$OUT/bt" --db "$DB" --out "$OUT/eval.json" \
    --shards "$SHARDS" --ctx "$CTX" --horizon "$HOR" --step "$STEP" \
    --nsamples "$NSAMPLES" --start "$START" \
    --kronos-ft "$OUT/ft.weights" --split-manifest "$OUT/split_manifest.json" \
    --summary-out "$SUMMARY" --k-frac 0.10

echo "== [4/4] summary =="
python3 -c "import json;d=json.load(open('$SUMMARY'));print(json.dumps(d.get('verdict'),indent=2))"
echo "wrote $SUMMARY"
