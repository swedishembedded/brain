#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Walk-forward backtest orchestrator: forecast the universe with the BASE and the
# FINE-TUNED decoders over the same held-out weeks (in-process argmax harness), then
# render the cumulative-return proof vs the SP500 (tools/backtest_diagram.py).
#
# Env:
#   DATA      csv dir (the backtest ranking universe, e.g. out/sp500-bt)
#   TOK       kronos tokenizer dir
#   BASE_DEC  base decoder dir (HF)
#   FT_DEC    fine-tuned decoder (.weights)
#   DB        trademiner stocks.db (for the ^gspc SP500 line)
#   OUT       output html (default out/backtest.html)
#   CTX/START/STEP/HOR  window shape + origin schedule (defaults below)
set -euo pipefail
BRAIN="${BRAIN_DIR:-/data/workspace/applications/edgeai/brain}"
: "${DATA:?set DATA=<csv dir>}"; : "${TOK:?set TOK}"; : "${BASE_DEC:?set BASE_DEC}"; : "${FT_DEC:?set FT_DEC}"; : "${DB:?set DB}"
OUT="${OUT:-$BRAIN/out/backtest.html}"
CTX="${CTX:-64}"; START="${START:-240}"; STEP="${STEP:-5}"; HOR="${HOR:-5}"
mkdir -p "$(dirname "$OUT")"
cd "$BRAIN"

run() { # $1 decoder  $2 out.json
  RANKIC_DATA="$DATA" RANKIC_OUT="$2" RANKIC_CTX="$CTX" RANKIC_HORIZON="$HOR" \
  RANKIC_STEP="$STEP" RANKIC_START="$START" RANKIC_ARGMAX=1 RANKIC_TICKERS=all \
  KRONOS_TOKENIZER_DIR="$TOK" KRONOS_DECODER_DIR="$1" \
    cargo test --release -p brain-kronos --test rankic_eval rankic_backtest -- --exact --nocapture 2>&1 \
    | grep -E "origin|wrote|records" | tail -3
}

echo "[backtest] base forecasts ($BASE_DEC) ..."
run "$BASE_DEC" /tmp/bt_base.json
echo "[backtest] fine-tuned forecasts ($FT_DEC) ..."
run "$FT_DEC" /tmp/bt_ft.json
echo "[backtest] rendering $OUT ..."
python3 "$BRAIN/tools/backtest_diagram.py" /tmp/bt_base.json /tmp/bt_ft.json "$DB" "$OUT"
