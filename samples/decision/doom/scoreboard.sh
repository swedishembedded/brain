#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Score one policy on every level of the episode and append a row per level to
# a CSV, so that progress can be watched per LEVEL across generations rather
# than as one averaged number that hides which levels moved.
#
# Generation 0 is the scripted teacher: pass no head.
#
# usage: scoreboard.sh <csv> <generation> [head.safetensors]
set -u
CSV=$1; GEN=$2; HEAD=${3:-}
: "${DOOM_BIN:?}" "${WAD:?}" "${ENC:?}" "${DBIN:?}"
SKILL=${SKILL:-3}
STEPS=${STEPS:-600}
EPISODES=${EPISODES:-2}
WHO=scripted
ARGS=()
if [ -n "$HEAD" ]; then WHO=policy; ARGS=(--head "$HEAD"); fi
[ -f "$CSV" ] || echo "generation,map,who,progress,kills,items,exits,deaths,steps" > "$CSV"
for M in 1 2 3 4 5 6 7 8 9; do
  line=$("$DBIN" eval --doom-bin "$DOOM_BIN" --wad "$WAD" --encoder "$ENC" \
    --map "$M" --skill "$SKILL" --mission clear --reward gauge \
    --max-steps "$STEPS" --eval-episodes "$EPISODES" "${ARGS[@]}" \
    --seed 11 --device "${GPU:-gpu0}" 2>/dev/null | grep -E "^  $WHO: " | tail -1)
  # "N episodes, return +X (...), K kills, I items, E exits, D deaths, S steps, ..., progress P"
  prog=$(sed -E 's/.*progress ([-0-9.]+).*/\1/'   <<<"$line")
  kills=$(sed -E 's/.*\), ([0-9.]+) kills.*/\1/'  <<<"$line")
  items=$(sed -E 's/.* ([0-9.]+) items.*/\1/'     <<<"$line")
  exits=$(sed -E 's/.* ([0-9]+) exits.*/\1/'      <<<"$line")
  deaths=$(sed -E 's/.* ([0-9]+) deaths.*/\1/'    <<<"$line")
  steps=$(sed -E 's/.* ([0-9]+) steps.*/\1/'      <<<"$line")
  echo "$GEN,$M,$WHO,$prog,$kills,$items,$exits,$deaths,$steps" >> "$CSV"
  echo "  E1M$M  progress $prog  kills $kills  exits $exits"
done
