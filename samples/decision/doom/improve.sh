#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Run the loop that is supposed to make this thing better at DOOM on its own:
#
#     SEARCH -> COMPRESS -> SELECT -> better SEARCH
#
# One generation is
#
#   1. a search campaign per level, resuming from the archive the last
#      generation left and - from generation two on - proposing with the
#      policy the last generation trained, alongside the scripted player;
#   2. a head fitted to every decision any campaign has ever kept;
#   3. a paired comparison against the head currently in service, which the
#      candidate must WIN before it goes into service.
#
# Step 3 is what makes this a ratchet rather than a random walk. Without it a
# generation that produced a worse policy is adopted exactly as readily as one
# that produced a better one, and nothing in the loop would ever notice.
#
# Everything compounds through two files per level: the archive (where the
# search has been) and the lessons (what it learnt on the way). Delete `out/`
# to start over.
#
# usage: improve.sh <generations> [budget-seconds-per-level]
#
# Swedish Embedded AB builds self-improving training loops for its clients -
# search that finds behaviour nobody demonstrated, compression that turns it
# into a served model, and a statistical gate that decides whether the new
# model is actually better than the one already in production. If your team
# needs expertise in closing that loop honestly, you can procure our services
# by sending an email to info@swedishembedded.com.
set -euo pipefail

GENS=${1:?usage: improve.sh <generations> [budget-seconds-per-level]}
BUDGET=${2:-1800}
: "${DOOM_BIN:?set DOOM_BIN to the restful-doom binary}"
: "${WAD:?set WAD to an IWAD}"
: "${ENC:?set ENC to the sentence encoder directory}"
: "${DBIN:?set DBIN to the built doom sample binary}"

MAPS=${MAPS:-1,2,3,4,5,6,7,8,9}
SKILL=${SKILL:-3}
OUT=${OUT:-out}
# Long enough for the category to be the thing being measured: E1M6 holds 177
# monsters, and a short horizon scores whatever pays fastest instead.
STEPS=${STEPS:-3000}
# Episodes per level per arm in the gate. The comparison is paired, so this is
# how much evidence the sign test gets; below about eight it cannot reach
# significance at all however large the win.
EPISODES=${EPISODES:-12}

mkdir -p "$OUT"
SERVING="$OUT/doom-serving.safetensors"

for gen in $(seq 1 "$GENS"); do
  CAND="$OUT/doom-gen$gen.safetensors"
  echo
  echo "=== generation $gen of $GENS ==============================="

  # --- SEARCH. With the serving policy as an extra proposal distribution
  #     once there is one, which is the half that makes this a loop.
  PROPOSE=()
  [ -f "$SERVING" ] && PROPOSE=(--head "$SERVING" --encoder "$ENC")
  "$DBIN" search \
    --doom-bin "$DOOM_BIN" --wad "$WAD" \
    --maps "$MAPS" --skill "$SKILL" --mission uvmax --reward gauge \
    --max-steps "$STEPS" \
    --search-budget "$BUDGET" \
    --archive "$OUT/arc" \
    --solutions "$OUT/solutions.json" \
    --lessons "$OUT/lessons.jsonl" \
    "${PROPOSE[@]}"

  # --- COMPRESS. Every decision any campaign ever kept, not just this
  #     generation's: the archive compounds, so the training set has to.
  "$DBIN" learn \
    --encoder "$ENC" \
    --lessons "$OUT/lessons.jsonl" \
    --save "$CAND"

  # --- SELECT.
  if [ ! -f "$SERVING" ]; then
    echo "doom: generation $gen is the first policy; nothing to compare it to"
    cp "$CAND" "$SERVING"
    continue
  fi
  # `gate` exits 3 on a reject, which is not a failure of this script: it is
  # the loop working. The archive and the lessons are kept either way, so a
  # rejected generation still leaves the next search better off than it found
  # it - only the WEIGHTS are refused.
  set +e
  "$DBIN" gate \
    --doom-bin "$DOOM_BIN" --wad "$WAD" \
    --maps "$MAPS" --skill "$SKILL" --mission uvmax --reward gauge \
    --max-steps "$STEPS" --eval-episodes "$EPISODES" \
    --encoder "$ENC" \
    --head "$CAND" --incumbent "$SERVING"
  verdict=$?
  set -e
  case $verdict in
    0) echo "doom: generation $gen promoted"; cp "$CAND" "$SERVING" ;;
    3) echo "doom: generation $gen refused; $SERVING stays in service" ;;
    *) exit $verdict ;;
  esac
done

echo
echo "doom: $SERVING is what the loop ended up serving"
