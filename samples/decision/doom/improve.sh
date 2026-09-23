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
# Everything compounds through the archives (where each search has been, one
# per level per seed) and the lessons (what all of them learnt, pooled into
# one file). Delete `out/` to start over.
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
# How many independent searches to run per level per generation.
#
# Each gets its own archive, because a trail is only a way back to a cell on
# the episode seed it was walked on - the seed reaches the engine, so the
# monsters do different things and the same actions lead somewhere else.
# Pooling the archives is therefore not an option; pooling what they LEARNT
# is, and that is what the shared lessons file does.
#
# Worth doing because the spread between seeds is wide: measured on E1M1 at
# one budget, the same binary scored 0.221 on one seed and 1.039 on another.
# A single seed makes a generation's result hostage to that draw. Several
# seeds buy independent draws at a rare event - a secret - at the cost of
# each one searching less deeply, which is the right trade early and the
# wrong one once a level is nearly solved.
SEEDS=${SEEDS:-3}
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
  for seed in $(seq 1 "$SEEDS"); do
    echo
    echo "--- generation $gen, seed $seed of $SEEDS ---"
    # One archive per seed, carried across generations under its own name.
    # Solutions are written per seed too and pooled below, because the best
    # run of a level may come from any of them.
    "$DBIN" search \
      --doom-bin "$DOOM_BIN" --wad "$WAD" \
      --maps "$MAPS" --skill "$SKILL" --mission uvmax --reward gauge \
      --max-steps "$STEPS" \
      --seed "$seed" \
      --search-budget "$BUDGET" \
      --archive "$OUT/arc-s$seed" \
      --solutions "$OUT/solutions-s$seed.json" \
      --lessons "$OUT/lessons.jsonl" \
      "${PROPOSE[@]}"
  done

  # --- COMPRESS. Every decision any campaign ever kept, from every seed and
  #     every generation: the archives compound separately and what they
  #     learnt pools here, which is the one place the seeds meet.
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
