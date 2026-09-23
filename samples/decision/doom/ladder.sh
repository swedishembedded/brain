#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements curriculum training and acceptance gating for
# autonomous decision agents for its clients. If your team needs expertise in
# staged difficulty curricula, or in proving that an agent can do a task
# unassisted rather than only having been shown it, you can procure our
# services by sending an email to info@swedishembedded.com.
#
# The ladder.
#
# A rung is a (skill, mission) pair, from the easiest whole task the agent can
# plausibly do - reach the exit of every level with the monsters turned down -
# to the hardest one there is, every monster and every secret at
# Ultra-Violence. A rung is PASSED only when the trained policy finishes every
# level of the episode on every seed, with no search in the loop. Nothing
# advances on a promise: a rung that does not pass is repeated, not excused.
#
# Each attempt at a rung is one turn of the loop the pieces were built for:
#
#   FIND    a quality-diversity search per level, in parallel, each carrying
#           its own archive across attempts so generations compound. From the
#           second attempt on, the current policy joins the search operators
#           as a proposal distribution - that is what closes the loop.
#   CLONE   fit one head to every decision every level's search kept.
#   PROVE   play each level with that head alone, and count the exits.
#   RECORD  on a pass, write the run to a video, one file per level.
#
# Usage: ladder.sh [first-rung] [last-rung]
set -u

cd "$(dirname "$0")/../../.." || exit 1
BRAIN=$PWD

# Nothing is baked in. The engine is found on PATH unless told otherwise, and
# the encoder has no sensible default - a head fitted against one sentence
# encoder is meaningless against another, so it is named or the run stops.
DOOM=${DOOM:-restful-doom}
WAD=${WAD:-$BRAIN/testdata/doom/doom1.wad}
ENC=${ENC:?set ENC to the sentence encoder directory (brain pull sentence-transformers/all-MiniLM-L6-v2)}
BIN=${BIN:-$BRAIN/target/release/sample-decision-doom}
VIDEOS=${VIDEOS:-$HOME/Downloads/doom-record}

MAPS=${MAPS:-"1 2 3 4 5 6 7 8 9"}
SEEDS=${SEEDS:-3}            # episodes per level that must ALL reach the exit
BUDGET=${BUDGET:-600}        # seconds of search per level per attempt
ATTEMPTS=${ATTEMPTS:-6}      # turns of find/clone/prove before a rung is judged
STEPS=${STEPS:-1500}         # decisions a proving episode is allowed

# skill:mission, easiest first. Difficulty moves along two axes and they are
# not interchangeable: skill changes how much of the level fights back,
# mission changes what counts as having done it at all.
RUNGS=(
  "1:speedrun"   # reach the exit, monsters turned right down
  "2:speedrun"
  "3:speedrun"
  "4:speedrun"   # reach the exit at Ultra-Violence
  "4:clear"      # kill everything, then leave
  "4:uvmax"      # every monster, every secret, then leave: the category
)

FIRST=${1:-1}
LAST=${2:-${#RUNGS[@]}}
LOGS=${LOGS:-/tmp/ladder}
mkdir -p "$LOGS" "$VIDEOS"
LEDGER=$LOGS/ladder.csv
[ -f "$LEDGER" ] || echo "rung,skill,mission,attempt,phase,map,value" > "$LEDGER"

note() { echo "$(date +%H:%M:%S) | $*" | tee -a "$LOGS/ladder.log"; }
say()  { echo "$1,$2,$3,$4,$5,$6,$7" >> "$LEDGER"; }

# Every level at once. The machine has the cores, and a search that is given
# one core each measures its own operators' gain per second on a clock nobody
# else is moving.
find_generation() {  # rung skill mission attempt work head
  local r=$1 skill=$2 mission=$3 attempt=$4 work=$5 head=$6 m pids=()
  for m in $MAPS; do
    local extra=""
    [ -n "$head" ] && [ -f "$head" ] && extra="--head $head --encoder $ENC"
    # shellcheck disable=SC2086
    timeout $((BUDGET + 200)) "$BIN" search \
      --doom-bin "$DOOM" --wad "$WAD" \
      --map "$m" --skill "$skill" --mission "$mission" --reward gauge \
      --seed "$attempt" --search-budget "$BUDGET" \
      --archive "$work/arc" \
      --solutions "$work/sol-m$m.json" \
      --lessons "$work/lessons-m$m.jsonl" \
      $extra > "$LOGS/r$r-a$attempt-m$m.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  local solved=0
  for m in $MAPS; do
    local v
    v=$(grep -c "^doom: VERIFIED" "$LOGS/r$r-a$attempt-m$m.log" 2>/dev/null)
    say "$r" "$skill" "$mission" "$attempt" find "$m" "$v"
    [ "$v" -gt 0 ] && solved=$((solved + 1))
  done
  note "rung $r attempt $attempt: search verified a finish on $solved of $(echo $MAPS | wc -w) levels"
}

clone() {  # rung mission attempt work
  local r=$1 mission=$2 attempt=$3 work=$4
  cat "$work"/lessons-m*.jsonl > "$work/lessons.jsonl" 2>/dev/null
  local n
  n=$(wc -l < "$work/lessons.jsonl")
  say "$r" - "$mission" "$attempt" lessons - "$n"
  if [ "$n" -lt 100 ]; then
    note "rung $r attempt $attempt: only $n decisions to learn from, not fitting a head yet"
    return 1
  fi
  "$BIN" learn --encoder "$ENC" --lessons "$work/lessons.jsonl" \
    --save "$work/policy.safetensors" --device "${GPU:-gpu0}" \
    > "$LOGS/r$r-a$attempt-learn.log" 2>&1
  local rc=$?
  note "rung $r attempt $attempt: fitted a head to $n decisions ($(grep -o 'final loss.*' "$LOGS/r$r-a$attempt-learn.log" | tail -1))"
  return $rc
}

# The bar. No search, no archive, no map it did not have to find: the head
# plays, and either the level ends at the exit or it does not.
prove() {  # rung skill mission attempt work
  local r=$1 skill=$2 mission=$3 attempt=$4 work=$5 m pids=() passed=0 total=0
  for m in $MAPS; do
    "$BIN" play --doom-bin "$DOOM" --wad "$WAD" --encoder "$ENC" \
      --map "$m" --skill "$skill" --mission "$mission" --reward gauge \
      --head "$work/policy.safetensors" --play "$SEEDS" --max-steps "$STEPS" \
      --seed "$attempt" --device "${GPU:-gpu0}" \
      > "$LOGS/r$r-a$attempt-prove-m$m.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  for m in $MAPS; do
    local exits
    exits=$(grep -oE "[0-9]+ exits" "$LOGS/r$r-a$attempt-prove-m$m.log" | tail -1 | cut -d' ' -f1)
    exits=${exits:-0}
    say "$r" "$skill" "$mission" "$attempt" prove "$m" "$exits/$SEEDS"
    total=$((total + exits))
    [ "$exits" -ge "$SEEDS" ] && passed=$((passed + 1))
  done
  note "rung $r attempt $attempt: the policy alone finished $total/$(( $(echo $MAPS | wc -w) * SEEDS )) episodes, clean on $passed levels"
  [ "$passed" -eq "$(echo $MAPS | wc -w)" ]
}

record() {  # rung skill mission work
  local r=$1 skill=$2 mission=$3 work=$4 m
  for m in $MAPS; do
    local out="$VIDEOS/rung$r-skill$skill-$mission-E1M$m.mp4"
    "$BIN" play --doom-bin "$DOOM" --wad "$WAD" --encoder "$ENC" \
      --map "$m" --skill "$skill" --mission "$mission" --reward gauge \
      --head "$work/policy.safetensors" --play 1 --max-steps "$STEPS" \
      --seed 1 --device "${GPU:-gpu0}" --record "$out" \
      > "$LOGS/r$r-record-m$m.log" 2>&1
    say "$r" "$skill" "$mission" - record "$m" "$(grep -oE '[0-9]+ exits' "$LOGS/r$r-record-m$m.log" | tail -1)"
    note "recorded $out"
  done
}

for (( r = FIRST; r <= LAST; r++ )); do
  rung=${RUNGS[$((r - 1))]}
  skill=${rung%%:*}
  mission=${rung##*:}
  work=out/ladder/r$r
  mkdir -p "$work"
  note "=== rung $r: skill $skill, $mission, levels $MAPS ==="
  passed=no
  for (( a = 1; a <= ATTEMPTS; a++ )); do
    find_generation "$r" "$skill" "$mission" "$a" "$work" "$work/policy.safetensors"
    clone "$r" "$mission" "$a" "$work" || continue
    if prove "$r" "$skill" "$mission" "$a" "$work"; then
      passed=yes
      break
    fi
  done
  if [ "$passed" = no ]; then
    note "rung $r did NOT pass in $ATTEMPTS attempts - stopping here rather than climbing on an unfinished rung"
    exit 1
  fi
  note "rung $r PASSED - every level, every seed, policy alone"
  record "$r" "$skill" "$mission" "$work"
done
note "the ladder is climbed"
