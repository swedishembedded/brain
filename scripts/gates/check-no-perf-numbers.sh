#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# No-unreviewed-perf-numbers-in-docs gate (`make check/scripts`).
#
# A perf number written into prose (docs/**/*.md) is a claim that outlives
# the hardware, driver, and code that produced it - the moment the kernel
# path changes or someone reads it on different silicon, the number quietly
# becomes a lie the reader has no way to detect. `docs/performance/overview.md`
# already says this explicitly ("not a promise for your hardware, always
# measure your own setup"), but that discipline only holds if bare numbers
# don't creep back in unreviewed. This gate makes that discipline mechanical:
# any bare number adjacent to a performance unit or claim is denied unless a
# human has deliberately signed off on it with an escape-hatch comment.
#
# DENIED (a number next to any of these means a measured wall-clock or
# throughput result is being asserted):
#   - durations:    ms, bare `s`, `min` (e.g. "72.2 s", "115 ms", "12 min")
#   - rates:        fps, tok/s, tokens/s, GB/s, MB/s, TFLOP[/s], GFLOP[/s]
#   - percentages:  N% - but ONLY when the surrounding few lines talk about
#                    measurement (see CONTEXT_RE below), so "30% token
#                    masking" (a training hyperparameter) is not confused
#                    with "reached 93-100% of its own datasheet peak" (a
#                    measured result)
#   - speedups:     N x / N x (e.g. "1.7x", "roughly 8x") - same
#                    context-gating as percentages, so "a 16x convolutional
#                    token compressor" (an architecture constant) is not
#                    confused with "measured 2.5-3.6x the throughput" (a
#                    measured result)
#   - percentiles:  p50/p95/p99 followed by an actual value (defining what a
#                    percentile IS, with no number attached, is fine - see
#                    docs/performance/benchmarking.md)
#
# NOT denied (these are not performance claims, so a bare number next to one
# of these is left alone): model/tensor dimensions, static file sizes
# (GiB/MB as a size, not a rate), port numbers, version numbers, and memory
# *requirements* ("needs 22 GiB RAM" is a requirement, not a measurement).
#
# Escape hatch: a line carrying an HTML comment `<!-- perf-number: <reason> -->`
# on the same line or the line immediately before it is let through - this
# turns an unavoidable claim into a reviewed, deliberate exception instead of
# an accident.
#
# Known imprecision: the percentage/speedup context-gate is a heuristic
# (nearby measurement vocabulary), not a parser. It errs toward flagging a
# real number that turns out to be a config ratio rather than silently
# missing a measured claim - false positives found in the wild belong on the
# escape-hatch list (or a gate refinement), never a reason to weaken the
# pattern itself.
#
# Usage: scripts/gates/check-no-perf-numbers.sh   (exits non-zero, listing
# every violation as file:line: matched text, not just the first)
set -u
cd "$(dirname "$0")/../.."

fail=0
tmp_hits=$(mktemp)
trap 'rm -f "$tmp_hits" "$tmp_hits.sorted"' EXIT

# "Hard" units: unambiguous performance vocabulary, no English word ever
# collides with them, so no context-gating needed.
HARD_RE='[0-9][0-9.,]*[[:space:]]?(ms|min|s\/(page|frame)|fps|tok(en)?s?\/s|[GM]B\/s|[TG]FLOP(S|s)?(\/s)?)\b'
HARD_RE_BARE_S='[0-9][0-9.,]*[[:space:]]?s\b'
# p50/p95/p99 only counts once an actual value trails it (not just the
# percentile named in prose, e.g. "latency p50/p99 + throughput").
PCTL_RE='\bp(50|95|99)\b[^0-9a-zA-Z\n]{0,20}[0-9][0-9.,]*[[:space:]]?(ms|s|%)\b'

# "Soft" patterns: a bare percentage or an Nx/N× multiplier. Both also occur
# as architecture/config constants (a 16x downsample, 30% token masking), so
# a hit only counts if measurement vocabulary shows up within a few lines -
# perf prose in this repo is written in wrapped paragraphs, so the cue word
# ("measured", "speedup", ...) is often a line or two away from the number
# itself, not on the exact same line.
SOFT_PCT_RE='[0-9][0-9.,]*[[:space:]]?%'
SOFT_X_RE='[0-9][0-9.,]*[×x](?![0-9a-zA-Z])'
CONTEXT_RE='measur|profil|\bwall\b|laten|throughput|speedup|speed-up|\bfaster\b|\bslower\b|benchmark|median|\bmean\b|\bpeak\b|regress(ed|ion)?|baseline|roofline|\bflop|decode (loop|step)|training step|step time|per-kernel|per-frame|per-page|inference time|resident instance|wall-clock|wall time'

is_escaped() {
  # $1 = file, $2 = 1-based line number: escaped if the marker is on this
  # line or the one immediately before it.
  local f="$1" ln="$2"
  sed -n "$((ln - 1 > 0 ? ln - 1 : 1)),${ln}p" "$f" | grep -q '<!-- perf-number:'
}

has_context() {
  # $1 = file, $2 = 1-based line number: true if measurement vocabulary
  # appears in the surrounding window (4 lines back, 2 forward - covers a
  # wrapped paragraph's cue word and a heading that introduces a table).
  local f="$1" ln="$2" start end
  start=$((ln - 4)); [ "$start" -lt 1 ] && start=1
  end=$((ln + 2))
  sed -n "${start},${end}p" "$f" | grep -qiE "$CONTEXT_RE"
}

while IFS= read -r -d '' file; do
  # Hard, unambiguous units.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$HARD_RE" "$file")

  # Bare seconds ("72.2 s") - same hard-unit treatment, kept as its own pass
  # only because it's expressed separately above for readability.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$HARD_RE_BARE_S" "$file")

  # p50/p95/p99 followed by an actual value.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noiP "$PCTL_RE" "$file")

  # Soft patterns: percentage and Nx/N×, gated on nearby measurement context.
  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    has_context "$file" "$ln" || continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$SOFT_PCT_RE" "$file")

  while IFS=: read -r ln rest; do
    [ -z "$ln" ] && continue
    is_escaped "$file" "$ln" && continue
    has_context "$file" "$ln" || continue
    printf '%s\n' "$file:$ln: $rest" >> "$tmp_hits"
  done < <(grep -noP "$SOFT_X_RE" "$file")
done < <(find docs -name '*.md' -print0 | sort -z)

if [ -s "$tmp_hits" ]; then
  # Sort by file/line for a stable, readable order. NOT `sort -u` restricted
  # to the file:line keys - two distinct matches (e.g. "1.3x" and "1.6x") can
  # legitimately share one line, and keying uniqueness on file:line alone
  # would silently drop the second one. `uniq` on the fully-sorted whole line
  # instead only collapses genuine exact-duplicate hits (the same match
  # caught by two overlapping patterns).
  sort -t: -k1,1 -k2,2n "$tmp_hits" | uniq > "$tmp_hits.sorted"
  while IFS= read -r hit; do
    echo "PERF NUMBER: $hit"
  done < "$tmp_hits.sorted"
  count=$(wc -l < "$tmp_hits.sorted")
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "Found $count bare perf number(s) in docs/**/*.md (see PERF NUMBER lines"
  echo "above, file:line). Each one is a wall-clock/throughput/percentage"
  echo "claim written into prose without a reviewed exception. Fix each by"
  echo "either:"
  echo "  1. Rephrasing to drop the specific bare number (point at"
  echo "     'brain perf run'/'brain flops' instead of a fixed figure), or"
  echo "  2. Marking it a deliberate, reviewed exception with"
  echo "     '<!-- perf-number: <reason> -->' on the same line or the line"
  echo "     immediately before it."
  exit 1
fi
echo "check-no-perf-numbers: no unreviewed perf numbers found in docs/**/*.md"
