#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Self-validation gate for scripts/ and tools/ (`make check/scripts`).
#
# The repo's own structure is supposed to make an unused/broken script
# impossible to miss — this is that check, automated. Three things:
#
#   1. SYNTAX  — every .sh parses (`bash -n`); every .py compiles
#      (`py_compile`), which also catches import-time typos (not missing
#      third-party packages — those are legitimately optional dev tooling).
#   2. ORPHANS — every tracked file under scripts/ and tools/ (except plain
#      data: .json/.txt/.md) must be named somewhere ELSE in the repo — a
#      Makefile target, a bats test, a Rust doc comment citing it as a golden
#      generator, another script that calls it, or a doc. A file nothing
#      references is exactly the "nobody remembers why this exists, does it
#      even still work" rot an unreferenced-scripts audit of this repo found.
#   3. ABSOLUTE PATHS — no non-overridable absolute machine path
#      (`/data/…`, `/home/…`, `/tmp/…`, …), mirroring the existing
#      `crates/**` grep gate in AGENTS.md. The one sanctioned exception is a
#      mirror-location default of the shape `${VAR:-/abs/path}` — an
#      overridable variable, not a baked-in path — which this check allows.
#
# Usage: scripts/gates/check-scripts.sh   (exits non-zero with every violation
# printed, not just the first, so one run tells you everything to fix)
set -u
cd "$(dirname "$0")/../.."

fail=0

echo "=== 1/3: syntax ==="
PYCACHE="$(mktemp -d)"
trap 'rm -rf "$PYCACHE"' EXIT
while IFS= read -r -d '' f; do
  bash -n "$f" || { echo "  SYNTAX FAIL: $f"; fail=1; }
done < <(git ls-files -z 'scripts/*.sh' 'tools/*.sh')
while IFS= read -r -d '' f; do
  PYTHONPYCACHEPREFIX="$PYCACHE" python3 -m py_compile "$f" || { echo "  SYNTAX FAIL: $f"; fail=1; }
done < <(git ls-files -z 'scripts/*.py' 'tools/*.py')
[ "$fail" -eq 0 ] && echo "  ok"

echo
echo "=== 2/3: orphans (every script/tool must be named somewhere else in the repo) ==="
orphans=0
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  # Search everywhere EXCEPT the file's own bytes; a hit anywhere else (Makefile
  # target, bats test, Rust doc comment, sibling script, doc) counts as "used".
  if ! git grep -qF -- "$base" -- . ":(exclude)$f" 2>/dev/null; then
    echo "  ORPHAN: $f (not named anywhere else in the repo)"
    orphans=$((orphans + 1))
  fi
done < <(git ls-files -z 'scripts/*.sh' 'scripts/*.py' 'tools/*.sh' 'tools/*.py' 'tools/*.txt' \
  | grep -zv '/forecast-perf-baselines/\|/wm-perf-baselines\.json$')
if [ "$orphans" -gt 0 ]; then
  echo "  $orphans orphan(s) — delete them, or wire them into a Makefile target /"
  echo "  bats test / doc / crate reference so the next run finds the citation."
  fail=1
else
  echo "  ok"
fi

echo
echo "=== 3/3: no non-overridable absolute machine paths ==="
# A literal absolute-path STRING that does not sit inside a shell default-value
# expansion (\${VAR:-/...}) or a Python os.environ.get(..., default) fallback —
# both of which are the sanctioned "overridable variable" shape. This is a
# heuristic, not a parser: it flags the string and requires a human look, same
# spirit as the crates/** gate.
abs=0
while IFS= read -r -d '' f; do
  # Full lines (NOT `grep -o`, which would truncate to just the path match and
  # throw away the `${VAR:-...}` / `.get(..., ...)` context this needs to see).
  while IFS=: read -r lineno full_line; do
    trimmed="${full_line#"${full_line%%[![:space:]]*}"}"
    # A full-line comment (# in bash or python) has no runtime effect — an
    # absolute path there can't misbehave on someone else's machine the way one
    # baked into executable code can. Skip it; docstring usage EXAMPLES should
    # still use a placeholder (see worldmirror2_dump_reference.py), but that's a
    # documentation-quality call, not this gate's job.
    case "$trimmed" in '#'*) continue ;; esac
    # ${VAR:-/abs/path} or os.environ.get(V, "/abs/path") — both an overridable
    # variable with a default, not a baked-in path — are sanctioned.
    if [[ "$full_line" =~ \$\{[A-Za-z_][A-Za-z0-9_]*:-/(data|home|tmp|opt|mnt|root)/ ]]; then
      continue
    fi
    if [[ "$full_line" =~ environ\.get\([^,]+,.*/(data|home|tmp|opt|mnt|root)/ ]]; then
      continue
    fi
    echo "  ABS PATH: $f:$lineno: $trimmed"
    abs=$((abs + 1))
  done < <(grep -nP '(?<![A-Za-z0-9_])/(data|home|tmp|opt|mnt|root)/[A-Za-z0-9_./-]+' "$f" || true)
done < <(git ls-files -z 'scripts/*.sh' 'scripts/*.py' 'tools/*.sh' 'tools/*.py')
if [ "$abs" -gt 0 ]; then
  echo "  $abs absolute path literal(s) — make them an overridable \${VAR:-/path}"
  echo "  default (see scripts/data/fetch-testdata.sh's BRAIN_*_MIRROR vars), or a"
  echo "  repo-relative path."
  fail=1
else
  echo "  ok"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "CHECK/SCRIPTS: PASS"
else
  echo "CHECK/SCRIPTS: FAIL"
fi
exit "$fail"
