#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Environment-write ratchet (`make check/env-writes`).
#
# The process environment is global state shared by every thread. Library code
# that sets a variable around a build and restores it afterwards lets another
# thread read the temporary value: two concurrent trainings of one seed once
# ran with different optimisers because one read the other's setting. Library
# code therefore passes a value down, or scopes it to its own thread
# (thread_local), and never writes the environment. Detection is
# scripts/gates/env-writes.py: `env::set_var` / `env::remove_var` outside test
# code, counted per file.
#
# Legitimate writers exist - a binary's `main` before any worker starts, a
# loader that must set `LD_LIBRARY_PATH` before `dlopen` - and each is listed in
# scripts/gates/env-writes.allow as `path<TAB>count<TAB>reason`. The gate fails on
#   (a) more writes in a file than its entry allows, or a file with none listed;
#   (b) an entry with no written reason; and
#   (c) an entry whose writes are gone or fewer - lower or delete it. The list
#       only shrinks.
#
# Usage: scripts/gates/check-env-writes.sh [--root DIR] [--allow FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/env-writes.allow"
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  *) echo "check-env-writes: unknown argument $1" >&2; exit 2 ;;
  esac
done

actual="$(LC_ALL=C python3 "$here/env-writes.py" --root "$root" | LC_ALL=C sort)" || exit 2

fail=0
declare -A allowed
while IFS=$'\t' read -r path count reason; do
  [ -z "$path" ] && continue
  allowed["$path"]="$count"
  if [ -z "${reason:-}" ]; then
    fail=1
    echo "check-env-writes: NO REASON for $path: every allowlisted writer says why it owns the process."
  fi
done <"$allow"

declare -A seen
while IFS=$'\t' read -r path count; do
  [ -z "$path" ] && continue
  seen["$path"]=1
  have="${allowed[$path]:-0}"
  if [ "$count" -gt "$have" ]; then
    fail=1
    echo "check-env-writes: NEW WRITE: $path writes the process environment $count time(s), $have allowed."
    echo "  Do not mutate the environment from library code: pass the value down the call path,"
    echo "  or scope it to the building thread (see model::offload). A binary's main that must"
    echo "  set a variable before any thread starts can be listed with its reason in $allow."
  elif [ "$count" -lt "$have" ]; then
    fail=1
    echo "check-env-writes: STALE: $path has $count write(s), $have allowed; lower the entry (the list only shrinks)."
  fi
done <<<"$actual"
for path in "${!allowed[@]}"; do
  if [ -z "${seen[$path]:-}" ]; then
    fail=1
    echo "check-env-writes: STALE: $path no longer writes the environment; delete its entry."
  fi
done
[ "$fail" -eq 0 ] && echo "check-env-writes: OK ($(printf '%s\n' "$actual" | grep -c .) allowlisted writers, none new)"
exit "$fail"
