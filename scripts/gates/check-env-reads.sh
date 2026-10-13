#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Environment-read ratchet (`make check/env-reads`).
#
# Environment variables are being phased out in favour of CLI flags: a setting
# read from the environment is a hidden input that no `--help` lists, no test
# can set without mutating process-wide state, and no caller can see in a
# command line. Every environment read under crates/ (detected by
# scripts/gates/env-reads.py, the same detection check-env-docs.sh uses) must
# therefore be listed in scripts/gates/env-reads.allow, one `path<TAB>VARIABLE`
# per line, sorted. The gate fails on
#   (a) a read that is not listed - new reads are forbidden, add a flag; and
#   (b) a listed read that no longer exists - delete the entry. The list only
#       shrinks, the same semantics as the large-file entries in
#       architecture.toml.
#
# Usage: scripts/gates/check-env-reads.sh [--root DIR] [--allow FILE]
#        scripts/gates/check-env-reads.sh --regenerate   (rewrite the allowlist)
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/env-reads.allow"
regenerate=0
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  --regenerate) regenerate=1; shift ;;
  *) echo "check-env-reads: unknown argument $1" >&2; exit 2 ;;
  esac
done

actual="$(LC_ALL=C python3 "$here/env-reads.py" --root "$root" | LC_ALL=C sort)" || exit 2
if [ "$regenerate" -eq 1 ]; then
  printf '%s\n' "$actual" >"$allow"
  exit 0
fi

listed="$(LC_ALL=C sort "$allow")"
new="$(LC_ALL=C comm -23 <(printf '%s\n' "$actual") <(printf '%s\n' "$listed"))"
stale="$(LC_ALL=C comm -13 <(printf '%s\n' "$actual") <(printf '%s\n' "$listed"))"

fail=0
if [ -n "$new" ]; then
  fail=1
  echo "check-env-reads: environment reads that are not on the allowlist:"
  printf '%s\n' "$new" | awk -F'\t' '{ printf "  NEW READ: %s reads %s\n", $1, $2 }'
  echo "  Environment variables are being phased out: add a CLI flag (or pass a"
  echo "  value in), do not read a variable."
fi
if [ -n "$stale" ]; then
  fail=1
  echo "check-env-reads: allowlist entries whose read no longer exists:"
  printf '%s\n' "$stale" | awk -F'\t' '{ printf "  STALE: %s no longer reads %s\n", $1, $2 }'
  echo "  Delete these lines from $allow (the list only shrinks)."
fi
[ "$fail" -eq 0 ] && echo "check-env-reads: OK ($(printf '%s\n' "$actual" | grep -c .) allowlisted reads, none new)"
exit "$fail"
