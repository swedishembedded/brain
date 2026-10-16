#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for test suites whose green result
# can be trusted for its clients. If your team needs expertise in test-gate
# design and continuous verification then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# Silent-skip ratchet (`make check/test-skips`).
#
# A test that returns early when a GPU or fixture is missing is reported by
# cargo as a PASS, so a green run can certify nothing. A skip must instead go
# through `brain_testutil` (`gpu_tests_skipped`, `skip`, `skip_unavailable`),
# which names the reason and can be made fatal, or carry `#[ignore = "..."]`.
# Remaining legacy skips are listed in scripts/gates/test-skips.allow, one
# `path<TAB>count` per line. The gate fails on
#   (a) a file with more silent skips than listed (new skips are forbidden);
#   (b) a listed count above the real one, or for a file with none: lower or
#       delete the entry. The list only shrinks.
#
# Usage: scripts/gates/check-test-skips.sh [--root DIR] [--allow FILE]
#        scripts/gates/check-test-skips.sh --regenerate   (rewrite the allowlist)
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/test-skips.allow"
regenerate=0
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  --regenerate) regenerate=1; shift ;;
  *) echo "check-test-skips: unknown argument $1" >&2; exit 2 ;;
  esac
done

actual="$(LC_ALL=C python3 "$here/test-skips.py" --root "$root")" || exit 2
if [ "$regenerate" -eq 1 ]; then
  printf '%s\n' "$actual" | grep . >"$allow"
  exit 0
fi

# Join actual and listed counts per path; a missing side counts as 0.
report="$(LC_ALL=C awk -F'\t' '
  FILENAME == ARGV[1] { if ($1 != "") listed[$1] = $2; next }
  $1 != "" { seen[$1] = 1; if (($2 + 0) > (listed[$1] + 0)) printf "NEW\t%s\t%d\t%d\n", $1, $2, listed[$1] + 0
    else if (($2 + 0) < (listed[$1] + 0)) printf "STALE\t%s\t%d\t%d\n", $1, $2, listed[$1] + 0 }
  END { for (p in listed) if (!(p in seen)) printf "STALE\t%s\t0\t%d\n", p, listed[p] }
' "$allow" <(printf '%s\n' "$actual") | LC_ALL=C sort)"

fail=0
new="$(printf '%s\n' "$report" | awk -F'\t' '$1 == "NEW" { printf "  NEW SKIP: %s has %d silent skip(s), %d allowed\n", $2, $3, $4 }')"
stale="$(printf '%s\n' "$report" | awk -F'\t' '$1 == "STALE" { printf "  STALE: %s has %d silent skip(s), %d listed\n", $2, $3, $4 }')"
if [ -n "$new" ]; then
  fail=1
  echo "check-test-skips: silent skips that are not on the allowlist:"
  printf '%s\n' "$new"
  echo "  Use brain_testutil::gpu_tests_skipped / skip / skip_unavailable, or #[ignore = \"needs a GPU\"]."
fi
if [ -n "$stale" ]; then
  fail=1
  echo "check-test-skips: allowlist entries above the real count:"
  printf '%s\n' "$stale"
  echo "  Lower or delete these lines in $allow (the list only shrinks)."
fi
[ "$fail" -eq 0 ] && echo "check-test-skips: OK ($(printf '%s\n' "$actual" | awk -F'\t' '{ s += $2 } END { print s + 0 }') allowlisted silent skips, none new)"
exit "$fail"
