#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# capability-report.sh - render the capability-skip ledger.
#
# `brain_testutil::skip_unvalidated_capability(cap, reason)` appends a row
# every time a hardware-gated code path runs on this box without the hardware
# it targets (FP8 tensor cores, AMX, AVX-512+VNNI, ...): the branch executed,
# but nothing here proved it correct. That is invisible in a plain `cargo
# test` pass/fail summary, so this renders the ledger those calls write to
# (`$BRAIN_CAPABILITY_LEDGER`, default `<repo>/out/capability-ledger.tsv`) as
# a table: which capabilities are unvalidated on this box, why, how many
# times each was hit, and by which call sites.
#
# Run tests first (the ledger only grows when a gated test actually runs),
# then:
#   scripts/gates/capability-report.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LEDGER=${BRAIN_CAPABILITY_LEDGER:-out/capability-ledger.tsv}

if [ ! -s "$LEDGER" ]; then
  echo "no capability skips recorded ($LEDGER is empty or absent)"
  echo "run the suite first; a gated test that lacks its hardware records itself there"
  exit 0
fi

awk -F'\t' '
  NF < 3 { next }
  {
    cap = $1; reason = $2; loc = $3
    count[cap]++
    rkey = cap SUBSEP reason
    if (!(rkey in reason_seen)) {
      reason_seen[rkey] = 1
      reasons[cap] = (cap in reasons) ? reasons[cap] "; " reason : reason
    }
    lkey = cap SUBSEP loc
    if (!(lkey in loc_seen)) {
      loc_seen[lkey] = 1
      sites[cap] = (cap in sites) ? sites[cap] ", " loc : loc
      nsites[cap]++
    }
  }
  END {
    printf "%-28s %6s  %s\n", "capability", "skips", "reason(s)"
    printf "%-28s %6s  %s\n", "----------", "-----", "---------"
    for (c in count) printf "%-28s %6d  %s\n", c, count[c], reasons[c]
    print ""
    print "call sites:"
    for (c in sites) printf "  %-28s (%d) %s\n", c, nsites[c], sites[c]
  }
' "$LEDGER"
