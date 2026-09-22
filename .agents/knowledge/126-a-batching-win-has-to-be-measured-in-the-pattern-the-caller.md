<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 126. A batching win has to be measured in the pattern the caller actually uses

Replacing many driver calls with one only pays if the host is not made to wait
somewhere else instead. Here the somewhere else is parameter staging: a replay
overwrites host memory the previous replay's copy nodes may not have read yet,
so a caller that submits in a tight loop without ever synchronising pays a
device wait per submission and measures roughly no improvement at all.

The same code in the loop the mechanism exists for - a decode step, which
reads its logits every token and has therefore already drained - pays nothing
and shows the full saving. Both numbers are true, and quoting either one
without the pattern is meaningless. The resolution is to count the wait
(`staging_waits`) so the two cases are distinguishable in the field, assert it
is zero in the test that claims the speedup, and say in the ledger which loop
the number belongs to.
