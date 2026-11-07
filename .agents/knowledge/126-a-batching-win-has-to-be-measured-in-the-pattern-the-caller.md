<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 126. A batching win has to be measured in the pattern the caller actually uses

**Lesson:** Replacing many driver calls with one only pays if the host is not made to wait somewhere
else instead. A speedup number is meaningless without the calling pattern it belongs to.

## What happened
Parameter staging is the somewhere else. A replay overwrites host memory the previous replay's copy
nodes may not have read yet.

A caller that submits in a tight loop without ever synchronising pays a device wait per submission
and measures roughly no improvement at all.

The same code in the loop the mechanism exists for pays nothing and shows the full saving. That
loop is a decode step, which reads its logits every token and has therefore already drained.

## Why
Both numbers are true. Quoting either one without the pattern is meaningless.

## What to do
- Count the wait (`staging_waits`) so the two cases are distinguishable in the field.
- Assert it is zero in the test that claims the speedup.
- Say in the ledger which loop the number belongs to.
