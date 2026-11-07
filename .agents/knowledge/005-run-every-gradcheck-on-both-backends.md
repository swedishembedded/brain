<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 5. Run every gradcheck on BOTH backends

**Lesson:** Run every gradcheck on BOTH backends; a GPU-only gate passes a `backend-cpu` bug
completely.

## What happened
A `var<workgroup>` + `workgroupBarrier()` reduction with no barrier-free sibling returns **all
zeros** on `backend-cpu`. There is no crash and no error: a trainable parameter's gradient is
silently dead and the loss curve still looks plausible. A GPU-only gate passes it completely.

## Why
`backend-cpu` reports `workgroup_reductions: false` and its split-at-barrier JIT mis-executes
the construct.

## What to do
Any per-channel or per-row reduction in a backward needs the **barrier-free + cooperative
pair**, selected on the *queried* `DeviceCaps::workgroup_reductions` (`prelu_bwd`/`prelu_bwd_wg`,
`gn_stats`/`gn_stats_wg`). That selection is a correctness gate, not a perf tweak.
