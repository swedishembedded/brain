<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 5. Run every gradcheck on BOTH backends

A `var<workgroup>` + `workgroupBarrier()` reduction with no barrier-free sibling
returns **all zeros** on `backend-cpu` - no crash, no error, a trainable
parameter whose gradient is silently dead and a loss curve that still looks
plausible. `backend-cpu` reports `workgroup_reductions: false` and its
split-at-barrier JIT mis-executes the construct. A GPU-only gate passes it
completely.

Any per-channel or per-row reduction in a backward therefore needs the
**barrier-free + cooperative pair**, selected on the *queried*
`DeviceCaps::workgroup_reductions` (`prelu_bwd`/`prelu_bwd_wg`,
`gn_stats`/`gn_stats_wg`). That selection is a correctness gate, not a perf tweak.
