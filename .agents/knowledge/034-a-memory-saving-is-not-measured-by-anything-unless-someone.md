<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 34. A memory saving is not measured by anything unless someone measures it

int8 paged KV had a real, careful quality gate before this workstream
started: a 3-way `fp32`/`int8`/`int8-calib` loss and token-accuracy table on
the real checkpoint. It had NO memory gate at all. The headline claim - "int8
KV is ~4x smaller" - lived as a single `/ 4` inside the allocator and a code
comment next to it. `crates/perf`'s `memory` block was hardcoded `null` for the
KV pool the whole time; the residency budget (`QwenResident::estimate`) only
ever counted checkpoint file size, so switching KV dtype changed the real
footprint by nearly 4x and changed the number `estimate()`/`crates/stats`/
braintop reported by exactly zero. Nothing was lying - there was simply no
path from "the allocator does `/4` here" to any artifact, gate, or UI a human
or a CI job would ever look at.

This is the memory-saving twin of lesson #1 (a gate that never runs is worse
than none): a saving that is only ever a mental derivation from reading the
allocator code is not verified, is not regression-tested, and cannot be
quoted in a doc without someone re-deriving it by hand each time - exactly
the failure mode `kv_pool_bytes_identity_holds_at_the_real_shape` exists to
close, by making the `4·head_dim / (head_dim + 4)` ratio an assertion instead
of a comment, and asserting it is DIFFERENT at `QwenConfig::tiny()`'s
`head_dim=8` (2.667x) than at the real `head_dim=128` (3.8788x) - a toy-fitted
number cannot stand in for the real one (lesson #18), including when the
"number" is a memory-savings ratio rather than a numeric result.

The fix generalizes: any optimization whose entire pitch is "uses less X"
needs X measured through the same artifact/gate path its quality claim
already gets, in the same commit - not as a follow-up, because the
follow-up is exactly the step that silently never happened here for however
many commits this shipped without one.
