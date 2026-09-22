<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 101. A guard test for one function's INPUT is not a test of what a DIFFERENT function does with it

`resolve::tests::a_no_weights_utility_model_with_no_brain_arch_row_still_resolves`
has existed for a while, pinning that `resolve()` classifies `imageops`/
`demo`/`imgpipe` as `Resolved::Arch` even though none of them has a
`brain_arch::Arch` row - exactly the shape `dispatch_arch` needs to route
them. That test passed the whole time. What it does not - and cannot -
cover is what happens a few calls later: `dispatch_arch` unconditionally
calls `supply::ensure_env_weights(arch)` whenever the verb is a real one
(`wants_weight_acquisition` checks the architecture's own capability
manifest, which `imageops`'s `draw_boxes` action is genuinely listed in),
and `ensure_env_weights_with` treated ANY `brain_arch::by_id` miss as proof
of a dispatch bug - a reasonable assumption for every architecture that
migrated onto `ARCH_TO_MODEL` from a real crate, and simply false for the
three that were never backed by one. `brain imageops draw_boxes` - the
single most basic, no-weights, deterministic command in the whole CLI -
failed outright, in a class of command real enough that it was needed
mid-session to visualize a trained detector's boxes.

Two lessons. First: a green test proves the function it calls behaves as
asserted, not that every function downstream of the code path it exercises
does. Tracing one call one level deep is not the same as running the
command. Second: `NO_ARCH_ROW` existed, but only inside the one test that
happened to need it, at file-private scope, its exception never propagated
to the other function whose OWN doc comment stated the exact assumption
that exception violates. A documented exception that lives next to only
one of its several consumers is one that the others will contradict.
