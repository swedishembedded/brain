<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 29. `make kernels-regen` had been broken since the script moved

`scripts/build/kernels-regen.sh` computed its repo root relative to its own
location, which was right when it lived directly under `scripts/` and became
wrong once it moved a directory deeper. Every invocation died with a missing
path.

Nobody noticed because the failure mode is *silent at the level that matters*:
you add a `.wgsl`, the regen fails, and you hand-append the two lines to
`crates/kernels/src/lib.rs` instead. The registry stays correct, so no test ever
fails - it just stops being mechanically derivable. Re-running the fixed script
produced a nontrivial diff over kernels added since the breakage: hand-written
doc comments replaced by the generated form, and several consts out of sort
order.

The lesson is about the class, not the typo: **a generator whose output can be
produced by hand will be, and its breakage is invisible until someone needs it
to be authoritative.** `make check/scripts` verifies every script *parses*; it
cannot verify one still *works*. A generator wants a regen-is-a-no-op check.
