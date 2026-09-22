<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 99. A placement is a label until something acts on it

`Home::Cpu` was returned by the placer, printed in the placement line,
asserted in tests - and did nothing. `Homes::run` matched `Some(Home::Gpu(i))`
and let every other case fall through to `_ => Ok(f())`, which runs the
closure UNSCOPED, so every `Gpu::new` inside still built on whatever card
was ambient. A part "placed on the host tier" allocated exactly the VRAM
the placer had just decided it could not have. `flux1::pipeline::run_on_home`
had an identical `_ =>` arm, so `BRAIN_FLUX1_TE_DEVICE=cpu` did not put
T5-XXL on the CPU either.

It survived because the enum arm that does nothing and the enum arm that is
not applicable look the same when both are spelled `_`. The GPU arm was
tested (a part on card 1 while card 0 was ambient - an assertion that CAN
fail if scoping is forgotten); the CPU arm was tested only for what the
placer RETURNED, never for what running under it did.

Two rules fall out. Match placement enums exhaustively, so "this variant
has no implementation yet" is a compile error rather than a silent
fall-through. And when testing a placement decision, assert on the
OBSERVABLE consequence (which backend got built, which card the scope
resolves to) rather than on the decision value - the value being right is
what the bug looked like from the outside the whole time.
