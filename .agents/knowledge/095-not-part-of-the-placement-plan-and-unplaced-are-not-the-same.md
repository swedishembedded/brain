<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 95. "Not part of the placement plan" and "unplaced" are not the same thing

`pulid::caps::Bundle::load` builds ArcFace/BiSeNet/EVA-CLIP/IDFormer with a
bare `Gpu::new(...)`, and the comment above it was correct on its own
terms: these towers run once per identity, never push a `Step` into the
DiT's dispatch list, and so have no reason to be a `Need` in
`flux1::pipeline::part_needs`. What the comment did not say, because
nobody had asked yet, is WHICH device a bare `Gpu::new` actually lands on.
On this box it is not "gpu0, always" or "whatever's least loaded" - it is
just the ambient default, and on the very run that finally exercised this
code with real weights, that default coincided with "te", where a ~21 GiB
fp32 T5-XXL was already resident. Two moderate allocations shared one
card and the second one that touched the device (T5-XXL, loaded lazily at
`encode()` time) OOM'd.

"Not part of the plan" was read as "doesn't need a home." It needed one,
just not a NEW one - `homes.run("dit", || Gpu::new(...))` (the same
`Homes` `plan_flux1` already computed for the DiT) rides this modest stack
on the DiT's card, which has real headroom (an int8 DiT + resident
`PulidCa` is ~17.5 GiB against 24), instead of trusting an unnamed
default. The general shape: a comment justifying why something is
EXCLUDED from a sizing/placement decision is not the same claim as "this
thing's location doesn't matter" - the first is often true while the
second silently is not.
