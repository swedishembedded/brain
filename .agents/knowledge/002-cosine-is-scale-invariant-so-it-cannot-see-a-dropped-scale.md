<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 2. Cosine is scale-invariant, so it cannot see a dropped scale factor

**Lesson:** Cosine is scale-invariant, so it cannot see a dropped scale factor; gate `rel_l2` (or
`max_abs`) alongside it.

## What happened
`got = 1.05 * want` scores cosine 1.0. A dropped `output_scale_factor`, a doubled attention
scale, or a mis-read GroupNorm gain all pass a cosine-only ladder.

## What to do
Gate `rel_l2` (or `max_abs`) alongside cosine. `crates/unet`'s parity does, after a reviewer
noticed the ladder computed `rel_l2` at every stage and never asserted it.
