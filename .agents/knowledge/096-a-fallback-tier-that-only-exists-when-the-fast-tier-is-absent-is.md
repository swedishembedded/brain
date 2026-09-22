<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 96. A fallback tier that only exists when the fast tier is ABSENT is not a fallback

`residency::place::pick_device` refuses to put a VRAM-costed model on the
CPU whenever a GPU exists, and that is right: its `None` is the signal that
makes `ResidencyManager::claim` EVICT a card rather than quietly park a
model in RAM at a hundredth of the speed. The rule was written for a caller
that has an eviction path.

It was then copied, verbatim, into three callers that do not.
`residency::plan::plan` gates its host-tier branch on
`b.gpus().is_empty()`; `could_ever_fit` mirrors the same expression;
`crates/cli`'s `probe_free_vram` dropped a card with zero free bytes from
the GPU list entirely. Together those produced a machine that was strictly
better off the MORE contended it was: with both cards fully consumed the
GPU list emptied, `gpus().is_empty()` became true, and the host tier
answered; with a single free byte on either card the GPU class stayed
"non-empty" and the whole run died - `cannot place 'dit' (14.3 GiB device)
... free: gpu0=4.9 GiB gpu1=7.8 GiB cpu=45.6 GiB`, with 45.6 GiB of host
RAM sitting idle and a working CPU backend three function calls away.

Two distinct questions had been collapsed into one predicate. "Should this
go to the CPU *now*?" (no - evict a card, you have one) and "*Could* this
ever run at all?" (yes - the CPU backend exists) have different answers,
and `could_ever_fit`'s answer is what decides `TooLarge`, which the
executor treats as permanent and uses to fail every queued job for that
model forever. The general shape: when you copy a policy predicate into a
new caller, copy the REASON with it and check the reason still holds. "No
GPU exists" was never the real condition; "the caller has no way to make
room" was.
