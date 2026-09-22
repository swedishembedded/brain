<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 109. A coverage ratchet written BEFORE the thing it ratchets exists cannot be a floor

This engine's proven answer to "a number nothing checks goes stale" is a
covered/total floor (`gpu_core::cost`'s per-kernel cost-formula ratchet). Build
the same instrumentation before the first item exists and the floor starts at
zero, where `assert!(covered >= 0)` on an unsigned count is not merely weak -
it is a deny-by-default clippy lint (`absurd_extreme_comparisons`), so it does
not compile in a gated workspace. The lint is right: a comparison against a
type's minimum asserts nothing.

Two spellings work, and the choice is a statement about the data. An EQUALITY
(`assert_eq!(covered, N)`) is right where each item is a deliberate, rare,
individually-reviewed commitment - a performance contract, a supported format -
because then having to edit the number is the point: it makes a deletion
visible. A floor is right where items land constantly and incidentally, like a
cost formula per new kernel, where forcing a test edit per addition is friction
with no signal in it. Starting a floor at 1 instead - "one item exists, so the
comparison is non-trivial" - is the worst of both: it ratchets nothing and
reads like it does.
