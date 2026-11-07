<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 109. A coverage ratchet written BEFORE the thing it ratchets exists cannot be a floor
**Lesson:** A coverage ratchet built before the first item exists starts at zero and cannot be a
floor. Choose between an equality and a floor according to how items land, and never start a floor
at 1 just to make the comparison non-trivial.

## What happened

This engine's proven answer to "a number nothing checks goes stale" is a covered/total floor
(`gpu_core::cost`'s per-kernel cost-formula ratchet). Building the same instrumentation before the
first item exists starts the floor at zero.

## Why

`assert!(covered >= 0)` on an unsigned count is not merely weak. It is a deny-by-default clippy lint
(`absurd_extreme_comparisons`), so it does not compile in a gated workspace. The lint is right: a
comparison against a type's minimum asserts nothing.

## What to do

Two spellings work, and the choice is a statement about the data:

- An EQUALITY (`assert_eq!(covered, N)`) is right where each item is a deliberate, rare,
  individually-reviewed commitment (a performance contract, a supported format). Having to edit the
  number is then the point: it makes a deletion visible.
- A floor is right where items land constantly and incidentally, like a cost formula per new
  kernel, where forcing a test edit per addition is friction with no signal in it.

Starting a floor at 1 instead ("one item exists, so the comparison is non-trivial") is the worst of
both: it ratchets nothing and reads like it does.
