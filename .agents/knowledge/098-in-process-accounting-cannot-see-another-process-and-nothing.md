<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 98. In-process accounting cannot see another process, and nothing about that is a rounding error
**Lesson:** Memory shared with the rest of the machine has to be MEASURED, repeatedly, not modelled
once. Anything that only re-reads the world when its own work completes cannot recover from a
change it did not cause.

## What happened

`brain serve` budgeted every card from `nvidia-smi --query-gpu=memory.total` once, at startup, and
then tracked only its own allocations. On a card with 18 GiB held by a neighbouring job it
therefore believed 22 GiB were usable, placed a 16 GiB model, and aborted inside the driver, with
the scheduler's own accounting reporting a successful placement.

The one-shot CLI path, in the same process at the same instant, read `memory.free` and budgeted
the same card at 5 GiB. Two capacity models, both frozen, disagreeing by 17 GiB.

## Why

Freezing was the deeper half. Budgets were pure in-process integers, so VRAM freed by a foreign
process produced no message, no budget change, and no wake-up.

The dispatcher blocked on `rx.recv()` with no timer, so the only thing that ever re-ran scheduling
was one of this daemon's own jobs finishing. A daemon could sit for a week believing a card was
full that a neighbour released in its first minute.

## What to do

Three changes fixed it:

- One probe (`crates/cli/src/capacity.rs`) feeding both consumers.
- A TTL on the one-shot placer's snapshot.
- An idle tick on the dispatcher that re-measures and re-assigns.
