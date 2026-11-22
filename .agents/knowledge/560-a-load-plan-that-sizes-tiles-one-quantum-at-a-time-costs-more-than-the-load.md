<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 560. A load plan that sizes tiles one quantum at a time costs more than the load

**Lesson:** time each phase of a load (plan, read, decode, encode, write) before tuning any of them. In brain's packed tile
executor the slowest part of loading a bf16 27B was the plan: it grew the tile by one tile quantum per step until the
staging cap was reached. A bf16 tensor's quantum is one element, so a 4 MB cap meant about two million steps per tensor,
16 s in total. A larger cap made it worse (a 64 MB cap: 235 s). The per-tile timers did not show it, because planning
happens before the first tile.

## Two smaller costs found the same way

- **A buffer cleared before a "fill to length" helper is reallocated on every tile.** The executor cleared its f32
  decode buffer, so `sized_for_fill` saw a length mismatch and mapped fresh zeroed pages each time. That was most of an
  8 s decode for 25.5 G elements.
- **Small tasks on a 288-thread pool cost more in dispatch than in work.** Encoding one 8 M-element tile to int8 took
  20 ms on one thread, 5.6 ms split into 64 K-element tasks, and 2.2 ms as one fused decode-and-code pass in tasks of
  256 K elements (8192 blocks). At 16 K-element tasks it was slower than at 64 K.

## What to do

Find the largest tile that fits by binary search over the quantum count, which is monotone in its staging cost
(`checkpoint::packed::load`, test `the_largest_tile_under_the_cap_is_planned_without_a_step_per_quantum`). Let a decoder
overwrite the previous tile's buffer in place. Convert a plain float tile to int8 in one parallel pass
(`Codec::transcode_tile`); the result must stay byte-identical to decoding then encoding.

## Evidence (2026-11-22, n442, 53.8 GB bf16 Qwen3.8-27B export)

Build of `qwen35 generate`: `int8` went from 61 s to 13.2 s, and bf16 from 21 s to 4.8 s. Full numbers are in
`.agents/roadmap/distributed-execution.md`, Measured.
