<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 502. A decode step recorded only for heads reduced on the device left every sampled step launched kernel by kernel

**Lesson:** a recording keyed to the head a step ends in covers only the heads someone recorded. When the head ends in
the logits that the host reads or draws from, the layers can still replay, with the readout run eagerly after them.
Check every decode entry point an engine serves, not only the greedy one, and measure a CPU figure as A/B pairs before
acting on it.

## What happened

Question (Qwen3.6-35B-A3B Q8_0 GGUF, `brain serve`, 64 streams, closed loop of 300-token chat completions, gpu3): one
measurement had the serving process at a whole core under greedy load. Is the core held, and by what?

- On the same commit, and with the binary that measurement used, greedy held 7-9 % of a core over four runs, with the
  card 93-98 % busy. The whole-core figure did not reproduce; greedy steps replay their recording and the host waits
  block (`CU_CTX_SCHED_BLOCKING_SYNC`).
- The sampled path (temperature 1: `forward_batched_drawn`, also `forward_batched_logit_rows`) was not recorded.
  `qwen35moe::serve::Engine::decode` recorded steps ending in the greedy or top-k head only. Every sampled step built and
  launched its ~1600 dispatches one by one. nsys showed 190K kernel gaps of 5-20 us (2.2 s of a 20 s window), a
  23.8 ms step against 19.8 ms greedy and kernels 79 % busy.

## Why

The tape key held `topk: Option<u32>`, so a head that hands its logits to the host had no key. The engine fell back to
the eager path, which is correct but launch-bound at 64 rows.

## What to do

- Done: the recorded step can end in the logits (`DecodeHead::Rows` / `Drawn`, `TapeOut::Logits`, one `HeadKey::Logits`
  recording per batch shape). After the replay the keyed draws run on the device (`model::readout::draw_rows`), and the
  undrawn rows are read back whole. The draws and rows are bit-identical to the built step
  (`qwen35moe` `decode_tapes::a_sampled_step_replays_its_recording_and_draws_what_the_built_step_draws`).
- Admission's first token is picked by the device head (`admit_greedy` / `admit_topk`), not a `[vocab]` readback that
  the host scans or sorts.
- Interleaved A/B (main 407765756 against the change, 2 reps): temperature 1 at 2066 / 2088 -> 2320 / 2324 tok/s,
  serving process 21 / 20 % -> 9 / 9 % of a core; greedy 2440 / 2454 -> 2424 / 2437 tok/s (noise), 7 % either way.
- Open: the hybrid engines' decode window is 1 (a host round-trip a token, about 0.76 ms idle a step at 64 rows
  greedy), and `draw_rows` builds a `RowDrawer` (eight buffers) every step.
