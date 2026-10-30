<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 390. A recorded decode step saved nothing until its graph was cut short-first and its buffers stopped being fenced one by one, so the gap between tokens fell from 550 to 150 us

Question: knowledge 388 left ~470 us between a token's top-k read and the next step's first kernel, and ~2500 dispatches rebuilt per token.
What does recording the paged decode step (layers + head) per (rows, context bucket, head) buy, and what stops it buying it?

Measured (gpu1, FP8 Qwen3.8-27B, `--max-batch 64 --kv-cache int8 --speculative off`, same client, nsys `-t cuda,nvtx --cuda-graph-trace=node`, c = 1):

1. **Recording alone made the gap worse (550 -> 1200 us).** `Qwen35GgufInstance::decode_planned` records the step (a `Tape`: layers, final norm, head, top-k / argmax; inputs in
   `DecodeInputs`, answer in persistent buffers) once a key `(rows, attention capacity bucket, head)` has held for 3 steps, and replays it for 2 host writes and one
   launch. One graph of 2500 nodes costs 690 us of `cuGraphLaunch` before the card sees a kernel; the held pass it replaced issued a 64-step graph first.
   `Frozen` is now cut like a held pass (64 steps, then 256): 690 us -> 53 us to the first kernel.
2. **498 `cuStreamWaitEvent` a launch (350 us) were waits for events that had long completed.** `fence_in` scans every buffer a program binds (each weight) and waited
   on the foreign stream's last-write event again every launch. Fences now carry a per-stream sequence number and a handle skips an event at or before the latest record of that
   stream it has waited for; a frozen program skips the scan while no other handle has published a fence since its last launch.
3. `cuMemGetInfo` (26 us, 39 calls in the run: one per mixed/prefill round, none per decode step) was `Gpu::reclaim_ceiling_bytes` read to size a drain budget that only a
   backend with pending reclaim bytes (wgpu) uses; `Gpu::reclaim_due` asks for the ceiling only when something is pending.

The eager decode step now also builds at the bucketed capacity (`Qwen35::paged_step_shape`: longest live sequence rounded to 128 keys, block table that wide), the shape a recording is keyed on, so recorded and eager steps are bit-equal
(`tests/decode_plans.rs`: 110 steps over a bucket boundary, a shrinking batch and a head switching between greedy and top-k, bit-for-bit).

Served (same session, base run first), output tok/s at c = 1 / 4 / 16 / 64: 74.0 / 287.8 / 536.3 / 777.6 -> 80.4 / 305.9 / 580.8 / 836.6; median ITL 12.13 / 13.23 / 16.58 -> 11.32 / 12.53 / 15.61 ms;
31K TTFT 4.27 -> 4.12 s. Four greedy chats of 400 tokens are byte-identical to the base binary. c = 1 gaps above 20 us between steps: peak 1100-1300 us (monolithic graph) -> 500-600 (cut graph) -> 100-200 us (no redundant fences).

Not done, by weight: a device sampler and K-step decode (the remaining ~150 us is the top-k read, host sampling, the input writes and 3 graph launches; the sampler must reproduce
temperature/top-k/top-p/penalties/seeded draws exactly), multi-stage plans (a stage hand-off is a card-to-card copy, not a dispatch a tape holds; the tape is per stage so a cross-card plan is a list of them), `fresh` GDN segments
(2 ms per prompt-admitting iteration), the mixed tail (decode rows could go from the round's output to the head on the card), prefill attention row offsets, a lifetime-planned arena for recorded steps (each recording holds its own activations, the 64-row logits block is 63 MB; 8 are kept).
