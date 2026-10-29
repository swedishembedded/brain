<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 375. A prompt read in beside decoding sequences needs its own slice of the card, not a share of it

Question: after #374 removed the scheduler-thread stalls, `perf run sweep --workload chat --ladder 1,4,16` (Qwen3.8-27B
Q8_0, GH200 gpu1, seed 1) still gave inter-token p99 46 / 121 / 180 ms against the 50 ms interactive budget, goodput 0 from
concurrency 4. What is left, and what does each part cost?

Measured (per-iteration host timestamps of the scheduler's prefill and decode phases, temporary; `nsys` for the driver
calls):

1. **Decode steps share the card with a recorded prefill queue.** With no prefill, plain decode steps are 14 / 20 / 22 ms
   at 1 / 4 / 8 sequences; with a prompt going in beside them 35-40 ms (b=4), whatever the pump's host budget. The pump
   recorded up to 16 ms before and again after every step, so a gap was pump + a slowed step + pump (up to 90 ms).
2. **Speculation was the larger share at 4+ sequences.** A speculative step (draft + verify) is 50-60 ms at b=4 against
   20 ms plain; the gate ranked them by tokens a second, so every burst of tokens arrived 50-60 ms apart. Plain-only
   (temporary switch) gave p99 17 ms and goodput 57.9 at concurrency 1 (64.5 tok/s with speculation, goodput 36).
3. **The MTP prompt feed was 42 ms synchronous per prompt, and most of it was host work:** the embedding rows one
   `tensor_range` at a time (3.4 ms per 128 rows) and 22 ms of scalar norms for 1000 rows on one thread; the device part
   is 1.1 ms of projection and 1.4 ms of layer per 128 rows.
4. **A 1731-token prompt alone went in at half speed in the background** because the round was 2048 rows wide:
   2048 rows 1.56 s, 1024 rows 0.97 s, 512 rows 0.75 s, 256 rows 0.86 s (sliced at 512: 0.76 s). The widest round is
   not the fastest one on this card.
5. **Dead-uniform `cuMemFree`s are not a contributor.** `release_dead_uniforms` freed ~700 blocks in 2-3.5 ms, 12 times in
   24 requests, never behind a long queue; `nsys` over a 12-request run shows 22 `cuMemFree_v2` calls in the whole
   process (median 7.5 us; the one slow call, 84 ms, is outside the token loop) and no `cuMemFreeHost`.

Changes: (a) background rounds are at most 512 rows (`BACKGROUND_ROUND_ROWS`); (b) `PrefillPace` gives a prompt one
slice of device time per scheduler iteration - what `PrefillPolicy::step_target` (45 ms) leaves after the measured decode
iteration, at least a quarter of it - recorded as that many layers of the round and *waited for* before the decode step
(`settle_prefill`), so the step runs on an idle card; the layer cost is timed per power-of-two round size, because a
100-row round costs 0.6 ms a layer for 512 rows' 2.9 ms and a per-row rate is wrong for both; the pumps inside the
decode step are gone; (c) the MTP feed's host side runs on several threads with one parallel embedding gather
(42 -> 22 ms) and is begun at once and run 256 rows with each slice (`DeferredFeed`), the slot neither drafting nor
recording steps until it is done and the prefix-cache entries filed from the prompt marked valid for the head only then;
(d) `CostGate` does not speculate where a speculative step exceeds `PagedDecoder::step_latency_target` and a plain one
does not (retried 1 in 384 steps), and a prompt's drafter preparation is then not paid for.

Result (same command, 48 requests, ladder 1,4,16; inter-token p99 / max ms, output tok/s, TTFT p99 ms, goodput req/s):
46 / 76, 121 / 170, 180 / 198 ms; 64.5, 138.5, 181.6; 473, 1969, 9477; 36.3, 2.9, 0.0 became 39 / 77, 63 / 97, 68 / 125 ms;
58.2, 102.2, 120.8; 478, 3023, 14842; 47.3, 2.1, 0.0. The cadence now holds at concurrency 1 and is near it at 4
(p95 47 ms); the price is prefill throughput: closed-loop chat at 4-16 clients is prefill-bound (0.45 s of prefill and 0.24 s
of decode share per request), so a prompt given 40-50 % of the card queues the next ones longer. A 1024-token prompt cannot
meet a 2 s TTFT and a 50 ms interval at 4 simultaneous arrivals on one card; the `step_target` trades one for the other
(60 and 90 ms runs were not monotonic within the sweep's noise, so no curve is claimed).

Not done: iterations in which a decode step is 40-60 ms against 18 ms for no visible reason (about 5 % of iterations at
b=5-6, alternating with fast ones: a graph recapture on a changed batch is the suspect, not confirmed); `step_target` is a
policy default, not a serve flag yet; the device-side sharing itself (a decode step beside a running round) is avoided, not
reduced.

Rule: time-slice the card between a latency loop and a throughput stream when sharing slows the first by 1.5-2x, and size
the slice from the measured iteration and a layer cost kept per round size; time a slice only when it ran the whole
number of layers it asked for.
