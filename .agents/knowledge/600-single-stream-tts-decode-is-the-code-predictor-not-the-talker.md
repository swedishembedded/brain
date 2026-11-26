<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 600. Single-stream TTS decode time is mostly the code predictor, not the Talker

**Lesson:** Cutting the Talker does little for one request's latency. A Qwen3-TTS frame is one Talker step plus 15
sequential code-predictor steps, each a row of one. At this size the code predictor's per-step latency, not its FLOPs,
sets the frame time.

## What was measured
`brain serve` on one GH200 (n105, CUDA, brain `e8dc292b3`), with resident batched decode (`qwen3tts::batched`: decode
tapes as CUDA graphs) and one `design` request at a time. nsys `--trace=cuda` was used, and the time per frame was read
from the CUDA graph launches (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`):

| per frame | Sten-TTS-1.7B | Sten-TTS-950M (24 layers, MLP 2560) |
|---|---:|---:|
| Talker step graph(s) | 1.59 + 0.63 ms | 1.30 + 0.34 ms |
| code predictor: 15 x (step graph + head/draw graph) | 15 x (0.273 + 0.059) = 4.98 ms | 15 x (0.270 + 0.058) = 4.92 ms |
| GPU busy | 8.0 ms | about 6.6 ms |
| wall (span of the request's graphs) | 10.9 ms | about 9.9 ms |

- The kernels launched outside graphs were busy only about 8 % of their span.
- Nearly all GPU time is inside graphs. About 2.9 ms a frame is host time between graphs: the cb0 draw on the host and
  a stream synchronisation each frame.
- One code-predictor step reads about 160 MB of bf16 weights in 273 us. That is about 15 % of HBM bandwidth: the
  step is bound by kernel latency, not by bandwidth.

## What follows
- A cut that leaves the code predictor alone cannot speak one request much faster than the full model: 1.09x for
  Sten-TTS-950M.
- What would help, generically:
  - fusing the code predictor's 15 steps (and its per-layer kernels) into fewer, larger launches;
  - drawing cb0 on the device, so a frame needs no host round trip;
  - narrowing or shortening the code predictor in the plan (it is sensitive, see the roadmap's profile).
- Batched serving amortises all of this across streams. It is the better measure of a variant's serving cost.
