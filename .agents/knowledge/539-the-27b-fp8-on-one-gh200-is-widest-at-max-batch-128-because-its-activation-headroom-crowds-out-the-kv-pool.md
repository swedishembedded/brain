<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 539. The 27B-FP8 on one GH200 is widest at --max-batch 128, because its activation headroom crowds out the KV pool

**Lesson:** on one 96 GB GH200, brain-5e25d5b82 serves `Qwen/Qwen3.8-27B-FP8` usefully only up to `--max-batch 128`.
The placement reserves activation headroom that grows with the batch: 28 GB at 128, 36 GB at 160 and 44 GB at 192. That
headroom comes out of the shared KV pool, which goes from 340k tokens at 128 to 143k at 160 and to 4k (a 2,560-token
context) at 192. At 256 the model is not placed at all. Under load at 128, with up to 192 agent trials in flight, the
card used only 59-63 GB of its 97 GB, so about 30 GB of the reserved headroom stayed idle. Whether the estimate is
pessimistic for this hybrid (GDN plus attention) model has not been measured kernel by kernel. The figures are what
`brain serve -v` printed and what nvidia-smi showed.

## Evidence (lane VSFT, 2026-11-21, gpu0, default --kv-cache)

| --max-batch | --reserve-gb | activation headroom placed | KV pool (tokens) | default --ctx |
|---:|---:|---:|---:|---:|
| 128 | default (2) | 28 GB | 339,968 | 333,824 |
| 160 | default | 36 GB | 143,360 | 137,216 |
| 192 | default | 44 GB | 4,096 | 2,560 |
| 192 | 6 | - | not placed ("does not fit ... even with a one-token context") | - |
| 256 | 6 | - | not placed | - |

- A `--reserve-gb N` that is given replaces only the generic headroom of a model that does not estimate its own
  activations. At 192 and 256 it did not make the 27B fit, so the model's own estimate is what bounds the batch.
- At 128, the teacher capture ran 1,192 vim agent trials, about 70% of them graded passes, with no server errors. The
  harbor concurrency above 100 was limited by bench's gateway pool (bench 070), not by brain.
- Open: measure the 27B's real activation peak at batch 128-256 (nsys or the allocator's high-water mark) and size the
  headroom from it. If the estimate is about 2x the real peak, batch 192 with a pool of about 250k tokens would fit.
