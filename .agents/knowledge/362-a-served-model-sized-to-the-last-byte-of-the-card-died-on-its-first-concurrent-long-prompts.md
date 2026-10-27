<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 362. A served model sized to the last byte of the card died on its first concurrent long prompts

Symptom: `brain serve` on a 98 GB GH200 logged `default --ctx 106496: the longest
context the cards hold`, answered a short prompt, then panicked the lane with
`cuMemAllocFromPoolAsync failed: out of memory` (48 MiB, in a prefill round) on
the first agent-sized requests (5-8k tokens, several at once). `--ctx 65536`
worked.

Cause: the placement charged weights, KV and recurrent state only. A fixed 1 GiB
per card stood in for the transient activations of a prefill round, but a round
of 1024 rows at the 27B shape peaks at 1.06 GiB of live device memory (measured
with the allocator's own `peak_live`, because the host queues layers ahead of the
device and several layers' buffers are live together), on top of ~1.5 GiB the
process holds outside the pool (context, library handles). The automatic context
therefore left less than the workspace, and the operator-fixed `--ctx` was not
checked against it at all.

Fix: `qwen35::int8_gguf_resident::prefill_workspace_bytes` derives the workspace
from the config and the rows of the largest prefill forward
(`ServingShape::prefill_rows`, widened by the batched-prefill setting), and the
planner keeps it free on every card for the automatic context, for an explicit
`--ctx` (refused with the arithmetic) and for the adapter-bank placement.
`tests/gguf_resident_real.rs` measures a real round and fails if the planned
figure ever falls under it.

Rules: a placement that sizes persistent state to the card must subtract the
transient peak of every path that runs after load, as a function of the same
knobs that size that path; a constant headroom is a guess that holds until the
shape changes. Measure the peak with the allocator's counter around the real
path, not by sampling the card.
