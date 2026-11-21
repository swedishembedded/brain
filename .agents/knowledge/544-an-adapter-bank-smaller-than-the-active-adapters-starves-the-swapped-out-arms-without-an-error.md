<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 544. An adapter bank smaller than the active adapters starves the swapped-out arms without an error

**Lesson:** `--adapter-slots N` swaps an adapter in only when a resident one is idle. When more than N adapters have
requests in flight at once and every resident one stays busy, the others never become resident: their requests wait
with no error, no refusal and no log line, while the resident arms run at full speed. Measured 2026-11-21 (lane EXP,
brain-bcd67e5ad, one GH200, Qwen3-4B `--max-batch 128 --adapter-slots 8 --serve-adapters DIR`): 18 LoRA arms, 6 agent
trials in flight each, 108 in all. After 8 minutes, 6 arms had 15-26 trials graded and the other 12 had 0, each
holding its 6 requests unanswered. The caller's exam was invalid and was discarded.

**How to apply:** until the server refuses or queues fairly (serving-runtime gap G13), give a server at least as many
adapter slots as adapters it serves concurrently (`--adapter-slots` >= active adapters; rank-16 slots are about 130 MB
each), or split arms across servers. An exam whose arms share a server must check that every arm is progressing
before reading any result.
