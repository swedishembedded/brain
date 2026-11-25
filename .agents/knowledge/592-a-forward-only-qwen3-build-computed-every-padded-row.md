<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 592. A forward-only qwen3 build computed every padded row

**Lesson:** `Qwen::needed_len` and `needed_rows` trimmed a batch to its last supervised row only on a build with a
recorded backward, so a forward-only (inference) build fed the same batch as a training build computed the whole
`[b, t]` block, and laid its hidden states out at `s * t + r` where the trained build had `s * run + r`. A frozen
teacher run beside a trimming student therefore paid for every padding row (a length-bucketed speech batch is mostly
padding at the block's length, #586) and its rows did not line up with the student's. A forward-only build fed input
rows (`enable_input_rows`, a decoder whose every row is assembled by its caller) now trims exactly as a training build
does, records no backward for a trimmed run, and the distillation checks that the teacher computed the student's
layout (`computed_layout`) before it reads a row.

Other forward-only builds still keep the whole row: their callers (`logits_all`, `score_positions`) read every row.

Where: `crates/qwen3/src/model.rs` (`trains_or_reads_rows`, `run_tape`, `new_frozen_on`).
