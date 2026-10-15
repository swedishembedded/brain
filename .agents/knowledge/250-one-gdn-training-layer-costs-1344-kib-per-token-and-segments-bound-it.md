<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 250. One Gated DeltaNet training layer costs about 1.3 MiB per token, so the whole-sequence pair caps the sequence length; segments make the cost a constant

What was measured. `crates/model/tests/gdn_mixer_train_cost.rs` (ignored, CUDA)
runs one layer's `gdn_mixer_fwd` + `gdn_mixer_bwd` at the Qwen3.5-27B linear
attention width (48 value heads, 16 key heads, dk = dv = 128, conv kernel 4,
chunk 64) and reports the peak of the CUDA backend's exact outstanding-byte
counter above the pre-run baseline, the host kernel launches and the wall
time. One GH200, build `dev`, one run per row:

| T | path | peak MiB | KiB per token | launches | seconds |
|---|------|---------:|--------------:|---------:|--------:|
| 2048 | one-shot | 2714 | 1357 | 1312 | 1.9 |
| 8192 | one-shot | 10771 | 1346 | 4768 | 1.4 |
| 32768 | one-shot | 42999 | 1344 | 18592 | 4.1 |
| 8192 | segments of 2048 | 3646 | 456 | 6923 | 2.2 |
| 32768 | segments of 2048 | 5804 | 181 | 27695 | 7.0 |
| 131072 | segments of 2048 | 14438 | 113 | 110783 | 26.6 |

The one-shot peak is linear in T at about 1.31 MiB per token (forward alone
holds about 0.78 MiB per token; the rest is the backward's own scratch on top
of the saved set). A recompute-per-layer trainer therefore still needs one
layer's peak to fit, which is 43 GB at 32K tokens and 86 GB at 64K: the
sequence-length ceiling of the layer is set here, not by the weights.

What segmenting does. `model::gdn_segment::{gdn_mixer_seg_fwd, gdn_mixer_seg_bwd}`
run the layer over segments of `segment_tokens` rows (rounded down to a chunk
multiple, last segment shorter). The forward is the cheap non-training pass and
keeps only the recurrent state and conv history entering each segment (a few
MiB). The backward walks the segments in reverse, replays one segment's
training forward from its snapshot, and runs `gdn_mixer_stream_bwd` with the
state gradient the following segment produced. The conv history gradient
(`K-1` rows) is added into the previous segment's last rows. The residual
growth with T in the table (113 KiB per token at 131072) is the whole-sequence
interface buffers (`gated`, `d_mixed_qkv`, `d_z`, ...) the caller needs anyway,
not layer activations: the segment itself costs the same at every T.

Cost of segmenting. The backward pays one extra forward of the layer (here
7.0 s against 4.1 s at 32K for the pair). Segment boundaries add no kernel; the
row windows are `concat_split` (gather) and `splice_add` (scatter) with the
window as a kernel Param, because a storage-bind offset must be a multiple of
256 bytes (finding 216).

Launch overhead does not dominate. At 32768 tokens the one-shot pair issues
about 18.6K host launches in about 4 s; at a few microseconds per launch that is
a few percent of the wall time, and the per-layer time grows linearly with T
like the launch count does. A fused native train loop would therefore not move
the time; the work is in the batched matrix products. No fusion was built.

Boundary rule. A segment must hold at least `conv_kernel - 1` rows so the conv
history of segment s is entirely inside segment s-1; `gdn_segment` asserts it
(the first segment is the shortest only when it is also the last).
