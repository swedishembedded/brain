<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 571. A query-tiled attention keeps a row's bits only on key tiles at absolute positions

**Lesson:** a serving engine computes one position in whatever dispatch its prompt chunk or decode step lands in, and
a cached or chunked engine is held to the whole-sequence forward bit for bit. An online-softmax attention that rescales
once per key tile reassociates its sums by the tile boundaries, so a row's bits are a function of where its tiles
start. Tiles that start at the first key a dispatch reads (the chunk's window start, `lo` of its first row) move with
the chunking; tiles at absolute positions (`[128 c, 128 c + 128)`, a key outside the row's range masked to an exact zero
in the sum and skipped in the value mix) do not. Two kernels a selection switches between by row count (a per-row form
for decode, a query-tiled form for prompt chunks) must share that grid and do every per-row operation in the same
order: the dot product over the head ascending, the bias, the tile maximum, the exp-sum's exact tree, `l * corr + sum`,
the rescale and the value mix key by key. Extra all-masked tiles a block carries for its other rows are exact no-ops
(corr is exactly 1, the sum exactly 0), and so are staged zero columns past the head width (a sum that starts at +0 is
never -0). Hold it by a gate that runs every candidate the selection can name whole, in chunks and row by row and
compares bits, not by an oracle tolerance: the oracle gates pass a kernel that breaks it.

## Evidence (2026-11-22)

- A first query-tiled `flash_attn_gqa_ext` (16 rows x 32-key tiles starting at the block's first key) passed the f64
  oracle gates on CUDA and Vulkan and failed `model` `serve_visit_state::served_logits_equal_the_stateless_whole_
  sequence_forward`.
- `flash_attn_gqa_ext_tiled` (32 rows, 128-key tiles at absolute positions) and `flash_attn_gqa_ext_rows` (tiles moved
  to the same grid) give identical bits on CUDA and on Vulkan (GH200) whole, in chunks of 1 to 130 rows, row by row and
  through a gathered ring (`model` `flash_ext::every_forward_candidate_computes_a_row_the_same_however_its_prompt_is_cut`);
  starting the tiled kernel's tiles at the block's first key instead fails it (26 of 8,704 values of a 17-row chunk).
- The CUDA backend compiles WGSL with `--fmad=false`, which makes two differently shaped kernels' `a * b + c` the same
  two roundings; a backend that contracts multiply-adds on its own is held by the same gate.
