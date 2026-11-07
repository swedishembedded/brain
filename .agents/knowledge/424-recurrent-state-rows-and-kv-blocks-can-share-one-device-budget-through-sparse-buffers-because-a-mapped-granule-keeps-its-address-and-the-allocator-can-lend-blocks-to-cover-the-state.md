<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 424. Recurrent-state rows and KV blocks can share one device budget through sparse buffers, because a mapped granule keeps its address and the block allocator can lend blocks to cover the state

**Lesson:** Recurrent-state rows and KV blocks can share one device budget without touching a
kernel, if both live in sparse buffers whose granules are mapped on demand at stable addresses and
the block allocator lends blocks to cover the state.

## What happened

Question (the brief of 2026-11-01): the 27B engine reserved a state row (150 MB f32, 75 MB bf16, 48
layers) for each of 64 slots at load, so 48 idle slots held 7 GB (110K tokens of bf16 KV) while 64K
x 16 was bound by KV capacity. vLLM holds both in one pool. How do the KV planes and the state pools
share one budget without touching a kernel?

## What to do

- **One arena, kernels re-addressed**: rejected. Kernels index a plane with `u32` elements (4 GiB),
  and a 50 GB arena does not fit; every attention and state kernel would take a stride.
- **vLLM's equal pages** (a block padded to a state page, layers grouped so a tensor serves one
  attention and three GDN layers): rejected here, K and V are separate planes and the state kernels
  address `pool[row * len]`.
- **Sparse buffers (kept)**: the KV data planes and the state pools are address ranges reserved at
  load (`Backend::alloc_sparse`, CUDA virtual memory management) with 2 MiB granules backed on
  demand. Addresses never change, so kernels, recorded graphs and `u32` indexing are untouched. A KV
  block is one granule of each plane (1024 rows bf16, 2048 int8/fp8:
  `KvTier::block_rows_for_granule`); a state row maps the granules of its byte range (3 MiB f32 rows
  share a granule with their neighbour: 2, 3, 5, 6 granules for 1, 2, 3, 4 rows).
- **One budget by lending**: the engine lends the allocator's coldest free blocks to cover the
  memory the backed state granules take (`BlockAllocator::lend`, `unipool::StateLedger`), so backed
  blocks + backed state never exceed the planned pool, and the generic scheduler needs one defaulted
  method (`PagedDecoder::sequence_state_blocks`, charged to a sequence that has no state row yet).

## Evidence

### Measured (GH200, gpu1)

- `cuMemMap` takes a whole physical allocation (offset must be 0): one handle backs one granule.
  Create 19 us (zeroing), map 1 us, set access 35 us: backing a block (32 planes) 1.8 ms, unbacking
  1.3 ms; paid once per block (a freed block keeps its memory).
- Pool tokens, bf16 KV / f32 state, 64 slots: static 622592 at any number running; unified 749568
  with none, minus the state of the sequences running (3, 5, 8, 9 blocks for 1, 2, 3, 4 rows; 3 x
  1024 tokens a sequence at most). fp8 KV / bf16 state: 1187840 static; 1294336 unified with none
  running.
- Exactness: a unified pool serves the greedy tokens of the separate pools at the same block size,
  through slots that are reused (`unified_pool_real.rs`).

### Rules

- A small tiny-model test cannot assert token equality across schedules: the scheduler's charge
  changes the batch composition, and this model's lone-row kernel differs from its batched one
  (`legacy running 1` differed from 4 at the same pool).
- `prompts_sharing_a_document_resume_from_where_they_part` fails on origin/main at 76243a969 already
  (warm vs cold hidden row differ bitwise); not caused by this change.
