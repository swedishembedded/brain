<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 555. A brain checkpoint read without its packed view has no stored tier, so a bf16 export is built at fp32

**Lesson:** open a brain `.safetensors` checkpoint through `checkpoint::packed::stored_source`, never as a bare
`WeightReader`, whenever anything downstream asks what a tensor is stored as. A bare reader answers every
`packed_descriptor` with `None`: `model::ops::stored_tier_policy` then returns an empty policy (every tensor fp32) and
uploads go through a whole-tensor f32 decode. Nothing fails; the model is simply built at twice the bytes.

## What happened

`qwen35 generate` on the bf16 checkpoint a parameter-sharded 27B fine-tune exports needed more than 200 GB of host
memory: it read the file with `checkpoint::load` (every tensor as f32, about 108 GB) and cloned the map with
`by_role` (another 108 GB). Streaming the file instead (`qwen35::base_source::open_base`) fixed the host peak, but the
new default precision, "as stored", ran out of device memory on a 96 GB card: `open_base` handed out a bare
`WeightReader`, the stored-tier policy came back empty, and every linear and table was built at fp32 (about 108 GB).
`brain serve` already wrapped the same file in `stored_source` and built it at bf16, so the two paths disagreed about
one file.

## What to do

`open_base` now returns `stored_source(reader)`, the same view `brain serve` reads. A test that relies on a stored
tier asserts it (`model::ops::stored_tier(src, name) == Some(BF16)`), as qwen35's `exported_checkpoint` test does,
instead of trusting greedy tokens to tell fp32 from bf16: on a small model they often agree.

## Evidence (2026-11-22, n442 card 3, Qwen3.8-27B in the sharded export layout, 53.8 GB)

Host peak (`ru_maxrss`) of `brain qwen35 generate`, 16 greedy tokens: 198.4 GiB at `--precision int8` before; after,
0.49 GiB (default, bf16), 0.46-0.60 GiB (int8), 0.53 GiB (bf16), with the same 16 tokens. With the bare reader the
default build failed with `cuMemAlloc failed: out of memory`.
