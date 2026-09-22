<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 69. Two limits, not one: a weight can be allocatable and still unbindable, and 5 GB is neither

Placing `Qwen3.8-27B`'s two largest tensors - `tok.weight` and
`lm_head.weight`, `[248320, 5120]` fp32, 5,085,593,600 bytes each - as a
pipeline stage's `embed`/`head` endpoint (the way training's `Shard::whole`
already treats them) failed inside `paramstore::upload`, not at the planner:
5.09 GB exceeds this card's `max_buffer_size` (4,292,870,144 bytes, a real
driver/hardware ALLOCATION ceiling) by itself, and separately exceeds the
2047 MiB (2,147,483,648-byte) storage-buffer BINDING limit - the WebGPU
guarantee this whole engine is written against - by 2.4x. These are not the
same number and not the same failure mode: a buffer under the allocation
ceiling can still be too large for any ONE shader binding to reference
whole, and a cost model (or a human) that checks only the first will build a
plan that allocates fine and then cannot be bound by any kernel that needs
to read it in one dispatch.

Consequence for `int8_gguf_resident`: the vocab endpoints cannot live inside
a pipeline stage at all, sharded or not, at this model's real vocab width.
The resident holds them itself, outside the shard loop - the embedding read
one row at a time straight from the mapping (`MmapGguf::tensor_range`, the
GGUF twin of `MmapSafetensors::tensor_f32_range`, needed for exactly this),
the head as INT8 (`stream::quantize_i8_rows` + `stream::head_logits_on`,
generalized out of `crate::stream`, which had already reached the same
conclusion about the same two tensors from the training side). `LayerBytes.
embed` is correctly `0` as a result - the embedding is never resident device
memory at all, so it costs nothing to place.

The general shape: "does it fit on the card" has (at least) two independent
answers on this engine - allocatable and bindable - and a size that clears
one by a wide margin can still fail the other. Check both, separately,
before assuming a large single tensor can be a stage's own resident weight.
