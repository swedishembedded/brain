<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 216. A storage-buffer bind offset that is not a multiple of 256 bytes is rejected, so a row window is a kernel Param

What holds. A storage-buffer binding offset (what `step_sliced` passes for a
sliced buffer) that is not a multiple of 256 bytes is rejected by wgpu: it
validates the offset against the device's `min_storage_buffer_offset_alignment`,
which is 256 bytes on the devices this tree targets. A row of a `u32` slot
vector is 4 bytes and a row of an activation is `k * 4` bytes, so an arbitrary
first row of a window is almost never a legal offset.

What the kernels do instead. Every buffer is bound whole, and the window
(`row0`, `rows`) travels as a uniform parameter that the kernel adds to its
row index. `model::dispatch::lora_bank_rows` documents this at its binding
site, and `crates/model/tests/lora_bank_rows.rs` exercises a window that
starts inside a larger batch.

Consequence for callers. A per-row index that must be addressable from any
row (a slot vector, a row-to-sequence map) has to be a whole-buffer binding
plus a Param offset, never a sliced binding. Slicing is safe only where the
offset is a compile-time multiple of 256 bytes, such as a head-row chunk of
64 rows of a `u32` or `f32` per-row buffer (`HEAD_ROW_ALIGN` in
`crates/qwen3/src/model.rs`). The serving engine's slot vector follows the
rule: `crates/qwen3/src/serve/adapters.rs` binds it whole.
