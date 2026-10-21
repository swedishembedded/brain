<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 348. A stored int8 linear was requantized on load because its scale sidecar had two names, and the serving residents expanded every checkpoint to f32

Symptom (read from the code, then reproduced by a red test): the Qwen3.8 dense and MoE catalog residents opened a
brain checkpoint with `checkpoint::load(path)` and `by_role("")`, one `Vec<f32>` per tensor for the whole model, and
built the engine from that map. A bf16 checkpoint was held as f32 on the host (twice the file), and a packed-int8
checkpoint (`u32` words plus `<name>.scale`) panicked in `tensor_f32` because a `U32` tensor has no f32 decode.

Two causes, one of them hidden behind the other:

1. The residents had no packed view of a brain-native file. `qwen3::open_checkpoint` gave an F8 file a `PackedSource` and
   every other file a plain `WeightReader`, so bf16 and int8 tensors were only reachable through the f32 interface.
2. Brain's int8-native checkpoints (`weightio::PACKED_INT8_LAYOUT`) name the group-scale sibling `<name>.scale`; the
   shared codec's `QuantEncoding::int8_blocked` named it `<name>_scale`. `QuantEncoding` equality includes the
   sidecar suffix, so `Loader::plan` saw "int8 to int8" as a different encoding, took the Convert path and requantized the
   decoded values. A group whose largest code is under 127 comes back with a different scale and different codes
   (the regression test sees the requantized words differ from the stored ones), so serving "the same int8
   weights" silently served another quantization of them.

Fix: `int8_blocked` uses `.scale` (`INT8_SCALE_SUFFIX`), `PackedReader` recognises the layout from the file alone (a
`U32` tensor with an f32 `.scale` sibling, logical width four times the stored one), and `checkpoint::packed::stored_source`
opens any safetensors file with reduced-precision, FP8 or packed-int8 tensors through that view. The residents build with
`model::ops::stored_tier_policy`, which holds each linear at the tier of its stored encoding (`stored_tier`), and
`model::int8::quantize_from` returns a stored int8 tensor's own codes. The tests are `catalog/tests/resident_packed.rs`
(logits equal the decoded-weights engine for f32, bf16 and int8 files on the host tier; the heap held beyond the engine
stays under a quarter of the model's f32 size where the old whole-container load holds all of it) and
`model/tests/upload_from_source.rs`.

Trap for the next reader: the host tier (`gpu_core::devices::with_host_tier`, the copy a residency fallback builds) has no
packed-int8 dot, so `encoding_of` refuses the int8 tier there and an int8 checkpoint widens to f32 on the CPU backend by
device capability, one tensor at a time. That is correct and bounded, not a leak; the int8 residency is asserted on a
device with `numeric.int8_dot`. A GGUF is not a brain checkpoint: `open_stored_checkpoint` refuses it by name instead of
panicking inside the loader.
