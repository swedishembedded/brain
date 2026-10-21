<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 349. A brain safetensors file with block-scaled FP8 tensors, opened through the plain reader, returned the raw 8-bit values without their scales

Symptom: a model exported in FP8 by the resident full fine-tune (`Qwen::export_resident`) and reloaded with
`qwen3::open_checkpoint` computed a different loss from the model it was exported from (3.5138 against 3.3998 on a tiny
model), while the same model exported in bf16 matched exactly. Reading one linear back through the source showed values
up to 448 away from the exported master: the FP8 codes with no scale applied.

Cause: `open_checkpoint` opened a HF directory through the packed view (`PackedSource`, which applies the
`weight_scale_inv` sidecar) but a single brain file through `WeightReader` directly, whose `tensor`/`with_tensor` decode
`F8_E4M3` to its unscaled values. The scale sidecar tensors were read as unrelated tensors. Nothing failed: the weights
were finite and merely wrong.

Fix: a file that holds any `F8_*` tensor is opened through `PackedSource::identity_lenient` over the `PackedReader`, so the
scales are applied; every other file keeps the plain reader (and its zero-copy word path). Test:
`the_exported_model_reloads_and_computes_the_trained_loss` (`fp8 to fp8` is exact).

Related, found while measuring the same run: `Rule::update` recomputed AdamW's two bias corrections (two `powi`) for every
element; they are now computed once per step (`StepScalars::bias_corrections`).
