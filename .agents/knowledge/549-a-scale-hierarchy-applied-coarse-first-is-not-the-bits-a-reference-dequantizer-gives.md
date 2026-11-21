<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 549. A scale hierarchy applied coarse level first is not the bits a reference dequantizer gives

**Lesson:** `checkpoint::packed`'s codec multiplied a decoded tile by its scale levels in declaration order, coarse to
fine: NVFP4 as `(e2m1 * global) * e4m3`. Reference loaders compute `(e2m1 * e4m3) * global`. The two are equal in
exact arithmetic but not in f32: `e2m1 * e4m3` is exact (two narrow mantissas), while `e2m1 * global` rounds a 24-bit
global, so the coarse-first order differed from the reference in 938,913 of 5,308,416 values of one real Inkling 2
Mini expert slab (layer 0 `w13`, expert 5). Applying the finest level first makes every narrow product exact and
leaves one rounding, at the f32 level; the slab then matches bit for bit, and so do the WGSL NVFP4 MoE kernels, which
already folded in that order. Found 2026-11-21 with `a_decoded_expert_slab_equals_the_reference_dequantization_bit_for_bit`.

The same release showed two more facts a loader must not paper over:
- Its `hf_quant_config.json` declares MXFP8 experts with `weight_block_shape: [32, 32]`, but the stored E8M0 scale is
  one per row per 32 columns (`[experts, rows, cols / 32]`) and differs between rows of one 32-row band. The stored
  grid is what decodes; the declared block's leading extent is only checked to divide the rows.
- `config.json`'s global `quantization_config` (`modelopt`, `NVFP4`) is wrong for most tensors; the per-tensor
  `parameter_groups` are the declaration, and a tensor they do not name is refused, not guessed.

**How to apply:** a multi-level scale is applied finest first (`apply_scales` iterates levels in reverse); a device
decode that claims parity with the codec folds in the same order. When a release declares its quantization per tensor,
cross-check every declaration against the stored dtype and each sidecar's shape, and read granularity from the stored
grid, never from a declared tile alone.
