// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side blockwise-FP8 weight dequantization - the import-time-only
//! counterpart to [`crate::int8`]'s device inference tier.
//!
//! DeepSeek-V3-style checkpoints (Qwen3.5/3.8's own FP8 release included:
//! `quantization_config: {quant_method: "fp8", fmt: "e4m3", weight_block_size:
//! [128, 128]}`) store each large 2-D weight as raw `E4M3` bytes (decoded to
//! their OWN unscaled f32 value by `checkpoint::safetensors`'s `F8_E4M3` dtype
//! arm - a byte-decode concern, not this module's) PLUS a companion
//! `<name>.weight_scale_inv` tensor: one BF16 scale per **128x128 block** of
//! the weight, `ceil(rows/128) x ceil(cols/128)` in shape. [`dequant_block128`]
//! is the second, and only remaining, step: `dequant[r,c] = raw[r,c] *
//! scale_inv[r/128, c/128]`.
//!
//! **This module stays import-time-only, host-only** - [`dequant_block128`]
//! is still the PARITY ORACLE every device-side consumer is checked against,
//! not one of two competing implementations. What changed (M8.6): a
//! checkpoint that ships FP8 no longer HAS to be converted to f32 at import
//! to be usable on a device -
//! `backend_api::DType::F8E4M3`/`F8E5M2` plus `kernels::template::
//! f8e4m3_decode_expr`/`f8e5m2_decode_expr` are a real, portable (decode-to-
//! f32-inline, plain integer/bitcast WGSL, no device feature) device tier
//! that can hold the raw bytes plus this module's own `weight_scale_inv`
//! layout resident on the device and decode inline, the same "storage tier"
//! shape `BF16`/`F16` already have. See `matmul_gemv_f8e4m3.wgsl`/
//! `matmul_gemv_f8e5m2.wgsl`'s own headers for the device kernel and
//! `crates/model/tests/matmul_fp8_gemm.rs` for its parity test against THIS
//! module's `dequant_block128`. Still out of scope, and still needing
//! hardware this repo's own boxes do not have: a NATIVE tensor-core FP8 GEMM
//! (Hopper+/Blackwell) - a precision change is not automatically a speed
//! change, and that tier is deferred to whichever future session can measure
//! it for real, per `brain_testutil::skip_unvalidated_capability`'s own
//! hardware-harness contract.

/// `(row_blocks, col_blocks)` for an `[rows, cols]` weight at the given
/// (square) block size - `ceil(rows/block), ceil(cols/block)`, matching the
/// checkpoint's own `weight_scale_inv` shape (e.g. a real `[5120, 17408]`
/// weight's scale is `[40, 136]` = `[ceil(5120/128), ceil(17408/128)]`).
pub fn scale_shape(rows: usize, cols: usize, block: usize) -> (usize, usize) {
    (rows.div_ceil(block), cols.div_ceil(block))
}

/// Fill one row's worth (`cols` elements) of the dequantized output from
/// that row's own raw slice and the shared scale grid. Walks column BLOCKS,
/// not columns, so the block index (`bc = c / block`) is computed once per
/// block rather than once per element - `import_layer`'s real `[5120,
/// 17408]` `mlp.down_proj` has 136 column blocks but 17408 columns, so this
/// is a 128x reduction in integer divisions on the hot path, before any
/// parallelism or vectorization. The per-block inner loop is then a plain
/// contiguous `out[c] = raw[c] * scale` - exactly the shape LLVM
/// auto-vectorizes into packed SIMD multiplies without any hand-written
/// intrinsics, matching this crate's `matvec`/`hostmath.rs` precedent of
/// preferring an already-vectorizable loop shape over a hand-rolled kernel
/// when the compiler already gets there.
#[inline]
fn dequant_row(raw_row: &[f32], scale_inv: &[f32], cb: usize, br: usize, block: usize, cols: usize, out_row: &mut [f32]) {
    let mut c = 0usize;
    while c < cols {
        let bc = c / block;
        let scale = scale_inv[br * cb + bc];
        let end = ((bc + 1) * block).min(cols);
        for cc in c..end {
            out_row[cc] = raw_row[cc] * scale;
        }
        c = end;
    }
}

/// Multiply every element of a raw (unscaled) `[rows, cols]` FP8-decoded
/// weight by its `128x128`-block scale, producing the final f32 weight.
/// `scale_inv` is `[scale_shape(rows, cols, block)]`, row-major, already
/// decoded to f32 (`checkpoint::safetensors`'s `BF16` dtype arm does that -
/// the scale tensor itself is plain BF16, never FP8).
///
/// Panics on a shape mismatch - a wrong block size or a raw/scale tensor
/// pairing mix-up is exactly the "params struct wrong order" class of bug
/// that must fail loudly, not silently compute over the wrong slice.
///
/// Native (non-wasm32) builds fan the row loop out across
/// [`backend_cpu::par`]'s pool - every row is an independent write with no
/// cross-row reduction, so this changes nothing about the arithmetic itself
/// (same per-element `raw * scale`, same fp order within a row), only the
/// schedule. wasm32 (no `backend-cpu` dependency there, see this crate's
/// `Cargo.toml`) keeps the equivalent sequential loop over the same
/// per-block-not-per-element addressing.
pub fn dequant_block128(raw: &[f32], scale_inv: &[f32], rows: usize, cols: usize, block: usize) -> Vec<f32> {
    assert_eq!(raw.len(), rows * cols, "dequant_block128: raw len {} != rows*cols {}", raw.len(), rows * cols);
    let (rb, cb) = scale_shape(rows, cols, block);
    assert_eq!(scale_inv.len(), rb * cb, "dequant_block128: scale_inv len {} != {rb}*{cb}", scale_inv.len());
    let mut out = vec![0f32; raw.len()];

    #[cfg(not(target_arch = "wasm32"))]
    {
        backend_cpu::par::rows_mut(&mut out, cols, |r, out_row| {
            let br = r / block;
            let raw_row = &raw[r * cols..r * cols + cols];
            dequant_row(raw_row, scale_inv, cb, br, block, cols, out_row);
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        for r in 0..rows {
            let br = r / block;
            let raw_row = &raw[r * cols..r * cols + cols];
            let out_row = &mut out[r * cols..r * cols + cols];
            dequant_row(raw_row, scale_inv, cb, br, block, cols, out_row);
        }
    }
    out
}

/// Decode an `E5M2` (OCP FP8) byte to its OWN f32 value: 1 sign, 5 exponent
/// (bias 15), 2 mantissa bits - unlike [`checkpoint::safetensors::e4m3fn_to_f32`]'s E4M3FN
/// (this checkpoint format's usual raw byte, decoded by that function), E5M2
/// HAS real infinities (`exponent==31, mantissa==0`), only `mantissa!=0` at
/// `exponent==31` is NaN. No checkpoint import path in this tree reads E5M2
/// today (`checkpoint::safetensors`'s own `F8_E5M2` arm is a loud `Err`) -
/// this exists as the HOST-side parity oracle for `kernels::template::
/// f8e5m2_decode_expr`'s device decode (M8.6), the same role
/// [`checkpoint::safetensors::e4m3fn_to_f32`] already plays for E4M3.
///
/// Direct sign/exponent/mantissa field reconstruction (not a bit trick),
/// same style as `checkpoint::safetensors::e4m3fn_to_f32_scalar` - this is
/// the ORACLE the device decode is checked against, so it must be the most
/// obviously-correct implementation available, not the fastest one.
pub fn e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (b >> 2) & 0x1F;
    let mant = (b & 0x03) as f32;
    if exp == 0x1F {
        if mant == 0.0 {
            return sign * f32::INFINITY;
        }
        return f32::NAN;
    }
    if exp == 0 {
        // subnormal: mant/4 * 2^(1-bias), bias=15 -> 2^-14
        return sign * (mant / 4.0) * 2f32.powi(-14);
    }
    // normal: (1 + mant/4) * 2^(exp-bias)
    sign * (1.0 + mant / 4.0) * 2f32.powi(exp as i32 - 15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_shape_matches_the_real_checkpoints_own_reported_shape() {
        // mlp.down_proj.weight [5120, 17408] -> weight_scale_inv [40, 136],
        // read directly off the real Qwen3.8-27B-FP8 checkpoint's tensor
        // headers.
        assert_eq!(scale_shape(5120, 17408, 128), (40, 136));
    }

    #[test]
    fn single_block_dequant_is_a_plain_scalar_multiply() {
        let raw = vec![1.0, -2.0, 4.0, 0.5];
        let out = dequant_block128(&raw, &[3.0], 2, 2, 128);
        assert_eq!(out, vec![3.0, -6.0, 12.0, 1.5]);
    }

    #[test]
    fn each_block_gets_its_own_independent_scale() {
        // 2 blocks of width 2 side by side (rows=1, cols=4, block=2):
        // columns 0-1 use scale_inv[0], columns 2-3 use scale_inv[1].
        let raw = vec![1.0, 1.0, 1.0, 1.0];
        let out = dequant_block128(&raw, &[2.0, 5.0], 1, 4, 2);
        assert_eq!(out, vec![2.0, 2.0, 5.0, 5.0]);
    }

    #[test]
    fn two_dimensional_block_grid_indexes_row_and_column_blocks_independently() {
        // 4x4 raw at block=2 -> a 2x2 grid of scales. Row block and column
        // block must each select the RIGHT scale, not just "a" scale - this
        // is exactly the kind of axis-order bug a same-value fixture would
        // hide, so every block gets a distinct value.
        #[rustfmt::skip]
        let raw = vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        // scale_inv row-major [2,2]: top-left=10, top-right=20, bot-left=30, bot-right=40.
        let scale_inv = vec![10.0, 20.0, 30.0, 40.0];
        let out = dequant_block128(&raw, &scale_inv, 4, 4, 2);
        #[rustfmt::skip]
        let expect = vec![
            10.0, 10.0, 20.0, 20.0,
            10.0, 10.0, 20.0, 20.0,
            30.0, 30.0, 40.0, 40.0,
            30.0, 30.0, 40.0, 40.0,
        ];
        assert_eq!(out, expect);
    }

    #[test]
    fn round_trip_through_a_real_quantize_recipe_stays_within_e4m3_precision() {
        // Mirrors the real quantization recipe (amax-per-block scale to the
        // e4m3fn dynamic range, ~448) rather than asserting against
        // hand-picked numbers - this is the "settle with an experiment"
        // gate for the multiply-not-divide `weight_scale_inv` convention
        // (DeepSeek-V3/Qwen3.5-FP8's own documented format): if the
        // direction were backwards, this round trip would be off by the
        // SQUARE of the scale, not just imprecise, and this assertion would
        // fail by orders of magnitude rather than by e4m3's few percent.
        let rows = 4;
        let cols = 4;
        let block = 2;
        let original: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 8.0) * 1.3).collect();
        let (rb, cb) = scale_shape(rows, cols, block);
        let mut scale = vec![0f32; rb * cb];
        let mut raw = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let br = r / block;
                let bc = c / block;
                let mut amax = 0f32;
                for rr in br * block..((br + 1) * block).min(rows) {
                    for cc in bc * block..((bc + 1) * block).min(cols) {
                        amax = amax.max(original[rr * cols + cc].abs());
                    }
                }
                scale[br * cb + bc] = amax / 448.0;
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                let br = r / block;
                let bc = c / block;
                let s = scale[br * cb + bc].max(1e-12);
                raw[r * cols + c] = (original[r * cols + c] / s).clamp(-448.0, 448.0);
            }
        }
        let dequant = dequant_block128(&raw, &scale, rows, cols, block);
        for (o, d) in original.iter().zip(dequant.iter()) {
            let rel = (o - d).abs() / o.abs().max(1.0);
            assert!(rel < 0.05, "round trip {o} -> {d} exceeds e4m3 precision (rel {rel})");
        }
    }

    #[test]
    #[should_panic(expected = "raw len")]
    fn wrong_raw_length_panics_loudly() {
        dequant_block128(&[1.0, 2.0], &[1.0], 2, 2, 128);
    }

    #[test]
    #[should_panic(expected = "scale_inv len")]
    fn wrong_scale_length_panics_loudly() {
        dequant_block128(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0], 2, 2, 1);
    }

    /// Spot-checks the same edge cases `checkpoint::safetensors`'s own
    /// `e4m3fn_decode_known_values` test pins for E4M3, for E5M2's
    /// (different) layout - the two formats disagree on all of these (E5M2
    /// has real infinities, a different max finite value, a different
    /// smallest-normal/subnormal), which is the whole point of them being
    /// separate `DType` tiers, not one decode with two labels.
    #[test]
    fn e5m2_decode_known_values() {
        assert_eq!(e5m2_to_f32(0x00), 0.0);
        assert!(e5m2_to_f32(0x00).is_sign_positive());
        assert_eq!(e5m2_to_f32(0x80), -0.0);
        assert!(e5m2_to_f32(0x80).is_sign_negative());
        assert_eq!(e5m2_to_f32(0x3C), 1.0); // exp=15(bias 0), mant=0
        assert_eq!(e5m2_to_f32(0xBC), -1.0);
        assert_eq!(e5m2_to_f32(0x04), 2f32.powi(-14)); // smallest normal
        assert_eq!(e5m2_to_f32(0x01), 2f32.powi(-16)); // smallest subnormal
        assert_eq!(e5m2_to_f32(0x7B), 57344.0); // largest finite (exp=30,mant=3 -> 1.75*2^15)
        assert_eq!(e5m2_to_f32(0x7C), f32::INFINITY); // E5M2 HAS infinities, unlike E4M3FN
        assert_eq!(e5m2_to_f32(0xFC), f32::NEG_INFINITY);
        assert!(e5m2_to_f32(0x7D).is_nan());
        assert!(e5m2_to_f32(0x7F).is_nan());
        assert!(e5m2_to_f32(0xFF).is_nan());
    }
}
