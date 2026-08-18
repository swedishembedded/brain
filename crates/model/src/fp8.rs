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
//! Import-time-only, host-only, by design: every existing device inference
//! path (fp32, [`crate::int8`]'s DP4A tier) reads plain f32/int8 - a
//! checkpoint that ships FP8 is converted to f32 (or re-quantized to int8) at
//! import, never carried compressed onto the device. A native device-side FP8
//! GEMM is deferred to a performance milestone and only if profiling says
//! arithmetic (not memory) is the limiter - a precision change is not
//! automatically a speed change.

/// `(row_blocks, col_blocks)` for an `[rows, cols]` weight at the given
/// (square) block size - `ceil(rows/block), ceil(cols/block)`, matching the
/// checkpoint's own `weight_scale_inv` shape (e.g. a real `[5120, 17408]`
/// weight's scale is `[40, 136]` = `[ceil(5120/128), ceil(17408/128)]`).
pub fn scale_shape(rows: usize, cols: usize, block: usize) -> (usize, usize) {
    (rows.div_ceil(block), cols.div_ceil(block))
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
pub fn dequant_block128(raw: &[f32], scale_inv: &[f32], rows: usize, cols: usize, block: usize) -> Vec<f32> {
    assert_eq!(raw.len(), rows * cols, "dequant_block128: raw len {} != rows*cols {}", raw.len(), rows * cols);
    let (rb, cb) = scale_shape(rows, cols, block);
    assert_eq!(scale_inv.len(), rb * cb, "dequant_block128: scale_inv len {} != {rb}*{cb}", scale_inv.len());
    let mut out = vec![0f32; raw.len()];
    for r in 0..rows {
        let br = r / block;
        for c in 0..cols {
            let bc = c / block;
            out[r * cols + c] = raw[r * cols + c] * scale_inv[br * cb + bc];
        }
    }
    out
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
}
