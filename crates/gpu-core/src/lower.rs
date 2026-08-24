// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Conv-as-GEMM lowering: the scratch budget and the chunk arithmetic every
//! lowering in this tree shares.
//!
//! A convolution lowered to a GEMM materialises an `im2col` operand that is
//! `K` (1D), `K*K` (2D) or `KT*KH*KW` (3D) times the input. For real shapes
//! that operand runs to gigabytes - over
//! [`Gpu::max_storage_binding_bytes`](crate::Gpu::max_storage_binding_bytes)
//! on cards that report well under 2 GiB, so it is not merely wasteful but
//! *unbindable*. Every lowering therefore does the same two things: cap the
//! scratch at a budget, and process `floor(budget / row_floats)` output
//! positions per GEMM.
//!
//! That arithmetic lived in three private copies (`vae::blocks`,
//! `vae::blocks3d`, and now `audio::conv`), two of them byte-identical. It is
//! here so a caller tuning device memory tunes every lowering at once, and so
//! the "snap the chunk to a whole GEMM tile" rule cannot be got right in one
//! copy and wrong in another.

/// Ceiling on the im2col scratch, in f32 words.
///
/// `BRAIN_CONV_COL_MIB` overrides it; `BRAIN_VAE_COL_MIB` is honoured as the
/// original name of the same knob, from when only the VAE lowerings had one.
/// Bigger means fewer, larger GEMMs and more resident scratch - the trade is
/// device memory, which is why it is a knob at all: a lowered conv running
/// beside a resident DiT has far less of it to spend.
pub fn col_budget_floats(default_mib: u64) -> u64 {
    let mib = std::env::var("BRAIN_CONV_COL_MIB")
        .or_else(|_| std::env::var("BRAIN_VAE_COL_MIB"))
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_mib);
    mib.max(1) * 1024 * 1024 / 4
}

/// Output positions per lowered GEMM: as many rows of `row_floats` as the
/// budget holds, snapped DOWN to a whole `tile` (so no chunk leaves a GEMM
/// tile part-empty except the last), then held to at least `tile` and never
/// more than `max(total, tile)` - the caller still takes `min(chunk, total -
/// pos)` per chunk, so a `total` under one tile is a single short chunk.
///
/// `row_floats` is the im2col row width (`Cin*K`, `Cin*K*K`, `Cin*KT*KH*KW`);
/// `tile` is the GEMM's row-tile height (128 for the `matmul_reg*` family).
/// Returning at least one tile matters: a budget too small for a single row
/// would otherwise yield a zero-length chunk and an infinite loop, and one
/// oversized scratch allocation is a far better failure than that.
pub fn col_chunk_rows(budget_floats: u64, row_floats: u64, tile: u32, total: u32) -> u32 {
    let rows = budget_floats / row_floats.max(1);
    let snapped = (rows / u64::from(tile)) * u64::from(tile);
    snapped.clamp(u64::from(tile), u64::from(total).max(u64::from(tile))) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_whole_tiles_bounded_by_the_budget_and_the_work() {
        // 1000 rows fit; snapped down to 7 whole 128-tiles.
        assert_eq!(col_chunk_rows(10_000, 10, 128, 100_000), 896);
        // Never more than the work there is.
        assert_eq!(col_chunk_rows(10_000, 10, 128, 300), 300);
        // A budget too small for one tile still yields one tile, never zero -
        // a zero chunk is an infinite loop at the call site.
        assert_eq!(col_chunk_rows(8, 10, 128, 100_000), 128);
        // ... including when `total` is itself under one tile, which the
        // caller resolves with its own `min(chunk, total - pos)`.
        assert_eq!(col_chunk_rows(8, 10, 128, 64), 128);
    }

    #[test]
    fn the_budget_is_megabytes_of_f32() {
        assert_eq!(col_budget_floats(512), 512 * 1024 * 1024 / 4);
    }
}
