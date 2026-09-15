// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vision-token composition (`_encode_image`'s tail, everything after
//! `forward_features_unpool`): learned 2D position embed + cosine temporal
//! embed, spatial-average pooling, projection to `d_model=768`, LayerNorm.
//! Turns DaViT's `[576,1024]` unpooled sequence into Florence-2's actual
//! `[577,768]` vision-token input to the BART encoder (1 pooled
//! `spatial_avg_pool` token + 576 `temporal_avg_pool` tokens - identity at
//! `T=1`, per the reference's `image_feature_source=["spatial_avg_pool",
//! "temporal_avg_pool"]` concat order).
//!
//! Two pieces the checkpoint doesn't ship in a directly usable shape - the
//! CALLER building this module's `ParamStore` source MUST synthesize them
//! (see [`build_pos_embed_table`] and [`transpose_2d`]) before construction,
//! same convention as `ChannelAttn::new`'s pre-scaled qkv weight:
//!
//! - `image_pos_embed.{row,column}_embeddings.weight` are `[50,512]` lookup
//!   tables (`num_pos=50` covers up to a 50x50 grid); the reference builds a
//!   `[h,w,1024]` table at forward time by broadcasting column embeddings
//!   across rows and row embeddings across columns, then concatenating on
//!   the channel axis. Florence-2-base's grid is always `24x24` (fixed by
//!   [`DavitConfig::florence2_base`]'s stage schedule), so this table is
//!   entirely static per checkpoint - [`build_pos_embed_table`] materializes
//!   it once on the host into a synthetic `{prefix}.pos_embed_table` tensor.
//! - `image_projection` is a bare `[1024,768]` (`in,out`) `nn.Parameter`,
//!   not an `nn.Linear` (no bias) - `matmul_rows`' convention needs weight
//!   rows to BE output features (`[out,in]`), so [`transpose_2d`] must
//!   produce a `{prefix}.image_projection_t` `[768,1024]` synthetic tensor.
//!
//! `visual_temporal_embed.pos_idx_to_embed` needs no transform: at `T=1` the
//! reference indexes only row 0 of this `[100,1024]` table, which is already
//! a contiguous 1024-float run at the checkpoint tensor's own offset 0 - used
//! directly as `bias_add`'s per-channel operand.

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{layernorm_fwd, LayerNormIds};
use model::vit::row_index_buffer;

/// Broadcasts `column[0..grid_w]` (fastest-varying token axis) and
/// `row[0..grid_h]` into the reference's `[grid_h, grid_w, 2*half_dim]`
/// position table, row-major flattened to match DaViT's own token order
/// (token `t = h*grid_w + w`): channels `0..half_dim` come from
/// `column[w]`, channels `half_dim..2*half_dim` from `row[h]`.
///
/// `column`/`row` are the checkpoint's full `[num_pos, half_dim]` embedding
/// tables (row-major); only the first `grid_w`/`grid_h` rows are read.
pub fn build_pos_embed_table(column: &[f32], row: &[f32], grid_h: u32, grid_w: u32, half_dim: u32) -> Vec<f32> {
    let (gh, gw, hd) = (grid_h as usize, grid_w as usize, half_dim as usize);
    let mut table = vec![0.0f32; gh * gw * 2 * hd];
    for h in 0..gh {
        for w in 0..gw {
            let out = (h * gw + w) * 2 * hd;
            table[out..out + hd].copy_from_slice(&column[w * hd..(w + 1) * hd]);
            table[out + hd..out + 2 * hd].copy_from_slice(&row[h * hd..(h + 1) * hd]);
        }
    }
    table
}

/// Row-major `[rows,cols]` -> `[cols,rows]` transpose (`image_projection`'s
/// `[in,out]` -> the `matmul_rows`/`matmul` convention's `[out,in]`).
pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

pub struct ImageProjectKernelIds {
    pub layernorm: usize,
    pub matmul: usize,
    pub matmul_rows: usize,
    pub bias_add: usize,
    pub add2: usize,
    pub nlc_nchw: usize,
    pub row_scatter: usize,
}

impl ImageProjectKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> ImageProjectKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        ImageProjectKernelIds {
            layernorm: k("layernorm"),
            matmul: k("matmul"),
            matmul_rows: k("matmul_rows"),
            bias_add: k("bias_add"),
            add2: k("add2"),
            nlc_nchw: k("nlc_nchw"),
            row_scatter: k("row_scatter"),
        }
    }
}

pub struct ImageProject {
    prefix: String,
    grid_h: u32,
    grid_w: u32,
    dim: u32,
    out_dim: u32,
    eps: f32,
    ones_row: DeviceBuffer,
    idx_pooled: DeviceBuffer,
    idx_rest: DeviceBuffer,
    with_pos: DeviceBuffer,
    x_t: DeviceBuffer,
    pooled: DeviceBuffer,
    cat: DeviceBuffer,
    projected: DeviceBuffer,
    normed: DeviceBuffer,
}

impl ImageProject {
    /// `dim`: DaViT's final stage width (`1024` for Florence-2-base).
    /// `out_dim`: BART `d_model` (`768`). `grid_h`/`grid_w`: DaViT's final
    /// stage token grid (`24x24`) - both the position-table shape and the
    /// pooling token count `grid_h*grid_w`.
    pub fn new(gpu: &Gpu, prefix: &str, grid_h: u32, grid_w: u32, dim: u32, out_dim: u32, eps: f32) -> ImageProject {
        let n = grid_h * grid_w;
        let ones_row = gpu.storage(n as u64);
        gpu.write_f32(&ones_row, &vec![1.0 / n as f32; n as usize]);
        let idx_pooled = row_index_buffer(gpu, "florence2_pool_idx0", &[0]);
        let rest: Vec<u32> = (1..=n).collect();
        let idx_rest = row_index_buffer(gpu, "florence2_pool_idx_rest", &rest);
        ImageProject {
            prefix: prefix.to_string(),
            grid_h,
            grid_w,
            dim,
            out_dim,
            eps,
            ones_row,
            idx_pooled,
            idx_rest,
            with_pos: gpu.storage((n * dim) as u64),
            x_t: gpu.storage((dim * n) as u64),
            pooled: gpu.storage(dim as u64),
            cat: gpu.storage(((n + 1) * dim) as u64),
            projected: gpu.storage(((n + 1) * out_dim) as u64),
            normed: gpu.storage(((n + 1) * out_dim) as u64),
        }
    }

    /// `unpooled`: DaViT's `[grid_h*grid_w, dim]` `forward_features_unpool`
    /// output. Returns the final `[grid_h*grid_w+1, out_dim]` vision-token
    /// sequence (Florence-2-base: `[577,768]`) fed to the BART encoder.
    pub fn forward(&self, gpu: &Gpu, k: &ImageProjectKernelIds, ps: &paramstore::ParamStore, unpooled: &DeviceBuffer) -> &DeviceBuffer {
        let (n, dim, out_dim) = (self.grid_h * self.grid_w, self.dim, self.out_dim);

        let pos_table = ps.w(&format!("{}.pos_embed_table", self.prefix));
        let temporal_row0 = ps.w(&format!("{}.visual_temporal_embed.pos_idx_to_embed", self.prefix));
        let proj_t = ps.w(&format!("{}.image_projection_t", self.prefix));
        let norm_w = ps.w(&format!("{}.image_proj_norm.weight", self.prefix));
        let norm_b = ps.w(&format!("{}.image_proj_norm.bias", self.prefix));

        let ln = LayerNormIds::resolve_fwd(gpu, k.layernorm);

        // x = unpooled + pos_embed_table (per-token) + temporal row 0
        // (per-channel, broadcast over every token - T=1 collapses the
        // reference's per-frame temporal embed to a single constant offset).
        let s1 = vec![
            gpu.step(k.add2, &[unpooled, pos_table, &self.with_pos], &[n * dim], n * dim),
            gpu.step(k.bias_add, &[&self.with_pos, temporal_row0], &[n, dim], n * dim),
        ];
        gpu.submit(&[], &s1);

        // spatial_avg_pool: transpose to [dim,n], mean over n via a constant
        // 1/n row-vector matmul (matmul's A@W^T convention: ones[1,n] @
        // x_t[dim,n]^T = per-channel mean over tokens).
        let s2 = vec![
            gpu.step(k.nlc_nchw, &[&self.with_pos, &self.x_t], &[n * dim, dim, n], n * dim),
            gpu.step(k.matmul, &[&self.ones_row, &self.x_t, &self.pooled], &[1, n, dim], dim),
        ];
        gpu.submit(&[], &s2);

        // Concat [pooled(1); with_pos(n)] -> [n+1,dim], matching the
        // reference's image_feature_source=["spatial_avg_pool",
        // "temporal_avg_pool"] order (temporal_avg_pool is `with_pos`
        // itself, unchanged - T=1 makes that pool an identity).
        let s3 = vec![
            gpu.step(k.row_scatter, &[&self.idx_pooled, &self.pooled, &self.cat], &[1, dim, n + 1], dim),
            gpu.step(k.row_scatter, &[&self.idx_rest, &self.with_pos, &self.cat], &[n, dim, n + 1], n * dim),
        ];
        gpu.submit(&[], &s3);

        // x @ image_projection (no bias) -> LayerNorm.
        let s4 = vec![
            gpu.step(k.matmul_rows, &[&self.cat, proj_t, &self.projected], &[n + 1, dim, out_dim], (n + 1).div_ceil(8) * out_dim),
            layernorm_fwd(gpu, &ln, &self.projected, norm_w, norm_b, &self.normed, out_dim, n + 1, self.eps),
        ];
        gpu.submit(&[], &s4);

        &self.normed
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_embed_table_places_column_then_row_channels_in_token_major_order() {
        // grid 2x2, half_dim 2: column table rows are [w=0]=[10,11], [w=1]=[20,21];
        // row table rows are [h=0]=[100,101], [h=1]=[200,201].
        let column = vec![10.0, 11.0, 20.0, 21.0];
        let row = vec![100.0, 101.0, 200.0, 201.0];
        let table = build_pos_embed_table(&column, &row, 2, 2, 2);
        // token (h=0,w=0): col[0] ++ row[0]
        assert_eq!(&table[0..4], &[10.0, 11.0, 100.0, 101.0]);
        // token (h=0,w=1): col[1] ++ row[0]
        assert_eq!(&table[4..8], &[20.0, 21.0, 100.0, 101.0]);
        // token (h=1,w=0): col[0] ++ row[1]
        assert_eq!(&table[8..12], &[10.0, 11.0, 200.0, 201.0]);
        // token (h=1,w=1): col[1] ++ row[1]
        assert_eq!(&table[12..16], &[20.0, 21.0, 200.0, 201.0]);
    }

    #[test]
    fn transpose_2d_is_its_own_inverse_for_a_rectangular_matrix() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let t = transpose_2d(&data, 2, 3); // [3,2]
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let back = transpose_2d(&t, 3, 2);
        assert_eq!(back, data);
    }
}
