// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DaViT's `SpatialBlock` windowed self-attention sublayer: standard
//! multi-head self-attention (`1/sqrt(head_dim)` scale - the ordinary
//! variant, unlike `channel_attn`'s `1/sqrt(N)`), restricted to
//! non-overlapping `window_size x window_size` windows.
//!
//! `WindowPlan::new` (not `::padded`) - Florence-2-base's four stage grids
//! (192/96/48/24) are all exact multiples of `window_size=12`, so no
//! padding sentinel row is ever needed for this checkpoint; `WindowPlan::
//! new`'s own doc names this exact "DaViT's local window stage" case.
//! Windowing is row permutation only (`window_partition`/`window_reverse`,
//! `model::vit::gather_rows` under the hood) - attention itself reuses
//! `model::vit::chunked_attn_fwd` with one span per window, `chunk == span
//! len` (whole window in one dispatch - 144 tokens at window_size=12, small
//! enough not to need further chunking).

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::vit::{self, VitKernelIds, VitShape, WindowIndex, WindowPlan};

pub struct WindowAttnKernelIds {
    pub vit: VitKernelIds,
    pub permute_embed: usize,
    pub permute_row_scatter: usize,
}

impl WindowAttnKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> WindowAttnKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        WindowAttnKernelIds {
            vit: VitKernelIds {
                layernorm: k("layernorm"),
                matmul: k("matmul"),
                matmul_rows: k("matmul_rows"),
                bias_add: k("bias_add"),
                mlp_act: k("gelu_erf"),
                scale_chan: vit::UNREGISTERED,
                add2: k("add2"),
                attn_scores_cross: k("attn_scores_cross"),
                attn_softmax_cross: k("attn_softmax_cross"),
                attn_apply_cross: k("attn_apply_cross"),
                kv_k_headt: vit::UNREGISTERED,
                attn_scores_cross_kt: vit::UNREGISTERED,
                ln_head: vit::UNREGISTERED,
                rope2d: vit::UNREGISTERED,
            },
            permute_embed: k("embed"),
            permute_row_scatter: k("row_scatter"),
        }
    }
}

pub struct WindowAttn {
    prefix: String,
    dim: u32,
    eps: f32,
    plan: WindowPlan,
    index: WindowIndex,
    normed: DeviceBuffer,
    qkv: DeviceBuffer,
    qkv_win: DeviceBuffer,
    ctxb: DeviceBuffer,
    ctx_grid: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    kt: DeviceBuffer,
    proj_out: DeviceBuffer,
    out: DeviceBuffer,
}

impl WindowAttn {
    pub fn new(gpu: &Gpu, prefix: &str, dim: u32, heads: u32, window_size: u32, grid_h: u32, grid_w: u32, eps: f32) -> WindowAttn {
        assert!(
            grid_h.is_multiple_of(window_size) && grid_w.is_multiple_of(window_size),
            "WindowAttn: grid {grid_h}x{grid_w} must be an exact multiple of window_size {window_size} (WindowPlan::padded not implemented here)"
        );
        let plan = WindowPlan::new(grid_h, grid_w, window_size, window_size);
        let index = WindowIndex::new(gpu, &plan);
        let rows = grid_h * grid_w;
        let max_span = (window_size * window_size) as u64;
        let head_dim = dim / heads;
        let _ = head_dim;

        WindowAttn {
            prefix: prefix.to_string(),
            dim,
            eps,
            plan,
            index,
            normed: gpu.storage((rows * dim) as u64),
            qkv: gpu.storage(3 * (rows * dim) as u64),
            qkv_win: gpu.storage(3 * (rows * dim) as u64),
            ctxb: gpu.storage((rows * dim) as u64),
            ctx_grid: gpu.storage((rows * dim) as u64),
            scores: gpu.storage(heads as u64 * max_span * max_span),
            probs: gpu.storage(heads as u64 * max_span * max_span),
            kt: gpu.storage(dim as u64 * max_span),
            proj_out: gpu.storage((rows * dim) as u64),
            out: gpu.storage((rows * dim) as u64),
        }
    }

    pub fn forward(&self, gpu: &Gpu, k: &WindowAttnKernelIds, ps: &paramstore::ParamStore, x_in: &DeviceBuffer, rows: u32, heads: u32) -> &DeviceBuffer {
        let c = self.dim;
        let sh = VitShape { dim: c, heads, mlp: 0, eps: self.eps };
        // PreNorm's `norm` is a SIBLING of the wrapped `fn` module in the
        // reference (`window_attn.norm.*` vs `window_attn.fn.{qkv,proj}.*`),
        // confirmed against the real checkpoint's own tensor names - NOT
        // nested under `.fn` the way a naive PreNorm(norm, WindowAttention)
        // reading might suggest.
        let ln = model::block::LayerNormIds::resolve_fwd(gpu, k.vit.layernorm);
        let norm_w = ps.w(&format!("{}.norm.weight", self.prefix));
        let norm_b = ps.w(&format!("{}.norm.bias", self.prefix));
        let qkv_w = ps.w(&format!("{}.fn.qkv.weight", self.prefix));
        let qkv_b = ps.w(&format!("{}.fn.qkv.bias", self.prefix));
        let proj_w = ps.w(&format!("{}.fn.proj.weight", self.prefix));
        let proj_b = ps.w(&format!("{}.fn.proj.bias", self.prefix));

        let s: Vec<Step> = vec![
            model::block::layernorm_fwd(gpu, &ln, x_in, norm_w, norm_b, &self.normed, c, rows, self.eps),
            gpu.step(k.vit.matmul_rows, &[&self.normed, qkv_w, &self.qkv], &[rows, c, 3 * c], rows.div_ceil(8) * 3 * c),
            gpu.step(k.vit.bias_add, &[&self.qkv, qkv_b], &[rows, 3 * c], rows * 3 * c),
        ];
        gpu.submit(&[], &s);

        // Permute rows into window-major order (q/k/v travel together since
        // they're one fused [rows,3C] buffer - the permutation is per-row,
        // channel-agnostic).
        let permute_ids = model::vit::VitPermuteIds { embed: k.permute_embed, row_scatter: k.permute_row_scatter };
        let s2 = vec![vit::window_partition(gpu, &permute_ids, &self.index, &self.qkv, &self.qkv_win, 3 * c)];
        gpu.submit(&[], &s2);

        // One span per window, one chunk per span (144 tokens at window_size=12
        // - small enough for a single dispatch, no further chunking needed).
        let n_windows = self.plan.n_windows();
        let win_len = self.plan.win_h * self.plan.win_w;
        let spans: Vec<(u32, u32)> = (0..n_windows).map(|w| (w * win_len, win_len)).collect();

        let mut s3: Vec<Step> = Vec::new();
        vit::chunked_attn_fwd(gpu, &k.vit, &sh, &self.qkv_win, &self.ctxb, &self.scores, &self.probs, &self.kt, &spans, win_len, &mut s3);
        gpu.submit(&[], &s3);

        let s4 = vec![vit::window_reverse(gpu, &permute_ids, &self.index, &self.ctxb, &self.ctx_grid, c)];
        gpu.submit(&[], &s4);

        let s5: Vec<Step> = vec![
            gpu.step(k.vit.matmul_rows, &[&self.ctx_grid, proj_w, &self.proj_out], &[rows, c, c], rows.div_ceil(8) * c),
            gpu.step(k.vit.bias_add, &[&self.proj_out, proj_b], &[rows, c], rows * c),
            gpu.step(k.vit.add2, &[x_in, &self.proj_out, &self.out], &[rows * c], rows * c),
        ];
        gpu.submit(&[], &s5);
        &self.out
    }
}
