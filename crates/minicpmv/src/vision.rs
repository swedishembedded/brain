// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The NaViT SigLIP tower, its in-tower window merger and the merger, on one
//! device, composed from `model::vit`'s shared ViT blocks.
//!
//! A request is a list of VIEWS (an image's source view and its slices, or a
//! video frame's), each with its own `h x w` patch grid. All views are packed
//! into one `[N, C]` token sequence (`N = sum h*w`) and attention stays inside
//! each view (one span per view). Per view:
//!
//! 1. Patch embedding: a `[C, 3*p*p]` linear over the host-flattened patches
//!    (the stride-`p` convolution), plus the position embedding gathered at
//!    the NEAREST cell of the learned `pos_side x pos_side` table.
//! 2. Encoder layers `0..=insert_layer_id`.
//! 3. `16x` only - the window merger: rows reordered window-major (2x2
//!    windows, row-major, each window's patches row-major), attention inside
//!    each 4-row window, then each window's 4 rows concatenated and projected
//!    back to one row plus their mean. The rows that come out are the halved
//!    grid in row-major order, so the remaining layers see an ordinary view.
//! 4. The remaining encoder layers, the post-layernorm.
//! 5. The merger: each 2x2 block concatenated, LayerNorm, Linear, GELU(erf),
//!    Linear to the decoder width.
//!
//! Each step is a gather/GEMM/elementwise dispatch an existing kernel already
//! implements; there is no MiniCPM-specific kernel.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::vit::{self, VitBlockWeights, VitKernelIds, VitPermuteIds, VitScratch, VitShape};
use paramstore::{ParamStore, Role};

use crate::config::{Downsample, MiniCpmVConfig};

/// The kernels the tower dispatches, by name.
pub const VISION_PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("row_scatter", kernels::ROW_SCATTER),
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("matmul", kernels::MATMUL),
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("bias_add", kernels::BIAS_ADD),
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
    ("gelu", kernels::GELU),
    ("gelu_erf", kernels::GELU_ERF),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("segment_sum_rows", kernels::SEGMENT_SUM_ROWS),
];

/// Most patch rows one tower pass packs; [`Tower::encode`] splits a larger
/// request into passes of whole views.
pub const ENCODE_ROWS: u32 = 16384;

/// Score-slab budget for the encoder's span attention.
const ATTN_BUDGET_BYTES: u64 = 256 << 20;

/// One view's patch grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    pub h: u32,
    pub w: u32,
}

impl View {
    pub fn patches(self) -> u32 {
        self.h * self.w
    }
}

/// Intermediate activations a parity test compares, read back on request.
#[derive(Default)]
pub struct Taps {
    pub embeddings: Vec<f32>,
    /// `(layer, [rows, C])` for each requested encoder layer.
    pub layers: Vec<(u32, Vec<f32>)>,
    pub window_merger: Vec<f32>,
    pub post_norm: Vec<f32>,
}

/// The canonical parameter manifest: `(name, numel)`.
pub fn tensor_manifest(cfg: &MiniCpmVConfig) -> Vec<(String, usize)> {
    let v = &cfg.vision;
    let (c, m, p) = (v.hidden as usize, v.mlp as usize, v.patch as usize);
    let (wc, wm) = (v.window_hidden() as usize, v.window_mlp() as usize);
    let mc = (c * (cfg.merge * cfg.merge) as usize, cfg.text.d_model as usize);
    let mut out = vec![
        ("patch.weight".to_string(), c * 3 * p * p),
        ("patch.bias".to_string(), c),
        ("pos.weight".to_string(), (v.pos_side * v.pos_side) as usize * c),
    ];
    let block = |prefix: &str, out: &mut Vec<(String, usize)>| {
        for (leaf, n) in [
            ("norm1.weight", c),
            ("norm1.bias", c),
            ("attn.qkv.weight", 3 * c * c),
            ("attn.qkv.bias", 3 * c),
            ("attn.proj.weight", c * c),
            ("attn.proj.bias", c),
        ] {
            out.push((format!("{prefix}.{leaf}"), n));
        }
    };
    for l in 0..v.n_layers {
        block(&format!("blocks.{l}"), &mut out);
        for (leaf, n) in [("norm2.weight", c), ("norm2.bias", c), ("mlp.fc1.weight", m * c), ("mlp.fc1.bias", m), ("mlp.fc2.weight", c * m), ("mlp.fc2.bias", c)] {
            out.push((format!("blocks.{l}.{leaf}"), n));
        }
    }
    out.push(("post_norm.weight".to_string(), c));
    out.push(("post_norm.bias".to_string(), c));
    block("wm", &mut out);
    for (leaf, n) in [("pre_norm.weight", wc), ("pre_norm.bias", wc), ("fc1.weight", wm * wc), ("fc1.bias", wm), ("fc2.weight", c * wm), ("fc2.bias", c)] {
        out.push((format!("wm.{leaf}"), n));
    }
    for (leaf, n) in [
        ("pre_norm.weight", mc.0),
        ("pre_norm.bias", mc.0),
        ("fc1.weight", mc.0 * mc.0),
        ("fc1.bias", mc.0),
        ("fc2.weight", mc.1 * mc.0),
        ("fc2.bias", mc.1),
    ] {
        out.push((format!("merger.{leaf}"), n));
    }
    out
}

/// Position ids of a `h x w` view: each patch's fractional coordinate mapped
/// to the nearest cell of the `side x side` table - the reference's
/// `bucketize(i / h, arange(1/side, 1, 1/side), right=True)`, in its own f32
/// arithmetic (the boundaries are computed in f64 and rounded, the coordinate
/// is an f32 division).
pub fn nearest_position_ids(view: View, side: u32) -> Vec<u32> {
    let step = 1.0f64 / side as f64;
    let n = ((1.0 - step) / step).ceil() as usize;
    let bounds: Vec<f32> = (0..n).map(|k| (step + k as f64 * step) as f32).collect();
    let bucket = |i: u32, len: u32| bounds.partition_point(|&b| b <= i as f32 / len as f32) as u32;
    let mut out = Vec::with_capacity(view.patches() as usize);
    for y in 0..view.h {
        let by = bucket(y, view.h);
        for x in 0..view.w {
            out.push(by * side + bucket(x, view.w));
        }
    }
    out
}

/// The row permutation that makes each `k x k` block of a `h x w` grid
/// contiguous (blocks row-major, each block's cells row-major), offset by
/// `row0`. `h` and `w` must be multiples of `k`.
fn block_major(view: View, k: u32, row0: u32) -> Vec<u32> {
    assert!(view.h.is_multiple_of(k) && view.w.is_multiple_of(k), "a {}x{} grid does not tile into {k}x{k} blocks", view.h, view.w);
    vit::WindowPlan::new(view.h, view.w, k, k).perm().iter().map(|&r| r + row0).collect()
}

/// One span per view over a packed sequence.
fn spans(views: &[View]) -> Vec<(u32, u32)> {
    let mut row0 = 0;
    views
        .iter()
        .map(|v| {
            let s = (row0, v.patches());
            row0 += v.patches();
            s
        })
        .collect()
}

pub struct Tower {
    pub gpu: Gpu,
    pub cfg: MiniCpmVConfig,
    pub ps: ParamStore,
}

impl Tower {
    /// The tower with every parameter frozen, its weights under the names of
    /// [`tensor_manifest`].
    pub fn new_on(gpu: Gpu, cfg: MiniCpmVConfig, weights: &dyn checkpoint::TensorSource) -> Tower {
        let roles = tensor_manifest(&cfg).into_iter().map(|(n, k)| (n, k, Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, weights);
        let t = Tower { gpu, cfg, ps };
        for (name, _) in VISION_PIPELINES.iter().filter(|(n, _)| *n != "matmul_reg3" && *n != "layernorm_rows") {
            t.k(name);
        }
        t
    }

    fn k(&self, name: &str) -> usize {
        self.gpu
            .kernel_index(name)
            .unwrap_or_else(|| panic!("minicpmv: vision kernel `{name}` is not registered on this handle - chain minicpmv::vision::VISION_PIPELINES into its kernel list"))
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn shape(&self) -> VitShape {
        let v = &self.cfg.vision;
        VitShape { dim: v.hidden, heads: v.heads, mlp: v.mlp, eps: v.eps }
    }

    fn block(&self, l: u32) -> VitBlockWeights<'_> {
        let p = |leaf: &str| self.w(&format!("blocks.{l}.{leaf}"));
        VitBlockWeights {
            norm1_w: p("norm1.weight"),
            norm1_b: p("norm1.bias"),
            qkv_w: p("attn.qkv.weight"),
            qkv_b: p("attn.qkv.bias"),
            qk_norm: None,
            rope: None,
            proj_w: p("attn.proj.weight"),
            proj_b: p("attn.proj.bias"),
            ls1: None,
            norm2_w: p("norm2.weight"),
            norm2_b: p("norm2.bias"),
            fc1_w: p("mlp.fc1.weight"),
            fc1_b: p("mlp.fc1.bias"),
            fc2_w: p("mlp.fc2.weight"),
            fc2_b: p("mlp.fc2.bias"),
            ls2: None,
        }
    }

    fn ln(&self) -> model::block::LayerNormIds {
        model::block::LayerNormIds::resolve_fwd(&self.gpu, self.k("layernorm"))
    }

    fn gemm(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        self.gpu.step(self.k("matmul_rows"), &[x, w, out], &[m, k, n], m.div_ceil(8) * n)
    }

    fn bias(&self, x: &DeviceBuffer, b: &DeviceBuffer, rows: u32, cols: u32) -> Step {
        self.gpu.step(self.k("bias_add"), &[x, b], &[rows, cols], rows * cols)
    }

    /// Visual tokens for `views` in decoder width: `[sum tokens(view), d_model]`
    /// in view order, `tokens(view) = h*w / mode.divisor()`. `patches` is
    /// `[N, 3*p*p]`, each row one patch flattened channel-major (`c, ky, kx`),
    /// views concatenated, each view's patches row-major.
    ///
    /// Views are encoded in groups of at most [`ENCODE_ROWS`] patches (a view
    /// never splits; each is independent of the others), so the scratch a
    /// long video or many images need stays bounded.
    pub fn encode(&self, patches: &[f32], views: &[View], mode: Downsample) -> Vec<f32> {
        let pp = (3 * self.cfg.vision.patch * self.cfg.vision.patch) as usize;
        let mut out = Vec::new();
        let (mut start, mut at) = (0usize, 0usize);
        while start < views.len() {
            let mut end = start;
            let mut rows = 0u32;
            while end < views.len() && (end == start || rows + views[end].patches() <= ENCODE_ROWS) {
                rows += views[end].patches();
                end += 1;
            }
            let len = rows as usize * pp;
            out.extend(self.encode_taps(&patches[at..at + len], &views[start..end], mode, None));
            at += len;
            start = end;
        }
        assert_eq!(at, patches.len(), "minicpmv: {} patch values for the views given", patches.len());
        out
    }

    /// [`Self::encode`], also reading back the activations `taps` asks for
    /// (its `layers` field lists the encoder layers to tap, values empty).
    pub fn encode_taps(&self, patches: &[f32], views: &[View], mode: Downsample, mut taps: Option<&mut Taps>) -> Vec<f32> {
        let g = &self.gpu;
        let v = &self.cfg.vision;
        let (c, pp) = (v.hidden, 3 * v.patch * v.patch);
        let n: u32 = views.iter().map(|v| v.patches()).sum();
        assert!(n > 0, "minicpmv: no views to encode");
        assert_eq!(patches.len(), (n * pp) as usize, "minicpmv: {} patch values for {n} patches of {pp}", patches.len());
        let step = self.cfg.merge * if mode == Downsample::X16 { v.window } else { 1 };
        for view in views {
            assert!(view.h % step == 0 && view.w % step == 0, "minicpmv: a {}x{} view does not reduce by {step}x{step} ({} mode)", view.h, view.w, mode.name());
        }
        let sh = self.shape();
        let vk = VitKernelIds::by_name(g, "gelu");
        let perm_ids = VitPermuteIds { embed: self.k("embed"), row_scatter: self.k("row_scatter") };
        let mut spans_now = spans(views);
        let max_span = views.iter().map(|v| v.patches()).max().unwrap_or(1);
        let chunk = vit::attn_chunk_for(&sh, max_span, ATTN_BUDGET_BYTES);
        let scr = VitScratch::new(g, &sh, n, chunk, max_span);
        let wanted: Vec<u32> = taps.as_ref().map(|t| t.layers.iter().map(|(l, _)| *l).collect()).unwrap_or_default();

        // 1. Patch embedding + nearest position embedding.
        let px = g.storage_init("minicpmv.patches", patches);
        let pos: Vec<u32> = views.iter().flat_map(|&view| nearest_position_ids(view, v.pos_side)).collect();
        let pos_idx = vit::row_index_buffer(g, "minicpmv.pos_ids", &pos);
        let mut x = g.storage((n * c) as u64);
        g.submit(
            &[],
            &[
                self.gemm(&px, self.w("patch.weight"), &scr.res, n, pp, c),
                self.bias(&scr.res, self.w("patch.bias"), n, c),
                vit::gather_rows(g, &perm_ids, &pos_idx, self.w("pos.weight"), &scr.ln, n, c),
                g.step(self.k("add2"), &[&scr.res, &scr.ln, &x], &[n * c], n * c),
            ],
        );
        if let Some(t) = taps.as_deref_mut() {
            t.embeddings = g.read(&x, (n * c) as usize);
        }

        // 2-4. Encoder, with the window merger after `insert_layer_id`.
        let mut rows = n;
        let mut grid: Vec<View> = views.to_vec();
        for l in 0..v.n_layers {
            let mut s = Vec::new();
            vit::vit_block_fwd(g, &vk, &sh, &self.block(l), &x, rows, &spans_now, chunk, &scr, &mut s);
            g.submit(&[], &s);
            if wanted.contains(&l) {
                let data = g.read(&x, (rows * c) as usize);
                if let Some(t) = taps.as_deref_mut() {
                    t.layers.iter_mut().find(|(tl, _)| *tl == l).expect("requested").1 = data;
                }
            }
            if l == v.insert_layer_id && mode == Downsample::X16 {
                x = self.window_merger(&x, &grid, &perm_ids);
                rows /= v.window * v.window;
                grid = grid.iter().map(|g| View { h: g.h / v.window, w: g.w / v.window }).collect();
                spans_now = spans(&grid);
                if let Some(t) = taps.as_deref_mut() {
                    t.window_merger = g.read(&x, (rows * c) as usize);
                }
            }
        }
        let ln = self.ln();
        let post = g.storage((rows * c) as u64);
        g.submit(&[], &[model::block::layernorm_fwd(g, &ln, &x, self.w("post_norm.weight"), self.w("post_norm.bias"), &post, c, rows, v.eps)]);
        if let Some(t) = taps {
            t.post_norm = g.read(&post, (rows * c) as usize);
        }

        // 5. Merger.
        self.merger(&post, &grid, &perm_ids)
    }

    /// The window merger over `x` (`[rows, C]`, views packed per `grid`):
    /// returns `[rows / window^2, C]`, each view's halved grid row-major.
    fn window_merger(&self, x: &DeviceBuffer, grid: &[View], perm_ids: &VitPermuteIds) -> DeviceBuffer {
        let g = &self.gpu;
        let v = &self.cfg.vision;
        let (c, wk) = (v.hidden, v.window);
        let ww = wk * wk;
        let rows: u32 = grid.iter().map(|v| v.patches()).sum();
        let windows = rows / ww;
        let (heads, hd) = (v.heads, v.head_dim());
        let ln = self.ln();
        let mut perm = Vec::with_capacity(rows as usize);
        let mut row0 = 0;
        for &view in grid {
            perm.extend(block_major(view, wk, row0));
            row0 += view.patches();
        }
        let perm = vit::row_index_buffer(g, "minicpmv.window_perm", &perm);
        let xp = g.storage((rows * c) as u64);
        let ln1 = g.storage((rows * c) as u64);
        let qkv = g.storage((rows * 3 * c) as u64);
        let scores = g.storage((windows * heads * ww * ww) as u64);
        let probs = g.storage((windows * heads * ww * ww) as u64);
        let ctx = g.storage((rows * c) as u64);
        let proj = g.storage((rows * c) as u64);
        let hw = g.storage((rows * c) as u64);
        let mean = g.storage((windows * c) as u64);
        let ones = vit::row_index_buffer(g, "minicpmv.window_mask", &vec![1u32; rows as usize]);
        let (wc, wm) = (v.window_hidden(), v.window_mlp());
        let flat_ln = g.storage((windows * wc) as u64);
        let h1 = g.storage((windows * wm) as u64);
        let h2 = g.storage((windows * wm) as u64);
        let out = g.storage((windows * c) as u64);
        let stride = 3 * c;
        let s = vec![
            vit::gather_rows(g, perm_ids, &perm, x, &xp, rows, c),
            model::block::layernorm_fwd(g, &ln, &xp, self.w("wm.norm1.weight"), self.w("wm.norm1.bias"), &ln1, c, rows, v.eps),
            self.gemm(&ln1, self.w("wm.attn.qkv.weight"), &qkv, rows, c, stride),
            self.bias(&qkv, self.w("wm.attn.qkv.bias"), rows, stride),
            g.step(self.k("attn_scores_bidir"), &[&qkv, &scores], &[windows, heads, ww, hd, stride, 0, c], windows * heads * ww * ww),
            g.step(self.k("attn_softmax_bidir"), &[&scores, &probs], &[windows, heads, ww], windows * heads * ww),
            g.step(self.k("attn_apply_bidir"), &[&probs, &qkv, &ctx], &[windows, heads, ww, hd, stride, 2 * c, c], rows * c),
            self.gemm(&ctx, self.w("wm.attn.proj.weight"), &proj, rows, c, c),
            self.bias(&proj, self.w("wm.attn.proj.bias"), rows, c),
            g.step(self.k("add2"), &[&xp, &proj, &hw], &[rows * c], rows * c),
            // The window's 4 rows are contiguous, so `[rows, C]` IS `[windows, 4C]`.
            g.step(self.k("segment_sum_rows"), &[&ones, &hw, &mean], &[windows, ww, c], windows * c),
            model::block::layernorm_fwd(g, &ln, &hw, self.w("wm.pre_norm.weight"), self.w("wm.pre_norm.bias"), &flat_ln, wc, windows, v.eps),
            self.gemm(&flat_ln, self.w("wm.fc1.weight"), &h1, windows, wc, wm),
            self.bias(&h1, self.w("wm.fc1.bias"), windows, wm),
            g.step(self.k("gelu"), &[&h1, &h2], &[windows * wm], windows * wm),
            self.gemm(&h2, self.w("wm.fc2.weight"), &out, windows, wm, c),
            self.bias(&out, self.w("wm.fc2.bias"), windows, c),
            // + the window mean (the segment sum scaled by 1/4).
            g.step(self.k("axpy"), &[&out, &mean], &[windows * c, f(1.0 / ww as f32)], windows * c),
        ];
        g.submit(&[], &s);
        out
    }

    /// The merger over the post-norm tokens (`[rows, C]`, views packed per
    /// `grid`): `[rows / merge^2, d_model]`.
    fn merger(&self, x: &DeviceBuffer, grid: &[View], perm_ids: &VitPermuteIds) -> Vec<f32> {
        let g = &self.gpu;
        let v = &self.cfg.vision;
        let (c, k) = (v.hidden, self.cfg.merge);
        let rows: u32 = grid.iter().map(|v| v.patches()).sum();
        let out_rows = rows / (k * k);
        let (mc, d) = (c * k * k, self.cfg.text.d_model);
        let mut perm = Vec::with_capacity(rows as usize);
        let mut row0 = 0;
        for &view in grid {
            perm.extend(block_major(view, k, row0));
            row0 += view.patches();
        }
        let perm = vit::row_index_buffer(g, "minicpmv.merge_perm", &perm);
        let ln = self.ln();
        let xp = g.storage((rows * c) as u64);
        let xn = g.storage((out_rows * mc) as u64);
        let h1 = g.storage((out_rows * mc) as u64);
        let h2 = g.storage((out_rows * mc) as u64);
        let out = g.storage((out_rows * d) as u64);
        let s = vec![
            vit::gather_rows(g, perm_ids, &perm, x, &xp, rows, c),
            model::block::layernorm_fwd(g, &ln, &xp, self.w("merger.pre_norm.weight"), self.w("merger.pre_norm.bias"), &xn, mc, out_rows, 1e-6),
            self.gemm(&xn, self.w("merger.fc1.weight"), &h1, out_rows, mc, mc),
            self.bias(&h1, self.w("merger.fc1.bias"), out_rows, mc),
            g.step(self.k("gelu_erf"), &[&h1, &h2], &[out_rows * mc], out_rows * mc),
            self.gemm(&h2, self.w("merger.fc2.weight"), &out, out_rows, mc, d),
            self.bias(&out, self.w("merger.fc2.bias"), out_rows, d),
        ];
        g.submit(&[], &s);
        g.read(&out, (out_rows * d) as usize)
    }
}

/// The patch rows for one view from its `[3, h*p, w*p]` normalized pixels
/// (channel-major planes): row `y*w + x` holds the `(c, ky, kx)` block at
/// patch `(y, x)`, the layout the patch-embedding weight expects.
pub fn patchify(pixels: &[f32], view: View, patch: u32) -> Vec<f32> {
    let (h, w, p) = (view.h as usize, view.w as usize, patch as usize);
    let (ih, iw) = (h * p, w * p);
    assert_eq!(pixels.len(), 3 * ih * iw, "patchify: {} values for a 3x{ih}x{iw} view", pixels.len());
    let mut out = Vec::with_capacity(pixels.len());
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                for ky in 0..p {
                    let row = &pixels[c * ih * iw + (y * p + ky) * iw + x * p..][..p];
                    out.extend_from_slice(row);
                }
            }
        }
    }
    out
}

/// Map the checkpoint's vision and merger tensors onto [`tensor_manifest`]'s
/// names: the attention projections fused into `qkv` (q, k, v row blocks),
/// the patch convolution flattened to `[C, 3*p*p]`. Two-way: every manifest
/// tensor must be produced at its size, and every source tensor under
/// [`crate::import::VISION_PREFIXES`] (`source_names`) must be consumed.
pub fn import(source_names: &[String], read: &mut dyn FnMut(&str) -> Option<Vec<f32>>, cfg: &MiniCpmVConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut used = std::collections::HashSet::new();
    let mut get = |name: &str| {
        used.insert(name.to_string());
        read(name).ok_or_else(|| format!("minicpmv: the checkpoint has no {name}"))
    };
    let mut out = HashMap::new();
    let vt = "model.vision_tower";
    out.insert("patch.weight".into(), get(&format!("{vt}.embeddings.patch_embedding.weight"))?);
    out.insert("patch.bias".into(), get(&format!("{vt}.embeddings.patch_embedding.bias"))?);
    out.insert("pos.weight".into(), get(&format!("{vt}.embeddings.position_embedding.weight"))?);
    let attn = |get: &mut dyn FnMut(&str) -> Result<Vec<f32>, String>, src: &str, dst: &str, out: &mut HashMap<String, Vec<f32>>| -> Result<(), String> {
        for part in ["weight", "bias"] {
            let mut fused = get(&format!("{src}.self_attn.q_proj.{part}"))?;
            fused.extend(get(&format!("{src}.self_attn.k_proj.{part}"))?);
            fused.extend(get(&format!("{src}.self_attn.v_proj.{part}"))?);
            out.insert(format!("{dst}.attn.qkv.{part}"), fused);
            out.insert(format!("{dst}.attn.proj.{part}"), get(&format!("{src}.self_attn.out_proj.{part}"))?);
            out.insert(format!("{dst}.norm1.{part}"), get(&format!("{src}.layer_norm1.{part}"))?);
        }
        Ok(())
    };
    for l in 0..cfg.vision.n_layers {
        let src = format!("{vt}.encoder.layers.{l}");
        attn(&mut get, &src, &format!("blocks.{l}"), &mut out)?;
        for part in ["weight", "bias"] {
            out.insert(format!("blocks.{l}.norm2.{part}"), get(&format!("{src}.layer_norm2.{part}"))?);
            out.insert(format!("blocks.{l}.mlp.fc1.{part}"), get(&format!("{src}.mlp.fc1.{part}"))?);
            out.insert(format!("blocks.{l}.mlp.fc2.{part}"), get(&format!("{src}.mlp.fc2.{part}"))?);
        }
    }
    attn(&mut get, &format!("{vt}.vit_merger"), "wm", &mut out)?;
    for part in ["weight", "bias"] {
        out.insert(format!("post_norm.{part}"), get(&format!("{vt}.post_layernorm.{part}"))?);
        out.insert(format!("wm.pre_norm.{part}"), get(&format!("{vt}.vit_merger.pre_norm.{part}"))?);
        out.insert(format!("wm.fc1.{part}"), get(&format!("{vt}.vit_merger.linear_1.{part}"))?);
        out.insert(format!("wm.fc2.{part}"), get(&format!("{vt}.vit_merger.linear_2.{part}"))?);
        out.insert(format!("merger.pre_norm.{part}"), get(&format!("model.merger.mlp.0.pre_norm.{part}"))?);
        out.insert(format!("merger.fc1.{part}"), get(&format!("model.merger.mlp.0.linear_1.{part}"))?);
        out.insert(format!("merger.fc2.{part}"), get(&format!("model.merger.mlp.0.linear_2.{part}"))?);
    }
    let want = tensor_manifest(cfg);
    let mut problems: Vec<String> = want
        .iter()
        .filter_map(|(name, n)| match out.get(name) {
            None => Some(format!("missing {name}")),
            Some(v) if v.len() != *n => Some(format!("{name} has {} elements, expected {n}", v.len())),
            Some(_) => None,
        })
        .collect();
    problems.extend(
        source_names
            .iter()
            .filter(|n| crate::import::VISION_PREFIXES.iter().any(|p| n.starts_with(p)) && !used.contains(n.as_str()))
            .map(|n| format!("{n} is in the checkpoint but maps to nothing")),
    );
    if !problems.is_empty() {
        problems.sort();
        return Err(format!("minicpmv: vision import ({} problems): {}", problems.len(), problems.join("; ")));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_position_ids_cover_the_table_corners_and_keep_row_major_order() {
        let side = 70;
        let ids = nearest_position_ids(View { h: 70, w: 70 }, side);
        assert_eq!(ids, (0..70 * 70).collect::<Vec<u32>>(), "a native-size view reads the table in order");
        let ids = nearest_position_ids(View { h: 2, w: 3 }, side);
        // i/h in {0, 1/2}, j/w in {0, 1/3, 2/3}: buckets 0 and 35; 0, 23, 46.
        assert_eq!(ids, vec![0, 23, 46, 35 * 70, 35 * 70 + 23, 35 * 70 + 46]);
    }

    #[test]
    fn block_major_orders_windows_then_cells_row_major() {
        assert_eq!(block_major(View { h: 2, w: 4 }, 2, 10), vec![10, 11, 14, 15, 12, 13, 16, 17]);
    }

    #[test]
    fn patchify_lays_each_patch_out_channel_then_row_major() {
        // A 1x2-patch view of patch 2: pixel value = c*100 + y*10 + x.
        let (h, w) = (2usize, 4usize);
        let px: Vec<f32> = (0..3).flat_map(|c| (0..h).flat_map(move |y| (0..w).map(move |x| (c * 100 + y * 10 + x) as f32))).collect();
        let rows = patchify(&px, View { h: 1, w: 2 }, 2);
        assert_eq!(&rows[..12], &[0., 1., 10., 11., 100., 101., 110., 111., 200., 201., 210., 211.]);
        assert_eq!(&rows[12..16], &[2., 3., 12., 13.]);
    }
}
