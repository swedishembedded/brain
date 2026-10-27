// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training the vision side: the backward of [`Tower`]'s forward, and LoRA
//! on the tower's linears.
//!
//! **Which parameters train** is decided by the tower's parameter store:
//! a tensor with a gradient buffer is differentiated, one without is not.
//! The merger trains alone when only `merger.*` has gradients (the tower's
//! backward is then skipped entirely); the tower trains when every tower
//! tensor has one. Gradients accumulate (`+=`); [`Tower::zero_grads`] clears
//! them.
//!
//! **Memory**: the forward keeps only each encoder layer's INPUT; the
//! backward re-runs each block's cached forward from it before differentiating
//! it (gradient checkpointing), so a full-resolution request does not hold
//! 27 layers of attention probabilities. The window merger and the merger are
//! small and keep their activations.
//!
//! **LoRA** ([`TowerLora`]) folds `W = W0 + (alpha/rank) B A` into the
//! tower's targeted weights before a forward and turns the dense weight
//! gradient the backward produced into the factors' gradients after it:
//! `dB = s dW A^T`, `dA = s B^T dW` - exact, since `W` is linear in `B A`.
//! Existing GEMM kernels do all three products.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu};
use model::saved::SavedPolicy;
use model::vit::{self, VitBlockCache, VitBlockGrads, VitBwdIds, VitBwdScratch, VitKernelIds};
use paramstore::{ParamStore, Role};

use crate::config::{Downsample, MiniCpmVConfig};
use crate::vision::{MergerActs, Saved, Tower, View, VISION_PIPELINES};

/// The tower's forward kernels plus every backward and optimiser kernel a
/// training build dispatches.
pub fn train_pipelines() -> &'static [(&'static str, &'static str)] {
    static ALL: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| {
        let mut v: Vec<(&'static str, &'static str)> = VISION_PIPELINES.to_vec();
        v.extend_from_slice(&[
            ("layernorm_dx", kernels::LAYERNORM_DX),
            ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
            ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
            ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
            ("ln_stats", kernels::LN_STATS),
            ("ln_stats_rows", kernels::LN_STATS_ROWS),
            ("matmul_dx", kernels::MATMUL_DX),
            ("matmul_dw", kernels::MATMUL_DW),
            ("bias_grad", kernels::BIAS_GRAD),
            ("gelu_bwd", kernels::GELU_BWD),
            ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
            ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
            ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
            ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
            ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
            ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
            ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
            ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
            ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
            ("region_copy", kernels::REGION_COPY),
            ("emb_bwd", kernels::EMB_BWD),
            ("segment_bcast_rows", kernels::SEGMENT_BCAST_ROWS),
            ("adamw", kernels::ADAMW),
            ("gradnorm_sq", kernels::GRADNORM_SQ),
            ("grad_scale", kernels::GRAD_SCALE),
            ("clip_coef", kernels::CLIP_COEF),
            ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ]);
        v
    })
}

/// A training forward's state: its output and what its backward reads.
pub struct TrainPass {
    saved: Saved,
    merger: MergerActs,
    mode: Downsample,
}

impl TrainPass {
    /// How many visual rows (`[n, d_model]`) the pass produced.
    pub fn out_rows(&self) -> u32 {
        self.merger.out_rows
    }
}

impl Tower {
    /// A tower whose parameters named by `trainable` carry gradient buffers
    /// (`Role::Trainable`); every other parameter is frozen.
    pub fn new_train_on(gpu: Gpu, cfg: MiniCpmVConfig, weights: &dyn checkpoint::TensorSource, trainable: &dyn Fn(&str) -> bool) -> Tower {
        let roles = crate::vision::tensor_manifest(&cfg).into_iter().map(|(n, k)| {
            let role = if trainable(&n) { Role::Trainable } else { Role::Frozen };
            (n, k, role)
        });
        let ps = ParamStore::new_with_roles_src(&gpu, roles.collect(), weights);
        Tower { gpu, cfg, ps }
    }

    fn gr(&self, name: &str) -> Option<&DeviceBuffer> {
        self.ps.grad.get(name)
    }

    /// Whether the encoder below the merger is differentiated.
    fn tower_trains(&self) -> bool {
        self.gr("blocks.0.norm1.weight").is_some()
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// The encoder's backward always rebuilds each block from the block's input (see the module doc): recomputing
    /// is accepted as what it already does, keeping every block and a narrow saved format are refused by name.
    pub fn set_saved_policy(&self, policy: SavedPolicy) -> Result<(), String> {
        model::saved::recompute_only("minicpmv vision tower", policy)
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.ps.w(name), data);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    /// Names with a gradient buffer.
    pub fn grad_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.ps.grad.keys().cloned().collect();
        v.sort();
        v
    }

    /// [`Self::encode`] for training: one packed pass (no splitting into
    /// groups - a training example is one request), keeping what
    /// [`Self::backward`] needs.
    pub fn encode_train(&self, patches: &[f32], views: &[View], mode: Downsample) -> (Vec<f32>, TrainPass) {
        let (merger, saved) = self.forward(patches, views, mode, None, true);
        let rows = self.gpu.read(&merger.out, (merger.out_rows * self.cfg.text.d_model) as usize);
        (rows, TrainPass { saved: saved.expect("a saving forward"), merger, mode })
    }

    /// Accumulate the gradient of every differentiated parameter given the
    /// loss's gradient with respect to the pass's visual rows (`[rows,
    /// d_model]`).
    pub fn backward(&self, pass: &TrainPass, d_rows: &[f32]) {
        let g = &self.gpu;
        let d = self.cfg.text.d_model;
        assert_eq!(d_rows.len(), (pass.merger.out_rows * d) as usize, "minicpmv: one gradient row per visual row");
        let d_out = g.storage_init("minicpmv.bwd.d_rows", d_rows);
        let d_post = self.merger_bwd(&pass.merger, &d_out);
        if !self.tower_trains() {
            return;
        }
        let v = &self.cfg.vision;
        let (c, s) = (v.hidden, &pass.saved);
        let ln = self.ln_ids();
        // post-layernorm
        let mut dx = g.storage((s.rows_last * c) as u64);
        self.ln_bwd(&ln, &s.x_last, "post_norm", &d_post, &dx, c, s.rows_last, v.eps);
        // encoder layers, last to first, the window merger between
        let sh = self.vit_shape();
        let (vk, kb) = (VitKernelIds::by_name(g, "gelu"), VitBwdIds::by_name(g, "gelu_bwd"));
        let n = s.n;
        let mut cache = VitBlockCache::new(g, &sh, n, s.max_span);
        let sb = VitBwdScratch::new(g, &sh, n, s.max_span);
        let scr_tmp = g.storage((n * c) as u64);
        let scores = g.storage(sh.heads as u64 * s.max_span as u64 * s.max_span as u64);
        let x_out = g.storage((n * c) as u64);
        for l in (0..v.n_layers).rev() {
            if pass.mode == Downsample::X16 && l == v.insert_layer_id {
                let acts = s.wm.as_ref().expect("a 16x pass keeps the window merger");
                dx = self.window_merger_bwd(acts, &dx);
            }
            let input = &s.inputs[l as usize];
            cache.x_in = input.x.clone();
            let w = self.block_ref(l);
            let gr = self.block_grads(l);
            let d_in = g.storage((input.rows * c) as u64);
            let mut steps = Vec::new();
            vit::vit_block_fwd_cached(g, &vk, &kb, &sh, &w, &cache, &x_out, input.rows, &input.spans, &scr_tmp, &scores, &mut steps);
            g.submit(&[&cache.qkv], &steps);
            let mut steps = Vec::new();
            vit::vit_block_bwd(g, &vk, &kb, &sh, &w, &gr, &cache, &dx, &d_in, input.rows, &input.spans, &sb, &mut steps);
            g.submit(&[&sb.d_qkv, &sb.d_qkv_pre], &steps);
            dx = d_in;
        }
        // embeddings: patch linear and the position table
        let pp = 3 * v.patch * v.patch;
        let mut steps = Vec::new();
        if let Some(gw) = self.gr("patch.weight") {
            steps.push(g.step(self.k("matmul_dw"), &[&dx, &s.px, gw], &[n, pp, c], c * pp));
        }
        if let Some(gb) = self.gr("patch.bias") {
            steps.push(g.step(self.k("bias_grad"), &[&dx, gb], &[n, c], c));
        }
        if let Some(gp) = self.gr("pos.weight") {
            steps.push(g.step(self.k("emb_bwd"), &[&s.pos_idx, &dx, gp], &[n, c, v.pos_side * v.pos_side], v.pos_side * v.pos_side * c));
        }
        g.submit(&[], &steps);
    }

    fn ln_ids(&self) -> model::block::LayerNormIds {
        model::block::LayerNormIds::resolve(&self.gpu, self.k("layernorm"), self.k("ln_stats"), self.k("layernorm_dx"))
    }

    fn vit_shape(&self) -> vit::VitShape {
        let v = &self.cfg.vision;
        vit::VitShape { dim: v.hidden, heads: v.heads, mlp: v.mlp, eps: v.eps }
    }

    fn block_ref(&self, l: u32) -> vit::VitBlockWeights<'_> {
        let p = |leaf: &str| self.ps.w(&format!("blocks.{l}.{leaf}"));
        vit::VitBlockWeights {
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

    fn block_grads(&self, l: u32) -> VitBlockGrads<'_> {
        let p = |leaf: &str| {
            let name = format!("blocks.{l}.{leaf}");
            self.gr(&name).unwrap_or_else(|| panic!("minicpmv: the tower trains but {name} has no gradient - a tower trains whole or not at all"))
        };
        VitBlockGrads {
            norm1_w: p("norm1.weight"),
            norm1_b: p("norm1.bias"),
            qkv_w: p("attn.qkv.weight"),
            qkv_b: p("attn.qkv.bias"),
            q_norm_w: None,
            q_norm_b: None,
            k_norm_w: None,
            k_norm_b: None,
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

    /// LayerNorm backward over `rows` rows of width `width`: the parameter
    /// gradients of `<prefix>.weight`/`.bias` (when they have buffers) and
    /// `dx` (written).
    fn ln_bwd(&self, ln: &model::block::LayerNormIds, x: &DeviceBuffer, prefix: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, width: u32, rows: u32, eps: f32) {
        let g = &self.gpu;
        let gamma = self.ps.w(&format!("{prefix}.weight"));
        let mut steps = Vec::new();
        if let (Some(gw), Some(gb)) = (self.gr(&format!("{prefix}.weight")), self.gr(&format!("{prefix}.bias"))) {
            let (mean, inv) = (g.storage(rows as u64), g.storage(rows as u64));
            steps.push(model::block::ln_stats_fwd(g, ln, x, &mean, &inv, width, rows, eps));
            steps.push(g.step(self.k("layernorm_dgamma"), &[dy, x, &mean, &inv, gw], &[width, rows], width));
            steps.push(g.step(self.k("layernorm_dbeta"), &[dy, gb], &[width, rows], width));
            g.submit(&[], &steps);
            steps = Vec::new();
        }
        steps.push(model::block::layernorm_dx_bwd(g, ln, x, gamma, dy, dx, width, rows, eps));
        g.submit(&[], &steps);
    }

    /// A biased linear's backward: `out = x W^T + b` over `rows` rows,
    /// accumulating `dW`/`db` when the parameter has a gradient buffer, and
    /// returning `dx = dy W` (`[rows, k]`).
    fn linear_bwd(&self, prefix: &str, x: &DeviceBuffer, dy: &DeviceBuffer, rows: u32, k: u32, n: u32) -> DeviceBuffer {
        let g = &self.gpu;
        let mut steps = Vec::new();
        if let Some(gw) = self.gr(&format!("{prefix}.weight")) {
            steps.push(g.step(self.k("matmul_dw"), &[dy, x, gw], &[rows, k, n], n * k));
        }
        if let Some(gb) = self.gr(&format!("{prefix}.bias")) {
            steps.push(g.step(self.k("bias_grad"), &[dy, gb], &[rows, n], n));
        }
        let dx = g.storage((rows * k) as u64);
        steps.push(g.step(self.k("matmul_dx"), &[dy, self.ps.w(&format!("{prefix}.weight")), &dx], &[rows, k, n, 0], rows * k));
        g.submit(&[], &steps);
        dx
    }

    /// The merger's backward: parameter gradients, and the gradient with
    /// respect to the post-layernorm output (`[rows, C]`, pass order).
    fn merger_bwd(&self, a: &MergerActs, d_out: &DeviceBuffer) -> DeviceBuffer {
        let g = &self.gpu;
        let v = &self.cfg.vision;
        let (c, k) = (v.hidden, self.cfg.merge);
        let (mc, d, r) = (c * k * k, self.cfg.text.d_model, a.out_rows);
        let d_h2 = self.linear_bwd("merger.fc2", &a.h2, d_out, r, mc, d);
        let d_h1 = g.storage((r * mc) as u64);
        g.submit(&[], &[g.step(self.k("gelu_erf_bwd"), &[&a.h1, &d_h2, &d_h1], &[r * mc], r * mc)]);
        let d_xn = self.linear_bwd("merger.fc1", &a.xn, &d_h1, r, mc, mc);
        let d_xp = g.storage((a.rows * c) as u64);
        self.ln_bwd(&self.ln_ids(), &a.xp, "merger.pre_norm", &d_xn, &d_xp, mc, r, crate::vision::MERGER_EPS);
        let d_post = g.storage((a.rows * c) as u64);
        let ids = vit::VitPermuteIds { embed: self.k("embed"), row_scatter: self.k("row_scatter") };
        g.submit(&[], &[vit::scatter_rows(g, &ids, &a.perm, &d_xp, &d_post, a.rows, c, a.rows)]);
        d_post
    }

    /// The window merger's backward: parameter gradients, and the gradient
    /// with respect to its input (`[rows, C]`, pass order).
    fn window_merger_bwd(&self, a: &crate::vision::WmActs, d_out: &DeviceBuffer) -> DeviceBuffer {
        let g = &self.gpu;
        let v = &self.cfg.vision;
        let (c, ww) = (v.hidden, v.window * v.window);
        let (wc, wm) = (v.window_hidden(), v.window_mlp());
        let (rows, windows) = (a.rows, a.windows);
        let (heads, hd) = (v.heads, v.head_dim());
        let ln = self.ln_ids();
        // out = fc2(gelu(fc1(LN(hw)))) + mean(hw)
        let d_h2 = self.linear_bwd("wm.fc2", &a.h2, d_out, windows, wm, c);
        let d_h1 = g.storage((windows * wm) as u64);
        g.submit(&[], &[g.step(self.k("gelu_bwd"), &[&a.h1, &d_h2, &d_h1], &[windows * wm], windows * wm)]);
        let d_flat = self.linear_bwd("wm.fc1", &a.flat_ln, &d_h1, windows, wc, wm);
        let d_hw = g.storage((rows * c) as u64);
        self.ln_bwd(&ln, &a.hw, "wm.pre_norm", &d_flat, &d_hw, wc, windows, v.eps);
        let bcast = g.storage((rows * c) as u64);
        g.submit(
            &[],
            &[
                g.step(self.k("segment_bcast_rows"), &[&a.ones, d_out, &bcast], &[windows, ww, c], rows * c),
                g.step(self.k("axpy"), &[&d_hw, &bcast], &[rows * c, f(1.0 / ww as f32)], rows * c),
            ],
        );
        // hw = xp + proj(attn(LN1(xp)))
        let d_ctx = self.linear_bwd("wm.attn.proj", &a.ctx, &d_hw, rows, c, c);
        let stride = 3 * c;
        let d_scores = g.storage((windows * heads * ww * ww) as u64);
        let d_qkv = g.storage((rows * stride) as u64);
        g.submit(
            &[&d_qkv],
            &[
                g.step(self.k("attn_bwd_dscores_bidir"), &[&d_ctx, &a.qkv, &a.probs, &d_scores], &[windows, heads, ww, hd, stride, 2 * c, c], windows * heads * ww),
                g.step(self.k("attn_bwd_dq_bidir"), &[&d_scores, &a.qkv, &d_qkv], &[windows, heads, ww, hd, stride, 0, c], rows * c),
                g.step(self.k("attn_bwd_dk_bidir"), &[&d_scores, &a.qkv, &d_qkv], &[windows, heads, ww, hd, stride, 0, c], rows * c),
                g.step(self.k("attn_bwd_dv_bidir"), &[&a.probs, &d_ctx, &d_qkv], &[windows, heads, ww, hd, stride, 2 * c, c], rows * c),
            ],
        );
        let d_ln1 = self.linear_bwd("wm.attn.qkv", &a.ln1, &d_qkv, rows, c, stride);
        let d_xp_ln = g.storage((rows * c) as u64);
        self.ln_bwd(&ln, &a.xp, "wm.norm1", &d_ln1, &d_xp_ln, c, rows, v.eps);
        g.submit(&[], &[g.step(self.k("axpy"), &[&d_hw, &d_xp_ln], &[rows * c, f(1.0)], rows * c)]);
        let d_in = g.storage((rows * c) as u64);
        let ids = vit::VitPermuteIds { embed: self.k("embed"), row_scatter: self.k("row_scatter") };
        g.submit(&[], &[vit::scatter_rows(g, &ids, &a.perm, &d_hw, &d_in, rows, c, rows)]);
        d_in
    }
}

/// The tower linears LoRA can target, by leaf name.
pub const LORA_LEAVES: &[&str] = &["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"];

/// The tower's own kernel indices in the shared fold's shape.
fn fold_kernels(tower: &Tower) -> model::adapter::fold::FoldKernels {
    model::adapter::fold::FoldKernels { matmul: tower.k("matmul"), matmul_dx: tower.k("matmul_dx"), matmul_dw: tower.k("matmul_dw"), axpy: tower.k("axpy") }
}

/// LoRA adapters on the tower's linears: frozen base weights `W0` (kept
/// aside), trainable factors `A [r, in]` and `B [out, r]` in their own store,
/// `W = W0 + (alpha / r) B A` folded into the tower before each forward.
pub struct TowerLora {
    pub rank: u32,
    pub alpha: f32,
    /// `(weight name, out, in)` of each targeted linear.
    targets: Vec<(String, u32, u32)>,
    base: HashMap<String, DeviceBuffer>,
    pub ps: ParamStore,
}

impl TowerLora {
    /// Adapters on every `leaves` linear of every encoder block of `tower`
    /// (whose targeted weights must carry gradient buffers): `A` from
    /// `a_init` (`[r, in]` values per target, in target order), `B` zero, so
    /// the adapted tower starts as the base.
    pub fn new(tower: &Tower, leaves: &[&str], rank: u32, alpha: f32, a_init: &mut dyn FnMut(usize) -> Vec<f32>) -> Result<TowerLora, String> {
        if rank == 0 {
            return Err("minicpmv: a LoRA rank must be at least 1".into());
        }
        let v = &tower.cfg.vision;
        let dims = |leaf: &str| match leaf {
            "attn.qkv" => Ok((3 * v.hidden, v.hidden)),
            "attn.proj" => Ok((v.hidden, v.hidden)),
            "mlp.fc1" => Ok((v.mlp, v.hidden)),
            "mlp.fc2" => Ok((v.hidden, v.mlp)),
            other => Err(format!("minicpmv: {other:?} is not a tower LoRA target (one of {LORA_LEAVES:?})")),
        };
        let mut targets = Vec::new();
        for l in 0..v.n_layers {
            for leaf in leaves {
                let (o, i) = dims(leaf)?;
                let name = format!("blocks.{l}.{leaf}.weight");
                if tower.gr(&name).is_none() {
                    return Err(format!("minicpmv: tower LoRA needs a gradient buffer on {name}"));
                }
                targets.push((name, o, i));
            }
        }
        let mut init = HashMap::new();
        let mut params = Vec::new();
        for (name, o, i) in &targets {
            let a = a_init((rank * i) as usize);
            if a.len() != (rank * i) as usize {
                return Err(format!("minicpmv: LoRA A init for {name} has {} values, expected {}", a.len(), rank * i));
            }
            init.insert(format!("{name}.lora_a"), a);
            init.insert(format!("{name}.lora_b"), vec![0.0; (o * rank) as usize]);
            params.push((format!("{name}.lora_a"), (rank * i) as usize));
            params.push((format!("{name}.lora_b"), (o * rank) as usize));
        }
        let ps = ParamStore::new(&tower.gpu, params, &init);
        let base = targets
            .iter().map(|(n, o, i)| (n.clone(), tower.copy(tower.ps.w(n), o * i))).collect();
        Ok(TowerLora { rank, alpha, targets, base, ps })
    }

    fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Write `W0 + s B A` into every targeted weight of `tower`, then
    /// project the backward's dense gradients back onto the factors - the
    /// shared `model::adapter::fold` choreography, over this store's buffers.
    pub fn fold(&self, tower: &Tower) {
        let g = &tower.gpu;
        let k = fold_kernels(tower);
        let targets: Vec<model::adapter::fold::Target<'_>> = self
            .targets
            .iter()
            .map(|(name, o, i)| model::adapter::fold::Target {
                weight: tower.ps.w(name),
                base: &self.base[name],
                a: self.ps.w(&format!("{name}.lora_a")),
                b: self.ps.w(&format!("{name}.lora_b")),
                out: *o,
                inn: *i,
            })
            .collect();
        model::adapter::fold::fold(g, &k, &targets, self.rank, self.scale());
    }

    /// Accumulate the factors' gradients from the dense weight gradients the
    /// tower's backward left in its targeted weights' gradient buffers.
    pub fn project(&self, tower: &Tower) {
        let g = &tower.gpu;
        let k = fold_kernels(tower);
        let targets: Vec<model::adapter::fold::GradTarget<'_>> = self
            .targets
            .iter()
            .map(|(name, o, i)| {
                let (a, b) = (format!("{name}.lora_a"), format!("{name}.lora_b"));
                model::adapter::fold::GradTarget {
                    dw: tower.gr(name).expect("checked at construction"),
                    a: self.ps.w(&a),
                    b: self.ps.w(&b),
                    ga: self.ps.g(&a),
                    gb: self.ps.g(&b),
                    out: *o,
                    inn: *i,
                }
            })
            .collect();
        model::adapter::fold::project(g, &k, &targets, self.rank, self.scale());
    }


    /// Every adapter tensor, host-side, by name.
    pub fn tensors(&self, gpu: &Gpu) -> Vec<(String, Vec<f32>)> {
        self.ps.params.iter().map(|(n, _)| (n.clone(), self.ps.read_weight(gpu, n))).collect()
    }
}
