// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The bidirectional state encoder's forward.
//!
//! ```text
//! x0     = LN_emb(embed(ids) + embed(pos_ids) + embed(type_ids))
//! per layer (POST-LayerNorm, which is where BERT differs from CLIP's tower):
//!   qkv  = x @ Wqkv^T + b                                   [rows, 3H]
//!   ctx  = bidirectional self-attention, per span           [rows, H]
//!   res  = LN1(x + (ctx @ Wo^T + bo))
//!   x'   = LN2(res + (gelu(res @ W1^T + b1) @ W2^T + b2))
//! ```
//!
//! **Sequences are packed, not padded.** Every sequence in a call is laid end
//! to end in one flat `[rows, H]` buffer and described by a span `(row0, len)`;
//! `block::chunked_bidir_fwd` self-attends within each span independently. That
//! is what lets one forward carry a batch of different lengths - state windows
//! and option slots in the same call - with no mask buffer, no padded rows to
//! compute, and no wasted attention area.
//!
//! It also sidesteps the trap that makes a padded bidirectional encoder subtly
//! wrong: with no causal structure, an unmasked pad position contributes to
//! every real position's attention. Packing removes the pad positions rather
//! than remembering to mask them.
//!
//! Position ids are supplied per row rather than derived from a fixed stride,
//! for the same reason - `pos_add` assumes every sequence has the same length.
//!
//! The step list is rebuilt when the spans change, because `chunked_bidir_fwd`
//! bakes them in. That is host-side work proportional to the span count, not a
//! device cost, and a caller whose spans are stable (the realtime path, whose
//! window size and option set repeat) rebuilds nothing.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::EncoderConfig;

// ---- kernel indices (order matches PIPELINES) ----
const K_EMBED: usize = 0;
const K_MATMUL: usize = 1;
const K_MATMUL_REG3: usize = 2;
const K_BIAS_ADD: usize = 3;
const K_ADD2: usize = 4;
const K_GELU_ERF: usize = 5;
const K_LAYERNORM: usize = 6;
const K_LN_STATS: usize = 7;
const K_LAYERNORM_DX: usize = 8;
const K_SCORES_CROSS: usize = 9;
const K_SOFTMAX_CROSS: usize = 10;
const K_APPLY_CROSS: usize = 11;

/// Every kernel this model dispatches. `layernorm_rows` has no index of its own
/// because `block::LayerNormIds::resolve` picks it BY NAME when the device
/// supports workgroup reductions; it must be registered for that lookup to
/// find it.
pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("bias_add", kernels::BIAS_ADD),
    ("add2", kernels::ADD2),
    ("gelu_erf", kernels::GELU_ERF),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
];

/// Attention-slab budget for the chunked path: the `[heads, chunk, len]` score
/// and probability slabs are sized against it, so `chunk` falls as the longest
/// span grows and the allocation stays bounded whatever a caller asks for.
const SLAB_BUDGET: u64 = 256 << 20;

struct LayerBufs {
    /// Fused `[rows, 3H]` - q at 0, k at H, v at 2H.
    qkv: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    /// `x + attn_out`, before LN1.
    res_pre: DeviceBuffer,
    res: DeviceBuffer,
    h: DeviceBuffer,
    h_act: DeviceBuffer,
    mlp_out: DeviceBuffer,
    /// `res + mlp_out`, before LN2.
    ffn_pre: DeviceBuffer,
}

pub struct Encoder {
    pub gpu: Gpu,
    pub cfg: EncoderConfig,
    pub ps: ParamStore,
    /// Capacity in rows; a call may use fewer.
    cap_rows: u32,
    rows: u32,
    spans: Vec<(u32, u32)>,
    chunk: u32,
    ids: DeviceBuffer,
    pos_ids: DeviceBuffer,
    type_ids: DeviceBuffer,
    e_tok: DeviceBuffer,
    e_pos: DeviceBuffer,
    e_type: DeviceBuffer,
    sum1: DeviceBuffer,
    sum2: DeviceBuffer,
    /// `x[0]` = embedding output, `x[i+1]` = layer `i`'s output.
    x: Vec<DeviceBuffer>,
    layers: Vec<LayerBufs>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    steps: Vec<Step>,
}

impl Encoder {
    /// Build on an existing device, sized for at most `cap_rows` packed tokens
    /// and a longest span of `max_span`. Every parameter is `Frozen`: this is
    /// the inference graph the parity ladder gates.
    pub fn new_on(
        gpu: Gpu,
        cfg: EncoderConfig,
        cap_rows: u32,
        max_span: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> Encoder {
        assert!(
            max_span <= cfg.max_positions,
            "span {max_span} > max_positions {} - a window may not outrun the learned position table",
            cfg.max_positions
        );
        assert!(max_span <= cap_rows, "max_span {max_span} > cap_rows {cap_rows}");
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = cap_rows as u64;
        let h = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        // `chunk` query rows at a time against a whole span's keys.
        let per_row = cfg.n_heads as u64 * max_span as u64 * 4;
        let chunk = ((SLAB_BUDGET / per_row.max(1)).max(1) as u32).min(max_span.max(1));
        let slab = cfg.n_heads as u64 * chunk as u64 * max_span as u64;

        let idbuf = |name: &str| {
            gpu.buffer(name, n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
        };
        let layers: Vec<LayerBufs> = (0..cfg.n_layers)
            .map(|_| LayerBufs {
                qkv: gpu.storage(n * 3 * h),
                ctx: gpu.storage(n * h),
                attn_out: gpu.storage(n * h),
                res_pre: gpu.storage(n * h),
                res: gpu.storage(n * h),
                h: gpu.storage(n * ff),
                h_act: gpu.storage(n * ff),
                mlp_out: gpu.storage(n * h),
                ffn_pre: gpu.storage(n * h),
            })
            .collect();
        let mut e = Encoder {
            cap_rows,
            rows: 0,
            spans: Vec::new(),
            chunk,
            ids: idbuf("ids"),
            pos_ids: idbuf("pos_ids"),
            type_ids: idbuf("type_ids"),
            e_tok: gpu.storage(n * h),
            e_pos: gpu.storage(n * h),
            e_type: gpu.storage(n * h),
            sum1: gpu.storage(n * h),
            sum2: gpu.storage(n * h),
            x: (0..=cfg.n_layers).map(|_| gpu.storage(n * h)).collect(),
            layers,
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            steps: Vec::new(),
            gpu,
            cfg,
            ps,
        };
        e.rows = cap_rows;
        e.spans = vec![(0, cap_rows.min(max_span))];
        e.steps = e.build_steps();
        e
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG3, false)
    }

    /// Load one packed call: `ids`/`type_ids` are the flat token and segment
    /// streams, `spans` the `(row0, len)` of each sequence within them.
    ///
    /// Position ids are derived here - `0..len` within each span - so a caller
    /// never has to know that a packed row's position is not its row index.
    pub fn set_batch(&mut self, ids: &[u32], type_ids: &[u32], spans: &[(u32, u32)]) {
        assert_eq!(ids.len(), type_ids.len(), "ids and type_ids must be the same length");
        assert!(ids.len() <= self.cap_rows as usize, "{} rows > capacity {}", ids.len(), self.cap_rows);
        let covered: u32 = spans.iter().map(|&(_, l)| l).sum();
        assert_eq!(covered as usize, ids.len(), "spans cover {covered} rows but {} were supplied", ids.len());
        let mut pos = vec![0u32; ids.len()];
        for &(row0, len) in spans {
            assert!(
                len <= self.cfg.max_positions,
                "span of {len} rows > max_positions {}",
                self.cfg.max_positions
            );
            for i in 0..len {
                pos[(row0 + i) as usize] = i;
            }
        }
        self.gpu.write(&self.ids, ids);
        self.gpu.write(&self.pos_ids, &pos);
        self.gpu.write(&self.type_ids, type_ids);
        let changed = self.rows != ids.len() as u32 || self.spans != spans;
        self.rows = ids.len() as u32;
        if changed {
            self.spans = spans.to_vec();
            self.steps = self.build_steps();
        }
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let n = self.rows;
        let h = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim();
        let ln = block::LayerNormIds::resolve(g, K_LAYERNORM, K_LN_STATS, K_LAYERNORM_DX);
        let cross = block::CrossIds { scores: K_SCORES_CROSS, softmax: K_SOFTMAX_CROSS, apply: K_APPLY_CROSS };
        // ---- embeddings ----
        // `embed` Params: [width, rows]; bufs [index(u32), table, out]. The
        // position and segment tables are gathered the same way as the token
        // table rather than added by stride, because packed spans do not share
        // one sequence length (see the module docs).
        let mut s = vec![
            g.step(K_EMBED, &[&self.ids, self.w("tok.weight"), &self.e_tok], &[h, n], n * h),
            g.step(K_EMBED, &[&self.pos_ids, self.w("pos.weight"), &self.e_pos], &[h, n], n * h),
            g.step(K_EMBED, &[&self.type_ids, self.w("type.weight"), &self.e_type], &[h, n], n * h),
        ];
        s.push(g.step(K_ADD2, &[&self.e_tok, &self.e_pos, &self.sum1], &[n * h], n * h));
        s.push(g.step(K_ADD2, &[&self.sum1, &self.e_type, &self.sum2], &[n * h], n * h));
        s.push(block::layernorm_fwd(
            g,
            &ln,
            &self.sum2,
            self.w("emb_ln.weight"),
            self.w("emb_ln.bias"),
            &self.x[0],
            h,
            n,
            c.eps,
        ));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = format!("blocks.{l}");

            let (mk, mt) = self.gemm(n, 3 * h);
            s.push(g.step(mk, &[&self.x[l], self.w(&format!("{p}.qkv.weight")), &lb.qkv], &[n, h, 3 * h], mt));
            s.push(g.step(K_BIAS_ADD, &[&lb.qkv, self.w(&format!("{p}.qkv.bias"))], &[n, 3 * h], n * 3 * h));

            // Self-attention within each span, independently. q/k/v live at
            // 0/H/2H of the fused row.
            block::chunked_bidir_fwd(
                g,
                &cross,
                None,
                c.n_heads,
                hd,
                h,
                &lb.qkv,
                3 * h,
                0,
                h,
                2 * h,
                &lb.ctx,
                &self.scores,
                &self.probs,
                &self.spans,
                self.chunk,
                None,
                &mut s,
            );

            let (mk, mt) = self.gemm(n, h);
            s.push(g.step(mk, &[&lb.ctx, self.w(&format!("{p}.proj.weight")), &lb.attn_out], &[n, h, h], mt));
            s.push(g.step(K_BIAS_ADD, &[&lb.attn_out, self.w(&format!("{p}.proj.bias"))], &[n, h], n * h));
            // POST-LayerNorm: the residual is added first and normalized after.
            s.push(g.step(K_ADD2, &[&self.x[l], &lb.attn_out, &lb.res_pre], &[n * h], n * h));
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &lb.res_pre,
                self.w(&format!("{p}.ln1.weight")),
                self.w(&format!("{p}.ln1.bias")),
                &lb.res,
                h,
                n,
                c.eps,
            ));

            let (mk, mt) = self.gemm(n, ff);
            s.push(g.step(mk, &[&lb.res, self.w(&format!("{p}.fc1.weight")), &lb.h], &[n, h, ff], mt));
            s.push(g.step(K_BIAS_ADD, &[&lb.h, self.w(&format!("{p}.fc1.bias"))], &[n, ff], n * ff));
            s.push(g.step(K_GELU_ERF, &[&lb.h, &lb.h_act], &[n * ff], n * ff));
            let (mk, mt) = self.gemm(n, h);
            s.push(g.step(mk, &[&lb.h_act, self.w(&format!("{p}.fc2.weight")), &lb.mlp_out], &[n, ff, h], mt));
            s.push(g.step(K_BIAS_ADD, &[&lb.mlp_out, self.w(&format!("{p}.fc2.bias"))], &[n, h], n * h));
            s.push(g.step(K_ADD2, &[&lb.res, &lb.mlp_out, &lb.ffn_pre], &[n * h], n * h));
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &lb.ffn_pre,
                self.w(&format!("{p}.ln2.weight")),
                self.w(&format!("{p}.ln2.bias")),
                &self.x[l + 1],
                h,
                n,
                c.eps,
            ));
        }
        s
    }

    // ---- parity / inference taps ----

    /// The final hidden states, `[rows, H]` row-major over the PACKED rows.
    pub fn hidden(&self) -> Vec<f32> {
        self.read(&self.x[self.cfg.n_layers as usize])
    }

    /// Layer `l`'s output (`l == 0` is the first block's output; use
    /// [`Encoder::embeddings`] for the pre-block residual).
    pub fn layer_out(&self, l: usize) -> Vec<f32> {
        self.read(&self.x[l + 1])
    }

    /// The post-embedding residual, after its LayerNorm.
    pub fn embeddings(&self) -> Vec<f32> {
        self.read(&self.x[0])
    }

    /// Layer `l`'s attention context, before the output projection.
    pub fn attn_ctx(&self, l: usize) -> Vec<f32> {
        self.read(&self.layers[l].ctx)
    }

    /// Layer `l`'s post-attention residual, after LN1.
    pub fn attn_out(&self, l: usize) -> Vec<f32> {
        self.read(&self.layers[l].res)
    }

    /// Layer `l`'s FFN hidden after its GELU, `[rows, d_ff]`.
    pub fn ffn_act(&self, l: usize) -> Vec<f32> {
        self.gpu.read(&self.layers[l].h_act, (self.rows * self.cfg.d_ff) as usize)
    }

    /// Mean of the final hidden states over each span - the sentence-transformer
    /// pooling head. `[spans, H]`.
    ///
    /// Packing is what makes this exact: there are no pad rows in the mean, so
    /// nothing has to be excluded from it.
    pub fn pooled_mean(&self) -> Vec<f32> {
        let h = self.cfg.d_model as usize;
        let hid = self.hidden();
        let mut out = Vec::with_capacity(self.spans.len() * h);
        for &(row0, len) in &self.spans {
            let mut acc = vec![0.0f32; h];
            for r in 0..len as usize {
                let base = (row0 as usize + r) * h;
                for (a, v) in acc.iter_mut().zip(&hid[base..base + h]) {
                    *a += v;
                }
            }
            let inv = 1.0 / len.max(1) as f32;
            out.extend(acc.into_iter().map(|v| v * inv));
        }
        out
    }

    /// Read the live prefix of a `[cap_rows, H]` buffer - the rows this call
    /// actually packed, not the capacity.
    fn read(&self, b: &DeviceBuffer) -> Vec<f32> {
        self.gpu.read(b, (self.rows * self.cfg.d_model) as usize)
    }
}
