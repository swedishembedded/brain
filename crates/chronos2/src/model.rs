// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chronos-2 forward (inference) — composes the preprocessing and the WGSL
//! kernels into the univariate forecast path.
//!
//! Univariate Phase 1: a single series, so `group_ids = arange(B)` with `B=1`.
//! Group (cross-variate) attention then operates on a length-1 sequence, so
//! softmax is trivially 1 and its output collapses to `o_proj(v_proj(RMSNorm(x)))`
//! per token — no attention kernel needed for the group sublayer (the spec
//! explicitly endorses this degeneration). The time (temporal) attention is the
//! real bidirectional MHA over the token axis.
//!
//! Op sequence per token sequence `S = n_ctx + 1(REG) + n_out`:
//! ```text
//! emb = [ input_patch_embedding(ctx) , shared[REG] , input_patch_embedding(future) ]
//! for each of L blocks:
//!   time-attn:  h += o_proj( attend( rope(q),rope(k), v ) of RMSNorm(h) )
//!   group-attn: h += o_proj( v_proj( RMSNorm(h) ) )            # B=1 degenerate
//!   ffn:        h += wo( relu( wi( RMSNorm(h) ) ) )
//! h = final_layer_norm(h)
//! qp = output_patch_embedding(h[-n_out:])                       # [n_out, 21*16]
//! rearrange b n (q p) -> [21, n_out*16]                         # q OUTER, p inner
//! denorm: sinh then affine  (InstanceNorm.inverse)
//! ```
//!
//! Correctness of every kernel this calls is covered by `tests/kernels.rs`
//! (isolation) and the preprocessing by `preprocess.rs` tests; the model test
//! validates the *composition/wiring* end to end.

use crate::config::Chronos2Config;
use crate::preprocess;
use gpu_core::{f, DeviceBuffer, Gpu};
use std::collections::HashMap;

// Kernel pipeline indices (order must match PIPELINES).
const MATMUL: usize = 0;
const BIAS_ADD: usize = 1;
const RELU: usize = 2;
const ADD: usize = 3;
const RMSNORM: usize = 4;
const ROPE_NEOX: usize = 5;
const ATTN_SCORES_FULL: usize = 6;
const ATTN_SOFTMAX_FULL: usize = 7;
const ATTN_APPLY_FULL: usize = 8;
const MATMUL_TILED: usize = 9;

const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("relu_inplace", kernels::RELU_INPLACE),
    ("add", kernels::ADD),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_neox", kernels::ROPE_NEOX),
    ("attn_scores_full", kernels::ATTN_SCORES_FULL),
    ("attn_softmax_full", kernels::ATTN_SOFTMAX_FULL),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("matmul_tiled", kernels::MATMUL_TILED),
];

/// Workgroups for the tiled GEMM (32×32 output tile) → invocation count.
#[inline]
fn tiled_threads(m: usize, n: usize) -> u32 {
    (m.div_ceil(32) * n.div_ceil(32) * 64) as u32
}

/// A loaded Chronos-2 model ready for inference.
pub struct Chronos2 {
    gpu: Gpu,
    cfg: Chronos2Config,
    w: HashMap<String, DeviceBuffer>,
}

impl Chronos2 {
    /// Build from host-side weights (name → values), keyed by the reference
    /// `state_dict` names (see [`Chronos2Config::param_list`]). Missing weights
    /// are a hard error.
    pub fn from_weights(
        cfg: Chronos2Config,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Result<Chronos2, String> {
        let gpu = Gpu::new(PIPELINES);
        let mut w = HashMap::new();
        for (name, shape) in cfg.param_list() {
            let numel: usize = shape.iter().product();
            let data = weights
                .get(&name)
                .ok_or_else(|| format!("chronos2: missing weight {name}"))?;
            if data.len() != numel {
                return Err(format!(
                    "chronos2: {name} has {} elems, expected {numel}",
                    data.len()
                ));
            }
            w.insert(name.clone(), gpu.storage_init(&name, data));
        }
        Ok(Chronos2 { gpu, cfg, w })
    }

    /// Load a model from a brain `.weights` container (see [`crate::import`]).
    pub fn load(path: &str) -> Result<Chronos2, String> {
        let c = checkpoint::load(path);
        let cfg = Chronos2Config::from_hf(&c.header["config"])?;
        let weights = c.by_role("");
        Chronos2::from_weights(cfg, &weights)
    }

    /// The config this model was built with.
    pub fn config(&self) -> &Chronos2Config {
        &self.cfg
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("chronos2: weight {name} not loaded"))
    }

    // -- per-op device helpers (immediate submit; buffers are local vars) -----

    fn mm(&self, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: usize, k: usize, n: usize) {
        // The tiled GEMM wins only when there are enough rows to fill its 32-row
        // tiles; for the small-m matmuls (patch/head, m≈S) the padding waste +
        // barrier overhead make it slower, so those keep the naive kernel. (On
        // this GPU even large-m is ~neutral — it is dispatch-overhead bound — but
        // the gate avoids the small-m regression and helps on beefier GPUs.)
        let (kind, threads) =
            if m >= 64 { (MATMUL_TILED, tiled_threads(m, n)) } else { (MATMUL, (m * n) as u32) };
        let s = self.gpu.step(kind, &[x, self.w(wname), out], &[m as u32, k as u32, n as u32], threads);
        self.gpu.submit(&[], &[s]);
    }
    fn bias(&self, out: &DeviceBuffer, bname: &str, m: usize, n: usize) {
        let s = self.gpu.step(BIAS_ADD, &[out, self.w(bname)], &[m as u32, n as u32], (m * n) as u32);
        self.gpu.submit(&[], &[s]);
    }
    fn relu(&self, buf: &DeviceBuffer, total: usize) {
        let s = self.gpu.step(RELU, &[buf], &[total as u32], total as u32);
        self.gpu.submit(&[], &[s]);
    }
    fn add(&self, src: &DeviceBuffer, dst: &DeviceBuffer, total: usize) {
        let s = self.gpu.step(ADD, &[src, dst], &[total as u32], total as u32);
        self.gpu.submit(&[], &[s]);
    }
    fn rms(&self, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, d: usize, rows: usize) {
        let s = self.gpu.step(RMSNORM, &[x, self.w(wname), out], &[d as u32, rows as u32], rows as u32);
        self.gpu.submit(&[], &[s]);
    }

    /// A `ResidualBlock` (biased): `output(relu(hidden(x))) + residual(x)`.
    /// Returns the `[rows, out_dim]` result buffer.
    fn residual_block(
        &self,
        prefix: &str,
        x: &DeviceBuffer,
        rows: usize,
        in_dim: usize,
        h: usize,
        out_dim: usize,
    ) -> DeviceBuffer {
        let hid = self.gpu.storage((rows * h) as u64);
        self.mm(x, &format!("{prefix}.hidden_layer.weight"), &hid, rows, in_dim, h);
        self.bias(&hid, &format!("{prefix}.hidden_layer.bias"), rows, h);
        self.relu(&hid, rows * h);

        let o1 = self.gpu.storage((rows * out_dim) as u64);
        self.mm(&hid, &format!("{prefix}.output_layer.weight"), &o1, rows, h, out_dim);
        self.bias(&o1, &format!("{prefix}.output_layer.bias"), rows, out_dim);

        let res = self.gpu.storage((rows * out_dim) as u64);
        self.mm(x, &format!("{prefix}.residual_layer.weight"), &res, rows, in_dim, out_dim);
        self.bias(&res, &format!("{prefix}.residual_layer.bias"), rows, out_dim);

        self.add(&o1, &res, rows * out_dim); // res += o1
        res
    }

    /// One encoder block, in place on `emb` `[S, D]`. `time_mask` is `[S]`. The
    /// univariate path: time attention → group attention (B=1 degeneration) → FFN.
    fn block(&self, b: usize, emb: &DeviceBuffer, s: usize, time_mask: &DeviceBuffer) {
        let pre = format!("encoder.block.{b}");
        self.time_attention(&pre, emb, s, time_mask);
        self.group_degenerate(&pre, emb, s);
        self.ffn(&pre, emb, s);
    }

    /// Time self-attention sublayer (bidirectional MHA with NeoX RoPE), residual,
    /// in place on `emb` `[S, D]`.
    fn time_attention(&self, pre: &str, emb: &DeviceBuffer, s: usize, time_mask: &DeviceBuffer) {
        let d = self.cfg.d_model;
        let inner = self.cfg.inner_dim();
        let heads = self.cfg.num_heads;
        let hd = self.cfg.d_kv;
        let xn = self.gpu.storage((s * d) as u64);
        self.rms(emb, &format!("{pre}.layer.0.layer_norm.weight"), &xn, d, s);
        let q = self.gpu.storage((s * inner) as u64);
        let k = self.gpu.storage((s * inner) as u64);
        let v = self.gpu.storage((s * inner) as u64);
        self.mm(&xn, &format!("{pre}.layer.0.self_attention.q.weight"), &q, s, d, inner);
        self.mm(&xn, &format!("{pre}.layer.0.self_attention.k.weight"), &k, s, d, inner);
        self.mm(&xn, &format!("{pre}.layer.0.self_attention.v.weight"), &v, s, d, inner);
        self.rope(&q, s, heads, hd);
        self.rope(&k, s, heads, hd);
        let scores = self.gpu.storage((heads * s * s) as u64);
        let sc = self.gpu.step(
            ATTN_SCORES_FULL,
            &[&q, &k, time_mask, &scores],
            &[1, heads as u32, s as u32, hd as u32, inner as u32],
            (heads * s * s) as u32,
        );
        self.gpu.submit(&[], &[sc]);
        let probs = self.gpu.storage((heads * s * s) as u64);
        let sm = self.gpu.step(ATTN_SOFTMAX_FULL, &[&scores, &probs], &[1, heads as u32, s as u32], (heads * s) as u32);
        self.gpu.submit(&[], &[sm]);
        let ctx = self.gpu.storage((s * d) as u64);
        let ap = self.gpu.step(
            ATTN_APPLY_FULL,
            &[&probs, &v, &ctx],
            &[1, heads as u32, s as u32, hd as u32, inner as u32, d as u32],
            (heads * s * hd) as u32,
        );
        self.gpu.submit(&[], &[ap]);
        let o = self.gpu.storage((s * d) as u64);
        self.mm(&ctx, &format!("{pre}.layer.0.self_attention.o.weight"), &o, s, inner, d);
        self.add(&o, emb, s * d); // residual
    }

    /// Group self-attention, B=1 degeneration: with a single series the softmax
    /// over the group axis is 1, so the sublayer reduces to
    /// `emb += o_proj(v_proj(RMSNorm(emb)))` per token.
    fn group_degenerate(&self, pre: &str, emb: &DeviceBuffer, s: usize) {
        let d = self.cfg.d_model;
        let inner = self.cfg.inner_dim();
        let xn2 = self.gpu.storage((s * d) as u64);
        self.rms(emb, &format!("{pre}.layer.1.layer_norm.weight"), &xn2, d, s);
        let vg = self.gpu.storage((s * inner) as u64);
        self.mm(&xn2, &format!("{pre}.layer.1.self_attention.v.weight"), &vg, s, d, inner);
        let og = self.gpu.storage((s * d) as u64);
        self.mm(&vg, &format!("{pre}.layer.1.self_attention.o.weight"), &og, s, inner, d);
        self.add(&og, emb, s * d);
    }

    /// Feed-forward sublayer (ReLU MLP), residual, in place on `emb` `[S, D]`.
    fn ffn(&self, pre: &str, emb: &DeviceBuffer, s: usize) {
        let d = self.cfg.d_model;
        let f = self.cfg.d_ff;
        let xn3 = self.gpu.storage((s * d) as u64);
        self.rms(emb, &format!("{pre}.layer.2.layer_norm.weight"), &xn3, d, s);
        let hid = self.gpu.storage((s * f) as u64);
        self.mm(&xn3, &format!("{pre}.layer.2.mlp.wi.weight"), &hid, s, d, f);
        self.relu(&hid, s * f);
        let ff = self.gpu.storage((s * d) as u64);
        self.mm(&hid, &format!("{pre}.layer.2.mlp.wo.weight"), &ff, s, f, d);
        self.add(&ff, emb, s * d);
    }

    fn rope(&self, buf: &DeviceBuffer, s: usize, heads: usize, hd: usize) {
        let row_stride = heads * hd;
        let st = self.gpu.step(
            ROPE_NEOX,
            &[buf],
            &[s as u32, heads as u32, hd as u32, row_stride as u32, 0, f(self.cfg.rope_theta)],
            (s * heads * (hd / 2)) as u32,
        );
        self.gpu.submit(&[], &[st]);
    }

    /// The transformer core: given assembled token embeddings `emb` (`[S, D]`
    /// row-major) and an additive per-key `kmask` (`[S]`), run the encoder stack,
    /// final norm, and the quantile head on the trailing `n_out` tokens, and
    /// return the raw head output `[n_out, head_out]` (before the rearrange +
    /// denorm). This is exactly what the ONNX / NPU graph computes, so it is the
    /// parity reference for the NPU export. `forecast_quantiles` composes this
    /// with the host-side scaler/patch/embed and rearrange/denorm.
    pub fn core_forward(&self, emb: &[f32], kmask: &[f32], n_out: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let head_out = cfg.head_out_dim();
        let s = kmask.len();
        assert_eq!(emb.len(), s * d, "emb must be [S, D]");

        let emb_buf = self.gpu.storage_init("emb", emb);
        let mask_buf = self.gpu.storage_init("mask", kmask);
        for b in 0..cfg.num_layers {
            self.block(b, &emb_buf, s, &mask_buf);
        }
        let normed = self.gpu.storage((s * d) as u64);
        self.rms(&emb_buf, "encoder.final_layer_norm.weight", &normed, d, s);
        let normed_host = self.gpu.read(&normed, s * d);
        let head_in = self.gpu.storage_init("head_in", &normed_host[(s - n_out) * d..]);
        let qp = self.residual_block("output_patch_embedding", &head_in, n_out, d, cfg.d_ff, head_out);
        self.gpu.read(&qp, n_out * head_out)
    }

    /// Forecast the quantile paths for a single context series over `horizon`
    /// steps. Returns `[num_quantiles, horizon]` row-major (quantile-major).
    pub fn forecast_quantiles(&self, context: &[f32], horizon: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let patch = cfg.input_patch_size;
        let q = cfg.num_quantiles;
        let head_out = cfg.head_out_dim();

        // 1. standardize + patch
        let (ls, scaled) = preprocess::instance_norm(context, cfg.use_arcsinh);
        let ctx = preprocess::context_features(&scaled, patch, cfg.time_encoding_scale);
        let fut = preprocess::future_features(horizon, patch, cfg.time_encoding_scale);
        let n_ctx = ctx.n_patches;
        let n_out = fut.n_patches;
        let s = n_ctx + 1 + n_out; // + REG token

        // 2. embed all patch tokens (context + future) through the shared block
        let mut feats = ctx.features.clone();
        feats.extend_from_slice(&fut.features);
        let feat_buf = self.gpu.storage_init("feats", &feats);
        let patch_emb = self.residual_block("input_patch_embedding", &feat_buf, n_ctx + n_out, 3 * patch, cfg.d_ff, d);
        let patch_emb_host = self.gpu.read(&patch_emb, (n_ctx + n_out) * d);

        // 3. assemble the token sequence with the REG embedding in the middle
        let reg = &self.w_host_row("shared.weight", cfg.reg_token_id, d);
        let mut emb_host = Vec::with_capacity(s * d);
        emb_host.extend_from_slice(&patch_emb_host[..n_ctx * d]);
        emb_host.extend_from_slice(reg);
        emb_host.extend_from_slice(&patch_emb_host[n_ctx * d..]);
        let emb = self.gpu.storage_init("emb", &emb_host);

        // 4. time-attention mask: ctx patches per their observed-mask, REG + future
        //    always attendable. Additive: 0 attend, large-negative masked.
        let mut mask = vec![0.0f32; s];
        for p in 0..n_ctx {
            if ctx.attn_mask[p] == 0.0 {
                mask[p] = -3.4e38;
            }
        }
        let mask_buf = self.gpu.storage_init("mask", &mask);

        // 5. encoder blocks
        for b in 0..cfg.num_layers {
            self.block(b, &emb, s, &mask_buf);
        }

        // 6. final norm
        let normed = self.gpu.storage((s * d) as u64);
        self.rms(&emb, "encoder.final_layer_norm.weight", &normed, d, s);

        // 7. quantile head on the trailing future tokens
        let normed_host = self.gpu.read(&normed, s * d);
        let head_in = self.gpu.storage_init("head_in", &normed_host[(n_ctx + 1) * d..]);
        let qp = self.residual_block("output_patch_embedding", &head_in, n_out, d, cfg.d_ff, head_out);
        let qp_host = self.gpu.read(&qp, n_out * head_out); // [n_out, q*patch]

        // 8. rearrange b n (q p) -> [q, n*patch] (q OUTER, p inner), denorm per row
        let hlen = n_out * patch;
        let mut out = vec![0.0f32; q * hlen];
        for n in 0..n_out {
            for qi in 0..q {
                for pp in 0..patch {
                    out[qi * hlen + n * patch + pp] = qp_host[n * head_out + qi * patch + pp];
                }
            }
        }
        // denorm each quantile row, then trim to the requested horizon
        let mut trimmed = vec![0.0f32; q * horizon];
        for qi in 0..q {
            let row = &out[qi * hlen..qi * hlen + hlen];
            let dn = preprocess::instance_norm_inverse(row, ls, cfg.use_arcsinh);
            trimmed[qi * horizon..qi * horizon + horizon].copy_from_slice(&dn[..horizon]);
        }
        trimmed
    }

    /// Read a full weight tensor to the host (small; used by the host-side group
    /// attention in the multivariate path).
    fn read_w(&self, name: &str, numel: usize) -> Vec<f32> {
        self.gpu.read(self.w(name), numel)
    }

    /// Group self-attention across the `bc` series at each token position, in
    /// place on the host state `[bc, S, D]` (series-major). Every series is in one
    /// group, so all series attend to each other (no group mask). Multi-head,
    /// **unscaled**, **no RoPE** — the reference `GroupSelfAttention`. B=1
    /// reduces to `o(v(rmsnorm(x)))`, matching [`group_degenerate`].
    fn group_attention_host(&self, pre: &str, state: &mut [f32], bc: usize, s: usize) {
        let d = self.cfg.d_model;
        let inner = self.cfg.inner_dim();
        let heads = self.cfg.num_heads;
        let hd = self.cfg.d_kv;
        let eps = self.cfg.layer_norm_epsilon;
        let gain = self.read_w(&format!("{pre}.layer.1.layer_norm.weight"), d);
        let wq = self.read_w(&format!("{pre}.layer.1.self_attention.q.weight"), inner * d);
        let wk = self.read_w(&format!("{pre}.layer.1.self_attention.k.weight"), inner * d);
        let wv = self.read_w(&format!("{pre}.layer.1.self_attention.v.weight"), inner * d);
        let wo = self.read_w(&format!("{pre}.layer.1.self_attention.o.weight"), d * inner);

        // one token position at a time: the "sequence" is the `bc` series.
        for pos in 0..s {
            // RMSNorm each series' D-vector at this position, then q/k/v projections.
            let mut q = vec![0f32; bc * inner];
            let mut k = vec![0f32; bc * inner];
            let mut v = vec![0f32; bc * inner];
            for b in 0..bc {
                let x = &state[(b * s + pos) * d..(b * s + pos) * d + d];
                let mut ms = 0f32;
                for &val in x {
                    ms += val * val;
                }
                let rms = 1.0 / (ms / d as f32 + eps).sqrt();
                let mut nx = vec![0f32; d];
                for i in 0..d {
                    nx[i] = x[i] * rms * gain[i];
                }
                for o in 0..inner {
                    let (mut sq, mut sk, mut sv) = (0f32, 0f32, 0f32);
                    let base = o * d;
                    for i in 0..d {
                        sq += nx[i] * wq[base + i];
                        sk += nx[i] * wk[base + i];
                        sv += nx[i] * wv[base + i];
                    }
                    q[b * inner + o] = sq;
                    k[b * inner + o] = sk;
                    v[b * inner + o] = sv;
                }
            }
            // per-head attention over the `bc` series (unscaled softmax), then o_proj + residual.
            let mut ctx = vec![0f32; bc * inner];
            for h in 0..heads {
                let off = h * hd;
                for bi in 0..bc {
                    // scores over bj, softmax
                    let mut sc = vec![0f32; bc];
                    let mut mx = f32::NEG_INFINITY;
                    for bj in 0..bc {
                        let mut dot = 0f32;
                        for dd in 0..hd {
                            dot += q[bi * inner + off + dd] * k[bj * inner + off + dd];
                        }
                        sc[bj] = dot; // UNSCALED
                        mx = mx.max(dot);
                    }
                    let mut sum = 0f32;
                    for x in sc.iter_mut() {
                        *x = (*x - mx).exp();
                        sum += *x;
                    }
                    for dd in 0..hd {
                        let mut acc = 0f32;
                        for bj in 0..bc {
                            acc += (sc[bj] / sum) * v[bj * inner + off + dd];
                        }
                        ctx[bi * inner + off + dd] = acc;
                    }
                }
            }
            // o projection + residual back into state
            for b in 0..bc {
                let cx = &ctx[b * inner..b * inner + inner];
                let dst = &mut state[(b * s + pos) * d..(b * s + pos) * d + d];
                for dd in 0..d {
                    let mut acc = 0f32;
                    let base = dd * inner;
                    for j in 0..inner {
                        acc += cx[j] * wo[base + j];
                    }
                    dst[dd] += acc;
                }
            }
        }
    }

    /// Multivariate forecast: `series[0]` is the target, `series[1..]` are past
    /// covariates. All series share one group, so the encoder's group attention
    /// lets the target attend to the covariates at every patch position (the
    /// reference's multivariate / past-covariate path with `group_ids` all equal
    /// and unknown futures). Returns the target's `[num_quantiles, horizon]`.
    ///
    /// With a single series this equals [`forecast_quantiles`] up to float noise
    /// (the group sublayer degenerates identically).
    pub fn forecast_quantiles_mv(&self, series: &[&[f32]], horizon: usize) -> Vec<f32> {
        let none: Vec<Option<&[f32]>> = vec![None; series.len()];
        self.forecast_quantiles_mv_kf(series, &none, horizon)
    }

    /// As [`forecast_quantiles_mv`], but each series may carry a **known-future**
    /// path in `futures[bi]` (values over the horizon, in raw units) — a
    /// known-future covariate. Its future patches carry the values (normalized by
    /// that series' own context scale) with the mask set; `None` means the future
    /// is unknown (the target and past-only covariates). Length of `futures` must
    /// equal `series`.
    pub fn forecast_quantiles_mv_kf(
        &self,
        series: &[&[f32]],
        futures: &[Option<&[f32]>],
        horizon: usize,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let patch = cfg.input_patch_size;
        let q = cfg.num_quantiles;
        let head_out = cfg.head_out_dim();
        let bc = series.len();
        assert!(bc >= 1, "need at least the target series");
        assert_eq!(futures.len(), bc, "futures must be parallel to series");

        let n_out = horizon.div_ceil(patch);

        // per-series: scale, patch, embed, assemble [S, D] with the REG token.
        let reg = self.w_host_row("shared.weight", cfg.reg_token_id, d);
        let mut emb_bufs: Vec<DeviceBuffer> = Vec::with_capacity(bc);
        let mut mask_bufs: Vec<DeviceBuffer> = Vec::with_capacity(bc);
        let mut ls0 = None;
        let mut s_len = 0usize;
        let mut n_ctx0 = 0usize;
        for (bi, ser) in series.iter().enumerate() {
            let (ls, scaled) = preprocess::instance_norm(ser, cfg.use_arcsinh);
            let ctx = preprocess::context_features(&scaled, patch, cfg.time_encoding_scale);
            let n_ctx = ctx.n_patches;
            let s = n_ctx + 1 + n_out;
            if bi == 0 {
                ls0 = Some(ls);
                s_len = s;
                n_ctx0 = n_ctx;
            } else {
                assert_eq!(s, s_len, "chronos2 mv: all series must share context length");
            }
            // per-series future features: known-future covariates carry their
            // (context-scaled) future values; everything else is unknown (zeros).
            let fut = match futures[bi] {
                Some(fv) => {
                    let fscaled = preprocess::instance_norm_apply(fv, ls, cfg.use_arcsinh);
                    preprocess::future_features_with_values(horizon, patch, cfg.time_encoding_scale, Some(&fscaled))
                }
                None => preprocess::future_features(horizon, patch, cfg.time_encoding_scale),
            };
            let mut feats = ctx.features.clone();
            feats.extend_from_slice(&fut.features);
            let feat_buf = self.gpu.storage_init("feats", &feats);
            let patch_emb = self.residual_block("input_patch_embedding", &feat_buf, n_ctx + n_out, 3 * patch, cfg.d_ff, d);
            let patch_emb_host = self.gpu.read(&patch_emb, (n_ctx + n_out) * d);
            let mut emb_host = Vec::with_capacity(s * d);
            emb_host.extend_from_slice(&patch_emb_host[..n_ctx * d]);
            emb_host.extend_from_slice(&reg);
            emb_host.extend_from_slice(&patch_emb_host[n_ctx * d..]);
            emb_bufs.push(self.gpu.storage_init("emb", &emb_host));
            let mut mask = vec![0.0f32; s];
            for p in 0..n_ctx {
                if ctx.attn_mask[p] == 0.0 {
                    mask[p] = -3.4e38;
                }
            }
            mask_bufs.push(self.gpu.storage_init("mask", &mask));
        }
        let s = s_len;

        // encoder blocks: time attention (per series) → group attention (across
        // series, host) → FFN (per series).
        for b in 0..cfg.num_layers {
            let pre = format!("encoder.block.{b}");
            for bi in 0..bc {
                self.time_attention(&pre, &emb_bufs[bi], s, &mask_bufs[bi]);
            }
            // read all series into one [bc, S, D] host buffer, mix, write back.
            let mut state = vec![0.0f32; bc * s * d];
            for bi in 0..bc {
                let h = self.gpu.read(&emb_bufs[bi], s * d);
                state[bi * s * d..(bi + 1) * s * d].copy_from_slice(&h);
            }
            self.group_attention_host(&pre, &mut state, bc, s);
            for bi in 0..bc {
                emb_bufs[bi] = self.gpu.storage_init("emb", &state[bi * s * d..(bi + 1) * s * d]);
            }
            for bi in 0..bc {
                self.ffn(&pre, &emb_bufs[bi], s);
            }
        }

        // target series only: final norm → head on trailing future tokens.
        let normed = self.gpu.storage((s * d) as u64);
        self.rms(&emb_bufs[0], "encoder.final_layer_norm.weight", &normed, d, s);
        let normed_host = self.gpu.read(&normed, s * d);
        let head_in = self.gpu.storage_init("head_in", &normed_host[(n_ctx0 + 1) * d..]);
        let qp = self.residual_block("output_patch_embedding", &head_in, n_out, d, cfg.d_ff, head_out);
        let qp_host = self.gpu.read(&qp, n_out * head_out);

        // rearrange b n (q p) -> [q, n*patch], denorm per quantile row (target scale).
        let hlen = n_out * patch;
        let mut out = vec![0.0f32; q * hlen];
        for n in 0..n_out {
            for qi in 0..q {
                for pp in 0..patch {
                    out[qi * hlen + n * patch + pp] = qp_host[n * head_out + qi * patch + pp];
                }
            }
        }
        let mut trimmed = vec![0.0f32; q * horizon];
        for qi in 0..q {
            let row = &out[qi * hlen..qi * hlen + hlen];
            let dn = preprocess::instance_norm_inverse(row, ls0.unwrap(), cfg.use_arcsinh);
            trimmed[qi * horizon..qi * horizon + horizon].copy_from_slice(&dn[..horizon]);
        }
        trimmed
    }

    /// Read one row of an embedding weight straight from the host copy in the
    /// param store (used for the REG token).
    fn w_host_row(&self, name: &str, row: usize, cols: usize) -> Vec<f32> {
        let full = self.gpu.read(self.w(name), self.cfg.vocab_size * cols);
        let _ = name;
        full[row * cols..(row + 1) * cols].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// With all weights zero, every linear/bias is zero, so the hidden state
    /// stays zero through the whole net and the quantile head emits zero in the
    /// standardized space. Denorm (`sinh(0)=0`, then `*scale + loc`) maps that to
    /// exactly the series mean (`loc`) for every quantile and step. This is a
    /// full end-to-end wiring test that needs no real weights.
    #[test]
    fn zero_weights_forecast_the_series_mean() {
        if skip() {
            return;
        }
        let cfg = Chronos2Config::tiny();
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        let model = Chronos2::from_weights(cfg.clone(), &weights).unwrap();

        let context: Vec<f32> = (0..40).map(|i| 3.0 + (i as f32) * 0.5).collect();
        let mean = context.iter().sum::<f32>() / context.len() as f32;
        let horizon = 6;
        let out = model.forecast_quantiles(&context, horizon);

        assert_eq!(out.len(), cfg.num_quantiles * horizon);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output {i} not finite");
            assert!((v - mean).abs() < 1e-2, "zero-weight forecast should be the mean {mean}, got {v}");
        }
    }

    /// The multivariate path with a single series must reproduce the
    /// parity-exact univariate path: its host group attention (B=1) degenerates
    /// to the same `o(v(rmsnorm(x)))` the device `group_degenerate` computes.
    /// Uses non-trivial deterministic weights so the group sublayer is exercised.
    #[test]
    fn mv_single_series_matches_univariate() {
        if skip() {
            return;
        }
        let cfg = Chronos2Config::tiny();
        // deterministic non-zero weights so every sublayer (incl. group) is active.
        let weights: HashMap<String, Vec<f32>> = cfg
            .param_list()
            .into_iter()
            .map(|(k, s)| {
                let n: usize = s.iter().product();
                let seed = k.len();
                let data = (0..n).map(|i| (((i + seed) as f32) * 0.1).sin() * 0.05).collect();
                (k, data)
            })
            .collect();
        let model = Chronos2::from_weights(cfg.clone(), &weights).unwrap();

        let context: Vec<f32> = (0..40).map(|i| 3.0 + (i as f32 * 0.3).sin()).collect();
        let horizon = 6;
        let uni = model.forecast_quantiles(&context, horizon);
        let mv = model.forecast_quantiles_mv(&[&context], horizon);
        assert_eq!(uni.len(), mv.len());
        let mut dot = 0f32;
        let mut nu = 0f32;
        let mut nm = 0f32;
        let mut worst = 0f32;
        for (a, b) in uni.iter().zip(mv.iter()) {
            dot += a * b;
            nu += a * a;
            nm += b * b;
            worst = worst.max((a - b).abs());
        }
        let cos = dot / (nu.sqrt() * nm.sqrt() + 1e-12);
        assert!(cos > 0.9999, "mv(B=1) vs univariate cosine={cos} worst={worst}");
        assert!(worst < 1e-2, "mv(B=1) vs univariate worst abs diff {worst}");
    }
}
