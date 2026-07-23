// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FinCast forward (inference) — composes preprocessing and the WGSL kernels
//! into the patched-decoder + top-2 MoE + PQ-head forecast path.
//!
//! Device work (runs on CPU or GPU via `gpu-core`): every GEMM (patch/head
//! ResidualBlocks, fused qkv, o_proj, per-expert gate/down projections), plus
//! RMSNorm / LayerNorm / SiLU / ReLU / bias / add. Host work (small,
//! data-dependent): the per-dim query scaling (`softplus(scaling)`), the causal
//! multi-head attention over the ≤`context_len/patch_len` patch tokens, and the
//! top-2 MoE gating + combine. This mirrors the reference exactly (see
//! `docs/models/fincast/status.md`), with **deterministic** top-2 routing (the parity
//! trap).
//!
//! Per decoder block on `emb [s,d]` (`s = n_patches`):
//! ```text
//! xn   = rmsnorm(emb, input_layernorm.weight)
//! qkv  = xn @ qkv_proj + bias        ; split q,k,v [s,inner]
//! q   *= scale * softplus(scaling)   ; per head_dim
//! ctx  = causal_mha(q,k,v, padmask)  ; host
//! emb += ctx @ o_proj + bias
//! p    = rmsnorm(emb, moe_prenorm.gamma)
//! g    = softmax(p @ to_gates); top-2 -> weights w (renormalized)
//! emb += p + Σ_{e∈top2} w_e * (down_proj(relu(gate_proj(layernorm_e(p)))))
//! ```
//! then `horizon_ff_layer(emb) -> [s, horizon_len*(1+num_quantiles)]`.

use crate::config::FincastConfig;
use crate::preprocess::{self, LocScale};
use gpu_core::{DeviceBuffer, Gpu};
use std::collections::HashMap;

const MATMUL: usize = 0;
const BIAS_ADD: usize = 1;
const RELU: usize = 2;
const SILU: usize = 3;
const ADD: usize = 4;
const RMSNORM: usize = 5;
const LAYERNORM: usize = 6;

// Naive `matmul` is used throughout (not the tiled GEMM): it JIT-compiles on the
// CPU backend (the tiled kernel needs work-group barriers the Cranelift JIT
// rejects), so the same forward runs on both CPU and GPU. A tiled/register-blocked
// fast path is a perf follow-up (see docs/models/fincast/status.md).
const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("relu_inplace", kernels::RELU_INPLACE),
    ("silu", kernels::SILU),
    ("add", kernels::ADD),
    ("rmsnorm", kernels::RMSNORM),
    ("layernorm", kernels::LAYERNORM),
];

#[inline]
fn softplus(x: f32) -> f32 {
    // numerically-stable ln(1+e^x)
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// A loaded FinCast model ready for inference.
pub struct Fincast {
    gpu: Gpu,
    cfg: FincastConfig,
    w: HashMap<String, DeviceBuffer>,
}

impl Fincast {
    /// Build from host-side weights (name → values), keyed by the reference
    /// `state_dict` names (see [`FincastConfig::param_list`]).
    pub fn from_weights(cfg: FincastConfig, weights: &HashMap<String, Vec<f32>>) -> Result<Fincast, String> {
        let gpu = Gpu::new(PIPELINES);
        let mut w = HashMap::new();
        for (name, shape) in cfg.param_list() {
            let numel: usize = shape.iter().product();
            let data = weights.get(&name).ok_or_else(|| format!("fincast: missing weight {name}"))?;
            if data.len() != numel {
                return Err(format!("fincast: {name} has {} elems, expected {numel}", data.len()));
            }
            w.insert(name.clone(), gpu.storage_init(&name, data));
        }
        Ok(Fincast { gpu, cfg, w })
    }

    /// Load from a brain `.weights` container (see [`crate::import`]).
    pub fn load(path: &str) -> Result<Fincast, String> {
        let c = checkpoint::load(path);
        let cfg = FincastConfig::from_json(&c.header["config"])?;
        let weights = c.by_role("");
        Fincast::from_weights(cfg, &weights)
    }

    pub fn config(&self) -> &FincastConfig {
        &self.cfg
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("fincast: weight {name} not loaded"))
    }

    // -- device op helpers (immediate submit) --------------------------------

    fn mm(&self, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: usize, k: usize, n: usize) {
        let s = self.gpu.step(MATMUL, &[x, self.w(wname), out], &[m as u32, k as u32, n as u32], (m * n) as u32);
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
    fn silu(&self, x: &DeviceBuffer, out: &DeviceBuffer, total: usize) {
        let s = self.gpu.step(SILU, &[x, out], &[total as u32], total as u32);
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
    fn layernorm(&self, x: &DeviceBuffer, gname: &str, bname: &str, out: &DeviceBuffer, d: usize, rows: usize) {
        let s = self.gpu.step(
            LAYERNORM,
            &[x, self.w(gname), self.w(bname), out],
            &[d as u32, rows as u32, gpu_core::f(self.cfg.rms_norm_eps)],
            rows as u32,
        );
        self.gpu.submit(&[], &[s]);
    }

    /// A `ResidualBlock` with a SiLU hidden nonlinearity (the FinCast form):
    /// `output(silu(hidden(x))) + residual(x)`. `hidden_layer` is a
    /// `Sequential(Linear, SiLU)` so its Linear is keyed `.hidden_layer.0`.
    fn residual_block(&self, prefix: &str, x: &DeviceBuffer, rows: usize, in_dim: usize, h: usize, out_dim: usize) -> DeviceBuffer {
        let hid = self.gpu.storage((rows * h) as u64);
        self.mm(x, &format!("{prefix}.hidden_layer.0.weight"), &hid, rows, in_dim, h);
        self.bias(&hid, &format!("{prefix}.hidden_layer.0.bias"), rows, h);
        let hact = self.gpu.storage((rows * h) as u64);
        self.silu(&hid, &hact, rows * h);

        let o1 = self.gpu.storage((rows * out_dim) as u64);
        self.mm(&hact, &format!("{prefix}.output_layer.weight"), &o1, rows, h, out_dim);
        self.bias(&o1, &format!("{prefix}.output_layer.bias"), rows, out_dim);

        let res = self.gpu.storage((rows * out_dim) as u64);
        self.mm(x, &format!("{prefix}.residual_layer.weight"), &res, rows, in_dim, out_dim);
        self.bias(&res, &format!("{prefix}.residual_layer.bias"), rows, out_dim);

        self.add(&o1, &res, rows * out_dim);
        res
    }

    /// One expert's MLP (WITHOUT the internal `+x` residual, which the caller
    /// folds into the weighted combine): `down(relu(gate(layernorm(x))))`.
    fn expert_mlp(&self, e: usize, layer: usize, p: &DeviceBuffer, s: usize) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        let pre = format!("stacked_transformer.layers.{layer}.moe.moe.experts.experts.{e}");
        let ln = self.gpu.storage((s * d) as u64);
        self.layernorm(p, &format!("{pre}.layer_norm.weight"), &format!("{pre}.layer_norm.bias"), &ln, d, s);
        let g = self.gpu.storage((s * d) as u64);
        self.mm(&ln, &format!("{pre}.gate_proj.weight"), &g, s, d, d);
        self.bias(&g, &format!("{pre}.gate_proj.bias"), s, d);
        self.relu(&g, s * d);
        let out = self.gpu.storage((s * d) as u64);
        self.mm(&g, &format!("{pre}.down_proj.weight"), &out, s, d, d);
        self.bias(&out, &format!("{pre}.down_proj.bias"), s, d);
        self.gpu.read(&out, s * d)
    }

    /// The transformer core + head: assembled token embeddings `emb [s,d]` and a
    /// per-patch padding mask `padmask[s]` (1.0 = padded) → the horizon-head
    /// output `[s, head_out]`. Builds the additive causal+padding `[s,s]` mask and
    /// delegates to [`core_forward_amask`](Self::core_forward_amask).
    pub fn core_forward(&self, emb: &[f32], padmask: &[f32]) -> Vec<f32> {
        let s = padmask.len();
        self.core_forward_amask(emb, &Self::causal_amask(padmask, s))
    }

    /// The additive `[s,s]` attention mask for a per-patch padding mask: `0` where
    /// key `j<=i` and unpadded, a large negative otherwise. This is exactly the
    /// `amask` the NPU graph consumes.
    pub fn causal_amask(padmask: &[f32], s: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i || padmask[j] > 0.5 {
                    m[i * s + j] = -1.0e9;
                }
            }
        }
        m
    }

    /// The transformer core + head over an explicit additive `[s,s]` mask. This is
    /// the parity reference for the NPU export (`emb`+`amask` → `qhead`).
    pub fn core_forward_amask(&self, emb: &[f32], amask: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.hidden_size;
        let s = (amask.len() as f64).sqrt() as usize;
        assert_eq!(amask.len(), s * s, "amask must be [s, s]");
        assert_eq!(emb.len(), s * d, "emb must be [s, d]");
        // MoE strategy: the default gathers each expert's routed tokens and runs
        // the expert MLP on just that subset (top_n/num_experts of the dense work,
        // bit-identical because every expert op is row-independent). Set
        // `FINCAST_MOE_DENSE=1` to force the reference dense-compute-then-mask path
        // (used by the parity test).
        let dense_moe = std::env::var("FINCAST_MOE_DENSE").map(|v| v != "0").unwrap_or(false);
        let emb_buf = self.gpu.storage_init("emb", emb);
        for b in 0..cfg.num_layers {
            self.block(b, &emb_buf, s, amask, dense_moe);
        }
        let hz = self.residual_block("horizon_ff_layer", &emb_buf, s, d, cfg.intermediate_size, cfg.head_out_dim());
        self.gpu.read(&hz, s * cfg.head_out_dim())
    }

    /// One decoder block, in place on `emb [s,d]`. `amask` is the additive `[s,s]`
    /// attention mask.
    fn block(&self, b: usize, emb: &DeviceBuffer, s: usize, amask: &[f32], dense_moe: bool) {
        let cfg = &self.cfg;
        let d = cfg.hidden_size;
        let inner = cfg.inner_dim();
        let pre = format!("stacked_transformer.layers.{b}");

        // -- attention --
        let xn = self.gpu.storage((s * d) as u64);
        self.rms(emb, &format!("{pre}.input_layernorm.weight"), &xn, d, s);
        let qkv = self.gpu.storage((s * cfg.qkv_dim()) as u64);
        self.mm(&xn, &format!("{pre}.self_attn.qkv_proj.weight"), &qkv, s, d, cfg.qkv_dim());
        self.bias(&qkv, &format!("{pre}.self_attn.qkv_proj.bias"), s, cfg.qkv_dim());
        let qkv_host = self.gpu.read(&qkv, s * cfg.qkv_dim());
        let ctx = self.host_attention(b, &qkv_host, s, amask);
        let ctx_buf = self.gpu.storage_init("ctx", &ctx);
        let o = self.gpu.storage((s * d) as u64);
        self.mm(&ctx_buf, &format!("{pre}.self_attn.o_proj.weight"), &o, s, inner, d);
        self.bias(&o, &format!("{pre}.self_attn.o_proj.bias"), s, d);
        self.add(&o, emb, s * d); // residual

        // -- MoE --
        let p = self.gpu.storage((s * d) as u64);
        self.rms(emb, &format!("{pre}.moe.moe_prenorm.gamma"), &p, d, s);
        let p_host = self.gpu.read(&p, s * d);
        // gate logits [s, E]
        let e = cfg.num_experts;
        let glog = self.gpu.storage((s * e) as u64);
        self.mm(&p, &format!("{pre}.moe.moe.gate.to_gates.weight"), &glog, s, d, e);
        let glog_host = self.gpu.read(&glog, s * e);
        let weights = self.gate_weights(&glog_host, s, e);
        // expert compute + weighted combine on host. moe_out starts at p (experts'
        // internal +x residual, weights sum to 1 over the top-n).
        let mut combine = p_host.clone();
        for ei in 0..e {
            // routed tokens for this expert; skip the expert entirely if none.
            let rows: Vec<usize> = (0..s).filter(|&t| weights[t * e + ei] != 0.0).collect();
            if rows.is_empty() {
                continue;
            }
            if dense_moe {
                // reference path: compute the expert MLP over ALL s tokens, then
                // mask on combine — computes num_experts×s token-MLPs.
                let mlp = self.expert_mlp(ei, b, &p, s);
                for &t in &rows {
                    let wt = weights[t * e + ei];
                    for c in 0..d {
                        combine[t * d + c] += wt * mlp[t * d + c];
                    }
                }
            } else {
                // gather the routed tokens, run the expert on just that subset,
                // scatter the weighted result back. Every op in `expert_mlp`
                // (layernorm / GEMM / ReLU / bias) is row-independent, so this is
                // bit-identical to the dense path but does only top_n×s token-MLPs
                // across the whole layer instead of num_experts×s.
                let ne = rows.len();
                let mut gathered = vec![0.0f32; ne * d];
                for (r, &t) in rows.iter().enumerate() {
                    gathered[r * d..r * d + d].copy_from_slice(&p_host[t * d..t * d + d]);
                }
                let gbuf = self.gpu.storage_init("moe_expert_in", &gathered);
                let mlp = self.expert_mlp(ei, b, &gbuf, ne); // [ne, d]
                for (r, &t) in rows.iter().enumerate() {
                    let wt = weights[t * e + ei];
                    for c in 0..d {
                        combine[t * d + c] += wt * mlp[r * d + c];
                    }
                }
            }
        }
        // block_out = combine + emb (outer residual)
        let combine_buf = self.gpu.storage_init("moe_out", &combine);
        self.add(&combine_buf, emb, s * d);
    }

    /// Per-token top-2 gating weights `[s, E]` (renormalized to sum 1 over the
    /// top-2), deterministic. Softmax over experts, take the two largest.
    fn gate_weights(&self, glog: &[f32], s: usize, e: usize) -> Vec<f32> {
        let top = self.cfg.gating_top_n.min(e);
        let mut w = vec![0.0f32; s * e];
        for t in 0..s {
            let row = &glog[t * e..t * e + e];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - mx).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / denom).collect();
            // indices of the top-`top` probs
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let sel = &idx[..top];
            let ssum: f32 = sel.iter().map(|&i| probs[i]).sum::<f32>().max(1e-9);
            for &i in sel {
                w[t * e + i] = probs[i] / ssum;
            }
        }
        w
    }

    /// Causal multi-head attention over the patch tokens, on the host. `qkv` is
    /// `[s, qkv_dim]` (fused, `[q | k | v]`); `amask` is the additive `[s,s]` mask
    /// (`0` attend, large-negative forbid). Returns `ctx [s, inner]`. Applies the
    /// per-dim query scaling `scale * softplus(scaling)`.
    fn host_attention(&self, layer: usize, qkv: &[f32], s: usize, amask: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let heads = cfg.num_heads;
        let hd = cfg.head_dim;
        let inner = cfg.inner_dim();
        let qkvd = cfg.qkv_dim();
        let scaling = self.gpu.read(self.w(&format!("stacked_transformer.layers.{layer}.self_attn.scaling")), hd);
        let base = 1.442695041f32 / (hd as f32).sqrt();
        let qscale: Vec<f32> = (0..hd).map(|dd| base * softplus(scaling[dd])).collect();

        let q_off = 0usize;
        let k_off = inner;
        let v_off = 2 * inner; // num_kv_heads == num_heads -> kv_size == inner

        let mut ctx = vec![0.0f32; s * inner];
        for h in 0..heads {
            let ho = h * hd;
            for i in 0..s {
                let mut sc = vec![0.0f32; s];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..s {
                    let mut dot = amask[i * s + j];
                    for dd in 0..hd {
                        let qv = qkv[i * qkvd + q_off + ho + dd] * qscale[dd];
                        dot += qv * qkv[j * qkvd + k_off + ho + dd];
                    }
                    sc[j] = dot;
                    mx = mx.max(dot);
                }
                if mx <= -3.0e38 {
                    continue; // all keys masked
                }
                let mut sum = 0.0f32;
                for j in 0..s {
                    sc[j] = (sc[j] - mx).exp();
                    sum += sc[j];
                }
                let inv = 1.0 / sum;
                for j in 0..s {
                    let pw = sc[j] * inv;
                    for dd in 0..hd {
                        ctx[i * inner + ho + dd] += pw * qkv[j * qkvd + v_off + ho + dd];
                    }
                }
            }
        }
        ctx
    }

    /// Assemble the patch-embedded token sequence for one context: preprocess →
    /// `input_ff_layer` ResidualBlock → `+ freq_emb[freq]`. Returns
    /// `(emb[s,d], padmask[s], loc_scale)`.
    fn assemble(&self, context: &[f32], freq: usize) -> (Vec<f32>, Vec<f32>, LocScale) {
        let cfg = &self.cfg;
        let d = cfg.hidden_size;
        let pp = preprocess::preprocess(cfg, context);
        let s = pp.n_patches;
        let feat_buf = self.gpu.storage_init("feat", &pp.features);
        let emb_buf = self.residual_block("input_ff_layer", &feat_buf, s, cfg.patch_feat_dim(), cfg.intermediate_size, d);
        let mut emb = self.gpu.read(&emb_buf, s * d);
        // + freq_emb[freq] broadcast over tokens
        let femb = self.gpu.read(self.w("freq_emb.weight"), 3 * d);
        let fr = freq.min(2);
        for t in 0..s {
            for c in 0..d {
                emb[t * d + c] += femb[fr * d + c];
            }
        }
        (emb, pp.patch_padding, pp.loc_scale)
    }

    /// Forecast `[horizon, 1+num_quantiles]` (mean + 9 quantiles, step-major) for
    /// one context series and frequency bucket (0 high / 1 med / 2 low). Runs the
    /// reference autoregressive decode with `output_patch_len == horizon_len`,
    /// feeding the mean forecast back for horizons beyond one patch. The
    /// transformer core runs on this model's device via [`core_forward_amask`](Self::core_forward_amask).
    pub fn forecast_full(&self, context: &[f32], freq: usize, horizon: usize) -> Vec<f32> {
        self.forecast_full_with_core(context, freq, horizon, |emb, amask| self.core_forward_amask(emb, amask))
    }

    /// As [`forecast_full`](Self::forecast_full), but the transformer core runs
    /// through a pluggable `core(emb[s,d], amask[s,s]) -> qhead[s, head_out]` — swap
    /// in an NPU-backed core (the exported ONNX graph) or reuse the device core.
    pub fn forecast_full_with_core<F>(&self, context: &[f32], freq: usize, horizon: usize, core: F) -> Vec<f32>
    where
        F: Fn(&[f32], &[f32]) -> Vec<f32>,
    {
        let cfg = &self.cfg;
        let no = cfg.num_outputs();
        let hlen = cfg.horizon_len;
        let mut series = context.to_vec();
        let mut out: Vec<f32> = Vec::with_capacity(horizon * no);
        let n_steps = horizon.div_ceil(hlen);
        for _ in 0..n_steps {
            let (emb, padmask, ls) = self.assemble(&series, freq);
            let s = padmask.len();
            let amask = Self::causal_amask(&padmask, s);
            let head = core(&emb, &amask); // [s, head_out]
            let head_out = cfg.head_out_dim();
            // last patch row -> [horizon_len, num_outputs]; reference layout is
            // view(n, horizon_len, num_outputs) then reverse-transform.
            let last = &head[(s - 1) * head_out..s * head_out];
            for hstep in 0..hlen {
                for k in 0..no {
                    out.push(preprocess::denorm(last[hstep * no + k], ls));
                }
            }
            // feed the mean forecast (index 0) back for the next AR step
            for hstep in 0..hlen {
                series.push(preprocess::denorm(last[hstep * no], ls));
            }
        }
        out.truncate(horizon * no);
        out
    }

    /// Convenience: the native quantile matrix `[num_quantiles, horizon]`
    /// (quantile-major), dropping the mean channel. Used by the forecaster.
    pub fn forecast_quantiles(&self, context: &[f32], freq: usize, horizon: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let no = cfg.num_outputs();
        let q = cfg.num_quantiles;
        let full = self.forecast_full(context, freq, horizon); // [horizon, no]
        let mut out = vec![0.0f32; q * horizon];
        for t in 0..horizon {
            for qi in 0..q {
                out[qi * horizon + t] = full[t * no + 1 + qi];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// With all weights zero, every linear/bias/gate is zero. The MoE gate logits
    /// are all zero → uniform softmax → top-2 weights sum to 1 but each expert MLP
    /// is zero, so the MoE contributes only its prenorm residual, and the horizon
    /// head emits zero in standardized space. Denorm (`*sigma + mu`) maps that to
    /// exactly the series' first-patch mean for every output. Full end-to-end
    /// wiring test needing no real weights.
    #[test]
    fn zero_weights_forecast_the_mean() {
        if skip() {
            return;
        }
        let cfg = FincastConfig::tiny();
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        let model = Fincast::from_weights(cfg.clone(), &weights).unwrap();

        let context: Vec<f32> = (0..32).map(|i| 3.0 + (i as f32) * 0.5).collect();
        // reference standardizes by the FIRST patch -> denorm target is that
        // patch's mean (mu).
        let pp = preprocess::preprocess(&cfg, &context);
        let mu = pp.loc_scale.mu;
        let horizon = 4;
        let out = model.forecast_full(&context, 0, horizon);
        assert_eq!(out.len(), horizon * cfg.num_outputs());
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output {i} not finite");
            assert!((v - mu).abs() < 1e-2, "zero-weight forecast should be mu={mu}, got {v}");
        }
    }

    /// `gate_weights` produces a valid deterministic top-2 distribution: exactly
    /// `top_n` non-zero entries per token, summing to 1, picking the largest
    /// logits.
    #[test]
    fn gate_weights_are_top2_and_normalized() {
        if skip() {
            return;
        }
        let cfg = FincastConfig::tiny();
        let weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![0.0; s.iter().product()])).collect();
        let model = Fincast::from_weights(cfg.clone(), &weights).unwrap();
        // logits for 2 tokens, 3 experts
        let glog = vec![3.0, 1.0, 2.0, /*t1*/ 0.0, 5.0, 4.0];
        let w = model.gate_weights(&glog, 2, 3);
        // token0: experts 0 (3.0) and 2 (2.0) selected, expert1 = 0
        assert_eq!(w[1], 0.0);
        assert!(w[0] > 0.0 && w[2] > 0.0);
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-5);
        // token1: experts 1 (5.0) and 2 (4.0)
        assert_eq!(w[3], 0.0);
        assert!((w[4] + w[5] - 1.0).abs() < 1e-5);
        assert!(w[4] > w[5], "higher logit gets more weight");
    }

    /// The default gather/scatter MoE (compute only each expert's routed tokens)
    /// is bit-identical to the reference dense-compute-then-mask path. Gates the
    /// speed optimization: same math, less work.
    #[test]
    fn moe_gather_scatter_matches_dense() {
        if skip() {
            return;
        }
        let cfg = FincastConfig::tiny();
        // Varied non-zero weights so per-token gating selects different experts,
        // exercising the gather/scatter across tokens (not just the skip path).
        let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        for (k, shp) in cfg.param_list() {
            let n: usize = shp.iter().product();
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                h ^= h << 13;
                h ^= h >> 7;
                h ^= h << 17;
                v.push(((h >> 40) as f32 / 16_777_216.0 - 0.5) * 0.1);
            }
            weights.insert(k, v);
        }
        let model = Fincast::from_weights(cfg.clone(), &weights).unwrap();
        let context: Vec<f32> = (0..40).map(|i| 10.0 + (i as f32 * 0.3).sin() * 2.0).collect();
        let horizon = 6;

        std::env::set_var("FINCAST_MOE_DENSE", "1");
        let dense = model.forecast_full(&context, 0, horizon);
        std::env::set_var("FINCAST_MOE_DENSE", "0");
        let sparse = model.forecast_full(&context, 0, horizon);
        std::env::remove_var("FINCAST_MOE_DENSE");

        assert_eq!(dense.len(), sparse.len());
        let worst = dense.iter().zip(&sparse).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "gather/scatter MoE diverged from dense: worst abs diff {worst}");
    }
}
