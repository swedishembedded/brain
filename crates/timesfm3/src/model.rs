// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! TimesFM-3 device forward: the stacked mixing transformer core, from a
//! patched-and-embedded input (`pre_transformer_resblock`'s output boundary)
//! through `num_layers` mixing blocks to the raw (pre-RevIN-reverse) output
//! head logits. Preprocessing (patching, RevIN, CPM refinement, linear
//! detrending, stitching) is host-side math in `preprocess.rs`, the same
//! split `fincast`/`chronos2` use.
//!
//! # The attention scale is folded into the query projection, not applied here
//!
//! The reference computes attention as `softmax(PerDimScale(RMSNorm(q)) ·
//! RMSNorm(k)ᵀ)`, where `PerDimScale(x) = x * 1.442695041 / sqrt(head_dim) *
//! softplus(w)` and the score itself is additionally scaled by `sqrt(head_dim)`
//! (`use_memory_efficient_attention=true` → `rescale_logits=false` → SDPA is
//! called with `scale=sqrt(head_dim)`, NOT the usual `1/sqrt(head_dim)`).
//! Multiplying those together, the `sqrt(head_dim)` terms cancel exactly and
//! every per-logit factor collapses to `1.442695041 * softplus(w)` - a plain
//! per-head-dim elementwise scale with NO dependence on `head_dim` at all.
//!
//! RMS normalization commutes with a subsequent elementwise gain multiply
//! (`x/rms(x) * g * s == x/rms(x) * (g*s)`), so this is folded once, on the
//! host, into an EFFECTIVE query gain (`query_ln.weight * 1.442695041 *
//! softplus(per_dim_scale)`) at model-load time, and the query's own QK-norm
//! dispatch uses that instead of `query_ln.weight` alone. The attention-scores
//! kernel is then called with `scale=1.0` and no separate scale parameter
//! exists anywhere downstream to get wrong - the highest-risk detail in this
//! port (lesson: cosine similarity cannot see a dropped/doubled scale factor)
//! is eliminated by construction rather than gated after the fact.
//!
//! Forward-only fold: a later backward must differentiate the UNFOLDED
//! reference form (RMSNorm, then a separate elementwise `softplus`-scale
//! multiply) rather than back-propagate through this fused gain, per the
//! usual "exploit structure, but differentiate the unfolded form" rule.
//!
//! # Layout
//!
//! Every per-token buffer this core produces is `[rows, d]` row-major, never a
//! fused qkv - the one new kernel this port added
//! (`attn_scores_qk_kmask`/see [`kernels::ATTN_SCORES_QK_KMASK`]) and the
//! existing `attn_softmax_full`/`attn_apply_full` all read separate q/k/v
//! buffers directly, so nothing needs assembling into one strided buffer.
//! `rows` is `b*v*n` in (batch, variate, patch-position) order for sequence
//! attention (patch-position innermost, so `pos = row % n` gives the right
//! sequence position); variate attention needs variates contiguous instead,
//! so its q/k/v (and the attention output) are permuted to `(b, n, v)` order
//! and back via [`kernels::SWAP_AXES12_VEC`] - the one new kernel besides the
//! scores kernel this port needed.

use crate::config::Timesfm3Config;
use gpu_core::{f, DeviceBuffer, Gpu};
use model::block;
use std::collections::HashMap;

const MATMUL: usize = 0;
const MATMUL_REG3: usize = 1;
const BIAS_ADD: usize = 2;
const RELU: usize = 3;
const ADD: usize = 4;
const RMSNORM: usize = 5;
const RMSNORM_ROWS: usize = 6;
const ROPE_PARTIAL: usize = 7;
const ATTN_SCORES_QK_KMASK: usize = 8;
const ATTN_SOFTMAX_FULL: usize = 9;
const ATTN_APPLY_FULL: usize = 10;
const SWAP_AXES12_VEC: usize = 11;
const SOFTMAX_ROWS: usize = 12;

pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("bias_add", kernels::BIAS_ADD),
    ("relu_inplace", kernels::RELU_INPLACE),
    ("add_inplace", kernels::ADD_INPLACE),
    ("rmsnorm", kernels::RMSNORM),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rope_partial", kernels::ROPE_PARTIAL),
    ("attn_scores_qk_kmask", kernels::ATTN_SCORES_QK_KMASK),
    ("attn_softmax_full", kernels::ATTN_SOFTMAX_FULL),
    ("attn_apply_full", kernels::ATTN_APPLY_FULL),
    ("swap_axes12_vec", kernels::SWAP_AXES12_VEC),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
];

/// RoPE base (theta) the reference hardcodes for sequence-attention rotation.
const ROPE_THETA: f32 = 10000.0;

/// A masked-out key's additive score contribution, matching the reference's
/// own manual (non-SDPA) masked-attention constant exactly (`model.py`'s
/// `-1e9` fill, not `attn_scores_qk_kmask`'s own `-3.4e38` causal constant -
/// the two are numerically indistinguishable after softmax in f32, but this
/// is the value that actually reproduces the reference's masking path).
const MASK_NEG: f32 = -1.0e9;

/// A loaded TimesFM-3 model ready for inference.
pub struct Timesfm3 {
    gpu: Gpu,
    cfg: Timesfm3Config,
    w: HashMap<String, DeviceBuffer>,
}

impl Timesfm3 {
    /// Build from host-side weights (name -> values), keyed by the reference
    /// `state_dict` names (see [`Timesfm3Config::param_list`]). Missing
    /// weights are a hard error. The query gain fold (see the module docs)
    /// happens here, once, rather than per forward call.
    pub fn from_weights(cfg: Timesfm3Config, weights: &HashMap<String, Vec<f32>>) -> Result<Timesfm3, String> {
        Timesfm3::from_weights_on(Gpu::new(PIPELINES), cfg, weights)
    }

    pub fn from_weights_on(gpu: Gpu, cfg: Timesfm3Config, weights: &HashMap<String, Vec<f32>>) -> Result<Timesfm3, String> {
        let mut w = HashMap::new();
        for (name, shape) in cfg.param_list() {
            let numel: usize = shape.iter().product();
            let data = weights.get(&name).ok_or_else(|| format!("timesfm3: missing weight {name}"))?;
            if data.len() != numel {
                return Err(format!("timesfm3: {name} has {} elems, expected {numel}", data.len()));
            }
            if name.ends_with(".query_ln.weight") {
                // Fold PerDimScale into the effective query gain (see module
                // docs). `{seq,var}_attn.query_ln.weight` and
                // `{seq,var}_attn.per_dim_scale.per_dim_scale` are always a
                // matched pair at this point in `param_list()`'s naming.
                let scale_name = name.replace(".query_ln.weight", ".per_dim_scale.per_dim_scale");
                let scale = weights.get(&scale_name).ok_or_else(|| format!("timesfm3: missing {scale_name}"))?;
                let folded: Vec<f32> = data.iter().zip(scale).map(|(&g, &s)| g * 1.442_695_1 * softplus(s)).collect();
                w.insert(name.clone(), gpu.storage_init(&name, &folded));
            } else if name.ends_with(".per_dim_scale.per_dim_scale") {
                continue; // consumed above, folded into the paired query_ln - not a separate device buffer.
            } else {
                w.insert(name.clone(), gpu.storage_init(&name, data));
            }
        }
        Ok(Timesfm3 { gpu, cfg, w })
    }

    /// Load a model from EITHER a brain `.safetensors` container (see
    /// [`crate::import::import`]) or a raw fetched checkpoint directory
    /// (`brain pull google/timesfm-3.0-pytorch`'s own output - `config.json`
    /// plus `model.safetensors`, never converted). A directory is always the
    /// raw form and a brain container is always a single file - `import`'s
    /// own output convention - so that alone tells the two apart, the same
    /// way `kronos::import::load_decoder` accepts both without the caller
    /// naming which one it has. This is what makes `brain pull` +
    /// `BRAIN_TIMESFM3=<fetched dir>` work with no manual `brain forecast
    /// import` step in between.
    pub fn load(path: &str) -> Result<Timesfm3, String> {
        Timesfm3::load_on(Gpu::new(PIPELINES), path)
    }

    pub fn load_on(gpu: Gpu, path: &str) -> Result<Timesfm3, String> {
        if std::path::Path::new(path).is_dir() {
            let cfg = crate::import::load_config(path)?;
            let weights = crate::import::load_hf(&cfg, path)?;
            return Timesfm3::from_weights_on(gpu, cfg, &weights);
        }
        let c = checkpoint::load(path);
        let cfg = Timesfm3Config::from_json(&c.header["config"])?;
        let weights = c.by_role("");
        Timesfm3::from_weights_on(gpu, cfg, &weights)
    }

    pub fn config(&self) -> &Timesfm3Config {
        &self.cfg
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("timesfm3: unbound weight {name}"))
    }

    fn gemm(&self, m: usize, n: usize) -> (usize, u32) {
        block::pick_gemm(m, n, MATMUL, MATMUL_REG3, false)
    }

    /// `out[m,n] = x[m,k] @ w[n,k]^T` (`w` is a PyTorch `nn.Linear` weight,
    /// `[out_features, in_features]`), no bias.
    fn linear(&self, x: &DeviceBuffer, weight_name: &str, out: &DeviceBuffer, m: usize, k: usize, n: usize) -> gpu_core::Step {
        let (kind, threads) = self.gemm(m, n);
        self.gpu.step(kind, &[x, self.w(weight_name), out], &[m as u32, k as u32, n as u32], threads)
    }

    /// `rms_variant` picks between the reference (one thread per row) and the
    /// cooperative `rmsnorm_rows` (one WORKGROUP of 64 per row) kernel; the two
    /// need DIFFERENT dispatch thread counts (`rows` vs `rows*64`), which is
    /// exactly what `rms_variant` returns alongside the kind - `block::
    /// rmsnorm_eps_fwd` hardcodes `rows` regardless of kind (it is meant for
    /// the reference kernel only), so the step is built directly here instead
    /// of through that helper, using the thread count `rms_variant` actually
    /// picked for the selected kernel.
    fn rmsnorm(&self, x: &DeviceBuffer, weight: &DeviceBuffer, out: &DeviceBuffer, dim: usize, rows: usize) -> gpu_core::Step {
        let coop = Some(RMSNORM_ROWS);
        let (kind, threads) = block::rms_variant(&self.gpu, RMSNORM, coop, rows as u32, dim as u32);
        self.gpu.step(kind, &[x, weight, out], &[dim as u32, rows as u32, f(self.cfg.rms_norm_eps)], threads)
    }

    fn rmsnorm_named(&self, x: &DeviceBuffer, weight_name: &str, out: &DeviceBuffer, dim: usize, rows: usize) -> gpu_core::Step {
        self.rmsnorm(x, self.w(weight_name), out, dim, rows)
    }

    fn add_inplace(&self, dst: &DeviceBuffer, addend: &DeviceBuffer, total: usize) -> gpu_core::Step {
        self.gpu.step(ADD, &[dst, addend], &[total as u32], total as u32)
    }

    fn relu_inplace(&self, x: &DeviceBuffer, total: usize) -> gpu_core::Step {
        self.gpu.step(RELU, &[x], &[total as u32], total as u32)
    }

    /// RoPE, NeoX/half-split, applied to the full `head_dim` (`rot_dim ==
    /// head_dim`). `rows = b*v*n`, `tcols = n` - `rope_partial`'s own
    /// `pos = row % tcols` is exactly the "position resets every `n` rows"
    /// semantics one dispatch needs to cover every (batch, variate) group at
    /// once, since rows are laid out (b, v, n) with n innermost.
    fn rope_seq(&self, buf: &DeviceBuffer, rows: usize, tcols: usize) -> gpu_core::Step {
        let hd = self.cfg.head_dim as u32;
        self.gpu.step(
            ROPE_PARTIAL,
            &[buf],
            &[rows as u32, self.cfg.num_heads as u32, hd, hd * self.cfg.num_heads as u32, 0, tcols as u32, f(ROPE_THETA), hd],
            rows as u32 * self.cfg.num_heads as u32 * (hd / 2),
        )
    }

    /// One attention sublayer's scores->softmax->apply, separate q/k/v
    /// buffers throughout (see module docs), `scale=1.0` always (folded into
    /// `q` ahead of this call). Softmax goes through `backend_api::select`'s
    /// `Op::Softmax` (`block::softmax_variant`, the same seam `wan`/`ltxv`
    /// adopted) instead of a fixed `attn_softmax_full` dispatch, so this
    /// model gets the cooperative `softmax_rows` kernel wherever the device
    /// supports workgroup reductions.
    fn attention(&self, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, kmask: &DeviceBuffer, ctx: &DeviceBuffer, bsz: usize, tcols: usize, causal: bool) -> Vec<gpu_core::Step> {
        let (h, hd, d) = (self.cfg.num_heads, self.cfg.head_dim, self.cfg.model_dims);
        let scores = self.gpu.storage((bsz * h * tcols * tcols) as u64);
        let probs = self.gpu.storage((bsz * h * tcols * tcols) as u64);
        let (sk, st) = block::softmax_variant(&self.gpu, ATTN_SOFTMAX_FULL, Some(SOFTMAX_ROWS), (bsz * h * tcols) as u32, tcols as u32);
        let softmax_step = if sk == SOFTMAX_ROWS {
            self.gpu.step(sk, &[&scores, &probs], &[(bsz * h * tcols) as u32, tcols as u32], st)
        } else {
            self.gpu.step(sk, &[&scores, &probs], &[bsz as u32, h as u32, tcols as u32], st)
        };
        vec![
            self.gpu.step(
                ATTN_SCORES_QK_KMASK,
                &[q, k, kmask, &scores],
                &[bsz as u32, h as u32, tcols as u32, hd as u32, d as u32, causal as u32, f(1.0)],
                (bsz * h * tcols * tcols) as u32,
            ),
            softmax_step,
            self.gpu.step(ATTN_APPLY_FULL, &[&probs, v, ctx], &[bsz as u32, h as u32, tcols as u32, hd as u32, d as u32, d as u32], (bsz * h * tcols * hd) as u32),
        ]
    }

    /// `[b,a1,a2,d] -> [b,a2,a1,d]` (see module docs - the sequence<->variate
    /// axis swap variate attention needs on both sides of its attention call).
    fn swap12(&self, src: &DeviceBuffer, dst: &DeviceBuffer, a0: usize, a1: usize, a2: usize, d: usize) -> gpu_core::Step {
        self.gpu.step(SWAP_AXES12_VEC, &[src, dst], &[a0 as u32, a1 as u32, a2 as u32, d as u32], (a0 * a1 * a2 * d) as u32)
    }

    /// The core forward: `pre_transformer_resblock`'s input through the
    /// stacked mixing transformer to the RAW output-head logits (before
    /// RevIN-reverse, before CPM refinement substitutes the horizon's
    /// running stats, before stitching - all host-side postprocessing in
    /// `preprocess.rs`).
    ///
    /// `resblock_input`: `[b*v*n, resblock_in_dim]`, `patch_mask`: `[b, v, n]`
    /// (`true` = masked/invalid - the reference's own convention, inverted
    /// from "attend"). Returns `[b*v*n, output_patch_len*num_quantiles]` raw
    /// logits.
    pub fn core_forward(&self, resblock_input: &[f32], patch_mask: &[bool], b: usize, v: usize, n: usize) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.model_dims;
        let rows = b * v * n;
        let g = &self.gpu;
        let mut steps = Vec::new();

        // ---- pre_transformer_resblock: out = output(relu(hidden(x))) + residual(x) ----
        let x = g.storage_init("resblock_input", resblock_input);
        let hidden = g.storage((rows * d) as u64);
        steps.push(self.linear(&x, "pre_transformer_resblock.hidden_layer.weight", &hidden, rows, c.resblock_in_dim(), d));
        steps.push(self.relu_inplace(&hidden, rows * d));
        let mut h = g.storage((rows * d) as u64);
        steps.push(self.linear(&hidden, "pre_transformer_resblock.output_layer.weight", &h, rows, d, d));
        let resid = g.storage((rows * d) as u64);
        steps.push(self.linear(&x, "pre_transformer_resblock.residual_layer.weight", &resid, rows, c.resblock_in_dim(), d));
        steps.push(self.add_inplace(&h, &resid, rows * d));

        // Additive key masks, both layouts, built once and reused by every
        // layer (matches the reference: computed once outside the layer
        // loop). Small host arrays - cheap regardless of context length.
        let seq_kmask: Vec<f32> = patch_mask.iter().map(|&m| if m { MASK_NEG } else { 0.0 }).collect();
        let mut var_kmask = vec![0.0f32; b * n * v];
        for bi in 0..b {
            for vi in 0..v {
                for ni in 0..n {
                    if patch_mask[(bi * v + vi) * n + ni] {
                        var_kmask[(bi * n + ni) * v + vi] = MASK_NEG;
                    }
                }
            }
        }
        let seq_kmask = g.storage_init("seq_kmask", &seq_kmask);
        let var_kmask = g.storage_init("var_kmask", &var_kmask);

        for l in 0..c.num_layers {
            let p = format!("transformer_stack.layers.{l}");

            // ---- sequence attention: causal, per-(b,v) sequence over n, RoPE ----
            let seq_in = g.storage((rows * d) as u64);
            steps.push(self.rmsnorm_named(&h, &format!("{p}.pre_seq_attn_ln.weight"), &seq_in, d, rows));
            let (q, k, vv) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(self.linear(&seq_in, &format!("{p}.seq_attn.query_proj.weight"), &q, rows, d, d));
            steps.push(self.linear(&seq_in, &format!("{p}.seq_attn.key_proj.weight"), &k, rows, d, d));
            steps.push(self.linear(&seq_in, &format!("{p}.seq_attn.value_proj.weight"), &vv, rows, d, d));
            steps.push(self.rope_seq(&q, rows, n));
            steps.push(self.rope_seq(&k, rows, n));
            let (qn, kn) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(self.rmsnorm_named(&q, &format!("{p}.seq_attn.query_ln.weight"), &qn, c.head_dim, rows * c.num_heads));
            steps.push(self.rmsnorm_named(&k, &format!("{p}.seq_attn.key_ln.weight"), &kn, c.head_dim, rows * c.num_heads));
            let ctx = g.storage((rows * d) as u64);
            steps.extend(self.attention(&qn, &kn, &vv, &seq_kmask, &ctx, b * v, n, true));
            let seq_out = g.storage((rows * d) as u64);
            steps.push(self.linear(&ctx, &format!("{p}.seq_attn.out_proj.weight"), &seq_out, rows, d, d));
            let seq_normed = g.storage((rows * d) as u64);
            steps.push(self.rmsnorm_named(&seq_out, &format!("{p}.post_seq_attn_ln.weight"), &seq_normed, d, rows));
            steps.push(self.add_inplace(&seq_normed, &h, rows * d));
            let h1 = seq_normed;

            // ---- variate attention: non-causal, per-(b,position) over v, no RoPE ----
            let var_in = g.storage((rows * d) as u64);
            steps.push(self.rmsnorm_named(&h1, &format!("{p}.pre_var_attn_ln.weight"), &var_in, d, rows));
            let (q2, k2, v2) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(self.linear(&var_in, &format!("{p}.var_attn.query_proj.weight"), &q2, rows, d, d));
            steps.push(self.linear(&var_in, &format!("{p}.var_attn.key_proj.weight"), &k2, rows, d, d));
            steps.push(self.linear(&var_in, &format!("{p}.var_attn.value_proj.weight"), &v2, rows, d, d));
            let (q2n, k2n) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(self.rmsnorm_named(&q2, &format!("{p}.var_attn.query_ln.weight"), &q2n, c.head_dim, rows * c.num_heads));
            steps.push(self.rmsnorm_named(&k2, &format!("{p}.var_attn.key_ln.weight"), &k2n, c.head_dim, rows * c.num_heads));
            let (q2t, k2t, v2t) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            steps.push(self.swap12(&q2n, &q2t, b, v, n, d));
            steps.push(self.swap12(&k2n, &k2t, b, v, n, d));
            steps.push(self.swap12(&v2, &v2t, b, v, n, d));
            let ctx2t = g.storage((rows * d) as u64);
            steps.extend(self.attention(&q2t, &k2t, &v2t, &var_kmask, &ctx2t, b * n, v, false));
            let ctx2 = g.storage((rows * d) as u64);
            steps.push(self.swap12(&ctx2t, &ctx2, b, n, v, d));
            let var_out = g.storage((rows * d) as u64);
            steps.push(self.linear(&ctx2, &format!("{p}.var_attn.out_proj.weight"), &var_out, rows, d, d));
            let var_normed = g.storage((rows * d) as u64);
            steps.push(self.rmsnorm_named(&var_out, &format!("{p}.post_var_attn_ln.weight"), &var_normed, d, rows));
            steps.push(self.add_inplace(&var_normed, &h1, rows * d));
            let h2 = var_normed;

            // ---- feedforward: ReLU, hidden width == model_dims (no 4x) ----
            let ff_in = g.storage((rows * d) as u64);
            steps.push(self.rmsnorm_named(&h2, &format!("{p}.pre_ff_ln.weight"), &ff_in, d, rows));
            let ff_h = g.storage((rows * c.hidden_dims) as u64);
            steps.push(self.linear(&ff_in, &format!("{p}.ff0.weight"), &ff_h, rows, d, c.hidden_dims));
            steps.push(self.relu_inplace(&ff_h, rows * c.hidden_dims));
            let ff_out = g.storage((rows * d) as u64);
            steps.push(self.linear(&ff_h, &format!("{p}.ff1.weight"), &ff_out, rows, c.hidden_dims, d));
            let ff_normed = g.storage((rows * d) as u64);
            steps.push(self.rmsnorm_named(&ff_out, &format!("{p}.post_ff_ln.weight"), &ff_normed, d, rows));
            steps.push(self.add_inplace(&ff_normed, &h2, rows * d));
            h = ff_normed;
        }

        // ---- output head: biased linear, no norm ----
        let head_out = c.head_out_dim();
        let logits = g.storage((rows * head_out) as u64);
        steps.push(self.linear(&h, "output_head.weight", &logits, rows, d, head_out));
        steps.push(g.step(BIAS_ADD, &[&logits, self.w("output_head.bias")], &[rows as u32, head_out as u32], (rows * head_out) as u32));

        g.submit(&[], &steps);
        g.read(&logits, rows * head_out)
    }
}

/// `log(1 + exp(x))`, numerically stable for large `|x|` - the reference's
/// `torch.nn.functional.softplus` default (`beta=1, threshold=20`).
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    #[test]
    fn softplus_matches_the_naive_formula_away_from_the_overflow_edge() {
        for x in [-5.0f32, -1.0, 0.0, 1.0, 5.0, 19.9] {
            let want = (1.0 + x.exp()).ln();
            assert!((softplus(x) - want).abs() < 1e-4, "x={x}");
        }
        assert!(softplus(50.0).is_finite(), "must not overflow like the naive formula would");
    }

    #[test]
    fn from_weights_folds_per_dim_scale_into_the_query_gain() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let mut weights: HashMap<String, Vec<f32>> =
            cfg.param_list().into_iter().map(|(k, s)| (k, vec![1.0; s.iter().product()])).collect();
        // Distinctive per_dim_scale so the fold is checkable, not just "ran".
        let per_dim_scale = vec![0.0f32; cfg.head_dim]; // softplus(0) = ln(2)
        weights.insert("transformer_stack.layers.0.seq_attn.per_dim_scale.per_dim_scale".into(), per_dim_scale);

        let m = Timesfm3::from_weights(cfg.clone(), &weights).unwrap();
        let got = m.gpu.read(m.w("transformer_stack.layers.0.seq_attn.query_ln.weight"), cfg.head_dim);
        let want = 1.0 * 1.442_695_1 * (2f32.ln()); // gain(1.0) * 1.442695 * softplus(0)
        for v in got {
            assert!((v - want).abs() < 1e-4, "{v} vs {want}");
        }
        // The raw per_dim_scale buffer must not exist as its own weight -
        // it was consumed into the fold, not carried through separately.
        assert!(!m.w.contains_key("transformer_stack.layers.0.seq_attn.per_dim_scale.per_dim_scale"));
    }

    #[test]
    fn core_forward_runs_at_tiny_scale_and_produces_finite_output() {
        if skip() {
            return;
        }
        let cfg = Timesfm3Config::tiny();
        let weights: HashMap<String, Vec<f32>> = cfg
            .param_list()
            .into_iter()
            .enumerate()
            .map(|(i, (k, s))| {
                let n: usize = s.iter().product();
                let data: Vec<f32> = (0..n).map(|j| (((i * 131 + j * 17) % 23) as f32 - 11.0) * 0.01).collect();
                (k, data)
            })
            .collect();
        let m = Timesfm3::from_weights(cfg.clone(), &weights).unwrap();
        let (b, v, n) = (2, 3, 4); // context patches only, no masking, for this smoke test
        let rows = b * v * n;
        let resblock_input: Vec<f32> = (0..rows * cfg.resblock_in_dim()).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let mask = vec![false; b * v * n];
        let out = m.core_forward(&resblock_input, &mask, b, v, n);
        assert_eq!(out.len(), rows * cfg.head_out_dim());
        assert!(out.iter().all(|x| x.is_finite()), "core_forward produced a non-finite value");
        assert!(out.iter().any(|&x| x != 0.0), "core_forward produced an all-zero output");
    }
}
