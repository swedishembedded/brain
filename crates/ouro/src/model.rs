// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements looped-transformer execution - one shared
// stack run for a configurable number of passes with per-pass readouts - for
// its clients. If your team needs expertise in recurrent-depth inference or in
// training through repeated uses of the same weights then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Ouro's execution graph.
//!
//! One prefill forward runs the shared layer stack `total_ut_steps` times over
//! the whole sequence. Every pass is a full causal transformer pass over the
//! same weights; pass `p` starts from the residual stream pass `p-1` left
//! behind, and each pass ends with the final RMSNorm plus the exit gate.
//!
//! The graph is recorded once at construction, exactly like every other model
//! in this workspace: one `Step` list for the trunk (all passes), and one head
//! per pass so a caller can read that pass's logits without materializing all
//! `T · vocab` rows at once.
//!
//! Boundary residuals are kept per `(pass, layer)` - `T · L + 1` buffers - so
//! the parity taps can hand back any pass's residual stream and the backward
//! pass (added next) can rebuild a layer from its input.

use std::cell::Cell;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use paramstore::ParamStore;

use crate::config::OuroConfig;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const RMS_INV: usize = 3;
const RMSNORM_DX: usize = 4;
const RMSNORM_DW: usize = 5;
const ROPE: usize = 6;
const ROPE_BWD: usize = 7;
const GQA_SCORES: usize = 8;
const ATTN_SOFTMAX: usize = 9;
const GQA_APPLY: usize = 10;
const GQA_DSCORES: usize = 11;
const GQA_DV: usize = 12;
const GQA_DQ: usize = 13;
const GQA_DK: usize = 14;
const SILU_MUL: usize = 15;
const SILU_DA: usize = 16;
const SILU_DB: usize = 17;
const ADD2: usize = 18;
const BIAS_ADD: usize = 19;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("add2", kernels::ADD2),
    ("bias_add", kernels::BIAS_ADD),
];

fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        rmsnorm_dx: RMSNORM_DX,
        rmsnorm_dw: RMSNORM_DW,
        rope: ROPE,
        rope_bwd: ROPE_BWD,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: GQA_DSCORES,
        gqa_dv: GQA_DV,
        gqa_dq: GQA_DQ,
        gqa_dk: GQA_DK,
        silu_mul: SILU_MUL,
        silu_da: SILU_DA,
        silu_db: SILU_DB,
        rmsnorm_rows: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
    }
}

/// Per-layer scratch. Reused by every layer and every pass: a layer consumes
/// its input boundary residual and produces the next one, so internal
/// activations are dead as soon as the next layer starts.
struct Scratch {
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xn1_2: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    mlp_h: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn2_2: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

pub struct Ouro {
    pub gpu: Gpu,
    pub cfg: OuroConfig,
    pub ps: ParamStore,
    b: u32,
    t: u32,

    tokens: DeviceBuffer,
    /// `T * L + 1` boundary residuals, `[n, d]` each.
    res: Vec<DeviceBuffer>,
    /// `T` pass outputs: the final-normed hidden state each pass ends on.
    hs: Vec<DeviceBuffer>,
    /// `T` exit-gate logits, `[n]` each (pre-sigmoid).
    gates: Vec<DeviceBuffer>,
    /// One `[n, vocab]` logit buffer, written by whichever pass head is run.
    logits: DeviceBuffer,

    scratch: Scratch,
    trunk: Vec<Step>,
    heads: Vec<Vec<Step>>,
    /// Last pass selection made by [`Ouro::forward`], per row.
    exit: Cell<Vec<u32>>,
}

impl Ouro {
    /// Frozen-weight model over a checkpoint source (a brain checkpoint, an HF
    /// directory source, or an in-memory init map).
    pub fn new(cfg: OuroConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource) -> Ouro {
        cfg.validate().expect("ouro config");
        assert!(t <= cfg.block_size, "sequence {t} exceeds positional capacity {}", cfg.block_size);
        assert!(b > 0 && t > 0);

        let gpu = Gpu::new(PIPELINES);
        let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);

        let n = (b * t) as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let v = cfg.vocab as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let st = |x: u64| gpu.storage(x);

        let tokens = gpu.buffer("tokens", n * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let boundaries = (cfg.total_ut_steps * cfg.n_layers + 1) as usize;
        let res = (0..boundaries).map(|_| st(n * d)).collect();
        let hs = (0..cfg.total_ut_steps).map(|_| st(n * d)).collect();
        let gates = (0..cfg.total_ut_steps).map(|_| st(n)).collect();
        let logits = st(n * v);

        let bht2 = (b as u64) * cfg.n_heads as u64 * (t as u64) * (t as u64);
        assert!(bht2 <= u32::MAX as u64, "materialized attention exceeds u32 index space");
        let scratch = Scratch {
            xn1: st(n * d),
            q: st(n * hq),
            k: st(n * hkv),
            v: st(n * hkv),
            ctx: st(n * hq),
            proj: st(n * d),
            xn1_2: st(n * d),
            xmid: st(n * d),
            xn2: st(n * d),
            gate_pre: st(n * ff),
            up: st(n * ff),
            mlp_h: st(n * ff),
            mlp_out: st(n * d),
            xn2_2: st(n * d),
            scores: st(bht2),
            probs: st(bht2),
        };

        let mut m = Ouro {
            gpu,
            cfg: cfg.clone(),
            ps,
            b,
            t,
            tokens,
            res,
            hs,
            gates,
            logits,
            scratch,
            trunk: Vec::new(),
            heads: Vec::new(),
            exit: Cell::new(vec![0; b as usize]),
        };
        m.trunk = m.trunk_steps();
        m.heads = (0..cfg.total_ut_steps).map(|p| m.head_steps(p as usize)).collect();
        m
    }

    pub fn batch(&self) -> u32 {
        self.b
    }
    pub fn seq(&self) -> u32 {
        self.t
    }
    pub fn steps(&self) -> u32 {
        self.cfg.total_ut_steps
    }
    fn rows(&self) -> u32 {
        self.b * self.t
    }
    fn k(&self) -> KernelIds {
        ids()
    }

    /// Upload token ids for the next forward.
    pub fn set_tokens(&self, tokens: &[u32]) {
        assert_eq!(tokens.len(), self.rows() as usize, "tokens must be b*t");
        self.gpu.write(&self.tokens, bytemuck::cast_slice(tokens));
    }

    /// The trunk: embed once, then `T` passes of the shared stack, each ending
    /// in the final norm and the exit gate. The embedding is applied exactly
    /// once - pass `p` consumes the residual stream, never `embed(x)` again.
    fn trunk_steps(&self) -> Vec<Step> {
        let cfg = &self.cfg;
        let ids = self.k();
        let n = self.rows();
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let t = cfg.total_ut_steps;
        let eps = cfg.norm_eps;
        let theta = cfg.rope_theta;
        let s = &self.scratch;
        let ga = Gqa { b: self.b, t: self.t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };

        let mut steps = vec![self.gpu.step(EMBED, &[&self.tokens, self.ps.w("tok.weight"), &self.res[0]], &[d, n], n * d)];
        for p in 0..t as usize {
            for l in 0..cfg.n_layers as usize {
                let i = p * cfg.n_layers as usize + l;
                // The reference reassigns `hidden_states = norm(hidden_states)`
                // at the end of every pass, so pass p > 0 consumes the previous
                // pass's NORMED output, not its pre-norm residual.
                let h_in: &DeviceBuffer = if l == 0 && p > 0 { &self.hs[p - 1] } else { &self.res[i] };
                let h_out = &self.res[i + 1];
                steps.push(block::rmsnorm_fwd(&self.gpu, &ids, h_in, self.ps.w(&format!("blocks.{l}.ln1.weight")), &s.xn1, d, n, eps));
                steps.push(self.gpu.step(MATMUL, &[&s.xn1, self.ps.w(&format!("blocks.{l}.attn.wq.weight")), &s.q], &[n, d, hq], n * hq));
                steps.push(self.gpu.step(MATMUL, &[&s.xn1, self.ps.w(&format!("blocks.{l}.attn.wk.weight")), &s.k], &[n, d, hkv], n * hkv));
                steps.push(self.gpu.step(MATMUL, &[&s.xn1, self.ps.w(&format!("blocks.{l}.attn.wv.weight")), &s.v], &[n, d, hkv], n * hkv));
                steps.push(block::rope_fwd(&self.gpu, &ids, &s.q, n, nh, hd, hq, self.t, theta));
                steps.push(block::rope_fwd(&self.gpu, &ids, &s.k, n, nkv, hd, hkv, self.t, theta));
                steps.extend(block::gqa_fwd(&self.gpu, &ids, &ga, &s.q, &s.k, &s.v, &s.scores, &s.probs, &s.ctx));
                steps.push(self.gpu.step(MATMUL, &[&s.ctx, self.ps.w(&format!("blocks.{l}.attn.wo.weight")), &s.proj], &[n, hq, d], n * d));
                // Sandwich normalization: the attention output is normalized
                // before the residual add, and so is the MLP output.
                steps.push(block::rmsnorm_fwd(&self.gpu, &ids, &s.proj, self.ps.w(&format!("blocks.{l}.ln1_2.weight")), &s.xn1_2, d, n, eps));
                steps.push(self.gpu.step(ADD2, &[h_in, &s.xn1_2, &s.xmid], &[n * d], n * d));
                steps.push(block::rmsnorm_fwd(&self.gpu, &ids, &s.xmid, self.ps.w(&format!("blocks.{l}.ln2.weight")), &s.xn2, d, n, eps));
                steps.push(self.gpu.step(MATMUL, &[&s.xn2, self.ps.w(&format!("blocks.{l}.mlp.gate.weight")), &s.gate_pre], &[n, d, ff], n * ff));
                steps.push(self.gpu.step(MATMUL, &[&s.xn2, self.ps.w(&format!("blocks.{l}.mlp.up.weight")), &s.up], &[n, d, ff], n * ff));
                steps.push(block::swiglu_fwd(&self.gpu, &ids, &s.gate_pre, &s.up, &s.mlp_h, n * ff));
                steps.push(self.gpu.step(MATMUL, &[&s.mlp_h, self.ps.w(&format!("blocks.{l}.mlp.down.weight")), &s.mlp_out], &[n, ff, d], n * d));
                steps.push(block::rmsnorm_fwd(&self.gpu, &ids, &s.mlp_out, self.ps.w(&format!("blocks.{l}.ln2_2.weight")), &s.xn2_2, d, n, eps));
                steps.push(self.gpu.step(ADD2, &[&s.xmid, &s.xn2_2, h_out], &[n * d], n * d));
            }
            let last = &self.res[(p + 1) * cfg.n_layers as usize];
            steps.push(block::rmsnorm_fwd(&self.gpu, &ids, last, self.ps.w("norm.weight"), &self.hs[p], d, n, eps));
            // Exit gate: one linear over the model width, plus its bias.
            steps.push(self.gpu.step(MATMUL, &[&self.hs[p], self.ps.w("gate.weight"), &self.gates[p]], &[n, d, 1], n));
            steps.push(self.gpu.step(BIAS_ADD, &[&self.gates[p], self.ps.w("gate.bias")], &[n, 1], n));
        }
        steps
    }

    fn head_steps(&self, pass: usize) -> Vec<Step> {
        let n = self.rows();
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        vec![self.gpu.step(MATMUL, &[&self.hs[pass], self.ps.w("lm_head.weight"), &self.logits], &[n, d, v], n * v)]
    }

    /// Run the trunk only (all passes and gates).
    pub fn trunk_forward(&self) {
        self.gpu.submit(&[], &self.trunk);
    }

    /// Run the trunk, then the LM head for pass `pass`.
    pub fn forward_pass(&self, pass: usize) {
        self.trunk_forward();
        self.head_forward(pass);
    }

    /// Run the LM head for `pass` against the already-computed pass output
    /// (the trunk must have run for these hidden states to be current).
    pub fn head_forward(&self, pass: usize) {
        assert!(pass < self.cfg.total_ut_steps as usize, "pass {pass} out of range");
        self.gpu.submit(&[], &self.heads[pass]);
        self.exit.set(vec![pass as u32; self.b as usize]);
    }

    /// The per-row exit pass implied by the gates and `early_exit_threshold`
    /// (the reference's CDF rule): the first pass whose cumulative exit mass
    /// reaches the threshold, else the final pass. Requires the trunk to have
    /// run.
    pub fn exit_steps(&self) -> Vec<u32> {
        let n = self.rows() as usize;
        let t = self.cfg.total_ut_steps as usize;
        let mut out = vec![t as u32 - 1; n];
        let mut remaining = vec![1.0f64; n];
        let mut cdf = vec![0.0f64; n];
        for pass in 0..t {
            let g = self.gpu.read(&self.gates[pass], n);
            for r in 0..n {
                let lambda = 1.0 / (1.0 + (-(g[r] as f64)).exp());
                let p = if pass + 1 < t { lambda * remaining[r] } else { remaining[r] };
                remaining[r] *= 1.0 - lambda;
                cdf[r] += p;
                if cdf[r] >= self.cfg.early_exit_threshold as f64 {
                    out[r] = pass as u32;
                }
            }
        }
        out
    }

    /// Inference forward: trunk, per-row exit selection, head for the selected
    /// pass. `exit_steps` must be uniform across rows for this entry point
    /// (batch 1, or equal prompts); a heterogeneous batch selects per row in
    /// the serving engine.
    pub fn forward(&self) -> Vec<f32> {
        self.trunk_forward();
        let steps = self.exit_steps();
        let step = steps[0];
        assert!(steps.iter().all(|s| *s == step), "heterogeneous exit steps {steps:?} need the serving path");
        self.gpu.submit(&[], &self.heads[step as usize]);
        self.exit.set(steps);
        self.read_logits()
    }

    pub fn read_logits(&self) -> Vec<f32> {
        self.gpu.read(&self.logits, self.rows() as usize * self.cfg.vocab as usize)
    }

    /// `[n, d]` final-normed hidden state of pass `pass`.
    pub fn read_hidden(&self, pass: usize) -> Vec<f32> {
        self.gpu.read(&self.hs[pass], self.rows() as usize * self.cfg.d_model as usize)
    }

    /// `[n]` exit-gate logits of pass `pass` (pre-sigmoid).
    pub fn read_gate(&self, pass: usize) -> Vec<f32> {
        self.gpu.read(&self.gates[pass], self.rows() as usize)
    }

    /// `[n, d]` residual stream entering layer `layer` of pass `pass` (layer
    /// `n_layers` is that pass's output).
    pub fn read_res(&self, pass: usize, layer: usize) -> Vec<f32> {
        let i = pass * self.cfg.n_layers as usize + layer;
        self.gpu.read(&self.res[i], self.rows() as usize * self.cfg.d_model as usize)
    }

    pub fn logits_all(&self, tokens: &[u32]) -> Vec<f32> {
        self.set_tokens(tokens);
        self.forward()
    }
}

impl model::ModelConfig for OuroConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        OuroConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        OuroConfig::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        OuroConfig::from_json(v).expect("ouro config")
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn block_size(&self) -> u32 {
        self.block_size
    }
    fn finalize_for_dataset(mut self, vocab: u32, block_size: u32) -> Self {
        self.vocab = vocab;
        self.block_size = block_size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hand-numbered slots index `PIPELINES` positionally; inserting a
    /// kernel without updating them would dispatch the wrong WGSL silently on
    /// the CPU backend. Pin the slots the graph actually uses.
    #[test]
    fn kernel_slots_match_the_pipeline_list() {
        for (slot, want) in [
            (EMBED, "embed"),
            (MATMUL, "matmul"),
            (RMSNORM, "rmsnorm"),
            (ROPE, "rope_base"),
            (GQA_SCORES, "gqa_scores"),
            (ATTN_SOFTMAX, "attn_softmax"),
            (GQA_APPLY, "gqa_apply"),
            (SILU_MUL, "silu_mul"),
            (ADD2, "add2"),
            (BIAS_ADD, "bias_add"),
        ] {
            assert_eq!(PIPELINES[slot].0, want, "slot {slot} is not {want}");
        }
    }

    #[test]
    fn every_registered_kernel_has_a_cost_formula() {
        for (name, _) in PIPELINES {
            assert!(gpu_core::cost::covers(name), "kernel {name} has no cost formula");
        }
    }
}
