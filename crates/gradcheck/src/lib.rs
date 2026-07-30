// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Numerical gradient checker — brain's backprop correctness gate.
//!
//! With the PyTorch oracle dropped (brain is pure-Rust), this validates each
//! model's analytic WGSL gradients against finite differences of its own
//! forward pass. We use a **directional** check: for each parameter tensor pick
//! a random direction `v` and compare the analytic directional derivative
//! `⟨∇L, v⟩` to the central difference `(L(w+εv) − L(w−εv)) / 2ε`. Summing over
//! the tensor's entries averages out per-entry fp32 round-off, giving a stable
//! signal even on a software GPU.

use data::rng::Rng;

/// A model the checker can drive: a fixed batch must already be set.
pub trait CheckModel {
    fn param_names(&self) -> Vec<String>;
    fn read_weight(&self, name: &str) -> Vec<f32>;
    fn write_weight(&self, name: &str, data: &[f32]);
    fn read_grad(&self, name: &str) -> Vec<f32>;
    /// Run the forward pass on the fixed batch and return the scalar loss that
    /// [`Self::backward`] differentiates.
    fn loss(&self) -> f32;
    fn zero_grads(&self);
    fn backward(&self);
}

/// One parameter tensor's directional-derivative comparison.
#[derive(Clone, Debug)]
pub struct Check {
    pub param: String,
    pub analytic: f32,
    pub numeric: f32,
    pub abs_err: f32,
    pub rel_err: f32,
}

impl Check {
    /// `allclose`-style: `|a − n| ≤ atol + rtol·max(|a|, |n|)`. A pure relative
    /// metric is ill-conditioned when the directional derivative is ~0 (a random
    /// direction nearly orthogonal to ∇L), so we combine an absolute floor.
    pub fn within(&self, atol: f32, rtol: f32) -> bool {
        self.abs_err <= atol + rtol * self.analytic.abs().max(self.numeric.abs())
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    pub fn max_rel(&self) -> f32 {
        self.checks.iter().map(|c| c.rel_err).fold(0.0, f32::max)
    }
    /// True iff every tensor passes the combined tolerance.
    pub fn all_within(&self, atol: f32, rtol: f32) -> bool {
        self.checks.iter().all(|c| c.within(atol, rtol))
    }
    pub fn failures(&self, atol: f32, rtol: f32) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.within(atol, rtol)).collect()
    }
    pub fn print(&self) {
        for c in &self.checks {
            println!(
                "  {:<32} analytic={:+.5e} numeric={:+.5e} abs={:.2e} rel={:.2e}",
                c.param, c.analytic, c.numeric, c.abs_err, c.rel_err
            );
        }
    }
}

/// Directional gradient check over every parameter tensor. `eps` ≈ 5e-3 suits
/// fp32; `n_dirs` random directions are tried per tensor and the best-agreeing
/// one is reported — a real backprop bug fails *every* direction, while a random
/// direction nearly orthogonal to ∇L only makes finite differences ill-
/// conditioned (the directional derivative ≈ 0). A fixed batch must already be
/// set on `m`.
pub fn directional_check<M: CheckModel>(m: &M, eps: f32, n_dirs: usize, seed: u64) -> Report {
    // Analytic gradients for the current batch (computed once).
    m.zero_grads();
    let _ = m.loss();
    m.backward();

    let mut rng = Rng::new(seed);
    let mut checks = Vec::new();

    for name in m.param_names() {
        let w0 = m.read_weight(&name);
        let g = m.read_grad(&name);
        let mut best: Option<Check> = None;

        for _ in 0..n_dirs.max(1) {
            let v: Vec<f32> = (0..w0.len())
                .map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 })
                .collect();
            let analytic: f32 = g.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();

            let wp: Vec<f32> = w0.iter().zip(&v).map(|(&w, &vi)| w + eps * vi).collect();
            m.write_weight(&name, &wp);
            let lp = m.loss();
            let wm: Vec<f32> = w0.iter().zip(&v).map(|(&w, &vi)| w - eps * vi).collect();
            m.write_weight(&name, &wm);
            let lm = m.loss();

            let numeric = (lp - lm) / (2.0 * eps);
            let abs_err = (analytic - numeric).abs();
            let denom = analytic.abs().max(numeric.abs()).max(1e-3);
            let cand = Check { param: name.clone(), analytic, numeric, abs_err, rel_err: abs_err / denom };
            // Keep the direction with the smallest relative error (best conditioned).
            if best.as_ref().is_none_or(|b| cand.rel_err < b.rel_err) {
                best = Some(cand);
            }
        }
        m.write_weight(&name, &w0); // restore
        checks.push(best.unwrap());
    }
    Report { checks }
}

// ---- CheckModel for ANY architecture-agnostic Model (ADR §8) ----
//
// The `model::Model` trait already exposes exactly the parameter-access +
// forward/backward surface the checker needs, so one blanket impl gradient-checks
// every model (GPT, MoE, PID, and future seq2seq/autoencoder) by construction —
// closing the TESTING.md gap where only GPT was checked. `loss()` is the model's
// scalar `forward()` (the objective `backward()` differentiates).
impl<M: model::Model> CheckModel for M {
    fn param_names(&self) -> Vec<String> {
        model::Model::param_names(self)
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        model::Model::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        model::Model::write_weight(self, name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        model::Model::read_grad(self, name)
    }
    fn loss(&self) -> f32 {
        model::Model::forward(self)
    }
    fn zero_grads(&self) {
        model::Model::zero_grads(self);
    }
    fn backward(&self) {
        model::Model::backward(self);
    }
}

/// Build a tiny GPT, set a fixed batch, and gradient-check it. Returns the report.
pub fn check_gpt(seed: u64) -> Report {
    use gpt::{Gpt, GptConfig};
    let cfg = GptConfig { vocab: 23, block_size: 12, n_layers: 2, d_model: 16, n_heads: 2, d_ff: 32 };
    let init = gpt::init_weights(&cfg, seed);
    let model = Gpt::new(cfg, 2, 6, &init);
    // Fixed batch (no masking → every position contributes).
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny Qwen3 decoder, set a fixed batch, and gradient-check it. This is
/// the correctness gate for the GQA attention (fwd + the dq/dk/dv backward with
/// kv-head accumulation), the per-head QK-RMSNorm, the half-split RoPE-base, the
/// SwiGLU MLP, and the tied-embedding grad accumulation — all through the blanket
/// `CheckModel for model::Model`. The tiny config uses 4 query / 2 kv heads
/// (group 2) and a decoupled `head_dim` (8, vs d_model/n_heads = 4). Returns the
/// report.
pub fn check_qwen(seed: u64) -> Report {
    use qwen::{Qwen, QwenConfig};
    let cfg = QwenConfig::tiny();
    let init = qwen::init_weights(&cfg, seed);
    let model = Qwen::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny LFM2.5 encoder (conv + attention + conv layer stack) and
/// gradient-check it. This is the correctness gate for the bidirectional
/// attention backward through the GQA→MHA expansion (`kv_expand_bwd`
/// group-sum), the gated depthwise symmetric-pad conv mixer (in_proj
/// row-thirds, `mul` gating, permute adjoints, conv1d dx/dw), the eps-aware
/// RMSNorm family, and the tied MLM head with UNSHIFTED masked-CE targets —
/// only some positions are supervised (the MLM label pattern), the rest are
/// IGNORE, exercising the masking path.
pub fn check_lfm(seed: u64) -> Report {
    use lfm::{Lfm, LfmConfig};
    let cfg = LfmConfig::tiny();
    let init = lfm::init::init_weights(&cfg, seed);
    let model = Lfm::new_train(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    // Unshifted MLM targets: supervise every other position with the original
    // token (as if corrupted), IGNORE the rest.
    let y: Vec<u32> = x
        .iter()
        .enumerate()
        .map(|(i, &t)| if i % 2 == 0 { t } else { lfm::model::IGNORE })
        .collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny **LoRA** Qwen3 decoder and gradient-check the adapters. The
/// checker walks only the trainable params (`*.lora_a`/`*.lora_b`); the base
/// weights are frozen. A few AdamW steps run first so the zero-initialised `B`
/// adapter (and hence `A`'s gradient) is non-trivial before the FD comparison —
/// this validates the LoRA forward (`axpy` fusion) and the A/B backward path.
pub fn check_qwen_lora(seed: u64) -> Report {
    use qwen::{LoraCfg, Qwen, QwenConfig};
    let cfg = QwenConfig { lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
    let init = qwen::init_weights(&cfg, seed);
    let model = Qwen::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    // Move the adapters off the B=0 init so both A and B carry real gradients.
    for step in 1..=5 {
        model.zero_grads();
        model.forward();
        model.backward();
        model.adamw_step(step, 5e-2, 0.0, Some(1.0), 1.0);
        model.poll_wait();
    }
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Gradient-check the **Qwen2** decoder variant: QK-norm **off**, q/k/v projection
/// **bias on** (the deltas vs Qwen3, used by FastVLM). Validates the bias forward
/// (`bias_add`) + backward (`bias_grad` row-sum) and the QK-norm-off routing (q/k
/// flow straight from the biased projection into RoPE/attention). Returns the report.
pub fn check_qwen2(seed: u64) -> Report {
    use qwen::{Qwen, QwenConfig};
    let cfg = QwenConfig { qk_norm: false, attn_bias: true, ..QwenConfig::tiny() };
    let init = qwen::init_weights(&cfg, seed);
    let model = Qwen::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Gradient-check the interleaved-M-RoPE decoder path (`Qwen::enable_mrope`),
/// which swaps the analytic rope_base for the table-driven `rope2d` on q/k. Uses
/// simple diagonal per-token position tables (so the rotation is non-trivial but
/// the interleaving — validated separately in `qwenvl::mrope` — is not needed
/// here); the check confirms `rope2d`'s forward and its sign=-1 backward are
/// correctly wired into the decoder's parameter gradients. Returns the report.
pub fn check_qwen_mrope(seed: u64) -> Report {
    use qwen::{Qwen, QwenConfig};
    let cfg = QwenConfig::tiny();
    let (hd, t) = (cfg.head_dim as usize, 6usize);
    let n = 2 * t; // b·t
    let theta = cfg.rope_theta;
    let init = qwen::init_weights(&cfg, seed);
    let mut model = Qwen::new(cfg, 2, 6, &init);
    model.enable_mrope();

    // Per-token cos/sin for position = row-within-sequence (row % t), i.e. the
    // ordinary sequential positions — a non-trivial rotation the rope2d path must
    // reproduce and differentiate.
    let half = hd / 2;
    let (mut cos, mut sin) = (vec![0f32; n * half], vec![0f32; n * half]);
    for r in 0..n {
        let posn = (r % t) as f32;
        for d in 0..half {
            let ang = posn * theta.powf(-2.0 * d as f32 / hd as f32);
            cos[r * half + d] = ang.cos();
            sin[r * half + d] = ang.sin();
        }
    }
    model.write_mrope_tables(&cos, &sin);

    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Gradient-check the vision-language embedding splice (`Qwen::enable_mm_splice`).
/// The spliced image embeddings are an INPUT, not a parameter, so this runs a
/// bespoke directional check on them: perturb `img_embeds` along random ±1
/// directions and compare the central difference of the loss to the analytic
/// gradient read back via `read_d_img_embeds`. This validates that `splice`
/// injects the embeddings into the residual stream and `splice_bwd` extracts
/// their gradient (and that the subsequent `emb_bwd` scatter, over rows the
/// splice backward zeroed, does not corrupt it). The image rows carry IGNORE
/// targets, so their only influence on the loss is through attention — a
/// non-trivial path exercising the full splice + decoder graph. Returns a
/// one-entry report ("img_embeds").
pub fn check_vlm_splice(seed: u64) -> Report {
    use qwen::{Qwen, QwenConfig, IGNORE};
    let cfg = QwenConfig::tiny();
    let d = cfg.d_model as usize;
    let init = qwen::init_weights(&cfg, seed);
    let mut model = Qwen::new(cfg, 2, 6, &init);
    let (row0, n_rows) = (1u32, 2u32);
    model.enable_mm_splice(row0, n_rows);

    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let mut y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    for r in row0..row0 + n_rows {
        y[r as usize] = IGNORE; // image tokens are never a prediction target
    }
    model.set_batch(&x, &y);

    let mut rng = Rng::new(seed ^ 0xA11CE);
    let img0: Vec<f32> = (0..n_rows as usize * d).map(|_| rng.next_f32() - 0.5).collect();
    model.write_img_embeds(&img0);

    // Analytic grad of the loss w.r.t. the spliced image embeddings.
    model.zero_grads();
    let _ = model.forward();
    model.backward();
    let g = model.read_d_img_embeds();

    // Directional central difference (same recipe as `directional_check`, but on
    // the input rather than a parameter).
    let (eps, n_dirs) = (5e-3f32, 4usize);
    let mut best: Option<Check> = None;
    for _ in 0..n_dirs {
        let v: Vec<f32> = (0..img0.len()).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
        let analytic: f32 = g.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();
        let ip: Vec<f32> = img0.iter().zip(&v).map(|(&w, &vi)| w + eps * vi).collect();
        model.write_img_embeds(&ip);
        let lp = model.forward();
        let im: Vec<f32> = img0.iter().zip(&v).map(|(&w, &vi)| w - eps * vi).collect();
        model.write_img_embeds(&im);
        let lm = model.forward();
        let numeric = (lp - lm) / (2.0 * eps);
        let abs_err = (analytic - numeric).abs();
        let denom = analytic.abs().max(numeric.abs()).max(1e-3);
        let cand = Check { param: "img_embeds".into(), analytic, numeric, abs_err, rel_err: abs_err / denom };
        if best.as_ref().is_none_or(|b| cand.rel_err < b.rel_err) {
            best = Some(cand);
        }
    }
    model.write_img_embeds(&img0); // restore
    Report { checks: vec![best.unwrap()] }
}

/// Build a tiny sparse-MoE Trainer, set a fixed batch, and gradient-check it
/// (validates RMSNorm/RoPE/router/SwiGLU/aux+z-loss backprop). Now that MoE
/// implements `model::Model`, the blanket `CheckModel` impl makes it checkable —
/// closing the TESTING.md gap where only GPT was gradient-checked. Returns the
/// report.
pub fn check_moe(seed: u64) -> Report {
    use moe::train::{Config, Trainer};
    // aux_coef/z_coef = 0: the FD check differentiates the model's scalar
    // `forward()`, which is the cross-entropy only (the load-balancing aux loss
    // and router z-loss are folded into the router gradient, not the returned
    // scalar — matching `validate`'s CE-only comparison vs the PyTorch
    // reference). Zeroing them makes the analytic router grad consistent with the
    // CE-only FD; the aux/z terms are gated separately by `train::validate`.
    let cfg = Config {
        vocab: 23,
        block_size: 12,
        n_layers: 2,
        d_model: 16,
        n_heads: 2,
        n_experts: 3,
        // top_k == n_experts: every expert is always selected, so the renormalised
        // gate is a smooth softmax over all experts with no hard top-k selection
        // boundary. That removes the discontinuity FD cannot see (perturbing the
        // router weight could otherwise flip *which* experts are in the top-k,
        // making the central difference ill-conditioned) while still exercising
        // the full router matmul + softmax + gate backprop.
        top_k: 3,
        d_ff: 32,
        aux_coef: 0.0,
        z_coef: 0.0,
    };
    let init = <Trainer as model::Model>::init_weights(&cfg, seed);
    let model = Trainer::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny GLM-5.2 (`glm_moe_dsa`) decoder, set a fixed batch, and
/// gradient-check it. This is the correctness gate for **MLA** (low-rank q/kv
/// down/up projections, the decoupled nope/rope head split, interleaved RoPE on
/// the rope slice + the shared MQA key `k_rot` grad summed over heads, the
/// `mla_scores`/`mla_bwd_*` kernels), the **sigmoid `noaux_tc` router**
/// (`router_gate_sigmoid`/`router_bwd_sigmoid` through the sigmoid + top-k
/// normalization), the **shared expert**, the **dense→MoE layer schedule**, and
/// the untied `lm_head` — all through the blanket `CheckModel for model::Model`.
///
/// `top_k == n_routed_experts`: every expert is always selected, so the group
/// top-k selection has no hard boundary (which FD cannot see) while the router's
/// sigmoid + renormalization backward is still fully exercised. The router
/// selection bias is Frozen (never in the trainable/checked set), matching the
/// reference where it is a load-balance heuristic, not a backprop target.
pub fn check_glm(seed: u64) -> Report {
    use glm::{Glm, GlmConfig};
    let cfg = GlmConfig { num_experts_per_tok: 3, ..GlmConfig::tiny() }; // top_k == n_routed_experts
    let init = glm::init_weights(&cfg, seed);
    let model = Glm::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny GLM with the **MTP** (multi-token-prediction) head enabled and
/// gradient-check it. Validates the added auxiliary t+2 path: the shared-embedding
/// input, the two RMSNorms, the split `eh_proj`, the position-wise SwiGLU block
/// with its residual, the shared-head norm, and the shared `lm_head` grad
/// accumulation (main + MTP) — plus that the MTP grad correctly flows back into
/// the final residual. `top_k == n_routed_experts` (smooth router).
pub fn check_glm_mtp(seed: u64) -> Report {
    use glm::{Glm, GlmConfig};
    let cfg = GlmConfig { mtp: true, num_experts_per_tok: 3, ..GlmConfig::tiny() };
    let init = glm::init_weights(&cfg, seed);
    let model = Glm::new(cfg, 2, 6, &init);
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny PID event/effect transformer, set a fixed (partially masked)
/// batch, and gradient-check it (validates LayerNorm-with-bias, biased linears,
/// SwiGLU, key-padding causal attention, and the separate u_head backprop). PID
/// is masked CE over the u_bins label space, so the fixed targets label a few
/// positions and IGNORE the rest. Now that PID implements `model::Model`, the
/// blanket `CheckModel` impl makes it checkable — closing the TESTING.md gap.
pub fn check_pid(seed: u64) -> Report {
    use pid::{Pid, PidConfig, IGNORE};
    let cfg = PidConfig {
        vocab: 20,
        block_size: 8,
        n_layers: 1,
        d_model: 16,
        n_heads: 2,
        d_ff: 32,
        u_bins: 8,
    };
    let init = pid::data::init_weights(&cfg, seed);
    let model = Pid::new(cfg, 2, 6, &init);
    // 2 sequences x 6 positions; label a few positions (rest IGNORE), no PAD so
    // every position is a valid attention key.
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 20).collect();
    let mut y = vec![IGNORE; 12];
    y[2] = 3;
    y[5] = 1;
    y[8] = 6;
    y[11] = 0;
    model.set_batch(&x, &y);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Build a tiny encoder-decoder Transformer, set a fixed seq2seq batch, and
/// gradient-check it. This is the correctness gate for the new architecture: it
/// validates the bidirectional encoder self-attention, the causal decoder
/// self-attention, the decoder->encoder cross-attention (whose backward splits
/// grads across the decoder-Q buffer and the encoder-memory K/V buffer), the
/// shared src/tgt embedding accumulation, and the masked-CE token head — all
/// through the blanket `CheckModel for model::Model`. Returns the report.
pub fn check_seq2seq(seed: u64) -> Report {
    use seq2seq::{Seq2Seq, Seq2SeqConfig, IGNORE};
    let cfg = Seq2SeqConfig {
        vocab: 23,
        block_size: 6,     // decoder (target) length
        src_block_size: 5, // encoder (source) length — exercises T_dec != T_enc
        n_enc: 2,
        n_dec: 2,
        d_model: 16,
        n_heads: 2,
        d_ff: 32,
    };
    let init = seq2seq::init_weights(&cfg, seed);
    // b=2 sequences; encoder length 5, decoder length 6.
    let model = Seq2Seq::new(cfg, 2, 6, &init);
    let src: Vec<u32> = (0..10).map(|i| (i * 7 + 1) % 23).collect(); // 2 x 5
    let tgt: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect(); // 2 x 6
    // Label a few decoder positions (rest IGNORE) to also exercise masking.
    let mut labels = vec![IGNORE; 12];
    labels[1] = 3;
    labels[3] = 7;
    labels[5] = 0;
    labels[7] = 11;
    labels[9] = 2;
    labels[11] = 5;
    model.set_batch(&src, &tgt, &labels);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

/// Gradient-check the FLUX.2 Klein **host** training reference at tiny dims:
/// the f64 instantiation of `flux2::modelgrad` (double + single block stacks,
/// joint attention, QK-RMSNorm, interleaved RoPE, SwiGLU, and the whole
/// conditioning path — timestep MLP → three global modulation linears + final
/// adaLN, with site grads accumulated across the block stack). Unlike the
/// device models above, flux2's trainer IS the host path (the f32
/// instantiation of the same code; a device trainer is future work), so the
/// FD check runs directly against the host forward under the rectified-flow
/// velocity-MSE loss — same directional-derivative recipe as
/// [`directional_check`], in f64.
pub fn check_flux2(seed: u64) -> Report {
    use flux2::modelgrad::{backward, forward, grad_views, init_model, loss, make_flow_batch, params_mut, Cfg};
    let cfg = Cfg::tiny();
    let w0 = init_model::<f64>(&cfg, seed);
    let mut rng = Rng::new(seed ^ 0xF1u64);
    let mut rf = || rng.next_f64() - 0.5;
    let x0: Vec<f64> = (0..cfg.n_img() * cfg.in_channels).map(|_| rf()).collect();
    let ctx: Vec<f64> = (0..cfg.txt_len * cfg.context_in_dim).map(|_| rf()).collect();
    let noise: Vec<f64> = (0..x0.len()).map(|_| rf()).collect();
    let b = make_flow_batch(&cfg, &x0, &ctx, 0.45, &noise);

    let run_loss = |w: &flux2::modelgrad::ModelWeights<f64>| -> f64 {
        let (pred, _) = forward(&cfg, w, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
        loss(&pred, &b.target).0
    };
    let (pred, cache) = forward(&cfg, &w0, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
    let (_l, dpred) = loss(&pred, &b.target);
    let g = backward(&cfg, &w0, &cache, &dpred);
    let analytic: Vec<(String, Vec<f64>)> = grad_views(&g).into_iter().map(|(n, v)| (n, v.clone())).collect();

    let eps = 1e-5;
    let mut checks = Vec::new();
    for (pi, (name, ga)) in analytic.iter().enumerate() {
        let v: Vec<f64> = (0..ga.len()).map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 }).collect();
        let an: f64 = ga.iter().zip(&v).map(|(&gi, &vi)| gi * vi).sum();
        let mut wp = w0.clone();
        for (p, &vi) in params_mut(&mut wp)[pi].1.iter_mut().zip(&v) {
            *p += eps * vi;
        }
        let lp = run_loss(&wp);
        let mut wm = w0.clone();
        for (p, &vi) in params_mut(&mut wm)[pi].1.iter_mut().zip(&v) {
            *p -= eps * vi;
        }
        let lm = run_loss(&wm);
        let numeric = (lp - lm) / (2.0 * eps);
        let abs_err = (an - numeric).abs();
        let denom = an.abs().max(numeric.abs()).max(1e-3);
        checks.push(Check {
            param: name.clone(),
            analytic: an as f32,
            numeric: numeric as f32,
            abs_err: abs_err as f32,
            rel_err: (abs_err / denom) as f32,
        });
    }
    Report { checks }
}

/// Build a tiny bottleneck autoencoder, set a fixed float batch, and
/// gradient-check it. This is the correctness gate for the `Regression` head
/// (ADR §6, PR-10): it validates the new `mse_value`/`mse_grad` loss kernels and
/// the encoder→bottleneck→decoder matmul/GELU/bias backprop, all through the
/// blanket `CheckModel for model::Model`. The objective is mean-squared
/// reconstruction error, so unlike the token-head models there is no masking —
/// every output element contributes. Returns the report.
pub fn check_autoencoder(seed: u64) -> Report {
    use autoencoder::{Autoencoder, AutoencoderConfig};
    let cfg = AutoencoderConfig { in_dim: 12, hidden: 16, z_dim: 4 };
    let init = autoencoder::init_weights(&cfg, seed);
    // b=2 items; reconstruct each item against itself (inputs == targets).
    let model = Autoencoder::new(cfg.clone(), 2, &init);
    let x: Vec<f32> = (0..(2 * cfg.in_dim)).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.13).collect();
    model.set_batch(&x, &x);
    directional_check(&model, 5e-3, 4, seed ^ 0x1234)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_gpt(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn lfm_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_lfm(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn qwen_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_qwen(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn qwen_lora_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_qwen_lora(7);
        report.print();
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "LoRA gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn qwen2_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_qwen2(7);
        report.print();
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "Qwen2 (qk-norm off, bias on) gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn qwen_mrope_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_qwen_mrope(7);
        report.print();
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "M-RoPE gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn vlm_splice_grad_matches_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_vlm_splice(7);
        report.print();
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "VLM splice gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn moe_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_moe(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn glm_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_glm(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn glm_mtp_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_glm_mtp(7);
        report.print();
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "MTP gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pid_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_pid(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn seq2seq_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_seq2seq(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn flux2_analytic_grads_match_finite_differences() {
        // Pure host f64 — no GPU, so no MOE_SKIP_GPU_TESTS gate.
        let report = check_flux2(7);
        report.print();
        // f64 central differences on the host reference are tight.
        let (atol, rtol) = (1e-6, 1e-4);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "FLUX.2 gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn autoencoder_analytic_grads_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let report = check_autoencoder(7);
        report.print();
        // fp32 directional FD on a software GPU: combined abs+rel tolerance.
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "gradient check failed for {:?}",
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
        );
    }
}
