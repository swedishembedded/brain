// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Bottleneck autoencoder (ADR 0001 §6), forward + backprop as WGSL compute
//! dispatches. Shares the engine seam (`gpu_core`, `paramstore`, `optim`,
//! `kernels`) with the GPT/MoE/PID/seq2seq models and implements
//! [`model::Model`] so the generic trainer and the blanket `gradcheck::CheckModel`
//! cover it.
//!
//! ## Objective: a `Regression` head (MSE), not a token head
//! This is the MAD **compression** model: encode a whole input sequence to ONE
//! compressed representation `z` (the bottleneck), then reconstruct the original
//! input from `z` alone. The objective is mean-squared reconstruction error
//! ([`model::Head::Regression`]), so it uses the new `mse_value`/`mse_grad`
//! kernels rather than the cross-entropy token head. There is no causal context
//! and no next-token prediction — reconstruction flows *only* through `z`, which
//! is exactly what a causal LM cannot express (ADR §6).
//!
//! ## Architecture (a symmetric 2-layer MLP autoencoder)
//!   x      [B, in_dim]                         (flattened input sequence)
//!   h  = GELU(x @ We^T + be)   [B, hidden]      (encoder)
//!   z  = h @ Wb^T + bb         [B, z_dim]       (BOTTLENECK — the single
//!                                                compressed representation;
//!                                                z_dim << in_dim is what forces
//!                                                compression)
//!   g  = GELU(z @ Wd^T + bd)   [B, hidden]      (decoder)
//!   out= g @ Wo^T + bo         [B, in_dim]      (reconstruction)
//!   loss = mean_i (out[i] - x[i])^2             (MSE over ALL B*in_dim elements)
//!
//! The "sequence -> one representation" framing of the MAD task is realized by
//! the caller flattening a length-`T` sequence of per-token feature vectors into
//! `in_dim = T * feat` floats; the bottleneck `z` (a single vector per item, far
//! narrower than `in_dim`) is the one compressed representation the whole input
//! must pass through, and the decoder MLP reconstructs all `T*feat` floats from
//! it. See `crates/bench/src/mad_compress.rs` for how tokens map to features and
//! back.
//!
//! Batch input is [`model::Batch::Tensor`] (`inputs` = `targets` = the flattened
//! sequence features; `tokens` unused). Every stage is a matmul / bias / GELU /
//! MSE — all individually finite-difference-checkable, so `check_autoencoder`
//! validates the whole graph through the blanket `CheckModel`.

use std::cell::Cell;
use std::collections::HashMap;

use serde_json::Value;

use gpu_core::{Gpu, Step};
use optim::Optim;
use paramstore::ParamStore;

// ---- kernel indices (order matches PIPELINES) ----
const MATMUL: usize = 0;
const BIAS_ADD: usize = 1;
const BIAS_GRAD: usize = 2;
const GELU: usize = 3;
const GELU_BWD: usize = 4;
const MATMUL_DX: usize = 5;
const MATMUL_DW: usize = 6;
const MSE_VALUE: usize = 7;
const MSE_GRAD: usize = 8;
const GRADNORM_SQ: usize = 9;
const GRAD_SCALE: usize = 10;
const ADAMW: usize = 11;
const CLIP_COEF: usize = 12;
const GRAD_SCALE_BUF: usize = 13;

const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("gelu", kernels::GELU),
    ("gelu_bwd", kernels::GELU_BWD),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("mse_value", kernels::MSE_VALUE),
    ("mse_grad", kernels::MSE_GRAD),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("adamw", kernels::ADAMW),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

/// Bottleneck-autoencoder configuration.
///
/// `in_dim` is the flattened input width (for MAD compress: `seq_len * feat`).
/// `z_dim` is the bottleneck width — keep it `<< in_dim` to force compression.
/// Following the [`model::ModelConfig`] convention, `block_size` carries the
/// input width (`in_dim`) and `vocab` is unused (there is no token head); both
/// exist only so the generic trainer can reason uniformly about the model.
#[derive(Clone, Debug)]
pub struct AutoencoderConfig {
    /// Flattened input/reconstruction width (`= seq_len * feat` for compress).
    pub in_dim: u32,
    /// Encoder/decoder hidden width.
    pub hidden: u32,
    /// Bottleneck width (the single compressed representation; `<< in_dim`).
    pub z_dim: u32,
}

impl AutoencoderConfig {
    /// A tiny config for tests / gradient checks.
    pub fn tiny() -> AutoencoderConfig {
        AutoencoderConfig { in_dim: 12, hidden: 16, z_dim: 4 }
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "autoencoder",
            "in_dim": self.in_dim, "hidden": self.hidden, "z_dim": self.z_dim
        })
    }

    pub fn from_json(c: &Value) -> AutoencoderConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        AutoencoderConfig {
            in_dim: g("in_dim", 12),
            hidden: g("hidden", 16),
            z_dim: g("z_dim", 4),
        }
    }

    /// Parameter list: `(name, numel)`. Weights follow the `out = x @ W^T`
    /// convention (`W` is `[out_features, in_features]` row-major).
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let i = self.in_dim as usize;
        let h = self.hidden as usize;
        let z = self.z_dim as usize;
        vec![
            ("enc.weight".to_string(), h * i), // [hidden, in]
            ("enc.bias".to_string(), h),
            ("bottleneck.weight".to_string(), z * h), // [z, hidden]
            ("bottleneck.bias".to_string(), z),
            ("dec.weight".to_string(), h * z), // [hidden, z]
            ("dec.bias".to_string(), h),
            ("out.weight".to_string(), i * h), // [in, hidden]
            ("out.bias".to_string(), i),
        ]
    }

    /// Fan-in (the `K` of `out = x @ W^T`) for a weight tensor, used by the
    /// initializer to scale the std. Biases are not passed here.
    pub fn fan_in_of(&self, name: &str) -> usize {
        match name {
            "enc.weight" => self.in_dim as usize,
            "bottleneck.weight" => self.hidden as usize,
            "dec.weight" => self.z_dim as usize,
            "out.weight" => self.hidden as usize,
            _ => 1,
        }
    }
}

pub struct Autoencoder {
    pub gpu: Gpu,
    pub cfg: AutoencoderConfig,
    pub ps: ParamStore,
    opt: Optim,
    b: u32,
    count: Cell<f32>, // total elements B*in_dim (the MSE denominator)

    // inputs / targets (the same flattened sequence features)
    x: gpu_core::DeviceBuffer,
    target: gpu_core::DeviceBuffer,

    // forward activations (SSA buffers)
    enc_pre: gpu_core::DeviceBuffer, // x @ We^T + be   [B, hidden]
    h: gpu_core::DeviceBuffer,       // GELU(enc_pre)   [B, hidden]
    z: gpu_core::DeviceBuffer,       // bottleneck      [B, z_dim]
    dec_pre: gpu_core::DeviceBuffer, // z @ Wd^T + bd   [B, hidden]
    g: gpu_core::DeviceBuffer,       // GELU(dec_pre)   [B, hidden]
    out: gpu_core::DeviceBuffer,     // reconstruction  [B, in_dim]
    mse_buf: gpu_core::DeviceBuffer, // per-element sq-err [B*in_dim]

    // backward temporaries
    d_out: gpu_core::DeviceBuffer, // grad wrt out    [B, in_dim]
    d_g: gpu_core::DeviceBuffer,   // grad wrt g      [B, hidden]
    d_dec_pre: gpu_core::DeviceBuffer, // grad wrt dec_pre [B, hidden]
    d_z: gpu_core::DeviceBuffer,   // grad wrt z      [B, z_dim]
    d_h: gpu_core::DeviceBuffer,   // grad wrt h      [B, hidden]
    d_enc_pre: gpu_core::DeviceBuffer, // grad wrt enc_pre [B, hidden]
    d_x: gpu_core::DeviceBuffer,   // grad wrt x (unused downstream) [B, in_dim]

    fwd_steps: Vec<Step>,
    bwd_steps: Vec<Step>,
}

impl Autoencoder {
    /// Load a model from a `.weights` checkpoint, sized for batch `b`.
    pub fn load(path: &str, b: u32) -> Autoencoder {
        let c = checkpoint::load(path);
        let cfg = AutoencoderConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Autoencoder::new(cfg, b, &init)
    }

    pub fn new(cfg: AutoencoderConfig, b: u32, init: &HashMap<String, Vec<f32>>) -> Autoencoder {
        Autoencoder::new_on(Gpu::new(PIPELINES), cfg, b, init)
    }

    /// Build on an existing device handle — see `Gpt::new_on`.
    fn new_on(gpu: Gpu, cfg: AutoencoderConfig, b: u32, init: &HashMap<String, Vec<f32>>) -> Autoencoder {
        let ps = ParamStore::new(&gpu, cfg.param_list(), init);
        let opt = Optim::new(ADAMW, GRADNORM_SQ, GRAD_SCALE, CLIP_COEF, GRAD_SCALE_BUF);

        let i = cfg.in_dim as u64;
        let h = cfg.hidden as u64;
        let z = cfg.z_dim as u64;
        let bi = b as u64 * i;
        let bh = b as u64 * h;
        let bz = b as u64 * z;
        let st = |x: u64| gpu.storage(x);
        let mk = |label: &str, n: u64| {
            gpu.buffer(label, n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
        };

        let mut m = Autoencoder {
            cfg,
            b,
            count: Cell::new(bi as f32),
            ps,
            opt,
            x: mk("x", bi),
            target: mk("target", bi),
            enc_pre: st(bh),
            h: st(bh),
            z: st(bz),
            dec_pre: st(bh),
            g: st(bh),
            out: st(bi),
            mse_buf: st(bi),
            d_out: st(bi),
            d_g: st(bh),
            d_dec_pre: st(bh),
            d_z: st(bz),
            d_h: st(bh),
            d_enc_pre: st(bh),
            d_x: st(bi),
            fwd_steps: Vec::new(),
            bwd_steps: Vec::new(),
            gpu,
        };
        m.fwd_steps = m.forward_steps();
        m.bwd_steps = m.build_backward_steps();
        m
    }

    /// Upload one batch. `inputs` and `targets` are both the flattened sequence
    /// features `[B*in_dim]`; the autoencoder reconstructs `inputs` against
    /// `targets` (normally identical — pass the same slice).
    pub fn set_batch(&self, inputs: &[f32], targets: &[f32]) {
        assert_eq!(inputs.len(), (self.b * self.cfg.in_dim) as usize, "input size mismatch");
        assert_eq!(targets.len(), inputs.len(), "target size mismatch");
        self.gpu.write(&self.x, bytemuck::cast_slice(inputs));
        self.gpu.write(&self.target, bytemuck::cast_slice(targets));
        self.count.set(inputs.len().max(1) as f32);
    }

    fn w(&self, name: &str) -> &gpu_core::DeviceBuffer {
        self.ps.w(name)
    }

    // Step-by-step append matches the engine's forward/backward graph-building
    // idiom (cf. `gpt`/`seq2seq`); the dispatch order is the readable structure.
    #[allow(clippy::vec_init_then_push)]
    fn forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let b = self.b;
        let i = c.in_dim;
        let h = c.hidden;
        let z = c.z_dim;
        let bi = b * i;
        let bh = b * h;
        let bz = b * z;
        let mut s: Vec<Step> = Vec::with_capacity(11);

        // encoder: h = GELU(x @ We^T + be)
        s.push(self.gpu.step(MATMUL, &[&self.x, self.w("enc.weight"), &self.enc_pre], &[b, i, h], bh));
        s.push(self.gpu.step(BIAS_ADD, &[&self.enc_pre, self.w("enc.bias")], &[b, h], bh));
        s.push(self.gpu.step(GELU, &[&self.enc_pre, &self.h], &[bh], bh));
        // bottleneck: z = h @ Wb^T + bb
        s.push(self.gpu.step(MATMUL, &[&self.h, self.w("bottleneck.weight"), &self.z], &[b, h, z], bz));
        s.push(self.gpu.step(BIAS_ADD, &[&self.z, self.w("bottleneck.bias")], &[b, z], bz));
        // decoder: g = GELU(z @ Wd^T + bd)
        s.push(self.gpu.step(MATMUL, &[&self.z, self.w("dec.weight"), &self.dec_pre], &[b, z, h], bh));
        s.push(self.gpu.step(BIAS_ADD, &[&self.dec_pre, self.w("dec.bias")], &[b, h], bh));
        s.push(self.gpu.step(GELU, &[&self.dec_pre, &self.g], &[bh], bh));
        // reconstruction: out = g @ Wo^T + bo
        s.push(self.gpu.step(MATMUL, &[&self.g, self.w("out.weight"), &self.out], &[b, h, i], bi));
        s.push(self.gpu.step(BIAS_ADD, &[&self.out, self.w("out.bias")], &[b, i], bi));
        // loss: per-element squared error / (B*in_dim)
        s.push(self.gpu.step(MSE_VALUE, &[&self.out, &self.target, &self.mse_buf], &[bi], bi));
        s
    }

    pub fn forward_submit(&self) {
        self.gpu.submit(&[], &self.fwd_steps);
    }

    pub fn loss(&self) -> f32 {
        let n = (self.b * self.cfg.in_dim) as usize;
        // mse_value already divides each element by n, so the host sum is the MSE.
        self.gpu.read(&self.mse_buf, n).iter().sum()
    }

    pub fn forward(&self) -> f32 {
        self.forward_submit();
        self.loss()
    }

    pub fn backward(&self) {
        self.gpu.submit(&[], &self.bwd_steps);
    }

    fn build_backward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let b = self.b;
        let i = c.in_dim;
        let h = c.hidden;
        let z = c.z_dim;
        let bi = b * i;
        let bh = b * h;
        let bz = b * z;
        let g = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::with_capacity(16);

        // d_out = d(loss)/d(out) = 2*(out - target)/(B*in_dim)
        s.push(self.gpu.step(MSE_GRAD, &[&self.out, &self.target, &self.d_out], &[bi], bi));
        // out = g @ Wo^T + bo  ->  d_g, dWo, dbo
        s.push(self.gpu.step(BIAS_GRAD, &[&self.d_out, g("out.bias")], &[b, i], i));
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_out, &self.g, g("out.weight")], &[b, h, i], i * h));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_out, self.w("out.weight"), &self.d_g], &[b, h, i, 0], bh));
        // g = GELU(dec_pre)  ->  d_dec_pre
        s.push(self.gpu.step(GELU_BWD, &[&self.dec_pre, &self.d_g, &self.d_dec_pre], &[bh], bh));
        // dec_pre = z @ Wd^T + bd  ->  d_z, dWd, dbd
        s.push(self.gpu.step(BIAS_GRAD, &[&self.d_dec_pre, g("dec.bias")], &[b, h], h));
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_dec_pre, &self.z, g("dec.weight")], &[b, z, h], h * z));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_dec_pre, self.w("dec.weight"), &self.d_z], &[b, z, h, 0], bz));
        // z = h @ Wb^T + bb  ->  d_h, dWb, dbb
        s.push(self.gpu.step(BIAS_GRAD, &[&self.d_z, g("bottleneck.bias")], &[b, z], z));
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_z, &self.h, g("bottleneck.weight")], &[b, h, z], z * h));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_z, self.w("bottleneck.weight"), &self.d_h], &[b, h, z, 0], bh));
        // h = GELU(enc_pre)  ->  d_enc_pre
        s.push(self.gpu.step(GELU_BWD, &[&self.enc_pre, &self.d_h, &self.d_enc_pre], &[bh], bh));
        // enc_pre = x @ We^T + be  ->  d_x (unused), dWe, dbe
        s.push(self.gpu.step(BIAS_GRAD, &[&self.d_enc_pre, g("enc.bias")], &[b, h], h));
        s.push(self.gpu.step(MATMUL_DW, &[&self.d_enc_pre, &self.x, g("enc.weight")], &[b, i, h], h * i));
        s.push(self.gpu.step(MATMUL_DX, &[&self.d_enc_pre, self.w("enc.weight"), &self.d_x], &[b, i, h, 0], bi));
        s
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, extra_scale);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.ps.read_weight(&self.gpu, name)
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write(self.w(name), bytemuck::cast_slice(data));
    }

    /// Reconstruct the current batch: returns the model output `[B*in_dim]`.
    pub fn reconstruct(&self, inputs: &[f32]) -> Vec<f32> {
        self.set_batch(inputs, inputs);
        self.forward_submit();
        self.gpu.poll_wait();
        self.gpu.read(&self.out, (self.b * self.cfg.in_dim) as usize)
    }

    pub fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .ps
            .params
            .iter()
            .map(|(name, _)| (name.clone(), vec![self.ps.numel(name) as u64], self.read_weight(name)))
            .collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
    }
}

// ---- the architecture-agnostic Model seam (ADR 0001 §2.2/§2.3) ----

impl model::ModelConfig for AutoencoderConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        AutoencoderConfig::param_list(self)
    }
    fn to_json(&self) -> Value {
        AutoencoderConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        AutoencoderConfig::from_json(v)
    }
    /// No token head: vocab is meaningless. Report 0.
    fn vocab(&self) -> u32 {
        0
    }
    /// `block_size` carries the flattened input width for the generic trainer.
    fn block_size(&self) -> u32 {
        self.in_dim
    }
    /// The autoencoder's input width is fixed by its own config (it is not a
    /// token model), so dataset vocab/block_size do not reshape it.
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> Self {
        self
    }
}

impl model::Model for Autoencoder {
    type Config = AutoencoderConfig;

    fn new(cfg: AutoencoderConfig, b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Autoencoder::new(cfg, b, init)
    }

    fn init_weights(cfg: &AutoencoderConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        crate::init::init_weights(cfg, seed)
    }

    fn config(&self) -> &AutoencoderConfig {
        &self.cfg
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Tensor { inputs, targets, .. } => Autoencoder::set_batch(self, inputs, targets),
            _ => panic!("autoencoder::Autoencoder only supports Batch::Tensor"),
        }
    }

    fn forward(&self) -> f32 {
        Autoencoder::forward(self)
    }
    fn backward(&self) {
        Autoencoder::backward(self)
    }
    fn zero_grads(&self) {
        Autoencoder::zero_grads(self)
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        Autoencoder::adamw_step(self, t, lr, wd, clip, extra_scale)
    }

    fn poll_wait(&self) {
        Autoencoder::poll_wait(self)
    }

    fn param_names(&self) -> Vec<String> {
        self.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        Autoencoder::read_weight(self, name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        Autoencoder::write_weight(self, name, data)
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        Autoencoder::read_grad(self, name)
    }

    /// Pure regression autoencoder: no token-classification head.
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }

    fn save(&self, path: &str) {
        Autoencoder::save(self, path)
    }
    fn config_json(&self) -> Value {
        self.cfg.to_json()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    #[test]
    fn param_list_shapes() {
        let cfg = AutoencoderConfig { in_dim: 12, hidden: 16, z_dim: 4 };
        let m: HashMap<_, _> = cfg.param_list().into_iter().collect();
        assert_eq!(m["enc.weight"], 16 * 12);
        assert_eq!(m["bottleneck.weight"], 4 * 16);
        assert_eq!(m["dec.weight"], 16 * 4);
        assert_eq!(m["out.weight"], 12 * 16);
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = AutoencoderConfig { in_dim: 20, hidden: 32, z_dim: 6 };
        let back = AutoencoderConfig::from_json(&cfg.to_json());
        assert_eq!(back.in_dim, 20);
        assert_eq!(back.hidden, 32);
        assert_eq!(back.z_dim, 6);
    }

    #[test]
    fn forward_finite_and_deterministic() {
        if gpu_disabled() {
            return;
        }
        let cfg = AutoencoderConfig::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let model = Autoencoder::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 3, &init);
        let x: Vec<f32> = (0..(3 * cfg.in_dim)).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        model.set_batch(&x, &x);
        let l1 = model.forward();
        let l2 = model.forward();
        assert!(l1.is_finite() && l1 > 0.0, "loss {l1}");
        assert!((l1 - l2).abs() < 1e-6, "not deterministic");
    }

    #[test]
    fn overfit_reduces_reconstruction_loss() {
        if gpu_disabled() {
            return;
        }
        let cfg = AutoencoderConfig { in_dim: 8, hidden: 16, z_dim: 4 };
        let init = crate::init::init_weights(&cfg, 11);
        let model = Autoencoder::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), 4, &init);
        let x: Vec<f32> = (0..(4 * cfg.in_dim)).map(|i| ((i * 3 % 11) as f32 - 5.0) * 0.15).collect();
        model.set_batch(&x, &x);
        let before = model.forward();
        for step in 1..=100 {
            model.zero_grads();
            model.forward();
            model.backward();
            model.adamw_step(step, 5e-3, 0.0, Some(1.0), 1.0);
            model.poll_wait();
        }
        let after = model.forward();
        assert!(after < before * 0.5, "autoencoder did not learn: {before} -> {after}");
    }
}
