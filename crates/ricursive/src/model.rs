// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Ripple prefill forward: embedding, the layer stack streamed one layer at
//! a time, the final norm and the untied `lm_head`.
//!
//! Swedish Embedded AB implements solutions for running very large decoders
//! within a fixed memory budget for its clients. If your team needs expertise
//! in layer-streamed inference on CPU and GPU then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The pieces are public so a serving engine can drive them itself: [`embed`]
//! (host gather, scaled), [`crate::Layer`] (one layer, explicit type and
//! window), [`Head`] (final norm and logits). [`RicursiveModel::forward`] is
//! the composition used by the parity tests; it keeps one layer's weights
//! resident at a time, so a model larger than device memory still runs.
//!
//! The value-residual source `x0` is the scaled embedding output, which every
//! layer reads unchanged.

use checkpoint::TensorSource;
use gpu_core::{f, DeviceBuffer, Gpu, Transient};
use model::ops::{Ops, TierPolicy, Weight};
use model::rope_table::{half_split_table, DeviceRope};
use paramstore::upload::Uploader;

use crate::config::{LayerType, RicursiveConfig};
use crate::layer::Layer;
use crate::ops::{self, pipelines, K_RMSNORM};
use crate::weights::Loader;

/// `embed_tokens(ids) * embed_scale`, `[ids.len(), hidden]`.
///
/// Reads only the rows it needs (a bounded range read per token), never the
/// whole table. An id outside the vocabulary is an error, not a clamp.
pub fn embed(src: &dyn TensorSource, cfg: &RicursiveConfig, ids: &[u32]) -> Result<Vec<f32>, String> {
    let h = cfg.hidden_size;
    let mut out = vec![0f32; ids.len() * h];
    for (row, &id) in ids.iter().enumerate() {
        if id as usize >= cfg.vocab_size {
            return Err(format!("ricursive: token id {id} at position {row} is outside the vocabulary of {}", cfg.vocab_size));
        }
        if !checkpoint::read_range_into(src, "model.embed_tokens.weight", id as usize * h, &mut out[row * h..(row + 1) * h]) {
            return Err("ricursive: source has no tensor model.embed_tokens.weight (or it is shorter than the vocabulary)".into());
        }
    }
    if let Some(scale) = cfg.embed_scale {
        out.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(out)
}

/// Final `norm` and `lm_head`, weights resident.
pub struct Head {
    ops: Ops,
    hidden: u32,
    vocab: u32,
    eps: f32,
    norm: DeviceBuffer,
    lm_head: Weight,
}

/// The head's outputs for each input row.
pub struct HeadOutput {
    /// `norm(x)`, `[rows, hidden]`.
    pub final_hidden: Vec<f32>,
    /// `[rows, vocab]`.
    pub logits: Vec<f32>,
}

impl Head {
    pub fn on<'g>(gpu: &'g Gpu, cfg: &RicursiveConfig, src: &dyn TensorSource, tiers: &TierPolicy) -> Result<Transient<'g, Head>, String> {
        let ops = Ops::new(gpu.share())?;
        let norm = Uploader::new(ops.gpu()).tensor(src, "model.norm.weight", cfg.hidden_size)?;
        let lm_head = Loader::new(&ops, src, tiers).linear("lm_head.weight", cfg.vocab_size, cfg.hidden_size)?;
        src.advise_drop("lm_head.weight");
        Ok(Transient::on(gpu, Head { ops, hidden: cfg.hidden_size as u32, vocab: cfg.vocab_size as u32, eps: cfg.rms_norm_eps, norm, lm_head }))
    }

    /// Normalize `x` (`[rows, hidden]`) and project it to the vocabulary.
    pub fn forward(&self, x: &[f32], rows: usize) -> HeadOutput {
        let n = rows * self.hidden as usize;
        assert_eq!(x.len(), n, "activations must be [rows, hidden]");
        let rows = rows as u32;
        let gpu = self.ops.gpu();
        let x_buf = gpu.storage(n as u64);
        gpu.write_f32(&x_buf, x);
        let normed = gpu.storage(n as u64);
        let mut steps = vec![gpu.step(K_RMSNORM, &[&x_buf, &self.norm, &normed], &[self.hidden, rows, f(self.eps)], rows)];
        let logits = ops::linear(&self.ops, &mut steps, &normed, &self.lm_head, rows);
        gpu.submit(&[], &steps);
        HeadOutput { final_hidden: gpu.read(&normed, n), logits: gpu.read(&logits, rows as usize * self.vocab as usize) }
    }
}

/// Everything a parity test bisects with.
pub struct Forward {
    /// Scaled embeddings, `[t, hidden]` (also `x0`).
    pub embeddings: Vec<f32>,
    /// Each layer's residual-stream output, `[t, hidden]` per layer.
    pub layer_outputs: Vec<Vec<f32>>,
    /// Each layer's `o_proj` output before its post-attention norm.
    pub attn_outputs: Vec<Vec<f32>>,
    /// After the final norm, `[t, hidden]`.
    pub final_hidden: Vec<f32>,
    /// `[t, vocab]`.
    pub logits: Vec<f32>,
}

/// A model bound to a checkpoint source and a device selection.
pub struct RicursiveModel<'s> {
    cfg: RicursiveConfig,
    src: &'s dyn TensorSource,
    device: Option<String>,
    tiers: TierPolicy,
}

impl<'s> RicursiveModel<'s> {
    /// `src` serves the canonical tensor names of `cfg` (see
    /// [`crate::open_safetensors`]). `device` is a `--device` selection;
    /// `None` takes the process default.
    pub fn new(cfg: RicursiveConfig, src: &'s dyn TensorSource, device: Option<&str>) -> RicursiveModel<'s> {
        RicursiveModel { cfg, src, device: device.map(str::to_string), tiers: TierPolicy::uniform(gpu_core::select::Dtype::F32) }
    }

    /// Hold the linears at the tiers `tiers` gives them instead of f32, the reference this model is by default
    /// ([`crate::resolve_tiers`] gives the tiers a checkpoint stores).
    pub fn with_tiers(mut self, tiers: TierPolicy) -> RicursiveModel<'s> {
        self.tiers = tiers;
        self
    }

    pub fn config(&self) -> &RicursiveConfig {
        &self.cfg
    }

    /// Prefill over `ids`: every position's logits.
    pub fn forward(&self, ids: &[u32]) -> Result<Forward, String> {
        let cfg = &self.cfg;
        if ids.is_empty() {
            return Err("ricursive: forward needs at least one token".into());
        }
        let t = ids.len();
        let embeddings = embed(self.src, cfg, ids)?;

        let gpu = Gpu::open(self.device.as_deref(), &pipelines(cfg, &[]));
        let rope = cfg.layer_types.contains(&LayerType::Sliding).then(|| DeviceRope::upload(&gpu, &half_split_table(cfg.head_dim, cfg.rope_theta, t)));

        let mut x = embeddings.clone();
        let mut layer_outputs = Vec::with_capacity(cfg.num_hidden_layers);
        let mut attn_outputs = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let started = std::time::Instant::now();
            let layer = Layer::on(&gpu, cfg, self.src, l, &self.tiers)?;
            let built = started.elapsed();
            let out = layer.forward(&x, &embeddings, rope.as_ref(), t);
            tracing::debug!(layer = l, kind = ?layer.kind(), build_ms = built.as_millis() as u64, total_ms = started.elapsed().as_millis() as u64, "ricursive layer");
            x = out.hidden.clone();
            layer_outputs.push(out.hidden);
            attn_outputs.push(out.attn_out);
        }

        let head = Head::on(&gpu, cfg, self.src, &self.tiers)?;
        let HeadOutput { final_hidden, logits } = head.forward(&x, t);
        Ok(Forward { embeddings, layer_outputs, attn_outputs, final_hidden, logits })
    }
}
