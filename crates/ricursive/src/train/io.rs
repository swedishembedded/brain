// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`model::Model`] face of the Ripple trainer: batches, the loss, the optimizer, parameter access and the
//! checkpoint it saves and reloads.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, from full-weight training to adapters over a 4-bit base. If your team needs expertise in
//! training hybrid-attention language models then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashMap;

use model::saved::SavedPolicy;

use super::{Ripple, TrainConfig, TrainOpts};

impl Ripple {
    /// The tensors a saved checkpoint holds, `(name, shape, values)`: every tensor the store owns. A build whose base
    /// linears are held packed owns only the adapter and the norms, embeddings and gates: its base is the source
    /// checkpoint it was built over, and the file says so.
    pub(super) fn saved_tensors(&self) -> Vec<(String, Vec<u64>, Vec<f32>)> {
        // A resident build's linears and tables are saved from their master copies.
        let resident: Vec<(String, usize)> = self.resident.as_ref().map(|ft| ft.borrow().specs().iter().map(|s| (s.name.clone(), s.n as usize * s.k as usize)).collect()).unwrap_or_default();
        self.ps.params.iter().cloned().chain(resident).map(|(name, numel)| (name.clone(), vec![numel as u64], Model::read_weight(self, &name))).collect()
    }

    /// The linears this build holds outside the store, at a packed tier.
    pub(super) fn external_base(&self) -> bool {
        self.cfg.shard_param_list(&self.shard).iter().any(|(name, _)| !self.ps.weight.contains_key(name) && self.resident_index(name).is_none())
    }

    /// The logits of the first `rows` positions of the batch last run forward. A training build holds one chunk of
    /// them at a time, so they are computed here chunk by chunk; an inference build holds them all.
    pub(super) fn read_logit_rows(&self, rows: usize, vocab: usize) -> Vec<f32> {
        let Some(head) = &self.head_ce else {
            let mut out = self.gpu.read(&self.logits, self.rows() as usize * vocab);
            out.truncate(rows * vocab);
            return out;
        };
        let mut out = Vec::with_capacity(rows * vocab);
        for (r0, n) in head.chunks(rows as u32) {
            let mut steps = Vec::new();
            head.logits_steps(&self.ops, &mut steps, &self.hs, r0, n, self.logit_scale);
            self.gpu.submit(&[], &steps);
            out.extend(self.gpu.read(head.logits(), n as usize * vocab));
        }
        out
    }

    /// Rebuild a model from a file written by [`Model::save`], sized for `(b, t)`. The file is read one tensor at a
    /// time (never the whole model as f32 on the host); AdamW moments restart, exactly as for a fresh optimizer.
    pub fn load(path: &str, b: u32, t: u32) -> Ripple {
        let reader = checkpoint::weightio::WeightReader::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
        let header = reader.config();
        let cfg = TrainConfig::from_json(&header).unwrap_or_else(|e| panic!("{path}: {e}"));
        if header.get("base").and_then(|v| v.as_str()) == Some("external") {
            panic!("{path}: this file holds an adapter over a packed base, not the base; build over the base checkpoint and load the adapter from it");
        }
        Ripple::build(cfg, b, t, &reader, &TrainOpts::default(), true).unwrap_or_else(|e| panic!("{path}: {e}"))
    }
}

use model::Model;

impl Model for Ripple {
    type Config = TrainConfig;

    fn new(cfg: TrainConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Ripple {
        Ripple::build(cfg, b, t, init, &TrainOpts::default(), true).unwrap_or_else(|e| panic!("{e}"))
    }

    fn init_weights(cfg: &TrainConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        cfg.init_weights(seed)
    }

    /// A stage without a global layer has nothing to sparsify; a model with none at all refuses by name.
    fn enable_native_sparse_attention(&mut self, sparse: model::NativeSparse) -> Result<(), String> {
        let c = &self.cfg.model;
        if !(0..c.num_hidden_layers).any(|l| c.layer_type(l) == crate::LayerType::Full) {
            return Err("ricursive: this model has no global-attention layer for native sparse attention to train".to_string());
        }
        if !(self.shard.start..self.shard.end).any(|l| c.layer_type(l) == crate::LayerType::Full) {
            return Ok(());
        }
        self.enable_native_lsa(sparse.cfg, sparse.long_range)
    }

    fn token_objectives(&self) -> Option<&dyn model::training::TokenObjectives> {
        Some(self)
    }

    fn token_objectives_mut(&mut self) -> Option<&mut dyn model::training::TokenObjectives> {
        Some(self)
    }

    fn config(&self) -> &TrainConfig {
        &self.cfg
    }

    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }

    fn set_batch(&self, batch: model::Batch) {
        match batch {
            model::Batch::Lm { tokens, targets } => {
                self.set_tokens_targets(tokens, targets);
                // A weighted build keeps its weights from the last weighted batch: a plain batch weighs every position one.
                if let Some(w) = &self.weighted {
                    w.write(&self.gpu, &vec![1.0; self.rows() as usize]);
                }
            }
            model::Batch::LmWeighted { tokens, targets, weights } => {
                let w = self.weighted.as_ref().expect("ricursive: Batch::LmWeighted needs enable_weighted_loss() first");
                self.set_tokens_targets(tokens, targets);
                w.write(&self.gpu, weights);
            }
            _ => panic!("ricursive: only Batch::Lm and Batch::LmWeighted are supported"),
        }
    }

    fn forward(&self) -> f32 {
        assert!(self.train, "forward on an inference-only build has no objective");
        self.run_forward();
        let rows = self.gpu.read(&self.ce_rows, self.rows() as usize);
        match &self.weighted {
            // The scalar the weighted gradient differentiates.
            Some(w) => w.loss(&self.gpu, &rows, self.count.get()),
            None => rows.iter().sum::<f32>() / self.count.get(),
        }
    }

    fn backward(&self) {
        assert!(self.train, "backward on an inference-only build");
        self.run_backward();
    }

    fn zero_grads(&self) {
        self.backward_pending.set(false);
        self.ps.zero_grads(&self.gpu);
        if let Some(ft) = &self.resident {
            self.gpu.flush();
            ft.borrow().zero_grads(&self.ops);
        }
    }

    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        // A resident build updates its linears and tables against their host masters through the generic resident
        // step, whose global clip spans them and the norms; that step also updates the norms.
        if let Some(ft) = &self.resident {
            assert!(!self.layer_streamed() || self.backward_pending.replace(false), "ricursive layer-streamed full fine-tune: adamw_step without a backward before it");
            let args = model::resident_ft::StepArgs { step: t, lr, wd, adam, clip, extra_scale };
            let norm = model::resident_ft::adamw_step(self, ft, &self.ops, &args).unwrap_or_else(|e| panic!("ricursive resident full fine-tune: update: {e}"));
            self.resident_norm.set(norm);
            return;
        }
        let opt = self.opt.as_ref().expect("adamw_step on an inference-only build");
        opt.step(&self.gpu, &self.ps, t, lr, wd, adam, clip, extra_scale);
    }

    /// A resident build that keeps every layer on the device updates itself from the global norm and multiplier a
    /// pipeline decides; a layer-streamed one runs its backward inside its update and cannot.
    fn stage_local_optimizer(&self) -> bool {
        self.resident.is_some() && !self.layer_streamed()
    }

    fn local_grad_sum_sq(&self) -> Option<f64> {
        use model::resident_ft::ResidentBackward;
        let ft = self.resident.as_ref().filter(|_| !self.layer_streamed())?;
        self.gpu.flush();
        let resident = ft.borrow_mut().grad_sum_sq(&self.ops).ok()?;
        Some(resident + self.remainder_grad_sum_sq(ft.borrow().device_norm()))
    }

    fn local_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, multiplier: f32) -> Result<(), String> {
        use model::resident_ft::ResidentBackward;
        let ft = self.resident.as_ref().filter(|_| !self.layer_streamed()).ok_or("ricursive: no stage-local optimizer on this build")?;
        self.gpu.flush();
        {
            let mut ft = ft.borrow_mut();
            ft.set_rule(optim::rule::Rule::AdamW { adam, wd });
            ft.step_scaled(&self.ops, lr, multiplier).map_err(|e| format!("ricursive resident full fine-tune: update: {e}"))?;
        }
        self.update_remainder(&model::resident_ft::StepArgs { step: t, lr, wd, adam, clip: None, extra_scale: 1.0 }, multiplier);
        Ok(())
    }

    fn off_device_state(&self) -> Option<&dyn model::OffDeviceState> {
        self.resident.is_some().then_some(self as &dyn model::OffDeviceState)
    }

    fn last_grad_norm(&self) -> Option<f32> {
        self.resident.as_ref().and_then(|_| self.resident_norm.get())
    }

    fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    fn pin_rounding_seed(&self, seed: u64) {
        if let Some(opt) = &self.opt {
            opt.set_rounding_seed(seed ^ 0x5eed_0000_1234_9876);
        }
    }

    fn optimizer_state(&mut self) -> Option<model::OptimizerState<'_>> {
        Some(model::OptimizerState { gpu: &self.gpu, store: &mut self.ps, optim: self.opt.as_ref()? })
    }

    fn optimizer_parts(&self) -> Option<model::OptimizerParts<'_>> {
        Some(model::OptimizerParts { gpu: &self.gpu, store: &self.ps, optim: self.opt.as_ref()? })
    }

    /// The residual between layers is the only thing the forward keeps, in f32 or (chosen when the model is built)
    /// narrowed; keeping every layer's intermediates is refused by name.
    fn set_saved_policy(&mut self, policy: SavedPolicy) -> Result<(), String> {
        let built = SavedPolicy::recompute_with(self.res.format());
        if !built.narrows_to(policy) {
            return Err(format!("ricursive recomputes each layer from its {:?} residual in the backward; {policy:?} is not available", self.res.format()));
        }
        if let SavedPolicy::Quantized(format) = policy {
            if policy != built {
                self.res.quantize(&self.gpu, format, self.rows(), self.cfg.model.hidden_size as u32)?;
                self.record();
            }
        }
        Ok(())
    }

    /// The optimised set: the weights of a full fine-tune, the adapter's factors under an adapter.
    fn param_names(&self) -> Vec<String> {
        let resident: Vec<String> = self.resident.as_ref().map(|ft| ft.borrow().specs().iter().map(|s| s.name.clone()).collect()).unwrap_or_default();
        self.ps.trained_params().map(|(n, _)| n.clone()).chain(resident).collect()
    }

    fn frozen_adapter_param_names(&self) -> Vec<String> {
        model::adapter::device::frozen_adapter_params(&self.ps)
    }

    fn adapter_sites(&self) -> Vec<model::adapter::LinearSite> {
        model::adapter::sites::LoraSites::lora_sites(&self.cfg)
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        if let Some(i) = self.resident_index(name) {
            // The master copy: the highest-precision weight, which is what a saved checkpoint should carry.
            self.gpu.flush();
            return self.resident.as_ref().expect("indexed").borrow_mut().read_master(&self.ops, i).unwrap_or_else(|e| panic!("ricursive resident full fine-tune: reading {name}: {e}"));
        }
        self.ps.read_weight(&self.gpu, name)
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        if let Some(i) = self.resident_index(name) {
            self.gpu.flush();
            self.resident.as_ref().expect("indexed").borrow_mut().write_master(&self.ops, i, data).unwrap_or_else(|e| panic!("ricursive resident full fine-tune: writing {name}: {e}"));
            return;
        }
        assert!(self.ps.weight.contains_key(name), "ricursive: {name} is not a tensor of this build's parameter store");
        self.gpu.write_f32(self.ps.w(name), data);
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        if let Some(grad) = self.read_resident_grad(name) {
            assert!(!self.layer_streamed(), "ricursive: {name} streams through a layer window; no gradient outlives its layer");
            return grad;
        }
        self.ps.read_grad(&self.gpu, name)
    }

    fn write_grad(&self, name: &str, data: &[f32]) -> Result<(), String> {
        if let Some(i) = self.resident_index(name) {
            // The accumulating buffer of the resident state, in its own dtype: what a pipeline writes back after summing
            // the gradient of a tensor several stages hold.
            self.gpu.flush();
            self.resident.as_ref().expect("indexed").borrow().write_grad(&self.ops, i, data);
            return Ok(());
        }
        if !self.is_trainable(name) {
            return Err(format!("{name}: not a trainable parameter of this ricursive model, so it has no gradient to reduce"));
        }
        self.ps.write_grad(&self.gpu, name, data);
        Ok(())
    }

    /// Logits for every position of one sequence (`b` must be 1 and the sequence no longer than `t`; a shorter one
    /// is padded and the padding sliced off, which causal attention makes exact).
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        let n = self.rows() as usize;
        let rows = tokens.len();
        assert!(self.b == 1 && rows <= self.t as usize, "ricursive logits_all: a single sequence of at most {} tokens, got {rows} on {} rows", self.t, self.b);
        let mut padded = tokens.to_vec();
        padded.resize(n, 0);
        self.set_tokens_targets(&padded, &vec![super::IGNORE; n]);
        self.run_forward();
        let vocab = self.cfg.model.vocab_size;
        Some(self.read_logit_rows(rows, vocab))
    }

    fn save(&self, path: &str) {
        let card = checkpoint::st::ModelCard::new("brain/ricursive", "ricursive");
        let mut config = self.cfg.to_json();
        if self.external_base() {
            config["base"] = serde_json::Value::String("external".into());
        }
        checkpoint::save_carded(path, config, &self.saved_tensors(), &card);
    }
}
