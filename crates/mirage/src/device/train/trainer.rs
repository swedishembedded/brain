// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable GPU fine-tuning of compact perception models for its clients. If your
// team needs expertise in full, head-only and low-rank fine-tuning with exact resume on accelerators then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The device trainer: the host trainer's modes (full, head-only, LoRA), settings and files, evaluated on a [`Gpu`].
//!
//! Parameters live in a [`ParamStore`]; each image's loss graph is built once on a [`DevTape`] and differentiated on the
//! device, gradients accumulate over the batch in the store, and the shared device AdamW (`optim::Optim`) applies the
//! mean gradient with the global clip. A run directory is exactly the host trainer's (`mirage.safetensors` in
//! checkpoint layout, `adapter.safetensors`, `state.safetensors` with the moments by checkpoint name), so a run can be
//! resumed on either evaluator and served unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use model::adapter::{kind, KeyStyle};
use model::saved::SavedPolicy;
use optim::{Adam, Optim};
use paramstore::{ParamStore, Role};

use super::params::{is_stacked, stacked_weights, upload_params};
use super::DevTape;
use crate::device::Device;
use crate::import::{Checkpoint, Metadata};
use crate::spec::{is_training_only, Manifest};
use crate::train::adapter;
use crate::train::graph::Graph;
use crate::train::net::{params_from_checkpoint, AdapterSet, Net, Params};
use crate::train::objective::sample_loss;
use crate::train::trainer::{FineTuner, Mode, Prepared, StepReport, TrainConfig};

/// The initial adapter and the store entries of each projection key's pair.
type Lora = (AdapterSet<f32>, BTreeMap<String, (String, String)>);

/// The device trainer's state.
pub struct DevTrainer {
    dev: Device,
    base: Checkpoint,
    store: ParamStore,
    optim: Optim,
    /// Base tensors a full or head-only run trains (empty for LoRA).
    trainable: BTreeSet<String>,
    /// LoRA pairs: projection key -> (A entry, B entry); `None` outside LoRA runs.
    lora: Option<Lora>,
    cfg: TrainConfig,
    step: u32,
    origin_step: u64,
}

impl DevTrainer {
    /// Start from a checkpoint on `dev`.
    pub fn new(dev: Device, base: Checkpoint, cfg: TrainConfig) -> Result<DevTrainer, String> {
        let mut params = params_from_checkpoint::<f32>(&base);
        params.retain(|name, _| !is_training_only(name));
        let origin_step = base.metadata.step;
        let manifest: Vec<String> = Manifest::new(&base.config).iter().filter(|s| !is_training_only(&s.name)).map(|s| s.name.clone()).collect();
        let (trainable, lora) = match &cfg.mode {
            Mode::Full => (manifest.into_iter().collect::<BTreeSet<_>>(), None),
            Mode::Heads(heads) => {
                if heads.is_empty() {
                    return Err("a head-only run needs at least one head group".into());
                }
                (manifest.into_iter().filter(|n| heads.iter().any(|h| h.trains(n))).collect(), None)
            }
            Mode::Lora { rank, alpha, seed } => {
                if *rank == 0 {
                    return Err("a LoRA run needs a positive rank".into());
                }
                let set = adapter::fresh(&base.config, *rank, *alpha, *seed);
                let mut names = BTreeMap::new();
                for (key, (a, b)) in &set.pairs {
                    let (an, bn) = (KeyStyle::Brain.format(key, kind::LORA_A), KeyStyle::Brain.format(key, kind::LORA_B));
                    params.insert(an.clone(), a.clone());
                    params.insert(bn.clone(), b.clone());
                    names.insert(key.clone(), (an, bn));
                }
                (BTreeSet::new(), Some((set, names)))
            }
        };
        let adapter_names: BTreeSet<String> = lora.iter().flat_map(|(_, names)| names.values().flat_map(|(a, b)| [a.clone(), b.clone()])).collect();
        let store = upload_params(&dev, base.config.dec_layers, &params, |name| if trainable.contains(name) || adapter_names.contains(name) { Role::Trainable } else { Role::Frozen });
        let idx = |n: &str| dev.gpu().kernel_index(n).ok_or_else(|| format!("the device lacks kernel `{n}`"));
        let optim = Optim::new(idx("adamw")?, idx("gradnorm_sq")?, idx("grad_scale")?, idx("clip_coef")?, idx("grad_scale_buf")?);
        Ok(DevTrainer { dev, base, store, optim, trainable, lora, cfg, step: 0, origin_step })
    }

    /// Change the learning rate for the steps that follow.
    pub fn set_lr(&mut self, lr: f32) {
        self.cfg.lr = lr;
    }

    /// Keep what `policy` names of the forward for the backward ([`TrainConfig::saved`]); the gradients do not
    /// depend on it. A narrow boundary format is refused: the recompute boundaries stay in full precision.
    pub fn set_saved_policy(&mut self, policy: SavedPolicy) -> Result<(), String> {
        crate::train::net::check_saved_policy(policy)?;
        self.cfg.saved = policy;
        Ok(())
    }

    /// The names of the trained base tensors (empty for LoRA).
    pub fn trainable(&self) -> &BTreeSet<String> {
        &self.trainable
    }

    fn tape(&self) -> DevTape<'_> {
        let mut tape = DevTape::new(&self.dev, &self.store);
        if let Some((set, names)) = &self.lora {
            for (key, (a, b)) in names {
                tape = tape.with_lora(key, a, b, set.rank, f64::from(set.scale));
            }
        }
        tape
    }

    /// Mean loss of `batch` without a step (the evaluation path; the same graph the step differentiates).
    pub fn eval_loss(&self, batch: &[Prepared]) -> f64 {
        let (mut total, mut n) = (0.0, 0usize);
        for p in batch {
            let mut net = Net::resident(&self.base.config, self.tape().without_grads());
            if let Some(l) = sample_loss(&mut net, &p.sample, p.target.as_ref(), &self.cfg.weights) {
                total += net.tape.scalar(l.total);
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            total / n as f64
        }
    }

    /// One optimiser step on the mean gradient of `batch` (gradients accumulate over the images in the store).
    pub fn step(&mut self, batch: &[Prepared]) -> Result<StepReport, String> {
        self.store.zero_grads(self.dev.gpu());
        let mut terms: BTreeMap<&'static str, f64> = BTreeMap::new();
        let (mut loss, mut samples) = (0.0f64, 0usize);
        for p in batch {
            let mut net = Net::resident(&self.base.config, self.tape()).with_saved_policy(self.cfg.saved)?;
            let Some(l) = sample_loss(&mut net, &p.sample, p.target.as_ref(), &self.cfg.weights) else { continue };
            loss += net.tape.scalar(l.total);
            l.terms.iter().for_each(|(k, v)| *terms.entry(k).or_insert(0.0) += v);
            net.backward(l.total);
            samples += 1;
        }
        if samples == 0 {
            return Err("no sample in the batch carries a label any active loss term can use".into());
        }
        self.step += 1;
        let inv = 1.0 / samples as f32;
        self.optim.step(self.dev.gpu(), &self.store, self.step, self.cfg.lr, self.cfg.weight_decay, Adam::default(), self.cfg.clip, inv);
        let grad_norm = self.optim.last_grad_norm(self.dev.gpu(), &self.store).map_or(0.0, f64::from);
        Ok(StepReport { loss: loss / samples as f64, terms: terms.into_iter().map(|(k, v)| (k, v / samples as f64)).collect(), grad_norm, samples })
    }

    /// The checkpoint with the trained tensors, in checkpoint layout (a LoRA run returns the base).
    pub fn checkpoint(&self) -> Checkpoint {
        let mut ck = self.base.clone();
        if !self.trainable.is_empty() {
            let weights = stacked_weights(&self.dev, &self.store);
            for name in &self.trainable {
                if let Some(t) = ck.tensors.get_mut(name) {
                    t.data.clone_from(&weights[name]);
                }
            }
        }
        let trained = if self.lora.is_some() { 0 } else { u64::from(self.step) };
        ck.metadata = Metadata { step: self.origin_step + trained, ..ck.metadata };
        ck
    }

    /// The adapter as trained so far (LoRA runs).
    pub fn adapter(&self) -> Option<AdapterSet<f32>> {
        let (set, names) = self.lora.as_ref()?;
        let gpu = self.dev.gpu();
        let pairs = names.iter().map(|(key, (a, b))| (key.clone(), (self.store.read_weight(gpu, a), self.store.read_weight(gpu, b)))).collect();
        Some(AdapterSet { scale: set.scale, rank: set.rank, pairs })
    }

    /// The checkpoint with the adapter folded in (the weights the serving path would run).
    pub fn merged(&self) -> Result<Checkpoint, String> {
        let mut ck = self.checkpoint();
        if let Some(set) = self.adapter() {
            adapter::fold(&mut ck, &set)?;
        }
        Ok(ck)
    }

    /// Store entries and checkpoint-layout names of the optimised tensors: `(name, entries in layer order)`.
    fn moment_groups(&self) -> Vec<(String, Vec<String>)> {
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (key, _) in self.store.trained_params() {
            let name = match key.rsplit_once('#') {
                Some((n, l)) if l.parse::<usize>().is_ok() => n.to_string(),
                _ => key.clone(),
            };
            groups.entry(name).or_default().push(key.clone());
        }
        // Layer order: numeric, not lexicographic.
        for entries in groups.values_mut() {
            entries.sort_by_key(|k| k.rsplit_once('#').and_then(|(_, l)| l.parse::<usize>().ok()).unwrap_or(0));
        }
        groups.into_iter().collect()
    }

    /// Write the run to `dir` in the host trainer's layout (weights, adapter, `state.safetensors`).
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        crate::import::export(&self.checkpoint(), &dir.join("mirage.safetensors")).map_err(|e| e.to_string())?;
        if let Some(set) = self.adapter() {
            adapter::save(&dir.join(adapter::FILE), &set, &self.base.config)?;
        }
        let mut tensors = Vec::new();
        for (name, entries) in self.moment_groups() {
            let (mut m, mut v) = (Vec::new(), Vec::new());
            for e in &entries {
                let (em, ev) = optim::read_moments(self.dev.gpu(), &self.store, e).ok_or_else(|| format!("`{e}` has no device moments"))?;
                m.extend(em);
                v.extend(ev);
            }
            tensors.push((format!("{name}.m"), vec![m.len() as u64], m));
            tensors.push((format!("{name}.v"), vec![v.len() as u64], v));
        }
        checkpoint::st::save_safetensors(dir.join("state.safetensors").to_str().ok_or("path is not UTF-8")?, &tensors, &serde_json::json!({"model": "mirage-train-state", "step": self.step}), None).map_err(|e| e.to_string())
    }

    /// Resume from a directory written by either trainer's `save`.
    pub fn resume(dev: Device, dir: &Path, cfg: TrainConfig) -> Result<DevTrainer, String> {
        let base = crate::import::load(&dir.join("mirage.safetensors"), &crate::import::ImportOptions::default()).map_err(|e| e.to_string())?;
        let loaded_step = base.metadata.step;
        let mut t = DevTrainer::new(dev, base, cfg)?;
        let gpu = t.dev.gpu();
        if let Some((set, names)) = t.lora.as_mut() {
            let loaded = adapter::load(&dir.join(adapter::FILE), &t.base.config)?;
            for (key, (a, b)) in &loaded.pairs {
                let (an, bn) = names.get(key).ok_or_else(|| format!("the adapter has a pair `{key}` this run does not"))?;
                gpu.write_f32(t.store.w(an), a);
                gpu.write_f32(t.store.w(bn), b);
            }
            *set = loaded;
        }
        let state_path = dir.join("state.safetensors");
        let path = state_path.to_str().ok_or("path is not UTF-8")?;
        let meta = checkpoint::st::read_metadata(path).map_err(|e| e.to_string())?;
        let config: serde_json::Value = meta.get(checkpoint::st::CONFIG_KEY).ok_or("state.safetensors has no config header").and_then(|s| serde_json::from_str(s).map_err(|_| "unreadable state header"))?;
        t.step = config.get("step").and_then(serde_json::Value::as_u64).ok_or("state.safetensors has no step")? as u32;
        t.origin_step = if t.lora.is_some() { loaded_step } else { loaded_step.saturating_sub(u64::from(t.step)) };
        let tensors: BTreeMap<String, Vec<f32>> = checkpoint::safetensors::read(path)?.into_iter().map(|x| (x.name, x.data)).collect();
        for (name, entries) in t.moment_groups() {
            let (Some(m), Some(v)) = (tensors.get(&format!("{name}.m")), tensors.get(&format!("{name}.v"))) else { continue };
            let unit = m.len() / entries.len();
            if m.len() != v.len() || unit * entries.len() != m.len() || (!is_stacked(&name) && entries.len() != 1) {
                return Err(format!("state for `{name}` holds {} and {} values, which its {} device entries do not split", m.len(), v.len(), entries.len()));
            }
            for (e, (pm, pv)) in entries.iter().zip(m.chunks_exact(unit).zip(v.chunks_exact(unit))) {
                if unit != t.store.numel(e) {
                    return Err(format!("state for `{name}` has {unit} values per entry, expected {}", t.store.numel(e)));
                }
                optim::write_moments(t.dev.gpu(), &t.store, e, pm, pv)?;
            }
        }
        Ok(t)
    }
}

impl FineTuner for DevTrainer {
    fn step(&mut self, batch: &[Prepared]) -> Result<StepReport, String> {
        DevTrainer::step(self, batch)
    }
    fn eval_loss(&self, batch: &[Prepared]) -> f64 {
        DevTrainer::eval_loss(self, batch)
    }
    fn checkpoint(&self) -> Checkpoint {
        DevTrainer::checkpoint(self)
    }
    fn merged(&self) -> Result<Checkpoint, String> {
        DevTrainer::merged(self)
    }
    fn adapter_set(&self) -> Option<AdapterSet<f32>> {
        self.adapter()
    }
    fn save(&self, dir: &Path) -> Result<(), String> {
        DevTrainer::save(self, dir)
    }
    fn steps(&self) -> u32 {
        self.step
    }
    fn set_saved_policy(&mut self, policy: SavedPolicy) -> Result<(), String> {
        DevTrainer::set_saved_policy(self, policy)
    }
}

/// Parameters in checkpoint layout, for callers that hold a store of their own.
pub fn checkpoint_params(dev: &Device, store: &ParamStore) -> Params<f32> {
    stacked_weights(dev, store)
}
