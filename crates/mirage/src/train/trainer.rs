// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable fine-tuning of compact perception models for its clients. If your team
// needs expertise in full, head-only and low-rank fine-tuning with exact resume then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The host trainer: full fine-tune, head-only (a parameter filter) and LoRA over the shared AdamW
//! (`optim::NamedMoments`), exact resume, and checkpoints the serving path loads unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use model::adapter::{kind, KeyStyle};
use optim::{grad_multiplier, Adam, NamedMoments};

use super::adapter;
use super::net::{params_from_checkpoint, AdapterSet, Net, Params};
use super::objective::{sample_loss, Sample, TextTarget, TrainWeights};
use crate::import::{Checkpoint, Metadata};
use crate::spec::{is_training_only, Manifest};

/// A group of parameters a head-only run trains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Head {
    /// The text reader.
    Reader,
    /// The dense detection head and the mask branches.
    Dense,
    /// The decoder stack, its embedding, engrams and the slot projections.
    Decoder,
}

impl Head {
    fn owns(self, name: &str) -> bool {
        match self {
            Head::Reader => name.starts_with("reader/"),
            Head::Dense => name.starts_with("dense/"),
            Head::Decoder => name.starts_with("stack/") || name.starts_with("embedding/") || name.starts_with("engrams_") || name.starts_with("slot_") || name == "pe_gate",
        }
    }
}

/// What the run trains.
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    /// Every tensor the deployment model uses.
    Full,
    /// Only the named head groups; everything else is frozen.
    Heads(Vec<Head>),
    /// Low-rank adapters on the dense projections; the base is frozen.
    Lora { rank: usize, alpha: f32, seed: u64 },
}

/// A run's settings.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// What to train.
    pub mode: Mode,
    /// Loss term weights.
    pub weights: TrainWeights,
    /// Learning rate.
    pub lr: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Global gradient-norm clip.
    pub clip: Option<f32>,
}

impl TrainConfig {
    /// A run of `mode` with the default weights, learning rate 1e-4 and clip 1.0.
    pub fn new(mode: Mode) -> TrainConfig {
        TrainConfig { mode, weights: TrainWeights::default(), lr: 1e-4, weight_decay: 0.0, clip: Some(1.0) }
    }
}

/// What one optimiser step saw.
#[derive(Debug, Clone)]
pub struct StepReport {
    /// Mean loss over the batch's samples.
    pub loss: f64,
    /// Mean of each term.
    pub terms: BTreeMap<&'static str, f64>,
    /// Gradient norm before clipping.
    pub grad_norm: f64,
    /// Samples that produced a loss.
    pub samples: usize,
}

/// One sample with its decoder target.
pub struct Prepared {
    /// The image and labels.
    pub sample: Sample,
    /// The decoder sequence, when text supervision applies.
    pub target: Option<TextTarget>,
}

/// The trainer's state.
pub struct Trainer {
    base: Checkpoint,
    params: Params<f32>,
    adapter: Option<AdapterSet<f32>>,
    trainable: BTreeSet<String>,
    moments: NamedMoments,
    cfg: TrainConfig,
    step: u32,
}

impl Trainer {
    /// Start from a checkpoint.
    pub fn new(base: Checkpoint, cfg: TrainConfig) -> Result<Trainer, String> {
        let params = params_from_checkpoint::<f32>(&base);
        let (adapter, trainable) = match &cfg.mode {
            Mode::Full => (None, Manifest::new(&base.config).iter().filter(|s| !is_training_only(&s.name)).map(|s| s.name.clone()).collect()),
            Mode::Heads(heads) => {
                if heads.is_empty() {
                    return Err("a head-only run needs at least one head group".into());
                }
                (None, Manifest::new(&base.config).iter().filter(|s| !is_training_only(&s.name) && heads.iter().any(|h| h.owns(&s.name))).map(|s| s.name.clone()).collect())
            }
            Mode::Lora { rank, alpha, seed } => {
                if *rank == 0 {
                    return Err("a LoRA run needs a positive rank".into());
                }
                (Some(adapter::fresh(&base.config, *rank, *alpha, *seed)), BTreeSet::new())
            }
        };
        Ok(Trainer { base, params, adapter, trainable, moments: NamedMoments::new(Adam::default()), cfg, step: 0 })
    }

    /// Change the learning rate for the steps that follow.
    pub fn set_lr(&mut self, lr: f32) {
        self.cfg.lr = lr;
    }

    /// Completed optimiser steps.
    pub fn steps(&self) -> u32 {
        self.step
    }

    /// The adapter (LoRA runs).
    pub fn adapter(&self) -> Option<&AdapterSet<f32>> {
        self.adapter.as_ref()
    }

    /// The names of the trained base tensors (empty for LoRA).
    pub fn trainable(&self) -> &BTreeSet<String> {
        &self.trainable
    }

    /// Mean loss of `batch` without a step (the evaluation path; the same graph the step differentiates).
    pub fn eval_loss(&self, batch: &[Prepared]) -> f64 {
        let (mut total, mut n) = (0.0, 0usize);
        for p in batch {
            let mut net = Net::<f32>::new(&self.base.config, &self.params, self.adapter.as_ref());
            if let Some(l) = sample_loss(&mut net, &p.sample, p.target.as_ref(), &self.cfg.weights) {
                total += f64::from(net.tape.val(l.total)[0]);
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            total / n as f64
        }
    }

    /// One optimiser step on the mean gradient of `batch`.
    pub fn step(&mut self, batch: &[Prepared]) -> Result<StepReport, String> {
        let mut grads: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut terms: BTreeMap<&'static str, f64> = BTreeMap::new();
        let (mut loss, mut samples) = (0.0f64, 0usize);
        for p in batch {
            let mut net = Net::<f32>::new(&self.base.config, &self.params, self.adapter.as_ref());
            let Some(l) = sample_loss(&mut net, &p.sample, p.target.as_ref(), &self.cfg.weights) else { continue };
            loss += f64::from(net.tape.val(l.total)[0]);
            l.terms.iter().for_each(|(k, v)| *terms.entry(k).or_insert(0.0) += v);
            let mut g = net.tape.backward(l.total);
            let mut sample_grads: BTreeMap<String, Vec<f32>> = if self.adapter.is_some() {
                net.adapter_grads(&mut g)
                    .into_iter()
                    .flat_map(|(key, (da, db))| [(KeyStyle::Brain.format(&key, kind::LORA_A), da), (KeyStyle::Brain.format(&key, kind::LORA_B), db)])
                    .collect()
            } else {
                net.param_grads(&mut g).into_iter().filter(|(n, _)| self.trainable.contains(n)).collect()
            };
            for (name, g) in std::mem::take(&mut sample_grads) {
                match grads.get_mut(&name) {
                    Some(acc) => acc.iter_mut().zip(&g).for_each(|(a, b)| *a += b),
                    None => {
                        grads.insert(name, g);
                    }
                }
            }
            samples += 1;
        }
        if samples == 0 {
            return Err("no sample in the batch carries a label any active loss term can use".into());
        }
        // The summed gradient is averaged by `inv`; the clip coefficient is formed on the averaged gradient.
        let inv = 1.0 / samples as f32;
        let raw_sq: f64 = grads.values().flatten().map(|g| f64::from(*g).powi(2)).sum();
        let scale = grad_multiplier(raw_sq, self.cfg.clip, inv);
        self.step += 1;
        let (lr, wd, t) = (self.cfg.lr, self.cfg.weight_decay, self.step);
        for (name, g) in &grads {
            let w: &mut Vec<f32> = if let Some(set) = self.adapter.as_mut() {
                adapter_slot(set, name, &self.base.config)?
            } else {
                self.params.get_mut(name).ok_or_else(|| format!("no parameter `{name}`"))?
            };
            self.moments.step(t, lr, wd, scale, name, w, g)?;
        }
        Ok(StepReport { loss: loss / samples as f64, terms: terms.into_iter().map(|(k, v)| (k, v / samples as f64)).collect(), grad_norm: raw_sq.sqrt() * f64::from(inv), samples })
    }

    /// The checkpoint with the trained tensors (a LoRA run returns the base; see [`Trainer::merged`]).
    pub fn checkpoint(&self) -> Checkpoint {
        let mut ck = self.base.clone();
        for name in &self.trainable {
            if let Some(t) = ck.tensors.get_mut(name) {
                t.data.clone_from(&self.params[name]);
            }
        }
        ck.metadata = Metadata { step: ck.metadata.step + u64::from(self.step), ..ck.metadata };
        ck
    }

    /// The checkpoint with the adapter folded in (the weights the serving path would run).
    pub fn merged(&self) -> Result<Checkpoint, String> {
        let mut ck = self.checkpoint();
        if let Some(set) = &self.adapter {
            adapter::fold(&mut ck, set)?;
        }
        Ok(ck)
    }

    /// Write the run to `dir`: the deployable weights (`mirage.safetensors`, the base for a LoRA run), the adapter
    /// (`adapter.safetensors`, LoRA runs) and the training state (`state.safetensors`: step and moments).
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        crate::import::export(&self.checkpoint(), &dir.join("mirage.safetensors")).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("config.json"), self.base.config.to_json()).map_err(|e| e.to_string())?;
        if let Some(set) = &self.adapter {
            adapter::save(&dir.join("adapter.safetensors"), set, &self.base.config)?;
        }
        let mut tensors = Vec::new();
        for (name, _) in self.moment_names() {
            if let Some((m, v)) = self.moments.moments(&name) {
                tensors.push((format!("{name}.m"), vec![m.len() as u64], m.to_vec()));
                tensors.push((format!("{name}.v"), vec![v.len() as u64], v.to_vec()));
            }
        }
        checkpoint::st::save_safetensors(dir.join("state.safetensors").to_str().ok_or("path is not UTF-8")?, &tensors, &serde_json::json!({"model": "mirage-train-state", "step": self.step}), None).map_err(|e| e.to_string())
    }

    fn moment_names(&self) -> Vec<(String, usize)> {
        match &self.adapter {
            Some(set) => set.pairs.iter().flat_map(|(k, (a, b))| [(KeyStyle::Brain.format(k, kind::LORA_A), a.len()), (KeyStyle::Brain.format(k, kind::LORA_B), b.len())]).collect(),
            None => self.trainable.iter().map(|n| (n.clone(), self.params[n].len())).collect(),
        }
    }

    /// Resume from a directory written by [`Trainer::save`]: the weights come from `base` (the original
    /// checkpoint is not needed: the saved weights replace it), the adapter, moments and step from the files.
    pub fn resume(dir: &Path, cfg: TrainConfig) -> Result<Trainer, String> {
        let base = crate::import::load(&dir.join("mirage.safetensors"), &crate::import::ImportOptions::default()).map_err(|e| e.to_string())?;
        let mut t = Trainer::new(base, cfg)?;
        if t.adapter.is_some() {
            t.adapter = Some(adapter::load(&dir.join("adapter.safetensors"), &t.base.config)?);
        }
        let state_path = dir.join("state.safetensors");
        let path = state_path.to_str().ok_or("path is not UTF-8")?;
        let meta = checkpoint::st::read_metadata(path).map_err(|e| e.to_string())?;
        let config: serde_json::Value = meta.get(checkpoint::st::CONFIG_KEY).ok_or("state.safetensors has no config header").and_then(|s| serde_json::from_str(s).map_err(|_| "unreadable state header"))?;
        t.step = config.get("step").and_then(serde_json::Value::as_u64).ok_or("state.safetensors has no step")? as u32;
        let tensors: BTreeMap<String, Vec<f32>> = checkpoint::safetensors::read(path)?.into_iter().map(|x| (x.name, x.data)).collect();
        for (name, len) in t.moment_names() {
            let (Some(m), Some(v)) = (tensors.get(&format!("{name}.m")), tensors.get(&format!("{name}.v"))) else { continue };
            if m.len() != len || v.len() != len {
                return Err(format!("state for `{name}` holds {} and {} values, expected {len}", m.len(), v.len()));
            }
            t.moments.restore(&name, m.clone(), v.clone())?;
        }
        Ok(t)
    }
}

/// The values of the adapter tensor a gradient name refers to (`<key>.lora_a` or `.lora_b`).
fn adapter_slot<'a>(set: &'a mut AdapterSet<f32>, name: &str, _cfg: &crate::config::MirageConfig) -> Result<&'a mut Vec<f32>, String> {
    for suffix in [kind::LORA_A, kind::LORA_B] {
        if let Some(key) = name.strip_suffix(suffix) {
            let pair = set.pairs.get_mut(key).ok_or_else(|| format!("no adapter pair `{key}`"))?;
            return Ok(if suffix == kind::LORA_A { &mut pair.0 } else { &mut pair.1 });
        }
    }
    Err(format!("`{name}` is not an adapter tensor"))
}
