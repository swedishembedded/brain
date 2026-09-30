// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reinforcement-learning rollouts on the paged serving engine: a
//! [`model::rollout::SyncRollout`] whose policy is an [`Engine`] built from a
//! base checkpoint plus whatever the trainer last synced.
//!
//! The trainer (`rl::objective::grpo::Grpo::with_rollout`) syncs its
//! trainable tensors every few groups. For a LoRA run those are the adapters:
//! each `<linear>.lora_a`/`.lora_b` pair is folded into a copy of its base
//! linear (`W + (alpha / rank) * B A`), and only the adapted linears are ever
//! duplicated. For a full fine-tune they are every parameter, and replace the
//! base outright. Either way the engine is rebuilt from the result, freeing
//! the previous one first, so a sampled group always comes from the policy
//! the trainer held at the last sync.
//!
//! Swedish Embedded AB implements reinforcement-learning pipelines for
//! language models like this for its clients. If your team needs expertise
//! in training models on your own hardware, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use data::rng::Rng;
use model::rollout::{Completion, PagedRollout, Rollout, RolloutParams, SyncRollout};

use crate::config::{LoraCfg, QwenConfig};
use crate::serve::Engine;
use crate::Dtype;

/// The paged cache an [`EngineRollout`] builds its engines with.
#[derive(Clone, Copy, Debug)]
pub struct EngineGeometry {
    pub block_size: u32,
    pub num_blocks: u32,
    /// Sequences decoded together: a GRPO group's size.
    pub max_batch: u32,
    pub max_blocks_per_seq: u32,
    pub max_prefill: u32,
    /// The storage tier of the engine's linears.
    pub tier: Dtype,
    /// The card the engine goes on (`None`: the ambient device).
    pub gpu: Option<u32>,
}

pub struct EngineRollout {
    cfg: QwenConfig,
    base: HashMap<String, Vec<f32>>,
    lora: Option<LoraCfg>,
    geometry: EngineGeometry,
    inner: Option<PagedRollout<Engine>>,
}

impl EngineRollout {
    /// A rollout over the decoder `cfg` (its `lora` field ignored: the engine
    /// serves plain linears) with weights `base`, folding adapters of shape
    /// `lora` on each sync. No engine exists until the first sync.
    pub fn new(cfg: QwenConfig, base: HashMap<String, Vec<f32>>, lora: Option<LoraCfg>, geometry: EngineGeometry) -> EngineRollout {
        EngineRollout { cfg: QwenConfig { lora: None, ..cfg }, base, lora, geometry, inner: None }
    }

    /// The engine, once synced.
    pub fn engine(&self) -> Option<&Engine> {
        self.inner.as_ref().map(|r| r.decoder())
    }

    fn build(&self, weights: &HashMap<String, Vec<f32>>) -> Result<Engine, String> {
        let g = self.geometry;
        let make = || Engine::from_map_tier(self.cfg.clone(), weights, g.block_size, g.num_blocks, g.max_batch, g.max_blocks_per_seq, g.max_prefill, false, g.tier);
        match g.gpu {
            Some(i) => gpu_core::devices::with_gpu(i, make),
            None => Ok(make()),
        }
    }
}

impl Rollout for EngineRollout {
    fn sample_n(&mut self, prompt: &[u32], n: usize, params: &RolloutParams, rng: &mut Rng) -> Vec<Completion> {
        self.inner.as_mut().expect("EngineRollout: sync before sampling").sample_n(prompt, n, params, rng)
    }
}

impl SyncRollout for EngineRollout {
    fn sync(&mut self, trained: &HashMap<String, Vec<f32>>) -> Result<(), String> {
        // Free the previous engine before the next one allocates.
        self.inner = None;
        // Swap each adapted (or replaced) tensor into the base map for the
        // build, and back out afterwards, so only those are ever copied.
        let mut swapped: Vec<(String, Vec<f32>)> = Vec::new();
        let restore = |base: &mut HashMap<String, Vec<f32>>, swapped: Vec<(String, Vec<f32>)>| {
            for (name, orig) in swapped {
                base.insert(name, orig);
            }
        };
        let result = (|| -> Result<(), String> {
            for (name, value) in trained {
                if name.ends_with(".lora_b") {
                    continue;
                }
                let (base_name, new) = if let Some(base_name) = name.strip_suffix(".lora_a") {
                    let lora = self.lora.as_ref().ok_or_else(|| format!("{name}: an adapter, but this rollout was built without a LoRA shape"))?;
                    let b = trained.get(&format!("{base_name}.lora_b")).ok_or_else(|| format!("{name} has no .lora_b"))?;
                    let w = self.base.get(base_name).ok_or_else(|| format!("{base_name}: the adapter's base is not a weight of this model"))?;
                    let r = lora.rank as usize;
                    let (inn, out) = (value.len() / r, b.len() / r);
                    if out * inn != w.len() {
                        return Err(format!("{base_name}: a rank-{r} adapter of [{out}, {inn}] on a weight of {} values", w.len()));
                    }
                    let mut folded = w.clone();
                    let scale = lora.alpha / lora.rank as f32;
                    model::lora::Pair::from_ab(out, inn, r, value.clone(), b.clone()).delta(scale, &mut folded);
                    (base_name.to_string(), folded)
                } else {
                    if !self.base.contains_key(name) {
                        return Err(format!("{name} is not a weight of this model"));
                    }
                    (name.clone(), value.clone())
                };
                let orig = self.base.insert(base_name.clone(), new).expect("checked present");
                swapped.push((base_name, orig));
            }
            let engine = self.build(&self.base)?;
            self.inner = Some(PagedRollout::new(engine));
            Ok(())
        })();
        restore(&mut self.base, std::mem::take(&mut swapped));
        result
    }
}
