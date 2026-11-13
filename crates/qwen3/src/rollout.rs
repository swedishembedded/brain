// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reinforcement-learning rollouts on the paged serving engine: a
//! [`model::rollout::SyncRollout`] whose policy is an [`Engine`] built once
//! from a base checkpoint, refit by whatever the trainer last synced.
//!
//! The trainer (`rl::objective::grpo::Grpo::with_rollout`) syncs its
//! trainable tensors every few groups. For a LoRA run those are the adapters:
//! each `<linear>.lora_a`/`.lora_b` pair is published to the engine's adapter
//! bank as the next policy version and sampled under from then on; the base
//! weights are never rebuilt or copied, a refit costs the factors' upload, and
//! the previous version's cached prompts are purged with it. For a full
//! fine-tune the trained tensors are every parameter and replace the base
//! outright: the engine is rebuilt from the result (freeing the previous one
//! first), the one case a bank cannot express. Either way a sampled group
//! comes from the policy the trainer held at the last sync and every
//! completion carries that policy's version.
//!
//! Swedish Embedded AB implements reinforcement-learning pipelines for
//! language models like this for its clients. If your team needs expertise
//! in training models on your own hardware, you can procure our services by
//! sending an email to info@swedishembedded.com.

use model::adapter::kind::{LORA_A, LORA_B};
use std::collections::HashMap;

use data::rng::Rng;
use model::rollout::{Completion, Rollout, RolloutParams, SchedulerRollout, SyncRollout};

use crate::config::{LoraCfg, QwenConfig};
use crate::lora::{AdapterSite, DeviceAdapter};
use crate::serve::{AdapterBankConfig, Engine, LoraSite, VerifiedAdapter};
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
    inner: Option<SchedulerRollout<Engine>>,
    /// Policies published so far, across engine rebuilds: the version the next
    /// refit takes is this plus one.
    versions: u64,
}

impl EngineRollout {
    /// A rollout over the decoder `cfg` (its `lora` field ignored: the engine
    /// serves plain linears) with weights `base`, publishing adapters of shape
    /// `lora` on each sync. No engine exists until the first sync.
    pub fn new(cfg: QwenConfig, base: HashMap<String, Vec<f32>>, lora: Option<LoraCfg>, geometry: EngineGeometry) -> EngineRollout {
        EngineRollout { cfg: QwenConfig { lora: None, ..cfg }, base, lora, geometry, inner: None, versions: 0 }
    }

    /// The engine, once synced.
    pub fn engine(&self) -> Option<&Engine> {
        self.inner.as_ref().map(|r| r.decoder())
    }

    /// The rollout's scheduler, once synced: its prefix-cache accounting.
    pub fn scheduler(&self) -> Option<&model::serve::Scheduler<Engine>> {
        self.inner.as_ref().map(|r| r.scheduler())
    }

    fn build(&self, weights: &HashMap<String, Vec<f32>>) -> Result<Engine, String> {
        let g = self.geometry;
        let make = || Engine::from_map_tier(self.cfg.clone(), weights, g.block_size, g.num_blocks, g.max_batch, g.max_blocks_per_seq, g.max_prefill, false, g.tier);
        match g.gpu {
            Some(i) => gpu_core::devices::with_gpu(i, make),
            None => Ok(make()),
        }
    }

    /// An engine over the unmodified base with an adapter bank for the shape
    /// the trainer adapts: two slots, the policy being published and the one
    /// it replaces.
    fn build_banked(&self, lora: &LoraCfg) -> Result<Engine, String> {
        let sites: Vec<LoraSite> = lora.targets.iter().map(|leaf| LoraSite::from_leaf(leaf).ok_or_else(|| format!("{leaf} is not a linear of a Qwen3 block"))).collect::<Result<_, _>>()?;
        let mut engine = self.build(&self.base)?;
        engine.enable_adapters(AdapterBankConfig::new(2, lora.rank, &sites)).map_err(|e| format!("the rollout engine's adapter bank: {e:?}"))?;
        Ok(engine)
    }

    /// The trained adapters as the bank takes them: every `<linear>.lora_a` /
    /// `.lora_b` pair, unscaled, with the scale carried once as `alpha / rank`.
    fn adapter_of(&self, lora: &LoraCfg, trained: &HashMap<String, Vec<f32>>) -> Result<VerifiedAdapter, String> {
        let mut sites: Vec<AdapterSite> = Vec::new();
        for (name, a) in trained {
            let Some(base_name) = name.strip_suffix(LORA_A) else { continue };
            let b = trained.get(&format!("{base_name}{LORA_B}")).ok_or_else(|| format!("{name} has no {LORA_B}"))?;
            if !self.base.contains_key(base_name) {
                return Err(format!("{base_name}: the adapter's base is not a weight of this model"));
            }
            sites.push(AdapterSite { base: base_name.to_string(), a: a.clone(), b: b.clone() });
        }
        sites.sort_by(|x, y| x.base.cmp(&y.base));
        let version = self.versions + 1;
        let adapter = DeviceAdapter { rank: lora.rank, alpha: lora.scale() * lora.rank as f32, sites, saved: Vec::new() };
        // The identity of a policy is its version: every sync is new weights.
        Ok(VerifiedAdapter { label: format!("policy-{version}"), digest: format!("policy-{version}"), adapter })
    }

    /// A LoRA run: publish the trained adapters to the bank.
    fn publish(&mut self, lora: LoraCfg, trained: &HashMap<String, Vec<f32>>) -> Result<(), String> {
        let adapter = self.adapter_of(&lora, trained)?;
        if self.inner.is_none() {
            let engine = self.build_banked(&lora)?;
            self.inner = Some(SchedulerRollout::new(engine, self.geometry.max_batch as usize));
        }
        let rollout = self.inner.as_mut().expect("built above");
        rollout.publish_adapter(adapter).map_err(|e| format!("publishing the adapters to the rollout engine: {e:?}"))?;
        self.versions = rollout.policy_version();
        Ok(())
    }

    /// A full fine-tune: the trained tensors replace the base's, and the engine
    /// is rebuilt from them.
    fn rebuild(&mut self, trained: &HashMap<String, Vec<f32>>) -> Result<(), String> {
        // Free the previous engine before the next one allocates.
        self.inner = None;
        let mut swapped: Vec<(String, Vec<f32>)> = Vec::new();
        let result = (|| -> Result<(), String> {
            for (name, value) in trained {
                if !self.base.contains_key(name) {
                    return Err(format!("{name} is not a weight of this model"));
                }
                let orig = self.base.insert(name.clone(), value.clone()).expect("checked present");
                swapped.push((name.clone(), orig));
            }
            let engine = self.build(&self.base)?;
            let mut rollout = SchedulerRollout::new(engine, self.geometry.max_batch as usize);
            self.versions += 1;
            rollout.set_adapter(None, self.versions);
            self.inner = Some(rollout);
            Ok(())
        })();
        // Back out the replaced tensors, so only those are ever copied.
        for (name, orig) in swapped {
            self.base.insert(name, orig);
        }
        result
    }
}

impl Rollout for EngineRollout {
    fn sample_n(&mut self, prompt: &[u32], n: usize, params: &RolloutParams, rng: &mut Rng) -> Vec<Completion> {
        self.inner.as_mut().expect("EngineRollout: sync before sampling").sample_n(prompt, n, params, rng)
    }
}

impl SyncRollout for EngineRollout {
    fn sync(&mut self, trained: &HashMap<String, Vec<f32>>) -> Result<(), String> {
        let is_adapter = |name: &String| name.ends_with(LORA_A) || name.ends_with(LORA_B);
        let adapters_only = !trained.is_empty() && trained.keys().all(is_adapter);
        if !adapters_only && trained.keys().any(is_adapter) {
            return Err("the trained tensors mix adapter factors with whole tensors; the rollout engine takes either, not both".to_string());
        }
        match (&self.lora, adapters_only) {
            (Some(lora), true) => self.publish(lora.clone(), trained),
            (None, true) => Err("the trained tensors are adapters, but this rollout was built without a LoRA shape".to_string()),
            (_, false) => self.rebuild(trained),
        }
    }
}
