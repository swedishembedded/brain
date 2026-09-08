// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) over the `GLVControl` trunk's attention and MLP
//! projections, over the generic `model::adapter` substrate.
//!
//! Every targeted linear `W [out×in]` gets `W_eff = W + (α/r)·B·A`, `A
//! [r×in]`, `B [out×r]` - the standard shape [`model::lora::Pair`] already
//! implements once for every model in this workspace (`s3dit`, `flux2`,
//! `wan`, `qwen3`, ...). This module keeps only what is SUPIR-specific: which
//! tensors are targeted and their naming.
//!
//! ## Targets, discovered from the manifest rather than hand-listed
//! [`crate::config::trunk_manifest`] is the single source of truth for the
//! trunk's tensor names; [`linear_sites`] filters it to the eight linear
//! suffixes [`sdxlunet::model::Rec::transformer_block`] emits (`attn1.qkv`,
//! `attn1.to_out`, `attn2.to_q`, `attn2.kv`, `attn2.to_out`, `ff.hidden`,
//! `ff.gate`, `ff.out`) - the trunk's every `BasicTransformerBlock`, at
//! whatever depth [`crate::config::SupirConfig`] the caller built with has.
//! Naming-driven discovery, not a hand-enumeration of block indices, so a
//! bigger/smaller trunk config is covered automatically. Shapes come from
//! the manifest itself, not from a hand-derived `(dim, ...)` tuple - this
//! crate does not need to know its own dimensions to target its own linears.
//!
//! The frozen SDXL backbone and the 12 adaptors are NOT targeted here - the
//! trunk is SUPIR's own trainable copy of the encoder, the natural site for a
//! parameter-efficient adapter, matching the plan's own framing ("LoRA
//! adapters on the trunk's attention/MLP projections").
//!
//! ## `apply` vs `fold`
//! [`SupirLora::apply`] clones a base [`Tensors`] map and adds every targeted
//! delta, for building a training graph's initial weight set.
//! [`SupirLora::fold_into`] does the identical per-tensor math IN PLACE, for
//! folding a trained adapter into an existing map (e.g. before serving).
//! `apply` is defined as "clone, then `fold_into`" - the same function runs
//! either way, so the two are bit-identical by construction; the module's own
//! test guards that invariant against a future refactor splitting them.
//!
//! ## Training loop shape
//! Unlike a live device-side adapter, [`crate::train::SupirTrainer`] bakes
//! weights into its recorded graph at construction - there is no `B·A`
//! decomposition on the device. So a LoRA step recomputes `W_eff` on the HOST
//! from the frozen base plus the current `A,B`, and
//! [`crate::train::SupirTrainer::write_weight`] overwrites the graph's
//! existing buffer directly - no re-recording. `dL/dW_eff`, read back via
//! `read_grad`, is projected to `(dA, dB)` exactly as every other model in
//! this workspace does it. Unlike the pre-migration version of this module, a
//! step that is missing a targeted tensor's gradient is now a hard error, not
//! a silent skip - the generic engine makes the full target-name set knowable
//! up front, so there is no reason left to tolerate a caller supplying only a
//! subset.
//!
//! Cloning the whole base [`Tensors`] map per `apply`/`fold_into` call is
//! fine at [`crate::config::SupirConfig::tiny`]'s toy scale (what this
//! module's own tests and `crate::finetune`'s overfit gates run at); a
//! production-scale LoRA fine-tune over the real 15 GB manifest would want a
//! more surgical partial-clone - out of scope here, and unlike
//! `s3dit`/`flux2`'s LoRA (their `apply()` rebuilds a full host weight
//! struct every step too, at their own real scale), SUPIR's training entry
//! points in this port are explicitly gated at reduced configs already (see
//! `crate::finetune`'s module doc).

use std::collections::HashMap;

use model::adapter::{AdapterKind, AdapterSet, KeyStyle, LinearSite, TargetHp, TargetSpec};
pub use model::lora::LoraCfg;
use model::lora::{LoraGrads, LoraPair};
use sdxlunet::import::Tensors;

use crate::config::SupirConfig;

/// The linear suffixes `sdxlunet::model::Rec::transformer_block` emits, in
/// the trunk's own (diffusers-style) naming.
const TARGET_SUFFIXES: [&str; 8] =
    ["attn1.qkv", "attn1.to_out", "attn2.to_q", "attn2.kv", "attn2.to_out", "ff.hidden", "ff.gate", "ff.out"];

/// Every linear in `cfg`'s trunk manifest this crate offers to a PEFT
/// adapter, discovered from the manifest rather than hand-listed.
pub fn linear_sites(cfg: &SupirConfig) -> Vec<LinearSite> {
    let manifest = crate::config::trunk_manifest(&cfg.trunk);
    let mut sites = Vec::new();
    for (name, shape) in &manifest {
        let Some(stem) = name.strip_suffix(".weight") else { continue };
        let Some(leaf) = TARGET_SUFFIXES.iter().find(|s| stem.ends_with(*s)) else { continue };
        assert_eq!(shape.len(), 2, "supir lora: {name} is not a 2D linear weight: {shape:?}");
        sites.push(LinearSite { name: name.clone(), leaf, layer: None, spec: TargetSpec::whole(shape[0], shape[1]) });
    }
    sites
}

/// A LoRA adapter over the trunk's attention/MLP projections.
pub struct SupirLora {
    set: AdapterSet<LoraPair>,
    hp: TargetHp,
}

impl SupirLora {
    /// Fresh adapter (`B = 0` -> initial no-op) over every targeted linear in
    /// `cfg.trunk`.
    pub fn new(cfg: &SupirConfig, lc: LoraCfg) -> SupirLora {
        let sites = linear_sites(cfg);
        assert!(!sites.is_empty(), "supir lora: no targeted linears found in the trunk manifest");
        let hp = TargetHp::from(lc);
        let mut rng: u64 = lc.seed ^ 0x5350_4952_4c6f_5241;
        let mut init = move || (model::lora::randn(&mut rng) * 0.02) as f32;
        let set = AdapterSet::build(sites, hp, KeyStyle::Brain, &mut init);
        SupirLora { set, hp }
    }

    /// Every base tensor name this adapter targets.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.set.iter().map(|(site, _)| site.name.as_str())
    }

    pub fn rank(&self) -> usize {
        self.hp.rank
    }

    pub fn alpha(&self) -> f32 {
        self.hp.alpha
    }

    /// `base_w + scale·B·A` for the ONE named tensor - the per-tensor
    /// primitive both [`Self::apply`]/[`Self::fold_into`] (over a whole
    /// [`Tensors`] map) and `crate::finetune`'s per-step weight refresh (over
    /// one already-read-back host `Vec<f32>`) reduce to.
    pub fn apply_one(&self, name: &str, base_w: &[f32]) -> Vec<f32> {
        let mut w = base_w.to_vec();
        let (_, k) = self.set.iter().find(|(site, _)| site.name == name).unwrap_or_else(|| panic!("supir lora: {name} is not targeted"));
        k.delta_into(1.0, &mut w);
        w
    }

    /// Add every targeted delta onto `base`, IN PLACE.
    pub fn fold_into(&self, base: &mut Tensors) {
        self.set.fold_into(base, 1.0).unwrap_or_else(|e| panic!("supir lora: {e}"));
    }

    /// A cloned copy of `base` with every targeted delta added - see the
    /// module doc for why this is bit-identical to a clone followed by
    /// [`Self::fold_into`] (it literally IS that).
    pub fn apply(&self, base: &Tensors) -> Tensors {
        let mut out = base.clone();
        self.fold_into(&mut out);
        out
    }

    /// One optimisation step: project `grads[name] = dL/dW_eff` onto each
    /// targeted pair's `(dA, dB)` and Adam-update it. A name `grads` does
    /// not carry is now a hard error, not a silent skip.
    pub fn step(&mut self, grads: &HashMap<String, Vec<f32>>, lr: f32) {
        let projected: Vec<LoraGrads> = self
            .set
            .iter()
            .map(|(site, k)| {
                let dw = grads.get(&site.name).unwrap_or_else(|| {
                    panic!("supir lora: step: missing gradient for {:?} - every targeted tensor's gradient is required", site.name)
                });
                k.project(dw)
            })
            .collect();
        self.set.step_projected(&projected, lr);
    }

    /// Serialise to `(name, shape, data)` tensors - `<stem>.lora_{a,b}`.
    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.set.to_tensors()
    }

    /// Reload an adapter (weights only; Adam state reset) from
    /// [`Self::to_tensors`]'s output - a fresh adapter of the right shape
    /// with `A,B` overwritten.
    pub fn from_tensors(cfg: &SupirConfig, lc: LoraCfg, tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<SupirLora, String> {
        let sites = linear_sites(cfg);
        assert!(!sites.is_empty(), "supir lora: no targeted linears found in the trunk manifest");
        let hp = TargetHp::from(lc);
        let set = AdapterSet::from_tensors(sites, hp, KeyStyle::Brain, tensors)?;
        Ok(SupirLora { set, hp })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SupirConfig {
        SupirConfig::tiny()
    }

    /// `linear_sites` must target exactly the same set of base tensor names
    /// the pre-migration `HashMap`-keyed enumeration did - the manifest
    /// filter itself did not change, only its container.
    #[test]
    fn linear_sites_matches_the_target_suffix_filter_over_the_manifest() {
        let cfg = cfg();
        let manifest = crate::config::trunk_manifest(&cfg.trunk);
        let mut want: Vec<String> = manifest
            .iter()
            .filter_map(|(name, _)| {
                let stem = name.strip_suffix(".weight")?;
                TARGET_SUFFIXES.iter().any(|s| stem.ends_with(s)).then(|| name.clone())
            })
            .collect();
        want.sort();
        let mut got: Vec<String> = linear_sites(&cfg).into_iter().map(|s| s.name).collect();
        got.sort();
        assert_eq!(want, got);
    }

    /// `B = 0` at construction, so [`SupirLora::apply`] must reproduce every
    /// targeted tensor's bytes EXACTLY - not merely "close".
    #[test]
    fn fresh_adapter_is_a_bit_exact_no_op() {
        let cfg = cfg();
        let base = crate::init::init_weights(&cfg, 5);
        let lora = SupirLora::new(&cfg, LoraCfg::new(4));
        assert!(lora.names().count() > 0, "no linears targeted - test is vacuous");
        let applied = lora.apply(&base);
        for name in lora.names() {
            let (_, want) = base.get(name).unwrap();
            let (_, got) = applied.get(name).unwrap();
            assert_eq!(want.len(), got.len(), "{name}");
            let differing = want.iter().zip(got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differing, 0, "{name}: {differing} of {} elements differ at B=0", want.len());
        }
    }

    /// `apply` (clone + fold) and a manual clone-then-`fold_into` must agree
    /// bit-for-bit - the invariant the module doc claims "by construction".
    #[test]
    fn apply_and_fold_into_agree_bit_for_bit() {
        let cfg = cfg();
        let base = crate::init::init_weights(&cfg, 6);
        let mut lora = SupirLora::new(&cfg, LoraCfg::new(4));
        // Perturb A/B away from the no-op point so a real delta is exercised.
        let grads: HashMap<String, Vec<f32>> =
            lora.names().map(|n| (n.to_string(), vec![0.37f32; base.get(n).unwrap().1.len()])).collect();
        lora.step(&grads, 0.05);

        let via_apply = lora.apply(&base);
        let mut via_fold = base.clone();
        lora.fold_into(&mut via_fold);

        for name in lora.names() {
            let (_, a) = via_apply.get(name).unwrap();
            let (_, b) = via_fold.get(name).unwrap();
            let differing = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(differing, 0, "{name}: apply vs fold_into disagree in {differing} elements");
        }
    }

    /// LoRA-only training (the base frozen, only `A,B` moving) drives the
    /// loss down - the third gate the plan asks for, alongside the no-op-at
    /// -init and fold-vs-apply checks above. Mirrors `crate::finetune`'s own
    /// overfit loop shape, but recomputes each targeted weight from the
    /// FROZEN base plus the current `(A,B)` every step
    /// ([`SupirLora::apply_one`]) rather than letting an Adam update drift
    /// the graph's own weight buffer directly - the base must never move
    /// for this to actually test "LoRA-only".
    #[test]
    fn lora_only_training_reduces_loss() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        use data::rng::Rng;

        let (cfg, base) = crate::train::tiny_setup(77);
        let mut lora = SupirLora::new(&cfg, LoraCfg::new(4));
        let folded = lora.apply(&base); // no-op at B=0

        // Matches `crate::finetune`'s and `gradcheck::supir`'s own scale -
        // see `crate::finetune`'s test module doc for why `H=W=8`.
        let (h, w, t_enc) = (8u32, 8u32, 5u32);
        let gpu = gpu_core::testgpu::dev(crate::train::TRAIN_PIPELINES);
        let trainer = crate::train::SupirTrainer::new(gpu, cfg.clone(), &folded, h, w, t_enc, 0.7);

        let c = cfg.backbone.clone();
        let mut rng = Rng::new(0xF00D_1234);
        let mut r = |n: usize| -> Vec<f32> { (0..n).map(|_| 2.0 * rng.next_f32() - 1.0).collect() };
        let sample = r((c.in_channels * h * w) as usize);
        let hint = r((4 * h * w) as usize);
        let enc = r((t_enc * c.cross_attention_dim) as usize);
        let pooled = r(c.pooled_dim() as usize);
        let time_ids = vec![64.0, 64.0, 0.0, 0.0, 64.0, 64.0];
        let target = r((c.out_channels * h * w) as usize);
        trainer.set_inputs(&sample, &hint, &enc, 601.0, &pooled, &time_ids, &target);

        let l0 = trainer.forward();
        let mut last = l0;
        let mut lmin = l0;
        for step in 1..=100 {
            trainer.zero_grads();
            last = trainer.forward();
            trainer.backward();
            let grads: HashMap<String, Vec<f32>> = lora.names().map(|n| (n.to_string(), trainer.read_grad(n))).collect();
            lora.step(&grads, 0.02);
            for name in lora.names().collect::<Vec<_>>() {
                let (_, base_w) = base.get(name).unwrap();
                let w_eff = lora.apply_one(name, base_w);
                trainer.write_weight(name, &w_eff);
            }
            lmin = lmin.min(last);
            if step % 20 == 0 {
                eprintln!("  lora step {step:3}: loss = {last:.4e} (min {lmin:.4e})");
            }
        }
        // A rank-4 adapter over only the trunk's attn/MLP linears (the
        // backbone, the adaptors' own zero-convs and the trunk's conv/norm
        // layers stay untouched) cannot fully fit an arbitrary target the
        // way full-capacity training can - it converges toward a capacity
        // floor and then jitters around it (Adam noise at a fixed lr), so
        // the gate checks the MINIMUM loss reached, not the last one -
        // matching `s3dit::lora`'s own established test shape for exactly
        // this convergence pattern.
        eprintln!("LoRA-only overfit: {l0:.4e} -> {last:.4e} (min {lmin:.4e})");
        assert!(lmin < l0 * 0.5, "LoRA-only training did not reduce the loss: {l0:.4e} -> min {lmin:.4e}");
    }

    /// Save/load round-trips the adapter weights exactly.
    #[test]
    fn save_load_round_trips() {
        let cfg = cfg();
        let base = crate::init::init_weights(&cfg, 9);
        let mut lora = SupirLora::new(&cfg, LoraCfg::new(4));
        let grads: HashMap<String, Vec<f32>> =
            lora.names().map(|n| (n.to_string(), vec![0.21f32; base.get(n).unwrap().1.len()])).collect();
        lora.step(&grads, 0.05);

        let tensors: HashMap<String, (Vec<usize>, Vec<f32>)> =
            lora.to_tensors().into_iter().map(|(n, s, d)| (n, (s, d))).collect();
        let reloaded = SupirLora::from_tensors(&cfg, LoraCfg::new(4), &tensors).expect("reload");

        let (wa, wb) = (lora.apply(&base), reloaded.apply(&base));
        for name in lora.names() {
            let (_, a) = wa.get(name).unwrap();
            let (_, b) = wb.get(name).unwrap();
            let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(diff < 1e-6, "{name}: save/load changed the effective weight (max diff {diff:.2e})");
        }
    }
}
