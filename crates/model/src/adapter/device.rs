// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training any parameter-
// efficient adapter method on any model architecture from one shared adapter
// schema, for its clients. If your team needs expertise in PEFT training
// infrastructure then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The device-side (`ParamStore`) adapter family, driven by the parameter set
//! of each adapter kind rather than by tensor names.
//!
//! A device model stores an adapter as extra named tensors beside the base
//! weight (`{base}{suffix}`). Which suffixes exist, their shapes, how they
//! start and whether and how fast they train are all answered by the kind's
//! [`AdapterDefinition`] and [`AdapterOptimization`] - the model crates ask the
//! functions below and never spell a suffix. A new kind joins by listing
//! itself in [`kinds`]; the role assignment, init, save filter and parameter
//! lists of every model follow.
//!
//! [`LoraCfg`] is the one device-side configuration: rank, alpha, targets and
//! the hyper-parameters the device trainers wire (rsLoRA, LoRA+, LoRA-FA).

use paramstore::{ParamStore, Role};
use serde_json::Value;

use super::kind::{AdapterDefinition, AdapterOptimization, ParamGroup, ParamInit};
use super::{TargetHp, TargetSpec};
use crate::lora::LoraPair;

/// The suffix sets of every adapter kind the device family knows.
fn kinds() -> [&'static [&'static str]; 1] {
    [LoraPair::suffixes()]
}

/// The suffix of `name`'s adapter tensor, if `name` is one.
fn suffix_of(name: &str) -> Option<&'static str> {
    kinds().into_iter().flatten().copied().find(|s| name.ends_with(s))
}

/// Is `name` one of an adapter kind's own tensors (as opposed to a base weight)?
pub fn is_adapter_param(name: &str) -> bool {
    suffix_of(name).is_some()
}

/// The name of `base`'s adapter tensor `suffix`.
pub fn param_name(base: &str, suffix: &str) -> String {
    format!("{base}{suffix}")
}

/// The names of the LoRA down and up projections of `base`: what a LoRA
/// forward/backward reads from the parameter store.
pub fn lora_names(base: &str) -> (String, String) {
    (param_name(base, super::kind::LORA_A), param_name(base, super::kind::LORA_B))
}

/// How the adapter tensor `name` starts; `None` for a base weight.
pub fn init_of(name: &str) -> Option<ParamInit> {
    let suffix = suffix_of(name)?;
    // Init does not depend on dimensions: any target stands for all.
    LoraPair::params(&TargetHp::new(1, 1.0), &TargetSpec::whole(1, 1)).into_iter().find(|p| p.suffix == suffix).map(|p| p.init)
}

/// The training group of the adapter tensor `name` under `hp`; `None` for a
/// base weight.
pub fn group_of(name: &str, hp: &TargetHp) -> Option<ParamGroup> {
    let suffix = suffix_of(name)?;
    LoraPair::groups(hp).into_iter().find(|g| g.suffix == suffix)
}

/// The `ParamStore` role of `name` in an adapter-training build: the adapter's
/// trainable tensors train, its frozen ones (LoRA-FA's `A`) and every base
/// weight are frozen.
pub fn role_of(name: &str, hp: &TargetHp) -> Role {
    match group_of(name, hp) {
        Some(g) if g.trainable => Role::Trainable,
        _ => Role::Frozen,
    }
}

/// Apply every adapter tensor's learning-rate multiplier (LoRA+) to `ps`.
/// A multiplier of exactly 1.0 is left unset, so a plain adapter's optimizer
/// descriptors are bit-identical to a build that never heard of groups.
pub fn apply_groups(ps: &mut ParamStore, hp: &TargetHp) {
    let names: Vec<String> = ps.trainable.iter().map(|(n, _)| n.clone()).collect();
    for name in names {
        if let Some(g) = group_of(&name, hp) {
            if g.lr_mult != 1.0 {
                ps.set_lr_mult(&name, g.lr_mult);
            }
        }
    }
}

/// The adapter tensors `ps` holds without optimising them (LoRA-FA's `A`):
/// what [`crate::Model::frozen_adapter_param_names`] reports.
pub fn frozen_adapter_params(ps: &ParamStore) -> Vec<String> {
    ps.params.iter().map(|(n, _)| n).filter(|n| is_adapter_param(n) && !ps.grad.contains_key(*n)).cloned().collect()
}

/// The shape of the adapter tensor `name` holding `len` elements at `rank`,
/// for a file that stored it flat.
pub fn shape_of(name: &str, rank: usize, len: usize) -> Option<Vec<usize>> {
    LoraPair::shape_from_len(suffix_of(name)?, rank, len)
}

/// The `(name, numel)` entries a model's parameter list adds for the adapter
/// of base weight `name` with `out x inn` dimensions.
pub fn param_list(name: &str, hp: &TargetHp, out: usize, inn: usize) -> Vec<(String, usize)> {
    LoraPair::params(hp, &TargetSpec::whole(out, inn)).into_iter().map(|p| (param_name(name, p.suffix), p.numel())).collect()
}

/// Device-side adapter configuration: the targets and hyper-parameters of an
/// adapter-training build of a model.
///
/// `PartialEq`/`Debug` because model configs that carry it derive them.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraCfg {
    pub rank: u32,
    pub alpha: f32,
    /// Which projections get adapters (matched by the leaf name, e.g. "wq").
    pub targets: Vec<String>,
    /// rsLoRA: scale `alpha/sqrt(rank)` instead of `alpha/rank`.
    pub rank_stabilized: bool,
    /// LoRA+: `B` trains at `lr_ratio x lr`. 1.0 = plain LoRA.
    pub lr_ratio: f32,
    /// LoRA-FA: `A` stays at its init and only `B` trains.
    pub freeze_a: bool,
}

impl LoraCfg {
    /// Plain LoRA on `targets`.
    pub fn new(rank: u32, alpha: f32, targets: Vec<String>) -> LoraCfg {
        LoraCfg { rank, alpha, targets, rank_stabilized: false, lr_ratio: 1.0, freeze_a: false }
    }

    /// The default Qwen LoRA target set: the four attention projections.
    pub fn attn(rank: u32, alpha: f32) -> LoraCfg {
        LoraCfg::new(rank, alpha, ["wq", "wk", "wv", "wo"].iter().map(|s| s.to_string()).collect())
    }

    pub fn targets_leaf(&self, leaf: &str) -> bool {
        self.targets.iter().any(|t| t == leaf)
    }

    /// The per-target hyper-parameters every kind's schema, scale and groups
    /// are computed from.
    pub fn hp(&self) -> TargetHp {
        TargetHp { rank_stabilized: self.rank_stabilized, lr_ratio: self.lr_ratio, freeze_a: self.freeze_a, ..TargetHp::new(self.rank as usize, self.alpha) }
    }

    /// The factor the adapter's product is scaled by.
    pub fn scale(&self) -> f32 {
        self.hp().scale()
    }

    /// The alpha an adapter file records: `scale() * rank`, so every reader
    /// that computes `alpha / rank` (fold, serve, export) applies exactly the
    /// scale this adapter trained with, rsLoRA or not.
    pub fn card_alpha(&self) -> f32 {
        self.scale() * self.rank as f32
    }

    /// The JSON a model config carries so a checkpoint rebuilds its adapter
    /// shape. Hyper-parameters at their defaults are omitted, so a plain
    /// LoRA's config is byte-identical to what earlier versions wrote.
    pub fn to_json(&self) -> Value {
        let mut v = serde_json::json!({ "rank": self.rank, "alpha": self.alpha, "targets": self.targets });
        if self.rank_stabilized {
            v["rank_stabilized"] = true.into();
        }
        if self.lr_ratio != 1.0 {
            v["lr_ratio"] = self.lr_ratio.into();
        }
        if self.freeze_a {
            v["freeze_a"] = true.into();
        }
        v
    }

    /// The inverse of [`Self::to_json`]; an absent hyper-parameter is its
    /// default, and an absent rank or alpha is 0 as it always was.
    pub fn from_json(v: &serde_json::Map<String, Value>) -> LoraCfg {
        LoraCfg {
            rank: v["rank"].as_u64().unwrap_or(0) as u32,
            alpha: v["alpha"].as_f64().unwrap_or(0.0) as f32,
            targets: v["targets"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default(),
            rank_stabilized: v.get("rank_stabilized").and_then(Value::as_bool).unwrap_or(false),
            lr_ratio: v.get("lr_ratio").and_then(Value::as_f64).map_or(1.0, |r| r as f32),
            freeze_a: v.get("freeze_a").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::kind::{LORA_B};

    #[test]
    fn names_are_recognised_by_the_kinds_own_suffixes() {
        assert!(is_adapter_param("blocks.0.attn.wq.weight.lora_a"));
        assert!(is_adapter_param(&param_name("w", LORA_B)));
        assert!(!is_adapter_param("blocks.0.attn.wq.weight"));
        assert!(!is_adapter_param("lora_a_norm.weight"));
    }

    #[test]
    fn init_roles_and_shapes_follow_the_definition() {
        assert_eq!(init_of("w.lora_a"), Some(ParamInit::Drawn));
        assert_eq!(init_of("w.lora_b"), Some(ParamInit::Zeros));
        assert_eq!(init_of("w"), None);
        assert_eq!(shape_of("w.lora_a", 2, 10), Some(vec![2, 5]));
        assert_eq!(shape_of("w.lora_b", 2, 12), Some(vec![6, 2]));
        assert_eq!(shape_of("w.lora_b", 3, 10), None, "not a whole rank-3 matrix");
        let list = param_list("w", &TargetHp::new(2, 2.0), 6, 5);
        assert_eq!(list, vec![("w.lora_a".to_string(), 10), ("w.lora_b".to_string(), 12)]);
    }

    #[test]
    fn lora_fa_freezes_a_and_lora_plus_scales_only_b() {
        let plain = TargetHp::new(2, 2.0);
        assert_eq!((role_of("w.lora_a", &plain), role_of("w.lora_b", &plain)), (Role::Trainable, Role::Trainable));
        let fa = TargetHp { freeze_a: true, lr_ratio: 4.0, ..plain };
        assert_eq!((role_of("w.lora_a", &fa), role_of("w.lora_b", &fa), role_of("w", &fa)), (Role::Frozen, Role::Trainable, Role::Frozen));
        assert_eq!(group_of("w.lora_a", &fa).unwrap().lr_mult, 1.0);
        assert_eq!(group_of("w.lora_b", &fa).unwrap().lr_mult, 4.0);
    }

    #[test]
    fn config_round_trips_and_a_plain_lora_writes_the_old_json() {
        let plain = LoraCfg::attn(4, 8.0);
        assert_eq!(plain.to_json(), serde_json::json!({ "rank": 4, "alpha": 8.0, "targets": ["wq", "wk", "wv", "wo"] }));
        let tuned = LoraCfg { rank_stabilized: true, lr_ratio: 16.0, freeze_a: true, ..plain.clone() };
        let back = LoraCfg::from_json(tuned.to_json().as_object().unwrap());
        assert_eq!(back, tuned);
        assert_eq!(LoraCfg::from_json(plain.to_json().as_object().unwrap()), plain);
    }

    #[test]
    fn the_card_alpha_carries_the_trained_scale_to_every_reader() {
        let rs = LoraCfg { rank_stabilized: true, ..LoraCfg::attn(4, 8.0) };
        assert_eq!(rs.scale(), 4.0);
        assert_eq!(rs.card_alpha() / rs.rank as f32, rs.scale(), "a reader's alpha/rank is the trained scale");
        let plain = LoraCfg::attn(4, 8.0);
        assert_eq!(plain.card_alpha(), 8.0);
    }
}
