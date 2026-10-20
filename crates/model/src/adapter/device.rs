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

use super::kind::{DeviceKind, ParamGroup, ParamInit};
use super::{TargetHp, TargetSpec};

/// The kind that owns the adapter tensor `name`, by its suffix.
pub fn kind_of(name: &str) -> Option<DeviceKind> {
    DeviceKind::ALL.into_iter().find(|k| k.suffixes().iter().any(|s| name.ends_with(s)))
}

/// Is `name` one of an adapter kind's own tensors (as opposed to a base weight)?
pub fn is_adapter_param(name: &str) -> bool {
    kind_of(name).is_some()
}

/// The suffix of `name`'s adapter tensor, if `name` is one.
fn suffix_of(name: &str) -> Option<&'static str> {
    kind_of(name)?.suffixes().iter().copied().find(|s| name.ends_with(s))
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
    let (kind, suffix) = (kind_of(name)?, suffix_of(name)?);
    // Init does not depend on dimensions: any target stands for all.
    kind.params(&TargetHp::new(1, 1.0), &TargetSpec::whole(1, 1)).into_iter().find(|p| p.suffix == suffix).map(|p| p.init)
}

/// The training group of the adapter tensor `name` under `hp`; `None` for a
/// base weight.
pub fn group_of(name: &str, hp: &TargetHp) -> Option<ParamGroup> {
    let (kind, suffix) = (kind_of(name)?, suffix_of(name)?);
    kind.groups(hp).into_iter().find(|g| g.suffix == suffix)
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
    kind_of(name)?.shape_from_len(suffix_of(name)?, rank, len)
}

/// The `(name, numel)` entries a model's parameter list adds for the adapter
/// of base weight `name` with `out x inn` dimensions.
pub fn param_list(name: &str, kind: DeviceKind, hp: &TargetHp, out: usize, inn: usize) -> Vec<(String, usize)> {
    kind.params(hp, &TargetSpec::whole(out, inn)).into_iter().map(|p| (param_name(name, p.suffix), p.numel())).collect()
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
    /// The adapter method (default LoRA).
    pub kind: DeviceKind,
    /// Targets that take their own rank and/or alpha instead of the shared
    /// ones, first match wins.
    pub overrides: Vec<TargetOverride>,
}

/// One target's own rank and alpha. `leaf` is matched like a target: the whole
/// name, a trailing dotted suffix of it, or its leaf component.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetOverride {
    pub leaf: String,
    pub rank: Option<u32>,
    pub alpha: Option<f32>,
}

/// The adapter hyper-parameters a fine-tune run chooses on top of rank, alpha
/// and targets: what `--lora-variant` and `--lora-plus-ratio` set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdapterTuning {
    /// rsLoRA: scale `alpha/sqrt(rank)`.
    pub rank_stabilized: bool,
    /// LoRA+: `B` trains at this multiple of the learning rate.
    pub lr_ratio: f32,
    /// LoRA-FA: `A` stays at its init.
    pub freeze_a: bool,
    /// The adapter method.
    pub kind: DeviceKind,
}

impl Default for AdapterTuning {
    fn default() -> AdapterTuning {
        AdapterTuning { rank_stabilized: false, lr_ratio: 1.0, freeze_a: false, kind: DeviceKind::Lora }
    }
}

impl AdapterTuning {
    /// The tuning the model-agnostic options name: `variant` is a comma list of
    /// `plain`, `rs` (rsLoRA) and `fa` (LoRA-FA), `plus_ratio` the LoRA+ ratio
    /// for `B`, `kind` the adapter method. Every front end parses these three
    /// values with this one function.
    pub fn parse(variant: Option<&str>, plus_ratio: Option<&str>, kind: Option<&str>) -> Result<AdapterTuning, String> {
        let mut t = AdapterTuning::default();
        if let Some(k) = kind {
            t.kind = DeviceKind::from_name(k).ok_or_else(|| format!("--adapter-kind {k:?}: expected lora, ia3 or lokr"))?;
        }
        for word in variant.into_iter().flat_map(|v| v.split(',')).map(str::trim).filter(|w| !w.is_empty()) {
            match word {
                "plain" => {}
                "rs" => t.rank_stabilized = true,
                "fa" => t.freeze_a = true,
                other => return Err(format!("--lora-variant {other:?}: expected a comma list of plain, rs, fa")),
            }
        }
        if let Some(r) = plus_ratio {
            t.lr_ratio = r.parse().map_err(|_| format!("--lora-plus-ratio {r:?}: not a valid number"))?;
        }
        t.validate()?;
        Ok(t)
    }

    /// Refuse a tuning no run can honour (a non-positive or non-finite LoRA+
    /// ratio).
    pub fn validate(&self) -> Result<(), String> {
        if !(self.lr_ratio.is_finite() && self.lr_ratio > 0.0) {
            return Err(format!("the LoRA+ learning-rate ratio must be positive, got {}", self.lr_ratio));
        }
        if self.kind != DeviceKind::Lora && (self.rank_stabilized || self.freeze_a || self.lr_ratio != 1.0) {
            return Err(format!("rsLoRA, LoRA-FA and LoRA+ are LoRA variants; a {} adapter takes none of them", self.kind.name()));
        }
        Ok(())
    }
}

/// Do two alphas name the same scale? Equal up to the rounding of the
/// `alpha -> alpha/sqrt(rank) -> card alpha` round trip of rsLoRA.
pub fn alpha_matches(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0)
}

/// Does `name` (a base weight name, or a bare leaf) belong to `target`?
fn target_matches(name: &str, target: &str) -> bool {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let hit = |n: &str| n == target || n.ends_with(&format!(".{target}"));
    hit(name) || hit(stem)
}

impl LoraCfg {
    /// Plain LoRA on `targets`.
    pub fn new(rank: u32, alpha: f32, targets: Vec<String>) -> LoraCfg {
        LoraCfg { rank, alpha, targets, rank_stabilized: false, lr_ratio: 1.0, freeze_a: false, kind: DeviceKind::Lora, overrides: Vec::new() }
    }

    /// The default Qwen LoRA target set: the four attention projections.
    pub fn attn(rank: u32, alpha: f32) -> LoraCfg {
        LoraCfg::new(rank, alpha, ["wq", "wk", "wv", "wo"].iter().map(|s| s.to_string()).collect())
    }

    pub fn targets_leaf(&self, leaf: &str) -> bool {
        self.targets.iter().any(|t| t == leaf)
    }

    /// This config with adapter method `kind`.
    pub fn with_kind(mut self, kind: DeviceKind) -> LoraCfg {
        self.kind = kind;
        self
    }

    /// The `(name, numel)` parameter entries of the adapter on base weight
    /// `name` of `out x inn`: this config's kind at the target's own
    /// hyper-parameters.
    pub fn param_list(&self, name: &str, out: usize, inn: usize) -> Vec<(String, usize)> {
        param_list(name, self.kind, &self.hp_for(name), out, inn)
    }

    /// This config with the run's hyper-parameters applied.
    pub fn with_tuning(mut self, tuning: AdapterTuning) -> LoraCfg {
        self.rank_stabilized = tuning.rank_stabilized;
        self.lr_ratio = tuning.lr_ratio;
        self.freeze_a = tuning.freeze_a;
        self.kind = tuning.kind;
        self
    }

    /// The hyper-parameters this config carries, as a run chooses them.
    pub fn tuning(&self) -> AdapterTuning {
        AdapterTuning { rank_stabilized: self.rank_stabilized, lr_ratio: self.lr_ratio, freeze_a: self.freeze_a, kind: self.kind }
    }

    /// This config with `leaf` at its own rank and/or alpha.
    pub fn with_override(mut self, leaf: &str, rank: Option<u32>, alpha: Option<f32>) -> LoraCfg {
        self.overrides.push(TargetOverride { leaf: leaf.to_string(), rank, alpha });
        self
    }

    /// The hyper-parameters of the target `name` (a base weight name or a leaf):
    /// the shared ones with that target's own rank and alpha applied.
    pub fn hp_for(&self, name: &str) -> TargetHp {
        let mut hp = self.hp();
        if let Some(o) = self.overrides.iter().find(|o| target_matches(name, &o.leaf)) {
            hp.rank = o.rank.map_or(hp.rank, |r| r as usize);
            hp.alpha = o.alpha.unwrap_or(hp.alpha);
        }
        hp
    }

    /// The widest rank any target takes: what a scratch buffer is sized for.
    pub fn max_rank(&self) -> u32 {
        self.overrides.iter().filter_map(|o| o.rank).fold(self.rank, u32::max)
    }

    /// Do all targets share one rank and scale?
    pub fn is_uniform(&self) -> bool {
        self.overrides.is_empty()
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
        if self.rank_stabilized {
            self.scale() * self.rank as f32
        } else {
            self.alpha
        }
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
        if self.kind != DeviceKind::Lora {
            v["kind"] = self.kind.name().into();
        }
        if !self.overrides.is_empty() {
            v["overrides"] = self.overrides.iter().map(|o| serde_json::json!({ "leaf": o.leaf, "rank": o.rank, "alpha": o.alpha })).collect::<Vec<_>>().into();
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
            kind: v.get("kind").and_then(Value::as_str).and_then(DeviceKind::from_name).unwrap_or_default(),
            overrides: v
                .get("overrides")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|o| {
                            Some(TargetOverride {
                                leaf: o.get("leaf")?.as_str()?.to_string(),
                                rank: o.get("rank").and_then(Value::as_u64).map(|r| r as u32),
                                alpha: o.get("alpha").and_then(Value::as_f64).map(|a| a as f32),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default(),
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
        let list = param_list("w", DeviceKind::Lora, &TargetHp::new(2, 2.0), 6, 5);
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
    fn the_model_agnostic_options_parse_into_one_tuning() {
        assert_eq!(AdapterTuning::parse(None, None, None).unwrap(), AdapterTuning::default());
        let t = AdapterTuning::parse(Some("rs,fa"), Some("16"), None).unwrap();
        assert_eq!(t, AdapterTuning { rank_stabilized: true, lr_ratio: 16.0, freeze_a: true, kind: DeviceKind::Lora });
        assert!(AdapterTuning::parse(Some("dora"), None, None).unwrap_err().contains("--lora-variant"));
        assert!(AdapterTuning::parse(None, Some("0"), None).unwrap_err().contains("LoRA+"));
        assert!(AdapterTuning::parse(None, Some("fast"), None).unwrap_err().contains("--lora-plus-ratio"));
        assert_eq!(AdapterTuning::parse(None, None, Some("lokr")).unwrap().kind, DeviceKind::Lokr);
        assert!(AdapterTuning::parse(None, None, Some("oft")).unwrap_err().contains("--adapter-kind"));
        assert!(AdapterTuning::parse(Some("rs"), None, Some("ia3")).unwrap_err().contains("LoRA variants"));
    }

    #[test]
    fn a_target_can_take_its_own_rank_and_alpha() {
        let cfg = LoraCfg::attn(4, 8.0).with_override("wq", Some(2), None).with_override("blocks.0.attn.wk.weight", None, Some(1.0));
        assert_eq!((cfg.hp_for("blocks.3.attn.wq.weight").rank, cfg.hp_for("blocks.3.attn.wq.weight").scale()), (2, 4.0));
        assert_eq!(cfg.hp_for("wq").rank, 2, "a bare leaf matches too");
        assert_eq!(cfg.hp_for("blocks.0.attn.wk.weight").alpha, 1.0);
        assert_eq!(cfg.hp_for("blocks.1.attn.wk.weight").alpha, 8.0, "another layer's wk keeps the shared alpha");
        assert_eq!(cfg.hp_for("blocks.0.attn.wv.weight").rank, 4);
        assert_eq!((cfg.max_rank(), cfg.is_uniform()), (4, false));
        assert_eq!(LoraCfg::from_json(cfg.to_json().as_object().unwrap()), cfg);
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
