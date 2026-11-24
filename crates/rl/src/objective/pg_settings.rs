// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training services whose every setting is declared once and reaches the
// API, the library and the command line alike, for its clients. If your team needs expertise in reinforcement-learning
// trainers that resume exactly and never drop a setting between front ends then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The policy-gradient settings, declared once: one row per field of [`PolicyGradientConfig`], from which every front
//! end takes what it shows and accepts.
//!
//! A row ([`PgSetting`]) names the field (its name is the wire name, the settings key and the field's name), the help
//! and range the served `brain/train` action declares, how the CLI takes it, how it enters the run identity an exact
//! resume is keyed by and the settings a run reports, and a value off its default. Its value moves in one form, the
//! wire form ([`PgSetting::get`], [`PgSetting::set`]): a number, a name (`z-score`), a correction (`truncate:2`).
//! [`PgSettingsPatch`] is what a request names, row by row, laid over a configuration (the default, or a named
//! algorithm's) by [`PgSettingsPatch::apply`]. [`identity`] and [`settings`] are generated from the rows.
//!
//! **Compatibility.** The identity of a run that could be started before this table existed is byte-identical to what
//! brain wrote then: each row that existed carries the encoding it had ([`InIdentity::Bits`], [`InIdentity::Wire`] or
//! a [`InIdentity::Legacy`] key and value). A row added later is [`InIdentity::WhenNotDefault`] and
//! [`InReport::WhenNotDefault`]: it joins the identity and the settings only when its value is not the default, so a
//! run that does not use it keeps the identity, and the adapter card, it had.
//!
//! **Adding a setting.** Add the field to [`PolicyGradientConfig`] (or its [`PgLoss`]) and to [`FIELDS`] (a field
//! missing there does not compile), and one row here: its served param, CLI flag and identity follow. The table's own test refuses a field
//! without a row, a row without a field, and a surface left out without a reason.

use serde_json::{json, Value};

use super::advantage::{AdvantageFloor, AdvantageNorm};
use super::external::{BehaviourSource, ImportanceCorrection, PolicyGradientConfig, ReferencePolicy};
use super::grpo::LossAgg;
use super::numerics::NumericsBounds;
use super::pg_loss::PgLoss;
use data::chat_rollout::TruncatedRollouts;
use data::tokenizer_identity::TokenizerIdentity;
use model::kv_tier::KvElement;

/// How a setting's value is typed.
#[derive(Clone, Copy, Debug)]
pub enum SettingKind {
    /// A number (an `f32`), bounded to `[min, max]` on the served action.
    Float { min: f64, max: f64 },
    /// One of the named values.
    Choice(fn() -> Vec<&'static str>),
    /// Text the setting reads itself (`truncate:2`, `start`).
    Text,
    /// `true` or `false`.
    Switch,
    /// A JSON object (the numerics gate's bounds).
    Document,
    /// Worked out by the trainer, never set.
    Computed,
}

/// Whether the served `brain/train` action takes the setting.
#[derive(Clone, Copy, Debug)]
pub enum Served {
    /// A param of its name, laid over the configuration by [`PgSettingsPatch`].
    Param,
    /// A param of its name that the action resolves itself, together with a param of its own (the aggregation and
    /// `loss_agg_length`): the value it declares for a configuration.
    Own(fn(&PolicyGradientConfig) -> Value),
    /// Not on the served action, and why.
    No(&'static str),
}

/// How `brain finetune` takes the setting.
#[derive(Clone, Copy, Debug)]
pub enum Flag {
    /// This flag, with a value in the wire form (or the row's `flag_value`).
    Named(&'static str),
    /// This flag, which the CLI parses itself with flags of its own (the numerics gate's bounds, the aggregation's
    /// length).
    Own(&'static str),
    /// No flag, and why.
    No(&'static str),
}

/// How the setting enters the run identity ([`identity`]).
#[derive(Clone, Copy, Debug)]
pub enum InIdentity {
    /// `<name>_bits`: the `f32`'s bits (`null` when unset).
    Bits,
    /// `<name>`: the wire form.
    Wire,
    /// The key and the value the identity carried for it before this table.
    Legacy(&'static str, fn(&PolicyGradientConfig) -> Value),
    /// `<name>`: the wire form, only when it is not the default's: a setting added after runs were started.
    WhenNotDefault,
    /// Not part of the identity, and why.
    Excluded(&'static str),
}

/// How the setting enters the settings a run reports ([`settings`]: the adapter card, the result).
#[derive(Clone, Copy, Debug)]
pub enum InReport {
    /// `<name>`: the wire form.
    Always,
    /// `<name>`: the wire form, only when it is not the default's: a setting added after runs were started.
    WhenNotDefault,
    /// Not reported, and why.
    Excluded(&'static str),
}

/// One setting of the policy gradient.
pub struct PgSetting {
    /// The field, the served param, the settings key.
    pub name: &'static str,
    /// What the served action says about it.
    pub help: &'static str,
    pub kind: SettingKind,
    pub served: Served,
    pub flag: Flag,
    pub identity: InIdentity,
    pub report: InReport,
    /// A value off the default, in the wire form, that a configuration otherwise at its default accepts (none for a
    /// computed setting): what the front ends' tests set.
    pub example: Option<&'static str>,
    get: fn(&PolicyGradientConfig) -> Value,
    set: fn(&mut PolicyGradientConfig, &Value) -> Result<(), String>,
    /// The flag's text in the wire form, where it is not the text itself (or the number it reads as).
    flag_value: Option<FlagReader>,
}

/// How a row reads its flag's text into the wire form.
type FlagReader = fn(&str) -> Result<Value, String>;

impl std::fmt::Debug for PgSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PgSetting").field("name", &self.name).field("kind", &self.kind).field("served", &self.served).field("flag", &self.flag).field("identity", &self.identity).field("report", &self.report).finish_non_exhaustive()
    }
}

impl PgSetting {
    /// The setting's value in `config`, in the wire form (`null` when unset).
    pub fn get(&self, config: &PolicyGradientConfig) -> Value {
        (self.get)(config)
    }

    /// Sets the field from its wire form.
    ///
    /// # Errors
    /// The value is not the setting's type, or names none of its values; the error names the setting.
    pub fn set(&self, config: &mut PolicyGradientConfig, value: &Value) -> Result<(), String> {
        (self.set)(config, value).map_err(|why| format!("{}: {why}", self.name))
    }

    /// A flag's text in the wire form: a number for a numeric setting, the text otherwise (or what the row reads it as).
    ///
    /// # Errors
    /// A numeric setting's text is not a number (`None`), or the row's own reading refuses it (`Some`).
    pub fn flag_value(&self, text: &str) -> Result<Value, Option<String>> {
        if let Some(read) = self.flag_value {
            return read(text).map_err(Some);
        }
        match self.kind {
            // Read as the f32 it is: a decimal read as f64 first could round twice.
            SettingKind::Float { .. } => text.parse::<f32>().ok().filter(|v| v.is_finite()).map(|v| json!(v)).ok_or(None),
            SettingKind::Switch => text.parse::<bool>().map(Value::from).map_err(|_| None),
            _ => Ok(Value::from(text)),
        }
    }

    /// The flag the CLI takes the setting by, if any.
    pub fn flag_name(&self) -> Option<&'static str> {
        match self.flag {
            Flag::Named(flag) | Flag::Own(flag) => Some(flag),
            Flag::No(_) => None,
        }
    }

    fn differs_from_default(&self, config: &PolicyGradientConfig) -> bool {
        self.get(config) != self.get(&PolicyGradientConfig::default())
    }
}

/// A value's wire form.
trait Wire: Sized {
    fn to_wire(&self) -> Value;
    fn from_wire(value: &Value) -> Result<Self, String>;
}

impl Wire for f32 {
    fn to_wire(&self) -> Value {
        // `json!` of an f32: the form the settings were always written in.
        json!(self)
    }
    fn from_wire(value: &Value) -> Result<f32, String> {
        value.as_f64().map(|v| v as f32).ok_or_else(|| format!("{value} is not a number"))
    }
}

impl Wire for bool {
    fn to_wire(&self) -> Value {
        json!(self)
    }
    fn from_wire(value: &Value) -> Result<bool, String> {
        value.as_bool().ok_or_else(|| format!("{value} is not true or false"))
    }
}

impl<T: Wire> Wire for Option<T> {
    fn to_wire(&self) -> Value {
        self.as_ref().map_or(Value::Null, Wire::to_wire)
    }
    fn from_wire(value: &Value) -> Result<Option<T>, String> {
        if value.is_null() {
            return Ok(None);
        }
        T::from_wire(value).map(Some)
    }
}

fn text(value: &Value) -> Result<&str, String> {
    value.as_str().ok_or_else(|| format!("{value} is not text"))
}

/// A type whose wire form is its `name()`, read back by `parse`.
macro_rules! named_wire {
    ($type:ty, $parse:expr) => {
        impl Wire for $type {
            fn to_wire(&self) -> Value {
                json!(self.name())
            }
            fn from_wire(value: &Value) -> Result<$type, String> {
                text(value).and_then($parse)
            }
        }
    };
}

named_wire!(ReferencePolicy, |s: &str| s.parse());
named_wire!(BehaviourSource, |s: &str| s.parse());
named_wire!(AdvantageNorm, AdvantageNorm::parse);
named_wire!(AdvantageFloor, AdvantageFloor::parse);
named_wire!(TruncatedRollouts, TruncatedRollouts::parse);

/// A type whose wire form is its `Display`, read back by `FromStr`.
macro_rules! displayed_wire {
    ($type:ty) => {
        impl Wire for $type {
            fn to_wire(&self) -> Value {
                json!(self.to_string())
            }
            fn from_wire(value: &Value) -> Result<$type, String> {
                text(value)?.parse()
            }
        }
    };
}

displayed_wire!(ImportanceCorrection);
displayed_wire!(LossAgg);

/// The KV encodings by their debug names, the form the settings always reported.
const KV_ELEMENTS: [KvElement; 4] = [KvElement::F32, KvElement::Bf16, KvElement::Int8, KvElement::Fp8E4m3];

impl Wire for KvElement {
    fn to_wire(&self) -> Value {
        json!(format!("{self:?}"))
    }
    fn from_wire(value: &Value) -> Result<KvElement, String> {
        let name = text(value)?;
        KV_ELEMENTS.into_iter().find(|e| format!("{e:?}") == name).ok_or_else(|| format!("{name:?} is none of {}", KV_ELEMENTS.map(|e| format!("{e:?}")).join(", ")))
    }
}

impl Wire for NumericsBounds {
    fn to_wire(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
    fn from_wire(value: &Value) -> Result<NumericsBounds, String> {
        serde_json::from_value(value.clone()).map_err(|e| e.to_string())
    }
}

impl Wire for TokenizerIdentity {
    fn to_wire(&self) -> Value {
        json!(self.to_string())
    }
    fn from_wire(_: &Value) -> Result<TokenizerIdentity, String> {
        Err("the trainer works it out by comparing the teacher model's tokenizer with the base's; name the teacher model instead".to_string())
    }
}

/// One row: the field at `$path`, read and written through its wire form.
macro_rules! row {
    ($name:literal, $($path:ident).+, $kind:expr, $help:literal, served: $served:expr, flag: $flag:expr, identity: $identity:expr, report: $report:expr, example: $example:expr $(, flag_value: $flag_value:expr)?) => {
        PgSetting {
            name: $name,
            help: $help,
            kind: $kind,
            served: $served,
            flag: $flag,
            identity: $identity,
            report: $report,
            example: $example,
            get: |c| Wire::to_wire(&c.$($path).+),
            set: |c, v| {
                c.$($path).+ = Wire::from_wire(v)?;
                Ok(())
            },
            flag_value: row!(@flag_value $($flag_value)?),
        }
    };
    (@flag_value) => { None };
    (@flag_value $flag_value:expr) => { Some($flag_value) };
}

/// Why a served setting has no CLI flag: `brain finetune` never offered one.
const NOT_ON_THE_CLI: &str = "brain finetune never took it (the served action and the SDK do); a flag is a CLI contract change, one `Flag::Named` here";

/// Every setting, in the order the served action lists them.
pub static SETTINGS: &[PgSetting] = &[
    row!("clip_eps", loss.clip_eps, SettingKind::Float { min: 0.0, max: 1.0 }, "policy gradient: half-width of the clip band of the importance ratio",
        served: Served::Param, flag: Flag::Named("--ratio-clip"), identity: InIdentity::Bits, report: InReport::Always, example: Some("0.3")),
    row!("clip_eps_high", loss.clip_eps_high, SettingKind::Float { min: 0.0, max: 10.0 }, "policy gradient: the upper edge of the clip band is 1 + this (clip-higher), at least clip_eps; omitted, the band is symmetric",
        served: Served::Param, flag: Flag::Named("--ratio-clip-high"), identity: InIdentity::Bits, report: InReport::Always, example: Some("0.35")),
    row!("dual_clip", loss.dual_clip, SettingKind::Float { min: 1.0, max: 1e9 }, "policy gradient: dual-clip PPO's bound c on a negative advantage's term, max(min(r*A, clip(r)*A), c*A): a token whose ratio ran past c costs at most c*|A| and gives no gradient (each step reports the share as clip_fraction_dual); above 1",
        served: Served::Param, flag: Flag::Named("--dual-clip"), identity: InIdentity::Bits, report: InReport::Always, example: Some("4.0")),
    row!("importance", importance, SettingKind::Text, "policy gradient: the correction for the sampling engine's gap to the trainer: `none`, `truncate:CAP` or `mask:CAP`; needs recorded behaviour log-probabilities",
        served: Served::Param, flag: Flag::Named("--importance"), identity: InIdentity::Legacy("importance", |c| serde_json::to_value(c.importance).unwrap_or_default()), report: InReport::Always, example: Some("\"truncate:2\"")),
    row!("max_step_kl", loss.max_step_kl, SettingKind::Float { min: 0.0, max: 1e9 }, "policy gradient: skip (and log) an optimiser step whose mean KL to the reference is over this",
        served: Served::Param, flag: Flag::Named("--max-step-kl"), identity: InIdentity::Bits, report: InReport::Always, example: Some("0.1")),
    row!("max_step_grad_norm", loss.max_step_grad_norm, SettingKind::Float { min: 0.0, max: 1e12 }, "policy gradient: skip (and log) an optimiser step whose gradient norm is over this",
        served: Served::Param, flag: Flag::Named("--max-step-grad-norm"), identity: InIdentity::Bits, report: InReport::Always, example: Some("5.0")),
    row!("kl_beta", loss.kl_beta, SettingKind::Float { min: 0.0, max: 1000.0 }, "policy gradient: coefficient of the KL term to the reference policy",
        served: Served::Param, flag: Flag::Named("--kl-beta"), identity: InIdentity::Bits, report: InReport::Always, example: Some("0.05")),
    row!("reference", reference, SettingKind::Choice(|| ReferencePolicy::ALL.map(ReferencePolicy::name).to_vec()), "policy gradient: the policy the KL term is taken against",
        served: Served::Param, flag: Flag::No(NOT_ON_THE_CLI), identity: InIdentity::Wire, report: InReport::Always, example: Some("\"start\"")),
    row!("behaviour", behaviour, SettingKind::Choice(|| BehaviourSource::ALL.map(BehaviourSource::name).to_vec()), "policy gradient: where the behaviour log-probabilities of the importance ratio come from: `recorded` (default) the sampler's own, required on every supervised token; `start_of_run_proxy` the start-of-run policy's where none was recorded, exact only if the rollouts were sampled from it",
        served: Served::Param, flag: Flag::Named("--behaviour"), identity: InIdentity::Wire, report: InReport::Always, example: Some("\"start_of_run_proxy\"")),
    row!("loss_agg", loss.loss_agg, SettingKind::Choice(|| vec!["seq-mean", "token-sum", "constant-length"]), "policy gradient: how the token terms are normalised: `seq-mean` (each sequence by its own token count), `token-sum` (every token of a step by the step's count) or `constant-length` (each sequence's sum by loss_agg_length, averaged over the sequences: Dr. GRPO)",
        served: Served::Own(|c| json!(c.loss.loss_agg.name())), flag: Flag::Own("--loss-agg"), identity: InIdentity::Legacy("loss_agg", |c| json!(format!("{:?}", c.loss.loss_agg))), report: InReport::Always, example: Some("\"token-sum\"")),
    row!("temperature", temperature, SettingKind::Float { min: 0.0, max: 100.0 }, "policy gradient: the temperature the rollouts were sampled at",
        served: Served::Param, flag: Flag::No(NOT_ON_THE_CLI), identity: InIdentity::Bits, report: InReport::Always, example: Some("0.7")),
    row!("teacher", teacher, SettingKind::Text, "policy gradient with teacher contexts: the frozen policy that scores the student's tokens in the privileged context, `base` or `start`; distill: the policy of the same base that teaches, `base`, `start` (the start_adapter) or `adapter:<blake3 digest>` (an adapter in the artifact store)",
        served: Served::Param, flag: Flag::No("brain finetune's --teacher is the distill objective's teacher"), identity: InIdentity::Wire, report: InReport::Always, example: Some("\"start\"")),
    row!("advantage_bound", advantage_bound, SettingKind::Float { min: 0.0, max: 1e6 }, "policy gradient with teacher contexts: each token's advantage is clipped to [-bound, bound]",
        served: Served::Param, flag: Flag::No(NOT_ON_THE_CLI), identity: InIdentity::Bits, report: InReport::Always, example: Some("2.0")),
    row!("sequence_advantage_cap", sequence_advantage_cap, SettingKind::Float { min: 0.0, max: 1e9 }, "policy gradient with teacher contexts: the most the magnitudes of a sequence's token advantages may sum to",
        served: Served::Param, flag: Flag::No(NOT_ON_THE_CLI), identity: InIdentity::Bits, report: InReport::Always, example: Some("3.0")),
    row!("advantage_floor", loss.advantage_floor, SettingKind::Choice(|| AdvantageFloor::ALL.map(AdvantageFloor::name).to_vec()), "policy gradient: `none` (default) or `zero`: each advantage clamped at zero after it is computed against its whole group, so a rollout under the baseline sets it but carries no policy gradient (a positive-only run)",
        served: Served::Param, flag: Flag::Named("--advantage-floor"), identity: InIdentity::Wire, report: InReport::Always, example: Some("\"zero\"")),
    row!("truncated", truncated, SettingKind::Choice(|| TruncatedRollouts::ALL.map(TruncatedRollouts::name).to_vec()), "policy gradient: what a rollout marked `truncated` does: `mask` (never trained, its reward takes no part in its group: DAPO's overlong filtering) or `keep` (trained, its reward counted)",
        served: Served::Param, flag: Flag::Named("--truncated"), identity: InIdentity::Wire, report: InReport::Always, example: Some("\"keep\"")),
    row!("advantage_norm", loss.advantage_norm, SettingKind::Choice(|| AdvantageNorm::ALL.map(AdvantageNorm::name).to_vec()), "policy gradient: how each group's rewards become the advantages of its rollouts: `mean-centred` (r - mean), `z-score` ((r - mean) / (std + 1e-4)) or `leave-one-out` (r_i - the mean of the others)",
        served: Served::Param, flag: Flag::Named("--advantage-norm"), identity: InIdentity::Wire, report: InReport::Always, example: Some("\"z-score\"")),
    row!("numerics", numerics, SettingKind::Document, "policy gradient: run as the numerics gate, no optimiser step, judged against these bounds",
        served: Served::No("the numerics gate is a measurement run (no optimiser step, no adapter worth serving) that brain finetune --numerics-gate runs on the trainer's host"),
        flag: Flag::Own("--numerics-gate"), identity: InIdentity::Legacy("numerics_gate", |c| serde_json::to_value(c.numerics).unwrap_or_default()),
        report: InReport::Excluded("a gate run's bounds and what it measured are the run's own numerics report"), example: Some("{\"max_abs_gap\": 1.5}")),
    row!("kv_roundtrip", kv_roundtrip, SettingKind::Text, "policy gradient: read the trainer's attention keys and values through the encoding of the KV cache the rollouts were sampled with",
        served: Served::No("not offered on the served action: it trains in the trainer's own precision"), flag: Flag::Named("--kv-roundtrip"), identity: InIdentity::Wire, report: InReport::Always, example: Some("\"Int8\""),
        flag_value: |tier| model::kv_tier::KvTier::parse(tier).map(|t| t.element().to_wire())),
    row!("log_kl", loss.log_kl, SettingKind::Switch, "policy gradient: measure and report the KL to the reference when kl_beta is zero",
        served: Served::No("the served action always measures the KL (the default)"), flag: Flag::No("brain finetune always measures the KL (the default)"),
        identity: InIdentity::Excluded("a measurement: it changes no weight by itself, and the KL bound it enables (max_step_kl) is in the identity"),
        report: InReport::Excluded("a measurement: each step's kl is null when it was not measured"), example: Some("false")),
    row!("teacher_tokenizer", teacher_tokenizer, SettingKind::Computed, "policy gradient: how the tokenizer of the teacher model whose scores the records carry compares with the student's",
        served: Served::No("the server compares the tokenizer of distill_teacher, the teacher model a request names"), flag: Flag::No("worked out from the teacher model, which the SDK names"), identity: InIdentity::Wire, report: InReport::Always, example: None),
];

/// Every field of [`PolicyGradientConfig`] and of its [`PgLoss`], by name. `_EVERY_FIELD` destructures the config with
/// each of them and no `..`, so a field added to either does not compile until it is listed here, and the table's test
/// then asks for its row.
macro_rules! fields {
    (loss { $($loss:ident),* $(,)? }, $($field:ident),* $(,)?) => {
        /// Every field of [`PolicyGradientConfig`] and of its [`PgLoss`], by name.
        pub const FIELDS: &[&str] = &[$(stringify!($loss),)* $(stringify!($field)),*];
        const _EVERY_FIELD: fn(PolicyGradientConfig) = |c| {
            let PolicyGradientConfig { loss: PgLoss { $($loss: _),* }, $($field: _),* } = c;
        };
    };
}

fields!(
    loss { clip_eps, clip_eps_high, dual_clip, kl_beta, log_kl, loss_agg, advantage_norm, advantage_floor, max_step_kl, max_step_grad_norm },
    importance, numerics, kv_roundtrip, reference, behaviour, temperature, teacher, advantage_bound, sequence_advantage_cap, truncated, teacher_tokenizer,
);

/// The row of the setting `name`.
pub fn setting(name: &str) -> Option<&'static PgSetting> {
    SETTINGS.iter().find(|s| s.name == name)
}

/// What a configuration contributes to the identity of the run that trains with it: every row by its
/// [`InIdentity`].
pub fn identity(config: &PolicyGradientConfig) -> Value {
    let mut entries = serde_json::Map::new();
    for s in SETTINGS {
        let (key, value) = match s.identity {
            InIdentity::Bits => (format!("{}_bits", s.name), s.get(config).as_f64().map_or(Value::Null, |v| json!((v as f32).to_bits()))),
            InIdentity::Wire => (s.name.to_string(), s.get(config)),
            InIdentity::Legacy(key, value) => (key.to_string(), value(config)),
            InIdentity::WhenNotDefault if s.differs_from_default(config) => (s.name.to_string(), s.get(config)),
            InIdentity::WhenNotDefault | InIdentity::Excluded(_) => continue,
        };
        entries.insert(key, value);
    }
    Value::Object(entries)
}

/// Every setting a run trained with, in the wire form, by each row's [`InReport`], and the estimator the teacher's
/// tokenizer selects ([`super::onpolicy_distill::DistillEstimator`]).
pub fn settings(config: &PolicyGradientConfig) -> Value {
    let mut entries = serde_json::Map::new();
    for s in SETTINGS {
        let reported = match s.report {
            InReport::Always => true,
            InReport::WhenNotDefault => s.differs_from_default(config),
            InReport::Excluded(_) => false,
        };
        if reported {
            entries.insert(s.name.to_string(), s.get(config));
        }
    }
    entries.insert("distill_estimator".to_string(), json!(config.teacher_tokenizer.map(|t| super::onpolicy_distill::DistillEstimator::for_tokenizers(&t).name())));
    Value::Object(entries)
}

/// The settings a request names, each in the wire form, in the order named: laid over a configuration by
/// [`Self::apply`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PgSettingsPatch(Vec<(&'static str, Value)>);

impl PgSettingsPatch {
    /// Names `value` for the setting `name`: a later value of one setting replaces the earlier.
    ///
    /// # Errors
    /// No setting is named `name`, or the value is not one it takes (checked against the default configuration, so
    /// the refusal comes when the request is read). The error is the reason alone: the caller names the setting as
    /// its surface does (a param, a flag).
    pub fn insert(&mut self, name: &str, value: Value) -> Result<(), String> {
        let row = setting(name).ok_or_else(|| format!("{name} is not a policy-gradient setting"))?;
        (row.set)(&mut PolicyGradientConfig::default(), &value)?;
        self.0.retain(|(n, _)| *n != row.name);
        self.0.push((row.name, value));
        Ok(())
    }

    /// The settings named, in the order named.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.0.iter().map(|(name, _)| *name)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The value named for `name`, if any.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.0.iter().find(|(n, _)| *n == name).map(|(_, v)| v)
    }

    /// Lays every named setting over `config`.
    ///
    /// # Errors
    /// A value the setting does not take; the error names the setting.
    pub fn apply(&self, config: &mut PolicyGradientConfig) -> Result<(), String> {
        for (name, value) in &self.0 {
            setting(name).expect("a patch holds only settings of the table").set(config, value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table and the config agree: every field has exactly one row and every row a field; each surface a row
    /// leaves out says why; a row added after the table existed joins the identity and the report only off its
    /// default.
    #[test]
    fn every_field_has_one_row_and_every_omission_a_reason() {
        let rows: Vec<&str> = SETTINGS.iter().map(|s| s.name).collect();
        for field in FIELDS {
            assert_eq!(rows.iter().filter(|r| *r == field).count(), 1, "{field}: one row per field");
        }
        for s in SETTINGS {
            assert!(FIELDS.contains(&s.name), "{}: a row names a field", s.name);
            for (surface, reason) in [("served", matches!(s.served, Served::No(r) if r.is_empty())), ("flag", matches!(s.flag, Flag::No(r) if r.is_empty())), ("identity", matches!(s.identity, InIdentity::Excluded(r) if r.is_empty())), ("report", matches!(s.report, InReport::Excluded(r) if r.is_empty()))] {
                assert!(!reason, "{}: left out of the {surface} without a reason", s.name);
            }
            if matches!(s.served, Served::Param | Served::Own(_)) {
                assert!(s.help.starts_with("policy gradient"), "{}: the served help says what it configures", s.name);
            }
        }
    }

    /// Each row reads and writes its own field: its example is off the default, is accepted, and reads back as
    /// itself; the identity and the report then differ from the default's where the row enters them.
    #[test]
    fn every_row_round_trips_its_example_through_its_own_field() {
        let default = PolicyGradientConfig::default();
        for s in SETTINGS {
            let Some(example) = s.example else {
                assert!(matches!(s.kind, SettingKind::Computed), "{}: a settable row has an example", s.name);
                assert!(s.set(&mut { default }, &json!("identical")).is_err(), "{}: a computed setting is never set", s.name);
                continue;
            };
            let value: Value = serde_json::from_str(example).unwrap();
            let mut config = default;
            s.set(&mut config, &value).unwrap_or_else(|e| panic!("{e}"));
            assert_ne!(config, default, "{}: the example is off the default", s.name);
            for other in SETTINGS.iter().filter(|o| o.name != s.name) {
                assert_eq!(other.get(&config), other.get(&default), "{}: setting it changed {}", s.name, other.name);
            }
            let read = s.get(&config);
            let mut again = default;
            s.set(&mut again, &read).unwrap();
            assert_eq!(again, config, "{}: the wire form reads back", s.name);
            if !matches!(s.identity, InIdentity::Excluded(_)) {
                assert_ne!(identity(&config), identity(&default), "{}: in the identity", s.name);
            }
            if !matches!(s.report, InReport::Excluded(_)) {
                assert_ne!(settings(&config), settings(&default), "{}: in the settings", s.name);
            }
        }
    }

    /// A patch takes only settings of the table, refuses a value its setting does not take when it is named, and
    /// lays the named settings over the configuration it is applied to.
    #[test]
    fn a_patch_names_settings_and_lays_them_over_a_configuration() {
        let mut patch = PgSettingsPatch::default();
        assert!(patch.insert("ratio_clip", json!(0.1)).unwrap_err().contains("ratio_clip"));
        assert!(patch.insert("advantage_norm", json!("standardised")).unwrap_err().contains("leave-one-out"), "the setting's own reason");
        patch.insert("kl_beta", json!(0.5)).unwrap();
        patch.insert("truncated", json!("keep")).unwrap();
        patch.insert("kl_beta", json!(0.25)).unwrap();
        assert_eq!(patch.names().collect::<Vec<_>>(), ["truncated", "kl_beta"]);
        let mut config = super::super::external::PolicyGradientAlgorithm::Dapo.config(64);
        patch.apply(&mut config).unwrap();
        assert_eq!((config.loss.kl_beta, config.truncated, config.loss.loss_agg), (0.25, TruncatedRollouts::Keep, LossAgg::TokenSum));
    }
}
