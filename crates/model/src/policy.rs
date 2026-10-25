// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One model-agnostic serving policy: how a model's weights and KV cache are
//! stored, which arithmetic its GEMMs use, and the serving limits it is sized
//! by. A caller (the CLI, the SDK, a server config) states it once, in
//! architecture-neutral terms; every model receives it through its own
//! [`PolicySupport`] declaration, which names what that model can do and
//! refuses the rest by model name and supported set - never a silent fallback.
//!
//! Swedish Embedded AB implements solutions for serving many model families
//! behind one precision and capacity contract for its clients. If your team
//! needs expertise in quantized serving configuration across architectures
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Precision knobs (an unsupported choice is an error)
//! - `--weight-format SPEC` ([`ServePolicy::weights`]): the weight storage format,
//!   `f32|bf16|f16|int8|q4|nf4` or a per-leaf list `q4,in_proj_a.weight=f32`
//!   ([`crate::ops::TierPolicy`]'s grammar). Unset: the model's own default.
//! - `--kv-cache TIER` ([`ServePolicy::kv_cache`]): `f32|bf16|int8|fp8`
//!   ([`crate::kv_tier::KvTier`]).
//! - `--gemm auto|fp32` ([`ServePolicy::gemm`]) and `--no-fuse-norm`,
//!   `--kv-calibration`, `--exact-decode`: arithmetic, calibration and
//!   batch-independence switches. `--exact-decode` makes a decode row's result
//!   independent of its co-batch (the row-count-gated int8 kernels run in
//!   bit-identical groups, attention is sized from the configured context);
//!   it costs an extra pass over the weights per group and is off by default.
//! # Limits: only what a person decides is a flag
//! `--ctx` (cap the window below the model's declared one), `--max-batch` (the
//! concurrency the service must carry) and `--kv-offload-gb` (host RAM granted
//! for preempted sessions) are real choices. Everything else a server sizes is
//! derived from the model's metadata, the device and the memory plan, and the
//! chosen values are reported at startup: the KV pool size, the prefix-cache
//! slots, the prefill slice and batch, the prefill row cap and the norm fusion
//! are therefore NOT flags. [`ServeLimits`] still carries them so the deprecated
//! architecture-prefixed spellings ([`rewrite_legacy`]) keep launch lines
//! working until their callers have moved; they are removed with those aliases.

use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;

use crate::kv_tier::KvTier;
use crate::ops::TierPolicy;

/// Sizing limits. `None` leaves the model's own default (they differ per model:
/// a decoder resident sizes its batch and context from the memory it has).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ServeLimits {
    /// `--ctx N`: per-sequence token ceiling (prompt plus generated).
    pub ctx: Option<u32>,
    /// `--max-batch N`: sequences decoded together.
    pub max_batch: Option<u32>,
    /// KV pool size (derived by default; set only by a deprecated alias): the KV pool's size in tokens, shared by all sequences.
    pub kv_tokens: Option<u32>,
    /// `--kv-offload-gb G`: host RAM for preempted sessions' KV (fractional).
    pub kv_offload_gb: Option<f64>,
    /// Prefix-cache slots (derived from the batch by default): cached prompts kept for prefix reuse; 0 is off.
    pub prefix_slots: Option<u32>,
    /// Cap on prompt rows prefilled per forward.
    pub max_prefill: Option<u32>,
    /// Prompt tokens one sequence gets of each prefill round.
    pub prefill_slice: Option<u32>,
    /// Prompt rows prefilled across sequences in one forward.
    pub prefill_batch: Option<u32>,
}

/// The typed policy. All fields default to "the model's own choice".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ServePolicy {
    pub weights: Option<TierPolicy>,
    pub kv_cache: Option<KvTier>,
    /// Opt in to a calibrated KV clip table beside the checkpoint.
    pub kv_calibration: bool,
    pub gemm: Option<GemmPrecision>,
    /// `Some(false)` turns the residual-add plus norm fusion off (an A/B switch).
    pub fuse_norm: Option<bool>,
    /// `--exact-decode`: make a decode row's result independent of which other
    /// sequences share its batch (see [`PolicySupport`] models that honour it).
    /// Off by default: today's batched numerics, unchanged.
    pub exact_decode: bool,
    pub limits: ServeLimits,
}

/// The public flags that take a value.
pub const VALUE_FLAGS: &[&str] = &["--weight-format", "--kv-cache", "--gemm", "--ctx", "--max-batch", "--kv-offload-gb"];
/// The public switches (no value).
pub const SWITCH_FLAGS: &[&str] = &["--kv-calibration", "--exact-decode"];
/// Settings that are derived by default and exist only as the target of a
/// deprecated alias (see the module docs). Not accepted on the command line.
pub const DERIVED_SETTINGS: &[&str] = &["kv-tokens", "prefix-slots", "max-prefill", "prefill-slice", "prefill-batch", "no-fuse-norm"];

fn count(flag: &str, v: &str) -> Result<u32, String> {
    v.trim().parse::<u32>().map_err(|_| format!("{flag} {v:?}: expected a non-negative integer"))
}

impl ServePolicy {
    /// Apply one value flag (or, from a deprecated alias, one derived setting
    /// named without dashes). Unknown names are an error - callers test
    /// [`VALUE_FLAGS`] first.
    pub fn set(&mut self, flag: &str, value: &str) -> Result<(), String> {
        match flag {
            "--weight-format" => self.weights = Some(TierPolicy::parse(&canonical_weights(value)).map_err(|e| format!("--weight-format {value:?}: {e}"))?),
            "--kv-cache" => self.kv_cache = Some(KvTier::parse(value).map_err(|e| format!("--kv-cache {value:?}: {e}"))?),
            "--gemm" => {
                self.gemm = Some(match value.trim().to_ascii_lowercase().as_str() {
                    "auto" => GemmPrecision::Auto,
                    "fp32" | "f32" => GemmPrecision::Fp32Reference,
                    other => return Err(format!("--gemm {other:?}: expected auto or fp32")),
                })
            }
            "--ctx" => self.limits.ctx = Some(count(flag, value)?),
            "--max-batch" => self.limits.max_batch = Some(count(flag, value)?),
            "kv-tokens" => self.limits.kv_tokens = Some(count(flag, value)?),
            "--kv-offload-gb" => {
                let gb: f64 = value.trim().parse().map_err(|_| format!("--kv-offload-gb {value:?}: expected a number of GiB"))?;
                if !(gb.is_finite() && gb >= 0.0) {
                    return Err(format!("--kv-offload-gb {value:?}: must be finite and not negative"));
                }
                self.limits.kv_offload_gb = Some(gb);
            }
            "prefix-slots" => self.limits.prefix_slots = Some(count(flag, value)?),
            "max-prefill" => self.limits.max_prefill = Some(count(flag, value)?),
            "prefill-slice" => self.limits.prefill_slice = Some(count(flag, value)?),
            "prefill-batch" => self.limits.prefill_batch = Some(count(flag, value)?),
            other => return Err(format!("{other} is not a serving policy flag")),
        }
        Ok(())
    }

    /// The explicitly chosen settings as one line, for the startup report.
    pub fn describe(&self) -> String {
        let mut v = Vec::new();
        if let Some(w) = &self.weights {
            v.push(format!("weights={}", w.describe()));
        }
        if let Some(k) = self.kv_cache {
            v.push(format!("kv-cache={}", k.as_str()));
        }
        if let Some(g) = self.gemm {
            v.push(format!("gemm={}", if g == GemmPrecision::Auto { "auto" } else { "fp32" }));
        }
        if self.kv_calibration {
            v.push("kv-calibration".into());
        }
        if self.exact_decode {
            v.push("exact-decode".into());
        }
        let l = &self.limits;
        for (name, x) in [("ctx", l.ctx), ("max-batch", l.max_batch), ("kv-tokens", l.kv_tokens), ("prefix-slots", l.prefix_slots), ("max-prefill", l.max_prefill), ("prefill-slice", l.prefill_slice), ("prefill-batch", l.prefill_batch)] {
            if let Some(x) = x {
                v.push(format!("{name}={x}"));
            }
        }
        if let Some(g) = l.kv_offload_gb {
            v.push(format!("kv-offload-gb={g}"));
        }
        if self.fuse_norm == Some(false) {
            v.push("norm-fusion=off".into());
        }
        v.join(" ")
    }

    /// Apply one switch.
    pub fn switch(&mut self, flag: &str) -> Result<(), String> {
        match flag {
            "--kv-calibration" => self.kv_calibration = true,
            "--exact-decode" => self.exact_decode = true,
            "no-fuse-norm" => self.fuse_norm = Some(false),
            other => return Err(format!("{other} is not a serving policy switch")),
        }
        Ok(())
    }
}

/// What one model can honour. Declared by the model (or its resident) as data;
/// [`PolicySupport::check`] is the one place an unsupported choice is refused.
#[derive(Clone, Copy, Debug)]
pub struct PolicySupport {
    /// The model's display name, used in every error.
    pub model: &'static str,
    /// Uniform weight formats accepted; empty means the model takes no weight format choice.
    pub weights: &'static [Dtype],
    /// Whether per-leaf `pattern=format` rules are honoured.
    pub weight_rules: bool,
    /// KV tiers accepted, by [`KvTier::as_str`] name; empty: no KV cache choice.
    pub kv_cache: &'static [&'static str],
    /// `--gemm` honoured.
    pub gemm: bool,
    /// `--no-fuse-norm` honoured.
    pub fuse_norm: bool,
    /// `--kv-calibration` honoured.
    pub kv_calibration: bool,
    /// `--exact-decode` honoured.
    pub exact_decode: bool,
}

impl PolicySupport {
    /// Refuse every precision choice in `policy` this model cannot honour, naming the model and
    /// the supported set. Limits are not checked here ([`Self::inapplicable_limits`]).
    pub fn check(&self, policy: &ServePolicy) -> Result<(), String> {
        if let Some(w) = &policy.weights {
            if self.weights.is_empty() {
                return Err(format!("{}: takes no weight format choice (--weight-format {})", self.model, w.describe()));
            }
            if !self.weight_rules && w.rule_count() > 0 {
                return Err(format!("{}: per-leaf weight rules are not supported (--weight-format {}); give one format of {}", self.model, w.describe(), names(self.weights)));
            }
            if !self.weights.contains(&w.default_dtype()) {
                return Err(format!("{}: weight format {} is not supported; supported: {}", self.model, pretty(w.default_dtype()), names(self.weights)));
            }
        }
        if let Some(t) = policy.kv_cache {
            if !self.kv_cache.contains(&t.as_str()) {
                return Err(format!("{}: KV cache {} is not supported; supported: {}", self.model, t.as_str(), if self.kv_cache.is_empty() { "none (no KV cache choice)".to_string() } else { self.kv_cache.join(", ") }));
            }
        }
        if policy.gemm.is_some() && !self.gemm {
            return Err(format!("{}: --gemm is not supported", self.model));
        }
        if policy.fuse_norm == Some(false) && !self.fuse_norm {
            return Err(format!("{}: disabling the residual-norm fusion is not supported", self.model));
        }
        if policy.kv_calibration && !self.kv_calibration {
            return Err(format!("{}: --kv-calibration is not supported", self.model));
        }
        if policy.exact_decode && !self.exact_decode {
            return Err(format!("{}: --exact-decode is not supported", self.model));
        }
        Ok(())
    }
}

/// The name a person writes for a weight format (`int8`, not the engine's `i8`).
pub fn format_name(d: Dtype) -> &'static str {
    match d {
        Dtype::I8 => "int8",
        Dtype::F8E4M3 => "fp8",
        Dtype::F8E5M2 => "fp8e5m2",
        other => other.name(),
    }
}

fn pretty(d: Dtype) -> &'static str {
    format_name(d)
}

fn names(d: &[Dtype]) -> String {
    d.iter().map(|d| pretty(*d)).collect::<Vec<_>>().join(", ")
}

/// `--weight-format` spellings people use (`int8`, `fp32`, ...) rewritten to the
/// engine's tier names; anything else passes through for `TierPolicy::parse`
/// to reject by name.
fn canonical_weights(spec: &str) -> String {
    let tier = |t: &str| match t.trim().to_ascii_lowercase().as_str() {
        "int8" => "i8".to_string(),
        "int4" => "q4".to_string(),
        "fp32" => "f32".to_string(),
        "fp16" => "f16".to_string(),
        "fp8" | "fp8e4m3" => "f8e4m3".to_string(),
        "fp8e5m2" => "f8e5m2".to_string(),
        other => other.to_string(),
    };
    spec.split(',')
        .map(|part| match part.split_once('=') {
            Some((pat, t)) => format!("{}={}", pat.trim(), tier(t)),
            None => tier(part),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// The frozen-base formats a LoRA trainer holds its base in, declared by the
/// model the way [`PolicySupport`] declares serving formats: one flag
/// (`--base-format`) is parsed once, and a format the model cannot train over is
/// refused naming the model and its supported set.
#[derive(Clone, Copy, Debug)]
pub struct TrainBases {
    /// The model's display name, used in every error.
    pub model: &'static str,
    /// The formats its LoRA trainer can hold the frozen base in.
    pub bases: &'static [Dtype],
}

impl TrainBases {
    /// `base` if this model can train over it.
    pub fn check(&self, base: Dtype) -> Result<Dtype, String> {
        if self.bases.contains(&base) {
            Ok(base)
        } else {
            Err(format!("{}: a LoRA base held as {} is not supported; supported: {}", self.model, format_name(base), names(self.bases)))
        }
    }

    /// A `--base-format` value (`f32|bf16|f16|int8|q4|nf4|fp8|fp8e5m2`), checked.
    pub fn parse(&self, spec: &str) -> Result<Dtype, String> {
        let tier = canonical_weights(spec);
        let base = Dtype::from_name(&tier).ok_or_else(|| format!("--base-format {spec:?}: not a weight format; {} supports {}", self.model, names(self.bases)))?;
        self.check(base)
    }
}

/// A deprecated architecture-prefixed flag and what replaces it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Replacement {
    /// Same value, generic name: `--qwen-ctx N` is `--ctx N`.
    Renamed(&'static str),
    /// A fixed value of a generic flag: `--qwen-kv-fp8` is `--kv-cache fp8`.
    Fixed(&'static str, &'static str),
    /// A generic switch: `--qwen-kv-calib` is `--kv-calibration`.
    Switch(&'static str),
    /// A value that is now derived from the model, device and memory plan. The
    /// alias still sets it (so a launch line behaves as before) until it is
    /// removed; the named setting is one of [`DERIVED_SETTINGS`].
    Derived(&'static str),
    /// A derived switch (see [`Replacement::Derived`]).
    DerivedSwitch(&'static str),
}

/// The deprecated spellings. Delete a row when its last caller has moved; the
/// layering gate's allowlist of architecture-prefixed flags names this file.
pub const LEGACY: &[(&str, Replacement)] = &[
    ("--qwen-weights", Replacement::Renamed("--checkpoint")),
    ("--qwen-ctx", Replacement::Renamed("--ctx")),
    ("--qwen-max-batch", Replacement::Renamed("--max-batch")),
    ("--qwen-kv-tokens", Replacement::Derived("kv-tokens")),
    ("--qwen-kv-offload-gb", Replacement::Renamed("--kv-offload-gb")),
    ("--qwen-max-prefill", Replacement::Derived("max-prefill")),
    ("--qwen-kv-fp32", Replacement::Fixed("--kv-cache", "f32")),
    ("--qwen-kv-fp8", Replacement::Fixed("--kv-cache", "fp8")),
    ("--qwen-kv-calib", Replacement::Switch("--kv-calibration")),
    ("--qwen-weights-int8", Replacement::Fixed("--weight-format", "int8")),
    ("--qwen-weights-fp32", Replacement::Fixed("--weight-format", "f32")),
    ("--qwen-weights-bf16", Replacement::Fixed("--weight-format", "bf16")),
    ("--qwen-no-fuse-norm", Replacement::DerivedSwitch("no-fuse-norm")),
    ("--qwen-gemm-fp32", Replacement::Fixed("--gemm", "fp32")),
    ("--qwen35-ctx", Replacement::Renamed("--ctx")),
    ("--qwen35-tier", Replacement::Renamed("--weight-format")),
    ("--qwen35-kv", Replacement::Renamed("--kv-cache")),
    ("--qwen35-prefix-slots", Replacement::Derived("prefix-slots")),
    ("--qwen35-prefill-slice", Replacement::Derived("prefill-slice")),
    ("--qwen35-prefill-batch", Replacement::Derived("prefill-batch")),
    ("--qwen35-fp32-ctx", Replacement::Renamed("--ctx")),
    ("--qwen35-fp32-max-batch", Replacement::Renamed("--max-batch")),
    ("--qwen35-max-batch", Replacement::Renamed("--max-batch")),
];

/// The replacement for a deprecated flag, if `flag` is one.
pub fn rewrite_legacy(flag: &str) -> Option<Replacement> {
    LEGACY.iter().find(|(old, _)| *old == flag).map(|(_, r)| *r)
}

/// The one-line warning to print when `old` was used.
pub fn deprecation_warning(old: &str, r: Replacement) -> String {
    match r {
        Replacement::Renamed(n) | Replacement::Switch(n) => format!("warning: {old} is deprecated and will be removed; use {n}"),
        Replacement::Fixed(n, v) => format!("warning: {old} is deprecated and will be removed; use {n} {v}"),
        Replacement::Derived(what) | Replacement::DerivedSwitch(what) => {
            format!("warning: {old} is deprecated and will be removed: the {what} setting is derived from the model, the device and the memory plan and reported at startup")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_train_base_is_parsed_once_and_an_unsupported_one_names_the_model_and_its_set() {
        const BASES: TrainBases = TrainBases { model: "toy-llm", bases: &[Dtype::F32, Dtype::I8, Dtype::NF4, Dtype::F8E4M3] };
        assert_eq!(BASES.parse("int8"), Ok(Dtype::I8));
        assert_eq!(BASES.parse("fp8"), Ok(Dtype::F8E4M3));
        assert_eq!(BASES.parse("NF4"), Ok(Dtype::NF4));
        let err = BASES.parse("bf16").unwrap_err();
        assert!(err.contains("toy-llm") && err.contains("bf16") && err.contains("f32, int8, nf4, fp8"), "{err}");
        assert!(BASES.parse("fp3").unwrap_err().contains("not a weight format"));
    }

    const LLM: PolicySupport = PolicySupport {
        model: "toy-llm",
        weights: &[Dtype::F32, Dtype::BF16, Dtype::I8],
        weight_rules: false,
        kv_cache: &["f32", "int8"],
        gemm: true,
        fuse_norm: false,
        kv_calibration: false,
        exact_decode: false,
    };

    #[test]
    fn every_flag_sets_its_field_and_a_bad_value_names_the_flag() {
        let mut p = ServePolicy::default();
        p.set("--weight-format", "int8").unwrap();
        p.set("--kv-cache", "fp8").unwrap();
        p.set("--gemm", "fp32").unwrap();
        p.set("--ctx", "4096").unwrap();
        p.set("--kv-offload-gb", "1.5").unwrap();
        p.switch("no-fuse-norm").unwrap();
        p.switch("--kv-calibration").unwrap();
        p.switch("--exact-decode").unwrap();
        assert_eq!(p.weights, Some(TierPolicy::uniform(Dtype::I8)));
        assert_eq!(p.kv_cache, Some(KvTier::FP8));
        assert_eq!(p.gemm, Some(GemmPrecision::Fp32Reference));
        assert_eq!((p.limits.ctx, p.limits.kv_offload_gb), (Some(4096), Some(1.5)));
        assert_eq!((p.fuse_norm, p.kv_calibration, p.exact_decode), (Some(false), true, true));
        assert!(p.describe().contains("exact-decode"), "{}", p.describe());
        for (flag, bad) in [("--ctx", "-1"), ("--kv-cache", "fp4"), ("--gemm", "tf32"), ("--kv-offload-gb", "nan"), ("--weight-format", "q9")] {
            let e = ServePolicy::default().set(flag, bad).unwrap_err();
            assert!(e.contains(flag), "{e}");
        }
    }

    #[test]
    fn an_unsupported_precision_choice_names_the_model_and_the_supported_set() {
        let mut p = ServePolicy::default();
        p.set("--kv-cache", "bf16").unwrap();
        let e = LLM.check(&p).unwrap_err();
        assert!(e.contains("toy-llm") && e.contains("bf16") && e.contains("f32, int8"), "{e}");

        let mut p = ServePolicy::default();
        p.set("--weight-format", "nf4").unwrap();
        let e = LLM.check(&p).unwrap_err();
        assert!(e.contains("toy-llm") && e.contains("nf4") && e.contains("f32, bf16, int8"), "{e}");

        let mut p = ServePolicy::default();
        p.set("--weight-format", "int8,up=f32").unwrap();
        assert!(LLM.check(&p).unwrap_err().contains("per-leaf"));

        let mut p = ServePolicy::default();
        p.switch("no-fuse-norm").unwrap();
        assert!(LLM.check(&p).unwrap_err().contains("fusion"));

        let mut p = ServePolicy::default();
        p.switch("--exact-decode").unwrap();
        assert!(LLM.check(&p).unwrap_err().contains("--exact-decode"));
    }

    #[test]
    fn a_supported_or_empty_policy_passes() {
        assert!(LLM.check(&ServePolicy::default()).is_ok());
        let mut p = ServePolicy::default();
        p.set("--weight-format", "bf16").unwrap();
        p.set("--kv-cache", "int8").unwrap();
        p.set("--gemm", "auto").unwrap();
        p.set("--max-batch", "8").unwrap();
        assert!(LLM.check(&p).is_ok());
    }

    #[test]
    fn derived_settings_are_not_public_flags_but_their_aliases_still_set_them() {
        for d in DERIVED_SETTINGS {
            assert!(!VALUE_FLAGS.iter().chain(SWITCH_FLAGS).any(|f| f.trim_start_matches("--") == *d), "{d} must not be a public flag");
        }
        let mut p = ServePolicy::default();
        p.set("prefix-slots", "16").unwrap();
        p.set("kv-tokens", "4096").unwrap();
        assert_eq!((p.limits.prefix_slots, p.limits.kv_tokens), (Some(16), Some(4096)));
        assert!(rewrite_legacy("--qwen35-prefix-slots").is_some());
        assert!(deprecation_warning("--qwen35-prefix-slots", Replacement::Derived("prefix-slots")).contains("derived"));
    }

    #[test]
    fn every_legacy_flag_maps_onto_a_generic_policy_flag() {
        for (old, r) in LEGACY {
            let known = match r {
                Replacement::Renamed(n) | Replacement::Fixed(n, _) => VALUE_FLAGS.contains(n) || *n == "--checkpoint",
                Replacement::Switch(n) => SWITCH_FLAGS.contains(n),
                Replacement::Derived(n) | Replacement::DerivedSwitch(n) => DERIVED_SETTINGS.contains(n),
            };
            assert!(known, "{old} maps to something that is not a policy setting");
            assert!(deprecation_warning(old, *r).contains(old));
        }
        assert_eq!(rewrite_legacy("--qwen-kv-fp8"), Some(Replacement::Fixed("--kv-cache", "fp8")));
        assert_eq!(rewrite_legacy("--ctx"), None);
    }
}
