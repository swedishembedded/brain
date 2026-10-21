// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models for its clients. If your
// team needs expertise in choosing and verifying the optimizer-state precision of a training run against a fixed
// accelerator budget, you can procure our services by sending an email to info@swedishembedded.com.

//! The training precision of a full fine-tune as ONE typed value, carried by [`crate::FitOpts::precision`].
//!
//! It replaces the thread-scoped policy a build used to read (a value set on the calling thread and read several
//! calls below, which two concurrent builds could not hold apart): a trainer hands [`TrainPrecision`] to the
//! model's builder, which resolves it parameter by parameter with [`TrainPrecision::resolve`] and returns an
//! `Err` naming the parameter when a choice is not one the trainer runs. Nothing is picked silently: the optimizer
//! state is narrowed by the memory planner only when the run authorized it ([`TrainPrecision::narrow_state`]), and
//! the report of that decision comes back with the effective precision.
//!
//! The textual form is model agnostic and parsed once, `state=int8,master=f32,value=bf16,grad=bf16,sr=on,narrow=on`
//! ([`TrainPrecision::parse`]).

use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, PrecisionPolicy, StateFormat, ValueDtype};

use crate::memplan::{self, Authorized, Capacity, Category, Lever, PlanReport, TrainingFootprint};

/// What a full fine-tune stores each trainable parameter as, and whether the planner may narrow its optimizer state.
#[derive(Clone, Debug)]
pub struct TrainPrecision {
    pub policy: PrecisionPolicy,
    /// Let the memory planner pick a narrower optimizer-state format (bf16, then int8 blocks) when the requested one
    /// does not fit the device. The decision is reported; without this a run that does not fit is refused.
    pub narrow_state: bool,
}

impl Default for TrainPrecision {
    /// The historical layout: everything fp32, no master, no narrowing.
    fn default() -> Self {
        TrainPrecision { policy: PrecisionPolicy::uniform(ParamPrecision::FP32), narrow_state: false }
    }
}

impl TrainPrecision {
    pub fn uniform(p: ParamPrecision) -> TrainPrecision {
        TrainPrecision { policy: PrecisionPolicy::uniform(p), narrow_state: false }
    }

    /// Whether this is the default (all fp32): a resume file only records a precision that departs from it.
    pub fn is_default(&self) -> bool {
        self.policy.rules.is_empty() && self.policy.default == ParamPrecision::FP32 && !self.narrow_state
    }

    /// Parse `key=value,...` with keys `state` (`f32`, `bf16`, `int8[:block]`, `log8[:block]`), `master`
    /// (`none`, `bf16`, `f32`), `value` and `grad` (`f32`, `bf16`), `sr` and `narrow` (`on`, `off`). Unnamed keys keep
    /// the fp32 default. The result is uniform over every parameter.
    pub fn parse(spec: &str) -> Result<TrainPrecision, String> {
        let mut p = ParamPrecision::FP32;
        let mut narrow_state = false;
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (key, value) = item.split_once('=').ok_or_else(|| format!("train precision '{item}': expected key=value"))?;
            let flag = |v: &str| match v {
                "on" | "true" => Ok(true),
                "off" | "false" => Ok(false),
                other => Err(format!("train precision '{key}': '{other}' is not on or off")),
            };
            match key {
                "state" => p.state = parse_state(value)?,
                "master" => {
                    p.master = match value {
                        "none" => MasterPolicy::None,
                        "bf16" => MasterPolicy::Bf16,
                        "f32" => MasterPolicy::F32,
                        other => return Err(format!("train precision 'master': '{other}' is not none, bf16 or f32")),
                    }
                }
                "value" => {
                    p.value = match value {
                        "f32" => ValueDtype::F32,
                        "bf16" => ValueDtype::Bf16,
                        other => return Err(format!("train precision 'value': '{other}' is not f32 or bf16")),
                    }
                }
                "grad" => {
                    p.grad = match value {
                        "f32" => GradDtype::F32,
                        "bf16" => GradDtype::Bf16,
                        other => return Err(format!("train precision 'grad': '{other}' is not f32 or bf16")),
                    }
                }
                "sr" => p.stochastic_rounding = flag(value)?,
                "narrow" => narrow_state = flag(value)?,
                other => return Err(format!("train precision: unknown key '{other}' (state, master, value, grad, sr, narrow)")),
            }
        }
        Ok(TrainPrecision { policy: PrecisionPolicy::uniform(p), narrow_state })
    }

    /// The precision of each of `params` under this policy, checked against what the trainer runs. The device
    /// training paths keep f32 weights and gradients with no master copy and run AdamW state in f32, bf16 or
    /// 64-element int8 blocks (the fused kernels); with the state offloaded to the host any state format is
    /// available. Anything else is an `Err` naming the parameter, never a silent fp32 fallback.
    pub fn resolve(&self, params: &[(String, usize)], host_offload: bool) -> Result<Vec<ParamPrecision>, String> {
        params
            .iter()
            .map(|(name, numel)| {
                let p = self.policy.resolve(name, *numel).precision;
                if p.value != ValueDtype::F32 || p.grad != GradDtype::F32 || p.master != MasterPolicy::None {
                    return Err(format!("{name}: this trainer keeps f32 device weights and gradients with no master copy; got {p:?}"));
                }
                let device_ok = matches!(p.state, StateFormat::F32 | StateFormat::Bf16) || p.state == (StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK });
                if !host_offload && !device_ok {
                    return Err(format!(
                        "{name}: the device optimiser keeps f32 or bf16 moments or int8 blocks of {}; {:?} moments need the host-offloaded optimizer state",
                        paramstore::DEVICE_INT8_BLOCK, p.state
                    ));
                }
                Ok(p)
            })
            .collect()
    }

    /// This precision with the optimizer state in `state` for every parameter (a uniform policy only).
    fn with_state(&self, state: StateFormat) -> TrainPrecision {
        TrainPrecision { policy: PrecisionPolicy::uniform(ParamPrecision { state, ..self.policy.default }), narrow_state: self.narrow_state }
    }

    /// Plan the device footprint of a full fine-tune before anything loads. The optimizer state is held in the
    /// requested format when it fits; when it does not, a run that authorized narrowing gets the first narrower
    /// format that does (reported in the returned [`PlanReport`]) and any other run gets the planner's refusal with
    /// its arithmetic. Only a uniform policy can be narrowed.
    pub fn plan(&self, footprint: &TrainingFootprint, capacity: Capacity) -> Result<PlannedPrecision, String> {
        let authorized = if self.narrow_state { Authorized::only(&[Lever::OptimizerState]) } else { Authorized::none() };
        if self.narrow_state && !self.policy.rules.is_empty() {
            return Err("narrowing the optimizer state needs a uniform precision policy (no per-group rules)".into());
        }
        let footprint = TrainingFootprint { precision: self.policy.default, ..footprint.clone() };
        let demands = memplan::training_demands(&footprint).map_err(|e| e.to_string())?;
        let report = memplan::plan(&demands, capacity, &authorized).map_err(|e| format!("the full fine-tune does not fit: {e}"))?;
        let ladder = paramstore::precision::optimizer_state_ladder(&footprint.trainable, footprint.moment_slots, footprint.int8_block);
        let chosen = report.variant(Category::OptimizerState).map(|v| v.label.as_str()).unwrap_or_default();
        let state = ladder.iter().map(|(f, _)| *f).find(|f| chosen == format!("{f:?}")).unwrap_or(self.policy.default.state);
        Ok(PlannedPrecision { precision: if state == self.policy.default.state { self.clone() } else { self.with_state(state) }, report })
    }
}

/// A precision the planner accepted, with the report of what it decided.
#[derive(Clone, Debug)]
pub struct PlannedPrecision {
    pub precision: TrainPrecision,
    pub report: PlanReport,
}

fn parse_state(v: &str) -> Result<StateFormat, String> {
    let (kind, block) = match v.split_once(':') {
        Some((k, b)) => (k, b.parse::<u32>().map_err(|_| format!("train precision 'state': '{b}' is not a block size"))?),
        None => (v, paramstore::DEVICE_INT8_BLOCK),
    };
    if block == 0 {
        return Err("train precision 'state': block size 0".into());
    }
    match kind {
        "f32" => Ok(StateFormat::F32),
        "bf16" => Ok(StateFormat::Bf16),
        "int8" => Ok(StateFormat::Int8Block { block }),
        "log8" => Ok(StateFormat::Log8Block { block }),
        other => Err(format!("train precision 'state': '{other}' is not f32, bf16, int8 or log8")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: the textual form round-trips every knob and rejects what it cannot read, naming the key.
    #[test]
    fn parse_reads_every_knob_and_names_what_it_rejects() {
        let t = TrainPrecision::parse("state=int8:128, master=f32,value=bf16,grad=bf16,sr=on,narrow=on").unwrap();
        let p = t.policy.default;
        assert_eq!((p.state, p.master, p.value, p.grad, p.stochastic_rounding, t.narrow_state), (StateFormat::Int8Block { block: 128 }, MasterPolicy::F32, ValueDtype::Bf16, GradDtype::Bf16, true, true));
        assert!(TrainPrecision::parse("").unwrap().is_default());
        assert!(TrainPrecision::parse("state=fp4").unwrap_err().contains("state"));
        assert!(TrainPrecision::parse("moment=f32").unwrap_err().contains("moment"));
        assert!(TrainPrecision::parse("master").unwrap_err().contains("key=value"));
    }

    /// Spec: a choice the trainer does not run is an error naming the parameter, not a panic; a host-offloaded
    /// state accepts every state format, the device only the fused ones.
    #[test]
    fn resolve_returns_an_error_naming_the_parameter() {
        let params = vec![("tok.weight".to_string(), 64usize)];
        let master = TrainPrecision::parse("master=f32").unwrap();
        assert!(master.resolve(&params, true).unwrap_err().contains("tok.weight"));
        let log8 = TrainPrecision::parse("state=log8").unwrap();
        assert!(log8.resolve(&params, false).unwrap_err().contains("tok.weight"));
        assert!(log8.resolve(&params, true).is_ok());
        assert!(TrainPrecision::parse("state=bf16").unwrap().resolve(&params, false).is_ok());
    }

    fn footprint(numel: usize) -> TrainingFootprint {
        TrainingFootprint {
            frozen_weight_bytes: 0,
            trainable: vec![("w".to_string(), numel)],
            precision: ParamPrecision::FP32,
            moment_slots: 2,
            int8_block: paramstore::DEVICE_INT8_BLOCK,
            optimizer_state_on_host: false,
            activation_bytes: 0,
            recompute_activation_bytes: None,
            saved_boundaries: None,
        }
    }

    /// Spec: a run that does not fit with fp32 state is refused with the arithmetic unless it authorized
    /// narrowing, in which case the first format that fits is chosen and the decision is in the report.
    #[test]
    fn the_state_is_narrowed_only_when_authorized_and_the_report_says_so() {
        let n = 1 << 20;
        // value + gradient (8 B/elem) + fp32 moments (8 B/elem) = 16 B/elem; bf16 moments make it 12, int8 ~10.
        let room = |bytes_per_elem: u64| Capacity { device_bytes: bytes_per_elem * n as u64 + 1024, host_bytes: u64::MAX };
        let strict = TrainPrecision::default();
        assert_eq!(strict.plan(&footprint(n), room(16)).unwrap().precision.policy.default.state, StateFormat::F32);
        let err = strict.plan(&footprint(n), room(13)).unwrap_err();
        assert!(err.contains("does not fit") && err.contains("OptimizerState"), "{err}");

        let may_narrow = TrainPrecision { narrow_state: true, ..TrainPrecision::default() };
        let bf16 = may_narrow.plan(&footprint(n), room(13)).unwrap();
        assert_eq!(bf16.precision.policy.default.state, StateFormat::Bf16);
        assert!(bf16.report.decisions.iter().any(|d| d.lever == Lever::OptimizerState && d.to.contains("Bf16")), "{}", bf16.report);
        let int8 = may_narrow.plan(&footprint(n), room(11)).unwrap();
        assert_eq!(int8.precision.policy.default.state, StateFormat::Int8Block { block: paramstore::DEVICE_INT8_BLOCK });
        assert!(may_narrow.plan(&footprint(n), room(9)).is_err(), "even int8 state does not fit");
    }
}
