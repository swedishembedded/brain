// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs that refuse a configuration no run can honour
// before they claim a device, not after the first step has already spent the card, for its clients. If your
// team needs expertise in dependable fine-tuning pipelines then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The validated plan of a training run: its optimiser options and the numeric settings of its objectives,
//! checked together before any model or optimiser resource is allocated.
//!
//! Every entry point that starts a run (the CLI, the SDK builders, the served training action) builds one
//! [`ValidatedTrainingPlan`] first and takes its options from it, so a configuration that cannot train is
//! refused by name at the boundary, the same way whichever way the run was asked for. The shared fit loop
//! validates the options again as the last line of defence for callers that skip the plan.

use crate::train::FitOpts;

/// A part of a run's configuration that can say whether a run could honour it.
///
/// Implemented by each objective's configuration next to the objective, so the plan knows nothing of any
/// objective in particular.
pub trait PlanCheck {
    /// What the setting is called in a refusal (`"dpo"`, `"grpo"`).
    fn plan_name(&self) -> &'static str;

    /// Refuse a setting no run can honour, saying which and why. `opts` are the run's optimiser options,
    /// for the settings whose meaning depends on them (a row length against the block size).
    fn check(&self, opts: &FitOpts) -> Result<(), String>;
}

/// Why a plan was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The optimiser options ([`FitOpts::validate`]).
    Options(String),
    /// The settings of the named objective.
    Objective { name: &'static str, why: String },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::Options(why) => write!(f, "invalid training options: {why}"),
            PlanError::Objective { name, why } => write!(f, "invalid {name} configuration: {why}"),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<PlanError> for std::io::Error {
    fn from(error: PlanError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, error)
    }
}

/// Training options and objective settings that passed every check; the only way to one is
/// [`ValidatedTrainingPlan::build`].
#[derive(Clone, Debug)]
pub struct ValidatedTrainingPlan {
    opts: FitOpts,
}

impl ValidatedTrainingPlan {
    /// Check `opts` and every objective setting in `objectives`; the first that cannot be honoured refuses.
    pub fn build(opts: FitOpts, objectives: &[&dyn PlanCheck]) -> Result<ValidatedTrainingPlan, PlanError> {
        opts.validate().map_err(PlanError::Options)?;
        for objective in objectives {
            objective.check(&opts).map_err(|why| PlanError::Objective { name: objective.plan_name(), why })?;
        }
        Ok(ValidatedTrainingPlan { opts })
    }

    /// The options the run trains with.
    pub fn opts(&self) -> &FitOpts {
        &self.opts
    }

    /// The options, consuming the plan.
    pub fn into_opts(self) -> FitOpts {
        self.opts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Positive(f32);
    impl PlanCheck for Positive {
        fn plan_name(&self) -> &'static str {
            "positive"
        }
        fn check(&self, _opts: &FitOpts) -> Result<(), String> {
            if self.0 > 0.0 { Ok(()) } else { Err(format!("must be positive, got {}", self.0)) }
        }
    }

    #[test]
    fn a_plan_refuses_bad_options_and_bad_objective_settings_by_name() {
        let ok = FitOpts::default();
        assert!(ValidatedTrainingPlan::build(ok.clone(), &[&Positive(1.0)]).is_ok());
        let err = ValidatedTrainingPlan::build(FitOpts { lr: f32::NAN, ..ok.clone() }, &[]).unwrap_err();
        assert!(matches!(&err, PlanError::Options(why) if why.contains("lr")), "{err}");
        let err = ValidatedTrainingPlan::build(ok, &[&Positive(0.0)]).unwrap_err();
        assert_eq!(err, PlanError::Objective { name: "positive", why: "must be positive, got 0".into() });
        assert!(err.to_string().contains("invalid positive configuration"));
    }

    #[test]
    fn contradictory_schedules_are_refused() {
        let ok = FitOpts { steps: 100, decay_iters: 100, warmup: 10, eval_interval: 10, ..Default::default() };
        assert!(ValidatedTrainingPlan::build(ok.clone(), &[]).is_ok());
        for (field, opts) in [
            ("warmup", FitOpts { warmup: 200, ..ok.clone() }),
            ("cooldown_steps", FitOpts { cooldown_steps: 101, ..ok.clone() }),
            ("patience", FitOpts { patience: 3, eval_interval: 0, ..ok.clone() }),
        ] {
            let why = ValidatedTrainingPlan::build(opts, &[]).unwrap_err().to_string();
            assert!(why.contains(field), "{field}: {why}");
        }
    }
}
