// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs whose every
// weight update can be trusted - a non-finite value never reaches the weights
// or the optimiser state without the run saying so - for its clients. If your
// team needs expertise in reliable model training then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The transactional gate in front of every optimiser update of the shared
//! fit loop.
//!
//! An update is committed (moments, applied-step count, weights) only after
//! the step's loss and its reduced gradients were checked; a step that fails
//! the check leaves all of them as they were. Objectives decide separately
//! whether a step holds anything to learn from ([`crate::StepVerdict`]); this
//! gate is the universal backstop for values that must never reach the
//! optimiser whatever the objective: a non-finite loss or gradient.
//!
//! Policy: a failing step is skipped and logged, [`MAX_CONSECUTIVE_NONFINITE`]
//! in a row abort the run, and in strict mode ([`crate::FitOpts::strict_updates`])
//! the first one does.

use std::fmt;

use crate::Model;

/// Consecutive optimiser steps with a non-finite loss or gradient after which
/// the run is aborted instead of skipping again.
pub const MAX_CONSECUTIVE_NONFINITE: u32 = 3;

/// Most floats of trainable gradient the gate reads back to the host and scans
/// before every update (a LoRA adapter is far below it). A run with more
/// trainable floats is checked after its update by the clip pass's gradient
/// norm instead, and says so once.
pub const MAX_SCANNED_GRAD_FLOATS: usize = 64 << 20;

/// The run was aborted because the loss or gradient was non-finite on
/// [`MAX_CONSECUTIVE_NONFINITE`] optimiser steps in a row (or on one step in
/// strict mode). Carried as the source of the `std::io::Error` the training
/// loop returns, so a caller can `downcast_ref` it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonFiniteSteps {
    pub consecutive: u32,
}

impl fmt::Display for NonFiniteSteps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the loss or gradient was not finite on {} optimiser steps in a row; the weights were left as they were", self.consecutive)
    }
}

impl std::error::Error for NonFiniteSteps {}

/// The gate of one run.
#[derive(Debug, Default)]
pub struct UpdateGate {
    strict: bool,
    consecutive: u32,
    warned_unscanned: bool,
}

impl UpdateGate {
    pub fn new(strict: bool) -> UpdateGate {
        UpdateGate { strict, ..Default::default() }
    }

    /// Whether the update of the step that produced `loss` may be committed:
    /// `Ok(None)` commit it, `Ok(Some(why))` skip it (nothing was changed),
    /// `Err` the run must stop. Call it after the gradients are accumulated
    /// and, in a distributed run, reduced - so every rank reaches the same
    /// answer from the same values.
    pub fn admit<M: Model>(&mut self, model: &M, loss: f32) -> std::io::Result<Option<String>> {
        let why = if !loss.is_finite() {
            Some(format!("non-finite loss {loss}"))
        } else {
            match self.gradients_finite(model) {
                Some(false) => Some("non-finite gradient".to_string()),
                Some(true) | None => None,
            }
        };
        let Some(why) = why else {
            self.consecutive = 0;
            return Ok(None);
        };
        self.consecutive += 1;
        if self.strict || self.consecutive >= MAX_CONSECUTIVE_NONFINITE {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, NonFiniteSteps { consecutive: self.consecutive }));
        }
        Ok(Some(format!("{why} ({} in a row)", self.consecutive)))
    }

    /// After a committed update: the gradient norm the clip pass measured must be finite. A model whose
    /// gradients are too large to scan before the update ([`MAX_SCANNED_GRAD_FLOATS`]) can only be checked
    /// here, when the update has already happened, so a failure is fatal and says the weights are suspect.
    pub fn verify_committed(&self, grad_norm: Option<f32>, step: u32) -> std::io::Result<()> {
        match grad_norm {
            Some(norm) if !norm.is_finite() => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("step {step}: the gradient norm was {norm} after the update was applied; the weights and optimiser state can no longer be trusted"),
            )),
            _ => Ok(()),
        }
    }

    /// `None`: the model does not expose its gradients, or they are too large to scan every step.
    fn gradients_finite<M: Model>(&mut self, model: &M) -> Option<bool> {
        let floats: Option<usize> = model.optimizer_parts().map(|p| p.store.opt_params().iter().map(|(_, n)| *n).sum());
        if floats.is_some_and(|n| n > MAX_SCANNED_GRAD_FLOATS) {
            if !std::mem::replace(&mut self.warned_unscanned, true) {
                eprintln!(
                    "update gate: {} trainable floats exceed the {MAX_SCANNED_GRAD_FLOATS}-float host scan; gradients are checked by the clip pass's norm after each update instead",
                    floats.unwrap_or_default()
                );
            }
            return None;
        }
        model.gradients_finite()
    }
}
