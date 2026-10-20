// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for adding a new parameter-efficient
// fine-tuning method to a training stack once, instead of once per model, for
// its clients. If your team needs expertise in PEFT infrastructure then you can
// procure our services by sending an email to info@swedishembedded.com.

//! One adapter kind, in three separable roles.
//!
//! * [`AdapterDefinition`] - WHAT a kind is: its name, which tensors it owns
//!   per target, their shapes and how they start. Stateless, so a device
//!   trainer that never builds a host adapter reads the same schema the host
//!   serializer does, and a loader can decide which tensors belong to an
//!   adapter without knowing any suffix spelling itself.
//! * [`AdapterExecution`] - what the adapter DOES to a linear: the delta it
//!   adds to a weight, the operator form it offers to a runtime that keeps the
//!   base untouched (any base storage), and the gradients it derives from a
//!   dense `dL/dW`.
//! * [`AdapterOptimization`] - how it TRAINS: which of its tensors are
//!   trainable, at which learning-rate multiplier (the parameter groups), and
//!   the update step.
//!
//! [`super::AdapterKind`] is their conjunction plus the instance lifecycle
//! (construct, serialize, load). Adding a kind means implementing the three
//! roles and listing its suffixes in [`super::device::KINDS`]; nothing else
//! names a tensor suffix.

use super::{LowRank, TargetHp, TargetSpec};

/// Canonical suffix of LoRA's down projection `A [rank, in]`.
pub const LORA_A: &str = ".lora_a";
/// Canonical suffix of LoRA's up projection `B [out, rank]`.
pub const LORA_B: &str = ".lora_b";

/// How a tensor of an adapter starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamInit {
    /// Drawn from the caller's distribution (LoRA's `A`), so every existing
    /// seed keeps reproducing the adapter it always did.
    Drawn,
    /// All zeros (LoRA's `B`), so a fresh adapter is invisible.
    Zeros,
}

/// One tensor an adapter kind owns for one target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamSpec {
    /// Canonical suffix relative to the base weight name.
    pub suffix: &'static str,
    /// Row-major shape.
    pub shape: Vec<usize>,
    pub init: ParamInit,
}

impl ParamSpec {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The schema of an adapter kind.
pub trait AdapterDefinition {
    fn kind_name() -> &'static str;

    /// The tensors this kind adds for ONE target. Both the device parameter
    /// list and the host serializer read this, so they cannot disagree about
    /// what a kind needs.
    fn params(hp: &TargetHp, spec: &TargetSpec) -> Vec<ParamSpec>;

    /// Every suffix [`Self::params`] can return, independent of any target:
    /// what lets a name be recognised as this kind's without a target in hand.
    fn suffixes() -> &'static [&'static str];

    /// The shape the tensor `suffix` has when it holds `len` elements at
    /// `rank`: how a file that stores flat data (the device family wrote
    /// `[len]` for a long time) gets its real shape back. `None` for a suffix
    /// this kind does not own or a `len` that is not a whole matrix.
    fn shape_from_len(suffix: &str, rank: usize, len: usize) -> Option<Vec<usize>>;
}

/// How an adapter acts on a linear.
pub trait AdapterExecution {
    type Grads: Sync;

    /// `dst[(row0+o)*row_stride + col0 + i] += strength * scale * delta[o,i]`.
    /// `strength` is the inference dial, multiplied ON TOP of the hp's scale.
    fn delta_into(&self, strength: f32, dst: &mut [f32]);

    /// Dense `dL/dW_eff` -> this kind's grads. Host dense-projection path
    /// only - device trainers compute their grads from low-rank
    /// intermediates and never form a dense `dW`.
    fn project(&self, dw: &[f32]) -> Self::Grads;

    /// This kind's correction as an operator on the linear's **input
    /// activation**, `Δy = scale·B·(A·x)`, when it has one - the form that
    /// can be delivered beside an unmodified base weight of ANY storage
    /// instead of being folded into it. `None` says the kind has no such
    /// form, which [`super::AdapterSet::delivery`] turns into a refusal
    /// rather than a fold onto a quantized grid.
    fn low_rank(&self) -> Option<LowRank<'_>>;
}

/// One group of an adapter's tensors that trains under one rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamGroup {
    pub suffix: &'static str,
    /// `false`: the tensor stays at its init (no gradient, no moments).
    pub trainable: bool,
    /// Multiplier on the run's learning rate for this tensor.
    pub lr_mult: f32,
}

/// How an adapter trains.
pub trait AdapterOptimization: AdapterExecution {
    /// One group per tensor of [`AdapterDefinition::params`], in the same
    /// order. The ONE place LoRA-FA (`A` frozen) and LoRA+ (`B` at
    /// `lr_ratio x lr`) are decided - a host step and a device parameter
    /// store both read it.
    fn groups(hp: &TargetHp) -> Vec<ParamGroup>
    where
        Self: Sized;

    /// One optimizer step at 1-based Adam counter `t`.
    fn step(&mut self, g: &Self::Grads, lr: f32, t: u64);

    /// The fused one-liner every per-linear walk calls - the exact body of
    /// [`crate::lora::proj_step`].
    fn proj_step(&mut self, dw: &[f32], lr: f32, t: u64) {
        let g = self.project(dw);
        self.step(&g, lr, t);
    }
}
