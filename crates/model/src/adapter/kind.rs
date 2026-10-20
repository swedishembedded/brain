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

/// The training groups of an adapter kind: static, so a device model that
/// never builds a host adapter reads the same rule a host step applies.
pub trait AdapterGroups {
    /// One group per tensor of [`AdapterDefinition::params`], in the same
    /// order. The ONE place LoRA-FA (`A` frozen) and LoRA+ (`B` at
    /// `lr_ratio x lr`) are decided - a host step and a device parameter
    /// store both read it.
    fn groups(hp: &TargetHp) -> Vec<ParamGroup>;
}

/// How an adapter trains: its groups and the update step.
pub trait AdapterOptimization: AdapterExecution + AdapterGroups {
    /// One optimizer step at 1-based Adam counter `t`.
    fn step(&mut self, g: &Self::Grads, lr: f32, t: u64);

    /// The fused one-liner every per-linear walk calls - the exact body of
    /// [`crate::lora::proj_step`].
    fn proj_step(&mut self, dw: &[f32], lr: f32, t: u64) {
        let g = self.project(dw);
        self.step(&g, lr, t);
    }
}

/// Which adapter method a device build trains. LoRA is a low-rank delta; IA3
/// scales a linear's output channels; LoKr adds a Kronecker-product delta that
/// is never formed densely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceKind {
    #[default]
    Lora,
    Ia3,
    Lokr,
}

impl DeviceKind {
    pub const ALL: [DeviceKind; 3] = [DeviceKind::Lora, DeviceKind::Ia3, DeviceKind::Lokr];

    /// The name an adapter file's card records.
    pub fn name(self) -> &'static str {
        match self {
            DeviceKind::Lora => "lora",
            DeviceKind::Ia3 => "ia3",
            DeviceKind::Lokr => "lokr",
        }
    }

    pub fn from_name(name: &str) -> Option<DeviceKind> {
        DeviceKind::ALL.into_iter().find(|k| k.name() == name)
    }

    pub fn params(self, hp: &TargetHp, spec: &TargetSpec) -> Vec<ParamSpec> {
        match self {
            DeviceKind::Lora => crate::lora::LoraPair::params(hp, spec),
            DeviceKind::Ia3 => Ia3::params(hp, spec),
            DeviceKind::Lokr => Lokr::params(hp, spec),
        }
    }

    pub fn groups(self, hp: &TargetHp) -> Vec<ParamGroup> {
        match self {
            DeviceKind::Lora => <crate::lora::LoraPair as AdapterGroups>::groups(hp),
            DeviceKind::Ia3 => Ia3::groups(hp),
            DeviceKind::Lokr => Lokr::groups(hp),
        }
    }

    pub fn suffixes(self) -> &'static [&'static str] {
        match self {
            DeviceKind::Lora => crate::lora::LoraPair::suffixes(),
            DeviceKind::Ia3 => Ia3::suffixes(),
            DeviceKind::Lokr => Lokr::suffixes(),
        }
    }

    pub fn shape_from_len(self, suffix: &str, rank: usize, len: usize) -> Option<Vec<usize>> {
        match self {
            DeviceKind::Lora => crate::lora::LoraPair::shape_from_len(suffix, rank, len),
            DeviceKind::Ia3 => Ia3::shape_from_len(suffix, rank, len),
            DeviceKind::Lokr => Lokr::shape_from_len(suffix, rank, len),
        }
    }
}

/// Suffix of IA3's output-channel scale: the stored value is the DELTA from 1
/// (`y' = y * (1 + d)`), so a fresh adapter is exactly zero and invisible.
pub const IA3_D: &str = ".ia3_d";
/// Suffixes of LoKr's two Kronecker factors, `W1 [a, b]` and `W2 [c, d]`.
pub const LOKR_W1: &str = ".lokr_w1";
pub const LOKR_W2: &str = ".lokr_w2";

/// How a linear of `out x inn` is factored for LoKr: `out = a * c` and
/// `inn = b * d`, each pair split as close to its square root as the
/// dimension's divisors allow, the smaller factor first. Returns `(a, c, b, d)`.
pub fn lokr_factors(out: usize, inn: usize) -> (usize, usize, usize, usize) {
    fn split(n: usize) -> (usize, usize) {
        let small = (1..=n).take_while(|f| f * f <= n).filter(|f| n.is_multiple_of(*f)).last().unwrap_or(1);
        (small, n / small)
    }
    let (a, c) = split(out);
    let (b, d) = split(inn);
    (a, c, b, d)
}

/// IA3: learned per-output-channel scales of a linear's output (Liu et al.
/// 2022, arXiv:2205.05638), stored as the delta from 1.
pub struct Ia3;

impl AdapterDefinition for Ia3 {
    fn kind_name() -> &'static str {
        "ia3"
    }
    fn params(_hp: &TargetHp, spec: &TargetSpec) -> Vec<ParamSpec> {
        vec![ParamSpec { suffix: IA3_D, shape: vec![spec.out], init: ParamInit::Zeros }]
    }
    fn suffixes() -> &'static [&'static str] {
        &[IA3_D]
    }
    fn shape_from_len(suffix: &str, _rank: usize, len: usize) -> Option<Vec<usize>> {
        (suffix == IA3_D && len > 0).then(|| vec![len])
    }
}

impl AdapterGroups for Ia3 {
    fn groups(_hp: &TargetHp) -> Vec<ParamGroup> {
        vec![ParamGroup { suffix: IA3_D, trainable: true, lr_mult: 1.0 }]
    }
}

/// LoKr: the delta is `scale * (W1 kron W2)` (Yeh et al. 2023, arXiv:2309.14859,
/// the LyCORIS Kronecker adapter), applied as two small GEMMs and two axis
/// swaps, so the Kronecker product is never formed. `W1` starts random and
/// `W2` at zero, so a fresh adapter is invisible and both factors get gradient
/// once `W2` moves.
pub struct Lokr;

impl AdapterDefinition for Lokr {
    fn kind_name() -> &'static str {
        "lokr"
    }
    fn params(_hp: &TargetHp, spec: &TargetSpec) -> Vec<ParamSpec> {
        let (a, c, b, d) = lokr_factors(spec.out, spec.inn);
        vec![
            ParamSpec { suffix: LOKR_W1, shape: vec![a, b], init: ParamInit::Drawn },
            ParamSpec { suffix: LOKR_W2, shape: vec![c, d], init: ParamInit::Zeros },
        ]
    }
    fn suffixes() -> &'static [&'static str] {
        &[LOKR_W1, LOKR_W2]
    }
    /// A flat factor has no recoverable shape without the linear's dimensions.
    fn shape_from_len(_suffix: &str, _rank: usize, _len: usize) -> Option<Vec<usize>> {
        None
    }
}

impl AdapterGroups for Lokr {
    fn groups(_hp: &TargetHp) -> Vec<ParamGroup> {
        vec![ParamGroup { suffix: LOKR_W1, trainable: true, lr_mult: 1.0 }, ParamGroup { suffix: LOKR_W2, trainable: true, lr_mult: 1.0 }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lokr_factors_split_each_dimension_near_its_square_root() {
        assert_eq!(lokr_factors(12, 20), (3, 4, 4, 5));
        assert_eq!(lokr_factors(7, 1), (1, 7, 1, 1), "a prime dimension degenerates to a vector factor");
        let (a, c, b, d) = lokr_factors(4096, 11008);
        assert_eq!((a * c, b * d), (4096, 11008));
    }

    #[test]
    fn kinds_own_disjoint_suffixes_and_round_trip_their_names() {
        let mut all: Vec<&str> = DeviceKind::ALL.iter().flat_map(|k| k.suffixes().iter().copied()).collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n);
        for k in DeviceKind::ALL {
            assert_eq!(DeviceKind::from_name(k.name()), Some(k));
        }
    }

    #[test]
    fn a_fresh_ia3_or_lokr_adapter_is_invisible() {
        let hp = TargetHp::new(1, 1.0);
        let spec = TargetSpec::whole(12, 20);
        assert!(Ia3::params(&hp, &spec).iter().all(|p| p.init == ParamInit::Zeros));
        let lokr = Lokr::params(&hp, &spec);
        assert_eq!((lokr[0].shape.clone(), lokr[1].shape.clone(), lokr[1].init), (vec![3, 4], vec![4, 5], ParamInit::Zeros));
    }
}
