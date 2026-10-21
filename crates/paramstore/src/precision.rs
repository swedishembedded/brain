// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in training large models on limited
// hardware then you can procure our services by sending an email to
// info@swedishembedded.com.

//! **Per-parameter training precision**: how each trainable tensor stores its
//! value, its gradient, its optimiser state and its optional master copy.
//! The four are independent - a bf16 weight can train with an fp32 master and
//! 8-bit moments, an fp32 norm with fp32 moments, in one model.
//!
//! This module is pure description and byte accounting. It never decides
//! silently: a [`PrecisionPolicy`] resolves each parameter by explicit rules
//! and returns a [`Resolution`] naming the rule that fired, so the caller can
//! report every choice. The state codecs and update rules that consume these
//! descriptions live in the `optim` crate.
//!
//! # Byte accounting
//!
//! [`ParamPrecision::bytes`] charges PHYSICAL bytes of every component,
//! including the per-block scales of 8-bit formats, so a budget built on it
//! cannot under-report the way an "elements x 4" model does.

/// Storage dtype of a trainable parameter's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDtype {
    F32,
    Bf16,
}

/// Dtype the backward pass accumulates this parameter's gradient in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradDtype {
    F32,
    Bf16,
}

/// Physical format of one optimiser-state tensor (a moment, a momentum or a
/// master copy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateFormat {
    F32,
    Bf16,
    /// 8-bit codes with one f32 scale per `block` consecutive elements. The
    /// code map depends on the slot's sign (see `optim::codec`).
    Int8Block { block: u32 },
    /// 8-bit codes on a logarithmic scale with one f32 scale (the block maximum) per `block` consecutive
    /// elements: a constant relative step over ~16 octaves below the maximum, where [`StateFormat::Int8Block`]
    /// spends its levels near the maximum. For heavy-tailed second moments.
    Log8Block { block: u32 },
}

/// Whether, and in which format, the optimiser keeps a higher-precision copy of
/// the weight that updates accumulate into. `None` updates the stored value
/// directly (stochastic rounding is then what keeps small updates alive for a
/// low-precision value).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterPolicy {
    None,
    Bf16,
    F32,
}

impl StateFormat {
    /// Block size of the scale hierarchy, or 1 for an unblocked format.
    pub fn block(self) -> usize {
        match self {
            StateFormat::Int8Block { block } | StateFormat::Log8Block { block } => block as usize,
            _ => 1,
        }
    }

    /// Physical bytes of `numel` elements in this format, scales included.
    pub fn bytes(self, numel: usize) -> u64 {
        match self {
            StateFormat::F32 => 4 * numel as u64,
            StateFormat::Bf16 => 2 * numel as u64,
            StateFormat::Int8Block { block } | StateFormat::Log8Block { block } => numel as u64 + 4 * numel.div_ceil(block as usize) as u64,
        }
    }
}

impl ValueDtype {
    pub fn bytes(self, numel: usize) -> u64 {
        match self {
            ValueDtype::F32 => 4 * numel as u64,
            ValueDtype::Bf16 => 2 * numel as u64,
        }
    }
}

impl GradDtype {
    pub fn bytes(self, numel: usize) -> u64 {
        match self {
            GradDtype::F32 => 4 * numel as u64,
            GradDtype::Bf16 => 2 * numel as u64,
        }
    }
}

impl MasterPolicy {
    /// The state format of the master copy, if there is one.
    pub fn format(self) -> Option<StateFormat> {
        match self {
            MasterPolicy::None => None,
            MasterPolicy::Bf16 => Some(StateFormat::Bf16),
            MasterPolicy::F32 => Some(StateFormat::F32),
        }
    }
}

/// Physical bytes of one parameter's training state, by component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParamBytes {
    pub value: u64,
    pub grad: u64,
    pub moments: u64,
    pub master: u64,
}

impl ParamBytes {
    pub fn total(&self) -> u64 {
        self.value + self.grad + self.moments + self.master
    }
}

impl std::ops::AddAssign for ParamBytes {
    fn add_assign(&mut self, o: ParamBytes) {
        self.value += o.value;
        self.grad += o.grad;
        self.moments += o.moments;
        self.master += o.master;
    }
}

/// The four independent precision choices of one trainable parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamPrecision {
    pub value: ValueDtype,
    pub grad: GradDtype,
    /// Format of every moment/momentum slot of the optimiser rule.
    pub state: StateFormat,
    pub master: MasterPolicy,
    /// Round to nearest-even (false) or stochastically (true) whenever a
    /// low-precision value, master or state is re-encoded.
    pub stochastic_rounding: bool,
}

impl ParamPrecision {
    /// The historical full-precision layout: everything fp32, no master.
    pub const FP32: ParamPrecision = ParamPrecision {
        value: ValueDtype::F32,
        grad: GradDtype::F32,
        state: StateFormat::F32,
        master: MasterPolicy::None,
        stochastic_rounding: false,
    };

    /// Physical bytes for `numel` elements with `moment_slots` state slots
    /// (2 for AdamW, 1 for Lion / SGD-momentum).
    pub fn bytes(&self, numel: usize, moment_slots: usize) -> ParamBytes {
        ParamBytes {
            value: self.value.bytes(numel),
            grad: self.grad.bytes(numel),
            moments: moment_slots as u64 * self.state.bytes(numel),
            master: self.master.format().map_or(0, |f| f.bytes(numel)),
        }
    }
}

/// What a rule matches: the parameter's name and element count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Matcher {
    Any,
    NameContains(String),
    NameSuffix(String),
    /// Parameters with at least this many elements (e.g. big matrices).
    MinNumel(usize),
    /// Every sub-matcher must match.
    All(Vec<Matcher>),
}

impl Matcher {
    pub fn matches(&self, name: &str, numel: usize) -> bool {
        match self {
            Matcher::Any => true,
            Matcher::NameContains(s) => name.contains(s.as_str()),
            Matcher::NameSuffix(s) => name.ends_with(s.as_str()),
            Matcher::MinNumel(n) => numel >= *n,
            Matcher::All(ms) => ms.iter().all(|m| m.matches(name, numel)),
        }
    }
}

/// Which rule chose a parameter's precision; `None` = the policy default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolution {
    pub precision: ParamPrecision,
    pub rule: Option<usize>,
}

/// Ordered per-group precision rules; the first matching rule wins, otherwise
/// the default. Rules are explicit - nothing here picks a format from hardware
/// or memory pressure on its own.
#[derive(Clone, Debug)]
pub struct PrecisionPolicy {
    pub default: ParamPrecision,
    pub rules: Vec<(Matcher, ParamPrecision)>,
}

impl PrecisionPolicy {
    pub fn uniform(p: ParamPrecision) -> PrecisionPolicy {
        PrecisionPolicy { default: p, rules: Vec::new() }
    }

    pub fn resolve(&self, name: &str, numel: usize) -> Resolution {
        match self.rules.iter().position(|(m, _)| m.matches(name, numel)) {
            Some(i) => Resolution { precision: self.rules[i].1, rule: Some(i) },
            None => Resolution { precision: self.default, rule: None },
        }
    }

    /// Human-readable one-line-per-parameter report of every choice and its
    /// physical bytes, plus a total. `moment_slots` is the optimiser rule's.
    pub fn report(&self, params: &[(String, usize)], moment_slots: usize) -> (String, ParamBytes) {
        let mut total = ParamBytes::default();
        let mut out = String::new();
        for (name, numel) in params {
            let r = self.resolve(name, *numel);
            let b = r.precision.bytes(*numel, moment_slots);
            total += b;
            let rule = r.rule.map_or("default".to_string(), |i| format!("rule {i}"));
            out.push_str(&format!(
                "{name}: {numel} elems, {rule}, value {:?}, grad {:?}, state {:?}, master {:?}, sr {} -> {} bytes\n",
                r.precision.value,
                r.precision.grad,
                r.precision.state,
                r.precision.master,
                r.precision.stochastic_rounding,
                b.total()
            ));
        }
        (out, total)
    }
}

/// The optimiser-state byte ladder of a parameter manifest: the physical total
/// of `moment_slots` state slots per parameter under each candidate format,
/// widest first. This is what a memory planner offers as the variants of its
/// optimiser-state category; every entry counts scales, so a narrower format
/// is never over-credited.
pub fn optimizer_state_ladder(params: &[(String, usize)], moment_slots: usize, int8_block: u32) -> Vec<(StateFormat, u64)> {
    [StateFormat::F32, StateFormat::Bf16, StateFormat::Int8Block { block: int8_block }]
        .into_iter()
        .map(|f| (f, params.iter().map(|(_, n)| moment_slots as u64 * f.bytes(*n)).sum()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: bytes are physical - an 8-bit block format pays its scales, a
    /// master is charged only when present, and the components are independent.
    #[test]
    fn bytes_are_physical_and_components_independent() {
        let p = ParamPrecision {
            value: ValueDtype::Bf16,
            grad: GradDtype::Bf16,
            state: StateFormat::Int8Block { block: 256 },
            master: MasterPolicy::F32,
            stochastic_rounding: true,
        };
        let b = p.bytes(1000, 2);
        assert_eq!(b.value, 2000);
        assert_eq!(b.grad, 2000);
        // 1000 codes + ceil(1000/256)=4 scales of 4 bytes, per slot.
        assert_eq!(b.moments, 2 * (1000 + 16));
        assert_eq!(b.master, 4000);
        assert_eq!(ParamPrecision::FP32.bytes(1000, 2).total(), 4000 * 4);
        assert_eq!(ParamPrecision { master: MasterPolicy::None, ..p }.bytes(1000, 1).master, 0);
    }

    /// Spec: the ladder is widest first, strictly decreasing, and pays scales.
    #[test]
    fn the_state_ladder_is_physical_and_decreasing() {
        let params = vec![("a".to_string(), 1000usize), ("b".to_string(), 24)];
        let l = optimizer_state_ladder(&params, 2, 256);
        assert_eq!(l[0], (StateFormat::F32, 2 * 4 * 1024));
        assert_eq!(l[1], (StateFormat::Bf16, 2 * 2 * 1024));
        assert_eq!(l[2], (StateFormat::Int8Block { block: 256 }, 2 * ((1000 + 16) + (24 + 4))));
        assert!(l[0].1 > l[1].1 && l[1].1 > l[2].1);
    }

    /// Spec: a per-group policy gives different precision to different
    /// tensors in one model, first matching rule wins, and the report names
    /// the rule that fired.
    #[test]
    fn policy_resolves_per_group_and_reports_the_rule() {
        let int8 = ParamPrecision { state: StateFormat::Int8Block { block: 64 }, ..ParamPrecision::FP32 };
        let policy = PrecisionPolicy {
            default: ParamPrecision::FP32,
            rules: vec![
                (Matcher::NameContains("norm".into()), ParamPrecision::FP32),
                (Matcher::MinNumel(1024), int8),
            ],
        };
        assert_eq!(policy.resolve("blk.0.norm.weight", 4096).rule, Some(0));
        let big = policy.resolve("blk.0.attn.weight", 4096);
        assert_eq!((big.rule, big.precision.state), (Some(1), StateFormat::Int8Block { block: 64 }));
        assert_eq!(policy.resolve("bias", 8).rule, None);
        let params = vec![("blk.0.attn.weight".to_string(), 4096), ("bias".to_string(), 8)];
        let (text, total) = policy.report(&params, 2);
        assert!(text.contains("rule 1") && text.contains("default"), "{text}");
        let mut expect = int8.bytes(4096, 2);
        expect += ParamPrecision::FP32.bytes(8, 2);
        assert_eq!(total, expect);
    }
}
