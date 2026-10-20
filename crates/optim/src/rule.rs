// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in optimiser design for large-model
// fine-tuning then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Element-wise update rules over decoded f32 state. This is the CPU
//! reference definition every fused device kernel (WGSL, CUDA) is gated
//! against. A rule says how many state slots it needs and of which sign
//! domain; it knows nothing about how those slots are stored.
//!
//! The rule is selected explicitly by the caller and never substituted.

use crate::codec::SlotKind;
use crate::Adam;

/// Per-step scalars shared by every element of a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepScalars {
    /// 1-based step index.
    pub t: u32,
    pub lr: f32,
    /// Multiplier already folding the accumulation average and the clip.
    pub grad_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// Decoupled-weight-decay Adam: slot 0 = first moment, slot 1 = second.
    AdamW { adam: Adam, wd: f32 },
    /// Lion (sign of an interpolated momentum): slot 0 = momentum.
    Lion { beta1: f32, beta2: f32, wd: f32 },
    /// SGD with momentum (coupled L2 weight decay, as torch): slot 0 = buffer.
    SgdMomentum { momentum: f32, wd: f32, nesterov: bool },
}

impl Rule {
    /// Sign domain of each state slot, in slot order.
    pub fn slots(&self) -> &'static [SlotKind] {
        match self {
            Rule::AdamW { .. } => &[SlotKind::Signed, SlotKind::NonNegative],
            Rule::Lion { .. } | Rule::SgdMomentum { .. } => &[SlotKind::Signed],
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Rule::AdamW { .. } => "adamw",
            Rule::Lion { .. } => "lion",
            Rule::SgdMomentum { .. } => "sgd_momentum",
        }
    }

    /// Update one element. `g` is the raw gradient (the rule applies
    /// `scalars.grad_scale`); `st` holds the decoded state slots.
    #[inline]
    pub fn update(&self, s: StepScalars, w: &mut f32, g: f32, st: &mut [f32; 2]) {
        let g = g * s.grad_scale;
        match *self {
            Rule::AdamW { adam, wd } => {
                let bc = adam.bias_corrections(s.t);
                let [m, v] = st;
                adam.update(bc, s.lr, wd, g, w, m, v);
            }
            Rule::Lion { beta1, beta2, wd } => {
                let m = &mut st[0];
                let c = beta1 * *m + (1.0 - beta1) * g;
                let u = if c > 0.0 {
                    1.0
                } else if c < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                *w -= s.lr * (u + wd * *w);
                *m = beta2 * *m + (1.0 - beta2) * g;
            }
            Rule::SgdMomentum { momentum, wd, nesterov } => {
                let g = g + wd * *w;
                let m = &mut st[0];
                *m = if s.t == 1 { g } else { momentum * *m + g };
                let d = if nesterov { g + momentum * *m } else { *m };
                *w -= s.lr * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalars(t: u32) -> StepScalars {
        StepScalars { t, lr: 0.1, grad_scale: 1.0 }
    }

    /// Spec: the AdamW rule is exactly the existing `Adam::update` the GPU
    /// kernel mirrors (one definition), including the grad scale.
    #[test]
    fn adamw_rule_is_the_adam_update() {
        let adam = Adam::default();
        let rule = Rule::AdamW { adam, wd: 0.01 };
        let (mut w, mut st) = (0.5f32, [0.1f32, 0.2]);
        rule.update(StepScalars { t: 3, lr: 0.01, grad_scale: 2.0 }, &mut w, 0.3, &mut st);
        let (mut w2, mut m2, mut v2) = (0.5f32, 0.1f32, 0.2f32);
        adam.update(adam.bias_corrections(3), 0.01, 0.01, 0.6, &mut w2, &mut m2, &mut v2);
        assert_eq!((w, st[0], st[1]), (w2, m2, v2));
    }

    /// Spec: Lion steps by lr in the sign of the interpolated momentum and
    /// then updates momentum with beta2; SGD momentum matches the torch
    /// recurrence (first step seeds the buffer with the gradient).
    #[test]
    fn lion_and_sgd_follow_their_definitions() {
        let lion = Rule::Lion { beta1: 0.9, beta2: 0.99, wd: 0.0 };
        let (mut w, mut st) = (1.0f32, [0.0f32, 0.0]);
        lion.update(scalars(1), &mut w, 0.5, &mut st);
        assert!((w - 0.9).abs() < 1e-6 && (st[0] - 0.005).abs() < 1e-7);

        let sgd = Rule::SgdMomentum { momentum: 0.5, wd: 0.0, nesterov: false };
        let (mut w, mut st) = (1.0f32, [0.0f32, 0.0]);
        sgd.update(scalars(1), &mut w, 1.0, &mut st);
        sgd.update(scalars(2), &mut w, 1.0, &mut st);
        assert!((st[0] - 1.5).abs() < 1e-6 && (w - (1.0 - 0.1 - 0.15)).abs() < 1e-6);
    }
}
