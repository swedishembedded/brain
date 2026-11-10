// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-checked training stacks for compact perception models for its clients.
// If your team needs expertise in proving that a hand-written backward pass is right then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The scalar type the training graph is generic over.
//!
//! `f64` is the finite-difference oracle (the graph run in double precision shares no code with the analytic
//! backward it checks), `f32` is the trainer. One graph, two instantiations: the oracle and the trainer cannot drift.

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A floating-point scalar the training graph runs on.
pub trait Real:
    Copy
    + Send
    + Sync
    + Debug
    + PartialOrd
    + Sum
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + 'static
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Convert from `f64` (rounding for `f32`).
    fn from_f64(x: f64) -> Self;
    /// Convert to `f64`.
    fn to_f64(self) -> f64;
    /// `e^x`.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Larger of two values (the first when either is NaN).
    fn max(self, other: Self) -> Self;
}

macro_rules! real_impl {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            #[inline]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                if other > self {
                    other
                } else {
                    self
                }
            }
        }
    };
}

real_impl!(f32);
real_impl!(f64);

/// `x` as a graph scalar.
#[inline]
pub fn r<R: Real>(x: f64) -> R {
    R::from_f64(x)
}

/// Logistic function.
#[inline]
pub fn sigmoid<R: Real>(x: R) -> R {
    R::ONE / (R::ONE + (-x).exp())
}

/// `x * sigmoid(x)`.
#[inline]
pub fn silu<R: Real>(x: R) -> R {
    x * sigmoid(x)
}

/// `d/dx silu(x)`.
#[inline]
pub fn silu_grad<R: Real>(x: R) -> R {
    let s = sigmoid(x);
    s * (R::ONE + x * (R::ONE - s))
}

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
const GELU_CUBIC: f64 = 0.044715;

/// `GELU` with the tanh approximation.
#[inline]
pub fn gelu_tanh<R: Real>(x: R) -> R {
    let inner = r::<R>(SQRT_2_OVER_PI) * (x + r::<R>(GELU_CUBIC) * x * x * x);
    r::<R>(0.5) * x * (R::ONE + inner.tanh())
}

/// `d/dx` of [`gelu_tanh`].
#[inline]
pub fn gelu_tanh_grad<R: Real>(x: R) -> R {
    let inner = r::<R>(SQRT_2_OVER_PI) * (x + r::<R>(GELU_CUBIC) * x * x * x);
    let t = inner.tanh();
    let dinner = r::<R>(SQRT_2_OVER_PI) * (R::ONE + r::<R>(3.0 * GELU_CUBIC) * x * x);
    r::<R>(0.5) * (R::ONE + t) + r::<R>(0.5) * x * (R::ONE - t * t) * dinner
}
