// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified training kernels for its clients. If your team needs expertise in
// gradient checking device backward passes, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The directional finite-difference check of a device backward ([`Directional::check`]): the analytic gradient
//! projected on a random ±1 direction against the central difference of the loss along it. One projection averages
//! per-element round-off into one well-conditioned number instead of probing every element.
//!
//! Two things make a naive version flaky, and both are handled here once:
//!
//! * the loss is taken in f64 (the caller's `loss_at` returns it), since an f32 sum over the outputs carries
//!   round-off of the order of the difference that is divided by `2 * eps`;
//! * a network with ReLUs has a piecewise smooth loss, and a step that crosses a kink measures no gradient at all.
//!   A direction whose two one-sided differences disagree by more than the tolerance has a kink inside its step and
//!   is drawn again, so the gradient is checked where the loss is differentiable. A wrong backward still fails:
//!   along a smooth direction its projection differs from the central difference.

/// Step and tolerance of a directional check: it passes when `|analytic - fd| <= abs_tol + rel_tol * scale`, `scale`
/// being the larger magnitude of the two (at least 1e-3).
#[derive(Clone, Copy, Debug)]
pub struct Directional {
    pub eps: f32,
    pub abs_tol: f32,
    pub rel_tol: f32,
}

/// Directions drawn before a check gives up on finding one whose step crosses no kink.
const DRAWS: usize = 8;

impl Directional {
    fn tolerance(&self, a: f32, b: f32) -> f32 {
        self.abs_tol + self.rel_tol * a.abs().max(b.abs()).max(1e-3)
    }

    /// Check `grad` (the analytic gradient of the loss wrt the weights `w0`) along directions from `draw` (±1 per
    /// element), `loss_at(w)` running the forward at weights `w` and returning the loss in f64. The weights last
    /// handed to `loss_at` are not `w0`: the caller restores them. Returns the compared `(analytic, fd)` pair.
    pub fn check(&self, grad: &[f32], w0: &[f32], mut draw: impl FnMut() -> Vec<f32>, mut loss_at: impl FnMut(&[f32]) -> f64) -> Result<(f32, f32), String> {
        let mut at = |step: f32, dir: &[f32]| loss_at(&w0.iter().zip(dir).map(|(w, d)| w + step * d).collect::<Vec<f32>>());
        let eps = self.eps as f64;
        for _ in 0..DRAWS {
            let dir = draw();
            let (l0, lp, lm) = (at(0.0, &dir), at(self.eps, &dir), at(-self.eps, &dir));
            let (ahead, behind) = (((lp - l0) / eps) as f32, ((l0 - lm) / eps) as f32);
            if (ahead - behind).abs() > self.tolerance(ahead, behind) {
                continue;
            }
            let analytic: f32 = grad.iter().zip(&dir).map(|(g, d)| g * d).sum();
            let fd = ((lp - lm) / (2.0 * eps)) as f32;
            let abs = (analytic - fd).abs();
            return if abs <= self.tolerance(analytic, fd) {
                Ok((analytic, fd))
            } else {
                Err(format!("analytic {analytic}, fd {fd} (abs {abs}, rel {})", abs / analytic.abs().max(fd.abs()).max(1e-3)))
            };
        }
        Err(format!("every one of {DRAWS} directions crosses a kink within a step of {}", self.eps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directions() -> impl FnMut() -> Vec<f32> {
        let mut s = 0x9E37_79B9u32;
        move || {
            (0..4)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    if s & 1 == 0 { 1.0 } else { -1.0 }
                })
                .collect()
        }
    }

    /// Spec: a correct gradient of a smooth loss passes, a 5% wrong one fails, and a loss with a kink inside every
    /// step is reported rather than compared.
    #[test]
    fn checks_smooth_directions_and_reports_kinks() {
        let check = Directional { eps: 1e-3, abs_tol: 1e-4, rel_tol: 1e-2 };
        let w0 = [0.5f32, -0.25, 1.0, 2.0];
        let loss = |w: &[f32]| w.iter().map(|&x| (x as f64).powi(3)).sum::<f64>();
        let grad: Vec<f32> = w0.iter().map(|x| 3.0 * x * x).collect();
        assert!(check.check(&grad, &w0, directions(), loss).is_ok());
        let wrong: Vec<f32> = grad.iter().map(|g| g * 1.05).collect();
        assert!(check.check(&wrong, &w0, directions(), loss).unwrap_err().contains("analytic"));
        let kinked = |w: &[f32]| w.iter().map(|&x| (x as f64 - 0.5).abs()).sum::<f64>();
        assert!(check.check(&grad, &w0, directions(), kinked).unwrap_err().contains("kink"));
    }
}
