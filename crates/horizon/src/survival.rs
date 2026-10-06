// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Closed-form survival and cumulative incidence from one subject's hazard
//! table - the model's answer at ANY horizon inside the knots, without
//! rolling a trajectory forward.
//!
//! Inside piece `p` every hazard is constant, so with `Lambda_p` the summed
//! hazard of the codes competing with code `k` (the absorbing codes, plus `k`
//! itself when `k` is not absorbing):
//!
//! ```text
//! S_k(t)   = exp(-sum_p Lambda_p * overlap_p(t))
//! CIF_k(t) = sum_p (lambda_kp / Lambda_p) * S_k(tau_{p-1}) * (1 - exp(-Lambda_p * overlap_p(t)))
//! ```
//!
//! For an absorbing code (a cause of death) the competitors are the other
//! causes, and the CIFs of all absorbing codes sum to `1 - S(t)`, never more.
//! A first-occurrence code (a diagnosis) competes with death: the probability
//! of being diagnosed before `t`, alive. Past the last knot the curves are
//! held at their value there; the model says nothing about later times.

/// One subject's hazards.
#[derive(Clone, Debug)]
pub struct Curves {
    knots: Vec<f32>,
    /// `[pieces][codes]` hazards (not log).
    hazard: Vec<Vec<f64>>,
    absorbing: Vec<bool>,
}

impl Curves {
    /// From the model's log-hazards `[pieces * codes]` row-major.
    pub fn new(log_hazards: &[f32], knots: &[f32], absorbing: &[bool]) -> Curves {
        let k = absorbing.len();
        let pieces = knots.len() - 1;
        assert_eq!(
            log_hazards.len(),
            pieces * k,
            "log-hazards for {pieces} pieces and {k} codes"
        );
        let hazard = (0..pieces)
            .map(|p| {
                (0..k)
                    .map(|c| (log_hazards[p * k + c] as f64).exp())
                    .collect()
            })
            .collect();
        Curves {
            knots: knots.to_vec(),
            hazard,
            absorbing: absorbing.to_vec(),
        }
    }

    /// The curves of the LAST `group` hazard columns of `log_hazards`
    /// (`[pieces * columns]` row-major, `outcomes` outcome columns first), every
    /// one absorbing: the next-event group, in which the codes compete for being
    /// first, so their cumulative incidences sum to the probability that any
    /// event has happened.
    pub fn first_events(
        log_hazards: &[f32],
        knots: &[f32],
        outcomes: usize,
        group: usize,
    ) -> Curves {
        let columns = outcomes + group;
        let pieces = knots.len() - 1;
        assert_eq!(
            log_hazards.len(),
            pieces * columns,
            "log-hazards for {pieces} pieces and {columns} columns"
        );
        let group_lh: Vec<f32> = (0..pieces)
            .flat_map(|p| log_hazards[p * columns + outcomes..(p + 1) * columns].iter().copied())
            .collect();
        Curves::new(&group_lh, knots, &vec![true; group])
    }

    /// The curves of the FIRST `outcomes` hazard columns of `log_hazards`
    /// (`[pieces * columns]`), with `absorbing` flagging which of them end
    /// follow-up.
    pub fn outcomes(
        log_hazards: &[f32],
        knots: &[f32],
        columns: usize,
        absorbing: &[bool],
    ) -> Curves {
        let pieces = knots.len() - 1;
        assert_eq!(
            log_hazards.len(),
            pieces * columns,
            "log-hazards for {pieces} pieces and {columns} columns"
        );
        let own: Vec<f32> = (0..pieces)
            .flat_map(|p| {
                log_hazards[p * columns..p * columns + absorbing.len()]
                    .iter()
                    .copied()
            })
            .collect();
        Curves::new(&own, knots, absorbing)
    }

    /// The hazard of `code` in `piece`.
    pub fn hazard(&self, piece: usize, code: usize) -> f64 {
        self.hazard[piece][code]
    }

    fn overlap(&self, p: usize, t: f64) -> f64 {
        (t.min(self.knots[p + 1] as f64) - self.knots[p] as f64).max(0.0)
    }

    fn competing(&self, p: usize, code: usize) -> f64 {
        let mut total: f64 = (0..self.absorbing.len())
            .filter(|&c| self.absorbing[c])
            .map(|c| self.hazard[p][c])
            .sum();
        if !self.absorbing[code] {
            total += self.hazard[p][code];
        }
        total
    }

    /// Probability of no absorbing event before `t` (for health: alive at `t`).
    pub fn survival(&self, t: f64) -> f64 {
        let cum: f64 = (0..self.hazard.len())
            .map(|p| {
                (0..self.absorbing.len())
                    .filter(|&c| self.absorbing[c])
                    .map(|c| self.hazard[p][c])
                    .sum::<f64>()
                    * self.overlap(p, t)
            })
            .sum();
        (-cum).exp()
    }

    /// Probability that `code` happens before `t` (and, for a non-absorbing
    /// code, before any absorbing one).
    pub fn cif(&self, code: usize, t: f64) -> f64 {
        let mut free = 1.0; // P(no competing event yet) at the piece start
        let mut cif = 0.0;
        for p in 0..self.hazard.len() {
            let dt = self.overlap(p, t);
            if dt <= 0.0 {
                break;
            }
            let lam = self.competing(p, code);
            if lam > 0.0 {
                let leave = 1.0 - (-lam * dt).exp();
                cif += self.hazard[p][code] / lam * free * leave;
                free *= 1.0 - leave;
            }
        }
        cif
    }

    /// The time at which survival falls to `q`, or `None` if it stays above
    /// it through the last knot (a median when `q = 0.5`).
    pub fn survival_quantile(&self, q: f64) -> Option<f64> {
        let mut s = 1.0;
        for p in 0..self.hazard.len() {
            let lam: f64 = (0..self.absorbing.len())
                .filter(|&c| self.absorbing[c])
                .map(|c| self.hazard[p][c])
                .sum();
            let width = (self.knots[p + 1] - self.knots[p]) as f64;
            let end = s * (-lam * width).exp();
            if end <= q && lam > 0.0 {
                return Some(self.knots[p] as f64 + (s / q).ln() / lam);
            }
            s = end;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_hazards_give_textbook_competing_risks() {
        // Two causes of death with hazards 0.1 and 0.3, one diagnosis at 0.2.
        let knots = [0.0f32, 1.0, 3.0, 10.0];
        let lh: Vec<f32> = (0..3)
            .flat_map(|_| [0.1f32.ln(), 0.3f32.ln(), 0.2f32.ln()])
            .collect();
        let c = Curves::new(&lh, &knots, &[true, true, false]);
        for t in [0.5f64, 2.0, 7.0] {
            let s = (-0.4 * t).exp();
            assert!((c.survival(t) - s).abs() < 1e-6);
            assert!(
                (c.cif(0, t) - 0.25 * (1.0 - s)).abs() < 1e-6,
                "cause 1 gets its share of the deaths"
            );
            assert!((c.cif(1, t) - 0.75 * (1.0 - s)).abs() < 1e-6);
            assert!(
                (c.survival(t) + c.cif(0, t) + c.cif(1, t) - 1.0).abs() < 1e-9,
                "the probabilities partition"
            );
            let dx = 0.2 / 0.6 * (1.0 - (-0.6 * t).exp());
            assert!((c.cif(2, t) - dx).abs() < 1e-6, "diagnosed before death");
        }
        assert_eq!(c.cif(0, 12.0), c.cif(0, 10.0), "held past the last knot");
        let med = c.survival_quantile(0.5).unwrap();
        assert!((med - 2f64.ln() / 0.4).abs() < 1e-6);
        assert!(
            Curves::new(&[-20.0, -20.0, -20.0], &[0.0, 1.0], &[true, true, false])
                .survival_quantile(0.5)
                .is_none()
        );
    }

    #[test]
    fn piecewise_hazards_chain_across_pieces() {
        let knots = [0.0f32, 2.0, 5.0];
        let lh = [0.05f32.ln(), 0.2f32.ln()];
        let c = Curves::new(&lh, &knots, &[true]);
        let s4 = (-(0.05 * 2.0 + 0.2 * 2.0f64)).exp();
        assert!((c.survival(4.0) - s4).abs() < 1e-6);
        assert!((c.cif(0, 4.0) - (1.0 - s4)).abs() < 1e-6);
    }
}
