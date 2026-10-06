// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Numerics and record-building shared by the survival generators
//! ([`super::single`], [`super::competing`], [`super::irregular`],
//! [`super::longitudinal`]): composite Simpson integration, inversion of a
//! cumulative hazard, and the one `timeline-v1` record shape they all emit.

use crate::timeline::{AtRisk, Event, Observation, Subject};

/// The common prediction time of the survival generators' subjects.
pub const ENTRY: f64 = 60.0;
/// The calendar time at [`ENTRY`].
pub const CALENDAR: f64 = 2010.0;

/// Composite Simpson integral of `f` over `[a, b]` with `n` (even) panels.
pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    debug_assert!(n >= 2 && n.is_multiple_of(2), "Simpson needs an even panel count");
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        sum += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// The time in `[0, hi]` at which the increasing `cumulative` reaches
/// `target` (bisection), or `None` if it is still below `target` at `hi`.
pub fn invert(cumulative: impl Fn(f64) -> f64, target: f64, hi: f64) -> Option<f64> {
    if cumulative(hi) < target {
        return None;
    }
    let (mut lo, mut up) = (0.0, hi);
    for _ in 0..80 {
        let mid = 0.5 * (lo + up);
        if cumulative(mid) < target {
            lo = mid;
        } else {
            up = mid;
        }
    }
    Some(0.5 * (lo + up))
}

/// An Exp(1) draw from a uniform one, kept off `ln(0)`.
pub fn exp1(u: f64) -> f64 {
    -u.max(f64::MIN_POSITIVE).ln()
}

/// One generated subject: `observations` at or before [`ENTRY`] (and any
/// later measurements), at most one absorbing `event` of `(time since entry,
/// code)`, and follow-up that ends at the event or at `censor` after entry.
pub fn subject(
    id: String,
    observations: Vec<Observation>,
    event: Option<(f64, &str)>,
    censor: f64,
) -> Subject {
    let (events, follow) = match event {
        Some((t, code)) => (
            vec![Event {
                t: ENTRY + t,
                code: code.into(),
            }],
            t,
        ),
        None => (Vec::new(), censor),
    };
    Subject {
        subject_id: id,
        group_id: None,
        weight: 1.0,
        source: "synthetic".into(),
        entry: ENTRY,
        calendar_at_entry: CALENDAR,
        observations,
        events,
        at_risk: vec![AtRisk {
            code: "*".into(),
            from: ENTRY,
            to: ENTRY + follow,
        }],
    }
}

/// `(time since entry, cause index)` of each subject's absorbing outcome, or
/// its censoring time, as the survival crate's observations.
pub fn outcomes(subjects: &[Subject], codes: &[&str]) -> Vec<(f64, Option<usize>)> {
    subjects
        .iter()
        .map(|s| {
            let ev = s
                .events
                .iter()
                .find_map(|e| codes.iter().position(|c| *c == e.code).map(|k| (e.t, k)));
            match ev {
                Some((t, k)) => (t - s.entry, Some(k)),
                None => (s.at_risk[0].to - s.entry, None),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simpson_integrates_a_smooth_function_and_invert_undoes_it() {
        let v = simpson(|x| x.exp(), 0.0, 2.0, 64);
        assert!((v - (2f64.exp() - 1.0)).abs() < 1e-6);
        let t = invert(|t| 0.3 * t * t, 1.2, 10.0).unwrap();
        assert!((t - 2.0).abs() < 1e-9);
        assert!(invert(|t| 0.3 * t, 100.0, 10.0).is_none());
    }
}
