// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does the gait measurement measure a gait?
//!
//! These run on SYNTHETIC traces, not on the fly, so they are fast and so the
//! right answer is known exactly. A metric validated only against real data
//! can be wrong in the same direction as the thing it is scoring and nobody
//! finds out.

use fly::gait::{analyse, Trace};

const DT: f64 = 0.002;

/// A trace of `seconds` at `hz`, with tripod 1 shifted by `phase` radians.
///
/// Two seconds rather than one throughout: `analyse` discards the first
/// quarter of a trace as the drive's onset transient, and what remains still
/// has to hold two cycles of the slowest frequency it looks at.
fn synthetic(hz: f64, phase: f64, seconds: f64) -> Trace {
    let mut t = Trace::new(DT);
    let n = (seconds / DT) as usize;
    for k in 0..n {
        let base = 2.0 * std::f64::consts::PI * hz * k as f64 * DT;
        let mut legs = [0.0f32; 6];
        for (i, (_, _, tripod)) in flybody::LEGS.iter().enumerate() {
            let p = base + if *tripod == 0 { 0.0 } else { phase };
            legs[i] = p.sin() as f32;
        }
        t.push(legs);
    }
    t
}

#[test]
fn a_perfect_alternating_tripod_reads_as_one() {
    let g = analyse(&synthetic(12.0, std::f64::consts::PI, 2.0)).expect("two seconds is enough");
    assert!((g.step_hz - 12.0).abs() <= 0.5, "expected 12 Hz, got {}", g.step_hz);
    assert!(g.tripod > 0.98, "a half-cycle shift is a perfect tripod, got {}", g.tripod);
    assert!(g.rhythmicity > 0.9, "a pure sine is entirely rhythmic, got {}", g.rhythmicity);
    assert!(g.score() > 0.9, "that is a gait; it scored {}", g.score());
}

#[test]
fn six_legs_moving_together_is_rhythmic_but_is_not_a_gait() {
    // THE CONTROL that separates the two measurements. Same frequency, same
    // amplitude, same rhythmicity - only the phase differs. Without this, a
    // metric that only ever looked at the spectrum would pass the test above.
    let g = analyse(&synthetic(12.0, 0.0, 2.0)).expect("two seconds is enough");
    assert!(g.tripod < -0.98, "all six legs in phase is the opposite of a tripod, got {}", g.tripod);
    assert!(g.score() < 0.02, "a six-legged hop is not walking; it scored {}", g.score());
}

#[test]
fn noise_is_not_mistaken_for_a_rhythm() {
    let mut t = Trace::new(DT);
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..1000 {
        let mut legs = [0.0f32; 6];
        for l in legs.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *l = (x as f32 / u64::MAX as f32) - 0.5;
        }
        t.push(legs);
    }
    let g = analyse(&t).expect("two seconds is enough");
    // THE CONTROL: the sine above scored above 0.9 on the same measure, so a
    // low number here is a statement about the signal and not about the scale.
    assert!(g.rhythmicity < 0.2, "white noise should have no dominant frequency, got {}", g.rhythmicity);
    assert!(g.power > 0.05, "the noise is not silent; power was {}", g.power);
}

#[test]
fn a_motionless_body_is_reported_as_motionless_rather_than_arrhythmic() {
    let mut t = Trace::new(DT);
    for _ in 0..1000 {
        t.push([0.0; 6]);
    }
    let g = analyse(&t).expect("two seconds is enough");
    assert_eq!(g.power, 0.0, "nothing moved, so there is no power");
    assert_eq!(g.score(), 0.0, "nothing moved, so nothing is a gait");
}

#[test]
fn a_trace_too_short_to_hold_a_cycle_is_refused_rather_than_guessed_at() {
    let t = synthetic(12.0, std::f64::consts::PI, 0.05);
    assert!(analyse(&t).is_none(), "50 ms cannot contain two cycles of the slowest band edge");
    // THE CONTROL: the same signal, long enough, must be accepted - otherwise
    // this passes against an analyse() that refuses everything.
    assert!(analyse(&synthetic(12.0, std::f64::consts::PI, 2.0)).is_some());
}

#[test]
fn the_frequency_band_is_a_slope_and_not_a_cliff() {
    // An optimiser at 2 Hz has to be able to tell it is getting warmer as it
    // approaches the band, or the score is unreachable from below.
    let far = analyse(&synthetic(2.0, std::f64::consts::PI, 2.0)).unwrap().score();
    let near = analyse(&synthetic(3.5, std::f64::consts::PI, 2.0)).unwrap().score();
    let inside = analyse(&synthetic(8.0, std::f64::consts::PI, 2.0)).unwrap().score();
    assert!(far < near, "2 Hz scored {far}, 3.5 Hz scored {near}; the approach is not monotone");
    assert!(near < inside, "3.5 Hz scored {near}, 8 Hz scored {inside}");
}

/// The analyser has to recognise a real fly walking.
///
/// Every gait number in this crate comes from `gait::analyse`, and until this
/// existed nothing had checked that it scores an actual walk highly. That is
/// the wrong way round, and it left two things unknown that this settles.
///
/// The CEILING. A real fly scores about 0.46, not 1.0, because the score is a
/// product of three terms none of which is ever perfect on a real animal. Every
/// simulated score in this repository should be read against 0.46 - which
/// makes an earlier result of 0.229 half of a real walk rather than a fifth of
/// one, and the current 0.017 about four percent.
///
/// And the SIGN. `tripod` comes out +0.65 on real walking, so the analyser's
/// two leg triangles are the right way round. Had it been negative, every gait
/// score ever reported here would have been upside down and the searches would
/// have been optimising for all six legs moving together.
///
/// Skips when the reference is not on the machine.
#[test]
fn the_analyser_recognises_a_real_fly_walking() {
    let (Ok(path), Ok(body)) = (std::env::var("BRAIN_FLY_REFERENCE"), std::env::var("BRAIN_FLYBODY_XML")) else {
        eprintln!("skipping: set BRAIN_FLY_REFERENCE and BRAIN_FLYBODY_XML");
        return;
    };
    let Ok(reference) = fly::reference::Reference::load(&path) else {
        eprintln!("skipping: {path} did not load");
        return;
    };
    let Ok(mj) = mujoco::MuJoCo::load() else {
        eprintln!("skipping: no MuJoCo");
        return;
    };
    let model = mujoco::Model::from_xml(&mj, std::path::Path::new(&body)).expect("the body loads");

    // The same six coordinates the simulated fly's gait is read from.
    let mut coxa = [usize::MAX; 6];
    for (i, (seg, side, _)) in flybody::LEGS.iter().enumerate() {
        coxa[i] = model
            .joint_qpos(&flybody::LegDof::Coxa.actuator(*seg, *side))
            .unwrap_or_else(|| panic!("the body has no coxa joint for leg {i}"));
    }

    let mut scored: Vec<fly::Gait> = Vec::new();
    for s in 0..reference.snippets() {
        let n = reference.len(s);
        if n < 700 {
            continue;
        }
        let mut trace = fly::Trace::new(reference.timestep());
        for i in 0..n {
            let Some((qpos, _)) = reference.frame(s, i) else { continue };
            let mut per_leg = [0.0f32; 6];
            for (slot, &q) in per_leg.iter_mut().zip(&coxa) {
                *slot = qpos.get(q).copied().unwrap_or(0.0);
            }
            trace.push(per_leg);
        }
        if let Some(g) = fly::analyse_gait(&trace) {
            scored.push(g);
        }
    }
    assert!(!scored.is_empty(), "no reference snippet was long enough to analyse");

    let n = scored.len() as f64;
    let mean = |f: fn(&fly::Gait) -> f64| scored.iter().map(f).sum::<f64>() / n;
    let (hz, tripod, score) = (mean(|g| g.step_hz), mean(|g| g.tripod), mean(|g| g.score()));
    eprintln!("real fly walking: {hz:.2} Hz, tripod {tripod:.3}, score {score:.3} over {n} snippets");

    assert!(tripod > 0.3, "a real tripod gait must score POSITIVE tripod, got {tripod:.3} - the triangles are swapped");
    assert!((4.0..=25.0).contains(&hz), "a real fly steps at about 7 Hz, and the analyser read {hz:.2}");
    assert!(score > 0.3, "the analyser scored a real walk at {score:.3}, so it does not recognise walking");
    assert!(score < 0.95, "a real walk scoring {score:.3} would mean the score has no headroom left to be wrong in");
}
