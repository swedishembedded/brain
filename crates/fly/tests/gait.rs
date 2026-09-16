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
    let g = analyse(&synthetic(12.0, std::f64::consts::PI, 1.0)).expect("a second is enough");
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
    let g = analyse(&synthetic(12.0, 0.0, 1.0)).expect("a second is enough");
    assert!(g.tripod < -0.98, "all six legs in phase is the opposite of a tripod, got {}", g.tripod);
    assert!(g.score() < 0.02, "a six-legged hop is not walking; it scored {}", g.score());
}

#[test]
fn noise_is_not_mistaken_for_a_rhythm() {
    let mut t = Trace::new(DT);
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..500 {
        let mut legs = [0.0f32; 6];
        for l in legs.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *l = (x as f32 / u64::MAX as f32) - 0.5;
        }
        t.push(legs);
    }
    let g = analyse(&t).expect("a second is enough");
    // THE CONTROL: the sine above scored above 0.9 on the same measure, so a
    // low number here is a statement about the signal and not about the scale.
    assert!(g.rhythmicity < 0.2, "white noise should have no dominant frequency, got {}", g.rhythmicity);
    assert!(g.power > 0.05, "the noise is not silent; power was {}", g.power);
}

#[test]
fn a_motionless_body_is_reported_as_motionless_rather_than_arrhythmic() {
    let mut t = Trace::new(DT);
    for _ in 0..500 {
        t.push([0.0; 6]);
    }
    let g = analyse(&t).expect("a second is enough");
    assert_eq!(g.power, 0.0, "nothing moved, so there is no power");
    assert_eq!(g.score(), 0.0, "nothing moved, so nothing is a gait");
}

#[test]
fn a_trace_too_short_to_hold_a_cycle_is_refused_rather_than_guessed_at() {
    let t = synthetic(12.0, std::f64::consts::PI, 0.05);
    assert!(analyse(&t).is_none(), "50 ms cannot contain two cycles of the slowest band edge");
    // THE CONTROL: the same signal, long enough, must be accepted - otherwise
    // this passes against an analyse() that refuses everything.
    assert!(analyse(&synthetic(12.0, std::f64::consts::PI, 1.0)).is_some());
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
