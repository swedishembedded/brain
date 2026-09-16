// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates for the reference reader and the imitation reward.

use fly::{ImitationReward, Reference};

/// Build a reference file in memory, so the parser is covered with no data on
/// the machine.
fn synth(count: usize, len: usize, nq: usize, nv: usize, timestep: f64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"BRNFLYW1");
    b.extend_from_slice(&timestep.to_le_bytes());
    b.extend_from_slice(&(nq as u32).to_le_bytes());
    b.extend_from_slice(&(nv as u32).to_le_bytes());
    b.extend_from_slice(&(count as u32).to_le_bytes());
    for _ in 0..count {
        b.extend_from_slice(&(len as u32).to_le_bytes());
    }
    for s in 0..count {
        for f in 0..len {
            for k in 0..(nq + nv) {
                b.extend_from_slice(&((s * 1000 + f * 10 + k) as f32).to_le_bytes());
            }
        }
    }
    b
}

#[test]
fn a_reference_round_trips_its_frames() {
    let r = Reference::parse(&synth(3, 4, 5, 4, 0.002)).unwrap();
    assert_eq!((r.snippets(), r.nq(), r.nv()), (3, 5, 4));
    assert_eq!(r.timestep(), 0.002);
    assert_eq!(r.len(0), 4);

    // Frame (snippet 2, frame 3) must be exactly the values written for it,
    // which is what catches an off-by-one in the per-snippet offsets - the
    // kind of error that yields a plausible trajectory from the wrong snippet.
    let (qpos, qvel) = r.frame(2, 3).unwrap();
    assert_eq!(qpos, &[2030.0, 2031.0, 2032.0, 2033.0, 2034.0]);
    assert_eq!(qvel, &[2035.0, 2036.0, 2037.0, 2038.0]);
    assert!(r.frame(2, 4).is_none(), "past the end must be None, not a wrapped frame");
    assert!(r.frame(9, 0).is_none());
}

#[test]
fn a_malformed_reference_is_refused_rather_than_misread() {
    assert!(Reference::parse(b"nope").is_err());
    let mut bad = synth(2, 3, 4, 4, 0.002);
    bad.truncate(bad.len() - 8);
    let err = Reference::parse(&bad).unwrap_err();
    assert!(err.contains("expected"), "a truncated file must say so: {err}");

    // A zero timestep would divide by zero downstream and is not a duration.
    let mut zero = synth(1, 1, 2, 2, 0.0);
    zero[8..16].copy_from_slice(&0.0f64.to_le_bytes());
    assert!(Reference::parse(&zero).is_err());
}

#[test]
fn a_reference_for_a_different_body_is_refused() {
    let r = Reference::parse(&synth(1, 2, 5, 4, 0.002)).unwrap();
    assert!(r.check_matches(5, 4).is_ok());
    let err = r.check_matches(109, 108).unwrap_err();
    assert!(err.contains("109"), "the error must name both bodies: {err}");
}

#[test]
fn a_perfect_match_scores_the_maximum_and_error_reduces_it_monotonically() {
    let rw = ImitationReward::default();
    let qpos = vec![0.1, 0.2, 0.3];
    let qvel = vec![1.0, 2.0, 3.0];
    let rq: Vec<f32> = qpos.iter().map(|v| *v as f32).collect();
    let rv: Vec<f32> = qvel.iter().map(|v| *v as f32).collect();

    let perfect = rw.total(&qpos, &qvel, &rq, &rv);
    assert!((perfect - rw.max()).abs() < 1e-12, "an exact match must score max(), got {perfect}");
    assert!((rw.max() - 21.0).abs() < 1e-12, "flybody weights are 20 for com and 1 for qvel");

    // Growing position error must reduce the reward, strictly and at every
    // step. A reward that plateaus gives a learning rule nothing to climb.
    let mut last = perfect;
    for k in 1..8 {
        let off: Vec<f64> = qpos.iter().map(|v| v + 0.02 * k as f64).collect();
        let s = rw.total(&off, &qvel, &rq, &rv);
        assert!(s < last, "error {k} scored {s}, not below {last}");
        last = s;
    }
    assert!(last < perfect * 0.5, "a large error should cost most of the reward, got {last}");
}

#[test]
fn the_two_factors_respond_to_their_own_feature_and_not_the_other() {
    let rw = ImitationReward::default();
    let q = vec![0.0, 0.0, 0.0];
    let v = vec![0.0; 6];
    let rq = vec![0.0f32; 3];
    let rv = vec![0.0f32; 6];

    let (c0, v0) = rw.factors(&q, &v, &rq, &rv);

    // Move only the centre of mass: the com factor must fall, the qvel factor
    // must not move at all. A reward that mixed its features would be
    // impossible to attribute.
    let (c1, v1) = rw.factors(&[0.05, 0.0, 0.0], &v, &rq, &rv);
    assert!(c1 < c0, "com error must reduce the com factor");
    assert_eq!(v1, v0, "com error must not touch the qvel factor");

    // And the converse.
    let (c2, v2) = rw.factors(&q, &[40.0, 0.0, 0.0, 0.0, 0.0, 0.0], &rq, &rv);
    assert_eq!(c2, c0, "velocity error must not touch the com factor");
    assert!(v2 < v0, "velocity error must reduce the qvel factor");
}

/// Load the real converted reference when it is present.
#[test]
fn the_published_reference_matches_this_body() {
    let Some(path) = std::env::var_os("BRAIN_FLY_REFERENCE").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_FLY_REFERENCE unset (run tools/convert/flybody_walking_reference.py)");
        return;
    };
    let r = Reference::load(std::path::PathBuf::from(path)).expect("the reference loads");
    eprintln!(
        "{} snippets, {} frames, {:.2} s of walking, timestep {:.3} ms",
        r.snippets(),
        (0..r.snippets()).map(|s| r.len(s)).sum::<usize>(),
        (0..r.snippets()).map(|s| r.len(s)).sum::<usize>() as f64 * r.timestep(),
        r.timestep() * 1e3
    );
    // The body it describes.
    r.check_matches(109, 108).expect("the reference is for flybody");
    // One reference frame per control tick: flybody's walking tasks run
    // control at 500 Hz and this dataset is sampled at exactly that, so
    // nothing is resampled and no interpolation error enters the reward.
    assert!((r.timestep() - 0.002).abs() < 1e-9, "expected a 2 ms reference timestep");
    assert!(r.snippets() > 0 && r.len(0) > 10);

    // The frames must be real data, not zeros: a converter that wrote the
    // header and no payload would satisfy every structural check above.
    //
    // Frame 0's VELOCITY is legitimately all zeros - the recordings start from
    // rest - so asserting otherwise would be asserting something the data does
    // not say. An earlier version of this test did exactly that and failed on
    // correct data. Position at frame 0 is real, and velocity is real later.
    let (q, v) = r.frame(0, 0).unwrap();
    assert!(q.iter().any(|x| *x != 0.0), "frame 0 position is all zeros");
    assert!(q.iter().all(|x| x.is_finite()) && v.iter().all(|x| x.is_finite()));
    let (_, v_mid) = r.frame(0, r.len(0) / 2).unwrap();
    assert!(v_mid.iter().any(|x| *x != 0.0), "the fly never moves in this snippet");

    // The recording tracks the LEGS and the body's own motion, and nothing
    // else: 48 of 108 velocity DoF, being the root free joint's 6 plus seven
    // DoF on each of six legs. The other 60 - head, rostrum, haustellum,
    // labrum, antennae, wings, halteres, abdomen - sit at zero for every
    // frame, because a walking recording does not track them.
    //
    // Pinned because the reward depends on it: scoring the untracked 60
    // against their zeros would penalise any movement of the wings and neck,
    // which this connectome also drives through 66 wing and 24 neck motor
    // neurons. That would make "learn to walk" also mean "hold everything
    // else rigid", which is not the same task.
    let moving = r.moving_dofs();
    eprintln!("  reference tracks {} of {} velocity DoF", moving.len(), r.nv());
    assert_eq!(moving.len(), 48, "expected 6 root DoF plus 7 on each of 6 legs");
    assert!(moving.iter().take(6).copied().eq(0..6), "the root's own motion must be tracked");
}
