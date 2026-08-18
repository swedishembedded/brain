// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `id_cond` composition convention, gated against the dumped reference.
//!
//! `tests/parity.rs` replays `id_cond` from the golden wholesale, so it cannot
//! catch a wrong *composition* — it never builds one. This file does, because
//! the convention is asymmetric and the asymmetry is invisible to every
//! structural check: `ArcFace(raw) ‖ EvaClip(L2-normalised)` and
//! `ArcFace(L2-normalised) ‖ EvaClip(L2-normalised)` have the same length, the
//! same dtype and the same finite values.
//!
//! brain's `arcface` `embed` action normalises (its output is meant to be
//! cosine-ready), so wiring *that* into PuLID is the natural mistake, and it
//! would leave the first 512 components ~20x too small.
//!
//! The reference settles it numerically: in `testdata/pulid/idformer.safetensors`
//! the dumped `id_cond` has `‖[:512]‖ = 20.11` and `‖[512:]‖ = 1.0000`.
//!
//! Fixtures resolve from `$BRAIN_TESTDATA` (default `<repo>/testdata`); the test
//! skips itself when the golden is absent, and `BRAIN_REQUIRE_FIXTURES=1` makes
//! that skip fatal.

use brain_testutil::testdata_path as testdata;

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// The dumped `id_cond`, or `None` with the skip booked here - both tests need
/// the same golden, so the reason it is absent is stated in one place.
fn id_cond() -> Option<Vec<f32>> {
    let p = testdata("pulid/idformer.safetensors");
    if !p.exists() {
        brain_testutil::skip(&format!(
            "{} absent (run tools/goldens/pulid_dump_reference.py)",
            p.display()
        ));
        return None;
    }
    let t = checkpoint::safetensors::read(p.to_str().unwrap()).expect("read golden");
    Some(t.into_iter().find(|x| x.name == "id_cond").expect("golden has id_cond").data)
}

/// The dumped `id_cond` must show the reference's asymmetry, and `compose` must
/// reproduce that exact vector from its own two halves.
#[test]
fn compose_reproduces_the_reference_id_cond() {
    let Some(want) = id_cond() else { return };

    let cfg = pulid::config::PulidConfig::v0_9_1();
    assert_eq!(want.len(), cfg.id_cond_dim);
    let d = pulid::idcond::ARCFACE_DIM;

    // 1. The reference itself is asymmetric — this is the fact everything else
    //    rests on, so assert it rather than cite it.
    let (n_arc, n_eva) = (norm(&want[..d]), norm(&want[d..]));
    assert!(
        (n_eva - 1.0).abs() < 1e-4,
        "reference EVA half should be L2-normalised, got {n_eva}"
    );
    assert!(
        n_arc > 5.0,
        "reference ArcFace half should be RAW (‖e‖ ~ 15-20), got {n_arc} — if this \
         ever approaches 1.0 the reference changed and idcond must follow"
    );

    // 2. Rebuilding from those halves must give back the same vector. The EVA
    //    half is already normalised so re-normalising is idempotent; the
    //    ArcFace half must survive byte-for-byte.
    let got = pulid::idcond::compose(&cfg, &want[..d], &want[d..]).expect("compose");
    let max_abs = got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-6, "compose must reproduce the reference id_cond, max_abs {max_abs:.3e}");
}

/// The mistake this module exists to prevent, stated as a test: normalising the
/// ArcFace half produces a vector that passes every structural check and is
/// still wrong.
#[test]
fn normalising_the_arcface_half_would_be_detectably_wrong() {
    let Some(want) = id_cond() else { return };
    let d = pulid::idcond::ARCFACE_DIM;

    let wrong = model::hostmath::l2_normalize(&want[..d]);
    let ratio = norm(&want[..d]) / norm(&wrong);
    assert!(
        ratio > 5.0,
        "the normalised ArcFace half should differ from the reference by a large \
         scale factor (got {ratio:.2}x) — if it does not, this guard is useless"
    );
}
