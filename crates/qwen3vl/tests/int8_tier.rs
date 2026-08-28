// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The int8 decoder tier, gated as the LOSSY thing it is.
//!
//! Kept in its own file, deliberately, and it asserts NOTHING about caption
//! equality. The exact-output claim this crate makes lives in
//! `vision_tower_parity.rs` and is about the fp32 path; folding an int8 number
//! in beside it would quietly turn "the caption did not change" into "the
//! caption did not change much", which is a different and far weaker
//! statement. Measured on real weights, int8 and fp32 captions of the same
//! photograph diverge more often than not, and the divergences are sometimes
//! substantive rather than cosmetic - so what can honestly be gated here is
//! not accuracy but **honesty about the tier**:
//!
//! * the tier a caller asks for is the tier that runs, or the caller is told
//!   it was promoted - a run that silently fell back to fp32 must not be
//!   indistinguishable from one that got int8, in either direction;
//! * the two tiers are separate residents, so an int8 request can never be
//!   served by an fp32 model that happens to already be loaded;
//! * `int8` is never reachable by default, from any spelling.
//!
//! What int8 numerically costs is bounded one level down, by
//! `qwen3`'s own `int8_kv_decode_tracks_fp32` (relative L2 of the decode
//! hidden state under 10%). That is a generous bound and it is the right one
//! to know before choosing this tier: a 10% perturbation of the hidden state
//! flips a greedy argmax readily, which is precisely why the captions differ.

use qwen3vl::caps::Precision;

/// `int8` must be unreachable except by asking for it in those words. A tier
/// that can be arrived at by an empty string, a typo, or a default is a tier
/// somebody ships training data from without deciding to.
#[test]
fn the_lossy_tier_is_never_the_default_and_never_a_typo() {
    assert_eq!(Precision::default(), Precision::F32);
    for spelling in ["", "fp32", "f32"] {
        assert_eq!(Precision::from_name(spelling).unwrap(), Precision::F32, "{spelling:?} must mean fp32");
    }
    for spelling in ["int8", "i8"] {
        assert_eq!(Precision::from_name(spelling).unwrap(), Precision::I8, "{spelling:?} must mean int8");
    }
    // Anything else is refused BY NAME rather than falling back to either
    // tier: silently giving fp32 wastes an operator's time, and silently
    // giving int8 corrupts their dataset.
    for bad in ["int4", "8", "INT8", "fp16", "true"] {
        let err = Precision::from_name(bad).unwrap_err();
        assert!(err.contains(bad), "the error must name the offending spelling: {err}");
        assert!(err.contains("fp32") && err.contains("int8"), "the error must list what IS accepted: {err}");
    }
}

/// The name a tier reports must round-trip to the tier itself, because that
/// string is what reaches the served action, the resident key and the log line
/// an operator reads to find out what they ran.
#[test]
fn tier_names_round_trip() {
    for p in [Precision::F32, Precision::I8] {
        assert_eq!(Precision::from_name(p.name()).unwrap(), p);
    }
}

/// On real weights: asking for int8 must GET int8, and the two tiers must be
/// separate residents.
///
/// Both halves matter and neither is about accuracy. `Weight::upload` promotes
/// a request the device cannot serve back to fp32, so `linear_dtype` is the
/// only honest answer to "what am I running"; and if the resident key ignored
/// precision, the second request in a process would silently be served by the
/// first request's model - which is how a labelling run ends up half in one
/// tier and half in the other with nothing in the output to say so.
#[test]
fn asking_for_int8_gets_int8_and_does_not_reuse_the_fp32_resident() {
    let dir = std::env::var("BRAIN_QWEN3VL_WEIGHTS").unwrap_or_default();
    if dir.is_empty() || !std::path::Path::new(&dir).join("config.json").exists() {
        brain_testutil::skip("BRAIN_QWEN3VL_WEIGHTS not set / checkpoint absent");
        return;
    }
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip("MOE_SKIP_GPU_TESTS set");
        return;
    }
    // A small budget: this test is about the tier, not about image detail, and
    // a full-size resident is minutes of load for no extra coverage.
    let px = 256 * 256;

    let f32_load = qwen3vl::caps::load_time(&dir, px, Precision::F32).expect("fp32 resident");
    assert_eq!(qwen3vl::caps::linear_dtype(&dir, px, Precision::F32).unwrap(), Some("fp32".to_string()));

    // A DIFFERENT tier must not be served by the resident just built. Building
    // is minutes, so a reused resident is unmistakable in the clock as well as
    // in the reported tier.
    let i8_load = qwen3vl::caps::load_time(&dir, px, Precision::I8).expect("int8 resident");
    let landed = qwen3vl::caps::linear_dtype(&dir, px, Precision::I8).unwrap();
    assert_eq!(
        landed,
        Some("int8".to_string()),
        "this device promoted the int8 request back to {landed:?}; the tier is unavailable here, \
         so this gate cannot say anything about it"
    );
    assert!(i8_load > 0.5, "the int8 request reused the fp32 resident (built in {i8_load:.3}s vs {f32_load:.1}s)");

    // And back again, so the swap is not one-way.
    assert_eq!(qwen3vl::caps::linear_dtype(&dir, px, Precision::F32).unwrap(), Some("fp32".to_string()));
}
