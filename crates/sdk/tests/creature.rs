// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end coverage for [`brain::Creature`]/[`brain::View`] - the second
//! SDK surface's counterpart to `tests/image_pipeline.rs`: a public pipeline
//! is not done without a real load -> call -> inspect -> (save) test, and
//! this surface had none.
//!
//! Two tiers, mirroring the split `crates/fly`'s own test suite already
//! uses for the exact same reason: builder-argument validation needs no real
//! data and always runs; the real build -> drive -> step -> inspect path
//! needs a real MANC connectome export and the flybody MJCF, so it is gated
//! on the same `BRAIN_CONNECTOME_DIR`/`BRAIN_FLYBODY_XML` env vars
//! `crates/fly/tests/loop_closes.rs`'s `rig()` uses, and skips cleanly (not
//! silently - see [`brain_testutil::skip_unavailable`]) when they are
//! absent. This workspace has no bundled connectome to ship as a fixture.

#![cfg(feature = "creature")]

use brain::Creature;

#[test]
fn build_without_a_connectome_names_the_missing_argument() {
    // `Creature` builds no `Debug` impl (it holds live GPU/MuJoCo handles,
    // the same reason `ImagePipeline` doesn't either - see that type's own
    // doc), so `unwrap_err()` isn't available here.
    let Err(err) = Creature::fruit_fly().build() else {
        panic!("a builder with neither .connectome() nor .body() must fail");
    };
    let msg = err.to_string();
    assert!(msg.contains(".connectome"), "{msg}");
}

#[test]
fn build_without_a_body_names_the_missing_argument_before_touching_disk() {
    // A path that does not exist: if this error came from `connectome::find`
    // failing to read it, the message would say so. It doesn't reach that
    // far - the body check runs first, with zero I/O.
    let Err(err) = Creature::fruit_fly().connectome("/nonexistent/path/for/this/test").build() else {
        panic!("a builder with no .body() must fail");
    };
    let msg = err.to_string();
    assert!(msg.contains(".body"), "{msg}");
}

/// A real connectome dir + a real flybody MJCF, or `None` - see this file's
/// module doc for why there is no bundled fallback.
fn real_paths() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let connectome = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty())?;
    let xml = std::env::var_os("BRAIN_FLYBODY_XML").filter(|v| !v.is_empty())?;
    Some((connectome.into(), xml.into()))
}

#[test]
fn build_drive_step_and_reset_a_real_fly() {
    let Some((connectome, xml)) = real_paths() else {
        brain_testutil::skip_unavailable("BRAIN_CONNECTOME_DIR/BRAIN_FLYBODY_XML unset - no bundled connectome ships with this workspace");
        return;
    };
    let mut fly = match Creature::fruit_fly().connectome(&connectome).body(&xml).build() {
        Ok(f) => f,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("creature failed to build on this box: {e}"));
            return;
        }
    };

    assert!(fly.neurons() > 0, "a real MANC export has neurons");
    assert!(!fly.wiring().is_empty(), "a real body has a motor map");

    let start = fly.position();
    fly.drive(2.0);
    let beat = fly.step_for(200).expect("stepping a built creature");
    assert!(beat.tick > 0);

    let moved = fly.position();
    assert_ne!(start, moved, "driving forward for 200 ticks should move the body");

    fly.reset();
    let after_reset = fly.position();
    assert!((after_reset[0] - start[0]).abs() < 1e-6, "reset returns to the start pose");
}

#[test]
fn a_view_renders_a_frame_of_the_requested_size() {
    let Some((connectome, xml)) = real_paths() else {
        brain_testutil::skip_unavailable("BRAIN_CONNECTOME_DIR/BRAIN_FLYBODY_XML unset");
        return;
    };
    let fly = match Creature::fruit_fly().connectome(&connectome).body(&xml).build() {
        Ok(f) => f,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("creature failed to build on this box: {e}"));
            return;
        }
    };
    let mut view = match brain::View::open(&fly, "sdk creature test", 64, 64) {
        Ok(v) => v,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no display available: {e}"));
            return;
        }
    };
    view.show(&fly, "test").expect("rendering a built creature");
    assert_eq!(view.frame().len(), 64 * 64 * 3, "one RGB8 frame at the requested size");
}
