// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain flops --model flux2|ltxv` prices a whole generation, and says so
//! honestly.
//!
//! Swedish Embedded AB implements analytic performance models for GPU
//! inference pipelines. If your team needs a cost figure it can act on before
//! the hardware exists, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Three properties, at toy dimensions so the gate costs seconds:
//!
//! * the derivation VERIFIES itself - the full-depth cost is extrapolated from
//!   probe builds, and the command refuses to print a total whose linearity it
//!   could not confirm at a point outside the basis;
//! * the offline recording and a real execution of the same graph agree
//!   EXACTLY, which is what makes "offline" a synonym for "the same thing,
//!   without running it" rather than for "an estimate";
//! * every dispatch is covered. A generation total with an uncovered kernel in
//!   it is a partial number presented as a complete one, and the whole design
//!   exists to refuse that.

use std::process::Command;

fn bin() -> String {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("brain");
    p.to_string_lossy().into_owned()
}

fn flops(args: &[&str]) -> String {
    let out = Command::new(bin()).arg("flops").args(args).output().expect("run brain flops");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "brain flops {args:?} exited {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", out.status.code());
    stdout
}

fn assert_has(out: &str, needle: &str) {
    assert!(out.contains(needle), "expected {needle:?} in:\n{out}");
}

#[test]
fn flux2_prices_every_stage_of_a_generation_and_checks_itself() {
    let out = flops(&["--model", "flux2", "--variant", "tiny", "--width", "128", "--height", "128", "--steps", "4", "--run"]);
    // The self-check that makes the extrapolation legitimate.
    assert_has(&out, "block-depth linearity check: EXACT");
    // A generation is stages, not one graph, and the VAE decode is one of them.
    assert_has(&out, "denoise (MMDiT forward)");
    assert_has(&out, "vae-decode (128x128 image)");
    assert_has(&out, "TOTAL (modelled stages)");
    // Offline is the same graph as online, not an approximation of it.
    assert_has(&out, "offline == online");
    // ...and the number describes the whole of it: this is a coverage
    // fraction over dispatches, not a rate.
    // perf-number: the string being asserted is the command's own coverage line
    assert_has(&out, "(100.0%)");
    assert!(!out.contains("UNCOVERED"), "an uncovered kernel would make the total partial:\n{out}");
    // Each stage says which roof it is against, which is what makes it
    // actionable rather than merely large.
    assert!(out.contains("compute") || out.contains("memory"), "no roofline classification:\n{out}");
}

#[test]
fn ltxv_prices_a_clip_and_reports_it_per_second_of_video() {
    let out = flops(&[
        "--model", "ltxv", "--variant", "tiny", "--width", "256", "--height", "256", "--frames", "17", "--fps", "24",
        "--steps", "8", "--run",
    ]);
    assert_has(&out, "block-depth linearity check: EXACT");
    assert_has(&out, "denoise (DiT forward)");
    // The 3D VAE decode is priced as the TILED decode the pipeline really
    // runs when the clip is big enough for one, not as a whole-volume graph
    // that would neither fit a card nor describe the same work.
    assert_has(&out, "vae-decode (3D, 17 frames");
    assert_has(&out, "offline == online");
    // perf-number: the string being asserted is the command's own coverage line
    assert_has(&out, "(100.0%)");
    // The temporal axis, stated: a video's cost is per second of output, which
    // is the number anyone deciding whether to run it reasons about.
    // perf-number: a clip LENGTH echoed back from the request, not a measured rate
    assert_has(&out, "17 frames at 24 fps = 0.71 s of video");
    assert_has(&out, "per second of video:");
    assert!(!out.contains("UNCOVERED"), "an uncovered kernel would make the total partial:\n{out}");
}

/// The FLUX.2 VAE decode is priced from a decoder built over SHAPE-ONLY
/// weights, and a wrong shape there does not fail loudly - it builds the graph
/// at the wrong dimensions, which is a wrong number wearing a right number's
/// clothes. `--vae` builds the same graph a second time from the real
/// checkpoint and requires the two to be identical dispatch for dispatch.
#[test]
fn the_vae_manifest_builds_the_same_graph_as_the_real_checkpoint() {
    let Ok(dir) = std::env::var("BRAIN_FLUX2_VAE") else {
        brain_testutil::skip("set BRAIN_FLUX2_VAE to the FLUX.2 vae/ dir");
        return;
    };
    let out = flops(&["--model", "flux2", "--variant", "tiny", "--width", "128", "--height", "128", "--vae", &dir]);
    assert_has(&out, "vae manifest check: EXACT");
}
