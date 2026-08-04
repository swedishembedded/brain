// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for the imaging workstream's four models.
//!
//! `crates/gradcheck/src/lib.rs`'s `check_*` entries are library functions; the
//! CLI only drives `check_gpt`. These tests are what actually runs the four new
//! ones in CI, on whichever backend `BRAIN_DEVICE` selects.
//!
//! Each is gated on `MOE_SKIP_GPU_TESTS` like every other GPU-touching test, and
//! reports through `Report::print` so a failure names the offending tensor.

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

macro_rules! check {
    ($name:ident, $call:path, $label:literal) => {
        #[test]
        fn $name() {
            if skip_gpu() {
                eprintln!("{}: skipped (MOE_SKIP_GPU_TESTS)", $label);
                return;
            }
            println!("--- {} ---", $label);
            let report = $call(1);
            report.print();
            assert!(report.all_within(4e-3, 8e-2), "{}: {:?}", $label, report.failures(4e-3, 8e-2));
        }
    };
}

check!(sam2_backward, gradcheck::check_sam2, "check_sam2");
check!(arcface_backward, gradcheck::check_arcface, "check_arcface");
check!(vqgan_backward, gradcheck::check_vqgan, "check_vqgan");
check!(clip_backward, gradcheck::check_clip, "check_clip");
