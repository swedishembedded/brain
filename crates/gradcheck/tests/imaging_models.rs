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
                brain_testutil::skip_unavailable(&format!("{}: MOE_SKIP_GPU_TESTS set", $label));
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
check!(vocoder_backward, gradcheck::check_vocoder, "check_vocoder");
check!(dit_backward, gradcheck::check_dit, "check_dit");
// `check_dit` builds its trainer on `backend-cpu` unconditionally, so it can
// never reach the capability-gated fast GEMM tier or a real card. The `_tiled`
// sibling runs the same backward on the pooled test device at dims past
// `block::pick_gemm`'s crossover - see its own doc.
check!(dit_tiled_backward, gradcheck::minimaxmusic3::check_dit_tiled, "check_dit_tiled");

// ---- phase 4c: the four newer models ----
//
// `check_t5` is the T5 encoder backward (`t5encoder::train::T5Trainer`). The two
// siblings are run because each isolates a failure mode the main gate cannot:
// `check_t5_one_block` removes the cross-block accumulation of the shared
// relative-position bias, and `check_t5_tiled` forces `block::pick_gemm` onto
// the register-tiled backward GEMMs (which on `backend-cpu` route to the AVX2
// fast path instead of the WGSL, so running the suite on both backends covers
// two different implementations of the same op).
check!(t5_backward, gradcheck::check_t5, "check_t5");
check!(t5_one_block_backward, gradcheck::t5::check_t5_one_block, "check_t5_one_block");
check!(t5_tiled_backward, gradcheck::t5::check_t5_tiled, "check_t5_tiled");
// `check_t5` does NOT cover the cross-block fold of the shared relative-position
// bias - measured: deleting the `axpy` leaves an error of a third in that tensor's
// gradient and `check_t5` still passes on both backends and both seeds. The
// per-ENTRY check below is what covers it.
check!(
    t5_rel_bias_elementwise,
    gradcheck::t5::check_t5_rel_bias_elementwise,
    "check_t5_rel_bias_elementwise"
);

// `check_codeformer` is CodeFormer's code-prediction Transformer under the
// code-token cross-entropy — stage II of the reference recipe, with the VQ
// autoencoder frozen (its own backward is `check_vqgan`, above). The
// single-layer sibling isolates the cross-layer `position_emb` accumulation.
check!(codeformer_backward, gradcheck::check_codeformer, "check_codeformer");
check!(
    codeformer_one_layer_backward,
    gradcheck::restore::check_codeformer_one_layer,
    "check_codeformer_one_layer"
);
