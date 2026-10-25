// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SpectralLM's backward against finite differences of its own forward: the
//! spectral mixer on the dense-basis path and on the FFT path, and the SwiGLU
//! control. The tied embedding is also checked element by element, since it
//! folds two gradient contributions and a directional check can hide a
//! partial one.

use gradcheck::{check_spectrallm, elementwise_check, Report};
use spectrallm::{Mixer, SpectralConfig, SpectralLm, TransformPath};

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8.0e-2;

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(fails.is_empty(), "{what}: gradient check failed for {:?}", fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    let dead = report.dead_gradients();
    assert!(dead.is_empty(), "{what}: silently-dead gradients {:?}", dead.iter().map(|c| (&c.param, c.analytic, c.numeric)).collect::<Vec<_>>());
}

fn spectral() -> Mixer {
    SpectralConfig::tiny().mixer
}

#[test]
fn spectral_mixer_dense_path_gradients_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(check_spectrallm(seed, spectral(), TransformPath::Dense), &format!("spectral dense, seed {seed}"));
    }
}

#[test]
fn spectral_mixer_fft_path_gradients_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let probe = SpectralLm::new_inference(SpectralConfig::tiny(), 1, 1, &spectrallm::init_weights(&SpectralConfig::tiny(), 1));
    let caps = probe.gpu.caps();
    if !caps.workgroup_reductions || caps.workgroup_mem_bytes < spectrallm::mixer::FFT_WORKGROUP_BYTES {
        brain_testutil::skip_unavailable("the FFT transform kernels need workgroup barriers this device does not run");
        return;
    }
    for seed in [3u64, 11] {
        gate(check_spectrallm(seed, spectral(), TransformPath::Fft), &format!("spectral fft, seed {seed}"));
    }
}

#[test]
fn swiglu_control_gradients_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    gate(check_spectrallm(5, SpectralConfig::tiny_swiglu().mixer, TransformPath::Auto), "swiglu control, seed 5");
}

#[test]
fn tied_embedding_gradient_is_complete_elementwise() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let cfg = SpectralConfig { transform: TransformPath::Dense, ..SpectralConfig::tiny() };
    let model = SpectralLm::new_train(cfg.clone(), 2, 6, &spectrallm::init_weights(&cfg, 7));
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 6) % 23).collect();
    model::Model::set_batch(&model, model::Batch::Lm { tokens: &x, targets: &y });
    let report = elementwise_check(&model, "tok.weight", 1e-3);
    let fails = report.failures(ATOL, RTOL);
    assert!(fails.is_empty(), "tok.weight elementwise: {} of {} entries off", fails.len(), report.checks.len());
}
