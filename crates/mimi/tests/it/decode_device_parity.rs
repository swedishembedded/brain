// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements neural audio codec inference on GPUs for its clients. If your team needs
// expertise in low-latency speech synthesis then you can procure our services by sending an email to
// info@swedishembedded.com.

//! `Codec::decode` on the ambient device against the same codec on the CPU
//! backend.
//!
//! The CPU backend always runs the direct conv kernels and the host-order
//! reductions, so it is the independent witness for the device pass: the GEMM
//! conv lowering, the device layout permutes and the device ConvNeXt
//! LayerNorm/GELU all have to land on the same waveform within the noise a
//! reassociated fp32 sum allows. Synthetic weights, widened past the GEMM
//! lowering's narrowest conv (`GEMM_CONV1D_MIN_COUT`) so the lowered kernels are
//! actually reached - the 6/8-channel [`tiny_cfg`] stays on the direct ones.

use std::collections::HashMap;

use mimi::{Codec, CodecConfig};

use crate::long_form_window::{assert_not_saturated, codes_for, tiny_cfg, tiny_weights, WINDOW};

/// [`tiny_cfg`] with the conv widths the lowering applies to: `pre_conv`
/// (`Cout = 32`), the SEANet head (`Cout = 64`) and its first two blocks
/// (`32`, `16` channels).
fn wide_cfg() -> CodecConfig {
    CodecConfig { latent_dim: 32, decoder_dim: 64, ..tiny_cfg() }
}

/// [`tiny_weights`] for [`wide_cfg`], with the output conv scaled down: the
/// wider stack drives the waveform into the final clamp, which would make the
/// comparison vacuous.
fn wide_weights(cfg: &CodecConfig) -> HashMap<String, Vec<f32>> {
    let mut weights = tiny_weights(cfg);
    weights.get_mut("decoder.6.conv.weight").expect("output conv weight").iter_mut().for_each(|w| *w *= 0.02);
    weights
}

fn relative_rms(a: &[f32], b: &[f32]) -> f64 {
    let energy: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
    let error: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    (error / energy).sqrt()
}

fn skip_without_device() -> bool {
    if brain_testutil::gpu_tests_skipped() {
        return true;
    }
    if gpu_core::backend_name() == "cpu" {
        eprintln!("ambient device is the CPU backend; nothing to compare against");
        return true;
    }
    false
}

#[test]
fn the_device_decode_matches_the_cpu_backend() {
    if skip_without_device() {
        return;
    }
    let cfg = wide_cfg();
    let weights = wide_weights(&cfg);
    let codes = codes_for(&cfg, 3 * WINDOW as usize);

    let want = Codec::from_weights(cfg.clone(), weights.clone()).decode(&codes);
    let got = Codec::from_weights_on(gpu_core::Gpu::new(mimi::PIPELINES), cfg, weights).decode(&codes);

    assert_eq!(got.len(), want.len(), "device output length differs from the CPU backend's");
    assert_not_saturated(&want, "cpu reference");
    let relative = relative_rms(&want, &got);
    eprintln!("codec decode on {}: relative rms error against the CPU backend {relative:.3e}", gpu_core::backend_name());
    assert!(relative < 1e-3, "relative rms error {relative}");
}

/// A codec decodes clips of different lengths back to back (a serving loop),
/// and decoding the same codes again gives the same samples: nothing carried
/// over from the previous pass - scratch, recycled blocks - may leak in.
#[test]
fn a_repeated_device_decode_of_clips_of_different_lengths_is_unchanged() {
    if skip_without_device() {
        return;
    }
    let cfg = wide_cfg();
    let weights = wide_weights(&cfg);
    let codec = Codec::from_weights_on(gpu_core::Gpu::new(mimi::PIPELINES), cfg.clone(), weights);
    let short = codes_for(&cfg, 2 * WINDOW as usize);
    let long = codes_for(&cfg, 5 * WINDOW as usize);

    let first = codec.decode(&short);
    let _ = codec.decode(&long);
    let again = codec.decode(&short);
    assert_eq!(first, again, "a decode after a longer clip differs from the first decode of the same codes");
}
