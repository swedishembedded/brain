// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched vision-encoder serving for clients who
// read documents at scale. If your team needs expertise in running a vision
// tower over many pages and tiles in one pass then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec of the serving forward: [`Resampler::encode_views`] produces, for every view of a
//! batch of 1, 2, 4, 8 or 16 and for both query banks, the rows the tap-shaped
//! [`Resampler::resample_view`] (the forward the golden parity and the gradient checks pin)
//! produces for that view alone. Batching changes how the work is dispatched, never a value.

use data::rng::Lcg;
use gpu_core::provider::library::GemmPrecision;
use deepseekocr2::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};
use deepseekocr2::encoder::{Resampler, PIPELINES};
use deepseekocr2::init::init_weights;

fn tiny() -> DeepseekOcr2VisionConfig {
    let sam = sam1::SamViTConfig { compress_out: 6, ..sam1::SamViTConfig::tiny() };
    // Every axis a different number, so a swapped stride cannot cancel.
    let encoder = Qwen2EncoderConfig { d_model: 6, n_layers: 3, n_heads: 3, n_kv_heads: 1, ffn_hidden: 9, rms_eps: 1e-6, rope_theta: 10_000.0, n_query_local: 2, n_query_global: 3 };
    DeepseekOcr2VisionConfig { sam, encoder, decoder_hidden: 7, lora: None }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn batched_views_equal_the_one_view_forward() {
    let cfg = tiny();
    let mut init = init_weights(&cfg, 7);
    // Larger than the fresh-init noise, so the comparison is about values and not about rounding of ~0.
    for (name, v) in init.iter_mut() {
        if !name.ends_with("norm.weight") && !name.ends_with("norm1.weight") && !name.ends_with("norm2.weight") {
            v.iter_mut().for_each(|x| *x *= 20.0);
        }
    }
    let mut resampler = Resampler::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &init, false);
    // The batching is the subject here, not the arithmetic: both forwards multiply in fp32 (`tests/tensor_core.rs` holds the rest).
    resampler.set_gemm_precision(GemmPrecision::Fp32Reference);
    let mut rng = Lcg::new(11);
    for local in [true, false] {
        let n_query = if local { cfg.encoder.n_query_local } else { cfg.encoder.n_query_global };
        for batch in [1usize, 2, 4, 8, 16] {
            let views: Vec<Vec<f32>> = (0..batch).map(|_| rng.vec_scaled((n_query * cfg.encoder.d_model) as usize, 1.0)).collect();
            let refs: Vec<&[f32]> = views.iter().map(Vec::as_slice).collect();
            let got = resampler.encode_views(&refs, local);
            assert_eq!(got.len(), batch);
            for (i, view) in views.iter().enumerate() {
                let want = resampler.resample_view(view, local).projected;
                let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                assert!(scale > 1e-3, "the comparison needs a signal, got scale {scale}");
                let err = max_abs(&got[i], &want);
                assert!(err <= 1e-4 * scale.max(1.0), "view {i} of {batch} (local={local}): max abs error {err} against scale {scale}");
            }
        }
    }
}
