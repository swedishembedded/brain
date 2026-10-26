// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in fused attention for ViT encoders with relative-position bias then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec of the fused attention: an inference build of the tower attends with `flash_attn_cross_relpos` (one
//! dispatch over every window of a block, the decomposed relative-position bias read from tables), and its
//! output equals the training build's materialized chain (scores, rel-pos add, softmax, apply per window and
//! query chunk) on the same weights and image - the chain the gradient checks and the llama.cpp parity pin.
//!
//! The tiny config makes every axis a different number (grid 13x7, window 4x3, so a padded window grid and
//! `kh != kw`), which is what exposes a swapped height/width in the bias; a second config adds a table
//! resampled to another grid (`SamViTConfig::at_grid`), the tile tower's shape.

use sam1::init::init_dense;
use sam1::{SamEncoder, SamViTConfig, PIPELINES};

fn run(cfg: &SamViTConfig, weights: &std::collections::HashMap<String, Vec<f32>>, train: bool, image: &[f32]) -> Vec<f32> {
    let enc = SamEncoder::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), weights, 0, train);
    enc.write_image(image);
    enc.run();
    enc.gpu.read(enc.output(), enc.out_len())
}

fn compare(label: &str, cfg: &SamViTConfig) {
    let weights = init_dense(cfg, 5);
    let mut rng = data::rng::Lcg::new(9);
    let image = rng.vec_scaled((3 * cfg.image_h() * cfg.image_w()) as usize, 1.0);
    let (fused, chain) = (run(cfg, &weights, false, &image), run(cfg, &weights, true, &image));
    let scale = chain.iter().fold(0f32, |m, x| m.max(x.abs()));
    let err = fused.iter().zip(&chain).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(scale > 1e-3, "{label}: the comparison needs a signal, got scale {scale}");
    eprintln!("{label}: max abs difference {err:.3e} against scale {scale:.3}");
    assert!(err <= 1e-4 * scale.max(1.0), "{label}: the fused attention differs from the materialized chain by {err} against scale {scale}");
}

#[test]
fn the_fused_attention_equals_the_materialized_chain() {
    let tiny = SamViTConfig::tiny();
    compare("tiny", &tiny);
    compare("tiny at another grid", &tiny.at_grid(tiny.grid_h - 3, tiny.grid_w + 2));
}
