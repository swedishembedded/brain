// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rotating a Qwen3 decoder's residual stream into an orthonormal basis is
//! exact when every direction is kept: the norm gains fold into the linears
//! that read the stream, the rotation into every map on it, and the rotated
//! model computes the same logits while its residual stream at every point
//! is the rotated original.

use qwen3::slice::Slicing;
use qwen3::{Qwen, QwenConfig, Shard};

#[test]
fn full_width_rotation_preserves_logits_and_rotates_the_residual_stream() {
    // Wider than a read block (512 rows), so the linears writing the stream
    // are accumulated over several blocks.
    let cfg = QwenConfig { d_model: 640, ..QwenConfig::tiny() };
    let (d, layers) = (cfg.d_model as usize, cfg.n_layers as usize);
    let mut w = qwen3::init_weights(&cfg, 5);
    // Gains away from one, so folding them is exercised.
    let g = model::hostmath::gaussian(4 * d * (layers + 1), 11);
    let mut draws = g.iter();
    for (name, v) in w.iter_mut() {
        if name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "norm.weight" {
            v.iter_mut().for_each(|x| *x = 1.0 + 0.3 * draws.next().unwrap());
        }
    }
    // A random orthonormal basis: the eigenvectors of a random symmetric matrix.
    let r = model::hostmath::gaussian(d * d, 13);
    let sym: Vec<f64> = (0..d * d).map(|i| (r[i] + r[(i % d) * d + i / d]) as f64).collect();
    let basis: Vec<f32> = model::eigen::symmetric_eigen(&sym, d).1.into_iter().flatten().map(|v| v as f32).collect();

    let tokens: Vec<u32> = (0..cfg.block_size).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let original = Qwen::new_shard(cfg.clone(), 1, cfg.block_size, &w, false, Shard::whole(layers));
    let want = original.logits_all(&tokens);
    let sliced = Slicing::new(&cfg, &w, &basis, d).unwrap();
    assert!(!sliced.config().tie_embeddings, "the folded final-norm gain unties the head");
    let rotated = Qwen::new_shard(sliced.config().clone(), 1, cfg.block_size, &sliced, false, Shard::whole(layers));
    let got = rotated.logits_all(&tokens);
    let err = want.iter().zip(&got).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(err < 1e-4, "rotated logits differ by {err:e}");

    for point in 0..=2 * layers {
        let x = original.read_residual(point);
        let y = rotated.read_residual(point);
        let rotated_x = model::hostmath::linear_rows(&x, &basis, tokens.len(), d, d);
        let err = rotated_x.iter().zip(&y).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-4, "residual point {point} is not the rotated original (error {err:e})");
    }
}
