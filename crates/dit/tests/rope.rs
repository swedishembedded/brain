// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-axis RoPE parity vs Z-Image's `RopeEmbedder.precompute_freqs_cis`.
//!
//! Goldens are closed-form values (`freqs[k]=theta^(-2k/d)`, `angle=pos·freqs`,
//! `(cos,sin)`) for Z-Image's config, verified by hand. No torch needed.

use dit::rope::{tables_for_ids, RopeConfig};

fn zimage_cfg() -> RopeConfig {
    RopeConfig { axes_dims: vec![32, 48, 48], axes_lens: vec![1024, 512, 512], theta: 256.0 }
}

fn close(a: f32, b: f32, what: &str) {
    assert!((a - b).abs() <= 2e-6, "{what}: {a} != {b}");
}

#[test]
fn head_dim_and_half() {
    let c = zimage_cfg();
    assert_eq!(c.head_dim(), 128);
    assert_eq!(c.half(), 64);
}

#[test]
fn zimage_freqs_match_closed_form() {
    let cfg = zimage_cfg();
    // Two tokens: position (0,0,0) and (1,0,0).
    let ids: Vec<u32> = vec![0, 0, 0, 1, 0, 0];
    let t = tables_for_ids(&cfg, &ids, 3);
    assert_eq!(t.seq, 2);
    assert_eq!(t.half, 64);

    // Token 0 = origin: every pair is (cos 0, sin 0) = (1, 0).
    for j in 0..64 {
        close(t.cos[j], 1.0, "tok0.cos");
        close(t.sin[j], 0.0, "tok0.sin");
    }

    // Token 1 = (pos 1 on axis0, origin on axes 1&2).
    // axis0 d=32: freqs[k] = 256^(-2k/32) = 2^(-k/2); angle = 1·freqs[k].
    let base = 64; // token 1 offset into cos/sin
    close(t.cos[base], (1.0f64).cos() as f32, "tok1.pair0.cos"); // freqs[0]=1
    close(t.sin[base], (1.0f64).sin() as f32, "tok1.pair0.sin");
    let f1 = 2f64.powf(-0.5); // freqs[1]
    close(t.cos[base + 1], f1.cos() as f32, "tok1.pair1.cos");
    close(t.sin[base + 1], f1.sin() as f32, "tok1.pair1.sin");
    close(t.cos[base + 2], (0.5f64).cos() as f32, "tok1.pair2.cos"); // freqs[2]=2^-1
    close(t.sin[base + 2], (0.5f64).sin() as f32, "tok1.pair2.sin");

    // Pairs 16..63 come from axes 1&2 at origin → (1,0).
    for j in 16..64 {
        close(t.cos[base + j], 1.0, "tok1.axis12.cos");
        close(t.sin[base + j], 0.0, "tok1.axis12.sin");
    }
}
