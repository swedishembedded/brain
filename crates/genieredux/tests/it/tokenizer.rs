// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full tokenizer forward assembly (patch-embed -> encoder -> cosine VQ ->
//! decoder -> to_pixels). The individual stages are host-verified elsewhere;
//! this checks the end-to-end wiring: output shape, finiteness, determinism,
//! and that the VQ produces in-range codebook indices.
use gpu_core::Gpu;
use crate::tiny;
use genieredux::{kernel_sources, tokenizer_forward};

#[test]
fn tokenizer_forward_end_to_end() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (b, c, f, hh, ww, p) = (1usize, 3usize, 3usize, 8usize, 8usize, 4usize);
    let (dim, heads, hd) = (16usize, 2usize, 8usize);
    let (cd, k) = (8usize, 16usize);
    let hw = (hh/p)*(ww/p);
    let w = tiny::tokenizer();
    let gpu = Gpu::new_cpu(&kernel_sources());
    let video = tiny::rand(1, b*c*f*hh*ww);
    let (recon, idx) = tokenizer_forward(&gpu, &video, &w,
        b as u32, c as u32, f as u32, hh as u32, ww as u32, p as u32, dim as u32, heads as u32, hd as u32, cd as u32, k as u32);

    assert_eq!(recon.len(), b*c*f*hh*ww, "recon shape");
    assert_eq!(idx.len(), b*f*hw, "index count = tokens");
    assert!(recon.iter().all(|v| v.is_finite()), "non-finite reconstruction");
    assert!(idx.iter().all(|&i| (i as usize) < k), "codebook index out of range");

    // deterministic
    let (recon2, idx2) = tokenizer_forward(&gpu, &video, &w,
        b as u32, c as u32, f as u32, hh as u32, ww as u32, p as u32, dim as u32, heads as u32, hd as u32, cd as u32, k as u32);
    assert_eq!(idx, idx2);
    let max = recon.iter().zip(&recon2).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert_eq!(max, 0.0, "non-deterministic reconstruction");
}
