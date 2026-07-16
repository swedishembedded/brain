// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full tokenizer forward assembly (patch-embed -> encoder -> cosine VQ ->
//! decoder -> to_pixels). The individual stages are host-verified elsewhere;
//! this checks the end-to-end wiring: output shape, finiteness, determinism,
//! and that the VQ produces in-range codebook indices.
use gpu_core::Gpu;
use wm_genie::bias::CpbLayer;
use wm_genie::{
    kernel_sources, tokenizer_forward, AttnWeights, FfWeights, PatchEmbedWeights, PegWeights,
    StBlockWeights, StTransformerWeights, ToPixelsWeights, TokenizerWeights, VqWeights, ff_inner,
};

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}
fn mk_attn(dim: usize, inner: usize, hd: usize, s: u64) -> AttnWeights {
    AttnWeights { norm_gamma: rand(s, dim).iter().map(|v| v+1.0).collect(),
        to_q: rand(s+1, inner*dim), to_k: rand(s+2, inner*dim), to_v: rand(s+3, inner*dim),
        to_out: rand(s+4, dim*inner),
        q_scale: rand(s+5, hd).iter().map(|v| v+1.0).collect(),
        k_scale: rand(s+6, hd).iter().map(|v| v+1.0).collect() }
}
fn mk_ff(dim: usize, inner: usize, s: u64) -> FfWeights {
    FfWeights { norm_gamma: rand(s, dim).iter().map(|v| v+1.0).collect(),
        norm_beta: rand(s+7, dim),
        w_x: rand(s+1, inner*dim), w_gate: rand(s+2, inner*dim), w_out: rand(s+3, dim*inner) }
}
fn mk_peg(dim: usize, s: u64) -> PegWeights {
    PegWeights { dsconv: rand(s, dim*27).iter().map(|v| v*0.3).collect(), bias: rand(s+1, dim) }
}
fn mk_block(dim: usize, inner: usize, hd: usize, ffi: usize, s: u64) -> StBlockWeights {
    StBlockWeights { spatial_peg: mk_peg(dim, s), spatial_attn: mk_attn(dim, inner, hd, s+10), spatial_ff: mk_ff(dim, ffi, s+20),
        temporal_peg: mk_peg(dim, s+30), temporal_attn: mk_attn(dim, inner, hd, s+40), temporal_ff: mk_ff(dim, ffi, s+50) }
}
fn mk_stack(dim: usize, inner: usize, hd: usize, ffi: usize, n: usize, s: u64) -> StTransformerWeights {
    StTransformerWeights { layers: (0..n).map(|i| mk_block(dim, inner, hd, ffi, s + i as u64 * 100)).collect(),
        norm_out_gamma: rand(s+9000, dim).iter().map(|v| v+1.0).collect() }
}
fn mk_patch(pf: usize, dim: usize, s: u64) -> PatchEmbedWeights {
    PatchEmbedWeights { ln1_g: rand(s, pf).iter().map(|v| v+1.0).collect(), ln1_b: rand(s+1, pf),
        lin_w: rand(s+2, dim*pf), lin_b: rand(s+3, dim),
        ln2_g: rand(s+4, dim).iter().map(|v| v+1.0).collect(), ln2_b: rand(s+5, dim) }
}

#[test]
fn tokenizer_forward_end_to_end() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (b, c, f, hh, ww, p) = (1usize, 3usize, 3usize, 8usize, 8usize, 4usize);
    let (dim, heads, hd) = (16usize, 2usize, 8usize);
    let (cd, k) = (8usize, 16usize);
    let inner = heads*hd; let ffi = ff_inner(dim as u32) as usize;
    let pf = c*p*p; // 48
    let hw = (hh/p)*(ww/p);

    let w = TokenizerWeights {
        patch_first: mk_patch(pf, dim, 10), patch_rest: mk_patch(pf, dim, 20),
        encoder: mk_stack(dim, inner, hd, ffi, 2, 1000),
        vq: VqWeights { project_in_w: rand(30, cd*dim), project_in_b: rand(31, cd),
            codebook: rand(32, k*cd), project_out_w: rand(33, dim*cd), project_out_b: rand(34, dim) },
        decoder: mk_stack(dim, inner, hd, ffi, 2, 5000),
        to_pixels_first: ToPixelsWeights { lin_w: rand(40, pf*dim), lin_b: rand(41, pf) },
        to_pixels_rest: ToPixelsWeights { lin_w: rand(42, pf*dim), lin_b: rand(43, pf) },
        cpb_net: vec![
            CpbLayer { w: rand(50, dim*2), b: rand(51, dim), in_dim: 2, out_dim: dim },
            CpbLayer { w: rand(52, dim*dim), b: rand(53, dim), in_dim: dim, out_dim: dim },
            CpbLayer { w: rand(54, heads*dim), b: rand(55, heads), in_dim: dim, out_dim: heads },
        ],
    };
    let gpu = Gpu::new_cpu(&kernel_sources());
    let video = rand(1, b*c*f*hh*ww);
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
