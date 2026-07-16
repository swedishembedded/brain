// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! STTransformer stack: verify the block loop + final LayerNorm wire correctly
//! by comparing a 2-layer stack against manually chaining the (already
//! host-verified) stblock_forward twice and applying the same final norm.
use gpu_core::Gpu;
use wm_genie::{
    ff_inner, kernel_sources, stblock_forward, sttransformer_forward, AttnWeights, FfWeights,
    PegWeights, StBlockWeights, StTransformerWeights,
};

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}
fn h_layernorm(x: &[f32], g: &[f32], dim: usize) -> Vec<f32> {
    let rows = x.len()/dim; let mut o = vec![0.0f32; x.len()];
    for r in 0..rows { let s=&x[r*dim..(r+1)*dim];
        let m: f32 = s.iter().sum::<f32>()/dim as f32;
        let va: f32 = s.iter().map(|v| (v-m)*(v-m)).sum::<f32>()/dim as f32;
        let inv = 1.0/(va+1e-5).sqrt();
        for c in 0..dim { o[r*dim+c] = (s[c]-m)*inv*g[c]; }
    }
    o
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
        w_x: rand(s+1, inner*dim), w_gate: rand(s+2, inner*dim), w_out: rand(s+3, dim*inner) }
}
fn mk_peg(dim: usize, s: u64) -> PegWeights {
    PegWeights { dsconv: rand(s, dim*27).iter().map(|v| v*0.3).collect(), bias: rand(s+1, dim) }
}
fn mk_block(dim: usize, inner: usize, hd: usize, ffi: usize, s: u64) -> StBlockWeights {
    StBlockWeights {
        spatial_peg: mk_peg(dim, s), spatial_attn: mk_attn(dim, inner, hd, s+10), spatial_ff: mk_ff(dim, ffi, s+20),
        temporal_peg: mk_peg(dim, s+30), temporal_attn: mk_attn(dim, inner, hd, s+40), temporal_ff: mk_ff(dim, ffi, s+50),
    }
}

#[test]
fn sttransformer_stacks_blocks_and_norms() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (b, t, h, w, dim, heads, hd) = (1usize, 3usize, 2usize, 2usize, 16usize, 2usize, 8usize);
    let (inner, ffi) = (heads*hd, ff_inner(dim as u32) as usize);
    let gpu = Gpu::new_cpu(&kernel_sources());
    let x = rand(1, b*t*h*w*dim);
    let sb = rand(2, heads*(h*w)*(h*w));
    let tb = rand(3, heads*t*t);
    let norm_g: Vec<f32> = rand(4, dim).iter().map(|v| v+1.0).collect();
    let wts = StTransformerWeights {
        layers: vec![mk_block(dim, inner, hd, ffi, 1000), mk_block(dim, inner, hd, ffi, 2000)],
        norm_out_gamma: norm_g.clone(),
    };
    let (bu,tu,hu,wu) = (b as u32, t as u32, h as u32, w as u32);
    let got = sttransformer_forward(&gpu, &x, bu, tu, hu, wu, dim as u32, heads as u32, hd as u32, &wts, &sb, &tb, true, false);

    // reference: two device STBlocks then the same LayerNorm.
    let a = stblock_forward(&gpu, &x, bu, tu, hu, wu, dim as u32, heads as u32, hd as u32, &wts.layers[0], &sb, &tb, true, false);
    let bb = stblock_forward(&gpu, &a, bu, tu, hu, wu, dim as u32, heads as u32, hd as u32, &wts.layers[1], &sb, &tb, true, false);
    let want = h_layernorm(&bb, &norm_g, dim);

    assert_eq!(got.len(), b*t*h*w*dim);
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-5, "sttransformer vs manual stack max abs {max}");
}
