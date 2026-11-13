// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! STTransformer stack: verify the block loop + final LayerNorm wire correctly
//! by comparing a 2-layer stack against manually chaining the (already
//! host-verified) stblock_forward twice and applying the same final norm.
use gpu_core::Gpu;
use crate::tiny::{self, rand};
use genieredux::{kernel_sources, stblock_forward, sttransformer_forward, StTransformerWeights};

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
#[test]
fn sttransformer_stacks_blocks_and_norms() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (b, t, h, w, dim, heads, hd) = (1usize, 3usize, 2usize, 2usize, 16usize, 2usize, 8usize);
    let gpu = Gpu::new_cpu(&kernel_sources());
    let x = rand(1, b*t*h*w*dim);
    let sb = rand(2, heads*(h*w)*(h*w));
    let tb = rand(3, heads*t*t);
    let norm_g: Vec<f32> = rand(4, dim).iter().map(|v| v+1.0).collect();
    let wts = StTransformerWeights {
        layers: vec![tiny::block(dim, heads, hd, 1000), tiny::block(dim, heads, hd, 2000)],
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
