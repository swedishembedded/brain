// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P3: brain's ZipDepth must match the reference PyTorch at a RECTANGULAR input —
//! the aspect-preserving shape the reference actually feeds (not a padded square).
//! Env-gated on a reference dump (BRAIN_REF_RECT) + the checkpoint.
use std::collections::HashMap;
use zipdepth::{import, ZipConfig, ZipDepth};
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::Ctx;

#[test]
fn matches_reference_at_384x512() {
    let (Ok(pth), Ok(refbin)) = (std::env::var("ZIPDEPTH_NPU_PTH"), std::env::var("BRAIN_REF_RECT")) else {
        eprintln!("SKIP"); return;
    };
    let (h, w) = (384usize, 512usize);
    let cfg = ZipConfig { upsample_unfold: false, ..ZipConfig::base() };
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let init: HashMap<String, Vec<f32>> = import::load(&pth, &cfg).unwrap();
    let params: Vec<(String,usize)> = cfg.param_list().into_iter().map(|(n,s)|(n,s.iter().product())).collect();
    let ps = ParamStore::new(&gpu, params, &init);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = ZipDepth::build_hw(&ctx, cfg, 1, h as u32, w as u32, false);
    m.set_eval(true);

    let mut chw = vec![0f32; 3*h*w];
    let hwc: Vec<f32> = (0..(h*w*3)).map(|i| ((i*37 % 251) as f32)/251.0).collect();
    for y in 0..h { for x in 0..w { for c in 0..3 {
        chw[c*h*w + y*w + x] = hwc[(y*w+x)*3 + c];
    }}}
    let xb = gpu.storage_init("x", &chw);
    m.forward(&ctx, &ps, &xb);
    let brn = gpu.read(m.out(), h*w);

    let refb = std::fs::read(&refbin).unwrap();
    let refd: Vec<f32> = refb.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    assert_eq!(refd.len(), brn.len());
    let dot: f32 = refd.iter().zip(&brn).map(|(a,b)|a*b).sum();
    let (na,nb): (f32,f32) = (refd.iter().map(|x|x*x).sum::<f32>().sqrt(), brn.iter().map(|x|x*x).sum::<f32>().sqrt());
    let cos = dot/(na*nb+1e-12);
    let maxd = refd.iter().zip(&brn).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
    eprintln!("cosine(reference, brain) @384x512 = {cos:.6}  max|Δ| = {maxd:.6}");
    assert!(cos > 0.9999, "brain must match the reference at rectangular sizes (cosine {cos:.6})");
}
