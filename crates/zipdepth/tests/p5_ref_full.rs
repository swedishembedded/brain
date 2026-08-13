// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P5: brain's FULL predict pipeline (aspect-preserving resize -> model -> resize
//! back) must match the reference PyTorch's full pipeline on a real image.
//! Env-gated: NATIVE_PPM (input), REF_FULL_BIN (reference depth at native res),
//! ZIPDEPTH_NPU_PTH.
use std::collections::HashMap;
use zipdepth::{import, Predictor, ZipConfig};
use gpu_core::Gpu;
use paramstore::ParamStore;

#[test]
fn full_pipeline_matches_reference() {
    let (Ok(ppm), Ok(refb), Ok(pth)) = (
        std::env::var("NATIVE_PPM"), std::env::var("REF_FULL_BIN"), std::env::var("ZIPDEPTH_NPU_PTH"),
    ) else { eprintln!("SKIP"); return; };

    let img = imaging::load(&ppm).unwrap();
    let (hwc, w, h) = (img.to_hwc_unit(), img.w, img.h);
    let cfg = ZipConfig { upsample_unfold: false, ..ZipConfig::base() };
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let init: HashMap<String, Vec<f32>> = import::load(&pth, &cfg).unwrap();
    let params: Vec<(String,usize)> = cfg.param_list().into_iter().map(|(n,s)|(n,s.iter().product())).collect();
    let ps = ParamStore::new(&gpu, params, &init);
    let pred = Predictor::new(&gpu, cfg, ps);
    let brn = pred.predict(&hwc, w, h);

    let rb = std::fs::read(&refb).unwrap();
    let refd: Vec<f32> = rb.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    assert_eq!(refd.len(), brn.len(), "depth sizes differ");
    let dot: f32 = refd.iter().zip(&brn).map(|(a,b)|a*b).sum();
    let (na,nb): (f32,f32) = (refd.iter().map(|x|x*x).sum::<f32>().sqrt(), brn.iter().map(|x|x*x).sum::<f32>().sqrt());
    let cos = dot/(na*nb+1e-12);
    // normalized-depth agreement (scale-invariant, what the user perceives).
    let (rm, bm) = (refd.iter().sum::<f32>()/refd.len() as f32, brn.iter().sum::<f32>()/brn.len() as f32);
    let cov: f32 = refd.iter().zip(&brn).map(|(a,b)|(a-rm)*(b-bm)).sum::<f32>();
    let (rv,bv): (f32,f32) = (refd.iter().map(|a|(a-rm).powi(2)).sum::<f32>().sqrt(), brn.iter().map(|b|(b-bm).powi(2)).sum::<f32>().sqrt());
    let pearson = cov/(rv*bv+1e-12);
    eprintln!("full-pipeline cosine = {cos:.5}   pearson = {pearson:.5}   ref_mean={rm:.4} brain_mean={bm:.4}");
    assert!(cos > 0.99, "brain's pipeline must match the reference (cosine {cos:.5})");
    assert!(pearson > 0.99, "depth structure must match the reference (pearson {pearson:.5})");
}
