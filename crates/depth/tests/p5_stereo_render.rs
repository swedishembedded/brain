// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Render dots + textured autostereograms from a real depth map, for visual check.
//! Env-gated: STEREO_PPM, STEREO_OUT_DOTS, STEREO_OUT_TEX, ZIPDEPTH_NPU_PTH.
use std::collections::HashMap;
use depth::{autostereogram, autostereogram_textured, import, Predictor, StereoOpts, ZipConfig};
use depth::viz::Bounds;
use gpu_core::Gpu;
use paramstore::ParamStore;

fn write_ppm(p: &str, rgb: &[u8], w: u32, h: u32) {
    imaging::save_ppm(p, &imaging::Rgb8::new(w, h, rgb.to_vec()).unwrap()).unwrap();
}

#[test]
fn render() {
    let (Ok(ppm),Ok(pth))=(std::env::var("STEREO_PPM"),std::env::var("ZIPDEPTH_NPU_PTH")) else { eprintln!("SKIP"); return; };
    let img = imaging::load(&ppm).unwrap();
    let (hwc, w, h) = (img.to_hwc_unit(), img.w, img.h);
    let cfg=ZipConfig{upsample_unfold:false,..ZipConfig::base()};
    let gpu=Gpu::new_cpu(depth::net::PIPELINES);
    let init:HashMap<String,Vec<f32>>=import::load(&pth,&cfg).unwrap();
    let params:Vec<(String,usize)>=cfg.param_list().into_iter().map(|(n,s)|(n,s.iter().product())).collect();
    let ps=ParamStore::new(&gpu,params,&init);
    let pred=Predictor::new(&gpu,cfg,ps);
    let depth=pred.predict(&hwc,w,h);
    let b=Bounds::from_percentiles(&depth,0.02,0.98);
    let opts=StereoOpts::for_width(w);
    if let Ok(o)=std::env::var("STEREO_OUT_DOTS"){ write_ppm(&o,&autostereogram(&depth,w,h,b,&opts),w,h); }
    if let Ok(o)=std::env::var("STEREO_OUT_TEX"){
        let rgb8:Vec<u8>=hwc.iter().map(|&v|(v.clamp(0.0,1.0)*255.0).round() as u8).collect();
        write_ppm(&o,&autostereogram_textured(&depth,w,h,b,&opts,&rgb8),w,h);
    }
    eprintln!("rendered {w}x{h}");
}
