// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Render an autostereogram from a real depth map, for visual confirmation.
//! Env-gated: STEREO_PPM (input image), STEREO_OUT (output ppm), ZIPDEPTH_NPU_PTH.
use std::collections::HashMap;
use depth::{autostereogram, import, Predictor, StereoOpts, ZipConfig};
use depth::viz::Bounds;
use gpu_core::Gpu;
use paramstore::ParamStore;

fn read_ppm(p:&str)->(Vec<f32>,u32,u32){let d=std::fs::read(p).unwrap();let mut i=2;let mut n=vec![];while n.len()<3{while d[i].is_ascii_whitespace(){i+=1;}let s=i;while !d[i].is_ascii_whitespace(){i+=1;}n.push(std::str::from_utf8(&d[s..i]).unwrap().parse::<u32>().unwrap());}i+=1;(d[i..].iter().map(|&b|b as f32/255.0).collect(),n[0],n[1])}

#[test]
fn render() {
    let (Ok(ppm),Ok(out),Ok(pth))=(std::env::var("STEREO_PPM"),std::env::var("STEREO_OUT"),std::env::var("ZIPDEPTH_NPU_PTH")) else { eprintln!("SKIP"); return; };
    let (hwc,w,h)=read_ppm(&ppm);
    let cfg=ZipConfig{upsample_unfold:false,..ZipConfig::base()};
    let gpu=Gpu::new_cpu(depth::net::PIPELINES);
    let init:HashMap<String,Vec<f32>>=import::load(&pth,&cfg).unwrap();
    let params:Vec<(String,usize)>=cfg.param_list().into_iter().map(|(n,s)|(n,s.iter().product())).collect();
    let ps=ParamStore::new(&gpu,params,&init);
    let pred=Predictor::new(&gpu,cfg,ps);
    let depth=pred.predict(&hwc,w,h);
    let b=Bounds::from_percentiles(&depth,0.02,0.98);
    let img=autostereogram(&depth,w,h,b,&StereoOpts::for_width(w));
    let mut bytes=format!("P6\n{w} {h}\n255\n").into_bytes(); bytes.extend_from_slice(&img);
    std::fs::write(&out,&bytes).unwrap();
    eprintln!("wrote {out} {w}x{h}");
}
