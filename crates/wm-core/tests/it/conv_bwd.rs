// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! conv2d_dx input-gradient FD check at conv_out's shape. The DIAMOND trainer
//! is grad-correct at cin=8 (tiny) but corrupts everything upstream of conv_out
//! at cin=64 (real) — this pins whether conv2d_dx is cin-sensitive.
use gpu_core::Gpu;

fn rnd(seed: u64, n: usize, s: f32) -> Vec<f32> {
    let mut z=seed; (0..n).map(|_|{z=z.wrapping_add(0x9E3779B97F4A7C15);let mut x=z;
        x=(x^(x>>30)).wrapping_mul(0xBF58476D1CE4E5B9);x=(x^(x>>27)).wrapping_mul(0x94D049BB133111EB);
        (((x^(x>>31))>>40) as f32/(1u64<<24) as f32-0.5)*2.0*s}).collect()
}

fn check_dx(cin: u32) {
    let (cout, k, h, w) = (3u32, 3u32, 8u32, 8u32);
    let (ho, wo) = (h, w); // stride 1 pad 1
    let gpu = Gpu::new_cpu(&[
        ("conv_bias_reg", kernels::CONV_BIAS_REG),
        ("conv2d_dx", kernels::CONV2D_DX),
    ]);
    let x = rnd(1, (cin*h*w) as usize, 1.0);
    let wt = rnd(2, (cout*cin*k*k) as usize, 0.2);
    let bias = vec![0.0f32; cout as usize];
    let dy = rnd(3, (cout*ho*wo) as usize, 1.0);
    let wf=|b:&gpu_core::DeviceBuffer,d:&[f32]|gpu.write(b,&d.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
    let xb=gpu.storage_init("x",&x); let wb=gpu.storage_init("w",&wt);
    let bb=gpu.storage_init("b",&bias); let yb=gpu.storage((cout*ho*wo) as u64);
    let dyb=gpu.storage_init("dy",&dy); let dxb=gpu.storage((cin*h*w) as u64);
    let dims=[1,cin,h,w,cout,k,1,1,ho,wo];
    let fwd=|gpu:&Gpu|{ let th=cout.div_ceil(8)*(ho*wo).div_ceil(4);
        gpu.submit(&[],&[gpu.step(0,&[&xb,&wb,&bb,&yb],&dims,th)]); };
    fwd(&gpu);
    gpu.submit(&[],&[gpu.step(1,&[&dyb,&wb,&dxb],&dims,cin*h*w)]);
    let dxg=gpu.read(&dxb,(cin*h*w) as usize);
    let dir=rnd(4,(cin*h*w) as usize,1.0);
    let analytic:f32=dxg.iter().zip(&dir).map(|(a,b)|a*b).sum();
    let eps=1e-3f32;
    let loss=|gpu:&Gpu|{fwd(gpu); gpu.read(&yb,(cout*ho*wo) as usize).iter().zip(&dy).map(|(a,b)|a*b).sum::<f32>()};
    let xp:Vec<f32>=x.iter().zip(&dir).map(|(v,d)|v+eps*d).collect(); wf(&xb,&xp); let lp=loss(&gpu);
    let xm:Vec<f32>=x.iter().zip(&dir).map(|(v,d)|v-eps*d).collect(); wf(&xb,&xm); let lm=loss(&gpu);
    let numeric=(lp-lm)/(2.0*eps);
    assert!((analytic-numeric).abs()<4e-3+8e-2*analytic.abs().max(numeric.abs()),
        "conv2d_dx cin={cin}: analytic {analytic} vs FD {numeric}");
}

#[test] fn conv2d_dx_cin8() { if std::env::var("MOE_SKIP_GPU_TESTS").is_ok(){return;} check_dx(8); }
#[test] fn conv2d_dx_cin64() { if std::env::var("MOE_SKIP_GPU_TESTS").is_ok(){return;} check_dx(64); }
