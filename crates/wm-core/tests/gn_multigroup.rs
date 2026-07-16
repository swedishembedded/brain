// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! gn_stats (serial) vs gn_part+gn_stats2 (parallel) must agree at multi-group
//! with large channels-per-group — the trainer uses the parallel pair, but the
//! tested backward was only ever paired with gn_stats at cpg=2. Regression for
//! the multi-group training-grad explosion.
use gpu_core::{f, Gpu};

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        (((z^(z>>31))>>40) as f32/(1u64<<24) as f32 - 0.5)*4.0 }).collect()
}

#[test]
fn gn_stats_serial_vs_parallel_agree_multigroup() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (n, c, h, w, g, gp) = (1u32, 64u32, 8u32, 8u32, 2u32, 64u32);
    let gpu = Gpu::new_cpu(&[
        ("gn_stats", kernels::GN_STATS),
        ("gn_part", kernels::GN_PART),
        ("gn_stats2", kernels::GN_STATS2),
    ]);
    let x = rand(9, (n*c*h*w) as usize);
    let xb = gpu.storage_init("x", &x);
    let eps = 1e-5f32;
    let s_serial = gpu.storage((2*n*g) as u64);
    gpu.submit(&[], &[gpu.step(0, &[&xb, &s_serial], &[n,c,h,w,g,f(eps)], n*g)]);
    let part = gpu.storage((2*n*g*gp) as u64);
    let s_par = gpu.storage((2*n*g) as u64);
    gpu.submit(&[], &[
        gpu.step(1, &[&xb, &part], &[n,c,h,w,g,gp], n*g*gp),
        gpu.step(2, &[&part, &s_par], &[n,c,h,w,g,gp,f(eps)], n*g),
    ]);
    let a = gpu.read(&s_serial, (2*n*g) as usize);
    let b = gpu.read(&s_par, (2*n*g) as usize);
    let max = a.iter().zip(&b).map(|(x,y)|(x-y).abs()).fold(0.0f32, f32::max);
    assert!(max < 1e-3, "gn_stats vs gn_stats2 diverge at C={c} G={g}: {a:?} vs {b:?} (max {max})");
}

/// Full GN fwd (gn_part/gn_stats2/gn_apply) + bwd (scale_chan/gn_dsum/gn_dx)
/// FD-checked at C=64,G=2 — the trainer's exact sequence and shape.
#[test]
fn gn_multigroup_backward_fd() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (n, c, h, w, g, gp) = (1u32, 64u32, 8u32, 8u32, 2u32, 64u32);
    let ne = (n*c*h*w) as usize;
    let gpu = Gpu::new_cpu(&[
        ("gn_part", kernels::GN_PART), ("gn_stats2", kernels::GN_STATS2),
        ("gn_apply", kernels::GN_APPLY), ("scale_chan", kernels::SCALE_CHAN),
        ("gn_dsum", kernels::GN_DSUM), ("gn_dx", kernels::GN_DX),
    ]);
    let x = rand(3, ne);
    let mut gb = rand(4, (2*c) as usize); for v in gb[..c as usize].iter_mut() { *v = *v*0.3+1.0; }
    let dy = rand(5, ne); // upstream grad
    let eps = 1e-5f32;
    let wf = |b:&gpu_core::DeviceBuffer, d:&[f32]| gpu.write(b,&d.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
    let xb=gpu.storage_init("x",&x); let gbb=gpu.storage_init("gb",&gb);
    let part=gpu.storage((2*n*g*gp) as u64); let stats=gpu.storage((2*n*g) as u64);
    let y=gpu.storage(ne as u64); let dyb=gpu.storage_init("dy",&dy);
    let dyg=gpu.storage(ne as u64); let sums=gpu.storage((4*n*g) as u64); let dx=gpu.storage(ne as u64);
    let fwd = |gpu:&Gpu| { gpu.submit(&[],&[
        gpu.step(0,&[&xb,&part],&[n,c,h,w,g,gp],n*g*gp),
        gpu.step(1,&[&part,&stats],&[n,c,h,w,g,gp,f(eps)],n*g),
        gpu.step(2,&[&xb,&stats,&gbb,&y],&[n,c,h,w,g],ne as u32),
    ]); };
    // analytic dx = d(sum y*dy)/dx
    fwd(&gpu);
    gpu.submit(&[],&[
        gpu.step(3,&[&dyb,&gbb,&dyg],&[ne as u32,c,h*w],ne as u32),
        gpu.step(4,&[&xb,&dyg,&stats,&sums],&[n,c,h,w,g],n*g),
        gpu.step(5,&[&xb,&dyg,&sums,&dx],&[n,c,h,w,g],ne as u32),
    ]);
    let dxg = gpu.read(&dx, ne);
    // FD on a random direction of x
    let dir = rand(6, ne);
    let analytic: f32 = dxg.iter().zip(&dir).map(|(a,b)|a*b).sum();
    let epsd=1e-3f32;
    let loss = |gpu:&Gpu| { fwd(gpu); gpu.read(&y,ne).iter().zip(&dy).map(|(a,b)|a*b).sum::<f32>() };
    let xp:Vec<f32>=x.iter().zip(&dir).map(|(v,d)|v+epsd*d).collect(); wf(&xb,&xp); let lp=loss(&gpu);
    let xm:Vec<f32>=x.iter().zip(&dir).map(|(v,d)|v-epsd*d).collect(); wf(&xb,&xm); let lm=loss(&gpu);
    let numeric=(lp-lm)/(2.0*epsd);
    let _=&mut gb;
    assert!((analytic-numeric).abs() < 4e-3+8e-2*analytic.abs().max(numeric.abs()),
        "GN C=64 G=2 dx: analytic {analytic} vs FD {numeric}");
}
