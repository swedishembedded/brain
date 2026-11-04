// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-row L2-normalize + learnable per-dim scale (GenieRedux QK-norm). Forward
//! vs an exact host reference; input (dx) and scale (dg) gradients FD-checked.
use gpu_core::Gpu;

const K: [(&str, &str); 3] = [
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
    ("l2norm_scale_dg", kernels::L2NORM_SCALE_DG),
];

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}

const N: usize = 12; // tokens*heads
const D: usize = 8;  // head_dim
const EPS: f32 = 1e-6;

fn host_fwd(x: &[f32], g: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; N*D];
    for n in 0..N {
        let s: f32 = (0..D).map(|k| x[n*D+k]*x[n*D+k]).sum();
        let r = 1.0/(s+EPS).sqrt();
        for d in 0..D { y[n*D+d] = x[n*D+d]*r*g[d]; }
    }
    y
}

#[test]
fn l2norm_scale_forward_and_grads() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let gpu = Gpu::new_cpu(&K);
    let x = rand(1, N*D);
    let g = rand(2, D).iter().map(|v| v+1.5).collect::<Vec<_>>(); // keep g away from 0
    let dy = rand(3, N*D);
    let ep = [N as u32, D as u32, EPS.to_bits()];
    let wf = |b: &gpu_core::DeviceBuffer, d: &[f32]| gpu.write(b, &d.iter().map(|v| v.to_bits()).collect::<Vec<_>>());

    let xb = gpu.storage_init("x", &x);
    let gb = gpu.storage_init("g", &g);
    let yb = gpu.storage((N*D) as u64);
    let dyb = gpu.storage_init("dy", &dy);
    let dxb = gpu.storage((N*D) as u64);
    let dgb = gpu.storage(D as u64);

    gpu.submit(&[], &[gpu.step(0, &[&xb,&gb,&yb], &ep, (N*D) as u32)]);
    let y = gpu.read(&yb, N*D);
    let want = host_fwd(&x,&g);
    let fmax = y.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(fmax < 1e-5, "l2norm_scale fwd max abs {fmax}");

    gpu.submit(&[], &[
        gpu.step(1, &[&xb,&gb,&dyb,&dxb], &ep, (N*D) as u32),
        gpu.step(2, &[&xb,&dyb,&dgb], &ep, D as u32),
    ]);
    let dxg = gpu.read(&dxb, N*D);
    let dgg = gpu.read(&dgb, D);

    let loss = |x: &[f32], g: &[f32]| -> f32 { host_fwd(x,g).iter().zip(&dy).map(|(a,b)| a*b).sum() };
    let eps = 1e-3f32;
    // dx
    let dir = rand(4, N*D);
    let ax: f32 = dxg.iter().zip(&dir).map(|(a,b)| a*b).sum();
    let xp: Vec<f32> = x.iter().zip(&dir).map(|(v,d)| v+eps*d).collect();
    let xm: Vec<f32> = x.iter().zip(&dir).map(|(v,d)| v-eps*d).collect();
    let nx = (loss(&xp,&g)-loss(&xm,&g))/(2.0*eps);
    assert!((ax-nx).abs() < 4e-3 + 8e-2*ax.abs().max(nx.abs()), "dx: {ax} vs {nx}");
    // dg
    let dirg = rand(5, D);
    let ag: f32 = dgg.iter().zip(&dirg).map(|(a,b)| a*b).sum();
    let gp: Vec<f32> = g.iter().zip(&dirg).map(|(v,d)| v+eps*d).collect();
    let gm: Vec<f32> = g.iter().zip(&dirg).map(|(v,d)| v-eps*d).collect();
    let ng = (loss(&x,&gp)-loss(&x,&gm))/(2.0*eps);
    assert!((ag-ng).abs() < 4e-3 + 8e-2*ag.abs().max(ng.abs()), "dg: {ag} vs {ng}");
    let _ = wf;
}
