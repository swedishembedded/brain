// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cosine-similarity VQ forward vs a host reference: project_in -> L2-normalize
//! -> argmax cosine against the normalized codebook -> gather codebook[idx] ->
//! project_out. Checks the quantized output and the chosen indices.
use gpu_core::Gpu;
use wm_genie::{kernel_sources, vq_quantize, VqWeights};

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}
fn linb(x: &[f32], w: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m*n];
    for i in 0..m { for j in 0..n { let mut a=b[j]; for t in 0..k { a += x[i*k+t]*w[j*k+t]; } o[i*n+j]=a; }}
    o
}
fn l2(v: &[f32]) -> Vec<f32> { let s: f32 = v.iter().map(|x| x*x).sum(); let r=1.0/s.sqrt(); v.iter().map(|x| x*r).collect() }

const N: usize = 20; const DIM: usize = 16; const CD: usize = 8; const K: usize = 32;

#[test]
fn vq_quantize_matches_host() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let gpu = Gpu::new_cpu(&kernel_sources());
    let x = rand(1, N*DIM);
    let w = VqWeights {
        project_in_w: rand(2, CD*DIM), project_in_b: rand(3, CD),
        codebook: rand(4, K*CD),
        project_out_w: rand(5, DIM*CD), project_out_b: rand(6, DIM),
    };
    let (got, idx) = vq_quantize(&gpu, &x, &w, N as u32, DIM as u32, CD as u32, K as u32);

    // host reference: normalize the input only, argmax dot against the RAW
    // codebook (matching the reference cosine codebook), gather raw codebook.
    let z = linb(&x, &w.project_in_w, &w.project_in_b, N, DIM, CD);
    let mut want_idx = vec![0u32; N];
    let mut q = vec![0.0f32; N*CD];
    for n in 0..N {
        let zn = l2(&z[n*CD..(n+1)*CD]);
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for j in 0..K {
            let dot: f32 = zn.iter().zip(&w.codebook[j*CD..(j+1)*CD]).map(|(a,b)| a*b).sum();
            if dot > best { best = dot; bi = j; }
        }
        want_idx[n] = bi as u32;
        q[n*CD..(n+1)*CD].copy_from_slice(&w.codebook[bi*CD..(bi+1)*CD]); // raw codebook
    }
    let want = linb(&q, &w.project_out_w, &w.project_out_b, N, CD, DIM);

    assert_eq!(idx, want_idx, "chosen codebook indices differ");
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-4, "vq quantize max abs {max}");
}
