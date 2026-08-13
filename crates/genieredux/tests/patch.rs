// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Patch-embed and to_pixels (the video <-> token boundary) vs host references.
//! Patchify is 4x4x3=48 with feature order (c p1 p2); LN/Linear carry bias.
use gpu_core::Gpu;
use genieredux::{kernel_sources, patch_embed, to_pixels, PatchEmbedWeights, ToPixelsWeights};

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}
fn ln(x: &[f32], g: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let rows = x.len()/dim; let mut o = vec![0.0f32; x.len()];
    for r in 0..rows { let s=&x[r*dim..(r+1)*dim];
        let m: f32 = s.iter().sum::<f32>()/dim as f32;
        let va: f32 = s.iter().map(|v| (v-m)*(v-m)).sum::<f32>()/dim as f32;
        let inv = 1.0/(va+1e-5).sqrt();
        for c in 0..dim { o[r*dim+c] = (s[c]-m)*inv*g[c]+b[c]; }
    }
    o
}
fn linb(x: &[f32], w: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m*n];
    for i in 0..m { for j in 0..n { let mut a=b[j]; for t in 0..k { a += x[i*k+t]*w[j*k+t]; } o[i*n+j]=a; }}
    o
}

const B: usize = 2; const C: usize = 3; const T: usize = 3; const HH: usize = 8; const WW: usize = 8;
const P: usize = 4; const DIM: usize = 16;
const PF: usize = C*P*P; // 48

fn h_patchify(video: &[f32]) -> Vec<f32> {
    let (h, w) = (HH/P, WW/P);
    let mut out = vec![0.0f32; B*T*h*w*PF];
    for bb in 0..B { for tt in 0..T { for hy in 0..h { for wx in 0..w {
        for cc in 0..C { for p1 in 0..P { for p2 in 0..P {
            let src = ((((bb*C+cc)*T+tt)*HH + hy*P+p1)*WW) + wx*P+p2;
            let pidx = (cc*P+p1)*P+p2;
            let dst = ((((bb*T+tt)*h+hy)*w+wx)*PF)+pidx;
            out[dst] = video[src];
        }}}
    }}}}
    out
}

#[test]
fn patch_embed_matches_host() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (h, w) = (HH/P, WW/P);
    let gpu = Gpu::new_cpu(&kernel_sources());
    let video = rand(1, B*C*T*HH*WW);
    let we = PatchEmbedWeights {
        ln1_g: rand(2, PF).iter().map(|v| v+1.0).collect(), ln1_b: rand(3, PF),
        lin_w: rand(4, DIM*PF), lin_b: rand(5, DIM),
        ln2_g: rand(6, DIM).iter().map(|v| v+1.0).collect(), ln2_b: rand(7, DIM),
    };
    let got = patch_embed(&gpu, &video, &we, B as u32, C as u32, T as u32, HH as u32, WW as u32, P as u32, DIM as u32);
    // host
    let patches = h_patchify(&video);
    let n1 = ln(&patches, &we.ln1_g, &we.ln1_b, PF);
    let lin = linb(&n1, &we.lin_w, &we.lin_b, B*T*h*w, PF, DIM);
    let want = ln(&lin, &we.ln2_g, &we.ln2_b, DIM);
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-4, "patch_embed max abs {max}");
}

#[test]
fn to_pixels_matches_host() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (h, w) = (HH/P, WW/P);
    let gpu = Gpu::new_cpu(&kernel_sources());
    let tokens = rand(1, B*T*h*w*DIM);
    let we = ToPixelsWeights { lin_w: rand(2, PF*DIM), lin_b: rand(3, PF) };
    let got = to_pixels(&gpu, &tokens, &we, B as u32, T as u32, h as u32, w as u32, DIM as u32, C as u32, P as u32);
    // host: linear then unpatch
    let patched = linb(&tokens, &we.lin_w, &we.lin_b, B*T*h*w, DIM, PF);
    let mut want = vec![0.0f32; B*C*T*HH*WW];
    for bb in 0..B { for tt in 0..T { for hy in 0..h { for wx in 0..w {
        for cc in 0..C { for p1 in 0..P { for p2 in 0..P {
            let pidx = (cc*P+p1)*P+p2;
            let src = ((((bb*T+tt)*h+hy)*w+wx)*PF)+pidx;
            let dst = ((((bb*C+cc)*T+tt)*HH + hy*P+p1)*WW) + wx*P+p2;
            want[dst] = patched[src];
        }}}
    }}}}
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-4, "to_pixels max abs {max}");
}
