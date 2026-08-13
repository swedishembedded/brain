// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GenieRedux STBlock sub-modules vs exact host references implementing the
//! reference forward (models/components/attention.py): the QK-normalized biased
//! Attention (num_null_kv=0, bidir + causal) and the GEGLU FeedForward.
use gpu_core::Gpu;
use genieredux::{attn_forward, geglu_forward, kernel_sources, AttnWeights, FfWeights};

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}

fn layernorm(x: &[f32], g: &[f32], dim: usize) -> Vec<f32> {
    let rows = x.len()/dim;
    let mut o = vec![0.0f32; x.len()];
    for r in 0..rows {
        let s = &x[r*dim..(r+1)*dim];
        let m: f32 = s.iter().sum::<f32>()/dim as f32;
        let va: f32 = s.iter().map(|v| (v-m)*(v-m)).sum::<f32>()/dim as f32;
        let inv = 1.0/(va+1e-5).sqrt();
        for c in 0..dim { o[r*dim+c] = (s[c]-m)*inv*g[c]; }
    }
    o
}
// y[m,n] = x[m,k] @ W[n,k]^T
fn matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m*n];
    for i in 0..m { for j in 0..n {
        let mut a = 0.0; for t in 0..k { a += x[i*k+t]*w[j*k+t]; }
        o[i*n+j] = a;
    }}
    o
}
fn l2n(v: &[f32], g: &[f32]) -> Vec<f32> {
    let s: f32 = v.iter().map(|x| x*x).sum();
    let r = 1.0/(s+1e-6).sqrt();
    v.iter().zip(g).map(|(x,gd)| x*r*gd).collect()
}

#[allow(clippy::too_many_arguments)]
fn host_attn(x: &[f32], w: &AttnWeights, bias: &[f32], b: usize, n: usize, dim: usize,
             heads: usize, hd: usize, causal: bool) -> Vec<f32> {
    let inner = heads*hd;
    let rows = b*n;
    let xn = layernorm(x, &w.norm_gamma, dim);
    let q = matmul(&xn, &w.to_q, rows, dim, inner);
    let k = matmul(x, &w.to_k, rows, dim, inner);   // k,v from RAW x
    let v = matmul(x, &w.to_v, rows, dim, inner);
    let mut out = vec![0.0f32; rows*inner];
    for bb in 0..b { for h in 0..heads {
        let hslice = |t: &[f32], i: usize| t[(bb*n+i)*inner + h*hd .. (bb*n+i)*inner + h*hd + hd].to_vec();
        for i in 0..n {
            let qn = l2n(&hslice(&q,i), &w.q_scale);
            let mut sc = vec![f32::NEG_INFINITY; n];
            for j in 0..n {
                if causal && j>i { continue; }
                let kn = l2n(&hslice(&k,j), &w.k_scale);
                let dot: f32 = qn.iter().zip(&kn).map(|(a,b)| a*b).sum();
                sc[j] = dot*8.0 + bias[(h*n+i)*n+j];
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den=0.0; let mut e=vec![0.0f32;n];
            for j in 0..n { if sc[j]>f32::NEG_INFINITY { e[j]=(sc[j]-mx).exp(); den+=e[j]; } }
            for d in 0..hd {
                let mut o=0.0;
                for j in 0..n { o += (e[j]/den) * v[(bb*n+j)*inner + h*hd + d]; }
                out[(bb*n+i)*inner + h*hd + d] = o;
            }
        }
    }}
    matmul(&out, &w.to_out, rows, inner, dim)
}

fn make_attn(dim: usize, inner: usize, hd: usize) -> AttnWeights {
    AttnWeights {
        norm_gamma: rand(10, dim).iter().map(|v| v+1.0).collect(),
        to_q: rand(11, inner*dim), to_k: rand(12, inner*dim), to_v: rand(13, inner*dim),
        to_out: rand(14, dim*inner),
        q_scale: rand(15, hd).iter().map(|v| v+1.0).collect(),
        k_scale: rand(16, hd).iter().map(|v| v+1.0).collect(),
    }
}

fn run_attn(causal: bool) {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (b, n, dim, heads, hd) = (2usize, 4usize, 16usize, 2usize, 8usize);
    let inner = heads*hd;
    let gpu = Gpu::new_cpu(&kernel_sources());
    let x = rand(1, b*n*dim);
    let bias = rand(2, heads*n*n);
    let w = make_attn(dim, inner, hd);
    let got = attn_forward(&gpu, &x, b as u32, n as u32, dim as u32, heads as u32, hd as u32, &w, &bias, causal);
    let want = host_attn(&x, &w, &bias, b, n, dim, heads, hd, causal);
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-4, "causal={causal} genie attn max abs {max}");
}

#[test] fn genie_attn_bidir() { run_attn(false); }
#[test] fn genie_attn_causal() { run_attn(true); }

#[test]
fn genie_geglu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (rows, dim, inner) = (6usize, 16usize, 32usize);
    let gpu = Gpu::new_cpu(&kernel_sources());
    let x = rand(1, rows*dim);
    let w = FfWeights {
        norm_gamma: rand(2, dim).iter().map(|v| v+1.0).collect(),
        norm_beta: rand(6, dim),
        w_x: rand(3, inner*dim), w_gate: rand(4, inner*dim), w_out: rand(5, dim*inner),
    };
    let got = geglu_forward(&gpu, &x, rows as u32, dim as u32, inner as u32, &w);

    let mut xn = layernorm(&x, &w.norm_gamma, dim);
    for r in 0..rows { for c in 0..dim { xn[r*dim+c] += w.norm_beta[c]; } } // FF LayerNorm has bias
    let xp = matmul(&xn, &w.w_x, rows, dim, inner);
    let gate = matmul(&xn, &w.w_gate, rows, dim, inner);
    let erf = |x: f32| { let s=x.signum(); let ax=x.abs(); let t=1.0/(1.0+0.3275911*ax);
        let poly=((((1.061_405_4*t-1.453_152_1)*t+1.421_413_8)*t-0.284_496_72)*t+0.254_829_6)*t;
        s*(1.0-poly*(-ax*ax).exp()) };
    let gelu = |v: f32| 0.5*v*(1.0+erf(v*0.707_106_77));
    let act: Vec<f32> = gate.iter().zip(&xp).map(|(g,xv)| gelu(*g)*xv).collect();
    let want = matmul(&act, &w.w_out, rows, inner, dim);
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-4, "genie geglu max abs {max}");
}
