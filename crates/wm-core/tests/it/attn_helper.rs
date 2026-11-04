// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `wm_core::attn::BiasedAttn` helper end-to-end: QK-normalize q and k,
//! then biased scale-8 attention (bidir + causal), vs an exact host reference
//! that recomputes the same pipeline. Exercises the whole ST-transformer
//! attention seam the way the model will drive it.
use gpu_core::Gpu;
use wm_core::attn::BiasedAttn;

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}

const EPS: f32 = 1e-6;

// L2-normalize a head slice over hd, times per-dim g.
fn l2n(v: &[f32], g: &[f32]) -> Vec<f32> {
    let s: f32 = v.iter().map(|x| x*x).sum();
    let r = 1.0/(s+EPS).sqrt();
    v.iter().zip(g).map(|(x,gd)| x*r*gd).collect()
}

fn host(qkv: &[f32], bias: &[f32], gq: &[f32], gk: &[f32],
        b: usize, heads: usize, t: usize, hd: usize, scale: f32, causal: bool) -> Vec<f32> {
    let c = heads*hd;
    let mut out = vec![0.0f32; b*t*c];
    for bb in 0..b { for h in 0..heads {
        let slice = |off: usize, i: usize| {
            let base = (bb*t+i)*3*c + off + h*hd;
            qkv[base..base+hd].to_vec()
        };
        for i in 0..t {
            let qn = l2n(&slice(0,i), gq);
            let mut sc = vec![f32::NEG_INFINITY; t];
            for j in 0..t {
                if causal && j>i { continue; }
                let kn = l2n(&slice(c,j), gk);
                let dot: f32 = qn.iter().zip(&kn).map(|(a,b)| a*b).sum();
                sc[j] = dot*scale + bias[(h*t+i)*t+j];
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den=0.0; let mut e=vec![0.0f32;t];
            for j in 0..t { if sc[j]>f32::NEG_INFINITY { e[j]=(sc[j]-mx).exp(); den+=e[j]; } }
            for d in 0..hd {
                let mut o=0.0;
                for j in 0..t { o += (e[j]/den) * qkv[(bb*t+j)*3*c + 2*c + h*hd + d]; }
                out[(bb*t+i)*c + h*hd + d] = o;
            }
        }
    }}
    out
}

fn run(causal: bool) {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (b, heads, t, hd) = (2usize, 3usize, 5usize, 8usize);
    let c = heads*hd;
    let scale = 8.0f32;
    let att = BiasedAttn::seq();
    let gpu = Gpu::new_cpu(&BiasedAttn::kernel_sources());

    let qkv = rand(1, b*t*3*c);
    let bias = rand(2, heads*t*t);
    let gq: Vec<f32> = rand(3, hd).iter().map(|v| v+1.2).collect();
    let gk: Vec<f32> = rand(4, hd).iter().map(|v| v+1.2).collect();

    let bias_b = gpu.storage_init("bias", &bias);
    let gq_b = gpu.storage_init("gq", &gq);
    let gk_b = gpu.storage_init("gk", &gk);
    let scores = gpu.storage((b*heads*t*t) as u64);
    let probs = gpu.storage((b*heads*t*t) as u64);
    let out = gpu.storage((b*t*c) as u64);

    // QK-norm writes the normalized q,k BACK into a fresh fused buffer's q/k
    // regions; v is copied through unchanged. Build that packed buffer here the
    // way the model would (q/k normalized in place, v as-is).
    // rows for l2norm = B*T*heads, dim = hd; but the q region is strided inside
    // [.,3C]. Normalize contiguous [B*T*heads, hd] copies, then scatter back.
    let rows = (b*t*heads) as u32;
    // gather q region -> contiguous
    let gather = |off: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(b*t*heads*hd);
        for bb in 0..b { for i in 0..t { for h in 0..heads {
            let base = (bb*t+i)*3*c + off + h*hd;
            v.extend_from_slice(&qkv[base..base+hd]);
        }}}
        v
    };
    let q_c = gpu.storage_init("qc", &gather(0));
    let k_c = gpu.storage_init("kc", &gather(c));
    let qn_c = gpu.storage((b*t*heads*hd) as u64);
    let kn_c = gpu.storage((b*t*heads*hd) as u64);
    gpu.submit(&[], &[
        att.step_l2norm(&gpu, rows, hd as u32, EPS, &q_c, &gq_b, &qn_c),
        att.step_l2norm(&gpu, rows, hd as u32, EPS, &k_c, &gk_b, &kn_c),
    ]);
    // scatter normalized q,k back into a packed [B,T,3C] buffer (v unchanged)
    let qn = gpu.read(&qn_c, b*t*heads*hd);
    let kn = gpu.read(&kn_c, b*t*heads*hd);
    let mut packed = qkv.clone();
    let mut p = 0usize;
    for bb in 0..b { for i in 0..t { for h in 0..heads {
        let base = (bb*t+i)*3*c;
        packed[base + h*hd .. base + h*hd + hd].copy_from_slice(&qn[p..p+hd]);
        packed[base + c + h*hd .. base + c + h*hd + hd].copy_from_slice(&kn[p..p+hd]);
        p += hd;
    }}}
    let packed_b = gpu.storage_init("packed", &packed);

    gpu.submit(&[], &[
        att.step_scores(&gpu, b as u32, heads as u32, t as u32, hd as u32, scale, causal, &packed_b, &bias_b, &scores),
        att.step_softmax(&gpu, b as u32, heads as u32, t as u32, &scores, &probs),
        att.step_apply(&gpu, b as u32, heads as u32, t as u32, hd as u32, &probs, &packed_b, &out),
    ]);
    let got = gpu.read(&out, b*t*c);
    let want = host(&qkv, &bias, &gq, &gk, b, heads, t, hd, scale, causal);
    let max = got.iter().zip(&want).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-4, "causal={causal} helper attention max abs {max}");
}

#[test] fn biased_attn_helper_bidir() { run(false); }
#[test] fn biased_attn_helper_causal() { run(true); }
