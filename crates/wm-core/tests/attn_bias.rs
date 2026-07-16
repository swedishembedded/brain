// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Biased / configurable-scale attention (the GenieRedux ST-transformer
//! primitive): scores = (q·k)*scale + bias[h,i,j], with a bidirectional
//! (spatial) and a causal (temporal) variant. Forward is checked against an
//! exact host reference; the dq/dk/dbias backward is finite-difference checked.
//! The softmax + apply + dscores + dv stages reuse the existing bidir kernels
//! (the causal mask is carried by the -1e30 the scores kernel writes for j>i,
//! which softmax turns into probability 0).
use gpu_core::Gpu;

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}

const KS: usize = 0;   // attn_scores_{bidir,causal}_bias (index chosen per-run)
const KSM: usize = 1;  // attn_softmax_bidir
const KAP: usize = 2;  // attn_apply_bidir
const KDS: usize = 3;  // attn_bwd_dscores_bidir
const KDV: usize = 4;  // attn_bwd_dv_bidir
const KDQ: usize = 5;  // attn_bwd_dq_bias
const KDK: usize = 6;  // attn_bwd_dk_bias
const KDB: usize = 7;  // attn_bwd_dbias

fn gpu_for(scores_bias: (&'static str, &'static str)) -> Gpu {
    Gpu::new_cpu(&[
        scores_bias,
        ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
        ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
        ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
        ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
        ("attn_bwd_dq_bias", kernels::ATTN_BWD_DQ_BIAS),
        ("attn_bwd_dk_bias", kernels::ATTN_BWD_DK_BIAS),
        ("attn_bwd_dbias", kernels::ATTN_BWD_DBIAS),
    ])
}

// Exact host forward: out[b,i,h,d] = sum_j softmax_j((q·k)*scale + bias)[.] * v.
fn host_fwd(qkv: &[f32], bias: &[f32], b: usize, heads: usize, t: usize, hd: usize, scale: f32, causal: bool) -> Vec<f32> {
    let c = heads * hd;
    let mut out = vec![0.0f32; b * t * c];
    for bb in 0..b {
        for h in 0..heads {
            let g = |off: usize, i: usize, d: usize| qkv[(bb*t + i)*3*c + off + h*hd + d];
            for i in 0..t {
                let mut sc = vec![f32::NEG_INFINITY; t];
                for j in 0..t {
                    if causal && j > i { continue; }
                    let mut s = 0.0; for d in 0..hd { s += g(0,i,d)*g(c,j,d); }
                    sc[j] = s*scale + bias[(h*t + i)*t + j];
                }
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0.0; let mut e = vec![0.0f32; t];
                for j in 0..t { if sc[j] > f32::NEG_INFINITY { e[j] = (sc[j]-mx).exp(); den += e[j]; } }
                for d in 0..hd {
                    let mut o = 0.0; for j in 0..t { o += (e[j]/den) * g(2*c,j,d); }
                    out[(bb*t + i)*c + h*hd + d] = o;
                }
            }
        }
    }
    out
}

struct Run { qkv: Vec<f32>, bias: Vec<f32>, d_out: Vec<f32>, b: usize, heads: usize, t: usize, hd: usize, scale: f32, causal: bool }

// Runs fwd+bwd on the GPU/CPU backend; returns (out, d_qkv, d_bias).
fn gpu_fwd_bwd(r: &Run, ks_idx: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (b, heads, t, hd) = (r.b, r.heads, r.t, r.hd);
    let c = heads * hd;
    let name = if r.causal { ("attn_scores_causal_bias", kernels::ATTN_SCORES_CAUSAL_BIAS) }
               else { ("attn_scores_bidir_bias", kernels::ATTN_SCORES_BIDIR_BIAS) };
    let _ = ks_idx;
    let gpu = gpu_for(name);
    let qkv_b = gpu.storage_init("qkv", &r.qkv);
    let bias_b = gpu.storage_init("bias", &r.bias);
    let scores = gpu.storage((b*heads*t*t) as u64);
    let probs = gpu.storage((b*heads*t*t) as u64);
    let out = gpu.storage((b*t*c) as u64);
    let d_out_b = gpu.storage_init("dout", &r.d_out);
    let d_scores = gpu.storage((b*heads*t*t) as u64);
    let d_qkv = gpu.storage((b*t*3*c) as u64);
    let d_bias = gpu.storage((heads*t*t) as u64);
    let cz = if r.causal {1u32} else {0u32};
    let sc_bits = r.scale.to_bits();
    // scores_bias: [bsz, heads, t, hd, 3c, q_off=0, k_off=c, scale]
    let sp = [b as u32, heads as u32, t as u32, hd as u32, (3*c) as u32, 0, c as u32, sc_bits];
    // apply/dscores/dv reuse the [.., v_off=2c, c] param convention.
    let ap = [b as u32, heads as u32, t as u32, hd as u32, (3*c) as u32, (2*c) as u32, c as u32];
    // dq/dk_bias: [.., q_off=0, k_off=c, scale, causal]
    let dqp = [b as u32, heads as u32, t as u32, hd as u32, (3*c) as u32, 0, c as u32, sc_bits, cz];
    let dbp = [b as u32, heads as u32, t as u32, cz];
    gpu.submit(&[], &[
        gpu.step(KS,  &[&qkv_b,&bias_b,&scores], &sp, (b*heads*t*t) as u32),
        gpu.step(KSM, &[&scores,&probs], &[b as u32,heads as u32,t as u32], (b*heads*t) as u32),
        gpu.step(KAP, &[&probs,&qkv_b,&out], &ap, (b*heads*t*hd) as u32),
        gpu.step(KDS, &[&d_out_b,&qkv_b,&probs,&d_scores], &ap, (b*heads*t) as u32),
        gpu.step(KDV, &[&probs,&d_out_b,&d_qkv], &ap, (b*heads*t*hd) as u32),
        gpu.step(KDQ, &[&d_scores,&qkv_b,&d_qkv], &dqp, (b*heads*t*hd) as u32),
        gpu.step(KDK, &[&d_scores,&qkv_b,&d_qkv], &dqp, (b*heads*t*hd) as u32),
        gpu.step(KDB, &[&d_scores,&d_bias], &dbp, (heads*t*t) as u32),
    ]);
    (gpu.read(&out, b*t*c), gpu.read(&d_qkv, b*t*3*c), gpu.read(&d_bias, heads*t*t))
}

fn run_case(causal: bool) {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (b, heads, t, hd) = (2usize, 4usize, 6usize, 8usize);
    let c = heads * hd;
    let scale = 8.0f32; // GenieRedux constant scale, not 1/sqrt(hd)
    let r = Run {
        qkv: rand(1, b*t*3*c), bias: rand(2, heads*t*t), d_out: rand(3, b*t*c),
        b, heads, t, hd, scale, causal,
    };
    let (out, d_qkv, d_bias) = gpu_fwd_bwd(&r, KS);

    // Forward parity.
    let want = host_fwd(&r.qkv, &r.bias, b, heads, t, hd, scale, causal);
    let fmax = out.iter().zip(&want).map(|(a,x)|(a-x).abs()).fold(0.0f32,f32::max);
    assert!(fmax < 1e-4, "causal={causal} forward max abs {fmax}");

    // FD loss = sum(out * d_out).
    let loss = |qkv: &[f32], bias: &[f32]| -> f32 {
        host_fwd(qkv, bias, b, heads, t, hd, scale, causal).iter().zip(&r.d_out).map(|(a,g)| a*g).sum()
    };
    let eps = 1e-3f32;
    // dq: perturb the q region of qkv along a random direction.
    let dir = rand(4, b*t*3*c);
    let analytic: f32 = d_qkv.iter().zip(&dir).map(|(a,x)| a*x).sum();
    let qp: Vec<f32> = r.qkv.iter().zip(&dir).map(|(v,d)| v+eps*d).collect();
    let qm: Vec<f32> = r.qkv.iter().zip(&dir).map(|(v,d)| v-eps*d).collect();
    let numeric = (loss(&qp,&r.bias) - loss(&qm,&r.bias)) / (2.0*eps);
    assert!((analytic-numeric).abs() < 4e-3 + 8e-2*analytic.abs().max(numeric.abs()),
        "causal={causal} d_qkv: {analytic} vs {numeric}");
    // dbias: perturb bias.
    let dirb = rand(5, heads*t*t);
    let analytic_b: f32 = d_bias.iter().zip(&dirb).map(|(a,x)| a*x).sum();
    let bp: Vec<f32> = r.bias.iter().zip(&dirb).map(|(v,d)| v+eps*d).collect();
    let bm: Vec<f32> = r.bias.iter().zip(&dirb).map(|(v,d)| v-eps*d).collect();
    let numeric_b = (loss(&r.qkv,&bp) - loss(&r.qkv,&bm)) / (2.0*eps);
    assert!((analytic_b-numeric_b).abs() < 4e-3 + 8e-2*analytic_b.abs().max(numeric_b.abs()),
        "causal={causal} d_bias: {analytic_b} vs {numeric_b}");
}

#[test] fn attn_bidir_bias_fwd_and_grad() { run_case(false); }
#[test] fn attn_causal_bias_fwd_and_grad() { run_case(true); }
