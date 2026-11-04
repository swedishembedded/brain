// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Blockwise causal grouped-query attention on the library's tensor cores
//! against an f64 oracle: the forward context and log-sum-exp, and the three
//! gradients, over several query blocks, a ragged last block and a KV group of
//! more than one head.
//!
//! Swedish Embedded AB implements long-sequence attention training for its
//! clients. If your team needs expertise in training transformers at sequence
//! lengths where the attention matrix dominates the step, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::provider::attention::AttnShape;
use gpu_core::provider::library::{GemmPrecision, LibraryGemmProvider};
use gpu_core::Gpu;

static NO_KERNELS: &[(&str, &str)] = &[];

fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn values(n: usize, scale: f32, salt: u64) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(salt + 1);
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((((z >> 40) as f32) / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
        })
        .collect()
}

struct Oracle {
    ctx: Vec<f64>,
    lse: Vec<f64>,
    dq: Vec<f64>,
    dk: Vec<f64>,
    dv: Vec<f64>,
}

/// Dense f64 causal GQA attention and its gradients over (bf16-rounded) inputs.
fn oracle(sh: &AttnShape, q: &[f32], k: &[f32], v: &[f32], d_ctx: &[f32]) -> Oracle {
    let (t, nh, nkv, hd) = (sh.t as usize, sh.n_heads as usize, sh.n_kv_heads as usize, sh.head_dim as usize);
    let (qd, kvd, g) = (nh * hd, nkv * hd, nh / nkv);
    let scale = 1.0 / (hd as f64).sqrt();
    let mut o = Oracle { ctx: vec![0.0; t * qd], lse: vec![0.0; nh * t], dq: vec![0.0; t * qd], dk: vec![0.0; t * kvd], dv: vec![0.0; t * kvd] };
    for h in 0..nh {
        let kvh = h / g;
        for i in 0..t {
            let dot = |a: &[f32], ao: usize, b: &[f32], bo: usize| (0..hd).map(|d| a[ao + d] as f64 * b[bo + d] as f64).sum::<f64>();
            let s: Vec<f64> = (0..=i).map(|j| scale * dot(q, i * qd + h * hd, k, j * kvd + kvh * hd)).collect();
            let m = s.iter().cloned().fold(f64::MIN, f64::max);
            let sum: f64 = s.iter().map(|x| (x - m).exp()).sum();
            o.lse[h * t + i] = m + sum.ln();
            let p: Vec<f64> = s.iter().map(|x| (x - m).exp() / sum).collect();
            for d in 0..hd {
                o.ctx[i * qd + h * hd + d] = (0..=i).map(|j| p[j] * v[j * kvd + kvh * hd + d] as f64).sum();
            }
            // Backward for this row.
            let dsum: f64 = (0..hd).map(|d| d_ctx[i * qd + h * hd + d] as f64 * o.ctx[i * qd + h * hd + d]).sum();
            for j in 0..=i {
                let dp = dot(d_ctx, i * qd + h * hd, v, j * kvd + kvh * hd);
                let ds = p[j] * (dp - dsum) * scale;
                for d in 0..hd {
                    o.dq[i * qd + h * hd + d] += ds * k[j * kvd + kvh * hd + d] as f64;
                    o.dk[j * kvd + kvh * hd + d] += ds * q[i * qd + h * hd + d] as f64;
                    o.dv[j * kvd + kvh * hd + d] += p[j] * d_ctx[i * qd + h * hd + d] as f64;
                }
            }
        }
    }
    o
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    let rms = (want.iter().map(|x| x * x).sum::<f64>() / want.len() as f64).sqrt();
    let worst = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    // bf16 operands carry a 2^-8 relative rounding in every probability and score
    // gradient; the worst element of a few hundred thousand lands at a few percent
    // of the RMS, while the whole tensor must stay aligned with the oracle.
    assert!(worst <= 6e-2 * rms.max(1e-6), "{what}: worst error {worst} exceeds 6e-2 of the RMS {rms}");
    let dot: f64 = got.iter().zip(want).map(|(g, w)| *g as f64 * w).sum();
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
    let cosine = dot / (norm(&mut got.iter().map(|g| *g as f64)) * norm(&mut want.iter().copied()));
    assert!(cosine > 0.9995, "{what}: cosine {cosine}");
}

#[test]
fn blockwise_attention_matches_an_f64_oracle_forward_and_backward() {
    let Ok(gpu) = Gpu::try_new_cuda(NO_KERNELS) else {
        eprintln!("cuda_library_attention: no CUDA device on this box - nothing to run");
        return;
    };
    let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
    if !provider.is_active() {
        eprintln!("cuda_library_attention: this card runs the fp32 path - nothing to run");
        return;
    }
    // Two KV groups of three heads; 72 tokens in blocks of 32 (a ragged last block),
    // then 256 in blocks of 64.
    for (t, rows, hd) in [(72u32, 32u32, 64u32), (256, 64, 256)] {
        let sh = AttnShape { b: 1, t, n_heads: 6, n_kv_heads: 2, head_dim: hd, block_rows: Some(rows) };
        let (n, qd, kvd) = (t as usize, (sh.n_heads * hd) as usize, (sh.n_kv_heads * hd) as usize);
        let r = |v: Vec<f32>| v.iter().map(|x| round_bf16(*x)).collect::<Vec<_>>();
        let (q, k, v, d_ctx) = (r(values(n * qd, 1.0, 1)), r(values(n * kvd, 1.0, 2)), r(values(n * kvd, 1.0, 3)), r(values(n * qd, 1.0, 4)));
        let want = oracle(&sh, &q, &k, &v, &d_ctx);

        let (qb, kb, vb) = (gpu.storage_init("q", &q), gpu.storage_init("k", &k), gpu.storage_init("v", &v));
        let (ctx, lse) = (gpu.storage((n * qd) as u64), gpu.storage((sh.n_heads * t) as u64));
        let steps = provider.attention_fwd(&gpu, &sh, &qb, &kb, &vb, &ctx, &lse).expect("the forward was declined");
        gpu.submit(&[], &steps);
        let what = format!("t={t} hd={hd}");
        assert_close(&gpu.read(&ctx, n * qd), &want.ctx, &format!("{what}: ctx"));
        let lse_got = gpu.read(&lse, (sh.n_heads * t) as usize);
        let worst = lse_got.iter().zip(&want.lse).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
        assert!(worst < 3e-2, "{what}: lse error {worst}");

        let dob = gpu.storage_init("dctx", &d_ctx);
        let (dq, dk, dv) = (gpu.storage((n * qd) as u64), gpu.storage((n * kvd) as u64), gpu.storage((n * kvd) as u64));
        let steps = provider.attention_bwd(&gpu, &sh, &qb, &kb, &vb, &ctx, &lse, &dob, &dq, &dk, &dv).expect("the backward was declined");
        gpu.submit(&[], &steps);
        assert_close(&gpu.read(&dq, n * qd), &want.dq, &format!("{what}: dq"));
        assert_close(&gpu.read(&dk, n * kvd), &want.dk, &format!("{what}: dk"));
        assert_close(&gpu.read(&dv, n * kvd), &want.dv, &format!("{what}: dv"));
    }
}
