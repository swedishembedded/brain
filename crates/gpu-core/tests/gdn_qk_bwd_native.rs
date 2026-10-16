// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `gdn_qk_bwd_tok` against an f64 transcription of the six dispatches it replaces: the
//! chunk-major to token-major permute of `d_query` and `d_key`, the sum over the value heads
//! that share a key head (the adjoint of the head repeat) and the backward of the L2
//! normalisation with a unit gain. Ragged chunks, two sequences and a 3:1 head repeat are
//! covered; every output is poisoned first so an element the kernel forgets is a failure.
//!
//! Swedish Embedded AB implements memory-bound training kernels for linear-attention
//! language models for its clients. If your team needs expertise in collapsing the layout
//! and normalisation glue around a recurrent layer's backward into one pass, you can
//! procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;

const D: usize = 128;
const EPS: f64 = 1e-6;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.signed()).collect()
}

/// `dx` of `y = x * rsqrt(sum(x^2) + eps)` given `dy`, in f64 (the `l2norm_scale_dx` formula, unit gain).
fn l2norm_dx(x: &[f32], dy: &[f64]) -> Vec<f64> {
    let s: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
    let r = 1.0 / (s + EPS).sqrt();
    let adotx: f64 = x.iter().zip(dy).map(|(xv, d)| d * *xv as f64).sum();
    x.iter().zip(dy).map(|(xv, d)| r * (d - *xv as f64 * r * r * adotx)).collect()
}

fn case(gpu: &Gpu, name: &str, b: usize, h: usize, group: usize, n_chunks: usize, c: usize) {
    let (nkh, t) = (h / group, n_chunks * c);
    let tokens = b * t;
    let bhc = n_chunks * b * h;
    let d_query_cm = values(bhc * c * D, 71);
    let d_key_cm = values(bhc * c * D, 72);
    let query = values(tokens * nkh * D, 73);
    let key = values(tokens * nkh * D, 74);

    let up = |n: &str, v: &[f32]| gpu.storage_init(n, v);
    let (dq_cm, dk_cm, q, k) = (up("dq_cm", &d_query_cm), up("dk_cm", &d_key_cm), up("q", &query), up("k", &key));
    let poison = |n: usize| gpu.storage_init("out", &vec![f32::NAN; n]);
    let (dq, dk) = (poison(tokens * nkh * D), poison(tokens * nkh * D));
    let layout = gpu_core::provider::cuda::GdnQkBwdLayout { tokens: tokens as u32, kv_heads: nkh as u32, group: group as u32, heads: h as u32, c_len: c as u32, seq_len: t as u32, batch: b as u32, eps: EPS as f32 };
    let step = gpu_core::provider::cuda::gdn_qk_bwd_tok_step(gpu, &[&dq_cm, &dk_cm, &q, &k, &dq, &dk], layout).expect("the fused query/key backward must be offered on a CUDA device with tensor cores");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();

    for (what, cm, raw, got) in [("d_query", &d_query_cm, &query, gpu.read(&dq, tokens * nkh * D)), ("d_key", &d_key_cm, &key, gpu.read(&dk, tokens * nkh * D))] {
        let mut worst = 0f64;
        let mut scale = 1f64;
        let mut want_all = Vec::with_capacity(got.len());
        for tok in 0..tokens {
            let (bb, ti) = (tok / t, tok % t);
            let (ci, r) = (ti / c, ti % c);
            for hs in 0..nkh {
                let mut dn = vec![0f64; D];
                for gi in 0..group {
                    let head = hs * group + gi;
                    let blk = ci * b * h + bb * h + head;
                    for (e, slot) in dn.iter_mut().enumerate() {
                        *slot += cm[(blk * c + r) * D + e] as f64;
                    }
                }
                want_all.extend(l2norm_dx(&raw[(tok * nkh + hs) * D..][..D], &dn));
            }
        }
        for w in &want_all {
            scale = scale.max(w.abs());
        }
        for (g, w) in got.iter().zip(&want_all) {
            assert!(g.is_finite(), "{name} {what}: non-finite output {g}");
            worst = worst.max((*g as f64 - w).abs() / scale);
        }
        println!("  {name} {what}: max error {worst:.2e}");
        assert!(worst <= 1e-5, "{name} {what}: error {worst:.3e} over 1e-5");
    }
}

fn the_fused_query_key_backward_matches_the_host_pipeline() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if gpu.kind() != "cuda" || !gpu_core::native_kernels_enabled() || !gpu_core::provider::cuda::tensor_core_kernels_enabled(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device with tensor cores");
        return;
    }
    // (batch, value heads, heads per key head, chunks, rows per chunk).
    for (name, b, h, group, n_chunks, c) in [("c64 group3", 2usize, 6usize, 3usize, 3usize, 64usize), ("c5 group3", 1, 6, 3, 4, 5), ("c32 group1", 2, 4, 1, 2, 32)] {
        case(&gpu, name, b, h, group, n_chunks, c);
    }
}

gpu_core::card_tests!(the_fused_query_key_backward_matches_the_host_pipeline,);
