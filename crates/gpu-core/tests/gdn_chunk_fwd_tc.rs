// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tensor-core form of Gated DeltaNet's state-independent chunk forward
//! (`gdn_chunk_prep_tc`, steps 1-9 of `model::gdn` plus the intra-chunk score
//! precompute) against an f64 transcription of the portable algorithm:
//! cumulative decay, decay mask, the Gram product, the unit-triangular solve,
//! `u`, `w` and the intra-chunk scores, for a full 64-row chunk, a ragged
//! 5-row chunk, a chunk whose keys are nearly parallel (where a Neumann-series
//! solve would cancel to garbage), and the training entry's extra tensors.
//!
//! Swedish Embedded AB implements tensor-core training kernels for linear-
//! attention language models for its clients. If your team needs expertise in
//! fusing a chain of memory-bound launches into one tensor-core kernel while
//! keeping the model's numerics under a stated bound, then you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;

fn is_tensor_core_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu_core::native_kernels_enabled() && gpu_core::provider::cuda::tensor_core_kernels_enabled(gpu)
}

const D: usize = 128;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.signed()).collect()
}

/// `rows` unit-norm rows of width `D`; `parallel` mixes in a shared direction so
/// neighbouring rows have a cosine near 1.
fn unit_rows(rows: usize, seed: u64, parallel: f32) -> Vec<f32> {
    let common = values(D, seed + 1000);
    let mut v = values(rows * D, seed);
    for row in v.chunks_mut(D) {
        for (x, c) in row.iter_mut().zip(&common) {
            *x = *x * (1.0 - parallel) + c * parallel * 8.0;
        }
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row.iter_mut().for_each(|x| *x /= n);
    }
    v
}

struct Inputs {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    raw_g: Vec<f32>,
    beta: Vec<f32>,
}

fn inputs(bhc: usize, c: usize, seed: u64, parallel: f32) -> Inputs {
    Inputs {
        query: unit_rows(bhc * c, seed + 1, 0.0),
        key: unit_rows(bhc * c, seed + 2, parallel),
        value: values(bhc * c * D, seed + 3),
        raw_g: values(bhc * c, seed + 4).iter().map(|v| -0.4 * v.abs()).collect(),
        beta: values(bhc * c, seed + 5).iter().map(|v| 0.5 + 0.45 * v).collect(),
    }
}

/// Everything the kernel writes, in f64.
#[derive(Default)]
struct Reference {
    g_cs: Vec<f64>,
    exp_g_cs: Vec<f64>,
    u: Vec<f64>,
    w: Vec<f64>,
    intra: Vec<f64>,
    k_beta: Vec<f64>,
    v_beta: Vec<f64>,
    k_beta_decay: Vec<f64>,
    decay_mask: Vec<f64>,
    raw_attn0: Vec<f64>,
    t_mat: Vec<f64>,
    raw_intra: Vec<f64>,
}

fn reference(bhc: usize, c: usize, x: &Inputs) -> Reference {
    let scale = 1.0 / (D as f64).sqrt();
    let mut r = Reference::default();
    for b in 0..bhc {
        let row = |i: usize| (b * c + i) * D;
        let mut g = vec![0f64; c];
        let mut acc = 0f64;
        for (i, gi) in g.iter_mut().enumerate() {
            acc += x.raw_g[b * c + i] as f64;
            *gi = acc;
        }
        let dm = |i: usize, j: usize| if j <= i { (g[i] - g[j]).exp() } else { 0.0 };
        let dot = |a: &[f32], ai: usize, bb: &[f32], bi: usize| (0..D).map(|k| a[ai + k] as f64 * bb[bi + k] as f64).sum::<f64>();
        let beta = |i: usize| x.beta[b * c + i] as f64;
        // attn0 = strict_lower(-(k_beta @ key^T) * decay)
        let mut a0 = vec![vec![0f64; c]; c];
        let mut raw_a0 = vec![vec![0f64; c]; c];
        for i in 0..c {
            for j in 0..c {
                raw_a0[i][j] = -(beta(i) * dot(&x.key, row(i), &x.key, row(j)));
                if j < i {
                    a0[i][j] = raw_a0[i][j] * dm(i, j);
                }
            }
        }
        // T = (I - attn0)^-1 by forward substitution.
        let mut t = vec![vec![0f64; c]; c];
        for i in 0..c {
            t[i][i] = 1.0;
            for j in 0..i {
                t[i][j] = a0[i][j] + (j + 1..i).map(|k| a0[i][k] * t[k][j]).sum::<f64>();
            }
        }
        for i in 0..c {
            r.g_cs.push(g[i]);
            r.exp_g_cs.push(g[i].exp());
            for j in 0..c {
                r.decay_mask.push(dm(i, j));
                r.raw_attn0.push(raw_a0[i][j]);
                r.t_mat.push(t[i][j]);
                let raw = scale * dot(&x.query, row(i), &x.key, row(j));
                r.raw_intra.push(raw);
                r.intra.push(raw * dm(i, j));
            }
            for d in 0..D {
                r.k_beta.push(x.key[row(i) + d] as f64 * beta(i));
                r.v_beta.push(x.value[row(i) + d] as f64 * beta(i));
                r.k_beta_decay.push(x.key[row(i) + d] as f64 * beta(i) * g[i].exp());
                r.u.push((0..=i).map(|k| t[i][k] * x.value[row(k) + d] as f64 * beta(k)).sum());
                r.w.push((0..=i).map(|k| t[i][k] * x.key[row(k) + d] as f64 * beta(k) * g[k].exp()).sum());
            }
        }
    }
    r
}

/// The largest absolute error, relative to the larger of 1 and the tensor's own
/// largest magnitude.
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| {
        assert!(g.is_finite(), "non-finite output {g}");
        m.max((*g as f64 - w).abs() / scale)
    })
}

#[allow(clippy::too_many_arguments)]
fn check(gpu: &Gpu, case: &str, bhc: usize, c: usize, parallel: f32, tol: f64, train: bool) {
    let x = inputs(bhc, c, 7 + c as u64, parallel);
    let want = reference(bhc, c, &x);
    let up = |name: &str, v: &[f32]| gpu.storage_init(name, v);
    let (q, k, v, g, beta) = (up("q", &x.query), up("k", &x.key), up("v", &x.value), up("g", &x.raw_g), up("beta", &x.beta));
    // Poison every output so an element the kernel forgets to write is a failure, not a lucky zero.
    let out = |n: usize| gpu.storage_init("out", &vec![f32::NAN; n]);
    let (g_cs, exp_g_cs, u, w, intra) = (out(bhc * c), out(bhc * c), out(bhc * c * D), out(bhc * c * D), out(bhc * c * c));
    let extras: Vec<_> = [bhc * c * D, bhc * c * D, bhc * c * D, bhc * c * c, bhc * c * c, bhc * c * c, bhc * c * c].map(out).into_iter().collect();
    let scale = 1.0 / (D as f32).sqrt();
    let step = if train {
        let bufs = [&q, &k, &v, &g, &beta, &g_cs, &exp_g_cs, &u, &w, &intra, &extras[0], &extras[1], &extras[2], &extras[3], &extras[4], &extras[5], &extras[6]];
        gpu_core::provider::cuda::gdn_chunk_prep_train_step(gpu, &bufs, bhc as u32, c as u32, D as u32, D as u32, scale)
    } else {
        let bufs = [&q, &k, &v, &g, &beta, &g_cs, &exp_g_cs, &u, &w, &intra];
        gpu_core::provider::cuda::gdn_chunk_prep_step(gpu, &bufs, bhc as u32, c as u32, D as u32, D as u32, scale)
    }
    .expect("the tensor-core chunk forward must be offered on a CUDA device with tensor cores");
    gpu.submit(&[], &[step]);
    gpu.poll_wait();

    let mut named: Vec<(&str, Vec<f32>, &Vec<f64>)> = vec![
        ("g_cs", gpu.read(&g_cs, bhc * c), &want.g_cs),
        ("exp_g_cs", gpu.read(&exp_g_cs, bhc * c), &want.exp_g_cs),
        ("u", gpu.read(&u, bhc * c * D), &want.u),
        ("w", gpu.read(&w, bhc * c * D), &want.w),
        ("intra_scores", gpu.read(&intra, bhc * c * c), &want.intra),
    ];
    if train {
        let n = [bhc * c * D, bhc * c * D, bhc * c * D, bhc * c * c, bhc * c * c, bhc * c * c, bhc * c * c];
        let w_ = [&want.k_beta, &want.v_beta, &want.k_beta_decay, &want.decay_mask, &want.raw_attn0, &want.t_mat, &want.raw_intra];
        let names = ["k_beta", "v_beta", "k_beta_decay", "decay_mask", "raw_attn0", "t_mat", "raw_intra"];
        for i in 0..7 {
            named.push((names[i], gpu.read(&extras[i], n[i]), w_[i]));
        }
    }
    for (name, got, want) in &named {
        let e = rel_err(got, want);
        println!("  {case} {} {name}: max error {e:.2e} (bound {tol:.0e})", if train { "train" } else { "fwd  " });
        assert!(e <= tol, "{case} {name}: error {e:.3e} over {tol:.0e}");
    }
}

fn the_tensor_core_chunk_forward_matches_the_f64_reference() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_tensor_core_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device with tensor cores");
        return;
    }
    // (head-chunks, rows per chunk, key parallelism, bound): the real chunk, a ragged
    // short one, a single row, and nearly parallel keys.
    for (case, bhc, c, parallel, tol) in [("c64", 5usize, 64usize, 0.0f32, 2e-5f64), ("c5", 3, 5, 0.0, 2e-5), ("c1", 2, 1, 0.0, 2e-5), ("c64 near-parallel", 4, 64, 0.97, 5e-4)] {
        for train in [false, true] {
            check(&gpu, case, bhc, c, parallel, tol, train);
        }
    }
}

gpu_core::card_tests!(
    the_tensor_core_chunk_forward_matches_the_f64_reference,
);
