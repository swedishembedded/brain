// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `topk_mask.wgsl` at a row length (`T=100`) well past the trivial sizes
//! every existing glmdsa integration test uses (`GlmConfig::tiny()`'s
//! `block_size<=16`), which cannot tell a correct `(b,s,t)` cell
//! decomposition from a broken one (`s`/`t` swapped, wrong divisor order)
//! since a tiny `T` makes most rows degenerate. Checked against an
//! INDEPENDENTLY written host oracle (the kernel's own doc formula, not a
//! copy of the kernel body) at every cell, across both the sparse
//! (`index_topk < T`) and all-pass (`index_topk >= T`) regimes.
//!
//! ```text
//! cargo test -p brain-glmdsa --test topk_mask_kernel
//! BRAIN_DEVICE=cpu cargo test -p brain-glmdsa --test topk_mask_kernel
//! ```

use data::rng::Lcg;
use gpu_core::Gpu;

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Independently re-derived from `topk_mask.wgsl`'s own header formula, not
/// copied from its body: `mask[b,s,t] = 0` iff `t<=s` AND `score[b,s,t]`'s
/// rank (count of causal scores strictly greater) is `< min(index_topk,s+1)`.
fn host_oracle(bsz: usize, t_len: usize, index_topk: u32, scores: &[f32]) -> Vec<f32> {
    let mut mask = vec![0f32; bsz * t_len * t_len];
    for b in 0..bsz {
        for s in 0..t_len {
            let causal_len = (s + 1) as u32;
            let count = index_topk.min(causal_len);
            for t in 0..t_len {
                let base = (b * t_len + s) * t_len;
                mask[base + t] = if t > s {
                    -3.4e38
                } else if count >= causal_len {
                    0.0
                } else {
                    let v = scores[base + t];
                    let greater = (0..=s).filter(|&t2| scores[base + t2] > v).count() as u32;
                    if greater < count {
                        0.0
                    } else {
                        -3.4e38
                    }
                };
            }
        }
    }
    mask
}

fn run(g: &Gpu, bsz: usize, t_len: usize, index_topk: u32, scores_h: &[f32]) -> Vec<f32> {
    let scores = g.storage_init("scores", scores_h);
    let mask = g.storage((bsz * t_len * t_len) as u64);
    let step = g.step(idx(g, "topk_mask"), &[&scores, &mask], &[bsz as u32, t_len as u32, index_topk], (bsz * t_len * t_len) as u32);
    g.submit(&[], &[step]);
    g.read(&mask, bsz * t_len * t_len)
}

#[test]
fn topk_mask_matches_host_oracle_at_a_nontrivial_row_length() {
    let g = gpu_core::testgpu::dev(glmdsa::model::PIPELINES);
    let (bsz, t_len) = (1usize, 100usize);
    let mut rng = Lcg::new(20260902);
    let scores_h: Vec<f32> = (0..bsz * t_len * t_len).map(|_| rng.scaled(2.0) - 1.0).collect();

    for &index_topk in &[3u32, 17, 64, 999] {
        let got = run(&g, bsz, t_len, index_topk, &scores_h);
        let want = host_oracle(bsz, t_len, index_topk, &scores_h);
        for (i, (&g_, &w_)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g_, w_, "index_topk={index_topk} cell {i}: got {g_}, want {w_}");
        }
    }
}
