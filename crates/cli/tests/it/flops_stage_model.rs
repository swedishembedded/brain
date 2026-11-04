// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two structural facts `brain flops --model flux2|ltxv` rests on.
//!
//! Swedish Embedded AB implements analytic performance models for GPU
//! inference pipelines. If your team needs to know what a generation will cost
//! on hardware it has never run on, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Pricing a 4B-parameter denoiser offline means never holding 4B weights, so
//! the cost model DERIVES the full-depth graph from probe builds of the same
//! config at one or zero blocks. That is exact only if the graph is affine in
//! the block counts:
//!
//!   cost(nd, ns) = cost(0, 0) + nd*(cost(1,0) - cost(0,0)) + ns*(cost(0,1) - cost(0,0))
//!
//! which is an assertion about the model, not about the arithmetic, and is
//! gated here at a point the basis does not contain.
//!
//! The second fact is the one that decides whether the model can PREDICT at
//! all rather than merely reproduce: attention must come out quadratic in the
//! token count and the GEMMs linear. A cost model with the right constant and
//! the wrong exponent is worse than none, because it is only wrong far away
//! from where it was checked.

use std::collections::BTreeMap;

use flux2::{position_ids, Flux2Config, Flux2Model, Precision};
use gpu_core::cost::{Cost, CostReport, Recording};

/// A deliberately asymmetric toy config: every axis differs from every other,
/// so a formula that confused two of them cannot pass.
fn cfg(depth_double: usize, depth_single: usize) -> Flux2Config {
    Flux2Config {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 16,
        n_heads: 2,
        depth_double,
        depth_single,
        axes_dim: [2, 2, 2, 2],
        txt_len: 8,
        ..Flux2Config::klein_4b()
    }
}

fn zeros_for(c: &Flux2Config) -> flux2::Tensors {
    let mut ts = flux2::Tensors::new();
    for (name, shape) in c.tensor_manifest() {
        let n: usize = shape.iter().product();
        ts.insert(name, (shape, vec![0.0f32; n]));
    }
    ts
}

/// The DiT graph for one denoise evaluation at `lh*lw` latent tokens, recorded
/// without executing anything.
fn record(c: &Flux2Config, lh: usize, lw: usize) -> CostReport {
    let ni = lh * lw;
    let ts = zeros_for(c);
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let m = Flux2Model::new_with(c, &ts, gpu, (c.txt_len + ni) as u32, Precision::F32);
    let ids = position_ids(c.txt_len, lh, lw, &[]);
    let img = vec![0.0f32; ni * c.in_channels];
    let ctx = vec![0.0f32; c.txt_len * c.context_in_dim];
    let rec = Recording::dry();
    let _ = m.forward(&img, &ctx, 0.7, &ids, ni);
    rec.take()
}

fn assert_same(what: &str, a: &CostReport, b: &CostReport) {
    assert_eq!(a.total, b.total, "{what}: totals differ");
    assert_eq!(a.steps, b.steps, "{what}: dispatch counts differ");
    assert_eq!(a.uncovered, b.uncovered, "{what}: uncovered sets differ");
    let names: Vec<&String> = a.by_kernel.keys().chain(b.by_kernel.keys()).collect();
    for n in names {
        let (x, y) = (a.by_kernel.get(n), b.by_kernel.get(n));
        let (x, y) = (x.map(|k| (k.calls, k.cost)), y.map(|k| (k.calls, k.cost)));
        assert_eq!(x, y, "{what}: kernel {n} differs");
    }
}

#[test]
fn the_flux2_graph_is_affine_in_block_depth() {
    let (lh, lw) = (4, 4);
    let c00 = record(&cfg(0, 0), lh, lw);
    let c10 = record(&cfg(1, 0), lh, lw);
    let c01 = record(&cfg(0, 1), lh, lw);
    assert!(c00.steps > 0 && c10.steps > c00.steps && c01.steps > c00.steps, "probes must be nested and non-empty");

    let per_double = c10.checked_sub(&c00).expect("a 1-double build contains the 0-block build");
    let per_single = c01.checked_sub(&c00).expect("a 1-single build contains the 0-block build");

    // The point the basis does not contain: the double -> single transition.
    let mut predicted = c00.clone();
    predicted.merge(&per_double);
    predicted.merge(&per_single);
    assert_same("predicted (1,1) vs recorded", &predicted, &record(&cfg(1, 1), lh, lw));

    // ...and a point where BOTH counts are above one, which is what would
    // catch a per-block cost that depended on the block's index.
    let mut p22 = c00.clone();
    p22.merge(&per_double.scaled(2));
    p22.merge(&per_single.scaled(2));
    assert_same("predicted (2,2) vs recorded", &p22, &record(&cfg(2, 2), lh, lw));
}

/// FLOPs per kernel kind at one token count.
fn by_kind(r: &CostReport) -> BTreeMap<String, Cost> {
    r.by_kernel.iter().map(|(n, k)| (n.clone(), k.cost)).collect()
}

#[test]
fn attention_is_quadratic_in_tokens_and_the_gemms_are_linear() {
    // Two latent grids whose JOINT token counts (txt + img) differ by a clean
    // ratio: 8 + 4*4 = 24 and 8 + 8*8 = 72, so n triples.
    let c = cfg(2, 2);
    let small = by_kind(&record(&c, 4, 4));
    let big = by_kind(&record(&c, 8, 8));
    let (n_small, n_big) = (8.0 + 16.0, 8.0 + 64.0);
    let r: f64 = n_big / n_small;

    // The EXPONENT, not the constant. A cost model fitted at one size can get
    // the constant right and the exponent wrong, and it is then only wrong far
    // away from where it was checked - which is the one place a predictor is
    // for.
    let exponent = |kind: &str| -> f64 {
        let a = small.get(kind).unwrap_or_else(|| panic!("{kind} not dispatched at the small size")).flops as f64;
        let b = big.get(kind).unwrap_or_else(|| panic!("{kind} not dispatched at the big size")).flops as f64;
        assert!(a > 0.0, "{kind} costs nothing at the small size");
        println!("{kind:<26} {a:>14} -> {b:>14} flops  ({:.4}x for {r:.4}x tokens)", b / a);
        (b / a).ln() / r.ln()
    };

    // Attention over the joint sequence: every query attends every key, so the
    // pair count - and with it the score/apply MACs - grows as n^2. The per-row
    // softmax terms make it very slightly sub-quadratic, which is why the
    // assertion is on the exponent with a band and not on an exact ratio.
    let attn = exponent("flash_attn_bidir_reg2");
    assert!((attn - 2.0).abs() < 0.05, "bidirectional attention must be ~quadratic in tokens, measured exponent {attn:.4}");

    // The projections contract fixed [d, d] / [d, mlp] weights against the
    // rows, so they are degree 1. Not exactly proportional: the text rows are
    // a fixed count, so the total is affine in n, not a multiple of it.
    let mm = exponent("matmul_reg3");
    assert!((mm - 1.0).abs() < 0.02, "the projection GEMMs must be ~linear in tokens, measured exponent {mm:.4}");

    // And the two must be TOLD APART - a model that got attention's exponent
    // wrong would still match at the size it was fitted on.
    assert!(attn - mm > 0.8, "attention must outgrow the GEMMs as tokens rise ({attn:.4} vs {mm:.4})");
}
