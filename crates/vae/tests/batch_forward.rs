// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `vae::blocks::Builder::set_batch` correctness: a real `n`-image graph must
//! match `n` independent single-image graphs run with the SAME weights,
//! tensor for tensor.
//!
//! Swedish Embedded AB implements batched inference for edge-AI image models
//! for its clients. If your team needs expertise in threading a real batch
//! dimension through a hand-written GPU kernel graph without a rewrite, you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! Three cases, each printed on its own line:
//!
//! 1. A plain resnet block at `n=4` vs four `n=1` replays.
//! 2. The same, but one of the four images is a large-magnitude outlier -
//!    this is what catches GroupNorm statistics leaking across the batch axis
//!    (case 1 alone can pass that bug by coincidence, since four
//!    similar-magnitude images give similar-enough per-group statistics
//!    either way).
//! 3. A resnet block followed by a self-attention block (`attn_resolutions`
//!    non-empty) - this is what catches self-attention wrongly attending
//!    across batch elements, and it exercises the `nchw_to_rows`/
//!    `rows_to_nchw` batch-total fix directly.
//!
//! Cosine alone is scale-invariant (a uniformly mis-scaled batch slice would
//! still read cosine 1.0), so every case asserts BOTH cosine and relative L2.

use std::collections::HashMap;

use data::rng::Lcg;
use gpu_core::Gpu;
use vae::blocks::{BlockNames, Builder, Tensors, KERNELS};

const CIN: u32 = 4;
// 16, not some smaller channel count: the attention block's GEMM lowering
// (`Builder::attn`'s `gemm_attn` path) slices its `qkv` buffer with a
// `C*T`-sized offset, and a real wgpu backend requires storage-buffer bind
// offsets to be a multiple of 64 words (256 bytes,
// `min_storage_buffer_offset_alignment`). `C*T = 16*16 = 256` is; a smaller
// `C` (found the hard way: `COUT = 6` gave `C*T = 96`, not a multiple of 64,
// and every real config in this tree happens to avoid the case) is not, and
// is a PRE-EXISTING constraint of that lowering, unrelated to batching -
// this picks dimensions that do not hit it, rather than this item silently
// papering over it.
const COUT: u32 = 16;
const GROUPS: u32 = 2;
const H: u32 = 4;
const W: u32 = 4;
const RES: &str = "res";
const ATTN: &str = "attn";

fn conv_w(rng: &mut Lcg, cout: u32, cin: u32, k: u32) -> (Vec<usize>, Vec<f32>) {
    let n = (cout * cin * k * k) as usize;
    (vec![cout as usize, cin as usize, k as usize, k as usize], rng.vec_scaled(n, 0.3))
}

fn conv_b(rng: &mut Lcg, cout: u32) -> (Vec<usize>, Vec<f32>) {
    (vec![cout as usize], rng.vec_scaled(cout as usize, 0.1))
}

/// Gamma centered on 1 (not on 0) so GroupNorm's affine step does not zero out
/// the very signal the test is trying to compare.
fn norm_w(rng: &mut Lcg, c: u32) -> (Vec<usize>, Vec<f32>) {
    (vec![c as usize], (0..c).map(|_| 0.8 + 0.4 * rng.unit()).collect())
}

fn norm_b(rng: &mut Lcg, c: u32) -> (Vec<usize>, Vec<f32>) {
    (vec![c as usize], rng.vec_scaled(c as usize, 0.1))
}

/// One resnet block's weights (`cin != cout`, so the shortcut conv is
/// exercised too) plus, when `attn`, one self-attention block's on top -
/// `BlockNames::vqgan()` naming (three separate q/k/v tensors, the path most
/// exercised by the fused-qkv concatenation logic in `Builder::attn`).
fn tensors(seed: u64, attn: bool) -> Tensors {
    let mut rng = Lcg::new(seed);
    let mut t: Tensors = HashMap::new();
    t.insert(format!("{RES}.norm1.weight"), norm_w(&mut rng, CIN));
    t.insert(format!("{RES}.norm1.bias"), norm_b(&mut rng, CIN));
    t.insert(format!("{RES}.conv1.weight"), conv_w(&mut rng, COUT, CIN, 3));
    t.insert(format!("{RES}.conv1.bias"), conv_b(&mut rng, COUT));
    t.insert(format!("{RES}.norm2.weight"), norm_w(&mut rng, COUT));
    t.insert(format!("{RES}.norm2.bias"), norm_b(&mut rng, COUT));
    t.insert(format!("{RES}.conv2.weight"), conv_w(&mut rng, COUT, COUT, 3));
    t.insert(format!("{RES}.conv2.bias"), conv_b(&mut rng, COUT));
    t.insert(format!("{RES}.conv_out.weight"), conv_w(&mut rng, COUT, CIN, 1));
    t.insert(format!("{RES}.conv_out.bias"), conv_b(&mut rng, COUT));
    if attn {
        t.insert(format!("{ATTN}.norm.weight"), norm_w(&mut rng, COUT));
        t.insert(format!("{ATTN}.norm.bias"), norm_b(&mut rng, COUT));
        for leaf in ["q", "k", "v", "proj_out"] {
            t.insert(format!("{ATTN}.{leaf}.weight"), conv_w(&mut rng, COUT, COUT, 1));
            t.insert(format!("{ATTN}.{leaf}.bias"), conv_b(&mut rng, COUT));
        }
    }
    t
}

/// Run the resnet (+ optional attention) graph at batch `n` over `x_host`
/// (`[n, CIN, H, W]`, row-major), returning `[n, COUT, H, W]`.
fn run(gpu: &Gpu, t: &Tensors, attn: bool, n: u32, x_host: &[f32]) -> Vec<f32> {
    assert_eq!(x_host.len(), (n * CIN * H * W) as usize, "input is not [n,CIN,H,W]");
    let mut b = Builder::new(gpu, t, 1e-6, GROUPS, BlockNames::vqgan(), false);
    b.set_batch(n);
    let x = gpu.storage(x_host.len() as u64);
    gpu.write_f32(&x, x_host);
    let r = b.resnet(RES, CIN, COUT, H, W, &x);
    let y = if attn { b.attn(ATTN, COUT, H, W, &r) } else { r };
    let out_len = (n * COUT * H * W) as usize;
    let (steps, _) = b.finish();
    gpu.submit(&[], &steps);
    gpu.read(&y, out_len)
}

/// `(cosine, rel_l2)` of `a` against the reference `b` - `rel_l2 = |a-b| /
/// |b|`, in f64 throughout so the comparison is not itself limited by fp32.
fn cosine_and_rel_l2(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut diff2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff2 += (x - y) * (x - y);
    }
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    let rel = diff2.sqrt() / nb.sqrt().max(1e-12);
    (cos, rel)
}

/// Every per-image slice of `batched` (`[n, COUT, H, W]`) against its own
/// independently-run `n=1` reference, returning the worst `(cosine, rel_l2)`
/// pair over the images in `check`.
fn worst_against_singles(
    gpu: &Gpu,
    t: &Tensors,
    attn: bool,
    x_all: &[f32],
    batched: &[f32],
    check: &[u32],
) -> (f64, f64) {
    let (per_in, per_out) = ((CIN * H * W) as usize, (COUT * H * W) as usize);
    let (mut worst_cos, mut worst_rel) = (f64::INFINITY, 0.0f64);
    for &i in check {
        let i = i as usize;
        let slice = &x_all[i * per_in..(i + 1) * per_in];
        let single = run(gpu, t, attn, 1, slice);
        let b_slice = &batched[i * per_out..(i + 1) * per_out];
        let (cos, rel) = cosine_and_rel_l2(b_slice, &single);
        worst_cos = worst_cos.min(cos);
        worst_rel = worst_rel.max(rel);
    }
    (worst_cos, worst_rel)
}

const COS_FLOOR: f64 = 1.0 - 1e-9;
const REL_L2_CEIL: f64 = 1e-6;

#[test]
fn batch_n4_matches_four_independent_n1_runs() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let n = 4u32;
    let t = tensors(0xBA7C_0001, false);
    let mut rng = Lcg::new(0xBA7C_1000);
    let x_all = rng.vec_scaled((CIN * H * W * n) as usize, 1.0);

    let batched = run(&gpu, &t, false, n, &x_all);
    let (worst_cos, worst_rel) = worst_against_singles(&gpu, &t, false, &x_all, &batched, &[0, 1, 2, 3]);
    println!("batch n=4 vs 4x n=1: worst cosine {worst_cos:.9}, worst rel_l2 {worst_rel:.3e}");
    assert!(worst_cos >= COS_FLOOR, "cosine floor missed: {worst_cos}");
    assert!(worst_rel <= REL_L2_CEIL, "rel_l2 ceiling missed: {worst_rel:e}");
}

#[test]
fn batch_outlier_does_not_leak_across_groupnorm_statistics() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let n = 4u32;
    let t = tensors(0xBA7C_0002, false);
    let mut rng = Lcg::new(0xBA7C_2000);
    let mut x_all = rng.vec_scaled((CIN * H * W * n) as usize, 1.0);
    // Image 2 is a large-magnitude outlier. If GroupNorm's per-image
    // statistics leaked across the batch axis, images 0/1/3 would come out
    // shifted/rescaled by its presence even though their OWN pixels did not
    // change - exactly what a batched mean/var reduction over the wrong axis
    // would do, and exactly what four similar-magnitude images (case 1) could
    // pass by coincidence.
    let outlier = 2usize;
    let per_in = (CIN * H * W) as usize;
    for v in &mut x_all[outlier * per_in..(outlier + 1) * per_in] {
        *v *= 50.0;
    }

    let batched = run(&gpu, &t, false, n, &x_all);
    let (worst_cos, worst_rel) = worst_against_singles(&gpu, &t, false, &x_all, &batched, &[0, 1, 3]);
    println!("batch outlier (image {outlier}): worst cosine over the OTHER images {worst_cos:.9}, worst rel_l2 {worst_rel:.3e}");
    assert!(worst_cos >= COS_FLOOR, "cosine floor missed: {worst_cos}");
    assert!(worst_rel <= REL_L2_CEIL, "rel_l2 ceiling missed: {worst_rel:e}");
}

#[test]
fn batch_n4_with_attention_matches_four_independent_n1_runs() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let n = 4u32;
    let t = tensors(0xBA7C_0003, true);
    let mut rng = Lcg::new(0xBA7C_3000);
    let x_all = rng.vec_scaled((CIN * H * W * n) as usize, 1.0);

    let batched = run(&gpu, &t, true, n, &x_all);
    let (worst_cos, worst_rel) = worst_against_singles(&gpu, &t, true, &x_all, &batched, &[0, 1, 2, 3]);
    println!("batch n=4 with attention vs 4x n=1: worst cosine {worst_cos:.9}, worst rel_l2 {worst_rel:.3e}");
    assert!(worst_cos >= COS_FLOOR, "cosine floor missed: {worst_cos}");
    assert!(worst_rel <= REL_L2_CEIL, "rel_l2 ceiling missed: {worst_rel:e}");
}
