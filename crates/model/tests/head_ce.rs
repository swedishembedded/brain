// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-lean fine-tuning of large language
// models on a single GPU for its clients. If your team needs expertise in
// training with 250k-token vocabularies without materialising the logits then
// you can procure our services by sending an email to info@swedishembedded.com.

//! `model::head_ce::HeadCe`, the chunked LM-head cross-entropy.
//!
//! Spec: loss, per-token log-probabilities and the gradient w.r.t. the hidden
//! states equal an f64 host reference computed from the head weight the tier
//! really holds, for every float-read head tier (F32, BF16, NF4), with chunk
//! sizes that divide neither the token count nor the vocabulary and with the
//! vocabulary split over several weight windows; chunking changes nothing
//! (<= 1e-5 against the unchunked run of the same kernels); the extra memory
//! is O(rows_per_chunk x vocab), not O(n x vocab); and the gradient agrees
//! with finite differences of the loss at distinct dimensions.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::head_ce::{HeadCe, HeadCeConfig, HeadKernels, IGNORE};
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, NF4_LUT};
use gpu_core::provider::library::GemmPrecision;
use model::ops::{training_kernel_list, Ops, Weight};

fn all_kernels() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| training_kernel_list().iter().chain(model::head_ce::kernels()).copied().collect())
}

fn ops() -> Ops {
    let mut ops = Ops::new(Gpu::new(all_kernels())).expect("Ops::new");
    // These tests state exact (bit-identical or finite-difference) properties: the fp32 reference tier.
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    ops
}

/// What the head weight holds after the tier's rounding, as f32 `[v, d]`.
fn effective(raw: &[f32], v: usize, d: usize, tier: Dtype) -> Vec<f32> {
    match tier {
        Dtype::F32 => raw.to_vec(),
        Dtype::BF16 => raw.iter().map(|&x| f32::from_bits((model::half::f32_to_bf16(x) as u32) << 16)).collect(),
        Dtype::NF4 => {
            let (packed, scales) = quantize_weight_lut4(raw, v, d, &NF4_LUT);
            dequantize_weight_lut4(&packed, &scales, v, d, &NF4_LUT)
        }
        other => panic!("{other:?} is not a head tier under test"),
    }
}

struct Reference {
    nll: Vec<f64>,
    d_hidden: Vec<f64>,
    d_w: Vec<f64>,
}

/// f64 loss and gradients of `sum_i w_i * CE(scale * h_i W^T, y_i)`, times `d_scale`.
#[allow(clippy::too_many_arguments)]
fn reference(h: &[f32], w: &[f32], n: usize, d: usize, v: usize, labels: &[u32], weights: Option<&[f32]>, logit_scale: f32, d_scale: f32) -> Reference {
    let mut nll = vec![0f64; n];
    let mut d_hidden = vec![0f64; n * d];
    let mut d_w = vec![0f64; v * d];
    for i in 0..n {
        if labels[i] == IGNORE {
            continue;
        }
        let logits: Vec<f64> = (0..v).map(|j| logit_scale as f64 * (0..d).map(|c| h[i * d + c] as f64 * w[j * d + c] as f64).sum::<f64>()).collect();
        let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
        let sum: f64 = logits.iter().map(|&l| (l - mx).exp()).sum();
        nll[i] = mx + sum.ln() - logits[labels[i] as usize];
        let row_w = weights.map_or(1.0, |ws| ws[i] as f64);
        for j in 0..v {
            let g = ((logits[j] - mx).exp() / sum - if j as u32 == labels[i] { 1.0 } else { 0.0 }) * row_w * d_scale as f64 * logit_scale as f64;
            for c in 0..d {
                d_hidden[i * d + c] += g * w[j * d + c] as f64;
                d_w[j * d + c] += g * h[i * d + c] as f64;
            }
        }
    }
    Reference { nll, d_hidden, d_w }
}

fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f64, |a, &v| a.max(v.abs())).max(1e-6);
    let worst = got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).abs()).fold(0f64, f64::max);
    assert!(worst <= tol * scale, "{what}: worst abs error {worst:e} exceeds {tol:e} of the scale {scale:e}");
}

struct Problem {
    n: usize,
    d: usize,
    v: usize,
    hidden: Vec<f32>,
    raw: Vec<f32>,
    labels: Vec<u32>,
}

/// Distinct, non-square dimensions; a few ignored labels.
fn problem(n: usize, d: usize, v: usize) -> Problem {
    let mut rng = Lcg::random();
    let hidden = rng.vec_scaled(n * d, 1.0);
    let raw = rng.vec_scaled(v * d, 0.4);
    let labels = (0..n).map(|i| if i % 11 == 3 { IGNORE } else { (rng.next_u64() % v as u64) as u32 }).collect();
    Problem { n, d, v, hidden, raw, labels }
}

fn head(ops: &Ops, p: &Problem, tier: Dtype) -> Weight {
    let w = Weight::upload(ops, &p.raw, p.v, p.d, tier);
    assert_eq!(w.dtype(), tier, "the head must stay at the tier under test");
    w
}

/// `rows` x `vocab_rows_per_window`: with a budget the weight is cut into
/// several windows (a budget of `k` rows of the tier's words).
fn build(ops: &Ops, p: &Problem, w: &Weight, rows: u32, windows: bool) -> HeadCe {
    let row_words = match w.dtype() {
        Dtype::F32 => p.d,
        Dtype::BF16 => p.d / 2,
        _ => p.d / 8,
    } as u64;
    let cfg = HeadCeConfig { rows_per_chunk: rows, tile_budget_words: windows.then_some(row_words * 250) };
    let hc = HeadCe::new(ops, p.d as u32, p.v as u32, w, cfg, HeadKernels::from_gpu(ops.gpu()).expect("head kernels")).expect("HeadCe::new");
    assert_eq!(hc.window_count() > 1, windows, "the budget must (not) split the {:?} head into windows", w.dtype());
    hc
}

fn run(ops: &Ops, hc: &HeadCe, p: &Problem, weights: Option<&[f32]>, logit_scale: f32, d_scale: f32) -> (model::head_ce::HeadLoss, Vec<f32>) {
    let g = ops.gpu();
    let hidden = g.storage_init("hidden", &p.hidden);
    let loss = hc.forward(ops, &hidden, p.n as u32, &p.labels, weights, logit_scale);
    let d_hidden = g.storage((p.n * p.d) as u64);
    hc.backward(ops, &hidden, &d_hidden, p.n as u32, &p.labels, weights, d_scale, logit_scale, None);
    (loss, g.read(&d_hidden, p.n * p.d))
}

const TIERS: [Dtype; 3] = [Dtype::F32, Dtype::BF16, Dtype::NF4];

#[test]
fn chunked_loss_logprobs_and_hidden_gradient_match_the_host_reference() {
    let ops = ops();
    // n = 150 and v = 1000 divide by neither the 64-row chunk nor the 250-row windows.
    let p = problem(150, 96, 1000);
    for tier in TIERS {
        let w = head(&ops, &p, tier);
        let eff = effective(&p.raw, p.v, p.d, tier);
        let want = reference(&p.hidden, &eff, p.n, p.d, p.v, &p.labels, None, 1.0, 1.0 / 140.0);
        for windows in [false, true] {
            let hc = build(&ops, &p, &w, 64, windows);
            let (loss, d_hidden) = run(&ops, &hc, &p, None, 1.0, 1.0 / 140.0);
            let what = format!("{tier:?} windows={windows}");
            assert_eq!(loss.count as usize, p.labels.iter().filter(|&&l| l != IGNORE).count(), "{what}: count");
            let sum: f64 = want.nll.iter().sum();
            assert!((loss.loss_sum - sum).abs() <= 1e-4 * sum.abs(), "{what}: loss {} vs {sum}", loss.loss_sum);
            assert_close(&loss.logprobs.iter().map(|l| -l).collect::<Vec<_>>(), &want.nll, 1e-4, &format!("{what} logprobs"));
            assert_close(&d_hidden, &want.d_hidden, 1e-4, &format!("{what} d_hidden"));
        }
    }
}

#[test]
fn chunking_and_windowing_change_nothing() {
    let ops = ops();
    let p = problem(150, 96, 1000);
    for tier in TIERS {
        let w = head(&ops, &p, tier);
        let whole = build(&ops, &p, &w, 192, false);
        let (l0, g0) = run(&ops, &whole, &p, None, 1.0, 0.01);
        for (rows, windows) in [(64, false), (64, true), (128, true)] {
            let hc = build(&ops, &p, &w, rows, windows);
            let (l1, g1) = run(&ops, &hc, &p, None, 1.0, 0.01);
            let scale = g0.iter().fold(0f32, |a, v| a.max(v.abs()));
            let worst = g0.iter().zip(&g1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst <= 1e-5 * scale.max(1e-6), "{tier:?} rows={rows} windows={windows}: gradient differs by {worst:e}");
            assert!((l0.loss_sum - l1.loss_sum).abs() <= 1e-5 * l0.loss_sum.abs(), "{tier:?} rows={rows} windows={windows}: loss");
        }
    }
}

#[test]
fn per_token_weights_and_logit_scale_enter_loss_and_gradient() {
    let ops = ops();
    let p = problem(70, 64, 300);
    let mut rng = Lcg::random();
    let weights: Vec<f32> = (0..p.n).map(|_| rng.unit() * 2.0).collect();
    let w = head(&ops, &p, Dtype::BF16);
    let eff = effective(&p.raw, p.v, p.d, Dtype::BF16);
    let (logit_scale, d_scale) = (0.7f32, 0.05f32);
    let want = reference(&p.hidden, &eff, p.n, p.d, p.v, &p.labels, Some(&weights), logit_scale, d_scale);
    let hc = build(&ops, &p, &w, 64, false);
    let (loss, d_hidden) = run(&ops, &hc, &p, Some(&weights), logit_scale, d_scale);
    let sum: f64 = want.nll.iter().zip(&weights).map(|(l, w)| l * *w as f64).sum();
    assert!((loss.loss_sum - sum).abs() <= 1e-4 * sum.abs(), "weighted loss {} vs {sum}", loss.loss_sum);
    assert_close(&d_hidden, &want.d_hidden, 1e-4, "weighted d_hidden");
}

/// d(loss)/d(hidden) by central differences of the device loss, at dims
/// (n, d, v) = (9, 96, 130), pairwise distinct and none a multiple of 64 or of the chunk.
#[test]
fn hidden_gradient_agrees_with_finite_differences_of_the_loss() {
    let ops = ops();
    let g = ops.gpu();
    let p = problem(9, 96, 130);
    for tier in TIERS {
        let w = head(&ops, &p, tier);
        let hc = build(&ops, &p, &w, 64, false);
        let count = p.labels.iter().filter(|&&l| l != IGNORE).count() as f32;
        let loss_of = |h: &[f32]| hc.forward(&ops, &g.storage_init("h", h), p.n as u32, &p.labels, None, 1.0).loss_sum / count as f64;
        let (_, analytic) = run(&ops, &hc, &p, None, 1.0, 1.0 / count);
        let eps = 1e-2f32;
        let scale = analytic.iter().fold(0f32, |a, v| a.max(v.abs()));
        for i in (0..p.n * p.d).step_by(13) {
            let (mut hi, mut lo) = (p.hidden.clone(), p.hidden.clone());
            hi[i] += eps;
            lo[i] -= eps;
            let fd = (loss_of(&hi) - loss_of(&lo)) / (2.0 * eps as f64);
            assert!((fd - analytic[i] as f64).abs() <= 2e-2 * scale as f64 + 1e-5, "{tier:?} element {i}: finite difference {fd} vs analytic {}", analytic[i]);
        }
    }
}

#[test]
fn an_f32_head_accumulates_its_weight_gradient() {
    let ops = ops();
    let g = ops.gpu();
    let p = problem(150, 96, 1000);
    let w = head(&ops, &p, Dtype::F32);
    let want = reference(&p.hidden, &p.raw, p.n, p.d, p.v, &p.labels, None, 1.0, 0.01);
    for windows in [false, true] {
        let hc = build(&ops, &p, &w, 64, windows);
        let hidden = g.storage_init("hidden", &p.hidden);
        let (d_hidden, dw) = (g.storage((p.n * p.d) as u64), g.storage((p.v * p.d) as u64));
        // Accumulates: two backward passes give twice the gradient.
        for _ in 0..2 {
            hc.backward(&ops, &hidden, &d_hidden, p.n as u32, &p.labels, None, 0.01, 1.0, Some(&dw));
        }
        let doubled: Vec<f64> = want.d_w.iter().map(|x| 2.0 * x).collect();
        assert_close(&g.read(&dw, p.v * p.d), &doubled, 1e-4, &format!("dW windows={windows}"));
    }
}

/// Spec: a trained bf16 head (a resident full fine-tune's table) accumulates its weight gradient into a bf16 buffer
/// equal to the f64 reference of the weight it holds accumulated the same way (one rounding per chunk of tokens), up to
/// one bf16 step per rounding, with the vocabulary in one window or several and
/// with a tile budget small enough that each window's gradient is produced in several bounded runs of rows.
#[test]
fn a_bf16_head_accumulates_a_bf16_weight_gradient() {
    use model::head_ce::{GradScale, HeadBackward, HeadGrad};
    use model::ops::GradDtype;
    let mut ops = ops();
    // 37 vocabulary rows of 96 floats per bounded run: neither the window nor the vocabulary is a multiple of it.
    ops.set_tile_budget_bytes(37 * 96 * 4);
    let g = ops.gpu();
    let p = problem(150, 96, 1000);
    let w = head(&ops, &p, Dtype::BF16);
    // The head accumulates one chunk of 64 tokens at a time into the bf16 gradient: the host reference adds each
    // chunk's f64 product and rounds to bf16 after it.
    let head_w = effective(&p.raw, p.v, p.d, Dtype::BF16);
    let mut expected = vec![0f32; p.v * p.d];
    // The largest value each element was rounded at: a rounding's error is half a bf16 step of the value rounded.
    let mut largest = vec![0f64; p.v * p.d];
    let mut roundings = 0;
    for r0 in (0..p.n).step_by(64) {
        let rows = 64.min(p.n - r0);
        let part = reference(&p.hidden[r0 * p.d..(r0 + rows) * p.d], &head_w, rows, p.d, p.v, &p.labels[r0..r0 + rows], None, 1.0, 0.01);
        for ((e, big), &x) in expected.iter_mut().zip(largest.iter_mut()).zip(&part.d_w) {
            let sum = *e as f64 + x;
            *big = big.max(sum.abs());
            *e = f32::from_bits((model::half::f32_to_bf16(sum as f32) as u32) << 16);
        }
        roundings += 1;
    }
    for windows in [false, true] {
        let hc = build(&ops, &p, &w, 64, windows);
        let hidden = g.storage_init("hidden", &p.hidden);
        let labels = g.storage(p.n as u64);
        g.write(&labels, &p.labels);
        let (d_hidden, dw) = (g.storage((p.n * p.d) as u64), g.storage((p.v * p.d / 2) as u64));
        let mut s = Vec::new();
        let grad = HeadGrad { buf: &dw, dtype: GradDtype::Bf16 };
        let b = HeadBackward { hidden: &hidden, d_hidden: &d_hidden, n: p.n as u32, labels: &labels, grad_scale: GradScale::Const(0.01), logit_scale: 1.0, row_weights: None, dw: Some(grad), forward_state_held: false };
        hc.backward_steps(&ops, &mut s, &b);
        g.submit(&[], &s);
        let got: Vec<f32> = g.read(&dw, p.v * p.d / 2).iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xFFFF_0000)]).collect();
        for (i, (&got, &want)) in got.iter().zip(&expected).enumerate() {
            // The same accumulation in f64 per chunk, rounded to bf16 after each: where the f32 product and the f64
            // reference round across a boundary differently, each rounding may differ by one bf16 step of the largest
            // value rounded.
            let step = f64::from(f32::from_bits(((largest[i] as f32).to_bits() & 0x7F80_0000).max(1 << 23))) / 128.0;
            assert!((got - want).abs() as f64 <= roundings as f64 * step, "dW[{i}] windows={windows}: {got} vs {want}");
        }
    }
}

/// The extra device memory is the per-chunk scratch only: it does not grow with
/// the number of tokens, and is far below the `n x vocab` logits it replaces.
#[test]
fn scratch_memory_is_per_chunk_not_per_token() {
    let ops = ops();
    let p = problem(8, 96, 1000);
    let w = head(&ops, &p, Dtype::BF16);
    let hc = build(&ops, &p, &w, 64, true);
    let words = hc.scratch_words();
    let (rows, v) = (64u64, 1000u64);
    assert!(words <= 2 * rows * v + 2 * rows + rows * 250, "scratch {words} words exceeds the per-chunk bound");
    for n in [150u64, 4096, 1 << 20] {
        assert!(words < n * v || n <= rows, "scratch {words} words must be below the n x vocab logits at n = {n}");
        assert_eq!(hc.scratch_words(), words, "scratch must not depend on the token count");
    }
}

/// Real-width device run (`gpu ... --ignored --nocapture` with
/// `BRAIN_DEVICE=gpu BRAIN_BACKEND=cuda`): 4096 tokens through a 248320 x 5120
/// bf16 head, forward and backward, chunks of 512 and 1024 rows agreeing, and
/// the time per pass.
#[test]
#[ignore = "needs a GPU: run under `gpu` with BRAIN_DEVICE=gpu BRAIN_BACKEND=cuda"]
fn bf16_head_at_real_width_runs_4096_tokens_and_reports_its_time() {
    let ops = ops();
    let g = ops.gpu();
    assert!(ops.caps().workgroup_reductions, "this test needs a device that runs the cooperative kernels, not the CPU JIT");
    let (n, d, v) = (4096usize, 5120usize, 248_320usize);
    let mut rng = Lcg::random();
    // A repeated random block keeps host-side generation of the 1.27G-element table cheap.
    let block = rng.vec_scaled(1 << 16, 0.05);
    let raw: Vec<f32> = block.iter().copied().cycle().take(v * d).collect();
    let w = Weight::upload(&ops, &raw, v, d, Dtype::BF16);
    drop(raw);
    let hidden = g.storage_init("hidden", &rng.vec_scaled(n * d, 1.0));
    let labels: Vec<u32> = (0..n).map(|_| (rng.next_u64() % v as u64) as u32).collect();
    let d_hidden = g.storage((n * d) as u64);
    let mut sums = Vec::new();
    for rows in [512u32, 1024] {
        let cfg = HeadCeConfig { rows_per_chunk: rows, tile_budget_words: None };
        let hc = HeadCe::new(&ops, d as u32, v as u32, &w, cfg, HeadKernels::from_gpu(g).unwrap()).unwrap();
        let t = std::time::Instant::now();
        let loss = hc.forward(&ops, &hidden, n as u32, &labels, None, 1.0);
        let fwd = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        hc.backward(&ops, &hidden, &d_hidden, n as u32, &labels, None, 1.0 / n as f32, 1.0, None);
        g.read(&d_hidden, 1);
        let bwd = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("rows_per_chunk={rows} windows={} scratch={:.2} GiB: forward {fwd:.0} ms, backward {bwd:.0} ms, loss_sum {:.3}", hc.window_count(), hc.scratch_words() as f64 * 4.0 / (1u64 << 30) as f64, loss.loss_sum);
        assert!(loss.loss_sum.is_finite() && loss.count as usize == n);
        sums.push(loss.loss_sum);
    }
    assert!((sums[0] - sums[1]).abs() <= 1e-4 * sums[0].abs(), "chunk size changed the loss: {sums:?}");
}
