// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The native CUDA backend must compute this decoder's forward logits.**
//!
//! Every other cross-backend test in this workspace compares the CPU JIT
//! against wgpu/Vulkan - all three of which reach a kernel through the same
//! WGSL front end and the same SPIR-V (or Cranelift) lowering. `backend-cuda`
//! reaches it through a fourth, independent path: `wgsl-cuda` emits CUDA C++
//! from the naga IR, NVRTC compiles it for the capability the device itself
//! reported, and the CUDA Driver API launches it. Nothing that compares two of
//! the first three can see a defect in that fourth path.
//!
//! So this file runs ONE model's forward on every backend this box has and
//! requires them to agree:
//!
//! * the CPU JIT - the numerical reference the gradcheck suite already pins;
//! * native Vulkan - the production GPU path, when an ICD is present;
//! * native CUDA - the tier under test.
//!
//! Skipped, never failed, when the box has no NVIDIA driver or no NVRTC
//! (`brain_testutil::skip_unavailable`), so `make test` is unaffected on a
//! machine that cannot run it at all.
//!
//! Swedish Embedded AB implements differential validation of accelerator
//! backends for its clients - holding a newly written execution path to the
//! answer an established one already produces, rather than to its own output.
//! If your team needs expertise in bringing up a GPU backend and proving it
//! equal to its reference, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this model, at this shape
//!
//! `wgsl-cuda` is a generated tier over a NAMED subset of the WGSL catalogue:
//! it refuses (never approximates) the register-blocked GEMMs and the flash
//! attention family, because a barrier inside a loop has no sound guarded-body
//! form. A dense GPT decoder at `GptConfig::tiny()` with `b * t` below
//! `select::GEMM_TILE_MIN_ROWS` selects the naive `matmul` at every linear, so
//! its whole forward - `embed`, `pos_add`, `ln_stats`/`layernorm`, `matmul`,
//! `bias_add`, `attn_scores`/`attn_softmax`/`attn_apply`, `gelu`, `add2` -
//! lands inside that subset without the test having to force a kernel choice.
//! It is the smallest complete forward pass in this workspace that does.
//!
//! The shape is deliberately tiny: this is a correctness gate, never a
//! benchmark, and it must not contend with anything else resident on the card.

use gpt2::model::{Gpt, GptConfig, PIPELINES};
use gpu_core::Gpu;

/// Per-logit bound inside which every pair of backends must agree, derived from the binary32 round-off bound
/// `k u S / (1 - k u)` ([`brain_testutil::rounding`]) for this model's own shapes and weights.
///
/// A logit is `sum_j W[v, j] h_j` over the final-norm output `h`. A normalised row has unit mean square, so
/// `|x_hat_j| <= sqrt(d)` and `|h_j| <= H = sqrt(d) max|g| + max|b|` for the norm's gain `g` and shift `b`; the terms
/// of logit `v` therefore sum in absolute value to `S_v <= H sum_j |W[v, j]|`. `k` is the number of rounded operations
/// on the longest chain from a token to a logit: every reduction of every stage in turn (norm statistics, projection
/// widths, attention over `t` keys, the 4d feed-forward contraction) plus the head's own `d` adds, and a few for each
/// elementwise step. The bound takes each stage to carry relative error through with gain at most one, which the norms
/// and the residual stream are there to make true; the full forward bound is the ledger item "derive the forward
/// bound". Two backends each within the bound of the exact logit are within twice the bound of each other.
///
/// A real miscompile (a barrier some threads never reach, a uniform member read at the wrong offset, a shift the PTX ISA
/// clamps) moves a logit by orders of magnitude more than this.
///
/// perf-number: a numerical-tolerance ratio, not a throughput claim.
fn logit_bounds(cfg: &GptConfig, init: &std::collections::HashMap<String, Vec<f32>>) -> Vec<f32> {
    let (d, ff, t) = (cfg.d_model as usize, cfg.d_ff as usize, T as usize);
    let hd = d / cfg.n_heads as usize;
    let norm = 2 * d + 4; // mean, variance, then the rsqrt and scale
    let layer = norm + d + (hd + t + 4 + t) + d + 1 + norm + d + 10 + ff + 1;
    let k = cfg.n_layers as usize * layer + norm + d;

    let max_abs = |name: &str| init[name].iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let h = (d as f64).sqrt() * max_abs("ln.weight") + max_abs("ln.bias");
    init["lm_head.weight"]
        .chunks(d)
        .map(|row| brain_testutil::rounding::agreement_bound(k, h * row.iter().map(|w| f64::from(w.abs())).sum::<f64>()) as f32)
        .collect()
}

/// Worst excess of `|a - b|` over its per-vocab bound `bounds[i % vocab]`, as `(excess, index)`; `excess <= 0` is agreement.
fn worst_excess(a: &[f32], b: &[f32], bounds: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "compared tensors differ in length");
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| ((x - y).abs() - bounds[i % bounds.len()], i))
        .fold((f32::NEG_INFINITY, 0), |m, e| if e.0 > m.0 { e } else { m })
}

/// `b * t` stays under `select::GEMM_TILE_MIN_ROWS` so every linear picks the
/// naive `matmul` - see the module doc.
const B: u32 = 1;
const T: u32 = 6;

fn config() -> GptConfig {
    GptConfig::tiny()
}

/// The token ids the forward runs on, plus the ignore-labels `set_batch`
/// needs (this test reads logits, never the loss).
fn batch(cfg: &GptConfig) -> (Vec<u32>, Vec<u32>) {
    let n = (B * T) as usize;
    let x = (0..n).map(|i| (i as u32 * 7 + 3) % cfg.vocab).collect();
    (x, vec![gpt2::model::IGNORE; n])
}

/// This model's forward logits on `gpu`.
fn logits_on(gpu: Gpu, cfg: &GptConfig, init: &std::collections::HashMap<String, Vec<f32>>) -> Vec<f32> {
    let (x, y) = batch(cfg);
    let m = Gpt::new_on(gpu, cfg.clone(), B, T, init);
    m.set_batch(&x, &y);
    m.forward_submit();
    m.logits_host()
}

/// Greedy argmax per position - the property a served run depends on. A
/// per-logit difference under the argmax margin is fp32 noise; one over it is
/// a different token.
fn argmax_ids(logits: &[f32], vocab: usize) -> Vec<usize> {
    logits
        .chunks(vocab)
        .map(|row| {
            let mut best = 0usize;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            best
        })
        .collect()
}

#[test]
fn the_cuda_forward_agrees_with_every_other_backend_on_this_box() {
    let cuda = match Gpu::try_new_cuda(PIPELINES) {
        Ok(g) => g,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            return;
        }
    };
    assert_eq!(cuda.kind(), "cuda", "try_new_cuda handed back some other backend");

    let cfg = config();
    let init = gpt2::init_weights(&cfg, 7);
    let vocab = cfg.vocab as usize;

    let want = logits_on(Gpu::new_cpu(PIPELINES), &cfg, &init);
    let got = logits_on(cuda, &cfg, &init);

    assert!(got.iter().all(|v| v.is_finite()), "the CUDA forward produced non-finite logits");
    let bounds = logit_bounds(&cfg, &init);
    let (excess, at) = worst_excess(&got, &want, &bounds);
    println!("gpt2 tiny forward, cpu vs cuda: worst excess over the derived bound {excess:e} (logit {at} of {})", got.len());
    assert!(excess <= 0.0, "the CUDA forward diverges from the CPU reference at logit {at}: {} vs {}, bound {}", got[at], want[at], bounds[at % bounds.len()]);
    assert_eq!(
        argmax_ids(&got, vocab),
        argmax_ids(&want, vocab),
        "CUDA and the CPU reference pick different argmax ids"
    );

    // Vulkan is the production GPU path. Its absence is not this test's
    // failure (a box may have a driver and no ICD), but where it IS present a
    // three-way agreement is what rules out "both new paths are wrong the
    // same way".
    match Gpu::try_new_vulkan(PIPELINES) {
        Ok(vk) => {
            let vk_logits = logits_on(vk, &cfg, &init);
            let (excess, at) = worst_excess(&got, &vk_logits, &bounds);
            println!("gpt2 tiny forward, vulkan vs cuda: worst excess over the derived bound {excess:e} (logit {at})");
            assert!(excess <= 0.0, "the CUDA forward diverges from Vulkan at logit {at}: {} vs {}", got[at], vk_logits[at]);
            assert_eq!(
                argmax_ids(&got, vocab),
                argmax_ids(&vk_logits, vocab),
                "CUDA and Vulkan pick different argmax ids"
            );
        }
        Err(e) => brain_testutil::skip_unavailable(&format!("no Vulkan device to cross-check against: {e}")),
    }
}

/// The derived bound must contain the difference of two backends that ARE on every box: the CPU JIT and the default GPU
/// device. Without it the CUDA comparison above, which skips without a card, would be the only evidence that `k` and
/// `S` are not too small.
#[test]
fn the_derived_bound_contains_the_cpu_vs_default_device_difference() {
    let cfg = config();
    let init = gpt2::init_weights(&cfg, 7);
    let want = logits_on(Gpu::new_cpu(PIPELINES), &cfg, &init);
    let got = logits_on(Gpu::new(PIPELINES), &cfg, &init);
    let bounds = logit_bounds(&cfg, &init);
    let (excess, at) = worst_excess(&got, &want, &bounds);
    println!("gpt2 tiny forward, cpu vs default device: worst excess {excess:e} at logit {at}, bound there {}", bounds[at % bounds.len()]);
    assert!(excess <= 0.0, "two backends differ beyond the round-off bound at logit {at}: {} vs {}", got[at], want[at]);
}
