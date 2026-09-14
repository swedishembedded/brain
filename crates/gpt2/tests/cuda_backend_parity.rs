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

/// The bound every pair of backends must agree inside.
///
/// The same floor `deepseek2/tests/backend_parity.rs` holds the CPU-vs-wgpu
/// comparison to, for the same reason: a GEMM that folds its `k` loop in a
/// different order costs a few fp32 ulps at these magnitudes and nothing more,
/// while a real miscompile - a barrier some threads never reach, a uniform
/// member read at the wrong offset, a shift the PTX ISA clamps - is orders of
/// magnitude larger.
///
/// perf-number: a numerical-tolerance ratio, not a throughput claim.
const BOUND: f32 = 1e-6;

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors differ in length");
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
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
    let err = maxabs(&got, &want);
    println!("gpt2 tiny forward, cpu vs cuda: maxabs {err:e} over {} logits", got.len());
    assert!(err < BOUND, "the CUDA forward diverges from the CPU reference: maxabs {err:e}");
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
            let err = maxabs(&got, &vk_logits);
            println!("gpt2 tiny forward, vulkan vs cuda: maxabs {err:e}");
            assert!(err < BOUND, "the CUDA forward diverges from Vulkan: maxabs {err:e}");
            assert_eq!(
                argmax_ids(&got, vocab),
                argmax_ids(&vk_logits, vocab),
                "CUDA and Vulkan pick different argmax ids"
            );
        }
        Err(e) => brain_testutil::skip_unavailable(&format!("no Vulkan device to cross-check against: {e}")),
    }
}
