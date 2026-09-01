// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolation test for the one WGSL kernel this port adds
//! (`attn_scores_qk_kmask`), dispatched on tiny hand-computed inputs via the
//! headless CPU backend. TimesFM-3 needs a scores kernel that is
//! simultaneously: separate q/k buffers (not fused), a caller-chosen scale
//! (its own attention scale is folded into the query projection ahead of
//! this kernel, not baked in here), an optional causal mask, and an additive
//! per-key mask (patch masking) - no existing kernel in this tree combines
//! all four, so this is the one new kernel the port adds. Both of its modes
//! (sequence attention: causal; variate attention: non-causal) are checked
//! against the same hand-computed 3-token, single-head case.

use gpu_core::Gpu;

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// 3 tokens, 1 head, head_dim=2: `q[i] = k[i]` = `[1,0]`, `[0,1]`, `[1,1]`, so
/// `q_i . k_j` is easy to hand-verify. `scale=2.0` (not 1.0) to prove the
/// param is actually applied, not hardcoded. `kmask = [0, 0, -1e9]` masks key
/// 2 for every query regardless of causal.
fn run(causal: u32) -> Vec<f32> {
    let gpu = Gpu::new_cpu(&[("attn_scores_qk_kmask", kernels::ATTN_SCORES_QK_KMASK)]);
    let qk = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let q = gpu.storage_init("q", &qk);
    let k = gpu.storage_init("k", &qk);
    let kmask = gpu.storage_init("kmask", &[0.0, 0.0, -1.0e9]);
    let scores = gpu.storage(9);
    // Params: bsz, n_heads, tcols, head_dim, qk_stride, causal, scale.
    let step = gpu.step(0, &[&q, &k, &kmask, &scores], &[1, 1, 3, 2, 2, causal, gpu_core::f(2.0)], 9);
    gpu.submit(&[], &[step]);
    gpu.read(&scores, 9)
}

const NEG: f32 = -3.4e38;

#[test]
fn causal_masks_future_keys_and_applies_scale_and_kmask() {
    if skip() {
        return;
    }
    let s = run(1);
    // row i=0: only j=0 visible.
    assert_eq!(s[0], 2.0, "i=0,j=0: (1*1+0*0)*2 + kmask[0] = 2");
    assert_eq!(s[1], NEG, "i=0,j=1: causal");
    assert_eq!(s[2], NEG, "i=0,j=2: causal");
    // row i=1: j=0,1 visible.
    assert_eq!(s[3], 0.0, "i=1,j=0: (0*1+1*0)*2 + 0 = 0");
    assert_eq!(s[4], 2.0, "i=1,j=1: (0*0+1*1)*2 + 0 = 2");
    assert_eq!(s[5], NEG, "i=1,j=2: causal");
    // row i=2: all keys visible, j=2 additionally kmasked.
    assert_eq!(s[6], 2.0, "i=2,j=0: (1*1+1*0)*2 + 0 = 2");
    assert_eq!(s[7], 2.0, "i=2,j=1: (1*0+1*1)*2 + 0 = 2");
    assert!(s[8] < -1.0e8, "i=2,j=2: (1*1+1*1)*2 + (-1e9) = 4-1e9, got {}", s[8]);
}

#[test]
fn non_causal_attends_every_key_and_still_applies_kmask() {
    if skip() {
        return;
    }
    let s = run(0);
    assert_eq!(s[0], 2.0);
    assert_eq!(s[1], 0.0, "i=0,j=1 is visible now (no causal restriction)");
    assert!(s[2] < -1.0e8);
    assert_eq!(s[3], 0.0);
    assert_eq!(s[4], 2.0);
    assert!(s[5] < -1.0e8);
    assert_eq!(s[6], 2.0);
    assert_eq!(s[7], 2.0);
    assert!(s[8] < -1.0e8);
}
