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

/// `[b=1,v=2,n=3,d=2] -> [b=1,n=3,v=2,d=2]` - the exact shape
/// `model::core_forward` needs to move between sequence attention (V-major)
/// and variate attention (N-major). Values are `100*v_idx + 10*n_idx + d_idx`
/// so every output position's expected source is unambiguous.
#[test]
fn swap_axes12_vec_moves_the_variate_axis_next_to_batch() {
    if skip() {
        return;
    }
    let gpu = Gpu::new_cpu(&[("swap_axes12_vec", kernels::SWAP_AXES12_VEC)]);
    let (v, n, d) = (2usize, 3usize, 2usize);
    let mut src = vec![0.0f32; v * n * d];
    for vi in 0..v {
        for ni in 0..n {
            for di in 0..d {
                src[(vi * n + ni) * d + di] = (100 * vi + 10 * ni + di) as f32;
            }
        }
    }
    let x = gpu.storage_init("x", &src);
    let y = gpu.storage(src.len() as u64);
    // Params: a0, a1, a2, d.
    let step = gpu.step(0, &[&x, &y], &[1, v as u32, n as u32, d as u32], src.len() as u32);
    gpu.submit(&[], &[step]);
    let out = gpu.read(&y, src.len());

    for ni in 0..n {
        for vi in 0..v {
            for di in 0..d {
                let got = out[(ni * v + vi) * d + di];
                let want = (100 * vi + 10 * ni + di) as f32;
                assert_eq!(got, want, "n={ni} v={vi} d={di}");
            }
        }
    }
}
