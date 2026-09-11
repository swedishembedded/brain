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

/// Both RMSNorm kernels this model registers must honour the epsilon the
/// model actually passes.
///
/// `block::rms_variant` picks between the per-element reference kernel and the
/// cooperative one-workgroup-per-row kernel purely from device capabilities,
/// so a device that reports `workgroup_reductions` runs a different kernel
/// than one that does not - and TimesFM-3's RMSNorm epsilon is
/// `f32::EPSILON`, roughly an order of magnitude below the 1e-6 that
/// `rmsnorm.wgsl` hardcodes. A reference kernel that ignored the epsilon
/// parameter would therefore normalize differently on a CPU backend than on a
/// GPU one, silently, for the same weights and input.
///
/// So this gates BOTH registered indices, not whichever one the current device
/// would select, and against a HOST reference rather than against each other
/// (two kernels wrong the same way would agree). The `dim`s are the two this
/// model's tapes actually dispatch at: `model_dims` for the sublayer norms and
/// `head_dim` for QK-norm, where a small row width makes the epsilon's
/// contribution largest.
#[test]
fn both_registered_rmsnorm_kernels_honour_the_configured_epsilon() {
    if skip() {
        return;
    }
    const RMSNORM: usize = 5;
    const RMSNORM_ROWS: usize = 6;
    let cfg = timesfm3::Timesfm3Config::tiny();
    let gpu = gpu_core::testgpu::dev(timesfm3::model::PIPELINES);
    // A row scale small enough that `mean(x^2)` is comparable to the epsilon
    // itself: that is where an epsilon of 1e-6 instead of `f32::EPSILON`
    // changes the answer by percent, not by rounding.
    for &(rows, dim) in &[(4usize, cfg.model_dims), (8, cfg.head_dim)] {
        let x: Vec<f32> = (0..rows * dim).map(|i| 1.0e-3 * (i as f32 * 0.7 + 0.1).sin()).collect();
        let w: Vec<f32> = (0..dim).map(|i| 0.5 * (i as f32 * 0.31 + 0.2).cos()).collect();
        let want = model::hostmath::rmsnorm_rows(&x, &w, rows, dim, cfg.rms_norm_eps);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);

        let xb = gpu.storage_init("rms_eps_x", &x);
        let wb = gpu.storage_init("rms_eps_w", &w);
        for &(kind, threads, what) in
            &[(RMSNORM, rows as u32, "reference"), (RMSNORM_ROWS, rows as u32 * 64, "cooperative")]
        {
            let ob = gpu.storage((rows * dim) as u64);
            let params = [dim as u32, rows as u32, gpu_core::f(cfg.rms_norm_eps)];
            gpu.submit(&[], &[gpu.step(kind, &[&xb, &wb, &ob], &params, threads)]);
            let got = gpu.read(&ob, rows * dim);
            let e = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
            assert!(e <= 2e-5, "{what} rmsnorm ({rows}x{dim}): relative error {e:e} exceeds 2e-5");
        }
    }
}
