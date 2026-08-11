// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::gdn::gdn_causal_conv1d_step` (the streaming, ring-buffer-state
//! causal depthwise Conv1d decode step) vs. `audio::conv::conv1d_fwd`'s
//! existing whole-sequence causal conv (`pad = K-1`, `stride=1`,
//! `dilation=1`, `groups=Cin` — the causal expression that module's own doc
//! names), on a real, non-degenerate shape (`N,C,K,L` all pairwise distinct).
//!
//! Run on both backends:
//!   cargo test -p brain-model --test causal_conv1d_step
//!   BRAIN_DEVICE=cpu cargo test -p brain-model --test causal_conv1d_step

use audio::conv::{conv1d_fwd, Conv1d, ConvKernels};
use data::rng::Lcg;
use model::gdn::{gdn_causal_conv1d_step, GdnConvIds, GdnConvShape};

const PIPES: &[(&str, &str)] = &[("conv1d", kernels::CONV1D), ("causal_conv1d_step", kernels::CAUSAL_CONV1D_STEP)];

#[test]
fn causal_conv1d_step_matches_conv1d_fwd() {
    let g = gpu_core::testgpu::dev(PIPES);
    let conv1d_idx = g.kernel_index("conv1d").unwrap_or_else(|| panic!("kernel 'conv1d' not registered"));
    let step_idx = g.kernel_index("causal_conv1d_step").unwrap_or_else(|| panic!("kernel 'causal_conv1d_step' not registered"));

    // Pairwise-distinct dims: N=2 sequences, C=5 channels (conv_dim), K=4
    // (Qwen3.5's actual linear_conv_kernel_dim), L=7 tokens.
    let (nn, cc, kk, ll) = (2usize, 5usize, 4usize, 7usize);

    let mut rng = Lcg::new(20260810);
    let w_h: Vec<f32> = rng.vec_scaled(cc * kk, 1.0); // [C, K] depthwise weight
    let x_h: Vec<f32> = rng.vec_scaled(nn * cc * ll, 1.0); // [N, C, L] (conv1d.wgsl's NCL layout)

    // ==================== Reference: conv1d_fwd, causal ====================
    let x_dev = g.storage_init("x", &x_h);
    let w_dev = g.storage_init("w", &w_h);
    let y_ref_dev = g.storage((nn * cc * ll) as u64);

    let conv_shape = Conv1d {
        n: nn as u32,
        cin: cc as u32,
        l: ll as u32,
        cout: cc as u32,
        k: kk as u32,
        stride: 1,
        pad: (kk - 1) as u32, // causal: left pad K-1, lo == l (module doc's own convention)
        dilation: 1,
        groups: cc as u32, // depthwise
        lo: ll as u32,
    };
    let conv_kernels = ConvKernels { fwd: conv1d_idx, dx: 0, dw: 0 };
    let ref_step = conv1d_fwd(&g, &conv_kernels, &conv_shape, &x_dev, &w_dev, &y_ref_dev);
    g.submit(&[], &[ref_step]);
    let y_ref = g.read(&y_ref_dev, nn * cc * ll);

    // ==================== Streaming: gdn_causal_conv1d_step, L times ====================
    let step_shape = GdnConvShape { n: nn as u32, c: cc as u32, k: kk as u32 };
    // hist MUST start zeroed (this module's doc: the same implicit left
    // zero-pad conv1d_fwd applies via its own `pad` parameter).
    let hist = g.storage_init("hist", &vec![0f32; step_shape.hist_len() as usize]);
    let ids = GdnConvIds { causal_conv1d_step: step_idx };

    let x_step = g.storage((nn * cc) as u64);
    let y_step = g.storage((nn * cc) as u64);

    let mut y_stream = vec![0f32; nn * cc * ll];
    for l in 0..ll {
        // x_seq[n,c,l] -> x_step[n,c] (conv1d.wgsl's NCL layout: idx = (n*C+c)*L+l).
        let x_step_h: Vec<f32> = (0..nn * cc).map(|nc| x_h[nc * ll + l]).collect();
        g.write_f32(&x_step, &x_step_h);

        let step = gdn_causal_conv1d_step(&g, &ids, &step_shape, &x_step, &w_dev, &hist, &y_step);
        g.submit(&[], &[step]);

        let y_step_h = g.read(&y_step, nn * cc);
        for nc in 0..nn * cc {
            y_stream[nc * ll + l] = y_step_h[nc];
        }
    }

    let tol = 1e-4;
    let mut worst = 0f64;
    for (i, (&got, &want)) in y_stream.iter().zip(&y_ref).enumerate() {
        let delta = (got as f64 - want as f64).abs();
        worst = worst.max(delta);
        assert!(delta < tol, "y[{i}]: got {got} want {want} (delta {delta})");
    }
    eprintln!("causal_conv1d_step_matches_conv1d_fwd: worst |delta| = {worst:e}");
}
