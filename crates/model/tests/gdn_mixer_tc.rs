// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Gated DeltaNet layer with the tensor-core chunk forward (query/key
//! normalisation, head repeat and chunk-major permutes folded into the prefix
//! kernel, tensor-core recurrence) against the same layer on the portable
//! dispatch sequence, at the real head width: the forward output, the gradients
//! of every input and the gradients of every weight agree to the TF32-split
//! bound, in the training form (which saves what the backward needs) and in the
//! inference form. A second test runs the layer in segments with a frozen convolution weight (a LoRA build), which
//! is the case the fused token-major convolution kernels serve end to end: the history it hands from one round to
//! the next, the gradient of that history, and the SiLU / split adjoint. (That the fold happened - the forward's five chunk-major
//! permutes, two normalisations and two head repeats are gone - is read off the
//! launch census of the training probe, recorded in `throughput-27b.md`.)
//!
//! Swedish Embedded AB implements tensor-core training kernels for linear-
//! attention language models for its clients. If your team needs expertise in
//! fusing the layout glue around a recurrent layer into its kernels without
//! changing the layer's gradients beyond a stated bound, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The kernels need a CUDA device with tensor cores; elsewhere this reports and
//! returns, and the portable sequence is what the other mixer tests gate.

mod gdn_kit;

use gdn_kit::{mixer_ids, DeviceWeights, Host, KERNELS};
use gpu_core::Gpu;
use model::gdn::{GdnFastIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_bwd, gdn_mixer_fwd, GdnMixerGrads, GdnMixerShape};
use model::gdn_segment::{gdn_mixer_seg_bwd, gdn_mixer_seg_fwd, GdnMixerInputs};

fn shape(t: u32, chunk: u32) -> GdnMixerShape {
    GdnMixerShape { gdn: GdnShape { b: 1, h: 6, t, dk: 128, dv: 128, chunk }, nkh: 2, conv_kernel: 4, rms_eps: 1e-6 }
}

struct Run {
    gated: Vec<f32>,
    grads: Vec<(&'static str, Vec<f32>)>,
}

fn run(g: &Gpu, s: &GdnMixerShape, h: &Host, tensor_core: bool, train: bool) -> Run {
    let n = s.gdn.t;
    let mut ids = mixer_ids(g);
    if tensor_core {
        ids.chunk.fast = Some(GdnFastIds { ut_fwd: gdn_kit::idx(g, "gdn_ut_fwd"), bmm_tiled: gdn_kit::idx(g, "bmm_tiled"), tensor_core: true });
    }
    let dw = DeviceWeights::upload(g, s, h);
    let up = |name: &str, v: &[f32]| g.storage_init(name, v);
    let (mixed, b, a, z, d_gated) = (up("mixed_qkv", &h.mixed_qkv), up("bproj", &h.bproj), up("aproj", &h.aproj), up("z", &h.z), up("d_gated", &h.d_gated));
    let (gated, acts) = gdn_mixer_fwd(g, &ids, s, &dw.weights(), &mixed, &b, &a, &z, n, train);
    let mut r = Run { gated: g.read(&gated, h.z.len()), grads: vec![] };
    if let Some(acts) = acts {
        let (dm, db, da, dz) = gdn_mixer_bwd(g, &ids, s, &dw.weights(), &dw.grads(), &acts, &d_gated, n);
        r.grads = vec![("d_mixed_qkv", g.read(&dm, h.mixed_qkv.len())), ("d_bproj", g.read(&db, h.bproj.len())), ("d_aproj", g.read(&da, h.aproj.len())), ("d_z", g.read(&dz, h.z.len()))];
        let wg = dw.read_grads(g, h);
        for (name, v) in ["conv1d.weight", "A_log", "dt_bias", "norm.weight"].into_iter().zip(wg) {
            r.grads.push((name, v));
        }
    }
    r
}

/// Largest absolute difference relative to the larger of 1 and the reference's own magnitude.
fn rel_diff(a: &[f32], reference: &[f32]) -> f64 {
    let scale = reference.iter().fold(1.0f64, |m, v| m.max(v.abs() as f64));
    a.iter().zip(reference).fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs() / scale))
}

#[test]
fn the_tensor_core_layer_matches_the_portable_layer_forward_and_backward() {
    let kernels: &'static [(&'static str, &'static str)] = {
        let mut v = KERNELS.to_vec();
        v.push(("gdn_ut_fwd", kernels::GDN_UT_FWD));
        v.push(("bmm_tiled", kernels::BMM_TILED));
        Box::leak(v.into_boxed_slice())
    };
    let Ok(g) = Gpu::try_new_cuda(kernels) else {
        eprintln!("gdn_mixer_tc: no CUDA device on this box - nothing to run");
        return;
    };
    if !model::gdn::use_fast_kernels(&g) {
        eprintln!("gdn_mixer_tc: this device does not take the fast kernels - nothing to run");
        return;
    }
    // Two chunks per head at the real chunk of 64 rows, a longer sequence, and one short chunk of a size that is no
    // multiple of the MMA tile (the remainder of a prompt slice, which serving runs as a chunk of its own).
    for (t, chunk) in [(128u32, 64u32), (320, 64), (36, 36), (13, 13), (63, 63)] {
        let s = shape(t, chunk);
        // seed: fixed fixture so a failure reproduces on the same values
        let h = Host::random(&s, 20261016);
        for train in [true, false] {
            let (want, got) = (run(&g, &s, &h, false, train), run(&g, &s, &h, true, train));
            assert!(want.gated.iter().any(|v| v.abs() > 1e-3), "degenerate fixture: forward output is ~0");
            let tol = 1e-4;
            let e = rel_diff(&got.gated, &want.gated);
            println!("t={t} chunk={chunk} train={train} gated: {e:.2e}");
            assert!(e <= tol, "t={t} train={train}: gated differs by {e:.3e}");
            for ((name, a), (_, b)) in got.grads.iter().zip(&want.grads) {
                let e = rel_diff(a, b);
                println!("t={t} chunk={chunk} train={train} {name}: {e:.2e}");
                assert!(e <= tol, "t={t} train={train}: {name} differs by {e:.3e}");
            }
        }
    }
}

/// One segmented layer, forward and backward, with the convolution weight frozen: the shape of a LoRA step.
fn run_segmented(g: &Gpu, s: &GdnMixerShape, h: &Host, tensor_core: bool, segment_tokens: u32) -> Vec<(&'static str, Vec<f32>)> {
    let mut ids = mixer_ids(g);
    if tensor_core {
        ids.chunk.fast = Some(GdnFastIds { ut_fwd: gdn_kit::idx(g, "gdn_ut_fwd"), bmm_tiled: gdn_kit::idx(g, "bmm_tiled"), tensor_core: true });
    }
    let dw = DeviceWeights::upload(g, s, h);
    let up = |name: &str, v: &[f32]| g.storage_init(name, v);
    let (mixed, b, a, z, d_gated) = (up("mixed_qkv", &h.mixed_qkv), up("bproj", &h.bproj), up("aproj", &h.aproj), up("z", &h.z), up("d_gated", &h.d_gated));
    let inp = GdnMixerInputs { mixed_qkv: &mixed, bproj: &b, aproj: &a, z: &z };
    let (gated, bounds) = gdn_mixer_seg_fwd(g, &ids, s, &dw.weights(), &inp, segment_tokens);
    let frozen = GdnMixerGrads { conv1d_weight: None, a_log: Some(&dw.a_log_g), dt_bias: Some(&dw.dt_bias_g), norm_weight: Some(&dw.norm_weight_g) };
    let (dm, db, da, dz) = gdn_mixer_seg_bwd(g, &ids, s, &dw.weights(), &frozen, &inp, &bounds, &d_gated);
    vec![
        ("gated", g.read(&gated, h.z.len())),
        ("d_mixed_qkv", g.read(&dm, h.mixed_qkv.len())),
        ("d_bproj", g.read(&db, h.bproj.len())),
        ("d_aproj", g.read(&da, h.aproj.len())),
        ("d_z", g.read(&dz, h.z.len())),
    ]
}

#[test]
fn the_fused_convolution_matches_the_portable_chain_in_a_segmented_layer_with_a_frozen_weight() {
    let kernels: &'static [(&'static str, &'static str)] = {
        let mut v = KERNELS.to_vec();
        v.push(("gdn_ut_fwd", kernels::GDN_UT_FWD));
        v.push(("bmm_tiled", kernels::BMM_TILED));
        Box::leak(v.into_boxed_slice())
    };
    let Ok(g) = Gpu::try_new_cuda(kernels) else {
        eprintln!("gdn_mixer_tc: no CUDA device on this box - nothing to run");
        return;
    };
    if !model::gdn::use_fast_kernels(&g) {
        eprintln!("gdn_mixer_tc: this device does not take the fast kernels - nothing to run");
        return;
    }
    // Three rounds of 128 rows (two chunks each) and an uneven tail round: every boundary carries a conv history.
    let s = shape(320, 64);
    // seed: fixed fixture so a failure reproduces on the same values
    let h = Host::random(&s, 20261017);
    for segment_tokens in [128u32, 192] {
        let (want, got) = (run_segmented(&g, &s, &h, false, segment_tokens), run_segmented(&g, &s, &h, true, segment_tokens));
        assert!(want[0].1.iter().any(|v| v.abs() > 1e-3), "degenerate fixture: forward output is ~0");
        for ((name, a), (_, b)) in got.iter().zip(&want) {
            let e = rel_diff(a, b);
            println!("segment_tokens={segment_tokens} {name}: {e:.2e}");
            assert!(e <= 1e-4, "segment_tokens={segment_tokens}: {name} differs by {e:.3e}");
        }
    }
}
