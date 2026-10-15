// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Running a Gated DeltaNet layer over a long sequence in SEGMENTS must give
//! the same forward output, the same input gradients and the same weight
//! gradients as the one-shot `gdn_mixer_fwd` / `gdn_mixer_bwd` pair, because
//! the only thing that changes is how much activation memory is live at once.
//! A segment boundary carries three things across it, and each one drops out
//! as an error that is localised to the rows next to the boundary (which a
//! whole-model comparison dilutes into noise):
//!
//! * the recurrent state forwards (outputs of the later segments),
//! * the gradient of that state backwards (inputs of the earlier segments),
//! * the causal conv's `K-1` row history forwards, and the gradient it sends
//!   back into the last `K-1` rows of the previous segment.
//!
//! The shapes are deliberately non-square and pairwise distinct (`h=2, dk=3,
//! dv=4, nkh=1`, conv kernel 3), the chunk is smaller than every segment so
//! the chunk recurrence is also exercised inside a segment, and one plan has
//! an uneven last segment.

mod gdn_kit;

use gdn_kit::{maxabs, mixer_ids, DeviceWeights, Host, KERNELS};
use gpu_core::Gpu;
use model::gdn::GdnShape;
use model::gdn_mixer::{gdn_mixer_bwd, gdn_mixer_fwd, GdnMixerShape};
use model::gdn_segment::{gdn_mixer_seg_bwd, gdn_mixer_seg_fwd, segment_spans, GdnMixerInputs};

fn shape(t: u32) -> GdnMixerShape {
    GdnMixerShape { gdn: GdnShape { b: 1, h: 2, t, dk: 3, dv: 4, chunk: 2 }, nkh: 1, conv_kernel: 3, rms_eps: 1e-6 }
}

/// Everything a layer hands back to its caller, host-side.
struct Run {
    gated: Vec<f32>,
    d_mixed_qkv: Vec<f32>,
    d_bproj: Vec<f32>,
    d_aproj: Vec<f32>,
    d_z: Vec<f32>,
    weight_grads: [Vec<f32>; 4],
}

fn upload(g: &Gpu, h: &Host) -> [gpu_core::DeviceBuffer; 5] {
    [
        g.storage_init("mixed_qkv", &h.mixed_qkv),
        g.storage_init("bproj", &h.bproj),
        g.storage_init("aproj", &h.aproj),
        g.storage_init("z", &h.z),
        g.storage_init("d_gated", &h.d_gated),
    ]
}

fn one_shot(g: &Gpu, s: &GdnMixerShape, h: &Host) -> Run {
    let n = s.gdn.t;
    let ids = mixer_ids(g);
    let dw = DeviceWeights::upload(g, s, h);
    let [mixed, b, a, z, d_gated] = upload(g, h);
    let (gated, acts) = gdn_mixer_fwd(g, &ids, s, &dw.weights(), &mixed, &b, &a, &z, n, true);
    let acts = acts.expect("train forward saves activations");
    let (dm, db, da, dz) = gdn_mixer_bwd(g, &ids, s, &dw.weights(), &dw.grads(), &acts, &d_gated, n);
    Run {
        gated: g.read(&gated, h.z.len()),
        d_mixed_qkv: g.read(&dm, h.mixed_qkv.len()),
        d_bproj: g.read(&db, h.bproj.len()),
        d_aproj: g.read(&da, h.aproj.len()),
        d_z: g.read(&dz, h.z.len()),
        weight_grads: dw.read_grads(g, h),
    }
}

fn segmented(g: &Gpu, s: &GdnMixerShape, h: &Host, segment_tokens: u32) -> Run {
    let ids = mixer_ids(g);
    let dw = DeviceWeights::upload(g, s, h);
    let [mixed, b, a, z, d_gated] = upload(g, h);
    let inp = GdnMixerInputs { mixed_qkv: &mixed, bproj: &b, aproj: &a, z: &z };
    let (gated, bounds) = gdn_mixer_seg_fwd(g, &ids, s, &dw.weights(), &inp, segment_tokens);
    let (dm, db, da, dz) = gdn_mixer_seg_bwd(g, &ids, s, &dw.weights(), &dw.grads(), &inp, &bounds, &d_gated);
    Run {
        gated: g.read(&gated, h.z.len()),
        d_mixed_qkv: g.read(&dm, h.mixed_qkv.len()),
        d_bproj: g.read(&db, h.bproj.len()),
        d_aproj: g.read(&da, h.aproj.len()),
        d_z: g.read(&dz, h.z.len()),
        weight_grads: dw.read_grads(g, h),
    }
}

#[test]
fn spans_tile_the_sequence_in_chunk_multiples_with_an_uneven_tail() {
    assert_eq!(segment_spans(10, 2, 4), vec![(0, 4), (4, 4), (8, 2)]);
    // A request that is not a chunk multiple rounds DOWN to one.
    assert_eq!(segment_spans(12, 2, 5), vec![(0, 4), (4, 4), (8, 4)]);
    // 0 and anything covering the whole sequence mean one segment.
    assert_eq!(segment_spans(12, 2, 0), vec![(0, 12)]);
    assert_eq!(segment_spans(12, 2, 64), vec![(0, 12)]);
}

#[test]
fn segmented_layer_matches_the_one_shot_layer_forward_and_backward() {
    let g = Gpu::new_cpu(KERNELS);
    let s = shape(10);
    // seed: fixed fixture so a failure reproduces on the same values
    let h = Host::random(&s, 20261016);
    let want = one_shot(&g, &s, &h);
    assert!(want.gated.iter().any(|v| v.abs() > 1e-3), "degenerate fixture: forward output is ~0");

    // [4,4,2] (uneven tail), [6,4], and the whole sequence as one segment.
    for segment_tokens in [4u32, 6, 0] {
        let got = segmented(&g, &s, &h, segment_tokens);
        let ctx = format!("segment_tokens={segment_tokens}");
        assert!(maxabs(&got.gated, &want.gated) < 1e-5, "{ctx}: gated {:e}", maxabs(&got.gated, &want.gated));
        assert!(maxabs(&got.d_mixed_qkv, &want.d_mixed_qkv) < 1e-4, "{ctx}: d_mixed_qkv {:e}", maxabs(&got.d_mixed_qkv, &want.d_mixed_qkv));
        assert!(maxabs(&got.d_bproj, &want.d_bproj) < 1e-4, "{ctx}: d_bproj {:e}", maxabs(&got.d_bproj, &want.d_bproj));
        assert!(maxabs(&got.d_aproj, &want.d_aproj) < 1e-4, "{ctx}: d_aproj {:e}", maxabs(&got.d_aproj, &want.d_aproj));
        assert!(maxabs(&got.d_z, &want.d_z) < 1e-5, "{ctx}: d_z {:e}", maxabs(&got.d_z, &want.d_z));
        for (i, name) in ["conv1d.weight", "A_log", "dt_bias", "norm.weight"].iter().enumerate() {
            let e = maxabs(&got.weight_grads[i], &want.weight_grads[i]);
            assert!(e < 1e-4, "{ctx}: {name} grad {e:e}");
        }
    }
}

/// Central finite differences of `L = sum(gated * d_gated)` through the
/// segmented layer, against its analytic backward. Probed elements sit on both
/// sides of every segment boundary of the `[4,4,2]` plan (rows 3/4 and 7/8)
/// and in the first rows, so a boundary gradient that is dropped or
/// mis-routed shows up as a probe that is off by O(1).
#[test]
fn segmented_backward_agrees_with_finite_differences_across_boundaries() {
    let g = Gpu::new_cpu(KERNELS);
    let s = shape(10);
    // seed: fixed fixture so a failure reproduces on the same values
    let h = Host::random(&s, 20261017);
    let base = segmented(&g, &s, &h, 4);

    let loss = |h: &Host| -> f64 {
        let got = segmented(&g, &s, h, 4);
        got.gated.iter().zip(&h.d_gated).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
    };
    let (conv_dim, value_dim, nvh) = (s.conv_dim() as usize, s.value_dim() as usize, s.gdn.h as usize);
    let eps = 1e-2f32;
    let mut worst = 0.0f64;
    let mut check = |name: &str, analytic: f32, plus: Host, minus: Host| {
        let fd = ((loss(&plus) - loss(&minus)) / (2.0 * eps as f64)) as f32;
        let tol = 3e-2 * fd.abs().max(analytic.abs()) + 3e-3;
        let err = (fd - analytic).abs();
        worst = worst.max((err / tol) as f64);
        assert!(err <= tol, "{name}: finite difference {fd} vs analytic {analytic}");
    };
    let bump = |h: &Host, f: &dyn Fn(&mut Host)| {
        let mut c = h.clone();
        f(&mut c);
        c
    };
    for row in [0usize, 3, 4, 7, 8, 9] {
        for ch in [0usize, conv_dim - 1, 5] {
            let i = row * conv_dim + ch;
            let p = bump(&h, &|c| c.mixed_qkv[i] += eps);
            let m = bump(&h, &|c| c.mixed_qkv[i] -= eps);
            check(&format!("mixed_qkv[{row},{ch}]"), base.d_mixed_qkv[i], p, m);
        }
        let i = row * nvh + (row % nvh);
        let p = bump(&h, &|c| c.bproj[i] += eps);
        let m = bump(&h, &|c| c.bproj[i] -= eps);
        check(&format!("bproj[{row}]"), base.d_bproj[i], p, m);
        let p = bump(&h, &|c| c.aproj[i] += eps);
        let m = bump(&h, &|c| c.aproj[i] -= eps);
        check(&format!("aproj[{row}]"), base.d_aproj[i], p, m);
        let i = row * value_dim + (row % value_dim);
        let p = bump(&h, &|c| c.z[i] += eps);
        let m = bump(&h, &|c| c.z[i] -= eps);
        check(&format!("z[{row}]"), base.d_z[i], p, m);
    }
    for k in 0..nvh {
        let p = bump(&h, &|c| c.a_log[k] += eps);
        let m = bump(&h, &|c| c.a_log[k] -= eps);
        check(&format!("A_log[{k}]"), base.weight_grads[1][k], p, m);
    }
    println!("segmented finite-difference worst error / tolerance = {worst:.3}");
}
