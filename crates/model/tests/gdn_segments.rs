// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Gated DeltaNet mixer over all the prompt segments of a round in one pass per layer
//! (`model::gdn_segments`) against the per-segment path it replaces: each segment run on its own state row as
//! pieces of full chunks and a short remainder, continuing a non-zero recurrent state and convolution history.
//! The gated output rows, every pool state row and every history row agree, rows outside the segments are
//! untouched in the pools, and a round is three launches of the segment kernels instead of a dozen per piece.
//!
//! Swedish Embedded AB implements solutions for serving many concurrent requests on one accelerator for its
//! clients. If your team needs expertise in batching the recurrent layers of a language model across requests
//! without a launch per request, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The kernels need a CUDA device with bf16 tensor cores; elsewhere this reports and returns.

mod gdn_kit;

use data::rng::Lcg;
use gdn_kit::{mixer_ids, DeviceWeights, Host, KERNELS};
use gpu_core::Gpu;
use model::gdn::{GdnFastIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_stream_fwd, GdnMixerShape, GdnStream};
use model::gdn_segments::{gdn_mixer_segments_fwd, GdnSegmentPools, GdnSegmentRound, SegmentSpan};

const HEADS: u32 = 6;

fn shape(t: u32, chunk: u32) -> GdnMixerShape {
    GdnMixerShape { gdn: GdnShape { b: 1, h: HEADS, t, dk: 128, dv: 128, chunk }, nkh: 2, conv_kernel: 4, rms_eps: 1e-6 }
}

fn max_rel(a: &[f32], reference: &[f32]) -> f64 {
    let scale = reference.iter().fold(1.0f64, |m, v| m.max(f64::from(v.abs())));
    a.iter().zip(reference).fold(0.0f64, |m, (x, y)| m.max((f64::from(*x) - f64::from(*y)).abs() / scale))
}

#[test]
fn a_round_of_segments_equals_each_segment_on_its_own_state_row() {
    let kernels: &'static [(&'static str, &'static str)] = {
        let mut v = KERNELS.to_vec();
        v.push(("gdn_ut_fwd", kernels::GDN_UT_FWD));
        v.push(("bmm_tiled", kernels::BMM_TILED));
        Box::leak(v.into_boxed_slice())
    };
    let Ok(g) = Gpu::try_new_cuda(kernels) else {
        brain_testutil::skip_unavailable("a CUDA device");
        return;
    };
    if !model::gdn::use_fast_kernels(&g) || g.caps().arch.compute_capability.is_none_or(|cc| cc < (8, 0)) {
        brain_testutil::skip_unavailable("a device with bf16 tensor cores");
        return;
    }
    gpu_core::force_candidate("gdn_prefill", "bf16");

    // Three decode rows lead the round; then segments of a short remainder only, whole chunks only, and both, in
    // pool rows that are neither adjacent nor in order.
    let lead = 3u32;
    let segs = [(70u32, 4u32), (64, 1), (21, 6), (130, 0)];
    let mut spans = Vec::new();
    let mut row = lead;
    for (rows, slot) in segs {
        spans.push(SegmentSpan { row0: row, rows, slot });
        row += rows;
    }
    let n = row;
    let s = shape(n, 64);
    let h = Host::random(&s, 20261031); // seed: fixed fixture so a failure reproduces on the same values
    let mut ids = mixer_ids(&g);
    ids.chunk.fast = Some(GdnFastIds { ut_fwd: gdn_kit::idx(&g, "gdn_ut_fwd"), bmm_tiled: gdn_kit::idx(&g, "bmm_tiled"), tensor_core: true });
    let dw = DeviceWeights::upload(&g, &s, &h);
    let w = dw.weights();

    let (conv_dim, value_dim) = (s.conv_dim() as usize, s.value_dim() as usize);
    let (sl, hl) = (HEADS as usize * 128 * 128, conv_dim * 3);
    let pool_slots = 8usize;
    let mut rng = Lcg::new(7); // seed: fixed fixture
    let state_h = rng.vec_scaled(pool_slots * sl, 0.05);
    let hist_h = rng.vec_scaled(pool_slots * hl, 1.0);

    let up = |name: &str, v: &[f32]| g.storage_init(name, v);
    let (mixed, b, a, z) = (up("mixed_qkv", &h.mixed_qkv), up("bproj", &h.bproj), up("aproj", &h.aproj), up("z", &h.z));

    // Reference: each segment alone on a working copy of its pool row, as pieces of full chunks and a remainder.
    let mut want_gated = vec![0.0f32; n as usize * value_dim];
    let mut want_state = state_h.clone();
    let mut want_hist = hist_h.clone();
    for sp in &spans {
        let (slot, rows, row0) = (sp.slot as usize, sp.rows as usize, sp.row0 as usize);
        let state = up("work_state", &state_h[slot * sl..(slot + 1) * sl]);
        let hist = up("work_hist", &hist_h[slot * hl..(slot + 1) * hl]);
        let full = rows / 64 * 64;
        let pieces = [(0usize, full, 64u32), (full, rows - full, (rows - full) as u32)];
        for (off, piece_rows, chunk) in pieces.into_iter().filter(|p| p.1 > 0) {
            let take = |v: &[f32], width: usize| v[(row0 + off) * width..(row0 + off + piece_rows) * width].to_vec();
            let (q, bb, aa, zz) = (up("q", &take(&h.mixed_qkv, conv_dim)), up("b", &take(&h.bproj, HEADS as usize)), up("a", &take(&h.aproj, HEADS as usize)), up("z", &take(&h.z, value_dim)));
            let ps = shape(piece_rows as u32, chunk);
            let (gated, _) = gdn_mixer_stream_fwd(&g, &ids, &ps, &w, &q, &bb, &aa, &zz, piece_rows as u32, false, Some(GdnStream { state: &state, hist: &hist }));
            let out = g.read(&gated, piece_rows * value_dim);
            want_gated[(row0 + off) * value_dim..(row0 + off + piece_rows) * value_dim].copy_from_slice(&out);
        }
        want_state[slot * sl..(slot + 1) * sl].copy_from_slice(&g.read(&state, sl));
        want_hist[slot * hl..(slot + 1) * hl].copy_from_slice(&g.read(&hist, hl));
    }

    // Under test: the whole round at once on the pools.
    let state_pool = up("state_pool", &state_h);
    let hist_pool = up("hist_pool", &hist_h);
    let round = GdnSegmentRound::new(&g, &spans).expect("the segment kernels are served on a bf16-prefill device");
    g.reset_kernel_times();
    g.set_kernel_timing(true);
    let out = gdn_mixer_segments_fwd(&g, &ids, &s, &w, &round, &GdnSegmentPools { state: &state_pool, hist: &hist_pool }, &mixed, &b, &a, &z, n, lead, true, false).expect("served");
    let gated = out.gated.clone();
    g.poll_wait();
    let ran: Vec<String> = g.kernel_times().unwrap_or_default().into_iter().map(|r| r.0).collect();
    g.set_kernel_timing(false);
    gpu_core::clear_forced_candidates();
    for kernel in ["gdn_conv_silu_seg", "gdn_chunk_prep_seg_bf16", "gdn_chunk_loop_seg_bf16", "gdn_gated_norm"] {
        assert_eq!(ran.iter().filter(|k| k.contains(kernel)).count(), 1, "{kernel} is one launch for the round; ran {ran:?}");
    }

    let got = g.read(&gated, n as usize * value_dim);
    let from = lead as usize * value_dim;
    let e_out = max_rel(&got[from..], &want_gated[from..]);
    let e_state = max_rel(&g.read(&state_pool, pool_slots * sl), &want_state);
    let e_hist = max_rel(&g.read(&hist_pool, pool_slots * hl), &want_hist);
    println!("segments vs per-segment path: gated {e_out:.2e}, state {e_state:.2e}, history {e_hist:.2e}");
    assert!(want_gated[from..].iter().any(|v| v.abs() > 1e-3), "degenerate fixture: forward output is ~0");
    assert!(e_out <= 1e-5, "gated rows differ by {e_out:.3e}");
    assert!(e_state <= 1e-5, "state rows differ by {e_state:.3e}");
    assert_eq!(e_hist, 0.0, "history rows are copies of inputs and must match exactly");

    // The E4M3 rows the pass wrote beside the product are what the standalone quantiser makes of it (rows of the segments).
    use gpu_core::provider::library::{quantize_e4m3_rows, Operand};
    let (q, sc) = out.e4m3.expect("asked for");
    let mut steps = Vec::new();
    let (q_ref, s_ref) = quantize_e4m3_rows(&g, &mut steps, &Operand { buf: &out.gated, range: (0, u64::from(n) * value_dim as u64) }, n, value_dim as u32).expect("offered");
    g.submit(&[], &steps);
    g.poll_wait();
    let (from_w, from_s) = (lead as usize * value_dim / 4, lead as usize * value_dim / 128);
    let bits = |v: Vec<f32>| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(g.read(&q, n as usize * value_dim / 4))[from_w..], bits(g.read(&q_ref, n as usize * value_dim / 4))[from_w..], "E4M3 bytes differ");
    assert_eq!(bits(g.read(&sc, n as usize * value_dim / 128))[from_s..], bits(g.read(&s_ref, n as usize * value_dim / 128))[from_s..], "E4M3 scales differ");
}

/// The value of an E4M3 byte (4 exponent bits, bias 7, 3 mantissa bits, no infinities).
fn e4m3_value(b: u8) -> f32 {
    let (sign, exp, man) = (if b & 0x80 != 0 { -1.0f32 } else { 1.0 }, i32::from((b >> 3) & 0xf), f32::from(b & 7));
    if exp == 0 { sign * man / 8.0 * 2f32.powi(-6) } else { sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7) }
}

/// `v` rounded to bf16 (nearest even) as an f32 holding the same value.
fn bf16_exact(v: &[f32]) -> Vec<f32> {
    v.iter().map(|f| f32::from_bits(((f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16) << 16)).collect()
}

/// `v` packed two bf16 to a word (low half first), the layout of bf16 rows, as the f32-typed words the upload takes.
fn bf16_words(v: &[f32]) -> Vec<f32> {
    let bits = |f: f32| (f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16;
    v.chunks(2).map(|p| f32::from_bits(bits(p[0]) | (bits(p[1]) << 16))).collect()
}

/// The round in bf16 (the FP8 projections' bf16 outputs in, bf16 between the kernels, the E4M3 rows out and no fp32 product
/// stored) against the same round in fp32 on the same bf16-valued projections: the E4M3 rows it writes dequantize to the
/// fp32 round's product within what the bf16 intermediates cost, and the recurrent state it leaves agrees to bf16 resolution.
#[test]
fn a_round_in_bf16_equals_the_fp32_round_to_bf16_resolution() {
    let kernels: &'static [(&'static str, &'static str)] = {
        let mut v = KERNELS.to_vec();
        v.push(("gdn_ut_fwd", kernels::GDN_UT_FWD));
        v.push(("bmm_tiled", kernels::BMM_TILED));
        Box::leak(v.into_boxed_slice())
    };
    let Ok(g) = Gpu::try_new_cuda(kernels) else {
        brain_testutil::skip_unavailable("a CUDA device");
        return;
    };
    if !model::gdn::use_fast_kernels(&g) || g.caps().arch.compute_capability.is_none_or(|cc| cc < (8, 0)) {
        brain_testutil::skip_unavailable("a device with bf16 tensor cores");
        return;
    }
    gpu_core::force_candidate("gdn_prefill", "bf16");
    let lead = 3u32;
    let segs = [(70u32, 4u32), (64, 1), (21, 6), (130, 0)];
    let mut spans = Vec::new();
    let mut row = lead;
    for (rows, slot) in segs {
        spans.push(SegmentSpan { row0: row, rows, slot });
        row += rows;
    }
    let n = row;
    let s = shape(n, 64);
    let h = Host::random(&s, 20261101); // seed: fixed fixture so a failure reproduces on the same values
    let mut ids = mixer_ids(&g);
    ids.chunk.fast = Some(GdnFastIds { ut_fwd: gdn_kit::idx(&g, "gdn_ut_fwd"), bmm_tiled: gdn_kit::idx(&g, "bmm_tiled"), tensor_core: true });
    let dw = DeviceWeights::upload(&g, &s, &h);
    let w = dw.weights();
    let (conv_dim, value_dim) = (s.conv_dim() as usize, s.value_dim() as usize);
    let (sl, hl) = (HEADS as usize * 128 * 128, conv_dim * 3);
    let pool_slots = 8usize;
    let mut rng = Lcg::new(11); // seed: fixed fixture
    let state_h = rng.vec_scaled(pool_slots * sl, 0.05);
    let hist_h = bf16_exact(&rng.vec_scaled(pool_slots * hl, 1.0));
    let up = |name: &str, v: &[f32]| g.storage_init(name, v);
    let (qkv_v, z_v) = (bf16_exact(&h.mixed_qkv), bf16_exact(&h.z));
    let (b, a) = (up("bproj", &h.bproj), up("aproj", &h.aproj));
    let round = GdnSegmentRound::new(&g, &spans).expect("served");

    let run = |bf16: bool| {
        let (state, hist) = (up("state_pool", &state_h), up("hist_pool", &hist_h));
        let (mixed, z) = if bf16 { (up("mixed16", &bf16_words(&qkv_v)), up("z16", &bf16_words(&z_v))) } else { (up("mixed32", &qkv_v), up("z32", &z_v)) };
        let out = gdn_mixer_segments_fwd(&g, &ids, &s, &w, &round, &GdnSegmentPools { state: &state, hist: &hist }, &mixed, &b, &a, &z, n, lead, true, bf16).unwrap_or_else(|| panic!("the round was not served (bf16: {bf16})"));
        g.poll_wait();
        (out, g.read(&state, pool_slots * sl), g.read(&hist, pool_slots * hl))
    };
    let (f32_out, f32_state, f32_hist) = run(false);
    let (bf_out, bf_state, bf_hist) = run(true);
    gpu_core::clear_forced_candidates();

    let from = lead as usize * value_dim;
    let truth = &g.read(&f32_out.gated, n as usize * value_dim)[from..];
    let dequant = |q: &DeviceBufferPair| -> Vec<f32> {
        let words = g.read(&q.0, n as usize * value_dim / 4);
        let scales = g.read(&q.1, n as usize * value_dim / 128);
        (from..n as usize * value_dim).map(|i| e4m3_value((words[i / 4].to_bits() >> (8 * (i % 4))) as u8) * scales[i / 128]).collect()
    };
    let rms = |a: &[f32], b: &[f32]| (a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt() / (b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
    let (e_f32, e_bf16) = (rms(&dequant(&f32_out.e4m3.clone().expect("asked for")), truth), rms(&dequant(&bf_out.e4m3.clone().expect("asked for")), truth));
    let e_state = max_rel(&bf_state, &f32_state);
    println!("E4M3 rows against the fp32 product: fp32 round {e_f32:.3e}, bf16 round {e_bf16:.3e}; state {e_state:.2e}");
    assert!(truth.iter().any(|v| v.abs() > 1e-3), "degenerate fixture: forward output is ~0");
    assert!(e_bf16 <= 1.5 * e_f32 + 5e-3, "the bf16 round's E4M3 rows are {e_bf16:.3e} from the product; the fp32 round's are {e_f32:.3e}");
    assert!(e_state <= 3e-2, "recurrent state differs by {e_state:.3e}");
    assert_eq!(bf_hist, f32_hist, "the convolution history holds bf16 values widened and must match exactly");
}

type DeviceBufferPair = (gpu_core::DeviceBuffer, gpu_core::DeviceBuffer);
