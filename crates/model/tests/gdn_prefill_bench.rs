// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The serving Gated DeltaNet prefill (convolution, prep, recurrence, gated norm: the four segment kernels of one layer)
//! at the production shape of the 27B (48 value heads, 16 key heads of 128, conv width 4), timed per kernel on the
//! device: one 8K-token segment and a 700-row round of three segments. The bench is run by hand
//! (`make test/one P=brain-model T=gdn_prefill_bench ARGS="--ignored --nocapture"`); the gate beside it holds the
//! warpgroup recurrence to the `mma.sync` one over a round.
//!
//! Swedish Embedded AB implements solutions for prefilling long prompts through recurrent layers at the tensor-core
//! roofline for its clients. If your team needs expertise in the throughput of linear-attention prefill on Hopper
//! accelerators, you can procure our services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use gpu_core::provider::cuda::{self, GdnConvLayout, GdnSegmentSpan, GdnSegmentTable, GdnTokenLayout};

const NVH: u32 = 48;
const NKH: u32 = 16;
const HD: u32 = 128;
const KEY_DIM: u32 = NKH * HD;
const VALUE_DIM: u32 = NVH * HD;
const CONV_DIM: u32 = 2 * KEY_DIM + VALUE_DIM;

fn bf16_words(v: &[f32]) -> Vec<f32> {
    let bits = |f: f32| (f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16;
    v.chunks(2).map(|p| f32::from_bits(bits(p[0]) | (bits(p[1]) << 16))).collect()
}

fn rand(rng: &mut Lcg, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale).collect()
}

/// Microseconds per kernel (the best of a few forwards) and the per-token-per-layer total.
fn bench(label: &str, spans: &[GdnSegmentSpan]) -> Option<(Vec<f32>, Vec<f32>)> {
    let g = gpu_core::testgpu::dev(&[]);
    if g.caps().arch.compute_capability.is_none_or(|cc| cc < (9, 0)) {
        brain_testutil::skip_unavailable("a Hopper device");
        return None;
    }
    eprintln!("device {} {:?}", g.kind(), g.caps().arch.compute_capability);
    let n: u32 = spans.iter().map(|s| s.row0 + s.rows).max().unwrap();
    let table = GdnSegmentTable::new(spans);
    let tbl = g.storage_uninit(table.words().len() as u64);
    g.write(&tbl, table.words());
    let mut rng = Lcg::new(7);
    let slots = spans.iter().map(|s| s.slot).max().unwrap() as usize + 1;
    let mixed = g.storage_init("qkv", &bf16_words(&rand(&mut rng, n as usize * CONV_DIM as usize, 1.0)));
    let conv_w = g.storage_init("conv_w", &rand(&mut rng, CONV_DIM as usize * 4, 0.5));
    let hist = g.storage_init("hist", &rand(&mut rng, slots * CONV_DIM as usize * 3, 0.5));
    let bproj = g.storage_init("b", &rand(&mut rng, n as usize * NVH as usize, 1.0));
    let aproj = g.storage_init("a", &rand(&mut rng, n as usize * NVH as usize, 1.0));
    let a_log = g.storage_init("a_log", &rand(&mut rng, NVH as usize, 1.0));
    let dt_bias = g.storage_init("dt", &rand(&mut rng, NVH as usize, 1.0));
    let state = g.storage_init("state", &rand(&mut rng, slots * (NVH * HD * HD) as usize, 0.05));
    let z = g.storage_init("z", &bf16_words(&rand(&mut rng, n as usize * VALUE_DIM as usize, 1.0)));
    let norm_w = g.storage_init("norm_w", &rand(&mut rng, HD as usize, 1.0));
    let rows = u64::from(n);
    let (query, key, value) = (g.storage_uninit(rows * u64::from(KEY_DIM) / 2), g.storage_uninit(rows * u64::from(KEY_DIM) / 2), g.storage_uninit(rows * u64::from(VALUE_DIM) / 2));
    let sl = u64::from(table.chunks() * NVH) * 64;
    let (g_cs, exp_g_cs, u, wm, intra) = (g.storage_uninit(sl * 3), g.storage_uninit(sl), g.storage_uninit(sl * 128), g.storage_uninit(sl * 128), g.storage_uninit(sl * 64));
    let (qcm, kcm) = (g.storage_uninit(sl * 128), g.storage_uninit(sl * 128));
    let out = g.storage_uninit(rows * u64::from(VALUE_DIM) / 2);
    let (gated, q8, s8) = (g.storage_uninit(rows * u64::from(VALUE_DIM)), g.storage_uninit(rows * u64::from(VALUE_DIM) / 4), g.storage_uninit(rows * u64::from(NVH)));

    let layout = GdnTokenLayout { bhc: 0, c_len: 64, n_chunks: 0, heads: NVH, group: NVH / NKH, head_dim: HD, scale: 1.0 / (HD as f32).sqrt(), eps: 1e-6 };
    let folds = cuda::gdn_seg_loop_folds(&g, true);
    let steps = || {
        vec![
            cuda::gdn_conv_silu_seg_step(&g, &[&mixed, &conv_w, &tbl, &hist, &query, &key, &value], GdnConvLayout { rows: n, conv_dim: CONV_DIM, seq_len: n, key_dim: KEY_DIM, value_dim: VALUE_DIM, kernel: 4 }, &table, true, false, 1e-6).expect("conv"),
            cuda::gdn_chunk_prep_seg_bf16_step(&g, &[&query, &key, &value, &bproj, &aproj, &a_log, &dt_bias, &tbl, &g_cs, &exp_g_cs, &u, &wm, &intra, &qcm, &kcm], layout, &table, true).expect("prep"),
            if folds { cuda::gdn_chunk_loop_seg_bf16_step(&g, &[&key, &query, &wm, &u, &g_cs, &intra, &tbl, &state, &out], layout, &table, true, false) } else { cuda::gdn_chunk_loop_seg_bf16_step(&g, &[&kcm, &qcm, &wm, &u, &g_cs, &intra, &tbl, &state, &out], layout, &table, true, false) }.expect("loop"),
            cuda::gdn_gated_norm_step(&g, &[&out, &norm_w, &z, &gated, &q8, &s8], n * NVH, 0, 1e-6, true, true).expect("norm"),
        ]
    };
    g.submit(&[], &steps());
    g.poll_wait();
    g.set_kernel_timing(true);
    let mut best: std::collections::BTreeMap<String, f64> = Default::default();
    for _ in 0..5 {
        g.reset_kernel_times();
        g.submit(&[], &steps());
        g.poll_wait();
        for (name, ms, _) in g.kernel_times().unwrap_or_default() {
            let e = best.entry(name).or_insert(f64::MAX);
            *e = e.min(ms);
        }
    }
    g.set_kernel_timing(false);
    let result = (g.read(&out, (rows * u64::from(VALUE_DIM) / 2) as usize), g.read(&state, slots * (NVH * HD * HD) as usize));
    let total: f64 = best.values().sum();
    println!("== {label}: {n} rows");
    for (name, ms) in &best {
        println!("   {name:<40} {:8.1} us   {:.4} us/token", ms * 1e3, ms * 1e3 / f64::from(n));
    }
    println!("   TOTAL {:.1} us = {:.4} us/token/layer", total * 1e3, total * 1e3 / f64::from(n));
    Some(result)
}

/// Two bf16 word buffers (rows two to a word) as relative rms error.
fn rel_rms(a: &[f32], reference: &[f32], bf16: bool) -> f64 {
    let wide = |v: &[f32]| -> Vec<f64> {
        if bf16 {
            v.iter().flat_map(|w| [f64::from(f32::from_bits(w.to_bits() << 16)), f64::from(f32::from_bits(w.to_bits() & 0xffff_0000))]).collect()
        } else {
            v.iter().map(|x| f64::from(*x)).collect()
        }
    };
    let (a, r) = (wide(a), wide(reference));
    let num: f64 = a.iter().zip(&r).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = r.iter().map(|y| y.powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Both tests pin the `gdn_prefill_loop` candidate, a process-wide table.
static FORCED_SLOT: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn round_of_three() -> Vec<GdnSegmentSpan> {
    vec![GdnSegmentSpan { row0: 0, rows: 300, slot: 0 }, GdnSegmentSpan { row0: 300, rows: 256, slot: 1 }, GdnSegmentSpan { row0: 556, rows: 144, slot: 2 }]
}

#[test]
fn the_warpgroup_recurrence_agrees_with_the_mma_one_over_a_round_of_segments() {
    let _forced = FORCED_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let spans = round_of_three();
    gpu_core::force_candidate("gdn_prefill_loop", "mma");
    let mma = bench("mma", &spans);
    gpu_core::force_candidate("gdn_prefill_loop", "wg");
    let wg = bench("wg", &spans);
    gpu_core::clear_forced_candidates();
    let (Some(mma), Some(wg)) = (mma, wg) else { return };
    let (out, state) = (rel_rms(&wg.0, &mma.0, true), rel_rms(&wg.1, &mma.1, false));
    assert!(out < 4.0 / 512.0 && state < 4.0 / 512.0, "output {out:.2e}, state {state:.2e}");
}

#[test]
#[ignore = "a bench at the production shape, run by hand"]
fn the_serving_prefill_layer_at_8k_tokens_and_in_a_700_row_round() {
    let _forced = FORCED_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let rounds: [(&str, Vec<GdnSegmentSpan>); 2] = [
        ("one 8K segment", vec![GdnSegmentSpan { row0: 0, rows: 8192, slot: 0 }]),
        ("700-row round", round_of_three()),
    ];
    for (label, spans) in &rounds {
        let mut results = Vec::new();
        for body in ["mma", "wg"] {
            gpu_core::force_candidate("gdn_prefill_loop", body);
            results.extend(bench(&format!("{label} ({body} recurrence)"), spans));
        }
        gpu_core::clear_forced_candidates();
        if let [mma, wg] = &results[..] {
            println!("   wg vs mma: output {:.2e}, state {:.2e}", rel_rms(&wg.0, &mma.0, true), rel_rms(&wg.1, &mma.1, false));
        }
    }
}

#[test]
fn the_measured_selection_runs_the_recurrence_on_warpgroups_on_a_hopper_card() {
    let _forced = FORCED_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let g = gpu_core::testgpu::dev(&[]);
    if g.caps().arch.compute_capability.is_none_or(|cc| cc < (9, 0)) {
        brain_testutil::skip_unavailable("a Hopper device");
        return;
    }
    g.prepare_gdn_prefill();
    let choice = g.gdn_prefill_choice().expect("measured");
    println!("measured selection: {choice:?}");
    assert!(choice.bf16 && choice.wg, "{choice:?}");
}

fn unpack_bf16(words: &[f32], n: usize) -> Vec<f32> {
    words.iter().flat_map(|w| [f32::from_bits(w.to_bits() << 16), f32::from_bits(w.to_bits() & 0xffff_0000)]).take(n).collect()
}

/// The bf16 prep (bf16 tensor-core products, `gdn_chunk_prep_tok_bf16`) against the exact one (three-pass TF32,
/// `gdn_chunk_prep_tok_tc`) on the same inputs: every tensor the recurrence reads agrees to a few bf16 units of relative
/// rms error (the cumulative decay, which both forms compute alike, exactly). The normalised rows the two forms hand the
/// recurrence are different tensors (the bf16 form scales them), so they are not compared.
#[test]
fn the_bf16_prep_agrees_with_the_exact_prep_to_bf16_resolution() {
    let g = gpu_core::testgpu::dev(&[]);
    if g.caps().arch.compute_capability.is_none_or(|cc| cc < (8, 0)) {
        brain_testutil::skip_unavailable("a device with bf16 tensor cores");
        return;
    }
    let (h, t, c, group) = (NVH, 1024u32, 64u32, NVH / NKH);
    let n_chunks = t / c;
    let bhc = h * n_chunks;
    let layout = GdnTokenLayout { bhc, c_len: c, n_chunks, heads: h, group, head_dim: HD, scale: 1.0 / (HD as f32).sqrt(), eps: 1e-6 };
    let mut rng = Lcg::new(11);
    let (ts, hs) = (t as usize, h as usize);
    let q = g.storage_init("q", &rand(&mut rng, ts * (hs / group as usize) * 128, 1.0));
    let k = g.storage_init("k", &rand(&mut rng, ts * (hs / group as usize) * 128, 1.0));
    let v = g.storage_init("v", &rand(&mut rng, ts * hs * 128, 1.0));
    let raw_g = g.storage_init("g", &rand(&mut rng, ts * hs, 0.2).iter().map(|x| -x.abs()).collect::<Vec<_>>());
    let beta = g.storage_init("beta", &rand(&mut rng, ts * hs, 0.5).iter().map(|x| x + 0.5).collect::<Vec<_>>());
    let len = bhc as usize * c as usize * 128;
    let run = |bf16: bool| {
        let (g_cs, exp_g_cs) = (g.storage(u64::from(bhc * c)), g.storage(u64::from(bhc * c)));
        let (u, w, intra) = (g.storage(len as u64), g.storage(len as u64), g.storage(u64::from(bhc * c * c)));
        let (qcm, kcm) = (g.storage(len as u64), g.storage(len as u64));
        let bufs = [&q, &k, &v, &raw_g, &beta, &g_cs, &exp_g_cs, &u, &w, &intra, &qcm, &kcm];
        let step = if bf16 { cuda::gdn_chunk_prep_tok_bf16_step(&g, &bufs, layout) } else { cuda::gdn_chunk_prep_tok_step(&g, &bufs, layout) }.expect("prep");
        g.submit(&[], &[step]);
        g.poll_wait();
        let wide = |b: &gpu_core::DeviceBuffer, n: usize| if bf16 { unpack_bf16(&g.read(b, n / 2), n) } else { g.read(b, n) };
        let cc = (bhc * c * c) as usize;
        (g.read(&g_cs, (bhc * c) as usize), g.read(&u, len), wide(&w, len), wide(&intra, cc))
    };
    let exact = run(false);
    let fast = run(true);
    let err = |a: &[f32], r: &[f32]| rel_rms(a, r, false);
    let report = [("g_cs", err(&fast.0, &exact.0), 1e-6), ("u", err(&fast.1, &exact.1), 4.0 / 512.0), ("w", err(&fast.2, &exact.2), 4.0 / 512.0), ("intra", err(&fast.3, &exact.3), 4.0 / 512.0)];
    for (name, e, limit) in report {
        println!("   prep {name}: relative rms error {e:.3e} (limit {limit:.1e})");
        assert!(e <= limit, "{name}: {e:.3e} > {limit:.1e}");
    }
}
