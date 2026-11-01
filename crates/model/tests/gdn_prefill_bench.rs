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

/// How long the back-to-back stress runs on each launch form of the recurrence.
const STRESS_SECONDS_A_FORM: u64 = 60;
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

/// A round of many short segments: more blocks than multiprocessors, the single-stage form of the warpgroup recurrence.
fn round_of_twelve() -> Vec<GdnSegmentSpan> {
    (0..12).map(|i| GdnSegmentSpan { row0: i * 70, rows: 70, slot: i }).collect()
}

/// One and two segments (up to two blocks a multiprocessor: the three-stage form of the warpgroup recurrence), of one to sixteen chunks and with
/// a short last chunk.
fn rounds_of_one_and_two() -> [Vec<GdnSegmentSpan>; 3] {
    [
        vec![GdnSegmentSpan { row0: 0, rows: 1000, slot: 0 }],
        vec![GdnSegmentSpan { row0: 0, rows: 130, slot: 0 }, GdnSegmentSpan { row0: 130, rows: 70, slot: 1 }],
        vec![GdnSegmentSpan { row0: 0, rows: 64, slot: 0 }, GdnSegmentSpan { row0: 64, rows: 320, slot: 1 }],
    ]
}

/// The warpgroup recurrence in each of its launch forms (a block a half of a head with a three-stage ring, a block a head on two warpgroups, a
/// single-stage block a half) against the `mma.sync` one over rounds of one to twelve segments, whole and short chunks.
#[test]
fn the_warpgroup_recurrence_agrees_with_the_mma_one_over_a_round_of_segments() {
    let _forced = FORCED_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    for spans in [round_of_three(), round_of_twelve()].into_iter().chain(rounds_of_one_and_two()) {
        gpu_core::force_candidate("gdn_prefill_loop", "mma");
        let mma = bench("mma", &spans);
        gpu_core::force_candidate("gdn_prefill_loop", "wg");
        for form in ["half", "full", "lean"] {
            gpu_core::force_candidate(cuda::GDN_LOOP_FORM_SLOT, form);
            let wg = bench(form, &spans);
            let (Some(mma), Some(wg)) = (&mma, wg) else { return };
            let (out, state) = (rel_rms(&wg.0, &mma.0, true), rel_rms(&wg.1, &mma.1, false));
            assert!(out < 4.0 / 512.0 && state < 4.0 / 512.0, "{} segments, {form}: output {out:.2e}, state {state:.2e}", spans.len());
        }
        gpu_core::clear_forced_candidates();
    }
}

#[test]
#[ignore = "a bench at the production shape, run by hand"]
fn the_serving_prefill_layer_at_8k_tokens_and_in_a_700_row_round() {
    let _forced = FORCED_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    let rounds: [(&str, Vec<GdnSegmentSpan>); 4] = [
        ("one 64-row segment", vec![GdnSegmentSpan { row0: 0, rows: 64, slot: 0 }]),
        ("one 320-row segment", vec![GdnSegmentSpan { row0: 0, rows: 320, slot: 0 }]),
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

/// The served 4096-row round as 1, 2, 3, 4 and 16 segments, on the selected (warpgroup) recurrence only: the shapes `ncu` profiles.
#[test]
#[ignore = "a bench at the production shape, run by hand"]
fn the_serving_prefill_layer_in_a_4096_row_round() {
    let _forced = FORCED_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    gpu_core::force_candidate("gdn_prefill_loop", "wg");
    for lens in [vec![3900u32, 196], vec![3500, 300, 296], vec![3000, 1096], vec![2600, 1496], vec![2048, 2048], vec![3584, 256, 128, 128], vec![1024, 1024, 1024, 1024]] {
        let mut row0 = 0;
        let spans: Vec<_> = lens.iter().enumerate().map(|(i, &rows)| { let s = GdnSegmentSpan { row0, rows, slot: i as u32 }; row0 += rows; s }).collect();
        bench(&format!("4096-row round of {lens:?} (wg recurrence)"), &spans);
    }
    for pieces in [1u32, 2, 3, 4, 16] {
        let rows = 4096 / pieces;
        let spans: Vec<_> = (0..pieces).map(|i| GdnSegmentSpan { row0: i * rows, rows, slot: i }).collect();
        bench(&format!("4096-row round as {pieces} segment(s) (wg recurrence)"), &spans);
    }
    gpu_core::clear_forced_candidates();
}

/// The four segment kernels of a layer (convolution, prep, recurrence, gated norm) launched back to back over a 4096-row round of two segments,
/// many layers at a time: every batch must end in the bytes of the first (the outputs poisoned and the state and history pools reset between
/// batches, so a launch that does not run cannot pass on the last one's results). Faults of the kind recorded in knowledge 410 show only in
/// uninstrumented back-to-back launches; it runs for `STRESS_SECONDS_A_FORM` on each launch form - compare `dmesg | grep -i xid` before and after.
#[test]
#[ignore = "a stress run of several minutes, by hand"]
fn the_segment_kernels_survive_back_to_back_launches() {
    let _forced = FORCED_SLOT.lock().unwrap_or_else(|e| e.into_inner());
    for form in ["half", "full", "lean", "alternate"] {
        stress_segment_kernels(form);
    }
}

fn stress_segment_kernels(form: &str) {
    let g = gpu_core::testgpu::dev(&[]);
    if g.caps().arch.compute_capability.is_none_or(|cc| cc < (9, 0)) {
        brain_testutil::skip_unavailable("a Hopper device");
        return;
    }
    gpu_core::force_candidate("gdn_prefill_loop", "wg");
    let spans = [GdnSegmentSpan { row0: 0, rows: 3000, slot: 0 }, GdnSegmentSpan { row0: 3000, rows: 1096, slot: 1 }];
    let n = 4096u32;
    let rows = u64::from(n);
    let table = GdnSegmentTable::new(&spans);
    let tbl = g.storage_uninit(table.words().len() as u64);
    g.write(&tbl, table.words());
    let mut rng = Lcg::new(5);
    let mixed = g.storage_init("qkv", &bf16_words(&rand(&mut rng, n as usize * CONV_DIM as usize, 1.0)));
    let conv_w = g.storage_init("conv_w", &rand(&mut rng, CONV_DIM as usize * 4, 0.5));
    let (hist0, state0) = (rand(&mut rng, 2 * CONV_DIM as usize * 3, 0.5), rand(&mut rng, 2 * (NVH * HD * HD) as usize, 0.05));
    let (hist, state) = (g.storage_init("hist", &hist0), g.storage_init("state", &state0));
    let (bproj, aproj) = (g.storage_init("b", &rand(&mut rng, n as usize * NVH as usize, 1.0)), g.storage_init("a", &rand(&mut rng, n as usize * NVH as usize, 1.0)));
    let (a_log, dt_bias) = (g.storage_init("a_log", &rand(&mut rng, NVH as usize, 1.0)), g.storage_init("dt", &rand(&mut rng, NVH as usize, 1.0)));
    let z = g.storage_init("z", &bf16_words(&rand(&mut rng, n as usize * VALUE_DIM as usize, 4.0)));
    let norm_w = g.storage_init("norm_w", &rand(&mut rng, HD as usize, 1.0));
    let (query, key, value) = (g.storage_uninit(rows * u64::from(KEY_DIM) / 2), g.storage_uninit(rows * u64::from(KEY_DIM) / 2), g.storage_uninit(rows * u64::from(VALUE_DIM) / 2));
    let sl = u64::from(table.chunks() * NVH) * 64;
    let (g_cs, exp_g_cs, u, wm, intra) = (g.storage_uninit(sl * 3), g.storage_uninit(sl), g.storage_uninit(sl * 128), g.storage_uninit(sl * 128), g.storage_uninit(sl * 64));
    let (qcm, kcm) = (g.storage_uninit(1), g.storage_uninit(1));
    let out = g.storage_uninit(rows * u64::from(VALUE_DIM) / 2);
    let (gated, q8, s8) = (g.storage_uninit(rows * u64::from(VALUE_DIM)), g.storage_uninit(rows * u64::from(VALUE_DIM) / 4), g.storage_uninit(rows * u64::from(NVH)));
    let layout = GdnTokenLayout { bhc: 0, c_len: 64, n_chunks: 0, heads: NVH, group: NVH / NKH, head_dim: HD, scale: 1.0 / (HD as f32).sqrt(), eps: 1e-6 };
    assert!(cuda::gdn_seg_loop_folds(&g, true));
    let conv_layout = GdnConvLayout { rows: n, conv_dim: CONV_DIM, seq_len: n, key_dim: KEY_DIM, value_dim: VALUE_DIM, kernel: 4 };
    let layers = || {
        let mut steps = Vec::new();
        for layer in 0..8 {
            // Alternate layers between two launch forms of the recurrence (the served rounds alternate them from round to round).
            gpu_core::force_candidate(cuda::GDN_LOOP_FORM_SLOT, if form == "alternate" { if layer % 2 == 0 { "full" } else { "half" } } else { form });
            steps.push(cuda::gdn_conv_silu_seg_step(&g, &[&mixed, &conv_w, &tbl, &hist, &query, &key, &value], conv_layout, &table, true, true, 1e-6).expect("conv"));
            steps.push(cuda::gdn_chunk_prep_seg_bf16_step(&g, &[&query, &key, &value, &bproj, &aproj, &a_log, &dt_bias, &tbl, &g_cs, &exp_g_cs, &u, &wm, &intra, &qcm, &kcm], layout, &table, true).expect("prep"));
            steps.push(cuda::gdn_chunk_loop_seg_bf16_step(&g, &[&key, &query, &wm, &u, &g_cs, &intra, &tbl, &state, &out], layout, &table, true, true).expect("loop"));
            steps.push(cuda::gdn_gated_norm_step(&g, &[&out, &norm_w, &z, &gated, &q8, &s8], n * NVH, 0, 1e-6, true, true).expect("norm"));
        }
        steps
    };
    let poisoned = [(&query, rows * u64::from(KEY_DIM) / 2), (&key, rows * u64::from(KEY_DIM) / 2), (&value, rows * u64::from(VALUE_DIM) / 2), (&u, sl * 128), (&wm, sl * 128), (&intra, sl * 64), (&out, rows * u64::from(VALUE_DIM) / 2), (&q8, rows * u64::from(VALUE_DIM) / 4), (&s8, rows * u64::from(NVH))];
    let (hist0_bits, state0_bits): (Vec<u32>, Vec<u32>) = (hist0.iter().map(|v| v.to_bits()).collect(), state0.iter().map(|v| v.to_bits()).collect());
    let reset = || {
        for (buf, len) in poisoned {
            g.write(buf, &vec![0xffff_ffffu32; len as usize]);
        }
        g.write(&hist, &hist0_bits);
        g.write(&state, &state0_bits);
    };
    let digest = || -> Vec<u32> { poisoned.iter().map(|(b, len)| (*b, *len as usize)).chain([(&state, state0.len())]).flat_map(|(b, len)| g.read(b, len).into_iter().map(f32::to_bits)).collect() };
    reset();
    g.submit(&[], &layers());
    g.poll_wait();
    let first = digest();
    // The scratch's slots hold only what the kernels write (a short chunk and the fp16 u leave words alone); the outputs must hold no poison.
    for (name, buf, len) in [("query", &query, rows * u64::from(KEY_DIM) / 2), ("key", &key, rows * u64::from(KEY_DIM) / 2), ("value", &value, rows * u64::from(VALUE_DIM) / 2), ("out", &out, rows * u64::from(VALUE_DIM) / 2), ("q8", &q8, rows * u64::from(VALUE_DIM) / 4), ("s8", &s8, rows * u64::from(NVH))] {
        assert!(!g.read(buf, len as usize).iter().any(|w| w.to_bits() == 0xffff_ffff), "{name}: the first batch left poison behind");
    }
    let start = std::time::Instant::now();
    let mut batches = 0u64;
    while start.elapsed().as_secs() < STRESS_SECONDS_A_FORM {
        reset();
        g.submit(&[], &layers());
        g.poll_wait();
        assert!(digest() == first, "batch {batches}: the results differ from the first batch's");
        batches += 1;
    }
    gpu_core::clear_forced_candidates();
    println!("{form}: {batches} batches of 8 layers (convolution, prep, recurrence, norm), identical results");
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
/// rms error (the cumulative decay, which the two forms sum in a different order, to a few fp32 ulps). The normalised rows the two forms hand the
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
    let report = [("g_cs", err(&fast.0, &exact.0), 1e-5), ("u", err(&fast.1, &exact.1), 4.0 / 512.0), ("w", err(&fast.2, &exact.2), 4.0 / 512.0), ("intra", err(&fast.3, &exact.3), 4.0 / 512.0)];
    for (name, e, limit) in report {
        println!("   prep {name}: relative rms error {e:.3e} (limit {limit:.1e})");
        assert!(e <= limit, "{name}: {e:.3e} > {limit:.1e}");
    }
}
