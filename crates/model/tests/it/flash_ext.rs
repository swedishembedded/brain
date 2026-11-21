// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::flash_ext`: causal GQA attention with an optional sliding window, sink logits, a distance-bank bias
//! and a log-position query scale, held to a host f64 oracle written straight from the reference math:
//!
//! * the forward (`out`, `lse`) for every combination of the options, at the window and `n_floor` boundaries;
//! * every gradient (`d_q`, `d_k`, `d_v`, `d_sink`, `d_rel`, `d_bank`) element by element against central finite
//!   differences of the f64 oracle, so the backward is the gradient of the REFERENCE forward and not merely of
//!   the kernels' own;
//! * one whole-sequence pass == a chunked prefill through a paged cache == token-by-token decode through the same
//!   cache, and == decode through a ring of `window` rows for a window layer, on f32 and bf16 planes.
//!
//! The kernels are the barrier-free reference tier, so these run on every backend, the CPU JIT included.
//!
//! Swedish Embedded AB implements memory-bounded attention for long-context models for its clients. If your team
//! needs expertise in verifying attention kernels against reference implementations then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::OnceLock;

use data::rng::Lcg;
use gpu_core::Gpu;
use model::flash_ext::{
    flash_ext_bwd, flash_ext_fwd, flash_ext_paged_fwd, log_position_scale, paged_index, AttnInputGrads, AttnInputs, AttnOptions, Contiguous, DistanceBias, FlashExtIds, HeadShape, KvPlanes, LogPositionScale, PagedKv, PagedSeq,
};
use model::kv_tier::{KvLayer, KvTier};

fn kernels() -> &'static [(&'static str, &'static str)] {
    static K: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    K.get_or_init(|| {
        let mut v = model::flash_ext::pipelines();
        model::ops::extend_kernel_list(&mut v, &model::gqa_mixer::ring_copy_pipelines());
        v
    })
    .as_slice()
}

fn device() -> (Gpu, FlashExtIds) {
    let g = gpu_core::testgpu::dev(kernels());
    let ids = FlashExtIds::resolve(&g);
    (g, ids)
}

/// One problem: shapes, options and its host inputs.
#[derive(Clone)]
struct Case {
    b: u32,
    t: u32,
    pos0: u32,
    heads: HeadShape,
    opts: AttnOptions,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    sink: Vec<f32>,
    rel: Vec<f32>,
    bank: Vec<f32>,
    d_out: Vec<f32>,
}

impl Case {
    fn new(b: u32, t: u32, pos0: u32, heads: HeadShape, opts: AttnOptions, rng: &mut Lcg) -> Case {
        let rows = (b * t) as usize;
        let (nh, nkv, hd) = (heads.n_heads as usize, heads.n_kv_heads as usize, heads.head_dim as usize);
        let (d_rel, extent) = opts.bias.map_or((0, 0), |b| (b.d_rel as usize, b.extent as usize));
        Case {
            b,
            t,
            pos0,
            heads,
            opts,
            q: rng.vec_scaled(rows * nh * hd, 1.0),
            k: rng.vec_scaled(rows * nkv * hd, 1.0),
            v: rng.vec_scaled(rows * nkv * hd, 1.0),
            sink: if opts.sink { rng.vec_scaled(rows * nh, 1.5) } else { Vec::new() },
            rel: rng.vec_scaled(rows * nh * d_rel, 0.7),
            bank: rng.vec_scaled(d_rel * extent, 1.0),
            d_out: rng.vec_scaled(rows * nh * hd, 1.0),
        }
    }

    fn rows(&self) -> usize {
        (self.b * self.t) as usize
    }
}

/// The f64 oracle: `(out, lse)` of `eager_attention_forward` with the window mask, the distance bias and the
/// concatenated sink, for the inputs of `c` (with `x` overriding the f32 values, for finite differences).
struct Host<'a> {
    q: &'a [f64],
    k: &'a [f64],
    v: &'a [f64],
    sink: &'a [f64],
    rel: &'a [f64],
    bank: &'a [f64],
}

fn host_fwd(c: &Case, x: &Host) -> (Vec<f64>, Vec<f64>) {
    let (nh, nkv, hd) = (c.heads.n_heads as usize, c.heads.n_kv_heads as usize, c.heads.head_dim as usize);
    let group = nh / nkv;
    let t = c.t as usize;
    let mut out = vec![0f64; c.rows() * nh * hd];
    let mut lse = vec![0f64; c.rows() * nh];
    for b in 0..c.b as usize {
        for i in 0..t {
            let row = b * t + i;
            let pos = c.pos0 + i as u32;
            let (tq, tb) = match c.opts.tau {
                Some(tau) => {
                    let f = 1.0 + f64::from(tau.alpha) * ((f64::from(pos) + 1.0) / f64::from(tau.n_floor)).max(1.0).ln();
                    (f, if tau.scales_bias { f } else { 1.0 })
                }
                None => (1.0, 1.0),
            };
            for h in 0..nh {
                let hkv = h / group;
                let qr = &x.q[(row * nh + h) * hd..][..hd];
                let mut logits = Vec::new();
                for j in 0..=i {
                    let dist = i - j;
                    if c.opts.window.is_some_and(|w| dist >= w as usize) {
                        continue;
                    }
                    let kr = &x.k[((b * t + j) * nkv + hkv) * hd..][..hd];
                    let dot: f64 = qr.iter().zip(kr).map(|(a, b)| a * b).sum();
                    let mut z = tq * f64::from(c.opts.scale) * dot;
                    if let Some(bias) = c.opts.bias {
                        if dist < bias.extent as usize {
                            let d_rel = bias.d_rel as usize;
                            let rel = &x.rel[(row * nh + h) * d_rel..][..d_rel];
                            z += tb * rel.iter().enumerate().map(|(r, w)| w * x.bank[r * bias.extent as usize + dist]).sum::<f64>();
                        }
                    }
                    logits.push((j, z));
                }
                let sink = c.opts.sink.then(|| x.sink[row * nh + h]);
                let m = logits.iter().map(|&(_, z)| z).chain(sink).fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = logits.iter().map(|&(_, z)| (z - m).exp()).chain(sink.map(|s| (s - m).exp())).sum();
                let o = &mut out[(row * nh + h) * hd..][..hd];
                for &(j, z) in &logits {
                    let p = (z - m).exp() / total;
                    let vr = &x.v[((b * t + j) * nkv + hkv) * hd..][..hd];
                    for d in 0..hd {
                        o[d] += p * vr[d];
                    }
                }
                lse[row * nh + h] = m + total.ln();
            }
        }
    }
    (out, lse)
}

fn wide(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

fn host_of(c: &Case) -> [Vec<f64>; 6] {
    [wide(&c.q), wide(&c.k), wide(&c.v), wide(&c.sink), wide(&c.rel), wide(&c.bank)]
}

fn eval(c: &Case, xs: &[Vec<f64>; 6]) -> (Vec<f64>, Vec<f64>) {
    host_fwd(c, &Host { q: &xs[0], k: &xs[1], v: &xs[2], sink: &xs[3], rel: &xs[4], bank: &xs[5] })
}

/// Central finite differences of `L = <d_out, out>` with respect to every element of input `which`.
fn fd_grad(c: &Case, which: usize) -> Vec<f64> {
    let mut xs = host_of(c);
    let d_out = wide(&c.d_out);
    let loss = |xs: &[Vec<f64>; 6]| -> f64 { eval(c, xs).0.iter().zip(&d_out).map(|(a, b)| a * b).sum() };
    let eps = 1e-5;
    (0..xs[which].len())
        .map(|e| {
            let x0 = xs[which][e];
            xs[which][e] = x0 + eps;
            let up = loss(&xs);
            xs[which][e] = x0 - eps;
            let down = loss(&xs);
            xs[which][e] = x0;
            (up - down) / (2.0 * eps)
        })
        .collect()
}

#[track_caller]
fn assert_close(what: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    let (worst, at) = got.iter().zip(want).enumerate().fold((0f64, 0usize), |(w, a), (e, (&g, &x))| {
        let d = (f64::from(g) - x).abs();
        if d > w || !g.is_finite() {
            (if g.is_finite() { d } else { f64::INFINITY }, e)
        } else {
            (w, a)
        }
    });
    assert!(worst <= tol * scale, "{what}: worst |diff| {worst:e} at {at} (got {}, want {}) over scale {scale:e}", got[at], want[at]);
}

struct DeviceRun {
    out: Vec<f32>,
    lse: Vec<f32>,
    grads: [Vec<f32>; 6],
}

fn u32s(g: &Gpu, words: &[u32]) -> gpu_core::DeviceBuffer {
    let b = g.storage(words.len() as u64);
    g.write(&b, words);
    b
}

fn upload(g: &Gpu, name: &str, x: &[f32]) -> gpu_core::DeviceBuffer {
    if x.is_empty() {
        g.storage(1)
    } else {
        g.storage_init(name, x)
    }
}

/// Forward and backward on the device through [`flash_ext_fwd`] / [`flash_ext_bwd`].
fn run_device(g: &Gpu, ids: &FlashExtIds, c: &Case) -> DeviceRun {
    let rows = c.rows() as u64;
    let (nh, nkv, hd) = (u64::from(c.heads.n_heads), u64::from(c.heads.n_kv_heads), u64::from(c.heads.head_dim));
    let (q, k, v, d_out) = (upload(g, "q", &c.q), upload(g, "k", &c.k), upload(g, "v", &c.v), upload(g, "d_out", &c.d_out));
    let (sink, rel, bank) = (upload(g, "sink", &c.sink), upload(g, "rel", &c.rel), upload(g, "bank", &c.bank));
    let inputs = AttnInputs { sink: c.opts.sink.then_some(&sink), rel: c.opts.bias.map(|_| &rel), bank: c.opts.bias.map(|_| &bank) };
    let layout = Contiguous { b: c.b, t: c.t, pos0: c.pos0 };
    let out = g.storage(rows * nh * hd);
    let lse = g.storage(rows * nh);
    g.submit(&[], &flash_ext_fwd(g, ids, &c.heads, &c.opts, &inputs, layout, &q, &k, &v, &out, &lse));

    let (d_q, d_k, d_v) = (g.storage(rows * nh * hd), g.storage(rows * nkv * hd), g.storage(rows * nkv * hd));
    let (d_sink, d_rel, d_bank) = (g.storage(c.sink.len().max(1) as u64), g.storage(c.rel.len().max(1) as u64), g.storage(c.bank.len().max(1) as u64));
    let grads = AttnInputGrads { d_sink: c.opts.sink.then_some(&d_sink), d_rel: c.opts.bias.map(|_| &d_rel), d_bank: c.opts.bias.map(|_| &d_bank) };
    g.submit(&[], &flash_ext_bwd(g, ids, &c.heads, &c.opts, &inputs, layout, &q, &k, &v, &out, &lse, &d_out, &d_q, &d_k, &d_v, &grads));
    DeviceRun {
        out: g.read(&out, (rows * nh * hd) as usize),
        lse: g.read(&lse, (rows * nh) as usize),
        grads: [
            g.read(&d_q, c.q.len()),
            g.read(&d_k, c.k.len()),
            g.read(&d_v, c.v.len()),
            if c.sink.is_empty() { Vec::new() } else { g.read(&d_sink, c.sink.len()) },
            if c.rel.is_empty() { Vec::new() } else { g.read(&d_rel, c.rel.len()) },
            if c.bank.is_empty() { Vec::new() } else { g.read(&d_bank, c.bank.len()) },
        ],
    }
}

/// Forward and every gradient of `c` against the oracle.
fn check_case(g: &Gpu, ids: &FlashExtIds, c: &Case, tag: &str) {
    let got = run_device(g, ids, c);
    let (out, lse) = eval(c, &host_of(c));
    assert_close(&format!("{tag} out"), &got.out, &out, 2e-5);
    assert_close(&format!("{tag} lse"), &got.lse, &lse, 2e-5);
    for (which, name) in ["d_q", "d_k", "d_v", "d_sink", "d_rel", "d_bank"].iter().enumerate() {
        let n = host_of(c)[which].len();
        if n == 0 {
            continue;
        }
        assert_close(&format!("{tag} {name}"), &got.grads[which], &fd_grad(c, which), 1e-4);
    }
}

fn opts(scale: f32, window: Option<u32>, sink: bool, bias: Option<DistanceBias>, tau: Option<LogPositionScale>) -> AttnOptions {
    AttnOptions { scale, window, sink, bias, tau }
}

/// Every combination of window, sink, bias (extent inside and beyond the sequence) and tau (with and without
/// scaling the bias), forward and all six gradients, on a GQA shape with `b > 1`.
#[test]
fn every_option_combination_matches_the_f64_oracle_forward_and_backward() {
    let (g, ids) = device();
    let heads = HeadShape { n_heads: 4, n_kv_heads: 2, head_dim: 8 };
    let scale = 8f32.powf(-0.5);
    let mut rng = Lcg::new(0xa77e_0001); // seed: fixed so a failing combination reproduces
    let biases = [None, Some(DistanceBias { extent: 3, d_rel: 3 }), Some(DistanceBias { extent: 16, d_rel: 2 })];
    for window in [None, Some(4)] {
        for sink in [false, true] {
            for bias in biases {
                let taus: Vec<Option<LogPositionScale>> = match bias {
                    // n_floor 4 around positions pos0 = 2 ..: the factor is 1 for the first rows and grows after.
                    Some(_) => vec![None, Some(LogPositionScale { alpha: 0.3, n_floor: 4.0, scales_bias: false }), Some(LogPositionScale { alpha: 0.3, n_floor: 4.0, scales_bias: true })],
                    None => vec![None, Some(LogPositionScale { alpha: 0.3, n_floor: 4.0, scales_bias: false })],
                };
                for tau in taus {
                    let o = opts(scale, window, sink, bias, tau);
                    let c = Case::new(2, 9, 2, heads, o, &mut rng);
                    check_case(&g, &ids, &c, &format!("{o:?}"));
                }
            }
        }
    }
}

/// The boundaries the brief names: T = 1, window - 1, window, window + 1 (window 4), positions on both sides of
/// `n_floor`, an MHA head of the real width and a GQA ratio of 3.
#[test]
fn boundaries_of_window_length_and_floor() {
    let (g, ids) = device();
    let mut rng = Lcg::new(0xa77e_0002); // seed: fixed so a failing boundary reproduces
    let all = |hd: u32, n_floor: f32| opts((hd as f32).powf(-0.5), Some(4), true, Some(DistanceBias { extent: 4, d_rel: 2 }), Some(LogPositionScale { alpha: 0.1, n_floor, scales_bias: true }));
    for t in [1u32, 3, 4, 5] {
        let c = Case::new(1, t, 0, HeadShape { n_heads: 3, n_kv_heads: 1, head_dim: 5 }, all(5, 3.0), &mut rng);
        check_case(&g, &ids, &c, &format!("t={t}"));
    }
    // Positions 6..=9 straddle n_floor 8: (P + 1) / 8 crosses 1 between P = 6 and P = 7.
    let c = Case::new(1, 4, 6, HeadShape { n_heads: 2, n_kv_heads: 2, head_dim: 6 }, all(6, 8.0), &mut rng);
    check_case(&g, &ids, &c, "positions around n_floor");
    assert_eq!(log_position_scale(6, 0.1, 8.0), 1.0, "below the floor the factor is exactly 1");
    assert!(log_position_scale(8, 0.1, 8.0) > 1.0);
    // A real head width (the kernels' 256-slot accumulators at 128), forward and backward.
    let c = Case::new(1, 6, 0, HeadShape { n_heads: 2, n_kv_heads: 1, head_dim: 128 }, all(128, 2.0), &mut rng);
    check_case(&g, &ids, &c, "head_dim 128");
}

/// Plain options reproduce plain causal attention: the same kernels with every option off equal the oracle with
/// every option off (the off-by-default contract).
#[test]
fn options_off_is_plain_causal_attention() {
    let (g, ids) = device();
    let mut rng = Lcg::new(0xa77e_0003); // seed: fixed so a failure reproduces
    let c = Case::new(2, 7, 0, HeadShape { n_heads: 6, n_kv_heads: 2, head_dim: 16 }, AttnOptions::plain(0.25), &mut rng);
    check_case(&g, &ids, &c, "plain");
}

/// A declared option without its input, or an input without its option, is refused - never ignored.
#[test]
#[should_panic(expected = "the sink option and the sink buffer must come together")]
fn a_sink_buffer_without_the_option_is_refused() {
    let (g, ids) = device();
    let heads = HeadShape { n_heads: 2, n_kv_heads: 1, head_dim: 4 };
    let x = g.storage(64);
    let (out, lse) = (g.storage(64), g.storage(8));
    let inputs = AttnInputs { sink: Some(&x), rel: None, bank: None };
    let _ = flash_ext_fwd(&g, &ids, &heads, &AttnOptions::plain(0.5), &inputs, Contiguous { b: 1, t: 4, pos0: 0 }, &x, &x, &x, &out, &lse);
}

// ---- paged, chunked, decode, ring ------------------------------------------------------------------------------

/// Round an f32 to the nearest-even bf16 and back (what a bf16 plane stores).
fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Write `rows` K/V rows (`kvd` words each) into pool rows `dst_rows[i]` of a layer's planes.
fn write_rows(g: &Gpu, layer: &KvLayer, tier: KvTier, k: &[f32], v: &[f32], kvd: usize, dst_rows: &[u32]) {
    for (src, &dst) in dst_rows.iter().enumerate() {
        for (plane, data) in [(&layer.k, k), (&layer.v, v)] {
            let row = &data[src * kvd..][..kvd];
            if tier == KvTier::F32 {
                g.write_f32_at(plane.data(), u64::from(dst) * kvd as u64, row);
            } else {
                // bf16 pairs, low half first; kvd is even.
                let words: Vec<u32> = row.chunks(2).map(|p| (bf16_round(p[0]).to_bits() >> 16) | (bf16_round(p[1]).to_bits() & 0xffff_0000)).collect();
                g.write_at(plane.data(), u64::from(dst) * kvd as u64 / 2, &words);
            }
        }
    }
}

/// `(out, lse)` of `n_q` rows of one sequence starting at query row `row0` (position `pos0`), through the paged
/// path over `layer` with `table`.
#[allow(clippy::too_many_arguments)]
fn paged_rows(g: &Gpu, ids: &FlashExtIds, c: &Case, layer: &KvLayer, seq: PagedSeq, table: &[u32], block_size: u32, ring: u32) -> (Vec<f32>, Vec<f32>) {
    let (nh, hd) = (c.heads.n_heads as usize, c.heads.head_dim as usize);
    let r0 = seq.q_row0 as usize;
    let n = seq.n_q as usize;
    let q = upload(g, "q.rows", &c.q[r0 * nh * hd..][..n * nh * hd]);
    let sink = upload(g, "sink.rows", if c.sink.is_empty() { &[] } else { &c.sink[r0 * nh..][..n * nh] });
    let d_rel = c.opts.bias.map_or(0, |b| b.d_rel as usize);
    let rel = upload(g, "rel.rows", if c.rel.is_empty() { &[] } else { &c.rel[r0 * nh * d_rel..][..n * nh * d_rel] });
    let bank = upload(g, "bank", &c.bank);
    let inputs = AttnInputs { sink: c.opts.sink.then_some(&sink), rel: c.opts.bias.map(|_| &rel), bank: c.opts.bias.map(|_| &bank) };
    let index = u32s(g, &paged_index(&[PagedSeq { q_row0: 0, ..seq }], &[table], table.len() as u32));
    let paged = PagedKv { index: &index, n_seqs: 1, max_q: seq.n_q, n_rows: seq.n_q, block_size, max_bt: table.len() as u32, ring };
    let (out, lse) = (g.storage((n * nh * hd) as u64), g.storage((n * nh) as u64));
    g.submit(&[], &flash_ext_paged_fwd(g, ids, &c.heads, &c.opts, &inputs, &paged, &q, KvPlanes::of(layer), &out, &lse));
    (g.read(&out, n * nh * hd), g.read(&lse, n * nh))
}

/// One sequence: the contiguous whole-sequence forward, a chunked prefill through a scattered block table, and
/// token-by-token decode through the same pages all agree; for a window layer, decode through a ring of `window`
/// rows (and a ring chunk against the gathered window) agrees too. f32 planes are held bit-close; bf16 planes are
/// compared with the oracle fed the same bf16-rounded keys and values.
#[test]
fn whole_chunked_decode_and_ring_agree() {
    let (g, ids) = device();
    let mut rng = Lcg::new(0xa77e_0004); // seed: fixed so a failure reproduces
    let heads = HeadShape { n_heads: 4, n_kv_heads: 2, head_dim: 8 };
    let (t, bs) = (13u32, 4u32);
    let kvd = (heads.n_kv_heads * heads.head_dim) as usize;
    let (nh, hd) = (heads.n_heads as usize, heads.head_dim as usize);
    for window in [None, Some(5)] {
        let o = opts(8f32.powf(-0.5), window, true, Some(DistanceBias { extent: 6, d_rel: 3 }), Some(LogPositionScale { alpha: 0.2, n_floor: 3.0, scales_bias: true }));
        for tier in [KvTier::F32, KvTier::BF16] {
            let mut c = Case::new(1, t, 0, heads, o, &mut rng);
            if tier == KvTier::BF16 {
                c.k = c.k.iter().map(|&x| bf16_round(x)).collect();
                c.v = c.v.iter().map(|&x| bf16_round(x)).collect();
            }
            let tag = format!("window {window:?} {tier}");
            let (want_out, want_lse) = eval(&c, &host_of(&c));
            let whole = run_device(&g, &ids, &c);
            assert_close(&format!("{tag}: whole out"), &whole.out, &want_out, 2e-5);

            // A scattered table: logical block b lives in physical block table[b].
            let table = [3u32, 0, 4, 1];
            let layer = KvLayer::new(&g, tier, u64::from(5 * bs), kvd as u64, u64::from(heads.head_dim));
            let pool_rows: Vec<u32> = (0..t).map(|p| table[(p / bs) as usize] * bs + p % bs).collect();
            write_rows(&g, &layer, tier, &c.k, &c.v, kvd, &pool_rows);

            let mut start = 0u32;
            for n in [5u32, 1, 4, 3] {
                let (out, lse) = paged_rows(&g, &ids, &c, &layer, PagedSeq { q_row0: start, n_q: n, q_pos0: start, kv_pos0: 0 }, &table, bs, 0);
                let r = start as usize;
                assert_close(&format!("{tag}: chunk at {start} out"), &out, &want_out[r * nh * hd..][..n as usize * nh * hd], 2e-5);
                assert_close(&format!("{tag}: chunk at {start} lse"), &lse, &want_lse[r * nh..][..n as usize * nh], 2e-5);
                start += n;
            }
            for p in 0..t {
                let (out, _) = paged_rows(&g, &ids, &c, &layer, PagedSeq { q_row0: p, n_q: 1, q_pos0: p, kv_pos0: 0 }, &table, bs, 0);
                assert_close(&format!("{tag}: decode at {p}"), &out, &want_out[p as usize * nh * hd..][..nh * hd], 2e-5);
            }

            if let Some(w) = window {
                // A ring of exactly `w` rows in physical block 2 (block size w): position p at row p % w, written as
                // decode goes, so the ring never holds more than the window.
                let ring = KvLayer::new(&g, tier, u64::from(3 * w), kvd as u64, u64::from(heads.head_dim));
                for p in 0..t {
                    let pr = p as usize;
                    write_rows(&g, &ring, tier, &c.k[pr * kvd..][..kvd], &c.v[pr * kvd..][..kvd], kvd, &[2 * w + p % w]);
                    let (out, lse) = paged_rows(&g, &ids, &c, &ring, PagedSeq { q_row0: p, n_q: 1, q_pos0: p, kv_pos0: 0 }, &[2], w, w);
                    assert_close(&format!("{tag}: ring decode at {p} out"), &out, &want_out[pr * nh * hd..][..nh * hd], 2e-5);
                    assert_close(&format!("{tag}: ring decode at {p} lse"), &lse, &want_lse[pr * nh..][..nh], 2e-5);
                }
                // A ring chunk: the last min(start, w) positions gathered before the chunk's own keys, attended
                // linearly from kv_pos0 = start - m (what `gqa_mixer`'s ring chunk does).
                let (start, n) = (7u32, 4u32);
                let m = start.min(w);
                let gathered = KvLayer::new(&g, tier, u64::from(m + n), kvd as u64, u64::from(heads.head_dim));
                let lo = (start - m) as usize;
                write_rows(&g, &gathered, tier, &c.k[lo * kvd..(start + n) as usize * kvd], &c.v[lo * kvd..(start + n) as usize * kvd], kvd, &(0..m + n).collect::<Vec<_>>());
                let (out, _) = paged_rows(&g, &ids, &c, &gathered, PagedSeq { q_row0: start, n_q: n, q_pos0: start, kv_pos0: start - m }, &[0], m + n, 0);
                assert_close(&format!("{tag}: ring chunk"), &out, &want_out[start as usize * nh * hd..][..n as usize * nh * hd], 2e-5);
            }
        }
    }
}

/// Two sequences in one paged dispatch (a decode batch with a prefill chunk beside it): each row reads its own
/// table, position and window.
#[test]
fn a_mixed_batch_reads_each_sequence_through_its_own_table() {
    let (g, ids) = device();
    let mut rng = Lcg::new(0xa77e_0005); // seed: fixed so a failure reproduces
    let heads = HeadShape { n_heads: 2, n_kv_heads: 1, head_dim: 4 };
    let o = opts(0.5, Some(3), true, Some(DistanceBias { extent: 2, d_rel: 2 }), Some(LogPositionScale { alpha: 0.5, n_floor: 2.0, scales_bias: false }));
    let (a, b) = (Case::new(1, 7, 0, heads, o, &mut rng), Case::new(1, 5, 0, heads, o, &mut rng));
    let kvd = (heads.n_kv_heads * heads.head_dim) as usize;
    let (nh, hd, bs) = (2usize, 4usize, 2u32);
    let layer = KvLayer::new(&g, KvTier::F32, 16, kvd as u64, hd as u64);
    let (ta, tb): (Vec<u32>, Vec<u32>) = (vec![7, 1, 5, 3], vec![0, 6, 2]);
    write_rows(&g, &layer, KvTier::F32, &a.k, &a.v, kvd, &(0..7).map(|p| ta[p / 2] * bs + p as u32 % bs).collect::<Vec<_>>());
    write_rows(&g, &layer, KvTier::F32, &b.k, &b.v, kvd, &(0..5).map(|p| tb[p / 2] * bs + p as u32 % bs).collect::<Vec<_>>());
    // Sequence a decodes its last row (position 6); sequence b prefills rows 2..5.
    let seqs = [PagedSeq { q_row0: 0, n_q: 1, q_pos0: 6, kv_pos0: 0 }, PagedSeq { q_row0: 1, n_q: 3, q_pos0: 2, kv_pos0: 0 }];
    let rows = |c: &Case, r0: usize, n: usize, w: usize| c.q[r0 * w..][..n * w].to_vec();
    let q: Vec<f32> = [rows(&a, 6, 1, nh * hd), rows(&b, 2, 3, nh * hd)].concat();
    let sink: Vec<f32> = [a.sink[6 * nh..7 * nh].to_vec(), b.sink[2 * nh..5 * nh].to_vec()].concat();
    let rel: Vec<f32> = [a.rel[6 * nh * 2..7 * nh * 2].to_vec(), b.rel[2 * nh * 2..5 * nh * 2].to_vec()].concat();
    // Both sequences share one bank (it is a layer parameter); b's draw is replaced by a's.
    let b = Case { bank: a.bank.clone(), ..b };
    let (qb, sb, rb, bk) = (upload(&g, "q", &q), upload(&g, "sink", &sink), upload(&g, "rel", &rel), upload(&g, "bank", &a.bank));
    let index = u32s(&g, &paged_index(&seqs, &[&ta, &tb], 4));
    let paged = PagedKv { index: &index, n_seqs: 2, max_q: 3, n_rows: 4, block_size: bs, max_bt: 4, ring: 0 };
    let (out, lse) = (g.storage((4 * nh * hd) as u64), g.storage((4 * nh) as u64));
    let inputs = AttnInputs { sink: Some(&sb), rel: Some(&rb), bank: Some(&bk) };
    g.submit(&[], &flash_ext_paged_fwd(&g, &ids, &heads, &o, &inputs, &paged, &qb, KvPlanes::of(&layer), &out, &lse));
    let got = g.read(&out, 4 * nh * hd);
    let (wa, _) = eval(&a, &host_of(&a));
    let (wb, _) = eval(&b, &host_of(&b));
    assert_close("sequence a decode", &got[..nh * hd], &wa[6 * nh * hd..7 * nh * hd], 2e-5);
    assert_close("sequence b chunk", &got[nh * hd..], &wb[2 * nh * hd..5 * nh * hd], 2e-5);
}

/// A window layer's prefill in chunks through its ring (`gqa_mixer::ring_chunk_attend`) with every option on equals
/// the whole-sequence pass, and leaves the ring holding exactly the last `window` positions, which a decode step
/// then reads in place. f32 and bf16 rings (the keys pre-rounded so the bf16 ring stores them exactly).
#[test]
fn ring_chunked_prefill_with_every_option_equals_the_whole_pass() {
    use model::gqa_mixer::{ring_chunk_attend, RingChunk, RingCopyIds};
    let (g, ids) = device();
    let mut rng = Lcg::new(0xa77e_0006); // seed: fixed so a failure reproduces
    let heads = HeadShape { n_heads: 4, n_kv_heads: 2, head_dim: 6 };
    let (nh, hd, kvd) = (4usize, 6usize, 12usize);
    let w = 5u32;
    let o = opts(6f32.powf(-0.5), Some(w), true, Some(DistanceBias { extent: 5, d_rel: 2 }), Some(LogPositionScale { alpha: 0.2, n_floor: 3.0, scales_bias: true }));
    for tier in [KvTier::F32, KvTier::BF16] {
        let mut c = Case::new(1, 15, 0, heads, o, &mut rng);
        c.k = c.k.iter().map(|&x| bf16_round(x)).collect();
        c.v = c.v.iter().map(|&x| bf16_round(x)).collect();
        let (want, _) = eval(&c, &host_of(&c));
        // Two rings in the pool; this sequence owns the second (rows w .. 2w).
        let layer = KvLayer::new(&g, tier, u64::from(2 * w), kvd as u64, hd as u64);
        let mut start = 0u32;
        for n in [3u32, 6, 1, 4] {
            let r = start as usize;
            let rows = |x: &[f32], width: usize| upload(&g, "rows", &x[r * width..][..n as usize * width]);
            let (q, k, v) = (rows(&c.q, nh * hd), rows(&c.k, kvd), rows(&c.v, kvd));
            let (sink, rel, bank) = (rows(&c.sink, nh), rows(&c.rel, nh * 2), upload(&g, "bank", &c.bank));
            let ring = RingChunk { layer: &layer, row0: w, window: w, start, flash: ids, copy: RingCopyIds::resolve(&g), ring_attn: None };
            let inputs = AttnInputs { sink: Some(&sink), rel: Some(&rel), bank: Some(&bank) };
            let ctx = ring_chunk_attend(&g, &ring, &heads, &o, &inputs, &q, &k, &v, n);
            assert_close(&format!("{tier}: ring chunk at {start}"), &g.read(&ctx, n as usize * nh * hd), &want[r * nh * hd..][..n as usize * nh * hd], 2e-5);
            start += n;
        }
        // The last query again, as a decode step over the ring in place (the ring holds positions 10..15).
        let p = start - 1;
        let (out, _) = paged_rows(&g, &ids, &c, &layer, PagedSeq { q_row0: p, n_q: 1, q_pos0: p, kv_pos0: 0 }, &[1], w, w);
        assert_close(&format!("{tier}: decode over the prefilled ring"), &out, &want[p as usize * nh * hd..][..nh * hd], 2e-5);
    }
}
