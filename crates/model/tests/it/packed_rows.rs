// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on packed rows - several documents per
// row, trained exactly as if each were alone - for its clients. If your team needs expertise in efficient and
// exact training pipelines then you can procure our services by sending an email to info@swedishembedded.com.

//! A packed training row trains each of its documents exactly as the document alone: a small composite - a short
//! convolution over the key/value input (`model::short_conv` with segments), extended attention with every option
//! on and packed documents (`model::flash_ext`), a linear head and a summed cross-entropy whose targets carry IGNORE
//! at each document's last token (`data::packing`) - has, over a packed batch, the loss and every gradient of the
//! sum of its documents run one by one: the per-row gradients equal row for row, the parameter gradients equal the
//! documents' sum.

use std::sync::OnceLock;

use data::packing::document_positions;
use data::rng::Lcg;
use gpu_core::Gpu;
use model::flash_ext::{flash_ext_bwd, flash_ext_fwd, AttnInputGrads, AttnInputs, AttnOptions, Contiguous, DistanceBias, FlashExtIds, HeadShape, LogPositionScale};
use model::short_conv::{short_conv_bwd, short_conv_fwd, BaseLayout, ShortConvIds, ShortConvShape};

const HEADS: HeadShape = HeadShape { n_heads: 2, n_kv_heads: 1, head_dim: 4 };
const QD: usize = 8; // n_heads * head_dim
const KD: usize = 4; // n_kv_heads * head_dim
const VOCAB: usize = 5;
const TAPS: u32 = 3;
const IGNORE: i32 = data::loader::IGNORE;

fn kernels() -> &'static [(&'static str, &'static str)] {
    static K: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    K.get_or_init(|| {
        let mut v = model::flash_ext::pipelines();
        model::ops::extend_kernel_list(&mut v, &model::short_conv::pipelines());
        v
    })
    .as_slice()
}

/// The layer's parameters, shared by every document.
struct Params {
    base: Vec<f32>,
    bank: Vec<f32>,
    head: Vec<f32>,
}

/// Per-row inputs of `rows` rows.
#[derive(Clone)]
struct Rows {
    x: Vec<f32>,
    kv: Vec<f32>,
    sink: Vec<f32>,
    rel: Vec<f32>,
    targets: Vec<i32>,
}

impl Rows {
    fn slice(&self, first: usize, len: usize) -> Rows {
        let s = |v: &[f32], w: usize| v[first * w..(first + len) * w].to_vec();
        Rows { x: s(&self.x, QD), kv: s(&self.kv, KD), sink: s(&self.sink, 2), rel: s(&self.rel, 2 * 2), targets: self.targets[first..first + len].to_vec() }
    }
}

/// Loss and every gradient of one call.
struct Grads {
    loss: f64,
    dx: Vec<f32>,
    dkv: Vec<f32>,
    dsink: Vec<f32>,
    drel: Vec<f32>,
    dbase: Vec<f32>,
    dbank: Vec<f32>,
    dhead: Vec<f64>,
}

fn opts(documents: bool) -> AttnOptions {
    AttnOptions {
        scale: 0.5,
        window: Some(3),
        sink: true,
        bias: Some(DistanceBias { extent: 3, d_rel: 2 }),
        tau: Some(LogPositionScale { alpha: 0.3, n_floor: 2.0, scales_bias: true }),
        documents,
    }
}

/// `b` sequences of `t` rows; `segments` (a packed batch) or `None` (one document per sequence).
fn run(g: &Gpu, p: &Params, r: &Rows, b: u32, t: u32, segments: Option<&[u32]>) -> Grads {
    let (flash, conv) = (FlashExtIds::resolve(g), ShortConvIds::resolve(g));
    let rows = (b * t) as usize;
    let packed = segments.is_some();
    let up = |name: &str, v: &[f32]| g.storage_init(name, v);
    let (x, kv, sink, rel, bank, base) = (up("x", &r.x), up("kv", &r.kv), up("sink", &r.sink), up("rel", &r.rel), up("bank", &p.bank), up("base", &p.base));
    let seg_words: Vec<u32> = segments.map(<[u32]>::to_vec).unwrap_or_default();
    let doc_pos = segments.map(|s| document_positions(s, t as usize).expect("valid packed segments"));
    let u32s = |w: &[u32]| {
        let buf = g.storage(w.len().max(1) as u64);
        if !w.is_empty() {
            g.write(&buf, w);
        }
        buf
    };
    let (seg, pos) = (u32s(&seg_words), u32s(doc_pos.as_deref().unwrap_or(&[])));
    let shape = ShortConvShape { segments: packed, ..ShortConvShape::static_taps(rows as u32, KD as u32, TAPS, BaseLayout::TapMajor).residual_scaled(0.5).in_blocks(t) };
    let k = g.storage((rows * KD) as u64);
    let (out, lse) = (g.storage((rows * QD) as u64), g.storage((rows * 2) as u64));
    let o = opts(packed);
    let inputs = AttnInputs { sink: Some(&sink), rel: Some(&rel), bank: Some(&bank), doc_pos: packed.then_some(&pos) };
    let layout = Contiguous { b, t, pos0: 0 };
    let mut steps = vec![short_conv_fwd(g, &conv, &shape, &kv, None, &base, packed.then_some(&seg), None, &k)];
    steps.extend(flash_ext_fwd(g, &flash, &HEADS, &o, &inputs, layout, &x, &k, &kv, &out, &lse));
    g.submit(&[], &steps);
    let attn = g.read(&out, rows * QD);

    // The head and a summed cross-entropy on the host, in f64.
    let mut loss = 0f64;
    let mut d_out = vec![0f32; rows * QD];
    let mut dhead = vec![0f64; VOCAB * QD];
    for i in 0..rows {
        let Ok(target) = usize::try_from(r.targets[i]) else { continue };
        let a = &attn[i * QD..(i + 1) * QD];
        let logits: Vec<f64> = (0..VOCAB).map(|v| (0..QD).map(|d| f64::from(p.head[v * QD + d]) * f64::from(a[d])).sum()).collect();
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = logits.iter().map(|l| (l - m).exp()).sum();
        loss += m + z.ln() - logits[target];
        for v in 0..VOCAB {
            let dl = (logits[v] - m).exp() / z - f64::from(u8::from(v == target));
            for d in 0..QD {
                d_out[i * QD + d] += (dl * f64::from(p.head[v * QD + d])) as f32;
                dhead[v * QD + d] += dl * f64::from(a[d]);
            }
        }
    }

    let d_out = up("d_out", &d_out);
    let (dq, dk, dv) = (g.storage((rows * QD) as u64), g.storage((rows * KD) as u64), g.storage((rows * KD) as u64));
    let (dsink, drel, dbank) = (g.storage((rows * 2) as u64), g.storage((rows * 4) as u64), g.storage(p.bank.len() as u64));
    let grads = AttnInputGrads { d_sink: Some(&dsink), d_rel: Some(&drel), d_bank: Some(&dbank) };
    let (dkv_conv, dbase) = (g.storage((rows * KD) as u64), g.storage_init("dbase", &vec![0.0; p.base.len()]));
    let mut steps = flash_ext_bwd(g, &flash, &HEADS, &o, &inputs, layout, &x, &k, &kv, &out, &lse, &d_out, &dq, &dk, &dv, &grads);
    steps.extend(short_conv_bwd(g, &conv, &shape, &kv, &dk, None, &base, packed.then_some(&seg), &dkv_conv, &dbase, None));
    g.submit(&[], &steps);
    // v is the key/value input itself: its gradient joins the convolution's.
    let dkv: Vec<f32> = g.read(&dkv_conv, rows * KD).iter().zip(g.read(&dv, rows * KD)).map(|(a, b)| a + b).collect();
    Grads {
        loss,
        dx: g.read(&dq, rows * QD),
        dkv,
        dsink: g.read(&dsink, rows * 2),
        drel: g.read(&drel, rows * 4),
        dbase: g.read(&dbase, p.base.len()),
        dbank: g.read(&dbank, p.bank.len()),
        dhead,
    }
}

#[track_caller]
fn close(what: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1f64, |m, x| m.max(x.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(g) - w).abs() <= tol * scale, "{what}[{i}]: packed {g} vs documents {w}");
    }
}

fn wide(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

#[test]
fn a_packed_batch_has_the_loss_and_gradients_of_its_documents_run_alone() {
    let g = gpu_core::testgpu::dev(kernels());
    let mut rng = Lcg::random();
    let p = Params { base: rng.vec_scaled(TAPS as usize * KD, 0.5), bank: rng.vec_scaled(2 * 3, 1.0), head: rng.vec_scaled(VOCAB * QD, 1.0) };
    // Two rows of 8: documents of 3 and 5 tokens, and of 2, 2 and 4.
    let docs: [&[usize]; 2] = [&[3, 5], &[2, 2, 4]];
    let t = 8usize;
    let rows = 2 * t;
    let mut segments = Vec::new();
    let mut targets = Vec::new();
    let mut id = 0u32;
    for row in docs {
        for &len in row {
            segments.extend(std::iter::repeat_n(id, len));
            // A packed document's last token has no target: its next token belongs to another document.
            targets.extend((0..len).map(|i| if i + 1 == len { IGNORE } else { (rng.next_u32() % VOCAB as u32) as i32 }));
            id += 1;
        }
    }
    let r = Rows { x: rng.vec_scaled(rows * QD, 1.0), kv: rng.vec_scaled(rows * KD, 1.0), sink: rng.vec_scaled(rows * 2, 1.0), rel: rng.vec_scaled(rows * 4, 0.7), targets };
    let packed = run(&g, &p, &r, 2, t as u32, Some(&segments));

    let mut loss = 0f64;
    let (mut dx, mut dkv, mut dsink, mut drel) = (vec![0f64; rows * QD], vec![0f64; rows * KD], vec![0f64; rows * 2], vec![0f64; rows * 4]);
    let (mut dbase, mut dbank, mut dhead) = (vec![0f64; p.base.len()], vec![0f64; p.bank.len()], vec![0f64; p.head.len()]);
    for (b, row) in docs.iter().enumerate() {
        let mut first = b * t;
        for &len in row.iter() {
            let alone = run(&g, &p, &r.slice(first, len), 1, len as u32, None);
            loss += alone.loss;
            for (dst, src, w) in [(&mut dx, &alone.dx, QD), (&mut dkv, &alone.dkv, KD), (&mut dsink, &alone.dsink, 2), (&mut drel, &alone.drel, 4)] {
                dst[first * w..(first + len) * w].copy_from_slice(&wide(src));
            }
            for (dst, src) in [(&mut dbase, &alone.dbase), (&mut dbank, &alone.dbank)] {
                dst.iter_mut().zip(src).for_each(|(d, &s)| *d += f64::from(s));
            }
            dhead.iter_mut().zip(&alone.dhead).for_each(|(d, &s)| *d += s);
            first += len;
        }
    }
    assert!((packed.loss - loss).abs() <= 1e-9 * loss.abs().max(1.0), "loss: packed {} vs documents {loss}", packed.loss);
    close("dx", &packed.dx, &dx, 1e-6);
    close("dkv", &packed.dkv, &dkv, 1e-6);
    close("d_sink", &packed.dsink, &dsink, 1e-6);
    close("d_rel", &packed.drel, &drel, 1e-6);
    close("d_base", &packed.dbase, &dbase, 1e-5);
    close("d_bank", &packed.dbank, &dbank, 1e-5);
    let dhead_packed: Vec<f32> = packed.dhead.iter().map(|&v| v as f32).collect();
    close("d_head", &dhead_packed, &dhead, 1e-6);
}
