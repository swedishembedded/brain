// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does an **int8 frozen base** still produce faithful adapter gradients?
//!
//! It is the question with the big payoff: at fp32 a klein-9B base is larger
//! than one 24 GiB card and the trainer has to be split across two; at int8 it
//! would fit one. So it is worth answering with a measurement rather than an
//! assumption, and worth answering on the REAL checkpoint - quantization error
//! is a property of the actual weight distribution, and a synthetic Gaussian
//! would understate it.
//!
//! Method: take one real double block out of the released GGUF, build the
//! device block engine twice - once on the fp32 weights, once on the same
//! weights round-tripped through brain's own per-output-row int8 grid
//! (`model::int8::row_scale`/`pack_row`, exactly what `matmul_i8_dyn` consumes)
//! - and run the identical backward through both. What is compared is the
//! **adapter** gradient, since that is the only thing a LoRA run consumes.
//!
//! Scope, stated because it bounds the conclusion: this measures the
//! **weight**-quantization term. A real int8 kernel additionally quantizes the
//! activation per token, which adds error this test does not see. So the
//! number here is an upper bound on int8's fidelity, not a promise about it -
//! if it is already poor, the kernel work is not worth doing; if it is good,
//! the activation term still has to be measured separately.
//!
//! Skip-if-absent: needs `BRAIN_FLUX2_DIT` (a klein GGUF/safetensors) and
//! `BRAIN_DEV_GPU=1`. Ignored by default - it reads gigabytes off disk.

use flux2::devgrad::{BlockDev, DoubleDev, N_SITES};
use flux2::grad::{Dims, Mod};
use flux2::Flux2Config;
use model::lora::Pair;

/// One backward's adapter gradients: `(dA, dB)` per targeted linear.
type PairGrads = Vec<(Vec<f32>, Vec<f32>)>;

const RANK: usize = 16;
const NT: usize = 512;
const NI: usize = 256;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}
fn vof(n: usize, r: &mut impl FnMut() -> f32, s: f32) -> Vec<f32> {
    (0..n).map(|_| r() * s).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-300)
}
fn rel_l2(dev: &[f32], host: &[f32]) -> f64 {
    let nh: f64 = host.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let df: f64 = dev.iter().zip(host).map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64)).sum::<f64>().sqrt();
    df / nh.max(1e-30)
}

/// `w` through brain's per-output-row symmetric int8 grid and back -
/// `matmul_i8_dyn`'s exact weight representation.
fn int8_round_trip(w: &[f32], out: usize, inn: usize) -> Vec<f32> {
    let mut q = vec![0.0f32; w.len()];
    for o in 0..out {
        let row = &w[o * inn..(o + 1) * inn];
        let s = model::int8::row_scale(row);
        for (i, &v) in row.iter().enumerate() {
            q[o * inn + i] = (v / s).round().clamp(-127.0, 127.0) * s;
        }
    }
    q
}

#[test]
#[ignore = "reads the real checkpoint off disk; run explicitly"]
fn an_int8_frozen_base_and_its_effect_on_adapter_gradients() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the int8 frozen-base measurement");
        return;
    }
    let Ok(dit) = std::env::var("BRAIN_FLUX2_DIT") else {
        brain_testutil::skip_unavailable("set BRAIN_FLUX2_DIT to a klein checkpoint for the int8 frozen-base measurement");
        return;
    };
    let variant = std::env::var("BRAIN_FLUX2_TRAIN_VARIANT").unwrap_or_else(|_| "klein-9b".into());
    let fc = Flux2Config::from_name(&variant).expect("variant");
    let (d, mlp, hd) = (fc.hidden, fc.mlp_hidden(), fc.head_dim());
    let dm = Dims { nt: NT, ni: NI, d, nh: fc.n_heads, mlp };
    let n = dm.n();

    let gguf = checkpoint::gguf::MmapGguf::open(&dit).expect("open checkpoint");
    let src = flux2::DitWeights::gguf(&gguf);

    // One real double block, split from the fused checkpoint layout exactly as
    // `ModelWeights::from_tensors` does.
    let rows = |v: &[f32], cols: usize, r0: usize, r1: usize| v[r0 * cols..r1 * cols].to_vec();
    let mut tensors: Vec<(String, usize, usize, Vec<f32>)> = Vec::new();
    for s in ["img", "txt"] {
        let p = format!("double_blocks.0.{s}");
        let qkv = src.with_f32(&format!("{p}_attn.qkv.weight"), |v| v.to_vec());
        let m0 = src.with_f32(&format!("{p}_mlp.0.weight"), |v| v.to_vec());
        tensors.push((format!("{s}.wq"), d, d, rows(&qkv, d, 0, d)));
        tensors.push((format!("{s}.wk"), d, d, rows(&qkv, d, d, 2 * d)));
        tensors.push((format!("{s}.wv"), d, d, rows(&qkv, d, 2 * d, 3 * d)));
        tensors.push((format!("{s}.wo"), d, d, src.with_f32(&format!("{p}_attn.proj.weight"), |v| v.to_vec())));
        tensors.push((format!("{s}.w1"), mlp, d, rows(&m0, d, 0, mlp)));
        tensors.push((format!("{s}.w3"), mlp, d, rows(&m0, d, mlp, 2 * mlp)));
        tensors.push((format!("{s}.w2"), d, mlp, src.with_f32(&format!("{p}_mlp.2.weight"), |v| v.to_vec())));
    }
    let nq: Vec<Vec<f32>> = ["img", "txt"].iter().map(|s| src.with_f32(&format!("double_blocks.0.{s}_attn.norm.query_norm.scale"), |v| v.to_vec())).collect();
    let nk: Vec<Vec<f32>> = ["img", "txt"].iter().map(|s| src.with_f32(&format!("double_blocks.0.{s}_attn.norm.key_norm.scale"), |v| v.to_vec())).collect();

    // ---- (1) how much the int8 grid itself moves the weights ----
    eprintln!("\n{variant} double block 0 - per-output-row int8 round trip:");
    let mut q8: Vec<Vec<f32>> = Vec::with_capacity(tensors.len());
    let mut worst_w = 0.0f64;
    for (name, out, inn, w) in &tensors {
        let r = int8_round_trip(w, *out, *inn);
        let e = rel_l2(&r, w);
        worst_w = worst_w.max(e);
        eprintln!("  {name:8} [{out},{inn}]  rel_l2 {e:.4e}  cosine {:.9}", cosine(&r, w));
        q8.push(r);
    }

    // ---- (2) what that does to the adapter gradients ----
    let eng = BlockDev::new(n, d, fc.n_heads, mlp, RANK);
    let build = |ws: &[Vec<f32>]| DoubleDev {
        img: eng.stream(&ws[0], &ws[1], &ws[2], &ws[3], &ws[4], &ws[5], &ws[6], &nq[0], &nk[0]),
        txt: eng.stream(&ws[7], &ws[8], &ws[9], &ws[10], &ws[11], &ws[12], &ws[13], &nq[1], &nk[1]),
    };
    let f32w: Vec<Vec<f32>> = tensors.iter().map(|(_, _, _, w)| w.clone()).collect();
    let dev_f32 = build(&f32w);
    let dev_i8 = build(&q8);
    drop(tensors);

    let mut r = rng(0xc0ffee);
    let scale = 1.0f32; // alpha = rank
    let pairs: Vec<Pair> = f32w.iter().enumerate().map(|(i, _)| {
        let (out, inn) = match i % 7 {
            4 | 5 => (mlp, d),
            6 => (d, mlp),
            _ => (d, d),
        };
        Pair::from_ab(out, inn, RANK, vof(RANK * inn, &mut r, 0.02), vof(out * RANK, &mut r, 0.02))
    }).collect();
    drop(f32w);
    drop(q8);

    let site = |r: &mut dyn FnMut() -> f32| Mod { shift: (0..d).map(|_| r() * 0.05).collect(), scale: (0..d).map(|_| r() * 0.05).collect(), gate: (0..d).map(|_| r() * 0.2).collect() };
    let zero = Mod { shift: vec![0.0; d], scale: vec![0.0; d], gate: vec![0.0; d] };
    let sites: [Mod<f32>; N_SITES] = [site(&mut r), site(&mut r), site(&mut r), site(&mut r), zero.clone(), zero];
    let x = vof(n * d, &mut r, 1.0);
    let dout = vof(n * d, &mut r, 1.0);
    let half = hd / 2;
    let cos: Vec<f32> = (0..n * half).map(|i| (i as f32 * 0.017).cos()).collect();
    let sin: Vec<f32> = (0..n * half).map(|i| (i as f32 * 0.017).sin()).collect();

    let xb = eng.slab(n);
    let doutb = eng.slab(n);
    let dxb = eng.slab(n);
    eng.gpu().write_f32(&xb, &x);
    eng.gpu().write_f32(&doutb, &dout);
    eng.upload_mods(&sites);
    eng.upload_rope(&cos, &sin);

    // (per-pair (dA, dB), dx) - named so the closure's return type is readable.
    let run = |dev: &DoubleDev| -> (PairGrads, Vec<f32>) {
        let mut i = 0;
        for st in [&dev.img, &dev.txt] {
            for l in [&st.wq, &st.wk, &st.wv, &st.wo, &st.w1, &st.w3, &st.w2] {
                eng.upload_lora(l, &pairs[i].a, &pairs[i].b, scale);
                i += 1;
            }
        }
        eng.double_backward(dev, dm, &xb, &doutb, &dxb);
        let mut g = Vec::new();
        for st in [&dev.img, &dev.txt] {
            for l in [&st.wq, &st.wk, &st.wv, &st.wo, &st.w1, &st.w3, &st.w2] {
                g.push(eng.lin_grads(l, scale));
            }
        }
        (g, eng.gpu().read(&dxb, n * d))
    };
    let (gf, dxf) = run(&dev_f32);
    let (gq, dxq) = run(&dev_i8);

    let names = ["wq", "wk", "wv", "wo", "w1", "w3", "w2"];
    let (mut wc, mut wr, mut at) = (1.0f64, 0.0f64, String::new());
    eprintln!("\nadapter gradients, int8 frozen base vs fp32 frozen base ({} joint tokens):", n);
    for (i, ((qa, qb), (fa, fb))) in gq.iter().zip(&gf).enumerate() {
        let tag = format!("{}.{}", if i < 7 { "img" } else { "txt" }, names[i % 7]);
        for (part, (q, f)) in [("dA", (qa, fa)), ("dB", (qb, fb))] {
            let c = cosine(q, f);
            let e = rel_l2(q, f);
            if c < wc {
                wc = c;
                at = format!("{tag}.{part}");
            }
            wr = wr.max(e);
            eprintln!("  {tag:8}.{part}  cosine {c:.6}  rel_l2 {e:.4e}");
        }
    }
    let dxc = cosine(&dxq, &dxf);
    eprintln!("  dx           cosine {dxc:.6}  rel_l2 {:.4e}", rel_l2(&dxq, &dxf));
    eprintln!("\nINT8 FROZEN BASE: weight round trip worst rel_l2 {worst_w:.4e}; adapter-gradient worst cosine {wc:.6} ({at}), worst rel_l2 {wr:.4e}");
    // No pass/fail threshold: this test exists to REPORT the number, and a
    // threshold here would be a guess dressed as a requirement. What it does
    // assert is that the measurement ran on two genuinely different bases.
    assert!(worst_w > 0.0, "the int8 round trip must actually change the weights");
}
