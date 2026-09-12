// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An adapter on a QUANTIZED FLUX.2 build must arrive in full.
//!
//! Swedish Embedded AB implements low-rank adapter serving on quantized
//! transformers for its clients. If your team needs expertise in INT8
//! inference or LoRA deployment then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! ## What these gate
//!
//! Deploying a LoRA by folding it - rebuild `W' = W + s·B·A`, quantize, run -
//! is exact while the destination is fp32 and destructive once it is int8.
//! `tests/lora_requant_int8.rs` measures the weight-level damage on fixtures:
//! at a delta 0.5% of the weight magnitude, 94.8% of adapted weights round
//! straight back to their base code and the surviving remainder has cosine
//! 0.44 to the delta that was intended. The user-visible result is an adapted
//! generation that is mostly un-adapted, with a token-constant bias where the
//! adapter did land.
//!
//! So an int8 build must not fold at all: the base weight stays exactly what
//! the checkpoint says (the same bit-exact Q8_0 route an unadapted run takes)
//! and the adapter is evaluated beside it, `y = W·x + s·B·(A·x)`, in fp32.
//!
//! Two gates, one per half of that claim:
//!
//! 1. **The checkpoint is not touched** and every rectangle the fold would
//!    have written is accounted for as a runtime correction instead, at the
//!    same scale - asserted against the fold itself, so the two descriptions
//!    of one adapter cannot drift.
//! 2. **The correction is dispatched over the right rows of the right
//!    linears**, checked by comparing a runtime-adapted int8 forward with the
//!    same adapter's exact fp32 result.
//!
//! The arithmetic itself - that a runtime correction delivers the delta whole
//! where a fold delivers a fraction of it - is settled one linear at a time in
//! `crates/model/tests/lora_runtime_delta.rs`, where the activation
//! quantization can be held identical across the two deployments and the
//! comparison is therefore exact.

use flux2::lora::{fold_adapters, runtime_adapters, save_adapter, LoraAdapter, LoraCfg};
use flux2::modelgrad::Cfg;
use flux2::{AdapterSpec, DitWeights, Flux2Config, Flux2Model, Precision};

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}

/// The smallest FLUX.2 that meets the int8 slicing alignment (txt_len, hidden
/// and mlp all multiples of 64) - the same dims `model_smoke`'s int8 case uses.
fn tiny_fc() -> Flux2Config {
    Flux2Config {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 64,
        n_heads: 2,
        depth_double: 2,
        depth_single: 2,
        axes_dim: [8, 8, 8, 8],
        txt_len: 64,
        ..Flux2Config::klein_4b()
    }
}

fn manifest_tensors(fc: &Flux2Config, seed: u64) -> flux2::Tensors {
    let mut r = rng(seed);
    let mut ts = flux2::Tensors::new();
    for (name, shape) in fc.tensor_manifest() {
        let n: usize = shape.iter().product();
        let (base, scale) = if name.ends_with("norm.scale") { (1.0, 0.1) } else { (0.0, 0.05) };
        let data: Vec<f32> = (0..n).map(|_| base + r() * scale).collect();
        ts.insert(name, (shape, data));
    }
    ts
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain-flux2-lora-runtime-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

/// A trained-looking brain-native adapter whose folded delta has an RMS of
/// `frac` times the RMS of the base weights - the operating point that matters
/// is the SIZE of the delta relative to the int8 grid, so it is set here
/// rather than left to whatever a random draw happens to produce.
///
/// A fresh adapter is `B = 0` (no delta at all), so `B` is filled directly.
fn brain_adapter(fc: &Flux2Config, ts: &flux2::Tensors, frac: f64, seed: u64, name: &str) -> String {
    let c = Cfg::from_flux2(fc, 1, 1);
    let mut ad = LoraAdapter::new(&c, LoraCfg::new(4));
    let mut r = rng(seed);
    for p in ad.pairs_mut() {
        for v in p.b.iter_mut() {
            *v = r();
        }
    }
    // Measure what that draw actually produces on the real target tensors,
    // then rescale `B` (which scales `B·A` linearly) to hit `frac`.
    let probe = "double_blocks.0.img_attn.qkv.weight";
    let mut folded = ts.clone();
    ad.fold_into_tensors(&mut folded).expect("fold onto the tiny manifest");
    let base = &ts[probe].1;
    let delta: Vec<f32> = folded[probe].1.iter().zip(base).map(|(a, b)| a - b).collect();
    let want = rms(base) * frac;
    let k = (want / rms(&delta)) as f32;
    for p in ad.pairs_mut() {
        for v in p.b.iter_mut() {
            *v *= k;
        }
    }
    let path = tmp(name);
    save_adapter(path.to_str().unwrap(), &ad);
    path.to_str().unwrap().to_string()
}

/// **An int8 build reads an adapter without touching the checkpoint.**
///
/// `runtime_adapters` must describe exactly the deltas `fold_adapters` would
/// have written - every rectangle, at the same scale, including the caller's
/// `--lora-scale` strength - while leaving the tensor map byte for byte as it
/// was. One walk (`LoraAdapter::placements`) feeds both, and this is what pins
/// that the two dispositions of one adapter stay the same adapter.
#[test]
fn runtime_corrections_describe_the_fold_without_writing_it() {
    let fc = tiny_fc();
    let ts = manifest_tensors(&fc, 0xA11CE);
    let path = brain_adapter(&fc, &ts, 0.005, 0x5EED, "describe.brain");

    // A rectangle of a row-major `[_, stride]` tensor, copied out.
    let rect = |w: &[f32], stride: usize, row0: usize, out: usize, col0: usize, inn: usize| -> Vec<f32> {
        (0..out).flat_map(|i| w[(row0 + i) * stride + col0..(row0 + i) * stride + col0 + inn].to_vec()).collect()
    };

    for strength in [1.0f32, 0.35] {
        let specs = [AdapterSpec { path: path.clone(), scale: strength }];
        let mut folded = ts.clone();
        fold_adapters(&fc, &mut folded, &specs).expect("fold");
        assert_ne!(folded, ts, "the fixture's adapter must actually move the weights");

        let (rt, reports) = runtime_adapters(&fc, &specs).expect("read").expect("a brain adapter is low-rank");
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].pairs, fc.depth_double * 14 + fc.depth_single * 7);
        assert_eq!(rt.len(), reports[0].pairs, "every pair is one correctable rectangle");
        assert_eq!(rt.max_rank(), 4);
        rt.validate(&|n| ts.get(n).map(|(_, d)| d.len())).expect("targets exist at the right size");

        // Take each correction's own rectangle out of the PRISTINE map, fold
        // the correction into it, and it must equal the same rectangle of the
        // folded map - bit for bit, at either strength.
        for d in rt.deltas() {
            let (shape, base) = &ts[&d.key];
            assert_eq!(*shape.last().unwrap(), d.row_stride, "{}: stride", d.key);
            let mut got = rect(base, d.row_stride, d.row0, d.pair.out, d.col0, d.pair.inn);
            d.fold_into(&mut got);
            let want = rect(&folded[&d.key].1, d.row_stride, d.row0, d.pair.out, d.col0, d.pair.inn);
            assert_eq!(got, want, "{} rows {}.. cols {}..", d.key, d.row0, d.col0);
        }
    }
    std::fs::remove_file(&path).ok();
}

/// **The correction is wired to the right linears, at the right rows.**
///
/// Whether the delta arrives *intact* is settled exactly, one linear at a
/// time, in `crates/model/tests/lora_runtime_delta.rs` (measured: rel_l2 5e-6
/// delivered at runtime against 7e-1 folded). What that test cannot see is
/// whether a whole DiT dispatches each correction over the rows its base
/// linear just wrote - an off-by-one row base, a correction attached to the
/// wrong projection, or one never dispatched at all.
///
/// So this compares a runtime-adapted INT8 forward against the same adapter on
/// the same model at fp32, where folding is exact and the answer is therefore
/// known: the two must respond to the adapter in the same direction and by the
/// same amount. A missing correction gives ratio 0; a misrouted one decorrelates.
///
/// The delta is deliberately larger here (5% of the weight magnitude) than the
/// regime the defect lives in: at toy width (hidden 64) the int8 tier's own
/// activation-quantization jitter is itself several percent of the output, so a
/// 0.5% delta would be measuring the tier's noise floor rather than this
/// wiring. GPU only (int8 is DP4A).
#[test]
fn an_int8_runtime_adapted_forward_tracks_its_fp32_reference() {
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable(&format!("int8 needs a GPU backend, current is {}", gpu.kind()));
        return;
    }
    let fc = tiny_fc();
    let ts = manifest_tensors(&fc, 0xB0B);
    let path = brain_adapter(&fc, &ts, 0.05, 0xC0FFEE, "deliver.brain");
    let specs = [AdapterSpec::new(path.clone())];

    let n_img = 4 * 4;
    let n_max = (fc.txt_len + n_img) as u32;
    let ids = flux2::position_ids(fc.txt_len, 4, 4, &[]);
    let img: Vec<f32> = (0..n_img * fc.in_channels).map(|i| (i as f32 * 0.7).sin() * 0.5).collect();
    let ctx: Vec<f32> = (0..fc.txt_len * fc.context_in_dim).map(|i| (i as f32 * 0.3).cos() * 0.5).collect();
    let run = |m: &Flux2Model| m.forward(&img, &ctx, 0.7, &ids, n_img);

    let mut folded = ts.clone();
    fold_adapters(&fc, &mut folded, &specs).expect("fold");
    let f32_base = run(&Flux2Model::new(&fc, &ts, gpu.share(), n_max));
    let f32_adapted = run(&Flux2Model::new(&fc, &folded, gpu.share(), n_max));

    let i8_base = run(&Flux2Model::new_with(&fc, &ts, gpu.share(), n_max, Precision::Int8));
    let (rt, _) = runtime_adapters(&fc, &specs).expect("read").expect("low-rank");
    let src = DitWeights::Map(&ts);
    let i8_adapted = run(&Flux2Model::new_adapted(&fc, &src, gpu.share(), n_max, 1, Precision::Int8, Some(&rt)));

    let diff = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
    let want = diff(&f32_adapted, &f32_base);
    let got = diff(&i8_adapted, &i8_base);
    assert!(rms(&want) > 0.0, "the fp32 reference adapter must move the prediction at all");
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(&want) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let ratio = rms(&got) / rms(&want);
    eprintln!("adapter response: cosine {cos:.4}, magnitude ratio {ratio:.4} (fp32 reference rms {:.3e})", rms(&want));
    assert!(cos > 0.97, "the int8 build's response to the adapter points elsewhere: cosine {cos:.4}");
    assert!((0.9..1.12).contains(&ratio), "the int8 build delivers {ratio:.4}x of the adapter's effect");
    std::fs::remove_file(&path).ok();
}
