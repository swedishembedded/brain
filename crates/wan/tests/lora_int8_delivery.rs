// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How a trained adapter reaches an INT8 Wan DiT - and why it may not reach it
//! by being folded into the weights the quantizer then re-rounds.
//!
//! Swedish Embedded AB implements quantized inference and low-rank adapter
//! deployment for its clients. If your team needs expertise in INT8 weight
//! quantization or LoRA serving then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! ## The seam
//!
//! `pipeline::run` folds `--adapter` into the host tensor map, and
//! `WanDitDev::build_dtype`'s `Int8`/`Int4` branch then quantizes that map.
//! The two steps were written independently and neither knows about the other,
//! so an adapted INT8 build quantizes weights that have a LoRA delta in them.
//!
//! INT8 has 256 levels. A trained LoRA delta is typically a fraction of one
//! level, so `round((w + δ)/s)` returns the BASE code and the adapter is
//! discarded for that weight - silently, on a run that reports "adapter
//! loaded". The first test measures exactly that on a real fold through
//! `wan::lora`'s own placement code.
//!
//! The fix is not a better quantizer. It is to never put the delta on the
//! weight grid: keep `Q(W)` exactly what the checkpoint says (the same
//! bit-exact tensor the no-adapter path already uses) and apply the low-rank
//! correction beside it, `y = Q(W)·x + s·B·(A·x)`, in fp32. The remaining
//! tests pin that contract at the SHARED layer (`model::adapter`), because the
//! fold-then-quantize shape is a property of the shared fold, not of wan.

use std::collections::HashMap;

use model::adapter::{BaseStorage, Delivery};
use model::int8::{dequantize_weight, quantize_weight};
use wan::config::WanConfig;
use wan::import::dit_manifest;
use wan::lora::{LoraAdapter, LoraCfg};
use wan::model::Tensors;
use wan::modelgrad::{grads, make_flow_batch, Batch, Cfg, ModelWeights};

/// The ten leaves an adapter targets per block. Spelled out here rather than
/// read from the crate, so this test pins the public contract independently.
const LEAVES: [&str; 10] =
    ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"];

/// `Cfg::tiny()` widened to INT8's grouping: `model::int8` quantizes in groups
/// of 32 along the reduction axis and refuses a `k` that is not a multiple of
/// one, so both widths have to be. `head_dim` stays 8, which is what the
/// inherited `rope_axes` are sized for.
fn tiny_q8_cfg() -> Cfg {
    Cfg { dim: 32, ffn_dim: 64, n_heads: 4, ..Cfg::tiny() }
}

fn tiny_wan(c: &Cfg) -> WanConfig {
    WanConfig {
        name: "tiny-lora-int8",
        dim: c.dim,
        ffn_dim: c.ffn_dim,
        num_heads: c.n_heads,
        num_layers: c.n_layers,
        in_channels: c.in_channels,
        out_channels: c.out_channels,
        text_dim: c.text_dim,
        text_len: c.text_len,
        freq_dim: c.freq_dim,
        ..WanConfig::t2v_1_3b()
    }
}

/// Base weights that are ALREADY on the INT8 grid, the way a Q8_0 checkpoint's
/// decoded tensors are: requantizing them is then the identity, so every
/// difference the tests below see comes from the fold and nothing else.
fn q8_grid_weights(cfg: &WanConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x1234_5678_9abc_def0;
    for (name, shape) in dit_manifest(cfg) {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(0.2 * (((state >> 33) as u32) as f32 / (1u64 << 31) as f32 - 0.5));
        }
        if name.contains("norm_q") || name.contains("norm_k") || name.ends_with("norm3.weight") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        // Only the quantized leaves need to sit on the grid; everything else
        // (norms, biases, the conditioning path) is never quantized.
        if shape.len() == 2 && LEAVES.iter().any(|l| name.ends_with(&format!("{l}.weight"))) {
            let (n_rows, k) = (shape[0], shape[1]);
            let (p, sw) = quantize_weight(&v, n_rows, k);
            v = dequantize_weight(&p, &sw, n_rows, k);
        }
        t.insert(name, (shape, v));
    }
    t
}

fn fixed_batch(cfg: &Cfg) -> Batch<f32> {
    let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let rows = cfg.text_len - 1;
    let ctx: Vec<f32> = (0..rows * cfg.text_dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    make_flow_batch(cfg, &x0, &ctx, rows, 0.5, &noise)
}

/// A genuinely trained adapter - real optimisation steps against real
/// gradients, so `B` is non-zero and the deltas have a trained adapter's
/// magnitude rather than a hand-picked one.
fn trained_adapter(cfg: &Cfg, base: &ModelWeights<f32>, rank: usize) -> LoraAdapter {
    let mut ad = LoraAdapter::new(cfg, LoraCfg::new(rank));
    let b = fixed_batch(cfg);
    for _ in 0..5 {
        let (_l, g) = grads(cfg, &ad.apply(base), &b);
        ad.step(&g, 5e-3);
    }
    ad
}

fn unpack_codes(packed: &[u32]) -> Vec<i32> {
    packed.iter().flat_map(|w| (0..4).map(move |b| (((w >> (8 * b)) as u8) as i8) as i32)).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na = a.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let nb = b.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// **The bug, measured on wan's own weights and wan's own fold**, as a
/// function of how big the delta is compared with one INT8 step.
///
/// The base weight sits exactly on the grid (it came off a Q8_0 block), so
/// `round((w + δ)/s)` returns the base code unchanged for every element whose
/// `δ` is under half a step: that weight is deployed as if the adapter did not
/// exist. The delta is not merely noisy after a fold - it is itself quantized
/// to the BASE weight's grid, at roughly `|δ|/s` levels of resolution.
///
/// `strength` - the same dial an inference caller turns - is what sweeps that
/// ratio here, so the claim does not rest on one hand-picked adapter
/// magnitude. A real trained adapter's delta is a fraction of a percent of the
/// weight magnitude, i.e. the LOW end of this sweep.
///
/// This is evidence, not a regression guard on a behaviour we want: it is why
/// the tests below require the fold never to happen for a quantized base.
#[test]
fn folding_into_an_int8_base_quantizes_the_delta_onto_the_base_weights_grid() {
    let cfg = tiny_q8_cfg();
    let wc = tiny_wan(&cfg);
    let ts = q8_grid_weights(&wc);
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let ad = trained_adapter(&cfg, &base, 4);

    let keys: Vec<String> =
        (0..cfg.n_layers).flat_map(|l| LEAVES.iter().map(move |leaf| format!("blocks.{l}.{leaf}.weight"))).collect();

    let mut sweep = Vec::new();
    for strength in [0.02f32, 0.06, 0.2, 0.6] {
        let mut folded = ts.clone();
        match ad.delivery(BaseStorage::Dense, strength).expect("a dense base always has a delivery") {
            Delivery::Fold(f) => f.into_tensors(&mut folded).expect("fold"),
            Delivery::Runtime(_) => panic!("a dense base must fold"),
        }

        let (mut total, mut unmoved) = (0usize, 0usize);
        let (mut intended, mut delivered) = (Vec::new(), Vec::new());
        let (mut dmax, mut step_sum, mut groups) = (0f32, 0f64, 0usize);
        for key in &keys {
            let (shape, w0) = &ts[key];
            let (_, w1) = &folded[key];
            let (rows, k) = (shape[0], shape[1]);
            let (p0, s0) = quantize_weight(w0, rows, k);
            let (p1, s1) = quantize_weight(w1, rows, k);
            let deq = dequantize_weight(&p1, &s1, rows, k);
            total += w0.len();
            unmoved += unpack_codes(&p0).into_iter().zip(unpack_codes(&p1)).filter(|(a, b)| a == b).count();
            dmax = w1.iter().zip(w0).fold(dmax, |m, (a, b)| m.max((a - b).abs()));
            step_sum += s0.iter().map(|&s| s as f64).sum::<f64>();
            groups += s0.len();
            intended.extend(w1.iter().zip(w0).map(|(a, b)| a - b));
            delivered.extend(deq.iter().zip(w0).map(|(a, b)| a - b));
        }

        let cos = cosine(&delivered, &intended);
        let share = unmoved as f64 / total as f64;
        let steps = dmax as f64 / (step_sum / groups as f64);
        eprintln!(
            "strength {strength}: |delta|max = {steps:.2} int8 steps -> {:.1}% of adapted weights keep the BASE int8 code, cosine(delivered, intended) {cos:.4}",
            100.0 * share
        );
        sweep.push((steps, cos, share));
    }

    // A delta around one step is mostly thrown away: those weights receive no
    // correction at all, and what survives is badly distorted.
    let small = sweep[0];
    assert!(small.0 < 2.0, "the smallest strength should put the delta around one int8 step, got {:.2}", small.0);
    assert!(small.2 > 0.5, "expected most weights to receive no delta at all, got {:.1}%", 100.0 * small.2);
    assert!(small.1 < 0.9, "expected the surviving delta to be badly distorted, cosine {:.4}", small.1);
    // And it is the ratio to the step that decides: fidelity climbs and the
    // discarded share shrinks as the delta grows past one.
    for w in sweep.windows(2) {
        assert!(w[1].1 > w[0].1, "delta fidelity did not improve as the delta grew: {:?} -> {:?}", w[0], w[1]);
        assert!(w[1].2 < w[0].2, "the discarded share did not shrink as the delta grew: {:?} -> {:?}", w[0], w[1]);
    }
}

/// **The structural guarantee.** A quantized base has no safe fold, and the
/// shared layer is where that is decided: [`model::adapter::AdapterSet`] hands
/// back a [`Delivery`] the caller must match on, and there is no way to obtain
/// the fold permit for [`BaseStorage::Quantized`]. A model crate therefore
/// cannot reach the fold-then-quantize path by forgetting about it.
#[test]
fn a_quantized_base_is_never_offered_a_fold() {
    let cfg = tiny_q8_cfg();
    let wc = tiny_wan(&cfg);
    let ts = q8_grid_weights(&wc);
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let ad = trained_adapter(&cfg, &base, 4);

    match ad.delivery(BaseStorage::Quantized, 1.0).expect("lora has a runtime form") {
        Delivery::Runtime(deltas) => {
            assert_eq!(deltas.len(), cfg.n_layers * LEAVES.len(), "every targeted linear needs a runtime correction");
        }
        Delivery::Fold(_) => panic!("a quantized base was offered a fold - the delta would be re-rounded onto the weight grid"),
    }
    // ...and a dense base still folds, because there nothing re-rounds it.
    match ad.delivery(BaseStorage::Dense, 1.0).expect("a dense base always has a delivery") {
        Delivery::Fold(_) => {}
        Delivery::Runtime(_) => panic!("a dense base must still fold - the runtime path costs two extra GEMMs for nothing"),
    }
}

/// **The numbers that matter.** With the base kept exactly as the checkpoint
/// says and the correction applied at runtime, the adapter's effect on a
/// layer's output is delivered to fp32 fidelity - against a fold, which loses
/// most of it (first test). Measured in OUTPUT space, which is what a
/// generation sees.
#[test]
fn the_runtime_correction_delivers_the_adapters_output_change() {
    let cfg = tiny_q8_cfg();
    let wc = tiny_wan(&cfg);
    let ts = q8_grid_weights(&wc);
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let ad = trained_adapter(&cfg, &base, 4);

    let mut folded = ts.clone();
    ad.fold_into_tensors(&mut folded, BaseStorage::Dense).expect("fold");

    let deltas = match ad.delivery(BaseStorage::Quantized, 1.0).expect("lora has a runtime form") {
        Delivery::Runtime(d) => d,
        Delivery::Fold(_) => panic!("a quantized base must not fold"),
    };

    let rows = 24usize;
    let mut want_cos = f64::INFINITY;
    let mut fold_cos = f64::INFINITY;
    for d in deltas.deltas() {
        let (shape, w0) = &ts[&d.key];
        let (n, k) = (shape[0], shape[1]);
        assert_eq!(d.pair.out, n, "{}: runtime correction is sized for the wrong output width", d.key);
        assert_eq!(d.pair.inn, k, "{}: runtime correction is sized for the wrong input width", d.key);

        // A smooth activation block, the way a flat image region reaches a
        // linear: one profile plus a small per-token wobble.
        let x: Vec<f32> = (0..rows * k)
            .map(|i| {
                let (t, c) = (i / k, i % k);
                let prof = 0.8 + 0.4 * ((c as f32 / k as f32) * std::f32::consts::TAU).sin();
                prof * (1.0 + 0.02 * ((t as f32 * 0.7).sin()))
            })
            .collect();
        let matvec = |w: &[f32], t: usize| -> Vec<f32> { (0..n).map(|o| (0..k).map(|c| w[o * k + c] * x[t * k + c]).sum::<f32>()).collect() };

        // The intended change: fp32 base + fp32 delta, minus fp32 base.
        let (_, wf) = &folded[&d.key];
        // What the base weights become on the device - bit-exact, because the
        // fixture is already on the grid and no delta was folded in.
        let (p, sw) = quantize_weight(w0, n, k);
        let deployed = dequantize_weight(&p, &sw, n, k);
        assert_eq!(&deployed, w0, "{}: an unfolded int8 base must requantize bit-exactly", d.key);

        // And the same tensor WITH the fold, which is what ships today.
        let (pf, swf) = quantize_weight(wf, n, k);
        let deployed_folded = dequantize_weight(&pf, &swf, n, k);

        let (mut intended, mut runtime, mut folded_out) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..rows {
            let yb = matvec(w0, t);
            let ye = matvec(wf, t);
            let yq = matvec(&deployed, t);
            let yf = matvec(&deployed_folded, t);
            // `h = A·x`, then `y += scale·B·h` - the runtime correction, exactly
            // as the device kernel computes it.
            let r = d.pair.r;
            let h: Vec<f32> = (0..r).map(|j| (0..k).map(|c| d.pair.a[j * k + c] * x[t * k + c]).sum::<f32>()).collect();
            for o in 0..n {
                // `B` already carries the delta scale - `RuntimeDelta` defines it
                // that way so an upload and a late fold cannot disagree about it.
                let corr: f32 = (0..r).map(|j| d.pair.b[o * r + j] * h[j]).sum::<f32>();
                intended.push(ye[o] - yb[o]);
                runtime.push(yq[o] + corr - yb[o]);
                folded_out.push(yf[o] - yb[o]);
            }
        }
        want_cos = want_cos.min(cosine(&runtime, &intended));
        fold_cos = fold_cos.min(cosine(&folded_out, &intended));
    }

    eprintln!("output-space delivery over {} linears: runtime cosine {want_cos:.6}, fold-then-quantize cosine {fold_cos:.6}", deltas.len());
    assert!(want_cos > 0.9999, "the runtime correction must deliver the adapter's output change, cosine {want_cos:.6}");
    assert!(fold_cos < want_cos, "the fold is not supposed to be the better of the two (fold {fold_cos:.6} vs runtime {want_cos:.6})");
}
