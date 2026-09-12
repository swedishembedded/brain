// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What INT8 requantization does to a LoRA-**folded** weight - the seam every
//! brain-native adapter's generation crosses and the direct Q8_0 route never
//! does.
//!
//! Swedish Embedded AB implements quantized inference and low-rank adapter
//! deployment for its clients. If your team needs expertise in INT8 weight
//! quantization or LoRA serving then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! ## The asymmetry these tests pin
//!
//! `Pipeline::build_dit` takes the streamed Q8_0 route only when every adapter
//! is a third-party `.safetensors`; with brain's own `.brain` container (or
//! none at all) the map route runs instead. With NO adapter the two are the
//! same model to the bit, because dequantizing a Q8_0 block and requantizing
//! it reproduces the block exactly (`tests/gguf_direct_int8.rs`). Fold a
//! delta in and that identity is gone: the group's absmax moves, a new scale
//! is derived, and **every** value in the group is re-rounded.
//!
//! So the base run's weights are the checkpoint's own, exactly, while an
//! adapted run's weights carry a full round of INT8 rounding error on top of
//! the delta. That is a property of the pipeline, not a property of LoRA, and
//! it is why "the artifact only appears with an adapter" is not by itself
//! evidence that the adapter is at fault.
//!
//! These tests keep the arithmetic honest (an independent reference recomputes
//! scales and codes from the spec, never by calling the function under test)
//! and bound what the error can look like, so a future change that makes the
//! requantization structured - periodic along the reduction axis, say - fails
//! here instead of in an image.

use model::int8::{dequantize_weight, quantize_weight, GROUP};
use model::lora::{fold_placements, Pair, Placement};

/// Deterministic values with both signs and a magnitude ramp along the row, so
/// neighbouring 32-element groups get genuinely different scales.
fn filler(seed: u64, n: usize, k: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n * k)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 32) as u32 as f32) / (u32::MAX as f32);
            (u - 0.5) * 2.0 * (1.0 + (i % k) as f32 / k as f32)
        })
        .collect()
}

/// The f32 a Q8_0 checkpoint actually decodes to: already on the INT8 grid, so
/// requantizing it is the identity. This is what the fold starts from.
fn q8_decoded(seed: u64, n: usize, k: usize) -> Vec<f32> {
    let raw = filler(seed, n, k);
    let (p, sw) = quantize_weight(&raw, n, k);
    dequantize_weight(&p, &sw, n, k)
}

/// A trained-looking rank-`r` pair. `b` is nonzero (a fresh `Pair::new` has
/// `b = 0`, i.e. no delta at all), `a` is the small init.
fn trained_pair(out: usize, inn: usize, r: usize, seed: u64, amp: f32) -> Pair {
    let f = |sd: u64, n: usize| -> Vec<f32> {
        let mut s = sd.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(7);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5) * 2.0 * amp
            })
            .collect()
    };
    Pair::from_ab(out, inn, r, f(seed, r * inn), f(seed ^ 0xABCD, out * r))
}

/// The reference: the spec of `group_scales`/`pack_row`, written out here
/// rather than called, so this test can disagree with the implementation.
fn reference_codes(row: &[f32]) -> (Vec<f32>, Vec<i32>) {
    let mut scales = Vec::new();
    let mut codes = Vec::new();
    for blk in row.chunks(GROUP) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = amax.max(1e-8) / 127.0;
        scales.push(s);
        for &v in blk {
            codes.push((v / s).round().clamp(-127.0, 127.0) as i32);
        }
    }
    (scales, codes)
}

/// Unpack the production `[n, k/4]` u32 words to signed codes.
fn unpack_codes(packed: &[u32]) -> Vec<i32> {
    packed
        .iter()
        .flat_map(|w| (0..4).map(move |b| (((w >> (8 * b)) as u8) as i8) as i32))
        .collect()
}

/// Fold `pair` into row-block `row0` of a fused `[3n, k]` tensor exactly the
/// way `flux2::lora`'s `placements()` does for `*_attn.qkv.weight`, then hand
/// back the whole fused tensor.
fn fold_qkv_rect(base_fused: &[f32], pair: &Pair, scale: f32, n: usize, k: usize, row0: usize) -> Vec<f32> {
    let key = "double_blocks.0.img_attn.qkv.weight";
    let mut ts = flux2::Tensors::new();
    ts.insert(key.to_string(), (vec![3 * n, k], base_fused.to_vec()));
    let p = Placement::fused(key.to_string(), pair, 3 * n * k, row0, k, 0);
    fold_placements(&mut ts, scale, &[p]).expect("fold");
    ts.remove(key).unwrap().1
}

/// The headline asymmetry, in numbers: requantizing the UNMODIFIED Q8_0-decoded
/// tensor is bit-exact (the no-adapter path loses nothing), and requantizing
/// the SAME tensor with a LoRA delta folded in re-rounds every value.
#[test]
fn a_fold_turns_an_exact_int8_requantization_into_a_lossy_one() {
    let (n, k, r) = (64usize, 256usize, 8usize);
    let base = q8_decoded(0x51a7, 3 * n, k);

    // (a) no delta: dequantize -> requantize is the identity, to the bit.
    let (p0, s0) = quantize_weight(&base[..n * k], n, k);
    let back = dequantize_weight(&p0, &s0, n, k);
    assert_eq!(&back[..], &base[..n * k], "requantizing an unmodified Q8_0-decoded tensor must be bit-exact");
    let (p0b, s0b) = quantize_weight(&back, n, k);
    assert_eq!(p0, p0b, "second requantization changed the packed words");
    assert_eq!(s0, s0b, "second requantization changed the scales");

    // (b) a trained adapter's delta, folded through the real placement code.
    let pair = trained_pair(n, k, r, 0xBEEF, 0.05);
    let folded = fold_qkv_rect(&base, &pair, 1.0, n, k, 0);
    let delta: Vec<f32> = folded.iter().zip(&base).map(|(f, b)| f - b).collect();
    let dmax = delta.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let wmax = base.iter().fold(0f32, |m, &v| m.max(v.abs()));
    assert!(dmax > 0.0, "fixture must actually fold a nonzero delta");
    // Rows n.. are outside the placement's rectangle and must be untouched.
    assert_eq!(&folded[n * k..], &base[n * k..], "fold escaped its row rectangle");

    let (p1, s1) = quantize_weight(&folded[..n * k], n, k);
    let deq = dequantize_weight(&p1, &s1, n, k);
    let err: Vec<f32> = deq.iter().zip(&folded[..n * k]).map(|(q, f)| q - f).collect();
    let emax = err.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let erms = (err.iter().map(|&e| (e as f64) * (e as f64)).sum::<f64>() / err.len() as f64).sqrt();
    let drms = (delta[..n * k].iter().map(|&d| (d as f64) * (d as f64)).sum::<f64>() / (n * k) as f64).sqrt();
    let step = s1.iter().sum::<f32>() / s1.len() as f32;

    // What the forward pass actually sees: a matmul reduces over k, so for a
    // smooth activation the per-row SUM of the weight error is the term that
    // survives. Report it against the delta's own row sum.
    let row_sum = |v: &[f32]| -> f64 {
        (0..n).map(|ro| v[ro * k..ro * k + k].iter().map(|&x| x as f64).sum::<f64>().abs()).sum::<f64>() / n as f64
    };

    // How much of the adapter actually reaches the device: the DELIVERED
    // delta is `dequantized(requantized(base + δ)) − base`, not δ.
    let delivered: Vec<f32> = deq.iter().zip(&base[..n * k]).map(|(q, b)| q - b).collect();
    let dot: f64 = delivered.iter().zip(&delta[..n * k]).map(|(x, y)| *x as f64 * *y as f64).sum();
    let nd = (delivered.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt();
    let ni = (delta[..n * k].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt();
    let cos = dot / (nd * ni).max(1e-30);
    let rel_l2 = (delivered.iter().zip(&delta[..n * k]).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>()).sqrt() / ni;
    // The cleanest "the adapter did not move this weight" measure: the int8
    // code the device receives is the one the BASE weight would have had.
    let same = unpack_codes(&p0).into_iter().zip(unpack_codes(&p1)).filter(|(a, b)| a == b).count();
    eprintln!(
        "delivered delta vs intended: cosine {cos:.4}, rel-L2 {rel_l2:.4}, \
         {same}/{} weights keep the BASE int8 code ({:.1}%) - the adapter did not move them at all",
        n * k,
        100.0 * same as f32 / (n * k) as f32,
    );

    eprintln!(
        "weight |max| {wmax:.4}, mean int8 step {step:.3e}\n\
         LoRA delta: |max| {dmax:.3e} ({:.1}% of |w|max), rms {drms:.3e}\n\
         requant error after fold: |max| {emax:.3e}, rms {erms:.3e} ({:.1}% of the delta's rms)\n\
         mean |Σ_k| per row: delta {:.3e}, requant error {:.3e} ({:.1}% of the delta's)",
        100.0 * dmax / wmax,
        100.0 * erms / drms,
        row_sum(&delta[..n * k]),
        row_sum(&err),
        100.0 * row_sum(&err) / row_sum(&delta[..n * k]),
    );

    // Correctness: every element's error is bounded by half of ITS OWN group's
    // step, as round-to-nearest requires - the groups have different scales, so
    // a single mean step is not the bound. Anything larger is a quantizer bug.
    let gs = k / GROUP;
    let worst = (0..n * k)
        .map(|i| err[i].abs() / (0.5 * s1[(i / k) * gs + (i % k) / GROUP]))
        .fold(0f32, f32::max);
    assert!(worst <= 1.0 + 1e-4, "some weight's requantization error is {worst:.4}x half its own group step");
    // ...and it is NOT zero, which is the whole point: the same tensor without
    // the fold requantized exactly.
    assert!(erms > 0.0, "a folded tensor requantized without any error - fixture is not exercising the seam");
    // A large share of the weights the adapter moved are deployed unchanged:
    // their delta was under half of their group's step.
    assert!(same * 2 > n * k, "expected most weights to keep the base int8 code, got {same}/{}", n * k);
}

/// How much of the adapter survives the requantization, as a function of how
/// large its delta is compared with one INT8 step.
///
/// The base weight sits exactly on the INT8 grid (it came off a Q8_0 block),
/// so `round((w + δ)/s)` returns the base code unchanged for every element
/// whose δ is under half a step: that element is deployed as if the adapter
/// did not exist. The delta is not merely noisy after a fold - it is itself
/// quantized to the BASE weight's grid, at roughly `|δ|/s` levels of
/// resolution.
///
/// This is the number that matters for a real run: a klein-9b Q8_0 group's
/// step is `max|w|/127`, so a delta worth a few tenths of a percent of the
/// weight magnitude is simply gone.
#[test]
fn a_fold_quantizes_the_lora_delta_itself_onto_the_base_weights_int8_grid() {
    let (n, k, r) = (64usize, 256usize, 8usize);
    let base = q8_decoded(0x51a7, 3 * n, k);
    let (p0, s0) = quantize_weight(&base[..n * k], n, k);
    let base_codes = unpack_codes(&p0);
    let step = s0.iter().sum::<f32>() / s0.len() as f32;

    let mut survival = Vec::new();
    for amp in [0.03f32, 0.08, 0.2, 0.45] {
        let pair = trained_pair(n, k, r, 0xBEEF, amp);
        let folded = fold_qkv_rect(&base, &pair, 1.0, n, k, 0);
        let delta: Vec<f32> = folded[..n * k].iter().zip(&base).map(|(f, b)| f - b).collect();
        let (p, sw) = quantize_weight(&folded[..n * k], n, k);
        let deq = dequantize_weight(&p, &sw, n, k);
        let delivered: Vec<f32> = deq.iter().zip(&base[..n * k]).map(|(q, b)| q - b).collect();
        let ni = delta.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let dot: f64 = delivered.iter().zip(&delta).map(|(x, y)| *x as f64 * *y as f64).sum();
        let nd = delivered.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let cos = if nd == 0.0 { 0.0 } else { dot / (nd * ni) };
        let unmoved = base_codes.iter().zip(unpack_codes(&p)).filter(|(a, b)| **a == *b).count();
        let dmax = delta.iter().fold(0f32, |m, &v| m.max(v.abs()));
        eprintln!(
            "delta |max| {dmax:.3e} = {:.2} int8 steps -> cosine(delivered, intended) {cos:.4}, {:.1}% of weights keep the base int8 code",
            dmax / step,
            100.0 * unmoved as f32 / (n * k) as f32
        );
        survival.push((dmax / step, cos, unmoved as f32 / (n * k) as f32));
    }

    // The smallest delta is under half a step for most weights, so most of
    // them are deployed unchanged - the adapter is partly a no-op.
    let small = survival[0];
    assert!(small.0 < 2.0, "fixture's smallest delta should be around one int8 step, got {:.2}", small.0);
    assert!(small.2 > 0.5, "expected most weights to receive no delta at all, got {:.1}%", 100.0 * small.2);
    // Fidelity improves as the delta grows past a step, and the share of
    // weights the adapter cannot move at all shrinks.
    for w in survival.windows(2) {
        assert!(w[1].1 > w[0].1, "delta fidelity did not improve as the delta grew: {:?} -> {:?}", w[0], w[1]);
        assert!(w[1].2 < w[0].2, "the dropped share did not shrink as the delta grew: {:?} -> {:?}", w[0], w[1]);
    }
    let big = survival.last().unwrap();
    assert!(big.0 > 10.0, "fixture's largest delta should be many int8 steps, got {:.1}", big.0);
    assert!(big.1 > 0.95, "a delta of {:.1} steps should survive well, cosine {:.4}", big.0, big.1);
}

/// The production requantization of a folded tensor agrees, code for code and
/// scale for scale, with the spec recomputed independently in this file.
#[test]
fn a_folded_tensors_group_scales_and_codes_match_an_independent_reference() {
    let (n, k, r) = (48usize, 192usize, 4usize);
    let base = q8_decoded(0xC0FFEE, 3 * n, k);
    let pair = trained_pair(n, k, r, 0x1234, 0.08);
    let folded = fold_qkv_rect(&base, &pair, 2.0, n, k, 0);

    let (packed, sw) = quantize_weight(&folded[..n * k], n, k);
    let codes = unpack_codes(&packed);
    let gs = k / GROUP;
    for ro in 0..n {
        let (rs, rc) = reference_codes(&folded[ro * k..ro * k + k]);
        assert_eq!(&sw[ro * gs..ro * gs + gs], &rs[..], "row {ro}: group scales differ from the reference");
        assert_eq!(&codes[ro * k..ro * k + k], &rc[..], "row {ro}: packed codes differ from the reference");
        // And the scale really is max|folded| / 127 over each 32-element block
        // of the FOLDED row - recomputed once more, the long way.
        for (g, &s) in rs.iter().enumerate() {
            let blk = &folded[ro * k + g * GROUP..ro * k + g * GROUP + GROUP];
            let mut amax = 0f32;
            for &v in blk {
                if v.abs() > amax {
                    amax = v.abs();
                }
            }
            assert_eq!(s, amax.max(1e-8) / 127.0, "row {ro} group {g}: scale is not max|folded|/127");
        }
    }

    // Round-trip: every value lands within half its own group's step.
    let deq = dequantize_weight(&packed, &sw, n, k);
    for ro in 0..n {
        for c in 0..k {
            let s = sw[ro * gs + c / GROUP];
            let e = (deq[ro * k + c] - folded[ro * k + c]).abs();
            assert!(e <= 0.5 * s + 1e-7, "row {ro} col {c}: error {e:.3e} > half a step {:.3e}", 0.5 * s);
        }
    }
}

/// Does the requantization of a folded tensor inject POSITION-DEPENDENT error?
///
/// A 4-pixel-period image artifact would need a weight error that is periodic
/// along the reduction axis, since that is the only axis the INT8 grouping
/// structures (one scale per 32 consecutive `k`). The delta here is smooth in
/// the column index by construction - a rank-2 product of low-frequency rows,
/// with no period-2, -4 or -32 content of its own - so any such peak in the
/// error would have been created by the quantizer.
///
/// The test measures the column-mean error's spectrum and requires no
/// frequency to dominate: in particular the period-4, period-2 and period-32
/// (group) lines must stay in the same band as the rest.
#[test]
fn requantization_error_after_a_fold_carries_no_periodic_structure() {
    let (n, k, r) = (128usize, 512usize, 2usize);
    let base = q8_decoded(0x5EED, 3 * n, k);

    // Smooth, low-frequency A rows (periods k and k/3) and a smooth B.
    let a: Vec<f32> = (0..r * k)
        .map(|i| {
            let (row, c) = (i / k, (i % k) as f32 / k as f32);
            0.06 * if row == 0 { (std::f32::consts::TAU * c).cos() } else { (std::f32::consts::TAU * 3.0 * c).sin() }
        })
        .collect();
    let b: Vec<f32> = (0..n * r)
        .map(|i| {
            let (o, j) = (i / r, i % r);
            0.9 * (0.3 + 0.7 * (o as f32 / n as f32)) * if j == 0 { 1.0 } else { -0.6 }
        })
        .collect();
    let pair = Pair::from_ab(n, k, r, a, b);
    let folded = fold_qkv_rect(&base, &pair, 1.0, n, k, 0);

    let (packed, sw) = quantize_weight(&folded[..n * k], n, k);
    let deq = dequantize_weight(&packed, &sw, n, k);
    let err: Vec<f32> = deq.iter().zip(&folded[..n * k]).map(|(q, f)| q - f).collect();

    // Column-mean error: averaging over rows kills the per-element rounding
    // noise and leaves exactly the position-dependent component a periodic
    // image artifact would need.
    let colmean: Vec<f64> = (0..k).map(|c| (0..n).map(|ro| err[ro * k + c] as f64).sum::<f64>() / n as f64).collect();

    // Naive DFT magnitude at every frequency; period p is frequency k/p.
    let mag = |f: usize| -> f64 {
        let (mut re, mut im) = (0f64, 0f64);
        for (c, &v) in colmean.iter().enumerate() {
            let th = -std::f64::consts::TAU * (f as f64) * (c as f64) / (k as f64);
            re += v * th.cos();
            im += v * th.sin();
        }
        (re * re + im * im).sqrt()
    };
    let mags: Vec<f64> = (1..k / 2).map(mag).collect();
    let mut sorted = mags.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = sorted[sorted.len() / 2];
    let peak = *sorted.last().unwrap();

    let p4 = mag(k / 4); // period 4
    let p2 = mag(k / 2); // period 2 (Nyquist-adjacent)
    let p32 = mag(k / GROUP); // period 32 - the INT8 group itself

    // Per-residue means, the other way the same claim can be read: if the
    // quantizer favoured one sub-position of every 4, this would show.
    let residue = |m: usize| -> Vec<f64> {
        (0..m).map(|j| colmean.iter().skip(j).step_by(m).sum::<f64>() / (k / m) as f64).collect()
    };
    let r4 = residue(4);
    let r2 = residue(2);
    let rms_col = (colmean.iter().map(|v| v * v).sum::<f64>() / k as f64).sqrt();

    eprintln!(
        "column-mean requant error: rms {rms_col:.3e}\n\
         DFT magnitudes: median {median:.3e}, max {peak:.3e}, period-4 {p4:.3e}, period-2 {p2:.3e}, period-32 {p32:.3e}\n\
         residue-4 means {r4:?}\n\
         residue-2 means {r2:?}"
    );

    // No line may dominate: a genuine periodic injection would stand several
    // times above the noise floor of the other 254 frequencies.
    for (name, v) in [("period-4", p4), ("period-2", p2), ("period-32", p32)] {
        assert!(
            v < 0.25 * peak.max(8.0 * median),
            "{name} line ({v:.3e}) dominates the column-mean error spectrum (median {median:.3e}, max {peak:.3e}) - the requantizer is injecting position-dependent error"
        );
    }
    // And no sub-position of a 4- or 2-wide pattern carries a systematic bias.
    for (m, means) in [(4usize, &r4), (2usize, &r2)] {
        for (j, &v) in means.iter().enumerate() {
            assert!(
                v.abs() < 0.5 * rms_col,
                "residue {j} mod {m} carries a systematic error bias ({v:.3e} vs column rms {rms_col:.3e})"
            );
        }
    }
}

/// What the requantization error does downstream: for the near-constant
/// activations of a SMOOTH image region it acts as a token-independent BIAS on
/// the layer's output, not as noise.
///
/// `y = W·x` reduces over `k`, so a weight error `E` contributes `E·x`. When
/// neighbouring tokens carry nearly the same `x` - which is what a flat wall
/// or a carpet is - `E·x` is nearly the same vector for all of them: a
/// constant offset added to every token's residual stream. A constant offset
/// is exactly the input that makes a decoder with nearest-2x upsampling
/// (`vae::VaeConfig::upscale_factor` = 8, three nearest-2x stages) draw its own
/// grid, which is why an error that looks like harmless rounding on the weight
/// can look like a hard periodic pattern in the image.
///
/// The test measures the ratio of the error's token-mean to its token-to-token
/// spread, and how big it is against the adapter's own intended effect.
#[test]
fn the_requantization_error_acts_as_a_constant_bias_on_smooth_activations() {
    let (n, k, r, t) = (64usize, 256usize, 8usize, 32usize);
    let base = q8_decoded(0x51a7, 3 * n, k);
    let pair = trained_pair(n, k, r, 0xBEEF, 0.05);
    let folded = fold_qkv_rect(&base, &pair, 1.0, n, k, 0);
    let (packed, sw) = quantize_weight(&folded[..n * k], n, k);
    let deployed = dequantize_weight(&packed, &sw, n, k);

    // A smooth region: every token's activation is the same profile plus a
    // 2% token-to-token wobble.
    let x: Vec<f32> = (0..t * k)
        .map(|i| {
            let (tok, c) = (i / k, i % k);
            let prof = 0.8 + 0.4 * ((c as f32 / k as f32) * std::f32::consts::TAU).sin();
            prof * (1.0 + 0.02 * ((tok as f32 * 0.7).sin()))
        })
        .collect();
    let matvec = |w: &[f32], tok: usize| -> Vec<f32> {
        (0..n)
            .map(|o| (0..k).map(|c| w[o * k + c] * x[tok * k + c]).sum::<f32>())
            .collect()
    };

    let mut err = vec![0f32; t * n];
    let mut intended = vec![0f32; t * n];
    for tok in 0..t {
        let ye = matvec(&folded[..n * k], tok);
        let yd = matvec(&deployed, tok);
        let yb = matvec(&base[..n * k], tok);
        for o in 0..n {
            err[tok * n + o] = yd[o] - ye[o];
            intended[tok * n + o] = ye[o] - yb[o];
        }
    }
    let mean_o: Vec<f64> = (0..n).map(|o| (0..t).map(|tok| err[tok * n + o] as f64).sum::<f64>() / t as f64).collect();
    let sd_o: Vec<f64> = (0..n)
        .map(|o| {
            let m = mean_o[o];
            ((0..t).map(|tok| (err[tok * n + o] as f64 - m).powi(2)).sum::<f64>() / t as f64).sqrt()
        })
        .collect();
    let bias = (mean_o.iter().map(|m| m * m).sum::<f64>() / n as f64).sqrt();
    let wobble = (sd_o.iter().map(|s| s * s).sum::<f64>() / n as f64).sqrt();
    let int_rms = (intended.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / intended.len() as f64).sqrt();
    let err_rms = (err.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / err.len() as f64).sqrt();

    eprintln!(
        "output-space requantization error on a smooth region: rms {err_rms:.4e}\n\
         token-constant component {bias:.4e} vs token-to-token spread {wobble:.4e} (ratio {:.1}x)\n\
         the adapter's own intended output change: rms {int_rms:.4e} - the error is {:.1}% of it",
        bias / wobble.max(1e-30),
        100.0 * err_rms / int_rms
    );
    assert!(
        bias > 5.0 * wobble,
        "expected the error to be a near-constant bias across tokens (bias {bias:.3e} vs spread {wobble:.3e})"
    );
}

/// Which tensors a brain-native fold actually moves. The 2x2-patch structure
/// (`vae::latent`'s 4-channel packing) only ever meets the DiT at `img_in` and
/// `final_layer.linear`; if a LoRA never touches those, no INT8 grouping of
/// their rows or columns can beat against a patch sub-position.
#[test]
fn a_lora_fold_never_touches_the_patch_boundary_tensors() {
    let c = flux2::modelgrad::Cfg::tiny();
    let fc = tiny_fc();
    let mut ad = flux2::lora::LoraAdapter::new(&c, flux2::lora::LoraCfg::new(4));
    // A fresh adapter has B = 0 and folds nothing; one step gives it a delta.
    let base = flux2::modelgrad::init_model::<f32>(&c, 0x51a7);
    let mut s = 0x1234_5678u64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    };
    let x0: Vec<f32> = (0..c.n_img() * c.in_channels).map(|_| rnd()).collect();
    let ctx: Vec<f32> = (0..c.txt_len * c.context_in_dim).map(|_| rnd()).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|_| rnd()).collect();
    let b = flux2::modelgrad::make_flow_batch(&c, &x0, &ctx, 0.4, &noise);
    let (_, g) = flux2::modelgrad::grads(&c, &ad.apply(&base), &b);
    ad.step(&g, 0.05);

    let mut ts = flux2::Tensors::new();
    let mut r = 0x00D0_0D00u64;
    let mut rn = move || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        ((r >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    };
    for (name, shape) in fc.tensor_manifest() {
        let n: usize = shape.iter().product();
        let (b0, sc) = if name.ends_with("norm.scale") { (1.0, 0.1) } else { (0.0, 0.2) };
        ts.insert(name, (shape, (0..n).map(|_| b0 + rn() * sc).collect()));
    }
    let before = ts.clone();
    ad.fold_into_tensors(&mut ts).expect("fold");

    let mut moved: Vec<&str> = Vec::new();
    for (name, (_, data)) in &ts {
        if data != &before[name].1 {
            moved.push(name.as_str());
        }
    }
    moved.sort_unstable();
    assert!(!moved.is_empty(), "the fold moved nothing - fixture broken");
    for name in &moved {
        let ok = name.ends_with("_attn.qkv.weight")
            || name.ends_with("_attn.proj.weight")
            || name.ends_with("_mlp.0.weight")
            || name.ends_with("_mlp.2.weight")
            || name.ends_with(".linear1.weight")
            || name.ends_with(".linear2.weight");
        assert!(ok, "a fold moved '{name}', which is not one of the block linears");
    }
    for name in before.keys() {
        let patch_boundary = name.starts_with("img_in") || name.starts_with("final_layer");
        assert!(
            !(patch_boundary && moved.contains(&name.as_str())),
            "a fold moved the patch-boundary tensor '{name}' - the 4-channel 2x2 packing DOES meet a LoRA delta"
        );
    }
    eprintln!("brain-native fold moves {} tensors, all block linears; img_in / final_layer untouched.", moved.len());
}

fn tiny_fc() -> flux2::Flux2Config {
    flux2::Flux2Config {
        in_channels: 4,
        context_in_dim: 6,
        hidden: 16,
        n_heads: 2,
        depth_double: 2,
        depth_single: 2,
        mlp_ratio: 0.75,
        axes_dim: [2, 2, 2, 2],
        txt_len: 3,
        ..flux2::Flux2Config::klein_4b()
    }
}
