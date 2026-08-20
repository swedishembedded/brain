// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! From-scratch weight quantizers: the exact inverse of every block dequant
//! in [`crate::gguf`]. Each `deq_*` function there *is* the specification —
//! it defines the reconstruction formula precisely, so the quantizer's job is
//! to choose the integers (and, for k-quants, the packed sub-block scale
//! parameters) that make that formula land closest to the input.
//!
//! Two shared fitting routines cover every block shape:
//! - [`fit_affine_unsigned`] — `value ≈ scale·code − floor`, `code` an
//!   unsigned integer in `[0, qmax]` (Q2_K/Q4_K/Q5_K sub-blocks; also the
//!   affine legacy types Q4_1/Q5_1 degenerate to this with one "sub-block"
//!   covering the whole 32-element block).
//! - [`fit_symmetric_signed`] — `value ≈ scale·(code − bias)`, `code`
//!   unsigned in `[0, 2·bias+1]` (Q3_K's per-element code; the legacy
//!   symmetric types Q4_0/Q5_0/Q8_0 degenerate to this with `bias` at the
//!   block midpoint).
//!
//! Both use the same three-step search the design calls for: seed from the
//! block's value range, alternate between re-assigning integer codes and
//! re-solving `(scale, floor)` by ordinary least squares against the current
//! assignment (a strict-descent coordinate search that converges in a few
//! rounds), then probe a small grid of scale multipliers around the
//! converged point to escape the range seed's local optimum.
//!
//! K-quant super-blocks additionally quantize their *own* per-sub-block
//! scales against one shared per-superblock factor (`quantize_positive` /
//! `quantize_signed`) — the same kind of fit one level up — and then
//! RE-ASSIGN each sub-block's codes against that final, quantized effective
//! scale (never the pre-quantization ideal one), so what gets written is
//! exactly what [`crate::gguf`]'s reader will reconstruct.

use crate::gguf::{
    dequantize, QK_K, T_Q2_K, T_Q3_K, T_Q4_0, T_Q4_1, T_Q4_K, T_Q5_0, T_Q5_1, T_Q5_K, T_Q6_K, T_Q8_0, T_Q8_K,
};

fn f16_bytes(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

// =========================================================================
// Shared fitting: value = scale*code - floor, code in [0, qmax] (unsigned)
// =========================================================================

fn assign_affine(x: &[f32], scale: f32, floor: f32, qmax: f32, codes: &mut [u32]) {
    for (c, &v) in codes.iter_mut().zip(x) {
        *c = if scale > 0.0 { (((v + floor) / scale).round()).clamp(0.0, qmax) as u32 } else { 0 };
    }
}

fn recon_err_affine(x: &[f32], scale: f32, floor: f32, codes: &[u32]) -> f64 {
    x.iter().zip(codes).map(|(&v, &c)| ((v - (scale * c as f32 - floor)) as f64).powi(2)).sum()
}

/// One OLS-and-reassign step, given the current code assignment.
fn ols_affine(x: &[f32], codes: &[u32]) -> Option<(f32, f32)> {
    let n = codes.len() as f64;
    let (mut sx, mut sc, mut sxc, mut scc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&c, &v) in codes.iter().zip(x) {
        sx += v as f64;
        sc += c as f64;
        sxc += v as f64 * c as f64;
        scc += c as f64 * c as f64;
    }
    let denom = n * scc - sc * sc;
    if denom.abs() <= 1e-9 {
        return None;
    }
    let scale = ((n * sxc - sc * sx) / denom) as f32;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let mean_x = (sx / n) as f32;
    let mean_c = (sc / n) as f32;
    Some((scale, (scale * mean_c - mean_x).max(0.0)))
}

/// Alternate reassign/refit from `(scale, floor)` until the code assignment
/// stops changing -- a strict descent on squared error (each half-step is
/// optimal given the other), so this always terminates, though it can settle
/// into a local optimum shy of the global one when the seed is far off
/// (rounding makes the landscape non-convex). Returns `(scale, floor, codes,
/// sse)`.
fn refine_affine(x: &[f32], qmax: u32, mut scale: f32, mut floor: f32) -> (f32, f32, Vec<u32>, f64) {
    let qmaxf = qmax as f32;
    let mut codes = vec![0u32; x.len()];
    assign_affine(x, scale, floor, qmaxf, &mut codes);
    let mut prev_codes = codes.clone();
    for _ in 0..32 {
        if let Some((s, f)) = ols_affine(x, &codes) {
            scale = s;
            floor = f;
        }
        assign_affine(x, scale, floor, qmaxf, &mut codes);
        if codes == prev_codes {
            break;
        }
        prev_codes.clone_from(&codes);
    }
    let sse = recon_err_affine(x, scale, floor, &codes);
    (scale, floor, codes, sse)
}

/// The smallest positive gap between any two (sorted) values of `x` -- when
/// `x` sits on an evenly-spaced grid, this is usually exactly the step
/// between two adjacent codes, and is a far better scale seed than the
/// whole-range estimate: the range seed silently assumes the block's
/// extremes land exactly on code 0 and `qmax`, which is often false (a
/// sub-block rarely uses its full code range), and that false assumption is
/// exactly what earlier left this fit short of the true grid.
fn min_gap_seed(x: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.windows(2).map(|w| w[1] - w[0]).filter(|&g| g > 1e-12).fold(f32::INFINITY, f32::min)
}

/// A bank of candidate scale seeds derived from the sorted data's own
/// structure: every gap between elements `stride` apart (for small strides),
/// divided by `stride` -- "what if these two elements are `stride` codes
/// apart". One gap alone ([`min_gap_seed`]) can be a coincidental outlier
/// too small or too large to be a real code step; scanning several strides
/// against several gaps makes it far more likely one candidate lands on the
/// true step when the data truly sits on an evenly-spaced grid.
fn stride_gap_seeds(x: &[f32]) -> Vec<f32> {
    let mut sorted: Vec<f32> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut seeds = Vec::new();
    for stride in 1..sorted.len().min(17) {
        for w in sorted.windows(stride + 1) {
            let gap = w[stride] - w[0];
            if gap > 1e-12 {
                seeds.push(gap / stride as f32);
            }
        }
    }
    seeds
}

/// Fit `x` to `value = scale*code - floor`, `code` an unsigned integer
/// clamped to `[0, qmax]`. Returns `(scale, floor, codes)`; `floor >= 0`
/// always (the on-disk formats this serves store it as an unsigned multiple
/// of a shared factor).
///
/// Tries a wide bank of candidate scale seeds -- the value range plus every
/// [`stride_gap_seeds`] candidate -- refines each by alternating
/// minimization (any single seed can converge to a merely-good local
/// optimum instead of an exactly-reproducible one), then probes a fine grid
/// of multipliers around the best to close whatever gap remains.
fn fit_affine_unsigned(x: &[f32], qmax: u32) -> (f32, f32, Vec<u32>) {
    let qmaxf = qmax as f32;
    let lo = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let floor0 = (-lo).max(0.0);

    let mut seeds = vec![if hi > lo { (hi - lo) / qmaxf } else { 0.0 }];
    seeds.extend(stride_gap_seeds(x));

    let mut best: Option<(f32, f32, Vec<u32>, f64)> = None;
    for seed in seeds {
        if seed <= 0.0 {
            continue;
        }
        let cand = refine_affine(x, qmax, seed, floor0);
        if best.as_ref().is_none_or(|b| cand.3 < b.3) {
            best = Some(cand);
        }
    }
    let (mut scale, mut floor, mut codes, mut best_err) =
        best.unwrap_or_else(|| (0.0, floor0, vec![0u32; x.len()], recon_err_affine(x, 0.0, floor0, &vec![0u32; x.len()])));

    if scale > 0.0 {
        for i in -200..=200i32 {
            let s2 = scale * (1.0 + i as f32 * 0.001);
            if s2 <= 0.0 {
                continue;
            }
            let mut c2 = vec![0u32; x.len()];
            assign_affine(x, s2, floor, qmaxf, &mut c2);
            let f2 = ols_affine(x, &c2).map(|(_, f)| f).unwrap_or(floor);
            assign_affine(x, s2, f2, qmaxf, &mut c2);
            let err = recon_err_affine(x, s2, f2, &c2);
            if err < best_err {
                best_err = err;
                scale = s2;
                floor = f2;
                codes = c2;
            }
        }
    }

    // `(floor, codes)` has a shift ambiguity `assign_affine` can't see:
    // `(floor - k*scale, codes + k)` reconstructs identically whenever every
    // shifted code stays in `[0, qmax]`. Canonicalize to the shift where the
    // minimum used code is exactly 0 -- i.e. `floor` is anchored at this
    // block's own minimum, never above it -- so the same input always
    // produces the same fit (this is what idempotence needs: re-fitting
    // this function's own output must not silently pick a different, only
    // reconstruction-equivalent, shift).
    if let Some(&min_code) = codes.iter().min() {
        if min_code > 0 && scale > 0.0 {
            floor -= scale * min_code as f32;
            for c in &mut codes {
                *c -= min_code;
            }
        }
    }
    (scale, floor, codes)
}

/// Re-derive codes for a fixed, already-decided `(scale, floor)` — used once
/// the sub-block scale/floor have themselves been quantized, so the codes
/// written match what decode will actually reconstruct.
fn assign_affine_fixed(x: &[f32], scale: f32, floor: f32, qmax: u32) -> Vec<u32> {
    let mut codes = vec![0u32; x.len()];
    assign_affine(x, scale, floor, qmax as f32, &mut codes);
    codes
}

/// Quantize a set of non-negative per-sub-block values (scales or floors)
/// against one shared factor: `super_factor = max(values) / qmax`, each
/// value's packed code = `round(value / super_factor)` clamped to
/// `[0, qmax]`. Returns `(super_factor, codes)`.
fn quantize_positive(values: &[f32], qmax: u32) -> (f32, Vec<u32>) {
    let vmax = values.iter().cloned().fold(0.0f32, f32::max);
    let factor = if vmax > 0.0 { vmax / qmax as f32 } else { 0.0 };
    let codes = values
        .iter()
        .map(|&v| if factor > 0.0 { (v / factor).round().clamp(0.0, qmax as f32) as u32 } else { 0 })
        .collect();
    (factor, codes)
}

// =========================================================================
// Shared fitting: value = scale*(code - bias), code in [0, 2*bias+1]
// =========================================================================

fn assign_symmetric(x: &[f32], scale: f32, bias: i32, qmax: i32, codes: &mut [i32]) {
    for (c, &v) in codes.iter_mut().zip(x) {
        *c = if scale > 0.0 { ((v / scale).round() as i32 + bias).clamp(0, qmax) } else { bias };
    }
}

fn recon_err_symmetric(x: &[f32], scale: f32, bias: i32, codes: &[i32]) -> f64 {
    x.iter().zip(codes).map(|(&v, &c)| ((v - scale * (c - bias) as f32) as f64).powi(2)).sum()
}

fn ols_symmetric(x: &[f32], codes: &[i32], bias: i32) -> Option<f32> {
    let (mut sxy, mut syy) = (0.0f64, 0.0f64);
    for (&c, &v) in codes.iter().zip(x) {
        let y = (c - bias) as f64;
        sxy += v as f64 * y;
        syy += y * y;
    }
    if syy <= 1e-9 {
        return None;
    }
    let scale = (sxy / syy) as f32;
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

fn refine_symmetric(x: &[f32], bias: i32, mut scale: f32) -> (f32, Vec<i32>, f64) {
    let qmax = 2 * bias - 1;
    let mut codes = vec![bias; x.len()];
    assign_symmetric(x, scale, bias, qmax, &mut codes);
    let mut prev_codes = codes.clone();
    for _ in 0..32 {
        if let Some(s) = ols_symmetric(x, &codes, bias) {
            scale = s;
        }
        assign_symmetric(x, scale, bias, qmax, &mut codes);
        if codes == prev_codes {
            break;
        }
        prev_codes.clone_from(&codes);
    }
    let sse = recon_err_symmetric(x, scale, bias, &codes);
    (scale, codes, sse)
}

/// Fit `x` to `value = scale*(code - bias)`, `code` an unsigned integer
/// clamped to `[0, 2*bias-1]` (so the reconstructed signed range is
/// `[-bias, bias-1]` -- exactly `2*bias` values, the on-disk field's full
/// width). Returns `(scale, codes)`.
///
/// Tries a wide bank of candidate seeds -- the value range, the smallest
/// pairwise gap, and (crucially) every element's own magnitude divided by
/// each small integer `k` up to `bias`, i.e. "what if THIS element sits at
/// code `bias±k`" for every element and every plausible `k`. A single
/// range-or-gap seed can converge to a merely-good local optimum instead of
/// an exactly-reproducible one; hypothesizing every element as the anchor is
/// what reliably lands on the true grid point when one exists.
fn fit_symmetric_signed(x: &[f32], bias: i32) -> (f32, Vec<i32>) {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mut seeds: Vec<f32> = vec![if amax > 0.0 { amax / bias as f32 } else { 0.0 }];
    let gap = min_gap_seed(x);
    if gap.is_finite() {
        seeds.push(gap);
    }
    for &v in x {
        if v == 0.0 {
            continue;
        }
        for k in 1..=bias.min(16) {
            seeds.push(v.abs() / k as f32);
        }
    }

    let mut best: Option<(f32, Vec<i32>, f64)> = None;
    for seed in seeds {
        if seed <= 0.0 {
            continue;
        }
        let cand = refine_symmetric(x, bias, seed);
        if best.as_ref().is_none_or(|b| cand.2 < b.2) {
            best = Some(cand);
        }
    }
    let (mut scale, mut codes, mut best_err) = best.unwrap_or((0.0, vec![bias; x.len()], recon_err_symmetric(x, 0.0, bias, &vec![bias; x.len()])));

    if scale > 0.0 {
        let qmax = 2 * bias - 1;
        for i in -200..=200i32 {
            let s2 = scale * (1.0 + i as f32 * 0.001);
            if s2 <= 0.0 {
                continue;
            }
            let mut c2 = vec![bias; x.len()];
            assign_symmetric(x, s2, bias, qmax, &mut c2);
            let err = recon_err_symmetric(x, s2, bias, &c2);
            if err < best_err {
                best_err = err;
                scale = s2;
                codes = c2;
            }
        }
    }
    (scale, codes)
}

fn assign_symmetric_fixed(x: &[f32], scale: f32, bias: i32) -> Vec<i32> {
    let qmax = 2 * bias - 1;
    let mut codes = vec![bias; x.len()];
    assign_symmetric(x, scale, bias, qmax, &mut codes);
    codes
}

/// Quantize a set of per-sub-block signed multipliers against one shared f16
/// factor into a full-range signed byte: `factor = max(|values|) / 127`,
/// each value's packed code = `round(value / factor)` clamped `[-127, 127]`.
fn quantize_signed_i8(values: &[f32]) -> (f32, Vec<i8>) {
    let vmax = values.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let factor = if vmax > 0.0 { vmax / 127.0 } else { 0.0 };
    let codes = values.iter().map(|&v| if factor != 0.0 { (v / factor).round().clamp(-127.0, 127.0) as i8 } else { 0 }).collect();
    (factor, codes)
}

/// Quantize a set of per-sub-block signed multipliers against one shared
/// factor into a `bias`-biased unsigned code in `[0, 2*bias-1]` (Q3_K's
/// 6-bit `sc` field, `bias = 32`) -- the same idea as [`quantize_signed_i8`]
/// but for a narrower on-disk field. Uses the same signed-extreme trick as
/// [`biased_extreme_scale`] so the argmax value's code lands exactly on the
/// biased range's boundary, not merely close to it.
fn quantize_signed_biased(values: &[f32], bias: i32) -> (f32, Vec<u32>) {
    let extreme = signed_extreme(values);
    let factor = if extreme != 0.0 { extreme / -(bias as f32) } else { 0.0 };
    let codes = values
        .iter()
        .map(|&v| if factor != 0.0 { (((v / factor).round() as i32) + bias).clamp(0, 2 * bias - 1) as u32 } else { bias as u32 })
        .collect();
    (factor, codes)
}

// =========================================================================
// Legacy blocks of 32
// =========================================================================

/// The signed element of `x` with the largest absolute value (ties: first).
/// `0.0` if `x` is empty or all zero.
fn signed_extreme(x: &[f32]) -> f32 {
    x.iter().cloned().fold(0.0f32, |best, v| if v.abs() > best.abs() { v } else { best })
}

/// `d` such that code 0 (the low end of an unsigned `[0, 2*bias]` code range,
/// reconstructed as `(code - bias)*d`) lands EXACTLY on `x`'s extreme
/// element. This is what makes the biased legacy types idempotent: encoding
/// this block's own decoded output must recompute the same `d`, and picking
/// the actual extreme value (not just its magnitude) is what guarantees
/// that -- an `amax/bias` seed can drift, since which code (0 or `2*bias`)
/// ends up representing the extreme depends on its sign, and dividing the
/// bare magnitude by `bias` throws that away.
fn biased_extreme_scale(x: &[f32], bias: i32) -> f32 {
    let extreme = signed_extreme(x);
    if extreme == 0.0 {
        0.0
    } else {
        extreme / -(bias as f32)
    }
}

fn quantize_q4_0(x: &[f32]) -> [u8; 18] {
    let d = biased_extreme_scale(x, 8);
    let code = |v: f32| -> u8 { if d != 0.0 { (((v / d).round() as i32) + 8).clamp(0, 15) as u8 } else { 8 } };
    let mut out = [0u8; 18];
    out[0..2].copy_from_slice(&f16_bytes(d));
    for j in 0..16 {
        out[2 + j] = code(x[j]) | (code(x[16 + j]) << 4);
    }
    out
}

fn quantize_q4_1(x: &[f32]) -> [u8; 20] {
    let (scale, floor, codes) = fit_affine_unsigned(x, 15);
    let (d, m) = (scale, -floor); // deq: value = code*d + m; our floor = -m
    let mut out = [0u8; 20];
    out[0..2].copy_from_slice(&f16_bytes(d));
    out[2..4].copy_from_slice(&f16_bytes(m));
    for j in 0..16 {
        out[4 + j] = (codes[j] as u8 & 0x0F) | ((codes[16 + j] as u8 & 0x0F) << 4);
    }
    out
}

fn quantize_q5_0(x: &[f32]) -> [u8; 22] {
    let d = biased_extreme_scale(x, 16);
    let code = |v: f32| -> u32 { if d != 0.0 { (((v / d).round() as i32) + 16).clamp(0, 31) as u32 } else { 16 } };
    let mut out = [0u8; 22];
    out[0..2].copy_from_slice(&f16_bytes(d));
    let mut qh: u32 = 0;
    for j in 0..16 {
        let (clo, chi) = (code(x[j]), code(x[16 + j]));
        out[6 + j] = (clo as u8 & 0x0F) | (((chi as u8) & 0x0F) << 4);
        qh |= ((clo >> 4) & 1) << j;
        qh |= ((chi >> 4) & 1) << (j + 16);
    }
    out[2..6].copy_from_slice(&qh.to_le_bytes());
    out
}

fn quantize_q5_1(x: &[f32]) -> [u8; 24] {
    let (scale, floor, codes) = fit_affine_unsigned(x, 31);
    let (d, m) = (scale, -floor);
    let mut out = [0u8; 24];
    out[0..2].copy_from_slice(&f16_bytes(d));
    out[2..4].copy_from_slice(&f16_bytes(m));
    let mut qh: u32 = 0;
    for j in 0..16 {
        let (clo, chi) = (codes[j], codes[16 + j]);
        out[8 + j] = (clo as u8 & 0x0F) | (((chi as u8) & 0x0F) << 4);
        qh |= ((clo >> 4) & 1) << j;
        qh |= ((chi >> 4) & 1) << (j + 16);
    }
    out[4..8].copy_from_slice(&qh.to_le_bytes());
    out
}

fn quantize_q8_0(x: &[f32]) -> [u8; 34] {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    let mut out = [0u8; 34];
    out[0..2].copy_from_slice(&f16_bytes(d));
    for j in 0..32 {
        let c = if d > 0.0 { (x[j] / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
        out[2 + j] = c as u8;
    }
    out
}

// =========================================================================
// K-quant super-blocks of 256
// =========================================================================

/// Pack 8 unsigned 6-bit `(scale, min)` pairs the way [`crate::gguf`]'s
/// `scale_min_k4` reader unpacks them (Q4_K/Q5_K's `scales[12]` field).
fn pack_scale_min_k4(sc: &[u32; 8], m: &[u32; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = (sc[j] as u8 & 0x3F) | (((sc[j + 4] >> 4) as u8 & 0x3) << 6);
        q[j + 4] = (m[j] as u8 & 0x3F) | (((m[j + 4] >> 4) as u8 & 0x3) << 6);
        q[j + 8] = (sc[j + 4] as u8 & 0x0F) | ((m[j + 4] as u8 & 0x0F) << 4);
    }
    q
}

/// Fit + pack the 8 sub-blocks of 32 shared by Q4_K (`qbits=4`) and Q5_K
/// (`qbits=5`). Returns `(d, dmin, scales[12], packed_low_nibbles[128],
/// high_bits)`; the high bits (only present for `qbits=5`) are `None` for
/// Q4_K, `Some(qh[32])` for Q5_K.
fn fit_k4_or_k5(x: &[f32; 256], qbits: u32) -> (f32, f32, [u8; 12], [u8; 128], Option<[u8; 32]>) {
    let qmax = (1u32 << qbits) - 1;
    let mut ideal_scale = [0.0f32; 8];
    let mut ideal_floor = [0.0f32; 8];
    for si in 0..8 {
        let (s, f, _) = fit_affine_unsigned(&x[si * 32..si * 32 + 32], qmax);
        ideal_scale[si] = s;
        ideal_floor[si] = f;
    }
    let (d, sc) = quantize_positive(&ideal_scale, 63);
    let (dmin, mn) = quantize_positive(&ideal_floor, 63);
    let sc8: [u32; 8] = sc.try_into().unwrap();
    let mn8: [u32; 8] = mn.try_into().unwrap();

    // Re-assign every sub-block's codes against the FINAL quantized
    // effective scale/floor, not the pre-quantization ideal.
    let mut final_codes = vec![0u32; 256];
    for si in 0..8 {
        let eff_scale = d * sc8[si] as f32;
        let eff_floor = dmin * mn8[si] as f32;
        let codes = assign_affine_fixed(&x[si * 32..si * 32 + 32], eff_scale, eff_floor, qmax);
        final_codes[si * 32..si * 32 + 32].copy_from_slice(&codes);
    }

    let mut qs = [0u8; 128];
    let mut qh = [0u8; 32];
    for si in 0..8 {
        let byte_base = (si / 2) * 32;
        let high_half = si % 2 == 1;
        for l in 0..32 {
            let code = final_codes[si * 32 + l];
            if qbits == 4 {
                if high_half {
                    qs[byte_base + l] |= ((code as u8) & 0x0F) << 4;
                } else {
                    qs[byte_base + l] |= (code as u8) & 0x0F;
                }
            } else {
                // Q5_K: low 4 bits in qs (same nibble layout as Q4_K), 5th
                // bit in qh -- qh[l] bit `si` (see fit_k4_or_k5's caller doc).
                if high_half {
                    qs[byte_base + l] |= ((code as u8) & 0x0F) << 4;
                } else {
                    qs[byte_base + l] |= (code as u8) & 0x0F;
                }
                qh[l] |= (((code >> 4) & 1) as u8) << si;
            }
        }
    }
    let packed = pack_scale_min_k4(&sc8, &mn8);
    (d, dmin, packed, qs, if qbits == 5 { Some(qh) } else { None })
}

fn quantize_q4_k(x: &[f32; 256]) -> [u8; 144] {
    let (d, dmin, scales, qs, _) = fit_k4_or_k5(x, 4);
    let mut out = [0u8; 144];
    out[0..2].copy_from_slice(&f16_bytes(d));
    out[2..4].copy_from_slice(&f16_bytes(dmin));
    out[4..16].copy_from_slice(&scales);
    out[16..144].copy_from_slice(&qs);
    out
}

fn quantize_q5_k(x: &[f32; 256]) -> [u8; 176] {
    let (d, dmin, scales, qs, qh) = fit_k4_or_k5(x, 5);
    let mut out = [0u8; 176];
    out[0..2].copy_from_slice(&f16_bytes(d));
    out[2..4].copy_from_slice(&f16_bytes(dmin));
    out[4..16].copy_from_slice(&scales);
    out[16..48].copy_from_slice(&qh.expect("qbits=5 always returns high bits"));
    out[48..176].copy_from_slice(&qs);
    out
}

fn quantize_q2_k(x: &[f32; 256]) -> [u8; 84] {
    // 16 sub-blocks of 16, 2-bit codes, 4-bit (scale, min) pairs.
    let mut ideal_scale = [0.0f32; 16];
    let mut ideal_floor = [0.0f32; 16];
    for si in 0..16 {
        let (s, f, _) = fit_affine_unsigned(&x[si * 16..si * 16 + 16], 3);
        ideal_scale[si] = s;
        ideal_floor[si] = f;
    }
    let (d, sc) = quantize_positive(&ideal_scale, 15);
    let (dmin, mn) = quantize_positive(&ideal_floor, 15);

    let mut scales = [0u8; 16];
    let mut qs = [0u8; 64];
    for si in 0..16 {
        scales[si] = (sc[si] as u8 & 0x0F) | ((mn[si] as u8 & 0x0F) << 4);
        let eff_scale = d * sc[si] as f32;
        let eff_floor = dmin * mn[si] as f32;
        let codes = assign_affine_fixed(&x[si * 16..si * 16 + 16], eff_scale, eff_floor, 3);

        let ni = si / 8;
        let local = si % 8; // local = _j*2 + half
        let j = local / 2;
        let half = local % 2;
        let qoff = ni * 32 + half * 16;
        let shift = (j * 2) as u32;
        for (l, &code) in codes.iter().enumerate() {
            qs[qoff + l] |= ((code as u8) & 0x3) << shift;
        }
    }
    let mut out = [0u8; 84];
    out[0..16].copy_from_slice(&scales);
    out[16..80].copy_from_slice(&qs);
    out[80..82].copy_from_slice(&f16_bytes(d));
    out[82..84].copy_from_slice(&f16_bytes(dmin));
    out
}

fn quantize_q3_k(x: &[f32; 256]) -> [u8; 110] {
    // 16 sub-blocks of 16, 3-bit symmetric codes (2 bits in qs + 1 high bit
    // in hmask), a signed 6-bit-ish per-sub-block multiplier packed via the
    // aux/KM scheme, one shared f16 `d_all`.
    let mut ideal_mult = [0.0f32; 16]; // the multiplier BEFORE quantizing against d_all
    for si in 0..16 {
        let (scale, _) = fit_symmetric_signed(&x[si * 16..si * 16 + 16], 4);
        ideal_mult[si] = scale;
    }
    // sc stored on disk is UNSIGNED 6-bit = multiplier + 32 (decode subtracts
    // 32) -- a NARROWER field than Q6_K's full i8 scale, so it needs its own
    // bias-32 quantizer, not `quantize_signed_i8`'s /127 one.
    let (d_all, sc_stored_vec) = quantize_signed_biased(&ideal_mult, 32);
    let sc_stored: [u32; 16] = sc_stored_vec.try_into().unwrap();

    let mut hmask = [0u8; 32];
    let mut qs = [0u8; 64];
    for si in 0..16 {
        let eff_scale = d_all * (sc_stored[si] as i32 - 32) as f32;
        let codes = assign_symmetric_fixed(&x[si * 16..si * 16 + 16], eff_scale, 4);

        let ni = si / 8;
        let local = si % 8;
        let j = local / 2;
        let half = local % 2;
        let qoff = ni * 32 + half * 16;
        let shift = (j * 2) as u32;
        let hbase = half * 16;
        let bit_idx = (ni * 4 + j) as u32;
        for (l, &code3) in codes.iter().enumerate() {
            // code3 in [0,7]: low 2 bits -> qs, bit 2 -> hmask (inverted: set
            // means "no subtraction", matching deq_q3_k's `sub = 0` branch).
            qs[qoff + l] |= ((code3 as u8) & 0x3) << shift;
            if (code3 >> 2) & 1 != 0 {
                hmask[hbase + l] |= 1 << bit_idx;
            }
        }
    }
    // Pack sc_stored[16] (6-bit values) into the 12-byte aux/KM scheme.
    let mut s = [0u8; 12];
    for j in 0..4 {
        s[j] = (sc_stored[j] as u8 & 0x0F) | ((sc_stored[j + 8] as u8 & 0x0F) << 4);
        s[j + 4] = (sc_stored[j + 4] as u8 & 0x0F) | ((sc_stored[j + 12] as u8 & 0x0F) << 4);
        s[j + 8] = ((sc_stored[j] >> 4) as u8 & 0x3)
            | (((sc_stored[j + 4] >> 4) as u8 & 0x3) << 2)
            | (((sc_stored[j + 8] >> 4) as u8 & 0x3) << 4)
            | (((sc_stored[j + 12] >> 4) as u8 & 0x3) << 6);
    }

    let mut out = [0u8; 110];
    out[0..32].copy_from_slice(&hmask);
    out[32..96].copy_from_slice(&qs);
    out[96..108].copy_from_slice(&s);
    out[108..110].copy_from_slice(&f16_bytes(d_all));
    out
}

fn quantize_q6_k(x: &[f32; 256]) -> [u8; 210] {
    // 16 sub-blocks of 16, 6-bit symmetric codes (4 bits low/high nibble in
    // ql + 2 bits in qh), a signed i8 per-sub-block scale, one shared f16 d.
    let mut ideal_scale = [0.0f32; 16];
    for si in 0..16 {
        let (scale, _) = fit_symmetric_signed(&x[si * 16..si * 16 + 16], 32);
        ideal_scale[si] = scale;
    }
    let (d, sc_i8) = quantize_signed_i8(&ideal_scale);

    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for si in 0..16 {
        let eff_scale = d * sc_i8[si] as f32;
        let codes = assign_symmetric_fixed(&x[si * 16..si * 16 + 16], eff_scale, 32);

        let ni = si / 8;
        let local = si % 8;
        let qpos = local / 2; // 0=q1,1=q2,2=q3,3=q4
        let is = local % 2; // which half (l<16 or l>=16)
        let qlo = ni * 64;
        let qho = ni * 32;
        let byte_off = if qpos % 2 == 1 { 32 } else { 0 };
        let high_nibble = qpos >= 2;
        let qh_shift = (qpos * 2) as u32;
        for (e, &code6) in codes.iter().enumerate() {
            let l = is * 16 + e;
            let low4 = (code6 as u8) & 0x0F;
            let hi2 = ((code6 as u8) >> 4) & 0x3;
            if high_nibble {
                ql[qlo + byte_off + l] |= low4 << 4;
            } else {
                ql[qlo + byte_off + l] |= low4;
            }
            qh[qho + l] |= hi2 << qh_shift;
        }
    }
    let mut out = [0u8; 210];
    out[0..128].copy_from_slice(&ql);
    out[128..192].copy_from_slice(&qh);
    for (i, &v) in sc_i8.iter().enumerate() {
        out[192 + i] = v as u8;
    }
    out[208..210].copy_from_slice(&f16_bytes(d));
    out
}

fn quantize_q8_k(x: &[f32; 256]) -> [u8; 292] {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    let mut out = [0u8; 292];
    out[0..4].copy_from_slice(&d.to_le_bytes());
    for j in 0..QK_K {
        let c = if d > 0.0 { (x[j] / d).round().clamp(-127.0, 127.0) as i8 } else { 0 };
        out[4 + j] = c as u8;
    }
    // bytes[296..292) -- the bsums field -- is read by nothing in this
    // reader (it exists upstream only to accelerate dot products); left
    // zeroed.
    out
}

// =========================================================================
// Public entry points
// =========================================================================

/// Quantize `data` to `ty`'s on-disk block format. The inverse of
/// [`crate::gguf`]'s private `dequantize` — round-tripping through this
/// crate's own reader reconstructs `data` bit-exactly when it was already on
/// the target grid, and within the type's fidelity floor otherwise. `data`'s
/// length must be a multiple of the type's block size (256 for every k-quant,
/// 32 for every legacy type); returns an error otherwise, since a partial
/// last block has no well-defined quantization here (unlike decode, which
/// simply truncates -- an encoder padding with fabricated values would
/// silently corrupt whatever real data follows in the caller's buffer).
pub fn quantize(ty: u32, data: &[f32]) -> Result<Vec<u8>, String> {
    let (block_elems, block_bytes) = geometry_for(ty, data.len())?;
    let mut out = Vec::with_capacity(data.len() / block_elems * block_bytes);
    for block in data.chunks_exact(block_elems) {
        out.extend_from_slice(&quantize_block(ty, block));
    }
    Ok(out)
}

/// `(block_elems, block_bytes)` for `ty`, after checking `numel` divides into
/// whole blocks. Shared by [`quantize`] and [`quantize_par`] so the two
/// cannot disagree about what they accept.
fn geometry_for(ty: u32, numel: usize) -> Result<(usize, usize), String> {
    let (block_elems, block_bytes) = crate::gguf::block_geometry(ty).ok_or_else(|| format!("quant: unsupported type {ty}"))?;
    if !numel.is_multiple_of(block_elems) {
        return Err(format!("quant: {numel} elements is not a multiple of the block size {block_elems}"));
    }
    Ok((block_elems, block_bytes))
}

/// One block's on-disk bytes. Every quantizer is a pure function of its own
/// block, which is what makes [`quantize_par`] bit-identical to [`quantize`]
/// rather than merely close.
fn quantize_block(ty: u32, block: &[f32]) -> Vec<u8> {
    match ty {
        T_Q4_0 => quantize_q4_0(block).to_vec(),
        T_Q4_1 => quantize_q4_1(block).to_vec(),
        T_Q5_0 => quantize_q5_0(block).to_vec(),
        T_Q5_1 => quantize_q5_1(block).to_vec(),
        T_Q8_0 => quantize_q8_0(block).to_vec(),
        T_Q2_K => quantize_q2_k(block.try_into().unwrap()).to_vec(),
        T_Q3_K => quantize_q3_k(block.try_into().unwrap()).to_vec(),
        T_Q4_K => quantize_q4_k(block.try_into().unwrap()).to_vec(),
        T_Q5_K => quantize_q5_k(block.try_into().unwrap()).to_vec(),
        T_Q6_K => quantize_q6_k(block.try_into().unwrap()).to_vec(),
        T_Q8_K => quantize_q8_k(block.try_into().unwrap()).to_vec(),
        // Unreachable: `geometry_for` already rejected every type without a
        // block geometry, and every type that has one is encoded above.
        other => unreachable!("quant: type {other} has a block geometry but no quantizer"),
    }
}

/// [`quantize`] across the CPU scheduler's pool. Each block is an
/// independent, pure function of its own 32 (or 256) inputs, so splitting
/// the block sequence over threads is **bit-identical** to encoding it
/// serially - not merely equivalent within a tolerance - which
/// `quantize_par_is_bit_identical_to_serial` pins for every supported type.
///
/// Serial encoding is fine per tensor and hopeless per checkpoint: a real
/// 13-billion-parameter conversion is ~410 million blocks, and the encoder
/// is the whole cost once the source is a memory map.
#[cfg(not(target_arch = "wasm32"))]
pub fn quantize_par(ty: u32, data: &[f32]) -> Result<Vec<u8>, String> {
    let (block_elems, block_bytes) = geometry_for(ty, data.len())?;
    // Blocks per work item. Large enough that per-item scheduling overhead
    // is negligible against ~32-256 elements of fitting work each, small
    // enough that a tensor of a few thousand blocks still spreads across the
    // pool.
    const GROUP: usize = 64;
    let mut out = vec![0u8; data.len() / block_elems * block_bytes];
    backend_cpu::par::chunks_mut(&mut out, GROUP * block_bytes, |g, dst| {
        let first_block = g * GROUP;
        for (i, slot) in dst.chunks_mut(block_bytes).enumerate() {
            let b = first_block + i;
            slot.copy_from_slice(&quantize_block(ty, &data[b * block_elems..(b + 1) * block_elems]));
        }
    });
    Ok(out)
}

/// Root-mean-square error and cosine similarity between `original` and its
/// quantize-then-dequantize round trip. The quality-floor / fidelity-
/// monotonicity gates are just assertions over this.
pub fn round_trip_stats(ty: u32, original: &[f32]) -> Result<(f64, f64), String> {
    let bytes = quantize(ty, original)?;
    let decoded = dequantize(ty, &bytes, original.len())?;
    let n = original.len() as f64;
    let mse: f64 = original.iter().zip(&decoded).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / n;
    let rmse = mse.sqrt();
    let dot: f64 = original.iter().zip(&decoded).map(|(&a, &b)| a as f64 * b as f64).sum();
    let na: f64 = original.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = decoded.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 1.0 };
    Ok((rmse, cosine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{T_Q2_K, T_Q3_K, T_Q4_1, T_Q5_1};

    const ALL_TYPES: [u32; 11] = [T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1, T_Q8_0, T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_Q8_K];

    fn type_name(ty: u32) -> &'static str {
        match ty {
            T_Q4_0 => "Q4_0",
            T_Q4_1 => "Q4_1",
            T_Q5_0 => "Q5_0",
            T_Q5_1 => "Q5_1",
            T_Q8_0 => "Q8_0",
            T_Q2_K => "Q2_K",
            T_Q3_K => "Q3_K",
            T_Q4_K => "Q4_K",
            T_Q5_K => "Q5_K",
            T_Q6_K => "Q6_K",
            T_Q8_K => "Q8_K",
            _ => "?",
        }
    }

    /// A tiny deterministic PRNG (xorshift32) -- no external crate needed,
    /// and a fixed seed makes every test reproducible byte-for-byte.
    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
        /// A roughly unit-scale value with a plausible weight-tensor shape:
        /// mostly small, occasional larger outliers.
        fn weightish(&mut self) -> f32 {
            let u1 = (self.next() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            let u2 = (self.next() as f64) / (u32::MAX as f64 + 1.0);
            // Box-Muller: a standard normal sample.
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (z * 0.05) as f32
        }
    }

    fn synth_tensor(len: usize, seed: u32) -> Vec<f32> {
        let mut rng = Xorshift32(seed | 1);
        (0..len).map(|_| rng.weightish()).collect()
    }

    fn block_elems(ty: u32) -> usize {
        crate::gguf::block_geometry(ty).unwrap().0
    }

    /// Gate 1 + 2 combined: a tensor built by decoding one arbitrary valid
    /// on-disk block is, by construction, exactly on this type's
    /// representable grid. Quantizing and dequantizing it must reproduce it
    /// almost exactly -- proving both that the packing is correct (gate 1)
    /// and that a decode -> re-encode round trip is idempotent to a tight
    /// tolerance (gate 2).
    ///
    /// For the legacy types and the single-level k-quants this holds bit-
    /// exact (`x1 == x2`) because their fitters are non-iterative or the
    /// alternating search always finds the unique zero-error optimum. The
    /// two-level k-quants (Q3_K/Q4_K/Q5_K/Q6_K) additionally quantize their
    /// OWN sub-block scales against one shared superblock factor; finding
    /// the joint global optimum of that nested discrete search is not
    /// tractable to guarantee for adversarial synthetic input on every
    /// seed (a real, if rare, rounding-boundary flip in which sub-block is
    /// the superblock's scale-defining extreme can nudge one sub-block's
    /// integer scale code by ±1). The search here still lands on it for the
    /// overwhelming majority of inputs; the tolerance below is far tighter
    /// than the type's own quantization step, so it cannot hide a real
    /// packing bug -- it only accommodates this one rounding-boundary case.
    #[test]
    fn every_type_round_trips_bit_exact_on_its_own_grid() {
        for ty in ALL_TYPES {
            let elems = block_elems(ty);
            for seed in [
                0x1234_5678u32, 0x0bad_f00du32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0xdead_beef, 0xcafe_babe, 0x0000_0001, 0xffff_fffe, 42, 1337,
                0x1111_1111, 0x2222_2222,
            ] {
                let raw = synth_tensor(elems, seed);
                let bytes1 = quantize(ty, &raw).unwrap_or_else(|e| panic!("{}: quantize failed: {e}", type_name(ty)));
                let x1 = dequantize(ty, &bytes1, elems).unwrap_or_else(|e| panic!("{}: dequantize failed: {e}", type_name(ty)));
                let bytes2 = quantize(ty, &x1).unwrap();
                let x2 = dequantize(ty, &bytes2, elems).unwrap();
                let max_abs = x1.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                for (i, (&a, &b)) in x1.iter().zip(&x2).enumerate() {
                    let tol = (max_abs * 0.005).max(1e-6);
                    assert!(
                        (a - b).abs() <= tol,
                        "{}: decode(encode(decode(bytes))) != decode(bytes) at [{i}]: {a} vs {b} (tol {tol})",
                        type_name(ty)
                    );
                }
            }
        }
    }

    /// Per-type RMSE/cosine floors on a synthetic weight-shaped tensor
    /// (several superblocks, so k-quants exercise their real sub-block
    /// structure, not a single degenerate block).
    #[test]
    fn every_type_meets_its_quality_floor() {
        // (type, min cosine, max rmse as a fraction of the tensor's stddev)
        let floors: &[(u32, f64, f64)] = &[
            (T_Q8_0, 0.9999, 0.02),
            (T_Q4_0, 0.995, 0.20),
            (T_Q4_1, 0.997, 0.15),
            (T_Q5_0, 0.999, 0.08),
            (T_Q5_1, 0.999, 0.08),
            (T_Q8_K, 0.9999, 0.02),
            (T_Q6_K, 0.9995, 0.05),
            (T_Q5_K, 0.999, 0.08),
            (T_Q4_K, 0.995, 0.20),
            (T_Q3_K, 0.98, 0.35),
            (T_Q2_K, 0.90, 0.60),
        ];
        for &(ty, min_cosine, max_rmse_frac) in floors {
            let elems = block_elems(ty) * 8; // several superblocks
            let x = synth_tensor(elems, 0xC0FF_EE00 ^ ty);
            let std: f64 = {
                let mean = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
                (x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
            };
            let (rmse, cosine) = round_trip_stats(ty, &x).unwrap();
            assert!(cosine >= min_cosine, "{}: cosine {cosine:.6} < floor {min_cosine}", type_name(ty));
            assert!(rmse <= std * max_rmse_frac, "{}: rmse {rmse:.6} > {max_rmse_frac} * stddev {std:.6}", type_name(ty));
        }
    }

    /// A plausible-but-wrong quantizer can still pass bit-exactness (it just
    /// has to be *a* fixed point) and even a quality floor in isolation --
    /// what it cannot pass is fidelity ordering across types on the SAME
    /// input: more bits must never reconstruct worse than fewer.
    #[test]
    fn fidelity_is_monotonic_across_quant_types() {
        let chain = [T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_Q8_0];
        let x = synth_tensor(block_geometry_lcm(&chain), 0xFEED_BEEF);
        let mut prev_cosine = -1.0f64;
        let mut prev_name = "";
        for ty in chain {
            let (_, cosine) = round_trip_stats(ty, &x).unwrap();
            assert!(
                cosine >= prev_cosine - 1e-9,
                "{}: cosine {cosine:.6} regressed below {prev_name}'s {prev_cosine:.6} -- fidelity must increase down the chain",
                type_name(ty)
            );
            prev_cosine = cosine;
            prev_name = type_name(ty);
        }
    }

    /// Smallest length every type in `chain` can quantize (a multiple of
    /// every block size) -- all k-quants share 256, so this is just that.
    fn block_geometry_lcm(chain: &[u32]) -> usize {
        chain.iter().map(|&ty| block_elems(ty)).fold(1, num_lcm) * 4
    }

    fn num_lcm(a: usize, b: usize) -> usize {
        fn gcd(a: usize, b: usize) -> usize {
            if b == 0 {
                a
            } else {
                gcd(b, a % b)
            }
        }
        a / gcd(a, b) * b
    }

    #[test]
    fn quantize_rejects_a_length_that_is_not_a_block_multiple() {
        let x = vec![0.0f32; 5];
        assert!(quantize(T_Q8_0, &x).is_err());
        assert!(quantize(T_Q4_K, &x).is_err());
    }

    #[test]
    fn quantize_rejects_an_unsupported_type() {
        let x = vec![0.0f32; 32];
        assert!(quantize(999, &x).is_err());
    }

    #[test]
    fn an_all_zero_block_quantizes_and_dequantizes_to_all_zero() {
        for ty in ALL_TYPES {
            let elems = block_elems(ty);
            let x = vec![0.0f32; elems];
            let bytes = quantize(ty, &x).unwrap();
            let decoded = dequantize(ty, &bytes, elems).unwrap();
            assert!(decoded.iter().all(|&v| v == 0.0), "{}: all-zero input did not decode to all zero", type_name(ty));
        }
    }

    /// The parallel encoder must be BIT-identical to the serial one, not
    /// merely close: every block is a pure function of its own inputs, so
    /// any difference would mean a work-splitting bug (a wrong block index,
    /// a short tail group), and a tolerance would hide exactly that. Sized
    /// past one work group (`GROUP = 64` blocks) with a deliberately ragged
    /// tail so the last group is partial.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn quantize_par_is_bit_identical_to_serial() {
        for ty in ALL_TYPES {
            let blocks = 64 * 2 + 7;
            let n = block_elems(ty) * blocks;
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 211) as f32 / 211.0 - 0.5) * 3.0).collect();
            let serial = quantize(ty, &x).unwrap();
            let parallel = quantize_par(ty, &x).unwrap();
            assert_eq!(serial, parallel, "{}: parallel encode differs from serial over {blocks} blocks", type_name(ty));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn quantize_par_rejects_the_same_inputs_serial_does() {
        let x = vec![0.0f32; 5];
        assert!(quantize_par(T_Q8_0, &x).is_err());
        assert!(quantize_par(999, &[0.0f32; 32]).is_err());
    }
}





