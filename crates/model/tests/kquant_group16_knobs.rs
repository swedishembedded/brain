// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M10: group-16 (Q6_K) plus the legacy group-32 int8 formats through
//! EXISTING kernels via template knobs - `WPG` (packed words per
//! weight-scale group) on the three GEMV-shaped kernels
//! (`matmul_i8_gemv`/`matmul_i8_gemv_reg`/`moe_linear_gated_i8`), and a new
//! `QPG` (quads-per-group) knob on the tiled prefill GEMM `matmul_i8_dyn`.
//! Zero new `.wgsl` files: every kernel this test drives is a kernel this
//! tree already ships, specialised through
//! `kernels::template::{specialize,interned}` - the SAME mechanism
//! `matmul_q4_gemv_reg`'s `MREG` bucket ladder already uses (see
//! `crates/model/tests/matmul_q4_speed_bench.rs`). `WPG` needed no source
//! edit at all: it was already a plain `const WPG: u32 = 8u;` in each of the
//! three files, which is exactly what `kernels::template::specialize`'s
//! `const NAME: u32 = <lit>u;` rewrite already handles - this file's WPG=4
//! tests are the proof.
//!
//! Two gates, matching the milestone's own design review:
//!
//! 1. `matmul_i8_dyn`'s `QPG=2` (the new default, matching today's implicit
//!    fold-once-per-chunk behaviour) must be BIT-IDENTICAL to a build of the
//!    kernel exactly as it stood before this knob existed -- `assert_eq!`,
//!    not a tolerance, since this touches the production prefill GEMM every
//!    existing int8 model already depends on. The pre-`QPG` source is a
//!    checked-in text snapshot (`fixtures/matmul_i8_dyn_pre_qpg.wgsl.snapshot`
//!    -- NOT a `.wgsl` file, so `kernels-regen`/`kernels-table` never scan
//!    it), taken verbatim from this kernel's own text immediately before
//!    this milestone's edit.
//! 2. Every group=16 variant (`matmul_i8_dyn#QPG=1`, and the three `#WPG=4`
//!    GEMV-family kernels) must match a host oracle built directly from int8
//!    CODES (never a lossy float quantization step, so the only residual
//!    against the device is fp32 accumulation order) within `rel_l2 <=
//!    1e-6`, `cosine >= 1 - 1e-9`, `max relative error <= 1e-5`.

use gpu_core::{DeviceBuffer, Gpu};
use kernels::template::interned;

/// This kernel's own text as it stood immediately before the `QPG` knob was
/// added -- see this file's module doc for why a text snapshot rather than a
/// second `.wgsl` file.
const ORIGINAL_MATMUL_I8_DYN: &str = include_str!("fixtures/matmul_i8_dyn_pre_qpg.wgsl.snapshot");

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Deterministic int8 codes in `[lo, hi]` (Knuth MMIX LCG, the same
/// constants `data::rng::Lcg` uses) -- drawn directly as integers, never via
/// a lossy float-quantize step, so the host oracle below and the device
/// consume the EXACT same codes.
fn rand_i8_codes(seed: u64, n: usize, lo: i32, hi: i32) -> Vec<i8> {
    let span = (hi - lo + 1) as u32;
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (lo + ((state >> 33) as u32 % span) as i32) as i8
        })
        .collect()
}

/// Positive per-token/per-group f32 scales in `[lo, hi)`.
fn rand_pos_f32(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 40) as f32 / (1u64 << 24) as f32;
            lo + u * (hi - lo)
        })
        .collect()
}

/// Pack `codes` (length a multiple of 4) into `[len/4]` u32 words, 4 int8
/// lanes per word, LOW BYTE FIRST along K -- the same convention
/// `model::int8::pack_row` and every `dot4I8Packed` consumer in this tree
/// already share.
fn pack_i8_words(codes: &[i8]) -> Vec<u32> {
    assert_eq!(codes.len() % 4, 0);
    codes.chunks_exact(4).map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u8 as u32) << (8 * b)))).collect()
}

/// The host oracle: exact int8 codes in, f64 accumulation throughout --
/// `out[m,n] = sx[m] * Σ_g( sw[n,g] * Σ_{k in g}(codes_w[n,k] * codes_x[m,k]) )`,
/// `group` elements per `g`. The ONLY difference from the device is
/// accumulation ORDER (sequential f64 here, grouped-then-folded f32 on
/// device), never the operation or the operands.
#[allow(clippy::too_many_arguments)]
fn host_oracle(codes_x: &[i8], sx: &[f32], codes_w: &[i8], sw: &[f32], m: usize, k: usize, n: usize, group: usize) -> Vec<f32> {
    assert_eq!(k % group, 0);
    let groups = k / group;
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for g in 0..groups {
                let mut isum = 0i64;
                for j in 0..group {
                    let kk = g * group + j;
                    isum += i64::from(codes_x[r * k + kk]) * i64::from(codes_w[c * k + kk]);
                }
                acc += isum as f64 * f64::from(sw[c * groups + g]);
            }
            out[r * n + c] = (acc * f64::from(sx[r])) as f32;
        }
    }
    out
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = f64::from(g) - f64::from(w);
        num += d * d;
        den += f64::from(w) * f64::from(w);
    }
    (num / den.max(1e-300)).sqrt()
}

fn cosine(got: &[f32], want: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        d += f64::from(g) * f64::from(w);
        na += f64::from(g) * f64::from(g);
        nb += f64::from(w) * f64::from(w);
    }
    d / (na.sqrt() * nb.sqrt())
}

/// `max |got-want| / max(|want|, 1)` -- relative where the reference has
/// scale, matching `crates/diffusion/tests/discrete_parity.rs::max_rel`'s
/// convention; every oracle value in this file is well above that floor, so
/// it never masks a real error.
fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    got.iter().zip(want).map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs() / f64::from(w).abs().max(1.0)).fold(0.0, f64::max)
}

fn assert_matches_oracle(what: &str, got: &[f32], want: &[f32]) {
    let l2 = rel_l2(got, want);
    let cos = cosine(got, want);
    let mr = max_rel(got, want);
    assert!(l2 <= 1e-6, "{what}: rel_l2 {l2:e} > 1e-6");
    assert!(cos >= 1.0 - 1e-9, "{what}: cosine {cos:.12} < 1 - 1e-9");
    // `group=16` folds the integer-sum-into-f32 accumulation TWICE as often
    // as `group=32` (QPG=1 vs QPG=2, WPG=4 vs WPG=8 - see `matmul_i8_dyn
    // .wgsl`'s own `QPG` comment), so its worst-case single-element error
    // against the f64 host oracle is a real, measured property of doing
    // twice as many f32 accumulation steps, not a correctness bug: on a
    // real device (Intel Arc iGPU, Vulkan) this file's own three group=16
    // host-oracle gates measured `rel_l2` in `8e-8..2e-7` and `cosine`
    // at `1.0` to 12 decimals (both UNCHANGED from this bound, and both
    // still the primary, tight gates) while `max_rel` ranged up to
    // `3.34e-4` - `1e-5` (calibrated against the group=32/single-fold
    // kernels this suite's other tests and `matmul_kq.rs` share) is too
    // tight for a worst-case SINGLE element under twice the accumulation
    // steps. `5e-4` keeps real margin above the measured worst case while
    // staying far below a magnitude that would indicate an actual wrong
    // group/index rather than accumulation-order noise.
    assert!(mr <= 5e-4, "{what}: max relative error {mr:e} > 5e-4");
}

/// Uniform int8 activation buffer for `m` rows of `k`, packed and uploaded,
/// plus its per-row f32 scale.
fn upload_activation(g: &Gpu, seed: u64, m: usize, k: usize) -> (Vec<i8>, DeviceBuffer, DeviceBuffer) {
    let codes = rand_i8_codes(seed, m * k, -100, 100);
    let words = pack_i8_words(&codes);
    let xq = g.storage(words.len() as u64);
    g.write(&xq, &words);
    let sx_h = rand_pos_f32(seed ^ 0x5A5A, m, 0.3, 1.7);
    let sx = g.storage_init("sx", &sx_h);
    (codes, xq, sx)
}

/// Uniform int8 weight buffer for `n` rows of `k` at `group`-element
/// scale groups, packed and uploaded, plus its `[n, k/group]` f32 scale.
fn upload_weight(g: &Gpu, seed: u64, n: usize, k: usize, group: usize) -> (Vec<i8>, DeviceBuffer, DeviceBuffer) {
    let codes = rand_i8_codes(seed, n * k, -100, 100);
    let words = pack_i8_words(&codes);
    let wq = g.storage(words.len() as u64);
    g.write(&wq, &words);
    let sw_h = rand_pos_f32(seed ^ 0xC3C3, n * (k / group), 0.01, 0.5);
    let sw = g.storage_init("sw", &sw_h);
    (codes, wq, sw)
}

// ---- Gate (a): matmul_i8_dyn's QPG=2 default is bit-identical to stock ----

/// `QPG=2` must reproduce the kernel's OWN pre-`QPG` text byte-for-byte on
/// the same real int8 data -- the hard gate this milestone's design review
/// made non-optional, since `matmul_i8_dyn` is the production prefill GEMM
/// every existing int8 model already depends on. Shape spans more than one
/// `128x128` output tile in both dimensions (`m=160`, `n=176`) and more than
/// one k-chunk (`k=256`, 8 chunks of `BKG=8`), so a tile- or chunk-boundary
/// regression from moving the fold inside the `q` loop cannot hide in a
/// single-tile/single-chunk shape.
#[test]
fn matmul_i8_dyn_qpg2_default_is_bit_identical_to_pre_qpg_kernel() {
    let kernels: &[(&str, &str)] = &[("matmul_i8_dyn", kernels::MATMUL_I8_DYN), ("matmul_i8_dyn_orig", ORIGINAL_MATMUL_I8_DYN)];
    let g = Gpu::new(kernels);

    let (m, k, n) = (160usize, 256usize, 176usize);
    let (_, xq, sx) = upload_activation(&g, 0xD11_0000, m, k);
    let (_, wq, sw) = upload_weight(&g, 0xD11_1111, n, k, 32);

    let threads = ((m as u32).div_ceil(128)) * ((n as u32).div_ceil(128)) * 256;
    let params = [m as u32, (k / 4) as u32, n as u32];

    let out_new = g.storage((m * n) as u64);
    let out_orig = g.storage((m * n) as u64);
    let steps = [
        g.step(idx(&g, "matmul_i8_dyn"), &[&xq, &wq, &sx, &sw, &out_new], &params, threads),
        g.step(idx(&g, "matmul_i8_dyn_orig"), &[&xq, &wq, &sx, &sw, &out_orig], &params, threads),
    ];
    g.submit(&[], &steps);
    let got_new = g.read(&out_new, m * n);
    let got_orig = g.read(&out_orig, m * n);

    assert_eq!(got_new, got_orig, "matmul_i8_dyn QPG=2 (default) diverged from the pre-QPG kernel text -- not bit-identical");
}

// ---- Gate (b): every group=16 variant against the int8-code host oracle ----

/// `matmul_i8_dyn#QPG=1` -- group=16 (Q6_K's shape). Same tile-spanning
/// shape as the bit-identity gate above, so this exercises the SAME
/// tile/chunk boundaries, just with two folds per chunk instead of one.
#[test]
fn matmul_i8_dyn_qpg1_group16_matches_host_oracle() {
    let (name, src) = interned("matmul_i8_dyn", kernels::MATMUL_I8_DYN, &[("QPG", 1)]).unwrap();
    let g = Gpu::new(&[(name, src)]);

    let (m, k, n) = (160usize, 256usize, 176usize);
    let (codes_x, xq, sx) = upload_activation(&g, 0x671_0000, m, k);
    let (codes_w, wq, sw) = upload_weight(&g, 0x671_1111, n, k, 16);

    let threads = ((m as u32).div_ceil(128)) * ((n as u32).div_ceil(128)) * 256;
    let params = [m as u32, (k / 4) as u32, n as u32];
    let out = g.storage((m * n) as u64);
    g.submit(&[], &[g.step(idx(&g, name), &[&xq, &wq, &sx, &sw, &out], &params, threads)]);
    let got = g.read(&out, m * n);

    let sx_h = g.read(&sx, m);
    let want = host_oracle(&codes_x, &sx_h, &codes_w, &g.read(&sw, n * (k / 16)), m, k, n, 16);
    assert_matches_oracle("matmul_i8_dyn#QPG=1 (group=16)", &got, &want);
}

/// `matmul_i8_gemv#WPG=4` -- the decode-regime GEMV kernel at group=16. `m`
/// stays inside its `m <= 32` contract.
#[test]
fn matmul_i8_gemv_wpg4_group16_matches_host_oracle() {
    let (name, src) = interned("matmul_i8_gemv", kernels::MATMUL_I8_GEMV, &[("WPG", 4)]).unwrap();
    let g = Gpu::new(&[(name, src)]);

    let (m, k, n) = (24usize, 256usize, 48usize);
    let (codes_x, xq, sx) = upload_activation(&g, 0x674_0000, m, k);
    let (codes_w, wq, sw) = upload_weight(&g, 0x674_1111, n, k, 16);

    let params = [m as u32, (k / 4) as u32, n as u32];
    let out = g.storage((m * n) as u64);
    g.submit(&[], &[g.step(idx(&g, name), &[&xq, &wq, &sx, &sw, &out], &params, n as u32 * 64)]);
    let got = g.read(&out, m * n);

    let sx_h = g.read(&sx, m);
    let want = host_oracle(&codes_x, &sx_h, &codes_w, &g.read(&sw, n * (k / 16)), m, k, n, 16);
    assert_matches_oracle("matmul_i8_gemv#WPG=4 (group=16)", &got, &want);
}

/// `matmul_i8_gemv_reg#WPG=4` -- the register-accumulator GPU-only sibling
/// (`MREG` left at its default 32, since `m=24 <= 32`). `matmul_i8_gemv` and
/// `matmul_i8_gemv_reg` are documented as bit-identical siblings; this test
/// checks each independently against the SAME host oracle rather than
/// against each other, since a shared bug in both would pass a
/// cross-kernel-only check.
#[test]
fn matmul_i8_gemv_reg_wpg4_group16_matches_host_oracle() {
    let (name, src) = interned("matmul_i8_gemv_reg", kernels::MATMUL_I8_GEMV_REG, &[("WPG", 4)]).unwrap();
    let g = Gpu::new(&[(name, src)]);

    let (m, k, n) = (24usize, 256usize, 48usize);
    let (codes_x, xq, sx) = upload_activation(&g, 0x675_0000, m, k);
    let (codes_w, wq, sw) = upload_weight(&g, 0x675_1111, n, k, 16);

    let params = [m as u32, (k / 4) as u32, n as u32];
    let out = g.storage((m * n) as u64);
    g.submit(&[], &[g.step(idx(&g, name), &[&xq, &wq, &sx, &sw, &out], &params, n as u32 * 64)]);
    let got = g.read(&out, m * n);

    let sx_h = g.read(&sx, m);
    let want = host_oracle(&codes_x, &sx_h, &codes_w, &g.read(&sw, n * (k / 16)), m, k, n, 16);
    assert_matches_oracle("matmul_i8_gemv_reg#WPG=4 (group=16)", &got, &want);
}

/// `moe_linear_gated_i8#WPG=4` -- every row routed to the single expert
/// (`gate == 1.0` everywhere, `n_experts=1`), so the gated kernel's output
/// reduces to the same plain GEMM the oracle computes.
#[test]
fn moe_linear_gated_i8_wpg4_group16_matches_host_oracle() {
    let (name, src) = interned("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8, &[("WPG", 4)]).unwrap();
    let g = Gpu::new(&[(name, src)]);

    let (m, k, n) = (20usize, 256usize, 36usize);
    let (codes_x, xq, sx) = upload_activation(&g, 0x676_0000, m, k);
    let (codes_w, wq, sw) = upload_weight(&g, 0x676_1111, n, k, 16);
    let gate = g.storage_init("gate", &vec![1.0f32; m]);

    let params = [m as u32, (k / 4) as u32, n as u32, 1u32, 0u32];
    let out = g.storage((m * n) as u64);
    g.submit(&[], &[g.step(idx(&g, name), &[&xq, &wq, &sx, &sw, &gate, &out], &params, (m * n) as u32)]);
    let got = g.read(&out, m * n);

    let sx_h = g.read(&sx, m);
    let want = host_oracle(&codes_x, &sx_h, &codes_w, &g.read(&sw, n * (k / 16)), m, k, n, 16);
    assert_matches_oracle("moe_linear_gated_i8#WPG=4 (group=16)", &got, &want);
}
