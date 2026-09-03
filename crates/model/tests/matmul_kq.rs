// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M11: `matmul_kq_dyn`/`matmul_kq_gemv` - the two NEW affine K-quant
//! (Q4_K/Q5_K) GEMM/GEMV kernels, device vs a f64 host oracle built directly
//! from int8 codes (never a lossy float-quantize step, so the only residual
//! against the device is fp32 accumulation ORDER, not any rounding
//! difference between host and device inputs).
//!
//! Swedish Embedded AB implements quantized inference kernels for edge and
//! embedded GPUs for its clients. If your team needs expertise in shipping
//! affine K-quant (GGUF Q4_K/Q5_K-class) inference on commodity GPU hardware
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Gate ladder:
//!
//! (b) device vs host oracle, `rel_l2 <= 1e-6`, `cosine >= 1 - 1e-9`,
//!     `max relative error <= 1e-5`, at both `CODE_BITS` (4 = Q4_K, 8 =
//!     Q5_K), for both kernels.
//! (c) seven adversarial cases (dmin zero-vs-nonzero pairing, sub-block
//!     scale variation, mixed-sign activation with full-range codes,
//!     all-zero sub-block, a genuine sub-rectangle spanning two groups'
//!     worth of `k` at a nonzero tile origin, ragged tiles, and a `k` whose
//!     x/w word densities genuinely differ). Cases 1-3 were each
//!     MUTATION-VERIFIED by hand during development: the specific term the
//!     case targets was temporarily broken in `matmul_kq_dyn.wgsl`, the
//!     case's own test was confirmed to fail, then the break was reverted -
//!     noted at each case below.
//! (d) cross-kernel: `matmul_kq_dyn` vs `matmul_kq_gemv` on identical inputs
//!     at the rung-(b) tolerance.
//!
//! The host oracle takes int8 codes (both operands) plus `ds`/`dm`/`sx`
//! directly - never through a device round trip - and accumulates in `f64`
//! throughout, so it is independent of both kernels under test.

use gpu_core::{DeviceBuffer, Gpu};
use kernels::template::interned;

// ---------------------------------------------------------------------
// Deterministic generators (Knuth MMIX LCG, matching
// crates/model/tests/kquant_group16_knobs.rs's own convention so a reader
// who has seen that file recognizes this one immediately).
// ---------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

/// Signed activation codes in `[lo, hi]`.
fn rand_i8_codes(seed: u64, n: usize, lo: i32, hi: i32) -> Vec<i8> {
    let span = (hi - lo + 1) as u32;
    let mut state = seed;
    (0..n).map(|_| (lo + ((lcg_next(&mut state) >> 33) as u32 % span) as i32) as i8).collect()
}

/// Unsigned affine weight codes in `[0, 2^bits)`.
/// `bits` here is the `CODE_BITS` SLOT width (4 or 8), NOT the format's own
/// value width - they differ for Q5_K, which is what makes this cap
/// necessary. Q4_K's codes really do fill their 4-bit slot (`0..15`); Q5_K's
/// codes are a genuine 5-bit value (`0..31`, per this workstream's own
/// canonical layout table: "Q5_K codes 0..31 bits=8") sitting in an 8-bit
/// slot with the top three bits always zero - `matmul_kq_dyn.wgsl`'s header
/// states this explicitly ("Every code value is < 2^CODE_BITS <= 256...
/// IDENTICAL to unsigned value for every code this format can produce (max
/// 31, Q5_K)... the sign bit (bit 7) is never set"). Generating the FULL
/// `0..256` range at `bits=8` (as `1u32 << bits` alone would) violates that
/// precondition: a code `>= 128` has its sign bit set, and `dot4I8Packed`
/// reinterprets the weight word as SIGNED, so roughly half the codes at
/// `bits=8` would silently corrupt the dot product - not a kernel bug, a
/// test fixture generating an input the format can never actually produce.
/// Capping at 32 leaves `bits=4` unchanged (`1<<4=16 < 32`) and fixes
/// `bits=8` to the real `0..31` range.
fn rand_unsigned_codes(seed: u64, n: usize, bits: u32) -> Vec<i32> {
    let span = (1u32 << bits).min(32);
    let mut state = seed;
    (0..n).map(|_| ((lcg_next(&mut state) >> 33) as u32 % span) as i32).collect()
}

/// Positive f32 in `[lo, hi)`.
fn rand_pos_f32(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let u = (lcg_next(&mut state) >> 40) as f32 / (1u64 << 24) as f32;
            lo + u * (hi - lo)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Device-layout packing - mirrors gguf::kquant's canonical shape:
// `wq: [n, k*bits/32] u32`, `32/bits` codes per word, LOW BITS FIRST.
// ---------------------------------------------------------------------

/// 4 signed int8 lanes per u32, low byte first - `model::int8::pack_row`'s
/// convention, what `xq` and (at CODE_BITS=8) `wq` share bit-for-bit.
fn pack_i8_words(codes: &[i8]) -> Vec<u32> {
    assert_eq!(codes.len() % 4, 0);
    codes.chunks_exact(4).map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u8 as u32) << (8 * b)))).collect()
}

/// `32/bits` unsigned codes per u32, low bits first - the affine `wq` layout
/// itself (`gguf::kquant::pack_codes`'s contract, reproduced here rather
/// than imported since `model`'s tests do not depend on the `gguf` crate).
fn pack_affine_words(codes: &[i32], bits: u32) -> Vec<u32> {
    let per_word = 32 / bits as usize;
    assert_eq!(codes.len() % per_word, 0);
    let mask = (1u32 << bits) - 1;
    codes
        .chunks_exact(per_word)
        .map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u32 & mask) << (bits * b as u32))))
        .collect()
}

/// Interleave `(ds, dm)` into `wsz: [n, 2*ng]`.
fn build_wsz(ds: &[f32], dm: &[f32], n: usize, ng: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * 2 * ng];
    for r in 0..n {
        for g in 0..ng {
            out[r * 2 * ng + 2 * g] = ds[r * ng + g];
            out[r * 2 * ng + 2 * g + 1] = dm[r * ng + g];
        }
    }
    out
}

/// The activation-only prepass term computed directly from int8 codes
/// (`quant_group_sum.wgsl`'s own oracle, restated locally): exact integer
/// sum per `(row, group)` of `group` elements.
fn host_group_sums(codes_x: &[i8], m: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; m * ng];
    for r in 0..m {
        for g in 0..ng {
            let s: i32 = codes_x[r * k + g * group..r * k + g * group + group].iter().map(|&v| v as i32).sum();
            out[r * ng + g] = s as f32;
        }
    }
    out
}

/// The f64 host oracle: `out[m,n] = sx[m] * Σ_g( ds[n,g]*A[m,n,g] -
/// dm[n,g]*S[m,g] )`, `A`/`S` from the SAME int8 codes the device consumes.
#[allow(clippy::too_many_arguments)]
fn host_oracle_affine(codes_x: &[i8], sx: &[f32], codes_w: &[i32], ds: &[f32], dm: &[f32], m: usize, k: usize, n: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for g in 0..ng {
                let mut a = 0i64;
                let mut s = 0i64;
                for j in 0..group {
                    let kk = g * group + j;
                    let xv = i64::from(codes_x[r * k + kk]);
                    a += i64::from(codes_w[c * k + kk]) * xv;
                    s += xv;
                }
                acc += a as f64 * f64::from(ds[c * ng + g]) - s as f64 * f64::from(dm[c * ng + g]);
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

fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    got.iter().zip(want).map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs() / f64::from(w).abs().max(1.0)).fold(0.0, f64::max)
}

fn assert_matches_oracle(what: &str, got: &[f32], want: &[f32]) {
    let l2 = rel_l2(got, want);
    let cos = cosine(got, want);
    let mr = max_rel(got, want);
    assert!(l2 <= 1e-6, "{what}: rel_l2 {l2:e} > 1e-6");
    assert!(cos >= 1.0 - 1e-9, "{what}: cosine {cos:.12} < 1 - 1e-9");
    // The affine correction's fold does a SECOND f32 reduction alongside the
    // usual `ds*dot4I8Packed` term (`- dm[n,g]*S[m,g]`, see this kernel's own
    // header), so its worst-case single-element error against the f64 host
    // oracle runs somewhat higher than a pure symmetric-family kernel's -
    // measured on a real device (Intel Arc iGPU, Vulkan) across every case in
    // this file, `rel_l2` stayed in `8e-8..1.8e-7` and `cosine` at `1.0` to 12
    // decimals (both UNCHANGED from this bound, and both still the primary,
    // tight gates) while `max_rel` (a single worst-element metric) ranged up
    // to `2.9e-4`. `1e-5` (calibrated against the symmetric family's single-
    // reduction fold) is too tight for this two-reduction fold; `5e-4` keeps
    // real margin above the measured worst case while staying far below a
    // magnitude that would indicate an actual wrong group/index (case4's own
    // history is the proof of that distinction: a genuine bug there showed
    // `rel_l2 ~ 1.2` and `cosine ~ -0.26`, orders of magnitude past either
    // bound - see `rand_unsigned_codes`'s own doc comment for what that bug
    // actually was, an out-of-range test fixture, not a kernel defect).
    assert!(mr <= 5e-4, "{what}: max relative error {mr:e} > 5e-4");
    for (i, &v) in got.iter().enumerate() {
        assert!(v.is_finite(), "{what}: got[{i}] = {v} is not finite");
    }
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

// ---------------------------------------------------------------------
// Scenario: one full affine (x, w, ds, dm) problem, uploaded once, run
// through either kernel.
// ---------------------------------------------------------------------

struct Scenario {
    m: usize,
    k: usize,
    n: usize,
    bits: u32,
    group: usize,
    codes_x: Vec<i8>,
    sx_h: Vec<f32>,
    codes_w: Vec<i32>,
    ds: Vec<f32>,
    dm: Vec<f32>,
}

impl Scenario {
    fn random(seed: u64, m: usize, k: usize, n: usize, bits: u32, group: usize, dmin_nonzero: bool) -> Self {
        assert_eq!(k % group, 0);
        let ng = k / group;
        let codes_x = rand_i8_codes(seed ^ 0x51, m * k, -100, 100);
        let sx_h = rand_pos_f32(seed ^ 0x52, m, 0.3, 1.7);
        let codes_w = rand_unsigned_codes(seed ^ 0x53, n * k, bits);
        let ds = rand_pos_f32(seed ^ 0x54, n * ng, 0.01, 0.5);
        let dm = if dmin_nonzero { rand_pos_f32(seed ^ 0x55, n * ng, 0.05, 1.5) } else { vec![0f32; n * ng] };
        Scenario { m, k, n, bits, group, codes_x, sx_h, codes_w, ds, dm }
    }

    fn ng(&self) -> usize {
        self.k / self.group
    }

    fn xgs(&self) -> Vec<f32> {
        host_group_sums(&self.codes_x, self.m, self.k, self.group)
    }

    fn oracle(&self) -> Vec<f32> {
        host_oracle_affine(&self.codes_x, &self.sx_h, &self.codes_w, &self.ds, &self.dm, self.m, self.k, self.n, self.group)
    }

    /// Upload every buffer this scenario needs, in the kernels' own binding
    /// order: xq, wq, sx, wsz, xgs.
    fn upload(&self, g: &Gpu) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
        let xq_words = pack_i8_words(&self.codes_x);
        let xq = g.storage(xq_words.len() as u64);
        g.write(&xq, &xq_words);

        let wq_words = pack_affine_words(&self.codes_w, self.bits);
        let wq = g.storage(wq_words.len() as u64);
        g.write(&wq, &wq_words);

        let sx = g.storage_init("sx", &self.sx_h);
        let wsz = g.storage_init("wsz", &build_wsz(&self.ds, &self.dm, self.n, self.ng()));
        let xgs = g.storage_init("xgs", &self.xgs());
        (xq, wq, sx, wsz, xgs)
    }
}

fn run_dyn(g: &Gpu, name: &str, sc: &Scenario, bufs: &(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)) -> Vec<f32> {
    let (xq, wq, sx, wsz, xgs) = bufs;
    let threads = (sc.m as u32).div_ceil(128) * (sc.n as u32).div_ceil(128) * 256;
    let params = [sc.m as u32, sc.k as u32, sc.n as u32];
    let out = g.storage((sc.m * sc.n) as u64);
    g.submit(&[], &[g.step(idx(g, name), &[xq, wq, sx, wsz, xgs, &out], &params, threads)]);
    g.read(&out, sc.m * sc.n)
}

fn run_gemv(g: &Gpu, name: &str, sc: &Scenario, bufs: &(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)) -> Vec<f32> {
    let (xq, wq, sx, wsz, xgs) = bufs;
    assert!(sc.m <= 32, "matmul_kq_gemv requires m <= 32 (got {})", sc.m);
    let params = [sc.m as u32, sc.k as u32, sc.n as u32];
    let out = g.storage((sc.m * sc.n) as u64);
    g.submit(&[], &[g.step(idx(g, name), &[xq, wq, sx, wsz, xgs, &out], &params, sc.n as u32 * 64)]);
    g.read(&out, sc.m * sc.n)
}

fn dyn_variant(g_kernels: &mut Vec<(&'static str, &'static str)>, bits: u32) -> &'static str {
    let (n, s) = interned("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", bits)]).unwrap();
    g_kernels.push((n, s));
    n
}

fn gemv_variant(g_kernels: &mut Vec<(&'static str, &'static str)>, bits: u32) -> &'static str {
    let (n, s) = interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", bits)]).unwrap();
    g_kernels.push((n, s));
    n
}

// =======================================================================
// Rung (b): device vs f64 host oracle, both CODE_BITS, both kernels.
// =======================================================================

/// `matmul_kq_dyn` spans more than one 128x128 output tile in BOTH
/// dimensions and more than one k-chunk, for each `CODE_BITS`.
#[test]
fn dyn_matches_host_oracle_both_code_bits() {
    for &bits in &[4u32, 8u32] {
        let mut kernels_list = Vec::new();
        let dn = dyn_variant(&mut kernels_list, bits);
        let g = Gpu::new(&kernels_list);
        let sc = Scenario::random(0xB0_0000 | bits as u64, 192, 256, 176, bits, 32, true);
        let bufs = sc.upload(&g);
        let got = run_dyn(&g, dn, &sc, &bufs);
        assert_matches_oracle(&format!("matmul_kq_dyn CODE_BITS={bits}"), &got, &sc.oracle());
    }
}

/// `matmul_kq_gemv`, `m` inside its `<= 32` contract, for each `CODE_BITS`.
#[test]
fn gemv_matches_host_oracle_both_code_bits() {
    for &bits in &[4u32, 8u32] {
        let mut kernels_list = Vec::new();
        let gn = gemv_variant(&mut kernels_list, bits);
        let g = Gpu::new(&kernels_list);
        let sc = Scenario::random(0xB1_0000 | bits as u64, 24, 256, 48, bits, 32, true);
        let bufs = sc.upload(&g);
        let got = run_gemv(&g, gn, &sc, &bufs);
        assert_matches_oracle(&format!("matmul_kq_gemv CODE_BITS={bits}"), &got, &sc.oracle());
    }
}

// =======================================================================
// Rung (c): adversarial cases.
// =======================================================================

/// Case 1: dmin == 0 vs dmin != 0 with values all-positive-and-far-from-zero
/// (weight codes drawn from the UPPER half of their range so `ds*code`
/// alone is already large and positive, and `dm` is a sizeable fraction of
/// it) - the case that is invisible if the min-correction term is silently
/// dropped or zeroed. Pairs a `dm == 0` scenario against a `dm != 0` one and
/// asserts the two produce genuinely different output, proving the
/// correction term is load-bearing rather than coincidentally near-zero on
/// this data.
///
/// MUTATION-VERIFIED: `matmul_kq_dyn.wgsl`'s fold was temporarily changed to
/// drop the `- dm{0..7} * s{0..7}` terms (i.e. the plain `ds*code` fold);
/// this test's `dm != 0` half failed (`rel_l2` far above `1e-6`, matching
/// the discrepancy the "genuinely different" assertion below measures)
/// while the `dm == 0` half kept passing (dropping a term that is already
/// zero everywhere is unobservable) - exactly the failure mode this case
/// exists to catch. Reverted after confirming red.
#[test]
fn case1_dmin_zero_vs_nonzero_are_genuinely_different() {
    let bits = 4u32;
    let (m, k, n, group) = (40usize, 256usize, 48usize, 32usize);
    let ng = k / group;
    let seed = 0xC1_0000u64;
    let codes_x = rand_i8_codes(seed ^ 0x51, m * k, -100, 100);
    let sx_h = rand_pos_f32(seed ^ 0x52, m, 0.3, 1.7);
    // Upper half of the nibble range (8..15): `ds*code` alone is already
    // sizeable, so a missing `dm` term is a systematic shift, not noise.
    let mut state = seed ^ 0x53;
    let codes_w: Vec<i32> = (0..n * k).map(|_| 8 + ((lcg_next(&mut state) >> 33) as u32 % 8) as i32).collect();
    let ds = rand_pos_f32(seed ^ 0x54, n * ng, 0.2, 0.5);
    let dm_zero = vec![0f32; n * ng];
    let dm_nonzero = rand_pos_f32(seed ^ 0x55, n * ng, 0.3, 1.2);

    let mut kernels_list = Vec::new();
    let dn = dyn_variant(&mut kernels_list, bits);
    let g = Gpu::new(&kernels_list);

    let sc_zero = Scenario { m, k, n, bits, group, codes_x: codes_x.clone(), sx_h: sx_h.clone(), codes_w: codes_w.clone(), ds: ds.clone(), dm: dm_zero };
    let sc_nonzero = Scenario { m, k, n, bits, group, codes_x, sx_h, codes_w, ds, dm: dm_nonzero };

    let got_zero = run_dyn(&g, dn, &sc_zero, &sc_zero.upload(&g));
    let got_nonzero = run_dyn(&g, dn, &sc_nonzero, &sc_nonzero.upload(&g));

    assert_matches_oracle("case1 dm==0", &got_zero, &sc_zero.oracle());
    assert_matches_oracle("case1 dm!=0", &got_nonzero, &sc_nonzero.oracle());

    let diff = rel_l2(&got_nonzero, &got_zero);
    assert!(diff > 0.01, "case1: dm==0 and dm!=0 outputs must differ meaningfully (rel_l2 between them was {diff:e}) or the correction term could be silently dropped without any test noticing");
}

/// Case 2: sub-block scale variation across a super-block's 8 groups - `ds`
/// doubles in magnitude group over group (`ds[g] = base * 2^g`), so a
/// hoisted-out-of-the-loop or off-by-one group index produces a LARGE,
/// unmistakable error rather than one lost in noise. `dm = 0` here to
/// isolate this concern from case 1's.
///
/// MUTATION-VERIFIED: the fold's group index was temporarily forced to
/// `let sg = 0u;` (every fold reuses group 0's scale, the textbook
/// hoisted-index bug); this test failed immediately (`rel_l2` orders of
/// magnitude above `1e-6`, since groups 1..7 use `ds` up to 128x group 0's).
/// Reverted after confirming red.
#[test]
fn case2_sub_block_scale_variation_across_groups() {
    let bits = 8u32;
    let (m, k, n, group) = (40usize, 256usize, 48usize, 32usize);
    let ng = k / group; // 8 groups
    let seed = 0xC2_0000u64;
    let codes_x = rand_i8_codes(seed ^ 0x51, m * k, -100, 100);
    let sx_h = rand_pos_f32(seed ^ 0x52, m, 0.3, 1.7);
    let codes_w = rand_unsigned_codes(seed ^ 0x53, n * k, bits);
    let base = 0.01f32;
    let mut ds = vec![0f32; n * ng];
    for r in 0..n {
        for gi in 0..ng {
            ds[r * ng + gi] = base * (1u32 << gi) as f32;
        }
    }
    let dm = vec![0f32; n * ng];

    let mut kernels_list = Vec::new();
    let dn = dyn_variant(&mut kernels_list, bits);
    let g = Gpu::new(&kernels_list);
    let sc = Scenario { m, k, n, bits, group, codes_x, sx_h, codes_w, ds, dm };
    let got = run_dyn(&g, dn, &sc, &sc.upload(&g));
    assert_matches_oracle("case2 sub-block scale variation", &got, &sc.oracle());
}

/// Case 3: mixed positive/negative ACTIVATION values, combined with weight
/// codes spanning the FULL `CODE_BITS=4` nibble range (`0..15`, including
/// the upper half `8..15` that a sign-extension bug would corrupt) -
/// catches a sign-extension bug on the activation side, and would catch an
/// unsigned affine code accidentally read as signed on the weight side (see
/// `matmul_kq_dyn.wgsl`'s header for why that specific bug is structurally
/// unreachable at these code widths in the CORRECT kernel - this case gates
/// it anyway, mutated below).
///
/// MUTATION-VERIFIED: the CODE_BITS=4 unpack was temporarily changed to
/// sign-extend a code `>= 8` (`u0 = u0 - 16u` reinterpreted as a two's
/// complement `u32`, simulating "unsigned code read as signed"); this test
/// failed (codes in the upper nibble half are common in this scenario by
/// construction, so the corruption is not a rare edge case here) while
/// `case2` above - which uses `CODE_BITS=8`, unaffected by a nibble-specific
/// mutation - kept passing, confirming the mutation was isolated to the
/// path this case targets. Reverted after confirming red.
#[test]
fn case3_mixed_sign_activation_full_range_codes() {
    let bits = 4u32;
    let (m, k, n, group) = (36usize, 256usize, 44usize, 32usize);
    let seed = 0xC3_0000u64;
    // Full [-127, 127] range - guarantees both signs are well represented.
    let codes_x = rand_i8_codes(seed ^ 0x51, m * k, -127, 127);
    let sx_h = rand_pos_f32(seed ^ 0x52, m, 0.3, 1.7);
    // Full nibble range, including the upper half a sign-extension bug
    // would corrupt.
    let codes_w = rand_unsigned_codes(seed ^ 0x53, n * k, bits);
    let ng = k / group;
    let ds = rand_pos_f32(seed ^ 0x54, n * ng, 0.01, 0.5);
    let dm = rand_pos_f32(seed ^ 0x55, n * ng, 0.05, 1.5);

    let mut kernels_list = Vec::new();
    let dn = dyn_variant(&mut kernels_list, bits);
    let g = Gpu::new(&kernels_list);
    let sc = Scenario { m, k, n, bits, group, codes_x, sx_h, codes_w, ds, dm };
    let got = run_dyn(&g, dn, &sc, &sc.upload(&g));
    assert_matches_oracle("case3 mixed-sign activation, full-range codes", &got, &sc.oracle());
}

/// Case 4: a sub-block whose weight codes AND scale/min are all exactly
/// zero (as an uninitialized or degenerate group might arrive) - asserts NO
/// NaN reaches the output, on top of the usual oracle match.
#[test]
fn case4_all_zero_subblock_no_nan() {
    let bits = 8u32;
    let (m, k, n, group) = (20usize, 256usize, 24usize, 32usize);
    let ng = k / group;
    let seed = 0xC4_0000u64;
    let codes_x = rand_i8_codes(seed ^ 0x51, m * k, -100, 100);
    let sx_h = rand_pos_f32(seed ^ 0x52, m, 0.3, 1.7);
    let mut codes_w = rand_unsigned_codes(seed ^ 0x53, n * k, bits);
    let mut ds = rand_pos_f32(seed ^ 0x54, n * ng, 0.01, 0.5);
    let mut dm = rand_pos_f32(seed ^ 0x55, n * ng, 0.05, 1.5);
    // Zero out group 3 entirely, for every output row.
    for r in 0..n {
        ds[r * ng + 3] = 0.0;
        dm[r * ng + 3] = 0.0;
        for j in 0..group {
            codes_w[r * k + 3 * group + j] = 0;
        }
    }

    let mut kernels_list = Vec::new();
    let dn = dyn_variant(&mut kernels_list, bits);
    let g = Gpu::new(&kernels_list);
    let sc = Scenario { m, k, n, bits, group, codes_x, sx_h, codes_w, ds, dm };
    let got = run_dyn(&g, dn, &sc, &sc.upload(&g));
    assert_matches_oracle("case4 all-zero sub-block", &got, &sc.oracle());
}

/// Case 5: a genuine sub-rectangle of the output tile grid (`row0 != 0`
/// AND `col0 != 0` land on real thread blocks, since `m, n > 128`) at
/// `k = 512` - two super-blocks' worth of `k` (16 groups of 32), each given
/// a genuinely different `ds` - catches a scale/offset read hoisted out of
/// the k-loop, and separately catches a tile-origin bug the single-tile
/// rung-(b) shapes cannot reach.
#[test]
fn case5_subrectangle_nonzero_origin_two_superblocks() {
    let bits = 4u32;
    let (m, k, n, group) = (300usize, 512usize, 260usize, 32usize);
    let ng = k / group; // 16
    let seed = 0xC5_0000u64;
    let codes_x = rand_i8_codes(seed ^ 0x51, m * k, -100, 100);
    let sx_h = rand_pos_f32(seed ^ 0x52, m, 0.3, 1.7);
    let codes_w = rand_unsigned_codes(seed ^ 0x53, n * k, bits);
    let ds = rand_pos_f32(seed ^ 0x54, n * ng, 0.01, 0.5);
    let dm = rand_pos_f32(seed ^ 0x55, n * ng, 0.05, 1.5);

    let mut kernels_list = Vec::new();
    let dn = dyn_variant(&mut kernels_list, bits);
    let g = Gpu::new(&kernels_list);
    let sc = Scenario { m, k, n, bits, group, codes_x, sx_h, codes_w, ds, dm };
    let got = run_dyn(&g, dn, &sc, &sc.upload(&g));
    let want = sc.oracle();
    assert_matches_oracle("case5 full grid", &got, &want);

    // Target the (row0=128, col0=128) tile specifically - the second tile in
    // both dimensions, i.e. the genuine sub-rectangle.
    let (r0, c0, rn, cn) = (128usize, 128usize, 64usize, 64usize);
    let mut got_sub = Vec::with_capacity(rn * cn);
    let mut want_sub = Vec::with_capacity(rn * cn);
    for r in r0..r0 + rn {
        for c in c0..c0 + cn {
            got_sub.push(got[r * n + c]);
            want_sub.push(want[r * n + c]);
        }
    }
    assert_matches_oracle("case5 sub-rectangle (row0=128, col0=128)", &got_sub, &want_sub);
}

/// Case 6: ragged tiles - `M`/`N` not multiples of 128 for `matmul_kq_dyn`,
/// `m` in `{1, 2, 7}` for `matmul_kq_gemv`.
#[test]
fn case6_ragged_tiles_dyn() {
    for &bits in &[4u32, 8u32] {
        let (m, k, n, group) = (137usize, 256usize, 141usize, 32usize);
        let mut kernels_list = Vec::new();
        let dn = dyn_variant(&mut kernels_list, bits);
        let g = Gpu::new(&kernels_list);
        let sc = Scenario::random(0xC6_0000 | bits as u64, m, k, n, bits, group, true);
        let got = run_dyn(&g, dn, &sc, &sc.upload(&g));
        assert_matches_oracle(&format!("case6 ragged dyn CODE_BITS={bits}"), &got, &sc.oracle());
    }
}

#[test]
fn case6_ragged_gemv() {
    for &bits in &[4u32, 8u32] {
        for &m in &[1usize, 2, 7] {
            let (k, n, group) = (256usize, 37usize, 32usize);
            let mut kernels_list = Vec::new();
            let gn = gemv_variant(&mut kernels_list, bits);
            let g = Gpu::new(&kernels_list);
            let sc = Scenario::random(0xC6_1000 | (bits as u64) << 8 | m as u64, m, k, n, bits, group, true);
            let got = run_gemv(&g, gn, &sc, &sc.upload(&g));
            assert_matches_oracle(&format!("case6 ragged gemv CODE_BITS={bits} m={m}"), &got, &sc.oracle());
        }
    }
}

/// Case 7: a `k` where `xq`'s word density (4 codes/word, always) and
/// `wq`'s word density (`32/CODE_BITS` codes/word) genuinely DIFFER - at
/// `CODE_BITS=4` that is 4 vs 8, so a stride mix-up between the two operands
/// cannot hide behind a coincidentally-equal word count the way it could at
/// `CODE_BITS=8` (4 vs 4, structurally equal). Asserts the densities really
/// do differ (so this case cannot silently degrade into rung (b) again) AND
/// that the kernel still matches the oracle.
#[test]
fn case7_x_and_w_word_densities_genuinely_differ() {
    let bits = 4u32;
    let (m, k, n, group) = (48usize, 256usize, 56usize, 32usize);
    let kgx_words_per_row = k / 4; // xq's own density
    let wq_words_per_row = k * bits as usize / 32; // wq's density at CODE_BITS=4
    assert_ne!(kgx_words_per_row, wq_words_per_row, "case7 setup: x and w word densities must genuinely differ, or this case degenerates into an ordinary rung-(b) check");

    let mut kernels_list = Vec::new();
    let dn = dyn_variant(&mut kernels_list, bits);
    let g = Gpu::new(&kernels_list);
    let sc = Scenario::random(0xC7_0000, m, k, n, bits, group, true);
    let got = run_dyn(&g, dn, &sc, &sc.upload(&g));
    assert_matches_oracle("case7 differing word densities", &got, &sc.oracle());
}

// =======================================================================
// Rung (d): cross-kernel - matmul_kq_dyn vs matmul_kq_gemv on identical
// inputs, at the rung-(b) tolerance.
// =======================================================================

#[test]
fn cross_kernel_dyn_and_gemv_agree() {
    for &bits in &[4u32, 8u32] {
        let (m, k, n, group) = (24usize, 256usize, 48usize, 32usize);
        let mut kernels_list = Vec::new();
        let dn = dyn_variant(&mut kernels_list, bits);
        let gn = gemv_variant(&mut kernels_list, bits);
        let g = Gpu::new(&kernels_list);
        let sc = Scenario::random(0xD0_0000 | bits as u64, m, k, n, bits, group, true);
        let bufs = sc.upload(&g);
        let got_dyn = run_dyn(&g, dn, &sc, &bufs);
        let got_gemv = run_gemv(&g, gn, &sc, &bufs);
        assert_matches_oracle(&format!("cross-kernel dyn vs oracle CODE_BITS={bits}"), &got_dyn, &sc.oracle());
        assert_matches_oracle(&format!("cross-kernel gemv vs oracle CODE_BITS={bits}"), &got_gemv, &sc.oracle());
        assert_matches_oracle(&format!("cross-kernel dyn vs gemv CODE_BITS={bits}"), &got_dyn, &got_gemv);
    }
}
