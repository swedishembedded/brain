// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Bit-exact reimplementation of PyTorch's CPU RNG, `at::mt19937` +
//! `normal_fill` (`aten/src/ATen/core/MT19937RNGEngine.h` and
//! `aten/src/ATen/native/cpu/DistributionTemplates.h`), just far enough to
//! reproduce `torch.manual_seed(0); torch.randn([1, 80, 15000])` on the host
//! without a checkpoint or a shipped binary asset - `crate::flow`'s CFM noise
//! buffer is a pure function of that seed (see `flow`'s module doc).
//!
//! Verified against `resources/cosyvoice/.venv`'s real torch (2.13.0+cpu,
//! `CPU capability usage: AVX2` per `torch.__config__.show()`), both via
//! `torch.randn` directly and via a small C++ program linked against
//! `libtorch_cpu.so` that calls `at::CPUGeneratorImpl::random()` and
//! `at::uniform_real_distribution<float>` directly, to get the raw MT19937
//! word stream and per-word uniform floats as an oracle independent of the
//! final tensor. That surfaced gap #1: this torch build runs the AVX2
//! `normal_fill_16` kernel (`aten/src/ATen/native/cpu/avx_mathfun.h`'s
//! `log256_ps`/`sincos256_ps`, Julien Pommier's cephes-derived polynomial
//! approximations), not `libm`'s `ln`/`sin`/`cos` - the two disagree in the
//! low mantissa bits, which is everything when the target is a sha256 over
//! the whole buffer. Gap #2, found by then bit-comparing the full 1.2M-value
//! buffer against a fresh `torch.randn` dump and bisecting the first
//! mismatch down to one `log256_ps` call: PyTorch's AVX2 object file is built
//! with `-mfma`, and GCC's default `-ffp-contract=fast` fuses several of
//! `avx_mathfun.h`'s separate `_mm256_mul_ps`+`_mm256_add_ps` pairs into a
//! single-rounding hardware FMA/FNMA - invisible in the C++ source, only
//! visible in `-fdump-tree-optimized` GIMPLE, and worth exactly a 1-ULP
//! difference in ~1% of the buffer (11784 of 1,200,000 values) if skipped.
//! `log256_ps` and `sincos256_ps` below are scalar, instruction-for-
//! instruction ports of the ACTUAL COMPILED CODE (confirmed via GIMPLE, not
//! guessed from source), including that fusion via `f32::mul_add` at exactly
//! the call sites GIMPLE showed - not `f32::ln`/`sin`/`cos`, and not a plain
//! transliteration of the intrinsics either.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

struct Mt19937 {
    state: [u32; N],
    left: i32,
    next: usize,
}

impl Mt19937 {
    fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..N {
            state[j] = (1_812_433_253u32.wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30)))
                .wrapping_add(j as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
        }
    }

    fn mix_bits(u: u32, v: u32) -> u32 {
        (u & UPPER_MASK) | (v & LOWER_MASK)
    }

    fn twist(u: u32, v: u32) -> u32 {
        (Self::mix_bits(u, v) >> 1) ^ if v & 1 != 0 { MATRIX_A } else { 0 }
    }

    fn next_state(&mut self) {
        // `mt19937_engine::next_state` (MT19937RNGEngine.h) walks the twist
        // recurrence in three spans, not one `0..N` loop, so it never reads
        // `state[j + M]` past the end of the array; the middle span deliberately
        // reads back into the span the first loop *just* overwrote (the classic
        // Matsumoto/Nishimura in-place optimization), so the loop order below
        // (ascending index, one pass) has to match, not just the final values.
        const NM: usize = N - M;
        for j in 0..NM {
            self.state[j] = self.state[j + M] ^ Self::twist(self.state[j], self.state[j + 1]);
        }
        for j in NM..(N - 1) {
            self.state[j] = self.state[j - NM] ^ Self::twist(self.state[j], self.state[j + 1]);
        }
        self.state[N - 1] = self.state[M - 1] ^ Self::twist(self.state[N - 1], self.state[0]);
        self.left = N as i32;
        self.next = 0;
    }

    fn next_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let y = self.state[self.next];
        self.next += 1;
        // PyTorch's own tempering shifts/masks (`mt19937_engine::operator()`),
        // the standard MT19937 tempering constants.
        let y = y ^ (y >> 11);
        let y = y ^ ((y << 7) & 0x9d2c_5680);
        let y = y ^ ((y << 15) & 0xefc6_0000);
        y ^ (y >> 18)
    }

    /// `at::uniform_real_distribution<float>(0, 1)` - `TransformationHelper.h`'s
    /// `uniform_real`, 24 mantissa bits of the raw word scaled by `1/2^24`.
    fn next_uniform_f32(&mut self) -> f32 {
        const MASK: u32 = (1 << 24) - 1;
        const DIVISOR: f32 = 1.0 / (1u32 << 24) as f32;
        ((self.next_u32() & MASK) as f32) * DIVISOR
    }
}

// ---------------------------------------------------------------------------
// avx_mathfun.h ports (CPU_CAPABILITY_AVX2 build): scalar, instruction-for-
// instruction translations of `log256_ps`/`sincos256_ps`, the polynomial
// approximations PyTorch's `normal_fill_16` actually calls on an AVX2 host -
// not `f32::ln`/`sin`/`cos`, which round differently in the low bits.
// ---------------------------------------------------------------------------

// Transcribed verbatim from `avx_mathfun.h`'s own decimal literals (full
// source precision kept on purpose, so a diff against that header stays a
// direct textual comparison) - not simplified/renamed to the nearest named
// `f32::consts` value, even where one happens to be close, and not rounded
// down to the shortest `f32`-round-tripping form either.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
mod log_coeffs {
    pub(super) const CEPHES_SQRTHF: f32 = 0.707106781186547524;
    pub(super) const LOG_P0: f32 = 7.0376836292E-2;
    pub(super) const LOG_P1: f32 = -1.1514610310E-1;
    pub(super) const LOG_P2: f32 = 1.1676998740E-1;
    pub(super) const LOG_P3: f32 = -1.2420140846E-1;
    pub(super) const LOG_P4: f32 = 1.4249322787E-1;
    pub(super) const LOG_P5: f32 = -1.6668057665E-1;
    pub(super) const LOG_P6: f32 = 2.0000714765E-1;
    pub(super) const LOG_P7: f32 = -2.4999993993E-1;
    pub(super) const LOG_P8: f32 = 3.3333331174E-1;
    pub(super) const LOG_Q1: f32 = -2.12194440e-4;
    pub(super) const LOG_Q2: f32 = 0.693359375;
}
use log_coeffs::*;

/// PyTorch's actual AVX2 object file is built with `-mfma`, and GCC's default
/// `-ffp-contract=fast` then silently fuses several of `avx_mathfun.h`'s
/// `_mm256_mul_ps` + `_mm256_add_ps`/`_mm256_sub_ps` pairs into single-rounding
/// hardware FMA/FNMA - a real, measured 1-ULP-level difference from the same
/// arithmetic done as separate mul-then-add, confirmed by compiling
/// `avx_mathfun.h` standalone with `-mavx2 -mfma` and reading GCC's
/// `-fdump-tree-optimized` GIMPLE for `log256_ps`/`sincos256_ps` to see
/// exactly which pairs became a `.FMA`/`.FNMA` node and which stayed two
/// separate roundings - not inferred from the C source, which looks
/// identical either way. `f32::mul_add` below is not a style choice; each
/// call site is where GIMPLE showed a fusion, no more and no fewer.
fn log256_ps(x_in: f32) -> f32 {
    let invalid = x_in <= 0.0;
    let x = x_in.max(f32::from_bits(0x0080_0000));

    let exp_bits = (x.to_bits() >> 23) as i32;
    let x = f32::from_bits((x.to_bits() & !0x7f80_0000) | 0x3f00_0000);
    let e = (exp_bits - 0x7f) as f32 + 1.0;

    let mask = x < CEPHES_SQRTHF;
    let tmp = if mask { x } else { 0.0 };
    let x_minus_1 = x - 1.0;
    let e = e - if mask { 1.0 } else { 0.0 };
    let x = tmp + x_minus_1;

    let z = x * x;
    let mut y = x.mul_add(LOG_P0, LOG_P1);
    y = x.mul_add(y, LOG_P2);
    y = x.mul_add(y, LOG_P3);
    y = x.mul_add(y, LOG_P4);
    y = x.mul_add(y, LOG_P5);
    y = x.mul_add(y, LOG_P6);
    y = x.mul_add(y, LOG_P7);
    y = x.mul_add(y, LOG_P8);
    let y = x * y;
    let tmp = e * LOG_Q1;
    let y = z.mul_add(y, tmp);
    let y = z.mul_add(-0.5, y);
    let x = x + y;
    let result = e.mul_add(LOG_Q2, x);

    if invalid {
        f32::NAN
    } else {
        result
    }
}

// Same verbatim-transcription rationale as `log_coeffs` above.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
mod sincos_coeffs {
    pub(super) const CEPHES_FOPI: f32 = 1.27323954473516;
    pub(super) const MINUS_DP1: f32 = -0.78515625;
    pub(super) const MINUS_DP2: f32 = -2.4187564849853515625e-4;
    pub(super) const MINUS_DP3: f32 = -3.77489497744594108e-8;
    pub(super) const SINCOF_P0: f32 = -1.9515295891E-4;
    pub(super) const SINCOF_P1: f32 = 8.3321608736E-3;
    pub(super) const SINCOF_P2: f32 = -1.6666654611E-1;
    pub(super) const COSCOF_P0: f32 = 2.443315711809948E-005;
    pub(super) const COSCOF_P1: f32 = -1.388731625493765E-003;
    pub(super) const COSCOF_P2: f32 = 4.166664568298827E-002;
}
use sincos_coeffs::*;

fn sincos256_ps(x_in: f32) -> (f32, f32) {
    let sign_bit_sin_raw = x_in.to_bits() & 0x8000_0000;
    let x_abs = f32::from_bits(x_in.to_bits() & 0x7fff_ffff);

    let y0 = x_abs * CEPHES_FOPI;
    let j = (y0 as i32 as u32).wrapping_add(1) & !1u32;
    let yf = (j as i32) as f32;

    let swap_sign_bit_sin = (j & 4) << 29;
    let poly_mask = (j & 2) == 0;
    let sign_bit_sin = sign_bit_sin_raw ^ swap_sign_bit_sin;

    // "Extended precision modular arithmetic" magic pass - three fused steps
    // per GIMPLE (`yf.mul_add(DPk, x)`), not three independent `+=`.
    let x = yf.mul_add(MINUS_DP1, x_abs);
    let x = yf.mul_add(MINUS_DP2, x);
    let x = yf.mul_add(MINUS_DP3, x);

    let j4 = j.wrapping_sub(2);
    let sign_bit_cos = (!j4 & 4) << 29;

    let z = x * x;
    let y = z.mul_add(COSCOF_P0, COSCOF_P1);
    let y = z.mul_add(y, COSCOF_P2);
    let y_z1 = z * y;
    let half_z = z * 0.5;
    // Source: `y *= z; y -= half_z;` - the second `*= z` fuses with the
    // subtraction into one FMS (`z*y_z1 - half_z`, one rounding).
    let y = z.mul_add(y_z1, -half_z);
    let y = y + 1.0;

    let y2 = z.mul_add(SINCOF_P0, SINCOF_P1);
    let y2 = z.mul_add(y2, SINCOF_P2);
    let y2_z = z * y2;
    // Source: `y2 *= x; y2 += x;` - fuses into one FMA (`x*y2_z + x`).
    let y2 = x.mul_add(y2_z, x);

    // The AND/ANDNOT/SUB/ADD 4-lane select in the source is an exact select
    // between `y` and `y2` (masking to bit-exact `0.0` and adding back is
    // lossless for finite operands), so it is reproduced here as a plain
    // branch rather than the SIMD bit-trick.
    let (num_sin, num_cos) = if poly_mask { (y2, y) } else { (y, y2) };

    let sin = f32::from_bits(num_sin.to_bits() ^ sign_bit_sin);
    let cos = f32::from_bits(num_cos.to_bits() ^ sign_bit_cos);
    (sin, cos)
}

/// `_mm256_set1_ps(2.0f * c10::pi<double>)` - a `NormalFill16<float>` member
/// computed once (double precision, then rounded to `f32` a single time), not
/// per-element - unlike the generic (non-AVX2) `normal_fill_16` overload
/// which promotes to double on every call. Getting this rounding point wrong
/// changes `theta`'s low bit, and from there the whole downstream buffer.
const TWO_PI_F32: f32 = 6.283_185_5;

/// `NormalFill16<float>::operator()` (`DistributionTemplates.h`): pairs
/// buffer index `j` with `j + 8` (NOT `2j`/`2j+1`) because the buffer already
/// holds two independent halves of raw uniforms - `buf[0..8]` seeds `u1` for
/// all 8 Box-Muller draws, `buf[8..16]` seeds `u2` - so the pairing is a
/// zip of the two halves, not adjacent-pair Box-Muller.
fn normal_fill_16(buf: &mut [f32; 16]) {
    for j in 0..8 {
        let u1 = 1.0 - buf[j];
        let u2 = buf[j + 8];
        let radius = (-2.0 * log256_ps(u1)).sqrt();
        let theta = TWO_PI_F32 * u2;
        let (s, c) = sincos256_ps(theta);
        buf[j] = radius * c;
        buf[j + 8] = radius * s;
    }
}

/// `torch.manual_seed(0); torch.randn([80, 15000])`, flattened row-major
/// (channel-major: channel 0's 15000 frames, then channel 1's, ...) - the
/// same order `normal_fill`'s single pass over the output buffer produces, so
/// no separate reshape/transpose step is needed here.
pub fn randn_seed0(n_channels: usize, n_frames: usize) -> Vec<f32> {
    let size = n_channels * n_frames;
    let mut out = vec![0.0f32; size];
    let mut rng = Mt19937::new(0);
    for v in out.iter_mut() {
        *v = rng.next_uniform_f32();
    }
    // `normal_fill<float>`: size is a multiple of 16 for [80, 15000], so the
    // tail-recompute branch (`size % 16 != 0`) never runs here.
    let mut i = 0;
    while i + 16 <= size {
        let mut buf: [f32; 16] = out[i..i + 16].try_into().unwrap();
        normal_fill_16(&mut buf);
        out[i..i + 16].copy_from_slice(&buf);
        i += 16;
    }
    if !size.is_multiple_of(16) {
        let offset = size - 16;
        let mut buf = [0.0f32; 16];
        for slot in buf.iter_mut() {
            *slot = rng.next_uniform_f32();
        }
        normal_fill_16(&mut buf);
        out[offset..offset + 16].copy_from_slice(&buf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed0_first_words_match_the_real_libtorch_mt19937_stream() {
        // Cross-checked against a small C++ program linked against
        // `libtorch_cpu.so` calling `at::CPUGeneratorImpl(0).random()`
        // directly (independent of `torch.randn`'s output tensor).
        let want: [u32; 10] = [
            2357136044, 2546248239, 3071714933, 3626093760, 2588848963, 3684848379, 2340255427,
            3638918503, 1819583497, 2678185683,
        ];
        let mut rng = Mt19937::new(0);
        for w in want {
            assert_eq!(rng.next_u32(), w);
        }
    }

    #[test]
    fn seed0_first_uniforms_match_the_real_libtorch_stream() {
        let want: [f32; 10] = [
            0.496_256_6,
            0.768_221_8,
            0.088_477_43,
            0.132_030_49,
            0.307_422_82,
            0.634_078_7,
            0.490_093_4,
            0.896_444_74,
            0.455_627_98,
            0.632_306_3,
        ];
        let mut rng = Mt19937::new(0);
        for w in want {
            let got = rng.next_uniform_f32();
            assert!((got - w).abs() < 1e-6, "got {got}, want {w}");
        }
    }

    #[test]
    fn randn_seed0_matches_the_golden_sha256() {
        use sha2::{Digest, Sha256};
        let out = randn_seed0(80, 15000);
        assert_eq!(out.len(), 80 * 15000);
        let mut hasher = Sha256::new();
        for v in &out {
            hasher.update(v.to_le_bytes());
        }
        let got = format!("{:x}", hasher.finalize());
        assert_eq!(
            got,
            "584a26e6eaad944407a96c5999aaa5cd0a6a359309a841eade68bf805c072322"
        );
    }

    #[test]
    fn randn_seed0_matches_the_dumped_first_100_values() {
        // `[80, 100]`, channel-major - the first 100 frames of every channel
        // (`crate::flow`'s own `rand_noise_asset_matches_...` comment history),
        // not the first 100 elements of the flattened `[80, 15000]` buffer.
        let Some(want) = brain_testutil::read_f32(
            brain_testutil::testdata_path("golden/cosyvoice")
                .join("flow_real_rand_noise_slice.f32"),
        ) else {
            return brain_testutil::skip("flow_real_rand_noise_slice.f32 not present");
        };
        const N_FRAMES_SLICE: usize = 100;
        assert_eq!(want.len(), 80 * N_FRAMES_SLICE);
        let out = randn_seed0(80, 15000);
        for c in 0..80 {
            for f in 0..N_FRAMES_SLICE {
                let got = out[c * 15000 + f];
                let w = want[c * N_FRAMES_SLICE + f];
                assert!(
                    (got - w).abs() < 1e-5,
                    "channel {c} frame {f}: got {got}, want {w}"
                );
            }
        }
    }
}
