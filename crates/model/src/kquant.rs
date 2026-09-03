// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The canonical brain K-quant DEVICE layout - ONE shape, three
//! instantiations - stated once here so every device-side consumer (the new
//! affine `matmul_kq_dyn`/`matmul_kq_gemv` kernels, the existing-kernel reuse
//! via a `QPG` template knob, and the dtype-selection seam) agrees on it
//! without re-deriving it. `gguf::kquant::KqLayout` states the identical
//! shape from the HOST relayout side of the crate boundary - `model` does not
//! depend on the `gguf` crate, so this is a deliberate restatement for the
//! device-dispatch call sites that live in (or under) this crate, not a
//! re-export.
//!
//! Swedish Embedded AB implements quantized inference kernels for edge and
//! embedded GPUs for its clients. If your team needs expertise in shipping
//! K-quant (GGUF Q4_K/Q5_K/Q6_K-class) inference on commodity GPU hardware
//! without an intermediate fp32 detour then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! ## The one shape
//!
//! `wq: [n, k*bits/32] u32` - codes, K-contiguous, `32/bits` codes packed per
//! word, low bits first (code `b` of word `w` covers element `w*(32/bits)+b`
//! and occupies bits `[bits*b, bits*b+bits)`). Unsigned raw value for the
//! affine family; signed, bias-folded, low-bits two's complement for the
//! symmetric family.
//!
//! `wsz: [n, 2*k/G] f32` - interleaved `(scale, min)` pairs, one pair per
//! `G`-element group of the reduction axis `k`. `min == 0.0` whenever the
//! type is symmetric (`affine == false`).
//!
//! ## The three instantiations
//!
//! | GGUF type | bits | G  | affine | reaches the device via                                    |
//! |-----------|------|----|--------|------------------------------------------------------------|
//! | Q4_K      | 4    | 32 | yes    | new `matmul_kq_dyn`/`matmul_kq_gemv`                        |
//! | Q5_K      | 8    | 32 | yes    | new `matmul_kq_dyn`/`matmul_kq_gemv`                        |
//! | Q6_K      | 8    | 16 | no     | EXISTING `matmul_i8_*`, a new `QPG` (quads-per-group) knob  |
//! | Q5_0      | 8    | 32 | no     | EXISTING `matmul_i8_*`, unchanged                           |
//! | Q4_0      | 4    | 32 | no     | EXISTING `matmul_i8_*`, unchanged                           |
//! | Q8_0      | 8    | 32 | n/a    | `gguf::int8_direct::try_i8_rect` - a DIFFERENT, already-solved layout, unrelated to this one |
//!
//! ## The affine correction this milestone's prepass feeds
//!
//! For an affine type (Q4_K/Q5_K), an affine weight group reconstructs as
//! `ds*code - dm` rather than the symmetric family's `ds*code`, so the GEMM's
//! per-output value carries a second reduction alongside the usual int8 dot
//! product:
//!
//! `out[m,n] = sx[m] * Σ_g( ds[n,g]*A[m,n,g] - dm[n,g]*S[m,g] )`
//!
//! - `A[m,n,g] = Σ_{k in g}( q[n,k]*xq[m,k] )` is the existing
//!   `dot4I8Packed`-shaped integer dot product every int8 GEMM already
//!   computes in its inner loop.
//! - `S[m,g] = Σ_{k in g}( xq[m,k] )` is activation-only - independent of
//!   `n` - so recomputing it per output column (inside the GEMM's k-loop)
//!   would redo the same sum `N` times. `quant_group_sum.wgsl` computes it
//!   ONCE per activation instead, wired through [`crate::int8::QuantRows`]'s
//!   `xgs` seam as an optional THIRD step alongside the existing
//!   `max_abs_row`/`quant_pack` pair. `sx` factors out of both terms, so
//!   nothing about the existing per-token epilogue changes.
//!
//! `S` is computed from the int8 activation `xq`, never from the f32
//! activation directly - mixing them would be a systematic bias proportional
//! to `dm`, not a rounding difference, because the correction has to match
//! exactly what the GEMM's `A` term consumes.

/// The device K-quant layout parameters one weight tensor was packed with -
/// the same `(bits, group, affine)` triple `gguf::kquant::KqLayout` (a
/// different crate) computes from the GGUF type, restated here so a device
/// dispatch site can size buffers and dispatch geometry without this crate
/// depending on the `gguf` crate at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KqDeviceLayout {
    /// Output rows (`n`).
    pub n: usize,
    /// Reduction-axis length (`k`), in elements.
    pub k: usize,
    /// Bits per code in `wq` (4 or 8 - Q5_K's 5-bit code sits in an 8-bit
    /// slot, so this is the SLOT width, not necessarily the format's own bit
    /// width).
    pub bits: u32,
    /// Elements per weight-scale group along `k` (32, except Q6_K's 16).
    pub group: usize,
    /// Whether reconstruction needs `ds*code - dm` (true) or just `ds*code`
    /// (false, `dm` is always `0.0`).
    pub affine: bool,
}

impl KqDeviceLayout {
    /// `wq` words per output row: `k*bits/32`.
    pub fn words_per_row(&self) -> usize {
        self.k * self.bits as usize / 32
    }
    /// `wsz` groups per output row: `k/group`. `wsz` itself holds `2*` this
    /// many f32s per row (interleaved scale, min).
    pub fn groups_per_row(&self) -> usize {
        self.k / self.group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q4_K/Q5_K's shared geometry: 4-bit or 8-bit codes, 32-element groups,
    /// affine. `words_per_row`/`groups_per_row` must land on the sizes a
    /// device buffer allocation actually needs.
    #[test]
    fn affine_32_group_shape() {
        let q4k = KqDeviceLayout { n: 5, k: 256, bits: 4, group: 32, affine: true };
        assert_eq!(q4k.words_per_row(), 32, "256*4/32");
        assert_eq!(q4k.groups_per_row(), 8, "256/32");

        let q5k = KqDeviceLayout { n: 5, k: 256, bits: 8, group: 32, affine: true };
        assert_eq!(q5k.words_per_row(), 64, "256*8/32");
        assert_eq!(q5k.groups_per_row(), 8, "256/32");
    }

    /// Q6_K's odd one out: an 8-bit code like Q5_K/Q8_0, but a 16-element
    /// group - the property that rules out reusing the `BKG=8` invariant
    /// (32 int8 = one group) every OTHER int8 kernel in this tree assumes.
    #[test]
    fn q6_k_group16_shape_does_not_match_bkg8() {
        let q6k = KqDeviceLayout { n: 5, k: 256, bits: 8, group: 16, affine: false };
        assert_eq!(q6k.words_per_row(), 64, "256*8/32");
        assert_eq!(q6k.groups_per_row(), 16, "256/16, NOT 256/32");
        assert_ne!(q6k.groups_per_row(), q6k.words_per_row() / 8, "one group is 4 words here, not 8");
    }
}
