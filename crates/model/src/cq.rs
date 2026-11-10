// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for
// its clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Rotated-codebook weight quantisation (`CQ2`, `CQ3`, `CQ4`) - the host half of `Dtype::Cq2`/`Cq3`/`Cq4`.
//!
//! A weight row is cut into groups of [`GROUP`] input lanes. Each group stores one `bits`-wide index per lane into a
//! shared Lloyd-Max codebook plus one fp16 norm; the group's weights are
//!
//! ```text
//! w_g = (codebook[idx_g] * norm_g) @ H        H = orthonormal natural-order Walsh matrix, symmetric
//! ```
//!
//! so `y = W x = sum_g norm_g * (codebook[idx_g] . (H x_g))`: the activation row is rotated once per group (a fast
//! Walsh transform, shared by every output row) and the weight is read as a lookup-table dot product. The rotation
//! spreads outliers over the whole group, which is what lets two bits per weight work.
//!
//! Layout (the layout of the `.cact` deployment container, so its bytes upload unchanged): `codes` holds `n` rows of
//! `kpad * bits / 32` words, indices packed LSB-first into a continuous bit stream per row; `norms` holds the fp16
//! norms `[n, kpad / GROUP]` flat, two to a word, low half first. `k` is padded up to whole groups (`kpad`); the
//! padding lanes quantise zeros and the rotation kernel pads the activation with zeros.
//!
//! Lower precision than the dense tiers: a `Cq` weight is only ever built on request (a container that stores one, or
//! [`Cq::quantize`]), never by promoting another tier.

/// Input lanes per quantisation group (one fp16 norm, one Walsh transform).
pub const GROUP: usize = 128;

/// Why a set of CQ parts was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CqError(pub String);

impl std::fmt::Display for CqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cq: {}", self.0)
    }
}

impl std::error::Error for CqError {}

/// One quantised `[n, k]` matrix in the device layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Cq {
    /// Index width: 2, 3 or 4.
    pub bits: u32,
    /// Output rows.
    pub n: usize,
    /// Logical input columns.
    pub k: usize,
    /// Packed indices, `n * kpad * bits / 32` words.
    pub codes: Vec<u32>,
    /// fp16 norms, two to a word, `ceil(n * groups / 2)` words.
    pub norms: Vec<u32>,
    /// The `2^bits` codebook levels.
    pub book: Vec<f32>,
}

/// `k` rounded up to whole groups.
pub const fn padded(k: usize) -> usize {
    k.div_ceil(GROUP) * GROUP
}

/// In-place unnormalised natural-order Walsh-Hadamard transform (`x.len()` a power of two).
pub fn walsh_natural(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for block in x.chunks_exact_mut(h * 2) {
            let (a, b) = block.split_at_mut(h);
            for (l, r) in a.iter_mut().zip(b) {
                let (sum, diff) = (*l + *r, *l - *r);
                *l = sum;
                *r = diff;
            }
        }
        h *= 2;
    }
}

/// `1 / sqrt(GROUP)`: the orthonormal scale of the group transform.
fn scale() -> f32 {
    1.0 / (GROUP as f32).sqrt()
}

/// fp16 bit patterns two to a word, low half first.
fn pack_halves(halves: &[u16]) -> Vec<u32> {
    halves.chunks(2).map(|c| u32::from(c[0]) | c.get(1).map_or(0, |&h| u32::from(h) << 16)).collect()
}

fn f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn f16_value(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

impl Cq {
    /// Columns after padding to whole groups.
    pub fn kpad(&self) -> usize {
        padded(self.k)
    }

    /// Groups per row.
    pub fn groups(&self) -> usize {
        self.kpad() / GROUP
    }

    /// Words of packed indices per row.
    pub fn row_words(&self) -> usize {
        self.kpad() * self.bits as usize / 32
    }

    /// Resident bytes of the device form (indices, norms and the codebook).
    pub fn bytes(&self) -> usize {
        4 * (self.codes.len() + self.norms.len() + self.book.len())
    }

    /// Build from the container's parts: `packed` the index rows as bytes, `norm_bytes` the fp16 norms little-endian.
    pub fn from_parts(n: usize, k: usize, bits: u32, packed: &[u8], norm_bytes: &[u8], book: &[f32]) -> Result<Cq, CqError> {
        if !(2..=4).contains(&bits) {
            return Err(CqError(format!("{bits}-bit indices are not supported (2, 3 or 4)")));
        }
        if book.len() != 1 << bits {
            return Err(CqError(format!("a {bits}-bit codebook has {} levels, found {}", 1 << bits, book.len())));
        }
        if n == 0 || k == 0 {
            return Err(CqError(format!("an empty {n}x{k} matrix")));
        }
        let kpad = padded(k);
        let want_packed = n * kpad * bits as usize / 8;
        let want_norms = n * (kpad / GROUP) * 2;
        if packed.len() != want_packed || norm_bytes.len() != want_norms {
            return Err(CqError(format!(
                "a {n}x{k} {bits}-bit matrix needs {want_packed} index bytes and {want_norms} norm bytes, found {} and {}",
                packed.len(),
                norm_bytes.len()
            )));
        }
        let words = |bytes: &[u8]| -> Vec<u32> { bytes.chunks(4).map(|c| c.iter().rev().fold(0u32, |w, &b| (w << 8) | u32::from(b))).collect() };
        Ok(Cq { bits, n, k, codes: words(packed), norms: words(norm_bytes), book: book.to_vec() })
    }

    /// Rows `first .. first + count` as a matrix of their own (the packed rows and norms are contiguous, so this
    /// copies words and decodes nothing).
    pub fn rows(&self, first: usize, count: usize) -> Cq {
        assert!(first + count <= self.n, "Cq::rows: rows {first}+{count} of {}", self.n);
        let (w, g) = (self.row_words(), self.groups());
        let norms = (first * g..(first + count) * g).map(|flat| self.norm_bits(flat)).collect::<Vec<u16>>();
        Cq { bits: self.bits, n: count, k: self.k, codes: self.codes[first * w..(first + count) * w].to_vec(), norms: pack_halves(&norms), book: self.book.clone() }
    }

    /// The matrices stacked row-wise (same width, index width and codebook), the packed form of concatenating the
    /// outputs of projections that share an input.
    pub fn concat_rows(parts: &[&Cq]) -> Result<Cq, CqError> {
        let first = parts.first().ok_or_else(|| CqError("nothing to concatenate".to_string()))?;
        if parts.iter().any(|p| p.bits != first.bits || p.k != first.k || p.book != first.book) {
            return Err(CqError("the parts differ in index width, input width or codebook".to_string()));
        }
        let norms: Vec<u16> = parts.iter().flat_map(|p| (0..p.n * p.groups()).map(|flat| p.norm_bits(flat))).collect();
        Ok(Cq { bits: first.bits, n: parts.iter().map(|p| p.n).sum(), k: first.k, codes: parts.iter().flat_map(|p| p.codes.iter().copied()).collect(), norms: pack_halves(&norms), book: first.book.clone() })
    }

    fn norm_bits(&self, flat: usize) -> u16 {
        (self.norms[flat / 2] >> ((flat % 2) * 16)) as u16
    }

    /// The index bytes in the container layout (`n * kpad * bits / 8`).
    pub fn packed_bytes(&self) -> Vec<u8> {
        self.codes.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The norm bytes in the container layout (`n * groups * 2`).
    pub fn norm_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = self.norms.iter().flat_map(|w| w.to_le_bytes()).collect();
        out.truncate(self.n * self.groups() * 2);
        out
    }

    /// The norm of group `g` of row `r`.
    pub fn norm(&self, r: usize, g: usize) -> f32 {
        let flat = r * self.groups() + g;
        f16_value((self.norms[flat / 2] >> ((flat % 2) * 16)) as u16)
    }

    /// The codebook index of lane `c` (within the padded row) of row `r`.
    pub fn index(&self, r: usize, c: usize) -> usize {
        let bit = c * self.bits as usize;
        let at = r * self.row_words() * 32 + bit;
        let (word, shift) = (at / 32, at % 32);
        let mut v = u64::from(self.codes[word]) >> shift;
        if shift + self.bits as usize > 32 {
            v |= u64::from(self.codes[word + 1]) << (32 - shift);
        }
        (v & ((1 << self.bits) - 1)) as usize
    }

    /// The matrix as the device computes with it: row-major `[n, k]` f32.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.n * self.k];
        let mut group = [0f32; GROUP];
        for (r, row) in out.chunks_exact_mut(self.k).enumerate() {
            for g in 0..self.groups() {
                let norm = self.norm(r, g);
                for (j, v) in group.iter_mut().enumerate() {
                    *v = self.book[self.index(r, g * GROUP + j)] * norm;
                }
                walsh_natural(&mut group);
                for (j, v) in group.iter().enumerate() {
                    if g * GROUP + j < self.k {
                        row[g * GROUP + j] = v * scale();
                    }
                }
            }
        }
        out
    }

    /// Quantise a row-major `[n, k]` matrix against `book` (ascending levels of a unit-variance source): rotate each
    /// group, store its L2 norm (rounded to fp16, as it is stored) and pick the nearest level for every lane.
    pub fn quantize(raw: &[f32], n: usize, k: usize, bits: u32, book: &[f32]) -> Result<Cq, CqError> {
        if raw.len() != n * k {
            return Err(CqError(format!("{} values for a {n}x{k} matrix", raw.len())));
        }
        let (kpad, groups) = (padded(k), padded(k) / GROUP);
        let mut packed = vec![0u8; n * kpad * bits as usize / 8];
        let mut norms = vec![0u8; n * groups * 2];
        let mut group = [0f32; GROUP];
        for r in 0..n {
            let row = &raw[r * k..(r + 1) * k];
            let mut stream = 0u64;
            let mut held = 0usize;
            let mut at = r * kpad * bits as usize / 8;
            for g in 0..groups {
                group.fill(0.0);
                let live = (k - g * GROUP).min(GROUP);
                group[..live].copy_from_slice(&row[g * GROUP..g * GROUP + live]);
                walsh_natural(&mut group);
                for v in &mut group {
                    *v *= scale();
                }
                // The stored norm is the group's L2 norm (the rotation preserves it) and the codebook holds levels on
                // the unit sphere (about 1/sqrt(GROUP) per lane), so a lane's target is its share of the norm.
                let l2 = group.iter().map(|v| v * v).sum::<f32>().sqrt();
                let stored = f16_bits(l2);
                norms[(r * groups + g) * 2..][..2].copy_from_slice(&stored.to_le_bytes());
                let norm = f16_value(stored);
                for &v in &group {
                    let target = if norm > 0.0 { v / norm } else { 0.0 };
                    let code = nearest(book, target) as u64;
                    stream |= code << held;
                    held += bits as usize;
                    while held >= 8 {
                        packed[at] = stream as u8;
                        at += 1;
                        stream >>= 8;
                        held -= 8;
                    }
                }
            }
            debug_assert_eq!(held, 0, "a row is a whole number of bytes");
        }
        Cq::from_parts(n, k, bits, &packed, &norms, book)
    }

    /// The rotated activation rows `[m, kpad]`: each group of `x` (`[m, k]`, zero padded) through the orthonormal
    /// Walsh transform. The device `cq_rotate` kernel computes the same.
    pub fn rotate(x: &[f32], m: usize, k: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * k, "Cq::rotate: {} values for {m} rows of {k}", x.len());
        let kpad = padded(k);
        let mut out = vec![0f32; m * kpad];
        for (src, dst) in x.chunks_exact(k).zip(out.chunks_exact_mut(kpad)) {
            dst[..k].copy_from_slice(src);
            for g in dst.as_chunks_mut::<GROUP>().0 {
                walsh_natural(g);
                g.iter_mut().for_each(|v| *v *= scale());
            }
        }
        out
    }

    /// `y[m, n] = x[m, k] W^T` evaluated the way the device kernels do (rotate once, then lookup-table dots with the
    /// norm applied per group), the host oracle of the stored-bytes products.
    pub fn matmul_rotated(&self, x: &[f32], m: usize) -> Vec<f32> {
        let xr = Cq::rotate(x, m, self.k);
        let kpad = self.kpad();
        let mut y = vec![0f32; m * self.n];
        for r in 0..self.n {
            for g in 0..self.groups() {
                let norm = self.norm(r, g);
                let levels: Vec<f32> = (0..GROUP).map(|j| self.book[self.index(r, g * GROUP + j)]).collect();
                for i in 0..m {
                    let dot: f32 = levels.iter().zip(&xr[i * kpad + g * GROUP..]).map(|(l, v)| l * v).sum();
                    y[i * self.n + r] += norm * dot;
                }
            }
        }
        y
    }
}

/// Index of the level of ascending `book` nearest to `v`.
fn nearest(book: &[f32], v: f32) -> usize {
    let mut best = 0;
    for (i, &level) in book.iter().enumerate() {
        if (level - v).abs() < (book[best] - v).abs() {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    /// The 2-bit Lloyd-Max levels of a unit Gaussian, scaled to the unit sphere of a 128-lane group.
    const BOOK2: [f32; 4] = [-0.133_5, -0.040_02, 0.040_02, 0.133_5];

    fn book(bits: u32) -> Vec<f32> {
        // Evenly spaced unit-sphere levels are enough to exercise every code path.
        let levels = 1usize << bits;
        (0..levels).map(|i| (i as f32 - (levels as f32 - 1.0) / 2.0) * 0.07).collect()
    }

    #[test]
    fn packing_round_trips_through_the_container_bytes() {
        for bits in [2u32, 3, 4] {
            let mut rng = Lcg::new(7 + u64::from(bits));
            let raw = rng.vec_scaled(5 * 200, 1.0);
            let cq = Cq::quantize(&raw, 5, 200, bits, &book(bits)).expect("quantises");
            let again = Cq::from_parts(5, 200, bits, &cq.packed_bytes(), &cq.norm_bytes(), &cq.book).expect("parts");
            assert_eq!(again.dequantize(), cq.dequantize(), "{bits}-bit parts round trip");
            assert_eq!(cq.packed_bytes().len(), 5 * 256 * bits as usize / 8);
        }
    }

    #[test]
    fn dequantising_approximates_the_source_matrix() {
        // Gaussian weights against the 2-bit Lloyd-Max levels: the relative error is the quantiser's distortion
        // (about 0.12 of the energy for 4 levels, so about 0.35 in norm), not a scale error.
        let mut rng = Lcg::new(21);
        let raw: Vec<f32> = (0..4 * 256).map(|_| (0..12).map(|_| rng.signed()).sum::<f32>() * 0.5).collect();
        let cq = Cq::quantize(&raw, 4, 256, 2, &BOOK2).expect("quantises");
        let (num, den) = cq.dequantize().iter().zip(&raw).fold((0f64, 0f64), |(n, d), (a, b)| (n + f64::from(a - b).powi(2), d + f64::from(*b).powi(2)));
        assert!((num / den).sqrt() < 0.45, "relative error {}", (num / den).sqrt());
    }

    #[test]
    fn quantising_a_decoded_matrix_again_is_exact() {
        let mut rng = Lcg::new(11);
        let raw = rng.vec_scaled(3 * 256, 1.0);
        let cq = Cq::quantize(&raw, 3, 256, 2, &BOOK2).expect("quantises");
        let again = Cq::quantize(&cq.dequantize(), 3, 256, 2, &BOOK2).expect("quantises");
        assert_eq!(again.packed_bytes(), cq.packed_bytes(), "the codes are a fixed point");
    }

    #[test]
    fn rotated_lookup_product_equals_the_dequantised_matmul() {
        for (bits, k) in [(2u32, 256usize), (3, 200), (4, 128)] {
            let (n, m) = (9, 4);
            let mut rng = Lcg::new(u64::from(bits) * 31);
            let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, &book(bits)).expect("quantises");
            let x = rng.vec_scaled(m * k, 1.0);
            let w = cq.dequantize();
            let got = cq.matmul_rotated(&x, m);
            for i in 0..m {
                for r in 0..n {
                    let want: f32 = (0..k).map(|c| x[i * k + c] * w[r * k + c]).sum();
                    assert!((got[i * n + r] - want).abs() <= 1e-4 * (1.0 + want.abs()), "{bits}-bit k={k} ({i},{r}): {} vs {want}", got[i * n + r]);
                }
            }
        }
    }

    #[test]
    fn row_slices_and_concatenation_are_exact() {
        let mut rng = Lcg::new(5);
        let a = Cq::quantize(&rng.vec_scaled(3 * 200, 1.0), 3, 200, 2, &BOOK2).expect("quantises");
        let b = Cq::quantize(&rng.vec_scaled(5 * 200, 1.0), 5, 200, 2, &BOOK2).expect("quantises");
        let both = Cq::concat_rows(&[&a, &b]).expect("same width");
        assert_eq!(both.dequantize(), [a.dequantize(), b.dequantize()].concat());
        assert_eq!(both.rows(3, 5).dequantize(), b.dequantize());
        assert_eq!(both.rows(1, 2).dequantize(), a.dequantize()[200..].to_vec());
        let other = Cq::quantize(&rng.vec_scaled(200, 1.0), 1, 200, 3, &book(3)).expect("quantises");
        assert!(Cq::concat_rows(&[&a, &other]).is_err(), "different index widths do not stack");
    }

    #[test]
    fn malformed_parts_are_refused() {
        assert!(Cq::from_parts(1, 128, 5, &[0; 80], &[0; 2], &book(4)).is_err(), "5-bit indices");
        assert!(Cq::from_parts(1, 128, 2, &[0; 31], &[0; 2], &BOOK2).is_err(), "short index bytes");
        assert!(Cq::from_parts(1, 128, 2, &[0; 32], &[0; 2], &book(3)).is_err(), "codebook of the wrong width");
    }
}
