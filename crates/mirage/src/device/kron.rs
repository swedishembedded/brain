// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kronecker-factored row transforms on the device: the sequency Walsh transform of the HConvGLU mixers and the
//! three rotations of the Hadamard MLP, both run by `mirage_kron_rows` on the original parameters instead of an
//! `n x n` matrix per layer.
//!
//! Swedish Embedded AB implements solutions for running structured-matrix models without paying for their dense
//! expansion for its clients. If your team needs expertise in memory-lean accelerator inference then you can
//! procure our services by sending an email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Step};

use super::Device;

/// One resident `out[r, o] (+)= post[o] * K[r, perm[o]]`, `K = a^T (pre * x) b` over rows viewed as `[fa, fb]`.
#[derive(Clone)]
pub struct Kron {
    fa: u32,
    fb: u32,
    b: DeviceBuffer,
    a: DeviceBuffer,
    pre: DeviceBuffer,
    perm: DeviceBuffer,
    post: DeviceBuffer,
}

/// The parameters of a [`Kron`], host side. `None` means the identity (scale of one, identity permutation).
pub struct KronParams<'a> {
    /// `[fa, fa]` factor applied along the row index.
    pub a: &'a [f32],
    /// `[fb, fb]` factor applied along the column index.
    pub b: &'a [f32],
    /// Scale of the input, `[n]`.
    pub pre: Option<&'a [f32]>,
    /// Gather permutation of the output, `[n]`.
    pub perm: Option<&'a [u32]>,
    /// Scale of the output, `[n]`.
    pub post: Option<&'a [f32]>,
}

/// Largest transform one workgroup holds (the kernel's workgroup array).
pub const MAX_WIDTH: usize = 1024;

impl Kron {
    /// Upload `p` for factors `fa x fb`.
    pub fn upload(dev: &Device, (fa, fb): (usize, usize), p: &KronParams) -> Kron {
        let n = fa * fb;
        assert!(n <= MAX_WIDTH, "a Kronecker transform of width {n} exceeds the {MAX_WIDTH} one workgroup holds");
        assert!(p.a.len() == fa * fa && p.b.len() == fb * fb, "factors are {fa}x{fa} and {fb}x{fb}");
        let ones = vec![1.0f32; n];
        let perm: Vec<f32> = match p.perm {
            Some(perm) => perm.iter().map(|&v| v as f32).collect(),
            None => (0..n).map(|v| v as f32).collect(),
        };
        // Permutation entries ride in f32 words; they are exact below 2^24.
        assert!(perm.len() == n && perm.iter().all(|&v| (v as usize) < n), "a permutation of 0..{n}");
        let up = |name: &str, v: &[f32]| dev.param_buffer(name, v);
        Kron {
            fa: fa as u32,
            fb: fb as u32,
            b: up("kron_b", p.b),
            a: up("kron_a", p.a),
            pre: up("kron_pre", p.pre.unwrap_or(&ones)),
            perm: up("kron_perm", &perm),
            post: up("kron_post", p.post.unwrap_or(&ones)),
        }
    }

    /// The normalised sequency Walsh transform of rows of width `n` (a power of two).
    pub fn walsh(dev: &Device, n: usize) -> Kron {
        assert!(n.is_power_of_two() && n >= 2, "the Walsh transform needs a power-of-two width, not {n}");
        let bits = n.trailing_zeros();
        let (fa, fb) = (1usize << (bits / 2), 1usize << (bits - bits / 2));
        let sylvester = |m: usize| -> Vec<f32> { (0..m * m).map(|e| if ((e / m) & (e % m)).count_ones().is_multiple_of(2) { 1.0 } else { -1.0 }).collect() };
        // Row k of the sequency matrix is natural row bitreverse(gray(k)).
        let perm: Vec<u32> = (0..n as u32).map(|k| (k ^ (k >> 1)).reverse_bits() >> (32 - bits)).collect();
        let post = vec![1.0 / (n as f32).sqrt(); n];
        Kron::upload(dev, (fa, fb), &KronParams { a: &sylvester(fa), b: &sylvester(fb), pre: None, perm: Some(&perm), post: Some(&post) })
    }

    /// Bytes resident.
    pub fn bytes(&self) -> usize {
        4 * (self.fa * self.fa + self.fb * self.fb + 3 * self.fa * self.fb) as usize
    }

    /// Width of a row.
    pub fn width(&self) -> u32 {
        self.fa * self.fb
    }
}

impl Device {
    /// Append the transform of `rows` rows of `x` into `out` (added to `out` when `acc`).
    pub fn kron(&self, s: &mut Vec<Step>, k: &Kron, x: &DeviceBuffer, rows: u32, out: &DeviceBuffer, acc: bool) {
        let bufs = [x, &k.b, &k.a, &k.pre, &k.perm, &k.post, out];
        s.push(self.gpu().dispatch(self.k("mirage_kron_rows"), &bufs, &[rows, k.fa, k.fb, u32::from(acc)], gpu_core::Dispatch::Workgroups(rows)));
    }

    /// Gate stage of HConvGLU on tap-major bands.
    pub fn hconv_band_glu(&self, s: &mut Vec<Step>, u: &DeviceBuffer, gate: &DeviceBuffer, value: &DeviceBuffer, bias: &DeviceBuffer, (rows, n, k): (u32, u32, u32), z: &DeviceBuffer) {
        s.push(self.gpu().step(self.k("mirage_hconv_band_glu"), &[u, gate, value, bias, z], &[rows, n, k], rows * n));
    }

    /// Banded channel convolution with tap-major weights.
    pub fn band_cols(&self, s: &mut Vec<Step>, u: &DeviceBuffer, w: &DeviceBuffer, (rows, n, k): (u32, u32, u32), y: &DeviceBuffer) {
        s.push(self.gpu().step(self.k("mirage_band_cols"), &[u, w, y], &[rows, n, k], rows * n));
    }
}
