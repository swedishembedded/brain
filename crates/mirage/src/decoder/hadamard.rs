// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Hadamard MLP: three Kronecker-factored orthogonal-initialised
//! rotations of the model width, a permutation after the first two, and a
//! rank-8 softmax conditioning between the first and second.
//!
//! The two permutations are fixed by the training code: the permutations of
//! `0..n` produced by a Mersenne-Twister generator seeded 11 and 13 with the
//! legacy shuffle. [`legacy_permutation`] reproduces them so a checkpoint
//! needs nothing but its safetensors weights.

use backend_cpu::par;
use model::hostmath;

use crate::config::{MirageConfig, HADAMARD_COND_RANK};
use crate::vision::ops::Linear;
use crate::vision::weights::Weights;

/// Seeds of the two permutations.
pub const PERMUTATION_SEEDS: (u32, u32) = (11, 13);

/// MT19937 with the classic 32-bit seeding.
struct Mt19937 {
    state: [u32; 624],
    pos: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { state, pos: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
                if y & 1 == 1 {
                    v ^= 0x9908_b0df;
                }
                self.state[i] = v;
            }
            self.pos = 0;
        }
        let mut y = self.state[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// `numpy.random.RandomState(seed).permutation(n)`: a Fisher-Yates shuffle
/// from the end, drawing each index by masked rejection sampling.
pub fn legacy_permutation(seed: u32, n: usize) -> Vec<u32> {
    let mut rng = Mt19937::new(seed);
    let mut x: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let mut mask = i as u32;
        for shift in [1, 2, 4, 8, 16] {
            mask |= mask >> shift;
        }
        let j = loop {
            let v = rng.next_u32() & mask;
            if v as usize <= i {
                break v as usize;
            }
        };
        x.swap(i, j);
    }
    x
}

/// One layer's Hadamard MLP parameters.
pub struct HadamardMlp {
    n: usize,
    factors: (usize, usize),
    d: [Vec<f32>; 4],
    b2: Vec<f32>,
    w: [(Vec<f32>, Vec<f32>); 3],
    cond_v: Linear,
    cond_u: Vec<f32>,
}

impl HadamardMlp {
    /// Layer `layer` of the stacked decoder tensors.
    pub fn new(cfg: &MirageConfig, w: &Weights, layer: usize) -> HadamardMlp {
        let p = |name: &str| w.layer(&format!("stack/layers/block/hadamard_mlp/{name}"), layer).to_vec();
        let n = cfg.hadamard_width();
        HadamardMlp {
            n,
            factors: cfg.hadamard_factors(),
            d: [p("d1"), p("d2"), p("d3"), p("d4")],
            b2: p("b2"),
            w: [(p("w1a"), p("w1b")), (p("w2a"), p("w2b")), (p("w3a"), p("w3b"))],
            cond_v: Linear::from_flax(&p("cond_v"), cfg.dim, HADAMARD_COND_RANK),
            cond_u: p("cond_u"),
        }
    }

    /// Kronecker product applied to each row viewed as `[fa, fb]`:
    /// `out = a^T Z b` for factors `a: [fa, fa]`, `b: [fb, fb]`.
    fn kron(&self, z: &mut [f32], a: &[f32], b: &[f32]) {
        let (fa, fb) = self.factors;
        par::rows_mut(z, self.n, |_, row| {
            let mut tmp = vec![0f32; self.n];
            for i in 0..fa {
                for l in 0..fb {
                    tmp[i * fb + l] = (0..fb).map(|j| row[i * fb + j] * b[j * fb + l]).sum();
                }
            }
            for k in 0..fa {
                for l in 0..fb {
                    row[k * fb + l] = (0..fa).map(|i| a[i * fa + k] * tmp[i * fb + l]).sum();
                }
            }
        });
    }

    /// `x: [rows, n]` (already normalised) -> `[rows, n]`.
    pub fn forward(&self, x: &[f32], rows: usize, perms: &(Vec<u32>, Vec<u32>)) -> Vec<f32> {
        let n = self.n;
        let mut logits = self.cond_v.apply(x, rows);
        par::rows_mut(&mut logits, HADAMARD_COND_RANK, |_, r| hostmath::softmax(r));
        let mut z: Vec<f32> = x.iter().enumerate().map(|(i, v)| self.d[0][i % n] * v).collect();
        self.kron(&mut z, &self.w[0].0, &self.w[0].1);
        let mut z = permute_rows(&z, n, &perms.0);
        par::rows_mut(&mut z, n, |r, row| {
            let probs = &logits[r * HADAMARD_COND_RANK..(r + 1) * HADAMARD_COND_RANK];
            for (c, v) in row.iter_mut().enumerate() {
                let cond = 1.0 + (0..HADAMARD_COND_RANK).map(|k| probs[k] * self.cond_u[k * n + c]).sum::<f32>();
                *v = hostmath::silu(self.d[1][c] * cond * *v + self.b2[c]);
            }
        });
        self.kron(&mut z, &self.w[1].0, &self.w[1].1);
        let mut z = permute_rows(&z, n, &perms.1);
        par::rows_mut(&mut z, n, |_, row| row.iter_mut().enumerate().for_each(|(c, v)| *v *= self.d[2][c]));
        self.kron(&mut z, &self.w[2].0, &self.w[2].1);
        par::rows_mut(&mut z, n, |_, row| row.iter_mut().enumerate().for_each(|(c, v)| *v *= self.d[3][c]));
        z
    }
}

/// `out[i] = z[perm[i]]` per row.
fn permute_rows(z: &[f32], n: usize, perm: &[u32]) -> Vec<f32> {
    let mut out = vec![0f32; z.len()];
    par::rows_mut(&mut out, n, |r, row| {
        for (i, o) in row.iter_mut().enumerate() {
            *o = z[r * n + perm[i] as usize];
        }
    });
    out
}
