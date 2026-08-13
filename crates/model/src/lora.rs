// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared host-side LoRA (low-rank adapter) machinery: the generic
//! `W_eff = W + (α/r)·B·A` pair — init, delta apply (plain and
//! strided-into-fused-tensor), the `dW → (dA, dB)` projection, and the Adam
//! moments - hoisted from `flux2::lora` / `s3dit::lora`, which carried it as
//! two near-verbatim copies (the next `chw_to_hwc`, per the hoist-and-migrate
//! policy). Each model keeps only what genuinely differs: its block walk
//! (which linears are targeted, fused-tensor offsets), serialization naming,
//! and its own init distribution (passed into [`Pair::new`] as a closure, so
//! existing seeds keep producing bit-identical adapters).
//!
//! `crates/qwen/src/lora.rs` is a legitimately different design (device-side
//! param-list adapters, not host pair math) and deliberately not folded in.

/// LoRA hyper-parameters. `alpha/rank` is the delta scale ([`LoraCfg::scale`]).
#[derive(Clone, Copy)]
pub struct LoraCfg {
    pub rank: usize,
    pub alpha: f32,
    pub seed: u64,
}

impl LoraCfg {
    pub fn new(rank: usize) -> LoraCfg {
        LoraCfg { rank, alpha: rank as f32, seed: 0 }
    }
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// A single linear's adapter: `A [r×in]`, `B [out×r]`, plus Adam moments.
/// Weights are public so a model's serializer can read/overwrite `a`/`b`
/// directly (`to_tensors`/`from_tensors`); the moments stay private — they
/// are reset on reload by design.
#[derive(Clone)]
pub struct Pair {
    pub out: usize,
    pub inn: usize,
    pub r: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    ma: Vec<f32>,
    va: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Pair {
    /// Standard LoRA init: `A` drawn from `init` (the caller's own small
    /// random distribution — kept caller-side so each model's existing seeds
    /// reproduce bit-identical adapters), `B = 0` (initial no-op).
    pub fn new(out: usize, inn: usize, r: usize, mut init: impl FnMut() -> f32) -> Pair {
        let a: Vec<f32> = (0..r * inn).map(|_| init()).collect();
        Pair {
            out,
            inn,
            r,
            a,
            b: vec![0.0; out * r],
            ma: vec![0.0; r * inn],
            va: vec![0.0; r * inn],
            mb: vec![0.0; out * r],
            vb: vec![0.0; out * r],
        }
    }

    /// `w += scale·B·A` in `W`'s `[out×in]` row-major layout.
    pub fn delta(&self, scale: f32, w: &mut [f32]) {
        self.delta_strided(scale, w, 0, self.inn, 0);
    }

    /// `out_buf[(row0+o)·row_stride + col0 + i] += scale·(B·A)[o,i]` — the
    /// fused-tensor fold: row slices use `row0`, a column split uses
    /// `col0`/`row_stride`.
    pub fn delta_strided(&self, scale: f32, out_buf: &mut [f32], row0: usize, row_stride: usize, col0: usize) {
        for o in 0..self.out {
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            let wrow = &mut out_buf[(row0 + o) * row_stride + col0..(row0 + o) * row_stride + col0 + self.inn];
            for (k, &bk) in brow.iter().enumerate() {
                let bok = bk * scale;
                if bok == 0.0 {
                    continue;
                }
                let arow = &self.a[k * self.inn..(k + 1) * self.inn];
                for i in 0..self.inn {
                    wrow[i] += bok * arow[i];
                }
            }
        }
    }

    /// Project the base-weight grad `dW [out×in]` to `(dA [r×in], dB [out×r])`:
    /// `dA = scale·Bᵀ·dW`, `dB = scale·dW·Aᵀ`.
    pub fn project(&self, dw: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
        let mut da = vec![0.0f32; self.r * self.inn];
        let mut db = vec![0.0f32; self.out * self.r];
        for o in 0..self.out {
            let dwrow = &dw[o * self.inn..(o + 1) * self.inn];
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            for k in 0..self.r {
                let arow = &self.a[k * self.inn..(k + 1) * self.inn];
                let mut acc = 0.0f32;
                for i in 0..self.inn {
                    acc += dwrow[i] * arow[i];
                }
                db[o * self.r + k] = acc * scale;
                let bok = brow[k] * scale;
                if bok != 0.0 {
                    let darow = &mut da[k * self.inn..(k + 1) * self.inn];
                    for i in 0..self.inn {
                        darow[i] += bok * dwrow[i];
                    }
                }
            }
        }
        (da, db)
    }

    /// One Adam step on `A,B` (β 0.9/0.999, eps 1e-8, no weight decay).
    pub fn adam_step(&mut self, da: &[f32], db: &[f32], lr: f32, t: u64) {
        adam(&mut self.a, &mut self.ma, &mut self.va, da, lr, t);
        adam(&mut self.b, &mut self.mb, &mut self.vb, db, lr, t);
    }
}

/// In-place bias-corrected Adam (β 0.9/0.999, eps 1e-8, no weight decay).
pub fn adam(p: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32], lr: f32, t: u64) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        p[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
    }
}

/// Project `dw` onto `p`'s adapter grads and Adam-step them — the one-liner
/// every per-linear walk calls.
pub fn proj_step(p: &mut Pair, dw: &[f32], scale: f32, lr: f32, t: u64) {
    let (da, db) = p.project(dw, scale);
    p.adam_step(&da, &db, lr, t);
}

/// A cheap deterministic standard-normal (xorshift + Box–Muller half) — the
/// init distribution `s3dit::lora` seeds `A` with.
pub fn randn(s: &mut u64) -> f64 {
    let mut nx = || {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    };
    let (u1, u2) = (nx(), nx());
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
