// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vector-quantization host helpers: nearest-codebook assignment dispatch
//! (Euclidean / cosine) and the straight-through / commitment / EMA host math
//! for VQ-VAE tokenizers (GenieRedux, iVideoGPT).
//!
//! The device produces only the ASSIGNMENT (`vq_argmin`/`vq_argmax_dot` ->
//! packed `[idx, score]` per query). Everything differentiable composes from
//! existing kernels:
//!
//! - forward: `embed(idx, codebook) -> q` (the quantized vectors)
//! - straight-through backward: route `d_q` straight into the encoder's `d_z`
//!   (the argmin is treated as identity); the codebook receives its own grad
//!   via `emb_bwd(idx, d_codebook)` OR an EMA update ([`ema_update`]).
//! - commitment loss `beta * ||sg[q] - z||^2` and codebook loss
//!   `||sg[z] - q||^2` are plain MSE terms the caller adds.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Kernel-table indices of the VQ assignment kernels.
#[derive(Clone, Copy, Debug)]
pub struct Vq {
    pub argmin: usize,
    pub argmax_dot: usize,
}

impl Vq {
    /// `(name, source)` pairs for `Gpu::new`, matching [`Vq::seq`].
    pub fn kernel_sources() -> [(&'static str, &'static str); 2] {
        [("vq_argmin", kernels::VQ_ARGMIN), ("vq_argmax_dot", kernels::VQ_ARGMAX_DOT)]
    }
    pub fn seq() -> Vq {
        Vq { argmin: 0, argmax_dot: 1 }
    }

    /// Euclidean assignment: `out[2m]=argmin_k ||x[m]-cb[k]||^2`,
    /// `out[2m+1]=min dist`. `m` queries, `k` codes, `d` dims. `out` len `2m`.
    pub fn step_argmin(
        &self,
        gpu: &Gpu,
        m: u32,
        k: u32,
        d: u32,
        x: &DeviceBuffer,
        cb: &DeviceBuffer,
        out: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.argmin, &[x, cb, out], &[m, k, d], m)
    }

    /// Cosine assignment (inputs pre-normalised): `out[2m]=argmax_k <x,cb>`,
    /// `out[2m+1]=max dot`.
    pub fn step_argmax_dot(
        &self,
        gpu: &Gpu,
        m: u32,
        k: u32,
        d: u32,
        x: &DeviceBuffer,
        cb: &DeviceBuffer,
        out: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.argmax_dot, &[x, cb, out], &[m, k, d], m)
    }
}

/// Extract the integer code indices from the packed `[idx, score]` output.
pub fn indices(packed: &[f32]) -> Vec<u32> {
    packed.chunks_exact(2).map(|c| c[0] as u32).collect()
}

/// EMA codebook update (van den Oord VQ-VAE): keeps a running cluster size
/// `n_k` and sum `m_k`, then `codebook[k] = m_k / n_k`. `decay` in (0,1).
/// Dead codes (never assigned) are left unchanged. Host-side; the caller
/// stores `n`/`m` across steps and marks the codebook `Role::Frozen`.
pub fn ema_update(
    codebook: &mut [f32],
    cluster_n: &mut [f32],
    cluster_m: &mut [f32],
    z: &[f32],
    idx: &[u32],
    k: usize,
    d: usize,
    decay: f32,
    eps: f32,
) {
    let mut count = vec![0.0f32; k];
    let mut sum = vec![0.0f32; k * d];
    for (m, &ci) in idx.iter().enumerate() {
        let c = ci as usize;
        count[c] += 1.0;
        for i in 0..d {
            sum[c * d + i] += z[m * d + i];
        }
    }
    for c in 0..k {
        cluster_n[c] = cluster_n[c] * decay + count[c] * (1.0 - decay);
        for i in 0..d {
            cluster_m[c * d + i] = cluster_m[c * d + i] * decay + sum[c * d + i] * (1.0 - decay);
        }
    }
    // Laplace-smoothed normalisation so empty clusters don't divide by zero.
    let total: f32 = cluster_n.iter().sum::<f32>().max(1.0);
    for c in 0..k {
        let n = (cluster_n[c] + eps) / (total + k as f32 * eps) * total;
        if cluster_n[c] > 0.5 {
            for i in 0..d {
                codebook[c * d + i] = cluster_m[c * d + i] / n;
            }
        }
    }
}

/// Codebook usage: fraction of codes used at least once, and the perplexity
/// `exp(-sum p log p)` of the assignment distribution — the standard
/// collapse diagnostics.
pub fn usage_stats(idx: &[u32], k: usize) -> (f32, f32) {
    let mut counts = vec![0u32; k];
    for &c in idx {
        counts[c as usize] += 1;
    }
    let used = counts.iter().filter(|&&c| c > 0).count();
    let n = idx.len().max(1) as f32;
    let mut ent = 0.0f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / n;
            ent -= p * p.ln();
        }
    }
    (used as f32 / k as f32, ent.exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::Gpu;

    fn brute_argmin(x: &[f32], cb: &[f32], m: usize, k: usize, d: usize) -> Vec<(u32, f32)> {
        (0..m)
            .map(|mi| {
                let mut bk = 0u32;
                let mut bd = f32::MAX;
                for ki in 0..k {
                    let dist: f32 =
                        (0..d).map(|i| (x[mi * d + i] - cb[ki * d + i]).powi(2)).sum();
                    if dist < bd {
                        bd = dist;
                        bk = ki as u32;
                    }
                }
                (bk, bd)
            })
            .collect()
    }

    fn rnd(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                (((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
            })
            .collect()
    }

    #[test]
    fn vq_argmin_matches_brute_force() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (m, k, d) = (37usize, 19usize, 8usize);
        let gpu = Gpu::new_cpu(&Vq::kernel_sources());
        let vq = Vq::seq();
        let x = rnd(1, m * d);
        let cb = rnd(2, k * d);
        let xb = gpu.storage_init("x", &x);
        let cbb = gpu.storage_init("cb", &cb);
        let out = gpu.storage((2 * m) as u64);
        gpu.submit(&[], &[vq.step_argmin(&gpu, m as u32, k as u32, d as u32, &xb, &cbb, &out)]);
        let got = gpu.read(&out, 2 * m);
        let want = brute_argmin(&x, &cb, m, k, d);
        for mi in 0..m {
            assert_eq!(got[2 * mi] as u32, want[mi].0, "idx m={mi}");
            assert!((got[2 * mi + 1] - want[mi].1).abs() < 1e-4, "dist m={mi}");
        }
    }

    #[test]
    fn vq_argmin_ties_pick_lowest_index() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        // Two identical codebook rows: the query must map to the LOWER index.
        let (m, k, d) = (1usize, 3usize, 2usize);
        let gpu = Gpu::new_cpu(&Vq::kernel_sources());
        let vq = Vq::seq();
        let x = vec![1.0, 1.0];
        let cb = vec![5.0, 5.0, 1.0, 1.0, 1.0, 1.0]; // codes 1,2 both equal x
        let xb = gpu.storage_init("x", &x);
        let cbb = gpu.storage_init("cb", &cb);
        let out = gpu.storage(2);
        gpu.submit(&[], &[vq.step_argmin(&gpu, m as u32, k as u32, d as u32, &xb, &cbb, &out)]);
        let got = gpu.read(&out, 2);
        assert_eq!(got[0] as u32, 1, "tie must resolve to lowest index");
    }

    #[test]
    fn vq_argmax_dot_is_scale_invariant_after_norm() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        // Cosine picks the best-aligned code regardless of magnitude when
        // inputs are normalised. Here codebook already unit-ish; scaling a
        // query by a positive constant must not change the argmax.
        let (m, k, d) = (1usize, 4usize, 3usize);
        let gpu = Gpu::new_cpu(&Vq::kernel_sources());
        let vq = Vq::seq();
        let cb = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.577, 0.577, 0.577];
        let cbb = gpu.storage_init("cb", &cb);
        let pick = |x: Vec<f32>| -> u32 {
            let xb = gpu.storage_init("x", &x);
            let out = gpu.storage(2);
            gpu.submit(&[], &[vq.step_argmax_dot(&gpu, m as u32, k as u32, d as u32, &xb, &cbb, &out)]);
            gpu.read(&out, 2)[0] as u32
        };
        let a = pick(vec![0.1, 0.9, 0.2]);
        let b = pick(vec![0.5, 4.5, 1.0]); // 5x scaled
        assert_eq!(a, b, "cosine argmax must be scale invariant");
    }

    #[test]
    fn vq_usage_stats_hand_values() {
        // idx = [0,0,1] over k=4: 2 codes used (0.5), perplexity of
        // {2/3, 1/3} = exp(-(2/3 ln(3/2) + 1/3 ln 3)) = exp(0.6365) = 1.8899.
        let (used, ppl) = usage_stats(&[0, 0, 1], 4);
        assert!((used - 0.5).abs() < 1e-6);
        assert!((ppl - 1.8899).abs() < 1e-3, "ppl={ppl}");
    }

    #[test]
    fn vq_ema_moves_codes_toward_assigned_means() {
        let (k, d) = (2usize, 2usize);
        let mut cb = vec![0.0, 0.0, 10.0, 10.0];
        let mut n = vec![0.0f32; k];
        let mut m = vec![0.0f32; k * d];
        // two queries near code 0, one near code 1
        let z = vec![1.0, 1.0, 1.2, 0.8, 9.0, 11.0];
        let idx = vec![0u32, 0, 1];
        for _ in 0..50 {
            ema_update(&mut cb, &mut n, &mut m, &z, &idx, k, d, 0.9, 1e-5);
        }
        // code 0 -> ~mean of first two queries ([1.1, 0.9]); code 1 -> [9,11].
        assert!((cb[0] - 1.1).abs() < 0.2 && (cb[1] - 0.9).abs() < 0.2, "cb0 {:?}", &cb[..2]);
        assert!((cb[2] - 9.0).abs() < 0.5 && (cb[3] - 11.0).abs() < 0.5, "cb1 {:?}", &cb[2..]);
    }
}
