// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fp16 tensor-core training forward of causal grouped-query flash attention agrees with the portable
//! fp32 `flash_attn_causal_gqa` and with an f64 host oracle, context AND the row log-sum-exp the backward
//! rebuilds its weights from, at Qwen3's head width (128, 32 query heads over 8 KV heads) and the 256-wide
//! head of the Qwen3.8 full-attention layers, over ragged lengths and several samples.
//!
//! Swedish Embedded AB implements tensor-core training stacks that agree with their fp32 reference. If your
//! team needs expertise in moving the attention of a training step onto tensor cores without losing the
//! gradient, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The bar is the paged prefill's: an absolute error relative to the output RMS (fp16 rounding of Q, K, V and
//! the softmax weights, fp32 accumulation). Skipped without a CUDA device.

use backend_api::DeviceBuffer;
use gpu_core::Gpu;

const REL_TOL: f32 = 1.2e-2;
const LSE_TOL: f64 = 8e-3;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f32 {
        let u = |r: &mut Rng| ((r.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32;
        let (a, b) = (u(self), u(self));
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

struct Shape {
    b: u32,
    t: u32,
    n_heads: u32,
    n_kv: u32,
    hd: u32,
}

impl Shape {
    fn params(&self) -> [u32; 6] {
        [self.b, self.n_heads, self.n_kv, self.t, self.hd, self.n_heads / self.n_kv]
    }
    fn q_len(&self) -> usize {
        (self.b * self.t * self.n_heads * self.hd) as usize
    }
    fn kv_len(&self) -> usize {
        (self.b * self.t * self.n_kv * self.hd) as usize
    }
    fn lse_len(&self) -> usize {
        (self.b * self.n_heads * self.t) as usize
    }
    /// f64 causal softmax attention of `(sample, row, head)`: the context row and the log-sum-exp.
    fn oracle(&self, q: &[f32], k: &[f32], v: &[f32], sample: usize, row: usize, head: usize) -> (Vec<f64>, f64) {
        let (hd, t) = (self.hd as usize, self.t as usize);
        let hkv = head / (self.n_heads / self.n_kv) as usize;
        let (q_row, kv_row) = (self.n_heads as usize * hd, self.n_kv as usize * hd);
        let qv = &q[(sample * t + row) * q_row + head * hd..][..hd];
        let kat = |j: usize| &k[(sample * t + j) * kv_row + hkv * hd..][..hd];
        let vat = |j: usize| &v[(sample * t + j) * kv_row + hkv * hd..][..hd];
        let scores: Vec<f64> = (0..=row).map(|j| (0..hd).map(|d| qv[d] as f64 * kat(j)[d] as f64).sum::<f64>() / (hd as f64).sqrt()).collect();
        let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
        let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let z: f64 = w.iter().sum();
        let ctx = (0..hd).map(|d| (0..=row).map(|j| w[j] * vat(j)[d] as f64).sum::<f64>() / z).collect();
        (ctx, mx + z.ln())
    }
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn open() -> Option<Gpu> {
    let list: &[(&str, &str)] = &[("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA)];
    match Gpu::try_new_cuda(list) {
        Ok(g) if g.caps().arch.compute_capability.is_some_and(|cc| kernels_cuda::get("flash_attn_fwd_f16_hd128").is_some_and(|k| k.min_cc <= cc)) => Some(g),
        _ => {
            brain_testutil::skip_unavailable("cuda_flash_attn_fwd: no CUDA device at the kernel's capability floor");
            None
        }
    }
}

fn run(gpu: &Gpu, s: &Shape, seed: u64) {
    let what = format!("b={} t={} heads={}/{} hd={}", s.b, s.t, s.n_heads, s.n_kv, s.hd);
    let mut r = Rng(seed);
    let q: Vec<f32> = (0..s.q_len()).map(|_| r.gauss()).collect();
    let k: Vec<f32> = (0..s.kv_len()).map(|_| r.gauss()).collect();
    let v: Vec<f32> = (0..s.kv_len()).map(|_| r.gauss()).collect();
    let (qb, kb, vb) = (gpu.storage_init("q", &q), gpu.storage_init("k", &k), gpu.storage_init("v", &v));
    let outputs = || (gpu.storage(s.q_len() as u64), gpu.storage(s.lse_len() as u64));
    let read = |ctx: &DeviceBuffer, lse: &DeviceBuffer| {
        gpu.poll_wait();
        (gpu.read(ctx, s.q_len()), gpu.read(lse, s.lse_len()))
    };

    let (pc, pl) = outputs();
    let kind = gpu.kernel_index("flash_attn_causal_gqa").expect("registered");
    let rows_per_wg = if s.hd > 128 { 32 } else { 64 };
    gpu.submit(&[], &[gpu.dispatch(kind, &[&qb, &kb, &vb, &pc, &pl], &s.params(), gpu_core::Dispatch::Workgroups(s.b * s.n_heads * s.t.div_ceil(rows_per_wg)))]);
    let (want, want_lse) = read(&pc, &pl);

    let (nc, nl) = outputs();
    gpu.write(&nc, &vec![0x7fc0_0000u32; s.q_len()]);
    let step = gpu_core::provider::cuda::flash_causal_gqa_fwd_step(gpu, &[&qb, &kb, &vb, &nc, &nl], &s.params()).expect("the native forward takes this shape");
    gpu.submit(&[], &[step]);
    let (got, got_lse) = read(&nc, &nl);

    assert!(got.iter().all(|x| x.is_finite()) && got_lse.iter().all(|x| x.is_finite()), "{what}: non-finite output");
    let scale = rms(&want);
    let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let lse_diff = want_lse.iter().zip(&got_lse).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("cuda_flash_attn_fwd: {what}: max|tc - portable| ctx {maxdiff:e} ({:.2e} of rms), lse {lse_diff:e}", maxdiff / scale);
    assert!(maxdiff <= REL_TOL * scale, "{what}: context differs from the portable kernel by {maxdiff:e}");
    assert!((lse_diff as f64) <= LSE_TOL, "{what}: lse differs from the portable kernel by {lse_diff:e}");

    let mut rng = Rng(seed ^ 0xABCD);
    for _ in 0..40 {
        let (sample, row, head) = ((rng.next() % s.b as u64) as usize, (rng.next() % s.t as u64) as usize, (rng.next() % s.n_heads as u64) as usize);
        let (truth, truth_lse) = s.oracle(&q, &k, &v, sample, row, head);
        let hd = s.hd as usize;
        let base = ((sample * s.t as usize + row) * s.n_heads as usize + head) * hd;
        let err = (0..hd).map(|d| (got[base + d] as f64 - truth[d]).abs()).fold(0f64, f64::max);
        assert!(err <= (REL_TOL * scale) as f64, "{what}: sample {sample} row {row} head {head} is {err:e} from the f64 oracle");
        let lse_err = (got_lse[(sample * s.n_heads as usize + head) * s.t as usize + row] as f64 - truth_lse).abs();
        assert!(lse_err <= LSE_TOL, "{what}: sample {sample} row {row} head {head} lse is {lse_err:e} from the f64 oracle");
    }
}

#[test]
fn the_tensor_core_training_forward_agrees_with_the_portable_kernel_and_the_oracle() {
    let Some(gpu) = open() else { return };
    // Qwen3-8B's attention (head 128, 32/8) at ragged lengths, two samples; then a narrow GQA head and the
    // 256-wide head of the Qwen3.8 full-attention layers (24 over 4 KV heads).
    run(&gpu, &Shape { b: 2, t: 300, n_heads: 32, n_kv: 8, hd: 128 }, 1);
    run(&gpu, &Shape { b: 1, t: 64, n_heads: 8, n_kv: 2, hd: 128 }, 2);
    run(&gpu, &Shape { b: 1, t: 77, n_heads: 4, n_kv: 4, hd: 128 }, 3);
    run(&gpu, &Shape { b: 2, t: 200, n_heads: 24, n_kv: 4, hd: 256 }, 4);
}

#[test]
fn a_head_width_the_forward_was_not_written_for_is_declined() {
    let Some(gpu) = open() else { return };
    let s = Shape { b: 1, t: 64, n_heads: 4, n_kv: 4, hd: 96 };
    let (q, k, v, c, l) = (gpu.storage(1), gpu.storage(1), gpu.storage(1), gpu.storage(1), gpu.storage(1));
    assert!(gpu_core::provider::cuda::flash_causal_gqa_fwd_step(&gpu, &[&q, &k, &v, &c, &l], &s.params()).is_none());
}
