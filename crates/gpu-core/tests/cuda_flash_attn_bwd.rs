// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fp16 tensor-core backward of causal grouped-query flash attention (dQ, dK, dV at Qwen3's head width 128)
//! agrees with the portable fp32 `flash_attn_causal_gqa_bwd_{dq,dkv}` pair on the same forward's context and
//! log-sum-exp, over ragged lengths, several samples and a gradient as small as a mean-reduced loss makes it
//! (1e-6 of unit scale: far below fp16's normal range, which the call's power-of-two scale exists to handle).
//!
//! Swedish Embedded AB implements tensor-core training stacks that agree with their fp32 reference. If your
//! team needs expertise in moving the attention backward of a fine-tuning step onto tensor cores without losing
//! the gradient, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! Each gradient is compared by cosine and by its largest absolute error relative to its RMS. Skipped without
//! a CUDA device.

use gpu_core::provider::cuda::{flash_causal_gqa_bwd_steps, FlashBwdBufs};
use gpu_core::Gpu;

const HD: u32 = 128;
const REL_TOL: f32 = 2.5e-2;
const MIN_COSINE: f64 = 0.9995;

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

#[derive(Clone, Copy)]
struct Shape {
    b: u32,
    t: u32,
    n_heads: u32,
    n_kv: u32,
}

impl Shape {
    fn params(&self) -> [u32; 6] {
        [self.b, self.n_heads, self.n_kv, self.t, HD, self.n_heads / self.n_kv]
    }
    fn q_len(&self) -> usize {
        (self.b * self.t * self.n_heads * HD) as usize
    }
    fn kv_len(&self) -> usize {
        (self.b * self.t * self.n_kv * HD) as usize
    }
    fn lse_len(&self) -> usize {
        (self.b * self.n_heads * self.t) as usize
    }
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

fn open() -> Option<Gpu> {
    let list: &[(&str, &str)] = &[
        ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA),
        ("flash_attn_causal_gqa_bwd_dq", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DQ),
        ("flash_attn_causal_gqa_bwd_dkv", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DKV),
    ];
    match Gpu::try_new_cuda(list) {
        Ok(g) if g.caps().arch.compute_capability.is_some_and(|cc| kernels_cuda::get("flash_bwd_dq").is_some_and(|k| k.min_cc <= cc)) => Some(g),
        _ => {
            brain_testutil::skip_unavailable("cuda_flash_attn_bwd: no CUDA device at the kernel's capability floor");
            None
        }
    }
}

fn run(gpu: &Gpu, s: Shape, d_scale: f32, seed: u64) {
    let what = format!("b={} t={} heads={}/{} dO x {d_scale:e}", s.b, s.t, s.n_heads, s.n_kv);
    let mut r = Rng(seed);
    let q: Vec<f32> = (0..s.q_len()).map(|_| r.gauss()).collect();
    let k: Vec<f32> = (0..s.kv_len()).map(|_| r.gauss()).collect();
    let v: Vec<f32> = (0..s.kv_len()).map(|_| r.gauss()).collect();
    let d_o: Vec<f32> = (0..s.q_len()).map(|_| r.gauss() * d_scale).collect();
    let (qb, kb, vb, dob) = (gpu.storage_init("q", &q), gpu.storage_init("k", &k), gpu.storage_init("v", &v), gpu.storage_init("d_o", &d_o));
    let (ctx, lse) = (gpu.storage(s.q_len() as u64), gpu.storage(s.lse_len() as u64));
    let params = s.params();
    let wg = |heads: u32| gpu_core::Dispatch::Workgroups(s.b * heads * s.t.div_ceil(64));
    let id = |name: &str| gpu.kernel_index(name).expect("registered");
    // The portable forward gives both backwards their context and log-sum-exp.
    gpu.submit(&[], &[gpu.dispatch(id("flash_attn_causal_gqa"), &[&qb, &kb, &vb, &ctx, &lse], &params, wg(s.n_heads))]);

    let outputs = || (gpu.storage(s.q_len() as u64), gpu.storage(s.kv_len() as u64), gpu.storage(s.kv_len() as u64), gpu.storage(s.lse_len() as u64));
    let (rdq, rdk, rdv, rdsum) = outputs();
    gpu.submit(
        &[],
        &[
            gpu.dispatch(id("flash_attn_causal_gqa_bwd_dq"), &[&qb, &kb, &vb, &ctx, &dob, &lse, &rdq, &rdsum], &params, wg(s.n_heads)),
            gpu.dispatch(id("flash_attn_causal_gqa_bwd_dkv"), &[&qb, &kb, &vb, &dob, &lse, &rdsum, &rdk, &rdv], &params, wg(s.n_kv)),
        ],
    );
    gpu.poll_wait();
    let want = (gpu.read(&rdq, s.q_len()), gpu.read(&rdk, s.kv_len()), gpu.read(&rdv, s.kv_len()));

    let (ndq, ndk, ndv, ndsum) = outputs();
    for (b, n) in [(&ndq, s.q_len()), (&ndk, s.kv_len()), (&ndv, s.kv_len())] {
        gpu.write(b, &vec![0x7fc0_0000u32; n]);
    }
    let steps = flash_causal_gqa_bwd_steps(gpu, &FlashBwdBufs { q: &qb, k: &kb, v: &vb, o: &ctx, d_o: &dob, lse: &lse, dsum: &ndsum, d_q: &ndq, d_k: &ndk, d_v: &ndv }, &params).expect("the native backward takes this shape");
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let got = (gpu.read(&ndq, s.q_len()), gpu.read(&ndk, s.kv_len()), gpu.read(&ndv, s.kv_len()));

    for (name, g, w) in [("dQ", &got.0, &want.0), ("dK", &got.1, &want.1), ("dV", &got.2, &want.2)] {
        assert!(g.iter().all(|x| x.is_finite()), "{what}: {name} is not finite");
        let scale = rms(w);
        let maxdiff = g.iter().zip(w.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let cos = cosine(g, w);
        eprintln!("cuda_flash_attn_bwd: {what}: {name} cosine {cos:.6} max error {:.2e} of rms {scale:e}", maxdiff / scale);
        assert!(cos >= MIN_COSINE, "{what}: {name} cosine {cos} against the portable kernel");
        assert!(maxdiff <= REL_TOL * scale, "{what}: {name} differs from the portable kernel by {maxdiff:e} (bar {:e})", REL_TOL * scale);
    }
}

#[test]
fn the_tensor_core_backward_agrees_with_the_portable_pair() {
    let Some(gpu) = open() else { return };
    // Qwen3-8B's attention (32 query heads over 8 KV heads) at ragged lengths and two samples; a narrow GQA head
    // and plain MHA; each with a unit-scale gradient and one a million times smaller.
    for (i, (shape, d_scale)) in [
        (Shape { b: 2, t: 300, n_heads: 32, n_kv: 8 }, 1.0),
        (Shape { b: 2, t: 300, n_heads: 32, n_kv: 8 }, 1e-6),
        (Shape { b: 1, t: 64, n_heads: 8, n_kv: 2 }, 1.0),
        (Shape { b: 1, t: 77, n_heads: 4, n_kv: 4 }, 1e-4),
    ]
    .into_iter()
    .enumerate()
    {
        run(&gpu, shape, d_scale, 11 + i as u64);
    }
}

#[test]
fn a_head_width_the_backward_was_not_written_for_is_declined() {
    let Some(gpu) = open() else { return };
    let b = gpu.storage(1);
    let bufs = FlashBwdBufs { q: &b, k: &b, v: &b, o: &b, d_o: &b, lse: &b, dsum: &b, d_q: &b, d_k: &b, d_v: &b };
    assert!(flash_causal_gqa_bwd_steps(&gpu, &bufs, &[1, 4, 4, 64, 256, 1]).is_none());
}
