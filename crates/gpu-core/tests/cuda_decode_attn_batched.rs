// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native batched paged-KV decode attention (head_dim 128, int8 and fp32 pools) agrees with an f64 host oracle
//! over several sequences of ragged lengths read through scattered block tables, at Qwen3's group sizes (4 query
//! heads per kv head, and 2 and 1), and is declined for one sequence (the split-key kernel owns that) and for other
//! head widths.
//!
//! Swedish Embedded AB implements long-context inference throughput for its clients. If your team needs expertise in
//! making the decode attention of a batched, paged-KV serving stack run at memory bandwidth, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! An int8 pool is checked against the values it holds (int8 times the row scale); the result is fp32 arithmetic, so
//! the bar is tight. Skipped without a CUDA device.

use gpu_core::provider::cuda::paged_decode_batched_step;
use gpu_core::Gpu;

const HD: usize = 128;
const BS: u32 = 16;

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

fn open() -> Option<Gpu> {
    match Gpu::try_new_cuda(&[]) {
        Ok(g) if g.caps().arch.compute_capability.is_some_and(|cc| kernels_cuda::get("decode_attn_batched_i8").is_some_and(|k| k.min_cc <= cc)) => Some(g),
        _ => {
            brain_testutil::skip_unavailable("cuda_decode_attn_batched: no CUDA device");
            None
        }
    }
}

fn run(gpu: &Gpu, int8: bool, n_heads: usize, n_kv: usize, lens: &[u32], seed: u64) {
    let what = format!("{} heads {n_heads}/{n_kv} lens {lens:?}", if int8 { "int8" } else { "fp32" });
    let group = n_heads / n_kv;
    let bsz = lens.len();
    let max_len = *lens.iter().max().unwrap() as usize;
    let max_bt = max_len.div_ceil(BS as usize);
    // A pool larger than the sequences need, each sequence's blocks scattered (reversed order, interleaved).
    let blocks = bsz * max_bt + 3;
    let mut r = Rng(seed);
    let rows = blocks * BS as usize;
    let kv_row = n_kv * HD;
    let q: Vec<f32> = (0..bsz * n_heads * HD).map(|_| r.gauss()).collect();
    let kf: Vec<f32> = (0..rows * kv_row).map(|_| r.gauss()).collect();
    let vf: Vec<f32> = (0..rows * kv_row).map(|_| r.gauss()).collect();
    let tables: Vec<u32> = (0..bsz).flat_map(|b| (0..max_bt).map(move |i| (blocks - 1 - (i * bsz + b)) as u32)).collect();
    // Dequantised plane values the oracle reads, and what the device is given.
    let quant = |plane: &[f32]| -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let (mut words, mut scales, mut deq) = (Vec::new(), vec![0f32; rows * n_kv], vec![0f32; plane.len()]);
        let mut codes = vec![0i8; plane.len()];
        for rr in 0..rows * n_kv {
            let row = &plane[rr * HD..(rr + 1) * HD];
            let s = row.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
            scales[rr] = s;
            for d in 0..HD {
                let c = (row[d] / s).round().clamp(-127.0, 127.0);
                codes[rr * HD + d] = c as i8;
                deq[rr * HD + d] = c * s;
            }
        }
        for c in codes.chunks(4) {
            words.push(c.iter().enumerate().fold(0u32, |w, (j, &x)| w | ((x as u8 as u32) << (8 * j))));
        }
        (words, scales, deq)
    };
    let words = |w: &[u32]| {
        let b = gpu.storage(w.len().max(1) as u64);
        gpu.write(&b, w);
        b
    };
    let (qb, bt, lb) = (gpu.storage_init("q", &q), words(&tables), words(lens));
    let ctx = gpu.storage((bsz * n_heads * HD) as u64);
    let params = [bsz as u32, n_heads as u32, n_kv as u32, HD as u32, group as u32, BS, max_bt as u32];
    // The device buffers stay alive in `_pool` while the step runs.
    let (step, _pool, k_seen, v_seen) = if int8 {
        let ((kw, kscale, kd), (vw, vscale, vd)) = (quant(&kf), quant(&vf));
        let (k, v, ks, vs) = (words(&kw), words(&vw), gpu.storage_init("ks", &kscale), gpu.storage_init("vs", &vscale));
        let step = paged_decode_batched_step(gpu, true, &[&qb, &k, &v, &ks, &vs, &bt, &lb, &ctx], &params);
        (step, vec![k, v, ks, vs], kd, vd)
    } else {
        let (k, v) = (gpu.storage_init("k", &kf), gpu.storage_init("v", &vf));
        let step = paged_decode_batched_step(gpu, false, &[&qb, &k, &v, &bt, &lb, &ctx], &params);
        (step, vec![k, v], kf.clone(), vf.clone())
    };
    let step = step.expect("the native batched decode attention takes this shape");
    gpu.write(&ctx, &vec![0x7fc0_0000u32; bsz * n_heads * HD]);
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let got = gpu.read(&ctx, bsz * n_heads * HD);
    let rms = (got.iter().map(|x| x * x).sum::<f32>() / got.len() as f32).sqrt();
    assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite output");
    for b in 0..bsz {
        for h in 0..n_heads {
            let hkv = h / group;
            let len = lens[b] as usize;
            let qv = &q[(b * n_heads + h) * HD..][..HD];
            let row_of = |j: usize| tables[b * max_bt + j / BS as usize] as usize * BS as usize + j % BS as usize;
            let scores: Vec<f64> = (0..len).map(|j| (0..HD).map(|d| qv[d] as f64 * k_seen[row_of(j) * kv_row + hkv * HD + d] as f64).sum::<f64>() / (HD as f64).sqrt()).collect();
            let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            for d in 0..HD {
                let truth: f64 = (0..len).map(|j| w[j] * v_seen[row_of(j) * kv_row + hkv * HD + d] as f64).sum::<f64>() / z;
                let err = (got[(b * n_heads + h) * HD + d] as f64 - truth).abs();
                assert!(err <= 2e-4 * rms as f64 + 1e-6, "{what}: seq {b} head {h} channel {d} is {err:e} from the f64 oracle (rms {rms:e})");
            }
        }
    }
}

#[test]
fn the_batched_decode_attention_matches_the_f64_oracle() {
    let Some(gpu) = open() else { return };
    for int8 in [true, false] {
        run(&gpu, int8, 32, 8, &[1, 17, 130, 300, 5, 64, 16, 1000], 1);
        run(&gpu, int8, 16, 8, &[33, 70], 2);
        run(&gpu, int8, 8, 8, &[200, 9, 2000], 3);
    }
}

#[test]
fn one_sequence_and_other_head_widths_are_declined() {
    let Some(gpu) = open() else { return };
    let b = gpu.storage(1);
    let bufs = [&b, &b, &b, &b, &b, &b];
    assert!(paged_decode_batched_step(&gpu, false, &bufs, &[1, 32, 8, 128, 4, 16, 4]).is_none());
    assert!(paged_decode_batched_step(&gpu, false, &bufs, &[4, 32, 8, 256, 4, 16, 4]).is_none());
    assert!(paged_decode_batched_step(&gpu, false, &bufs, &[4, 64, 4, 128, 16, 16, 4]).is_none());
}
