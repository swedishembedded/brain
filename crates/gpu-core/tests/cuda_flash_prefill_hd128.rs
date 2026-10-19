// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fp16 tensor-core paged flash-prefill kernel at head_dim 128 (Qwen3: 32 query heads, 8 KV
//! heads) agrees with the portable fp32 `paged_flash_prefill` and with an f64 host oracle, over an
//! fp32 and an int8-KV cache, with ragged chunk lengths and non-zero starts; a round in the decode
//! regime keeps the portable kernel.
//!
//! Swedish Embedded AB implements long-context inference throughput for its clients. If your team
//! needs expertise in proving that a reduced-precision attention kernel is faster AND numerically
//! faithful to the fp32 path it replaces, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The bar is head_dim 256's ([`REL_TOL`] there too): an absolute error relative to the output RMS (fp16
//! rounding of Q, K, V and the softmax weights, fp32 accumulation). Skipped without a CUDA device.

use backend_api::DeviceBuffer;
use gpu_core::Gpu;

const HD: usize = 128;
const N_HEADS: u32 = 32;
const N_KV: u32 = 8;
const REL_TOL: f32 = 1.2e-2;

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
        (-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()
    }
}

/// One sequence: `n` query rows at `start`, a paged pool with identity block order.
struct Problem {
    n: u32,
    start: u32,
    block_size: u32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Problem {
    fn new(n: u32, start: u32, block_size: u32, seed: u64) -> Problem {
        let mut r = Rng(seed);
        let rows = (start + n) as usize;
        let kv = rows.div_ceil(block_size as usize) * block_size as usize;
        Problem {
            n,
            start,
            block_size,
            q: (0..n as usize * N_HEADS as usize * HD).map(|_| r.gauss()).collect(),
            k: (0..kv * N_KV as usize * HD).map(|_| r.gauss()).collect(),
            v: (0..kv * N_KV as usize * HD).map(|_| r.gauss()).collect(),
        }
    }
    fn max_bt(&self) -> u32 {
        (self.start + self.n).div_ceil(self.block_size)
    }
    fn params(&self) -> [u32; 7] {
        [self.n, N_HEADS, N_KV, HD as u32, N_HEADS / N_KV, self.block_size, self.max_bt()]
    }
    /// Causal softmax attention of one (row, head) in f64 over the pool values `k`/`v`.
    fn oracle(&self, k: &[f32], v: &[f32], row: usize, head: usize) -> Vec<f64> {
        let len = self.start as usize + row + 1;
        let hkv = head / (N_HEADS / N_KV) as usize;
        let kv_row = N_KV as usize * HD;
        let qv = &self.q[(row * N_HEADS as usize + head) * HD..][..HD];
        let scores: Vec<f64> = (0..len).map(|j| (0..HD).map(|d| qv[d] as f64 * k[j * kv_row + hkv * HD + d] as f64).sum::<f64>() / (HD as f64).sqrt()).collect();
        let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
        let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let z: f64 = w.iter().sum();
        (0..HD).map(|d| (0..len).map(|j| w[j] * v[j * kv_row + hkv * HD + d] as f64).sum::<f64>() / z).collect()
    }
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn gpu() -> Option<Gpu> {
    let list: &[(&str, &str)] = &[("paged_flash_prefill", kernels::PAGED_FLASH_PREFILL)];
    match Gpu::try_new_cuda(list) {
        Ok(g) if g.caps().arch.compute_capability.is_some_and(|cc| kernels_cuda::get("flash_prefill_f16_hd128").is_some_and(|k| k.min_cc <= cc)) => Some(g),
        _ => {
            brain_testutil::skip_unavailable("cuda_flash_prefill_hd128: no CUDA device at the kernel's capability floor");
            None
        }
    }
}

struct Buffers {
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    bt: DeviceBuffer,
    lens: DeviceBuffer,
    ctx: DeviceBuffer,
}

impl Buffers {
    fn upload(gpu: &Gpu, p: &Problem, k: DeviceBuffer, v: DeviceBuffer) -> Buffers {
        // Every row of a tile reads the table row of the tile's first row.
        let bt: Vec<u32> = (0..p.n).flat_map(|_| 0..p.max_bt()).collect();
        let lens: Vec<u32> = (0..p.n).map(|i| p.start + i + 1).collect();
        let words = |w: &[u32]| {
            let b = gpu.storage(w.len() as u64);
            gpu.write(&b, w);
            b
        };
        Buffers {
            q: gpu.storage_init("q", &p.q),
            k,
            v,
            bt: words(&bt),
            lens: words(&lens),
            ctx: gpu.storage(p.n as u64 * N_HEADS as u64 * HD as u64),
        }
    }
    fn six(&self) -> [&DeviceBuffer; 6] {
        [&self.q, &self.k, &self.v, &self.bt, &self.lens, &self.ctx]
    }
    fn run(&self, gpu: &Gpu, step: gpu_core::Step, p: &Problem) -> Vec<f32> {
        gpu.write(&self.ctx, &vec![0x7fc0_0000u32; p.n as usize * N_HEADS as usize * HD]);
        gpu.submit(&[], &[step]);
        gpu.read(&self.ctx, p.n as usize * N_HEADS as usize * HD)
    }
}

fn check_against_oracle(got: &[f32], p: &Problem, k: &[f32], v: &[f32], what: &str) {
    let scale = rms(got);
    assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite output");
    let mut rng = Rng(7);
    for _ in 0..48 {
        let (row, head) = ((rng.next() % p.n as u64) as usize, (rng.next() % N_HEADS as u64) as usize);
        let truth = p.oracle(k, v, row, head);
        let base = (row * N_HEADS as usize + head) * HD;
        let err = (0..HD).map(|d| (got[base + d] as f64 - truth[d]).abs()).fold(0f64, f64::max);
        assert!(err <= (REL_TOL * scale) as f64, "{what}: row {row} head {head} is {err:e} from the f64 oracle (bar {:e})", REL_TOL * scale);
    }
}

const CASES: &[(u32, u32, u32)] = &[(256, 0, 4096), (77, 1000, 4096), (200, 300, 128), (64, 31, 16), (33, 5, 16)];

#[test]
fn the_head_dim_128_kernel_agrees_with_the_portable_kernel_and_the_oracle_on_an_fp32_cache() {
    let Some(gpu) = gpu() else { return };
    for (i, &(n, start, bs)) in CASES.iter().enumerate() {
        let what = format!("fp32 case {i} (n={n} start={start} block={bs})");
        let p = Problem::new(n, start, bs, 0xBEEF + i as u64);
        let b = Buffers::upload(&gpu, &p, gpu.storage_init("k", &p.k), gpu.storage_init("v", &p.v));
        let blocks = N_HEADS * n.div_ceil(64);
        let kind = gpu.kernel_index("paged_flash_prefill").expect("registered");
        let want = b.run(&gpu, gpu.dispatch(kind, &b.six(), &p.params(), gpu_core::Dispatch::Workgroups(blocks)), &p);
        let step = gpu_core::provider::cuda::paged_flash_prefill_step(&gpu, HD as u32, &b.six(), &p.params(), blocks).expect("head_dim 128 takes the tensor-core kernel");
        let got = b.run(&gpu, step, &p);
        let scale = rms(&want);
        let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("cuda_flash_prefill_hd128: {what}: max|tc - portable| = {maxdiff:e} ({:.2e} of rms)", maxdiff / scale);
        assert!(maxdiff <= REL_TOL * scale, "{what}: differs from the portable kernel by {maxdiff:e}");
        check_against_oracle(&got, &p, &p.k, &p.v, &what);
    }
}

#[test]
fn the_head_dim_128_kernel_reads_an_int8_cache() {
    let Some(gpu) = gpu() else { return };
    for (i, &(n, start, bs)) in CASES.iter().enumerate() {
        let what = format!("int8 case {i} (n={n} start={start} block={bs})");
        let p = Problem::new(n, start, bs, 0xCAFE + i as u64);
        let rows = p.k.len() / (N_KV as usize * HD);
        // One scale per (token, kv head) row; the oracle reads the dequantised values.
        let quantise = |plane: &[f32]| -> (Vec<u32>, Vec<f32>, Vec<f32>) {
            let (mut codes, mut scales, mut deq) = (vec![0i8; plane.len()], vec![0f32; rows * N_KV as usize], vec![0f32; plane.len()]);
            for r in 0..rows * N_KV as usize {
                let row = &plane[r * HD..(r + 1) * HD];
                let s = row.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
                scales[r] = s;
                for d in 0..HD {
                    let c = (row[d] / s).round().clamp(-127.0, 127.0);
                    codes[r * HD + d] = c as i8;
                    deq[r * HD + d] = c * s;
                }
            }
            let words = codes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (j, &x)| w | ((x as u8 as u32) << (8 * j)))).collect();
            (words, scales, deq)
        };
        let (kw, ks, kd) = quantise(&p.k);
        let (vw, vs, vd) = quantise(&p.v);
        let words = |w: &[u32]| {
            let b = gpu.storage(w.len() as u64);
            gpu.write(&b, w);
            b
        };
        let b = Buffers::upload(&gpu, &p, words(&kw), words(&vw));
        let (kscale, vscale) = (gpu.storage_init("ks", &ks), gpu.storage_init("vs", &vs));
        let blocks = N_HEADS * n.div_ceil(64);
        let s = b.six();
        let step = gpu_core::provider::cuda::paged_flash_prefill_kv8_step(&gpu, HD as u32, &[s[0], s[1], s[2], s[3], s[4], s[5], &kscale, &vscale], &p.params(), blocks).expect("head_dim 128 takes the int8-KV tensor-core kernel");
        let got = b.run(&gpu, step, &p);
        check_against_oracle(&got, &p, &kd, &vd, &what);
    }
}

#[test]
fn a_round_in_the_decode_regime_keeps_the_portable_kernel_at_head_dim_128() {
    let Some(gpu) = gpu() else { return };
    let p = Problem::new(32, 100, 4096, 1);
    let b = Buffers::upload(&gpu, &p, gpu.storage_init("k", &p.k), gpu.storage_init("v", &p.v));
    assert!(gpu_core::provider::cuda::paged_flash_prefill_step(&gpu, HD as u32, &b.six(), &p.params(), N_HEADS).is_none());
}
