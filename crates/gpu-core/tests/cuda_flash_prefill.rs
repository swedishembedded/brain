// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fp16 tensor-core paged flash-prefill kernel must agree with the portable
//! fp32 `paged_flash_prefill_hd256` it replaces on real Qwen3.8-27B attention
//! shapes (24 query heads, 4 KV heads, head_dim 256, flat and paged caches,
//! ragged chunk lengths and non-zero starts), and must be much faster.
//!
//! Swedish Embedded AB implements long-context inference throughput for its
//! clients. If your team needs expertise in proving that a reduced-precision
//! attention kernel is faster AND numerically faithful to the fp32 path it
//! replaces, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What "agrees" means here
//!
//! The portable kernel is fp32 end to end; the tensor-core kernel rounds Q, K,
//! V and the softmax weights to fp16 (fp32 accumulation). The two therefore
//! differ by fp16 rounding, ~5e-4 relative per operand, averaged down by the
//! sums that make up an attention output. The bar is stated relative to the
//! output's own RMS ([`REL_TOL`]), so it keeps meaning at another activation
//! scale, and an f64 host oracle over a sample of rows checks both kernels
//! against the truth so a bug the two would share cannot hide.
//!
//! Skipped without a device, or below the kernel's capability floor.

use backend_api::select::{self, Dtype};
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256)];

const N_HEADS: u32 = 24;
const N_KV: u32 = 4;
const HD: u32 = 256;

/// Largest `|fp16 tensor-core - fp32 portable|` allowed, as a fraction of the
/// RMS of the portable output. Measured values (printed by the run) sit around
/// 3-6e-3 of the RMS on unit-scale inputs (the worst single element of millions); the bar leaves margin over them
/// without being loose enough to hide a wrong mask, row or head.
const REL_TOL: f32 = 1.2e-2;

/// Outputs checked against the f64 oracle: `(row, head)` pairs, every element.
const ORACLE_SAMPLES: usize = 24;

/// How much faster than the portable kernel it must be by device time, at the
/// real prefill shape. Far under what is measured on an idle part, so a busy
/// shared card cannot fail it, but high enough that a kernel which fell back to
/// scalar work would.
// perf-number: the asserted floor of this gate; the achieved ratio is printed.
const SPEEDUP_FLOOR: f64 = 4.0;
const TRIALS: usize = 10;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Roughly unit-variance (sum of four uniforms, centred and scaled).
    fn gauss(&mut self) -> f32 {
        let u = |r: &mut Rng| (r.next() >> 40) as f32 / (1u32 << 24) as f32;
        (u(self) + u(self) + u(self) + u(self) - 2.0) * 1.732
    }
}

struct Case {
    /// Chunk length (query rows) and the number of cached rows before it.
    n: u32,
    start: u32,
    /// Rows per physical block, and whether the physical blocks are scattered.
    block_size: u32,
    scatter: bool,
}

struct Problem {
    c: Case,
    q: Vec<f32>,
    k_pool: Vec<f32>,
    v_pool: Vec<f32>,
    tables: Vec<u32>,
    seq_lens: Vec<u32>,
    max_bt: u32,
    /// pool row of sequence position `j`.
    row_of: Vec<u32>,
}

impl Problem {
    fn new(c: Case, seed: u64) -> Problem {
        let mut r = Rng(seed);
        let total = c.start + c.n;
        let max_bt = total.div_ceil(c.block_size);
        let num_blocks = max_bt + 2;
        // Logical block b -> physical block; scattered = reversed, offset by one.
        let phys: Vec<u32> = (0..max_bt).map(|b| if c.scatter { num_blocks - 1 - b } else { b }).collect();
        let kv_row = (N_KV * HD) as usize;
        let pool_rows = (num_blocks * c.block_size) as usize;
        let k_pool: Vec<f32> = (0..pool_rows * kv_row).map(|_| r.gauss()).collect();
        let v_pool: Vec<f32> = (0..pool_rows * kv_row).map(|_| r.gauss()).collect();
        let q: Vec<f32> = (0..(c.n * N_HEADS * HD) as usize).map(|_| r.gauss()).collect();
        let tables: Vec<u32> = (0..c.n).flat_map(|_| phys.clone()).collect();
        let seq_lens: Vec<u32> = (0..c.n).map(|i| c.start + i + 1).collect();
        let row_of = (0..total).map(|j| phys[(j / c.block_size) as usize] * c.block_size + j % c.block_size).collect();
        Problem { c, q, k_pool, v_pool, tables, seq_lens, max_bt, row_of }
    }

    /// f64 attention for one (row, head): the oracle.
    fn oracle(&self, i: usize, h: usize) -> Vec<f64> {
        let kv_row = (N_KV * HD) as usize;
        let hkv = h / (N_HEADS / N_KV) as usize;
        let len = self.seq_lens[i] as usize;
        let qv = &self.q[(i * N_HEADS as usize + h) * HD as usize..][..HD as usize];
        let scale = 1.0 / (HD as f64).sqrt();
        let scores: Vec<f64> = (0..len)
            .map(|j| {
                let kr = &self.k_pool[self.row_of[j] as usize * kv_row + hkv * HD as usize..][..HD as usize];
                qv.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale
            })
            .collect();
        let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
        let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let z: f64 = w.iter().sum();
        let mut out = vec![0f64; HD as usize];
        for (j, wj) in w.iter().enumerate() {
            let vr = &self.v_pool[self.row_of[j] as usize * kv_row + hkv * HD as usize..][..HD as usize];
            for (o, v) in out.iter_mut().zip(vr) {
                *o += wj / z * *v as f64;
            }
        }
        out
    }
}

struct Device {
    q: backend_api::DeviceBuffer,
    k: backend_api::DeviceBuffer,
    v: backend_api::DeviceBuffer,
    tables: backend_api::DeviceBuffer,
    seq_lens: backend_api::DeviceBuffer,
    ctx: backend_api::DeviceBuffer,
}

impl Device {
    fn upload(gpu: &Gpu, p: &Problem) -> Device {
        let words = |w: &[u32]| {
            let b = gpu.storage(w.len().max(1) as u64);
            gpu.write(&b, w);
            b
        };
        Device {
            q: gpu.storage_init("q", &p.q),
            k: gpu.storage_init("k", &p.k_pool),
            v: gpu.storage_init("v", &p.v_pool),
            tables: words(&p.tables),
            seq_lens: words(&p.seq_lens),
            ctx: gpu.storage((p.c.n * N_HEADS * HD) as u64),
        }
    }

    fn bufs(&self) -> [&backend_api::DeviceBuffer; 6] {
        [&self.q, &self.k, &self.v, &self.tables, &self.seq_lens, &self.ctx]
    }

    fn params(p: &Problem) -> [u32; 7] {
        [p.c.n, N_HEADS, N_KV, HD, N_HEADS / N_KV, p.c.block_size, p.max_bt]
    }

    fn portable(&self, gpu: &Gpu, p: &Problem) -> gpu_core::Step {
        let kind = gpu.kernel_index("paged_flash_prefill_hd256").expect("registered");
        gpu.dispatch(kind, &self.bufs(), &Self::params(p), gpu_core::Dispatch::Workgroups(N_HEADS * p.c.n.div_ceil(64)))
    }

    fn native(&self, gpu: &Gpu, p: &Problem) -> Option<gpu_core::Step> {
        gpu_core::provider::cuda::paged_flash_prefill_step(gpu, HD, &self.bufs(), &Self::params(p), N_HEADS * p.c.n.div_ceil(64))
    }

    fn run(&self, gpu: &Gpu, step: &gpu_core::Step, p: &Problem) -> Vec<f32> {
        gpu.submit(&[&self.ctx], std::slice::from_ref(step));
        gpu.read(&self.ctx, (p.c.n * N_HEADS * HD) as usize)
    }

    fn device_ms(&self, gpu: &Gpu, step: &gpu_core::Step, name_part: &str) -> f64 {
        assert!(gpu.set_kernel_timing(true));
        let mut best = f64::MAX;
        for _ in 0..TRIALS {
            gpu.reset_kernel_times();
            for _ in 0..3 {
                gpu.submit(&[], std::slice::from_ref(step));
            }
            gpu.poll_wait();
            let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().filter(|(n, _, _)| n.contains(name_part)).map(|(_, ms, _)| ms).sum();
            best = best.min(ms / 3.0);
        }
        gpu.set_kernel_timing(false);
        best
    }
}

fn device() -> Option<Gpu> {
    let Ok(gpu) = Gpu::try_new_cuda(KERNELS) else {
        eprintln!("cuda_flash_prefill: no CUDA device on this box - skipping");
        return None;
    };
    let cc = gpu.caps().arch.compute_capability.expect("the CUDA backend reports its capability");
    if kernels_cuda::find(select::Op::PagedAttentionFused, Dtype::F32, cc).is_none() {
        eprintln!("cuda_flash_prefill: compute capability {}.{} is below the kernel's floor - skipping", cc.0, cc.1);
        return None;
    }
    Some(gpu)
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt() as f32
}

#[test]
fn the_tensor_core_flash_prefill_agrees_with_the_portable_kernel_and_the_oracle() {
    let Some(gpu) = device() else { return };
    let cases = [
        Case { n: 256, start: 0, block_size: 4096, scatter: false },
        Case { n: 256, start: 256, block_size: 4096, scatter: false },
        Case { n: 77, start: 1000, block_size: 4096, scatter: false },
        Case { n: 1, start: 513, block_size: 4096, scatter: false },
        Case { n: 130, start: 0, block_size: 4096, scatter: false },
        Case { n: 200, start: 300, block_size: 128, scatter: true },
        Case { n: 64, start: 31, block_size: 16, scatter: true },
    ];
    let mut worst = 0f32;
    for (i, c) in cases.into_iter().enumerate() {
        let desc = format!("case {i} (n={} start={} block={} scatter={})", c.n, c.start, c.block_size, c.scatter);
        let p = Problem::new(c, 0xF1A5 + i as u64);
        let d = Device::upload(&gpu, &p);
        let want = d.run(&gpu, &d.portable(&gpu, &p), &p);
        let step = d.native(&gpu, &p).expect("the production helper must take a head_dim-256 request on this device");
        // The output must not depend on what the buffer held before.
        gpu.write(&d.ctx, &vec![0x7fc0_0000u32; (p.c.n * N_HEADS * HD) as usize]);
        let got = d.run(&gpu, &step, &p);

        let scale = rms(&want);
        let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(got.iter().all(|x| x.is_finite()), "{desc}: non-finite output");
        eprintln!("cuda_flash_prefill: {desc}: max|tc - fp32| = {maxdiff:e} ({:.2e} of rms {scale:e})", maxdiff / scale);
        assert!(maxdiff <= REL_TOL * scale, "{desc}: differs from the portable kernel by {maxdiff:e} (bar {:e})", REL_TOL * scale);
        worst = worst.max(maxdiff / scale);

        let mut rng = Rng(9 + i as u64);
        for _ in 0..ORACLE_SAMPLES {
            let (row, head) = ((rng.next() % p.c.n as u64) as usize, (rng.next() % N_HEADS as u64) as usize);
            let truth = p.oracle(row, head);
            let base = (row * N_HEADS as usize + head) * HD as usize;
            let err = (0..HD as usize).map(|d| (got[base + d] as f64 - truth[d]).abs()).fold(0f64, f64::max);
            assert!(err <= (REL_TOL * scale) as f64, "{desc}: row {row} head {head} is {err:e} from the f64 oracle");
        }
    }
    eprintln!("cuda_flash_prefill: worst relative difference {worst:.2e} (bar {REL_TOL:.0e})");
}

#[test]
fn a_head_width_the_kernel_was_not_written_for_is_declined() {
    let Some(gpu) = device() else { return };
    let p = Problem::new(Case { n: 8, start: 0, block_size: 64, scatter: false }, 3);
    let d = Device::upload(&gpu, &p);
    assert!(gpu_core::provider::cuda::paged_flash_prefill_step(&gpu, 128, &d.bufs(), &Device::params(&p), 24).is_none());
}

#[test]
fn the_tensor_core_flash_prefill_is_materially_faster_than_the_portable_kernel() {
    let Some(gpu) = device() else { return };
    for (n, start) in [(256u32, 1792u32), (256, 0)] {
        let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 21);
        let d = Device::upload(&gpu, &p);
        let portable = d.portable(&gpu, &p);
        let native = d.native(&gpu, &p).expect("native step");
        d.run(&gpu, &portable, &p); // compile outside the timed region
        d.run(&gpu, &native, &p);
        let t_ref = d.device_ms(&gpu, &portable, "paged_flash_prefill_hd256");
        let t_tc = d.device_ms(&gpu, &native, "flash_prefill_f16_hd256");
        // causal attention flops: 4 * sum_i len_i * hd * heads
        let flops: f64 = (0..n).map(|i| (start + i + 1) as f64).sum::<f64>() * 4.0 * HD as f64 * N_HEADS as f64;
        let speedup = t_ref / t_tc;
        eprintln!(
            "cuda_flash_prefill: n={n} start={start}: portable {t_ref:.3} ms ({:.1} TFLOPS), tensor-core {t_tc:.3} ms ({:.1} TFLOPS), {speedup:.1}x",
            flops / (t_ref * 1e-3) / 1e12,
            flops / (t_tc * 1e-3) / 1e12
        );
        assert!(speedup >= SPEEDUP_FLOOR, "n={n} start={start}: only {speedup:.1}x the portable kernel (floor {SPEEDUP_FLOOR}x)");
    }
}
