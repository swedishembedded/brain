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
//! The same kernel serves a bf16 and an int8 KV cache (planes widened while they are
//! staged); those tiers are checked against their own portable kernel and against the
//! f64 oracle over the values the cache actually holds.
//!
//! Skipped without a device, or below the kernel's capability floor.

use backend_api::select::{self, Dtype};
use gpu_core::Gpu;

/// The fp32 portable kernel and its bf16 and int8-KV rewrites: what a model that
/// offers the compact tiers registers (`model::kv_tier::kernel_list`).
fn kernel_list() -> &'static [(&'static str, &'static str)] {
    use kernels::template::{dtype_variant, fp8_kv_variant, int8_kv_variant};
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let src = kernels::PAGED_FLASH_PREFILL_HD256;
        let bf16_k = dtype_variant("paged_flash_prefill_hd256", src, "pool_k", Dtype::BF16).expect("pool_k is templatable");
        let bf16 = dtype_variant(bf16_k.0, bf16_k.1, "pool_v", Dtype::BF16).expect("pool_v is templatable");
        let i8_k = int8_kv_variant("paged_flash_prefill_hd256", src, "pool_k", "k_scales", "p.head_dim").expect("pool_k is templatable");
        let i8 = int8_kv_variant(i8_k.0, i8_k.1, "pool_v", "v_scales", "p.head_dim").expect("pool_v is templatable");
        let fp8_k = fp8_kv_variant("paged_flash_prefill_hd256", src, "pool_k", "k_scales", "p.head_dim").expect("pool_k is templatable");
        let fp8 = fp8_kv_variant(fp8_k.0, fp8_k.1, "pool_v", "v_scales", "p.head_dim").expect("pool_v is templatable");
        vec![("paged_flash_prefill_hd256", src), bf16, i8, fp8]
    })
}

/// How a compact KV cache stores its planes (the fp32 cache is the rest of this file).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kv {
    Bf16,
    Int8,
    Fp8,
}

impl Kv {
    fn portable_name(self) -> &'static str {
        match self {
            Kv::Bf16 => "paged_flash_prefill_hd256#pool_k=bf16#pool_v=bf16",
            Kv::Int8 => "paged_flash_prefill_hd256#pool_k=kv8#pool_v=kv8",
            Kv::Fp8 => "paged_flash_prefill_hd256#pool_k=kvfp8#pool_v=kvfp8",
        }
    }
}

/// The value of E4M3 magnitude code `c` (`0..0x7f`): 4 exponent bits (bias 7), 3 mantissa bits, subnormal at exponent 0.
fn e4m3_value(c: u32) -> f32 {
    let (e, m) = ((c >> 3) as i32, (c & 7) as f32);
    if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// One plane stored in `kv`: its device words, its row scales (int8 only) and the
/// f32 values a reader of the cache sees.
fn store_plane(kv: Kv, vals: &[f32]) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let kv_row = (N_KV * HD) as usize;
    match kv {
        Kv::Bf16 => {
            let bits: Vec<u16> = vals.iter().map(|v| bf16_bits(*v)).collect();
            let words = bits.chunks(2).map(|c| c[0] as u32 | ((c.get(1).copied().unwrap_or(0) as u32) << 16)).collect();
            (words, Vec::new(), bits.iter().map(|b| f32::from_bits((*b as u32) << 16)).collect())
        }
        Kv::Fp8 => {
            // E4M3 values, positive half: index = byte code (0x7f is NaN and never produced).
            let table: Vec<f32> = (0u32..0x7f).map(e4m3_value).collect();
            let (mut bytes, mut scales, mut seen) = (Vec::new(), Vec::new(), Vec::new());
            for row in vals.chunks(kv_row) {
                for head in row.chunks(HD as usize) {
                    let scale = head.iter().fold(0f32, |m, v| m.max(v.abs())) / 448.0;
                    scales.push(scale);
                    for v in head {
                        let y = (v / scale).abs();
                        let code = (0..table.len()).min_by(|&a, &b| (table[a] - y).abs().total_cmp(&(table[b] - y).abs())).unwrap() as u8;
                        let sign = if *v < 0.0 { 0x80u8 } else { 0 };
                        bytes.push(code | sign);
                        let mag = table[code as usize] * scale;
                        seen.push(if sign != 0 { -mag } else { mag });
                    }
                }
            }
            (bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), scales, seen)
        }
        Kv::Int8 => {
            let (mut bytes, mut scales, mut seen) = (Vec::new(), Vec::new(), Vec::new());
            for row in vals.chunks(kv_row) {
                for head in row.chunks(HD as usize) {
                    let scale = head.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
                    scales.push(scale);
                    for v in head {
                        let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
                        bytes.push(q as u8);
                        seen.push(q as f32 * scale);
                    }
                }
            }
            (bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), scales, seen)
        }
    }
}

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

/// The K and V planes as a cache stores them: each plane's device words and row scales.
type StoredPlanes = [(Vec<u32>, Vec<f32>); 2];

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

    /// The same problem with its K and V values as a cache in `kv` holds them.
    fn as_stored(&self, kv: Kv) -> (Problem, StoredPlanes) {
        let (kw, ks, kseen) = store_plane(kv, &self.k_pool);
        let (vw, vs, vseen) = store_plane(kv, &self.v_pool);
        let seen = Problem { c: Case { ..self.c }, q: self.q.clone(), k_pool: kseen, v_pool: vseen, tables: self.tables.clone(), seq_lens: self.seq_lens.clone(), max_bt: self.max_bt, row_of: self.row_of.clone() };
        (seen, [(kw, ks), (vw, vs)])
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

    /// Device ms of one dispatch of `step`: the mean of three back-to-back dispatches.
    fn trial_ms(&self, gpu: &Gpu, step: &gpu_core::Step, name_part: &str) -> f64 {
        gpu.reset_kernel_times();
        for _ in 0..3 {
            gpu.submit(&[], std::slice::from_ref(step));
        }
        gpu.poll_wait();
        let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().filter(|(n, _, _)| n.contains(name_part)).map(|(_, ms, _)| ms).sum();
        ms / 3.0
    }

    /// The best device ms of the portable and the native step, `(portable, native)`, their trials INTERLEAVED.
    ///
    /// On a card shared with a neighbour's job, a burst of its work that lands on the trials of one side only made
    /// that side look slow: timing every portable trial before every native one let a burst during the native half
    /// fail the floor while the same kernels alone clear it with room to spare. Contention only
    /// ever lengthens a trial, so the best of many interleaved trials is the stable estimate for each side, and a
    /// burst now lengthens trials of both. A ratio still under the floor is measured again, up to
    /// [`SPEEDUP_ATTEMPTS`] times, before the gate fails: a kernel that fell back to scalar work stays under it
    /// every time.
    fn best_ms_pair(&self, gpu: &Gpu, portable: &gpu_core::Step, native: &gpu_core::Step, floor: f64) -> (f64, f64) {
        assert!(gpu.set_kernel_timing(true));
        let (mut t_ref, mut t_tc) = (f64::MAX, f64::MAX);
        for _ in 0..SPEEDUP_ATTEMPTS {
            for _ in 0..TRIALS {
                t_ref = t_ref.min(self.trial_ms(gpu, portable, "paged_flash_prefill_hd256"));
                t_tc = t_tc.min(self.trial_ms(gpu, native, "flash_prefill_f16_hd256"));
            }
            if t_ref / t_tc >= floor {
                break;
            }
        }
        gpu.set_kernel_timing(false);
        (t_ref, t_tc)
    }
}

/// Rounds of [`TRIALS`] interleaved trials a speed gate measures before it fails (see `Device::best_ms_pair`).
const SPEEDUP_ATTEMPTS: usize = 3;

fn device() -> Option<Gpu> {
    // `BRAIN_DEVICE` picks the card (`gpu_core::testgpu::dev`); a CPU selection skips.
    let gpu = gpu_core::testgpu::dev(kernel_list());
    if gpu.kind() != "cuda" {
        eprintln!("cuda_flash_prefill: no CUDA device selected - skipping");
        return None;
    }
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
        Case { n: 33, start: 513, block_size: 4096, scatter: false },
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

/// A verify or ragged-tail round is not a prefill: it stays on the fp32 kernel so
/// speculative decoding keeps computing what the token-by-token tape computes.
#[test]
fn a_round_in_the_decode_regime_keeps_the_fp32_kernel() {
    let Some(gpu) = device() else { return };
    for n in [1u32, 8, 32] {
        let p = Problem::new(Case { n, start: 100, block_size: 4096, scatter: false }, 5);
        let d = Device::upload(&gpu, &p);
        assert!(d.native(&gpu, &p).is_none(), "a {n}-row round must not take the fp16 kernel");
    }
    let p = Problem::new(Case { n: 33, start: 100, block_size: 4096, scatter: false }, 5);
    let d = Device::upload(&gpu, &p);
    assert!(d.native(&gpu, &p).is_some(), "33 rows is past the decode regime and takes the kernel");
}

#[test]
fn a_head_width_the_kernel_was_not_written_for_is_declined() {
    let Some(gpu) = device() else { return };
    let p = Problem::new(Case { n: 64, start: 0, block_size: 64, scatter: false }, 3);
    let d = Device::upload(&gpu, &p);
    assert!(gpu_core::provider::cuda::paged_flash_prefill_step(&gpu, 96, &d.bufs(), &Device::params(&p), 24).is_none());
}

#[test]
fn the_tensor_core_flash_prefill_is_materially_faster_than_the_portable_kernel() {
    let Some(gpu) = device() else { return };
    // The deep shape is the gate. The shallow one (a first round, ~40 us of work
    // for the kernel) is printed but not asserted: at that size the event timer
    // and launch overhead are most of what is being measured.
    for (n, start, gate) in [(256u32, 1792u32, true), (256, 0, false)] {
        let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 21);
        let d = Device::upload(&gpu, &p);
        let portable = d.portable(&gpu, &p);
        let native = d.native(&gpu, &p).expect("native step");
        d.run(&gpu, &portable, &p); // compile outside the timed region
        d.run(&gpu, &native, &p);
        let (t_ref, t_tc) = d.best_ms_pair(&gpu, &portable, &native, if gate { SPEEDUP_FLOOR } else { 0.0 });
        // causal attention flops: 4 * sum_i len_i * hd * heads
        let flops: f64 = (0..n).map(|i| (start + i + 1) as f64).sum::<f64>() * 4.0 * HD as f64 * N_HEADS as f64;
        let speedup = t_ref / t_tc;
        eprintln!(
            "cuda_flash_prefill: n={n} start={start}: portable {t_ref:.3} ms ({:.1} TFLOPS), tensor-core {t_tc:.3} ms ({:.1} TFLOPS), {speedup:.1}x",
            flops / (t_ref * 1e-3) / 1e12,
            flops / (t_tc * 1e-3) / 1e12
        );
        assert!(!gate || speedup >= SPEEDUP_FLOOR, "n={n} start={start}: only {speedup:.1}x the portable kernel (floor {SPEEDUP_FLOOR}x)");
    }
}

/// A problem uploaded as a cache in one tier stores it.
struct TierDevice {
    kv: Kv,
    base: Device,
    k_scales: Option<backend_api::DeviceBuffer>,
    v_scales: Option<backend_api::DeviceBuffer>,
}

impl TierDevice {
    /// `seen` holds the values a reader of the stored cache sees (see [`Problem::as_stored`]).
    fn upload(gpu: &Gpu, seen: &Problem, kv: Kv, stored: &[(Vec<u32>, Vec<f32>); 2]) -> TierDevice {
        let mut base = Device::upload(gpu, seen);
        let words = |w: &[u32]| {
            let b = gpu.storage(w.len().max(1) as u64);
            gpu.write(&b, w);
            b
        };
        base.k = words(&stored[0].0);
        base.v = words(&stored[1].0);
        let scales = |s: &[f32]| (kv != Kv::Bf16).then(|| gpu.storage_init("scales", s));
        TierDevice { kv, base, k_scales: scales(&stored[0].1), v_scales: scales(&stored[1].1) }
    }

    fn bufs(&self) -> Vec<&backend_api::DeviceBuffer> {
        let mut b = self.base.bufs().to_vec();
        b.extend(self.k_scales.iter().chain(&self.v_scales));
        b
    }

    fn portable(&self, gpu: &Gpu, p: &Problem) -> gpu_core::Step {
        let kind = gpu.kernel_index(self.kv.portable_name()).expect("registered");
        gpu.dispatch(kind, &self.bufs(), &Device::params(p), gpu_core::Dispatch::Workgroups(N_HEADS * p.c.n.div_ceil(64)))
    }

    fn native(&self, gpu: &Gpu, p: &Problem) -> Option<gpu_core::Step> {
        let blocks = N_HEADS * p.c.n.div_ceil(64);
        let b = self.bufs();
        match self.kv {
            Kv::Bf16 => gpu_core::provider::cuda::paged_flash_prefill_bf16_step(gpu, HD, &[b[0], b[1], b[2], b[3], b[4], b[5]], &Device::params(p), blocks),
            Kv::Int8 => gpu_core::provider::cuda::paged_flash_prefill_kv8_step(gpu, HD, &[b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]], &Device::params(p), blocks),
            Kv::Fp8 => gpu_core::provider::cuda::paged_flash_prefill_kvfp8_step(gpu, HD, &[b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]], &Device::params(p), blocks),
        }
    }
}

/// A compact cache is read in place by the tensor-core kernel: it agrees with the
/// portable kernel of its tier and with an f64 oracle over the values the cache holds.
#[test]
fn the_tensor_core_flash_prefill_reads_a_bf16_an_int8_and_an_fp8_cache() {
    let Some(gpu) = device() else { return };
    let cases = [
        Case { n: 256, start: 0, block_size: 4096, scatter: false },
        Case { n: 77, start: 1000, block_size: 4096, scatter: false },
        Case { n: 200, start: 300, block_size: 128, scatter: true },
        Case { n: 64, start: 31, block_size: 16, scatter: true },
    ];
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        for (i, c) in cases.iter().enumerate() {
            let desc = format!("{kv:?} case {i} (n={} start={} block={} scatter={})", c.n, c.start, c.block_size, c.scatter);
            let p = Problem::new(Case { ..*c }, 0xC0DE + i as u64);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let want = d.base.run(&gpu, &d.portable(&gpu, &seen), &seen);
            let step = d.native(&gpu, &seen).expect("the production helper must take a head_dim-256 request on this device");
            gpu.write(&d.base.ctx, &vec![0x7fc0_0000u32; (seen.c.n * N_HEADS * HD) as usize]);
            let got = d.base.run(&gpu, &step, &seen);

            let scale = rms(&want);
            let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(got.iter().all(|x| x.is_finite()), "{desc}: non-finite output");
            eprintln!("cuda_flash_prefill: {desc}: max|tc - portable| = {maxdiff:e} ({:.2e} of rms {scale:e})", maxdiff / scale);
            assert!(maxdiff <= REL_TOL * scale, "{desc}: differs from the portable kernel by {maxdiff:e} (bar {:e})", REL_TOL * scale);
            let mut rng = Rng(77 + i as u64);
            for _ in 0..ORACLE_SAMPLES {
                let (row, head) = ((rng.next() % seen.c.n as u64) as usize, (rng.next() % N_HEADS as u64) as usize);
                let truth = seen.oracle(row, head);
                let base = (row * N_HEADS as usize + head) * HD as usize;
                let err = (0..HD as usize).map(|e| (got[base + e] as f64 - truth[e]).abs()).fold(0f64, f64::max);
                assert!(err <= (REL_TOL * scale) as f64, "{desc}: row {row} head {head} is {err:e} from the f64 oracle");
            }
        }
    }
}

/// Version 2 of the compact-cache kernel (one warp per 16 rows over the whole head, two blocks per SM) does the
/// arithmetic of the original with a different key tile (so a different running-max schedule and fp16 rounding of P),
/// so the two agree to rounding noise; this holds it to 2e-3 of the output scale (the bar against the portable fp32
/// kernel is 1e-2) on ragged and scattered shapes, and the NaN-filled output proves every row is written.
/// It flips the process-wide version switch; the other tests in this file are valid under either kernel.
#[test]
fn version_2_matches_the_original_kernel_on_a_bf16_an_int8_and_an_fp8_cache() {
    let Some(gpu) = device() else { return };
    let cases = [
        Case { n: 256, start: 0, block_size: 4096, scatter: false },
        Case { n: 77, start: 1000, block_size: 4096, scatter: false },
        Case { n: 200, start: 300, block_size: 128, scatter: true },
        Case { n: 33, start: 513, block_size: 16, scatter: true },
    ];
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        for (i, c) in cases.iter().enumerate() {
            let p = Problem::new(Case { ..*c }, 0xFA57 + i as u64);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let mut out = Vec::new();
            for v2 in [false, true] {
                gpu_core::provider::cuda::set_flash_prefill_v2(v2);
                let step = d.native(&gpu, &seen).expect("the production helper must take a head_dim-256 request on this device");
                gpu.write(&d.base.ctx, &vec![0x7fc0_0000u32; (seen.c.n * N_HEADS * HD) as usize]);
                out.push(d.base.run(&gpu, &step, &seen));
            }
            gpu_core::provider::cuda::set_flash_prefill_v2(true);
            assert!(out[1].iter().all(|x| x.is_finite()), "{kv:?} case {i}: version 2 left a row unwritten");
            let maxdiff = out[0].iter().zip(&out[1]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let bar = 2e-3 * rms(&out[0]);
            assert!(maxdiff <= bar, "{kv:?} case {i} (n={} start={}): version 2 differs from version 1 by {maxdiff:e} (bar {bar:e})", c.n, c.start);
        }
    }
}

/// The compact-tier kernels keep the decode-regime rule of the fp32 one.
#[test]
fn a_compact_cache_in_the_decode_regime_keeps_its_portable_kernel() {
    let Some(gpu) = device() else { return };
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        let p = Problem::new(Case { n: 8, start: 100, block_size: 4096, scatter: false }, 5);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        assert!(d.native(&gpu, &seen).is_none(), "{kv:?}: an 8-row round must not take the fp16 kernel");
    }
}

/// The reason the kernel exists: an int8 cache prefill on the portable kernel is the
/// slowest part of a served prefill.
#[test]
fn an_8_bit_cache_prefill_is_materially_faster_than_the_portable_kernel() {
    let Some(gpu) = device() else { return };
    for kv in [Kv::Int8, Kv::Fp8] {
        let (n, start) = (256u32, 1792u32);
        let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 21);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        let portable = d.portable(&gpu, &seen);
        let native = d.native(&gpu, &seen).expect("native step");
        d.base.run(&gpu, &portable, &seen);
        d.base.run(&gpu, &native, &seen);
        let (t_ref, t_tc) = d.base.best_ms_pair(&gpu, &portable, &native, SPEEDUP_FLOOR);
        let speedup = t_ref / t_tc;
        eprintln!("cuda_flash_prefill: {kv:?} cache n={n} start={start}: portable {t_ref:.3} ms, tensor-core {t_tc:.3} ms, {speedup:.1}x");
        assert!(speedup >= SPEEDUP_FLOOR, "only {speedup:.1}x the portable kernel (floor {SPEEDUP_FLOOR}x)");
    }
}

/// The serving shape of the 27B's agentic prefill (a 512-row chunk behind 3584 cached tokens, int8 cache): 192
/// blocks, i.e. more than one wave on 132 SMs, which the smaller cases above do not exercise. Reports the time of
/// the original kernel and of version 2 and how far their outputs differ; it asserts nothing about speed. It flips the
/// process-wide version switch, so run it alone (`--test-threads 1`, as `runs/b27/flash-bench.sh` does).
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_int8_cache_prefill_rates_at_the_serving_chunk() {
    let Some(gpu) = device() else { return };
    let (n, start) = (512u32, 3584u32);
    let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 23);
    let (seen, stored) = p.as_stored(Kv::Int8);
    let d = TierDevice::upload(&gpu, &seen, Kv::Int8, &stored);
    let mut out = Vec::new();
    for (v2, name) in [(false, "flash_prefill_f16_hd256_kv8"), (true, "flash_prefill_f16_hd256_v2_kv8")] {
        gpu_core::provider::cuda::set_flash_prefill_v2(v2);
        let native = d.native(&gpu, &seen).expect("native step");
        let ctx = d.base.run(&gpu, &native, &seen);
        assert!(gpu.set_kernel_timing(true));
        let ms = (0..TRIALS).map(|_| d.base.trial_ms(&gpu, &native, name)).fold(f64::MAX, f64::min);
        gpu.set_kernel_timing(false);
        eprintln!("cuda_flash_prefill: serving chunk int8 n={n} start={start}: {name} {ms:.3} ms");
        out.push(ctx);
    }
    gpu_core::provider::cuda::set_flash_prefill_v2(true);
    let diff = out[0].iter().zip(&out[1]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("cuda_flash_prefill: version 1 vs version 2 max |diff| {diff:.2e} on outputs of rms {:.2e}", rms(&out[0]));
}

/// Scores are exponentiated, so an error in `q . k` becomes a relative error in the attention weight: what matters is
/// the absolute error of the score, which grows with the operands' magnitude, and the unit-scale cases above hide that.
/// A real model's scaled scores reach several tens, so this scales queries and keys until the scores have a standard
/// deviation of `mag * mag` (up to 16) and holds both kernel versions to the portable fp32 kernel by relative L2 over
/// the whole output. Measured: 3e-4 at unit scale growing to 1.3e-3 at mag = 4; the bar is 3e-3. (Through the 27B's logits the
/// same kernels differ from fp32 by 0.1 - that is the W8A8 model's own sensitivity, `qwen35_gguf_logit_check` reports it
/// next to its floor - so the kernel is gated here, where the cause of an error can be told.)
#[test]
fn both_flash_versions_stay_close_to_fp32_at_large_score_magnitudes() {
    let Some(gpu) = device() else { return };
    for kv in [Kv::Bf16, Kv::Int8, Kv::Fp8] {
        for mag in [1.0f32, 2.0, 4.0] {
            let mut p = Problem::new(Case { n: 128, start: 1920, block_size: 4096, scatter: false }, 5);
            p.q.iter_mut().for_each(|x| *x *= mag);
            p.k_pool.iter_mut().for_each(|x| *x *= mag);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(&gpu, &seen, kv, &stored);
            let want = d.base.run(&gpu, &d.portable(&gpu, &seen), &seen);
            let norm = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
            for v2 in [false, true] {
                gpu_core::provider::cuda::set_flash_prefill_v2(v2);
                let step = d.native(&gpu, &seen).expect("the production helper must take a head_dim-256 request on this device");
                let got = d.base.run(&gpu, &step, &seen);
                gpu_core::provider::cuda::set_flash_prefill_v2(true);
                let rel = (want.iter().zip(&got).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / norm).sqrt();
                eprintln!("cuda_flash_prefill: {kv:?} score scale {}x v2={v2}: rel-L2 to the portable kernel {rel:.2e}", mag * mag);
                assert!(rel <= 3e-3, "{kv:?} at score scale {}x, v2={v2}: rel-L2 {rel:.2e} from the fp32 kernel (bar 3e-3)", mag * mag);
            }
        }
    }
}

/// The int8-cache prefill kernel at the serving chunk (512 rows) at growing context, for `ncu -k regex:flash_prefill -c 1`:
/// `cargo test --release -p brain-gpu-core --test cuda_flash_prefill -- --ignored --nocapture contexts`.
#[test]
#[ignore = "a timing report for kernel work, not a spec"]
fn the_int8_cache_prefill_rates_at_growing_contexts() {
    let Some(gpu) = device() else { return };
    gpu_core::provider::cuda::set_flash_prefill_v2(true);
    for (n, start) in [(512u32, 512u32), (512, 3584), (512, 7680), (512, 15872), (512, 31232)] {
        let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 23);
        let (seen, stored) = p.as_stored(Kv::Int8);
        let d = TierDevice::upload(&gpu, &seen, Kv::Int8, &stored);
        let native = d.native(&gpu, &seen).expect("native step");
        let _ = d.base.run(&gpu, &native, &seen);
        assert!(gpu.set_kernel_timing(true));
        let ms = (0..TRIALS).map(|_| d.base.trial_ms(&gpu, &native, "flash_prefill_f16_hd256_v2_kv8")).fold(f64::MAX, f64::min);
        gpu.set_kernel_timing(false);
        let flops = 4.0 * f64::from(n) * (f64::from(start) + f64::from(n) / 2.0) * 256.0 * 24.0;
        eprintln!("cuda_flash_prefill: int8 n={n} ctx={}: {ms:.3} ms ({:.0} TFLOPS causal)", start + n, flops / ms / 1e9);
    }
}
