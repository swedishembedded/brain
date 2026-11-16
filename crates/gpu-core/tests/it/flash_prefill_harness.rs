// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements long-context inference throughput for its clients. If your team needs expertise in
// proving that a reduced-precision attention kernel is faster AND numerically faithful to the fp32 path it replaces,
// you can procure our services by sending an email to info@swedishembedded.com.
//
// The harness of the paged flash-prefill kernel tests, one head shape per including module: `include!`d by
// `cuda_flash_prefill.rs` (head_dim 256) and `cuda_flash_prefill_hd128.rs` (head_dim 128), each of which defines
// `HD`, `N_HEADS` and `N_KV` first. The portable reference of every shape is `paged_flash_prefill_hd256` (head_dim
// up to 256) and its compact-tier rewrites; the native kernels are named `flash_prefill_f16_hd{HD}*`.

use backend_api::select::Dtype;
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

/// How a KV cache stores its planes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kv {
    F32,
    Bf16,
    Int8,
    Fp8,
}

impl Kv {
    fn portable_name(self) -> &'static str {
        match self {
            Kv::F32 => "paged_flash_prefill_hd256",
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
        Kv::F32 => (vals.iter().map(|v| v.to_bits()).collect(), Vec::new(), vals.to_vec()),
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
/// Back-to-back rounds of a timing report's sustained figure.
const SUSTAINED_REPS: usize = 400;

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
        self.run_steps(gpu, std::slice::from_ref(step), p)
    }

    fn run_steps(&self, gpu: &Gpu, steps: &[gpu_core::Step], p: &Problem) -> Vec<f32> {
        gpu.submit(&[&self.ctx], steps);
        gpu.read(&self.ctx, (p.c.n * N_HEADS * HD) as usize)
    }

    /// Device ms of one dispatch of `step`: the mean of three back-to-back dispatches.
    fn trial_ms(&self, gpu: &Gpu, step: &gpu_core::Step, name_part: &str) -> f64 {
        self.trial_steps_ms(gpu, std::slice::from_ref(step), name_part)
    }

    fn trial_steps_ms(&self, gpu: &Gpu, steps: &[gpu_core::Step], name_part: &str) -> f64 {
        gpu.reset_kernel_times();
        for _ in 0..3 {
            gpu.submit(&[], steps);
        }
        gpu.poll_wait();
        let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().filter(|(n, _, _)| n.contains(name_part)).map(|(_, ms, _)| ms).sum();
        ms / 3.0
    }

    /// Mean device ms of `reps` back-to-back submissions of `steps`: the rate the card holds once its clocks have settled
    /// under the load, which the best of a few short trials (above) does not show.
    fn sustained_steps_ms(&self, gpu: &Gpu, steps: &[gpu_core::Step], name_part: &str, reps: usize) -> f64 {
        gpu.reset_kernel_times();
        for _ in 0..reps {
            gpu.submit(&[], steps);
        }
        gpu.poll_wait();
        let ms: f64 = gpu.kernel_times().unwrap_or_default().iter().filter(|(n, _, _)| n.contains(name_part)).map(|(_, ms, _)| ms).sum();
        ms / reps as f64
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
                t_tc = t_tc.min(self.trial_ms(gpu, native, &format!("flash_prefill_f16_hd{HD}")));
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
        brain_testutil::skip_unavailable(&format!("cuda_flash_prefill hd{HD}: no CUDA device selected"));
        return None;
    }
    let cc = gpu.caps().arch.compute_capability.expect("the CUDA backend reports its capability");
    if kernels_cuda::get(&format!("flash_prefill_f16_hd{HD}")).is_none_or(|k| k.min_cc > cc) {
        brain_testutil::skip_unavailable(&format!("cuda_flash_prefill hd{HD}: compute capability {}.{} is below the kernel's floor", cc.0, cc.1));
        return None;
    }
    Some(gpu)
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt() as f32
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
        let scales = |s: &[f32]| matches!(kv, Kv::Int8 | Kv::Fp8).then(|| gpu.storage_init("scales", s));
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

    /// The warpgroup kernel's steps (attention and the merge of its key splits), for a compact tier that has one.
    fn native_wg(&self, gpu: &Gpu, p: &Problem) -> Option<Vec<gpu_core::Step>> {
        use gpu_core::provider::cuda::{Scaled8, WgKv};
        let kv = match self.kv {
            Kv::Int8 => WgKv::Scaled(Scaled8::Int8),
            Kv::Fp8 => WgKv::Scaled(Scaled8::Fp8E4m3),
            Kv::F32 => WgKv::F32,
            Kv::Bf16 => WgKv::Bf16,
        };
        gpu_core::provider::cuda::paged_flash_prefill_wg_steps(gpu, HD, &self.bufs(), &Device::params(p), p.c.start + p.c.n, kv, None)
    }

    fn native(&self, gpu: &Gpu, p: &Problem) -> Option<gpu_core::Step> {
        let blocks = N_HEADS * p.c.n.div_ceil(64);
        let b = self.bufs();
        match self.kv {
            Kv::F32 => gpu_core::provider::cuda::paged_flash_prefill_step(gpu, HD, &[b[0], b[1], b[2], b[3], b[4], b[5]], &Device::params(p), blocks),
            Kv::Bf16 => gpu_core::provider::cuda::paged_flash_prefill_bf16_step(gpu, HD, &[b[0], b[1], b[2], b[3], b[4], b[5]], &Device::params(p), blocks),
            Kv::Int8 => gpu_core::provider::cuda::paged_flash_prefill_kv8_step(gpu, HD, &[b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]], &Device::params(p), blocks),
            Kv::Fp8 => gpu_core::provider::cuda::paged_flash_prefill_kvfp8_step(gpu, HD, &[b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]], &Device::params(p), blocks),
        }
    }
}

/// The rates of the forms that serve `kv` at this head shape over the `(rows, cached keys)` rounds in `cases`: the
/// per-call kernel (`base`: version 2 at head_dim 256, the original kernel otherwise), at head_dim 256 over an 8-bit cache
/// the warpgroup form that widens in every block (`wg`), and the staged warpgroup form, which widens the round's keys once
/// first (its time is in the rate). Best of a few trials and sustained over many back-to-back rounds, causal TFLOPS.
fn prefill_rates(kv: Kv, cases: &[(u32, u32)]) {
    let Some(gpu) = device() else { return };
    gpu_core::provider::cuda::set_flash_prefill_v2(true);
    let mut forms: Vec<(&str, bool, u32)> = vec![("base", false, u32::MAX), ("staged", true, 1)];
    if HD == 256 && matches!(kv, Kv::Int8 | Kv::Fp8) {
        forms.insert(1, ("wg", true, u32::MAX));
    }
    // Every kernel of every form (the widen pass and the merge of the key splits included) carries this prefix.
    let name = format!("flash_prefill_f16_hd{HD}");
    for &(n, start) in cases {
        let p = Problem::new(Case { n, start, block_size: 4096, scatter: false }, 23);
        let (seen, stored) = p.as_stored(kv);
        let d = TierDevice::upload(&gpu, &seen, kv, &stored);
        let flops = 4.0 * f64::from(n) * (f64::from(start) + f64::from(n) / 2.0) * f64::from(HD) * f64::from(N_HEADS);
        assert!(gpu.set_kernel_timing(true));
        for &(label, wg, staged_min_rows) in &forms {
            gpu_core::provider::cuda::set_flash_prefill_wg(wg);
            gpu_core::provider::cuda::set_flash_prefill_staged_min_rows(staged_min_rows);
            let steps = if wg { d.native_wg(&gpu, &seen).expect("warpgroup step") } else { vec![d.native(&gpu, &seen).expect("native step")] };
            let _ = d.base.run_steps(&gpu, &steps, &seen);
            let ms = (0..TRIALS).map(|_| d.base.trial_steps_ms(&gpu, &steps, &name)).fold(f64::MAX, f64::min);
            let held = d.base.sustained_steps_ms(&gpu, &steps, &name, SUSTAINED_REPS);
            eprintln!("cuda_flash_prefill hd{HD}: {kv:?} {label} n={n} ctx={}: {ms:.3} ms ({:.0} TFLOPS causal), sustained over {SUSTAINED_REPS} rounds {held:.3} ms ({:.0} TFLOPS)", start + n, flops / ms / 1e9, flops / held / 1e9);
        }
        gpu.set_kernel_timing(false);
    }
    gpu_core::provider::cuda::set_flash_prefill_wg(true);
    gpu_core::provider::cuda::set_flash_prefill_staged_min_rows(gpu_core::provider::cuda::FLASH_PREFILL_STAGED_MIN_ROWS);
}

/// Set the row count from which a warpgroup round this thread builds is staged (every round of a narrower head than 256
/// is staged whatever the setting).
fn set_staged_min_rows(rows: u32) {
    gpu_core::provider::cuda::set_flash_prefill_staged_min_rows(rows);
}

/// The cases a warpgroup prefill test crosses: the shapes its structure distinguishes - one 128-row block and several, a
/// ragged tail, rows with no key to attend in a tile, scattered and tiny cache blocks, key ranges long enough that a block's
/// keys are split over several blocks and merged, and (from 2048 rows) a round that runs unsplit and has no merge step.
const WARPGROUP_CASES: [Case; 7] = [
    Case { n: 256, start: 0, block_size: 4096, scatter: false },
    Case { n: 77, start: 1000, block_size: 4096, scatter: false },
    Case { n: 200, start: 300, block_size: 128, scatter: true },
    Case { n: 64, start: 31, block_size: 16, scatter: true },
    Case { n: 512, start: 7680, block_size: 4096, scatter: false },
    Case { n: 129, start: 5000, block_size: 64, scatter: true },
    Case { n: 2176, start: 1500, block_size: 512, scatter: true },
];

/// The warpgroup prefill over each `(staged, tier)` of `tiers` (staged: the round widens its keys once into fp16 records,
/// whatever its length; unstaged: every block widens them) agrees with the portable kernel of its tier and with the f64
/// oracle over the values the cache holds, and writes every row (the output starts as NaN).
fn warpgroup_agrees(gpu: &Gpu, tiers: &[(bool, Kv)], cases: &[Case]) {
    for &(staged, kv) in tiers {
        set_staged_min_rows(if staged { 1 } else { u32::MAX });
        for (i, c) in cases.iter().enumerate() {
            let desc = format!("hd{HD} {kv:?} staged={staged} case {i} (n={} start={} block={} scatter={})", c.n, c.start, c.block_size, c.scatter);
            let p = Problem::new(Case { ..*c }, 0xA11CE + i as u64);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(gpu, &seen, kv, &stored);
            let want = d.base.run(gpu, &d.portable(gpu, &seen), &seen);
            let steps = d.native_wg(gpu, &seen).unwrap_or_else(|| panic!("{desc}: the warpgroup kernel must take the round on a Hopper part"));
            gpu.write(&d.base.ctx, &vec![0x7fc0_0000u32; (seen.c.n * N_HEADS * HD) as usize]);
            let got = d.base.run_steps(gpu, &steps, &seen);

            let scale = rms(&want);
            let maxdiff = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(got.iter().all(|x| x.is_finite()), "{desc}: non-finite output (a row was left unwritten)");
            eprintln!("cuda_flash_prefill: warpgroup {desc}: max|wg - portable| = {maxdiff:e} ({:.2e} of rms {scale:e})", maxdiff / scale);
            assert!(maxdiff <= REL_TOL * scale, "{desc}: differs from the portable kernel by {maxdiff:e} (bar {:e})", REL_TOL * scale);
            let mut rng = Rng(177 + i as u64);
            for _ in 0..ORACLE_SAMPLES {
                let (row, head) = ((rng.next() % seen.c.n as u64) as usize, (rng.next() % N_HEADS as u64) as usize);
                let truth = seen.oracle(row, head);
                let base = (row * N_HEADS as usize + head) * HD as usize;
                let err = (0..HD as usize).map(|e| (got[base + e] as f64 - truth[e]).abs()).fold(0f64, f64::max);
                assert!(err <= (REL_TOL * scale) as f64, "{desc}: row {row} head {head} is {err:e} from the f64 oracle");
            }
        }
    }
    set_staged_min_rows(gpu_core::provider::cuda::FLASH_PREFILL_STAGED_MIN_ROWS);
}

/// The staged warpgroup rounds of each tier submitted back to back many times without a wait between them (knowledge 410:
/// a TMA fault showed only under back-to-back launches) give the output of the first round, bit for bit, every time.
fn staged_rounds_repeat(gpu: &Gpu, tiers: &[Kv]) {
    set_staged_min_rows(1);
    for &kv in tiers {
        for (i, c) in [Case { n: 2176, start: 1500, block_size: 512, scatter: true }, Case { n: 200, start: 300, block_size: 128, scatter: true }].into_iter().enumerate() {
            let p = Problem::new(c, 0x5EED + i as u64);
            let (seen, stored) = p.as_stored(kv);
            let d = TierDevice::upload(gpu, &seen, kv, &stored);
            let steps = d.native_wg(gpu, &seen).expect("the warpgroup kernel must take the round on a Hopper part");
            let first = d.base.run_steps(gpu, &steps, &seen);
            for round in 0..20 {
                for _ in 0..50 {
                    gpu.submit(&[], &steps);
                }
                assert_eq!(d.base.run_steps(gpu, &steps, &seen), first, "hd{HD} {kv:?} case {i}: round {round} changed the output");
            }
            eprintln!("staged_back_to_back_stress: hd{HD} {kv:?} case {i}: 1000 rounds identical");
        }
    }
    set_staged_min_rows(gpu_core::provider::cuda::FLASH_PREFILL_STAGED_MIN_ROWS);
}

/// Relative L2 distance of the staged warpgroup form's output from `want` (the portable kernel's) on `seen`: the bar the
/// large-score-magnitude tests hold it to.
fn staged_rel_l2(gpu: &Gpu, d: &TierDevice, seen: &Problem, want: &[f32]) -> f64 {
    set_staged_min_rows(1);
    let steps = d.native_wg(gpu, seen).expect("the warpgroup kernel must take the round on a Hopper part");
    set_staged_min_rows(gpu_core::provider::cuda::FLASH_PREFILL_STAGED_MIN_ROWS);
    let got = d.base.run_steps(gpu, &steps, seen);
    let norm = want.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
    (want.iter().zip(&got).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / norm).sqrt()
}
