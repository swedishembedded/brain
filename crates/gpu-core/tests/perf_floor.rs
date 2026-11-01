// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The hot ops of serving, timed on the backend the process runs on, one machine-readable `PERF` line each.
//! `scripts/gates/perf-floor-gate.sh` runs this binary once per backend (`BRAIN_BACKEND=cpu|vulkan|cuda`) and
//! compares every line against the floor recorded for that device class.
//!
//! Swedish Embedded AB implements performance regression gates for GPU inference engines for its clients. If your
//! team needs expertise in keeping every backend of an engine at its roofline as the code changes, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! What is timed, and why these:
//!
//! * the portable (WGSL) ops, on every backend that can run them: the FP8 GEMV at decode widths 1 and 8, the
//!   paged flash prefill at a 4096-row round, and the split-key decode attention. The CPU JIT cannot run these
//!   cooperative kernels (its attention and int8 paths are native fast paths of the CPU backend, not WGSL), so
//!   for the CPU the gate times the f32 GEMM it does run; every backend times that GEMM.
//! * on CUDA the native candidates the autotuner may pick, by the same shapes and the best of what serves each:
//!   FP8 at decode widths 1/16/64 and at the 4096-row prefill round, the tensor-core flash prefill, the batched
//!   decode attention and the GDN chunk forward.
//!
//! Each op is a rate (higher is better): GB/s for the bandwidth-bound, TFLOPS for the compute-bound. Time is the
//! minimum over repeated back-to-back runs, the stable estimate on a card shared with a neighbour (kernels.md E.0a).
//! Not timed, because no single op exists to time on every backend: the GDN decode recurrence (it is a fused
//! pipeline of small kernels in the model) and bf16/NVFP4 weights.
//!
//! ```text
//! BRAIN_DEVICE=gpu1 BRAIN_BACKEND=cuda cargo test --release -p brain-gpu-core --test perf_floor -- --ignored --nocapture
//! ```

use std::time::Instant;

use gpu_core::{Dispatch, Gpu, NativeVerdict, Step};

const FP8: &str = "matmul_fp8_gemv";

/// Every portable kernel this gate dispatches, registered once on the shared test device.
fn kernel_set() -> &'static [(&'static str, &'static str)] {
    use kernels::template::gemv_variants;
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v: Vec<(&'static str, &'static str)> = vec![
            ("matmul", kernels::MATMUL),
            ("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256),
            ("paged_flash_decode_split", kernels::PAGED_FLASH_DECODE_SPLIT),
            ("paged_flash_decode_combine", kernels::PAGED_FLASH_DECODE_COMBINE),
        ];
        v.extend(gemv_variants().into_iter().filter(|(name, _)| name.starts_with(FP8)));
        v
    })
}

/// A deterministic word stream for filler, whose bytes are valid finite E4M3 codes (no NaN, no infinity).
struct Fill(u64);
impl Fill {
    fn word(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32) & 0x6f6f_6f6f
    }
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }
}

fn words(gpu: &Gpu, w: &[u32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(w.len().max(1) as u64);
    gpu.write(&b, w);
    b
}

/// Minimum seconds for ONE run of `steps`, over `trials` submits of `reps` back-to-back copies each.
fn best_secs(gpu: &Gpu, steps: &[Step], reps: usize, trials: usize) -> f64 {
    let all: Vec<Step> = (0..reps).flat_map(|_| steps.iter().cloned()).collect();
    let mut best = f64::MAX;
    for trial in 0..=trials {
        let t = Instant::now();
        gpu.submit(&[], &all);
        gpu.poll_wait();
        // Trial 0 compiles pipelines and ramps the clock; it is never the estimate.
        if trial > 0 {
            best = best.min(t.elapsed().as_secs_f64() / reps as f64);
        }
    }
    best
}

struct Out(Vec<(String, f64, &'static str)>);
impl Out {
    fn rate(&mut self, op: &str, value: f64, unit: &'static str) {
        println!("PERF {op} {value:.3} {unit}");
        self.0.push((op.to_owned(), value, unit));
    }
}

/// The plain f32 GEMM at a decode row and a small prefill round: the one op every backend runs, the CPU through its
/// native GEMM and the GPUs through the WGSL kernel (the CPU JIT cannot run the cooperative kernels below).
fn matmul_f32(gpu: &Gpu, out: &mut Out) {
    let (k, n) = (1024u32, 4096u32);
    let mut fill = Fill(9);
    let w = gpu.storage_init("w", &(0..(n * k) as usize).map(|_| fill.unit()).collect::<Vec<_>>());
    for m in [1u32, 256] {
        let x = gpu.storage_init("x", &(0..(m * k) as usize).map(|_| fill.unit()).collect::<Vec<_>>());
        let y = gpu.storage(u64::from(m * n));
        let step = gpu.step(gpu.kernel_index("matmul").expect("registered"), &[&x, &w, &y], &[m, k, n], m * n);
        let secs = best_secs(gpu, &[step], 4, 5);
        out.rate(&format!("matmul_f32_m{m}_{k}x{n}"), 2.0 * f64::from(m) * f64::from(k) * f64::from(n) / secs / 1e9, "GFLOPS");
    }
}

fn fp8_gemv_portable(gpu: &Gpu, out: &mut Out) {
    if !gpu.caps().workgroup_reductions {
        println!("PERF-SKIP fp8_gemv_portable the cooperative decode kernels cannot run on this device");
        return;
    }
    let (k, n, block) = (5120u32, 17408u32, 128u32);
    let cb = k.div_ceil(block);
    let mut fill = Fill(1);
    let w = words(gpu, &(0..n as usize * k as usize / 4).map(|_| fill.word()).collect::<Vec<_>>());
    let s = gpu.storage_init("s", &vec![0.01f32; (n.div_ceil(block) * cb) as usize]);
    for m in [1u32, 8] {
        let x = gpu.storage_init("x", &(0..(m * k) as usize).map(|_| fill.unit()).collect::<Vec<_>>());
        let y = gpu.storage((m * n) as u64);
        let idx = gpu.kernel_index(&kernels::template::gemv_kernel_name(FP8, m)).expect("FP8 GEMV registered");
        let tiles = n.div_ceil(4);
        let rpw = (tiles / 528).clamp(1, 16);
        let step = gpu.dispatch(idx, &[&x, &w, &s, &y], &[m, k, n, 0, block, cb, 0, rpw], Dispatch::Workgroups(tiles.div_ceil(rpw)));
        let secs = best_secs(gpu, &[step], 16, 5);
        out.rate(&format!("fp8_gemv_portable_m{m}_{k}x{n}"), f64::from(n) * f64::from(k) / secs / 1e9, "GB/s");
    }
}

/// The FP8 candidates the autotuner picks from on a Hopper CUDA device; the best of those that serve each shape.
fn fp8_native(gpu: &Gpu, out: &mut Out) {
    if !gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::WGMMA_MIN_CC) {
        println!("PERF-SKIP fp8_native the FP8 candidates are registered for a device with warpgroup tensor cores");
        return;
    }
    let (k, n) = (5120u32, 17408u32);
    for m in [1u32, 16, 64] {
        let best = gpu_core::native_time_fp8_shape(gpu, [m, k / 4, n]).into_iter().filter_map(|(name, v)| match v {
            NativeVerdict::Timed { gbs, .. } => Some((name, gbs)),
            NativeVerdict::Rejected(_) => None,
        });
        let (name, gbs) = best.max_by(|a, b| a.1.total_cmp(&b.1)).expect("a candidate serves the decode widths");
        println!("PERF-NOTE fp8_native_m{m} best candidate {name}");
        out.rate(&format!("fp8_native_m{m}_{k}x{n}"), gbs, "GB/s");
    }
    let m = 4096u32;
    let best = gpu_core::native_time_fp8_candidates(gpu, [m, k / 4, n], "tma").into_iter().filter_map(|(name, v)| match v {
        NativeVerdict::Timed { us, .. } => Some((name, 2.0 * f64::from(m) * f64::from(k) * f64::from(n) / us / 1e6)),
        NativeVerdict::Rejected(_) => None,
    });
    let (name, tflops) = best.max_by(|a, b| a.1.total_cmp(&b.1)).expect("a TMA candidate serves the prefill round");
    println!("PERF-NOTE fp8_native_m{m} best candidate {name}");
    out.rate(&format!("fp8_native_m{m}_{k}x{n}"), tflops, "TFLOPS");
}

const N_HEADS: u32 = 24;
const N_KV: u32 = 4;
const HD: u32 = 256;

/// Paged flash prefill of one `n`-row round from an empty cache: portable on every backend that runs the cooperative
/// kernels, tensor-core on CUDA.
fn flash_prefill(gpu: &Gpu, n: u32, out: &mut Out) {
    if !gpu.caps().workgroup_reductions {
        println!("PERF-SKIP flash_prefill the cooperative kernels cannot run on this device");
        return;
    }
    let mut fill = Fill(3);
    let kv_len = (n * N_KV * HD) as usize;
    let q = gpu.storage_init("q", &(0..(n * N_HEADS * HD) as usize).map(|_| fill.unit()).collect::<Vec<_>>());
    let k = gpu.storage_init("k", &(0..kv_len).map(|_| fill.unit()).collect::<Vec<_>>());
    let v = gpu.storage_init("v", &(0..kv_len).map(|_| fill.unit()).collect::<Vec<_>>());
    // One block holds the whole round; every query row sees its own table (the serving layout).
    let tables = words(gpu, &vec![0u32; n as usize]);
    let seq_lens = words(gpu, &(1..=n).collect::<Vec<_>>());
    let ctx = gpu.storage((n * N_HEADS * HD) as u64);
    let bufs = [&q, &k, &v, &tables, &seq_lens, &ctx];
    let params = [n, N_HEADS, N_KV, HD, N_HEADS / N_KV, n, 1];
    let blocks = N_HEADS * n.div_ceil(64);
    // Causal attention: 4 flops per (query, key, channel) over n(n+1)/2 pairs per head.
    let flops = f64::from(n) * f64::from(n + 1) / 2.0 * 4.0 * f64::from(HD) * f64::from(N_HEADS);
    let kind = gpu.kernel_index("paged_flash_prefill_hd256").expect("registered");
    let portable = gpu.dispatch(kind, &bufs, &params, Dispatch::Workgroups(blocks));
    let secs = best_secs(gpu, &[portable], 1, 3);
    out.rate(&format!("flash_prefill_portable_n{n}"), flops / secs / 1e12, "TFLOPS");
    if gpu.kind() == "cuda" {
        match gpu_core::provider::cuda::paged_flash_prefill_step(gpu, HD, &bufs, &params, blocks) {
            Some(native) => {
                let secs = best_secs(gpu, &[native], 2, 5);
                out.rate(&format!("flash_prefill_native_n{n}"), flops / secs / 1e12, "TFLOPS");
            }
            None => println!("PERF-SKIP flash_prefill_native the device is below the kernel's capability floor"),
        }
    }
}

/// Decode attention for `batch` sequences of `seq` cached tokens at head width 128 (the production decode shape):
/// the portable split-key pair on every backend with cooperative kernels, the batched native kernel on CUDA.
fn decode_attention(gpu: &Gpu, out: &mut Out) {
    let (batch, n_heads, n_kv, hd, bs, seq) = (16u32, 16u32, 8u32, 128u32, 16u32, 2048u32);
    let group = n_heads / n_kv;
    let mbs = seq / bs;
    let kv_stride = n_kv * hd;
    let mut fill = Fill(5);
    let pool = (batch * mbs * bs * kv_stride) as usize;
    let q = gpu.storage_init("q", &(0..(batch * n_heads * hd) as usize).map(|_| fill.unit()).collect::<Vec<_>>());
    let pk = gpu.storage_init("pk", &(0..pool).map(|_| fill.unit()).collect::<Vec<_>>());
    let pv = gpu.storage_init("pv", &(0..pool).map(|_| fill.unit()).collect::<Vec<_>>());
    let bt = words(gpu, &(0..batch * mbs).collect::<Vec<_>>());
    let sl = words(gpu, &vec![seq; batch as usize]);
    let ctx = gpu.storage((batch * n_heads * hd) as u64);
    // Both K and V planes are read once: the bytes the step must move.
    let bytes = 2.0 * f64::from(batch) * f64::from(seq) * f64::from(kv_stride) * 4.0;
    if gpu.caps().workgroup_reductions {
        let tiles_per_split = 16u32;
        let n_splits = (seq / 8).div_ceil(tiles_per_split).max(1);
        let (pm, pl) = (gpu.storage(u64::from(batch * n_heads * n_splits)), gpu.storage(u64::from(batch * n_heads * n_splits)));
        let po = gpu.storage(u64::from(batch * n_heads * n_splits * hd));
        let sp = [batch, n_heads, n_kv, hd, group, bs, mbs, n_splits, tiles_per_split];
        let cp = [batch, n_heads, hd, n_splits];
        let split = gpu.dispatch(gpu.kernel_index("paged_flash_decode_split").expect("registered"), &[&q, &pk, &pv, &bt, &sl, &pm, &pl, &po], &sp, Dispatch::Workgroups(batch * n_heads * n_splits));
        let combine = gpu.step(gpu.kernel_index("paged_flash_decode_combine").expect("registered"), &[&pm, &pl, &po, &ctx], &cp, batch * n_heads * 128);
        let secs = best_secs(gpu, &[split, combine], 8, 5);
        out.rate(&format!("decode_attn_portable_b{batch}_s{seq}"), bytes / secs / 1e9, "GB/s");
    } else {
        println!("PERF-SKIP decode_attn_portable the cooperative decode kernels cannot run on this device");
    }
    if gpu.kind() == "cuda" {
        let params = [batch, n_heads, n_kv, hd, group, bs, mbs];
        match gpu_core::provider::cuda::paged_decode_batched_step(gpu, false, &[&q, &pk, &pv, &bt, &sl, &ctx], &params) {
            Some(native) => {
                let secs = best_secs(gpu, &[native], 8, 5);
                out.rate(&format!("decode_attn_native_b{batch}_s{seq}"), bytes / secs / 1e9, "GB/s");
            }
            None => println!("PERF-SKIP decode_attn_native the device is below the kernel's capability floor"),
        }
    } else {
        println!("PERF-SKIP decode_attn_native CUDA only");
    }
}

/// The tensor-core GDN chunk forward at the model's chunk (64 rows, 128 channels) over `bhc` head-chunks.
fn gdn_chunk(gpu: &Gpu, out: &mut Out) {
    const D: u32 = 128;
    let (bhc, c) = (1536u32, 64u32);
    let mut fill = Fill(7);
    let mut up = |name: &str, n: u32, scale: f32| gpu.storage_init(name, &(0..n as usize).map(|_| fill.unit() * scale).collect::<Vec<_>>());
    let (q, k, v) = (up("q", bhc * c * D, 0.1), up("k", bhc * c * D, 0.1), up("v", bhc * c * D, 1.0));
    let (g, beta) = (up("g", bhc * c, -0.4), up("beta", bhc * c, 1.0));
    let blank = |n: u32| gpu.storage(u64::from(n).max(1));
    let (g_cs, exp_g_cs, u, w, intra) = (blank(bhc * c), blank(bhc * c), blank(bhc * c * D), blank(bhc * c * D), blank(bhc * c * c));
    let bufs = [&q, &k, &v, &g, &beta, &g_cs, &exp_g_cs, &u, &w, &intra];
    match gpu_core::provider::cuda::gdn_chunk_prep_step(gpu, &bufs, bhc, c, D, D, 1.0 / (D as f32).sqrt()) {
        Some(step) => {
            let secs = best_secs(gpu, &[step], 8, 5);
            // Operands read plus the five products written, f32.
            let bytes = 4.0 * f64::from(bhc) * f64::from(c) * (3.0 * f64::from(D) + 2.0 + 2.0 + 2.0 * f64::from(D) + f64::from(c));
            out.rate(&format!("gdn_chunk_native_bhc{bhc}_c{c}"), bytes / secs / 1e9, "GB/s");
        }
        None => println!("PERF-SKIP gdn_chunk_native the tensor-core chunk forward is not offered here"),
    }
}

#[test]
#[ignore = "a timing gate: make test/perf/backends"]
fn hot_ops_rates_on_this_backend() {
    let gpu = gpu_core::testgpu::dev(kernel_set());
    println!("PERF-DEVICE {}", gpu.kind());
    let mut out = Out(Vec::new());
    fp8_gemv_portable(&gpu, &mut out);
    matmul_f32(&gpu, &mut out);
    flash_prefill(&gpu, 4096, &mut out);
    decode_attention(&gpu, &mut out);
    if gpu.kind() == "cuda" {
        fp8_native(&gpu, &mut out);
        gdn_chunk(&gpu, &mut out);
    } else {
        println!("PERF-SKIP native ops CUDA only");
    }
    assert!(!out.0.is_empty(), "no op was timed on this backend");
}
