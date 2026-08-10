// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decoder-LM profiler: where a PREFILL and a DECODE step actually spend their
//! time, per kernel kind, graded against the device's measured roofline.
//!
//! Why this exists. `vqgan_bench`, `unet_bench` and `flux2_bench` cover the
//! conv/diffusion datapaths; the LLM datapath had no per-kernel-kind profiler at
//! all, so every recorded qwen number in `docs/performance/` came from
//! `BRAIN_PROFILE`'s timestamp table — and, more importantly, all of them were
//! measured on `qwen-synth:8x512x8` (47 M params, vocab 32 k). That target
//! **structurally cannot express Qwen3-0.6B**: it forces `head_dim = d/h` and
//! `d_ff = 4d`, while the real model has `head_dim` 128 (so `q_dim` 2048 ≠
//! `d_model` 1024) and `d_ff` 3072. Its vocab is 151936, not 32000 — 4.75× —
//! and the head is tied, so the LM head alone is 155.6 M params = 622 MB fp32,
//! about a quarter of every decode step's bytes. None of that was ever profiled.
//!
//! Random weights throughout: cost depends on shape, not values. Every number
//! this produces is **valid for cost and meaningless for output quality**.
//!
//! The roofline the rows are graded against is `gpu_core::roof`'s MEASURED one,
//! not a hardcoded peak — so the same table is meaningful on any device.
//!
//! Usage:
//!   qwen_bench                      # prefill T=512 + decode, fp32, at 0.6B
//!   qwen_bench prefill [T] [reps]
//!   qwen_bench decode  [ctx] [reps]
//!   qwen_bench serve   [rows] [reps] [ctx] [i8w] [kv8]
//!                                    # the PAGED serving tape (a different
//!                                    # kernel set from the batched forward);
//!                                    # `i8w` = int8 weights, `kv8` = int8 KV
//!   qwen_bench head    [reps]       # the tied 151936x1024 LM head alone
//!   qwen_bench cost                 # offline FLOP/byte accounting, no device
//!   qwen_bench gemm8   [m k n] [reps]      # A/B `matmul_i8_dyn` at one shape
//!   qwen_bench gemm8-sweep [k n] [reps]    # `matmul_i8_dyn` GOP/s across an
//!                                          # `m` sweep at fixed k,n — the D0
//!                                          # occupancy-hypothesis check
//!                                          # (§F.2) before touching any WGSL

use std::time::Instant;

use gpu_core::roof::Roofs;
use gpu_core::Gpu;
use qwen3::{init_weights, Qwen, QwenConfig};

/// One §F.1 pass profile against the device's measured roofline, plus the rows
/// that sit below their class's floor.
fn report(gpu: &Gpu, label: &str, steps: &[gpu_core::Step], reps: usize, roofs: Option<Roofs>) -> f64 {
    let p = gpu_core::profile::profile(gpu, label, steps, reps);
    p.print_top(roofs, 16);
    if let Some(r) = roofs {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<26} {:>5.1}% of its {} roof (floor {:.0}%) — {:.1}% of this pass",
                row.name,
                pct,
                bound.as_str(),
                bound.defect_pct(),
                100.0 * row.secs / p.summed_secs,
            );
        }
    }
    p.total_secs
}

fn banner(gpu: &Gpu) -> Option<Roofs> {
    let roofs = gpu_core::roof::ensure(gpu);
    match roofs {
        Some(r) => println!(
            "measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte",
            r.gflops,
            r.gbs,
            r.cache_gbs,
            r.ridge()
        ),
        None => println!("roofline unmeasured — utilisation columns print '-' rather than a guess"),
    }
    roofs
}

/// Weight bytes and the decode-step bandwidth ceiling they imply.
///
/// Decode reads every weight once per token, so on a memory-bound device the
/// token rate has a hard ceiling of `bandwidth / weight_bytes` no matter how
/// good the kernels are. Stating it first is what makes the profile
/// interpretable: a kernel-level fix that cannot move this number is not a
/// throughput fix.
fn weight_budget(cfg: &QwenConfig, roofs: Option<Roofs>) {
    let (d, ff, l) = (cfg.d_model as u64, cfg.d_ff as u64, cfg.n_layers as u64);
    let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
    let attn = d * hq + 2 * (d * hkv) + hq * d;
    let mlp = 2 * (d * ff) + ff * d;
    let per_layer = attn + mlp;
    let embed = cfg.vocab as u64 * d;
    // Tied embeddings mean the head IS the embedding table — counted once as
    // resident bytes, but read twice per step (gather + head GEMV).
    let total = l * per_layer + embed * if cfg.tie_embeddings { 1 } else { 2 };

    println!(
        "\n{} params = {:.2} GB fp32 / {:.2} GB int8   (layers {:.1} M x {l}, embed/head {:.1} M{})",
        total,
        total as f64 * 4.0 / 1e9,
        total as f64 / 1e9,
        per_layer as f64 / 1e6,
        embed as f64 / 1e6,
        if cfg.tie_embeddings { ", tied" } else { "" },
    );
    if let Some(r) = roofs {
        let bw = r.gbs as f64 * 1e9;
        println!(
            "decode is weight-bandwidth bound: ceiling {:.0} tok/s fp32, {:.0} tok/s int8 \
             (the head alone is {:.0}% of the bytes)",
            bw / (total as f64 * 4.0),
            bw / total as f64,
            100.0 * embed as f64 / total as f64,
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("all");

    let cfg = QwenConfig::qwen3_0_6b();

    if mode == "cost" {
        // Offline accounting only — no device, no probe, no weights uploaded.
        let t = 128u32;
        let init = init_weights(&cfg, 7);
        let m = Qwen::new(cfg.clone(), 1, t, &init);
        let c = m.cost_fwd();
        println!("{c}");
        return;
    }
    if mode == "gemm8" {
        let m: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
        let k: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
        let n: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(2048);
        let reps: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);
        gemm8_ab(m, k, n, reps);
        return;
    }
    if mode == "gemm8-sweep" {
        let k: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
        let n: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2048);
        let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
        gemm8_sweep(k, n, reps);
        return;
    }

    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    match mode {
        "decode" => {
            let ctx: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
            eprintln!("qwen_bench decode: Qwen3-0.6B, ctx {ctx}, {reps} reps (random weights)");
            let t0 = Instant::now();
            let init = init_weights(&cfg, 7);
            let m = Qwen::from_tensors_decode(cfg.clone(), &init, ctx);
            eprintln!("built in {:.1}s\n", t0.elapsed().as_secs_f32());
            let gpu = m.gpu().share();
            let roofs = banner(&gpu);
            weight_budget(&cfg, roofs);

            // Profile the step at the END of the context, where the attention
            // reduction is longest — the honest worst case for a decode step,
            // and the one a served request spends most of its tokens near.
            let steps = m.decode_steps(Some(1), ctx - 1, None, None);
            let secs = report(&gpu, &format!("DECODE @pos {}", ctx - 1), &steps, reps, roofs);
            println!(
                "\none decode step: {:.3} ms  ->  {:.1} tok/s (single stream, no LM head — \
                 the head is applied host-side on this path)",
                secs * 1e3,
                1.0 / secs
            );
        }
        "serve" => {
            // The paged serving tape is a DIFFERENT kernel set from the batched
            // forward — `qwen3::serve` registers its own pipelines — so a
            // finding on one does not transfer to the other, and the serving
            // path is what actually runs behind the HTTP/D-Bus surface.
            //
            // `rows` is what a chunked-prefill chunk or a concurrent decode
            // batch presents: the same tape serves both, with `seqlens[i]`
            // deciding which.
            let rows: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
            // `i8w` quantizes the 7 per-layer linears + the head at load
            // (A0). The ledger records +32% at c=16 on `qwen-synth`; it has
            // never been measured at the real 0.6B shape, where the tied
            // 622 MB head is a quarter of the weight bytes.
            let i8w = a.iter().any(|x| x == "i8w");
            let kv8 = a.iter().any(|x| x == "kv8");
            eprintln!(
                "qwen_bench serve: Qwen3-0.6B, {rows} rows, {reps} reps (random weights\
                 {}{})",
                if i8w { ", int8 weights" } else { "" },
                if kv8 { ", int8 KV" } else { "" }
            );
            let init = init_weights(&cfg, 7);
            let bs = 16u32;
            // Context per sequence. Every row gets its OWN blocks (see below),
            // so the paged pool is `rows * mbs` blocks and grows with BOTH —
            // 2 * n_layers * block_size * kv_stride * 4 B each, which is 3.6 MB
            // per block on this config. Default to a context that keeps the
            // pool inside a 24 GB card alongside the 2.4 GB of weights; the
            // guard below turns an over-large request into a message rather
            // than a wgpu "Out of Memory" panic ten seconds later.
            let seq: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
            let mbs = seq.div_ceil(bs);
            let blk_bytes =
                2 * cfg.n_layers as u64 * bs as u64 * cfg.kv_dim() as u64 * 4;
            let pool_gb = (rows as u64 * mbs as u64) as f64 * blk_bytes as f64 / 1e9;
            eprintln!(
                "paged pool: {} blocks x {:.1} MB = {:.1} GB (+2.4 GB weights)",
                rows * mbs,
                blk_bytes as f64 / 1e6,
                pool_gb
            );
            if pool_gb > 18.0 {
                eprintln!(
                    "refusing: that pool will not fit. Lower `rows` or the 4th arg (context, \
                     default 256): qwen_bench serve <rows> <reps> <ctx>"
                );
                return;
            }
            let t0 = Instant::now();
            let eng = qwen3::serve::Engine::from_map(
                cfg.clone(), &init, bs, rows * mbs, rows.max(8), mbs, rows.max(8), kv8, i8w,
            );
            eprintln!("built in {:.1}s\n", t0.elapsed().as_secs_f32());
            let gpu = eng.gpu().share();
            let roofs = banner(&gpu);
            weight_budget(&cfg, roofs);

            // Every row at FULL context, not a 1..rows ramp. Two reasons, and
            // they agree: it is the steady-state decode shape (the case worth
            // optimising), and it is the case `gpu_core::cost`'s paged formulas
            // are EXACT for — those use `cap` because `seq_lens` lives in a
            // storage buffer the cost model cannot see, so a ramp would make
            // the attention kernels report roughly double their real rate.
            let positions: Vec<u32> = (0..rows).map(|_| seq - 1).collect();
            let seqlens: Vec<u32> = (0..rows).map(|_| seq).collect();
            // Each row gets its OWN physical blocks. Sharing one block table
            // across every row (the obvious way to build this) makes all 128
            // sequences read the same few megabytes, so they hit cache and the
            // profile's streaming byte estimate becomes fiction — it reported
            // `paged_decode_scores_batched` at 2120% of the bandwidth roof,
            // which is the profiler catching the harness, not the kernel.
            let blocks: Vec<u32> = (0..rows).map(|i| i * mbs).collect();
            let offsets: Vec<u32> = (0..rows).map(|i| i % bs).collect();
            let bt: Vec<u32> = (0..rows).flat_map(|i| (0..mbs).map(move |b| i * mbs + b)).collect();

            let tokens: Vec<u32> = (0..rows).map(|i| (i * 131 + 7) % cfg.vocab).collect();
            let steps =
                eng.steps_for_profile(rows, &tokens, &positions, &seqlens, &blocks, &offsets, &bt);
            let secs = report(&gpu, &format!("SERVE {rows} rows"), &steps, reps, roofs);
            println!(
                "\none served step at {rows} rows: {:.2} ms  ->  {:.0} rows/s",
                secs * 1e3,
                rows as f64 / secs
            );
        }
        "prefill" | "all" | _ => {
            let t: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
            eprintln!("qwen_bench prefill: Qwen3-0.6B, T={t}, {reps} reps (random weights)");
            let t0 = Instant::now();
            let init = init_weights(&cfg, 7);
            let m = Qwen::new(cfg.clone(), 1, t, &init);
            eprintln!("built in {:.1}s\n", t0.elapsed().as_secs_f32());
            let gpu = m.gpu().share();
            let roofs = banner(&gpu);
            weight_budget(&cfg, roofs);

            let x: Vec<u32> = (0..t).map(|i| (i * 131 + 7) % cfg.vocab).collect();
            m.set_batch(&x, &x);
            let secs = report(&gpu, &format!("PREFILL T={t}"), m.fwd_steps(), reps, roofs);
            println!(
                "\nprefill T={t}: {:.2} ms  ->  {:.0} tok/s",
                secs * 1e3,
                t as f64 / secs
            );
        }
    }
}

/// Pack signed bytes 4-per-`u32`, little-endian — the exact layout
/// `dot4I8Packed` expects and `model::int8::quantize_weight`/`quant_pack`
/// produce. `as u8` preserves the two's-complement bit pattern (a sign
/// EXTEND here, rather than this truncating cast, would corrupt every
/// negative value — the trap `dot4i8packed_matches_host_reference` in
/// `crates/wgsl-cpu/src/lib.rs` exists to catch for the GEMV sibling).
fn pack_i8(v: &[i8]) -> Vec<u32> {
    v.chunks(4).map(|c| (c[0] as u8 as u32) | ((c[1] as u8 as u32) << 8) | ((c[2] as u8 as u32) << 16) | ((c[3] as u8 as u32) << 24)).collect()
}

/// Exact i32 host reference for an int8×int8 GEMM. The contraction is
/// integer, hence exact — no tolerance needed, unlike an fp32 GEMM oracle.
fn host_i8_gemm(x_i8: &[i8], w_i8: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc: i32 = 0;
            for ki in 0..k {
                acc += x_i8[mi * k + ki] as i32 * w_i8[ni * k + ki] as i32;
            }
            out[mi * n + ni] = acc as f32 * sx[mi] * sw[ni];
        }
    }
    out
}

/// A/B `matmul_i8_dyn` at one shape: exact correctness vs the host i32
/// reference above, plus GOP/s against the measured DP4A roof (never the
/// fp32 roof — `Roofs::compute_roof`/`utilisation_of` pick the DP4A one
/// automatically when `int_ops > flops`, the fix for the 4.15x-flattering
/// bug `docs/performance/overview.md`'s "int8 weights on the served path"
/// section records).
///
/// `qwen_bench gemm8 [m k n] [reps]` — defaults to the qkv-projection shape
/// at Qwen3-0.6B's 128-row served prefill (`d_model=1024, q_dim=2048`).
pub fn gemm8_ab(m: u32, k: u32, n: u32, reps: usize) {
    assert_eq!(k % 4, 0, "k must be a multiple of 4 (int8 packing)");
    let kg = k / 4;
    let gpu = Gpu::new(&[("matmul_i8_dyn", kernels::MATMUL_I8_DYN)]);
    let ki = gpu.kernel_index("matmul_i8_dyn").expect("matmul_i8_dyn registered above");
    let mut rng = data::rng::Rng::new(11);
    let rand_i8 = |rng: &mut data::rng::Rng, len: usize| -> Vec<i8> { (0..len).map(|_| ((rng.next_f32() * 254.0) as i32 - 127) as i8).collect() };
    let x_i8 = rand_i8(&mut rng, (m * k) as usize);
    let w_i8 = rand_i8(&mut rng, (n * k) as usize);
    let sx: Vec<f32> = (0..m).map(|_| rng.next_f32() * 0.1 + 0.01).collect();
    let sw: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.1 + 0.01).collect();

    let xq = pack_i8(&x_i8);
    let wq = pack_i8(&w_i8);
    let xb = gpu.storage(xq.len() as u64);
    gpu.write(&xb, &xq);
    let wb = gpu.storage(wq.len() as u64);
    gpu.write(&wb, &wq);
    let sxb = gpu.storage_init("sx", &sx);
    let swb = gpu.storage_init("sw", &sw);
    let out = gpu.storage((m * n) as u64);

    let tiles = m.div_ceil(128) * n.div_ceil(128) * 256;
    let st = vec![gpu.step(ki, &[&xb, &wb, &sxb, &swb, &out], &[m, kg, n], tiles)];
    let t = gpu_core::profile::best_of(&gpu, &st, reps);
    let got = gpu.read(&out, (m * n) as usize);

    let want = host_i8_gemm(&x_i8, &w_i8, &sx, &sw, m as usize, k as usize, n as usize);
    let max_abs = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_rel = got.iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0f32, f32::max);

    let roofs = gpu_core::roof::ensure(&gpu);
    let int_ops = 8u64 * m as u64 * kg as u64 * n as u64;
    let gops = int_ops as f64 / t / 1e9;
    let pct = roofs.and_then(|r| r.utilisation_of(0, int_ops, 0, t));
    println!(
        "matmul_i8_dyn [m={m:>5} k={k:>5} n={n:>5}]  {:>8.3} ms  {:>9.1} GOP/s  {:<20}  max|Δ|={:.3e} max_rel={:.3e}  wgs={tiles}",
        t * 1e3,
        gops,
        pct.map(|p| format!("{p:.1}% of int8 roof")).unwrap_or_else(|| "roof unmeasured".into()),
        max_abs,
        max_rel,
    );
}

/// D0 — the occupancy-hypothesis check (§F.2), run BEFORE touching any WGSL:
/// sweep `m` at fixed `k,n` and watch whether GOP/s climbs with it. Climbing
/// confirms the tile grid is occupancy-starved at small `m` for `matmul_i8_dyn`
/// (mirroring `matmul_reg3_splitk.wgsl`'s own header table for the fp32
/// sibling); flat means it doesn't, and that fix plan is wrong.
///
/// `qwen_bench gemm8-sweep [k n] [reps]`.
pub fn gemm8_sweep(k: u32, n: u32, reps: usize) {
    println!("D0 occupancy sweep: matmul_i8_dyn at k={k} n={n}, m swept\n");
    for &m in &[8u32, 32, 128, 256, 512, 1024, 2048] {
        gemm8_ab(m, k, n, reps);
    }
}
