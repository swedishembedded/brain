// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder attention A/B: `block::gemm_bidir_fwd` vs
//! `block::flash_bidir_fwd` (both flash variants) at the model's real 8k shape.
//!
//! Why this exists: `crates/lfm/src/model.rs` registered `flash_attn_bidir` and
//! never dispatched it, and the roadmap recorded "flash measured
//! ≈ naive here". That measurement predates `flash_attn_bidir_split`, which is
//! numerically identical (cosine 1.00000000) but 14.4× the baseline at
//! `head_dim = 64` — and 64 is exactly lfm's `head_dim`. So the recorded
//! conclusion does not transfer, and the choice has to be re-measured.
//!
//! The comparison is **like for like**: the GEMM path packs straight from the
//! narrow k/v projections (GQA replication folded into `head_pack`), while the
//! flash kernels read a fused `[q | k_exp | v_exp]` slab — so the flash timings
//! here INCLUDE the three `kv_expand` dispatches that build it, exactly as a
//! wired flash path would have to pay them.
//!
//! Shapes come from `LfmConfig::lfm25_encoder_350m` (d_model 1024, 16 heads, 8 kv
//! heads, head_dim 64) at T = 8192, with the chunk the chunked regime actually
//! picks from `caps::SLAB_BUDGET` (512 MiB / (H·T·4) = 1024).
//!
//! Usage: `lfm_attn_ab [t] [reps]`   (`BRAIN_GPU_INDEX=n` selects a card)

use std::time::Instant;

use gpu_core::{DeviceBuffer, Gpu, Step};
use lfm::config::LfmConfig;
use model::block;

/// Tesla P40 fp32 peak, for a "% of peak" column.
const P40_FP32_TFLOPS: f64 = 11.76;

const KERNELS: &[(&str, &str)] = &[
    ("kv_expand", kernels::KV_EXPAND),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("head_pack", kernels::HEAD_PACK),
    ("head_pack_t", kernels::HEAD_PACK_T),
    ("head_unpack", kernels::HEAD_UNPACK),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
];
const K_KV_EXPAND: usize = 0;
const K_FLASH: usize = 1;
const K_SPLIT: usize = 2;
const K_HEAD_PACK: usize = 3;
const K_HEAD_PACK_T: usize = 4;
const K_HEAD_UNPACK: usize = 5;
const K_SOFTMAX_ROWS: usize = 6;
const K_MATMUL: usize = 7;
const K_MATMUL_REG3: usize = 8;

fn time_steps(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
    gpu.submit(&[], steps);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

/// Deterministic filler — this bench is a timing harness, not a parity gate,
/// but the two paths must agree, so both read the SAME bytes.
fn fill(gpu: &Gpu, b: &DeviceBuffer, n: usize, phase: f64) {
    let v: Vec<f32> = (0..n).map(|i| ((i as f64 * 0.37 + phase).sin() * 0.5) as f32).collect();
    gpu.write(b, bytemuck::cast_slice(&v));
}

fn cosine(a: &[f32], b: &[f32]) -> (f64, f32) {
    let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f32);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
        mx = mx.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), mx)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let cfg = LfmConfig::lfm25_encoder_350m();
    let (d, nh, nkv, hd) = (cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (hq, hkv, group) = (cfg.q_dim(), cfg.kv_dim(), cfg.group());
    // The chunk the chunked regime picks: SLAB_BUDGET / (heads · T · 4 bytes).
    let chunk = (((512u64 << 20) / (nh as u64 * t as u64 * 4)) as u32).clamp(64, 4096).min(t);

    let gpu = Gpu::new(KERNELS);
    let caps = gpu.caps();
    println!(
        "device: {:?} | max_workgroup_size {} | workgroup_reductions {}",
        caps.class, caps.max_workgroup_size, caps.workgroup_reductions
    );
    println!("shape: T={t} d_model={d} heads={nh} kv_heads={nkv} head_dim={hd} group={group} chunk={chunk}");

    let n = t as u64;
    let q = gpu.storage(n * hq as u64);
    let k = gpu.storage(n * hkv as u64);
    let v = gpu.storage(n * hkv as u64);
    fill(&gpu, &q, (n * hq as u64) as usize, 0.0);
    fill(&gpu, &k, (n * hkv as u64) as usize, 1.7);
    fill(&gpu, &v, (n * hkv as u64) as usize, 3.1);

    // `qkv` doubles as the GEMM path's pack space (as it does in `model.rs`),
    // so it must be the larger of the two uses: [n, 3·d] vs 3 pack segments of
    // [n, hq]. With hq == d they are the same size.
    let qkv = gpu.storage(n * 3 * u64::from(d.max(hq)));
    let ctx_pack = gpu.storage(n * hq as u64);
    let slab = nh as u64 * chunk as u64 * t as u64;
    let scores = gpu.storage(slab);
    let probs = gpu.storage(slab);
    let ctx_gemm = gpu.storage(n * hq as u64);
    let ctx_flash = gpu.storage(n * hq as u64);
    let ctx_flash_base = gpu.storage(n * hq as u64);

    let spans = [(0u32, t)];

    // --- path B: GEMM attention (what lfm dispatches today) -----------------
    let gemm_ids = block::GemmAttnIds {
        head_pack: K_HEAD_PACK,
        head_pack_t: K_HEAD_PACK_T,
        head_unpack: K_HEAD_UNPACK,
        softmax_rows: K_SOFTMAX_ROWS,
        matmul: K_MATMUL,
        matmul_reg: K_MATMUL_REG3,
    };
    let mut gemm_steps: Vec<Step> = Vec::new();
    block::gemm_bidir_fwd(
        &gpu, &gemm_ids, nh, hd, group, &q, hq, (&k, &v), hkv, &ctx_gemm, hq, &qkv, &ctx_pack, &scores, &probs,
        &spans, chunk, false, &mut gemm_steps,
    );
    let t_gemm = time_steps(&gpu, &gemm_steps, reps);
    let out_gemm = gpu.read(&ctx_gemm, (n * hq as u64) as usize);

    // --- path A/A': flash attention, INCLUDING the kv_expand fusion ---------
    // `qkv` is reused, so the fused build must be re-emitted for each variant
    // (the GEMM run above left pack data in it).
    let expand = |dst: &DeviceBuffer| -> Vec<Step> {
        vec![
            block::kv_expand_fwd(&gpu, K_KV_EXPAND, &q, dst, t, nh, 1, hd, 3 * d, 0),
            block::kv_expand_fwd(&gpu, K_KV_EXPAND, &k, dst, t, nh, group, hd, 3 * d, d),
            block::kv_expand_fwd(&gpu, K_KV_EXPAND, &v, dst, t, nh, group, hd, 3 * d, 2 * d),
        ]
    };

    let mut split_steps = expand(&qkv);
    let split_expand_len = split_steps.len();
    block::flash_bidir_fwd(
        &gpu,
        block::FlashIds { bidir: K_FLASH, split: Some(K_SPLIT) },
        nh, hd, hq, &qkv, 3 * d, 0, d, 2 * d, &ctx_flash, &spans, &mut split_steps,
    );
    let t_split = time_steps(&gpu, &split_steps, reps);
    let t_expand = time_steps(&gpu, &split_steps[..split_expand_len], reps);
    let out_split = gpu.read(&ctx_flash, (n * hq as u64) as usize);

    let mut base_steps = expand(&qkv);
    block::flash_bidir_fwd(
        &gpu,
        block::FlashIds { bidir: K_FLASH, split: None },
        nh, hd, hq, &qkv, 3 * d, 0, d, 2 * d, &ctx_flash_base, &spans, &mut base_steps,
    );
    let t_base = time_steps(&gpu, &base_steps, reps);
    let out_base = gpu.read(&ctx_flash_base, (n * hq as u64) as usize);

    // --- report -------------------------------------------------------------
    // Attention FLOP for one bidirectional span: QKᵀ + PV = 2 · 2·T²·hd per head.
    let gf = 4.0 * t as f64 * t as f64 * hd as f64 * nh as f64 / 1e9;
    let row = |name: &str, s: f64| {
        println!(
            "{name:<34} {:>9.2} ms  {:>8.0} GFLOP/s  {:>6.2}% peak  {:>6.2}x gemm",
            s * 1e3,
            gf / s,
            100.0 * (gf / s) / (P40_FP32_TFLOPS * 1e3),
            t_gemm / s
        );
    };
    println!();
    row("gemm_bidir_fwd (current)", t_gemm);
    row("kv_expand + flash_bidir_split", t_split);
    row("kv_expand + flash_bidir (base)", t_base);
    println!("{:<34} {:>9.2} ms  (included in both flash rows)", "kv_expand alone", t_expand * 1e3);

    let (c_sg, m_sg) = cosine(&out_gemm, &out_split);
    let (c_bs, m_bs) = cosine(&out_base, &out_split);
    println!();
    println!("agreement gemm vs split : cosine {c_sg:.10}  max_abs {m_sg:.3e}");
    println!("agreement base vs split : cosine {c_bs:.10}  max_abs {m_bs:.3e}");
}
