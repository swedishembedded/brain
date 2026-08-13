// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Weight::{BF16,F16}` dual-backend roundtrip for the two genuine
//! weight-consuming inference kernels B8 added beyond the B4/B5 matmul
//! family: embedding-table gather (`Ops::embed`, `embed.wgsl`/
//! `embed_tile.wgsl`) and the sparse-MoE expert linear (`Ops::moe_linear`,
//! `moe_linear_gated.wgsl`). Same structure as `bf16_roundtrip.rs`/
//! `f16_roundtrip.rs`: pack a small host reference table/weight, dispatch
//! through the façade, compare against an f32 host oracle within a derived
//! tolerance, on BOTH `Gpu::new_cpu` and `Gpu::new_wgpu`.
//!
//! **Tolerance.** `embed` is a pure gather + dequant - no reduction, so each
//! output element's error is simply the stored value's own rounding error:
//! `<= 2^-8 * |value|` for bf16 (7 explicit mantissa bits), `<= 2^-11 *
//! |value|` for f16 (10 bits), plus a small absolute floor for near-zero
//! values. `moe_linear_gated` is `matmul.wgsl`'s exact math with a per-row
//! gate early-exit, so it reuses `bf16_roundtrip.rs`/`f16_roundtrip.rs`'s own
//! per-output-element sum-of-absolute-terms bound directly - the row-gating
//! only ever ZEROES a row (no reduction happens for it at all), never changes
//! the arithmetic of a live row.
//!
//! `embed_tile.wgsl` (the vocab-chunked sibling) is exercised directly
//! against the raw kernel (not through `Ops`, which only wraps the
//! single-binding `embed.wgsl` - see `model::ops`'s own doc comment on why
//! `embed_tile`'s extra `v0`/`v_count` tiling parameters make it a poor fit
//! for that one generic method) - the same "prove the templated KERNEL
//! itself round-trips" standard, just dispatched by hand.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{Ops, Weight};

/// The full façade kernel set `Ops::new` requires (mirrors
/// `model::ops::tests::kernel_list` and `bf16_roundtrip.rs`'s own copy),
/// extended with the B8 `embed`/`moe_linear_gated` bf16/f16 variants this
/// file's tests dispatch.
fn kernel_list() -> Vec<(&'static str, &'static str)> {
    let dv = kernels::template::dtype_variant;
    let bf16_matmul = dv("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
    let bf16_gemv = dv("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
    let bf16_reg3 = dv("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
    let f16_matmul = dv("matmul", kernels::MATMUL, "w", Dtype::F16).unwrap();
    let f16_gemv = dv("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::F16).unwrap();
    let f16_reg3 = dv("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::F16).unwrap();
    let bf16_embed = dv("embed", kernels::EMBED, "emb", Dtype::BF16).unwrap();
    let f16_embed = dv("embed", kernels::EMBED, "emb", Dtype::F16).unwrap();
    let bf16_moe = dv("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::BF16).unwrap();
    let f16_moe = dv("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::F16).unwrap();
    // B9: paged-KV append (write direction)/scores/apply (read direction)
    // bf16 tiers - additive, mechanical, same fix this file's own comment
    // above already needed when B8 landed.
    let bf16_kv_append = kernels::template::dtype_variant_store(
        "paged_kv_append_batched_word",
        kernels::PAGED_KV_APPEND_BATCHED_WORD,
        "pool",
        Dtype::BF16,
    )
    .unwrap();
    let bf16_decode_scores =
        dv("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED, "pool_k", Dtype::BF16).unwrap();
    let bf16_decode_apply =
        dv("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED, "pool_v", Dtype::BF16).unwrap();
    // B10: matmul_dx's bf16-weight-read backward variant. matmul_dw has no
    // bf16 variant at all (it never reads the weight).
    let bf16_matmul_dx = dv("matmul_dx", kernels::MATMUL_DX, "w", Dtype::BF16).unwrap();
    vec![
        ("matmul", kernels::MATMUL),
        ("matmul_gemv", kernels::MATMUL_GEMV),
        ("matmul_reg2", kernels::MATMUL_REG2),
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
        ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
        ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
        ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
        ("max_abs_row", kernels::MAX_ABS_ROW),
        ("quant_pack", kernels::QUANT_PACK),
        bf16_matmul,
        bf16_gemv,
        bf16_reg3,
        f16_matmul,
        f16_gemv,
        f16_reg3,
        ("embed", kernels::EMBED),
        bf16_embed,
        f16_embed,
        ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
        bf16_moe,
        f16_moe,
        ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
        bf16_kv_append,
        ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
        bf16_decode_scores,
        ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
        bf16_decode_apply,
        ("matmul_dx", kernels::MATMUL_DX),
        ("matmul_dw", kernels::MATMUL_DW),
        bf16_matmul_dx,
    ]
}

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

// ---------------------------------------------------------------- embed ---

fn host_embed(tokens: &[u32], table: &[f32], seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut out = vec![0f32; seq_len * d_model];
    for t in 0..seq_len {
        let row = tokens[t] as usize;
        out[t * d_model..(t + 1) * d_model].copy_from_slice(&table[row * d_model..(row + 1) * d_model]);
    }
    out
}

/// Per-element tolerance for a pure gather+dequant: no reduction, so the only
/// error is the stored value's own rounding. `bits` is the tier's explicit
/// mantissa bit count (7 for bf16, 10 for f16).
fn embed_tol(table: &[f32], tokens: &[u32], seq_len: usize, d_model: usize, bits: i32) -> Vec<f32> {
    let mut tol = vec![0f32; seq_len * d_model];
    for t in 0..seq_len {
        let row = tokens[t] as usize;
        for c in 0..d_model {
            let v = table[row * d_model + c] as f64;
            tol[t * d_model + c] = (v.abs() * 2f64.powi(-(bits + 1))) as f32 + 1e-5;
        }
    }
    tol
}

fn check_embed(gpu: Gpu, dt: Dtype, seq_len: usize, vocab: usize, d_model: usize, seed: u64, label: &str) {
    let bits = match dt {
        Dtype::BF16 => 7,
        Dtype::F16 => 10,
        other => panic!("check_embed: unexpected tier {other:?}"),
    };
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let mut rng = Lcg::new(seed);
    let table_h = rng.vec_scaled(vocab * d_model, 1.0);
    let tokens_h: Vec<u32> = (0..seq_len).map(|i| (i * 7 + 3) as u32 % vocab as u32).collect();

    let table = Weight::upload(&ops, &table_h, vocab, d_model, dt);
    assert_eq!(table.dtype(), dt, "{label}: device must report storage support for {dt:?}");

    let tokens = g.storage(seq_len as u64);
    g.write(&tokens, &tokens_h);
    let out = g.storage((seq_len * d_model) as u64);

    let mut steps = Vec::new();
    ops.embed(&mut steps, &table, &tokens, seq_len as u32, &out);
    g.submit(&[], &steps);
    let got = g.read(&out, seq_len * d_model);

    let want = host_embed(&tokens_h, &table_h, seq_len, d_model);
    let tol = embed_tol(&table_h, &tokens_h, seq_len, d_model, bits);
    assert_eq!(got.len(), want.len());
    let mut worst: f32 = 0.0;
    for i in 0..got.len() {
        let err = (got[i] - want[i]).abs();
        worst = worst.max(err / tol[i].max(1e-12));
        assert!(err <= tol[i], "{label}: elem {i} got {} want {} (err {err}, tol {})", got[i], want[i], tol[i]);
    }
    eprintln!("{label}: worst err/tol ratio {worst:.4}");
}

#[test]
fn embed_bf16_and_f16_match_f32_reference_on_cpu() {
    for dt in [Dtype::BF16, Dtype::F16] {
        let gpu = Gpu::new_cpu(&kernel_list());
        check_embed(gpu, dt, 17, 97, 32, 0xE3BED ^ dt as u64, &format!("cpu/embed/{dt:?}"));
    }
}

#[test]
fn embed_bf16_and_f16_match_f32_reference_on_gpu() {
    if skip_gpu() {
        eprintln!("embed_bf16_and_f16_match_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("embed_bf16_and_f16_match_f32_reference_on_gpu: running on a real wgpu device");
    for dt in [Dtype::BF16, Dtype::F16] {
        let gpu = Gpu::new_wgpu(&kernel_list());
        check_embed(gpu, dt, 17, 97, 32, 0xE3BED ^ dt as u64, &format!("gpu/embed/{dt:?}"));
    }
}

// ----------------------------------------------------------- embed_tile ---

/// `embed_tile.wgsl` dispatched directly (not through `Ops` - see this
/// file's module doc). `emb` here is only the CURRENT TILE's rows
/// (`[v0, v0+v_count)`), matching the kernel's own contract - a token outside
/// the tile is skipped (left however `out` was cleared, all zero here).
fn check_embed_tile(gpu: Gpu, dt: Dtype, seq_len: usize, d_model: usize, v0: u32, v_count: u32, seed: u64, label: &str) {
    let bits = match dt {
        Dtype::BF16 => 7,
        Dtype::F16 => 10,
        other => panic!("check_embed_tile: unexpected tier {other:?}"),
    };
    let (name, _src) = kernels::template::dtype_variant("embed_tile", kernels::EMBED_TILE, "emb", dt).unwrap();
    let idx = gpu.kernel_index(name).unwrap_or_else(|| panic!("{name} not registered"));

    let mut rng = Lcg::new(seed);
    let tile_h = rng.vec_scaled(v_count as usize * d_model, 1.0);
    // Every token deliberately falls inside this tile, so every output row is
    // written (a token outside the tile is a legitimate no-op this test does
    // not need to also cover - `embed.wgsl`'s own roundtrip already proves
    // the decode expression itself; this proves the SECOND templated kernel
    // shares it correctly).
    let tokens_h: Vec<u32> = (0..seq_len).map(|i| v0 + (i as u32 * 3 + 1) % v_count).collect();

    let packed = match dt {
        Dtype::BF16 => model::half::pack_bf16(&tile_h),
        Dtype::F16 => model::half::pack_f16(&tile_h),
        _ => unreachable!(),
    };
    let emb = gpu.storage(packed.len() as u64);
    gpu.write(&emb, &packed);
    let tokens = gpu.storage(seq_len as u64);
    gpu.write(&tokens, &tokens_h);
    let out = gpu.storage((seq_len * d_model) as u64);

    let steps = [gpu.step(idx, &[&tokens, &emb, &out], &[d_model as u32, seq_len as u32, v0, v_count], seq_len as u32 * d_model as u32)];
    gpu.submit(&[], &steps);
    let got = gpu.read(&out, seq_len * d_model);

    let mut want = vec![0f32; seq_len * d_model];
    let mut tol = vec![1e-5f32; seq_len * d_model];
    for (t, &tok) in tokens_h.iter().enumerate() {
        let row = (tok - v0) as usize;
        for c in 0..d_model {
            let v = tile_h[row * d_model + c];
            want[t * d_model + c] = v;
            tol[t * d_model + c] = (v as f64).abs() as f32 * 2f32.powi(-(bits + 1)) + 1e-5;
        }
    }
    for i in 0..got.len() {
        let err = (got[i] - want[i]).abs();
        assert!(err <= tol[i], "{label}: elem {i} got {} want {} (err {err}, tol {})", got[i], want[i], tol[i]);
    }
}

#[test]
fn embed_tile_bf16_and_f16_match_f32_reference_on_cpu() {
    for dt in [Dtype::BF16, Dtype::F16] {
        let (n0, s0) = kernels::template::dtype_variant("embed_tile", kernels::EMBED_TILE, "emb", dt).unwrap();
        let gpu = Gpu::new_cpu(&[(n0, s0)]);
        check_embed_tile(gpu, dt, 11, 24, 100, 40, 0x7113 ^ dt as u64, &format!("cpu/embed_tile/{dt:?}"));
    }
}

#[test]
fn embed_tile_bf16_and_f16_match_f32_reference_on_gpu() {
    if skip_gpu() {
        eprintln!("embed_tile_bf16_and_f16_match_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("embed_tile_bf16_and_f16_match_f32_reference_on_gpu: running on a real wgpu device");
    for dt in [Dtype::BF16, Dtype::F16] {
        let (n0, s0) = kernels::template::dtype_variant("embed_tile", kernels::EMBED_TILE, "emb", dt).unwrap();
        let gpu = Gpu::new_wgpu(&[(n0, s0)]);
        check_embed_tile(gpu, dt, 11, 24, 100, 40, 0x7113 ^ dt as u64, &format!("gpu/embed_tile/{dt:?}"));
    }
}

// ------------------------------------------------------ moe_linear_gated ---

fn host_moe_linear(x: &[f32], w: &[f32], gate: &[f32], m: usize, k: usize, n: usize, n_experts: u32, e_idx: u32) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        if gate[r * n_experts as usize + e_idx as usize] <= 0.0 {
            continue;
        }
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[j * k + i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

/// Same per-output-element sum-of-absolute-terms bound `bf16_roundtrip.rs`/
/// `f16_roundtrip.rs` derive for plain matmul - the gate only ever ZEROES a
/// row (a live row's arithmetic is byte-for-byte `matmul.wgsl`'s), so a
/// non-routed row's tolerance is a small absolute floor (its true value IS
/// exactly zero) rather than the sum-of-terms bound.
fn moe_linear_tol(x: &[f32], w: &[f32], gate: &[f32], m: usize, k: usize, n: usize, n_experts: u32, e_idx: u32, bits: i32) -> Vec<f32> {
    let mut tol = vec![0f32; m * n];
    for r in 0..m {
        if gate[r * n_experts as usize + e_idx as usize] <= 0.0 {
            for j in 0..n {
                tol[r * n + j] = 1e-6;
            }
            continue;
        }
        for j in 0..n {
            let mut abs_sum = 0f64;
            for i in 0..k {
                abs_sum += (x[r * k + i] as f64 * w[j * k + i] as f64).abs();
            }
            tol[r * n + j] = (abs_sum * 2f64.powi(-(bits + 1))) as f32 + 1e-5;
        }
    }
    tol
}

#[allow(clippy::too_many_arguments)]
fn check_moe_linear(gpu: Gpu, dt: Dtype, m: usize, n: usize, k: usize, n_experts: u32, e_idx: u32, seed: u64, label: &str) {
    let bits = match dt {
        Dtype::BF16 => 7,
        Dtype::F16 => 10,
        other => panic!("check_moe_linear: unexpected tier {other:?}"),
    };
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let mut rng = Lcg::new(seed);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);
    // Deterministic, non-degenerate routing: every other row is routed to
    // `e_idx`, the rest to a different (unmodelled) expert.
    let gate_h: Vec<f32> = (0..m * n_experts as usize)
        .map(|i| {
            let row = i / n_experts as usize;
            let col = i % n_experts as usize;
            if col as u32 == e_idx && row.is_multiple_of(2) {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let x = g.storage_init("x", &x_h);
    let gate = g.storage_init("gate", &gate_h);
    let weight = Weight::upload(&ops, &w_h, n, k, dt);
    assert_eq!(weight.dtype(), dt, "{label}: device must report storage support for {dt:?}");
    let out = g.storage((m * n) as u64);

    let mut steps = Vec::new();
    ops.moe_linear(&mut steps, &weight, &x, &gate, n_experts, e_idx, m as u32, &out);
    g.submit(&[], &steps);
    let got = g.read(&out, m * n);

    let want = host_moe_linear(&x_h, &w_h, &gate_h, m, k, n, n_experts, e_idx);
    let tol = moe_linear_tol(&x_h, &w_h, &gate_h, m, k, n, n_experts, e_idx, bits);
    assert_eq!(got.len(), want.len());
    let mut worst: f32 = 0.0;
    for i in 0..got.len() {
        let err = (got[i] - want[i]).abs();
        worst = worst.max(err / tol[i].max(1e-12));
        assert!(err <= tol[i], "{label}: elem {i} got {} want {} (err {err}, tol {})", got[i], want[i], tol[i]);
    }
    eprintln!("{label}: worst err/tol ratio {worst:.4}");
}

#[test]
fn moe_linear_bf16_and_f16_match_f32_reference_on_cpu() {
    for dt in [Dtype::BF16, Dtype::F16] {
        let gpu = Gpu::new_cpu(&kernel_list());
        check_moe_linear(gpu, dt, 12, 20, 16, 4, 2, 0x0E0E ^ dt as u64, &format!("cpu/moe_linear/{dt:?}"));
    }
}

#[test]
fn moe_linear_bf16_and_f16_match_f32_reference_on_gpu() {
    if skip_gpu() {
        eprintln!("moe_linear_bf16_and_f16_match_f32_reference_on_gpu: SKIPPED (MOE_SKIP_GPU_TESTS set)");
        return;
    }
    eprintln!("moe_linear_bf16_and_f16_match_f32_reference_on_gpu: running on a real wgpu device");
    for dt in [Dtype::BF16, Dtype::F16] {
        let gpu = Gpu::new_wgpu(&kernel_list());
        check_moe_linear(gpu, dt, 12, 20, 16, 4, 2, 0x0E0E ^ dt as u64, &format!("gpu/moe_linear/{dt:?}"));
    }
}
