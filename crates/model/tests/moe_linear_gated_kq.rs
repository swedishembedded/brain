// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M13: `moe_linear_gated_kq.wgsl` (the affine K-quant, Q4_K/Q5_K, MoE-gated
//! expert linear) and `model::moe::expert_fwd_kq`'s wiring of it.
//!
//! Swedish Embedded AB implements quantized inference kernels for edge and
//! embedded GPUs for its clients. If your team needs expertise in shipping
//! affine K-quant (GGUF Q4_K/Q5_K-class) sparse-MoE inference on commodity
//! GPU hardware then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Gate ladder:
//!
//! (b) `moe_linear_gated_kq` with every row routed to ONE expert must match a
//!     plain `matmul_kq_dyn`/`matmul_kq_gemv` dispatch on the SAME weight, at
//!     the SAME tolerance `matmul_kq.rs` (M11) established
//!     (`rel_l2 <= 1e-6`, `cosine >= 1 - 1e-9`, `max_rel <= 5e-4`), both
//!     `CODE_BITS`. Both are compared against an independent f64 host oracle
//!     built directly from int8 codes, so this is not merely kernel-vs-kernel
//!     agreement (which cannot tell you which one is wrong).
//! (w) `model::moe::expert_fwd_kq`/`MoeIdsKQ`/`LinKQ` wiring: a non-routed row
//!     (gate column 0) writes exactly zero, and two experts routed to every
//!     row combine by plain f32 ADDITION in the SAME order `scale_add.wgsl`'s
//!     own `accumulate` branch performs - so this is `assert_eq!` on the
//!     bits, not a tolerance, and catches a swapped/misrouted buffer or a
//!     wrong `k`/`n` param the way a purely-finite smoke test could not.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use kernels::template::interned;
use model::moe::{expert_fwd_kq, ExpertScratchKQ, LinKQ, MoeIdsKQ, MoeShape};

// ---------------------------------------------------------------------
// Deterministic generators - the same conventions `matmul_kq.rs` (M11) uses.
// ---------------------------------------------------------------------

fn rand_i8_codes(seed: u64, n: usize, lo: i32, hi: i32) -> Vec<i8> {
    let mut r = Lcg::new(seed);
    let span = (hi - lo + 1) as u32;
    (0..n).map(|_| (lo + (r.next_u32() % span) as i32) as i8).collect()
}

fn rand_unsigned_codes(seed: u64, n: usize, bits: u32) -> Vec<i32> {
    let span = (1u32 << bits).min(32);
    let mut r = Lcg::new(seed);
    (0..n).map(|_| (r.next_u32() % span) as i32).collect()
}

fn rand_pos_f32(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| lo + (r.next_u32() % 100_000) as f32 / 100_000.0 * (hi - lo)).collect()
}

fn pack_i8_words(codes: &[i8]) -> Vec<u32> {
    assert_eq!(codes.len() % 4, 0);
    codes.chunks_exact(4).map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u8 as u32) << (8 * b)))).collect()
}

fn pack_affine_words(codes: &[i32], bits: u32) -> Vec<u32> {
    let per_word = 32 / bits as usize;
    assert_eq!(codes.len() % per_word, 0);
    let mask = (1u32 << bits) - 1;
    codes
        .chunks_exact(per_word)
        .map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u32 & mask) << (bits * b as u32))))
        .collect()
}

/// M14: groups per `wd` super-block entry - `matmul_kq_dyn.wgsl`'s own fixed
/// `GPS=8` (256-element GGUF Q4_K/Q5_K super-block / 32-element group).
const GPS: usize = 8;

fn pack_wsm(sc: &[u8], mn: &[u8], n: usize, ng: usize) -> Vec<u32> {
    let words = ng.div_ceil(2);
    let mut out = vec![0u32; n * words];
    for r in 0..n {
        for g in 0..ng {
            let (w, shift) = (g / 2, if g % 2 == 0 { 0u32 } else { 16u32 });
            out[r * words + w] |= (sc[r * ng + g] as u32) << shift;
            out[r * words + w] |= (mn[r * ng + g] as u32) << (shift + 8);
        }
    }
    out
}

fn pack_wd(d: &[f32], dmin: &[f32], n: usize, spb_per_row: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * spb_per_row];
    for r in 0..n {
        for s in 0..spb_per_row {
            let db = half::f16::from_f32(d[r * spb_per_row + s]).to_bits() as u32;
            let dmb = half::f16::from_f32(dmin[r * spb_per_row + s]).to_bits() as u32;
            out[r * spb_per_row + s] = db | (dmb << 16);
        }
    }
    out
}

/// `Gpu::storage_init` is f32-only; `wsm`/`wd` are packed `u32` words, so
/// upload them through `storage` + `write` instead.
fn storage_u32(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let b = g.storage(data.len() as u64);
    g.write(&b, data);
    b
}

/// The decomposed `(sc, mn, d_super, dmin_super)` -> effective `(ds, dm)`
/// product `wsm`/`wd` reconstruct - restated from `crates/model/tests/
/// matmul_kq.rs`'s own `effective_ds_dm` (integration test binaries share no
/// code across files in this crate).
fn effective_ds_dm(sc: &[u8], mn: &[u8], d_super: &[f32], dmin_super: &[f32], n: usize, ng: usize) -> (Vec<f32>, Vec<f32>) {
    let spb_per_row = ng.div_ceil(GPS);
    let mut ds = vec![0f32; n * ng];
    let mut dm = vec![0f32; n * ng];
    for r in 0..n {
        for g in 0..ng {
            let s = r * spb_per_row + g / GPS;
            let dv = half::f16::from_f32(d_super[s]).to_f32();
            let dmv = half::f16::from_f32(dmin_super[s]).to_f32();
            ds[r * ng + g] = dv * sc[r * ng + g] as f32;
            dm[r * ng + g] = dmv * mn[r * ng + g] as f32;
        }
    }
    (ds, dm)
}

/// Random `(sc, mn, d_super, dmin_super)`, mirroring `matmul_kq.rs`'s own
/// `rand_kq_scale`.
fn rand_kq_scale(seed: u64, n: usize, ng: usize, dmin_nonzero: bool) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let spb = ng.div_ceil(GPS);
    let sc: Vec<u8> = rand_pos_f32(seed ^ 0x54, n * ng, 1.0, 63.0).into_iter().map(|v| v as u8).collect();
    let mn: Vec<u8> = if dmin_nonzero {
        rand_pos_f32(seed ^ 0x55, n * ng, 1.0, 63.0).into_iter().map(|v| v as u8).collect()
    } else {
        vec![0u8; n * ng]
    };
    let d_super = rand_pos_f32(seed ^ 0x56, n * spb, 0.001, 0.02);
    let dmin_super = if dmin_nonzero { rand_pos_f32(seed ^ 0x57, n * spb, 0.002, 0.04) } else { vec![0f32; n * spb] };
    (sc, mn, d_super, dmin_super)
}

fn host_group_sums(codes_x: &[i8], m: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; m * ng];
    for r in 0..m {
        for g in 0..ng {
            let s: i32 = codes_x[r * k + g * group..r * k + (g + 1) * group].iter().map(|&v| v as i32).sum();
            out[r * ng + g] = s as f32;
        }
    }
    out
}

/// The f64 host oracle - identical to `matmul_kq.rs`'s own
/// `host_oracle_affine`, duplicated here (integration test binaries share no
/// code across files in this crate).
#[allow(clippy::too_many_arguments)]
fn host_oracle_affine(codes_x: &[i8], sx: &[f32], codes_w: &[i32], ds: &[f32], dm: &[f32], m: usize, k: usize, n: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for g in 0..ng {
                let mut a = 0i64;
                let mut s = 0i64;
                for j in 0..group {
                    let kk = g * group + j;
                    let xv = i64::from(codes_x[r * k + kk]);
                    a += i64::from(codes_w[c * k + kk]) * xv;
                    s += xv;
                }
                acc += a as f64 * f64::from(ds[c * ng + g]) - s as f64 * f64::from(dm[c * ng + g]);
            }
            out[r * n + c] = (acc * f64::from(sx[r])) as f32;
        }
    }
    out
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = f64::from(g) - f64::from(w);
        num += d * d;
        den += f64::from(w) * f64::from(w);
    }
    (num / den.max(1e-300)).sqrt()
}

fn cosine(got: &[f32], want: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        d += f64::from(g) * f64::from(w);
        na += f64::from(g) * f64::from(g);
        nb += f64::from(w) * f64::from(w);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    got.iter().zip(want).map(|(&g, &w)| (f64::from(g) - f64::from(w)).abs() / f64::from(w).abs().max(1.0)).fold(0.0, f64::max)
}

/// M11's own tolerance (`matmul_kq.rs`'s `assert_matches_oracle`), restated:
/// the affine correction's two-reduction fold measured `rel_l2` in
/// `8e-8..1.8e-7` and `cosine` at `1.0` to 12 decimals on a real device, with
/// `max_rel` up to `2.9e-4` - `5e-4` keeps real margin while staying far
/// below a magnitude that would indicate a genuine wrong group/index.
fn assert_matches_oracle(what: &str, got: &[f32], want: &[f32]) {
    let l2 = rel_l2(got, want);
    let cos = cosine(got, want);
    let mr = max_rel(got, want);
    assert!(l2 <= 1e-6, "{what}: rel_l2 {l2:e} > 1e-6");
    assert!(cos >= 1.0 - 1e-9, "{what}: cosine {cos:.12} < 1 - 1e-9");
    assert!(mr <= 5e-4, "{what}: max relative error {mr:e} > 5e-4");
    for (i, &v) in got.iter().enumerate() {
        assert!(v.is_finite(), "{what}: got[{i}] = {v} is not finite");
    }
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

// =======================================================================
// Rung (b): moe_linear_gated_kq (all rows routed to one expert) vs
// matmul_kq_dyn / matmul_kq_gemv, both compared to the SAME f64 host oracle.
// =======================================================================

#[test]
fn moe_matches_dyn_and_gemv_when_fully_routed_both_code_bits() {
    for &bits in &[4u32, 8u32] {
        let (m, k, n, group, n_experts, e_idx) = (6usize, 256usize, 48usize, 32usize, 3u32, 1u32);
        let ng = k / group;

        let mut kernels_list = Vec::new();
        let (moe_name, moe_src) = interned("moe_linear_gated_kq", kernels::MOE_LINEAR_GATED_KQ, &[("CODE_BITS", bits)]).unwrap();
        let (dyn_name, dyn_src) = interned("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", bits)]).unwrap();
        let (gemv_name, gemv_src) = interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", bits)]).unwrap();
        kernels_list.push((moe_name, moe_src));
        kernels_list.push((dyn_name, dyn_src));
        kernels_list.push((gemv_name, gemv_src));
        let g = Gpu::new(&kernels_list);

        let codes_x = rand_i8_codes(0xB0_0000 | bits as u64, m * k, -100, 100);
        let sx = rand_pos_f32(0xB0_0001 | bits as u64, m, 0.3, 1.7);
        let codes_w = rand_unsigned_codes(0xB0_0002 | bits as u64, n * k, bits);
        let (sc_bytes, mn, d_super, dmin_super) = rand_kq_scale(0xB0_0003 | bits as u64, n, ng, true);
        let (ds, dm) = effective_ds_dm(&sc_bytes, &mn, &d_super, &dmin_super, n, ng);
        let want = host_oracle_affine(&codes_x, &sx, &codes_w, &ds, &dm, m, k, n, group);

        let xq_words = pack_i8_words(&codes_x);
        let xq = g.storage(xq_words.len() as u64);
        g.write(&xq, &xq_words);
        let wq_words = pack_affine_words(&codes_w, bits);
        let wq = g.storage(wq_words.len() as u64);
        g.write(&wq, &wq_words);
        let sxb = g.storage_init("sx", &sx);
        let wsm = storage_u32(&g, &pack_wsm(&sc_bytes, &mn, n, ng));
        let wd = storage_u32(&g, &pack_wd(&d_super, &dmin_super, n, ng.div_ceil(GPS)));
        let xgs = g.storage_init("xgs", &host_group_sums(&codes_x, m, k, group));

        // Every row routed to expert `e_idx` (n_experts columns, only the
        // e_idx one read); the OTHER columns are garbage on purpose - the
        // kernel must never read them.
        let mut gate_h = rand_pos_f32(0xB0_0005 | bits as u64, m * n_experts as usize, -9.0, -1.0);
        for r in 0..m {
            gate_h[r * n_experts as usize + e_idx as usize] = 1.0;
        }
        let gate = g.storage_init("gate", &gate_h);

        let out_moe = g.storage((m * n) as u64);
        let params = [m as u32, k as u32, n as u32, n_experts, e_idx];
        g.submit(&[], &[g.step(idx(&g, moe_name), &[&xq, &wq, &sxb, &wsm, &wd, &xgs, &gate, &out_moe], &params, (m * n) as u32)]);
        let got_moe = g.read(&out_moe, m * n);

        let tile_threads = (m as u32).div_ceil(128) * (n as u32).div_ceil(128) * 256;
        let out_dyn = g.storage((m * n) as u64);
        g.submit(&[], &[g.step(idx(&g, dyn_name), &[&xq, &wq, &sxb, &wsm, &wd, &xgs, &out_dyn], &[m as u32, k as u32, n as u32], tile_threads)]);
        let got_dyn = g.read(&out_dyn, m * n);

        let out_gemv = g.storage((m * n) as u64);
        g.submit(&[], &[g.step(idx(&g, gemv_name), &[&xq, &wq, &sxb, &wsm, &wd, &xgs, &out_gemv], &[m as u32, k as u32, n as u32], n as u32 * 64)]);
        let got_gemv = g.read(&out_gemv, m * n);

        assert_matches_oracle(&format!("moe_linear_gated_kq vs oracle CODE_BITS={bits}"), &got_moe, &want);
        assert_matches_oracle(&format!("matmul_kq_dyn vs oracle CODE_BITS={bits}"), &got_dyn, &want);
        assert_matches_oracle(&format!("matmul_kq_gemv vs oracle CODE_BITS={bits}"), &got_gemv, &want);
    }
}

// =======================================================================
// Rung (w): model::moe::expert_fwd_kq / MoeIdsKQ / LinKQ wiring.
// =======================================================================

const KERNELS_KQ: &[(&str, &str)] = &[
    ("moe_linear_gated_kq", kernels::MOE_LINEAR_GATED_KQ),
    ("silu_mul", kernels::SILU_MUL),
    ("scale_add", kernels::SCALE_ADD),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("quant_group_sum", kernels::QUANT_GROUP_SUM),
];

struct KqExpert {
    wq: DeviceBuffer,
    wsm: DeviceBuffer,
    wd: DeviceBuffer,
}

fn upload_kq_expert(g: &Gpu, seed: u64, n: usize, k: usize, group: usize) -> KqExpert {
    let ng = k / group;
    let codes_w = rand_unsigned_codes(seed, n * k, 8);
    let (sc, mn, d_super, dmin_super) = rand_kq_scale(seed, n, ng, true);
    let wq_words = pack_affine_words(&codes_w, 8);
    let wq = g.storage(wq_words.len() as u64);
    g.write(&wq, &wq_words);
    let wsm = storage_u32(g, &pack_wsm(&sc, &mn, n, ng));
    let wd = storage_u32(g, &pack_wd(&d_super, &dmin_super, n, ng.div_ceil(GPS)));
    KqExpert { wq, wsm, wd }
}

fn run_expert(
    g: &Gpu,
    ids: &MoeIdsKQ,
    shape: &MoeShape,
    xq: &DeviceBuffer,
    sx: &DeviceBuffer,
    xgs: &DeviceBuffer,
    gate: &DeviceBuffer,
    gate_w: &KqExpert,
    up_w: &KqExpert,
    down_w: &KqExpert,
    e_idx: u32,
) -> Vec<f32> {
    let (m, d, ff) = (shape.rows, shape.d_model, shape.moe_ff);
    let scratch = ExpertScratchKQ {
        gate_pre: &g.storage((m * ff) as u64),
        up: &g.storage((m * ff) as u64),
        h: &g.storage((m * ff) as u64),
        hq: &g.storage((m * ff / 4) as u64),
        sh: &g.storage(m as u64),
        hgs: &g.storage((m * ff / 32) as u64),
        expert_out: &g.storage((m * d) as u64),
    };
    let acc = g.storage((m * d) as u64);
    let steps = expert_fwd_kq(
        g,
        ids,
        shape,
        xq,
        sx,
        xgs,
        gate,
        LinKQ { wq: &gate_w.wq, wsm: &gate_w.wsm, wd: &gate_w.wd },
        LinKQ { wq: &up_w.wq, wsm: &up_w.wsm, wd: &up_w.wd },
        LinKQ { wq: &down_w.wq, wsm: &down_w.wsm, wd: &down_w.wd },
        &scratch,
        &acc,
        e_idx,
        false,
    );
    g.submit(&[], &steps);
    g.read(&acc, (m * d) as usize)
}

/// A row not routed to its ONLY expert (gate column 0) writes EXACTLY zero -
/// the row-level skip `moe_linear_gated_kq.wgsl`'s own header describes,
/// reached through the real `expert_fwd_kq` wiring rather than a direct
/// kernel dispatch.
#[test]
fn expert_fwd_kq_zeroes_a_non_routed_row() {
    let g = gpu_core::testgpu::dev(KERNELS_KQ);
    let shape = MoeShape { rows: 4, d_model: 64, moe_ff: 32, n_experts: 1, top_k: 1 };
    let ids = MoeIdsKQ {
        linear_gated_kq: idx(&g, "moe_linear_gated_kq"),
        silu_mul: idx(&g, "silu_mul"),
        scale_add: idx(&g, "scale_add"),
        quant: [idx(&g, "max_abs_row"), idx(&g, "quant_pack")],
        quant_group_sum: idx(&g, "quant_group_sum"),
    };
    let (m, d) = (shape.rows as usize, shape.d_model as usize);
    let codes_x = rand_i8_codes(0x7770, m * d, -100, 100);
    let xq_words = pack_i8_words(&codes_x);
    let xq = g.storage(xq_words.len() as u64);
    g.write(&xq, &xq_words);
    let sx = g.storage_init("sx", &rand_pos_f32(0x7771, m, 0.3, 1.7));
    let xgs = g.storage_init("xgs", &host_group_sums(&codes_x, m, d, 32));

    // Row 0 is NOT routed (gate = 0); every other row IS.
    let mut gate_h = vec![1.0f32; m];
    gate_h[0] = 0.0;
    let gate = g.storage_init("gate", &gate_h);

    let gate_w = upload_kq_expert(&g, 0x7772, shape.moe_ff as usize, d, 32);
    let up_w = upload_kq_expert(&g, 0x7773, shape.moe_ff as usize, d, 32);
    let down_w = upload_kq_expert(&g, 0x7774, d, shape.moe_ff as usize, 32);

    let out = run_expert(&g, &ids, &shape, &xq, &sx, &xgs, &gate, &gate_w, &up_w, &down_w, 0);
    for c in 0..d {
        assert_eq!(out[c], 0.0, "row 0 (gate=0) must write exactly zero at column {c}");
    }
    assert!(out[d..].iter().any(|&v| v != 0.0), "the routed rows must not ALSO be all-zero (a trivial pass)");
}

/// Two experts, both routed to every row, must combine by plain f32 ADDITION
/// in the SAME order `scale_add.wgsl`'s own `accumulate` branch performs
/// (`acc = contrib` then `acc = acc + contrib`) - `assert_eq!` on the bits,
/// not a tolerance, since IEEE addition of the SAME two f32 operands in the
/// SAME order is deterministic. Catches a swapped buffer or a mis-threaded
/// `accumulate` flag the way a purely-finite smoke test could not.
#[test]
fn expert_fwd_kq_combines_two_experts_by_plain_addition() {
    let g = gpu_core::testgpu::dev(KERNELS_KQ);
    let shape = MoeShape { rows: 4, d_model: 64, moe_ff: 32, n_experts: 2, top_k: 2 };
    let ids = MoeIdsKQ {
        linear_gated_kq: idx(&g, "moe_linear_gated_kq"),
        silu_mul: idx(&g, "silu_mul"),
        scale_add: idx(&g, "scale_add"),
        quant: [idx(&g, "max_abs_row"), idx(&g, "quant_pack")],
        quant_group_sum: idx(&g, "quant_group_sum"),
    };
    let (m, d) = (shape.rows as usize, shape.d_model as usize);
    let codes_x = rand_i8_codes(0x8880, m * d, -100, 100);
    let xq_words = pack_i8_words(&codes_x);
    let xq = g.storage(xq_words.len() as u64);
    g.write(&xq, &xq_words);
    let sx = g.storage_init("sx", &rand_pos_f32(0x8881, m, 0.3, 1.7));
    let xgs = g.storage_init("xgs", &host_group_sums(&codes_x, m, d, 32));
    // Both columns routed for every row.
    let gate = g.storage_init("gate", &vec![1.0f32; m * shape.n_experts as usize]);

    let e0_gate_w = upload_kq_expert(&g, 0x8882, shape.moe_ff as usize, d, 32);
    let e0_up_w = upload_kq_expert(&g, 0x8883, shape.moe_ff as usize, d, 32);
    let e0_down_w = upload_kq_expert(&g, 0x8884, d, shape.moe_ff as usize, 32);
    let e1_gate_w = upload_kq_expert(&g, 0x8885, shape.moe_ff as usize, d, 32);
    let e1_up_w = upload_kq_expert(&g, 0x8886, shape.moe_ff as usize, d, 32);
    let e1_down_w = upload_kq_expert(&g, 0x8887, d, shape.moe_ff as usize, 32);

    let out0 = run_expert(&g, &ids, &shape, &xq, &sx, &xgs, &gate, &e0_gate_w, &e0_up_w, &e0_down_w, 0);
    let out1 = run_expert(&g, &ids, &shape, &xq, &sx, &xgs, &gate, &e1_gate_w, &e1_up_w, &e1_down_w, 1);

    // Both experts into ONE accumulator, expert 0 first (accumulate=false),
    // expert 1 second (accumulate=true) - the real multi-expert call order.
    struct Owned {
        gate_pre: DeviceBuffer,
        up: DeviceBuffer,
        h: DeviceBuffer,
        hq: DeviceBuffer,
        sh: DeviceBuffer,
        hgs: DeviceBuffer,
        expert_out: DeviceBuffer,
    }
    let new_owned = |g: &Gpu| Owned {
        gate_pre: g.storage((m * shape.moe_ff as usize) as u64),
        up: g.storage((m * shape.moe_ff as usize) as u64),
        h: g.storage((m * shape.moe_ff as usize) as u64),
        hq: g.storage((m * shape.moe_ff as usize / 4) as u64),
        sh: g.storage(m as u64),
        hgs: g.storage((m * shape.moe_ff as usize / 32) as u64),
        expert_out: g.storage((m * d) as u64),
    };
    fn as_scratch(o: &Owned) -> ExpertScratchKQ<'_> {
        ExpertScratchKQ { gate_pre: &o.gate_pre, up: &o.up, h: &o.h, hq: &o.hq, sh: &o.sh, hgs: &o.hgs, expert_out: &o.expert_out }
    }
    let owned0 = new_owned(&g);
    let owned1 = new_owned(&g);
    let acc = g.storage((m * d) as u64);
    let mut steps = expert_fwd_kq(
        &g,
        &ids,
        &shape,
        &xq,
        &sx,
        &xgs,
        &gate,
        LinKQ { wq: &e0_gate_w.wq, wsm: &e0_gate_w.wsm, wd: &e0_gate_w.wd },
        LinKQ { wq: &e0_up_w.wq, wsm: &e0_up_w.wsm, wd: &e0_up_w.wd },
        LinKQ { wq: &e0_down_w.wq, wsm: &e0_down_w.wsm, wd: &e0_down_w.wd },
        &as_scratch(&owned0),
        &acc,
        0,
        false,
    );
    steps.extend(expert_fwd_kq(
        &g,
        &ids,
        &shape,
        &xq,
        &sx,
        &xgs,
        &gate,
        LinKQ { wq: &e1_gate_w.wq, wsm: &e1_gate_w.wsm, wd: &e1_gate_w.wd },
        LinKQ { wq: &e1_up_w.wq, wsm: &e1_up_w.wsm, wd: &e1_up_w.wd },
        LinKQ { wq: &e1_down_w.wq, wsm: &e1_down_w.wsm, wd: &e1_down_w.wd },
        &as_scratch(&owned1),
        &acc,
        1,
        true,
    ));
    g.submit(&[], &steps);
    let combined = g.read(&acc, m * d);

    let want: Vec<f32> = out0.iter().zip(&out1).map(|(&a, &b)| a + b).collect();
    assert_eq!(combined, want, "two fully-routed experts must combine by plain f32 addition, same order scale_add.wgsl performs");
}
