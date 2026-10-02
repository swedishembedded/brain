// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The kernels of the gather-layout int8 sparse-MoE pipeline, each against the
//! plainest statement of what it computes.
//!
//! Swedish Embedded AB implements fast sparse-MoE inference for its clients. If
//! your team needs expertise in serving large mixture-of-experts models at the
//! bandwidth of the memory system then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! * `moe_i8_gemv_gather` - a slot reads ITS expert's rows out of one fused
//!   bank. The reference is a host loop that performs the kernel's documented
//!   arithmetic in its documented order (integer sum per 32-group, converted
//!   once, scaled, ascending lane accumulation, ascending 16-lane fold), so the
//!   comparison is on raw BITS, not a tolerance: the order is the contract a
//!   native kernel must keep.
//! * `moe_router_topk` - the same routing as `router_gate` +
//!   `router_topk_compact` (ids equal as sets, weights to a few ulp), plus the
//!   shared expert's sigmoid slot.
//! * `moe_swiglu_quant` - bit-identical to `silu_mul` -> `max_abs_row` ->
//!   `quant_pack`.
//! * `moe_slot_combine` - the weighted sum of a row's slots.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("moe_i8_gemv_gather", kernels::MOE_I8_GEMV_GATHER),
    ("moe_router_topk", kernels::MOE_ROUTER_TOPK),
    ("moe_swiglu_quant", kernels::MOE_SWIGLU_QUANT),
    ("moe_slot_combine", kernels::MOE_SLOT_COMBINE),
    ("silu_mul", kernels::SILU_MUL),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("router_gate", kernels::ROUTER_GATE),
    ("router_topk_compact", kernels::ROUTER_TOPK_COMPACT),
];
const K_GATHER: usize = 0;
const K_ROUTER: usize = 1;
const K_SWIGLU_QUANT: usize = 2;
const K_COMBINE: usize = 3;
const K_SILU_MUL: usize = 4;
const K_MAX_ABS: usize = 5;
const K_QUANT_PACK: usize = 6;
const K_ROUTER_GATE: usize = 7;
const K_TOPK_COMPACT: usize = 8;

fn words_of(v: &[u32]) -> Vec<u32> {
    v.to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn read_u32(gpu: &Gpu, buf: &gpu_core::DeviceBuffer, n: usize) -> Vec<u32> {
    gpu.read(buf, n).iter().map(|f| f.to_bits()).collect()
}

fn upload_u32(gpu: &Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, &words_of(v));
    b
}

fn upload_f32(gpu: &Gpu, v: &[f32]) -> gpu_core::DeviceBuffer {
    gpu.storage_init("test", v)
}

/// Signed bytes over the full range (including -128 and 127), packed four to a
/// word the way `dot4I8Packed` reads them, and the bytes themselves.
fn random_packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = Lcg::new(seed);
    (0..words)
        .map(|_| {
            let mut w = 0u32;
            for lane in 0..4 {
                w |= u32::from((r.next_u32() % 256) as u8) << (8 * lane);
            }
            w
        })
        .collect()
}

fn random_scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5).collect()
}

fn dot4(x: u32, w: u32) -> i32 {
    (0..4).map(|b| i32::from((x >> (8 * b)) as u8 as i8) * i32::from((w >> (8 * b)) as u8 as i8)).sum()
}

/// The kernel's documented arithmetic, in its documented order.
#[allow(clippy::too_many_arguments)]
fn gather_reference(slots: usize, kg: usize, n: usize, xdiv: usize, xq: &[u32], sx: &[f32], ids: &[u32], wq: &[u32], sw: &[f32]) -> Vec<f32> {
    let ng = kg / 8;
    let mut out = vec![0f32; slots * n];
    for s in 0..slots {
        for col in 0..n {
            let row_w = ids[s] as usize * n + col;
            let mut lane_acc = [0f32; 16];
            for g in 0..ng {
                let d: i32 = (0..8).map(|j| dot4(xq[(s / xdiv) * kg + g * 8 + j], wq[row_w * kg + g * 8 + j])).sum();
                let term = d as f32 * sw[row_w * ng + g];
                lane_acc[g % 16] += term;
            }
            let mut total = 0f32;
            for a in lane_acc {
                total += a;
            }
            out[s * n + col] = total * sx[s / xdiv];
        }
    }
    out
}

fn run_gather(gpu: &Gpu, slots: usize, k: usize, n: usize, xdiv: usize, blocks: usize, seed: u64) {
    let kg = k / 4;
    let rows = slots.div_ceil(xdiv);
    let xq = random_packed(rows * kg, seed + 1);
    let sx = random_scales(rows, seed + 2);
    let mut rng = Lcg::new(seed + 3);
    let ids: Vec<u32> = (0..slots).map(|_| rng.next_u32() % blocks as u32).collect();
    let wq = random_packed(blocks * n * kg, seed + 4);
    let sw = random_scales(blocks * n * (kg / 8), seed + 5);
    let want = gather_reference(slots, kg, n, xdiv, &xq, &sx, &ids, &wq, &sw);

    let (xb, sxb, idb, wb, swb) = (upload_u32(gpu, &xq), upload_f32(gpu, &sx), upload_u32(gpu, &ids), upload_u32(gpu, &wq), upload_f32(gpu, &sw));
    let out = gpu.storage((slots * n) as u64);
    gpu.write(&out, &vec![0x7fc0_dead; slots * n]);
    let blocks_launched = slots as u32 * (n as u32).div_ceil(4);
    gpu.submit(&[], &[gpu.dispatch(K_GATHER, &[&xb, &sxb, &idb, &wb, &swb, &out], &[slots as u32, kg as u32, n as u32, xdiv as u32], Dispatch::Workgroups(blocks_launched))]);
    gpu.poll_wait();
    let got = read_u32(gpu, &out, slots * n);
    let want = bits(&want);
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).take(5).collect();
    assert!(bad.is_empty(), "slots={slots} k={k} n={n} xdiv={xdiv}: first mismatches at {bad:?}: got {:?} want {:?}", bad.iter().map(|&i| f32::from_bits(got[i])).collect::<Vec<_>>(), bad.iter().map(|&i| f32::from_bits(want[i])).collect::<Vec<_>>());
}

/// Every slot reads its own expert's rows out of the bank, the activation row it
/// names, and nothing else - at the widths the real model uses (`k = 2048, n =
/// 512` gate/up, `k = 512, n = 2048` down), at ragged `n`, and with `xdiv` both
/// 1 (down: a row per slot) and > 1 (gate/up: a row per token).
#[test]
fn the_gather_gemv_is_bit_identical_to_its_documented_arithmetic() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("needs a packed int8 dot");
    }
    // (slots, k, n, xdiv, bank blocks)
    for (i, &(slots, k, n, xdiv, blocks)) in [
        (9, 2048, 512, 9, 257), // gate/up at the real shape, one token, 8 routed + 1 shared
        (9, 512, 2048, 1, 257), // down
        (27, 2048, 512, 9, 257),
        (5, 64, 6, 1, 3),   // ragged n: 6 is not a multiple of the 4-row tile
        (3, 32, 4, 1, 2),   // one 32-group: fewer groups than lanes
        (4, 544, 12, 2, 3), // 17 groups: one lane takes a second
        (1, 96, 1, 1, 1),
    ]
    .iter()
    .enumerate()
    {
        run_gather(&gpu, slots, k, n, xdiv, blocks, 1000 + i as u64);
    }
}

fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}

/// Same routing as the dense `router_gate` + `router_topk_compact` pair (the
/// kernels this replaces), and the shared slot.
#[test]
fn the_router_selects_and_weights_like_router_gate_and_adds_the_shared_slot() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("needs workgroup reductions");
    }
    for &(rows, experts, top_k) in &[(1usize, 256usize, 8usize), (7, 256, 8), (3, 64, 4), (2, 17, 5)] {
        let mut rng = Lcg::new(77 + rows as u64);
        let width = experts + 1;
        let logits: Vec<f32> = (0..rows * width).map(|_| (rng.next_u32() % 20000) as f32 * 1e-3 - 10.0).collect();
        // The dense kernels see only the routed columns.
        let routed: Vec<f32> = (0..rows).flat_map(|r| logits[r * width..r * width + experts].to_vec()).collect();

        let (lb, rb) = (upload_f32(&gpu, &logits), upload_f32(&gpu, &routed));
        let (ids, weight) = (gpu.storage((rows * (top_k + 1)) as u64), gpu.storage((rows * (top_k + 1)) as u64));
        let (gate, dense_ids) = (gpu.storage((rows * experts) as u64), gpu.storage((rows * top_k) as u64));
        gpu.submit(
            &[],
            &[
                gpu.dispatch(K_ROUTER, &[&lb, &ids, &weight], &[rows as u32, experts as u32, top_k as u32, 1], Dispatch::Workgroups(rows as u32)),
                gpu.step(K_ROUTER_GATE, &[&rb, &gate], &[rows as u32, experts as u32, top_k as u32, 1, 1.0f32.to_bits()], rows as u32),
                gpu.step(K_TOPK_COMPACT, &[&gate, &dense_ids], &[rows as u32, experts as u32, top_k as u32], rows as u32),
            ],
        );
        gpu.poll_wait();
        let (got_ids, got_w) = (read_u32(&gpu, &ids, rows * (top_k + 1)), gpu.read(&weight, rows * (top_k + 1)));
        let (want_ids, dense_gate) = (read_u32(&gpu, &dense_ids, rows * top_k), gpu.read(&gate, rows * experts));
        for r in 0..rows {
            let slot = &got_ids[r * (top_k + 1)..(r + 1) * (top_k + 1)];
            assert_eq!(sorted(slot[..top_k].to_vec()), sorted(want_ids[r * top_k..(r + 1) * top_k].to_vec()), "row {r}: the selected experts");
            for s in 0..top_k {
                let e = slot[s] as usize;
                let want = dense_gate[r * experts + e];
                let got = got_w[r * (top_k + 1) + s];
                assert!((got - want).abs() <= 4.0 * f32::EPSILON * want.abs(), "row {r} expert {e}: weight {got} vs {want}");
            }
            let sum: f32 = got_w[r * (top_k + 1)..r * (top_k + 1) + top_k].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r}: the routed weights renormalise to 1, got {sum}");
            assert_eq!(slot[top_k], experts as u32, "the shared expert is bank block n_experts");
            let sig = 1.0 / (1.0 + (-logits[r * width + experts]).exp());
            assert!((got_w[r * (top_k + 1) + top_k] - sig).abs() < 1e-6, "row {r}: the shared slot's weight is sigmoid of the extra logit");
        }
    }
}

/// Two experts at exactly equal probability: the LOWER index wins, as in
/// `router_gate`'s strict `>` scan, and an expert is never selected twice.
#[test]
fn the_router_breaks_ties_toward_the_lower_expert() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("needs workgroup reductions");
    }
    let experts = 256usize;
    let mut logits = vec![0.0f32; experts + 1];
    logits[200] = 3.0;
    logits[5] = 3.0;
    logits[130] = 3.0;
    let lb = upload_f32(&gpu, &logits);
    let (ids, weight) = (gpu.storage(3), gpu.storage(3));
    gpu.submit(&[], &[gpu.dispatch(K_ROUTER, &[&lb, &ids, &weight], &[1, experts as u32, 2, 0], Dispatch::Workgroups(1))]);
    gpu.poll_wait();
    assert_eq!(&read_u32(&gpu, &ids, 2), &[5, 130], "equal probabilities pick the lowest indices first");
}

/// The fused kernel IS the three it replaces.
#[test]
fn swiglu_quant_is_bit_identical_to_silu_mul_then_row_quantisation() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("needs workgroup reductions");
    }
    for &(rows, k) in &[(9usize, 512usize), (1, 512), (3, 64), (5, 1024)] {
        let mut rng = Lcg::new(5 + k as u64);
        // Spread over magnitudes, with an occasional large value so the row scale is set by an outlier.
        let g: Vec<f32> = (0..rows * k).map(|i| ((rng.next_u32() % 4000) as f32 - 2000.0) * 1e-3 * if i % 97 == 0 { 8.0 } else { 1.0 }).collect();
        let u: Vec<f32> = (0..rows * k).map(|_| ((rng.next_u32() % 4000) as f32 - 2000.0) * 1e-3).collect();
        let (gb, ub) = (upload_f32(&gpu, &g), upload_f32(&gpu, &u));
        let (h, sx_ref, xq_ref) = (gpu.storage((rows * k) as u64), gpu.storage(rows as u64), gpu.storage((rows * k / 4) as u64));
        let (xq, sh) = (gpu.storage((rows * k / 4) as u64), gpu.storage(rows as u64));
        gpu.submit(
            &[],
            &[
                gpu.step(K_SILU_MUL, &[&gb, &ub, &h], &[(rows * k) as u32], (rows * k) as u32),
                gpu.step(K_MAX_ABS, &[&h, &sx_ref], &[rows as u32, k as u32], rows as u32),
                gpu.step(K_QUANT_PACK, &[&h, &sx_ref, &xq_ref], &[rows as u32, k as u32], (rows * k / 4) as u32),
                gpu.dispatch(K_SWIGLU_QUANT, &[&gb, &ub, &xq, &sh], &[rows as u32, k as u32], Dispatch::Workgroups(rows as u32)),
            ],
        );
        gpu.poll_wait();
        assert_eq!(read_u32(&gpu, &sh, rows), read_u32(&gpu, &sx_ref, rows), "rows={rows} k={k}: the row scales");
        assert_eq!(read_u32(&gpu, &xq, rows * k / 4), read_u32(&gpu, &xq_ref, rows * k / 4), "rows={rows} k={k}: the packed activations");
    }
}

#[test]
fn the_combine_is_the_weighted_sum_of_a_rows_slots_in_slot_order() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for &(rows, d, s) in &[(1usize, 2048usize, 9usize), (4, 33, 9), (3, 7, 1)] {
        let mut rng = Lcg::new(11 + d as u64);
        let y: Vec<f32> = (0..rows * s * d).map(|_| (rng.next_u32() % 2000) as f32 * 1e-3 - 1.0).collect();
        let w: Vec<f32> = (0..rows * s).map(|_| (rng.next_u32() % 1000) as f32 * 1e-3).collect();
        let (yb, wb, out) = (upload_f32(&gpu, &y), upload_f32(&gpu, &w), gpu.storage((rows * d) as u64));
        gpu.submit(&[], &[gpu.step(K_COMBINE, &[&yb, &wb, &out], &[rows as u32, d as u32, s as u32], (rows * d) as u32)]);
        gpu.poll_wait();
        let got = gpu.read(&out, rows * d);
        for r in 0..rows {
            for c in 0..d {
                let mut acc = 0f32;
                for k in 0..s {
                    acc += w[r * s + k] * y[(r * s + k) * d + c];
                }
                assert!((got[r * d + c] - acc).abs() <= 1e-6 * acc.abs().max(1.0), "row {r} col {c}: {} vs {acc}", got[r * d + c]);
            }
        }
    }
}
