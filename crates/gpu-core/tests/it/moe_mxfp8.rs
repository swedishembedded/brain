// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The MXFP8 stored-format products: the dense GEMV (`matmul_mxfp8_gemv`, every row-capacity build) and the sparse-MoE
//! expert GEMMs over a fused bank (`moe_mxfp8_gemv_gather`, `moe_mxfp8_grouped`).
//!
//! Swedish Embedded AB implements fast serving of MXFP8 mixture-of-experts models for its clients. If your team
//! needs expertise in microscaled expert GEMMs then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! * every product equals an f64 oracle of the decoded weights (E4M3 element x 2^(E8M0 - 127), the OCP MXFP8 layout)
//!   times the unquantised activations, to f32 rounding of the running sum;
//! * the three kernels share one order of arithmetic, so on one device a slot's gather output, its grouped output and
//!   the dense GEMV of its expert's rows are the same bits.
//!
//! The kernels have one top-level barrier and no workgroup table, so they run on the CPU JIT as well as on a GPU (on a
//! CUDA device through the WGSL translation); the routing table the grouped kernel reads is built on the host here.

use data::rng::{random_seed, Lcg};
use gpu_core::{Dispatch, Gpu};

fn kernels() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = vec![("moe_mxfp8_gemv_gather", kernels::MOE_MXFP8_GEMV_GATHER), ("moe_mxfp8_grouped", kernels::MOE_MXFP8_GROUPED)];
        for cap in kernels::template::GEMV_ROW_CAPACITIES {
            v.push(kernels::template::interned("matmul_mxfp8_gemv", kernels::MATMUL_MXFP8_GEMV, &[("MAXM", cap)]).expect("the GEMV declares MAXM"));
        }
        v
    })
}
const K_GATHER: usize = 0;
const K_GROUPED: usize = 1;
/// The GEMV build for `m` rows: the first capacity that holds them.
fn k_gemv(m: usize) -> usize {
    2 + kernels::template::GEMV_ROW_CAPACITIES.iter().position(|c| *c as usize >= m).expect("a capacity holds m")
}
const MR: usize = 8;
const COLS: u32 = 4;

fn e4m3(b: u8) -> f64 {
    let (e, m) = (u32::from(b >> 3) & 15, u32::from(b & 7));
    let v = if e == 0 { f64::from(m) / 512.0 } else { (1.0 + f64::from(m) / 8.0) * 2f64.powi(e as i32 - 7) };
    if b & 128 != 0 { -v } else { v }
}

struct Bank {
    bytes: Vec<u8>,
    scales: Vec<u8>,
}

/// A random bank of `[rows, k]` MXFP8 weight rows: E4M3 bytes without the NaN code, E8M0 scales near 1.
fn random_bank(rows: usize, k: usize, seed: u64) -> Bank {
    let mut r = Lcg::new(seed);
    let bytes = (0..rows * k).map(|_| loop { let b = r.next_u32() as u8; if b & 0x7f != 0x7f { break b; } }).collect();
    let scales = (0..rows * k / 32).map(|_| (120 + r.next_u32() % 12) as u8).collect();
    Bank { bytes, scales }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (i, b)| w | u32::from(*b) << (8 * i))).collect()
}

fn upload_u32(gpu: &Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, v);
    b
}

/// `(sum, sqrt(sum of squared terms))` of the dot product of weight row `row` with `x`, in f64.
fn oracle(bank: &Bank, row: usize, k: usize, x: &[f32]) -> (f64, f64) {
    let terms = (0..k).map(|i| e4m3(bank.bytes[row * k + i]) * 2f64.powi(i32::from(bank.scales[row * k / 32 + i / 32]) - 127) * f64::from(x[i]));
    let (sum, sq) = terms.fold((0.0, 0.0), |(s, q), t| (s + t, q + t * t));
    (sum, sq.sqrt())
}

fn close(label: &str, got: f32, (want, scale): (f64, f64)) {
    assert!(got.is_finite(), "{label}: not written");
    assert!((f64::from(got) - want).abs() <= 2e-5 * scale + 1e-9, "{label}: {got}, oracle {want}");
}

/// The routing table `moe_route_scan` / `moe_route_emit` produce: slot starts per expert, then MR-tile starts per
/// expert (each `ne + 1` long), and the slots ordered by expert.
fn route(ids: &[u32], ne: usize) -> (Vec<u32>, Vec<u32>) {
    let mut counts = vec![0u32; ne];
    ids.iter().for_each(|&e| counts[e as usize] += 1);
    let (mut slot_start, mut tile_start) = (vec![0u32; ne + 1], vec![0u32; ne + 1]);
    for e in 0..ne {
        slot_start[e + 1] = slot_start[e] + counts[e];
        tile_start[e + 1] = tile_start[e] + counts[e].div_ceil(MR as u32);
    }
    let perm = (0..ne as u32).flat_map(|e| (0..ids.len() as u32).filter(move |&s| ids[s as usize] == e)).collect();
    (slot_start.into_iter().chain(tile_start).collect(), perm)
}

struct Case {
    slots: usize,
    k: usize,
    n: usize,
    xdiv: usize,
    ne: usize,
}

fn run_case(gpu: &Gpu, c: &Case, seed: u64) {
    let Case { slots, k, n, xdiv, ne } = *c;
    let rows = slots.div_ceil(xdiv);
    let x = Lcg::new(seed).vec_scaled(rows * k, 1.0);
    let ids: Vec<u32> = (0..slots).map(|s| ((s as u64 * 7 + seed) % ne as u64) as u32).collect();
    let bank = random_bank(ne * n, k, seed + 1);
    let (tab, perm) = route(&ids, ne);
    let tiles = (ne + slots.div_ceil(MR)) as u32;
    let (xb, idb, tb, pb) = (gpu.storage_init("x", &x), upload_u32(gpu, &ids), upload_u32(gpu, &tab), upload_u32(gpu, &perm));
    let (wb, sb) = (upload_u32(gpu, &words(&bank.bytes)), upload_u32(gpu, &words(&bank.scales)));
    let (gather, grouped) = (gpu.storage((slots * n) as u64), gpu.storage((slots * n) as u64));
    gpu.write(&grouped, &vec![0x7fc0_dead; slots * n]);
    let col_tiles = (n as u32).div_ceil(COLS);
    gpu.submit(
        &[],
        &[
            gpu.dispatch(K_GATHER, &[&xb, &idb, &wb, &sb, &gather], &[slots as u32, k as u32, n as u32, xdiv as u32], Dispatch::Workgroups(slots as u32 * col_tiles)),
            gpu.dispatch(K_GROUPED, &[&xb, &tb, &pb, &wb, &sb, &grouped], &[tiles, k as u32, n as u32, xdiv as u32, ne as u32], Dispatch::Workgroups(tiles * col_tiles)),
        ],
    );
    gpu.poll_wait();
    let (gather, grouped) = (gpu.read(&gather, slots * n), gpu.read(&grouped, slots * n));
    let label = format!("slots={slots} k={k} n={n} xdiv={xdiv} ne={ne}");
    for s in 0..slots {
        let xs = &x[(s / xdiv) * k..(s / xdiv + 1) * k];
        for col in 0..n {
            close(&format!("gather {label} slot {s} col {col}"), gather[s * n + col], oracle(&bank, ids[s] as usize * n + col, k, xs));
        }
    }
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&grouped), bits(&gather), "grouped is the gather kernel's arithmetic, bit for bit ({label})");

    // The dense GEMV over expert 0's rows equals the gather output of a slot routed to expert 0.
    if let Some(s0) = ids.iter().position(|&e| e == 0) {
        let m = 1usize;
        let y = gpu.storage(n as u64);
        let xr0 = (s0 / xdiv) as u32;
        gpu.submit(&[], &[gpu.dispatch(k_gemv(m), &[&xb, &wb, &sb, &y], &[m as u32, k as u32, n as u32, 0, xr0, 0, 0, 0], Dispatch::Workgroups(col_tiles))]);
        gpu.poll_wait();
        assert_eq!(bits(&gpu.read(&y, n)), bits(&gather[s0 * n..(s0 + 1) * n]), "the dense GEMV is the gather's arithmetic ({label})");
    }
}

#[test]
fn the_mxfp8_expert_gemms_match_the_decoded_reference_and_each_other() {
    let gpu = gpu_core::testgpu::dev(kernels());
    let base = random_seed();
    for (i, c) in [
        Case { slots: 9, k: 2304, n: 40, xdiv: 9, ne: 17 },  // decode: one token's routed experts
        Case { slots: 90, k: 1152, n: 12, xdiv: 9, ne: 7 },  // ten tokens, the down projection's width
        Case { slots: 37, k: 128, n: 6, xdiv: 1, ne: 5 },    // one block-row of scales, ragged n
        Case { slots: 1, k: 384, n: 1, xdiv: 1, ne: 1 },
    ]
    .iter()
    .enumerate()
    {
        run_case(&gpu, c, base.wrapping_add(i as u64 * 31));
    }
}

/// Spec: every row-capacity build of the dense MXFP8 GEMV reads rows `first .. first + count` of the weight against
/// activation rows `xr0 .. xr0 + m` and equals the f64 oracle.
#[test]
fn the_mxfp8_gemv_matches_the_decoded_reference_at_every_row_count() {
    let gpu = gpu_core::testgpu::dev(kernels());
    let seed = random_seed();
    let (k, n, first, count, xr0) = (640usize, 23usize, 5usize, 13usize, 2usize);
    let bank = random_bank(n, k, seed);
    let (wb, sb) = (upload_u32(&gpu, &words(&bank.bytes)), upload_u32(&gpu, &words(&bank.scales)));
    for m in [1usize, 2, 3, 5, 8] {
        let x = Lcg::new(seed + m as u64).vec_scaled((xr0 + m) * k, 1.0);
        let (xb, y) = (gpu.storage_init("x", &x), gpu.storage((m * count) as u64));
        gpu.write(&y, &vec![0x7fc0_dead; m * count]);
        let params = [m as u32, k as u32, count as u32, first as u32, xr0 as u32, 0, 0, 0];
        gpu.submit(&[], &[gpu.dispatch(k_gemv(m), &[&xb, &wb, &sb, &y], &params, Dispatch::Workgroups((count as u32).div_ceil(COLS)))]);
        gpu.poll_wait();
        let got = gpu.read(&y, m * count);
        for r in 0..m {
            for c in 0..count {
                close(&format!("m={m} row {r} col {c}"), got[r * count + c], oracle(&bank, first + c, k, &x[(xr0 + r) * k..(xr0 + r + 1) * k]));
            }
        }
    }
}
