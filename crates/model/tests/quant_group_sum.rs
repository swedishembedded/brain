// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device gate for `quant_group_sum.wgsl` (`model::kquant`'s `S[m,g] =
//! Σ_{k in g} xq[m,k]` affine K-quant correction term) and its wiring into
//! `model::int8::QuantRows`/`quant_rows_steps`'s `xgs` seam.
//!
//! The kernel's whole claim is that it is EXACT: `dot4I8Packed` against
//! `0x01010101u` sums four signed int8 lanes with no rounding, so its output
//! must equal a host-computed integer sum of the SAME packed bytes bit for
//! bit - `assert_eq!`, never a tolerance.

use gpu_core::testgpu::dev;

const PIPES: &[(&str, &str)] = &[
    ("quant_group_sum", kernels::QUANT_GROUP_SUM),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
];

/// Pack `m*k` signed int8 values (K-contiguous, row-major, `k` a multiple of
/// 4) into `[m, k/4]` u32 words, little-endian along K - the same layout
/// `quant_pack.wgsl` produces and `quant_group_sum.wgsl` consumes. Chunking
/// the flat, already row-major array by 4 lands on row boundaries for free
/// because `k` is always a multiple of 4.
fn pack_i8(vals: &[i8]) -> Vec<u32> {
    vals.chunks(4).map(|b| u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8])).collect()
}

/// The host oracle: exact integer sum per `[row, group]` of 32 elements.
fn host_group_sums(vals: &[i8], m: usize, k: usize) -> Vec<f32> {
    let gs = k / 32;
    let mut out = vec![0f32; m * gs];
    for r in 0..m {
        for g in 0..gs {
            let s: i32 = vals[r * k + g * 32..r * k + g * 32 + 32].iter().map(|&v| v as i32).sum();
            out[r * gs + g] = s as f32;
        }
    }
    out
}

/// The kernel in isolation: feed a KNOWN packed int8 buffer (not derived via
/// device quantization, so there is no rounding path to reproduce) and
/// compare against the host's exact integer sum of those same bytes.
#[test]
fn quant_group_sum_matches_host_exact_integer_sum() {
    let g = dev(PIPES);
    // Several rows, several groups per row, values covering the full i8
    // range including -128 (the one value `-x` cannot represent) so a
    // sign-handling bug cannot hide.
    let (m, k) = (5usize, 128usize);
    let vals: Vec<i8> = (0..m * k).map(|i| ((i as i64 * 41 - 777) % 256) as i8).collect();
    let words = pack_i8(&vals);
    let xq = g.storage_init("xq", &words.iter().map(|&w| f32::from_bits(w)).collect::<Vec<f32>>());

    let gs = k / 32;
    let out = g.storage((m * gs) as u64);
    g.submit(&[&out], &[g.step(0, &[&xq, &out], &[m as u32, k as u32], (m * gs) as u32)]);
    let got = g.read(&out, m * gs);

    let want = host_group_sums(&vals, m, k);
    assert_eq!(got, want, "quant_group_sum must reproduce the host's exact integer per-group sum");
}

/// Every group's sum must land in the RIGHT row/group slot, not just have the
/// right multiset of values - a swapped row or group stride would still pass
/// a same-row test with symmetric data, so rows use visibly different totals
/// (row `r`'s values are all `r+1`, so row `r`'s every group sums to
/// `32*(r+1)`) and every group is checked individually.
#[test]
fn quant_group_sum_indexes_row_and_group_independently() {
    let g = dev(PIPES);
    let (m, k) = (4usize, 96usize); // 3 groups per row
    let vals: Vec<i8> = (0..m * k).map(|i| (i / k) as i8 + 1).collect();
    let words = pack_i8(&vals);
    let xq = g.storage_init("xq", &words.iter().map(|&w| f32::from_bits(w)).collect::<Vec<f32>>());

    let gs = k / 32;
    let out = g.storage((m * gs) as u64);
    g.submit(&[&out], &[g.step(0, &[&xq, &out], &[m as u32, k as u32], (m * gs) as u32)]);
    let got = g.read(&out, m * gs);

    for r in 0..m {
        for gi in 0..gs {
            let want = 32.0 * (r as f32 + 1.0);
            assert_eq!(got[r * gs + gi], want, "row {r} group {gi}");
        }
    }
}

/// The `xgs` seam end to end: `QuantRows { xgs: Some(..) }` must append the
/// `quant_group_sum` step and land on the same numbers as the isolated-kernel
/// tests above, reading the packed rows `quant_pack` itself just wrote (not a
/// separately-uploaded buffer) - and `xgs: None` must still dispatch exactly
/// the pre-existing two steps, byte-identically to before this seam existed.
#[test]
fn quant_rows_steps_wires_the_xgs_seam() {
    use model::int8::{quant_rows_steps, QuantRows};

    let g = dev(PIPES);
    let (m, k) = (3usize, 64usize);
    // Integer values in [-100, 100], plus one element per row forced to
    // exactly 127 so `max_abs_row` computes sx[r] = 127/127 = 1.0 EXACTLY
    // (no floating remainder). With sx == 1.0, `quant_pack`'s
    // `round(x * (1/sx))` is `round` of an already-integer value: never a
    // tie, so the packed bytes are known exactly without reimplementing
    // quant_pack's rounding on the host.
    let mut x_vals: Vec<f32> = (0..m * k).map(|i| (((i as i64 * 53) % 201) - 100) as f32).collect();
    for r in 0..m {
        x_vals[r * k] = 127.0;
    }
    let x = g.storage_init("x", &x_vals);
    let sx = g.storage(m as u64);

    let xq_none = g.storage((m * k / 4) as u64);
    let steps_none = quant_rows_steps(&g, QuantRows { kernels: [1, 2], x: &x, sx: &sx, xq: &xq_none, xgs: None }, 0, m as u32, k as u32);
    assert_eq!(steps_none.len(), 2, "xgs: None must dispatch exactly the pre-existing two steps");

    let xq = g.storage((m * k / 4) as u64);
    let xgs = g.storage((m * (k / 32)) as u64);
    let steps_some = quant_rows_steps(&g, QuantRows { kernels: [1, 2], x: &x, sx: &sx, xq: &xq, xgs: Some((0, &xgs)) }, 0, m as u32, k as u32);
    assert_eq!(steps_some.len(), 3, "xgs: Some(..) must append exactly one more step");
    g.submit(&[&xq, &xgs], &steps_some);

    let got = g.read(&xgs, m * (k / 32));
    let vals_i8: Vec<i8> = x_vals.iter().map(|&v| v.round().clamp(-127.0, 127.0) as i8).collect();
    let want = host_group_sums(&vals_i8, m, k);
    assert_eq!(got, want, "the xgs seam must sum the SAME packed bytes quant_pack wrote");
}
