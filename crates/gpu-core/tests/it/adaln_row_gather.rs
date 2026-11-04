// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `adaln_row`: extract one PixArt/adaLN-single modulation vector on the
//! device, GATHERING each token's own row out of a table that holds one row
//! per DISTINCT token timestep rather than one per token.
//!
//! Swedish Embedded AB implements bit-exact device kernels and the gates that
//! keep them bit-exact for its clients. If your team needs a compute kernel
//! whose result a downstream numerical gate can `assert_eq!` on rather than
//! tolerate, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Two things this has to hold, and only one of them is about arithmetic.
//!
//! **The gather picks the right row for the right token.** A wrong scatter -
//! rows mapped to the wrong tokens - is the failure mode of the whole
//! compact-table scheme, and it is invisible to any comparison whose two arms
//! share the row map. The reference here builds the DENSE table independently
//! and indexes it by token, so it does not.
//!
//! **The arithmetic is bit-identical to the host form it replaces**
//! (`dit::adaln::add_table` then a slice, then `1.0 + x` for the `1+scale`
//! rows). `assert_eq!` on raw bits, not a tolerance: nothing here
//! reassociates, so a tolerance would only hide a change. Cosine would be
//! weaker still - it is scale invariant, and an RMSNorm-epsilon mutation
//! elsewhere in this tree scored cosine 1.000000 and was caught only by a
//! relative-L2 check.
//!
//! It lives in `gpu-core` rather than in `ltxv` (the kernel's only caller)
//! because it must run on EVERY backend the kernel claims to support, and
//! `ltxv`'s own production path for it is int8, which the CPU JIT cannot
//! dispatch at all (`matmul_i8_dyn` has more than one top-level barrier). A
//! kernel declared `@cpu yes` needs a gate that actually runs it there -
//! `wgsl-cpu`'s `compile_all` proves only that it TRANSLATES.

const KERNELS: &[(&str, &str)] = &[("adaln_row", kernels::ADALN_ROW)];
const K_ADALN_ROW: usize = 0;

/// The nine `(row, plus_one)` pairs `ltxv::block::MOD_ROWS` dispatches - the
/// `1 + scale` rows are the ones with a second add.
const MOD_ROWS: [(u32, bool); 9] =
    [(0, false), (1, true), (2, false), (3, false), (4, true), (5, false), (6, false), (7, true), (8, false)];

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    r.vec_scaled(n, 1.3)
}

/// The host arithmetic the kernel reproduces, over the DENSE `[R, NR*D]`
/// table: `dit::adaln::add_table(dense, tbl, R, NR*D)` then row `row` sliced
/// out, plus `1.0 +` when asked. Written against the dense table on purpose: it is
/// the independent statement of which row belongs to which token.
fn reference(dense: &[f32], tbl: &[f32], r: usize, d: usize, nr: usize, row: usize, plus_one: bool) -> Vec<f32> {
    let mut out = vec![0f32; r * d];
    for ri in 0..r {
        for di in 0..d {
            let off = row * d + di;
            let v = tbl[off] + dense[ri * nr * d + off];
            out[ri * d + di] = if plus_one { 1.0 + v } else { v };
        }
    }
    out
}

/// `keys[i]` -> the distinct-row index it maps to, in first-appearance order,
/// plus the distinct count - the same relation `dit::adaln::distinct_rows`
/// produces, restated here so the gate does not inherit that function's own
/// idea of the answer.
fn row_map(keys: &[u32]) -> (Vec<u32>, usize) {
    let mut seen: Vec<u32> = Vec::new();
    let map = keys
        .iter()
        .map(|k| match seen.iter().position(|s| s == k) {
            Some(i) => i as u32,
            None => {
                seen.push(*k);
                (seen.len() - 1) as u32
            }
        })
        .collect();
    (map, seen.len())
}

/// Every distinct-row count a real per-token-timestep denoise step produces,
/// at `t` tokens - one row (plain text-to-video), two INTERLEAVED (an image
/// anchor or a long-form window's carried context), several reused, and none
/// shared at all. Interleaved rather than split into contiguous blocks: a
/// gather that happened to be right for a contiguous prefix would pass a
/// block-shaped case and fail this one.
fn patterns(t: usize) -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("uniform", vec![0; t]),
        ("two interleaved", (0..t).map(|i| u32::from(i % 3 == 1)).collect()),
        ("several reused", (0..t).map(|i| (i % 5) as u32).collect()),
        ("all distinct", (0..t as u32).collect()),
    ]
}

#[test]
fn adaln_row_gathers_each_tokens_own_row_and_matches_the_host_arithmetic() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (r, d, nr) = (37usize, 16usize, 9usize);
    let tbl = fill(nr * d, 0xADA1);

    for (kind, keys) in patterns(r) {
        let (map, u) = row_map(&keys);
        // The DISTINCT rows the device is given, and the dense per-token table
        // they stand for. Built in that order so the dense form is derived
        // from the compact one and cannot disagree about the VALUES - leaving
        // the row MAPPING as the only thing under test.
        let distinct = fill(u * nr * d, 0xD157 ^ u as u64);
        let mut dense = vec![0f32; r * nr * d];
        for (ri, chunk) in dense.chunks_mut(nr * d).enumerate() {
            let src = map[ri] as usize * nr * d;
            chunk.copy_from_slice(&distinct[src..src + nr * d]);
        }

        let tab_buf = gpu.storage((u * nr * d) as u64);
        gpu.write_f32(&tab_buf, &distinct);
        let tbl_buf = gpu.storage((nr * d) as u64);
        gpu.write_f32(&tbl_buf, &tbl);
        let map_buf = gpu.storage(r as u64);
        gpu.write_at(&map_buf, 0, &map);

        let outs: Vec<_> = MOD_ROWS.iter().map(|_| gpu.storage((r * d) as u64)).collect();
        let steps: Vec<_> = MOD_ROWS
            .iter()
            .zip(&outs)
            .map(|(&(row, plus_one), o)| {
                gpu.step(
                    K_ADALN_ROW,
                    &[&tab_buf, &tbl_buf, &map_buf, o],
                    &[r as u32, d as u32, nr as u32, row, u32::from(plus_one)],
                    (r * d) as u32,
                )
            })
            .collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();

        for ((row, plus_one), o) in MOD_ROWS.into_iter().zip(&outs) {
            let got = gpu.read(o, r * d);
            let want = reference(&dense, &tbl, r, d, nr, row as usize, plus_one);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{kind}: adaln_row row {row} (plus_one={plus_one}) disagrees with the host form it reproduces");
        }
    }
}

/// The patterns really do address different rows, so the agreement above is
/// not the vacuous "the map never mattered" statement - which is exactly what
/// a gather that always read row 0 would look like on the uniform case alone.
#[test]
fn the_row_map_changes_the_answer() {
    let (map_a, ua) = row_map(&patterns(37)[0].1);
    let (map_b, ub) = row_map(&patterns(37)[1].1);
    assert_eq!(ua, 1);
    assert_eq!(ub, 2);
    assert_ne!(map_a, map_b);
}
