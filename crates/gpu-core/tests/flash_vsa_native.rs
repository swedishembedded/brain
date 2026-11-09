// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `kernels_cuda`'s native video sparse attention
//! (`brain_flash_vsa_f32` in `cu/flash_attn_f32.cu`), which
//! `gpu_core::native_upgrade` substitutes for the WGSL `flash_attn_vsa` on a
//! CUDA device at `head_dim == 128`.
//!
//! Swedish Embedded AB implements block-sparse attention kernels that turn
//! skipped work into wall-clock savings. If your team needs expertise in
//! sparse attention for video diffusion on the GPUs it already has, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! ```text
//! cargo test -p brain-gpu-core --test flash_vsa_native -- --device gpu1 --backend cuda
//! ```
//!
//! The native kernel is `brain_flash_bidir_f32`'s core over a query tile's
//! gathered slots and the key tiles its mask row keeps, so it is held to that
//! gate's tolerance against an f64 oracle (`tests/flash_bidir_native.rs`):
//! `|out - oracle| <= 2^-17 * max|v|` for both tiers, the tiers within twice
//! that of each other, and the native error within twice the generated
//! tier's plus a floor. Slots past a tile's live count are never written.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("flash_attn_vsa", kernels::FLASH_ATTN_VSA), ("flash_attn_vsa_ref", kernels::FLASH_ATTN_VSA)];
const K_VSA: usize = 0;
const K_REF: usize = 1;
/// Slots per tile.
const TILE: usize = 64;
const HD: u32 = 128;
/// The stated tolerance, in units of the largest value magnitude.
const TOL: f64 = 1.0 / 131_072.0;
/// The slack beside twice the generated tier's own error.
const FLOOR: f64 = 1.0 / 4_194_304.0;
/// Written to every output word before the run, so a write to a row no live
/// slot names shows.
const SENTINEL: u32 = 0x7fc0_dead;

/// A tile layout with its keep mask: `slots[t * 64 + s]` is the packed row in
/// slot s of tile t, `live[t]` how many slots hold one, `mask[(h * n + q) * n
/// + k]` whether query tile q keeps key tile k for head h.
struct Layout {
    rows: usize,
    slots: Vec<u32>,
    live: Vec<u32>,
    mask: Vec<u32>,
}

impl Layout {
    fn tiles(&self) -> usize {
        self.live.len()
    }

    /// The packed rows of tile t, slot order.
    fn rows_of(&self, t: usize) -> &[u32] {
        &self.slots[t * TILE..t * TILE + self.live[t] as usize]
    }
}

/// `rows` packed rows scattered over tiles in a shuffled order, the tile
/// sizes drawn so that full, partial and empty tiles all occur, and a mask
/// that keeps a random subset (`keep_pct` percent) plus one non-empty tile
/// for every query tile so every live query attends something.
fn layout(rows: usize, heads: usize, keep_pct: u32, r: &mut data::rng::Lcg) -> Layout {
    let mut order: Vec<u32> = (0..rows as u32).collect();
    for i in (1..rows).rev() {
        order.swap(i, r.next_u32() as usize % (i + 1));
    }
    let (mut slots, mut live) = (Vec::new(), Vec::new());
    let mut next = 0;
    while next < rows {
        let take = match r.next_u32() % 4 {
            0 => 0,
            1 => 1 + r.next_u32() as usize % TILE,
            _ => TILE,
        }
        .min(rows - next);
        let mut tile = vec![0u32; TILE];
        tile[..take].copy_from_slice(&order[next..next + take]);
        slots.extend(tile);
        live.push(take as u32);
        next += take;
    }
    let n = live.len();
    let nonempty: Vec<usize> = (0..n).filter(|&t| live[t] > 0).collect();
    let mut mask = vec![0u32; heads * n * n];
    for h in 0..heads {
        for q in 0..n {
            for k in 0..n {
                mask[(h * n + q) * n + k] = u32::from(r.next_u32() % 100 < keep_pct);
            }
            mask[(h * n + q) * n + nonempty[r.next_u32() as usize % nonempty.len()]] = 1;
        }
    }
    Layout { rows, slots, live, mask }
}

/// A packed `[rows, 3 * heads * hd]` qkv slab: q and k normal with standard
/// deviation `sigma`, v uniform in [-1, 1).
fn slab(rows: usize, d: usize, sigma: f32, r: &mut data::rng::Lcg) -> Vec<f32> {
    let mut uni = || (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
    (0..rows * 3 * d)
        .map(|i| {
            if i % (3 * d) < 2 * d {
                let (u1, u2) = (uni().max(1e-7), uni());
                sigma * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            } else {
                2.0 * uni() - 1.0
            }
        })
        .collect()
}

/// `[tiles, n_heads, head_dim, qkv_stride, q_off, k_off, v_off, d_model]`.
fn params(tiles: usize, heads: usize) -> [u32; 8] {
    let d = heads as u32 * HD;
    [tiles as u32, heads as u32, HD, 3 * d, 0, d, 2 * d, d]
}

/// The f64 output of packed row `i` (in query tile `qt`) for head `h`.
fn oracle_row(l: &Layout, qkv: &[f32], heads: usize, h: usize, qt: usize, i: usize) -> Vec<f64> {
    let (n, hd, d) = (l.tiles(), HD as usize, heads * HD as usize);
    let at = |row: usize, off: usize, c: usize| f64::from(qkv[row * 3 * d + off + h * hd + c]);
    let keys: Vec<usize> = (0..n).filter(|&k| l.mask[(h * n + qt) * n + k] != 0).flat_map(|k| l.rows_of(k).iter().map(|&j| j as usize)).collect();
    let scale = 1.0 / (hd as f64).sqrt();
    let s: Vec<f64> = keys.iter().map(|&j| (0..hd).map(|c| at(i, 0, c) * at(j, d, c)).sum::<f64>() * scale).collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    (0..hd).map(|c| keys.iter().zip(&e).map(|(&j, w)| w * at(j, 2 * d, c)).sum::<f64>() / sum).collect()
}

fn run(gpu: &Gpu, kernel: usize, l: &Layout, heads: usize, qkv: &[f32]) -> Vec<f32> {
    let d = heads * HD as usize;
    let out_len = l.rows * d;
    let upload = |words: &[u32]| {
        let b = gpu.storage(words.len() as u64);
        gpu.write(&b, words);
        b
    };
    let qkv_buf = gpu.storage(qkv.len() as u64);
    gpu.write_f32(&qkv_buf, qkv);
    let (slots, live, mask) = (upload(&l.slots), upload(&l.live), upload(&l.mask));
    let out = upload(&vec![SENTINEL; out_len]);
    let p = params(l.tiles(), heads);
    // The grid the WGSL tier is dispatched with (two query tiles a workgroup); a redirect computes its own.
    let wg = (heads * l.tiles().div_ceil(2)) as u32;
    gpu.submit(&[], &[gpu.dispatch(kernel, &[&qkv_buf, &out, &slots, &live, &mask], &p, Dispatch::Workgroups(wg))]);
    gpu.poll_wait();
    gpu.read(&out, out_len)
}

fn check(gpu: &Gpu, rows: usize, heads: usize, keep_pct: u32, sigma: f32, seed: u64) {
    let mut r = data::rng::Lcg::new(seed);
    let l = layout(rows, heads, keep_pct, &mut r);
    let d = heads * HD as usize;
    let qkv = slab(rows, d, sigma, &mut r);
    let native = run(gpu, K_VSA, &l, heads, &qkv);
    let wgsl = run(gpu, K_REF, &l, heads, &qkv);
    let vmax = (0..rows).flat_map(|row| qkv[row * 3 * d + 2 * d..row * 3 * d + 3 * d].iter()).fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let bound = TOL * vmax;
    for (name, out) in [("native", &native), ("generated", &wgsl)] {
        assert!(out.iter().all(|v| v.to_bits() != SENTINEL && v.is_finite()), "{name}: a live row left unwritten or non-finite ({rows} rows, {heads} heads)");
    }
    let pair = native.iter().zip(&wgsl).map(|(a, b)| f64::from(*a - *b).abs()).fold(0.0, f64::max);
    assert!(pair <= 2.0 * bound, "native and generated tiers differ by {pair:e} (bound {:e})", 2.0 * bound);
    let (mut worst_n, mut worst_w) = (0.0f64, 0.0f64);
    for h in 0..heads {
        for qt in 0..l.tiles() {
            let picks = l.rows_of(qt);
            // The first and last slot of each tile (its edges), and one between.
            for &i in [picks.first(), picks.get(picks.len() / 2), picks.last()].into_iter().flatten() {
                let want = oracle_row(&l, &qkv, heads, h, qt, i as usize);
                let base = i as usize * d + h * HD as usize;
                for (c, w) in want.iter().enumerate() {
                    worst_n = worst_n.max((f64::from(native[base + c]) - w).abs());
                    worst_w = worst_w.max((f64::from(wgsl[base + c]) - w).abs());
                }
            }
        }
    }
    println!("rows {rows} heads {heads} keep {keep_pct}% sigma {sigma}: |native-f64| {worst_n:.2e}  |wgsl-f64| {worst_w:.2e}  |native-wgsl| {pair:.2e}  bound {bound:.2e}");
    assert!(worst_w <= bound, "the generated tier misses the stated tolerance: {worst_w:e} > {bound:e}");
    assert!(worst_n <= bound, "native sparse attention misses the f64 oracle: {worst_n:e} > {bound:e}");
    assert!(worst_n <= 2.0 * worst_w + FLOOR * vmax, "native sparse attention is less accurate than the generated tier");
}

fn the_native_kernel_serves_head_dim_128() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() {
        assert_eq!(gpu.native_kernel_for(K_VSA, &params(243, 56)), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(K_VSA, &params(243, 56)), Some("flash_vsa_f32"));
    let mut other = params(4, 2);
    other[2] = 64;
    assert_eq!(gpu.native_kernel_for(K_VSA, &other), None, "another head width keeps the WGSL tier");
    assert_eq!(gpu.native_kernel_for(K_REF, &params(243, 56)), None);
}

/// Sparse and dense masks, partial and empty tiles, a flat and a peaked
/// softmax, and a tile count that leaves the WGSL tier's pairing one short.
fn matches_the_f64_oracle_and_the_generated_tier() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() {
        brain_testutil::skip_unavailable("native sparse attention needs a CUDA device");
        return;
    }
    for (rows, heads, keep) in [(1usize, 1usize, 100u32), (65, 2, 50), (300, 3, 25), (700, 2, 10), (1000, 1, 100)] {
        for sigma in [1.0f32, 3.0] {
            check(&gpu, rows, heads, keep, sigma, (rows * 100 + heads * 10 + keep as usize) as u64);
        }
    }
}

gpu_core::card_tests!(the_native_kernel_serves_head_dim_128, matches_the_f64_oracle_and_the_generated_tier,);
