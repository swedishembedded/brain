// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for `kernels_cuda`'s fp16 tensor-core flash attention at head_dim 64
//! (`cu/flash_attn_f16_hd64.cu`), which `gpu_core::native_upgrade` substitutes for the WGSL
//! `flash_attn_cross_relpos` (SAM's relative-position-bias attention) and `flash_attn_bidir_reg2` (CLIP) on a CUDA
//! device.
//!
//! Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
//! needs expertise in running ViT attention on tensor cores with a stated accuracy contract then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! ```text
//! cargo test -p brain-gpu-core --test flash_attn_f16_native -- --device gpu0 --backend cuda
//! ```
//!
//! The kernel rounds Q, K, V and the probabilities to fp16 for the tensor cores (fp32 accumulation, fp32 softmax),
//! so it is not close to the fp32 WGSL tier bit for bit. Attention is a weighted mean of value rows, so the error
//! is stated in units of the largest value magnitude, against an f64 oracle:
//!
//! ```text
//! |out - oracle| <= TOL * max|v|        TOL = 2^-8 (about 3.9e-3)
//! ```
//!
//! An fp16 operand carries 11 significant bits; rounding q, k, v and p each contributes about 2^-11 relative, and
//! the weighted mean averages them down. The peaked case here (logit spread 4) is the worst: a logit of magnitude 10
//! moves by 0.005 under the rounding of its two operands, and the output by that fraction of the value spread. The
//! TF32 products the unfused chain used round q and k to the same 10 bits, so the redirect keeps the tower's accuracy
//! class. The fp32 WGSL tier is held to a much tighter bound on the same data, so
//! the redirect trades accuracy for tensor-core speed by a stated, bounded amount.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[("flash_attn_cross_relpos", kernels::FLASH_ATTN_CROSS_RELPOS), ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2)];
const K_CROSS: usize = 0;
const K_BIDIR: usize = 1;
const HD: u32 = 64;
/// Query rows per workgroup of the WGSL cross kernel (which sets its grid).
const WGSL_CROSS_BR: u32 = 128;
/// The stated tolerance, in units of the largest value magnitude.
const TOL: f64 = 1.0 / 256.0;
/// Written past the end of the output so an out-of-range write shows.
const SENTINEL: u32 = 0x7fc0_dead;

fn device() -> Option<Gpu> {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() {
        brain_testutil::skip_unavailable("native flash attention needs a CUDA device");
        return None;
    }
    Some(gpu)
}

fn normal(r: &mut data::rng::Lcg, sigma: f32) -> f32 {
    let mut uni = || (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
    let (u1, u2) = (uni().max(1e-7), uni());
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Rows of `width` normal(0, sigma) values.
fn normals(n: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| normal(&mut r, sigma)).collect()
}

/// A shape of the SAM kernel: `bsz` spans of `kh x kw` keys, queries = keys.
struct Cross {
    bsz: u32,
    heads: u32,
    kh: u32,
    kw: u32,
}

impl Cross {
    fn t(&self) -> u32 {
        self.kh * self.kw
    }
    fn d(&self) -> u32 {
        self.heads * HD
    }
    /// The WGSL kernel's `Params` for separate q/k/v buffers of row width `d` (offset 0).
    fn params(&self) -> [u32; 14] {
        let (t, d) = (self.t(), self.d());
        [self.bsz, self.heads, t, t, HD, d, 0, d, 0, d, 0, d, self.kh, self.kw]
    }
}

struct Data {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    rel_h: Vec<f32>,
    rel_w: Vec<f32>,
}

fn cross_data(c: &Cross, sigma: f32, seed: u64) -> Data {
    let rows = (c.bsz * c.t()) as usize;
    let bh = (c.bsz * c.heads * c.t()) as usize;
    let mut r = data::rng::Lcg::new(seed ^ 77);
    let mut uni = || (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
    Data {
        q: normals(rows * c.d() as usize, sigma, seed),
        k: normals(rows * c.d() as usize, sigma, seed + 1),
        v: (0..rows * c.d() as usize).map(|_| 2.0 * uni() - 1.0).collect(),
        rel_h: normals(bh * c.kh as usize, sigma, seed + 2),
        rel_w: normals(bh * c.kw as usize, sigma, seed + 3),
    }
}

/// The f64 output of query row `i` of span `b`, head `h`.
fn cross_oracle(c: &Cross, x: &Data, b: u32, h: u32, i: u32) -> Vec<f64> {
    let (t, d) = (c.t() as usize, c.d() as usize);
    let (b, h, i) = (b as usize, h as usize, i as usize);
    let at = |m: &[f32], row: usize, ch: usize| f64::from(m[(b * t + row) * d + h * HD as usize + ch]);
    let bias = |j: usize| {
        let base = (b * c.heads as usize + h) * t + i;
        f64::from(x.rel_h[base * c.kh as usize + j / c.kw as usize]) + f64::from(x.rel_w[base * c.kw as usize + j % c.kw as usize])
    };
    let s: Vec<f64> = (0..t).map(|j| (0..HD as usize).map(|ch| at(&x.q, i, ch) * at(&x.k, j, ch)).sum::<f64>() / f64::from(HD).sqrt() + bias(j)).collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
    let l: f64 = e.iter().sum();
    (0..HD as usize).map(|ch| (0..t).map(|j| e[j] * at(&x.v, j, ch)).sum::<f64>() / l).collect()
}

fn upload(gpu: &Gpu, v: &[f32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(v.len() as u64);
    gpu.write_f32(&b, v);
    b
}

fn fresh_out(gpu: &Gpu, len: usize) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(len as u64 + 8);
    gpu.write(&b, &vec![SENTINEL; len + 8]);
    b
}

fn read_out(gpu: &Gpu, b: &gpu_core::DeviceBuffer, len: usize, what: &str) -> Vec<f32> {
    let v = gpu.read(b, len + 8);
    assert!(v[len..].iter().all(|x| x.to_bits() == SENTINEL), "a write past the output at {what}");
    v[..len].to_vec()
}

fn check_cross(gpu: &Gpu, c: &Cross, sigma: f32, rows: u32, seed: u64) {
    let p = c.params();
    let x = cross_data(c, sigma, seed);
    let out_len = (c.bsz * c.t() * c.d()) as usize;
    let bufs = [upload(gpu, &x.q), upload(gpu, &x.k), upload(gpu, &x.v), upload(gpu, &x.rel_h), upload(gpu, &x.rel_w)];
    let out = fresh_out(gpu, out_len);
    let step = gpu.dispatch(K_CROSS, &[&bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4], &out], &p, Dispatch::Workgroups(c.bsz * c.heads * c.t().div_ceil(WGSL_CROSS_BR)));
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let got = read_out(gpu, &out, out_len, &format!("{p:?}"));
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output at {p:?}");
    let vmax = x.v.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let bound = TOL * vmax;
    let (t, d) = (c.t(), c.d() as usize);
    let mut r = data::rng::Lcg::new(seed ^ 0x5a5a);
    let mut worst = 0.0f64;
    for b in 0..c.bsz {
        for h in 0..c.heads {
            let picks: Vec<u32> = [0, t - 1].into_iter().chain((0..rows.saturating_sub(2)).map(|_| r.next_u32() % t)).collect();
            for i in picks {
                let want = cross_oracle(c, &x, b, h, i);
                let base = ((b * t + i) as usize) * d + (h * HD) as usize;
                for (ch, w) in want.iter().enumerate() {
                    worst = worst.max((f64::from(got[base + ch]) - w).abs());
                }
            }
        }
    }
    println!("cross bsz {} heads {} grid {}x{} sigma {sigma}: |native-f64| {worst:.2e} bound {bound:.2e}", c.bsz, c.heads, c.kh, c.kw);
    assert!(worst <= bound, "fp16 flash attention with bias misses the f64 oracle at {p:?}: {worst:e} > {bound:e}");
}

fn bidir_params(bsz: u32, heads: u32, t: u32) -> [u32; 9] {
    let d = heads * HD;
    [bsz, heads, t, HD, 3 * d, 0, d, 2 * d, d]
}

fn check_bidir(gpu: &Gpu, bsz: u32, heads: u32, t: u32, sigma: f32, rows: u32, seed: u64) {
    let p = bidir_params(bsz, heads, t);
    let d = (heads * HD) as usize;
    let nrows = (bsz * t) as usize;
    let mut r = data::rng::Lcg::new(seed);
    let mut uni = || (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
    let mut qkv = vec![0.0f32; nrows * 3 * d];
    let mut rn = data::rng::Lcg::new(seed ^ 9);
    for row in 0..nrows {
        for c in 0..3 * d {
            qkv[row * 3 * d + c] = if c < 2 * d { normal(&mut rn, sigma) } else { 2.0 * uni() - 1.0 };
        }
    }
    let out_len = nrows * d;
    let buf = upload(gpu, &qkv);
    let out = fresh_out(gpu, out_len);
    // The WGSL grid is only used when the native tier declines; the native one derives its own.
    let step = gpu.dispatch(K_BIDIR, &[&buf, &out], &p, Dispatch::Workgroups(bsz * heads * t.div_ceil(128)));
    gpu.submit(&[], &[step]);
    gpu.poll_wait();
    let got = read_out(gpu, &out, out_len, &format!("{p:?}"));
    let vmax = (0..nrows).flat_map(|row| qkv[row * 3 * d + 2 * d..row * 3 * d + 3 * d].iter()).fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let bound = TOL * vmax;
    let mut pick = data::rng::Lcg::new(seed ^ 0x77);
    let mut worst = 0.0f64;
    for b in 0..bsz as usize {
        for h in 0..heads as usize {
            let picks: Vec<u32> = [0, t - 1].into_iter().chain((0..rows.saturating_sub(2)).map(|_| pick.next_u32() % t)).collect();
            for i in picks {
                let at = |row: usize, off: usize, ch: usize| f64::from(qkv[(b * t as usize + row) * 3 * d + off + h * HD as usize + ch]);
                let i = i as usize;
                let s: Vec<f64> = (0..t as usize).map(|j| (0..HD as usize).map(|ch| at(i, 0, ch) * at(j, d, ch)).sum::<f64>() / f64::from(HD).sqrt()).collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                let l: f64 = e.iter().sum();
                for ch in 0..HD as usize {
                    let want = (0..t as usize).map(|j| e[j] * at(j, 2 * d, ch)).sum::<f64>() / l;
                    let g = f64::from(got[(b * t as usize + i) * d + h * HD as usize + ch]);
                    worst = worst.max((g - want).abs());
                }
            }
        }
    }
    println!("bidir bsz {bsz} heads {heads} T {t} sigma {sigma}: |native-f64| {worst:.2e} bound {bound:.2e}");
    assert!(worst <= bound, "fp16 flash attention misses the f64 oracle at {p:?}: {worst:e} > {bound:e}");
}

fn the_native_kernels_serve_head_dim_64() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let cross = Cross { bsz: 25, heads: 12, kh: 14, kw: 14 }.params();
    let bidir = bidir_params(1, 16, 257);
    if gpu.kind() != "cuda" || gpu.caps().arch.compute_capability.is_none() {
        assert_eq!(gpu.native_kernel_for(K_CROSS, &cross), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(K_CROSS, &cross), Some("flash_cross_relpos_f16_hd64"));
    assert_eq!(gpu.native_kernel_for(K_BIDIR, &bidir), Some("flash_bidir_f16_hd64"));
    let mut wide = cross;
    wide[4] = 128;
    assert_eq!(gpu.native_kernel_for(K_CROSS, &wide), None, "another head width keeps the WGSL tier");
    let mut short_grid = cross;
    short_grid[12] = 3;
    assert_eq!(gpu.native_kernel_for(K_CROSS, &short_grid), None, "a key grid smaller than the keys keeps the WGSL tier");
    let mut unaligned = cross;
    unaligned[5] += 2;
    assert_eq!(gpu.native_kernel_for(K_CROSS, &unaligned), None, "rows that are not whole float4s keep the WGSL tier");
    assert_eq!(gpu.native_kernel_for(K_CROSS, &cross[..13]), None, "short params");
    assert_eq!(gpu.native_kernel_for(K_BIDIR, &bidir_params(1, 4, 0)), None, "no rows");
}

/// SAM's two span shapes (14x14 windows over a padded grid, a whole 64x64 or 40x40 grid) and shapes on both sides of
/// every tile edge, with a flat softmax and a peaked one.
fn the_bias_kernel_matches_the_f64_oracle() {
    let Some(gpu) = device() else { return };
    for (bsz, heads, kh, kw) in [(1u32, 1u32, 1u32, 1u32), (2, 2, 3, 5), (1, 2, 8, 8), (3, 2, 7, 10), (2, 2, 14, 14), (4, 3, 5, 13), (1, 2, 20, 20), (1, 2, 32, 33)] {
        for sigma in [1.0f32, 2.0] {
            check_cross(&gpu, &Cross { bsz, heads, kh, kw }, sigma, 24, u64::from(bsz * 1000 + heads * 100 + kh * kw));
        }
    }
}

fn the_served_sam_shapes_match() {
    let Some(gpu) = device() else { return };
    check_cross(&gpu, &Cross { bsz: 25, heads: 12, kh: 14, kw: 14 }, 2.0, 4, 11);
    check_cross(&gpu, &Cross { bsz: 1, heads: 12, kh: 64, kw: 64 }, 2.0, 6, 12);
    check_cross(&gpu, &Cross { bsz: 1, heads: 12, kh: 40, kw: 40 }, 2.0, 6, 13);
}

fn the_packed_kernel_matches_the_f64_oracle() {
    let Some(gpu) = device() else { return };
    for (bsz, heads, t) in [(1u32, 1u32, 1u32), (1, 2, 7), (2, 2, 32), (1, 3, 63), (2, 2, 64), (1, 2, 65), (1, 2, 101), (1, 4, 129), (1, 16, 257), (6, 16, 257)] {
        for sigma in [1.0f32, 2.0] {
            check_bidir(&gpu, bsz, heads, t, sigma, 16, u64::from(bsz * 1000 + heads * 100 + t));
        }
    }
}

gpu_core::card_tests!(
    the_native_kernels_serve_head_dim_64,
    the_bias_kernel_matches_the_f64_oracle,
    the_served_sam_shapes_match,
    the_packed_kernel_matches_the_f64_oracle,
);
