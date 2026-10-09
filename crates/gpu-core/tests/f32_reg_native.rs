// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native fp32 GEMMs of `kernels_cuda`'s
//! `cu/matmul_f32_reg.cu` that `gpu_core::native_upgrade` substitutes on a
//! CUDA device: `matmul_f32_reg` for the WGSL `matmul_reg3` forward and
//! `matmul_f32_dx_reg` for `matmul_dx_reg`, its input gradient, and
//! `matmul_f32_dw_reg` for the accumulating weight gradient `matmul_dw_reg`.
//!
//! Swedish Embedded AB implements bit-exact native fp32 GEMMs. If your team
//! needs expertise in replacing a generated kernel with a hand-written one
//! without changing a single output bit, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! `matmul_reg3` is dispatched by dozens of models, so the substitution has to
//! be invisible: RAW-BIT identity with the WGSL tier. It is achievable because
//! both keep one fp32 accumulator per output that adds `a * b` - product and
//! sum each rounded - in ascending k, including the zero terms of a K that is
//! not a whole number of 8-wide chunks. The reference is `matmul_reg3_ref`, the
//! same WGSL source under a name the upgrade table does not know.

use gpu_core::{Dispatch, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_reg3_ref", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dx_reg_ref", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("matmul_dw_reg_ref", kernels::MATMUL_DW_REG),
];

const K_REG3: usize = 0;
const K_REF: usize = 1;
const K_DX: usize = 2;
const K_DX_REF: usize = 3;
const K_DW: usize = 4;
const K_DW_REF: usize = 5;
/// The WGSL kernel's output tile, which sets its dispatch geometry.
const WGSL_TILE: u32 = 128;
/// Written around every window so an out-of-window write shows.
const SENTINEL: u32 = 0x7fc0_dead;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some()
}

/// Values of both signs over several binades, with exact zeros sprinkled in,
/// so the products and sums really round.
fn values(n: usize, seed: u64) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n)
        .map(|_| {
            let u = r.next_u32();
            if u.is_multiple_of(17) {
                return 0.0f32.to_bits();
            }
            let mag = (1.0 + (u % 10_000) as f32 * 1e-4) * 2f32.powi((u >> 16) as i32 % 9 - 4);
            (if (u >> 8).is_multiple_of(2) { mag } else { -mag }).to_bits()
        })
        .collect()
}

struct Windowed {
    buf: backend_api::DeviceBuffer,
    lead: u64,
    len: usize,
    total: usize,
}

impl Windowed {
    fn new(gpu: &Gpu, words: &[u32], lead: u64) -> Windowed {
        let total = words.len() + lead as usize + 7;
        let buf = gpu.storage(total as u64);
        gpu.write(&buf, &vec![SENTINEL; total]);
        gpu.write_at(&buf, lead, words);
        Windowed { buf, lead, len: words.len(), total }
    }
    fn range(&self) -> (u64, u64) {
        (self.lead, self.len as u64)
    }
    fn bits(&self, gpu: &Gpu) -> Vec<u32> {
        gpu.read(&self.buf, self.total).iter().map(|f| f.to_bits()).collect()
    }
}

/// `(m, k, n, [x, w, out] leading words)`.
fn check(gpu: &Gpu, m: u32, k: u32, n: u32, lead: [u64; 3]) {
    let seed = u64::from(m) * 131 + u64::from(k) * 7 + u64::from(n);
    let x = Windowed::new(gpu, &values((m * k) as usize, seed + 1), lead[0]);
    let w = Windowed::new(gpu, &values((n * k) as usize, seed + 2), lead[1]);
    let a = Windowed::new(gpu, &vec![SENTINEL; (m * n) as usize], lead[2]);
    let b = Windowed::new(gpu, &vec![SENTINEL; (m * n) as usize], lead[2]);
    let tiles = m.div_ceil(WGSL_TILE) * n.div_ceil(WGSL_TILE);
    let dispatch = |kind: usize, o: &Windowed| {
        gpu.dispatch_sliced(kind, &[&x.buf, &w.buf, &o.buf], &[x.range(), w.range(), o.range()], &[m, k, n], Dispatch::Workgroups(tiles))
    };
    gpu.submit(&[], &[dispatch(K_REG3, &a), dispatch(K_REF, &b)]);
    gpu.poll_wait();
    let (ga, gb) = (a.bits(gpu), b.bits(gpu));
    let (lo, hi) = (a.lead as usize, a.lead as usize + a.len);
    assert!(ga[..lo].iter().chain(&ga[hi..]).all(|&v| v == SENTINEL), "native path wrote outside its window at {m}x{k}x{n} {lead:?}");
    let first = ga[lo..hi].iter().zip(&gb[lo..hi]).position(|(p, q)| p != q);
    assert!(
        first.is_none(),
        "native fp32 GEMM differs from matmul_reg3 at m={m} k={k} n={n} lead={lead:?}: first at {first:?} ({:?} vs {:?})",
        first.map(|i| f32::from_bits(ga[lo + i])),
        first.map(|i| f32::from_bits(gb[lo + i])),
    );
    for (name, win) in [("x", &x), ("w", &w)] {
        let bits = win.bits(gpu);
        assert!(bits[..win.lead as usize].iter().chain(&bits[win.lead as usize + win.len..]).all(|&v| v == SENTINEL), "input {name} modified");
    }
}

fn device() -> Option<Gpu> {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("native fp32 GEMM needs a CUDA device");
        return None;
    }
    Some(gpu)
}

fn the_native_kernel_is_selected_for_every_reg3_dispatch() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(K_REG3, &[128, 64, 128]), None, "only a CUDA device takes a native kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for p in [[1u32, 1, 1], [128, 64, 128], [1280, 9216, 3072], [37, 13, 5]] {
        assert_eq!(gpu.native_kernel_for(K_REG3, &p), Some("matmul_f32_reg"), "{p:?}");
    }
    assert_eq!(gpu.native_kernel_for(K_REG3, &[0, 64, 128]), None, "no rows");
    assert_eq!(gpu.native_kernel_for(K_REG3, &[128, 0, 128]), None, "no reduction");
    assert_eq!(gpu.native_kernel_for(K_REG3, &[128, 64]), None, "short params");
    assert_eq!(gpu.native_kernel_for(K_REF, &[128, 64, 128]), None);
}

/// Partial tiles, K tails of every residue mod 8, aligned and unaligned windows.
fn the_native_kernel_is_bit_identical_to_matmul_reg3() {
    let Some(gpu) = device() else { return };
    for k in [1u32, 3, 8, 13, 16, 31, 64] {
        for (m, n) in [(1u32, 1u32), (1, 130), (127, 129), (128, 128), (129, 3), (200, 257)] {
            for lead in [[0, 0, 0], [1, 3, 5], [4, 4, 4]] {
                check(&gpu, m, k, n, lead);
            }
        }
    }
}

fn random_shapes_and_windows_are_bit_identical() {
    let Some(gpu) = device() else { return };
    let mut r = data::rng::Lcg::new(0x0f32_5eed);
    for _ in 0..40 {
        let (m, k, n) = (1 + r.next_u32() % 400, 1 + r.next_u32() % 300, 1 + r.next_u32() % 400);
        let lead = [0, 1, 2].map(|_| u64::from(r.next_u32() % 8));
        check(&gpu, m, k, n, lead);
    }
}

/// Shapes the models this was written for dispatch: FLUX.2 klein-4B's fp32
/// linears (double-block mlp.2 over text and image rows, txt_in, final layer)
/// and a VAE attention projection and conv-as-GEMM chunk.
fn the_model_shapes_are_bit_identical() {
    let Some(gpu) = device() else { return };
    for (m, k, n) in [(512u32, 9216u32, 3072u32), (1280, 9216, 3072), (512, 7680, 3072), (1280, 3072, 128), (5120, 512, 512), (4096, 1152, 128)] {
        check(&gpu, m, k, n, [0; 3]);
    }
}

/// `dX[m,k] = sum_n dY[m,n] * W[n,k]`, optionally added to what `out` holds.
/// Both outputs start from the same prior contents so the accumulating mode
/// is compared on equal terms.
fn check_dx(gpu: &Gpu, m: u32, k: u32, n: u32, accumulate: u32, lead: [u64; 3]) {
    let seed = u64::from(m) * 37 + u64::from(k) * 11 + u64::from(n) + u64::from(accumulate);
    let dy = Windowed::new(gpu, &values((m * n) as usize, seed + 1), lead[0]);
    let w = Windowed::new(gpu, &values((n * k) as usize, seed + 2), lead[1]);
    let prior = values((m * k) as usize, seed + 3);
    let a = Windowed::new(gpu, &prior, lead[2]);
    let b = Windowed::new(gpu, &prior, lead[2]);
    let tiles = m.div_ceil(WGSL_TILE) * k.div_ceil(WGSL_TILE);
    let dispatch = |kind: usize, o: &Windowed| {
        gpu.dispatch_sliced(kind, &[&dy.buf, &w.buf, &o.buf], &[dy.range(), w.range(), o.range()], &[m, k, n, accumulate], Dispatch::Workgroups(tiles))
    };
    gpu.submit(&[], &[dispatch(K_DX, &a), dispatch(K_DX_REF, &b)]);
    gpu.poll_wait();
    let (ga, gb) = (a.bits(gpu), b.bits(gpu));
    let (lo, hi) = (a.lead as usize, a.lead as usize + a.len);
    assert!(ga[..lo].iter().chain(&ga[hi..]).all(|&v| v == SENTINEL), "native dx wrote outside its window at {m}x{k}x{n}");
    let first = ga[lo..hi].iter().zip(&gb[lo..hi]).position(|(p, q)| p != q);
    assert!(
        first.is_none(),
        "native dx GEMM differs from matmul_dx_reg at m={m} k={k} n={n} accumulate={accumulate} lead={lead:?}: first at {first:?} ({:?} vs {:?})",
        first.map(|i| f32::from_bits(ga[lo + i])),
        first.map(|i| f32::from_bits(gb[lo + i])),
    );
}

fn the_input_gradient_is_redirected_and_bit_identical() {
    let Some(gpu) = device() else { return };
    assert_eq!(gpu.native_kernel_for(K_DX, &[128, 64, 128, 0]), Some("matmul_f32_dx_reg"));
    assert_eq!(gpu.native_kernel_for(K_DX, &[128, 64, 128]), None, "short params");
    assert_eq!(gpu.native_kernel_for(K_DX_REF, &[128, 64, 128, 0]), None);
    for accumulate in [0u32, 1] {
        for n in [1u32, 5, 16, 33] {
            for (m, k) in [(1u32, 1u32), (1, 130), (127, 129), (128, 128), (129, 3), (200, 257)] {
                for lead in [[0, 0, 0], [1, 3, 5], [4, 4, 4]] {
                    check_dx(&gpu, m, k, n, accumulate, lead);
                }
            }
        }
    }
    let mut r = data::rng::Lcg::new(0xd0_0dad);
    for _ in 0..30 {
        let (m, k, n) = (1 + r.next_u32() % 400, 1 + r.next_u32() % 400, 1 + r.next_u32() % 300);
        let lead = [0, 1, 2].map(|_| u64::from(r.next_u32() % 8));
        check_dx(&gpu, m, k, n, r.next_u32() % 2, lead);
    }
    // FLUX.2 klein-4B training shapes: the input gradient of a single block's
    // fused linear1 and of linear2 over a paired 512 px joint sequence.
    for (m, k, n) in [(2560u32, 3072u32, 27648u32), (2560, 12288, 3072), (2560, 3072, 3072)] {
        check_dx(&gpu, m, k, n, 0, [0; 3]);
    }
}

/// `out[n,k] += sum_m a[m,n] * b[m,k]`, from identical prior contents.
fn check_dw(gpu: &Gpu, m: u32, k: u32, n: u32, lead: [u64; 3]) {
    let seed = u64::from(m) * 41 + u64::from(k) * 13 + u64::from(n);
    let a_in = Windowed::new(gpu, &values((m * n) as usize, seed + 1), lead[0]);
    let b_in = Windowed::new(gpu, &values((m * k) as usize, seed + 2), lead[1]);
    let prior = values((n * k) as usize, seed + 3);
    let a = Windowed::new(gpu, &prior, lead[2]);
    let b = Windowed::new(gpu, &prior, lead[2]);
    let tiles = n.div_ceil(WGSL_TILE) * k.div_ceil(WGSL_TILE);
    let dispatch = |kind: usize, o: &Windowed| {
        gpu.dispatch_sliced(kind, &[&a_in.buf, &b_in.buf, &o.buf], &[a_in.range(), b_in.range(), o.range()], &[m, k, n], Dispatch::Workgroups(tiles))
    };
    gpu.submit(&[], &[dispatch(K_DW, &a), dispatch(K_DW_REF, &b)]);
    gpu.poll_wait();
    let (ga, gb) = (a.bits(gpu), b.bits(gpu));
    let (lo, hi) = (a.lead as usize, a.lead as usize + a.len);
    assert!(ga[..lo].iter().chain(&ga[hi..]).all(|&v| v == SENTINEL), "native dw wrote outside its window at {m}x{k}x{n}");
    let first = ga[lo..hi].iter().zip(&gb[lo..hi]).position(|(p, q)| p != q);
    assert!(
        first.is_none(),
        "native dw GEMM differs from matmul_dw_reg at m={m} k={k} n={n} lead={lead:?}: first at {first:?} ({:?} vs {:?})",
        first.map(|i| f32::from_bits(ga[lo + i])),
        first.map(|i| f32::from_bits(gb[lo + i])),
    );
}

fn the_weight_gradient_is_redirected_and_bit_identical() {
    let Some(gpu) = device() else { return };
    assert_eq!(gpu.native_kernel_for(K_DW, &[128, 64, 128]), Some("matmul_f32_dw_reg"));
    assert_eq!(gpu.native_kernel_for(K_DW_REF, &[128, 64, 128]), None);
    for m in [1u32, 5, 16, 33] {
        for (n, k) in [(1u32, 1u32), (1, 130), (127, 129), (128, 128), (129, 3), (200, 257)] {
            for lead in [[0, 0, 0], [1, 3, 5], [4, 4, 4]] {
                check_dw(&gpu, m, k, n, lead);
            }
        }
    }
    let mut r = data::rng::Lcg::new(0xd_e1a7);
    for _ in 0..30 {
        let (m, k, n) = (1 + r.next_u32() % 400, 1 + r.next_u32() % 400, 1 + r.next_u32() % 300);
        let lead = [0, 1, 2].map(|_| u64::from(r.next_u32() % 8));
        check_dw(&gpu, m, k, n, lead);
    }
    // FLUX.2 klein-4B training: a head's attention weight gradient over a
    // paired 512 px joint sequence, and a full-width one.
    for (m, k, n) in [(2560u32, 128u32, 2560u32), (2560, 3072, 3072)] {
        check_dw(&gpu, m, k, n, [0; 3]);
    }
}

gpu_core::card_tests!(
    the_native_kernel_is_selected_for_every_reg3_dispatch,
    the_native_kernel_is_bit_identical_to_matmul_reg3,
    random_shapes_and_windows_are_bit_identical,
    the_model_shapes_are_bit_identical,
    the_input_gradient_is_redirected_and_bit_identical,
    the_weight_gradient_is_redirected_and_bit_identical,
);
