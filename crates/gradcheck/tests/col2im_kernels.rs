// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The GEMM-lowered conv backward must reproduce the direct one, exactly.
//!
//! `conv2d_dx` computes `dX` directly: one invocation per input pixel, reducing
//! over `Cout * K * K`. The lowered path computes the same value as
//!
//!     dcol[HW, CinKK] = dY[HW, Cout] . W[Cout, CinKK]     (matmul_dx_reg)
//!     dX              = col2im(dcol)                      (col2im, K*K per pixel)
//!
//! moving the `Cout` reduction into a register-tiled GEMM. That is only worth
//! doing if it is the SAME number, so this gates it against the kernel already
//! in use — the one every conv-net gradcheck in the tree currently validates.
//!
//! Run on BOTH backends: `col2im` is barrier-free by construction, so
//! `backend-cpu` must agree too, and if it ever does not the JIT is
//! mis-executing something the GPU tolerates.

use gpu_core::Gpu;

const KERNELS: [(&str, &str); 4] = [
    ("conv2d_dx", kernels::CONV2D_DX),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("col2im", kernels::COL2IM),
    ("nchw_nlc", kernels::NCHW_NLC),
];
const K_CONV_DX: usize = 0;
const K_MM_DX: usize = 1;
const K_COL2IM: usize = 2;
const K_NCHW_NLC: usize = 3;

struct Case {
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    k: u32,
    stride: u32,
    pad: u32,
}

impl Case {
    fn ho(&self) -> u32 {
        (self.h + 2 * self.pad - self.k) / self.stride + 1
    }
    fn wo(&self) -> u32 {
        (self.w + 2 * self.pad - self.k) / self.stride + 1
    }
}

/// `data::rng::Lcg` is the sanctioned test/fixture RNG (b3aa5cc) — this file
/// used to carry its own copy of the same constants (audit F40).
fn rnd(n: usize, seed: u64) -> Vec<f32> {
    data::rng::Lcg::new(seed).vec_scaled(n, 1.0)
}

/// `dX` both ways. Returns `(direct, lowered)`.
fn both_ways(gpu: &Gpu, c: &Case) -> (Vec<f32>, Vec<f32>) {
    let (ho, wo) = (c.ho(), c.wo());
    let hw = (ho * wo) as usize;
    let cinkk = (c.cin * c.k * c.k) as usize;
    let dy = rnd((c.cout * ho * wo) as usize, 7);
    let wt = rnd((c.cout * c.cin * c.k * c.k) as usize, 11);

    let dy_b = gpu.storage(dy.len() as u64);
    gpu.write_f32(&dy_b, &dy);
    let w_b = gpu.storage(wt.len() as u64);
    gpu.write_f32(&w_b, &wt);

    // ---- direct: conv2d_dx over [N,Cout,Ho,Wo] and [Cout,Cin,K,K].
    let n_in = (c.cin * c.h * c.w) as usize;
    let dx_direct = gpu.storage(n_in as u64);
    gpu.submit(
        &[],
        &[gpu.step(
            K_CONV_DX,
            &[&dy_b, &w_b, &dx_direct],
            &[1, c.cin, c.h, c.w, c.cout, c.k, c.stride, c.pad, ho, wo],
            n_in as u32,
        )],
    );
    gpu.poll_wait();
    let direct = gpu.read(&dx_direct, n_in);

    // ---- lowered: dY -> [HW, Cout], GEMM against W[Cout, CinKK], then col2im.
    let dy_nlc = gpu.storage(dy.len() as u64);
    let dcol = gpu.storage((hw * cinkk) as u64);
    let dx_low = gpu.storage(n_in as u64);
    gpu.submit(
        &[],
        &[
            // NCHW -> NLC: [1,Cout,Ho,Wo] -> [HW, Cout].
            gpu.step(K_NCHW_NLC, &[&dy_b, &dy_nlc], &[c.cout * hw as u32, c.cout, hw as u32], c.cout * hw as u32),
            // dcol[m,k] = sum_n dY[m,n] * W[n,k];  m=HW, n=Cout, k=CinKK.
            // `accumulate = 0`: dcol is written, not accumulated.
            gpu.step(
                K_MM_DX,
                &[&dy_nlc, &w_b, &dcol],
                &[hw as u32, cinkk as u32, c.cout, 0],
                (hw as u32).div_ceil(128) * (cinkk as u32).div_ceil(128) * 256,
            ),
            gpu.step(
                K_COL2IM,
                &[&dcol, &dx_low],
                &[1, c.cin, c.h, c.w, c.k, c.stride, c.pad, ho, wo, cinkk as u32],
                n_in as u32,
            ),
        ],
    );
    gpu.poll_wait();
    (direct, gpu.read(&dx_low, n_in))
}

fn check(gpu: &Gpu, label: &str, c: &Case) {
    let (a, b) = both_ways(gpu, c);
    let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    eprintln!("  {label:34} max|delta| = {max:.3e}   (scale {scale:.3e}, rel {:.3e})", max / scale);
    // fp32 reduction-order differences only: the GEMM sums Cout in a different
    // order than the direct kernel, so this is a tolerance, not an equality.
    assert!(max / scale < 1e-5, "{label}: lowered dX differs from conv2d_dx by rel {:.3e}", max / scale);
}

fn cases() -> Vec<(&'static str, Case)> {
    vec![
        // The VAE decoder's shape: 3x3, same padding, square.
        ("3x3 s1 p1  cin64 cout64 32x32", Case { cin: 64, cout: 64, h: 32, w: 32, k: 3, stride: 1, pad: 1 }),
        // Widths that all DIFFER, so a Cin/Cout swap cannot pass (lessons #4).
        ("3x3 s1 p1  cin8  cout16 9x7", Case { cin: 8, cout: 16, h: 9, w: 7, k: 3, stride: 1, pad: 1 }),
        // Strided + the asymmetric-output case, where the ho/wo arithmetic bites.
        ("3x3 s2 p1  cin8  cout16 9x7", Case { cin: 8, cout: 16, h: 9, w: 7, k: 3, stride: 2, pad: 1 }),
        // 1x1: no spatial reduction at all, the degenerate end.
        ("1x1 s1 p0  cin16 cout8  8x8", Case { cin: 16, cout: 8, h: 8, w: 8, k: 1, stride: 1, pad: 0 }),
        // Unpadded 5x5: every tap in range, no boundary masking.
        ("5x5 s1 p0  cin4  cout6  12x10", Case { cin: 4, cout: 6, h: 12, w: 10, k: 5, stride: 1, pad: 0 }),
    ]
}

#[test]
fn lowered_dx_matches_conv2d_dx() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    eprintln!("col2im lowering vs conv2d_dx:");
    for (label, c) in cases() {
        check(&gpu, label, &c);
    }
}

/// The lowering must be right where it is easiest to get wrong: the tap packing
/// `(ci*K + kh)*K + kw` has to match `im2col_at`'s exactly. A transposed read is
/// still a plausible number — it just belongs to a different pixel — so this
/// asserts against a shape where Cin, K and the spatial dims are all distinct.
#[test]
fn the_tap_packing_matches_im2col() {
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let c = Case { cin: 3, cout: 5, h: 7, w: 11, k: 3, stride: 1, pad: 1 };
    check(&gpu, "tap packing cin3 k3 7x11", &c);
}
