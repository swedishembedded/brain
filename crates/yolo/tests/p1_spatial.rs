// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference micro-checks for the P1 spatial + elementwise WGSL kernels
//! (conv-net primitives for the from-scratch YOLO detector). Every kernel is
//! loaded into the CPU backend (`Gpu::new_cpu`) and dispatched exactly like
//! gpu-core's own test (`storage_init` / `storage` / `step` / `submit` / `read`),
//! so these run headless with no GPU.
//!
//! Forward kernels are checked against plain-Rust references. Gradients are
//! checked against central differences: silu_bwd via a tolerance (smooth
//! nonlinearity), and the linear/selection ops (maxpool5_dx, upsample2_dx,
//! concat_split) near-exactly.

use gpu_core::Gpu;

macro_rules! wgsl {
    ($f:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/", $f))
    };
}

// ---------------------------------------------------------------------------
// Deterministic LCG (no rand crate) — produces values in (-1, 1).
// ---------------------------------------------------------------------------
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        // uniform in [-1, 1)
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.next_f32()).collect()
}

// ===========================================================================
// SiLU forward
// ===========================================================================
#[test]
fn silu_forward() {
    let gpu = Gpu::new_cpu(&[("silu", wgsl!("silu.wgsl"))]);
    let x = randvec(1, 257); // odd, > 64 to exercise the grid mask
    let xb = gpu.storage_init("x", &x);
    let ob = gpu.storage(x.len() as u64);
    let step = gpu.step(0, &[&xb, &ob], &[x.len() as u32], x.len() as u32);
    gpu.submit(&[], &[step]);
    let got = gpu.read(&ob, x.len());
    for (i, &v) in x.iter().enumerate() {
        let want = v / (1.0 + (-v).exp());
        assert!((got[i] - want).abs() < 1e-5, "silu[{i}]: {} vs {want}", got[i]);
    }
}

// ===========================================================================
// SiLU backward vs central differences
// ===========================================================================
#[test]
fn silu_backward_fd() {
    let n = 128usize;
    let x = randvec(2, n);
    let dy = randvec(3, n);

    let gpu = Gpu::new_cpu(&[("silu_bwd", wgsl!("silu_bwd.wgsl"))]);
    let xb = gpu.storage_init("x", &x);
    let dyb = gpu.storage_init("dy", &dy);
    let dxb = gpu.storage(n as u64);
    let step = gpu.step(0, &[&xb, &dyb, &dxb], &[n as u32], n as u32);
    gpu.submit(&[], &[step]);
    let dx = gpu.read(&dxb, n);

    // L = <dy, silu(x)>, dL/dx_i = dy_i * silu'(x_i). Central diff on silu(x_i).
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let eps = 1e-3f32;
    for i in 0..n {
        let fp = dy[i] * silu(x[i] + eps);
        let fm = dy[i] * silu(x[i] - eps);
        let fd = (fp - fm) / (2.0 * eps);
        let denom = fd.abs().max(1e-4);
        let rel = (dx[i] - fd).abs() / denom;
        assert!(rel < 2e-2, "silu_bwd[{i}]: {} vs fd {fd} (rel {rel})", dx[i]);
    }
}

// ===========================================================================
// maxpool5 forward + argmax correctness
// ===========================================================================
fn ref_maxpool5(x: &[f32], n: usize, c: usize, h: usize, w: usize, k: usize, pad: usize)
    -> (Vec<f32>, Vec<usize>) {
    let mut y = vec![0.0f32; n * c * h * w];
    let mut am = vec![0usize; n * c * h * w];
    for nn in 0..n {
        for cc in 0..c {
            for ho in 0..h {
                for wo in 0..w {
                    let mut best = f32::NEG_INFINITY;
                    let mut bi = 0usize;
                    for kh in 0..k {
                        let hi = ho as isize - pad as isize + kh as isize;
                        if hi < 0 || hi >= h as isize { continue; }
                        for kw in 0..k {
                            let wi = wo as isize - pad as isize + kw as isize;
                            if wi < 0 || wi >= w as isize { continue; }
                            let ii = ((nn * c + cc) * h + hi as usize) * w + wi as usize;
                            if x[ii] > best {
                                best = x[ii];
                                bi = ii;
                            }
                        }
                    }
                    let oi = ((nn * c + cc) * h + ho) * w + wo;
                    y[oi] = best;
                    am[oi] = bi;
                }
            }
        }
    }
    (y, am)
}

#[test]
fn maxpool5_forward_and_argmax() {
    let (n, c, h, w, k, pad) = (1usize, 2, 5, 5, 5, 2);
    let total = n * c * h * w;
    // Distinct values so the argmax is unambiguous.
    let x: Vec<f32> = randvec(7, total);

    let gpu = Gpu::new_cpu(&[("maxpool5", wgsl!("maxpool5.wgsl"))]);
    let xb = gpu.storage_init("x", &x);
    let yb = gpu.storage(total as u64);
    let amb = gpu.storage(total as u64);
    let params = [n as u32, c as u32, h as u32, w as u32, k as u32, pad as u32];
    let step = gpu.step(0, &[&xb, &yb, &amb], &params, total as u32);
    gpu.submit(&[], &[step]);
    let y = gpu.read(&yb, total);
    let am = gpu.read(&amb, total);

    let (ry, ram) = ref_maxpool5(&x, n, c, h, w, k, pad);
    for i in 0..total {
        assert!((y[i] - ry[i]).abs() < 1e-5, "maxpool y[{i}]: {} vs {}", y[i], ry[i]);
        let gi = am[i] as usize;
        // argmax points to the actual max element.
        assert_eq!(gi, ram[i], "maxpool argmax[{i}]: {gi} vs {}", ram[i]);
        assert!((x[gi] - ry[i]).abs() < 1e-5, "argmax[{i}] doesn't point to the max");
    }
}

// ===========================================================================
// maxpool5_dx — exact analytic gather grad.
// ===========================================================================
#[test]
fn maxpool5_dx_exact() {
    let (n, c, h, w, k, pad) = (1usize, 2, 5, 5, 5, 2);
    let total = n * c * h * w;
    let x: Vec<f32> = randvec(11, total);
    let dy: Vec<f32> = randvec(13, total);

    // Forward to obtain argmax.
    let (_, am) = ref_maxpool5(&x, n, c, h, w, k, pad);

    // Reference dx: scatter dy into the argmax input cell.
    let mut ref_dx = vec![0.0f32; total];
    for oi in 0..total {
        ref_dx[am[oi]] += dy[oi];
    }

    // Run the gather kernel (uses the GPU-produced argmax for self-consistency).
    let gpu = Gpu::new_cpu(&[
        ("maxpool5", wgsl!("maxpool5.wgsl")),
        ("maxpool5_dx", wgsl!("maxpool5_dx.wgsl")),
    ]);
    let xb = gpu.storage_init("x", &x);
    let yb = gpu.storage(total as u64);
    let amb = gpu.storage(total as u64);
    let params = [n as u32, c as u32, h as u32, w as u32, k as u32, pad as u32];
    let fwd = gpu.step(0, &[&xb, &yb, &amb], &params, total as u32);
    gpu.submit(&[], &[fwd]);

    let dyb = gpu.storage_init("dy", &dy);
    let dxb = gpu.storage(total as u64);
    let bwd = gpu.step(1, &[&dyb, &amb, &dxb], &params, total as u32);
    gpu.submit(&[], &[bwd]);
    let dx = gpu.read(&dxb, total);

    for i in 0..total {
        assert!((dx[i] - ref_dx[i]).abs() < 1e-4,
            "maxpool5_dx[{i}]: {} vs {}", dx[i], ref_dx[i]);
    }

    // Also cross-check the total grad mass is conserved (gather == scatter sum).
    let sum_dx: f32 = dx.iter().sum();
    let sum_dy: f32 = dy.iter().sum();
    assert!((sum_dx - sum_dy).abs() < 1e-3, "grad mass {sum_dx} vs {sum_dy}");
}

// ===========================================================================
// upsample2 forward
// ===========================================================================
#[test]
fn upsample2_forward() {
    let (n, c, h, w) = (1usize, 2, 3, 4);
    let total_in = n * c * h * w;
    let oh = 2 * h;
    let ow = 2 * w;
    let total_out = n * c * oh * ow;
    let x: Vec<f32> = randvec(17, total_in);

    let gpu = Gpu::new_cpu(&[("upsample2", wgsl!("upsample2.wgsl"))]);
    let xb = gpu.storage_init("x", &x);
    let yb = gpu.storage(total_out as u64);
    let params = [n as u32, c as u32, h as u32, w as u32];
    let step = gpu.step(0, &[&xb, &yb], &params, total_out as u32);
    gpu.submit(&[], &[step]);
    let y = gpu.read(&yb, total_out);

    for nn in 0..n {
        for cc in 0..c {
            for ho in 0..oh {
                for wo in 0..ow {
                    let oi = ((nn * c + cc) * oh + ho) * ow + wo;
                    let ii = ((nn * c + cc) * h + ho / 2) * w + wo / 2;
                    assert!((y[oi] - x[ii]).abs() < 1e-5, "upsample[{oi}]");
                }
            }
        }
    }
}

// ===========================================================================
// upsample2_dx — exact: dx == sum over 2x2 output block.
// ===========================================================================
#[test]
fn upsample2_dx_exact() {
    let (n, c, h, w) = (1usize, 2, 3, 4);
    let total_in = n * c * h * w;
    let oh = 2 * h;
    let ow = 2 * w;
    let total_out = n * c * oh * ow;
    let dy: Vec<f32> = randvec(19, total_out);

    // Reference: each input pixel sums its 2x2 output block.
    let mut ref_dx = vec![0.0f32; total_in];
    for nn in 0..n {
        for cc in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let mut acc = 0.0;
                    for dh in 0..2 {
                        for dw in 0..2 {
                            let ho = hi * 2 + dh;
                            let wo = wi * 2 + dw;
                            acc += dy[((nn * c + cc) * oh + ho) * ow + wo];
                        }
                    }
                    ref_dx[((nn * c + cc) * h + hi) * w + wi] = acc;
                }
            }
        }
    }

    let gpu = Gpu::new_cpu(&[("upsample2_dx", wgsl!("upsample2_dx.wgsl"))]);
    let dyb = gpu.storage_init("dy", &dy);
    let dxb = gpu.storage(total_in as u64);
    let params = [n as u32, c as u32, h as u32, w as u32];
    let step = gpu.step(0, &[&dyb, &dxb], &params, total_in as u32);
    gpu.submit(&[], &[step]);
    let dx = gpu.read(&dxb, total_in);

    for i in 0..total_in {
        assert!((dx[i] - ref_dx[i]).abs() < 1e-4,
            "upsample2_dx[{i}]: {} vs {}", dx[i], ref_dx[i]);
    }
}

// ===========================================================================
// concat2 forward
// ===========================================================================
#[test]
fn concat2_forward() {
    let (n, ca, cb, h, w) = (1usize, 2, 3, 3, 4);
    let na = n * ca * h * w;
    let nb = n * cb * h * w;
    let ctot = ca + cb;
    let ny = n * ctot * h * w;
    let a: Vec<f32> = randvec(23, na);
    let b: Vec<f32> = randvec(29, nb);

    let gpu = Gpu::new_cpu(&[("concat2", wgsl!("concat2.wgsl"))]);
    let ab = gpu.storage_init("a", &a);
    let bb = gpu.storage_init("b", &b);
    let yb = gpu.storage(ny as u64);
    let params = [n as u32, ca as u32, cb as u32, h as u32, w as u32];
    let step = gpu.step(0, &[&ab, &bb, &yb], &params, ny as u32);
    gpu.submit(&[], &[step]);
    let y = gpu.read(&yb, ny);

    for nn in 0..n {
        for cc in 0..ctot {
            for hh in 0..h {
                for ww in 0..w {
                    let oi = ((nn * ctot + cc) * h + hh) * w + ww;
                    let want = if cc < ca {
                        a[((nn * ca + cc) * h + hh) * w + ww]
                    } else {
                        b[((nn * cb + (cc - ca)) * h + hh) * w + ww]
                    };
                    assert!((y[oi] - want).abs() < 1e-5, "concat[{oi}]");
                }
            }
        }
    }
}

// ===========================================================================
// concat_split — splitting a concat2 grad back to a,b reconstructs dy exactly.
// ===========================================================================
#[test]
fn concat_split_reconstructs() {
    let (n, ca, cb, h, w) = (1usize, 2, 3, 3, 4);
    let ctot = ca + cb;
    let na = n * ca * h * w;
    let nb = n * cb * h * w;
    let ny = n * ctot * h * w;
    let dy: Vec<f32> = randvec(31, ny);

    let gpu = Gpu::new_cpu(&[("concat_split", wgsl!("concat_split.wgsl"))]);
    let dyb = gpu.storage_init("dy", &dy);

    // Split into `da` (Csrc=Ca, c_off=0).
    let dab = gpu.storage(na as u64);
    let pa = [n as u32, ctot as u32, ca as u32, 0u32, h as u32, w as u32];
    let sa = gpu.step(0, &[&dyb, &dab], &pa, na as u32);
    // Split into `db` (Csrc=Cb, c_off=Ca).
    let dbb = gpu.storage(nb as u64);
    let pb = [n as u32, ctot as u32, cb as u32, ca as u32, h as u32, w as u32];
    let sb = gpu.step(0, &[&dyb, &dbb], &pb, nb as u32);
    gpu.submit(&[], &[sa, sb]);

    let da = gpu.read(&dab, na);
    let db = gpu.read(&dbb, nb);

    // Reconstruct dy by re-concatenating the two halves; must equal dy exactly.
    for nn in 0..n {
        for cc in 0..ctot {
            for hh in 0..h {
                for ww in 0..w {
                    let yi = ((nn * ctot + cc) * h + hh) * w + ww;
                    let got = if cc < ca {
                        da[((nn * ca + cc) * h + hh) * w + ww]
                    } else {
                        db[((nn * cb + (cc - ca)) * h + hh) * w + ww]
                    };
                    assert!((got - dy[yi]).abs() < 1e-4,
                        "concat_split reconstruct[{yi}]: {got} vs {}", dy[yi]);
                }
            }
        }
    }
}
