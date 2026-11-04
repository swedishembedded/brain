// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1 conv-net primitive verification: forward correctness + finite-difference
//! micro-gradient-checks for conv2d / conv2d_dx / conv2d_dw on the CPU backend.
//!
//! No GPU required. Random data comes from a deterministic xorshift so failures
//! are reproducible. The backward kernels are checked against the numerical
//! gradient of the scalar loss  L = sum_j dy_j * y_j  (dy fixed-random), whose
//! analytic gradients w.r.t. the input and weights are exactly what conv2d_dx /
//! conv2d_dw compute.

const CONV2D: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/conv2d.wgsl"));
const CONV2D_DX: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/conv2d_dx.wgsl"));
const CONV2D_DW: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/conv2d_dw.wgsl"));

#[derive(Clone, Copy)]
struct Cfg {
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl Cfg {
    fn ho(&self) -> usize {
        (self.h + 2 * self.pad - self.k) / self.stride + 1
    }
    fn wo(&self) -> usize {
        (self.w + 2 * self.pad - self.k) / self.stride + 1
    }
    fn x_len(&self) -> usize {
        self.n * self.cin * self.h * self.w
    }
    fn w_len(&self) -> usize {
        self.cout * self.cin * self.k * self.k
    }
    fn y_len(&self) -> usize {
        self.n * self.cout * self.ho() * self.wo()
    }
    fn params(&self) -> [u32; 10] {
        [
            self.n as u32,
            self.cin as u32,
            self.h as u32,
            self.w as u32,
            self.cout as u32,
            self.k as u32,
            self.stride as u32,
            self.pad as u32,
            self.ho() as u32,
            self.wo() as u32,
        ]
    }
}

/// Deterministic xorshift32 -> values in roughly [-1, 1].
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        // map to [-1, 1)
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Plain-Rust reference forward convolution (implicit zero pad).
fn ref_conv(cfg: &Cfg, x: &[f32], w: &[f32]) -> Vec<f32> {
    let (ho, wo) = (cfg.ho(), cfg.wo());
    let mut y = vec![0.0f32; cfg.y_len()];
    for n in 0..cfg.n {
        for co in 0..cfg.cout {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = 0.0f32;
                    for ci in 0..cfg.cin {
                        for kh in 0..cfg.k {
                            for kw in 0..cfg.k {
                                let hi = oh * cfg.stride + kh;
                                let wi = ow * cfg.stride + kw;
                                if hi < cfg.pad || wi < cfg.pad {
                                    continue;
                                }
                                let hi = hi - cfg.pad;
                                let wi = wi - cfg.pad;
                                if hi >= cfg.h || wi >= cfg.w {
                                    continue;
                                }
                                let x_idx = ((n * cfg.cin + ci) * cfg.h + hi) * cfg.w + wi;
                                let w_idx = ((co * cfg.cin + ci) * cfg.k + kh) * cfg.k + kw;
                                acc += x[x_idx] * w[w_idx];
                            }
                        }
                    }
                    let y_idx = ((n * cfg.cout + co) * ho + oh) * wo + ow;
                    y[y_idx] = acc;
                }
            }
        }
    }
    y
}

/// Run the forward kernel on the CPU backend.
fn gpu_forward(cfg: &Cfg, x: &[f32], w: &[f32]) -> Vec<f32> {
    let gpu = gpu_core::Gpu::new_cpu(&[("conv2d", CONV2D)]);
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let yb = gpu.storage(cfg.y_len() as u64);
    let step = gpu.step(0, &[&xb, &wb, &yb], &cfg.params(), cfg.y_len() as u32);
    gpu.submit(&[], &[step]);
    gpu.read(&yb, cfg.y_len())
}

fn gpu_dx(cfg: &Cfg, dy: &[f32], w: &[f32]) -> Vec<f32> {
    let gpu = gpu_core::Gpu::new_cpu(&[("conv2d_dx", CONV2D_DX)]);
    let dyb = gpu.storage_init("dy", dy);
    let wb = gpu.storage_init("w", w);
    let dxb = gpu.storage(cfg.x_len() as u64);
    let step = gpu.step(0, &[&dyb, &wb, &dxb], &cfg.params(), cfg.x_len() as u32);
    gpu.submit(&[], &[step]);
    gpu.read(&dxb, cfg.x_len())
}

fn gpu_dw(cfg: &Cfg, dy: &[f32], x: &[f32]) -> Vec<f32> {
    let gpu = gpu_core::Gpu::new_cpu(&[("conv2d_dw", CONV2D_DW)]);
    let dyb = gpu.storage_init("dy", dy);
    let xb = gpu.storage_init("x", x);
    let dwb = gpu.storage(cfg.w_len() as u64);
    let step = gpu.step(0, &[&dyb, &xb, &dwb], &cfg.params(), cfg.w_len() as u32);
    // dw accumulates -> the buffer must be pre-zeroed via clears.
    gpu.submit(&[&dwb], &[step]);
    gpu.read(&dwb, cfg.w_len())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Loss L = <dy, conv(x, w)> recomputed via the forward kernel.
fn loss(cfg: &Cfg, x: &[f32], w: &[f32], dy: &[f32]) -> f32 {
    dot(&gpu_forward(cfg, x, w), dy)
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Combined error: passes if EITHER the relative error is within `REL_TOL` OR
/// the absolute error is within `ABS_TOL`. The absolute fallback matters when
/// the analytic gradient is small: the central difference at eps=1e-3 carries
/// ~O(eps^2) truncation plus f32 cancellation noise on the order of 1e-3, which
/// blows up the *relative* error of a ~1e-2 gradient even though the kernel is
/// exact (the dw checks and stride-2 dx checks confirm the index math).
fn grad_err(numeric: f32, analytic: f32) -> f32 {
    let abs = (numeric - analytic).abs();
    let rel = abs / analytic.abs().max(1e-12);
    // Effective error = min of the two normalised measures.
    rel.min(abs / ABS_TOL * REL_TOL)
}

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

fn check_forward(cfg: Cfg, seed: u32) {
    let mut rng = Rng::new(seed);
    let x = rng.vec(cfg.x_len());
    let w = rng.vec(cfg.w_len());
    let y_gpu = gpu_forward(&cfg, &x, &w);
    let y_ref = ref_conv(&cfg, &x, &w);
    let err = max_abs_err(&y_gpu, &y_ref);
    assert!(
        err < 1e-4,
        "forward mismatch (stride={}, pad={}): max abs err {err}",
        cfg.stride,
        cfg.pad
    );
}

#[test]
fn forward_stride1_pad1() {
    check_forward(
        Cfg { n: 1, cin: 2, h: 4, w: 4, cout: 3, k: 3, stride: 1, pad: 1 },
        0x1234_5678,
    );
}

#[test]
fn forward_stride1_pad0() {
    check_forward(
        Cfg { n: 2, cin: 2, h: 5, w: 5, cout: 3, k: 3, stride: 1, pad: 0 },
        0x0BAD_F00D,
    );
}

#[test]
fn forward_stride2_pad1() {
    check_forward(
        Cfg { n: 1, cin: 2, h: 5, w: 5, cout: 3, k: 3, stride: 2, pad: 1 },
        0xDEAD_BEEF,
    );
}

#[test]
fn forward_stride2_pad0() {
    check_forward(
        Cfg { n: 2, cin: 3, h: 6, w: 7, cout: 2, k: 3, stride: 2, pad: 0 },
        0xCAFE_BABE,
    );
}

// ---------------------------------------------------------------------------
// Gradient micro-checks (core deliverable)
// ---------------------------------------------------------------------------

const EPS: f32 = 1e-3;
const REL_TOL: f32 = 2e-2;
/// Absolute-error fallback for small-magnitude gradients (see `grad_err`).
const ABS_TOL: f32 = 2e-3;

/// Verify conv2d_dx against central finite differences on L = <dy, y>.
fn check_dx(cfg: Cfg, seed: u32) {
    let mut rng = Rng::new(seed);
    let x = rng.vec(cfg.x_len());
    let w = rng.vec(cfg.w_len());
    let dy = rng.vec(cfg.y_len());

    let dx = gpu_dx(&cfg, &dy, &w);

    // Sample several random input coordinates.
    let n_checks = 12.min(cfg.x_len());
    for _ in 0..n_checks {
        let i = (rng.next_u32() as usize) % cfg.x_len();
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += EPS;
        xm[i] -= EPS;
        let num = (loss(&cfg, &xp, &w, &dy) - loss(&cfg, &xm, &w, &dy)) / (2.0 * EPS);
        let ana = dx[i];
        let re = grad_err(num, ana);
        assert!(
            re < REL_TOL,
            "dx[{i}] (stride={}, pad={}): numeric {num}, analytic {ana}, rel err {re}",
            cfg.stride,
            cfg.pad
        );
    }
}

/// Verify conv2d_dw against central finite differences on L = <dy, y>.
fn check_dw(cfg: Cfg, seed: u32) {
    let mut rng = Rng::new(seed);
    let x = rng.vec(cfg.x_len());
    let w = rng.vec(cfg.w_len());
    let dy = rng.vec(cfg.y_len());

    let dw = gpu_dw(&cfg, &dy, &x);

    let n_checks = 12.min(cfg.w_len());
    for _ in 0..n_checks {
        let i = (rng.next_u32() as usize) % cfg.w_len();
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp[i] += EPS;
        wm[i] -= EPS;
        let num = (loss(&cfg, &x, &wp, &dy) - loss(&cfg, &x, &wm, &dy)) / (2.0 * EPS);
        let ana = dw[i];
        let re = grad_err(num, ana);
        assert!(
            re < REL_TOL,
            "dw[{i}] (stride={}, pad={}): numeric {num}, analytic {ana}, rel err {re}",
            cfg.stride,
            cfg.pad
        );
    }
}

#[test]
fn dx_stride1_pad1() {
    check_dx(
        Cfg { n: 1, cin: 2, h: 4, w: 4, cout: 3, k: 3, stride: 1, pad: 1 },
        0x1111_2222,
    );
}

#[test]
fn dx_stride2_pad1() {
    check_dx(
        Cfg { n: 1, cin: 2, h: 5, w: 5, cout: 3, k: 3, stride: 2, pad: 1 },
        0x3333_4444,
    );
}

#[test]
fn dx_stride2_pad0() {
    check_dx(
        Cfg { n: 2, cin: 2, h: 6, w: 6, cout: 2, k: 3, stride: 2, pad: 0 },
        0x5555_6666,
    );
}

#[test]
fn dw_stride1_pad1() {
    check_dw(
        Cfg { n: 1, cin: 2, h: 4, w: 4, cout: 3, k: 3, stride: 1, pad: 1 },
        0x7777_8888,
    );
}

#[test]
fn dw_stride2_pad1() {
    check_dw(
        Cfg { n: 1, cin: 2, h: 5, w: 5, cout: 3, k: 3, stride: 2, pad: 1 },
        0x9999_AAAA,
    );
}

#[test]
fn dw_stride2_pad0() {
    check_dw(
        Cfg { n: 2, cin: 2, h: 6, w: 6, cout: 2, k: 3, stride: 2, pad: 0 },
        0xBBBB_CCCC,
    );
}
