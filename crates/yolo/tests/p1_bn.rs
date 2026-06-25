// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference micro-checks for the BatchNorm WGSL kernels on the CPU
//! backend (no GPU). Verifies bn_train forward against a plain-Rust reference,
//! and the full backward chain (bn_dstats -> bn_dx / bn_dgamma / bn_dbeta)
//! against numerical gradients of the scalar loss L = sum_j dy_j * y_j.

use gpu_core::{f, Gpu};

const BN_STATS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_stats.wgsl"
));
const BN_TRAIN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_train.wgsl"
));
const BN_EVAL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_eval.wgsl"
));
const BN_RUNNING: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_running.wgsl"
));
const BN_DSTATS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_dstats.wgsl"
));
const BN_DX: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_dx.wgsl"
));
const BN_DGAMMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_dgamma.wgsl"
));
const BN_DBETA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../kernels/wgsl/bn_dbeta.wgsl"
));

// Kernel indices into the JIT set (order passed to new_cpu).
const K_STATS: usize = 0;
const K_TRAIN: usize = 1;
const K_EVAL: usize = 2;
const K_RUNNING: usize = 3;
const K_DSTATS: usize = 4;
const K_DX: usize = 5;
const K_DGAMMA: usize = 6;
const K_DBETA: usize = 7;

const EPS: f32 = 1e-5;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("bn_stats", BN_STATS),
        ("bn_train", BN_TRAIN),
        ("bn_eval", BN_EVAL),
        ("bn_running", BN_RUNNING),
        ("bn_dstats", BN_DSTATS),
        ("bn_dx", BN_DX),
        ("bn_dgamma", BN_DGAMMA),
        ("bn_dbeta", BN_DBETA),
    ]
}

// Deterministic LCG -> uniform in [-1, 1].
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.0 >> 33) as u32; // top 31 bits
        (bits as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }
}

// ---- Plain-Rust references (population stats, NCHW) ----

fn ref_stats(x: &[f32], n: usize, c: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let m = (n * h * w) as f32;
    let mut mean = vec![0.0f32; c];
    let mut var = vec![0.0f32; c];
    for cc in 0..c {
        let mut s = 0.0f32;
        for nn in 0..n {
            for hh in 0..h {
                for ww in 0..w {
                    s += x[((nn * c + cc) * h + hh) * w + ww];
                }
            }
        }
        let mu = s / m;
        let mut v = 0.0f32;
        for nn in 0..n {
            for hh in 0..h {
                for ww in 0..w {
                    let d = x[((nn * c + cc) * h + hh) * w + ww] - mu;
                    v += d * d;
                }
            }
        }
        mean[cc] = mu;
        var[cc] = v / m;
    }
    (mean, var)
}

fn ref_forward(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for nn in 0..n {
        for cc in 0..c {
            let inv = 1.0 / (var[cc] + EPS).sqrt();
            for hh in 0..h {
                for ww in 0..w {
                    let i = ((nn * c + cc) * h + hh) * w + ww;
                    y[i] = (x[i] - mean[cc]) * inv * gamma[cc] + beta[cc];
                }
            }
        }
    }
    y
}

// L(theta) using BATCH stats recomputed from x (so x perturbations move stats).
fn ref_loss(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    dy: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> f32 {
    let (mean, var) = ref_stats(x, n, c, h, w);
    let y = ref_forward(x, &mean, &var, gamma, beta, n, c, h, w);
    y.iter().zip(dy).map(|(a, b)| a * b).sum()
}

// Pack helpers matching the kernel ABIs.
fn pack_mv(mean: &[f32], var: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(2 * mean.len());
    for c in 0..mean.len() {
        v.push(mean[c]);
        v.push(var[c]);
    }
    v
}
fn pack_gb(gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(2 * gamma.len());
    for c in 0..gamma.len() {
        v.push(gamma[c]);
        v.push(beta[c]);
    }
    v
}
fn pack_mvg(mean: &[f32], var: &[f32], gamma: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(3 * mean.len());
    for c in 0..mean.len() {
        v.push(mean[c]);
        v.push(var[c]);
        v.push(gamma[c]);
    }
    v
}

// GPU forward via bn_stats + bn_train. Returns (mean, var, y).
fn gpu_train_forward(
    gpu: &Gpu,
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: u32,
    c: u32,
    h: u32,
    w: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = (n * c * h * w) as usize;
    let cc = c as usize;
    let xb = gpu.storage_init("x", x);
    let meanb = gpu.storage(c as u64);
    let varb = gpu.storage(c as u64);
    let s = gpu.step(K_STATS, &[&xb, &meanb, &varb], &[n, c, h, w], c);
    gpu.submit(&[], &[s]);
    let mean = gpu.read(&meanb, cc);
    let var = gpu.read(&varb, cc);

    let mv = gpu.storage_init("mv", &pack_mv(&mean, &var));
    let gb = gpu.storage_init("gb", &pack_gb(gamma, beta));
    let outb = gpu.storage(total as u64);
    let s2 = gpu.step(K_TRAIN, &[&xb, &mv, &gb, &outb], &[n, c, h, w], n * c * h * w);
    gpu.submit(&[], &[s2]);
    let y = gpu.read(&outb, total);
    (mean, var, y)
}

#[test]
fn bn_train_forward_matches_reference() {
    let (n, c, h, w) = (4usize, 3usize, 2usize, 2usize);
    let total = n * c * h * w;
    let mut rng = Lcg::new(0xDEAD_BEEF);
    let x: Vec<f32> = (0..total).map(|_| rng.next_f32() * 2.0).collect();
    let gamma: Vec<f32> = (0..c).map(|_| 0.5 + rng.next_f32()).collect();
    let beta: Vec<f32> = (0..c).map(|_| rng.next_f32()).collect();

    let (rmean, rvar) = ref_stats(&x, n, c, h, w);
    let yref = ref_forward(&x, &rmean, &rvar, &gamma, &beta, n, c, h, w);

    let gpu = Gpu::new_cpu(&kernels());
    let (gmean, gvar, y) =
        gpu_train_forward(&gpu, &x, &gamma, &beta, n as u32, c as u32, h as u32, w as u32);

    for cc in 0..c {
        assert!((gmean[cc] - rmean[cc]).abs() < 1e-4, "mean[{cc}]");
        assert!((gvar[cc] - rvar[cc]).abs() < 1e-4, "var[{cc}]");
    }
    for i in 0..total {
        assert!(
            (y[i] - yref[i]).abs() < 1e-4,
            "y[{i}]: gpu={} ref={}",
            y[i],
            yref[i]
        );
    }
}

#[test]
fn bn_eval_matches_running_stat_forward() {
    // bn_eval uses provided (running) stats directly — no recompute.
    let (n, c, h, w) = (2usize, 3usize, 2usize, 2usize);
    let total = n * c * h * w;
    let mut rng = Lcg::new(0x1234_5678);
    let x: Vec<f32> = (0..total).map(|_| rng.next_f32() * 2.0).collect();
    let run_mean: Vec<f32> = (0..c).map(|_| rng.next_f32()).collect();
    let run_var: Vec<f32> = (0..c).map(|_| 0.5 + rng.next_f32().abs()).collect();
    let gamma: Vec<f32> = (0..c).map(|_| 0.5 + rng.next_f32()).collect();
    let beta: Vec<f32> = (0..c).map(|_| rng.next_f32()).collect();

    let yref = ref_forward(&x, &run_mean, &run_var, &gamma, &beta, n, c, h, w);

    let gpu = Gpu::new_cpu(&kernels());
    let xb = gpu.storage_init("x", &x);
    let mv = gpu.storage_init("mv", &pack_mv(&run_mean, &run_var));
    let gb = gpu.storage_init("gb", &pack_gb(&gamma, &beta));
    let outb = gpu.storage(total as u64);
    let s = gpu.step(
        K_EVAL,
        &[&xb, &mv, &gb, &outb],
        &[n as u32, c as u32, h as u32, w as u32],
        (n * c * h * w) as u32,
    );
    gpu.submit(&[], &[s]);
    let y = gpu.read(&outb, total);
    for i in 0..total {
        assert!((y[i] - yref[i]).abs() < 1e-4, "y[{i}]");
    }
}

#[test]
fn bn_running_momentum_update() {
    let c = 4usize;
    let m = 0.1f32;
    let mut rng = Lcg::new(0xABCD);
    let mean: Vec<f32> = (0..c).map(|_| rng.next_f32()).collect();
    let var: Vec<f32> = (0..c).map(|_| 0.5 + rng.next_f32().abs()).collect();
    let run_mean0: Vec<f32> = (0..c).map(|_| rng.next_f32()).collect();
    let run_var0: Vec<f32> = (0..c).map(|_| 0.5 + rng.next_f32().abs()).collect();

    let gpu = Gpu::new_cpu(&kernels());
    let meanb = gpu.storage_init("mean", &mean);
    let varb = gpu.storage_init("var", &var);
    let rmb = gpu.storage_init("rm", &run_mean0);
    let rvb = gpu.storage_init("rv", &run_var0);
    let s = gpu.step(
        K_RUNNING,
        &[&meanb, &varb, &rmb, &rvb],
        &[c as u32, f(m)],
        c as u32,
    );
    gpu.submit(&[], &[s]);
    let rm = gpu.read(&rmb, c);
    let rv = gpu.read(&rvb, c);
    for cc in 0..c {
        let er = (1.0 - m) * run_mean0[cc] + m * mean[cc];
        let ev = (1.0 - m) * run_var0[cc] + m * var[cc];
        assert!((rm[cc] - er).abs() < 1e-5, "run_mean[{cc}]");
        assert!((rv[cc] - ev).abs() < 1e-5, "run_var[{cc}]");
    }
}

#[test]
fn bn_backward_finite_difference() {
    let (n, c, h, w) = (4usize, 3usize, 2usize, 2usize);
    let (nu, cu, hu, wu) = (n as u32, c as u32, h as u32, w as u32);
    let total = n * c * h * w;
    let mut rng = Lcg::new(0xF00D_CAFE);
    let x: Vec<f32> = (0..total).map(|_| rng.next_f32() * 2.0).collect();
    let gamma: Vec<f32> = (0..c).map(|_| 0.5 + rng.next_f32()).collect();
    let beta: Vec<f32> = (0..c).map(|_| rng.next_f32()).collect();
    let dy: Vec<f32> = (0..total).map(|_| rng.next_f32()).collect();

    let gpu = Gpu::new_cpu(&kernels());

    // --- Analytic backward via kernels ---
    let (mean, var, _y) =
        gpu_train_forward(&gpu, &x, &gamma, &beta, nu, cu, hu, wu);

    let xb = gpu.storage_init("x", &x);
    let dyb = gpu.storage_init("dy", &dy);
    let mvg = gpu.storage_init("mvg", &pack_mvg(&mean, &var, &gamma));
    let bpb = gpu.storage(5 * c as u64);
    let sd = gpu.step(K_DSTATS, &[&xb, &dyb, &mvg, &bpb], &[nu, cu, hu, wu], cu);
    gpu.submit(&[], &[sd]);

    // dx
    let dxb = gpu.storage(total as u64);
    let sdx = gpu.step(K_DX, &[&xb, &dyb, &bpb, &dxb], &[nu, cu, hu, wu], (n * c * h * w) as u32);
    gpu.submit(&[], &[sdx]);
    let dx = gpu.read(&dxb, total);

    // dgamma (accumulate kernel: pre-zero its out via submit clears)
    let mvb = gpu.storage_init("mv", &pack_mv(&mean, &var));
    let dgb = gpu.storage(c as u64);
    let sdg = gpu.step(K_DGAMMA, &[&xb, &dyb, &mvb, &dgb], &[nu, cu, hu, wu], cu);
    gpu.submit(&[&dgb], &[sdg]);
    let dgamma = gpu.read(&dgb, c);

    // dbeta (accumulate kernel: pre-zero its out via submit clears)
    let dbb = gpu.storage(c as u64);
    let sdb = gpu.step(K_DBETA, &[&dyb, &dbb], &[nu, cu, hu, wu], cu);
    gpu.submit(&[&dbb], &[sdb]);
    let dbeta = gpu.read(&dbb, c);

    // --- Numerical gradients of L = sum_j dy_j * y_j ---
    let fd = 1e-3f32;
    let rel = |num: f32, ana: f32| -> f32 {
        let denom = num.abs().max(ana.abs()).max(1e-6);
        (num - ana).abs() / denom
    };

    // dL/dx_i  (perturb x — stats recomputed inside ref_loss, as required)
    for i in 0..total {
        let mut xp = x.clone();
        xp[i] += fd;
        let lp = ref_loss(&xp, &gamma, &beta, &dy, n, c, h, w);
        let mut xm = x.clone();
        xm[i] -= fd;
        let lm = ref_loss(&xm, &gamma, &beta, &dy, n, c, h, w);
        let num = (lp - lm) / (2.0 * fd);
        assert!(
            rel(num, dx[i]) < 3e-2,
            "dx[{i}]: num={num} ana={} rel={}",
            dx[i],
            rel(num, dx[i])
        );
    }

    // dL/dgamma_c
    for cc in 0..c {
        let mut gp = gamma.clone();
        gp[cc] += fd;
        let lp = ref_loss(&x, &gp, &beta, &dy, n, c, h, w);
        let mut gm = gamma.clone();
        gm[cc] -= fd;
        let lm = ref_loss(&x, &gm, &beta, &dy, n, c, h, w);
        let num = (lp - lm) / (2.0 * fd);
        assert!(
            rel(num, dgamma[cc]) < 3e-2,
            "dgamma[{cc}]: num={num} ana={}",
            dgamma[cc]
        );
    }

    // dL/dbeta_c
    for cc in 0..c {
        let mut bp = beta.clone();
        bp[cc] += fd;
        let lp = ref_loss(&x, &gamma, &bp, &dy, n, c, h, w);
        let mut bm = beta.clone();
        bm[cc] -= fd;
        let lm = ref_loss(&x, &gamma, &bm, &dy, n, c, h, w);
        let num = (lp - lm) / (2.0 * fd);
        assert!(
            rel(num, dbeta[cc]) < 3e-2,
            "dbeta[{cc}]: num={num} ana={}",
            dbeta[cc]
        );
    }
}
