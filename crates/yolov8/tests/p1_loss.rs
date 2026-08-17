// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference micro-checks for the YOLOv8 loss + box-decode WGSL kernels
//! on the CPU backend (no GPU). Each gradient kernel is verified against central
//! differences of its matching value kernel / scalar loss.
//!
//! The riskiest gate is `ciou_grad`: its FD reference replicates the EXACT value
//! the `ciou.wgsl` kernel computes, including the atan polyfill and the detached
//! alpha (standard YOLO convention), so analytic == numerical.

use gpu_core::Gpu;

const DFL_DECODE: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/dfl_decode.wgsl"));
const DFL_GRAD: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/dfl_grad.wgsl"));
const DFL_LOSS: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/dfl_loss.wgsl"));
const DFL_LOSS_GRAD: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/dfl_loss_grad.wgsl"));
const CIOU: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/ciou.wgsl"));
const CIOU_GRAD: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/ciou_grad.wgsl"));
const BCE: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/bce_logits.wgsl"));
const BCE_GRAD: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/wgsl/bce_logits_grad.wgsl"));

const K_DFL_DECODE: usize = 0;
const K_DFL_GRAD: usize = 1;
const K_DFL_LOSS: usize = 2;
const K_DFL_LOSS_GRAD: usize = 3;
const K_CIOU: usize = 4;
const K_CIOU_GRAD: usize = 5;
const K_BCE: usize = 6;
const K_BCE_GRAD: usize = 7;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("dfl_decode", DFL_DECODE),
        ("dfl_grad", DFL_GRAD),
        ("dfl_loss", DFL_LOSS),
        ("dfl_loss_grad", DFL_LOSS_GRAD),
        ("ciou", CIOU),
        ("ciou_grad", CIOU_GRAD),
        ("bce_logits", BCE),
        ("bce_logits_grad", BCE_GRAD),
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
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }
    fn unit(&mut self) -> f32 {
        (self.next_f32() + 1.0) * 0.5 // [0,1]
    }
}

fn rel(num: f32, ana: f32) -> f32 {
    let denom = num.abs().max(ana.abs()).max(1e-6);
    (num - ana).abs() / denom
}

// Central differences in fp32 carry ~1e-3 absolute noise; for small-magnitude
// gradient entries that noise can exceed the relative band even though the
// analytic value is correct. A gradient entry is "good" if it passes EITHER the
// relative band (rel_tol) OR a small absolute band (abs_tol). This is the
// standard combined criterion for finite-difference gradient checks.
fn grad_ok(num: f32, ana: f32, rel_tol: f32, abs_tol: f32) -> bool {
    (num - ana).abs() <= abs_tol || rel(num, ana) <= rel_tol
}

// ---------------------------------------------------------------------------
// DFL decode: two-hot encode -> decode round trip + golden distance.
// ---------------------------------------------------------------------------

fn decode_gpu(gpu: &Gpu, logits: &[f32], a: u32, reg_max: u32) -> Vec<f32> {
    let lb = gpu.storage_init("logits", logits);
    let db = gpu.storage((a * 4) as u64);
    let s = gpu.step(K_DFL_DECODE, &[&lb, &db], &[a, reg_max], a * 4);
    gpu.submit(&[], &[s]);
    gpu.read(&db, (a * 4) as usize)
}

#[test]
fn dfl_decode_two_hot_roundtrip() {
    // target 3.25 -> bins 3:0.75, 4:0.25 -> decode 3.25.
    // Encode by setting logits = log(weights) (softmax recovers the weights).
    let reg_max = 8u32;
    let a = 1u32;
    let mut logits = vec![-30.0f32; (a * 4 * reg_max) as usize];
    // sides all encode 3.25 for simplicity.
    for side in 0..4u32 {
        let base = (side * reg_max) as usize;
        logits[base + 3] = 0.75f32.ln();
        logits[base + 4] = 0.25f32.ln();
    }
    let gpu = Gpu::new_cpu(&kernels());
    let dist = decode_gpu(&gpu, &logits, a, reg_max);
    for (side, &d) in dist.iter().enumerate() {
        assert!((d - 3.25).abs() < 1e-3, "side {side}: {d}");
    }
}

#[test]
fn dfl_decode_peaked() {
    // A logit distribution sharply peaked at bin 5 decodes to ~5.
    let reg_max = 8u32;
    let a = 1u32;
    let mut logits = vec![0.0f32; (a * 4 * reg_max) as usize];
    for side in 0..4u32 {
        let base = (side * reg_max) as usize;
        logits[base + 5] = 20.0; // dominates softmax
    }
    let gpu = Gpu::new_cpu(&kernels());
    let dist = decode_gpu(&gpu, &logits, a, reg_max);
    for (side, &d) in dist.iter().enumerate() {
        assert!((d - 5.0).abs() < 1e-3, "side {side}: {d}");
    }
}

// L = <dE, E(logits)>. dfl_grad must equal dL/dlogits.
#[test]
fn dfl_grad_finite_difference() {
    let a = 3u32;
    let reg_max = 8u32;
    let n = (a * 4 * reg_max) as usize;
    let mut rng = Lcg::new(0x1111_2222);
    let logits: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let de: Vec<f32> = (0..(a * 4) as usize).map(|_| rng.next_f32()).collect();

    let gpu = Gpu::new_cpu(&kernels());

    // analytic
    let lb = gpu.storage_init("logits", &logits);
    let deb = gpu.storage_init("de", &de);
    let dlb = gpu.storage(n as u64);
    let s = gpu.step(K_DFL_GRAD, &[&lb, &deb, &dlb], &[a, reg_max], a * 4);
    gpu.submit(&[], &[s]);
    let dl = gpu.read(&dlb, n);

    // numeric: L = sum_(anchor,side) de * E
    let loss = |lg: &[f32]| -> f32 {
        let e = decode_gpu(&gpu, lg, a, reg_max);
        e.iter().zip(&de).map(|(a, b)| a * b).sum()
    };
    let fd = 1e-3f32;
    for i in 0..n {
        let mut lp = logits.clone();
        lp[i] += fd;
        let mut lm = logits.clone();
        lm[i] -= fd;
        let num = (loss(&lp) - loss(&lm)) / (2.0 * fd);
        assert!(
            grad_ok(num, dl[i], 2e-2, 2e-3),
            "dfl_grad[{i}]: num={num} ana={} rel={}",
            dl[i],
            rel(num, dl[i])
        );
    }
}

// ---------------------------------------------------------------------------
// DFL loss value + gradient.
// ---------------------------------------------------------------------------

fn dfl_loss_gpu(gpu: &Gpu, logits: &[f32], tdist: &[f32], a: u32, reg_max: u32) -> Vec<f32> {
    let lb = gpu.storage_init("logits", logits);
    let tb = gpu.storage_init("tdist", tdist);
    let ob = gpu.storage(a as u64);
    let s = gpu.step(K_DFL_LOSS, &[&lb, &tb, &ob], &[a, reg_max], a);
    gpu.submit(&[], &[s]);
    gpu.read(&ob, a as usize)
}

#[test]
fn dfl_loss_grad_finite_difference() {
    let a = 3u32;
    let reg_max = 8u32;
    let n = (a * 4 * reg_max) as usize;
    let mut rng = Lcg::new(0x3333_4444);
    let logits: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    // targets in (0, reg_max-1) so two-hot stays interior.
    let tdist: Vec<f32> = (0..(a * 4) as usize)
        .map(|_| 0.5 + rng.unit() * (reg_max as f32 - 2.0))
        .collect();

    let gpu = Gpu::new_cpu(&kernels());
    let lb = gpu.storage_init("logits", &logits);
    let tb = gpu.storage_init("tdist", &tdist);
    let dlb = gpu.storage(n as u64);
    let s = gpu.step(K_DFL_LOSS_GRAD, &[&lb, &tb, &dlb], &[a, reg_max], a * 4);
    gpu.submit(&[], &[s]);
    let dl = gpu.read(&dlb, n);

    // numeric: total loss = sum_a out[a].
    let total = |lg: &[f32]| -> f32 { dfl_loss_gpu(&gpu, lg, &tdist, a, reg_max).iter().sum() };
    let fd = 1e-3f32;
    for i in 0..n {
        let mut lp = logits.clone();
        lp[i] += fd;
        let mut lm = logits.clone();
        lm[i] -= fd;
        let num = (total(&lp) - total(&lm)) / (2.0 * fd);
        assert!(
            grad_ok(num, dl[i], 2e-2, 2e-3),
            "dfl_loss_grad[{i}]: num={num} ana={} rel={}",
            dl[i],
            rel(num, dl[i])
        );
    }
}

// ---------------------------------------------------------------------------
// BCE-with-logits value + gradient.
// ---------------------------------------------------------------------------

fn bce_gpu(gpu: &Gpu, logits: &[f32], tgt: &[f32]) -> Vec<f32> {
    let total = logits.len() as u32;
    let lb = gpu.storage_init("logits", logits);
    let tb = gpu.storage_init("tgt", tgt);
    let ob = gpu.storage(total as u64);
    let s = gpu.step(K_BCE, &[&lb, &tb, &ob], &[total], total);
    gpu.submit(&[], &[s]);
    gpu.read(&ob, total as usize)
}

#[test]
fn bce_logits_value_golden() {
    // t=1, z=0 -> loss = log(2). t=0, z=0 -> log(2).
    let gpu = Gpu::new_cpu(&kernels());
    let out = bce_gpu(&gpu, &[0.0, 0.0], &[1.0, 0.0]);
    assert!((out[0] - 2.0f32.ln()).abs() < 1e-5);
    assert!((out[1] - 2.0f32.ln()).abs() < 1e-5);
}

#[test]
fn bce_logits_grad_finite_difference() {
    let total = 24usize;
    let mut rng = Lcg::new(0x5555_6666);
    let logits: Vec<f32> = (0..total).map(|_| rng.next_f32() * 3.0).collect();
    let tgt: Vec<f32> = (0..total).map(|_| rng.unit()).collect();

    let gpu = Gpu::new_cpu(&kernels());
    let lb = gpu.storage_init("logits", &logits);
    let tb = gpu.storage_init("tgt", &tgt);
    let dlb = gpu.storage(total as u64);
    let s = gpu.step(K_BCE_GRAD, &[&lb, &tb, &dlb], &[total as u32], total as u32);
    gpu.submit(&[], &[s]);
    let dl = gpu.read(&dlb, total);

    let sumloss = |lg: &[f32]| -> f32 { bce_gpu(&gpu, lg, &tgt).iter().sum() };
    let fd = 1e-3f32;
    for i in 0..total {
        let mut lp = logits.clone();
        lp[i] += fd;
        let mut lm = logits.clone();
        lm[i] -= fd;
        let num = (sumloss(&lp) - sumloss(&lm)) / (2.0 * fd);
        assert!(
            grad_ok(num, dl[i], 2e-2, 2e-3),
            "bce_grad[{i}]: num={num} ana={} rel={}",
            dl[i],
            rel(num, dl[i])
        );
    }
}

// ---------------------------------------------------------------------------
// CIoU value + gradient. The FD reference replicates ciou.wgsl exactly.
// ---------------------------------------------------------------------------

// Same atan polyfill as the kernel (for x >= 0).
fn atan_pos(xin: f32) -> f32 {
    let half_pi = 1.570_796_4_f32;
    let mut x = xin;
    let mut flip = false;
    if x > 1.0 {
        x = 1.0 / x;
        flip = true;
    }
    let z = x * x;
    let mut a = 0.0028662257f32;
    a = a * z - 0.016_165_737;
    a = a * z + 0.042_909_615;
    a = a * z - 0.075_289_64;
    a = a * z + 0.106_562_64;
    a = a * z - 0.142_089;
    a = a * z + 0.199_935_51;
    a = a * z - 0.333_331_47;
    a = a * z + 1.0;
    let mut r = a * x;
    if flip {
        r = half_pi - r;
    }
    r
}

// CIoU loss value for one box, matching ciou.wgsl (alpha live in the value;
// detachment only matters for the gradient).
fn ciou_loss_one(p: &[f32], g: &[f32]) -> f32 {
    let (px1, py1, px2, py2) = (p[0], p[1], p[2], p[3]);
    let (gx1, gy1, gx2, gy2) = (g[0], g[1], g[2], g[3]);
    let wp = px2 - px1;
    let hp = py2 - py1;
    let wg = gx2 - gx1;
    let hg = gy2 - gy1;
    let iw = (px2.min(gx2) - px1.max(gx1)).max(0.0);
    let ih = (py2.min(gy2) - py1.max(gy1)).max(0.0);
    let inter = iw * ih;
    let uni = (wp * hp + wg * hg - inter).max(1e-9);
    let iou = inter / uni;
    let cpx = (px1 + px2) * 0.5;
    let cpy = (py1 + py2) * 0.5;
    let cgx = (gx1 + gx2) * 0.5;
    let cgy = (gy1 + gy2) * 0.5;
    let rho2 = (cpx - cgx).powi(2) + (cpy - cgy).powi(2);
    let cw = px2.max(gx2) - px1.min(gx1);
    let ch = py2.max(gy2) - py1.min(gy1);
    let c2 = (cw * cw + ch * ch).max(1e-9);
    let atg = atan_pos(wg / hg.max(1e-9));
    let atp = atan_pos(wp / hp.max(1e-9));
    let diff = atg - atp;
    let k = 0.405_284_73_f32;
    let v = k * diff * diff;
    let alpha = v / ((1.0 - iou) + v).max(1e-9);
    let ciou = iou - rho2 / c2 - alpha * v;
    1.0 - ciou
}

fn ciou_value_gpu(gpu: &Gpu, pred: &[f32], tgt: &[f32], a: u32) -> Vec<f32> {
    let pb = gpu.storage_init("pred", pred);
    let tb = gpu.storage_init("tgt", tgt);
    let ob = gpu.storage(a as u64);
    let s = gpu.step(K_CIOU, &[&pb, &tb, &ob], &[a], a);
    gpu.submit(&[], &[s]);
    gpu.read(&ob, a as usize)
}

#[test]
fn ciou_identical_box_zero_loss() {
    let pred = vec![10.0, 20.0, 50.0, 80.0];
    let tgt = pred.clone();
    let gpu = Gpu::new_cpu(&kernels());
    let out = ciou_value_gpu(&gpu, &pred, &tgt, 1);
    assert!(out[0].abs() < 1e-4, "identical box loss = {}", out[0]);
}

#[test]
fn ciou_partial_overlap_golden() {
    // Two equal-size 2x2 boxes offset by (1,1): pred [0,0,2,2], tgt [1,1,3,3].
    // inter = 1, union = 4+4-1 = 7, IoU = 1/7.
    // centers (1,1) vs (2,2): rho2 = 2. enclosing [0,0,3,3]: c2 = 18.
    // aspect equal (square vs square) -> v = 0 -> CIoU = IoU - rho2/c2.
    let pred = vec![0.0, 0.0, 2.0, 2.0];
    let tgt = vec![1.0, 1.0, 3.0, 3.0];
    let expect = 1.0 - (1.0 / 7.0 - 2.0 / 18.0);
    let gpu = Gpu::new_cpu(&kernels());
    let out = ciou_value_gpu(&gpu, &pred, &tgt, 1);
    assert!((out[0] - expect).abs() < 1e-4, "ciou = {} expect {expect}", out[0]);
    // cross-check against the Rust reference too.
    assert!((out[0] - ciou_loss_one(&pred, &tgt)).abs() < 1e-4);
}

#[test]
fn ciou_grad_finite_difference() {
    // Random well-separated, positive-area boxes with PARTIAL overlap so the
    // gradient is smooth (avoid IoU=1 and disjoint-box degeneracies).
    let a = 16usize;
    let mut rng = Lcg::new(0x7777_8888);
    let mut pred = vec![0.0f32; a * 4];
    let mut tgt = vec![0.0f32; a * 4];
    for k in 0..a {
        // target box: random center, size in [4,8].
        let cx = rng.unit() * 20.0 + 10.0;
        let cy = rng.unit() * 20.0 + 10.0;
        let gw = 4.0 + rng.unit() * 4.0;
        let gh = 4.0 + rng.unit() * 4.0;
        tgt[k * 4] = cx - gw * 0.5;
        tgt[k * 4 + 1] = cy - gh * 0.5;
        tgt[k * 4 + 2] = cx + gw * 0.5;
        tgt[k * 4 + 3] = cy + gh * 0.5;
        // pred box: shifted by a fraction of the size (guarantees overlap),
        // with a different aspect ratio (exercises the v term).
        let sx = (rng.next_f32()) * gw * 0.4;
        let sy = (rng.next_f32()) * gh * 0.4;
        let pw = gw * (0.7 + rng.unit() * 0.6);
        let ph = gh * (0.7 + rng.unit() * 0.6);
        let pcx = cx + sx;
        let pcy = cy + sy;
        pred[k * 4] = pcx - pw * 0.5;
        pred[k * 4 + 1] = pcy - ph * 0.5;
        pred[k * 4 + 2] = pcx + pw * 0.5;
        pred[k * 4 + 3] = pcy + ph * 0.5;
    }

    let gpu = Gpu::new_cpu(&kernels());
    let pb = gpu.storage_init("pred", &pred);
    let tb = gpu.storage_init("tgt", &tgt);
    let dpb = gpu.storage((a * 4) as u64);
    let s = gpu.step(K_CIOU_GRAD, &[&pb, &tb, &dpb], &[a as u32], a as u32);
    gpu.submit(&[], &[s]);
    let dp = gpu.read(&dpb, a * 4);

    // numeric per-coordinate central difference, using the alpha-DETACHED loss
    // (matches the kernel's stop-gradient on alpha).
    let loss_detached = |p: &[f32], g: &[f32], alpha0: f32| -> f32 {
        let (px1, py1, px2, py2) = (p[0], p[1], p[2], p[3]);
        let (gx1, gy1, gx2, gy2) = (g[0], g[1], g[2], g[3]);
        let wp = px2 - px1;
        let hp = py2 - py1;
        let wg = gx2 - gx1;
        let hg = gy2 - gy1;
        let iw = (px2.min(gx2) - px1.max(gx1)).max(0.0);
        let ih = (py2.min(gy2) - py1.max(gy1)).max(0.0);
        let inter = iw * ih;
        let uni = (wp * hp + wg * hg - inter).max(1e-9);
        let iou = inter / uni;
        let cpx = (px1 + px2) * 0.5;
        let cpy = (py1 + py2) * 0.5;
        let cgx = (gx1 + gx2) * 0.5;
        let cgy = (gy1 + gy2) * 0.5;
        let rho2 = (cpx - cgx).powi(2) + (cpy - cgy).powi(2);
        let cw = px2.max(gx2) - px1.min(gx1);
        let ch = py2.max(gy2) - py1.min(gy1);
        let c2 = (cw * cw + ch * ch).max(1e-9);
        let atg = atan_pos(wg / hg.max(1e-9));
        let atp = atan_pos(wp / hp.max(1e-9));
        let diff = atg - atp;
        let k = 0.405_284_73_f32;
        let v = k * diff * diff;
        let ciou = iou - rho2 / c2 - alpha0 * v;
        1.0 - ciou
    };
    // alpha evaluated at the unperturbed point (detached constant).
    let alpha_of = |p: &[f32], g: &[f32]| -> f32 {
        let (px1, py1, px2, py2) = (p[0], p[1], p[2], p[3]);
        let (gx1, gy1, gx2, gy2) = (g[0], g[1], g[2], g[3]);
        let wp = px2 - px1;
        let hp = py2 - py1;
        let wg = gx2 - gx1;
        let hg = gy2 - gy1;
        let iw = (px2.min(gx2) - px1.max(gx1)).max(0.0);
        let ih = (py2.min(gy2) - py1.max(gy1)).max(0.0);
        let inter = iw * ih;
        let uni = (wp * hp + wg * hg - inter).max(1e-9);
        let iou = inter / uni;
        let atg = atan_pos(wg / hg.max(1e-9));
        let atp = atan_pos(wp / hp.max(1e-9));
        let diff = atg - atp;
        let k = 0.405_284_73_f32;
        let v = k * diff * diff;
        v / ((1.0 - iou) + v).max(1e-9)
    };

    let fd = 1e-3f32;
    let mut worst = 0.0f32;
    for k in 0..a {
        let p0 = &pred[k * 4..k * 4 + 4];
        let g0 = &tgt[k * 4..k * 4 + 4];
        let alpha0 = alpha_of(p0, g0);
        for c in 0..4 {
            let mut pp = p0.to_vec();
            pp[c] += fd;
            let mut pm = p0.to_vec();
            pm[c] -= fd;
            let num = (loss_detached(&pp, g0, alpha0) - loss_detached(&pm, g0, alpha0)) / (2.0 * fd);
            let ana = dp[k * 4 + c];
            let r = rel(num, ana);
            worst = worst.max(r);
            assert!(r < 2e-2, "ciou_grad box {k} coord {c}: num={num} ana={ana} rel={r}");
        }
    }
    eprintln!("ciou_grad worst rel error = {worst}");
}
