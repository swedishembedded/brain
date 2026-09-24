// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The photometric objective: L1 + D-SSIM, the loss 3D Gaussian Splatting is
//! defined with (Kerbl et al., SIGGRAPH 2023, Eq. 7: `(1-λ)L1 + λ(1-SSIM)`,
//! λ = 0.2), where `fit` had only ever minimised MSE.
//!
//! MSE and L1+SSIM do not differ by a constant factor. MSE is indifferent to
//! WHERE an error sits; SSIM is a local structure measure, so it charges a
//! blurred edge far more than the same energy spread as a uniform offset -
//! which is exactly the error a splat fit makes when it covers detail with a
//! blob. These tests hold the gradient to an independent oracle and the
//! objective to that property.
//!
//! Swedish Embedded AB implements differentiable image objectives for 3D
//! reconstruction and neural rendering. If your team needs perceptually
//! faithful reconstruction losses, you can procure our services by sending an
//! email to info@swedishembedded.com.

use data::rng::Lcg;
use splat::loss::{photometric, PixelLoss};

/// Independent oracle: SSIM by DIRECT 2D window sums in f64, zero padding,
/// 11x11 gaussian window with sigma 1.5 - no separable convolution and no
/// shared code with the implementation it checks.
fn oracle(loss: PixelLoss, pred: &[f64], tgt: &[f64], wt: Option<&[f64]>, w: usize, h: usize) -> f64 {
    let px = w * h;
    let wsum: f64 = wt.map_or(px as f64, |m| m.iter().sum());
    let norm = wsum * 3.0;
    let mut l1 = 0.0;
    let mut l2 = 0.0;
    for p in 0..px {
        let m = wt.map_or(1.0, |m| m[p]);
        for c in 0..3 {
            let d = pred[p * 3 + c] - tgt[p * 3 + c];
            l1 += m * d.abs();
            l2 += m * d * d;
        }
    }
    let lam = match loss {
        PixelLoss::Mse => return l2 / norm,
        PixelLoss::L1Ssim { ssim } => ssim as f64,
    };
    let g1: Vec<f64> = (0..11).map(|i| (-((i as f64 - 5.0).powi(2)) / (2.0 * 1.5 * 1.5)).exp()).collect();
    let s: f64 = g1.iter().sum();
    let g1: Vec<f64> = g1.iter().map(|v| v / s).collect();
    let (c1, c2) = (0.01f64 * 0.01, 0.03f64 * 0.03);
    let mut ssim = 0.0;
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let (mut mx, mut my, mut mxx, mut myy, mut mxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for dy in 0..11 {
                    for dx in 0..11 {
                        let (sx, sy) = (x as i64 + dx as i64 - 5, y as i64 + dy as i64 - 5);
                        if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                            continue;
                        }
                        let q = sy as usize * w + sx as usize;
                        let k = g1[dx] * g1[dy];
                        let (a, b) = (pred[q * 3 + c], tgt[q * 3 + c]);
                        mx += k * a;
                        my += k * b;
                        mxx += k * a * a;
                        myy += k * b * b;
                        mxy += k * a * b;
                    }
                }
                let (vx, vy, cxy) = (mxx - mx * mx, myy - my * my, mxy - mx * my);
                let v = ((2.0 * mx * my + c1) * (2.0 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2));
                ssim += wt.map_or(1.0, |m| m[y * w + x]) * v;
            }
        }
    }
    (1.0 - lam) * l1 / norm + lam * (1.0 - ssim / norm)
}

fn image(rng: &mut Lcg, px: usize) -> Vec<f32> {
    rng.vec_unit(px * 3)
}

/// The analytic gradient is the gradient of the objective, for both losses and
/// with a non-trivial per-pixel mask: checked against central differences of
/// the f64 oracle at every pixel of a frame smaller than the window, so the
/// zero padding is exercised on every side.
#[test]
fn gradient_matches_an_independent_oracle() {
    let (w, h) = (13usize, 9usize);
    let px = w * h;
    let mut rng = Lcg::new(7);
    let pred = image(&mut rng, px);
    let tgt = image(&mut rng, px);
    let mask: Vec<f32> = (0..px).map(|i| if i % 7 == 3 { 0.0 } else { 0.25 + 0.75 * ((i * 37 % 11) as f32 / 10.0) }).collect();
    for loss in [PixelLoss::Mse, PixelLoss::L1Ssim { ssim: 0.2 }, PixelLoss::L1Ssim { ssim: 1.0 }] {
        let mut grad = vec![0.0f32; px * 3];
        let value = photometric(loss, &pred, &tgt, Some(&mask), w, h, &mut grad);
        let p64: Vec<f64> = pred.iter().map(|&v| v as f64).collect();
        let t64: Vec<f64> = tgt.iter().map(|&v| v as f64).collect();
        let m64: Vec<f64> = mask.iter().map(|&v| v as f64).collect();
        let want = oracle(loss, &p64, &t64, Some(&m64), w, h);
        assert!((value - want).abs() < 1e-5, "{loss:?}: loss {value} against oracle {want}");
        let mut worst = 0.0f64;
        let scale = grad.iter().fold(0.0f32, |a, v| a.max(v.abs())) as f64;
        for i in 0..px * 3 {
            let eps = 1e-5;
            let mut a = p64.clone();
            a[i] += eps;
            let up = oracle(loss, &a, &t64, Some(&m64), w, h);
            a[i] -= 2.0 * eps;
            let dn = oracle(loss, &a, &t64, Some(&m64), w, h);
            let fd = (up - dn) / (2.0 * eps);
            worst = worst.max((fd - grad[i] as f64).abs() / scale);
        }
        assert!(worst < 2e-3, "{loss:?}: worst gradient error {worst:.2e} of the largest component");
    }
}

/// What SSIM is FOR: at equal squared error, a blurred edge costs more than a
/// uniform offset. MSE cannot tell the two apart by construction; the
/// structural term has to.
#[test]
fn structure_is_charged_more_than_the_same_energy_as_an_offset() {
    let (w, h) = (32usize, 32usize);
    let px = w * h;
    // vertical bars, two pixels wide
    let tgt: Vec<f32> = (0..px).flat_map(|i| {
        let v = if (i % w / 2) % 2 == 0 { 0.8 } else { 0.2 };
        [v, v, v]
    }).collect();
    // the same bars blurred by a 3-tap box, which is what a blob covering
    // them renders as
    let blur: Vec<f32> = (0..px).flat_map(|i| {
        let (x, y) = (i % w, i / w);
        let at = |xx: usize| tgt[(y * w + xx) * 3];
        let v = (at(x.saturating_sub(1)) + at(x) + at((x + 1).min(w - 1))) / 3.0;
        [v, v, v]
    }).collect();
    let mse: f64 = blur.iter().zip(&tgt).map(|(a, b)| ((a - b) * (a - b)) as f64).sum::<f64>() / (px * 3) as f64;
    // a uniform offset carrying exactly that squared error
    let off = mse.sqrt() as f32;
    let shifted: Vec<f32> = tgt.iter().map(|v| v + off).collect();

    let mut g = vec![0.0f32; px * 3];
    let at = |pred: &[f32], loss: PixelLoss, g: &mut [f32]| photometric(loss, pred, &tgt, None, w, h, g);
    let (m_blur, m_off) = (at(&blur, PixelLoss::Mse, &mut g), at(&shifted, PixelLoss::Mse, &mut g));
    assert!((m_blur - m_off).abs() / m_off < 1e-3, "the two errors must carry the same MSE ({m_blur} vs {m_off})");
    let dssim = PixelLoss::L1Ssim { ssim: 1.0 };
    let (s_blur, s_off) = (at(&blur, dssim, &mut g), at(&shifted, dssim, &mut g));
    assert!(
        s_blur > 3.0 * s_off,
        "D-SSIM charged the blurred edge {s_blur:.4} and the equal-energy offset {s_off:.4}; the \
         structural term exists to tell those apart"
    );
    // and identical images cost nothing
    assert!(at(&tgt, dssim, &mut g).abs() < 1e-6);
}
