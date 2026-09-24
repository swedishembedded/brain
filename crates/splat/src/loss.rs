// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-pixel photometric objective a fit minimises, with its gradient.
//!
//! [`PixelLoss::L1Ssim`] is the objective 3D Gaussian Splatting is defined
//! with (Kerbl et al., SIGGRAPH 2023, Eq. 7): `(1-λ)·L1 + λ·(1-SSIM)`, SSIM
//! over an 11x11 gaussian window of sigma 1.5 with `C1 = 0.01²`, `C2 = 0.03²`
//! (Wang et al., IEEE TIP 2004) and zero padding at the frame border.
//! [`PixelLoss::Mse`] is what `fit` minimised before it, kept so a fit that
//! asks for nothing new is the fit it always was.
//!
//! Every term is a weighted MEAN over the supervised pixels and channels, so a
//! masked frame's number is comparable with an unmasked one's, and SSIM is
//! averaged by the same per-pixel weights the L1 term uses.
//!
//! This runs on the host because the fit's loss already does: the rendered
//! frame is read back once per view and the upstream gradient uploaded once.
//! The SSIM window is applied as two separable passes, rows in parallel.
//!
//! Swedish Embedded AB implements differentiable image objectives for 3D
//! reconstruction and neural rendering. If your team needs perceptually
//! faithful reconstruction losses, you can procure our services by sending an
//! email to info@swedishembedded.com.

/// Which photometric objective a fit minimises.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum PixelLoss {
    /// Mean squared error.
    #[default]
    Mse,
    /// `(1-ssim)·L1 + ssim·(1-SSIM)`. `ssim = 0.2` is the 3DGS objective.
    L1Ssim { ssim: f32 },
}

impl PixelLoss {
    /// The 3DGS objective.
    pub const fn gaussian_splatting() -> PixelLoss {
        PixelLoss::L1Ssim { ssim: 0.2 }
    }
}

const RADIUS: usize = 5;
const C1: f32 = 0.01 * 0.01;
const C2: f32 = 0.03 * 0.03;

fn window() -> [f32; 2 * RADIUS + 1] {
    let mut g = [0.0f32; 2 * RADIUS + 1];
    for (i, v) in g.iter_mut().enumerate() {
        let d = i as f32 - RADIUS as f32;
        *v = (-(d * d) / (2.0 * 1.5 * 1.5)).exp();
    }
    let s: f32 = g.iter().sum();
    g.map(|v| v / s)
}

/// Zero-padded "same" convolution of a planar `w x h` image with the
/// separable symmetric window `g` - its own adjoint, which is what the SSIM
/// backward relies on.
fn blur(src: &[f32], w: usize, h: usize, g: &[f32; 2 * RADIUS + 1]) -> Vec<f32> {
    let mut rows = vec![0.0f32; w * h];
    backend_cpu::par::rows_mut(&mut rows, w, |y, out| {
        let line = &src[y * w..y * w + w];
        for (x, o) in out.iter_mut().enumerate() {
            let lo = x.saturating_sub(RADIUS);
            let hi = (x + RADIUS).min(w - 1);
            *o = (lo..=hi).map(|k| g[k + RADIUS - x] * line[k]).sum();
        }
    });
    let mut out = vec![0.0f32; w * h];
    backend_cpu::par::rows_mut(&mut out, w, |y, o| {
        let lo = y.saturating_sub(RADIUS);
        let hi = (y + RADIUS).min(h - 1);
        for k in lo..=hi {
            let wk = g[k + RADIUS - y];
            let r = &rows[k * w..k * w + w];
            for x in 0..w {
                o[x] += wk * r[x];
            }
        }
    });
    out
}

/// Evaluate `loss` of `pred` against `target` (both interleaved RGB
/// `[w*h*3]`), writing dLoss/dpred into `grad` (overwritten, same layout) and
/// returning the loss. `weight` is an optional per-pixel weight `[w*h]`; a
/// frame whose weights sum to zero returns 0 with a zero gradient.
pub fn photometric(
    loss: PixelLoss,
    pred: &[f32],
    target: &[f32],
    weight: Option<&[f32]>,
    w: usize,
    h: usize,
    grad: &mut [f32],
) -> f64 {
    let px = w * h;
    assert_eq!(pred.len(), px * 3);
    assert_eq!(target.len(), px * 3);
    assert_eq!(grad.len(), px * 3);
    if let Some(m) = weight {
        assert_eq!(m.len(), px);
    }
    let wt = |p: usize| weight.map_or(1.0, |m| m[p]);
    let wsum: f64 = weight.map_or(px as f64, |m| m.iter().map(|&v| v as f64).sum());
    grad.fill(0.0);
    if wsum <= 0.0 {
        return 0.0;
    }
    let norm = wsum * 3.0;
    let inv = (1.0 / norm) as f32;
    let lam = match loss {
        PixelLoss::Mse => {
            // `2/norm` rounded once, as `fit` always did, so an MSE fit stays
            // bit-identical to the one before this module existed.
            let scale = (2.0 / norm) as f32;
            let mut sum = 0.0f64;
            for p in 0..px {
                let m = wt(p);
                if m == 0.0 {
                    continue;
                }
                for c in 0..3 {
                    let d = pred[p * 3 + c] - target[p * 3 + c];
                    sum += (m * d * d) as f64;
                    grad[p * 3 + c] = scale * m * d;
                }
            }
            return sum / norm;
        }
        PixelLoss::L1Ssim { ssim } => ssim.clamp(0.0, 1.0),
    };
    let mut l1 = 0.0f64;
    for p in 0..px {
        let m = wt(p);
        if m == 0.0 {
            continue;
        }
        for c in 0..3 {
            let d = pred[p * 3 + c] - target[p * 3 + c];
            l1 += (m * d.abs()) as f64;
            // `f32::signum(0.0)` is 1; an exact match must pull nowhere.
            let sign = if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 };
            grad[p * 3 + c] = (1.0 - lam) * m * inv * sign;
        }
    }
    if lam == 0.0 {
        return l1 / norm;
    }
    let g = window();
    let mut ssim_sum = 0.0f64;
    for c in 0..3 {
        let x: Vec<f32> = (0..px).map(|p| pred[p * 3 + c]).collect();
        let y: Vec<f32> = (0..px).map(|p| target[p * 3 + c]).collect();
        let prod = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(u, v)| u * v).collect::<Vec<f32>>();
        let mx = blur(&x, w, h, &g);
        let my = blur(&y, w, h, &g);
        let mxx = blur(&prod(&x, &x), w, h, &g);
        let myy = blur(&prod(&y, &y), w, h, &g);
        let mxy = blur(&prod(&x, &y), w, h, &g);
        // Per-pixel partials of S = A1·A2 / (B1·B2) with respect to the five
        // window moments, each already multiplied by that pixel's weight and
        // by -λ/norm (the loss is 1 - mean SSIM). Only three are needed: the
        // moments of the TARGET alone carry no gradient.
        let mut g_mx = vec![0.0f32; px];
        let mut g_mxx = vec![0.0f32; px];
        let mut g_mxy = vec![0.0f32; px];
        let k = -lam * inv;
        for p in 0..px {
            let (ux, uy) = (mx[p], my[p]);
            let a1 = 2.0 * ux * uy + C1;
            let a2 = 2.0 * (mxy[p] - ux * uy) + C2;
            let b1 = ux * ux + uy * uy + C1;
            let b2 = (mxx[p] - ux * ux) + (myy[p] - uy * uy) + C2;
            let s = a1 * a2 / (b1 * b2);
            let m = wt(p);
            ssim_sum += (m * s) as f64;
            if m == 0.0 {
                continue;
            }
            let d_ux = (2.0 * uy * a2 - 2.0 * uy * a1) / (b1 * b2) - s * (2.0 * ux / b1 - 2.0 * ux / b2);
            g_mx[p] = k * m * d_ux;
            g_mxx[p] = k * m * (-s / b2);
            g_mxy[p] = k * m * (2.0 * a1 / (b1 * b2));
        }
        // The window is symmetric and zero padded, so its adjoint is itself.
        let (b_mx, b_mxx, b_mxy) = (blur(&g_mx, w, h, &g), blur(&g_mxx, w, h, &g), blur(&g_mxy, w, h, &g));
        for p in 0..px {
            grad[p * 3 + c] += b_mx[p] + 2.0 * x[p] * b_mxx[p] + y[p] * b_mxy[p];
        }
    }
    (1.0 - lam as f64) * l1 / norm + lam as f64 * (1.0 - ssim_sum / norm)
}
