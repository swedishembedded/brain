// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-pixel photometric objective a fit minimises, with its gradient,
//! on the device.
//!
//! [`PixelLoss::L1Ssim`] is the objective 3D Gaussian Splatting is defined
//! with (Kerbl et al., SIGGRAPH 2023, Eq. 7): `(1-λ)·L1 + λ·(1-SSIM)`, SSIM
//! over an 11x11 gaussian window of sigma 1.5 with `C1 = 0.01²`, `C2 = 0.03²`
//! (Wang et al., IEEE TIP 2004) and zero padding at the frame border.
//! [`PixelLoss::Mse`] is what `fit` minimised before it.
//!
//! Every term is a weighted MEAN over the supervised pixels and channels, so a
//! masked frame's number is comparable with an unmasked one's, and SSIM is
//! averaged by the same per-pixel weights the L1 term uses.
//!
//! It is four separable window passes (`l1ssim_*.wgsl`): moments along rows,
//! then along columns into the SSIM map and its partials, then the partials
//! carried back through the window, which is its own adjoint. It ran on the
//! host until a profile of a 300k-gaussian fit put it at a third of every
//! iteration (224 ms per 768x576 view).
//!
//! Swedish Embedded AB implements differentiable image objectives for 3D
//! reconstruction and neural rendering. If your team needs perceptually
//! faithful reconstruction losses, you can procure our services by sending an
//! email to info@swedishembedded.com.

use gpu_core::{f, DeviceBuffer, Gpu};

use crate::Kernels;

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

/// Device scratch for [`PixelLoss`] over frames of up to `max_px` pixels.
pub struct DeviceLoss {
    hsum: DeviceBuffer,
    part: DeviceBuffer,
    hpart: DeviceBuffer,
    loss: DeviceBuffer,
    /// A weight of 1 everywhere, bound when a frame has no mask.
    ones: DeviceBuffer,
    max_px: usize,
}

impl DeviceLoss {
    pub fn new(gpu: &Gpu, max_px: usize) -> DeviceLoss {
        let ones = gpu.storage(max_px as u64);
        gpu.write_f32(&ones, &vec![1.0; max_px]);
        DeviceLoss {
            hsum: gpu.storage(15 * max_px as u64),
            part: gpu.storage(9 * max_px as u64),
            hpart: gpu.storage(9 * max_px as u64),
            loss: gpu.storage(max_px as u64),
            ones,
            max_px,
        }
    }

    /// Evaluate `loss` of `pred` (RGBA `[w*h*4]`, alpha ignored) against
    /// `target` (RGB `[w*h*3]`), writing dLoss/dpred into `dimg` (RGBA
    /// `[w*h*4]`, alpha 0) and returning the loss. `weight` is the per-pixel
    /// weight `[w*h]` (None = all 1) and `wsum` its sum; a frame whose weights
    /// sum to zero returns 0 without touching `dimg`.
    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &self,
        gpu: &Gpu,
        ks: Kernels,
        loss: PixelLoss,
        pred: &DeviceBuffer,
        target: &DeviceBuffer,
        weight: Option<&DeviceBuffer>,
        wsum: f64,
        w: u32,
        h: u32,
        dimg: &DeviceBuffer,
    ) -> f64 {
        let px = (w * h) as usize;
        assert!(px <= self.max_px, "frame of {px} pixels past the loss scratch's {}", self.max_px);
        if wsum <= 0.0 {
            return 0.0;
        }
        let inv = (1.0 / (3.0 * wsum)) as f32;
        let wt = weight.unwrap_or(&self.ones);
        let (lam, mode) = match loss {
            PixelLoss::Mse => (0.0, 0u32),
            PixelLoss::L1Ssim { ssim } => (ssim.clamp(0.0, 1.0), 1u32),
        };
        let mut steps = Vec::new();
        if mode == 1 {
            steps.push(gpu.step(ks.l1ssim_moments_h, &[pred, target, &self.hsum], &[w, h], px as u32));
            steps.push(gpu.step(
                ks.l1ssim_map_v,
                &[&self.hsum, pred, target, wt, &self.part, &self.loss],
                &[w, h, f(lam), f(inv)],
                px as u32,
            ));
            steps.push(gpu.step(ks.l1ssim_partials_h, &[&self.part, &self.hpart], &[w, h], px as u32));
        }
        steps.push(gpu.step(
            ks.l1ssim_grad_v,
            &[&self.hpart, pred, target, wt, dimg, &self.loss],
            &[w, h, f(lam), f(inv), mode],
            px as u32,
        ));
        gpu.submit(&[], &steps);
        gpu.read(&self.loss, px).iter().map(|&v| v as f64).sum()
    }
}

/// Which pixels of a view to go on supervising when the photograph may hold
/// transient content (`crate::opt::FitCfg::transients`): 1 = supervise, 0 =
/// ignore, from each pixel's photometric residual `r` (`[W*H]`) against the
/// current scene (after RobustNeRF, Sabour et al. 2023).
///
/// A pixel is an outlier when its residual is above the view's 80th
/// percentile; a 3x3 majority vote removes isolated outliers - fine texture
/// the scene has not resolved yet scatters its error, a transient object
/// does not - and a 16x16 block that is at least 60% inliers is supervised
/// whole. What remains masked is a large coherent region the scene the
/// other views agree on cannot explain.
pub fn transient_mask(r: &[f32], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(r.len(), w * h);
    let mut sorted: Vec<f32> = r.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return vec![1.0; w * h];
    }
    let k = (sorted.len() * 8 / 10).min(sorted.len() - 1);
    let tau = *sorted.select_nth_unstable_by(k, f32::total_cmp).1;
    let inlier: Vec<bool> = r.iter().map(|v| *v <= tau).collect();
    let mut smooth = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut n, mut good) = (0, 0);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        n += 1;
                        good += inlier[yy as usize * w + xx as usize] as u32;
                    }
                }
            }
            smooth[y * w + x] = 2 * good >= n;
        }
    }
    let mut mask: Vec<f32> = smooth.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
    const BLOCK: usize = 16;
    for by in (0..h).step_by(BLOCK) {
        for bx in (0..w).step_by(BLOCK) {
            let (y1, x1) = ((by + BLOCK).min(h), (bx + BLOCK).min(w));
            let total = (y1 - by) * (x1 - bx);
            let good: usize = (by..y1).map(|y| (bx..x1).filter(|&x| smooth[y * w + x]).count()).sum();
            if 10 * good >= 6 * total {
                for y in by..y1 {
                    mask[y * w + bx..y * w + x1].fill(1.0);
                }
            }
        }
    }
    mask
}

