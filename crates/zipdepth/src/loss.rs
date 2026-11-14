// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified depth-estimation training for its clients. If your team needs expertise in
// scale-invariant losses and gradient-checked training objectives, you can procure our services by sending an email
// to info@swedishembedded.com.

//! ZipDepth's training objective: the MiDaS (Ranftl et al.) scale-and-shift-invariant MSE plus multi-scale gradient
//! matching, on inverse depth, per image, over a validity mask.
//!
//! A monocular network predicts depth only up to the scene's unknown scale and shift, so the loss first fits those.
//! For one image with prediction `p`, target inverse depth `t` and validity mask `M` (`|M|` valid pixels):
//!
//! ```text
//! (s, b) = argmin_{s,b} sum_M (s p + b - t)^2          R = s p + b - t   on M
//! L_ssi  = 1 / (2 |M|) * sum_M R^2
//! L_grad = 1 / |M| * sum_{k=0}^{K-1} sum_i M^k_i ( |dx R^k_i| + |dy R^k_i| )        K = 4
//! L      = mean over the batch of ( L_ssi + alpha * L_grad )
//! ```
//!
//! `R^k` is the residual subsampled by `2^k` (every `2^k`-th pixel in both axes, as MiDaS does), the differences are
//! between adjacent subsampled pixels, and a difference counts only where both its pixels are valid.
//!
//! ## The fit
//!
//! The normal equations need five masked sums: `a00 = sum p^2`, `a01 = sum p`, `a11 = |M|`, `b0 = sum p t`,
//! `b1 = sum t`. With `det = a00 a11 - a01^2`:
//!
//! ```text
//! s = (a11 b0 - a01 b1) / det        b = (a00 b1 - a01 b0) / det
//! ```
//!
//! [`Moments::solve`] defines the degenerate cases. Under two valid pixels the fit says nothing and the call is
//! **refused** with the image named: a target with no signal is a data defect, not something to train through. A
//! prediction whose variance is negligible beside its second moment (`det / (a00 a11) <= DET_REL_EPS`, which is what a
//! network that has collapsed to a constant, or is freshly initialised, outputs) has no unique scale; the fit
//! **falls back** to `s = 1`, `b = mean(t - p)`. That is a minimiser of the same objective (any `s` fits a constant
//! `p` equally, and `b` is then the exact minimiser given `s`), and the loss keeps a gradient to move the output
//! with. In that branch `s` and `b` are held constant in the gradient term below.
//!
//! ## The gradient
//!
//! **SSI.** `(s, b)` minimise exactly the objective `L_ssi` is built from, so by the envelope theorem the derivative
//! of `L_ssi` through them vanishes and only the explicit dependence of `R_i = s p_i + b - t_i` on `p_i` remains:
//!
//! ```text
//! dL_ssi / dp_i = s R_i / |M|
//! ```
//!
//! exactly, with no term through `s` or `b`.
//!
//! **Gradient matching.** `L_grad` is not what `(s, b)` minimise, so its gradient flows through them. Let
//! `g_i = dL_grad / dR_i` (the sum, over every scale-`k` difference pixel `i` takes part in, of `+-sign(d) / |M|`).
//! Since `R_i = s p_i + b - t_i` with `s, b` functions of `p`:
//!
//! ```text
//! dL_grad / dp_i = s g_i + (ds/dp_i) sum_j g_j p_j + (db/dp_i) sum_j g_j
//! ```
//!
//! (`g_j` is zero off the mask, so the sums are over `M`.) The derivatives come from the closed form. Moving `p_i`
//! changes `a00` by `2 p_i`, `a01` by `1` and `b0` by `t_i`, so
//!
//! ```text
//! d det / dp_i = 2 a11 p_i - 2 a01
//! ds / dp_i = ( a11 t_i - b1 - s * ddet/dp_i ) / det
//! db / dp_i = ( 2 b1 p_i - b0 - a01 t_i - b * ddet/dp_i ) / det
//! ```
//!
//! both affine in `(p_i, t_i)`. Put `G1 = sum_j g_j p_j`, `G0 = sum_j g_j`; the total gradient for one image, with the
//! batch mean folded into `c = 1 / (N |M|)`, is
//!
//! ```text
//! dL / dp_i = m_i ( c s R_i + alpha s g_i + w_t t_i + w_p p_i + w_0 )
//! w_t = alpha ( G1 a11 - G0 a01 ) / det
//! w_p = alpha ( G1 (-2 s a11) + G0 (2 b1 - 2 b a11) ) / det
//! w_0 = alpha ( G1 (2 s a01 - b1) + G0 (2 b a01 - b0) ) / det
//! ```
//!
//! [`weights`] is that fold and the kernel `ssi_grad` is the per-pixel affine combination. It is checked against
//! central differences of an independent f64 oracle in `tests/it/p4_loss.rs`.
//!
//! ## Where the work runs
//!
//! Per-pixel arithmetic is four WGSL kernels (`ssi_moments`, `ssi_resid`, `grad_match`, `ssi_grad`); the per-image
//! reductions are summed on the host in f64, as every global reduction in brain is (`masked_l1`'s numerator,
//! `mse_value`). The existing row-reduction kernel (`segment_sum_rows`) walks each segment serially in binary32,
//! which loses both throughput (a handful of threads for a whole image) and accuracy (the sum is the quantity the
//! determinant cancels), and the readback is small beside the net's forward and backward.

use gpu_core::{DeviceBuffer, Gpu};

/// Gradient-matching scales (`K`): the residual subsampled by 1, 2, 4 and 8. The kernel `grad_match` loops this
/// many times; the two must agree.
pub const SCALES: u32 = 4;

/// Below this `det / (a00 a11)` (the prediction's variance over its second moment) the scale is not determined by
/// the data to binary32 precision: the moment terms carry relative error around `6e-8` each, which the ratio
/// magnifies into `s`.
pub const DET_REL_EPS: f64 = 1e-6;

/// The five masked moments of one image's scale-and-shift fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    /// `sum p^2` over the valid pixels.
    pub a00: f64,
    /// `sum p`.
    pub a01: f64,
    /// `|M|`, the number of valid pixels.
    pub a11: f64,
    /// `sum p t`.
    pub b0: f64,
    /// `sum t`.
    pub b1: f64,
}

/// The solved fit of one image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub s: f64,
    pub b: f64,
    /// `false` when the prediction was too flat for a unique scale and `(s, b) = (1, mean(t - p))` was used.
    pub exact: bool,
}

impl Moments {
    /// Sum the five term planes `ssi_moments` wrote for one image (`[5, hw]`, term-major).
    fn from_planes(planes: &[f32]) -> Moments {
        let hw = planes.len() / 5;
        let sum = |j: usize| planes[j * hw..(j + 1) * hw].iter().map(|&v| v as f64).sum::<f64>();
        Moments { a00: sum(0), a01: sum(1), a11: sum(2), b0: sum(3), b1: sum(4) }
    }

    /// `det / (a00 a11)`: 0 for a constant prediction, 1 for one with no mean.
    pub fn conditioning(&self) -> f64 {
        (self.a00 * self.a11 - self.a01 * self.a01) / (self.a00 * self.a11)
    }

    fn det(&self) -> f64 {
        self.a00 * self.a11 - self.a01 * self.a01
    }

    /// The closed-form least-squares `(s, b)`; the degenerate cases are in the module docs.
    pub fn solve(&self) -> Result<Fit, String> {
        if ![self.a00, self.a01, self.a11, self.b0, self.b1].iter().all(|v| v.is_finite()) {
            return Err(format!("non-finite moments {self:?}: the prediction or target holds a NaN or infinity"));
        }
        if self.a11 < 2.0 {
            return Err(format!("{} valid pixel(s): a scale-and-shift fit needs at least 2", self.a11));
        }
        // `!(x > eps)` also catches NaN (a00 = 0 gives 0 / 0).
        if self.conditioning().partial_cmp(&DET_REL_EPS) != Some(std::cmp::Ordering::Greater) {
            return Ok(Fit { s: 1.0, b: (self.b1 - self.a01) / self.a11, exact: false });
        }
        let det = self.det();
        Ok(Fit {
            s: (self.a11 * self.b0 - self.a01 * self.b1) / det,
            b: (self.a00 * self.b1 - self.a01 * self.b0) / det,
            exact: true,
        })
    }
}

/// The five per-image weights `(w_r, w_g, w_t, w_p, w_0)` the `ssi_grad` kernel combines (module docs, "The
/// gradient"): `c` is `1 / (N |M|)`, `g1 = sum g_j p_j`, `g0 = sum g_j`.
pub fn weights(m: &Moments, fit: &Fit, alpha: f64, c: f64, g1: f64, g0: f64) -> [f64; 5] {
    let (w_r, w_g) = (c * fit.s, alpha * fit.s);
    if !fit.exact {
        // s and b are constants of the fallback: no derivative through them.
        return [w_r, w_g, 0.0, 0.0, 0.0];
    }
    let (s, b, det) = (fit.s, fit.b, m.det());
    let w_t = alpha * (g1 * m.a11 - g0 * m.a01) / det;
    let w_p = alpha * (g1 * (-2.0 * s * m.a11) + g0 * (2.0 * m.b1 - 2.0 * b * m.a11)) / det;
    let w_0 = alpha * (g1 * (2.0 * s * m.a01 - m.b1) + g0 * (2.0 * b * m.a01 - m.b0)) / det;
    [w_r, w_g, w_t, w_p, w_0]
}

/// The loss's two parts, each already a batch mean (the total is `ssi + alpha * grad`).
#[derive(Clone, Copy, Debug)]
pub struct LossValue {
    pub ssi: f32,
    pub grad: f32,
}

impl LossValue {
    pub fn total(&self, alpha: f32) -> f32 {
        self.ssi + alpha * self.grad
    }
}

/// Kernel indices, resolved by name against [`crate::net::PIPELINES`].
struct Kernels {
    moments: usize,
    resid: usize,
    grad_match: usize,
    grad: usize,
}

/// The loss for a fixed `[n, 1, h, w]` prediction: scratch buffers allocated once, reused every step.
pub struct DepthLoss {
    n: usize,
    hw: usize,
    h: u32,
    w: u32,
    k: Kernels,
    planes: DeviceBuffer,
    sb: DeviceBuffer,
    coef: DeviceBuffer,
    resid: DeviceBuffer,
    ssi_term: DeviceBuffer,
    gm_term: DeviceBuffer,
    g: DeviceBuffer,
    gp: DeviceBuffer,
    wts: DeviceBuffer,
}

impl DepthLoss {
    pub fn new(gpu: &Gpu, n: u32, h: u32, w: u32) -> DepthLoss {
        let (n, hw) = (n as usize, (h * w) as usize);
        let k = |name: &str| crate::net::PIPELINES.iter().position(|(m, _)| *m == name).expect("loss kernel missing");
        DepthLoss {
            n,
            hw,
            h,
            w,
            k: Kernels { moments: k("ssi_moments"), resid: k("ssi_resid"), grad_match: k("grad_match"), grad: k("ssi_grad") },
            planes: gpu.storage((5 * n * hw) as u64),
            sb: gpu.storage((2 * n) as u64),
            coef: gpu.storage(n as u64),
            resid: gpu.storage((n * hw) as u64),
            ssi_term: gpu.storage((n * hw) as u64),
            gm_term: gpu.storage((n * hw) as u64),
            g: gpu.storage((n * hw) as u64),
            gp: gpu.storage((n * hw) as u64),
            wts: gpu.storage((5 * n) as u64),
        }
    }

    /// Evaluate the loss of `pred` against `tgt` (inverse depth) under `mask` (exactly `0.0` or `1.0` per pixel,
    /// all `[n, h, w]`), weighting gradient matching by `alpha`, and write `dL/d pred` into `d_pred`.
    ///
    /// Fails, naming the image, when one has fewer than two valid pixels or holds a non-finite value.
    pub fn eval(
        &self,
        gpu: &Gpu,
        pred: &DeviceBuffer,
        tgt: &DeviceBuffer,
        mask: &DeviceBuffer,
        alpha: f32,
        d_pred: &DeviceBuffer,
    ) -> Result<LossValue, String> {
        let (total, hw) = ((self.n * self.hw) as u32, self.hw as u32);

        // 1. per-pixel moment terms; host sums per image and solves the fit.
        let s = gpu.step(self.k.moments, &[pred, tgt, mask, &self.planes], &[total, hw], total);
        gpu.submit(&[], &[s]);
        let planes = gpu.read(&self.planes, 5 * self.n * self.hw);
        let mut moments = Vec::with_capacity(self.n);
        let mut fits = Vec::with_capacity(self.n);
        for (img, plane) in planes.chunks(5 * self.hw).enumerate() {
            let m = Moments::from_planes(plane);
            fits.push(m.solve().map_err(|e| format!("image {img}: {e}"))?);
            moments.push(m);
        }

        // 2. residual under the fit, then the gradient-matching terms and dL_grad/dR.
        let sb: Vec<f32> = fits.iter().flat_map(|ft| [ft.s as f32, ft.b as f32]).collect();
        let c: Vec<f32> = moments.iter().map(|m| (1.0 / (self.n as f64 * m.a11)) as f32).collect();
        gpu.write(&self.sb, bytemuck::cast_slice(&sb));
        gpu.write(&self.coef, bytemuck::cast_slice(&c));
        let s1 = gpu.step(self.k.resid, &[pred, tgt, mask, &self.sb, &self.resid, &self.ssi_term], &[total, hw], total);
        let s2 = gpu.step(
            self.k.grad_match,
            &[&self.resid, pred, mask, &self.coef, &self.gm_term, &self.g, &self.gp],
            &[total, hw, self.h, self.w],
            total,
        );
        gpu.submit(&[], &[s1, s2]);
        let n_px = self.n * self.hw;
        let ssi_term = gpu.read(&self.ssi_term, n_px);
        let gm_term = gpu.read(&self.gm_term, n_px);
        let g = gpu.read(&self.g, n_px);
        let gp = gpu.read(&self.gp, n_px);

        // 3. per-image reductions, folded into the five weights of the gradient kernel.
        let img_sum = |v: &[f32], n: usize| v[n * self.hw..(n + 1) * self.hw].iter().map(|&x| x as f64).sum::<f64>();
        let mut wts = Vec::with_capacity(5 * self.n);
        let (mut ssi, mut grad) = (0.0f64, 0.0f64);
        for n in 0..self.n {
            let m = &moments[n];
            ssi += img_sum(&ssi_term, n) / m.a11;
            grad += img_sum(&gm_term, n);
            let c = 1.0 / (self.n as f64 * m.a11);
            wts.extend(weights(m, &fits[n], alpha as f64, c, img_sum(&gp, n), img_sum(&g, n)).map(|w| w as f32));
        }
        gpu.write(&self.wts, bytemuck::cast_slice(&wts));
        let s3 = gpu.step(self.k.grad, &[pred, tgt, mask, &self.resid, &self.g, &self.wts, d_pred], &[total, hw], total);
        gpu.submit(&[], &[s3]);
        Ok(LossValue { ssi: (ssi / self.n as f64) as f32, grad: grad as f32 })
    }
}
