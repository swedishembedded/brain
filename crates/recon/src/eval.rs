// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Judging a reconstruction on views it never saw.
//!
//! A fit is scored by the loss on its training views, and that number says
//! how well it memorized them - nothing more. A scene can reproduce every
//! photograph it was fitted to to the pixel while being a cloud of
//! view-specific billboards between them: measured on a 16-photograph
//! capture, a fit reached 30.96 dB on its training views and 13.83 dB on the
//! four it was not shown. Only views held out of the fit say whether the
//! scene is right, so this module is how a reconstruction is judged:
//!
//! * [`holdout`] - which registered photographs are withheld: every k-th,
//!   spread over the capture rather than its ends;
//! * [`score`] - PSNR and SSIM of each view's render against its photograph,
//!   over the pixels its lens images;
//! * [`path`] and [`stability`] - a smooth camera path through the capture
//!   and how much a render of it flickers (splats popping as their order
//!   flips, sub-pixel detail aliasing), which no still image shows.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines and the evaluation that keeps them honest. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::TargetView;
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

/// Which of `n` registered views are withheld from a fit: every `every`-th,
/// starting half a stride in so the withheld views are not the capture's
/// first and last. `every` of 0 or 1 withholds nothing.
pub fn holdout(n: usize, every: usize) -> Vec<bool> {
    (0..n).map(|i| every > 1 && i % every == every / 2).collect()
}

/// How one view's render compares with its photograph.
#[derive(Clone, Copy, Debug)]
pub struct ViewScore {
    /// Index into the views that were scored.
    pub view: usize,
    pub psnr: f64,
    pub ssim: f64,
}

/// Every scored view, and their means.
#[derive(Clone, Debug, Default)]
pub struct Scores {
    pub views: Vec<ViewScore>,
}

impl Scores {
    pub fn mean_psnr(&self) -> f64 {
        self.views.iter().map(|v| v.psnr).sum::<f64>() / self.views.len().max(1) as f64
    }
    pub fn mean_ssim(&self) -> f64 {
        self.views.iter().map(|v| v.ssim).sum::<f64>() / self.views.len().max(1) as f64
    }
}

/// A scene ready to render: the gaussians and the 3D filter they were fitted
/// under, on the device, with a renderer sized for them.
pub struct Viewer {
    gpu: Gpu,
    scene: Splats,
    splats: GpuSplats,
    renderer: Renderer,
    opts: RenderOpts,
}

impl Viewer {
    /// `filter3d` is the fit's per-gaussian 3D filter variance (empty =
    /// none); `opts` how the scene was fitted.
    pub fn new(gpu: &Gpu, scene: &Splats, filter3d: &[f32], opts: RenderOpts, max_w: u32, max_h: u32) -> Viewer {
        let mut splats = GpuSplats::upload(gpu, scene);
        if filter3d.len() == scene.len() && !scene.is_empty() {
            splats = splats.with_filter3d(gpu, filter3d);
        }
        Viewer {
            gpu: gpu.share(),
            scene: scene.clone(),
            splats,
            renderer: Renderer::new(gpu, Kernels::at(0), scene.len().max(1), max_w, max_h, 0).growable(),
            opts: RenderOpts { ray: true, ..opts },
        }
    }

    /// The scene through `cam`: interleaved RGB in [0,1].
    pub fn render(&mut self, cam: &Camera) -> Vec<f32> {
        if let Some(c) = splat::sh::shade(&self.scene, cam.eye()) {
            self.gpu.write_f32(&self.splats.colors, &c);
        }
        self.renderer.render(&self.gpu, &self.splats, cam, &self.opts);
        rgba_to_rgb(&self.renderer.read_rgba(&self.gpu, cam.width, cam.height))
    }
}

/// Score every view's render against its photograph over the pixels it
/// supervises. Returns the scores and the renders.
pub fn score(viewer: &mut Viewer, views: &[TargetView]) -> (Scores, Vec<Vec<f32>>) {
    let mut scores = Scores::default();
    let mut renders = Vec::with_capacity(views.len());
    for (i, v) in views.iter().enumerate() {
        let img = viewer.render(&v.cam);
        let (w, h) = (v.cam.width as usize, v.cam.height as usize);
        let mask = v.mask.as_deref();
        scores.views.push(ViewScore {
            view: i,
            psnr: splat::quality::psnr_masked(&img, &v.rgb, mask),
            ssim: splat::quality::ssim(&img, &v.rgb, w, h, mask),
        });
        renders.push(img);
    }
    (scores, renders)
}

/// A smooth path through `cams`: visited in a nearest-neighbour chain from
/// the first, `steps` interpolated cameras between consecutive ones
/// (rotation by slerp, centre and calibration linearly), each at the size of
/// the camera it leaves.
pub fn path(cams: &[Camera], steps: usize) -> Vec<Camera> {
    if cams.len() < 2 {
        return cams.to_vec();
    }
    let mut order = vec![0usize];
    let mut left: Vec<usize> = (1..cams.len()).collect();
    while !left.is_empty() {
        let last = cams[*order.last().expect("non-empty")].eye();
        let (j, _) = left
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                let e = cams[i].eye();
                (j, (e[0] - last[0]).powi(2) + (e[1] - last[1]).powi(2) + (e[2] - last[2]).powi(2))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty");
        order.push(left.remove(j));
    }
    let mut out = Vec::new();
    for w in order.windows(2) {
        let (a, b) = (&cams[w[0]], &cams[w[1]]);
        let (qa, qb) = (quat_of(&a.c2w), quat_of(&b.c2w));
        for s in 0..steps {
            let t = s as f32 / steps as f32;
            let q = slerp(qa, qb, t);
            let r = mat_of(q);
            let mut c2w = a.c2w;
            for i in 0..3 {
                for j in 0..3 {
                    c2w[i * 4 + j] = r[i * 3 + j];
                }
                c2w[i * 4 + 3] = a.c2w[i * 4 + 3] * (1.0 - t) + b.c2w[i * 4 + 3] * t;
            }
            out.push(Camera { c2w, ..*a });
        }
    }
    out.push(cams[*order.last().expect("non-empty")]);
    out
}

/// [`splat::quality::temporal_instability`] of the scene rendered along
/// `cams`.
pub fn stability(viewer: &mut Viewer, cams: &[Camera]) -> f64 {
    let frames: Vec<Vec<f32>> = cams.iter().map(|c| viewer.render(c)).collect();
    splat::quality::temporal_instability(&frames)
}

fn quat_of(m: &[f32; 16]) -> [f32; 4] {
    let (r00, r11, r22) = (m[0], m[5], m[10]);
    let tr = r00 + r11 + r22;
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [0.25 * s, (m[9] - m[6]) / s, (m[2] - m[8]) / s, (m[4] - m[1]) / s]
    } else if r00 > r11 && r00 > r22 {
        let s = (1.0 + r00 - r11 - r22).sqrt() * 2.0;
        [(m[9] - m[6]) / s, 0.25 * s, (m[1] + m[4]) / s, (m[2] + m[8]) / s]
    } else if r11 > r22 {
        let s = (1.0 + r11 - r00 - r22).sqrt() * 2.0;
        [(m[2] - m[8]) / s, (m[1] + m[4]) / s, 0.25 * s, (m[6] + m[9]) / s]
    } else {
        let s = (1.0 + r22 - r00 - r11).sqrt() * 2.0;
        [(m[4] - m[1]) / s, (m[2] + m[8]) / s, (m[6] + m[9]) / s, 0.25 * s]
    };
    let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    q.map(|v| v / n)
}

fn slerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let mut d = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>();
    let b = if d < 0.0 {
        d = -d;
        b.map(|v| -v)
    } else {
        b
    };
    if d > 0.9995 {
        let q: [f32; 4] = std::array::from_fn(|i| a[i] + t * (b[i] - a[i]));
        let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        return q.map(|v| v / n);
    }
    let th = d.acos();
    let (sa, sb) = (((1.0 - t) * th).sin() / th.sin(), (t * th).sin() / th.sin());
    std::array::from_fn(|i| sa * a[i] + sb * b[i])
}

fn mat_of(q: [f32; 4]) -> [f32; 9] {
    let [w, x, y, z] = q;
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kth_view_is_held_out_away_from_the_ends() {
        let h = holdout(16, 4);
        assert_eq!(h.iter().filter(|&&v| v).count(), 4);
        assert!(!h[0] && !h[15]);
        assert!(h[2] && h[6] && h[10] && h[14]);
        assert!(holdout(5, 0).iter().all(|&v| !v));
    }

    /// The path passes through every camera, and between two it turns and
    /// moves smoothly: the rotation at the midpoint is halfway.
    #[test]
    fn a_path_visits_every_camera_and_interpolates_between_them() {
        let a = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 60.0, 32, 24);
        let b = Camera::look_at([1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, -1.0, 0.0], 60.0, 32, 24);
        let p = path(&[a, b], 4);
        assert_eq!(p.len(), 5);
        assert!((p[2].eye()[0] - 0.5).abs() < 1e-5, "{:?}", p[2].eye());
        assert!((p[4].eye()[0] - 1.0).abs() < 1e-6);
        let q = quat_of(&a.c2w);
        let back = mat_of(q);
        for i in 0..3 {
            for j in 0..3 {
                assert!((back[i * 3 + j] - a.c2w[i * 4 + j]).abs() < 1e-5);
            }
        }
    }
}
