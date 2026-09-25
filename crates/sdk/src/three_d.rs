// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`Reconstruction`]: brain's 3D reconstruction surface. Photographs of a
//! static scene go in, and a 3D Gaussian Splatting scene comes out, with the
//! camera every photograph was taken from. No camera poses, no EXIF and no
//! learned model are needed: structure from motion recovers the cameras and a
//! sparse point cloud from the photographs themselves, and the fit grows that
//! cloud into the scene.
//!
//! Named for the `brain_arch::Domain` it belongs to (`three-d`, where the
//! `splat` rasterizer's own `arch!` row sits). Nothing is resolved from the
//! model store: the scene is learned from the photographs.
//!
//! ```no_run
//! let photos = ["a.jpg", "b.jpg", "c.jpg"].iter().map(brain::Image::open).collect::<brain::Result<Vec<_>>>()?;
//! let scene = brain::Reconstruction::builder().photos(photos).width(768).iterations(3000).run()?;
//! scene.save_ply("scene.ply")?;
//! scene.render(0)?.save("view0.png")?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Swedish Embedded AB implements photogrammetry pipelines, from a folder of
//! photographs to calibrated cameras and a radiance field, for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use splat::opt::FitCfg;
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

use crate::{Device, Error, Image, Result};

/// A reconstructed scene: the gaussians, the camera of every registered
/// photograph, and how well structure from motion explained them.
pub struct Reconstruction {
    splats: Splats,
    cameras: Vec<Camera>,
    registered: Vec<usize>,
    reprojection_rms_px: f64,
    render: RenderOpts,
    gpu: gpu_core::Gpu,
}

/// Hand-written, not derived: [`gpu_core::Gpu`] holds a live device handle.
impl std::fmt::Debug for Reconstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reconstruction")
            .field("gaussians", &self.splats.len())
            .field("registered", &self.registered)
            .field("reprojection_rms_px", &self.reprojection_rms_px)
            .finish()
    }
}

impl Reconstruction {
    pub fn builder() -> ReconstructionBuilder {
        ReconstructionBuilder {
            photos: Vec::new(),
            width: 1024,
            iterations: 3000,
            max_gaussians: 500_000,
            camera_model: false,
            device: Device::default(),
            progress: None,
        }
    }

    /// Gaussians in the scene.
    pub fn len(&self) -> usize {
        self.splats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.splats.is_empty()
    }

    /// How many photographs structure from motion placed; one camera each,
    /// in input order.
    pub fn views(&self) -> usize {
        self.cameras.len()
    }

    /// The input index of each camera's photograph. A photograph that
    /// overlapped too little with the others is left out, so this can skip
    /// indices.
    pub fn registered(&self) -> &[usize] {
        &self.registered
    }

    /// Structure from motion's reprojection error over the photographs, in
    /// their own pixels.
    pub fn reprojection_rms_px(&self) -> f64 {
        self.reprojection_rms_px
    }

    /// Write the scene as a standard 3D Gaussian Splatting PLY, which splat
    /// viewers and `brain splat view` open.
    pub fn save_ply(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref().to_str().ok_or_else(|| Error::Backend(format!("{}: not a UTF-8 path", path.as_ref().display())))?;
        splat::ply::write(path, &self.splats).map_err(Error::Backend)
    }

    /// Write the cameras as the `cameras.json` that `brain splat render
    /// --cameras` reads.
    pub fn save_cameras(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, splat::types::cameras_to_json(&self.cameras)).map_err(Error::Io)
    }

    /// The scene seen from camera `view` (`0..views()`), at the training
    /// resolution, rendered the way it was fitted.
    pub fn render(&self, view: usize) -> Result<Image> {
        let cam = self.cameras.get(view).ok_or_else(|| Error::Backend(format!("view {view} of {}", self.cameras.len())))?;
        let mut r = Renderer::new(&self.gpu, Kernels::at(0), self.splats.len().max(1), cam.width, cam.height, 0).growable();
        let gs = GpuSplats::upload(&self.gpu, &self.splats);
        if let Some(col) = splat::sh::shade(&self.splats, cam.eye()) {
            self.gpu.write_f32(&gs.colors, &col);
        }
        r.render(&self.gpu, &gs, cam, &self.render);
        let rgb = rgba_to_rgb(&r.read_rgba(&self.gpu, cam.width, cam.height));
        Image::from_rgb8(cam.width, cam.height, rgb.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect())
    }
}

/// The per-step progress callback: `(iteration, loss) -> keep going`.
type Progress = Box<dyn FnMut(usize, f32) -> bool>;

/// Builds a [`Reconstruction`] from photographs.
pub struct ReconstructionBuilder {
    photos: Vec<Image>,
    width: u32,
    iterations: usize,
    max_gaussians: usize,
    camera_model: bool,
    device: Device,
    progress: Option<Progress>,
}

impl ReconstructionBuilder {
    /// The photographs: one camera and one zoom, the whole scene static, each
    /// part of it in at least three of them.
    pub fn photos(mut self, photos: impl IntoIterator<Item = Image>) -> Self {
        self.photos.extend(photos);
        self
    }

    /// Training resolution across; the height follows the photographs'
    /// aspect. Structure from motion always runs on the full photographs.
    /// Default 1024.
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Optimizer steps. Default 3000.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// The most gaussians the scene may grow to. Default 500 000.
    pub fn max_gaussians(mut self, max_gaussians: usize) -> Self {
        self.max_gaussians = max_gaussians;
        self
    }

    /// Fit per-photograph exposure and white balance and the lens's
    /// vignetting alongside the scene. Off by default: on photographs taken at
    /// one exposure it only absorbs fit error.
    pub fn camera_model(mut self, on: bool) -> Self {
        self.camera_model = on;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Called after every optimizer step with the iteration and its loss;
    /// returning `false` stops the fit early with the scene as it is.
    pub fn progress(mut self, f: impl FnMut(usize, f32) -> bool + 'static) -> Self {
        self.progress = Some(Box::new(f));
        self
    }

    /// Recover the cameras and fit the scene. Blocks for as long as the fit
    /// takes: minutes for a few dozen photographs on one GPU.
    pub fn run(self) -> Result<Reconstruction> {
        let ReconstructionBuilder { photos, width, iterations, max_gaussians, camera_model, device, progress } = self;
        if photos.is_empty() {
            return Err(Error::MissingArgument("photos".into()));
        }
        if width == 0 || iterations == 0 || max_gaussians == 0 {
            return Err(Error::Backend(format!(
                "width ({width}), iterations ({iterations}) and max_gaussians ({max_gaussians}) must all be positive"
            )));
        }
        crate::device::resolve(&device)?;
        let photos: Vec<imaging::Rgb8> = photos.into_iter().map(Image::into_rgb8).collect();
        let sfm_cfg = sfm::incremental::SfmCfg::default();
        let set = recon::photogrammetry::training_set(&photos, width, 0.5, &sfm_cfg)
            .map_err(|e| Error::Backend(e.to_string()))?
            .upright();
        let preset = FitCfg::from_sparse_points(iterations, max_gaussians, set.targets.len());
        let cfg = FitCfg { log_every: 0, isp: if camera_model { Some(splat::isp::IspCfg::default()) } else { preset.isp }, ..preset };
        let gpu = gpu_core::Gpu::new(splat::PIPELINES);
        let mut progress = progress;
        let mut step = |it: usize, loss: f32| progress.as_mut().is_none_or(|f| f(it, loss));
        let fitted = splat::opt::fit_full(&gpu, Kernels::at(0), &set.init, &set.targets, &cfg, &mut step);
        Ok(Reconstruction {
            splats: fitted.scene,
            cameras: fitted.cams,
            registered: set.source,
            reprojection_rms_px: set.sfm.rms_px,
            render: RenderOpts { antialiased: cfg.antialiased, eps2d: cfg.eps2d, ..Default::default() },
            gpu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Everything knowable before any work starts is refused before any
    /// work starts.
    #[test]
    fn a_reconstruction_without_photographs_is_refused_up_front() {
        let err = Reconstruction::builder().run().unwrap_err();
        assert!(matches!(err, Error::MissingArgument(ref a) if a == "photos"), "{err}");
        let one = Image::from_rgb8(2, 2, vec![0; 12]).unwrap();
        let err = Reconstruction::builder().photos([one]).iterations(0).run().unwrap_err();
        assert!(err.to_string().contains("iterations (0)"), "{err}");
    }
}
