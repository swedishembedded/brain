// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`Reconstruction`]: brain's 3D reconstruction surface. Photographs of a
//! static scene go in, and a 3D Gaussian Splatting scene comes out, with the
//! camera every photograph was taken from. No camera poses, no EXIF and no
//! learned model are needed: structure from motion recovers every camera and
//! its real lens from the photographs themselves, multi-view stereo measures
//! the surfaces they show, and the fit grows a radiance field from that dense
//! start (`recon::photogrammetry::reconstruct`, the one pipeline the CLI and
//! this surface share).
//!
//! Named for the `brain_arch::Domain` it belongs to (`three-d`, where the
//! `splat` rasterizer's own `arch!` row sits). Nothing is resolved from the
//! model store: the scene is learned from the photographs.
//!
//! ```no_run
//! let photos = ["a.jpg", "b.jpg", "c.jpg"].iter().map(brain::Image::open).collect::<brain::Result<Vec<_>>>()?;
//! let scene = brain::Reconstruction::builder().photos(photos).run()?;
//! scene.save("scene")?;
//! scene.render(0)?.save("view0.png")?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Swedish Embedded AB implements photogrammetry pipelines, from a folder of
//! photographs to calibrated cameras and a radiance field, for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use recon::photogrammetry::{PhotoCfg, Reconstructed};
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::Kernels;

use crate::{Device, Error, Image, Result};

/// A reconstructed scene: the gaussians, the camera of every registered
/// photograph, and what the pipeline measured on the way.
pub struct Reconstruction {
    out: Reconstructed,
    gpu: gpu_core::Gpu,
}

/// Hand-written, not derived: [`gpu_core::Gpu`] holds a live device handle.
impl std::fmt::Debug for Reconstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reconstruction")
            .field("gaussians", &self.out.scene.len())
            .field("registered", &self.out.source)
            .field("reprojection_rms_px", &self.out.reprojection_rms_px)
            .finish()
    }
}

impl Reconstruction {
    pub fn builder() -> ReconstructionBuilder {
        ReconstructionBuilder {
            photos: Vec::new(),
            files: Vec::new(),
            cfg: PhotoCfg::default(),
            device: Device::default(),
            progress: None,
            log: None,
        }
    }

    /// Gaussians in the scene.
    pub fn len(&self) -> usize {
        self.out.scene.len()
    }

    pub fn is_empty(&self) -> bool {
        self.out.scene.is_empty()
    }

    /// How many photographs structure from motion placed; one camera each,
    /// in input order.
    pub fn views(&self) -> usize {
        self.out.cameras.len()
    }

    /// The input index of each camera's photograph. A photograph that
    /// overlapped too little with the others is left out, so this can skip
    /// indices.
    pub fn registered(&self) -> &[usize] {
        &self.out.source
    }

    /// Structure from motion's reprojection error over the photographs, in
    /// their own pixels.
    pub fn reprojection_rms_px(&self) -> f64 {
        self.out.reprojection_rms_px
    }

    /// Write the scene as a standard 3D Gaussian Splatting PLY, which splat
    /// viewers and `brain splat view` open; a fitted environment is baked in
    /// as a distant shell of gaussians.
    pub fn save_ply(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref().to_str().ok_or_else(|| Error::Backend(format!("{}: not a UTF-8 path", path.as_ref().display())))?;
        splat::ply::write(path, &self.out.export()).map_err(Error::Backend)
    }

    /// Write the cameras, each with its lens, as the `cameras.json` that
    /// `brain splat render --cameras` reads.
    pub fn save_cameras(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, splat::types::cameras_to_json(&self.out.cameras)).map_err(Error::Io)
    }

    /// Write the whole reconstruction to the directory `dir`: `scene.ply`,
    /// `cameras.json` and `reconstruction.json` (which photograph each camera
    /// is, how the scene was fitted and rendered, and what structure from
    /// motion and stereo measured).
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(Error::Io)?;
        self.save_ply(dir.join("scene.ply"))?;
        self.save_cameras(dir.join("cameras.json"))?;
        let o = &self.out;
        let meta = serde_json::json!({
            "gaussians": o.scene.len(),
            "source": o.source,
            "reprojection_rms_px": o.reprojection_rms_px,
            "iterations": o.iterations,
            "max_gaussians": o.max_gaussians,
            "loss": o.loss,
            "render": { "ray": o.render.ray, "antialiased": o.render.antialiased, "eps2d": o.render.eps2d },
            "stereo": o.dense.as_ref().map(|d| serde_json::json!({ "points": d.points, "coverage": d.coverage })),
            "camera_model": o.isp.as_ref().map(|i| i.summary()),
            "environment": o.env.as_ref().map(|e| serde_json::json!({ "degree": e.degree, "coeffs": e.coeffs })),
        });
        std::fs::write(dir.join("reconstruction.json"), serde_json::to_string_pretty(&meta).map_err(|e| Error::Backend(e.to_string()))?).map_err(Error::Io)
    }

    /// The scene seen from camera `view` (`0..views()`), at the training
    /// resolution, rendered the way it was fitted - through that
    /// photograph's own fitted camera model when the fit had one.
    pub fn render(&self, view: usize) -> Result<Image> {
        let cam = self.out.cameras.get(view).ok_or_else(|| Error::Backend(format!("view {view} of {}", self.out.cameras.len())))?;
        let scene = &self.out.scene;
        let mut r = Renderer::new(&self.gpu, Kernels::at(0), scene.len().max(1), cam.width, cam.height, 0).growable();
        let gs = GpuSplats::upload(&self.gpu, scene);
        if let Some(col) = splat::sh::shade(scene, cam.eye()) {
            self.gpu.write_f32(&gs.colors, &col);
        }
        r.render(&self.gpu, &gs, cam, &self.out.render);
        if let Some(env) = &self.out.env {
            splat::env::EnvDevice::new(&self.gpu, env).composite(&self.gpu, &Kernels::at(0), &r.img, cam, &self.out.render);
        }
        let mut rgb = rgba_to_rgb(&r.read_rgba(&self.gpu, cam.width, cam.height));
        if let Some(isp) = &self.out.isp {
            rgb = isp.forward(view, cam, &rgb);
        }
        Image::from_rgb8(cam.width, cam.height, rgb.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect())
    }
}

/// The per-step progress callback: `(iteration, loss) -> keep going`.
type Progress = Box<dyn FnMut(usize, f32) -> bool>;

/// Builds a [`Reconstruction`] from photographs.
pub struct ReconstructionBuilder {
    photos: Vec<Image>,
    files: Vec<std::path::PathBuf>,
    cfg: PhotoCfg,
    device: Device,
    progress: Option<Progress>,
    log: Option<Box<dyn FnMut(&str)>>,
}

impl ReconstructionBuilder {
    /// The photographs: one static scene, each part of it in at least three
    /// of them. They may come from several cameras and zoom settings.
    pub fn photos(mut self, photos: impl IntoIterator<Item = Image>) -> Self {
        self.photos.extend(photos);
        self
    }

    /// Photographs to read from disk, after any given as [`Image`]s. Read
    /// this way a photograph keeps what its file records beyond the pixels:
    /// its orientation, bit depth and colour encoding, the exposure it was
    /// shot at (so the camera model starts from the true brightness
    /// differences) and where it was taken (so the scene comes back in
    /// metres, upright, when the fixes are precise enough).
    pub fn photo_files<P: AsRef<std::path::Path>>(mut self, paths: impl IntoIterator<Item = P>) -> Self {
        self.files.extend(paths.into_iter().map(|p| p.as_ref().to_path_buf()));
        self
    }

    /// The widest training image: the photographs are halved exactly until
    /// they fit (never resampled at another factor). Structure from motion
    /// always runs on the full photographs. Default 2048.
    pub fn max_width(mut self, width: u32) -> Self {
        self.cfg.max_width = width;
        self
    }

    /// Optimizer steps. By default about five hundred visits per photograph,
    /// from 3000 to 30 000.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.cfg.iterations = Some(iterations);
        self
    }

    /// The most gaussians the scene may hold. By default what the dense
    /// start needs and a quarter more.
    pub fn max_gaussians(mut self, max_gaussians: usize) -> Self {
        self.cfg.max_gaussians = Some(max_gaussians);
        self
    }

    /// Start from multi-view stereo's dense surfaces, with its range and
    /// normals as priors (the default), or from the sparse structure-from-
    /// motion points alone.
    pub fn dense(mut self, on: bool) -> Self {
        self.cfg.dense = on.then(Default::default);
        self
    }

    /// The photographs may hold people, traffic or anything else that was
    /// not there in all of them: stop supervising what the scene cannot
    /// explain in one view. Off by default; on a static capture it can only
    /// withhold supervision from regions that are merely hard to fit.
    pub fn transients(mut self, on: bool) -> Self {
        self.cfg.transients = on;
        self
    }

    /// Fit the environment behind the scene - sky and distant scenery, as
    /// radiance by direction - at this spherical-harmonic degree (up to 8).
    pub fn environment(mut self, degree: u32) -> Self {
        self.cfg.environment = Some(degree);
        self
    }

    /// Fit the photometric camera alongside the scene: per-photograph
    /// exposure and white balance, the lens's vignetting, the sensor's colour
    /// matrix and response curve. On by default; on photographs all taken at
    /// one exposure it changes little, on a capture whose exposure or white
    /// balance changes it is what keeps the scene consistent.
    pub fn camera_model(mut self, on: bool) -> Self {
        self.cfg.camera_model = on.then(splat::isp::IspCfg::default);
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

    /// Called with one line per finished stage (structure from motion,
    /// stereo, the fit), saying what it found and how long it took.
    pub fn log(mut self, f: impl FnMut(&str) + 'static) -> Self {
        self.log = Some(Box::new(f));
        self
    }

    /// Recover the cameras and fit the scene. Blocks for as long as the fit
    /// takes: minutes to hours, with the photographs' count and resolution.
    pub fn run(self) -> Result<Reconstruction> {
        let ReconstructionBuilder { photos, files, cfg, device, progress, log } = self;
        if photos.is_empty() && files.is_empty() {
            return Err(Error::MissingArgument("photos".into()));
        }
        if cfg.max_width == 0 || cfg.iterations == Some(0) || cfg.max_gaussians == Some(0) {
            return Err(Error::Backend(format!(
                "max_width ({}), iterations ({}) and max_gaussians ({}) must all be positive",
                cfg.max_width,
                cfg.iterations.map_or("auto".into(), |v| v.to_string()),
                cfg.max_gaussians.map_or("auto".into(), |v| v.to_string()),
            )));
        }
        crate::device::resolve(&device)?;
        let mut all: Vec<imaging::Photo> = photos.into_iter().map(|i| imaging::Photo::from_rgb8(&i.into_rgb8())).collect();
        for f in &files {
            all.push(imaging::load_photo(f).map_err(Error::Backend)?);
        }
        let gpu = gpu_core::Gpu::new(&recon::photogrammetry::pipelines());
        let (mut progress, mut log) = (progress, log);
        let mut step = |it: usize, loss: f32| progress.as_mut().is_none_or(|f| f(it, loss));
        let mut line = |m: &str| {
            if let Some(f) = log.as_mut() {
                f(m)
            }
        };
        let out = recon::photogrammetry::reconstruct_photos(&gpu, &all, &cfg, &mut line, &mut step).map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Reconstruction { out, gpu })
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
        let one = Image::from_rgb8(2, 2, vec![0; 12]).unwrap();
        let err = Reconstruction::builder().photos([one]).max_width(0).run().unwrap_err();
        assert!(err.to_string().contains("max_width (0)"), "{err}");
    }
}
