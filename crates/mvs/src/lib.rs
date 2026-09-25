// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense multi-view stereo on the GPU, through any lens.
//!
//! From calibrated photographs (structure from motion's poses, its
//! [`camera::Intrinsics`] - a phone's ultra-wide as the Kannala-Brandt fisheye
//! it is - and its sparse tracks) to the geometry a splat fit needs: a range
//! and normal for most pixels of every view, a fused point cloud, and a
//! starting scene of thin gaussians lying on the surface.
//!
//! The method is PatchMatch stereo with per-pixel view selection
//! (Schönberger, Zheng, Pollefeys, Frahm, "Pixelwise View Selection for
//! Unstructured Multi-View Stereo", ECCV 2016) organised the way the ACMM
//! family runs it on a GPU (Xu & Tao, "Multi-Scale Geometric Consistency
//! Guided Multi-View Stereo", CVPR 2019; "Planar Prior Assisted PatchMatch
//! Multi-View Stereo", AAAI 2020): red-black adaptive checkerboard
//! propagation, multi-hypothesis joint view selection, a coarse-to-fine
//! pyramid, and geometric consistency between the views' maps, which is what
//! fills weakly textured surfaces. Implemented from the papers:
//!
//! 1. [`select`] - source views per reference view, and its range bounds,
//!    from the sparse tracks;
//! 2. [`stereo`] - the pyramid of PatchMatch passes (`mvs_pm.wgsl`), the
//!    consistency filter (`mvs_filter.wgsl`) and the result, per view, as a
//!    [`depth::DepthMap`];
//! 3. [`fuse`] - one point per surface sample across all views
//!    (`mvs_fuse_acc.wgsl`, `mvs_fuse_out.wgsl`);
//! 4. [`init`] - the fused points as surface-aligned gaussians.
//!
//! **No rectification, anywhere.** A hypothesis is a range along a pixel's
//! own unit ray (`lens_unproject`) and a normal; every patch sample is carried
//! through the plane into the source view and imaged through that view's real
//! lens (`lens_project`). Ranges are distances from the camera centre, the
//! quantity the splat renderer's depth prior compares - never z-depth.
//!
//! Working images are exact box halvings of the photographs (`mvs_prepare`),
//! so a level's intrinsics are the photograph's scaled by exactly `2^-h`.
//!
//! Swedish Embedded AB implements dense 3D reconstruction from photographs -
//! calibration, multi-view stereo and radiance fields - for its clients. If
//! your team needs accurate geometry from ordinary cameras, including wide and
//! fisheye lenses, you can procure our services by sending an email to
//! info@swedishembedded.com.

pub mod depth;
pub mod fuse;
pub mod init;
pub mod select;
pub mod stereo;

pub use depth::DepthMap;
pub use fuse::{fuse, FuseCfg, Fused};
pub use init::{to_splats, SplatInit};
pub use select::{SelectCfg, Track};
pub use stereo::{depth_maps, FilterCfg, Stereo, StereoCfg, View};

/// WGSL kernels this crate dispatches, in [`Kernels`] order. Pass to
/// `Gpu::new(..)` alone or appended to another pipeline list (then resolve
/// indices with [`Kernels::at`] at that base).
pub const PIPELINES: &[(&str, &str)] = &[
    ("region_copy", kernels::REGION_COPY),
    ("mvs_prepare", kernels::MVS_PREPARE),
    ("mvs_rays", kernels::MVS_RAYS),
    ("mvs_quad", kernels::MVS_QUAD),
    ("mvs_pm", kernels::MVS_PM),
    ("mvs_upsample", kernels::MVS_UPSAMPLE),
    ("mvs_filter", kernels::MVS_FILTER),
    ("mvs_fuse_acc", kernels::MVS_FUSE_ACC),
    ("mvs_fuse_out", kernels::MVS_FUSE_OUT),
];

/// Positional kernel indices into a `Gpu` whose pipeline list contains
/// [`PIPELINES`] starting at `base`.
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    pub region_copy: usize,
    pub prepare: usize,
    pub rays: usize,
    pub quad: usize,
    pub pm: usize,
    pub upsample: usize,
    pub filter: usize,
    pub fuse_acc: usize,
    pub fuse_out: usize,
}

impl Kernels {
    pub fn at(base: usize) -> Kernels {
        Kernels {
            region_copy: base,
            prepare: base + 1,
            rays: base + 2,
            quad: base + 3,
            pm: base + 4,
            upsample: base + 5,
            filter: base + 6,
            fuse_acc: base + 7,
            fuse_out: base + 8,
        }
    }
}

/// The world-to-camera pose of a splat camera, in f64.
pub(crate) fn pose_of(cam: &splat::types::Camera) -> sfm::camera::Pose {
    let m = cam.c2w.map(f64::from);
    let r = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
    let t = sfm::linalg::scale(sfm::linalg::mv(&r, [m[3], m[7], m[11]]), -1.0);
    sfm::camera::Pose { r, t }
}

/// Why multi-view stereo could not run.
#[derive(Debug, Clone, PartialEq)]
pub enum MvsError {
    /// A view's pixels do not match its camera's size.
    ImageSize { view: usize, expected: usize, got: usize },
    /// The per-reference image stack would exceed what one device binding
    /// may address; run at a coarser `halving` or with fewer sources.
    TooLarge { bytes: u64, limit: u64 },
    /// A configuration value is out of its range.
    Config(String),
    /// Inputs that must correspond do not.
    Mismatch(String),
}

impl std::fmt::Display for MvsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MvsError::ImageSize { view, expected, got } => {
                write!(f, "view {view}: {got} bytes of pixels, its camera needs {expected} (interleaved 8-bit RGB)")
            }
            MvsError::TooLarge { bytes, limit } => write!(
                f,
                "a reference view's image stack needs {bytes} bytes in one device binding, the device allows {limit}; \
                 raise `halving` or lower `max_sources`"
            ),
            MvsError::Config(m) => write!(f, "invalid configuration: {m}"),
            MvsError::Mismatch(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for MvsError {}

/// Wall-clock seconds per stage, measured around device syncs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timings {
    pub stages: Vec<(String, f64)>,
}

impl Timings {
    pub fn total(&self) -> f64 {
        self.stages.iter().map(|s| s.1).sum()
    }

    fn push(&mut self, name: impl Into<String>, t: std::time::Instant) {
        self.stages.push((name.into(), t.elapsed().as_secs_f64()));
    }
}

impl std::fmt::Display for Timings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (name, s) in &self.stages {
            writeln!(f, "  {name:<28} {s:8.3} s")?;
        }
        write!(f, "  {:<28} {:8.3} s", "total", self.total())
    }
}
