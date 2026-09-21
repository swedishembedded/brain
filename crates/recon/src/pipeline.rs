// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Running the plan: one pass per chunk, each registered onto the world the
//! earlier chunks built, then one global finish.

use imaging::Rgb8;
use splat::align::{apply_sim3, camera_residual, sim3_from_cameras, transform_c2w_sim3};
use splat::types::{Camera, Splats};

use crate::plan::{plan_chunks, ChunkPlan, MIN_OVERLAP};
use crate::select::{select, SelectOpts, Selection};
use crate::{ingest, Frame, PipelineError, ReconstructionModel, Source};

// ------------------------------------------------------- shape normalisation

/// The pixel size most of the capture is already at. A clip's hundreds of
/// frames outvote a handful of stills, which is the right way round: resizing
/// the few costs less total quality than reshaping the many.
///
/// Ties go to the earliest frame with that size, so the answer does not depend
/// on iteration order.
pub fn majority_size(frames: &[Frame]) -> (u32, u32) {
    let mut sizes: Vec<((u32, u32), usize, usize)> = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let s = (f.image.w, f.image.h);
        match sizes.iter_mut().find(|e| e.0 == s) {
            Some(e) => e.1 += 1,
            None => sizes.push((s, 1, i)),
        }
    }
    sizes
        .into_iter()
        .max_by_key(|&(_, count, first)| (count, std::cmp::Reverse(first)))
        .map(|(s, _, _)| s)
        .unwrap_or((0, 0))
}

/// Bring a mixed capture onto ONE pixel size, at the aspect ratio the model
/// asked for.
///
/// One pass takes one shape for the whole batch, and a folder of photographs
/// beside a clip has neither the same pixel size nor the same aspect ratio. So
/// every frame is centre-cropped to `aspect`'s ratio and resized onto the
/// majority frame's cropped size.
///
/// Cropping rather than padding: a model estimates each frame's field of view,
/// so a crop simply narrows it, while a pad would invent black borders the
/// model then has to explain as geometry.
///
/// A capture that is already uniform - the common case, one camera - comes
/// back untouched, pixel for pixel. That matters: the model does its own
/// resampling to its own grid, and a resample here would put a second,
/// lower-quality one in front of it.
pub fn normalize(frames: &[Frame], aspect: (u32, u32)) -> Vec<Frame> {
    if frames.is_empty() || aspect.0 == 0 || aspect.1 == 0 {
        return frames.to_vec();
    }
    let (mw, mh) = majority_size(frames);
    let (cw, ch) = crop_size(mw, mh, aspect);
    frames
        .iter()
        .map(|f| {
            let image = fit_to(&f.image, cw, ch);
            Frame { image, label: f.label.clone(), source: f.source }
        })
        .collect()
}

/// The largest centred window of `w x h` with `aspect`'s ratio.
fn crop_size(w: u32, h: u32, aspect: (u32, u32)) -> (u32, u32) {
    let (aw, ah) = (aspect.0 as u64, aspect.1 as u64);
    let (w64, h64) = (w as u64, h as u64);
    let (kw, kh) = if w64 * ah >= h64 * aw { ((h64 * aw) / ah, h64) } else { (w64, (w64 * ah) / aw) };
    ((kw.clamp(1, w64)) as u32, (kh.clamp(1, h64)) as u32)
}

/// Centre-crop to the target ratio, then resize onto it exactly.
fn fit_to(img: &Rgb8, cw: u32, ch: u32) -> Rgb8 {
    if (img.w, img.h) == (cw, ch) {
        return img.clone();
    }
    let (kw, kh) = crop_size(img.w, img.h, (cw, ch));
    let cropped = if (kw, kh) == (img.w, img.h) {
        img.clone()
    } else {
        let (w, x0, y0) = (img.w as usize, ((img.w - kw) / 2) as usize, ((img.h - kh) / 2) as usize);
        let mut px = Vec::with_capacity((kw * kh) as usize * 3);
        for y in 0..kh as usize {
            let row = ((y0 + y) * w + x0) * 3;
            px.extend_from_slice(&img.px[row..row + kw as usize * 3]);
        }
        Rgb8::new(kw, kh, px).expect("crop is w*h*3")
    };
    if (cropped.w, cropped.h) == (cw, ch) {
        return cropped;
    }
    let src: Vec<f32> = cropped.px.iter().map(|&v| v as f32).collect();
    let out = imaging::resize_bilinear_hwc(&src, 3, cropped.w, cropped.h, cw, ch);
    Rgb8::new(cw, ch, out.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect())
        .expect("resize is w*h*3")
}

// ------------------------------------------------------------- accumulation

/// How one chunk landed in the world. This is the honest read on the whole
/// reconstruction: a scene whose chunks all registered tightly is one scene,
/// and one with a loose join is two scenes overlaid.
#[derive(Clone, Copy, Debug)]
pub struct ChunkReport {
    pub chunk: usize,
    pub frames: usize,
    /// Frames shared with what was already placed in the world.
    pub shared: usize,
    /// Worst shared-camera miss, in world units.
    pub residual: f64,
    /// Spread of the shared cameras, in world units.
    pub span: f64,
    /// `residual / span`, which is what the gate uses.
    pub relative: f64,
    /// The scale this chunk's own world was solved to be at.
    pub scale: f64,
    pub gaussians: usize,
}

/// The finished scene.
#[derive(Clone, Default)]
pub struct Reconstruction {
    pub splats: Splats,
    /// One camera per planned frame, in the world frame.
    pub cameras: Vec<Camera>,
    pub reports: Vec<ChunkReport>,
}

impl Reconstruction {
    /// The worst relative registration residual over the whole capture - the
    /// one number that says whether this is one scene.
    pub fn worst_relative(&self) -> f64 {
        self.reports.iter().map(|r| r.relative).fold(0.0, f64::max)
    }
}

/// The accumulation and finishing knobs.
#[derive(Clone, Copy, Debug)]
pub struct ReconstructOpts {
    /// Reject a chunk whose shared cameras miss by more than this FRACTION of
    /// their own spread. Deliberately relative: a feed-forward chunk's world
    /// has no unit, so an absolute tolerance would mean a different thing in
    /// every capture.
    pub max_relative_residual: f64,
    /// Voxel size for duplicate fusion, in world units. `0` disables.
    pub voxel: f32,
    /// Keep only the heaviest N voxels in the final scene. `0` is unbounded.
    pub max_points: usize,
    /// Drop gaussians above this quantile of largest-axis scale. `>= 1` keeps
    /// everything.
    pub max_scale_quantile: f32,
    /// Land the finished scene in the frame the cameras describe.
    pub orient: bool,
}

impl Default for ReconstructOpts {
    fn default() -> ReconstructOpts {
        ReconstructOpts {
            max_relative_residual: 0.1,
            voxel: 0.002,
            max_points: 0,
            max_scale_quantile: 0.98,
            orient: true,
        }
    }
}

fn c2w64(c: &Camera) -> [f64; 16] {
    std::array::from_fn(|i| c.c2w[i] as f64)
}

fn moved(c: &Camera, m: &[f64; 16]) -> Camera {
    Camera { c2w: std::array::from_fn(|i| m[i] as f32), ..*c }
}

/// How far apart a set of cameras sit - the length a registration residual is
/// judged against.
fn camera_span(c: &[[f64; 16]]) -> f64 {
    let e: Vec<[f64; 3]> = c.iter().map(|m| [m[3], m[7], m[11]]).collect();
    let mut span: f64 = 0.0;
    for i in 0..e.len() {
        for j in i + 1..e.len() {
            let d = ((e[i][0] - e[j][0]).powi(2) + (e[i][1] - e[j][1]).powi(2) + (e[i][2] - e[j][2]).powi(2)).sqrt();
            span = span.max(d);
        }
    }
    span
}

/// Reconstruct every chunk of a plan and accumulate them into one world.
///
/// The FIRST chunk defines the world; it is not transformed. Every later chunk
/// is solved onto the cameras already placed THERE - onto the world, not onto
/// its predecessor's local frame - so the transform applied is `world <-
/// chunk`, taken as solved, never inverted and never composed by hand. That
/// direction is the thing this gets wrong when it is wrong, and it fails
/// quietly: the scene comes back mirrored in scale, or every chunk after the
/// first lands somewhere plausible but not where its own cameras say.
///
/// `frames` must already be normalised (see [`normalize`]) and must be exactly
/// the frames the plan was built for.
pub fn reconstruct_plan(
    frames: &[Frame],
    plan: &ChunkPlan,
    model: &mut impl ReconstructionModel,
    opts: &ReconstructOpts,
) -> Result<Reconstruction, PipelineError> {
    if frames.is_empty() {
        return Err(PipelineError::NoFrames);
    }
    if frames.len() != plan.frames {
        return Err(PipelineError::PlanMismatch { frames: frames.len(), planned: plan.frames });
    }

    let mut placed: Vec<Option<Camera>> = vec![None; plan.frames];
    let mut parts: Vec<Splats> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut reports = Vec::new();

    for (ci, chunk) in plan.chunks.iter().enumerate() {
        let scene = model
            .reconstruct(&frames[chunk.range()])
            .map_err(|reason| PipelineError::ChunkFailed { chunk: ci, reason })?;
        if scene.cameras.len() != chunk.len {
            return Err(PipelineError::ChunkFailed {
                chunk: ci,
                reason: format!("{} camera(s) for a {}-frame chunk", scene.cameras.len(), chunk.len),
            });
        }
        if scene.weights.len() != scene.splats.len() {
            return Err(PipelineError::ChunkFailed {
                chunk: ci,
                reason: format!(
                    "{} fusion weight(s) for {} gaussian(s)",
                    scene.weights.len(),
                    scene.splats.len()
                ),
            });
        }

        // Which of this chunk's frames already have a world pose. For a
        // forward plan that is its leading overlap, but taking whatever is
        // actually placed means a plan that reaches further back registers
        // against all of it.
        let shared: Vec<usize> = chunk.range().filter(|&f| placed[f].is_some()).collect();

        let (world_splats, world_cams, report) = if ci == 0 {
            let report = ChunkReport {
                chunk: ci,
                frames: chunk.len,
                shared: 0,
                residual: 0.0,
                span: camera_span(&scene.cameras.iter().map(c2w64).collect::<Vec<_>>()),
                relative: 0.0,
                scale: 1.0,
                gaussians: scene.splats.len(),
            };
            (scene.splats, scene.cameras, report)
        } else {
            if shared.len() < MIN_OVERLAP {
                return Err(PipelineError::NotEnoughSharedFrames { chunk: ci, shared: shared.len() });
            }
            let a: Vec<[f64; 16]> =
                shared.iter().map(|&f| c2w64(placed[f].as_ref().expect("placed"))).collect();
            let b: Vec<[f64; 16]> =
                shared.iter().map(|&f| c2w64(&scene.cameras[f - chunk.start])).collect();
            let span = camera_span(&a);
            let m = sim3_from_cameras(&a, &b).ok_or(PipelineError::RegistrationFailed {
                chunk: ci,
                shared: shared.len(),
                residual: f64::INFINITY,
                span,
                relative: f64::INFINITY,
                tolerance: opts.max_relative_residual,
            })?;
            let residual = camera_residual(&a, &b, &m);
            // Unitless, because the world has no unit. A span of zero means
            // every shared camera sits in one place, which fixes no scale
            // however many of them there are.
            let relative = if span > 0.0 { residual / span } else { f64::INFINITY };
            // NaN is a failure too: a residual that is not a number means the
            // solve produced nothing comparable, which is not a pass.
            if relative > opts.max_relative_residual || relative.is_nan() {
                return Err(PipelineError::RegistrationFailed {
                    chunk: ci,
                    shared: shared.len(),
                    residual,
                    span,
                    relative,
                    tolerance: opts.max_relative_residual,
                });
            }
            let cams: Vec<Camera> =
                scene.cameras.iter().map(|c| moved(c, &transform_c2w_sim3(&c2w64(c), &m))).collect();
            let report = ChunkReport {
                chunk: ci,
                frames: chunk.len,
                shared: shared.len(),
                residual,
                span,
                relative,
                scale: m.s,
                gaussians: scene.splats.len(),
            };
            (apply_sim3(&scene.splats, &m), cams, report)
        };

        // Fuse this chunk's own duplicates NOW, in world units, before it
        // joins the pile. A dense model emits about one gaussian per source
        // pixel, so a long capture's raw concatenation is the thing that runs
        // the host out of memory - fusing per chunk holds the peak at one
        // chunk plus the fused scene instead of the whole capture raw.
        let (part, part_w) = if opts.voxel > 0.0 {
            let fused = splat::prune::voxel_merge(&world_splats, &scene.weights, opts.voxel, 0);
            let w = fused.opacities.clone();
            (fused, w)
        } else {
            (world_splats, scene.weights.clone())
        };
        weights.extend_from_slice(&part_w);
        parts.push(part);

        for (k, cam) in world_cams.into_iter().enumerate() {
            // First placement wins: a frame's pose belongs to the chunk that
            // put it in the world, and re-placing it would move the anchor
            // every later chunk registers against.
            placed[chunk.start + k].get_or_insert(cam);
        }
        reports.push(report);
    }

    let mut splats = splat::align::concat(&parts);
    if opts.voxel > 0.0 {
        // Now across chunks: the overlap frames were reconstructed twice, so
        // every seam carries two copies of the same surface.
        splats = splat::prune::voxel_merge(&splats, &weights, opts.voxel, opts.max_points);
    }
    if opts.max_scale_quantile > 0.0 && opts.max_scale_quantile < 1.0 {
        splats = splat::prune::drop_largest_scales(&splats, opts.max_scale_quantile);
    }
    let mut cameras: Vec<Camera> = placed.into_iter().map(|c| c.expect("every frame placed")).collect();
    if opts.orient && cameras.len() >= 3 {
        let mats: Vec<[f64; 16]> = cameras.iter().map(c2w64).collect();
        let (r, centre) = splat::orient::frame_from_cameras(&mats, -1.0);
        cameras = cameras
            .iter()
            .zip(&mats)
            .map(|(c, m)| moved(c, &splat::orient::transform_c2w(m, &r, &centre)))
            .collect();
        splats = splat::orient::apply(&splats, &r, &centre);
    }
    Ok(Reconstruction { splats, cameras, reports })
}

// ------------------------------------------------------------- the whole run

/// Everything the caller chose, in one place.
#[derive(Clone, Debug)]
pub struct PipelineOpts {
    pub select: SelectOpts,
    /// Frames per pass. `None` asks the model
    /// ([`ReconstructionModel::frame_budget`]), which is the right answer
    /// unless the caller knows something about this machine that the model
    /// does not.
    pub budget: Option<usize>,
    /// Frames two consecutive chunks share. More overlap costs passes and
    /// buys registration accuracy.
    pub overlap: usize,
    pub scene: ReconstructOpts,
}

impl Default for PipelineOpts {
    fn default() -> PipelineOpts {
        PipelineOpts { select: SelectOpts::default(), budget: None, overlap: 4, scene: ReconstructOpts::default() }
    }
}

/// A whole capture, from sources to one scene.
#[derive(Clone)]
pub struct Capture {
    /// How many frames were ingested, before selection.
    pub ingested: usize,
    pub selection: Selection,
    /// The grid the model said it would run these frames at.
    pub grid: (u32, u32),
    /// Frames per pass, as planned.
    pub budget: usize,
    pub plan: ChunkPlan,
    pub scene: Reconstruction,
    /// The frames that were actually reconstructed, normalised, in order.
    pub frames: Vec<Frame>,
}

/// Ingest, select, normalise, plan, reconstruct, finish.
pub fn run(
    sources: &[Source],
    opts: &PipelineOpts,
    model: &mut impl ReconstructionModel,
) -> Result<Capture, PipelineError> {
    let all = ingest(sources)?;
    let selection = select(&all, &opts.select);
    let picked = selection.frames(&all);
    // Ask the model what shape it will run these at, then make the capture
    // that shape. Both halves are the model's rule, not this crate's.
    let (mw, mh) = majority_size(&picked);
    let grid = model.input_grid(mw, mh);
    let frames = normalize(&picked, grid);
    let budget = opts.budget.unwrap_or_else(|| model.frame_budget(grid));
    let plan = plan_chunks(frames.len(), budget, opts.overlap)?;
    let scene = reconstruct_plan(&frames, &plan, model, &opts.scene)?;
    Ok(Capture { ingested: all.len(), selection, grid, budget, plan, scene, frames })
}
