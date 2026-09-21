// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reconstructing a capture that does not fit in one forward pass.
//!
//! A feed-forward reconstruction model takes N frames and returns one scene.
//! N is bounded - by the model's cost curve, by the card - and a capture is
//! not: a folder of photographs, a walk through a building, or both together.
//! So a long capture is not one pass. It is a sequence of overlapping passes,
//! each returning a scene in its OWN world, registered onto each other
//! through the frames they share.
//!
//! This crate is that orchestration and nothing else:
//!
//! 1. [`ingest`] - photographs, a clip, or both, in one ordered frame set.
//! 2. [`select`] - which frames are worth a slot: sharp ones, and not two of
//!    the same viewpoint.
//! 3. [`plan_chunks`] - overlapping windows of the size the MODEL says it can
//!    take, guaranteeing every adjacent pair shares enough frames to be
//!    registered.
//! 4. [`reconstruct_plan`] - run each chunk, solve the similarity that puts it
//!    in the world the earlier chunks built, and REFUSE the ones that do not
//!    fit.
//! 5. the global finish - fuse duplicates, drop the runaway scales, and land
//!    the scene upright.
//!
//! ## It knows nothing about any model
//!
//! Everything model-shaped sits behind [`ReconstructionModel`]: the input grid
//! a model wants, how many frames it can hold in one pass, and the pass
//! itself. Neither of the first two is a universal truth - one model's
//! attention is global across frames and costs quadratically in how many it
//! holds, another takes two hundred happily - so the pipeline ASKS rather than
//! assumes. A special case for a particular model anywhere in this crate would
//! be a missing trait method.
//!
//! The consequence worth stating: the planner, the frame selection and the
//! accumulation all run with no weights and no GPU. The parts of a long
//! reconstruction that are easy to get silently wrong are exactly the parts
//! that cost nothing to run.
//!
//! ## The failure this crate exists to make loud
//!
//! Two chunks are joined through the frames they share, each of which has a
//! pose in both worlds. If those poses really do describe the same path, one
//! similarity explains all of them and `splat::align::camera_residual` is
//! small. If they do not - the capture jumped, the overlap was textureless,
//! one chunk's cameras drifted - a similarity is still SOLVED, it just does
//! not fit, and concatenating anyway gives a scene carrying two copies of the
//! same wall slightly apart. Nothing downstream can tell that from a wall that
//! is genuinely there twice. So the residual is measured, reported per chunk,
//! and a bad one is [`PipelineError::RegistrationFailed`] rather than a scene.
//!
//! Swedish Embedded AB implements 3D reconstruction pipelines that scale past
//! one model's context window - chunk planning, registration and the quality
//! gates that keep a long capture honest. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::fmt;

use splat::types::{Camera, Splats};

pub mod ingest;
pub mod pipeline;
pub mod plan;
pub mod select;

pub use ingest::{ingest, Frame, Source};
pub use pipeline::{
    majority_size, normalize, reconstruct_plan, run, Capture, ChunkReport, PipelineOpts, Reconstruction,
    ReconstructOpts,
};
pub use plan::{plan_chunks, Chunk, ChunkPlan, MIN_OVERLAP};
pub use select::{select, sharpness, viewpoint_change, Dropped, SelectOpts, Selection};

/// What one pass produced, in ITS OWN world frame.
///
/// A feed-forward model anchors its world to the frames it was given - the
/// first camera is the origin, and the scale comes from whatever that pass's
/// content normalised to - so two passes over one capture come back in two
/// different worlds. Putting them in one world is [`reconstruct_plan`]'s job.
#[derive(Clone, Default)]
pub struct ChunkScene {
    pub splats: Splats,
    /// Optional per-frame depth prior, in the chunk's own units and aligned
    /// with `cameras`. Empty when the model has nothing to say.
    ///
    /// This is the one thing a reconstruction knows that a photograph does
    /// not. A splat's image-plane gradient is orthogonal to its own viewing
    /// ray, so an RGB loss is structurally blind to distance: measured against
    /// analytic ground truth, a scene slid 5.7% along every ray while still
    /// rendering correctly was left at 6.21% error by an RGB-only fit and
    /// pulled back to 0.08% once the same depth was supervised. Handing the
    /// prior forward is what lets a fit see the axis it otherwise cannot.
    pub depth: Vec<DepthPrior>,
    /// One camera per frame of the chunk, in the chunk's own frame, in the
    /// order the frames were handed over.
    pub cameras: Vec<Camera>,
    /// Per-gaussian fusion weight - the evidence behind each gaussian, used
    /// when duplicates are merged. A model that has nothing better to say here
    /// should return its opacities.
    pub weights: Vec<f32>,
}

/// One frame's depth prior and how far each pixel of it is to be trusted.
///
/// The confidence is not a volume knob. A fit's step is normalised per
/// parameter, so scaling every pixel's trust by a constant changes nothing;
/// what it expresses is which pixels deserve more weight than which OTHERS. A
/// multi-view agreement count is exactly that signal, which is why the model
/// that has one should put it here rather than a flat 1.0.
#[derive(Clone, Debug, Default)]
pub struct DepthPrior {
    /// `w * h`, in the chunk's own units. 0 means "no prior for this pixel".
    pub depth: Vec<f32>,
    /// `w * h`, relative weight per pixel. Empty means "trust all of it
    /// equally", which is weaker than it sounds - see the type doc.
    pub conf: Vec<f32>,
}

/// What a pipeline needs from a reconstruction model. Implemented by the model
/// crate, never by this one.
///
/// The two questions that are NOT the pass itself are the ones that keep
/// model knowledge out of the pipeline, and they are asked for a reason:
///
/// * [`input_grid`](ReconstructionModel::input_grid) - a model has its own
///   preprocessing rule (a cap on the longest edge, a patch size the grid must
///   divide by, a square input). The pipeline must never reimplement it; it
///   asks, and uses the answer to normalise a MIXED capture onto one shape.
/// * [`frame_budget`](ReconstructionModel::frame_budget) - how many frames one
///   pass may hold. This is a property of the model's own cost curve at that
///   grid, which is why it takes the grid: the frames a model affords shrink
///   as the frames get bigger.
pub trait ReconstructionModel {
    /// For reports and errors.
    fn name(&self) -> &str;

    /// The pixel grid this model will run a `width x height` source frame at.
    ///
    /// The pipeline uses it for the frame budget and to bring a mixed capture
    /// onto ONE shape; the model still does its own resampling inside
    /// [`reconstruct`](ReconstructionModel::reconstruct), so a model whose
    /// resampler is part of its reference parity keeps that parity.
    fn input_grid(&self, width: u32, height: u32) -> (u32, u32);

    /// How many frames one pass may hold at `grid`.
    fn frame_budget(&self, grid: (u32, u32)) -> usize;

    /// One pass. Every frame handed over has the same pixel size.
    fn reconstruct(&mut self, frames: &[Frame]) -> Result<ChunkScene, String>;
}

/// So a caller that resolved its model BY NAME - holding a
/// `Box<dyn ReconstructionModel>` because the name is only known at runtime -
/// can hand it straight to [`run`] and [`reconstruct_plan`], which take
/// `impl ReconstructionModel`. The trait is object safe; this is what closes
/// the gap between that and the generic bound.
impl<T: ReconstructionModel + ?Sized> ReconstructionModel for &mut T {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn input_grid(&self, width: u32, height: u32) -> (u32, u32) {
        (**self).input_grid(width, height)
    }
    fn frame_budget(&self, grid: (u32, u32)) -> usize {
        (**self).frame_budget(grid)
    }
    fn reconstruct(&mut self, frames: &[Frame]) -> Result<ChunkScene, String> {
        (**self).reconstruct(frames)
    }
}

/// Everything this pipeline refuses to do. Each variant names the chunk or the
/// parameter at fault, because the alternative - carrying on - produces
/// something that looks like a reconstruction and is not one.
#[derive(Clone, Debug)]
pub enum PipelineError {
    /// No frames at all reached the planner.
    NoFrames,
    /// A source could not be read.
    Ingest { source: String, reason: String },
    /// The requested overlap cannot register two chunks.
    OverlapTooSmall { overlap: usize },
    /// The chunk budget leaves no room to advance past the overlap.
    ChunkTooSmall { budget: usize, overlap: usize },
    /// The plan and the frame set disagree.
    PlanMismatch { frames: usize, planned: usize },
    /// The model could not produce this chunk.
    ChunkFailed { chunk: usize, reason: String },
    /// This chunk shares too few placed frames with everything before it.
    NotEnoughSharedFrames { chunk: usize, shared: usize },
    /// The chunk was solved onto the world and does not fit there.
    RegistrationFailed {
        chunk: usize,
        shared: usize,
        /// Worst shared-camera miss, in world units.
        residual: f64,
        /// How far apart the shared cameras sit, in world units.
        span: f64,
        /// `residual / span` - the unitless form, which is what is gated.
        relative: f64,
        tolerance: f64,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::NoFrames => write!(f, "no frames to reconstruct"),
            PipelineError::Ingest { source, reason } => write!(f, "cannot read {source}: {reason}"),
            PipelineError::OverlapTooSmall { overlap } => write!(
                f,
                "an overlap of {overlap} frame(s) cannot register two chunks: at least {MIN_OVERLAP} \
                 shared cameras are needed, since a single one fixes no scale"
            ),
            PipelineError::ChunkTooSmall { budget, overlap } => write!(
                f,
                "a chunk budget of {budget} frame(s) with an overlap of {overlap} never advances; \
                 the budget must exceed the overlap"
            ),
            PipelineError::PlanMismatch { frames, planned } => {
                write!(f, "the plan covers {planned} frame(s) but {frames} were handed over")
            }
            PipelineError::ChunkFailed { chunk, reason } => {
                write!(f, "chunk {chunk} could not be reconstructed: {reason}")
            }
            PipelineError::NotEnoughSharedFrames { chunk, shared } => write!(
                f,
                "chunk {chunk} shares {shared} already-placed frame(s) with the scene, below the \
                 {MIN_OVERLAP} a similarity can be fitted to"
            ),
            PipelineError::RegistrationFailed { chunk, shared, residual, span, relative, tolerance } => write!(
                f,
                "chunk {chunk} did not register: over its {shared} shared camera(s) the best \
                 similarity still misses by {residual:.4} against a shared baseline of {span:.4} \
                 ({relative:.3} of it, tolerance {tolerance:.3}). Joining it anyway would put a \
                 second, displaced copy of that overlap into the scene. Give the chunks more \
                 overlap, or re-capture the join with more texture and less camera motion."
            ),
        }
    }
}

impl std::error::Error for PipelineError {}
