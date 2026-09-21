// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cutting a capture into the passes a model can actually run.
//!
//! The plan is the contract between what the model affords and what the
//! registration needs. Every frame must be reconstructed by some chunk, or
//! part of the capture is simply missing from the scene; and every adjacent
//! pair of chunks must share enough frames to be registered, or the second
//! one has nothing to be placed by.

use crate::PipelineError;

/// The fewest shared frames a similarity can be fitted to. One shared camera
/// gives a position and a heading but no second point, so nothing fixes the
/// relative SCALE of the two worlds, and they are free to sit at any size
/// relative to each other. `splat::align::sim3_from_cameras` returns `None`
/// below this.
pub const MIN_OVERLAP: usize = 2;

/// One pass: a contiguous window of the selected frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: usize,
    pub len: usize,
}

impl Chunk {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.end()
    }
}

/// How a capture is cut into passes.
#[derive(Clone, Debug)]
pub struct ChunkPlan {
    pub chunks: Vec<Chunk>,
    /// The overlap that was asked for. Adjacent chunks share AT LEAST this
    /// many frames; the last pair usually shares more, because the final
    /// window is pulled back to end on the last frame.
    pub overlap: usize,
    /// How many frames the plan covers.
    pub frames: usize,
}

impl ChunkPlan {
    /// The frames chunk `i` shares with chunk `i - 1`.
    pub fn shared(&self, i: usize) -> std::ops::Range<usize> {
        if i == 0 || i >= self.chunks.len() {
            return 0..0;
        }
        let (prev, cur) = (self.chunks[i - 1], self.chunks[i]);
        cur.start..prev.end().min(cur.end())
    }
}

/// Cut `frames` into overlapping windows of at most `budget` frames, each
/// sharing at least `overlap` frames with the one before it.
///
/// `budget` comes from the model
/// ([`ReconstructionModel::frame_budget`](crate::ReconstructionModel::frame_budget)),
/// because it is a property of that model's cost curve at that frame size and
/// of nothing here.
///
/// The last window is pulled BACK to end on the final frame rather than being
/// allowed to run short. A short tail chunk is the worst of both: it costs a
/// whole pass and reconstructs the end of the capture from fewer views than
/// anywhere else in it.
pub fn plan_chunks(frames: usize, budget: usize, overlap: usize) -> Result<ChunkPlan, PipelineError> {
    if frames == 0 {
        return Err(PipelineError::NoFrames);
    }
    if overlap < MIN_OVERLAP {
        return Err(PipelineError::OverlapTooSmall { overlap });
    }
    if budget <= overlap {
        return Err(PipelineError::ChunkTooSmall { budget, overlap });
    }
    if frames <= budget {
        return Ok(ChunkPlan { chunks: vec![Chunk { start: 0, len: frames }], overlap, frames });
    }
    let stride = budget - overlap;
    let last = frames - budget;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    loop {
        let s = start.min(last);
        chunks.push(Chunk { start: s, len: budget });
        if s == last {
            break;
        }
        start = s + stride;
    }
    Ok(ChunkPlan { chunks, overlap, frames })
}
