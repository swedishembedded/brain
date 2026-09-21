// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2 as a chunk of a longer capture.
//!
//! `brain-recon` orchestrates a capture of any length: it ingests, selects,
//! plans overlapping windows, registers each window onto the world the earlier
//! ones built, and refuses a window that does not fit. It knows nothing about
//! any model. This file is the other side of that seam - everything that IS
//! specific to this one:
//!
//! * the **input grid**. The reference preprocessing caps the longest edge and
//!   floors both sides to the 14-pixel patch grid the ViT tokenizes on. That
//!   rule belongs here, and the pipeline asks for it rather than reproducing it.
//! * the **frame budget**. This trunk alternates per-frame and GLOBAL
//!   attention, so one pass attends across every patch of every frame it
//!   holds: the cost grows quadratically in the frames, and the frames one
//!   pass can afford therefore fall as the frames get bigger. That is a fact
//!   about this architecture, not about reconstruction, so the pipeline asks
//!   instead of assuming it.
//! * the **pass** itself: preprocess, forward, assemble.
//!
//! The weights are loaded once and every chunk runs on the same resident
//! model. That is the whole reason [`MirrorRecon`] borrows a `Mirror` instead
//! of taking a path: on a long capture the checkpoint is the expensive part,
//! and re-importing it per chunk would eat what the chunking saved.
//!
//! Swedish Embedded AB implements feed-forward 3D reconstruction that survives
//! contact with real captures - long ones, mixed sources, and the registration
//! gates that keep the result one scene. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use recon::{ChunkScene, Frame, ReconstructionModel};

use crate::config::MirrorConfig;
use crate::gaussians::{assemble, AssembleOpts};
use crate::model::{largest_binding_bytes, Mirror};
use crate::preprocess;

/// Patch tokens one pass may hold, across ALL its frames.
///
/// Global attention makes a pass's cost a function of this product and not of
/// the frame count alone, so the budget is expressed in tokens and the frame
/// count falls out of it. The value is a wall-clock choice rather than a
/// correctness one: below it a pass is something a person waits through, above
/// it a pass is something a person schedules.
const MAX_PASS_PATCHES: usize = 14336;

/// Never plan a pass narrower than this. A chunk has to carry a real spread of
/// viewpoints AND leave room for the overlap that registers it to its
/// neighbour; below this the plan is mostly overlap and the reconstruction is
/// mostly seams.
const MIN_PASS_FRAMES: usize = 6;

/// Never plan a wider one either, whatever the arithmetic says: past this the
/// pass stops being the thing that fails first and the host-side assembly
/// does.
const MAX_PASS_FRAMES: usize = 24;

/// The pixel grid this checkpoint runs a `width x height` source frame at.
///
/// `cap` is the longest-edge cap the caller chose (the reference's inference
/// default, not the checkpoint's native grid - the normalized 2D RoPE is what
/// lets the two differ). Both returned sides are multiples of `cfg.patch`,
/// because the ViT tokenizes on that grid and a partial patch has no row.
pub fn input_grid(cfg: &MirrorConfig, cap: usize, width: u32, height: u32) -> (u32, u32) {
    let (w, h) = (width as usize, height as usize);
    let target = preprocess::adaptive_target(w, h, cap, cfg.patch);
    let (nw, nh) = preprocess::resize_dims(w, h, target, cfg.patch);
    (nw.min(target) as u32, nh.min(target) as u32)
}

/// How many frames one pass may hold at `grid`.
///
/// Two ceilings, and the lower wins. The first is the cost curve above. The
/// second is the device: `largest_binding_bytes` is what the widest single
/// storage binding of a pass costs, and a pass that exceeds the adapter's
/// limit does not run slowly, it does not run. `storage_limit = 0` means the
/// caller does not know the device, and only the cost ceiling applies.
pub fn frame_budget(cfg: &MirrorConfig, grid: (u32, u32), storage_limit: u64) -> usize {
    let (wp, hp) = (grid.0 as usize / cfg.patch, grid.1 as usize / cfg.patch);
    let per_frame = (wp * hp).max(1);
    let mut budget = (MAX_PASS_PATCHES / per_frame).clamp(MIN_PASS_FRAMES, MAX_PASS_FRAMES);
    if storage_limit > 0 {
        while budget > MIN_PASS_FRAMES && largest_binding_bytes(cfg, budget, hp, wp) > storage_limit {
            budget -= 1;
        }
    }
    budget
}

/// One resident WorldMirror-2, presented to `brain-recon` as a reconstruction
/// model.
pub struct MirrorRecon<'a> {
    model: &'a mut Mirror,
    cfg: MirrorConfig,
    /// Longest-edge cap for preprocessing.
    cap: usize,
    assemble: AssembleOpts,
}

impl<'a> MirrorRecon<'a> {
    pub fn new(model: &'a mut Mirror, cfg: MirrorConfig, cap: usize, assemble: AssembleOpts) -> MirrorRecon<'a> {
        MirrorRecon { model, cfg, cap, assemble }
    }

    /// The grid a chunk of these frames will run at.
    pub fn grid_for(&self, frames: &[Frame]) -> (u32, u32) {
        let (w, h) = recon::majority_size(frames);
        input_grid(&self.cfg, self.cap, w, h)
    }

    /// Frames as the trunk takes them: `[0,1]` CHW, one grid for the batch.
    fn to_chw(&self, frames: &[Frame]) -> Result<(Vec<f32>, usize, usize, usize), String> {
        if frames.is_empty() {
            return Err("no frames in this chunk".to_string());
        }
        let (gw, gh) = self.grid_for(frames);
        if gw == 0 || gh == 0 {
            return Err("frames are smaller than one patch".to_string());
        }
        let (gwu, ghu) = (gw as usize, gh as usize);
        let mut all = Vec::with_capacity(frames.len() * 3 * gwu * ghu);
        for f in frames {
            // The PIL-exact bicubic path, which is what this checkpoint's
            // parity is gated on - the reason resampling to the grid stays on
            // this side of the trait rather than being done by the pipeline.
            let img = if (f.image.w, f.image.h) == (gw, gh) {
                f.image.clone()
            } else {
                preprocess::resize_bicubic(&f.image, gwu, ghu)
            };
            for c in 0..3 {
                for y in 0..ghu {
                    for x in 0..gwu {
                        all.push(img.px[(y * gwu + x) * 3 + c] as f32 / 255.0);
                    }
                }
            }
        }
        Ok((all, frames.len(), ghu / self.cfg.patch, gwu / self.cfg.patch))
    }
}

impl ReconstructionModel for MirrorRecon<'_> {
    fn name(&self) -> &str {
        "worldmirror2"
    }

    fn input_grid(&self, width: u32, height: u32) -> (u32, u32) {
        input_grid(&self.cfg, self.cap, width, height)
    }

    fn frame_budget(&self, grid: (u32, u32)) -> usize {
        frame_budget(&self.cfg, grid, self.model.gpu().max_storage_binding_bytes())
    }

    fn reconstruct(&mut self, frames: &[Frame]) -> Result<ChunkScene, String> {
        let (chw, s, hp, wp) = self.to_chw(frames)?;
        let (w, h) = ((wp * self.cfg.patch) as u32, (hp * self.cfg.patch) as u32);
        let limit = self.model.gpu().max_storage_binding_bytes();
        let need = largest_binding_bytes(&self.cfg, s, hp, wp);
        if limit > 0 && need > limit {
            return Err(format!(
                "{s} frame(s) at {w}x{h} need {:.2} GiB in one storage binding against this \
                 device's {:.2} GiB limit",
                need as f64 / (1u64 << 30) as f64,
                limit as f64 / (1u64 << 30) as f64
            ));
        }
        self.model.forward(&chw, s, hp, wp);
        let (splats, cameras, weights) =
            assemble(self.model.gpu(), self.model, &chw, s, w, h, &self.assemble, None);
        Ok(ChunkScene { splats, cameras, weights })
    }
}
