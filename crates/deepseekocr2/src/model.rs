// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The composite: [`Resampler`] spliced into `deepseek2::model::DeepseekV2`,
//! the SAME unmodified decoder v1 already uses (M0's ledger).
//!
//! ```text
//! input_ids ──► token embedding ─────────────────────► res[0]
//!                                                          │  rows [row0, row0+n_rows)
//! SAM tokens, per tile/view ──► Resampler ──► gather_rows ─┘  overwritten
//!                                                          ▼
//!                                                   decoder blocks ──► lm_head
//! ```
//!
//! **Row layout is flat, not interleaved**: unlike v1 ([`crate`]'s sibling
//! crate `deepseek2ocr`), there is no `image_newline` here, so
//! [`crate::rows::row_plan`] and [`crate::encoder::gather_rows`]/
//! [`crate::encoder::scatter_rows`] are ordinary concatenation and splitting,
//! never an index gather. The tile-then-global-then-separator ORDER is a
//! design assumption pinned in `crate::rows`'s own doc, pending M6's
//! empirical check against a real forward - this module does not re-decide
//! it, it just consumes `RowPlan` so that a future change to the formula is
//! confined to `crate::rows::row_plan`.
//!
//! **SAM itself is still not invoked here** (`crate::encoder`'s own scope
//! note carries forward): `forward`'s `local_tiles`/`global` arguments are
//! each view's already-computed `[n_query, d_model]` token grid, exactly
//! what a real `sam1::SamEncoder` would hand back. Wiring a real SAM
//! instance and real image preprocessing in front of this composite is
//! later milestones' work (real-weight parity, M6).

use deepseek2::model::DeepseekV2;
use deepseek2::{DeepseekV2Config, IGNORE};

use crate::config::DeepseekOcr2VisionConfig;
use crate::encoder::{self, Resampler, TrainState};
use crate::rows::{row_plan, RowPlan, TileGrid};

/// Encoder + row-gather + decoder.
pub struct DeepseekOcr2 {
    vision_cfg: DeepseekOcr2VisionConfig,
    enc: Resampler,
    dec: DeepseekV2,
    plan: RowPlan,
    row0: u32,
}

impl DeepseekOcr2 {
    /// Build the composite for a `seq`-token sequence whose image block
    /// occupies rows `[row0, row0 + plan.len())`, where `plan` comes from
    /// `grid` and the vision config's own `n_query_local`/`n_query_global`.
    ///
    /// `gpu_vision`/`gpu_decoder` are independent `Gpu` handles, one built
    /// with [`crate::encoder::PIPELINES`] and one with `deepseek2::PIPELINES`,
    /// the same split v1's composite uses (`crates/deepseek2ocr/src/model.rs`'s
    /// header explains why: the splice crosses them as a host `Vec<f32>`, so
    /// the two towers need not share a device or a backend).
    #[allow(clippy::too_many_arguments)] // the composite genuinely has this many independent knobs
    pub fn new_on(
        gpu_vision: gpu_core::Gpu,
        gpu_decoder: gpu_core::Gpu,
        vision_cfg: DeepseekOcr2VisionConfig,
        decoder_cfg: DeepseekV2Config,
        vision_init: &dyn checkpoint::TensorSource,
        decoder_init: &dyn checkpoint::TensorSource,
        grid: TileGrid,
        seq: u32,
        row0: u32,
        train: bool,
    ) -> DeepseekOcr2 {
        vision_cfg.check();
        let plan = row_plan(grid, vision_cfg.encoder.n_query_local, vision_cfg.encoder.n_query_global);
        let n_rows = plan.len() as u32;
        assert!(row0 + n_rows <= seq, "the image block [{row0}, {}) does not fit a {seq}-token sequence", row0 + n_rows);
        assert_eq!(
            vision_cfg.decoder_hidden, decoder_cfg.shape.d_model,
            "the projector's output width must equal the decoder's own d_model"
        );

        let enc = Resampler::new_on(gpu_vision, vision_cfg.clone(), vision_init, train);
        let mut dec = DeepseekV2::new_on(gpu_decoder, decoder_cfg, 1, seq, decoder_init, train);
        dec.enable_mm_splice(row0, n_rows);
        DeepseekOcr2 { vision_cfg, enc, dec, plan, row0 }
    }

    pub fn encoder(&self) -> &Resampler {
        &self.enc
    }
    pub fn decoder(&self) -> &DeepseekV2 {
        &self.dec
    }
    /// `(row0, n_rows)` - the spliced image run.
    pub fn image_run(&self) -> (u32, u32) {
        (self.row0, self.plan.len() as u32)
    }
    pub fn row_plan(&self) -> &RowPlan {
        &self.plan
    }

    /// Set the decoder's input ids and next-token targets ([`IGNORE`] masks a
    /// position out of the loss).
    pub fn set_tokens(&self, ids: &[u32], targets: &[u32]) {
        self.dec.set_batch(ids, targets);
    }
    /// Convenience: no loss, every position masked - a forward-only run.
    pub fn set_tokens_unsupervised(&self, ids: &[u32]) {
        self.set_tokens(ids, &vec![IGNORE; ids.len()]);
    }

    /// Run every view's resampler, gather the spliced block, and run the
    /// decoder. `local_tiles` must be `self.row_plan().grid.tiles()` entries
    /// long, each `[n_query_local, d_model]`; `global` is
    /// `[n_query_global, d_model]`. Returns the masked cross-entropy loss and
    /// the per-view state [`Self::backward`] needs.
    pub fn forward(&self, local_tiles: &[Vec<f32>], global: &[f32]) -> (f32, CompositeState) {
        let e = &self.vision_cfg.encoder;
        assert_eq!(local_tiles.len() as u32, self.plan.grid.tiles(), "expected one SAM token grid per local tile");
        for t in local_tiles {
            assert_eq!(t.len(), (e.n_query_local * e.d_model) as usize, "a local tile's SAM grid is the wrong shape");
        }
        assert_eq!(global.len(), (e.n_query_global * e.d_model) as usize, "the global view's SAM grid is the wrong shape");

        let mut tile_states = Vec::with_capacity(local_tiles.len());
        let mut projected_tiles = Vec::with_capacity(local_tiles.len());
        for t in local_tiles {
            let (p, st) = self.enc.forward_train(t, true);
            projected_tiles.push(p);
            tile_states.push(st);
        }
        let (projected_global, global_state) = self.enc.forward_train(global, false);
        let separator = self.enc.read_weight("vision.view_separator");

        let block = encoder::gather_rows(&projected_tiles, &projected_global, &separator);
        assert_eq!(block.len(), self.plan.len() * self.vision_cfg.decoder_hidden as usize, "the gathered block does not match the row plan");

        self.dec.write_img_embeds(&block);
        let loss = self.dec.forward();
        (loss, CompositeState { tile_states, global_state })
    }

    /// Full backward of the loss [`Self::forward`] returned: the decoder's
    /// gradient reaches the spliced rows, [`crate::encoder::scatter_rows`]
    /// splits it back into each view's own gradient (and the separator's, by
    /// itself since it is read by exactly one row), and each view's
    /// gradient is walked back through the resampler. Returns the gradient
    /// w.r.t. each local tile's SAM grid and the global view's, in the same
    /// order [`Self::forward`] took them.
    pub fn backward(&self, st: CompositeState) -> (Vec<Vec<f32>>, Vec<f32>) {
        self.dec.backward();
        let d_block = self.dec.read_d_img_embeds();

        let e = &self.vision_cfg.encoder;
        let (d_tiles, d_global, d_separator) = encoder::scatter_rows(&d_block, st.tile_states.len(), e.n_query_local, e.n_query_global, self.vision_cfg.decoder_hidden);
        self.enc.write_view_separator_grad(&d_separator);

        let d_sam_tiles: Vec<Vec<f32>> = st.tile_states.into_iter().zip(d_tiles).map(|(state, d_proj)| self.enc.backward_train(state, &d_proj, true)).collect();
        let d_sam_global = self.enc.backward_train(st.global_state, &d_global, false);
        (d_sam_tiles, d_sam_global)
    }

    pub fn zero_grads(&self) {
        self.enc.zero_grads();
        self.dec.zero_grads();
    }

    /// `[seq, d_model]` - the decoder's residual stream after the splice.
    pub fn read_decoder_input(&self) -> Vec<f32> {
        self.dec.read_res(0)
    }
}

/// Everything [`DeepseekOcr2::forward`] must hand back to
/// [`DeepseekOcr2::backward`] - opaque outside this crate.
pub struct CompositeState {
    tile_states: Vec<TrainState>,
    global_state: TrainState,
}
