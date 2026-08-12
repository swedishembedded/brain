// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The composite: [`crate::DeepEncoder`] spliced into `deepseekv2::DeepseekV2`.
//!
//! ```text
//! input_ids ──► token embedding ──► res[0]
//!                                     │  rows [row0, row0 + image_tokens)
//! image ──► DeepEncoder ──────────────┘  overwritten by the projector output
//!                                        (`model::vlm::splice_fwd`)
//!                                     ▼
//!                              12 decoder blocks ──► RMSNorm ──► lm_head
//! ```
//!
//! **Two ways to fill that run**, and both ship:
//!
//! * [`DeepseekOcr::new`] / [`DeepseekOcr::new_split`] size the splice at
//!   `cfg.image_tokens()` and write the projector's output straight into it -
//!   256 contiguous rows at the real geometry. That is the checkpoint-free
//!   golden fixture's own scope (its reference dump has no newline/separator
//!   rows at all) and every parity number in this crate was measured on it.
//! * [`DeepseekOcr::new_with_prompt`] sizes the splice at the REAL layout's
//!   `n_rows` - 273 - and fills it through [`crate::layout::RowGather`], which
//!   interleaves the mmproj's `vision.image_newline` and `vision.view_separator`
//!   rows between the projector's. That is what the reference model feeds its
//!   decoder, and what [`crate::prompt::build_prompt`]'s ids describe.
//!
//! The two coexist: the first is the parity path, the second is the production
//! one. Nothing about the encoder, the projector or the decoder differs between
//! them - only which rows of the spliced block carry which vector.
//!
//! **Still one contiguous DECODER run** in both cases: the splice seam takes
//! exactly one `(row0, n_rows)`. A multi-view (Base/Gundam) layout is several
//! decoder runs and needs one splice call per run; `RowGather` itself is already
//! indifferent to that, `deepseekv2::DeepseekV2::enable_mm_splice` is not.
//!
//! **The backward is real and end to end**: the decoder's cross-entropy gradient
//! reaches the placeholder rows, the splice moves them into the encoder's
//! `d_out` and ZEROES them in `dres[0]` (so the placeholder token's embedding row
//! is never trained on the image), and the encoder walks them back through the
//! projector, the concat, CLIP's injected-token seam and the whole SAM tower to
//! the input pixels.

use deepseekv2::model::DeepseekV2;
use deepseekv2::IGNORE;

use crate::config::DeepseekOcrConfig;
use crate::encoder::DeepEncoder;
use crate::layout::RowGather;
use crate::prompt::Prompt;
use crate::rows::Src;
use crate::DeviceFactory;

/// Encoder + splice + decoder.
pub struct DeepseekOcr {
    pub cfg: DeepseekOcrConfig,
    enc: DeepEncoder,
    dec: DeepseekV2,
    /// `Some` on the real-layout path ([`DeepseekOcr::new_with_prompt`]);
    /// `None` on the contiguous parity path. This is the ONLY thing that
    /// differs between the two - see this module's header.
    rows: Option<RowGather>,
    row0: u32,
    n_rows: u32,
    seq: u32,
}

impl DeepseekOcr {
    /// Build the composite for a `seq`-token sequence whose image placeholders
    /// occupy rows `[row0, row0 + image_tokens)`.
    ///
    /// `init` is ONE source covering the SAM, CLIP, glue and decoder manifests;
    /// the four name spaces are disjoint by construction. It is a
    /// [`checkpoint::TensorSource`], so an eager `&HashMap<String, Vec<f32>>`
    /// coerces (the tiny fixture's shape) and a streaming reader works too.
    /// When the encoder and the decoder live in **different files** - which is
    /// what the shipped checkpoint is, an mmproj plus an LM GGUF - use
    /// [`Self::new_split`] rather than merging them into one map.
    ///
    /// See [`Self::new_split`] for what `train` decides.
    pub fn new(
        dev: DeviceFactory<'_>,
        cfg: DeepseekOcrConfig,
        init: &dyn checkpoint::TensorSource,
        seed: u64,
        seq: u32,
        row0: u32,
        train: bool,
    ) -> DeepseekOcr {
        DeepseekOcr::new_split(dev, cfg, init, init, seed, seq, row0, train)
    }

    /// [`Self::new`] with the encoder's and the decoder's weights coming from
    /// two independent sources - the shipped checkpoint's own shape (a 448 MB
    /// mmproj for the three vision stages, a 3.1 GB LM GGUF for the decoder).
    ///
    /// `train` is threaded, unchanged, into `DeepEncoder::new` and
    /// `DeepseekV2::new_on`. It decides two things at once, and at real scale
    /// both are measured in gigabytes: whether every parameter is
    /// `Role::Trainable` (weight + gradient + two AdamW moments, ~4x) or
    /// `Role::Frozen` (weight only), and whether the backward scratch and the
    /// reverse tape are built at all. A `train = false` composite cannot run
    /// [`Self::backward`]; that is the point of it.
    #[allow(clippy::too_many_arguments)] // the composite genuinely has this many independent knobs
    pub fn new_split(
        dev: DeviceFactory<'_>,
        cfg: DeepseekOcrConfig,
        vision: &dyn checkpoint::TensorSource,
        decoder: &dyn checkpoint::TensorSource,
        seed: u64,
        seq: u32,
        row0: u32,
        train: bool,
    ) -> DeepseekOcr {
        let n_rows = cfg.image_tokens();
        DeepseekOcr::build(dev, cfg, vision, decoder, seed, seq, row0, n_rows, None, train)
    }

    /// The **real-layout** composite: the splice is sized and filled from a
    /// [`Prompt`], not from `cfg.image_tokens()`.
    ///
    /// `prompt` is [`crate::prompt::build_prompt`]'s output. Its `n_rows` is the
    /// whole image block - 273 at the real geometry, the 256 projector rows with
    /// 16 `image_newline` rows and one `view_separator` interleaved - and its
    /// `plan.rows` says which row is which. The decoder's `enable_mm_splice` is
    /// sized at that `n_rows`, and every forward assembles the block through
    /// [`crate::layout::RowGather`] instead of writing the projector output
    /// straight in.
    ///
    /// `seq` still sizes the decoder and must leave room for whatever
    /// [`Self::generate_greedy`] will append: `prompt.len() + n_new <= seq`.
    /// Everything else - `train`, the two weight sources, the backward - is
    /// [`Self::new_split`]'s contract unchanged.
    #[allow(clippy::too_many_arguments)] // same knobs as new_split, plus the prompt
    pub fn new_with_prompt(
        dev: DeviceFactory<'_>,
        cfg: DeepseekOcrConfig,
        vision: &dyn checkpoint::TensorSource,
        decoder: &dyn checkpoint::TensorSource,
        seed: u64,
        seq: u32,
        prompt: &Prompt,
        train: bool,
    ) -> DeepseekOcr {
        assert_eq!(
            prompt.n_rows as usize,
            prompt.plan.rows.len(),
            "the prompt's image block and its row plan disagree ({} ids, {} plan rows)",
            prompt.n_rows,
            prompt.plan.rows.len()
        );
        assert!(prompt.len() <= seq as usize, "the prompt's {} ids do not fit a {seq}-token sequence", prompt.len());
        let (row0, n_rows) = prompt.image_run();
        DeepseekOcr::build(dev, cfg, vision, decoder, seed, seq, row0, n_rows, Some(&prompt.plan.rows), train)
    }

    /// The one constructor both public ones funnel through. `layout` decides
    /// which of the two fill paths this composite uses; everything else is
    /// identical, which is what keeps the parity path bit-identical.
    #[allow(clippy::too_many_arguments)]
    fn build(
        dev: DeviceFactory<'_>,
        cfg: DeepseekOcrConfig,
        vision: &dyn checkpoint::TensorSource,
        decoder: &dyn checkpoint::TensorSource,
        seed: u64,
        seq: u32,
        row0: u32,
        n_rows: u32,
        layout: Option<&[Src]>,
        train: bool,
    ) -> DeepseekOcr {
        cfg.check();
        assert!(row0 + n_rows <= seq, "image rows [{row0}, {}) do not fit a {seq}-token sequence", row0 + n_rows);

        let enc = DeepEncoder::new(dev, cfg.clone(), vision, seed, train);
        let rows = layout.map(|l| enc.row_gather(l));
        if let Some(rg) = &rows {
            assert_eq!(rg.rows(), n_rows, "the layout is {} rows but the splice was sized at {n_rows}", rg.rows());
        }
        let mut dec = DeepseekV2::new_on(dev(deepseekv2::PIPELINES), cfg.decoder.clone(), 1, seq, decoder, train);
        dec.enable_mm_splice(row0, n_rows);
        DeepseekOcr { cfg, enc, dec, rows, row0, n_rows, seq }
    }

    pub fn encoder(&self) -> &DeepEncoder {
        &self.enc
    }
    pub fn decoder(&self) -> &DeepseekV2 {
        &self.dec
    }
    /// `(row0, n_rows)` -- the spliced image run.
    pub fn image_run(&self) -> (u32, u32) {
        (self.row0, self.n_rows)
    }
    /// The row layout this composite splices, or `None` on the contiguous
    /// parity path (where the block IS the projector's output, in order).
    pub fn row_gather(&self) -> Option<&RowGather> {
        self.rows.as_ref()
    }

    /// Encode `image` and return the block that gets spliced --
    /// `[n_rows, d_model]`. The projector's output verbatim on the contiguous
    /// path; the interleaved layout on the real one.
    fn encode_block(&self, image: &[f32]) -> Vec<f32> {
        match &self.rows {
            Some(rg) => self.enc.forward_rows(image, rg),
            None => self.enc.forward(image),
        }
    }

    /// Set the decoder's input ids and next-token targets ([`IGNORE`] masks a
    /// position out of the loss). A parity run masks everything; a training or
    /// descent-smoke run does not.
    pub fn set_tokens(&self, ids: &[u32], targets: &[u32]) {
        assert_eq!(ids.len(), self.seq as usize, "ids must be one sequence of {} tokens", self.seq);
        assert_eq!(targets.len(), ids.len());
        self.dec.set_batch(ids, targets);
    }

    /// Convenience: no loss, every position masked -- what a forward-parity run
    /// wants.
    pub fn set_tokens_unsupervised(&self, ids: &[u32]) {
        self.set_tokens(ids, &vec![IGNORE; ids.len()]);
    }

    /// Encode `image` (`[3, image_h, image_w]` NCHW), splice, and run the
    /// decoder. Returns the masked cross-entropy loss (0 when every target is
    /// [`IGNORE`]).
    pub fn forward(&self, image: &[f32]) -> f32 {
        let embeds = self.encode_block(image);
        self.dec.write_img_embeds(&embeds);
        self.dec.forward()
    }

    /// Full backward of the loss [`Self::forward`] returned. Returns the
    /// gradient w.r.t. the input image.
    ///
    /// On the real-layout path the block's gradient is de-interleaved first:
    /// projector rows go on through the encoder, and the newline/separator rows
    /// are summed onto `vision.image_newline` / `vision.view_separator` - a
    /// shared row's gradient is the sum over every row that read it.
    pub fn backward(&self) -> Vec<f32> {
        self.dec.backward();
        self.dec.poll_wait();
        let d_img = self.dec.read_d_img_embeds();
        match &self.rows {
            Some(rg) => self.enc.backward_rows(&d_img, rg),
            None => self.enc.backward(&d_img),
        }
    }

    /// Encode `image`, splice it, then **greedily decode** `n_new` tokens
    /// continuing `prompt_ids`, returned as one sequence (prompt included).
    ///
    /// `prompt_ids` is the FULL leading sequence, image placeholder rows and
    /// all - the splice overwrites rows `[row0, row0 + image_tokens)` of the
    /// residual stream whatever ids sit there, so those positions' ids are
    /// arbitrary but must be present for the run to be inside the sequence.
    ///
    /// The image is encoded ONCE and its projected tokens stay in the decoder's
    /// splice buffer, so the `n_new` recomputed forwards re-splice the same
    /// embedding rather than re-running the 400 M-parameter DeepEncoder per
    /// step. Everything else is `DeepseekV2::generate_greedy`'s contract,
    /// including the `O(T²)` recompute and the sized-context requirement.
    ///
    /// Returns everything at once; [`Self::generate_greedy_cb`] is the same
    /// run with a per-token callback, for a caller that must observe tokens as
    /// they are produced.
    pub fn generate_greedy(&self, image: &[f32], prompt_ids: &[u32], n_new: u32) -> Vec<u32> {
        self.generate_greedy_cb(image, prompt_ids, n_new, |_| {})
    }

    /// [`Self::generate_greedy`] with a per-token callback - the seam a served
    /// path emits REAL streaming deltas from, forwarded straight to
    /// [`DeepseekV2::generate_greedy_cb`], so it fires once per generated token
    /// and exactly `n_new` times.
    ///
    /// The encode/splice half runs ONCE before the first callback, as in
    /// [`Self::generate_greedy`] - the vision tower is not a per-token cost and
    /// nothing about it is observable through this seam.
    pub fn generate_greedy_cb(&self, image: &[f32], prompt_ids: &[u32], n_new: u32, on_token: impl FnMut(u32)) -> Vec<u32> {
        assert!(
            self.row0 + self.n_rows <= prompt_ids.len() as u32,
            "the prompt's {} tokens do not contain the image run [{}, {})",
            prompt_ids.len(),
            self.row0,
            self.row0 + self.n_rows
        );
        assert!(
            prompt_ids.len() + n_new as usize <= self.seq as usize,
            "greedy decode of {} + {n_new} tokens exceeds the composite's {}-token sequence",
            prompt_ids.len(),
            self.seq
        );
        let embeds = self.encode_block(image);
        self.dec.write_img_embeds(&embeds);
        self.dec.generate_greedy_cb(prompt_ids, n_new, on_token)
    }

    /// [`Self::generate_greedy`] driven by the [`Prompt`] this composite was
    /// built for - the real-layout entry point.
    ///
    /// The prompt's ids ALREADY carry the image block (every row is `<image>`),
    /// so this is `generate_greedy(image, &prompt.ids, n_new)` with the one
    /// check that matters asserted rather than assumed: that the run the prompt
    /// describes is the run the splice was sized at. A prompt built at a
    /// different `tokens_per_side`, or against a different `text_before`, would
    /// otherwise splice the image over the wrong rows and still decode.
    pub fn generate_greedy_from_prompt(&self, image: &[f32], prompt: &Prompt, n_new: u32) -> Vec<u32> {
        self.generate_greedy_from_prompt_cb(image, prompt, n_new, |_| {})
    }

    /// [`Self::generate_greedy_from_prompt`] with the per-token callback of
    /// [`Self::generate_greedy_cb`] - same streaming contract (fires exactly
    /// `n_new` times, never for a prompt id, and the return value still carries
    /// the prompt ahead of the generated ids).
    pub fn generate_greedy_from_prompt_cb(&self, image: &[f32], prompt: &Prompt, n_new: u32, on_token: impl FnMut(u32)) -> Vec<u32> {
        assert_eq!(
            prompt.image_run(),
            self.image_run(),
            "this prompt's image run is not the one the splice was sized at"
        );
        self.generate_greedy_cb(image, &prompt.ids, n_new, on_token)
    }

    pub fn zero_grads(&self) {
        self.enc.zero_grads();
        self.dec.zero_grads();
    }

    /// `[seq, d_model]` -- the decoder's residual stream after the splice, i.e.
    /// the token embeddings with the image rows overwritten.
    pub fn read_decoder_input(&self) -> Vec<f32> {
        self.dec.read_res(0)
    }
    /// `[seq, vocab]`.
    pub fn read_logits(&self) -> Vec<f32> {
        self.dec.read_logits()
    }
}
