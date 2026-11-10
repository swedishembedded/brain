// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mirage's vision forward pass: preprocessing, the encoder, the dense
//! detection and mask heads, the text reader and the decoder's image memory.
//!
//! Swedish Embedded AB implements solutions for running compact perception
//! models on commodity CPUs for its clients. If your team needs expertise in
//! efficient on-device vision inference then you can procure our services by
//! sending an email to info@swedishembedded.com.

pub mod dense;
pub mod encoder;
pub mod masks;
pub mod memory;
pub mod ops;
pub mod preprocess;
pub mod reader;
pub mod weights;

use dense::{find_proposals, DenseHead, DenseOutput, Kind, Proposal, DEFAULT_THRESHOLD, MAX_OBJECTS, MAX_TEXT};
use encoder::{Encoder, EncoderOutput};
use memory::MemoryBuilder;
use reader::Reader;
use weights::Weights;

use crate::config::MirageConfig;
use crate::import::Checkpoint;

/// Everything the vision half computes for one image.
pub struct Analysis {
    /// Resized image height on the canvas.
    pub resized_h: usize,
    /// Resized image width on the canvas.
    pub resized_w: usize,
    /// Encoder outputs.
    pub encoder: EncoderOutput,
    /// Dense-head outputs.
    pub dense: DenseOutput,
    /// Object proposals in decoder reference order.
    pub objects: Vec<Proposal>,
    /// Text-line proposals in decoder reference order.
    pub text: Vec<Proposal>,
}

/// Confidence thresholds for proposals.
#[derive(Debug, Clone, Copy)]
pub struct ProposalOptions {
    /// Minimum object probability.
    pub object_threshold: f32,
    /// Minimum text probability.
    pub text_threshold: f32,
    /// Ignore cells outside the resized image when picking peaks.
    pub valid_only: bool,
}

impl Default for ProposalOptions {
    fn default() -> Self {
        ProposalOptions { object_threshold: DEFAULT_THRESHOLD, text_threshold: DEFAULT_THRESHOLD, valid_only: true }
    }
}

/// The vision half of the model, prepared from an imported checkpoint.
pub struct VisionModel {
    cfg: MirageConfig,
    /// The encoder.
    pub encoder: Encoder,
    /// The dense and mask heads.
    pub dense: DenseHead,
    /// The text reader.
    pub reader: Reader,
    /// The decoder-memory builder.
    pub memory: MemoryBuilder,
}

impl VisionModel {
    /// Prepare every vision component from `ck`.
    pub fn new(ck: &Checkpoint) -> VisionModel {
        let w = Weights::new(ck);
        VisionModel {
            cfg: ck.config.clone(),
            encoder: Encoder::new(&ck.config, &w),
            dense: DenseHead::new(&ck.config, &w),
            reader: Reader::new(&ck.config, &w),
            memory: MemoryBuilder::new(&ck.config, &w),
        }
    }

    /// Run the encoder, the dense head and proposal selection on a letterboxed image.
    pub fn analyze(&self, img: &preprocess::Resized, options: ProposalOptions) -> Analysis {
        let canvas = preprocess::canvas(img);
        let encoder = self.encoder.forward(&canvas, img.h, img.w);
        let dense = self.dense.forward(&encoder, img.h, img.w);
        let side = self.cfg.canvas / self.cfg.stride;
        let valid = options.valid_only.then(|| ops::valid_cells(img.h, img.w, side));
        let objects = find_proposals(&dense.head, side, Kind::Object, options.object_threshold, MAX_OBJECTS, valid.as_deref());
        let text = find_proposals(&dense.head, side, Kind::Text, options.text_threshold, MAX_TEXT, valid.as_deref());
        Analysis { resized_h: img.h, resized_w: img.w, encoder, dense, objects, text }
    }

    /// The configuration the model was prepared with.
    pub fn model_config(&self) -> &MirageConfig {
        &self.cfg
    }

    /// Reference-token embedding rows for object proposal `k` and text proposal `j`.
    pub fn reference_token(&self, kind: Kind, index: usize) -> usize {
        let base = self.cfg.text + crate::config::LANGUAGE_TAGS;
        match kind {
            Kind::Object => base + index,
            Kind::Text => base + self.cfg.detections + index,
        }
    }

    /// Decoder memory for an analysed image.
    pub fn decoder_memory(&self, a: &Analysis) -> Vec<f32> {
        let grid = self.cfg.grid();
        let valid = ops::valid_cells(a.resized_h, a.resized_w, grid);
        let slots: Vec<memory::Slot> = a
            .objects
            .iter()
            .enumerate()
            .map(|(k, p)| memory::Slot { bbox: p.bbox, token: self.reference_token(Kind::Object, k) })
            .chain(a.text.iter().enumerate().map(|(j, p)| memory::Slot { bbox: p.bbox, token: self.reference_token(Kind::Text, j) }))
            .collect();
        self.memory.build(&a.encoder.encoder, &valid, &a.dense.head_in, &slots)
    }
}
