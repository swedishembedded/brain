// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The whole model: image in, structured result out.
//!
//! [`Mirage`] owns the vision half, the decoder, the tokenizer and the OCR
//! charset, and runs the describe and extract requests end to end on the
//! host.
//!
//! Swedish Embedded AB implements solutions for deploying end-to-end visual
//! perception pipelines for its clients. If your team needs expertise in
//! turning images into structured data then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::fmt;
use std::path::Path;

use crate::cact::{CactError, CactFile};
use crate::cactmap::CactModel;
use crate::decoder::generate::{absent_references, generate, GenerateParams};
use crate::decoder::{Decoder, Positions};
use crate::describe::{assemble, Describe, Detection, Inputs, Options, DEFAULT_CONFIDENCE};
use crate::form::{self, Extract, FormError};
use crate::import::{self, Checkpoint, ImportError, ImportOptions};
use crate::tokenizer::{Tokenizer, TokenizerError};
use crate::vision::masks::{coefficient_cell, object_field, round_box, Frame, COEF_SIDE};
use crate::vision::preprocess::{letterbox, ImageError};
use crate::vision::reader::crop_rect;
use crate::vision::weights::Weights;
use crate::vision::{ProposalOptions, VisionModel};
use crate::config::MirageConfig;

/// Tokens generated when a request sets no budget.
pub const DEFAULT_MAX_TOKENS: usize = 512;

/// Why a request failed.
#[derive(Debug)]
pub enum ModelError {
    /// Loading the checkpoint failed.
    Import(ImportError),
    /// Reading the deployment file failed.
    Cact(CactError),
    /// The tokenizer is unusable.
    Tokenizer(TokenizerError),
    /// The image is unusable.
    Image(ImageError),
    /// The form is invalid.
    Form(FormError),
    /// The request is invalid.
    Request(String),
    /// The caller cancelled the request before it finished.
    Cancelled,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Import(e) => e.fmt(f),
            ModelError::Cact(e) => e.fmt(f),
            ModelError::Tokenizer(e) => e.fmt(f),
            ModelError::Image(e) => e.fmt(f),
            ModelError::Form(e) => e.fmt(f),
            ModelError::Request(m) => write!(f, "mirage request: {m}"),
            ModelError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for ModelError {}

macro_rules! from_error {
    ($($t:ty => $v:ident),*) => {$(impl From<$t> for ModelError { fn from(e: $t) -> Self { ModelError::$v(e) } })*};
}
from_error!(ImportError => Import, CactError => Cact, TokenizerError => Tokenizer, ImageError => Image, FormError => Form);

/// The complete model.
pub struct Mirage {
    pub(crate) cfg: MirageConfig,
    pub(crate) vision: VisionModel,
    pub(crate) decoder: Decoder,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) charset: Vec<char>,
}

impl Mirage {
    /// Build from an imported checkpoint, the tokenizer blob and the reader charset.
    pub fn new(ck: &Checkpoint, tokenizer_blob: &[u8], charset: Vec<char>) -> Result<Mirage, ModelError> {
        let weights = Weights::new(ck);
        Ok(Mirage {
            cfg: ck.config.clone(),
            vision: VisionModel::new(ck),
            decoder: Decoder::new(&ck.config, &weights),
            tokenizer: Tokenizer::from_blob(tokenizer_blob)?,
            charset,
        })
    }

    /// Load a released model directory (`config.json`, `mirage.safetensors`, `mirage.cact`).
    ///
    /// A directory that also holds `adapter.safetensors` (what a LoRA fine-tune writes) is served with the adapter
    /// folded into the weights, so every surface that loads a directory serves a fine-tune unchanged.
    pub fn open(dir: &Path) -> Result<Mirage, ModelError> {
        let mut ck = import::load_dir(dir, &ImportOptions::default())?;
        let adapter_path = dir.join(crate::train::adapter::FILE);
        if adapter_path.is_file() {
            let set = crate::train::adapter::load(&adapter_path, &ck.config).map_err(ModelError::Request)?;
            crate::train::adapter::fold(&mut ck, &set).map_err(ModelError::Request)?;
        }
        let cact = CactModel::new(CactFile::open(&dir.join("mirage.cact"))?, ck.config.clone())?;
        Mirage::new(&ck, cact.tokenizer_blob()?, cact.charset()?)
    }

    /// The vision half.
    pub fn vision(&self) -> &VisionModel {
        &self.vision
    }

    /// The decoder.
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub(crate) fn prompt(&self, language: Option<&str>) -> Result<Vec<u32>, ModelError> {
        let lang = language.unwrap_or("en");
        let tag = self.tokenizer.language_tag(lang).ok_or_else(|| ModelError::Request(format!("the tokenizer has no tag for language `{lang}`")))?;
        Ok(vec![self.cfg.bos, tag])
    }

    /// Analyse an `h x w` RGB image and decode a description of it.
    pub fn describe(&self, rgb: &[u8], h: usize, w: usize, opts: &Options) -> Result<Describe, ModelError> {
        if opts.form.is_some() {
            return Err(ModelError::Request("`form` requests extraction, use `extract`".to_string()));
        }
        let img = letterbox(rgb, h, w)?;
        let analysis = self.vision.analyze(
            &img,
            ProposalOptions {
                object_threshold: opts.conf.unwrap_or(DEFAULT_CONFIDENCE),
                text_threshold: opts.text_conf.unwrap_or(DEFAULT_CONFIDENCE),
                max_objects: opts.max_objects.unwrap_or(self.cfg.detections),
                max_text: opts.max_text_lines.unwrap_or(self.cfg.regions),
            },
        );
        let memory = self.vision.decoder_memory(&analysis);
        let cross = self.decoder.cross_memory(&memory, memory.len() / self.cfg.dim);
        let generation = generate(
            &self.decoder,
            &cross,
            &self.tokenizer,
            &GenerateParams {
                prompt: &self.prompt(opts.language.as_deref())?,
                banned: &absent_references(&self.cfg, analysis.objects.len(), analysis.text.len()),
                max_tokens: opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
                max_rank: opts.max_rank(),
                object_slots: self.cfg.detections,
                guard: true,
            },
        );

        let frame = Frame { resized_w: img.w, resized_h: img.h, width: w, height: h };
        let objects: Vec<Detection> = analysis.objects.iter().map(Detection::from).collect();
        let text: Vec<Detection> = analysis.text.iter().map(Detection::from).collect();
        let maps = std::cell::OnceCell::new();
        let field = |b: [f32; 4]| {
            let (protos, coefs): &(Vec<f32>, Vec<f32>) = maps.get_or_init(|| self.vision.dense.mask_maps(&analysis.dense.head_in, &analysis.encoder, img.h, img.w));
            let (cy, cx) = coefficient_cell(round_box(b));
            object_field(protos, &coefs[(cy * COEF_SIDE + cx) * 32..][..32], b)
        };
        let read = |j: usize| -> Option<String> {
            let det = text.get(j)?;
            let public = crate::describe::public_box(det.bbox, frame).map(|v| v as f32);
            self.vision.reader.read_line(rgb, h, w, crop_rect(public, w, h), &self.charset).ok()
        };
        Ok(assemble(&generation.text, &Inputs { objects: &objects, text: &text, frame, read: &read, field: &field }, opts))
    }

    /// Analyse an image and extract the fields of `opts.form`.
    pub fn extract(&self, rgb: &[u8], h: usize, w: usize, opts: &Options) -> Result<Extract, ModelError> {
        let spec = opts.form.as_ref().ok_or_else(|| ModelError::Request("`extract` needs a `form`".to_string()))?;
        let fields = form::parse_form(spec)?;
        let img = letterbox(rgb, h, w)?;
        let analysis = self.vision.analyze(
            &img,
            ProposalOptions {
                object_threshold: opts.conf.unwrap_or(DEFAULT_CONFIDENCE),
                text_threshold: opts.text_conf.unwrap_or(DEFAULT_CONFIDENCE),
                max_objects: opts.max_objects.unwrap_or(self.cfg.detections),
                max_text: opts.max_text_lines.unwrap_or(self.cfg.regions),
            },
        );
        let memory = self.vision.decoder_memory(&analysis);
        let cross = self.decoder.cross_memory(&memory, memory.len() / self.cfg.dim);
        let prompt = self.prompt(opts.language.as_deref())?;
        Ok(form::extract(&self.decoder, &cross, &self.tokenizer, &prompt, &fields, opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))?)
    }

    /// Teacher-forced full-sequence logits (for verification).
    pub fn logits(&self, memory: &[f32], tokens: &[u32]) -> Vec<f32> {
        let cross = self.decoder.cross_memory(memory, memory.len() / self.cfg.dim);
        self.decoder.forward(&mut self.decoder.new_state(), &cross, tokens, Positions::All)
    }
}
