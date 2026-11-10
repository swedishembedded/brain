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

use model::cq::CqActivations;

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

/// Which stored weights a model directory is loaded from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WeightSource {
    /// `mirage.safetensors`: FP32 weights, resident as such. The default whenever the file exists.
    #[default]
    Safetensors,
    /// `mirage.cact`: the deployment archive. Its CQ matrices stay packed on the device (2 to 4 bits per weight
    /// plus fp16 group norms); precision is the archive's, so this is an explicit opt-in whenever both files exist.
    Cact,
}

impl WeightSource {
    /// What a directory holds: FP32 safetensors if present, else the deployment archive.
    pub fn detect(dir: &Path) -> Result<WeightSource, ModelError> {
        if dir.join(import::WEIGHTS_FILE).is_file() {
            Ok(WeightSource::Safetensors)
        } else if dir.join(crate::resolver::DEPLOYMENT_FILE).is_file() {
            Ok(WeightSource::Cact)
        } else {
            Err(ModelError::Request(format!("{} holds neither {} nor {}", dir.display(), import::WEIGHTS_FILE, crate::resolver::DEPLOYMENT_FILE)))
        }
    }
}

/// The numeric precision a model is served at: which stored weights, and how the packed ones read their activations.
/// The `precision` parameter of the capability, the residency adapter's instance key and the CLI flag all name it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Precision {
    /// `fp32`: the safetensors weights, f32 activations. The default whenever the checkpoint exists.
    #[default]
    Fp32,
    /// `cq`: the packed deployment archive with f32 activations after the rotation.
    Cq,
    /// `cq-a8`: the packed archive with int8 activations (one scale per rotated 128-lane group, integer group dot
    /// products), as the vendor engine runs it. Lower precision than `cq`, so only on request.
    CqA8,
}

impl Precision {
    /// Every precision, in the order the parameter lists them.
    pub const ALL: [Precision; 3] = [Precision::Fp32, Precision::Cq, Precision::CqA8];

    /// The name a request writes.
    pub fn name(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Cq => "cq",
            Precision::CqA8 => "cq-a8",
        }
    }

    /// The precision named `name`.
    pub fn parse(name: &str) -> Option<Precision> {
        Precision::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The stored weights this precision loads.
    pub fn source(self) -> WeightSource {
        match self {
            Precision::Fp32 => WeightSource::Safetensors,
            Precision::Cq | Precision::CqA8 => WeightSource::Cact,
        }
    }

    /// How its packed matrices read their activations.
    pub fn activations(self) -> CqActivations {
        match self {
            Precision::CqA8 => CqActivations::A8,
            Precision::Fp32 | Precision::Cq => CqActivations::F32,
        }
    }
}

impl From<WeightSource> for Precision {
    /// The precision with f32 activations that loads `source`.
    fn from(source: WeightSource) -> Precision {
        match source {
            WeightSource::Safetensors => Precision::Fp32,
            WeightSource::Cact => Precision::Cq,
        }
    }
}

/// A model directory's weights, the tokenizer blob and the reader charset.
pub struct Loaded {
    /// The tensors (CQ matrices dequantised) and, for [`WeightSource::Cact`], the packed matrices.
    pub checkpoint: Checkpoint,
    /// The tokenizer blob of the deployment archive.
    pub tokenizer: Vec<u8>,
    /// The reader charset of the deployment archive.
    pub charset: Vec<char>,
}

/// Read `<dir>/config.json` and the weights `source` names; the deployment archive is always read for the tokenizer
/// and charset. An `adapter.safetensors` is folded into the weights, which is only possible from FP32.
pub fn load_dir(dir: &Path, source: WeightSource) -> Result<Loaded, ModelError> {
    let archive = |config: &MirageConfig| -> Result<CactModel, ModelError> { Ok(CactModel::new(CactFile::open(&dir.join(crate::resolver::DEPLOYMENT_FILE))?, config.clone())?) };
    let adapter_path = dir.join(crate::train::adapter::FILE);
    let (mut ck, cact) = match source {
        WeightSource::Safetensors => {
            let ck = import::load_dir(dir, &ImportOptions::default())?;
            let cact = archive(&ck.config)?;
            (ck, cact)
        }
        WeightSource::Cact => {
            if adapter_path.is_file() {
                return Err(ModelError::Request(format!("{} holds an adapter, which is folded into FP32 weights; load the safetensors instead", dir.display())));
            }
            let config_path = dir.join(import::CONFIG_FILE);
            let sidecar = crate::config::CheckpointConfig::from_json(&std::fs::read_to_string(&config_path).map_err(|e| ModelError::Request(format!("{}: {e}", config_path.display())))?).map_err(|e| ModelError::Request(e.to_string()))?;
            let cact = archive(&sidecar.architecture)?;
            let metadata = import::Metadata { source_sha256: sidecar.source_checkpoint_sha256.clone(), step: sidecar.training_step, format: import::FORMAT.to_string() };
            let ck = Checkpoint { config: sidecar.architecture, metadata, tensors: cact.canonical_tensors()?, packed: cact.packed_matrices()?, cq_activations: CqActivations::F32 };
            (ck, cact)
        }
    };
    if source == WeightSource::Safetensors && adapter_path.is_file() {
        let set = crate::train::adapter::load(&adapter_path, &ck.config).map_err(ModelError::Request)?;
        crate::train::adapter::fold(&mut ck, &set).map_err(ModelError::Request)?;
    }
    Ok(Loaded { tokenizer: cact.tokenizer_blob()?.to_vec(), charset: cact.charset()?, checkpoint: ck })
}

/// [`load_dir`] at `precision`: the weights it names, with its activation mode recorded for the host paths.
pub fn load_precision(dir: &Path, precision: Precision) -> Result<Loaded, ModelError> {
    let mut loaded = load_dir(dir, precision.source())?;
    loaded.checkpoint.cq_activations = precision.activations();
    Ok(loaded)
}

/// Quantise the FP32 checkpoint in `dir' (with its adapter folded in, if it holds one) into the deployment archive in
/// `out`, next to a copy of `config.json` so `out` loads like any model directory with [`WeightSource::Cact`].
///
/// The archive's plan (which tensors are CQ at which width, the codebooks, the derived entries and the tokenizer) is
/// taken from the `mirage.cact` already in `dir`; the weights come from the safetensors. Files are written through a
/// temporary name and renamed, so a failed run leaves no partial archive. Returns the archive's path.
pub fn quantize_dir(dir: &Path, out: &Path) -> Result<std::path::PathBuf, ModelError> {
    let io = |what: &str, e: std::io::Error| ModelError::Request(format!("{what}: {e}"));
    if !dir.join(import::WEIGHTS_FILE).is_file() {
        return Err(ModelError::Request(format!("{} holds no {} to quantise", dir.display(), import::WEIGHTS_FILE)));
    }
    let template_path = dir.join(crate::resolver::DEPLOYMENT_FILE);
    if !template_path.is_file() {
        return Err(ModelError::Request(format!("{} holds no {}: the released archive supplies the quantisation plan", dir.display(), crate::resolver::DEPLOYMENT_FILE)));
    }
    let fp32 = load_dir(dir, WeightSource::Safetensors)?;
    let template = CactModel::new(CactFile::open(&template_path)?, fp32.checkpoint.config.clone())?;
    let bytes = template.quantize(&fp32.checkpoint)?;
    std::fs::create_dir_all(out).map_err(|e| io(&out.display().to_string(), e))?;
    let target = out.join(crate::resolver::DEPLOYMENT_FILE);
    let tmp = out.join(format!("{}.tmp", crate::resolver::DEPLOYMENT_FILE));
    std::fs::write(&tmp, &bytes).map_err(|e| io(&tmp.display().to_string(), e))?;
    std::fs::rename(&tmp, &target).map_err(|e| io(&target.display().to_string(), e))?;
    let config = dir.join(import::CONFIG_FILE);
    if config.is_file() && out.canonicalize().ok() != dir.canonicalize().ok() {
        std::fs::copy(&config, out.join(import::CONFIG_FILE)).map_err(|e| io(&config.display().to_string(), e))?;
    }
    Ok(target)
}

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
        Mirage::open_with(dir, WeightSource::detect(dir)?)
    }

    /// [`Mirage::open`] from the weights `source` names.
    pub fn open_with(dir: &Path, source: WeightSource) -> Result<Mirage, ModelError> {
        Mirage::open_precision(dir, source.into())
    }

    /// [`Mirage::open`] at `precision`.
    pub fn open_precision(dir: &Path, precision: Precision) -> Result<Mirage, ModelError> {
        let loaded = load_precision(dir, precision)?;
        Mirage::new(&loaded.checkpoint, &loaded.tokenizer, loaded.charset)
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
