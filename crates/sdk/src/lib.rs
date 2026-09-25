// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain` -- the public, embeddable SDK facade over brain's model backends.
//!
//! This first slice covers [`ImagePipeline`]: resolve a model through
//! `crates/loader` (the same resolver `brain flux2 generate`/`brain do
//! z-image text2image` uses, never a parallel path), build a real,
//! resident flux2- or s3dit-backed pipeline, and generate + save images --
//! with no CLI binary, no capability-dispatch machinery, and no environment
//! variable required in the loop.
//!
//! ```no_run
//! let mut pipe = brain::ImagePipeline::from_pretrained("black-forest-labs/FLUX.2-klein-9B")?;
//! pipe.generate("a whale submarine")?.save("out.png")?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! `use brain::ImagePipeline;` is the whole point of this crate's name --
//! see `Cargo.toml` for why `brain` (not `brain-sdk`) is this workspace's
//! one deliberate exception to its `brain-<short>` package-naming
//! convention.
//!
//! ## Features
//!
//! This crate is the workspace's feature vocabulary. Name the **surfaces** you
//! use and you get their dependencies and nothing else:
//!
//! ```toml
//! brain = { version = "1", features = ["image"] }
//! ```
//!
//! | feature | what it adds |
//! |---|---|
//! | `image` | [`ImagePipeline`], [`Image`] -- text-to-image and image editing; [`UpscalePipeline`] -- super-resolution; [`RestorePipeline`] -- blind face restoration |
//! | `creature` | [`Creature`], [`View`] -- a connectome running a body, and a window onto it |
//! | `forecast` | [`ForecastPipeline`] -- time-series forecasting (kronos, timesfm3) |
//! | `text` | [`TextGenerationPipeline`] -- text generation (qwen3, from a local checkpoint path or a hub id); [`EmbeddingPipeline`]'s Qwen3/LFM2.5-Encoder backends (long-context text embedding); [`EmbeddingTrainer`] -- contrastive training of a frozen-backbone projection head; [`EncoderFineTuner`] -- full-encoder contrastive fine-tuning (LFM2.5-Encoder only, via its seeded backward pass) |
//! | `vision` | [`EmbeddingPipeline`] -- text embedding (CLIP); [`DetectionPipeline`] -- object detection (YOLOv8); [`SegmentPipeline`] -- promptable segmentation (SAM 2.1); [`DepthPipeline`] -- monocular depth (ZipDepth); [`GroundingPipeline`] -- open-vocabulary visual grounding (Florence-2); named for the `brain_arch::Domain` they resolve under, not the capability, since there is no `Embedding`/`Detection`/`Segmentation`/`Depth`/`Grounding` domain |
//! | `audio` | [`TranscribePipeline`] -- speech-to-text (qwen3-asr, offline); [`TtsPipeline`] -- text-to-speech (Qwen3-TTS: speak/clone_voice/design; CosyVoice: clone_voice); [`MusicPipeline`] -- lyrics+caption-to-song (MiniMax Music 3) |
//! | `video` | [`VideoPipeline`] -- text-to-video (Wan2.1 T2V) |
//! | `multimodal` | [`VisionLanguagePipeline`] -- vision-language (Qwen3-VL: 1-8 images + text in, text out) |
//! | `three-d` | [`Reconstruction`] -- photographs of a static scene to a 3D Gaussian Splatting scene and the camera of every photograph (structure from motion, then the fit; no weights) |
//! | `full` | every surface; this is `default` |
//!
//! `device` and `resolve` are infrastructure tiers that a surface selects for
//! you ([`Device`], [`DType`]); name a surface, not a tier. Features select
//! code and never carry values -- per-instance configuration is a builder
//! argument ([`ImagePipelineBuilder`]), never a feature name.
//!
//! ## Resource safety
//!
//! This crate is an in-process library, not a server: unlike
//! `crates/apiserve`/`crates/dbus`, it has no admission control, request
//! queue, or concurrency limit of its own -- the "resource safety &
//! backpressure" discipline those network-facing surfaces are held to does
//! not apply the same way to an in-process call. Each
//! `ImagePipeline` you build holds real, multi-gigabyte GPU/host memory for
//! as long as it lives; building several concurrently (or handing untrusted
//! callers direct control over how many get built, or over
//! [`ImageGenerationOptions`]'s `steps`/size, which are NOT range-checked
//! here -- see that type's own doc) is entirely this crate's caller's
//! responsibility to bound, exactly as it is a caller's responsibility to
//! bound any other in-process allocation. A service built ON TOP of this
//! crate that accepts requests from an untrusted network needs its own
//! admission/backpressure layer in front of it -- this crate does not, and
//! is not meant to, provide one.
//!
//! Swedish Embedded AB implements client-embeddable model inference for its
//! clients -- turning an internal research pipeline into a small, stable
//! library surface a product can link directly, with no CLI process and no
//! capability-dispatch server in the loop. If your team needs an SDK facade
//! over its own model stack, you can procure our services by sending an
//! email to info@swedishembedded.com.

/// Command-line option groups every application built on brain can compose -
/// see [`options`]. The mechanism and the hardware selection are shared with
/// the `brain` binary itself, so `--device` means one thing everywhere.
pub mod options;

/// A window and a canvas for a sample application - see [`viewport`]. Not a
/// model surface: it is how a run SHOWS what it is doing, on a desk or as a
/// PNG from a machine with no display.
#[cfg(feature = "viewport")]
pub use viewport;

#[cfg(feature = "audio")]
mod asr;
#[cfg(feature = "auto")]
mod auto;
#[cfg(feature = "creature")]
mod creature;
#[cfg(feature = "vision")]
mod depth;
#[cfg(feature = "vision")]
mod detect;
#[cfg(feature = "device")]
mod device;
mod error;
#[cfg(any(feature = "vision", feature = "text"))]
mod embedding;
#[cfg(feature = "text")]
mod embed_train;
#[cfg(feature = "text")]
mod embed_finetune;
#[cfg(feature = "forecast")]
mod forecast;
#[cfg(feature = "vision")]
mod ground;
#[cfg(feature = "imagetype")]
mod image;
#[cfg(feature = "image")]
mod pipeline;
#[cfg(feature = "audio")]
mod music;
/// `brain::DecisionPipeline` - calibrated probabilities over options supplied
/// per request. Its own surface because its output space lives in the CALL,
/// which no generative pipeline's shape can express.
#[cfg(feature = "decision")]
pub mod decision;
#[cfg(feature = "decision")]
pub use decision::{Choice, DecisionPipeline, DecisionPipelineBuilder, TrainSpec, DEFAULT_TRAIN_BATCH};

/// Learning to reach a known goal state by retracing random walks away from
/// it - a policy rolled out with one forward pass per action and no search.
///
/// Re-exported whole rather than wrapped: a caller supplies its own
/// `StateSpace`, so there is no pipeline here to hide behind a facade, and a
/// partial re-export would just mean the sample importing half a crate.
#[cfg(feature = "decision")]
pub use solve;

/// Teaching a model a batch of documents, and gating whether it learned them.
/// Its own surface because the unit is a STUDY - train, score against a
/// pre-registered bar, run a null-gate control beside it, publish only on a
/// promote - which no inference pipeline's shape can express.
#[cfg(feature = "study")]
pub mod chat_dataset;
#[cfg(feature = "study")]
pub mod study;

#[cfg(feature = "reader")]
pub mod reader;
#[cfg(feature = "reader")]
pub use audit::acceptance::LedgerFacts;
#[cfg(feature = "reader")]
pub use reader::{BatteryScore, BatteryTask, ContinualReader, ReadOutcome, MIN_EPISODE_CHARS};
#[cfg(feature = "study")]
pub use chat_dataset::{validate_chat_dataset, validate_chat_dataset_for, ChatDatasetSummary};
#[cfg(feature = "study")]
pub use study::{
    Cause, CycleOutcome, DatasetSummary, Decision, DocumentStudy, Environment, Improve, ImproveOptions, ImproveOutcome, Reward, Step, StepOutcome, StudyOutcome, Task, Verifier,
};
#[cfg(feature = "decision")]
pub mod control;
#[cfg(feature = "decision")]
pub use control::{
    Agreement, Candidates, ControlPipeline, ControlPipelineBuilder, ControlSpec,
    Counterfactual, Demonstration, Env, Rollout, Situation, Spend, ValueFit,
};
#[cfg(feature = "decision")]
pub mod conversion;
#[cfg(feature = "decision")]
pub use conversion::{
    sales_conversations, ConversionPipeline, ConversionPipelineBuilder, ConversionSpec,
    RoutingDecision, SalesConversation, SalesMessage, Verdict,
};

/// `brain::search` - the discovery runtime, re-exported unchanged.
///
/// A sample or a downstream consumer may depend only on this facade, so the
/// search half of `SEARCH -> VERIFY -> SELECT -> COMPRESS` has to be reachable
/// through it. Re-exported rather than wrapped: `brain-search` is a leaf that
/// knows nothing of models or devices, and anything this file added around it
/// would be a second spelling of an archive.
#[cfg(feature = "decision")]
pub use search;

/// `brain::promote` - the promote/reject decision, re-exported unchanged.
///
/// The SELECT half of `SEARCH -> VERIFY -> SELECT -> COMPRESS`, and the half
/// a loop that trains itself cannot do without: a generation that produces a
/// worse policy than the one before has to be REFUSED, or the loop has no
/// ratchet and wanders. `brain-promote` is a leaf in the training-substrate
/// layer that knows nothing of models - a paired sign test and four bars over
/// already-scored episodes - so it is re-exported rather than wrapped, for
/// the same reason `search` is.
#[cfg(feature = "decision")]
pub use promote;

/// `brain::RlcdPipeline` - training a decision model directly against exact
/// oracle posteriors (a soft target distribution, not a single gold index),
/// and auditing it on calibration AND cost-sensitive decision regret. Its
/// own surface because its training signal - a full distribution per
/// example - is what no other pipeline here takes.
#[cfg(feature = "decision")]
pub mod rlcd;
#[cfg(feature = "decision")]
pub use rlcd::{
    ada_ece, bayes_action, bayes_risk, check_information_refinement, classwise_ece, confidence,
    coverage_accuracy, decision_loss, decision_loss_soft, ece, failure_auroc, regret,
    reliability_bins, voi, witness_search, Answer, BayesAction, CostMatrix, DecisionContract,
    Distribution, Features, Learner, LossConfig, Observation, OracleKind, Opt, Question,
    ReliabilityBin, RlcdExample, RlcdPipeline, RlcdPipelineBuilder, RlcdSpec, WitnessFamily, World,
};

/// The stage chain every pipeline shares: `train`, `evaluate`, `save`, `ask`,
/// `tui`, `report`, `finish`. Written once, adapted per architecture through
/// [`flow::Stages`].
pub mod flow;
pub use flow::{EvalReport, Flow, Stages, TrainReport};

#[cfg(feature = "text")]
pub mod artifact;
#[cfg(feature = "resolve")]
mod resolve_policy;
#[cfg(feature = "image")]
mod restore;
#[cfg(feature = "vision")]
mod segment;
#[cfg(feature = "text")]
mod text;
#[cfg(feature = "three-d")]
mod three_d;
#[cfg(feature = "text")]
pub mod qa;
#[cfg(feature = "audio")]
mod tts;
#[cfg(feature = "image")]
mod upscale;
#[cfg(feature = "video")]
mod video;
#[cfg(feature = "creature")]
mod view;
#[cfg(feature = "multimodal")]
mod vlm;

pub use error::{Error, ForecastFailure};

/// A device/backend selection. Re-exported, not reinvented: the SAME type
/// `--device` parses into (`gpu_core::devices::DeviceSpec`). An empty
/// [`Device::default`] is the existing "auto" concept -- schedule on
/// whatever hardware the machine actually has.
#[cfg(feature = "device")]
pub use gpu_core::devices::DeviceSpec as Device;
/// A numeric tier. Re-exported, not reinvented: the SAME type flux2's own
/// `Pipeline::build_sized` takes (`model::dispatch::Precision`).
#[cfg(feature = "resolve")]
pub use model::dispatch::Precision as DType;
/// How a pipeline builder may use the network to resolve a `model_id`.
/// Re-exported, not reinvented: the SAME type `loader::supply::
/// ensure_default_weights` already takes. See
/// [`ImagePipelineBuilder::download_policy`] for the one builder that
/// exposes it as a knob today.
#[cfg(feature = "resolve")]
pub use loader::DownloadPolicy;

#[cfg(feature = "audio")]
pub use asr::{Transcript, TranscribePipeline, TranscribePipelineBuilder};
#[cfg(feature = "auto")]
pub use auto::{AutoPipeline, AutoPipelineBuilder};
#[cfg(feature = "creature")]
pub use creature::{Arena, Beat, Creature, CreatureBuilder, MotorMap, WingWiring};
#[cfg(feature = "vision")]
pub use depth::{DepthMap, DepthOptions, DepthPipeline, DepthPipelineBuilder};
#[cfg(feature = "three-d")]
pub use three_d::{Reconstruction, ReconstructionBuilder};
#[cfg(feature = "vision")]
pub use detect::{DetectOptions, Detection, DetectionPipeline, DetectionPipelineBuilder};
#[cfg(any(feature = "vision", feature = "text"))]
pub use embedding::{Embedding, EmbeddingOptions, EmbeddingPipeline, EmbeddingPipelineBuilder};
#[cfg(feature = "text")]
pub use embed_train::EmbeddingTrainer;
#[cfg(feature = "text")]
pub use embed_finetune::EncoderFineTuner;
#[cfg(feature = "forecast")]
pub use forecast::{ForecastPipeline, ForecastPipelineBuilder};
#[cfg(feature = "vision")]
pub use ground::{GroundedBox, GroundingOptions, GroundingPipeline, GroundingPipelineBuilder};
/// The forecasting domain types [`ForecastPipeline::forecast_with`] needs
/// for anything past the simple `.forecast(series, horizon)` call --
/// re-exported from `brain-forecast`, not reinvented (that crate IS the
/// model-agnostic seam; wrapping it again here would be a second,
/// competing representation of the same domain).
#[cfg(feature = "forecast")]
pub use ::forecast::{Block, Capabilities, Forecast, ForecastSpec, Item, Panel, Representation, TargetForecast, Variate};
#[cfg(feature = "imagetype")]
pub use image::Image;
#[cfg(feature = "image")]
pub use pipeline::{ImageGenerationOptions, ImagePipeline, ImagePipelineBuilder};
#[cfg(feature = "image")]
pub use restore::{RestoreOptions, RestorePipeline, RestorePipelineBuilder};
#[cfg(feature = "vision")]
pub use segment::{Mask, Prompt, SegmentOptions, SegmentPipeline, SegmentPipelineBuilder};
#[cfg(feature = "text")]
pub use text::{GeneratedText, TextGenerationOptions, TextGenerationPipeline, TextGenerationPipelineBuilder};
#[cfg(feature = "audio")]
pub use tts::{Audio, TtsOptions, TtsPipeline, TtsPipelineBuilder};
#[cfg(feature = "audio")]
pub use music::{MusicOptions, MusicPipeline, MusicPipelineBuilder, Song};
#[cfg(feature = "image")]
pub use upscale::{UpscaleOptions, UpscalePipeline, UpscalePipelineBuilder};
#[cfg(feature = "video")]
pub use video::{Video, VideoOptions, VideoPipeline, VideoPipelineBuilder};
#[cfg(feature = "creature")]
pub use view::{Steering, View};
#[cfg(feature = "multimodal")]
pub use vlm::{VisionLanguagePipeline, VisionLanguagePipelineBuilder, VlmOptions};

/// This crate's one `Result` alias -- every fallible public entry point
/// returns it.
pub type Result<T> = std::result::Result<T, Error>;
