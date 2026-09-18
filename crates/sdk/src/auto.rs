// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`AutoPipeline`]: hand [`AutoPipeline::from_pretrained`] ANY `<vendor>/
//! <repo>` this crate has a pipeline for, and get the right concrete
//! pipeline TYPE back - a caller who does not already know whether a model
//! id names an image generator, a text decoder, or a depth model no longer
//! has to know that up front, the same "auto" promise `transformers`'
//! `AutoModel` makes.
//!
//! An ENUM, one variant per pipeline type, not a trait object: every
//! pipeline in this crate has a genuinely different call shape
//! (`ImagePipeline::generate` returns `Image`, `EmbeddingPipeline::embed`
//! returns `Embedding`, `DetectionPipeline::detect` returns
//! `Vec<Detection>`, ...), so there is no common trait to erase behind that
//! would not either lose real capability or force every method onto every
//! pipeline whether it makes sense there or not - the same "one pipeline
//! type per capability" reasoning this crate's own design rules already
//! apply to `ImagePipeline` fronting flux2 AND s3dit. A caller matches on
//! the variant they get, once, and from there holds the SAME concrete type
//! [`ImagePipeline::from_pretrained`] (or any other pipeline's own
//! constructor) would have handed them directly - [`AutoPipeline`] decides
//! WHICH constructor to call, nothing past that.
//!
//! ```no_run
//! match brain::AutoPipeline::from_pretrained("skchen1993/ZipDepth")? {
//!     brain::AutoPipeline::Depth(pipe) => {
//!         let photo = brain::Image::open("street.jpg")?;
//!         let depth = pipe.predict(&photo)?;
//!         println!("{}x{} depth map", depth.width, depth.height);
//!     }
//!     other => println!("resolved to a different pipeline type: {other:?}"),
//! }
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! ## Dispatch, and why it is safe to resolve twice
//!
//! [`detect`] tries every known architecture's [`loader::resolve_structured`]
//! against the CURRENT model store - independent of `model_id` past its own
//! syntax, the same "classify what is on disk, do not filter by name" shape
//! every existing pipeline's own resolver call already has (none of them
//! read `model_id` back out of `overrides` either). Once [`detect`] names a
//! target, [`AutoPipelineBuilder::load`] builds it by calling that
//! pipeline's OWN public `builder(model_id)...load()` - so the classification
//! that actually decides what gets built is the SAME code
//! `DepthPipeline::from_pretrained` (or any other concrete constructor)
//! already runs, resolved a second time rather than duplicated. Two resolve
//! passes over an already-cached `brain_modelstore::inventory::scan` is a
//! cheap, bounded cost for a one-shot construction call, not a hot path -
//! the same trade-off `crate::resolve_policy::resolve_with_policy`'s own
//! `Missing`-then-retry already makes.
//!
//! A store holding more than one architecture's worth of real content picks
//! among them by TRY ORDER (`detect`'s own sequence), not by inspecting
//! `model_id` - a real, documented narrowing: this is the SAME limitation
//! `crate::pipeline::ImagePipelineBuilder`'s own flux2-then-s3dit tie-break
//! already has, generalized from two architectures to every one this crate
//! serves, not a new one introduced here.

use std::collections::BTreeMap;

use crate::{Device, Error, Result};

/// Which concrete pipeline type [`AutoPipeline::from_pretrained`] resolved
/// to. See this module's own doc for why an enum, not a trait object.
/// Every variant is boxed uniformly: the concrete pipeline types range from
/// under 100 bytes to over 16KB (`yolov8::Yolo`'s own inline config/weights
/// handles), so leaving even a few unboxed still sizes every `AutoPipeline`,
/// even a tiny `Text` or `Embedding` one, off whichever variant happens to
/// be largest today. One consistent rule beats re-litigating the threshold
/// every time an underlying pipeline's own size changes.
pub enum AutoPipeline {
    Image(Box<crate::ImagePipeline>),
    Upscale(Box<crate::UpscalePipeline>),
    Restore(Box<crate::RestorePipeline>),
    Forecast(Box<crate::ForecastPipeline>),
    Embedding(Box<crate::EmbeddingPipeline>),
    Detection(Box<crate::DetectionPipeline>),
    Segment(Box<crate::SegmentPipeline>),
    Depth(Box<crate::DepthPipeline>),
    Grounding(Box<crate::GroundingPipeline>),
    Transcribe(Box<crate::TranscribePipeline>),
    Tts(Box<crate::TtsPipeline>),
    Music(Box<crate::MusicPipeline>),
    Video(Box<crate::VideoPipeline>),
    VisionLanguage(Box<crate::VisionLanguagePipeline>),
    Text(Box<crate::TextGenerationPipeline>),
}

impl std::fmt::Debug for AutoPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            AutoPipeline::Image(_) => "Image",
            AutoPipeline::Upscale(_) => "Upscale",
            AutoPipeline::Restore(_) => "Restore",
            AutoPipeline::Forecast(_) => "Forecast",
            AutoPipeline::Embedding(_) => "Embedding",
            AutoPipeline::Detection(_) => "Detection",
            AutoPipeline::Segment(_) => "Segment",
            AutoPipeline::Depth(_) => "Depth",
            AutoPipeline::Grounding(_) => "Grounding",
            AutoPipeline::Transcribe(_) => "Transcribe",
            AutoPipeline::Tts(_) => "Tts",
            AutoPipeline::Music(_) => "Music",
            AutoPipeline::Video(_) => "Video",
            AutoPipeline::VisionLanguage(_) => "VisionLanguage",
            AutoPipeline::Text(_) => "Text",
        };
        write!(f, "AutoPipeline::{name}")
    }
}

impl AutoPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<AutoPipeline> {
        AutoPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> AutoPipelineBuilder {
        AutoPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default() }
    }
}

// No `capabilities()` delegation here: `ForecastPipeline::capabilities`
// returns `::forecast::Capabilities`, a genuinely different (and
// incompatible) type from every other pipeline's `capability::Manifest`,
// and `EmbeddingPipeline`/`TranscribePipeline`/`TextGenerationPipeline` have
// no `capabilities()` at all. Forcing one uniform method over types that do
// not uniformly support it is exactly the "force a method onto every
// pipeline whether it makes sense there or not" this module's own doc
// already rules out - match out the variant and call the concrete type's
// own introspection where it exists, the same way calling ANY
// pipeline-specific method already requires.

/// Which pipeline FAMILY [`detect`] found real evidence for - carries no
/// data of its own; [`AutoPipelineBuilder::load`] re-resolves through that
/// family's own real constructor (see this module's own doc for why that
/// second resolve is safe and cheap).
enum Target {
    Image,
    Upscale,
    Restore,
    Forecast,
    Embedding,
    Detection,
    Segment,
    Depth,
    Grounding,
    Transcribe,
    Tts,
    Music,
    Video,
    VisionLanguage,
    Text,
}

/// Try every known architecture's resolver against the CURRENT model store,
/// in a fixed order, and return the first family that actually resolves.
///
/// A store that resolves NOTHING reports whichever architecture found REAL
/// (if ambiguous) evidence of itself over one that found nothing at all -
/// the same "ambiguous beats missing" preference
/// `crate::resolve_policy::try_two` already applies pairwise, generalized to
/// every architecture this crate knows about. Failing that, the FIRST
/// `Missing` seen (an arbitrary tie-break over the try order, not a claim
/// that architecture is the more likely answer) - still a real, structured
/// answer naming actual roles, more useful than a bare "nothing matched".
fn detect(overrides: &BTreeMap<String, String>) -> Result<Target> {
    use brain_modelstore::resolve::Resolution;

    let mut first_ambiguous: Option<Box<brain_modelstore::resolve::Ambiguity>> = None;
    let mut first_missing: Option<Box<brain_modelstore::resolve::Missing>> = None;

    macro_rules! probe {
        ($arch:expr, $spec:expr, $target:expr) => {
            match loader::resolve_structured($arch, $spec, overrides).map_err(Error::Backend)? {
                Resolution::Resolved(_) => return Ok($target),
                Resolution::Ambiguous(a) => {
                    if first_ambiguous.is_none() {
                        first_ambiguous = Some(a);
                    }
                }
                Resolution::Missing(m) => {
                    if first_missing.is_none() {
                        first_missing = Some(m);
                    }
                }
            }
        };
    }

    probe!("flux2", &flux2::spec::Flux2Spec, Target::Image);
    probe!("s3dit", &s3dit::spec::S3ditSpec, Target::Image);
    probe!("rrdbnet", &rrdbnet::spec::RrdbnetSpec, Target::Upscale);
    probe!("codeformer", &codeformer::spec::CodeFormerSpec, Target::Restore);
    probe!("kronos", &kronos::spec::KronosSpec, Target::Forecast);
    probe!("timesfm3", &timesfm3::spec::Timesfm3Spec, Target::Forecast);
    probe!("clip", &clip::spec::ClipSpec, Target::Embedding);
    probe!("yolov8", &yolov8::spec::YoloSpec, Target::Detection);
    probe!("sam2", &sam2::spec::Sam2Spec, Target::Segment);
    probe!("zipdepth", &zipdepth::spec::ZipdepthSpec, Target::Depth);
    probe!("florence2", &florence2::spec::Florence2Spec, Target::Grounding);
    probe!("qwen3asr", &qwen3asr::spec::Qwen3AsrSpec, Target::Transcribe);
    probe!("qwen3tts", &qwen3tts::spec::Qwen3TtsSpec, Target::Tts);
    probe!("cosyvoice", &cosyvoice::spec::CosyVoiceSpec, Target::Tts);
    probe!("minimaxmusic3", &minimaxmusic3::spec::MinimaxMusic3Spec, Target::Music);
    probe!("wan", &wan::spec::WanSpec, Target::Video);
    probe!("qwen3vl", &qwen3vl::spec::Qwen3VlSpec, Target::VisionLanguage);
    probe!("qwen3", &qwen3::spec::Qwen3Spec, Target::Text);

    if let Some(a) = first_ambiguous {
        return Err(Error::Ambiguous(a));
    }
    if let Some(m) = first_missing {
        return Err(Error::Missing(m));
    }
    Err(Error::Backend("model store is empty: no architecture this crate has a pipeline for was found".to_string()))
}

/// Builds an [`AutoPipeline`]. `.device(...)`/`.download_policy(...)` are
/// the only knobs exposed - every pipeline-SPECIFIC option (dtype, tower,
/// precision, ...) stays on that pipeline's own builder: a caller who needs
/// one already knows which concrete type they want, which is exactly the
/// case [`AutoPipeline`] is not for.
pub struct AutoPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
}

impl AutoPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// How construction may use the network to resolve `model_id`. Applies
    /// to the target pipeline's OWN build once [`detect`] has picked one -
    /// see [`loader::DownloadPolicy`]'s own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve `model_id` and build a real [`AutoPipeline`].
    ///
    /// 1. `model_id` is parsed - a malformed reference is refused here,
    ///    before [`detect`] ever runs, the same "bad PATH fails regardless
    ///    of what the store holds" ordering every other builder in this
    ///    crate keeps.
    /// 2. [`detect`] tries every known architecture's resolver against the
    ///    current store and picks the first that resolves.
    /// 3. The matching concrete pipeline's own `builder(model_id)` is built
    ///    with this builder's `device`/`download_policy` carried over, and
    ///    `.load()` - the SAME public constructor calling that type's
    ///    `from_pretrained` directly would reach.
    pub fn load(self) -> Result<AutoPipeline> {
        let AutoPipelineBuilder { model_id, device, download_policy } = self;

        brain_modelref::ModelRef::parse(&model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;

        let overrides: BTreeMap<String, String> = BTreeMap::new();
        match detect(&overrides)? {
            Target::Image => Ok(AutoPipeline::Image(Box::new(crate::ImagePipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Upscale => Ok(AutoPipeline::Upscale(Box::new(crate::UpscalePipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Restore => Ok(AutoPipeline::Restore(Box::new(crate::RestorePipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Forecast => Ok(AutoPipeline::Forecast(Box::new(crate::ForecastPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Embedding => Ok(AutoPipeline::Embedding(Box::new(crate::EmbeddingPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Detection => Ok(AutoPipeline::Detection(Box::new(crate::DetectionPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Segment => Ok(AutoPipeline::Segment(Box::new(crate::SegmentPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Depth => Ok(AutoPipeline::Depth(Box::new(crate::DepthPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Grounding => Ok(AutoPipeline::Grounding(Box::new(crate::GroundingPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Transcribe => Ok(AutoPipeline::Transcribe(Box::new(crate::TranscribePipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Tts => Ok(AutoPipeline::Tts(Box::new(crate::TtsPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Music => Ok(AutoPipeline::Music(Box::new(crate::MusicPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Video => Ok(AutoPipeline::Video(Box::new(crate::VideoPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::VisionLanguage => Ok(AutoPipeline::VisionLanguage(Box::new(crate::VisionLanguagePipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
            Target::Text => Ok(AutoPipeline::Text(Box::new(crate::TextGenerationPipeline::builder(&model_id).device(device).download_policy(download_policy).load()?))),
        }
    }
}
