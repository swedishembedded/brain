// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`EmbeddingPipeline`]: brain's text-embedding surface, over CLIP's text
//! towers (CLIP-L / OpenCLIP-bigG, `vision` feature) and, natively long
//! context, the Qwen3 decoder used the way Qwen3-Embedding is meant to be -
//! last-token pooled, L2-normalized, over `qwen3::caps`'s decode-only
//! `Qwen::prefill` path (`text` feature). One `model_id` in, one
//! architecture resolved, per sdk-design rule 2 (two architectures doing the
//! same task are one type dispatching on the resolved arch, not two types).
//! Neither feature alone is required by the other; each compiles this module
//! standalone (`scripts/gates/check-sdk-features.sh` enforces exactly that).
//!
//! Selected by the `vision`/`text` Cargo features, not `embedding` - see
//! `Cargo.toml`'s own comment on the `vision` feature for why
//! (`brain_arch::Domain` has no `Embedding` variant; both backbones already
//! live under their own domain, CLIP's `Vision` and Qwen3's `Text`).
//!
//! Scoped to TEXT embedding only: ArcFace (face embedding) takes an image
//! plus a detected-and-aligned face, not a string, so it needs a genuinely
//! different call shape (`embed_image`, composed with SCRFD detection) -
//! tracked as a real, separate extension, not folded into this type by
//! pretending the inputs are the same. T5-XXL/umT5-XXL
//! (`crates/t5encoder`) are conditioning encoders another model's pipeline
//! consumes internally (FLUX.1's text conditioning), not a
//! caller-facing embedding endpoint in their own right.
//!
//! ```no_run
//! let pipe = brain::EmbeddingPipeline::from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")?;
//! let v = pipe.embed("a whale submarine")?;
//! println!("{} dims", v.len());
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! (CLIP text towers ship inside the same released SDXL-layout directory
//! several image-generation checkpoints already carry - see
//! [`EmbeddingPipelineBuilder::load`]'s own doc for exactly which role this
//! resolves.)
//!
//! A literal local `.safetensors`/`.gguf` checkpoint path is always resolved
//! as a Qwen3-Embedding checkpoint (CLIP's own resolution reads a released
//! SDXL-layout DIRECTORY, never a bare file), and its context can genuinely
//! reach 32768 tokens - a real document, not a query-sized string:
//!
//! ```no_run
//! let pipe = brain::EmbeddingPipeline::builder("/models/qwen3-embedding-0.6b.safetensors")
//!     .capacity(32768)
//!     .load()?;
//! let query = pipe.embed_with(
//!     "what does the report say about Q3 revenue",
//!     brain::EmbeddingOptions::new().instruction("Given a query, retrieve relevant passages"),
//! )?;
//! let passage = pipe.embed("Q3 revenue rose 12% year over year...")?;
//! println!("{:.3}", query.cosine_similarity(&passage));
//! # Ok::<(), brain::Error>(())
//! ```

#[cfg(any(feature = "vision", feature = "text"))]
use std::collections::BTreeMap;
#[cfg(feature = "text")]
use std::path::Path;

use crate::{Device, Error, Result};

/// One embedding vector. Wraps a plain `Vec<f32>` rather than exposing one
/// directly - a domain type, per this SDK's own rule, even though today it
/// adds only a name and a `dim()` accessor; raw access stays one call away
/// (`Embedding::as_slice`).
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }

    /// Cosine similarity against `other`. Scale-invariant (unaffected by
    /// either vector's own norm), so this is meaningful whether or not the
    /// pipeline that produced them normalizes - the one piece of retrieval
    /// math this type carries; a flat/ANN index over many vectors belongs in
    /// a caller's own code, not here. `0.0` when either vector is empty or a
    /// zero vector (no direction to compare), or when the two are different
    /// lengths (comparing embeddings from two different pipelines/dimensions
    /// is a caller error, not a number worth returning).
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.0.len() != other.0.len() || self.0.is_empty() {
            return 0.0;
        }
        let dot: f64 = self.0.iter().zip(&other.0).map(|(a, b)| *a as f64 * *b as f64).sum();
        let na: f64 = self.0.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        let nb: f64 = other.0.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        (dot / (na * nb)) as f32
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(v: Vec<f32>) -> Embedding {
        Embedding(v)
    }
}

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

/// Per-call embedding knobs, layered over each backend's own defaults - see
/// [`TextGenerationOptions`](crate::TextGenerationOptions) for the same
/// pattern this mirrors.
#[derive(Clone, Debug, Default)]
pub struct EmbeddingOptions {
    instruction: Option<String>,
    max_tokens: Option<u32>,
    dimensions: Option<usize>,
    normalize: Option<bool>,
}

impl EmbeddingOptions {
    pub fn new() -> EmbeddingOptions {
        EmbeddingOptions::default()
    }

    /// The Qwen3-Embedding query convention: rendered as
    /// `"Instruct: {instruction}\nQuery: {text}"` rather than the raw text.
    /// Asymmetric retrieval by design - a query carries an instruction, a
    /// passage/document does not. Qwen3-backed pipelines only; a CLIP-backed
    /// pipeline has no instruction concept and [`EmbeddingPipeline::embed_with`]
    /// refuses this option rather than silently ignoring it.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    /// Truncate the input to this many tokens before embedding. Qwen3-backed
    /// pipelines only, same refusal-not-silent-ignore rule as `instruction`.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Truncate the RETURNED vector to its first `n` dimensions, renormalized
    /// (unless [`EmbeddingOptions::normalize`] is explicitly set to `false`)
    /// so a truncated vector still has unit norm - deliberately differing
    /// from brain's `/v1/embeddings` HTTP endpoint, which documents that it
    /// does NOT re-project after truncating (`crates/apiserve` cannot afford
    /// a whole extra normalization pass on every request at HTTP scale the
    /// way one caller's own library call can).
    pub fn dimensions(mut self, n: usize) -> Self {
        self.dimensions = Some(n);
        self
    }

    /// Force (`true`) or refuse (`false`) the renormalize-after-truncate step
    /// [`EmbeddingOptions::dimensions`] applies by default. Has no effect
    /// without `dimensions` set - the full-length vector always comes back in
    /// each backend's own natural form (CLIP's raw projection, Qwen3's
    /// already-unit-norm pooled output), so there is nothing here to opt out
    /// of before truncation enters the picture.
    pub fn normalize(mut self, on: bool) -> Self {
        self.normalize = Some(on);
        self
    }
}

/// `brain`'s text-embedding pipeline. See this module's doc for scope and
/// how a `model_id` resolves to one backend or the other.
pub struct EmbeddingPipeline {
    backend: Backend,
}

enum Backend {
    #[cfg(feature = "vision")]
    Clip {
        session: clip::caps::Session,
        /// Which CLIP tower every call embeds with - a pipeline-wide choice
        /// (`.tower(...)` on the builder), not a per-call parameter: unlike
        /// `ImagePipeline`'s two structurally interchangeable backends,
        /// CLIP-L and OpenCLIP-bigG produce embeddings of DIFFERENT
        /// dimensionality, so mixing them per call on one pipeline would
        /// silently change what a caller's stored vectors mean.
        tower: String,
    },
    #[cfg(feature = "text")]
    Qwen3 {
        model: qwen3::Qwen,
        tok: data::qwen_tokenizer::QwenBpe,
        /// The context budget this pipeline was BUILT for - fixed at
        /// construction, like `TextGenerationPipeline::capacity`. A request
        /// past it is refused, not silently truncated or rebuilt.
        capacity: u32,
    },
    /// LFM2.5-Encoder: bidirectional, so - unlike Qwen3's decode-only
    /// KV-cache build, reusable at any length up to its built capacity -
    /// the graph must be built at the EXACT request length (unmasked padding
    /// corrupts bidirectional attention; see `lfm2::caps::EncoderAction`'s
    /// own doc, which this mirrors). `hot` is the resident (length, model)
    /// pair, rebuilt only when the request length changes - the same
    /// rebuild-on-length-change rule that capability action uses, behind a
    /// `Mutex` because `EmbeddingPipeline::embed`/`embed_batch` take `&self`.
    #[cfg(feature = "text")]
    Lfm2 {
        weights: String,
        tok: data::qwen_tokenizer::QwenBpe,
        /// The longest request this pipeline will build for - refused past
        /// this, never silently truncated. No fixed KV cache to reserve
        /// ahead of time the way Qwen3's `capacity` does; this bounds the
        /// chunked-attention slab math instead.
        capacity: u32,
        hot: std::sync::Mutex<Option<(u32, lfm2::Lfm)>>,
    },
}

impl std::fmt::Debug for EmbeddingPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backend {
            #[cfg(feature = "vision")]
            Backend::Clip { tower, .. } => f.debug_struct("EmbeddingPipeline").field("backend", &"clip").field("tower", tower).finish(),
            #[cfg(feature = "text")]
            Backend::Qwen3 { capacity, .. } => f.debug_struct("EmbeddingPipeline").field("backend", &"qwen3").field("capacity", capacity).finish(),
            #[cfg(feature = "text")]
            Backend::Lfm2 { capacity, .. } => f.debug_struct("EmbeddingPipeline").field("backend", &"lfm2").field("capacity", capacity).finish(),
        }
    }
}

impl EmbeddingPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<EmbeddingPipeline> {
        EmbeddingPipeline::builder(model_id).load()
    }

    pub fn builder(model_id: impl AsRef<str>) -> EmbeddingPipelineBuilder {
        EmbeddingPipelineBuilder {
            model_id: model_id.as_ref().to_string(),
            device: Device::default(),
            tower: DEFAULT_TOWER.to_string(),
            capacity: DEFAULT_CAPACITY,
            tokenizer: None,
            download_policy: loader::DownloadPolicy::default(),
        }
    }

    /// Embed one string, at every [`EmbeddingOptions`] default.
    pub fn embed(&self, text: &str) -> Result<Embedding> {
        self.embed_with(text, EmbeddingOptions::default())
    }

    /// [`EmbeddingPipeline::embed`] plus [`EmbeddingOptions`].
    pub fn embed_with(&self, text: &str, opts: EmbeddingOptions) -> Result<Embedding> {
        Ok(self.embed_batch_with(&[text], opts)?.into_iter().next().expect("one input in, one output out"))
    }

    /// Embed `texts` in ONE batched forward, at every [`EmbeddingOptions`]
    /// default - the genuine batched path `clip::caps::Session::embed_text_batch`
    /// already implements for the CLIP backend, not a serial loop.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        self.embed_batch_with(texts, EmbeddingOptions::default())
    }

    /// [`EmbeddingPipeline::embed_batch`] plus [`EmbeddingOptions`]. The Qwen3
    /// backend has no batched forward on its decode-only build (see
    /// `qwen3::caps`'s own `embed` action doc), so this loops one prefill per
    /// text on that backend; CLIP's own batched path is used unchanged.
    pub fn embed_batch_with(&self, texts: &[&str], opts: EmbeddingOptions) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out: Vec<Vec<f32>> = match &self.backend {
            #[cfg(feature = "vision")]
            Backend::Clip { session, tower } => {
                if opts.instruction.is_some() {
                    return Err(Error::Backend("embedding: 'instruction' is a Qwen3-Embedding option; this pipeline resolved to a CLIP text tower".to_string()));
                }
                if opts.max_tokens.is_some() {
                    return Err(Error::Backend("embedding: 'max_tokens' is a Qwen3-Embedding option; this pipeline resolved to a CLIP text tower".to_string()));
                }
                let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
                session.embed_text_batch(tower, &owned).map_err(Error::Backend)?
            }
            #[cfg(feature = "text")]
            Backend::Qwen3 { model, tok, capacity } => texts.iter().map(|t| qwen3_embed_one(model, tok, *capacity, t, &opts)).collect::<Result<Vec<_>>>()?,
            #[cfg(feature = "text")]
            Backend::Lfm2 { weights, tok, capacity, hot } => {
                if opts.instruction.is_some() {
                    return Err(Error::Backend("embedding: 'instruction' is a Qwen3-Embedding option; this pipeline resolved to the LFM2.5-Encoder backend".to_string()));
                }
                texts.iter().map(|t| lfm2_embed_one(weights, tok, *capacity, hot, t, &opts)).collect::<Result<Vec<_>>>()?
            }
        };

        if let Some(d) = opts.dimensions {
            let renormalize = opts.normalize.unwrap_or(true);
            for v in out.iter_mut() {
                if d > v.len() {
                    return Err(Error::Backend(format!("embedding: dimensions ({d}) exceeds this pipeline's embedding length ({})", v.len())));
                }
                v.truncate(d);
                if renormalize {
                    l2_normalize(v);
                }
            }
        }

        Ok(out.into_iter().map(Embedding::from).collect())
    }
}

/// One Qwen3-Embedding call: the Qwen3-Embedding query convention
/// (`instruction`), `Qwen::prefill`'s last-token pooled hidden state,
/// L2-normalized - the SAME sequence `qwen3::caps::EmbedAction` runs, built
/// here directly against `qwen3::Qwen` rather than through
/// `capability::Invocation` dispatch (this crate's "wrap, don't re-export"
/// convention - see `crate::text`'s `generate_with` for the same shape).
#[cfg(feature = "text")]
fn qwen3_embed_one(model: &qwen3::Qwen, tok: &data::qwen_tokenizer::QwenBpe, capacity: u32, text: &str, opts: &EmbeddingOptions) -> Result<Vec<f32>> {
    use data::tokenizer::Tokenizer;

    let content = match &opts.instruction {
        Some(instr) => format!("Instruct: {instr}\nQuery: {text}"),
        None => text.to_string(),
    };
    let mut ids: Vec<u32> = tok.encode(&content);
    if let Some(mt) = opts.max_tokens {
        ids.truncate(mt as usize);
    }
    // An EOS token gives the pooled position something the checkpoint was
    // actually trained to summarize into, rather than whatever content token
    // happens to end the input - same reasoning as `qwen3::caps::EmbedAction`.
    if let Some(eos) = tok.special_id("<|endoftext|>") {
        ids.push(eos);
    }
    if ids.is_empty() {
        return Err(Error::Backend("qwen3 embedding: empty input".to_string()));
    }
    let need = ids.len() as u32;
    if need > capacity {
        return Err(Error::Backend(format!(
            "qwen3 embedding: input is {need} tokens, past this pipeline's built capacity ({capacity} tokens) -- rebuild with .capacity({need}) or larger"
        )));
    }

    // A resident decode-only model's KV cache carries state (`dec_pos`)
    // across calls; reset it before every request, same reasoning as
    // `qwen3::caps::EmbedAction`.
    model.reset_cache();
    let inputs: Vec<qwen3::model::PrefillInput> = ids.iter().map(|&t| qwen3::model::PrefillInput::Token(t)).collect();
    let mut hidden = model.prefill(&inputs);
    l2_normalize(&mut hidden);
    Ok(hidden)
}

/// Attention-slab budget for LFM2's chunked long-context path - the same
/// value `lfm2::caps`'s own `EncoderAction` uses (that constant is private
/// to that module, so this is a second literal, not a shared one; see that
/// module's own doc for the derivation).
#[cfg(feature = "text")]
const LFM2_SLAB_BUDGET: u64 = 512 << 20;

/// One LFM2.5-Encoder call: mean pooling over every hidden-state row (the
/// SAME sequence `lfm2::caps::EncoderAction`'s `embed` action runs), always
/// L2-normalized here (unlike that capability action's own `normalize:
/// false` default - a served action owes existing callers byte-identical
/// output; this SDK surface has no such caller yet and matches
/// `qwen3_embed_one`'s own always-normalized contract instead, so
/// `Embedding::cosine_similarity` means the same thing regardless of which
/// backbone resolved).
///
/// Bidirectional attention means the graph is rebuilt at the EXACT request
/// length whenever it changes (`hot`'s own doc) - a real, different cost
/// profile from Qwen3's KV-cache reuse, which is why this is a `Mutex`, not
/// a plain field: [`EmbeddingPipeline::embed`]/`embed_batch` take `&self`.
#[cfg(feature = "text")]
fn lfm2_embed_one(weights: &str, tok: &data::qwen_tokenizer::QwenBpe, capacity: u32, hot: &std::sync::Mutex<Option<(u32, lfm2::Lfm)>>, text: &str, opts: &EmbeddingOptions) -> Result<Vec<f32>> {
    use data::tokenizer::Tokenizer;

    let mut ids: Vec<u32> = tok.template_prefix().to_vec();
    ids.extend(tok.encode(text));
    if let Some(mt) = opts.max_tokens {
        ids.truncate(mt as usize);
    }
    if ids.is_empty() {
        return Err(Error::Backend("lfm2 embedding: empty input".to_string()));
    }
    let need = ids.len() as u32;
    if need > capacity {
        return Err(Error::Backend(format!(
            "lfm2 embedding: input is {need} tokens, past this pipeline's built capacity ({capacity} tokens) -- rebuild with .capacity({need}) or larger"
        )));
    }

    let mut guard = hot.lock().map_err(|_| Error::Backend("lfm2 embedding: hot model lock poisoned".to_string()))?;
    let reuse = matches!(&*guard, Some((len, _)) if *len == need);
    if !reuse {
        let model = lfm2::Lfm::load_inference_chunked(weights, 1, need, LFM2_SLAB_BUDGET, 0);
        *guard = Some((need, model));
    }
    let (_, model) = guard.as_ref().expect("hot model present");

    model.set_tokens(&ids);
    model.forward();
    let d = model.cfg.d_model as usize;
    let hidden = &model.read_hidden()[..ids.len() * d];
    let mut mean = vec![0.0f32; d];
    for row in hidden.chunks_exact(d) {
        for (m, &x) in mean.iter_mut().zip(row) {
            *m += x;
        }
    }
    for x in &mut mean {
        *x /= ids.len() as f32;
    }
    l2_normalize(&mut mean);
    Ok(mean)
}

/// L2-normalize in place. A zero vector (degenerate input) is left as-is
/// rather than divided by zero.
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}

const DEFAULT_TOWER: &str = "clip_l";

/// The context budget an [`EmbeddingPipeline`] gets when
/// [`EmbeddingPipelineBuilder::capacity`] is never called: generous enough
/// for a paragraph-sized passage, small enough that the KV cache a Qwen3
/// backend reserves does not surprise a caller who never thought about it.
/// Override for a real document - up to 32768 on a Qwen3-Embedding-shaped
/// checkpoint, see this module's own doc for the long-context example.
/// (CLIP's own tower has a fixed context this constant does not affect.)
const DEFAULT_CAPACITY: u32 = 2048;

/// Builds an [`EmbeddingPipeline`]. `.device(...)`/`.tower(...)`/
/// `.capacity(...)`/`.tokenizer(...)` are the only knobs this milestone
/// exposes.
pub struct EmbeddingPipelineBuilder {
    model_id: String,
    device: Device,
    tower: String,
    capacity: u32,
    tokenizer: Option<String>,
    download_policy: loader::DownloadPolicy,
}

impl EmbeddingPipelineBuilder {
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Which CLIP text tower to embed with: `"clip_l"` (the default) or
    /// `"openclip_bigg"` - see [`EmbeddingPipeline`]'s own doc for why this
    /// is fixed per pipeline rather than a per-call choice. No effect on a
    /// pipeline that resolves to the Qwen3 backend.
    pub fn tower(mut self, tower: impl Into<String>) -> Self {
        self.tower = tower.into();
        self
    }

    /// The context budget (in tokens) to build a Qwen3 backend's KV cache
    /// for - see [`EmbeddingPipeline`]'s own doc for why this is fixed at
    /// build time rather than resized per call. No effect on a pipeline that
    /// resolves to the CLIP backend.
    pub fn capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// How [`EmbeddingPipelineBuilder::load`] may use the network to resolve
    /// `model_id`. Defaults to [`loader::DownloadPolicy::IfMissing`] -- see
    /// that type's own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// A tokenizer to use instead of one resolved automatically - required
    /// for a Qwen3 `.safetensors` checkpoint (brain's own format carries no
    /// embedded tokenizer, same rule as `TextGenerationPipelineBuilder::tokenizer`);
    /// ignored for a `.gguf` that already embeds one. No effect on a pipeline
    /// that resolves to the CLIP backend.
    pub fn tokenizer(mut self, path: impl AsRef<str>) -> Self {
        self.tokenizer = Some(path.as_ref().to_string());
        self
    }
}

/// `vision`-only build: unchanged from before the Qwen3 backend existed -
/// every `model_id` resolves against CLIP's `"towers"` role, exactly as
/// [`EmbeddingPipelineBuilder::load`]'s doc below describes.
#[cfg(all(feature = "vision", not(feature = "text")))]
impl EmbeddingPipelineBuilder {
    /// Resolve `model_id` and build a real [`EmbeddingPipeline`] over CLIP.
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]).
    /// 2. `model_id` is resolved against CLIP's `"towers"` role, trying
    ///    resolution BEFORE ever consulting `Store::local`/`plan` - see
    ///    [`load_clip`]'s own doc for why.
    /// 3. `clip::caps::Session::load` opens the resolved directory's
    ///    tokenizer(s); the text tower itself builds lazily, on first
    ///    [`EmbeddingPipeline::embed`]/[`EmbeddingPipeline::embed_batch`] call.
    pub fn load(self) -> Result<EmbeddingPipeline> {
        crate::device::apply(&self.device)?;
        Ok(EmbeddingPipeline { backend: load_clip(&self.model_id, self.tower, self.download_policy)? })
    }
}

/// `text`-only build: every `model_id` resolves against Qwen3 or LFM2 - see
/// [`resolve_text_backend`]'s own doc for how the two are told apart.
#[cfg(all(feature = "text", not(feature = "vision")))]
impl EmbeddingPipelineBuilder {
    /// Resolve `model_id` and build a real [`EmbeddingPipeline`] over Qwen3
    /// or LFM2. See [`resolve_text_backend`]'s own doc for the rule.
    pub fn load(self) -> Result<EmbeddingPipeline> {
        crate::device::apply(&self.device)?;
        Ok(EmbeddingPipeline { backend: resolve_text_backend(&self.model_id, self.capacity, self.tokenizer, self.download_policy)? })
    }
}

/// Both features on: a literal local file always resolves against Qwen3 or
/// LFM2 (CLIP's own resolution reads a directory, never a bare file - see
/// [`resolve_text_backend`]'s doc), otherwise CLIP's directory-shaped
/// resolution is tried first (preserving this pipeline's original CLIP-only
/// default behavior exactly, for a caller who never asked for the `text`
/// feature's new capability) and the text backends are the fallback when
/// CLIP's resolution reports the model genuinely missing.
#[cfg(all(feature = "vision", feature = "text"))]
impl EmbeddingPipelineBuilder {
    pub fn load(self) -> Result<EmbeddingPipeline> {
        crate::device::apply(&self.device)?;
        let EmbeddingPipelineBuilder { model_id, device: _, tower, capacity, tokenizer, download_policy } = self;

        if Path::new(&model_id).is_file() {
            return Ok(EmbeddingPipeline { backend: resolve_text_backend(&model_id, capacity, tokenizer, download_policy)? });
        }

        match load_clip(&model_id, tower, download_policy) {
            Ok(backend) => Ok(EmbeddingPipeline { backend }),
            Err(Error::Missing(_)) => Ok(EmbeddingPipeline { backend: resolve_text_backend(&model_id, capacity, tokenizer, download_policy)? }),
            Err(e) => Err(e),
        }
    }
}

/// Resolve `model_id` against CLIP's `"towers"` role and build a real
/// `Backend::Clip`.
///
/// Tries resolution FIRST, before ever consulting `Store::local`/`plan` -
/// deliberately the reverse of the naive "check `Store::local`, then
/// fetch-if-missing, then resolve" order, for the same real reason
/// `crate::tts::TtsPipelineBuilder::load`/`crate::depth::DepthPipelineBuilder::load`
/// already apply it: `ClipSpec::classify` reads a released SDXL-layout
/// directory's own subdirectories directly, a strictly wider net than
/// `Store::local`'s narrow "a compound `brain.manifest.json`, or a bare
/// `model.brain.safetensors`" shapes - and a real SDXL release (no
/// `brain_modelstore::recipe::FilesRecipe` entry exists for CLIP) has
/// neither: it carries `model_index.json`, not the top-level `config.json`
/// `plan()`'s `TransformersRecipe` catch-all looks for, so `Store::local`
/// never recognizes even an already-downloaded release and `plan()` fails
/// outright.
#[cfg(feature = "vision")]
fn load_clip(model_id: &str, tower: String, download_policy: loader::DownloadPolicy) -> Result<Backend> {
    let overrides: BTreeMap<String, String> = BTreeMap::new();
    let assembly = crate::resolve_policy::resolve_with_policy("clip", &clip::spec::ClipSpec, model_id, &overrides, download_policy)?;
    let dir = assembly.roles.get("towers").ok_or_else(|| Error::Backend(format!("clip: resolved assembly {:?} has no towers role", assembly.id)))?;

    let gpu = gpu_core::Gpu::new(clip::model::TEXT_PIPELINES);
    let session = clip::caps::Session::load(&dir.to_string_lossy(), gpu).map_err(Error::Backend)?;
    Ok(Backend::Clip { session, tower })
}

/// Resolve `model_id` against Qwen3 or LFM2 - the two `text`-feature
/// backbones.
///
/// A literal local file ([`Path::is_file`]) is classified by its
/// `ModelCard.family`: `"lfm"` ([`lfm2::spec::CARD_FAMILY`]) routes to
/// [`load_lfm2`]; everything else - including a `.gguf` (which carries no
/// brain `ModelCard` at all) or a `ModelCard`-less file - routes to
/// [`load_qwen3`], preserving that function's own pre-existing "let an
/// unlabeled checkpoint through" behavior exactly (`crate::text::check_local_weights_architecture`'s
/// own doc: only a POSITIVE, different-architecture marker is ever refused).
///
/// A hub id (not a local file) tries [`load_qwen3`]'s resolver FIRST - the
/// pre-existing default for every caller before LFM2 was a reachable
/// backend at all - and only falls back to [`load_lfm2`] when Qwen3's own
/// resolution reports the model genuinely `Missing`, the same two-way
/// tie-break [`EmbeddingPipelineBuilder::load`]'s CLIP-vs-text cascade
/// already uses one level up.
#[cfg(feature = "text")]
fn resolve_text_backend(model_id: &str, capacity: u32, tokenizer: Option<String>, download_policy: loader::DownloadPolicy) -> Result<Backend> {
    if Path::new(model_id).is_file() {
        if let Ok(Some(card)) = checkpoint::st::read_card(model_id) {
            if card.family == lfm2::spec::CARD_FAMILY {
                return load_lfm2(model_id, capacity, tokenizer, download_policy);
            }
        }
        return load_qwen3(model_id, capacity, tokenizer, download_policy);
    }

    match load_qwen3(model_id, capacity, tokenizer.clone(), download_policy) {
        Ok(backend) => Ok(backend),
        Err(Error::Missing(_)) => load_lfm2(model_id, capacity, tokenizer, download_policy),
        Err(e) => Err(e),
    }
}

/// Resolve `model_id` as a Qwen3-Embedding checkpoint and build a real
/// `Backend::Qwen3`, DECODE-ONLY (`Qwen::from_reader_decode`, `O(T)` memory) -
/// never the batched forward, which is `O(T^2)` in `scores`/`probs` and the
/// wrong shape for the long-context case this backend exists for. See
/// `qwen3::caps`'s own `embed` action doc for why.
///
/// A string that names a real file already on disk (`Path::new(s).is_file()`)
/// is ALWAYS a local path - the same rule `crate::text::TextGenerationPipelineBuilder::load`
/// uses, and this reuses `crate::text::check_local_weights_architecture`/
/// `resolve_hub_weights` directly rather than re-deriving the same qwen3
/// hub-resolution logic a second time. `tokenizer`: an explicit
/// [`EmbeddingPipelineBuilder::tokenizer`] always wins over anything
/// resolved, same precedence `TextGenerationPipelineBuilder::load` uses.
#[cfg(feature = "text")]
fn load_qwen3(model_id: &str, capacity: u32, tokenizer: Option<String>, download_policy: loader::DownloadPolicy) -> Result<Backend> {
    let (weights, resolved_tokenizer) = if Path::new(model_id).is_file() {
        crate::text::check_local_weights_architecture(model_id)?;
        (model_id.to_string(), None)
    } else {
        crate::text::resolve_hub_weights(model_id, download_policy)?
    };

    let reader = checkpoint::weightio::WeightReader::open(&weights).map_err(|e| Error::Backend(format!("qwen3 embedding: {weights}: {e}")))?;
    let cfg = qwen3::QwenConfig::from_json(&reader.config());

    let tok = if let Some(t) = &tokenizer {
        data::qwen_tokenizer::QwenBpe::from_file(t).map_err(Error::Backend)?
    } else if let Some(gt) = reader.tokenizer() {
        data::qwen_tokenizer::QwenBpe::from_gguf(&gt).map_err(Error::Backend)?
    } else if let Some(rt) = &resolved_tokenizer {
        data::qwen_tokenizer::QwenBpe::from_file(rt).map_err(Error::Backend)?
    } else {
        return Err(Error::MissingArgument(format!("{weights}: no tokenizer embedded (not a .gguf) and none resolved")));
    };

    if capacity > cfg.block_size {
        return Err(Error::Backend(format!(
            "qwen3 embedding: requested capacity ({capacity} tokens) exceeds this checkpoint's configured context ({} tokens)",
            cfg.block_size
        )));
    }

    let shard = qwen3::Shard::whole(cfg.n_layers as usize);
    // `reader` moves into the closure: `Qwen::from_reader_decode` borrows it
    // internally, and `place_and_build` calls the closure exactly once.
    let model = qwen3::footprint::place_and_build(&cfg, &shard, qwen3::Dtype::F32, 1, capacity, false, true, "qwen3", move || qwen3::Qwen::from_reader_decode(&reader, capacity))
        .map_err(Error::Backend)?;

    Ok(Backend::Qwen3 { model, tok, capacity })
}

/// Resolve `model_id` as an LFM2.5-Encoder checkpoint and build a real
/// `Backend::Lfm2`. The `Lfm` model itself builds LAZILY, on the first
/// [`EmbeddingPipeline::embed`]/`embed_batch` call - the same "resolve now,
/// build on first use" split [`load_clip`] already establishes, and a real
/// necessity here specifically: bidirectional attention needs the graph
/// sized to the EXACT request length (see `Backend::Lfm2`'s own doc), which
/// is not knowable at `load()` time at all.
///
/// Same local-path-vs-hub-id rule as [`load_qwen3`]: a literal local file is
/// used directly; otherwise `model_id` resolves through
/// [`lfm2::spec::Lfm2Spec`] (the model-store resolver). Unlike Qwen3, LFM2.5
/// has no embedded-tokenizer format (no GGUF path exists for this arch at
/// all - see [`lfm2::spec`]'s own doc) - an explicit
/// [`EmbeddingPipelineBuilder::tokenizer`] or the resolver's own pick is the
/// only two sources, checked in that order.
#[cfg(feature = "text")]
fn load_lfm2(model_id: &str, capacity: u32, tokenizer: Option<String>, download_policy: loader::DownloadPolicy) -> Result<Backend> {
    let (weights, resolved_tokenizer) = if Path::new(model_id).is_file() {
        (model_id.to_string(), None)
    } else {
        let overrides: BTreeMap<String, String> = BTreeMap::new();
        let assembly = crate::resolve_policy::resolve_with_policy("lfm2", &lfm2::spec::Lfm2Spec, model_id, &overrides, download_policy)?;
        let w = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("lfm2: resolved assembly {:?} has no weights role", assembly.id)))?;
        if !w.exists() {
            return Err(Error::Backend(format!("lfm2: resolved assembly {:?} is missing weights at {}", assembly.id, w.display())));
        }
        let t = assembly.roles.get("tokenizer").map(|p| p.to_string_lossy().into_owned());
        (w.to_string_lossy().into_owned(), t)
    };

    let tok_path = tokenizer
        .or(resolved_tokenizer)
        .ok_or_else(|| Error::MissingArgument(format!("{weights}: no tokenizer resolved for this LFM2.5 checkpoint; call .tokenizer(path)")))?;
    let tok = data::qwen_tokenizer::QwenBpe::from_file(&tok_path).map_err(Error::Backend)?;

    Ok(Backend::Lfm2 { weights, tok, capacity, hot: std::sync::Mutex::new(None) })
}
