// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mirage behind the generalized [`capability`] interface: what makes
//! `brain caps`, `brain do mirage describe|extract`, the HTTP and D-Bus
//! surfaces and the SDK work with no model-specific plumbing.
//!
//! Two actions, both taking one `image` blob (raw interleaved HWC `f32` in
//! `[0,1]`, [`capability::blob`]):
//!
//! * **`describe`**: the engine-shaped structured result (scene, objects with
//!   boxes, scores and masks, regions, text lines, relations, events,
//!   reasoning), one output per section that was requested and produced.
//! * **`extract`**: a `form` (a JSON object of choice, count and text fields)
//!   answered about the image, as `{form: {field: {value, probabilities}}}`.
//!
//! Every outcome also carries a `usage` object (images, decoder tokens and the
//! latency split of the batch the request ran in).
//!
//! # One implementation
//!
//! [`MirageSession`] is the single place a request is checked, batched and
//! answered. [`MirageProvider`] (one request at a time, loaded on first use)
//! and the residency adapter in `crates/catalog` (concurrent requests merged
//! into one device batch) both call [`MirageSession::run_batch`]. Requests of
//! one batch that share their options are one [`MirageModel::describe_batch_with`]
//! call; requests with different options are separate calls of the same batch.
//!
//! # Boundary
//!
//! No param names a path: the checkpoint directory comes from the model store
//! when the provider is built. Images are bounded by
//! [`capability::blob::MAX_IMAGE_PIXELS`]; counts and tokens by the ceilings
//! declared on the params; a form by [`MAX_FORM_BYTES`], [`MAX_FORM_FIELDS`]
//! and [`MAX_FORM_CANDIDATES`]. A request that breaks a bound is refused with
//! the name of the offending field before any device work.
//!
//! Swedish Embedded AB implements solutions for serving compact perception
//! models to many concurrent callers for its clients. If your team needs
//! expertise in batched vision inference then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use capability::{hot::Hot, Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use gpu_core::Gpu;
use serde_json::{json, Map, Value};

use crate::describe::{Options, LANGUAGES};
use crate::device::{BatchControl, DecoderMemory, Device, Image, MirageModel};
use crate::form::{self, FieldKind, FormError};
use crate::memory_cache::{MemoryCache, MemoryKey};
use crate::model::{Mirage, ModelError, WeightSource};
use crate::vision::dense::{MAX_OBJECTS, MAX_TEXT};

/// The model id used on the CLI (`brain do mirage …`), over the bus and in the residency manifest.
pub const MODEL: &str = "Cactus-Compute/mirage";

/// Version of the preprocessing (letterbox, canvas, normalisation) a cached decoder memory was computed with.
/// Bump it whenever those change, so no old memory is ever reused.
pub const PREPROCESSING_VERSION: u32 = 1;

/// The most images one device batch carries.
pub const MAX_BATCH: usize = 64;

/// The longest `form`, in bytes.
pub const MAX_FORM_BYTES: usize = 64 * 1024;

/// The most fields of one form.
pub const MAX_FORM_FIELDS: usize = 32;

/// The most candidate values over all fields of one form: each is scored by a decoder pass.
pub const MAX_FORM_CANDIDATES: usize = 1024;

/// The most tokens a request may decode (the model's context).
pub const MAX_DECODE_TOKENS: f64 = 1536.0;

/// Bytes of decoder memories one session keeps for reuse between actions.
pub const MEMORY_CACHE_BYTES: usize = 256 << 20;

/// Batches whose sizes the session remembers for [`MirageSession::stats`].
const SIZE_HISTORY: usize = 256;

const SECTIONS: [&str; 7] = ["scene", "objects", "regions", "text", "relations", "events", "reasoning"];
const FIELDS: [&str; 6] = ["bbox", "score", "mask", "position", "description", "count"];

fn proposal_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("conf", ParamType::Float, "minimum object confidence (default 0.3)").min(0.0).max(capability::bounds::UNIT),
        ParamSpec::new("text_conf", ParamType::Float, "minimum text-line confidence (default 0.3)").min(0.0).max(capability::bounds::UNIT),
        ParamSpec::new("max_objects", ParamType::Int, "most objects considered").min(1.0).max(MAX_OBJECTS as f64),
        ParamSpec::new("max_text_lines", ParamType::Int, "most text lines considered").min(1.0).max(MAX_TEXT as f64),
        ParamSpec::new("max_tokens", ParamType::Int, "decoding budget in tokens (default 512)").min(1.0).max(MAX_DECODE_TOKENS),
        ParamSpec::new("language", ParamType::Enum(LANGUAGES.iter().map(|l| l.to_string()).collect()), "language the model writes in (default en)"),
    ]
}

/// Names of the `precision` parameter and the weights each loads.
pub const PRECISIONS: [&str; 2] = ["fp32", "cq"];

fn precision_param() -> ParamSpec {
    ParamSpec::new("precision", ParamType::Enum(PRECISIONS.map(String::from).to_vec()), "weights to serve: fp32 (the safetensors checkpoint) or cq (the packed deployment archive mirage.cact, 2-4 bits per weight; default fp32 when the checkpoint exists)")
}

/// The weight source a request asks for: `None` leaves the choice to what the model directory holds.
pub fn weight_source_of(inv: &Invocation) -> Result<Option<crate::model::WeightSource>, String> {
    match inv.get_str("precision").as_deref() {
        None => Ok(None),
        Some("fp32") => Ok(Some(crate::model::WeightSource::Safetensors)),
        Some("cq") => Ok(Some(crate::model::WeightSource::Cact)),
        Some(other) => Err(format!("param 'precision' must be one of {}, not '{other}'", PRECISIONS.join(", "))),
    }
}

/// The `describe` action.
pub fn describe_spec() -> ActionSpec {
    let mut params = vec![
        ParamSpec::new("sections", ParamType::Str, &format!("comma-separated sections to return and decode, of {} (default all)", SECTIONS.join(","))),
        ParamSpec::new("fields", ParamType::Str, &format!("comma-separated per-item fields, of {} (default bbox,score,mask,position,description)", FIELDS.join(","))),
        ParamSpec::new("mask_threshold", ParamType::Float, "bitmap and polygon threshold (default 0.5)").min(0.0).max(capability::bounds::UNIT),
        ParamSpec::new("mask_format", ParamType::Enum(["largest", "merged", "polygons", "bitmap"].map(String::from).to_vec()), "mask rendering (default largest)"),
    ];
    params.extend(proposal_params());
    params.push(precision_param());
    ActionSpec::new("describe", "describe an image: scene, objects with boxes, scores and masks, regions, text lines, relations, events and reasoning")
        .params(params)
        .input(BlobSpec::new("image", Media::Image, "the image to describe (HWC f32 in [0,1])").required())
}

/// The `extract` action.
pub fn extract_spec() -> ActionSpec {
    let mut params = vec![ParamSpec::new(
        "form",
        ParamType::Str,
        &format!("JSON object of fields to answer: {{\"name\": {{\"enum\": [...]}} | {{\"type\": \"integer\", \"minimum\": a, \"maximum\": b}} | {{\"type\": \"string\", \"maxLength\": n}}}}; at most {MAX_FORM_BYTES} bytes, {MAX_FORM_FIELDS} fields, {MAX_FORM_CANDIDATES} candidate values"),
    )
    .required()];
    params.extend(proposal_params());
    params.push(precision_param());
    ActionSpec::new("extract", "answer the fields of a form about an image (choice, count and text fields) with the probability of every candidate")
        .params(params)
        .input(BlobSpec::new("image", Media::Image, "the image to read the form from (HWC f32 in [0,1])").required())
}

/// The full, static capability manifest: safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "Mirage visual perception: describe an image (objects, masks, text, relations) or extract a form from it.", vec![describe_spec(), extract_spec()])
}

// ===================== requests =====================

/// One request, checked and decoded.
struct Request {
    rgb: Vec<u8>,
    height: usize,
    width: usize,
    options: Options,
    /// Requests with equal keys share their options and may share a device batch.
    group: String,
    form: Option<Value>,
}

/// The request options an invocation carries, as the typed [`Options`].
fn options_of(inv: &Invocation) -> Result<(Options, String), String> {
    let mut map = Map::new();
    let params = inv.params.as_object().cloned().unwrap_or_default();
    for (name, value) in params {
        match name.as_str() {
            "sections" | "fields" => {
                let text = value.as_str().ok_or_else(|| format!("param '{name}' must be a comma-separated string"))?;
                let items: Vec<Value> = text.split(',').map(|s| Value::String(s.trim().to_string())).collect();
                map.insert(name, Value::Array(items));
            }
            // The form and the weights are not decoding options: the form is its own input and the precision picks the
            // session (see `MirageProvider`), so neither joins the batch group.
            "form" | "precision" => {}
            _ => {
                map.insert(name, value);
            }
        }
    }
    let group = Value::Object(map.clone()).to_string();
    let options = Options::from_json(&group).map_err(|e| e.0)?;
    Ok((options, group))
}

fn check_form(inv: &Invocation) -> Result<Value, String> {
    let text = inv.get_str("form").ok_or("missing required param 'form'")?;
    if text.len() > MAX_FORM_BYTES {
        return Err(format!("param 'form' is {} bytes, at most {MAX_FORM_BYTES}", text.len()));
    }
    let form: Value = serde_json::from_str(&text).map_err(|e| format!("param 'form' is not JSON: {e}"))?;
    let fields = form::parse_form(&form).map_err(|e| format!("param 'form': {e}"))?;
    if fields.len() > MAX_FORM_FIELDS {
        return Err(format!("param 'form' has {} fields, at most {MAX_FORM_FIELDS}", fields.len()));
    }
    let candidates: usize = fields
        .iter()
        .map(|f| match &f.kind {
            FieldKind::Choice(options) => options.len(),
            FieldKind::Count { min, max } => (max - min + 1) as usize,
            FieldKind::Text { .. } => 0,
        })
        .sum();
    if candidates > MAX_FORM_CANDIDATES {
        return Err(format!("param 'form' has {candidates} candidate values, at most {MAX_FORM_CANDIDATES}"));
    }
    Ok(form)
}

/// Declared size of the `image` blob, refused before it is decoded.
fn check_image(inv: &Invocation) -> Result<(), String> {
    let blob = inv.get_blob("image").ok_or("missing required input 'image'")?;
    let dim = |k: &str| blob.meta.get(k).and_then(Value::as_u64).ok_or_else(|| format!("'image' blob missing {k}"));
    let (w, h) = (dim("w")?, dim("h")?);
    if w == 0 || h == 0 || w.saturating_mul(h) > capability::blob::MAX_IMAGE_PIXELS {
        return Err(format!("'image' is {w}x{h}, at most {} pixels", capability::blob::MAX_IMAGE_PIXELS));
    }
    let expected = w * h * 3 * std::mem::size_of::<f32>() as u64;
    if blob.bytes.len() as u64 != expected {
        return Err(format!("'image' payload is {} bytes, a {w}x{h} RGB f32 image is {expected}", blob.bytes.len()));
    }
    Ok(())
}

/// Everything about a request that can be refused without touching the image's pixels or a device: the cheap check
/// a scheduler runs at the door.
pub fn check_request(action: &str, inv: &Invocation) -> Result<(), String> {
    match action {
        "describe" | "extract" => {}
        other => return Err(format!("mirage: unknown action '{other}'")),
    }
    check_image(inv)?;
    weight_source_of(inv)?;
    let (options, _) = options_of(inv)?;
    if action == "extract" {
        check_form(inv)?;
    } else if inv.get_str("form").is_some() {
        return Err("param 'form' belongs to `extract`".to_string());
    }
    let _ = options;
    Ok(())
}

fn request_of(action: &str, inv: &Invocation) -> Result<Request, String> {
    check_request(action, inv)?;
    let (options, group) = options_of(inv)?;
    let form = (action == "extract").then(|| check_form(inv)).transpose()?;
    let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
    let rgb = hwc.iter().map(|&v| if v.is_finite() { (v.clamp(0.0, 1.0) * 255.0).round() as u8 } else { 0 }).collect();
    Ok(Request { rgb, height: h as usize, width: w as usize, options, group, form })
}

// ===================== the session =====================

/// How the session's batches went.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchStats {
    /// Device batches run.
    pub batches: u64,
    /// Images over all batches.
    pub images: u64,
    /// Size of the widest batch.
    pub widest: usize,
    /// Sizes of the most recent batches, oldest first.
    pub sizes: Vec<usize>,
    /// Decoder memories served from the cache instead of the vision half.
    pub memory_hits: u64,
}

/// Mirage on one device: the single implementation of both actions.
pub struct MirageSession {
    model: MirageModel,
    revision: Arc<str>,
    cache: Mutex<MemoryCache>,
    stats: Mutex<BatchStats>,
}

impl MirageSession {
    /// Import `<dir>/{config.json, mirage.safetensors, mirage.cact}` and put the weights on `gpu`, which must have
    /// been created with [`Device::kernels`].
    pub fn load(dir: &str, gpu: Gpu) -> Result<MirageSession, String> {
        let source = crate::model::WeightSource::detect(std::path::Path::new(dir)).map_err(|e| e.to_string())?;
        MirageSession::load_with(dir, gpu, source)
    }

    /// [`MirageSession::load`] from the weights `source` names: with [`crate::model::WeightSource::Cact`] the
    /// deployment archive's CQ matrices stay packed on the device.
    pub fn load_with(dir: &str, gpu: Gpu, source: crate::model::WeightSource) -> Result<MirageSession, String> {
        let loaded = crate::model::load_dir(std::path::Path::new(dir), source).map_err(|e| e.to_string())?;
        let ck = &loaded.checkpoint;
        let host = Mirage::new(ck, &loaded.tokenizer, loaded.charset.clone()).map_err(|e| e.to_string())?;
        let revision: Arc<str> = Arc::from(ck.metadata.source_sha256.as_str());
        let model = MirageModel::new(host, gpu).map_err(|e| e.to_string())?;
        Ok(MirageSession { model, revision, cache: Mutex::new(MemoryCache::new(MEMORY_CACHE_BYTES)), stats: Mutex::new(BatchStats::default()) })
    }

    /// The kernels a session's [`Gpu`] is created with.
    pub fn kernels() -> Vec<(&'static str, &'static str)> {
        Device::kernels()
    }

    /// The model.
    pub fn model(&self) -> &MirageModel {
        &self.model
    }

    /// The revision (weights SHA-256) cached memories are keyed with.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// How the batches went so far.
    pub fn stats(&self) -> BatchStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Entries and bytes of decoder memory held for reuse.
    pub fn cached_memories(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        (cache.len(), cache.held_bytes())
    }

    fn record_batch(&self, size: usize, memory_hits: usize) {
        let mut s = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        s.batches += 1;
        s.images += size as u64;
        s.widest = s.widest.max(size);
        s.memory_hits += memory_hits as u64;
        s.sizes.push(size);
        if s.sizes.len() > SIZE_HISTORY {
            s.sizes.remove(0);
        }
    }

    fn key_of(&self, r: &Request) -> MemoryKey {
        let o = &r.options;
        MemoryKey::new(&r.rgb, r.height, r.width, PREPROCESSING_VERSION, &self.revision, o.conf.unwrap_or(crate::describe::DEFAULT_CONFIDENCE), o.text_conf.unwrap_or(crate::describe::DEFAULT_CONFIDENCE), o.max_objects.unwrap_or(usize::MAX), o.max_text_lines.unwrap_or(usize::MAX))
    }

    /// Run `invs` of one action as device batches: invalid requests fail alone, requests sharing their options share a
    /// batch (of at most [`MAX_BATCH`]), a request cancelled through its token fails with `cancelled`. Results align
    /// with `invs`.
    pub fn run_batch(&self, action: &str, invs: &[Invocation]) -> Vec<ActionResult> {
        let mut results: Vec<Option<ActionResult>> = (0..invs.len()).map(|_| None).collect();
        let mut groups: Vec<(String, Vec<(usize, Request)>)> = Vec::new();
        for (i, inv) in invs.iter().enumerate() {
            if inv.cancel.is_cancelled() {
                results[i] = Some(Err("cancelled".to_string()));
                continue;
            }
            match request_of(action, inv) {
                Err(e) => results[i] = Some(Err(e)),
                Ok(r) => match groups.iter_mut().find(|(g, _)| *g == r.group) {
                    Some((_, members)) => members.push((i, r)),
                    None => groups.push((r.group.clone(), vec![(i, r)])),
                },
            }
        }
        for (_, members) in groups {
            for chunk in members.chunks(MAX_BATCH) {
                let outcomes = match action {
                    "describe" => self.describe_chunk(chunk, invs),
                    _ => self.extract_chunk(chunk, invs),
                };
                for ((i, _), outcome) in chunk.iter().zip(outcomes) {
                    results[*i] = Some(outcome);
                }
            }
        }
        results.into_iter().map(|r| r.expect("every request answered")).collect()
    }

    fn usage(&self, size: usize, tokens: usize) -> Value {
        let t = self.model.last_timings();
        let ms = |s: f64| (s * 1e4).round() / 10.0;
        json!({"images": 1, "batch": size, "decoder_tokens": tokens, "vision_ms": ms(t.vision), "memory_ms": ms(t.memory), "decode_ms": ms(t.decode), "post_ms": ms(t.post)})
    }

    fn describe_chunk(&self, chunk: &[(usize, Request)], invs: &[Invocation]) -> Vec<ActionResult> {
        let images: Vec<Image> = chunk.iter().map(|(_, r)| Image { rgb: &r.rgb, height: r.height, width: r.width }).collect();
        let cancelled = |k: usize| invs[chunk[k].0].cancel.is_cancelled();
        let cap = self.cache.lock().unwrap_or_else(|e| e.into_inner()).held_bytes() < MEMORY_CACHE_BYTES;
        let described = self.model.describe_batch_with(&images, &chunk[0].1.options, &BatchControl { keep_memory: cap, cancelled: &cancelled });
        self.record_batch(chunk.len(), 0);
        described
            .into_iter()
            .zip(chunk)
            .map(|(result, (_, request))| {
                let d = result.map_err(model_error)?;
                if let Some(memory) = d.memory.clone() {
                    self.cache.lock().unwrap_or_else(|e| e.into_inner()).put(self.key_of(request), memory);
                }
                let mut outcome = Outcome::new();
                if let Value::Object(sections) = serde_json::to_value(&d.describe).map_err(|e| e.to_string())? {
                    for (name, value) in sections {
                        outcome = outcome.set(&name, value);
                    }
                }
                Ok(outcome.set("usage", self.usage(chunk.len(), d.decoder_tokens)))
            })
            .collect()
    }

    fn extract_chunk(&self, chunk: &[(usize, Request)], invs: &[Invocation]) -> Vec<ActionResult> {
        let images: Vec<Image> = chunk.iter().map(|(_, r)| Image { rgb: &r.rgb, height: r.height, width: r.width }).collect();
        let forms: Vec<Value> = chunk.iter().map(|(_, r)| r.form.clone().expect("an extract request has a form")).collect();
        let keys: Vec<MemoryKey> = chunk.iter().map(|(_, r)| self.key_of(r)).collect();
        let cached: Vec<Option<Arc<DecoderMemory>>> = {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            keys.iter().map(|k| cache.get(k)).collect()
        };
        let hits = cached.iter().flatten().count();
        let cancelled = |k: usize| invs[chunk[k].0].cancel.is_cancelled();
        let extracted = self.model.extract_batch_with(&images, &forms, &chunk[0].1.options, &cached, &BatchControl { keep_memory: false, cancelled: &cancelled });
        self.record_batch(chunk.len(), hits);
        extracted
            .into_iter()
            .zip(keys)
            .map(|(result, key)| {
                let e = result.map_err(model_error)?;
                if !e.reused {
                    self.cache.lock().unwrap_or_else(|p| p.into_inner()).put(key, e.memory.clone());
                }
                let value = serde_json::to_value(&e.extract).map_err(|e| e.to_string())?;
                let mut usage = self.usage(chunk.len(), 0);
                usage["memory_reused"] = json!(e.reused);
                Ok(Outcome::new().set("form", value["form"].clone()).set("usage", usage))
            })
            .collect()
    }
}

fn model_error(e: ModelError) -> String {
    match e {
        ModelError::Form(FormError::Unsupported(m)) => format!("unsupported: {m}"),
        other => other.to_string(),
    }
}

// ===================== the provider =====================

/// The executable model behind the manifest. Construction is free: the checkpoint loads on the first call and stays
/// resident.
pub struct MirageProvider {
    dir: String,
    hot: Arc<Hot<(String, Option<WeightSource>), MirageSession>>,
}

impl MirageProvider {
    /// `dir` holds the released `config.json`, `mirage.safetensors` and `mirage.cact`.
    pub fn new(dir: impl Into<String>) -> MirageProvider {
        MirageProvider { dir: dir.into(), hot: Arc::new(Hot::new()) }
    }
}

impl Provider for MirageProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let spec = manifest().actions.into_iter().find(|a| a.name == name)?;
        Some(Arc::new(MirageAction { spec, dir: self.dir.clone(), hot: self.hot.clone() }))
    }
}

struct MirageAction {
    spec: ActionSpec,
    dir: String,
    hot: Arc<Hot<(String, Option<WeightSource>), MirageSession>>,
}

impl Action for MirageAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let source = weight_source_of(inv)?;
        // One resident session at a time: a request for other weights replaces it.
        let load = || match source {
            Some(source) => MirageSession::load_with(&self.dir, Gpu::new(&MirageSession::kernels()), source),
            None => MirageSession::load(&self.dir, Gpu::new(&MirageSession::kernels())),
        };
        self.hot
            .try_get((self.dir.clone(), source), load, |s| s.run_batch(&self.spec.name, std::slice::from_ref(inv)).pop().expect("one result per request"))
    }
}

/// The kernel and option names a caller may use, for documentation and tests.
pub fn option_names() -> BTreeMap<&'static str, Vec<&'static str>> {
    BTreeMap::from([("sections", SECTIONS.to_vec()), ("fields", FIELDS.to_vec()), ("mask_format", vec!["largest", "merged", "polygons", "bitmap"])])
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Blob;

    fn image(w: u32, h: u32) -> Blob {
        capability::blob::image_blob(&vec![0.5; (w * h * 3) as usize], w, h, 3)
    }

    fn describe(extra: Value) -> Invocation {
        let mut inv = Invocation::new().blob("image", image(2, 2));
        for (k, v) in extra.as_object().unwrap() {
            inv = inv.set(k, v.clone());
        }
        inv
    }

    #[test]
    fn the_manifest_declares_two_actions_and_no_path_param() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["describe", "extract"]);
        for a in &m.actions {
            assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
            assert!(a.params.iter().all(|p| !p.host_resolved && p.host_env.is_none()), "no param names a host resource");
            assert!(a.params.iter().filter(|p| matches!(p.ty, ParamType::Int | ParamType::Float)).all(|p| p.max.is_some()), "every number has a ceiling");
        }
        assert!(m.actions[1].params.iter().any(|p| p.name == "form" && p.required));
        // The served view is the same manifest: nothing to drop.
        assert_eq!(m.clone().for_serving().actions.iter().map(|a| a.params.len()).collect::<Vec<_>>(), m.actions.iter().map(|a| a.params.len()).collect::<Vec<_>>());
    }

    #[test]
    fn precision_picks_the_weights_and_is_not_a_batching_option() {
        use crate::model::WeightSource;
        let spec = describe_spec();
        for (value, source) in [("fp32", Some(WeightSource::Safetensors)), ("cq", Some(WeightSource::Cact))] {
            let inv = describe(json!({"precision": value}));
            assert!(spec.validate(inv.clone()).is_ok(), "{value}");
            assert_eq!(weight_source_of(&inv), Ok(source));
            // Requests with different precisions are routed to different sessions, never grouped in one batch.
            assert_eq!(options_of(&inv).unwrap().1, options_of(&describe(json!({}))).unwrap().1);
        }
        assert_eq!(weight_source_of(&describe(json!({}))), Ok(None));
        assert!(spec.validate(describe(json!({"precision": "int4"}))).is_err());
        assert!(extract_spec().params.iter().any(|p| p.name == "precision"));
    }

    #[test]
    fn options_are_validated_at_the_boundary_and_name_the_field() {
        let spec = describe_spec();
        assert!(spec.validate(describe(json!({"conf": 1.5}))).is_err(), "above the declared ceiling");
        assert!(spec.validate(describe(json!({"max_tokens": 100_000}))).is_err());
        assert!(spec.validate(describe(json!({"language": "xx"}))).is_err());
        assert!(spec.validate(describe(json!({"weights": "/etc/passwd"}))).is_err(), "unknown param");
        assert!(spec.validate(describe(json!({"path": "/etc/passwd"}))).is_err());
        let ok = describe(json!({"sections": "objects, text", "fields": "bbox,mask", "mask_format": "bitmap", "conf": 0.5}));
        assert!(check_request("describe", &ok).is_ok());
        let err = check_request("describe", &describe(json!({"sections": "objects,bogus"}))).unwrap_err();
        assert!(err.contains("sections") || err.contains("bogus"), "{err}");
        let err = check_request("describe", &describe(json!({"fields": "bbox,color"}))).unwrap_err();
        assert!(err.contains("fields") || err.contains("color"), "{err}");
    }

    #[test]
    fn images_are_bounded_by_their_declared_size() {
        let huge = Blob::new(Media::Image, vec![0; 12]).with_meta(json!({"w": 100_000, "h": 100_000, "c": 3}));
        let err = check_request("describe", &Invocation::new().blob("image", huge)).unwrap_err();
        assert!(err.contains("pixels"), "{err}");
        let empty = Blob::new(Media::Image, Vec::new()).with_meta(json!({"w": 0, "h": 4, "c": 3}));
        assert!(check_request("describe", &Invocation::new().blob("image", empty)).is_err());
        assert!(check_request("describe", &Invocation::new()).is_err(), "the image is required");
        let short = Blob::new(Media::Image, vec![0; 12]).with_meta(json!({"w": 4, "h": 4, "c": 3}));
        assert!(check_request("describe", &Invocation::new().blob("image", short)).unwrap_err().contains("payload"));
    }

    fn extract(form: &str) -> Invocation {
        Invocation::new().blob("image", image(2, 2)).set("form", json!(form))
    }

    #[test]
    fn forms_are_bounded_in_size_fields_candidates_and_shape() {
        assert!(check_request("extract", &extract(r#"{"kind": {"enum": ["cat", "dog"]}, "n": {"type": "integer", "minimum": 0, "maximum": 5}, "t": {"type": "string", "maxLength": 8}}"#)).is_ok());
        assert!(check_request("extract", &extract("not json")).unwrap_err().contains("form"));
        assert!(check_request("extract", &extract("[]")).unwrap_err().contains("form"));
        assert!(check_request("extract", &extract("{}")).is_err(), "a form needs a field");
        let big = format!("{{\"a\": {{\"enum\": [\"{}\"]}}}}", "x".repeat(MAX_FORM_BYTES));
        assert!(check_request("extract", &extract(&big)).unwrap_err().contains("bytes"));
        let many: Map<String, Value> = (0..=MAX_FORM_FIELDS).map(|i| (format!("f{i}"), json!({"enum": ["a", "b"]}))).collect();
        assert!(check_request("extract", &extract(&Value::Object(many).to_string())).unwrap_err().contains("fields"));
        let values: Vec<Value> = (0..200).map(|i| json!(format!("v{i}"))).collect();
        let heavy: Map<String, Value> = (0..8).map(|i| (format!("f{i}"), json!({"enum": values}))).collect();
        assert!(check_request("extract", &extract(&Value::Object(heavy).to_string())).unwrap_err().contains("candidate"));
        let boxy = r#"{"b": {"type": "string", "format": "box"}}"#;
        assert!(check_request("extract", &extract(boxy)).is_err(), "box fields are unsupported");
        assert!(check_request("describe", &describe(json!({"form": "{}"}))).is_err(), "form belongs to extract");
        assert!(check_request("extract", &Invocation::new().blob("image", image(2, 2))).is_err(), "the form is required");
    }

    #[test]
    fn a_missing_checkpoint_is_a_clean_error() {
        let mut reg = capability::Registry::new();
        reg.register(Arc::new(MirageProvider::new("/nonexistent/mirage")));
        let err = reg.run(MODEL, "describe", describe(json!({})), &mut |_| {}).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn an_unknown_option_name_in_the_invocation_is_refused() {
        assert!(check_request("describe", &describe(json!({"max_tokens": 5, "bogus": 1}))).is_err());
    }
}
