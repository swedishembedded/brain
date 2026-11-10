// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`PerceptionPipeline`]: brain's structured visual-perception surface, over
//! Mirage. One call describes an image (scene, objects with boxes, scores and
//! masks, regions, text lines, relations, events, reasoning) and another
//! answers a form about it (choice, count and text fields with the
//! probability of every candidate).
//!
//! The results are the engine-shaped JSON the model emits, with typed
//! accessors for the parts most callers read. The CLI (`brain do mirage
//! describe`), the HTTP and D-Bus surfaces and the Python client return this
//! same JSON, because every one of them runs `mirage::caps::MirageSession`,
//! the single implementation of both actions.
//!
//! ```no_run
//! let pipe = brain::PerceptionPipeline::from_pretrained("Cactus-Compute/mirage")?;
//! let photo = brain::Image::open("cats.jpg")?;
//! let described = pipe.describe(&photo)?;
//! for object in described.objects() {
//!     println!("{} {:?}", object.name, object.bbox);
//! }
//! let form = serde_json::json!({"animal": {"enum": ["cat", "dog"]}});
//! let answered = pipe.extract(&photo, &form)?;
//! println!("{}", answered.json());
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;

use capability::Invocation;
use mirage::caps::MirageSession;
use serde_json::{json, Map, Value};

use crate::{Device, Error, Image, Result};

/// How a mask is rendered in a description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskFormat {
    /// The largest polygon as a flat `[x, y, ...]` list (the default).
    Largest,
    /// All polygons joined into one ring.
    Merged,
    /// Every polygon.
    Polygons,
    /// Column-major run-length bitmap at the source size.
    Bitmap,
}

impl MaskFormat {
    fn wire(self) -> &'static str {
        match self {
            MaskFormat::Largest => "largest",
            MaskFormat::Merged => "merged",
            MaskFormat::Polygons => "polygons",
            MaskFormat::Bitmap => "bitmap",
        }
    }
}

/// An `f32` setting as the number the caller wrote: `0.7`, not the nearest binary neighbour widened to `f64`.
fn unit(v: f32) -> Value {
    json!(v.to_string().parse::<f64>().unwrap_or(f64::from(v)))
}

/// Optional settings of a request; every unset field keeps the model's default.
/// The same names the `describe` and `extract` actions take as params.
#[derive(Clone, Debug, Default)]
pub struct PerceptionOptions {
    params: Map<String, Value>,
}

impl PerceptionOptions {
    /// The defaults.
    pub fn new() -> PerceptionOptions {
        PerceptionOptions::default()
    }

    fn with(mut self, name: &str, value: Value) -> Self {
        self.params.insert(name.to_string(), value);
        self
    }

    /// Sections to return and decode, of `scene objects regions text relations events reasoning` (`describe` only).
    pub fn sections(self, names: &[&str]) -> Self {
        self.with("sections", json!(names.join(",")))
    }

    /// Per-item fields, of `bbox score mask position description` (`describe` only).
    pub fn fields(self, names: &[&str]) -> Self {
        self.with("fields", json!(names.join(",")))
    }

    /// Minimum object confidence in `[0, 1]`.
    pub fn conf(self, v: f32) -> Self {
        self.with("conf", unit(v))
    }

    /// Minimum text-line confidence in `[0, 1]`.
    pub fn text_conf(self, v: f32) -> Self {
        self.with("text_conf", unit(v))
    }

    /// Most objects considered.
    pub fn max_objects(self, n: usize) -> Self {
        self.with("max_objects", json!(n))
    }

    /// Most text lines considered.
    pub fn max_text_lines(self, n: usize) -> Self {
        self.with("max_text_lines", json!(n))
    }

    /// Decoding budget in tokens.
    pub fn max_tokens(self, n: usize) -> Self {
        self.with("max_tokens", json!(n))
    }

    /// Bitmap and polygon threshold in `[0, 1]` (`describe` only).
    pub fn mask_threshold(self, v: f32) -> Self {
        self.with("mask_threshold", unit(v))
    }

    /// Mask rendering (`describe` only).
    pub fn mask_format(self, f: MaskFormat) -> Self {
        self.with("mask_format", json!(f.wire()))
    }

    /// The language the model writes in (`en es fr pt de it nl pl ru ar zh`).
    pub fn language(self, code: &str) -> Self {
        self.with("language", json!(code))
    }
}

/// What a request cost. The latencies are those of the device batch the
/// request ran in (a request served alone is a batch of one).
#[derive(Clone, Debug, PartialEq)]
pub struct Usage {
    /// Images in the request.
    pub images: u64,
    /// Images in the device batch.
    pub batch: u64,
    /// Tokens the decoder generated (0 for `extract`).
    pub decoder_tokens: u64,
    /// Letterbox, upload, encoder and dense head, in milliseconds.
    pub vision_ms: f64,
    /// Proposal slots and decoder memory, in milliseconds.
    pub memory_ms: f64,
    /// Decoder prefill and steps, in milliseconds.
    pub decode_ms: f64,
    /// Mask rendering, reader and JSON assembly, in milliseconds.
    pub post_ms: f64,
}

impl Usage {
    fn from_json(v: &Value) -> Usage {
        let n = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
        let f = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        Usage { images: n("images"), batch: n("batch"), decoder_tokens: n("decoder_tokens"), vision_ms: f("vision_ms"), memory_ms: f("memory_ms"), decode_ms: f("decode_ms"), post_ms: f("post_ms") }
    }
}

/// A detected object, in source-image pixel coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectedObject {
    /// Reference id relations and regions use.
    pub id: u64,
    /// What the model calls it.
    pub name: String,
    /// `[x1, y1, x2, y2]`, when boxes were requested.
    pub bbox: Option<[i64; 4]>,
    /// Detection score, when scores were requested.
    pub score: Option<f64>,
}

/// A line of text read from the image.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadLine {
    /// The text.
    pub string: String,
    /// `[x1, y1, x2, y2]`, when boxes were requested.
    pub bbox: Option<[i64; 4]>,
    /// Detection score, when scores were requested.
    pub score: Option<f64>,
}

fn bbox_of(v: &Value) -> Option<[i64; 4]> {
    let a = v.get("bbox")?.as_array()?;
    (a.len() == 4).then(|| [0, 1, 2, 3].map(|i| a[i].as_i64().unwrap_or(0)))
}

/// The result of [`PerceptionPipeline::describe`].
#[derive(Clone, Debug, PartialEq)]
pub struct Description {
    json: Value,
    usage: Usage,
}

impl Description {
    /// The engine-shaped result: one key per section that was requested and produced.
    pub fn json(&self) -> &Value {
        &self.json
    }

    /// The result, by value.
    pub fn into_json(self) -> Value {
        self.json
    }

    /// What the request cost.
    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    /// The `objects` section.
    pub fn objects(&self) -> Vec<DetectedObject> {
        let list = self.json.get("objects").and_then(Value::as_array).cloned().unwrap_or_default();
        list.iter()
            .filter_map(|o| Some(DetectedObject { id: o.get("id")?.as_u64()?, name: o.get("name")?.as_str()?.to_string(), bbox: bbox_of(o), score: o.get("score").and_then(Value::as_f64) }))
            .collect()
    }

    /// The `text` section.
    pub fn text_lines(&self) -> Vec<ReadLine> {
        let list = self.json.get("text").and_then(Value::as_array).cloned().unwrap_or_default();
        list.iter().filter_map(|t| Some(ReadLine { string: t.get("string")?.as_str()?.to_string(), bbox: bbox_of(t), score: t.get("score").and_then(Value::as_f64) })).collect()
    }

    /// The `reasoning` section.
    pub fn reasoning(&self) -> Option<&str> {
        self.json.get("reasoning").and_then(Value::as_str)
    }
}

/// The result of [`PerceptionPipeline::extract`].
#[derive(Clone, Debug, PartialEq)]
pub struct Extraction {
    json: Value,
    usage: Usage,
}

impl Extraction {
    /// `{"form": {field: {"value": ..., "probabilities": {candidate: p}}}}`.
    pub fn json(&self) -> &Value {
        &self.json
    }

    /// What the request cost.
    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    /// The answer to one field; `null` when the model abstained.
    pub fn value(&self, field: &str) -> Option<&Value> {
        self.json.get("form")?.get(field)?.get("value")
    }

    /// The probability of every candidate of one field, abstention last.
    pub fn probabilities(&self, field: &str) -> Vec<(String, f64)> {
        let Some(map) = self.json.get("form").and_then(|f| f.get(field)).and_then(|f| f.get("probabilities")).and_then(Value::as_object) else { return Vec::new() };
        map.iter().filter_map(|(k, v)| Some((k.clone(), v.as_f64()?))).collect()
    }
}

/// Mirage on one device.
pub struct PerceptionPipeline {
    session: MirageSession,
}

impl std::fmt::Debug for PerceptionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerceptionPipeline").field("revision", &self.session.revision()).finish()
    }
}

impl PerceptionPipeline {
    /// `builder(model_id).load()`.
    pub fn from_pretrained(model_id: impl AsRef<str>) -> Result<PerceptionPipeline> {
        PerceptionPipeline::builder(model_id).load()
    }

    /// A builder for the optional settings.
    pub fn builder(model_id: impl AsRef<str>) -> PerceptionPipelineBuilder {
        PerceptionPipelineBuilder { model_id: model_id.as_ref().to_string(), device: Device::default(), download_policy: loader::DownloadPolicy::default(), models_dir: None }
    }

    /// The capability manifest of the two actions, reflected from the model.
    pub fn capabilities(&self) -> capability::Manifest {
        mirage::caps::manifest()
    }

    fn run(&self, action: &str, image: &Image, params: &Map<String, Value>) -> Result<(Value, Usage)> {
        let hwc: Vec<f32> = image.pixels().iter().map(|&b| b as f32 / 255.0).collect();
        let mut inv = Invocation::new().blob("image", capability::blob::image_blob(&hwc, image.width(), image.height(), 3));
        for (k, v) in params {
            inv = inv.set(k, v.clone());
        }
        let outcome = self.session.run_batch(action, std::slice::from_ref(&inv)).pop().expect("one result per request").map_err(Error::Backend)?;
        let mut outputs = outcome.outputs;
        let usage = outputs.as_object_mut().and_then(|o| o.remove("usage")).map(|u| Usage::from_json(&u)).unwrap_or_else(|| Usage::from_json(&Value::Null));
        Ok((outputs, usage))
    }

    /// Describe `image` at the model's defaults.
    pub fn describe(&self, image: &Image) -> Result<Description> {
        self.describe_with(image, &PerceptionOptions::default())
    }

    /// [`Self::describe`] with options.
    pub fn describe_with(&self, image: &Image, options: &PerceptionOptions) -> Result<Description> {
        let (json, usage) = self.run("describe", image, &options.params)?;
        Ok(Description { json, usage })
    }

    /// Answer `form` (a JSON object of choice, count and text fields) about `image`.
    pub fn extract(&self, image: &Image, form: &Value) -> Result<Extraction> {
        self.extract_with(image, form, &PerceptionOptions::default())
    }

    /// [`Self::extract`] with options.
    pub fn extract_with(&self, image: &Image, form: &Value, options: &PerceptionOptions) -> Result<Extraction> {
        let mut params = options.params.clone();
        params.insert("form".to_string(), Value::String(form.to_string()));
        let (json, usage) = self.run("extract", image, &params)?;
        Ok(Extraction { json, usage })
    }
}

/// Builds a [`PerceptionPipeline`].
pub struct PerceptionPipelineBuilder {
    model_id: String,
    device: Device,
    download_policy: loader::DownloadPolicy,
    models_dir: Option<std::path::PathBuf>,
}

impl PerceptionPipelineBuilder {
    /// The device to run on.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// The models directory the model id is resolved in and fetched into. Unset uses the default store
    /// (`BRAIN_MODELS_DIR`, then the user's data directory).
    pub fn models_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// How [`Self::load`] may use the network to resolve the model id.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Resolve the model id through the model store (fetching it under the download policy when it is missing) and
    /// load the weights onto the device.
    pub fn load(self) -> Result<PerceptionPipeline> {
        let PerceptionPipelineBuilder { model_id, device, download_policy, models_dir } = self;
        crate::device::apply(&device)?;
        let spec = &mirage::resolver::MirageSpec;
        let assembly = match models_dir {
            Some(dir) => crate::resolve_policy::resolve_with_policy_in(Some(dir), "mirage", spec, &model_id, &BTreeMap::new(), download_policy)?,
            None => crate::resolve_policy::resolve_with_policy("mirage", spec, &model_id, &BTreeMap::new(), download_policy)?,
        };
        let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("mirage: resolved assembly {:?} has no weights role", assembly.id)))?;
        let dir = if weights.is_dir() { weights.as_path() } else { weights.parent().ok_or_else(|| Error::Backend(format!("mirage: {} has no directory", weights.display())))? };
        let session = MirageSession::load(&dir.to_string_lossy(), gpu_core::Gpu::new(&MirageSession::kernels())).map_err(Error::Backend)?;
        Ok(PerceptionPipeline { session })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_use_the_action_param_names() {
        let o = PerceptionOptions::new().sections(&["objects", "text"]).conf(0.5).max_objects(8).mask_format(MaskFormat::Bitmap).language("de");
        assert_eq!(Value::Object(o.params), json!({"sections": "objects,text", "conf": 0.5, "max_objects": 8, "mask_format": "bitmap", "language": "de"}));
        // Each setter overrides exactly its own field.
        let o = PerceptionOptions::new().conf(0.5).conf(0.7);
        assert_eq!(Value::Object(o.params), json!({"conf": 0.7}));
    }

    #[test]
    fn a_description_reads_objects_and_text_back_from_the_json() {
        let d = Description {
            json: json!({"objects": [{"id": 1, "name": "cat", "bbox": [1, 2, 30, 40], "score": 0.9}, {"id": 2, "name": "sofa"}], "text": [{"string": "EXIT", "bbox": [5, 6, 7, 8]}], "reasoning": "two cats"}),
            usage: Usage::from_json(&json!({"images": 1, "batch": 4, "decoder_tokens": 17, "decode_ms": 12.5})),
        };
        assert_eq!(d.objects(), vec![DetectedObject { id: 1, name: "cat".into(), bbox: Some([1, 2, 30, 40]), score: Some(0.9) }, DetectedObject { id: 2, name: "sofa".into(), bbox: None, score: None }]);
        assert_eq!(d.text_lines(), vec![ReadLine { string: "EXIT".into(), bbox: Some([5, 6, 7, 8]), score: None }]);
        assert_eq!(d.reasoning(), Some("two cats"));
        assert_eq!((d.usage().batch, d.usage().decoder_tokens), (4, 17));
    }

    #[test]
    fn an_extraction_reads_values_and_probabilities() {
        let e = Extraction { json: json!({"form": {"animal": {"value": "cat", "probabilities": {"cat": 0.8, "dog": 0.1, "-": 0.1}}, "n": {"value": null, "probabilities": {}}}}), usage: Usage::from_json(&Value::Null) };
        assert_eq!(e.value("animal"), Some(&json!("cat")));
        assert_eq!(e.value("n"), Some(&Value::Null));
        assert_eq!(e.value("missing"), None);
        assert_eq!(e.probabilities("animal").len(), 3);
    }
}
