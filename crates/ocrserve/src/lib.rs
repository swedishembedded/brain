// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in serving
// vision-language OCR models at scale then you can procure our services by
// sending an email to info@swedishembedded.com.

//! What the DeepSeek-OCR generations share to be **served**.
//!
//! DeepSeek-OCR and DeepSeek-OCR-2 differ in their vision encoder, their
//! preprocessing and their checkpoint layout. Everything past that is one
//! thing: a page request becomes prompt ids with the encoder's rows standing at
//! the image positions ([`frontend`]), a paged decoder engine batches those
//! requests ([`frontend::build_engine`]), the model is placed on the offered
//! cards ([`placement`]), and it is advertised as the same `generate` action
//! ([`caps`]) and held by the same residency adapter ([`resident`]). A model
//! implements [`OcrModel`] and gets all of it.

pub mod caps;
pub mod frontend;
pub mod placement;
pub mod policy;
pub mod resident;

use std::path::Path;

use capability::Assembly;
use residency::Instance;

pub use frontend::{build_engine, engine_config, instance, serving_gpu, OcrFrontend, OcrInstance, Encoded, PageEncoder, PageRequest, ReadOptions, MAX_BATCH, NGRAM_SIZE, NGRAM_WINDOW, POOL_TOKENS, PREFILL_CHUNK};
pub use policy::{config, configure, OcrServeConfig, MAX_BATCH_CEILING, OCR_POLICY};
pub use placement::{placement_for_process, plan_placement, plan_replicas, DecoderDevice, Footprint, Placement};

/// The instruction the reference models ship with.
///
/// The leading `<|grounding|>` marker is **semantically load-bearing, not
/// decoration**: it switches the model into grounding mode, where the decoded
/// text carries `<|ref|>`/`<|det|>` spans pairing each piece of recognised text
/// with its bounding box on the page. Dropping it turns that output off, which
/// is why this default keeps it. A request that wants plain markdown with no
/// box spans passes its own `prompt` without the marker.
pub const DEFAULT_INSTRUCTION: &str = "<|grounding|>Convert the document to markdown.";

/// The longest sequence (prompt plus generated) a request may reach: the
/// checkpoint's own architectural ceiling (`max_position_embeddings`). The KV
/// pool is paged ([`POOL_TOKENS`] across all sequences in flight), so this
/// bounds one sequence, not the memory every sequence holds.
pub const MAX_CONTEXT: u32 = 8192;

/// Default generated-token budget: real documents (tables, code listings,
/// multi-column pages) routinely need 500-2000+ output tokens; a page that
/// finishes sooner does not pay for the unused budget (EOS frees its KV blocks
/// that step). A caller that wants a different ceiling passes its own `max_new`.
pub const DEFAULT_MAX_NEW: i64 = 2048;

/// One OCR model generation, as the serving layer needs to know it.
pub trait OcrModel: Send + Sync + 'static {
    /// The catalog id. Case-exact, the real upstream repo.
    const ID: &'static str;
    /// One-paragraph description for the manifest.
    const SUMMARY: &'static str;
    /// What the model holds while hot, measured on the served build.
    const FOOTPRINT: Footprint;
    /// `Ok` when `dir` holds a checkpoint this model can load.
    fn locate(dir: &str) -> Result<(), String>;
    /// Build the whole served model from `dir`: the vision encoder on the ambient
    /// card, the decoder on `placement`.
    fn load(dir: &str, placement: DecoderDevice) -> Result<Box<dyn Instance>, String>;
}

/// The checkpoint DIRECTORY a resolved [`Assembly`]'s `dir` role names.
///
/// The two layouts name it differently and the difference is structural, not a
/// heuristic: an HF checkpoint is ONE directory record, so the role already holds
/// the directory, while a GGUF pair is recorded file-by-file, so the role holds
/// the LM GGUF and the directory is its parent. Testing whether the role itself
/// is a directory tells the two apart without either side knowing which recipe
/// fetched the checkpoint.
pub fn dir_from_assembly(assembly: &Assembly) -> Result<String, String> {
    let role = assembly.roles.get("dir").ok_or_else(|| format!("assembly '{}' has no dir role", assembly.id))?;
    if role.is_dir() {
        return Ok(role.to_string_lossy().into_owned());
    }
    role.parent().map(|p| p.to_string_lossy().into_owned()).ok_or_else(|| format!("{} has no parent directory", role.display()))
}

/// Whether an HF-shaped directory's `config.json` declares `architecture`.
///
/// Content, not filename: a store can hold many `transformers` checkpoints, and
/// the architecture string is the same one `brain_modelstore::plan` gated the
/// download on.
pub fn declares_architecture(dir: &Path, architecture: &str) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else { return false };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return false };
    json["architectures"].as_array().is_some_and(|a| a.iter().any(|v| v.as_str() == Some(architecture)))
}

pub use gpu_core::profile::stage_time;
