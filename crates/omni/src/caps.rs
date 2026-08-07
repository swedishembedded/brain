// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability` surface for Qwen3-Omni's Thinker: text generation only.
//!
//! Deliberately scoped to what is actually implemented: `generate` (text
//! prompt in, greedy text out, via `crate::generate::generate_greedy`). No
//! `converse`/`transcribe`/`speak` actions are declared here — those need
//! multimodal input splice, the Talker + code predictor + Code2Wav chained
//! together, and `accept_hidden_layer`/codec-id sampling, none of which are
//! wired into a serving-shaped loop yet (`docs/models/omni/status.md`'s M9
//! entry). Declaring an action whose `run()` can't actually do what its spec
//! promises is worse than not declaring it.
//!
//! `generate` is itself validation-tier, not production: no KV-cache (every
//! new token re-runs the full forward from scratch) and no int8/GPU-sharded
//! residency (`crate::generate`'s own module doc has the full reasoning) —
//! every layer's weights are streamed fresh from the checkpoint via
//! `checkpoint::weightio::WeightReader`, per generated token. Correct, slow.

use std::path::Path;
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use checkpoint::weightio::WeightReader;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;
use serde_json::json;

use crate::config::MoeTextConfig;
use crate::generate::generate_greedy;
use crate::thinker::thinker_pipelines;

/// Model name in the manifest.
pub const MODEL: &str = "brain/omni";

/// The `generate` action schema.
pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("generate", "Qwen3-Omni Thinker: greedy text completion (validation-tier -- no KV-cache; see this module's doc)")
        .param(ParamSpec::new("prompt", ParamType::Str, "the text prompt (encoded with the real BPE tokenizer; no chat template applied)").required())
        .param(ParamSpec::new("max_new_tokens", ParamType::Int, "max tokens to generate").default(json!(32)))
        .output(BlobSpec::new("text", Media::Text, "the generated continuation"))
}

/// The manifest (one `generate` action).
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Qwen3-Omni-30B-A3B Thinker -- text generation only (validation-tier: no KV-cache, no int8/GPU-sharded residency; multimodal input and speech output are not wired into a generation loop yet -- see docs/models/omni/status.md's M9 entry).",
        vec![generate_spec()],
    )
}

/// A loaded Thinker, ready to generate — real weights streamed on demand
/// from `reader`, not resident. `embed_table`/`lm_head` are the two tensors
/// every generated token needs (`thinker.model.embed_tokens.weight`,
/// `thinker.lm_head.weight`, untied — `tie_word_embeddings: false`), kept
/// host-resident once at load time rather than re-read from the mmap on
/// every token (unlike the 48 decoder layers, which really are re-streamed
/// per token — see `crate::generate`'s doc for why).
pub struct OmniInner {
    reader: WeightReader,
    gpu: Gpu,
    cfg: MoeTextConfig,
    tok: QwenBpe,
    embed_table: Vec<f32>,
    lm_head: Vec<f32>,
    eos_ids: Vec<u32>,
}

pub struct OmniProvider {
    inner: Arc<OmniInner>,
}

impl OmniProvider {
    /// Load from a real HF checkpoint directory (sharded or single-file —
    /// `WeightReader::open_hf_dir` handles both). No brain-native import
    /// step involved: this reads the raw checkpoint directly, the same
    /// pattern every real-weight test in this crate already uses.
    pub fn load(dir: &str) -> Result<OmniProvider, String> {
        let reader = WeightReader::open_hf_dir(Path::new(dir)).map_err(|e| format!("omni: open {dir}: {e}"))?;
        let config_json = std::fs::read_to_string(Path::new(dir).join("config.json")).map_err(|e| format!("omni: read config.json: {e}"))?;
        let root: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| format!("omni: parse config.json: {e}"))?;
        let cfg = MoeTextConfig::thinker_from_json(&root);
        let tok = QwenBpe::from_dir(dir)?;
        let embed_table = reader.tensor("thinker.model.embed_tokens.weight").ok_or("omni: missing thinker.model.embed_tokens.weight")?;
        let lm_head = reader.tensor("thinker.lm_head.weight").ok_or("omni: missing thinker.lm_head.weight")?;
        let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].into_iter().filter_map(|s| tok.special_id(s)).collect();
        let gpu = Gpu::new(thinker_pipelines());
        Ok(OmniProvider { inner: Arc::new(OmniInner { reader, gpu, cfg, tok, embed_table, lm_head, eos_ids }) })
    }

    /// The shared inner state — the seam `cli::resident_omni`'s
    /// `ResidentModel` uses to serve the same loaded model without a second
    /// (and much slower) checkpoint open + tokenizer load.
    pub fn inner(&self) -> Arc<OmniInner> {
        self.inner.clone()
    }
}

impl Provider for OmniProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "generate").then(|| Arc::new(GenerateAction { inner: self.inner.clone() }) as Arc<dyn Action>)
    }
}

impl OmniInner {
    /// The shared generate path (also used by the resident adapter,
    /// `crate::resident_omni` — matches `qwen_asr::caps::QwenAsrInner::transcribe`'s
    /// shared-between-Provider-and-resident shape).
    pub fn generate(&self, prompt: &str, max_new: u32) -> (String, Vec<u32>) {
        let prompt_ids = self.tok.encode(prompt);
        let out_ids = generate_greedy(&self.reader, &self.gpu, &self.cfg, &self.embed_table, &self.lm_head, &prompt_ids, max_new, &self.eos_ids);
        let new_ids = out_ids[prompt_ids.len()..].to_vec();
        let text = self.tok.decode(&new_ids);
        (text, new_ids)
    }
}

struct GenerateAction {
    inner: Arc<OmniInner>,
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        generate_spec()
    }
    fn run(&self, inv: &capability::Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let prompt = inv.get_str("prompt").ok_or("omni generate: missing 'prompt'")?;
        let max_new = inv.get_i64("max_new_tokens").unwrap_or(32).clamp(1, 4096) as u32;
        progress(Progress::step(0, max_new, "generating"));
        let (text, new_ids) = self.inner.generate(&prompt, max_new);
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(Outcome::new().set("text", json!(text)).set("tokens", json!(new_ids)).blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}
