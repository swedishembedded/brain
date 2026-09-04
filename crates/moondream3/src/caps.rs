// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for Moondream 3: an image + an instruction in,
//! generated text out.
//!
//! One action, `caption`, in the same shape `crates/fastvlm`'s and
//! `crates/deepseek2ocr`'s use - `messages`/`prompt`, `.streaming()`, a
//! `Media::Text` output - because `apiserve::catalog::api_caps` classifies a
//! model chat-capable on exactly that quadruple.
//!
//! ## What is real here
//!
//! * **The full reference front end.** [`crate::preprocess::overlap_crop_image`]
//!   turns the request's decoded pixels into the global crop plus the `h·w`
//!   local crops, which the ViT encodes and
//!   [`crate::preprocess::build_connector_input`] stitches back into the
//!   `[729, 2·dim]` connector input - not a single resized view.
//! * **The production loader.** [`crate::import::load`], the same function this
//!   crate's real-weight tests call, so a served run and a parity test cannot
//!   disagree about which tensors they loaded.
//! * **int8 by default.** The fp32 build is ~43 GiB at the released config and
//!   loads nowhere; [`crate::model::Precision::Int8`] is ~9 GiB. `precision` is
//!   a request parameter so fp32 stays reachable on a machine that has the
//!   room, but the default is the one that runs.
//!
//! ## What is not
//!
//! * **Decode IS KV-cached** (`MoondreamModel::generate_kv`): the prompt pays
//!   one batched masked forward that also seeds every layer's cache, and each
//!   token after that is one `O(pos)` incremental step rather than a full
//!   `O(T²)` recompute. Gated against the recompute path token-for-token.
//!   `max_new` still defaults low: the steps are cheap but not free, and the
//!   prefill over a 730-row image prefix is not.
//! * **No batching.** `run_batch` is the serial default and the resident says
//!   why: each request has its own image, so the ViT pass is per-request, and
//!   the decoder has no batch axis wired.
//! * **Greedy only**, batch 1, and the region/point/detect heads are recognized
//!   on import but not built.
//! * **No real-weight validation exists in this workspace.** The composed path
//!   is gated by checkpoint-free tests through this same code; token-for-token
//!   agreement with the reference is NOT claimed.

use std::sync::Mutex;

use capability::{
    last_user_text, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
    Provider,
};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use serde_json::json;

use crate::config::MoondreamConfig;
use crate::model::{MoondreamModel, Precision};

/// The catalog id.
///
/// A `brain/` placeholder, not the upstream repo name, because
/// [`DIR_VAR`] points at an ARBITRARY directory: the loader accepts any
/// checkpoint whose `config.json` matches the preview architecture, so the id
/// stands in for "whatever is configured" rather than naming one specific
/// release. `deepseek-ai/DeepSeek-OCR` is the one catalog id that does name its
/// upstream repo, and only because its weights are exactly one shipped GGUF
/// pair or nothing - a distinction `crates/cli/tests/model_ids.rs` enforces.
/// The upstream repo to FETCH from is a separate field, `crates/arch`'s
/// `default_ref`.
pub const MODEL: &str = "brain/moondream3";

/// `$BRAIN_MOONDREAM3_WEIGHTS` - the checkpoint DIRECTORY (`config.json`, the
/// safetensors shards, and `tokenizer.json`).
pub const DIR_VAR: &str = "BRAIN_MOONDREAM3_WEIGHTS";

/// The instruction used when a request carries neither `messages` nor `prompt`.
pub const DEFAULT_PROMPT: &str = "Describe this image.";

/// Built context: the image block (`1 + patches_per_crop` rows) plus room for an
/// instruction and a caption. A fixed, documented budget - every extra row costs
/// a `[seq, vocab]` logit slab and this decoder has no KV cache to amortise it.
pub const SEQ_LEN: u32 = 1 + 729 + 96;

/// Default generated-token budget, deliberately small: each token is a full
/// recompute of the sequence through 24 layers.
pub const DEFAULT_MAX_NEW: i64 = 32;

fn default_dir() -> String {
    std::env::var(DIR_VAR).unwrap_or_default()
}

pub fn caption_spec() -> ActionSpec {
    ActionSpec::new("caption", "Moondream 3: an image + an instruction in, generated text out (greedy, streamed per token)")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a raw instruction (alternative to messages)").default(json!(DEFAULT_PROMPT)))
        .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(DEFAULT_MAX_NEW)))
        .param(
            ParamSpec::new("precision", ParamType::Str, "int8 (default, ~9 GiB) or fp32 (~43 GiB - needs a very large machine)")
                .default(json!("int8")),
        )
        .param(ParamSpec::new("weights", ParamType::Str, "checkpoint DIRECTORY").host_env(DIR_VAR))
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required())
        .output(BlobSpec::new("text", Media::Text, "the generated text"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Moondream 3 -- image in, text out. SigLIP ViT with overlap multi-crop -> connector -> a \
         parallel-block sparse-MoE decoder. Greedy, batch 1, int8 experts by default.",
        vec![caption_spec()],
    )
    .with_max_context_tokens(SEQ_LEN as u64)
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP):
/// the checkpoint directory is service-side configuration ([`DIR_VAR`]), so
/// the served action carries only real per-request parameters - see
/// `glmdsa::caps::manifest_resident`'s doc for why a static, CLI-facing
/// manifest and a stripped resident one are two different things, not one
/// hidden behind deployment state. `crate::resident_moondream3::
/// Moondream3Resident::manifest` calls this rather than [`manifest`].
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// Parse the `precision` parameter. `Err` names the accepted values rather than
/// defaulting - silently building a 43 GiB fp32 model because of a typo is a
/// long wait followed by an allocation failure, not a useful error.
pub fn parse_precision(s: &str) -> Result<Precision, String> {
    match s {
        "int8" => Ok(Precision::Int8),
        "fp32" => Ok(Precision::Fp32),
        other => Err(format!("moondream3 caption: unknown precision '{other}' (expected 'int8' or 'fp32')")),
    }
}

/// One request's decoded pixels: `(hwc, width, height)`, or why they could not
/// be decoded. A per-request `Result` so one malformed blob fails on its own
/// rather than poisoning a shared vision pass.
type DecodedImage = Result<(Vec<f32>, u32, u32), String>;

/// A built model plus everything one request needs around it.
///
/// Public so `crates/cli/src/resident_moondream3.rs` can own one directly: the
/// residency adapter and the direct provider then run the SAME code and cannot
/// drift about preprocessing, prompt assembly or token accounting.
pub struct Session {
    dir: String,
    precision: Precision,
    model: MoondreamModel,
    tok: QwenBpe,
    eos: Option<u32>,
}

impl Session {
    /// Build from a checkpoint directory. Minutes, and (at int8) a ~9 GiB peak -
    /// this is the call `ResidentModel::activate` makes once.
    pub fn load(dir: &str, precision: Precision) -> Result<Session, String> {
        Session::load_on(dir, precision, None)
    }

    /// [`Session::load`] on a chosen physical card (`gpu_core::devices`'
    /// canonical index), or `None` for the CPU backend.
    ///
    /// Placement is a scoped registry selection, never an env write - a
    /// server-lifetime resident must not change the backend every other model
    /// builds on afterwards.
    pub fn load_on(dir: &str, precision: Precision, gpu: Option<u32>) -> Result<Session, String> {
        let path = std::path::Path::new(dir);
        let cfg = MoondreamConfig::from_dir(path)?;
        let (w, _cov) = crate::import::load(path, &cfg)?;
        let tok = QwenBpe::from_dir(dir).map_err(|e| format!("moondream3: tokenizer: {e}"))?;
        // The multi-crop path is what the reference runs, and it is what widens
        // the connector input to 2·dim.
        let conn_in = cfg.connector_in();
        let model = MoondreamModel::new_on(gpu, cfg, w.vision, w.connector, w.decoder, conn_in, SEQ_LEN, precision)?;
        let eos = tok.special_id("<|endoftext|>");
        Ok(Session { dir: dir.to_string(), precision, model, tok, eos })
    }

    pub fn dir(&self) -> &str {
        &self.dir
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Run one `caption` invocation.
    pub fn caption(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        // Vision: the full overlap multi-crop front end.
        let embeds = self.model.image_embeds_from_pixels(&hwc, w, h);
        self.caption_with_embeds(inv, &embeds, progress)
    }

    /// Decode a batch of requests, sharing ONE vision pass.
    ///
    /// The decoder half stays per-request - each has its own prompt, its own
    /// image embeddings and its own KV cache, and the block forward has no batch
    /// dimension - but the ViT does not, and at the released config it is the
    /// dominant per-request cost (1 global + up to 12 local crops of 729
    /// patches). One `SiglipEncoder::encode` over every request's crops replaces
    /// N of them.
    pub fn caption_batch(&self, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        // Decode every image first; a request whose blob is malformed fails on
        // its own rather than poisoning the shared pass.
        let decoded: Vec<DecodedImage> = invs.iter().map(|inv| capability::blob::decode_image(inv, "image")).collect();
        let good: Vec<(&[f32], u32, u32)> =
            decoded.iter().filter_map(|d| d.as_ref().ok()).map(|(px, w, h)| (px.as_slice(), *w, *h)).collect();
        let mut embeds = self.model.image_embeds_from_pixels_batch(&good).into_iter();

        decoded
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                Err(e) => Err(e.clone()),
                Ok(_) => {
                    let e: Vec<f32> = embeds.next().expect("one embedding per successfully decoded image");
                    self.caption_with_embeds(&invs[i], &e, &mut |p| progress(i, p))
                }
            })
            .collect()
    }

    /// The decoder half of [`Self::caption`], given image embeddings that may
    /// have come from a batched vision pass. ONE implementation, so the single
    /// and batched paths cannot drift about prompt assembly or token accounting.
    fn caption_with_embeds(&self, inv: &Invocation, embeds: &[f32], progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let instruction = {
            let t = last_user_text(inv);
            if t.trim().is_empty() {
                DEFAULT_PROMPT.to_string()
            } else {
                t
            }
        };
        let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).clamp(1, SEQ_LEN as i64) as usize;

        // Prompt: bos, then the image rows, then the instruction. The image rows
        // are POSITIONAL - the splice writes the connector output over them - so
        // any in-vocab id works as the placeholder.
        let cfg = self.model.config();
        let n_img = cfg.vision.patches_per_crop() as usize;
        let text = self.tok.encode(&instruction);
        let mut prompt = Vec::with_capacity(1 + n_img + text.len());
        prompt.push(0u32);
        prompt.extend(std::iter::repeat_n(0u32, n_img));
        prompt.extend(text.iter().copied());
        if prompt.len() >= self.model.seq_len() as usize {
            return Err(format!(
                "moondream3 caption: the instruction is too long - {} prompt tokens against a built context of {}",
                prompt.len(),
                self.model.seq_len()
            ));
        }
        let prompt_tokens = prompt.len();

        let budget = max_new.min(self.model.seq_len() as usize - prompt_tokens);
        // The KV path: one batched (masked, so the image prefix stays
        // bidirectional) prefill, then O(pos) steps. Gated against the O(T²)
        // recompute path token-for-token, which is what makes preferring it safe.
        let ids = self.model.generate_kv(&prompt, embeds, budget, self.eos)?;
        for (i, _) in ids.iter().enumerate() {
            progress(Progress::step(i as u32 + 1, budget as u32, ""));
        }
        let out = self.tok.decode(&ids);

        // Real token accounting: `apiserve::bridge::read_outcome` defaults these
        // to 0/0/"stop" when absent, i.e. an action that omits them reports zero
        // usage over the OpenAI and Anthropic surfaces.
        let finish = if ids.len() < budget { "stop" } else { "length" };
        Ok(Outcome::new()
            .set("text", json!(out.clone()))
            .set("prompt_tokens", json!(prompt_tokens))
            .set("completion_tokens", json!(ids.len()))
            .set("finish_reason", json!(finish))
            .blob("text", Blob::new(Media::Text, out.into_bytes())))
    }
}

/// Direct provider: builds (and caches) one [`Session`] per
/// `(directory, precision)` on first use.
pub struct Moondream3Provider;

impl Moondream3Provider {
    pub fn new() -> Moondream3Provider {
        Moondream3Provider
    }
}

impl Default for Moondream3Provider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for Moondream3Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<std::sync::Arc<dyn Action>> {
        (name == "caption").then(|| std::sync::Arc::new(CaptionAction) as std::sync::Arc<dyn Action>)
    }
}

struct CaptionAction;

/// One process-wide resident, keyed by `(dir, precision)` so switching either
/// rebuilds. The provider is registered once, and a ~9 GiB build is not
/// something to repeat per call.
static RESIDENT: Mutex<Option<Session>> = Mutex::new(None);

impl Action for CaptionAction {
    fn spec(&self) -> ActionSpec {
        caption_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(default_dir);
        if dir.is_empty() {
            return Err(format!("moondream3 caption: no checkpoint - set {DIR_VAR} or pass `weights`"));
        }
        let precision = parse_precision(&inv.get_str("precision").unwrap_or_else(|| "int8".to_string()))?;
        let mut guard = RESIDENT.lock().map_err(|_| "moondream3: resident lock poisoned")?;
        if !matches!(&*guard, Some(s) if s.dir() == dir && s.precision() == precision) {
            *guard = None; // drop the old build BEFORE the new one allocates
            *guard = Some(Session::load(&dir, precision)?);
        }
        guard.as_ref().expect("just loaded").caption(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_validates_without_weights() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let a = &m.actions[0];
        assert_eq!(a.name, "caption");
        assert!(a.streaming, "per-token Progress is what gives the perf harness TTFT/ITL");
        assert!(a.inputs.iter().any(|b| b.name == "image"));
        assert!(a.outputs.iter().any(|b| b.name == "text"));
    }

    /// The chat-capable quadruple `apiserve::catalog::api_caps` keys on. Losing
    /// any one of these silently removes the model from `/v1/chat/completions`
    /// while leaving it reachable over D-Bus `Run` - which dispatches any
    /// capability action generically, where the HTTP dialects additionally
    /// filter by action SHAPE. "Served" is not one fact, and the thing that
    /// decides is a bool in an `ActionSpec` builder.
    #[test]
    fn the_action_keeps_the_chat_capable_shape() {
        let a = &manifest().actions[0];
        assert!(a.params.iter().any(|p| p.name == "messages"));
        assert!(a.params.iter().any(|p| p.name == "prompt"));
        assert!(a.streaming);
        assert!(a.outputs.iter().any(|b| b.media == Media::Text));
    }

    #[test]
    fn precision_parses_both_and_names_an_unknown_one() {
        assert_eq!(parse_precision("int8").unwrap(), Precision::Int8);
        assert_eq!(parse_precision("fp32").unwrap(), Precision::Fp32);
        assert!(parse_precision("bf16").unwrap_err().contains("unknown precision"));
    }

    /// The default must be the precision that can actually load. fp32 is ~43 GiB
    /// at the released config; defaulting to it would make every unconfigured
    /// request a long wait ending in an allocation failure.
    #[test]
    fn the_default_precision_is_the_one_that_fits() {
        let spec = caption_spec();
        let p = spec.params.iter().find(|p| p.name == "precision").expect("precision param");
        assert_eq!(p.default.as_ref().and_then(|v| v.as_str()), Some("int8"));
    }

    #[test]
    fn a_missing_checkpoint_is_a_clean_error() {
        let inv = Invocation::new().set("weights", json!("/definitely/not/a/moondream/dir")).blob(
            "image",
            Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1})),
        );
        let err = CaptionAction.run(&inv, &mut |_| {}).err().unwrap_or_default();
        assert!(err.contains("moondream3"), "{err}");
    }

    /// The built context must hold the image block with room to spare, or every
    /// request fails on a prompt that has no instruction in it yet.
    #[test]
    fn the_context_holds_the_image_block_and_a_prompt() {
        let ppc = MoondreamConfig::preview().vision.patches_per_crop();
        assert!(SEQ_LEN > 1 + ppc, "SEQ_LEN {SEQ_LEN} cannot hold the {ppc}-row image block plus bos");
        assert!(SEQ_LEN - (1 + ppc) >= DEFAULT_MAX_NEW as u32, "no room left for the default generation budget");
    }
}
