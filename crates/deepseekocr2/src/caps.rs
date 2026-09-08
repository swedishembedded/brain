// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for DeepSeek-OCR-2: a document image + an
//! instruction in, decoded text out, streamed token by token.
//!
//! One action, `generate`, in the same chat-capable shape v1's
//! `crates/deepseek2ocr/src/caps.rs` and `crates/qwenvl/src/caps.rs` use -
//! `apiserve::catalog::api_caps` only classifies a model chat-capable on that
//! exact quadruple (`messages`/`prompt`, `.streaming()`, a `Media::Text`
//! output).
//!
//! ## Scope, honestly stated
//!
//! * **Global view only.** `crate::preprocess` fits any source image into the
//!   model's one `1024x1024` view; the multi-tile "Gundam" layout needs a SAM
//!   position-embedding resample this crate does not implement yet (M6's
//!   ledger entry) - the composite's row-gather/splice already handles
//!   multiple tiles correctly (`tests/tiny_ref.rs`, `tests/composite.rs`),
//!   only real SAM inference on a tile is blocked.
//! * **Decode is full-recompute, not KV-cached.** `M6`'s real-weight tests
//!   proved `DeepseekV2::generate_greedy_cb` correct end to end through this
//!   composite's splice; `generate_greedy_kv_cb`'s interaction with a splice
//!   built once at construction has not been independently verified for this
//!   crate, so this module does not reach for it yet. [`DEFAULT_MAX_NEW`] is
//!   sized for that cost, the same staged shape v1's own default followed
//!   before ITS KV cache landed (see that module's own doc history).
//! * **Greedy only, batch 1.**

use std::sync::Mutex;

use capability::{last_user_text, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::tokenizer::Tokenizer;
use deepseek2::DeepseekV2Config;
use sam1::SamEncoder;
use serde_json::json;

use crate::encoder::sam_tokens_from_nchw;
use crate::import::{self, Files};
use crate::model::DeepseekOcr2;
use crate::preprocess::{self, Fit};
use crate::prompt::{self, Prompt};
use crate::rows::TileGrid;

/// The catalog id. Case-exact, the real upstream repo - same carve-out
/// reasoning as v1's `MODEL` constant: this crate's weights are exactly one
/// specific checkpoint pair or nothing, never an arbitrary env-named
/// substitute a `brain/<family>` placeholder would abstract over.
pub const MODEL: &str = "deepseek-ai/DeepSeek-OCR-2";

/// `$BRAIN_DEEPSEEKOCR2_DIR` - the directory holding both shipped GGUFs.
pub const DIR_VAR: &str = "BRAIN_DEEPSEEKOCR2_DIR";

/// The instruction the reference model ships with. `<|grounding|>` is
/// semantically load-bearing (M0's ledger; same reserved marker v1 defines),
/// not decoration - it switches the model into box-annotated output.
pub const DEFAULT_INSTRUCTION: &str = "<|grounding|>Convert the document to markdown.";

/// Built context length: BOS + the global view's spliced rows (257: 256
/// query rows + 1 separator) + room for the instruction and the reply.
pub const SEQ_LEN: u32 = 512;

/// Conservative while decode is full-recompute (see this module's header) -
/// each token is a whole pass through 12 MoE decoder layers over the grown
/// sequence, not the `O(1)` step a KV cache would give. Measured on this
/// crate's own release build: a real end-to-end `brain deepseekocr2
/// generate` request with `max_new=16` completes; `max_new=40` did not
/// finish inside a 280 s budget on this box. 16 is the number that
/// completed, not a guess - raise it once decode moves to a KV cache the
/// way v1's `deepseek2::DeepseekV2::generate_greedy_kv` already did (see
/// this module's header), the same staged shape v1's own default followed.
pub const DEFAULT_MAX_NEW: i64 = 16;

fn default_dir() -> String {
    std::env::var(DIR_VAR).unwrap_or_default()
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("generate", "DeepSeek-OCR-2: a document image + an instruction in, decoded text out (greedy, streamed per token)")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(
            ParamSpec::new(
                "prompt",
                ParamType::Str,
                "what to do with the page, in plain English (alternative to messages). The default's leading <|grounding|> \
                 marker turns on grounding mode, pairing recognised text with its bounding box; drop it for plain markdown.",
            )
            .default(json!(DEFAULT_INSTRUCTION)),
        )
        .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(DEFAULT_MAX_NEW)))
        .param(ParamSpec::new("weights", ParamType::Str, "checkpoint DIRECTORY holding both DeepSeek-OCR-2 GGUFs (mmproj + LM)").host_env(DIR_VAR))
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required())
        .output(BlobSpec::new("text", Media::Text, "the decoded document text"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "DeepSeek-OCR-2 -- document image in, text/markdown out. SAM ViT-B feeding a 24-layer Qwen2 GQA \
         resampler under a prefix-LM mask, spliced into the unchanged DeepSeek-V2 MoE decoder. Greedy, \
         batch 1, global view only.",
        vec![generate_spec()],
    )
    .with_max_context_tokens(SEQ_LEN as u64)
}

/// The manifest a resident/scheduled server advertises: the checkpoint
/// directory is service-side configuration, so `weights` (a `host_env`
/// param) is stripped - see `deepseek2ocr::caps::manifest_resident`'s doc for
/// why the CLI-facing and resident manifests are two different things.
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// A built composite plus everything one request needs around it.
///
/// Public so `crates/cli/src/resident_deepseekocr2.rs` can own one directly -
/// the residency adapter and the direct `brain do` provider then run the
/// SAME code and cannot drift about preprocessing, prompt assembly, or the
/// image run they splice into.
pub struct Session {
    dir: String,
    vision_cfg: crate::config::DeepseekOcr2VisionConfig,
    model: DeepseekOcr2,
    /// The SAM tower, held resident across requests (built once from the same
    /// vision expansion the composite's encoder half was built from) - a
    /// fresh `SamEncoder` per request would re-upload ~1.9 GiB of weights on
    /// every call for no reason.
    sam: SamEncoder,
    tok: data::qwen_tokenizer::QwenBpe,
    eos: u32,
    gpu_pre: gpu_core::Gpu,
}

impl Session {
    /// Build the whole composite from a checkpoint directory - the global
    /// view's row plan (BOS, then 257 image rows) is fixed at construction,
    /// same shape as v1's `Session::load`.
    pub fn load(dir: &str) -> Result<Session, String> {
        let files = Files::locate(dir)?;
        let decoder_cfg = DeepseekV2Config::deepseek_ocr(1);
        let vision_cfg = import::vision_config(&files.mmproj, decoder_cfg.shape.d_model)?;

        let lm = files.lm.to_string_lossy().into_owned();
        let tok = prompt::tokenizer_from_gguf(&lm)?;
        let eos = tok.special_id(prompt::EOS).ok_or_else(|| format!("this tokenizer has no reserved {:?} token", prompt::EOS))?;

        let n_rows = vision_cfg.encoder.n_query_global + 1; // the global view's rows, plus the separator
        let shape = Self::build_prompt(&tok, &vision_cfg, DEFAULT_INSTRUCTION, n_rows)?;

        // `vision_init` is read twice (SAM's tensors, then the resampler's) -
        // `WeightReader` is a random-access source keyed by tensor name, not
        // a single-pass stream, the same reuse `tests/real_weight.rs` relies
        // on for the identical pair of constructions.
        let vision_init = import::vision_reader(&files)?;
        let decoder_init = import::decoder_reader(&files)?;
        let gpu_sam = gpu_core::Gpu::new_cpu(sam1::PIPELINES);
        let sam = SamEncoder::new_inference(gpu_sam, vision_cfg.sam.clone(), &vision_init, 0);
        let gpu_vision = gpu_core::Gpu::new_cpu(crate::encoder::PIPELINES);
        let gpu_decoder = gpu_core::Gpu::new_cpu(deepseek2::PIPELINES);
        let model = DeepseekOcr2::new_on(gpu_vision, gpu_decoder, vision_cfg.clone(), decoder_cfg.clone(), &vision_init, &decoder_init, TileGrid::none(), SEQ_LEN, shape.row0, false);
        drop(vision_init);
        drop(decoder_init);

        let gpu_pre = gpu_core::Gpu::new_cpu(preprocess::PIPELINES);
        Ok(Session { dir: dir.to_string(), vision_cfg, model, sam, tok, eos, gpu_pre })
    }

    pub fn dir(&self) -> &str {
        &self.dir
    }

    fn build_prompt(tok: &data::qwen_tokenizer::QwenBpe, vision_cfg: &crate::config::DeepseekOcr2VisionConfig, instruction: &str, n_rows: u32) -> Result<Prompt, String> {
        let _ = vision_cfg;
        prompt::build_prompt(tok, "", &format!("\n{instruction}"), n_rows)
    }

    /// Run one `generate` invocation.
    pub fn generate(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let instruction = {
            let t = last_user_text(inv);
            if t.trim().is_empty() {
                DEFAULT_INSTRUCTION.to_string()
            } else {
                t
            }
        };
        let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).clamp(1, SEQ_LEN as i64) as u32;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;

        let n_rows = self.vision_cfg.encoder.n_query_global + 1;
        let prompt = Self::build_prompt(&self.tok, &self.vision_cfg, &instruction, n_rows)?;
        if prompt.image_run() != self.model.image_run() {
            return Err(format!(
                "deepseek-ocr-2 generate: this instruction moves the image run to {:?}, but the splice is sized at {:?}",
                prompt.image_run(),
                self.model.image_run()
            ));
        }
        if prompt.len() + max_new as usize > SEQ_LEN as usize {
            return Err(format!(
                "deepseek-ocr-2 generate: prompt ({} tokens) + max_new ({max_new}) exceeds this model's built context {SEQ_LEN}",
                prompt.len()
            ));
        }

        // Real preprocessing: any extent -> the model's [3, S, S] global view.
        let image = preprocess::preprocess_image(&self.gpu_pre, &self.vision_cfg, &hwc, w, h, Fit::Pad);

        // Real SAM, on the caller's own image, through the resident tower -
        // the composite's forward downstream of this token grid is what
        // M3-M5 already gradient-checked against a synthetic grid; this is
        // where a real image enters that graph for the first time.
        self.sam.write_image(&image);
        self.sam.forward();
        let sam_nchw = self.sam.gpu.read(self.sam.output(), self.sam.out_len());
        let sam_tokens = sam_tokens_from_nchw(&sam_nchw, self.vision_cfg.sam.compress_out as usize, self.vision_cfg.encoder.n_query_global as usize);
        let _ = self.model.prime_vision(&[], &sam_tokens);

        progress(Progress::step(0, max_new, "generating"));
        let mut ids: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut step = 0u32;
        let mut stopped = false;
        let out = self.model.decoder().generate_greedy_cb(&prompt.ids, max_new, |tok_id| {
            step += 1;
            if stopped {
                return;
            }
            if tok_id == self.eos {
                stopped = true;
                return;
            }
            ids.push(tok_id);
            let full = self.tok.decode(&ids);
            let (delta, np) = qwen3::chat::stream_delta(&printed, &full);
            printed = np;
            if !delta.is_empty() {
                progress(Progress::token(step, max_new, delta));
            }
        });
        debug_assert_eq!(out.len(), prompt.len() + max_new as usize);

        let text = self.tok.decode(&ids);
        let finish = if stopped { "stop" } else { "length" };
        let completion = ids.len() + usize::from(stopped);
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(Outcome::new()
            .set("text", json!(text.clone()))
            .set("prompt_tokens", json!(prompt.len() as i64))
            .set("completion_tokens", json!(completion as i64))
            .set("finish_reason", json!(finish))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

pub struct DeepseekOcr2Provider {
    dir: String,
}

impl DeepseekOcr2Provider {
    pub fn from_env() -> Option<DeepseekOcr2Provider> {
        Self::new(default_dir())
    }
    pub fn new(dir: impl Into<String>) -> Option<DeepseekOcr2Provider> {
        let dir = dir.into();
        if dir.is_empty() {
            return None;
        }
        match Files::locate(&dir) {
            Ok(_) => Some(DeepseekOcr2Provider { dir }),
            Err(e) => {
                eprintln!("brain: deepseek-ocr-2 not served ({e})");
                None
            }
        }
    }
}

impl Provider for DeepseekOcr2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<std::sync::Arc<dyn Action>> {
        (name == "generate").then(|| std::sync::Arc::new(GenerateAction { dir: self.dir.clone() }) as std::sync::Arc<dyn Action>)
    }
}

/// One process-wide session, keyed by checkpoint directory.
static RESIDENT: Mutex<Option<Session>> = Mutex::new(None);

struct GenerateAction {
    dir: String,
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        generate_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(|| self.dir.clone());
        if dir.is_empty() {
            return Err(format!("deepseek-ocr-2 generate: no checkpoint directory (set 'weights' or ${DIR_VAR})"));
        }
        let mut guard = RESIDENT.lock().map_err(|_| "deepseek-ocr-2: resident lock poisoned")?;
        if !matches!(&*guard, Some(s) if s.dir == dir) {
            *guard = None;
            *guard = Some(Session::load(&dir)?);
        }
        guard.as_ref().expect("just built").generate(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_is_chat_capable_shaped() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.model, "deepseek-ai/DeepSeek-OCR-2");
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "generate");
        assert!(a.streaming);
        assert!(a.params.iter().any(|p| p.name == "messages"));
        assert!(a.params.iter().any(|p| p.name == "prompt"));
        assert!(a.params.iter().any(|p| p.name == "max_new"));
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert!(a.outputs.iter().any(|b| b.name == "text" && b.media == Media::Text));
    }

    #[test]
    fn an_unset_directory_yields_no_provider() {
        assert!(DeepseekOcr2Provider::new("").is_none());
        assert!(DeepseekOcr2Provider::new("/definitely/not/a/deepseek-ocr-2/dir").is_none());
    }

    #[test]
    fn only_generate_resolves() {
        let p = DeepseekOcr2Provider { dir: "/tmp".into() };
        assert!(p.action("generate").is_some());
        assert!(p.action("segment").is_none());
    }
}
