// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for DeepSeek-OCR: a document image + an instruction
//! in, decoded text out, streamed token by token.
//!
//! One action, `generate`, in the SAME chat-capable shape `crates/qwenvl/src/
//! caps.rs` and `crates/omni/src/caps.rs` use (`messages`/`prompt`,
//! `.streaming()`, a `Media::Text` output). That is required, not conventional:
//! `apiserve::catalog::api_caps` classifies a model chat-capable only on that
//! exact quadruple, and both HTTP handlers always populate `messages`, never a
//! bare `prompt`.
//!
//! ## What is real here
//!
//! * **Real preprocessing.** [`crate::preprocess::preprocess_image`] turns the
//!   request's decoded pixels into the `[3, 1024, 1024]` normalized tensor with
//!   the checkpoint's own `mean = std = 0.5` affine and the reference's
//!   aspect-preserving fit-and-pad - not a stretch, not a borrowed CLIP
//!   normalization.
//! * **The real 273-row prompt.** [`crate::prompt::build_prompt`] assembles
//!   `BOS ++ <image>×273 ++ text` with the LM GGUF's own tokenizer, and the
//!   composite is built through [`crate::DeepseekOcr::new_with_prompt`], so the
//!   16 `image_newline` rows and the one `view_separator` row carry the mmproj's
//!   learned vectors rather than being 17 missing rows.
//! * **Real per-token streaming**, off
//!   [`crate::DeepseekOcr::generate_greedy_kv_from_prompt_cb`], diffed with
//!   `qwen3::chat::stream_delta` so a multi-byte character never escapes
//!   half-decoded.
//! * **Real token accounting.** `prompt_tokens` / `completion_tokens` /
//!   `finish_reason` are set explicitly, because `apiserve::bridge::read_outcome`
//!   defaults them to `0`/`0`/`"stop"` when absent - i.e. an action that omits
//!   them reports zero usage over the OpenAI and Anthropic surfaces. (Both
//!   `omni::caps` and `qwenvl::caps` currently do omit them. That is a bug to
//!   fix there, not a precedent to copy.)
//!
//! ## What is not
//!
//! * **The decode loop does not stop at EOS.** `generate_greedy_kv_from_prompt_cb`
//!   always runs `max_new` steps (each one now `O(1)`, not a full recompute);
//!   this module truncates the result at the first end-of-sentence id and
//!   reports `finish_reason = "stop"`, but the *wall time* is always `max_new`
//!   steps. Early termination is a `crates/deepseekv2` change (the callback
//!   there is synchronous and infallible), not something a serving wrapper can
//!   fake.
//! * **Greedy only, batch 1, one contiguous image run.** No sampling, no
//!   Base/Gundam multi-tile layout (the decoder splice takes one run). The
//!   decode IS now KV-cached (`DeepseekV2::generate_greedy_kv`) - the prompt
//!   pays one batched forward, every generated token after that is one
//!   incremental step, not a full re-run of the whole sequence so far.
//! * **Split backend.** [`Session::load`] builds the vision encoder
//!   (SAM+CLIP+glue) on `gpu_core::Gpu::new_wgpu` and the decoder on
//!   `gpu_core::Gpu::new_cpu`, regardless of the ambient device selection.
//!   `crates/sam1`'s tower used to corrupt its per-block buffers on wgpu at
//!   1024x1024 with three or more blocks; that is fixed and confirmed at
//!   real-weight scale (see `crates/sam1/tests/wgpu_real_weight_parity.rs`),
//!   which is what let the vision half move off the CPU backend.

use std::sync::Mutex;

use capability::{
    last_user_text, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
    Provider,
};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use serde_json::json;

use crate::config::DeepseekOcrConfig;
use crate::import::{self, Files};
use crate::model::DeepseekOcr;
use crate::preprocess::{self, Fit};
use crate::prompt::{self, Prompt};

/// The catalog id. Case-exact, because it is a real upstream repo
/// (`deepseek-ai/DeepSeek-OCR`) rather than one of the `brain/<family>`
/// placeholders `crates/modelref/src/alias.rs` documents - those exist for
/// models whose weights come from an arbitrary env-named checkpoint, which this
/// one's do not: it is exactly the `ggml-org/DeepSeek-OCR-GGUF` pair or nothing.
pub const MODEL: &str = "deepseek-ai/DeepSeek-OCR";

/// `$BRAIN_DEEPSEEK_OCR_DIR` - the directory holding BOTH shipped GGUFs. One
/// variable for a multi-file checkpoint, matching `BRAIN_FACENET_DIR` /
/// `BRAIN_CLIP_DIR`.
pub const DIR_VAR: &str = "BRAIN_DEEPSEEK_OCR_DIR";

/// The instruction the reference model ships with, and the one
/// `tests/prompt_real.rs` pinned against the real tokenizer. Used when a request
/// carries neither `messages` nor `prompt`.
pub const DEFAULT_INSTRUCTION: &str = "<|grounding|>Convert the document to markdown.";

/// Built context length: the 273-row image block plus BOS plus a real
/// instruction plus room to generate. A fixed, documented budget (the same
/// sizing philosophy `qwenvl::caps::SEQ_LEN` uses), not the checkpoint's 8192
/// architectural ceiling - every extra row costs a `[seq, 129280]` logit slab,
/// and this model has no KV cache to amortise it.
pub const SEQ_LEN: u32 = 512;

/// Default generated-token budget. Deliberately small: each token is one FULL
/// recompute of the sequence through 12 MoE layers - **~22 s measured** on 22
/// CPU cores at the served context, so this default is already ~12 minutes.
pub const DEFAULT_MAX_NEW: i64 = 32;

/// `$BRAIN_DEEPSEEK_OCR_DIR`, or empty.
fn default_dir() -> String {
    std::env::var(DIR_VAR).unwrap_or_default()
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new(
        "generate",
        "DeepSeek-OCR: a document image + an instruction in, decoded text out (greedy, streamed per token)",
    )
    .streaming()
    .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
    .param(ParamSpec::new("prompt", ParamType::Str, "a raw instruction (alternative to messages)").default(json!(DEFAULT_INSTRUCTION)))
    .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(DEFAULT_MAX_NEW)))
    .param(
        ParamSpec::new("weights", ParamType::Str, "checkpoint DIRECTORY holding both DeepSeek-OCR GGUFs (mmproj + LM)")
            .default(json!(default_dir())),
    )
    .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required())
    .output(BlobSpec::new("text", Media::Text, "the decoded document text"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "DeepSeek-OCR -- document image in, text/markdown out. DeepEncoder (SAM ViT-B + 16x compressor \
         -> CLIP-L) spliced into a DeepSeek-V2 MoE decoder. Greedy, batch 1, vision on wgpu, decoder on CPU.",
        vec![generate_spec()],
    )
    .with_max_context_tokens(SEQ_LEN as u64)
}

/// A built composite plus everything one request needs around it.
///
/// Public so `crates/cli/src/resident_deepseekocr.rs` can own one directly:
/// the residency adapter and the direct `brain do` provider then run the SAME
/// code, and cannot drift about preprocessing, prompt assembly or token counts.
pub struct Session {
    dir: String,
    cfg: DeepseekOcrConfig,
    model: DeepseekOcr,
    tok: QwenBpe,
    /// The end-of-sentence id the generated ids are truncated at.
    eos: u32,
    /// The device the preprocessor dispatches on - its own handle, because its
    /// kernel list ([`preprocess::PIPELINES`]) is not any model stage's.
    pre: gpu_core::Gpu,
}

impl Session {
    /// Build the whole composite from a checkpoint directory. Minutes, and a
    /// ~22 GiB peak - this is the call `ResidentModel::activate` makes once.
    ///
    /// **The decoder is forced onto the CPU backend**, with `Gpu::new_cpu`, not
    /// by mutating `BRAIN_DEVICE`: this object lives for the life of a server
    /// process, and a process-global env write from inside one model's
    /// activation would silently change the backend every *other* resident
    /// builds on afterwards. It has no wgpu-corruption reason to move (that bug
    /// was `crates/sam1`'s tower, not the decoder) and no measured wgpu benefit
    /// either, so it stays put.
    ///
    /// **The vision encoder (SAM+CLIP+glue) now builds on `Gpu::new_wgpu`.**
    /// `crates/sam1`'s known wgpu corruption at 1024x1024 with three or more
    /// blocks (what pinned this whole model to the CPU backend originally) is
    /// fixed and confirmed at real-weight scale (`crates/sam1/tests/
    /// wgpu_real_weight_parity.rs`, `wgpu_block_count_corruption.rs`) - a prior
    /// pass measured the isolated CPU-vs-wgpu gap on this tower at ~3.6x, so
    /// moving it is a real per-page win, not a defensive no-op. The vision
    /// tower and the decoder are already separate `Gpu` handles - the splice
    /// crosses them as a host `Vec<f32>` (`DeepseekOcr::encode_block`), never a
    /// raw device buffer - so giving them different backends is a
    /// device-selection change, not an architectural one.
    pub fn load(dir: &str) -> Result<Session, String> {
        let t0 = std::time::Instant::now();
        let files = Files::locate(dir)?;
        let cfg = import::config(&files, 1)?;
        let tok = import::tokenizer(&files)?;
        let eos = tok.special_id(prompt::EOS).ok_or_else(|| format!("this tokenizer has no reserved {:?} token", prompt::EOS))?;

        // The prompt the splice is SIZED for: text_before is always empty, so
        // `row0` is 1 and `n_rows` is 273 whatever instruction a request
        // carries. Requests only vary the text AFTER the image block, which
        // does not move the run -- asserted per request by
        // `generate_greedy_from_prompt_cb` itself.
        let shape = Self::build_prompt(&tok, &cfg, DEFAULT_INSTRUCTION)?;
        stage_time("load: config+tokenizer+prompt", t0);

        let t1 = std::time::Instant::now();
        let dev_vision = |k: &'static [(&'static str, &'static str)]| gpu_core::Gpu::new_wgpu(k);
        let dev_decoder = |k: &'static [(&'static str, &'static str)]| gpu_core::Gpu::new_cpu(k);
        let vision = import::encoder_weights_from(&files.mmproj)?;
        stage_time("load: mmproj import (encoder weights)", t1);
        let t2 = std::time::Instant::now();
        let decoder = import::decoder_reader(&files)?;
        stage_time("load: decoder_reader open", t2);
        let t3 = std::time::Instant::now();
        let model = DeepseekOcr::new_with_prompt_devices(&dev_vision, &dev_decoder, cfg.clone(), &vision, &decoder, 0, SEQ_LEN, &shape, false);
        drop(decoder);
        drop(vision);
        stage_time("load: DeepseekOcr::new_with_prompt_devices (weight upload + tape build)", t3);

        let pre = gpu_core::Gpu::new_cpu(preprocess::PIPELINES);
        stage_time("load: TOTAL", t0);
        Ok(Session { dir: dir.to_string(), cfg, model, tok, eos, pre })
    }

    /// Which checkpoint directory this session was built from.
    pub fn dir(&self) -> &str {
        &self.dir
    }

    /// `BOS ++ <image>×273 ++ "\n" ++ instruction`.
    ///
    /// The newline belongs to the reference's own prompt string
    /// (`"<image>\n<|grounding|>Convert the document to markdown."`), i.e. it
    /// sits between the image block and the instruction - which is exactly the
    /// `text_after` side of `build_prompt`'s split.
    fn build_prompt(tok: &QwenBpe, cfg: &DeepseekOcrConfig, instruction: &str) -> Result<Prompt, String> {
        prompt::build_prompt(tok, "", &format!("\n{instruction}"), cfg.token_grid().0)
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

        let prompt = Self::build_prompt(&self.tok, &self.cfg, &instruction)?;
        if prompt.image_run() != self.model.image_run() {
            // Only reachable if an instruction somehow moved the image block,
            // which `text_before = ""` makes impossible -- but the splice would
            // otherwise land on text rows and still decode, so it is checked.
            return Err(format!(
                "deepseek-ocr generate: this instruction moves the image run to {:?}, but the splice is sized at {:?}",
                prompt.image_run(),
                self.model.image_run()
            ));
        }
        if prompt.len() + max_new as usize > SEQ_LEN as usize {
            return Err(format!(
                "deepseek-ocr generate: prompt ({} tokens, incl. the {}-row image block) + max_new ({max_new}) exceeds this model's context {SEQ_LEN}",
                prompt.len(),
                prompt.n_rows
            ));
        }

        // Real preprocessing: any extent -> [3, 1024, 1024], aspect-preserving
        // fit-and-pad, the checkpoint's own normalization.
        let t_pre = std::time::Instant::now();
        let image = preprocess::preprocess_image(&self.pre, &self.cfg, &hwc, w, h, Fit::Pad);
        stage_time("generate: preprocess", t_pre);

        progress(Progress::step(0, max_new, "generating"));
        let t_gen = std::time::Instant::now();
        // Real per-token deltas: re-decode the running id list each token and
        // emit the UTF-8-safe suffix (`qwen3::chat::stream_delta`), the same
        // loop `qwenvl::caps` runs.
        let mut ids: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut step = 0u32;
        let mut stopped = false;
        let out = self.model.generate_greedy_kv_from_prompt_cb(&image, &prompt, max_new, |tok_id| {
            step += 1;
            if stopped {
                return; // past EOS: the loop still runs, but nothing more is emitted
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
        stage_time("generate: encode+splice+decode (TOTAL)", t_gen);
        // A resident device never drops, so its BRAIN_PROFILE table would
        // otherwise never print -- same pattern `crates/fastvlm`'s caps.rs uses.
        self.model.gpu().dump_profile();

        let text = self.tok.decode(&ids);
        // "stop" when the model emitted EOS inside the budget, "length" when it
        // ran the budget out -- `qwen3::chat::SeqState::finish`'s own rule,
        // minus the tool-call/cancellation arms this model has no notion of.
        let finish = if stopped { "stop" } else { "length" };
        // Completion length is what the model actually produced as the
        // completion (EOS included, as OpenAI counts it), not the budget: the
        // ids past EOS are recompute this loop cannot skip, not output.
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

use crate::stage_time;

/// A stateless provider: it holds only the checkpoint directory, and builds (and
/// caches) the composite on the first `generate` - construction must stay cheap,
/// because `crates/cli/src/catalog.rs` constructs every provider just to list it.
pub struct DeepseekOcrProvider {
    dir: String,
}

impl DeepseekOcrProvider {
    /// `None` when `$BRAIN_DEEPSEEK_OCR_DIR` is unset or does not hold both
    /// shipped GGUFs - advertising a model whose every call would fail is worse
    /// than not advertising it.
    pub fn from_env() -> Option<DeepseekOcrProvider> {
        Self::new(default_dir())
    }

    /// Direct constructor (no env round-trip), for a caller that already has the
    /// path - the same seam every imaging resident exposes.
    pub fn new(dir: impl Into<String>) -> Option<DeepseekOcrProvider> {
        let dir = dir.into();
        if dir.is_empty() {
            return None;
        }
        match Files::locate(&dir) {
            Ok(_) => Some(DeepseekOcrProvider { dir }),
            Err(e) => {
                eprintln!("brain: deepseek-ocr not served ({e})");
                None
            }
        }
    }
}

impl Provider for DeepseekOcrProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<std::sync::Arc<dyn Action>> {
        (name == "generate").then(|| std::sync::Arc::new(GenerateAction { dir: self.dir.clone() }) as std::sync::Arc<dyn Action>)
    }
}

/// One process-wide session, keyed by checkpoint directory - the same shape
/// `qwenvl::caps`'s `RESIDENT` static uses, and for the same reason: this is a
/// ~24 GiB build, so a second one would not fit beside the first.
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
            return Err(format!("deepseek-ocr generate: no checkpoint directory (set 'weights' or ${DIR_VAR})"));
        }
        let mut guard = RESIDENT.lock().map_err(|_| "deepseek-ocr: resident lock poisoned")?;
        if !matches!(&*guard, Some(s) if s.dir == dir) {
            *guard = None; // free the old composite BEFORE building the new one
            *guard = Some(Session::load(&dir)?);
        }
        guard.as_ref().expect("just built").generate(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape `apiserve::catalog::api_caps` classifies as chat-capable.
    /// Weights-free, so it runs everywhere.
    #[test]
    fn the_manifest_is_chat_capable_shaped() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.model, "deepseek-ai/DeepSeek-OCR", "the id is the case-exact upstream repo");
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "generate");
        assert!(a.streaming, "streaming is required for the chat-capable classification");
        assert!(a.params.iter().any(|p| p.name == "messages"));
        assert!(a.params.iter().any(|p| p.name == "prompt"));
        assert!(a.params.iter().any(|p| p.name == "max_new"));
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert!(a.outputs.iter().any(|b| b.name == "text" && b.media == Media::Text));
        assert_eq!(m.max_context_tokens, Some(SEQ_LEN as u64));
    }

    /// An unconfigured provider must not exist at all, rather than exist and
    /// fail every call.
    #[test]
    fn an_unset_directory_yields_no_provider() {
        assert!(DeepseekOcrProvider::new("").is_none());
        assert!(DeepseekOcrProvider::new("/definitely/not/a/deepseek/dir").is_none());
    }

    /// The provider only answers its one action.
    #[test]
    fn only_generate_resolves() {
        // A provider built directly (the ctor's existence check is the env
        // path's job, not this one's).
        let p = DeepseekOcrProvider { dir: "/tmp".into() };
        assert!(p.action("generate").is_some());
        assert!(p.action("segment").is_none());
    }

    /// The image run must not depend on the instruction: `text_before` is empty,
    /// so the block always starts right after BOS and is always 273 rows. This
    /// is what lets ONE resident composite serve every request.
    #[test]
    fn the_image_run_is_instruction_independent() {
        // Checkpoint-free: a toy tokenizer carrying this model's reserved
        // strings is all `build_prompt` needs.
        let gt = checkpoint::gguf::GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("deepseek-v3".into()),
            tokens: vec![prompt::BOS.into(), prompt::EOS.into(), prompt::IMAGE.into(), prompt::GROUNDING.into(), "a".into(), "\n".into()],
            merges: Vec::new(),
            token_types: vec![3, 3, 3, 3, 1, 1],
            bos: Some(0),
            eos: Some(1),
            unk: None,
            pad: None,
        };
        let tok = QwenBpe::from_gguf(&gt).expect("toy tokenizer");
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        let a = Session::build_prompt(&tok, &cfg, DEFAULT_INSTRUCTION).expect("default instruction");
        let (short, long) = (
            Session::build_prompt(&tok, &cfg, "a").expect("short instruction"),
            Session::build_prompt(&tok, &cfg, "aa").expect("longer instruction"),
        );
        for p in [&a, &short, &long] {
            assert_eq!(p.image_run(), (1, 273), "BOS, then the whole 273-row global view -- whatever the instruction");
        }
        // ...and the instruction really does change the prompt, so the run's
        // invariance above is not vacuous. (Only the tail moves; `row0` cannot,
        // because `text_before` is empty by construction.)
        assert!(long.len() > short.len(), "a longer instruction must produce a longer prompt");
        assert!(a.len() + DEFAULT_MAX_NEW as usize <= SEQ_LEN as usize, "the default request must fit the built context");
    }
}
