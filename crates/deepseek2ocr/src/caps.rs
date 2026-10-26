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
//!   DeepEncoder's rows - the 16 `image_newline` rows and the one
//!   `view_separator` row carrying the mmproj's learned vectors - stand at the
//!   image positions ([`crate::serve::PageEncoder`]).
//! * **Continuous batching.** Requests in flight share the decoder's weight
//!   pass: [`crate::serve`] drives `model::serve::Scheduler` over
//!   `deepseek2::serve::Engine`, so pages join and leave the batch between
//!   steps and a short page is not held behind a long one.
//! * **Real per-token streaming**, diffed with `qwen3::chat::stream_delta` so a
//!   multi-byte character never escapes half-decoded, and **real EOS early
//!   stop**: a page that finishes frees its KV blocks that step.
//! * **Real n-gram anti-repetition.** `model::serve::NoRepeatNgram` - upstream's
//!   own `vllm-project/recipes` entry for this model (`ngram_size = 30`,
//!   `window_size = 90`, `<td>`/`</td>` exempt), applied per request.
//! * **Real token accounting.** `prompt_tokens` / `completion_tokens` /
//!   `finish_reason` are set explicitly, because `apiserve::bridge::read_outcome`
//!   defaults them to `0`/`0`/`"stop"` when absent - i.e. an action that omits
//!   them reports zero usage over the OpenAI and Anthropic surfaces.
//!
//! ## What is not
//!
//! * **Greedy only, one contiguous image run.** No sampling, no Base/Gundam
//!   multi-tile layout.
//! * **Both halves on the GPU, on a host that has one.** [`Session::load`]
//!   builds the vision encoder (SAM+CLIP+glue) with `gpu_core::Gpu::new`
//!   and picks the decoder's device with [`plan_placement`] - a real card by
//!   default, the CPU Cranelift JIT when there is none. Both backends are
//!   gated against the same llama.cpp reference
//!   (`crates/deepseek2/tests/{parity,generate}.rs`), so the choice is a
//!   placement decision, not a correctness one.

use std::sync::Mutex;

use capability::{
    Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress,
    Provider,
};
use residency::Instance;
use serde_json::json;

use crate::import::Files;
use crate::serve::OcrInstance;

/// The catalog id. Case-exact, because it is a real upstream repo
/// (`deepseek-ai/DeepSeek-OCR`) rather than one of the `brain/<family>`
/// placeholders `crates/modelref/src/alias.rs` documents - those exist for
/// models whose weights come from an arbitrary env-named checkpoint, which this
/// one's do not: it is exactly the `ggml-org/DeepSeek-OCR-GGUF` pair or nothing.
pub const MODEL: &str = "deepseek-ai/DeepSeek-OCR";

/// The instruction the reference model ships with, and the one
/// `tests/prompt_real.rs` pinned against the real tokenizer. Used when a request
/// carries neither `messages` nor `prompt`.
///
/// The leading [`crate::prompt::GROUNDING`] marker is **semantically
/// load-bearing, not decoration**: it is what switches the model into
/// grounding mode, where the decoded text carries `<|ref|>`/`<|det|>` spans
/// pairing each piece of recognised text with its bounding box on the page.
/// Dropping it does not merely tidy the string up - it turns that output off,
/// which is why this default keeps it despite being a reserved token a person
/// should never have to type. A request that wants plain markdown with no box
/// spans passes its own `prompt` without the marker; see the `prompt` param's
/// own help.
pub const DEFAULT_INSTRUCTION: &str = "<|grounding|>Convert the document to markdown.";

/// The longest sequence (prompt plus generated) a request may reach: the
/// checkpoint's own architectural ceiling (`max_position_embeddings`, 8192 -
/// `DeepseekV2Config::deepseek_ocr`'s shape). The KV pool is paged
/// ([`crate::serve::POOL_TOKENS`] across all sequences in flight), so this
/// bounds one sequence, not the memory every sequence holds.
pub const MAX_CONTEXT: u32 = 8192;

/// Default generated-token budget.
///
/// **This was 128**, calibrated to the OLD fixed 512-token context (the
/// 273-row image block plus BOS plus the instruction left room for at most
/// ~229 generated tokens, so 128 was picked from two real calibration
/// requests to fit comfortably under that ceiling while still producing real
/// multi-sentence markdown - 414 characters measured, with the served
/// median for prefill plus up to 128 KV-cached steps taken at calibration).
///
/// Raised now that [`MAX_CONTEXT`] is no longer capped near 512: real
/// documents (tables, code listings, multi-column pages) routinely need
/// 500-2000+ output tokens, which the old ceiling could never reach
/// regardless of what a caller asked for. 2048 is a provisional operator
/// default, not a throughput claim - decode speed itself (measured per step in the ledger, unrelated to this default) is a separate, tracked
/// follow-up; EOS early stop (`DeepseekV2::generate_greedy_kv_stream`) means
/// a real document that finishes sooner does not pay for the unused budget.
/// A caller that wants a different ceiling passes its own `max_new`.
pub const DEFAULT_MAX_NEW: i64 = 2048;

/// Device bytes the vision tower (SAM ViT-B at 1024² + the 16x compressor +
/// CLIP-L + the projector) holds while hot.
///
/// MEASURED on a Tesla P40, on the card, not inferred from host RSS: a real
/// page through the production [`Session::load`] peaks that card at
/// **6.27 GiB** (`nvidia-smi memory.used`, sampled for the life of the
/// process, with the decoder placed elsewhere so the figure is the tower's
/// alone). 7 GiB rounds that up.
///
/// `crates/catalog/src/resident_deepseekocr.rs` budgets it; the constant lives
/// here because [`plan_placement`] needs it to decide whether one card can
/// hold both halves.
pub const VISION_DEVICE_BYTES: u64 = 7u64 << 30;

/// Device bytes the decoder half holds while hot, at the default context.
///
/// MEASURED the same way, on the same run: **13.25 GiB** on its own Tesla P40
/// at `ctx = 8192`, `chunk = 512` (the served defaults). It decomposes as
/// ~11.4 GiB of weights (2234 tensors, 2.9 B parameters at fp32), ~1.0 GiB of
/// KV cache (12 layers x 2 x 8192 x 1280 fp32), ~0.5 GiB of per-expert MoE
/// activation scratch at a 512-row round, and ~0.3 GiB of attention
/// score/probability slabs. 14 GiB rounds that up.
///
/// A smaller `$BRAIN_DEEPSEEK_OCR_CTX` shrinks only the KV cache, which is
/// the smallest of those four terms - the weights do not move - so this is
/// close to a floor, not an average.
pub const DECODER_DEVICE_BYTES: u64 = 14u64 << 30;

/// Host bytes the whole composite holds once the decoder is on a card:
/// tokenizer, the transient mmproj import, the CPU-side preprocessor and the
/// per-token host work. MEASURED at **2.63 GiB** of VmHWM over a real page.
pub const HOST_BYTES_GPU_DECODER: u64 = 3u64 << 30;

/// Host bytes the whole composite holds with the decoder on the CPU backend -
/// the ~11.4 GiB of fp32 weights that would otherwise be on a card, plus
/// [`HOST_BYTES_GPU_DECODER`]'s working set. MEASURED at **13.51 GiB** of
/// VmHWM over the same page.
pub const HOST_BYTES_CPU_DECODER: u64 = 14u64 << 30;

/// Where [`Session::load`] builds the decoder half.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecoderDevice {
    /// The CPU Cranelift JIT (`gpu_core::Gpu::new_cpu`).
    Cpu,
    /// wgpu on the ambient card - the same one the vision tower resolves to,
    /// so both halves share that card's VRAM.
    SameCard,
    /// wgpu on card `i`, so the decoder's ~14 GiB and the tower's ~6 GiB come
    /// out of two different pools.
    Card(u32),
}

/// Where the two halves of the composite go.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Placement {
    /// The card the vision tower is built on, or `None` when no offered card can hold it.
    pub vision: Option<u32>,
    /// Where the decoder is built.
    pub decoder: DecoderDevice,
}

/// Place the composite on the `cards` it may use, each `(index, usable bytes)`.
///
/// The caller owns which cards those are: a server passes the cards it
/// schedules (`--device`), so a placement can never name a card the
/// residency budgets do not hold - a cost naming an unbudgeted card is
/// refused by the scheduler, and used to be parked in its queue forever.
///
/// 1. The vision tower goes on the card with the most usable bytes that can
///    hold [`VISION_DEVICE_BYTES`] at all (it is a single-image graph and
///    cannot be spread).
/// 2. The decoder goes on another card that can hold [`DECODER_DEVICE_BYTES`]
///    (largest first) so the two pools do not compete, else on the vision card
///    when it holds both, else on the CPU backend - a decoder that does not fit
///    is not one that runs slowly, it is one that fails to allocate.
/// 3. With no card for the tower, everything is host-side.
pub fn plan_placement(cards: &[(u32, u64)]) -> Placement {
    let vision = cards.iter().filter(|&&(_, usable)| usable >= VISION_DEVICE_BYTES).max_by_key(|&&(_, usable)| usable).map(|&(i, _)| i);
    let Some(v) = vision else { return Placement { vision: None, decoder: DecoderDevice::Cpu } };
    let other = cards.iter().filter(|&&(i, usable)| i != v && usable >= DECODER_DEVICE_BYTES).max_by_key(|&&(_, usable)| usable);
    let own = cards.iter().find(|&&(i, _)| i == v).map_or(0, |&(_, usable)| usable);
    let decoder = match other {
        Some(&(i, _)) => DecoderDevice::Card(i),
        None if own >= VISION_DEVICE_BYTES + DECODER_DEVICE_BYTES => DecoderDevice::SameCard,
        None => DecoderDevice::Cpu,
    };
    Placement { vision: Some(v), decoder }
}

/// The placements of every replica the offered `cards` can run at once: one self-contained replica (the
/// vision tower and the decoder on the same card) on each card that holds both, when there are two or
/// more such cards - the spare capacity is then used, and two replicas never wait on each other's card.
/// With fewer, the one placement of [`plan_placement`]. Never empty.
pub fn plan_replicas(cards: &[(u32, u64)]) -> Vec<Placement> {
    let whole: Vec<Placement> =
        cards.iter().filter(|&&(_, usable)| usable >= VISION_DEVICE_BYTES + DECODER_DEVICE_BYTES).map(|&(i, _)| Placement { vision: Some(i), decoder: DecoderDevice::SameCard }).collect();
    if whole.len() >= 2 {
        whole
    } else {
        vec![plan_placement(cards)]
    }
}

/// [`plan_placement`] for a one-shot process that has no scheduler: the cards
/// the process may use are the ambient pin when `--device gpuN` set one, else
/// every discrete card. Capacity, not free memory, because wgpu exposes no
/// portable free-VRAM query; a server uses its budgets instead.
fn placement_for_process() -> Placement {
    let pinned = gpu_core::devices::ambient_gpu();
    let cards: Vec<(u32, u64)> = gpu_core::devices::gpus()
        .iter()
        .filter(|d| d.identity.class == gpu_core::DeviceClass::DiscreteGpu && pinned.is_none_or(|p| p == d.index))
        .map(|d| (d.index, d.identity.vram_bytes))
        .collect();
    plan_placement(&cards)
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new(
        "generate",
        "DeepSeek-OCR: a document image + an instruction in, decoded text out (greedy, streamed per token)",
    )
    .streaming()
    .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
    .param(
        ParamSpec::new(
            "prompt",
            ParamType::Str,
            "what to do with the page, in plain English (alternative to messages). The default's leading <|grounding|> \
             marker is not decoration: it turns on grounding mode, where the output pairs each piece of recognised text \
             with its bounding box. Delete it for plain markdown with no boxes.",
        )
        .default(json!(DEFAULT_INSTRUCTION)),
    )
    .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(DEFAULT_MAX_NEW)))
    .param(
        ParamSpec::new(
            "weights",
            ParamType::Str,
            "checkpoint DIRECTORY holding both DeepSeek-OCR GGUFs (mmproj + LM); overrides the model-store resolver's own pick when set",
        )
        .host_resolved(),
    )
    .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required())
    .output(BlobSpec::new("text", Media::Text, "the decoded document text"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "DeepSeek-OCR -- document image in, text/markdown out. DeepEncoder (SAM ViT-B + 16x compressor \
         -> CLIP-L) spliced into a DeepSeek-V2 MoE decoder. Greedy, continuously batched, both halves on the GPU where one fits.",
        vec![generate_spec()],
    )
    .with_max_context_tokens(MAX_CONTEXT as u64)
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP):
/// the checkpoint directory is service-side configuration (resolved through
/// the model store, see `crate::spec::Deepseek2ocrSpec`), so
/// the served action carries only real per-request parameters - see
/// `glmdsa::caps::manifest_resident`'s doc for why a static, CLI-facing
/// manifest and a stripped resident one are two different things, not one
/// hidden behind deployment state. `crate::resident_deepseekocr::
/// DeepseekOcrResident::manifest` calls this rather than [`manifest`].
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// The served model: encoder, decoder engine and scheduler (`crate::serve`).
///
/// Public so `crates/catalog/src/resident_deepseekocr.rs` and the direct
/// `brain do` provider run the SAME code, and cannot drift about
/// preprocessing, prompt assembly, batching or token counts.
pub struct Session {
    dir: String,
    inst: OcrInstance,
}

impl Session {
    /// Build the whole model from a checkpoint directory. Tens of seconds, and
    /// roughly [`VISION_DEVICE_BYTES`] + [`DECODER_DEVICE_BYTES`] +
    /// [`HOST_BYTES_GPU_DECODER`] once it is up. The vision tower builds on the
    /// ambient card and the decoder where [`plan_placement`] puts it.
    ///
    /// Neither device is selected by mutating `BRAIN_DEVICE`: this object lives
    /// for the life of a server process, and a process-global env write from
    /// inside one model's activation would silently change the backend every
    /// *other* resident builds on afterwards. A specific card is entered with
    /// `gpu_core::devices::with_gpu`, a thread-scoped selection that ends with
    /// the call.
    pub fn load(dir: &str) -> Result<Session, String> {
        let placement = placement_for_process();
        match placement.vision {
            Some(i) => gpu_core::devices::with_gpu(i, || Session::load_with(dir, placement.decoder))?,
            None => Session::load_with(dir, placement.decoder),
        }
    }

    /// [`Session::load`] with the decoder's placement chosen by the caller
    /// rather than re-derived from this process's device registry.
    ///
    /// The scheduler needs this: `crates/catalog/src/resident_deepseekocr.rs`
    /// RESERVES device bytes against a specific set of devices before
    /// activating, and a second, independent placement decision inside the
    /// build could name a different card than the one that was reserved.
    pub fn load_with(dir: &str, placement: DecoderDevice) -> Result<Session, String> {
        Ok(Session { dir: dir.to_string(), inst: crate::serve::load(dir, placement)? })
    }

    /// Which checkpoint directory this session was built from.
    pub fn dir(&self) -> &str {
        &self.dir
    }

    /// Run one `generate` invocation to completion.
    pub fn generate(&mut self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.inst.run("generate", inv, progress)
    }
}

/// A stateless provider: it holds only the checkpoint directory, and builds (and
/// caches) the composite on the first `generate` - construction must stay cheap,
/// because `crates/catalog/src/lib.rs` constructs every provider just to list it.
pub struct DeepseekOcrProvider {
    dir: String,
}

impl DeepseekOcrProvider {
    /// `None` when `dir` is empty or does not hold both shipped GGUFs -
    /// advertising a model whose every call would fail is worse than not
    /// advertising it. `crates/catalog/src/lib.rs` builds `dir` from a
    /// resolved [`capability::Assembly`] (`crate::spec::Deepseek2ocrSpec`'s
    /// `dir` role) rather than an env var.
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
/// `qwen3vl::caps`'s `RESIDENT` static uses, and for the same reason: this is a
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
            return Err("deepseek-ocr generate: no checkpoint directory (pass 'weights', or configure one through the models directory)".to_string());
        }
        let mut guard = RESIDENT.lock().map_err(|_| "deepseek-ocr: resident lock poisoned")?;
        if !matches!(&*guard, Some(s) if s.dir == dir) {
            *guard = None; // free the old composite BEFORE building the new one
            *guard = Some(Session::load(&dir)?);
        }
        guard.as_mut().expect("just built").generate(inv, progress)
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
        assert_eq!(m.max_context_tokens, Some(MAX_CONTEXT as u64));
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

    const GIB: u64 = 1 << 30;

    /// The placement only ever names cards it was offered. A server offers the
    /// cards `--device` schedules; naming any other card made the residency
    /// claim wait for a budget that does not exist.
    #[test]
    fn placement_names_only_offered_cards() {
        let p = plan_placement(&[(0, 90 * GIB)]);
        assert_eq!(p, Placement { vision: Some(0), decoder: DecoderDevice::SameCard });
        let p = plan_placement(&[(1, 90 * GIB)]);
        assert_eq!(p, Placement { vision: Some(1), decoder: DecoderDevice::SameCard }, "an offered card keeps its own index");
    }

    /// Spare cards are used: a replica on every card that holds the whole model, but a single
    /// replica when only one does (its decoder may then go to a second card).
    #[test]
    fn a_replica_runs_on_every_card_that_holds_the_whole_model() {
        let whole = Placement { vision: Some(0), decoder: DecoderDevice::SameCard };
        assert_eq!(plan_replicas(&[(0, 90 * GIB), (1, 90 * GIB)]), vec![whole, Placement { vision: Some(1), decoder: DecoderDevice::SameCard }]);
        assert_eq!(plan_replicas(&[(0, 90 * GIB), (1, 12 * GIB)]), vec![plan_placement(&[(0, 90 * GIB), (1, 12 * GIB)])], "one whole card is one replica");
        assert_eq!(plan_replicas(&[(0, 90 * GIB), (1, 60 * GIB), (3, 8 * GIB)]).len(), 2, "a card that cannot hold the model is left out");
        assert_eq!(plan_replicas(&[]), vec![Placement { vision: None, decoder: DecoderDevice::Cpu }]);
    }

    #[test]
    fn the_decoder_prefers_a_second_offered_card_then_the_vision_card_then_the_cpu() {
        assert_eq!(plan_placement(&[(0, 24 * GIB), (1, 30 * GIB)]), Placement { vision: Some(1), decoder: DecoderDevice::Card(0) });
        assert_eq!(plan_placement(&[(0, 24 * GIB)]), Placement { vision: Some(0), decoder: DecoderDevice::SameCard });
        assert_eq!(plan_placement(&[(0, 10 * GIB)]).decoder, DecoderDevice::Cpu, "the tower fits, the decoder does not");
        assert_eq!(plan_placement(&[(0, 2 * GIB)]), Placement { vision: None, decoder: DecoderDevice::Cpu });
        assert_eq!(plan_placement(&[]), Placement { vision: None, decoder: DecoderDevice::Cpu });
    }
}
