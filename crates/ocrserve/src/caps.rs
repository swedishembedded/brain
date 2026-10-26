// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for an OCR model: a document image + an instruction
//! in, decoded text out, streamed token by token.
//!
//! One action, `generate`, in the chat-capable shape `apiserve::catalog::api_caps`
//! requires: it classifies a model chat-capable only on `messages`/`prompt`,
//! `.streaming()` and a `Media::Text` output, and both HTTP handlers always
//! populate `messages`, never a bare `prompt`.
//!
//! * **Real per-token streaming**, diffed with `qwen3::chat::stream_delta` so a
//!   multi-byte character never escapes half-decoded, and **real EOS early
//!   stop**: a page that finishes frees its KV blocks that step.
//! * **Real token accounting.** `prompt_tokens` / `completion_tokens` /
//!   `finish_reason` are set explicitly, because `apiserve::bridge::read_outcome`
//!   defaults them to `0`/`0`/`"stop"` when absent.
//! * **Continuous batching.** Requests in flight share the decoder's weight
//!   pass (`model::serve::Scheduler` over `deepseek2::serve::Engine`), so pages
//!   join and leave the batch between steps.

use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress, Provider};
use residency::Instance;
use serde_json::json;

use crate::{placement_for_process, DecoderDevice, OcrModel, DEFAULT_INSTRUCTION, DEFAULT_MAX_NEW, MAX_CONTEXT};

/// The one `generate` action every OCR model serves.
pub fn generate_spec<M: OcrModel>() -> ActionSpec {
    ActionSpec::new("generate", &format!("{}: a document image + an instruction in, decoded text out (greedy, streamed per token)", M::ID))
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
        .param(ParamSpec::new("weights", ParamType::Str, "checkpoint DIRECTORY; overrides the model-store resolver's own pick when set").host_resolved())
        .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required())
        .output(BlobSpec::new("text", Media::Text, "the decoded document text"))
}

/// The model's static manifest.
pub fn manifest<M: OcrModel>() -> Manifest {
    Manifest::new(M::ID, M::SUMMARY, vec![generate_spec::<M>()]).with_max_context_tokens(MAX_CONTEXT as u64)
}

/// The manifest for the RESIDENT/scheduled service: the checkpoint directory is
/// service-side configuration (resolved through the model store), so the served
/// action carries only real per-request parameters.
pub fn manifest_resident<M: OcrModel>() -> Manifest {
    manifest::<M>().for_serving()
}

/// The served model for one-shot callers (`brain do`): built once from a
/// checkpoint directory, on the cards the process may use.
pub struct Session {
    dir: String,
    inst: Box<dyn Instance>,
}

impl Session {
    /// Build the whole model from a checkpoint directory. The vision encoder
    /// builds on the card [`placement_for_process`] picks and the decoder where
    /// that placement puts it. No device is selected through the environment:
    /// a card is entered with `gpu_core::devices::with_gpu`, a thread-scoped
    /// selection that ends with the call.
    pub fn load<M: OcrModel>(dir: &str) -> Result<Session, String> {
        let placement = placement_for_process(&M::FOOTPRINT);
        let inst = match placement.vision {
            Some(i) => gpu_core::devices::with_gpu(i, || M::load(dir, placement.decoder))?,
            None => M::load(dir, DecoderDevice::Cpu),
        }?;
        Ok(Session { dir: dir.to_string(), inst })
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
/// caches) the model on the first `generate` - construction must stay cheap,
/// because `crates/catalog` constructs every provider just to list it.
pub struct OcrProvider<M: OcrModel> {
    dir: String,
    _model: std::marker::PhantomData<M>,
}

impl<M: OcrModel> OcrProvider<M> {
    /// `None` when `dir` is empty or holds no checkpoint this model loads -
    /// advertising a model whose every call would fail is worse than not
    /// advertising it.
    pub fn new(dir: impl Into<String>) -> Option<OcrProvider<M>> {
        let dir = dir.into();
        if dir.is_empty() {
            return None;
        }
        match M::locate(&dir) {
            Ok(()) => Some(OcrProvider { dir, _model: std::marker::PhantomData }),
            Err(e) => {
                eprintln!("brain: {} not served ({e})", M::ID);
                None
            }
        }
    }
}

impl<M: OcrModel> Provider for OcrProvider<M> {
    fn manifest(&self) -> Manifest {
        manifest::<M>()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "generate").then(|| Arc::new(GenerateAction::<M> { dir: self.dir.clone(), _model: std::marker::PhantomData }) as Arc<dyn Action>)
    }
}

/// One process-wide session per model id: a build is tens of GiB, so a second
/// one would not fit beside the first.
static RESIDENT: Mutex<Vec<(&'static str, Session)>> = Mutex::new(Vec::new());

struct GenerateAction<M: OcrModel> {
    dir: String,
    _model: std::marker::PhantomData<M>,
}

impl<M: OcrModel> Action for GenerateAction<M> {
    fn spec(&self) -> ActionSpec {
        generate_spec::<M>()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(|| self.dir.clone());
        if dir.is_empty() {
            return Err(format!("{} generate: no checkpoint directory (pass 'weights', or configure one through the models directory)", M::ID));
        }
        let mut guard = RESIDENT.lock().map_err(|_| format!("{}: resident lock poisoned", M::ID))?;
        let at = match guard.iter().position(|(id, _)| *id == M::ID) {
            Some(i) if guard[i].1.dir == dir => i,
            found => {
                // Free the old composite BEFORE building the new one.
                if let Some(i) = found {
                    guard.remove(i);
                }
                guard.push((M::ID, Session::load::<M>(&dir)?));
                guard.len() - 1
            }
        };
        guard[at].1.generate(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Footprint, OcrModel};

    struct Toy;
    impl OcrModel for Toy {
        const ID: &'static str = "toy/ocr";
        const SUMMARY: &'static str = "a toy";
        const FOOTPRINT: Footprint = Footprint { vision_device: 1, decoder_device: 1, host_gpu_decoder: 1, host_cpu_decoder: 1 };
        fn locate(_dir: &str) -> Result<(), String> {
            Err("no checkpoint".into())
        }
        fn load(_dir: &str, _placement: DecoderDevice) -> Result<Box<dyn Instance>, String> {
            Err("no checkpoint".into())
        }
    }

    /// The exact shape `apiserve::catalog::api_caps` classifies as chat-capable.
    #[test]
    fn the_manifest_is_chat_capable_shaped() {
        let m = manifest::<Toy>();
        assert_eq!(m.model, "toy/ocr");
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "generate");
        assert!(a.streaming, "streaming is required for the chat-capable classification");
        for p in ["messages", "prompt", "max_new"] {
            assert!(a.params.iter().any(|x| x.name == p), "{p}");
        }
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert!(a.outputs.iter().any(|b| b.name == "text" && b.media == Media::Text));
        assert_eq!(m.max_context_tokens, Some(MAX_CONTEXT as u64));
    }

    /// An unconfigured provider must not exist at all, rather than exist and fail every call.
    #[test]
    fn an_unset_or_unloadable_directory_yields_no_provider() {
        assert!(OcrProvider::<Toy>::new("").is_none());
        assert!(OcrProvider::<Toy>::new("/definitely/not/a/dir").is_none());
    }

    #[test]
    fn only_generate_resolves() {
        let p = OcrProvider::<Toy> { dir: "/tmp".into(), _model: std::marker::PhantomData };
        assert!(p.action("generate").is_some());
        assert!(p.action("segment").is_none());
    }
}
