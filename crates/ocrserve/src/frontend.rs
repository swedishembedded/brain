// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in turning
// concurrent page requests into batched decoder rows then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The scheduler-facing half of an OCR model: how a page request becomes the
//! prompt's token ids with the vision encoder's rows standing at the image
//! positions ([`model::serve::PromptEmbeds`]), and the decoder engine the
//! scheduler batches those requests onto.
//!
//! The encoder runs once per request; the decode then shares the decoder's
//! weight pass with every other request in flight ([`deepseek2::serve::Engine`]).
//! Admission, chunked prefill, batching, cancellation and streaming are
//! `model::serve` and `llmserve`, not this crate.
//!
//! The anti-repetition filter upstream recommends for long documents (vLLM's
//! no-repeat n-gram, 30 over a 90-token window, table-cell tags exempt) is the
//! scheduler's own [`model::serve::NoRepeatNgram`], set per request.

use capability::{last_user_text, Invocation};
use data::qwen_chat::TemplateFlavor;
use data::qwen_tokenizer::QwenBpe;
use deepseek2::serve::{pipelines, Engine, EngineConfig, Precision};
use deepseek2::DeepseekV2Config;
use gpu_core::Gpu;
use llmserve::{Frontend, ParsedRequest, ScheduledInstance};
use model::kv_tier::KvTier;
use model::serve::{NoRepeatNgram, PromptEmbeds, SampleParams, Scheduler};
use qwen3::chat::ToolChoice;

use crate::{DecoderDevice, DEFAULT_INSTRUCTION, DEFAULT_MAX_NEW, MAX_CONTEXT};

/// Pages decoded together: the batch the decoder's step buffers are sized for.
pub const MAX_BATCH: u32 = 32;
/// Tokens of K/V the pool holds across all pages in flight.
pub const POOL_TOKENS: u32 = 64 * 1024;
/// Prefill rows per forward.
pub const PREFILL_CHUNK: u32 = 512;
/// vLLM's recipe for this model family: no 30-gram may repeat within a 90-token window.
pub const NGRAM_SIZE: u32 = 30;
pub const NGRAM_WINDOW: u32 = 90;

/// The decoder engine's shape at the served defaults.
pub fn engine_config() -> EngineConfig {
    EngineConfig { max_seq: MAX_CONTEXT, pool_tokens: POOL_TOKENS, max_batch: MAX_BATCH, chunk: PREFILL_CHUNK, block_size: 64, kv: KvTier::F32, topk: 64, precision: Precision::Auto }
}

/// The vision half of a request: a page's pixels to the prompt and the rows
/// that stand at its image positions.
pub trait PageEncoder {
    /// The tokenizer prompts are built with and generated ids decode through.
    fn tok(&self) -> &QwenBpe;
    /// Preprocess `hwc` (`h x w` RGB in `[0, 1]`), run the vision encoder and lay
    /// its rows out as the decoder's image block: the prompt's ids for
    /// `instruction` and the rows that stand at its image positions.
    fn encode(&self, instruction: &str, hwc: &[f32], w: u32, h: u32) -> Result<(Vec<u32>, PromptEmbeds), String>;
}

/// An OCR request front end over any [`PageEncoder`].
pub struct OcrFrontend<E: PageEncoder> {
    enc: E,
    label: &'static str,
    stop: Vec<u32>,
    ngram: NoRepeatNgram,
}

impl<E: PageEncoder> OcrFrontend<E> {
    /// `eos` ends a generation; `allow` are the ids exempt from the n-gram filter (table-cell tags).
    pub fn new(enc: E, label: &'static str, eos: u32, allow: &[u32]) -> OcrFrontend<E> {
        OcrFrontend { enc, label, stop: vec![eos], ngram: NoRepeatNgram::new(NGRAM_SIZE, NGRAM_WINDOW, allow) }
    }
}

impl<E: PageEncoder> Frontend for OcrFrontend<E> {
    fn tok(&self) -> &QwenBpe {
        self.enc.tok()
    }
    fn stop(&self) -> &[u32] {
        &self.stop
    }
    fn label(&self) -> &str {
        self.label
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        self.parse_embedded(inv).map(|(req, _)| req)
    }
    fn parse_embedded(&self, inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        let instruction = {
            let t = last_user_text(inv);
            if t.trim().is_empty() {
                DEFAULT_INSTRUCTION.to_string()
            } else {
                t
            }
        };
        let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).clamp(1, MAX_CONTEXT as i64) as usize;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let (ids, embeds) = self.enc.encode(&instruction, &hwc, w, h)?;
        let req = ParsedRequest {
            ids,
            max_new,
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            sampler: SampleParams { no_repeat: self.ngram, ..SampleParams::greedy() },
            seed: 0,
            stops: Vec::new(),
            tool_choice: ToolChoice::Auto,
            flavor: TemplateFlavor::default(),
            thinking_open: false,
            raw: true,
        };
        Ok((req, Some(embeds)))
    }
    /// Building a request runs the vision encoder: it is not run ahead of the request's slot.
    fn encodes_inputs(&self) -> bool {
        true
    }
}

/// A served OCR model: the front end and the scheduler that batches requests onto the decoder engine.
pub type OcrInstance<E> = ScheduledInstance<Engine, OcrFrontend<E>>;

/// The decoder engine on `placement`'s device, from a decoder weight source under brain's names.
pub fn build_engine(cfg: DeepseekV2Config, decoder: &dyn checkpoint::TensorSource, placement: DecoderDevice) -> Result<Engine, String> {
    let kernels = pipelines();
    let gpu = match placement {
        DecoderDevice::Cpu => Gpu::new_cpu(&kernels),
        DecoderDevice::SameCard => Gpu::new(&kernels),
        DecoderDevice::Card(i) => gpu_core::devices::with_gpu(i, || Gpu::new(&kernels)).map_err(|e| format!("decoder on gpu{i}: {e}"))?,
    };
    Engine::new(gpu, cfg, decoder, engine_config())
}

/// Wrap `front` and `engine` in the continuously-batching instance.
pub fn instance<E: PageEncoder>(front: OcrFrontend<E>, engine: Engine) -> OcrInstance<E> {
    ScheduledInstance::new(front, Scheduler::new(engine, MAX_BATCH as usize), MAX_BATCH as usize)
}
