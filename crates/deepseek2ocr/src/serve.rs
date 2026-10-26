// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in turning
// concurrent page requests into batched decoder rows then you can procure our
// services by sending an email to info@swedishembedded.com.

//! DeepSeek-OCR through the generic continuous-batching scheduler.
//!
//! A request is an image and an instruction. [`OcrFrontend`] turns it into what
//! the scheduler batches: the prompt's token ids with the DeepEncoder's rows
//! standing at the image positions ([`model::serve::PromptEmbeds`]). The
//! encoder runs once per request, on the vision card; the decode then shares
//! the decoder's weight pass with every other request in flight
//! ([`deepseek2::serve::Engine`]). Admission, chunked prefill, batching,
//! cancellation and streaming are `model::serve` and `llmserve`, not this crate.
//!
//! The anti-repetition filter upstream recommends for long documents (vLLM's
//! no-repeat n-gram, 30 over a 90-token window, table-cell tags exempt) is the
//! scheduler's own [`model::serve::NoRepeatNgram`], set per request.

use capability::{last_user_text, Invocation};
use data::qwen_chat::TemplateFlavor;
use data::qwen_tokenizer::QwenBpe;
use deepseek2::serve::{pipelines, Engine, EngineConfig};
use gpu_core::Gpu;
use llmserve::{Frontend, ParsedRequest, ScheduledInstance};
use model::kv_tier::KvTier;
use model::serve::{NoRepeatNgram, PromptEmbeds, SampleParams, Scheduler};
use qwen3::chat::ToolChoice;

use crate::caps::{DecoderDevice, DEFAULT_INSTRUCTION, DEFAULT_MAX_NEW, MAX_CONTEXT};
use crate::config::DeepseekOcrConfig;
use crate::encoder::DeepEncoder;
use crate::import::{self, Files};
use crate::layout::RowGather;
use crate::preprocess::{self, Fit};
use crate::prompt::{self, Prompt};
use crate::stage_time;

/// Pages decoded together: the batch the decoder's step buffers are sized for.
pub const MAX_BATCH: u32 = 32;
/// Tokens of K/V the pool holds across all pages in flight.
pub const POOL_TOKENS: u32 = 64 * 1024;
/// Prefill rows per forward: the whole real prompt (BOS, 273 image rows and the
/// instruction) in one pass.
pub const PREFILL_CHUNK: u32 = 512;
/// vLLM's recipe for this model: no 30-gram may repeat within a 90-token window.
pub const NGRAM_SIZE: u32 = 30;
pub const NGRAM_WINDOW: u32 = 90;

/// The decoder engine's shape at the served defaults.
pub fn engine_config() -> EngineConfig {
    EngineConfig { max_seq: MAX_CONTEXT, pool_tokens: POOL_TOKENS, max_batch: MAX_BATCH, chunk: PREFILL_CHUNK, block_size: 64, kv: KvTier::F32, topk: 64 }
}

/// The vision half of a request: image to the DeepEncoder's rows, plus the
/// tokenizer and the prompt layout around them.
pub struct PageEncoder {
    cfg: DeepseekOcrConfig,
    enc: DeepEncoder,
    rows: RowGather,
    /// The CPU-side preprocessor's own device handle (its kernel list is not any model stage's).
    pre: Gpu,
    tok: QwenBpe,
}

impl PageEncoder {
    /// `BOS ++ <image>x273 ++ "\n" ++ instruction`.
    ///
    /// The newline belongs to the reference's own prompt string
    /// (`"<image>\n<|grounding|>Convert the document to markdown."`), i.e. it
    /// sits between the image block and the instruction - which is exactly the
    /// `text_after` side of `build_prompt`'s split.
    pub fn build_prompt(tok: &QwenBpe, cfg: &DeepseekOcrConfig, instruction: &str) -> Result<Prompt, String> {
        prompt::build_prompt(tok, "", &format!("\n{instruction}"), cfg.token_grid().0)
    }

    /// The tokenizer prompts are built with and generated ids decode through.
    pub fn tok(&self) -> &QwenBpe {
        &self.tok
    }

    /// Preprocess `hwc` (`h x w` RGB in `[0, 1]`), run the DeepEncoder and lay
    /// its rows out as the decoder's image block: the prompt for `instruction`
    /// and the rows that stand at its image positions.
    pub fn encode(&self, instruction: &str, hwc: &[f32], w: u32, h: u32) -> Result<(Prompt, PromptEmbeds), String> {
        let prompt = Self::build_prompt(&self.tok, &self.cfg, instruction)?;
        let (row0, n_rows) = prompt.image_run();
        if n_rows != self.rows.rows() {
            return Err(format!("deepseek-ocr: this instruction moves the image block to {n_rows} rows, but the layout is {}", self.rows.rows()));
        }
        // Real preprocessing: any extent -> [3, 1024, 1024], aspect-preserving
        // fit-and-pad, the checkpoint's own normalization.
        let t_pre = std::time::Instant::now();
        let image = preprocess::preprocess_image(&self.pre, &self.cfg, hwc, w, h, Fit::Pad);
        stage_time("encode request: preprocess", t_pre);
        let t_enc = std::time::Instant::now();
        let rows = self.enc.forward_rows(&image, &self.rows);
        stage_time("encode request: DeepEncoder", t_enc);
        let embeds = PromptEmbeds::run(self.cfg.decoder.d_model() as usize, row0, rows)?;
        Ok((prompt, embeds))
    }
}

/// The scheduler-facing half: how an OCR request becomes a prompt, and how its
/// tokens are read back.
pub struct OcrFrontend {
    enc: PageEncoder,
    stop: Vec<u32>,
    ngram: NoRepeatNgram,
}

impl Frontend for OcrFrontend {
    fn tok(&self) -> &QwenBpe {
        self.enc.tok()
    }
    fn stop(&self) -> &[u32] {
        &self.stop
    }
    fn label(&self) -> &str {
        "deepseek-ocr"
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
        let (prompt, embeds) = self.enc.encode(&instruction, &hwc, w, h)?;
        let req = ParsedRequest {
            ids: prompt.ids,
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
    /// Building a request runs the DeepEncoder: it is not run ahead of the request's slot.
    fn encodes_inputs(&self) -> bool {
        true
    }
}

/// DeepSeek-OCR resident: the encoder, the decoder engine and the scheduler
/// that batches requests onto it.
pub type OcrInstance = ScheduledInstance<Engine, OcrFrontend>;

/// Build the whole served model from `dir` (the GGUF pair or the Hugging Face
/// checkpoint): the vision tower on the ambient card, the decoder on `placement`.
pub fn load(dir: &str, placement: DecoderDevice) -> Result<OcrInstance, String> {
    let t0 = std::time::Instant::now();
    let files = Files::locate(dir)?;
    let cfg = import::config(&files, 1)?;
    let tok = import::tokenizer(&files)?;
    let eos = tok.special_id(prompt::EOS).ok_or_else(|| format!("this tokenizer has no reserved {:?} token", prompt::EOS))?;
    let allow: Vec<u32> = [prompt::TD_OPEN, prompt::TD_CLOSE].into_iter().filter_map(|s| tok.special_id(s)).collect();
    let shape = PageEncoder::build_prompt(&tok, &cfg, DEFAULT_INSTRUCTION)?;
    stage_time("load: config+tokenizer+prompt", t0);

    eprintln!("deepseek-ocr: vision on the ambient card, decoder on {placement:?}");
    let t1 = std::time::Instant::now();
    let vision = import::encoder_weights_for(&files, &cfg)?;
    let dev_vision = |k: &'static [(&'static str, &'static str)]| Gpu::new(k);
    let enc = DeepEncoder::new(&dev_vision, cfg.clone(), &vision, 0, false);
    drop(vision);
    let rows = enc.row_gather(&shape.plan.rows);
    stage_time("load: DeepEncoder (mmproj import, weight upload)", t1);

    let t2 = std::time::Instant::now();
    let reader = import::decoder_reader(&files, &cfg)?;
    let decoder = import::decoder_source(&files, &reader, &cfg)?;
    let kernels = pipelines();
    let gpu = match placement {
        DecoderDevice::Cpu => Gpu::new_cpu(&kernels),
        DecoderDevice::SameCard => Gpu::new(&kernels),
        DecoderDevice::Card(i) => gpu_core::devices::with_gpu(i, || Gpu::new(&kernels)).map_err(|e| format!("deepseek-ocr: decoder on gpu{i}: {e}"))?,
    };
    let engine = Engine::new(gpu, cfg.decoder.clone(), &decoder, engine_config())?;
    stage_time("load: decoder engine (weight upload, KV pool)", t2);

    let pre = Gpu::new_cpu(preprocess::PIPELINES);
    let front = OcrFrontend { enc: PageEncoder { cfg, enc, rows, pre, tok }, stop: vec![eos], ngram: NoRepeatNgram::new(NGRAM_SIZE, NGRAM_WINDOW, &allow) };
    stage_time("load: TOTAL", t0);
    Ok(ScheduledInstance::new(front, Scheduler::new(engine, MAX_BATCH as usize), MAX_BATCH as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            ..Default::default()
        };
        let tok = QwenBpe::from_gguf(&gt).expect("toy tokenizer");
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        let a = PageEncoder::build_prompt(&tok, &cfg, DEFAULT_INSTRUCTION).expect("default instruction");
        let (short, long) = (
            PageEncoder::build_prompt(&tok, &cfg, "a").expect("short instruction"),
            PageEncoder::build_prompt(&tok, &cfg, "aa").expect("longer instruction"),
        );
        for p in [&a, &short, &long] {
            assert_eq!(p.image_run(), (1, 273), "BOS, then the whole 273-row global view -- whatever the instruction");
        }
        // ...and the instruction really does change the prompt, so the run's
        // invariance above is not vacuous. (Only the tail moves; `row0` cannot,
        // because `text_before` is empty by construction.)
        assert!(long.len() > short.len(), "a longer instruction must produce a longer prompt");
        assert!(a.len() + DEFAULT_MAX_NEW as usize <= MAX_CONTEXT as usize, "the default request must fit the built context");
    }
}
