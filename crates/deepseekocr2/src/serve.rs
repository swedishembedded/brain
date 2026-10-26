// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in turning
// concurrent page requests into batched decoder rows then you can procure our
// services by sending an email to info@swedishembedded.com.

//! DeepSeek-OCR-2 through the generic continuous-batching scheduler: the vision
//! half ([`PageEncoder`]: SAM then the Qwen2 resampler then the projector) and the
//! loader that wires it to the shared decoder engine (`ocrserve`).

use data::qwen_tokenizer::QwenBpe;
use gpu_core::Gpu;
use model::serve::PromptEmbeds;
use ocrserve::{DecoderDevice, OcrFrontend, OcrInstance, PageEncoder as OcrPageEncoder};
use sam1::SamEncoder;

use crate::caps::DEFAULT_INSTRUCTION;
use crate::config::DeepseekOcr2VisionConfig;
use crate::encoder::{sam_tokens_from_nchw, Resampler};
use crate::import::{self, Files};
use crate::preprocess::{self, Fit};
use crate::prompt::{self, Prompt};

/// The vision half of a request: pixels to the rows that stand at the image
/// positions, plus the tokenizer and the prompt layout around them.
pub struct PageEncoder {
    cfg: DeepseekOcr2VisionConfig,
    sam: SamEncoder,
    resampler: Resampler,
    separator: Vec<f32>,
    /// The CPU-side preprocessor's own device handle.
    pre: Gpu,
    tok: QwenBpe,
}

impl PageEncoder {
    /// `BOS ++ <image> x (global rows + 1) ++ "\n" ++ instruction`.
    pub fn build_prompt(tok: &QwenBpe, cfg: &DeepseekOcr2VisionConfig, instruction: &str) -> Result<Prompt, String> {
        prompt::build_prompt(tok, "", &format!("\n{instruction}"), cfg.encoder.n_query_global + 1)
    }
}

impl OcrPageEncoder for PageEncoder {
    fn tok(&self) -> &QwenBpe {
        &self.tok
    }

    fn encode(&self, instruction: &str, hwc: &[f32], w: u32, h: u32) -> Result<(Vec<u32>, PromptEmbeds), String> {
        let prompt = Self::build_prompt(&self.tok, &self.cfg, instruction)?;
        let t_pre = std::time::Instant::now();
        let image = preprocess::preprocess_image(&self.pre, &self.cfg, hwc, w, h, Fit::Pad);
        ocrserve::stage_time("encode request: preprocess", t_pre);

        let t_enc = std::time::Instant::now();
        self.sam.write_image(&image);
        self.sam.run();
        let nchw = self.sam.gpu.read(self.sam.output(), self.sam.out_len());
        let tokens = sam_tokens_from_nchw(&nchw, self.cfg.sam.compress_out as usize, self.cfg.encoder.n_query_global as usize);
        let mut rows = self.resampler.resample_view(&tokens, false).projected;
        // v2 has no per-row newline: the global view's rows, then the one separator.
        rows.extend_from_slice(&self.separator);
        ocrserve::stage_time("encode request: SAM + resampler", t_enc);
        let (row0, _) = prompt.image_run();
        let embeds = PromptEmbeds::run(self.cfg.decoder_hidden as usize, row0, rows)?;
        Ok((prompt.ids, embeds))
    }
}

/// DeepSeek-OCR-2 resident: the encoder, the decoder engine and the scheduler that batches requests onto it.
pub type DeepseekOcr2Instance = OcrInstance<PageEncoder>;

/// Build the whole served model from `dir` (the Hugging Face checkpoint or a
/// community GGUF pair): the vision tower on the ambient card, the decoder on `placement`.
pub fn load(dir: &str, placement: DecoderDevice) -> Result<DeepseekOcr2Instance, String> {
    let t0 = std::time::Instant::now();
    let files = Files::locate(dir)?;
    let (cfg, dec_cfg) = import::configs(&files, 1)?;
    let tok = import::tokenizer(&files)?;
    let eos = tok.special_id(prompt::EOS).ok_or_else(|| format!("this tokenizer has no reserved {:?} token", prompt::EOS))?;
    let allow: Vec<u32> = ["<td>", "</td>"].into_iter().filter_map(|s| tok.special_id(s)).collect();
    // The prompt layout is validated once, here, so a tokenizer without the image marker fails the load, not the first request.
    PageEncoder::build_prompt(&tok, &cfg, DEFAULT_INSTRUCTION)?;
    ocrserve::stage_time("load: config+tokenizer+prompt", t0);

    eprintln!("deepseek-ocr-2: vision on the ambient card, decoder on {placement:?}");
    let t1 = std::time::Instant::now();
    let vision = import::vision_reader(&files)?;
    let sam = SamEncoder::new_inference(Gpu::new(sam1::PIPELINES), cfg.sam.clone(), &vision, 0);
    let resampler = Resampler::new_on(Gpu::new(crate::encoder::PIPELINES), cfg.clone(), &vision, false);
    let mut separator = Vec::new();
    use checkpoint::TensorSource;
    if !vision.with_tensor_chunks("vision.view_separator", usize::MAX, &mut |_, d| separator = d.to_vec()) {
        return Err("deepseek-ocr-2: the checkpoint has no view separator".to_string());
    }
    drop(vision);
    ocrserve::stage_time("load: vision weight upload", t1);

    let t2 = std::time::Instant::now();
    let decoder = import::decoder_reader(&files)?;
    let engine = ocrserve::build_engine(dec_cfg, &decoder, placement).map_err(|e| format!("deepseek-ocr-2: {e}"))?;
    ocrserve::stage_time("load: decoder engine (weight upload, KV pool)", t2);

    let pre = Gpu::new_cpu(preprocess::PIPELINES);
    let front = OcrFrontend::new(PageEncoder { cfg, sam, resampler, separator, pre, tok }, "deepseek-ocr-2", eos, &allow);
    ocrserve::stage_time("load: TOTAL", t0);
    Ok(ocrserve::instance(front, engine))
}
