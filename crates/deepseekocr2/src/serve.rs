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

use std::sync::Arc;

use data::qwen_tokenizer::QwenBpe;
use imaging::pixels::{hwc_to_rgb8, ChannelPolicy};
use imaging::views::{mosaic_tiles, pad_square, tile_grid, to_chw_half};
use model::serve::PromptEmbeds;
use ocrserve::{DecoderDevice, Encoded, OcrFrontend, OcrInstance, PageEncoder as OcrPageEncoder, PageRequest};
use sam1::resample::GridSource;
use sam1::SamEncoder;

use crate::caps::DEFAULT_INSTRUCTION;
use crate::config::DeepseekOcr2VisionConfig;
use crate::encoder::{gather_rows, sam_tokens_from_nchw, Resampler};
use crate::import::{self, Files};
use crate::prompt::{self, Prompt};

/// The global view's side: the tower's native input.
pub const GLOBAL_SIDE: u32 = 1024;
/// A local tile's side, and the largest image read as one small view.
pub const TILE_SIDE: u32 = 768;
/// The most tiles an image is cut into.
pub const MAX_TILES: u32 = 6;
/// The pad colour of a view (the reference's `mean * 255` truncated).
const PAD_FILL: u8 = 127;

/// How a page is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    /// Both sides at most [`TILE_SIDE`]: one aspect-fitted `768 x 768` view through the local query
    /// bank, 145 rows. A region crop costs about half a page, not a page.
    Small,
    /// The whole image as the one `1024 x 1024` global view (tiling off), 257 rows.
    Global,
    /// Larger images: the `cols x rows` mosaic of 768 px tiles, then the 1024 px global view, then the
    /// separator (the reference's dynamic crops).
    Gundam { cols: u32, rows: u32 },
}

impl Reading {
    /// How an image of `w x h` pixels is read; `max_tiles` caps the tiles (`Some(0)` turns tiling off).
    pub fn of(w: u32, h: u32, max_tiles: Option<u32>) -> Reading {
        if max_tiles == Some(0) {
            return Reading::Global;
        }
        match tile_grid(w, h, TILE_SIDE, max_tiles.unwrap_or(MAX_TILES).max(2)) {
            None => Reading::Small,
            Some((cols, rows)) => Reading::Gundam { cols, rows },
        }
    }

    /// Rows of the image block: every view's query rows, plus the one separator.
    pub fn rows(self, cfg: &DeepseekOcr2VisionConfig) -> u32 {
        let (local, global) = (cfg.encoder.n_query_local, cfg.encoder.n_query_global);
        1 + match self {
            Reading::Small => local,
            Reading::Global => global,
            Reading::Gundam { cols, rows } => cols * rows * local + global,
        }
    }
}

/// The vision half of a request: pixels to the rows that stand at the image
/// positions, plus the tokenizer and the prompt layout around them.
///
/// Two SAM towers share one checkpoint and one resampler: the global view's at
/// the native 1024 px grid and the tile's at 768 px (a 48x48 grid), which reads
/// the absolute position table resampled to it ([`GridSource`]) and resamples
/// the relative-position tables in the graph.
pub struct PageEncoder {
    cfg: DeepseekOcr2VisionConfig,
    sam_global: SamEncoder,
    sam_tile: SamEncoder,
    resampler: Resampler,
    separator: Vec<f32>,
    tok: Arc<QwenBpe>,
}

impl PageEncoder {
    /// `BOS ++ <image> x rows ++ "\n" ++ instruction` for `reading`.
    pub fn build_prompt(tok: &QwenBpe, cfg: &DeepseekOcr2VisionConfig, reading: Reading, instruction: &str) -> Result<Prompt, String> {
        prompt::build_prompt(tok, "", &format!("\n{instruction}"), reading.rows(cfg))
    }

    /// One view's SAM token grid, `[n_query, d_model]`.
    fn sam_tokens(&self, sam: &SamEncoder, chw: &[f32], n_query: u32) -> Vec<f32> {
        sam.write_image(chw);
        sam.run();
        let nchw = sam.gpu.read(sam.output(), sam.out_len());
        sam_tokens_from_nchw(&nchw, self.cfg.sam.compress_out as usize, n_query as usize)
    }
}

impl OcrPageEncoder for PageEncoder {
    fn tok(&self) -> &Arc<QwenBpe> {
        &self.tok
    }

    fn encode(&self, page: &PageRequest) -> Encoded {
        let (w, h) = (page.w, page.h);
        let reading = Reading::of(w, h, page.read.max_tiles);
        let prompt = Self::build_prompt(&self.tok, &self.cfg, reading, &page.instruction)?;
        let t_pre = std::time::Instant::now();
        let rgb = hwc_to_rgb8(&page.hwc, w, h, 3, ChannelPolicy::RequireRgb)?;
        ocrserve::stage_time("encode request: preprocess", t_pre);

        let t_enc = std::time::Instant::now();
        // The embedding order is the reference's: local tiles, the global view, the separator. Every
        // tile of a page goes through the resampler as one batch.
        let (nl, ng) = (self.cfg.encoder.n_query_local, self.cfg.encoder.n_query_global);
        let rows = match reading {
            Reading::Small => {
                let tokens = self.sam_tokens(&self.sam_tile, &to_chw_half(&pad_square(&rgb, TILE_SIDE, PAD_FILL)), nl);
                let mut rows = self.resampler.encode_views(&[&tokens], true).remove(0);
                rows.extend_from_slice(&self.separator);
                rows
            }
            Reading::Global => {
                let tokens = self.sam_tokens(&self.sam_global, &to_chw_half(&pad_square(&rgb, GLOBAL_SIDE, PAD_FILL)), ng);
                let mut rows = self.resampler.encode_views(&[&tokens], false).remove(0);
                rows.extend_from_slice(&self.separator);
                rows
            }
            Reading::Gundam { cols, rows } => {
                let tiles: Vec<Vec<f32>> = mosaic_tiles(&rgb, TILE_SIDE, cols, rows).iter().map(|t| self.sam_tokens(&self.sam_tile, &to_chw_half(t), nl)).collect();
                let tile_refs: Vec<&[f32]> = tiles.iter().map(Vec::as_slice).collect();
                let local = self.resampler.encode_views(&tile_refs, true);
                let global_tokens = self.sam_tokens(&self.sam_global, &to_chw_half(&pad_square(&rgb, GLOBAL_SIDE, PAD_FILL)), ng);
                let global = self.resampler.encode_views(&[&global_tokens], false).remove(0);
                gather_rows(&local, &global, &self.separator)
            }
        };
        ocrserve::stage_time("encode request: SAM + resampler", t_enc);
        let (row0, _) = prompt.image_run();
        let embeds = PromptEmbeds::run(self.cfg.decoder_hidden as usize, row0, rows)?;
        Ok((prompt.ids, embeds))
    }
}

/// DeepSeek-OCR-2 resident: the encoder, the decoder engine and the scheduler that batches requests onto it.
pub type DeepseekOcr2Instance = OcrInstance;

/// Build the whole served model from `dir` (the Hugging Face checkpoint or a
/// community GGUF pair): the vision tower on the ambient card, the decoder on `placement`.
pub fn load(dir: &str, placement: DecoderDevice) -> Result<DeepseekOcr2Instance, String> {
    ocrserve::config().check_servable()?;
    let t0 = std::time::Instant::now();
    let files = Files::locate(dir)?;
    let (cfg, dec_cfg) = import::configs(&files, 1)?;
    let tok = Arc::new(import::tokenizer(&files)?);
    let eos = tok.special_id(prompt::EOS).ok_or_else(|| format!("this tokenizer has no reserved {:?} token", prompt::EOS))?;
    let allow: Vec<u32> = ["<td>", "</td>"].into_iter().filter_map(|s| tok.special_id(s)).collect();
    // The layouts are validated once, here, so a tokenizer without the image marker fails the load, not the first request.
    PageEncoder::build_prompt(&tok, &cfg, Reading::Small, DEFAULT_INSTRUCTION)?;
    ocrserve::stage_time("load: config+tokenizer+prompt", t0);

    eprintln!("deepseek-ocr-2: vision on the ambient card, decoder on {placement:?}");
    let t1 = std::time::Instant::now();
    let vision = import::vision_reader(&files)?;
    let mut sam_global = SamEncoder::new_inference(ocrserve::serving_gpu(sam1::PIPELINES), cfg.sam.clone(), &vision, 0);
    let tile_grid = TILE_SIDE / cfg.sam.patch_size;
    let tile_weights = GridSource::new(&vision, (cfg.sam.grid_h, cfg.sam.grid_w), (tile_grid, tile_grid), cfg.sam.d_model);
    let mut sam_tile = SamEncoder::new_inference(ocrserve::serving_gpu(sam1::PIPELINES), cfg.sam.at_grid(tile_grid, tile_grid), &tile_weights, 0);
    let mut resampler = Resampler::new_on(ocrserve::serving_gpu(crate::encoder::PIPELINES), cfg.clone(), &vision, false);
    if let Some(precision) = ocrserve::config().gemm {
        sam_global.set_gemm_precision(precision);
        sam_tile.set_gemm_precision(precision);
        resampler.set_gemm_precision(precision);
    }
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

    let front = OcrFrontend::new(PageEncoder { cfg, sam_global, sam_tile, resampler, separator, tok }, "deepseek-ocr-2", eos, &allow);
    ocrserve::stage_time("load: TOTAL", t0);
    Ok(ocrserve::instance(front, engine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Qwen2EncoderConfig;

    fn cfg() -> DeepseekOcr2VisionConfig {
        DeepseekOcr2VisionConfig { sam: sam1::SamViTConfig::deepseek_ocr().at_grid(64, 64), encoder: Qwen2EncoderConfig::deepseek_ocr2(), decoder_hidden: 1280, lora: None }
    }

    /// The image block's size is a function of how the page is read: 144 query rows per local tile,
    /// 256 for the global view, one separator.
    #[test]
    fn the_image_block_follows_the_reading() {
        let c = cfg();
        assert_eq!(Reading::Small.rows(&c), 145);
        assert_eq!(Reading::Global.rows(&c), 257);
        assert_eq!(Reading::Gundam { cols: 2, rows: 3 }.rows(&c), 6 * 144 + 257);
    }

    /// Small images take the cheap single view; a page is cut into 768 px tiles, at most six.
    #[test]
    fn readings_follow_the_image_extent_and_the_tile_cap() {
        assert_eq!(Reading::of(700, 768, None), Reading::Small);
        assert_eq!(Reading::of(2480, 3508, None), Reading::Gundam { cols: 2, rows: 3 });
        assert_eq!(Reading::of(2480, 3508, Some(0)), Reading::Global);
        assert!(matches!(Reading::of(2480, 3508, Some(2)), Reading::Gundam { cols, rows } if cols * rows == 2));
    }
}
