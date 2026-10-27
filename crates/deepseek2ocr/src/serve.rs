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

use std::sync::Arc;

use data::qwen_tokenizer::QwenBpe;
use imaging::pixels::{hwc_to_rgb8, ChannelPolicy};
use imaging::views::{mosaic_tiles, pad_square, tile_grid, to_chw_half};
use model::serve::PromptEmbeds;
use ocrserve::{DecoderDevice, Encoded, OcrFrontend, OcrInstance, PageEncoder as OcrPageEncoder, PageRequest};
use sam1::resample::GridSource;

use crate::caps::DEFAULT_INSTRUCTION;
use crate::config::{DeepseekOcrConfig, IMAGE_NEWLINE, VIEW_SEPARATOR};
use crate::encoder::DeepEncoder;
use crate::import::{self, Files};
use crate::layout::assemble_block;
use crate::prompt::{self, Prompt};
use crate::rows::ViewGrid;
use crate::stage_time;

/// The global view's side: the tower's native input.
pub const GLOBAL_SIDE: u32 = 1024;
/// A local tile's side, and the largest image read as one view.
pub const TILE_SIDE: u32 = 640;
/// The most tiles an image is cut into.
pub const MAX_TILES: u32 = 9;
/// The pad colour of a view (the reference's `mean * 255` truncated).
const PAD_FILL: u8 = 127;

/// How a page is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    /// Both sides at most [`TILE_SIDE`]: one aspect-fitted `640 x 640` view, 111 rows. A
    /// region crop, a table cell or a line costs a tenth of a page, not a page (the
    /// reference's "Small" mode).
    Small,
    /// The whole image as the one `1024 x 1024` global view (tiling off), 273 rows.
    Global,
    /// Larger images: the `cols x rows` mosaic of 640 px tiles plus the 1024 px global view
    /// (the reference's "Gundam" mode).
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
}

/// The vision half of a request: image to the DeepEncoder's rows, plus the
/// tokenizer and the prompt layout around them.
///
/// Two encoders share one checkpoint: the global view's at the tower's native
/// 1024 px grid, and the tile's at 640 px (a 40x40 SAM grid, 10x10 tokens). The
/// tile encoder reads the absolute position table resampled to its grid
/// ([`GridSource`]) and resamples the relative-position tables in the graph.
pub struct PageEncoder {
    cfg: DeepseekOcrConfig,
    global: DeepEncoder,
    tile: DeepEncoder,
    newline: Vec<f32>,
    separator: Vec<f32>,
    tok: Arc<QwenBpe>,
}

impl PageEncoder {
    /// `BOS ++ <image> x n_rows ++ "\n" ++ instruction` for `reading`.
    ///
    /// The newline belongs to the reference's own prompt string
    /// (`"<image>\n<|grounding|>Convert the document to markdown."`), i.e. it
    /// sits between the image block and the instruction - which is exactly the
    /// `text_after` side of `build_prompt`'s split.
    pub fn build_prompt(tok: &QwenBpe, cfg: &DeepseekOcrConfig, reading: Reading, instruction: &str) -> Result<Prompt, String> {
        let after = format!("\n{instruction}");
        let (g, t) = (cfg.token_grid().0, cfg.at_sam_grid(TILE_SIDE / 16, TILE_SIDE / 16).token_grid().0);
        match reading {
            Reading::Small => prompt::build_prompt(tok, "", &after, t, t, ViewGrid::global_only()),
            Reading::Global => prompt::build_prompt(tok, "", &after, g, t, ViewGrid::global_only()),
            Reading::Gundam { cols, rows } => prompt::build_prompt(tok, "", &after, g, t, ViewGrid::new(cols, rows)),
        }
    }
}

impl OcrPageEncoder for PageEncoder {
    fn tok(&self) -> &Arc<QwenBpe> {
        &self.tok
    }

    /// Cut `hwc` (`h x w` RGB in `[0, 1]`) into its views the way the reference does
    /// (Pillow bicubic, aspect-fitted pad), run the DeepEncoder over each and lay the
    /// rows out as the decoder's image block: local tiles, the global view, the separator.
    fn encode(&self, page: &PageRequest) -> Encoded {
        let (w, h) = (page.w, page.h);
        let reading = Reading::of(w, h, page.read.max_tiles);
        let prompt = Self::build_prompt(&self.tok, &self.cfg, reading, &page.instruction)?;
        let t_pre = std::time::Instant::now();
        let rgb = hwc_to_rgb8(&page.hwc, w, h, 3, ChannelPolicy::RequireRgb)?;
        let (global_chw, tiles): (Vec<f32>, Vec<Vec<f32>>) = match reading {
            Reading::Small => (to_chw_half(&pad_square(&rgb, TILE_SIDE, PAD_FILL)), Vec::new()),
            Reading::Global => (to_chw_half(&pad_square(&rgb, GLOBAL_SIDE, PAD_FILL)), Vec::new()),
            Reading::Gundam { cols, rows } => {
                (to_chw_half(&pad_square(&rgb, GLOBAL_SIDE, PAD_FILL)), mosaic_tiles(&rgb, TILE_SIDE, cols, rows).iter().map(to_chw_half).collect())
            }
        };
        stage_time("encode request: preprocess", t_pre);

        let t_enc = std::time::Instant::now();
        let mut views = Vec::with_capacity(1 + tiles.len());
        views.push(match reading {
            Reading::Small => self.tile.forward(&global_chw),
            Reading::Global | Reading::Gundam { .. } => self.global.forward(&global_chw),
        });
        views.extend(tiles.iter().map(|t| self.tile.forward(t)));
        stage_time("encode request: DeepEncoder", t_enc);

        let (row0, _) = prompt.image_run();
        let embeds = PromptEmbeds::run(self.cfg.decoder.d_model() as usize, row0, assemble_block(&prompt.plan, &views, &self.newline, &self.separator))?;
        Ok((prompt.ids, embeds))
    }
}

/// DeepSeek-OCR resident: the encoder, the decoder engine and the scheduler
/// that batches requests onto it.
pub type DeepseekOcrInstance = OcrInstance;

/// Build the whole served model from `dir` (the GGUF pair or the Hugging Face
/// checkpoint): the vision towers on the ambient card, the decoder on `placement`.
pub fn load(dir: &str, placement: DecoderDevice) -> Result<DeepseekOcrInstance, String> {
    ocrserve::config().check_servable()?;
    let t0 = std::time::Instant::now();
    let files = Files::locate(dir)?;
    let cfg = import::config(&files, 1)?;
    let tok = Arc::new(import::tokenizer(&files)?);
    let eos = tok.special_id(prompt::EOS).ok_or_else(|| format!("this tokenizer has no reserved {:?} token", prompt::EOS))?;
    let allow: Vec<u32> = [prompt::TD_OPEN, prompt::TD_CLOSE].into_iter().filter_map(|s| tok.special_id(s)).collect();
    // The layouts are validated once, here, so a tokenizer without the image marker fails the load, not the first request.
    PageEncoder::build_prompt(&tok, &cfg, Reading::Small, DEFAULT_INSTRUCTION)?;
    PageEncoder::build_prompt(&tok, &cfg, Reading::Gundam { cols: 2, rows: 1 }, DEFAULT_INSTRUCTION)?;
    stage_time("load: config+tokenizer+prompt", t0);

    eprintln!("deepseek-ocr: vision on the ambient card, decoder on {placement:?}");
    let t1 = std::time::Instant::now();
    let vision = import::encoder_weights_for(&files, &cfg)?;
    let dev_vision = |k: &'static [(&'static str, &'static str)]| ocrserve::serving_gpu(k);
    let mut global = DeepEncoder::new(&dev_vision, cfg.clone(), &vision, 0, false);
    let tile_grid = TILE_SIDE / cfg.sam.patch_size;
    let tile_cfg = cfg.at_sam_grid(tile_grid, tile_grid);
    let tile_weights = GridSource::new(&vision, (cfg.sam.grid_h, cfg.sam.grid_w), (tile_grid, tile_grid), cfg.sam.d_model);
    let mut tile = DeepEncoder::new(&dev_vision, tile_cfg, &tile_weights, 0, false);
    if let Some(precision) = ocrserve::config().gemm {
        global.set_gemm_precision(precision);
        tile.set_gemm_precision(precision);
    }
    let newline = glue_vector(&global, IMAGE_NEWLINE);
    let separator = glue_vector(&global, VIEW_SEPARATOR);
    drop(vision);
    stage_time("load: DeepEncoders (mmproj import, weight upload)", t1);

    let t2 = std::time::Instant::now();
    let reader = import::decoder_reader(&files, &cfg)?;
    let decoder = import::decoder_source(&files, &reader, &cfg)?;
    let engine = ocrserve::build_engine(cfg.decoder.clone(), &decoder, placement).map_err(|e| format!("deepseek-ocr: {e}"))?;
    stage_time("load: decoder engine (weight upload, KV pool)", t2);

    let front = OcrFrontend::new(PageEncoder { cfg, global, tile, newline, separator, tok }, "deepseek-ocr", eos, &allow);
    stage_time("load: TOTAL", t0);
    Ok(ocrserve::instance(front, engine))
}

/// One of the encoder's learned `[d_model]` rows (the newline or the separator).
fn glue_vector(enc: &DeepEncoder, name: &str) -> Vec<f32> {
    enc.read_glue_weight(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tok() -> QwenBpe {
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
        QwenBpe::from_gguf(&gt).expect("toy tokenizer")
    }

    /// The image run must not depend on the instruction: `text_before` is empty,
    /// so the block always starts right after BOS, and its size is a function of
    /// how the page is read alone.
    #[test]
    fn the_image_run_depends_on_the_reading_and_not_on_the_instruction() {
        let tok = toy_tok();
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        for (reading, rows) in [(Reading::Small, 111), (Reading::Global, 273), (Reading::Gundam { cols: 1, rows: 2 }, 273 + 2 * 10 * 11), (Reading::Gundam { cols: 3, rows: 3 }, 273 + 30 * 31)] {
            let run = |i: &str| PageEncoder::build_prompt(&tok, &cfg, reading, i).expect("prompt");
            let (a, short, long) = (run(DEFAULT_INSTRUCTION), run("a"), run("aa"));
            for p in [&a, &short, &long] {
                assert_eq!(p.image_run(), (1, rows), "{reading:?}");
            }
            // ...and the instruction really does change the prompt, so the invariance is not vacuous.
            assert!(long.len() > short.len());
        }
    }

    /// Small images take the cheap single view, everything else the tiled reading.
    #[test]
    fn small_images_are_read_as_one_small_view() {
        assert_eq!(Reading::of(300, 40, None), Reading::Small, "a line crop");
        assert_eq!(Reading::of(640, 640, None), Reading::Small);
        assert!(matches!(Reading::of(1240, 1754, None), Reading::Gundam { cols: 2, rows: 3 }));
        assert_eq!(Reading::of(1240, 1754, Some(0)), Reading::Global, "tiling off reads the page as one view");
        assert_eq!(Reading::of(300, 40, Some(0)), Reading::Global);
        assert!(matches!(Reading::of(1240, 1754, Some(4)), Reading::Gundam { cols, rows } if cols * rows <= 4));
    }
}
