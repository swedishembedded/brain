// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for conditioning video generation
// on interleaved text, image and video prompts for its clients. If your team
// needs expertise in multimodal text encoders then you can procure our
// services by sending an email to info@swedishembedded.com.

//! MiniMax-H3's presentation of a request to its Qwen3-VL conditioner.
//!
//! Every request is ONE token stream, read after decoder layer
//! [`crate::pipeline::TEXT_ENCODER_LAYER`]: a label per reference, numbered
//! per modality - `"<Picture i>: "` plus an image vision block, `"<Audio j>:
//! "` alone (a waveform never reaches the conditioner), `"<Video k>: "` plus
//! one `"<{t:.1} seconds>"`-labelled vision block per merged pair of frames
//! sampled at 2 fps - then the prompt verbatim, with no chat template and no
//! special tokens. A video that carries sound is labelled `"<Audio j>: "`
//! first, mirroring the order its rows are packed in. `t2va` is the prompt
//! alone, `fl2va` labels its keyframes as pictures, `ref2va` its references
//! (`encoders.py`: `MiniMaxH3TextEncoderStep`, `MiniMaxH3FL2VATextEncoderStep`,
//! `MiniMaxH3Ref2VATextEncoderStep`).
//!
//! The rows of a vision block - its start/end markers included - are tagged
//! [`TAG_VIDEO`] rather than text: that is what the DiT's AdaLN modulation
//! keys off.

use gpu_core::select::Dtype;
use imaging::Rgb8;
use qwen3vl::encoder::VisionTower;
use qwen3vl::preprocess::ImageGeometry;
use qwen3vl::text_tower::VisionSplice;
use qwen3vl::{Qwen3VlConfig, TextTower};

use crate::config::{TAG_TEXT, TAG_VIDEO};
use crate::pipeline::{TextConditioning, FPS};

/// The rate the conditioner reads a reference video at
/// (`MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=2.0)`).
pub const VIDEO_SAMPLE_FPS: f64 = 2.0;

/// One item of a presentation, in the order it is labelled.
#[derive(Clone, Copy)]
pub enum Presented<'a> {
    /// An image reference or an `fl2va` keyframe.
    Image(&'a Rgb8),
    /// A video reference's frames at [`FPS`]; `with_audio` when it carries
    /// a soundtrack, which earns it an `"<Audio j>: "` label too.
    Video { frames: &'a [Rgb8], with_audio: bool },
    /// A lone soundtrack: a label, nothing else.
    Audio,
}

impl<'a> Presented<'a> {
    /// The presentation of a `ref2va` reference.
    pub fn of_reference(reference: &'a crate::references::NormalizedReference) -> Presented<'a> {
        use crate::references::NormalizedReference;
        match reference {
            NormalizedReference::Image(img) => Presented::Image(img),
            NormalizedReference::Video { frames, soundtrack } => Presented::Video { frames, with_audio: soundtrack.is_some() },
            NormalizedReference::Audio(_) => Presented::Audio,
        }
    }
}

/// The frames the conditioner reads from a `fps` clip of `num_frames` at
/// `sample_fps` - every `fps / sample_fps`-th, deduplicated - and the
/// timestamp of each vision block once Qwen3-VL merges them in groups of
/// `temporal_patch` (the last frame repeated to fill the final group; a
/// block is labelled with the mean of its first and last timestamp).
pub fn sample_video_frames(num_frames: usize, fps: f64, sample_fps: f64, temporal_patch: usize) -> Result<(Vec<usize>, Vec<f64>), String> {
    let stride = fps / sample_fps;
    let mut indices: Vec<usize> = Vec::new();
    let mut cursor = 0.0f64;
    while (cursor.round_ties_even() as usize) < num_frames {
        let index = cursor.round_ties_even() as usize;
        if indices.last().is_none_or(|&last| index > last) {
            indices.push(index);
        }
        cursor += stride;
    }
    if indices.len() < temporal_patch {
        let minimum = ((temporal_patch - 1) as f64 * stride).round_ties_even() as usize + 1;
        return Err(format!(
            "a reference video is read at {sample_fps} fps and its sampled frames are merged in groups of {temporal_patch}, so it must run at least {minimum} frames at {fps} fps, got {num_frames}"
        ));
    }
    let mut timestamps: Vec<f64> = (0..indices.len()).map(|i| i as f64 / sample_fps).collect();
    while !timestamps.len().is_multiple_of(temporal_patch) {
        timestamps.push(*timestamps.last().expect("at least one sampled frame"));
    }
    let blocks = timestamps.chunks(temporal_patch).map(|group| (group[0] + group[temporal_patch - 1]) / 2.0).collect();
    Ok((indices, blocks))
}

/// The tokens of a presentation, their [`TAG_TEXT`]/[`TAG_VIDEO`] tags, and
/// where each vision block's placeholder run sits.
#[derive(Debug, PartialEq)]
pub struct PresentationTokens {
    pub ids: Vec<u32>,
    pub tags: Vec<u32>,
    /// `(row0, n_rows)` of each vision block's placeholder run, in order.
    pub runs: Vec<(u32, u32)>,
}

/// The token ids a presentation needs besides its text.
#[derive(Clone, Copy, Debug)]
pub struct VisionTokenIds {
    pub vision_start: u32,
    pub vision_end: u32,
    pub image_pad: u32,
    pub video_pad: u32,
}

/// What one presented item contributes to the token stream: its image
/// block's token count, or its video blocks' timestamps and per-block token
/// count.
pub enum ItemBlocks {
    Image { tokens: u32 },
    Video { timestamps: Vec<f64>, tokens_per_block: u32 },
    None,
}

/// Tokenize a presentation (`_build_presentation`): `items` and `blocks`
/// pair up, in order; `text` tokenizes a string without special tokens.
pub fn build_presentation(text: &dyn Fn(&str) -> Vec<u32>, ids: VisionTokenIds, prompt: &str, items: &[Presented], blocks: &[ItemBlocks]) -> PresentationTokens {
    assert_eq!(items.len(), blocks.len(), "one block description per presented item");
    let mut out = PresentationTokens { ids: Vec::new(), tags: Vec::new(), runs: Vec::new() };
    let emit_text = |out: &mut PresentationTokens, s: &str| {
        let t = text(s);
        out.tags.extend(std::iter::repeat_n(TAG_TEXT, t.len()));
        out.ids.extend(t);
    };
    let emit_vision = |out: &mut PresentationTokens, pad: u32, n: u32| {
        out.ids.push(ids.vision_start);
        out.runs.push((out.ids.len() as u32, n));
        out.ids.extend(std::iter::repeat_n(pad, n as usize));
        out.ids.push(ids.vision_end);
        out.tags.extend(std::iter::repeat_n(TAG_VIDEO, n as usize + 2));
    };
    let (mut pictures, mut videos, mut audios) = (0, 0, 0);
    for (item, block) in items.iter().zip(blocks) {
        let with_audio = matches!(item, Presented::Audio | Presented::Video { with_audio: true, .. });
        if with_audio {
            audios += 1;
            emit_text(&mut out, &format!("<Audio {audios}>: "));
        }
        match (item, block) {
            (Presented::Image(_), ItemBlocks::Image { tokens }) => {
                pictures += 1;
                emit_text(&mut out, &format!("<Picture {pictures}>: "));
                emit_vision(&mut out, ids.image_pad, *tokens);
            }
            (Presented::Video { .. }, ItemBlocks::Video { timestamps, tokens_per_block }) => {
                videos += 1;
                emit_text(&mut out, &format!("<Video {videos}>: "));
                for t in timestamps {
                    emit_text(&mut out, &format!("<{t:.1} seconds>"));
                    emit_vision(&mut out, ids.video_pad, *tokens_per_block);
                }
            }
            (Presented::Audio, ItemBlocks::None) => {}
            _ => panic!("build_presentation: a presented item and its block description disagree"),
        }
    }
    emit_text(&mut out, prompt);
    out
}

/// One vision block's packed patches and patch grid; `video` for a
/// temporal group of a video, which Qwen3-VL positions as its own block.
struct VisionBlock {
    grid: (u32, u32),
    pixels: Vec<f32>,
    video: bool,
}

/// The Qwen3-VL conditioner a presentation is encoded with.
pub struct PresentationEncoder {
    text_encoder_dir: String,
    cfg: Qwen3VlConfig,
    tokenizer: data::qwen_tokenizer::QwenBpe,
    /// The image and video processor geometry, absent for a checkout
    /// without `processor/` configs - which can present a prompt alone.
    image: Option<ImageGeometry>,
    video: Option<ImageGeometry>,
    layer: usize,
    dt: Dtype,
    device: Option<String>,
}

const NO_PROCESSOR: &str = "minimaxh3 conditioner: images and videos are presented through the checkout's processor/ configs, which this checkout has none of";

fn read_json(path: &str) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("minimaxh3 conditioner: cannot read {path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("minimaxh3 conditioner: cannot parse {path}: {e}"))
}

impl PresentationEncoder {
    /// The conditioner of a checkout: the Qwen3-VL at `text_encoder_dir`,
    /// read after decoder layer `layer` with its linears at `dt`, the
    /// tokenizer at `tokenizer_dir`, and the image and video processor
    /// configs at `processor_dir` (needed only to present images or videos).
    /// The vision tower runs on `device`.
    pub fn open(text_encoder_dir: &str, tokenizer_dir: &str, processor_dir: Option<&str>, layer: usize, dt: Dtype, device: Option<&str>) -> Result<PresentationEncoder, String> {
        let cfg = Qwen3VlConfig::from_hf(&read_json(&format!("{text_encoder_dir}/config.json"))?);
        if cfg.text.n_layers as usize <= layer {
            return Err(format!("minimaxh3 conditioner: MiniMax-H3 reads hidden_states[{layer}], which needs more than {layer} decoder layers, but the checkout has {}", cfg.text.n_layers));
        }
        let tokenizer = data::qwen_tokenizer::QwenBpe::from_dir(tokenizer_dir).map_err(|e| format!("minimaxh3 conditioner: tokenizer: {e}"))?;
        let geometry = |file: &str| processor_dir.map(|dir| ImageGeometry::from_json(&read_json(&format!("{dir}/{file}"))?)).transpose();
        let (image, video) = (geometry("preprocessor_config.json")?, geometry("video_preprocessor_config.json")?);
        Ok(PresentationEncoder { text_encoder_dir: text_encoder_dir.to_string(), cfg, tokenizer, image, video, layer, dt, device: device.map(str::to_string) })
    }

    fn token_ids(&self) -> VisionTokenIds {
        VisionTokenIds { vision_start: self.cfg.vision_start_token_id, vision_end: self.cfg.vision_end_token_id, image_pad: self.cfg.image_token_id, video_pad: self.cfg.video_token_id }
    }

    /// The token stream of the presentation of `prompt` and `items`.
    pub fn tokens(&self, prompt: &str, items: &[Presented]) -> Result<PresentationTokens, String> {
        self.prepare(prompt, items).map(|(tokens, _)| tokens)
    }

    /// The token stream, and every vision block's packed patches in order.
    fn prepare(&self, prompt: &str, items: &[Presented]) -> Result<(PresentationTokens, Vec<VisionBlock>), String> {
        use data::Tokenizer;
        let m2 = self.cfg.vision.spatial_merge_size.pow(2);
        let patch = self.cfg.vision.patch_size;
        let mut vision_blocks: Vec<VisionBlock> = Vec::new();
        let mut item_blocks = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Presented::Image(img) => {
                    let p = self.image.as_ref().ok_or(NO_PROCESSOR)?.patches(img);
                    let grid = (p.h_bar / patch, p.w_bar / patch);
                    item_blocks.push(ItemBlocks::Image { tokens: grid.0 * grid.1 / m2 });
                    vision_blocks.push(VisionBlock { grid, pixels: p.pixels, video: false });
                }
                Presented::Video { frames, .. } => {
                    let video = self.video.as_ref().ok_or(NO_PROCESSOR)?;
                    let (indices, timestamps) = sample_video_frames(frames.len(), FPS as f64, VIDEO_SAMPLE_FPS, video.temporal_patch_size as usize)?;
                    let sampled: Vec<&Rgb8> = indices.iter().map(|&i| &frames[i]).collect();
                    let v = video.video_patches(&sampled);
                    if v.grid_t as usize != timestamps.len() {
                        return Err(format!("minimaxh3 conditioner: a reference video merged into {} vision blocks but is labelled with {}", v.grid_t, timestamps.len()));
                    }
                    let grid = (v.h_bar / patch, v.w_bar / patch);
                    let per_block = (grid.0 * grid.1) as usize * self.cfg.vision.patch_vec_dim() as usize;
                    for chunk in v.pixels.chunks_exact(per_block) {
                        vision_blocks.push(VisionBlock { grid, pixels: chunk.to_vec(), video: true });
                    }
                    item_blocks.push(ItemBlocks::Video { timestamps, tokens_per_block: grid.0 * grid.1 / m2 });
                }
                Presented::Audio => item_blocks.push(ItemBlocks::None),
            }
        }

        let text = |s: &str| self.tokenizer.encode(s);
        let tokens = build_presentation(&text, self.token_ids(), prompt, items, &item_blocks);
        if tokens.ids.is_empty() {
            return Err("minimaxh3 conditioner: the presentation tokenizes to nothing".to_string());
        }
        Ok((tokens, vision_blocks))
    }

    /// Encode the presentation of `prompt` and `items`.
    pub fn encode(&self, prompt: &str, items: &[Presented]) -> Result<TextConditioning, String> {
        let merge = self.cfg.vision.spatial_merge_size;
        let (tokens, vision_blocks) = self.prepare(prompt, items)?;
        let mut tower = TextTower::from_hf_dir(&self.text_encoder_dir, self.cfg.text.clone(), self.cfg.mrope_section, self.dt, self.layer, tokens.ids.len() as u32)?;
        tracing::info!(stages = tower.n_stages(), tokens = tokens.ids.len(), vision_blocks = vision_blocks.len(), "minimaxh3 conditioner placed");
        if vision_blocks.is_empty() {
            return Ok(TextConditioning { embeds: tower.encode(&tokens.ids), token_tags: tokens.tags });
        }

        let reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(&self.text_encoder_dir)).map_err(|e| format!("minimaxh3 conditioner: open {}: {e}", self.text_encoder_dir))?;
        let gpu = gpu_core::Gpu::open(self.device.as_deref(), qwen3vl::encoder::vision_pipelines());
        let vision = VisionTower::from_reader(&reader, gpu, self.cfg.vision.clone(), self.cfg.text.d_model)?;
        let mut visual = Vec::new();
        let mut deepstack: Vec<Vec<f32>> = vec![Vec::new(); vision.n_deepstack()];
        let (mut image_grids, mut video_grids) = (Vec::new(), Vec::new());
        for block in &vision_blocks {
            let f = vision.features(block.grid.0, block.grid.1, &block.pixels);
            visual.extend(f.visual);
            for (level, rows) in f.deepstack.into_iter().enumerate() {
                deepstack[level].extend(rows);
            }
            let llm_grid = (1, block.grid.0 / merge, block.grid.1 / merge);
            if block.video { video_grids.push(llm_grid) } else { image_grids.push(llm_grid) }
        }
        drop(vision);
        let positions = qwen3vl::mrope::get_rope_index_multi(&tokens.ids, &[(self.cfg.image_token_id, &image_grids), (self.cfg.video_token_id, &video_grids)]);
        let embeds = tower.encode_multimodal(&tokens.ids, &positions, &VisionSplice { runs: &tokens.runs, visual: &visual, deepstack: &deepstack });
        Ok(TextConditioning { embeds, token_tags: tokens.tags })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long presentation's text tower holds no more on a card than its
    /// placement plans for it: real layers, built at a reference image's
    /// token count, fit inside what `text_tower::tower_cost` prices them at.
    #[test]
    fn a_real_text_tower_stage_holds_no_more_than_its_placement_plans() {
        let Some(root) = crate::caps::store_root() else {
            brain_testutil::skip("no MiniMax-H3 checkout in the model store");
            return;
        };
        let Some(before) = gpu_core::capacity::available_for(0) else {
            brain_testutil::skip("no GPU to measure");
            return;
        };
        let dir = format!("{root}/text_encoder");
        let cfg = Qwen3VlConfig::from_hf(&read_json(&format!("{dir}/config.json")).unwrap());
        let dt = crate::caps::best_linear_dtype();
        let (rows, layers) = (4200u32, 8usize);
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(&dir)).unwrap();
        let source = qwen3vl::import::decoder_source(&reader);
        let shard = model::shard::Shard { start: 0, end: layers, embed: true, head: false, gpu_index: 0 };
        let tower = TextTower::with_shards(&source, cfg.text.clone(), cfg.mrope_section, dt, vec![shard], rows);
        let used = before.saturating_sub(gpu_core::capacity::available_for(0).unwrap());
        let (cost, reserve) = qwen3vl::text_tower::tower_cost(&cfg.text, layers, dt, rows);
        let planned = cost.embed + cost.per_layer.iter().sum::<u64>() + reserve;
        eprintln!("text tower, {layers} layers + embed at {rows} rows, {dt:?}: {used} bytes on the card, {planned} planned (per layer {}, per stage {reserve}, embed {})", cost.per_layer[0], cost.embed);
        assert!(used <= planned, "the stage holds {used} bytes, its placement planned {planned}");
        drop(tower);
    }

    /// The 2 fps sampling of a 24 fps clip, and its round-half-to-even block
    /// labels: 30 frames sample 0, 12 and 24, merged as (0, 12) and (24, 24).
    #[test]
    fn video_frames_are_sampled_at_two_fps_and_labelled_per_block() {
        let (indices, blocks) = sample_video_frames(30, 24.0, 2.0, 2).unwrap();
        assert_eq!(indices, [0, 12, 24]);
        assert_eq!(blocks, [0.25, 1.0]);
        assert_eq!(format!("{:.1}", blocks[0]), "0.2", "Python's `{{:.1f}}` rounds the tie to even");
        assert!(sample_video_frames(12, 24.0, 2.0, 2).unwrap_err().contains("at least 13 frames"));
    }

    /// The ref2va presentation of a tiny random Qwen3-VL, against the
    /// installed diffusers + transformers over the same references, the real
    /// tokenizer and the real processor configs: the token stream and its
    /// tags exactly, the hidden state to float rounding.
    #[test]
    fn a_ref2va_presentation_matches_the_reference() {
        let dir = brain_testutil::testdata_path("golden/minimaxh3/presentation");
        let file = dir.join("minimaxh3_presentation.safetensors");
        if !file.is_file() {
            brain_testutil::skip(&format!("{} not found - run tools/goldens/minimaxh3_presentation_dump_reference.py --out {}", file.display(), dir.display()));
            return;
        }
        let Some(h3) = brain_testutil::model_dir("MiniMaxAI/MiniMax-H3").filter(|d| std::path::Path::new(&format!("{d}/processor/video_preprocessor_config.json")).is_file()) else {
            brain_testutil::skip("no MiniMaxAI/MiniMax-H3 processor/tokenizer in the model store");
            return;
        };
        let Some(src) = brain_testutil::golden::Source::open(&dir, "tools/goldens/minimaxh3_presentation_dump_reference.py") else {
            return;
        };
        if !src.require(&[("layer", 3), ("fps_x1000", (FPS * 1000.0) as i64), ("hidden", 64)]) {
            return;
        }
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        let prompt = manifest["prompt"].as_str().unwrap();
        let fx: std::collections::HashMap<String, checkpoint::safetensors::StTensor> = checkpoint::safetensors::read(file.to_str().unwrap()).unwrap().into_iter().map(|t| (t.name.clone(), t)).collect();
        let rgb = |t: &checkpoint::safetensors::StTensor, f: usize| {
            let (h, w) = (t.shape[t.shape.len() - 3], t.shape[t.shape.len() - 2]);
            Rgb8 { w: w as u32, h: h as u32, px: t.data[f * h * w * 3..(f + 1) * h * w * 3].iter().map(|&v| v as u8).collect() }
        };
        let image = rgb(&fx["image"], 0);
        let frames: Vec<Rgb8> = (0..fx["video"].shape[0]).map(|f| rgb(&fx["video"], f)).collect();
        let items = [Presented::Image(&image), Presented::Audio, Presented::Video { frames: &frames, with_audio: true }];

        let encoder = PresentationEncoder::open(dir.join("text_encoder").to_str().unwrap(), &format!("{h3}/tokenizer"), Some(&format!("{h3}/processor")), 3, Dtype::F32, Some("cpu")).unwrap();
        let as_u32 = |name: &str| fx[name].data.iter().map(|&v| v as u32).collect::<Vec<u32>>();
        assert_eq!(encoder.tokens(prompt, &items).unwrap().ids, as_u32("token_ids"), "token ids");
        let got = encoder.encode(prompt, &items).unwrap();
        assert_eq!(got.token_tags, as_u32("token_tags"), "token tags");
        let mut r = brain_testutil::parity::Report::new(0.9999);

        // The vision tower's own outputs first, so a vision-side drift is
        // told apart from a language-side one.
        let (_, blocks) = encoder.prepare(prompt, &items).unwrap();
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir.join("text_encoder")).unwrap();
        let vision = VisionTower::from_reader(&reader, gpu_core::Gpu::open(Some("cpu"), qwen3vl::encoder::vision_pipelines()), encoder.cfg.vision.clone(), 64).unwrap();
        for kind in ["image", "video"] {
            let mut visual = Vec::new();
            let mut deepstack = vec![Vec::new(); vision.n_deepstack()];
            for b in blocks.iter().filter(|b| b.video == (kind == "video")) {
                let f = vision.features(b.grid.0, b.grid.1, &b.pixels);
                visual.extend(f.visual);
                f.deepstack.into_iter().zip(&mut deepstack).for_each(|(rows, acc)| acc.extend(rows));
            }
            r.check(&format!("{kind} visual"), &visual, &fx[&format!("{kind}_visual")].data);
            for (level, rows) in deepstack.iter().enumerate() {
                r.check(&format!("{kind} deepstack {level}"), rows, &fx[&format!("{kind}_deepstack_{level}")].data);
            }
        }
        r.check("hidden_states[3]", &got.embeds, &fx["hidden"].data);
        r.finish("minimaxh3 ref2va presentation vs the reference");
    }
}
