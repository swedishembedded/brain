// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Janus-Pro text-to-image: classifier-free-guided autoregressive sampling of
//! VQ-16 image tokens, decoded to pixels.
//!
//! Every image is two decoder sequences: the conditional prompt
//! (`<|User|>: {prompt}\n\n<|Assistant|>:<begin_of_image>`) and an
//! unconditional one, the same ids with everything between the first and
//! the last replaced by the pad id. All `2 * parallel` sequences run as one
//! batch on `brain-qwen3`'s paged serving engine (one weight copy; identical
//! prompts share their prefilled blocks through its prefix cache). At each of
//! the 576 steps the generation head scores every sequence's last hidden
//! state, each image's two rows are blended (`uncond + w * (cond - uncond)`),
//! a token is sampled from the blend, and that token's `gen_aligner` row is
//! fed to both of the image's sequences. The finished 24x24 grids go through
//! the VQ-16 decoder in one batch.
//!
//! The engine keeps the decoder at the checkpoint's own bf16. Guidance
//! amplifies the gap between the two branches five-fold, and with it any
//! error in either: an int8 decoder, within 0.1% of the reference on each
//! branch's logits alone, lands at a cosine of 0.85 on the guided ones.

use std::path::Path;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use imaging::pixels::Rgb8;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen3::model::PrefillInput;
use qwen3::serve::Engine;
use vqgan::config::VqganConfig;
use vqgan::model::Vqgan;

use crate::gen::GenHeads;

/// One text-to-image request.
#[derive(Clone, Debug)]
pub struct Request<'a> {
    pub prompt: &'a str,
    /// Classifier-free guidance weight (the reference uses 5).
    pub cfg_weight: f32,
    /// Softmax temperature; `0` takes the most likely token.
    pub temperature: f32,
    pub seed: u64,
}

pub struct TextToImage {
    engine: Engine,
    heads: GenHeads,
    vq: Vqgan,
    tokenizer: QwenBpe,
    bos: String,
    image_start: String,
    pad_id: u32,
    parallel: u32,
    /// Image tokens per image, and the grid side they fill.
    tokens: usize,
    grid: u32,
    image_size: u32,
}

/// Paged-cache geometry: 16-token blocks, room for the prompt plus the image.
const BLOCK: u32 = 16;
const CONTEXT: u32 = 1024;
const MAX_PREFILL: u32 = 256;

impl TextToImage {
    /// Load the generation path from `dir` for batches of `parallel` images,
    /// the decoder's linears stored at `tier` (`BF16`, the checkpoint's own,
    /// keeps its values exactly; see this module's doc before choosing `I8`).
    pub fn load(dir: &Path, parallel: u32, tier: qwen3::Dtype) -> Result<TextToImage, String> {
        if parallel == 0 {
            return Err("at least one image per batch".into());
        }
        let (cfg, rd) = crate::model::open(dir)?;
        let dcfg = qwen3::hf::decoder_config_as(&cfg.language.to_string(), "llama")?;
        let rows = 2 * parallel;
        let weights = {
            let src = qwen3::import::nested_source(&rd, deepseekvl::import::DECODER, &dcfg)?;
            Engine::tensors_from(&dcfg, &src)?
        };
        let per_seq = CONTEXT.div_ceil(BLOCK);
        let engine = Engine::from_map_tier(dcfg, &weights, BLOCK, rows * per_seq, rows, per_seq, MAX_PREFILL, false, tier);
        drop(weights);
        let heads = GenHeads::load(&rd, &cfg, rows)?;

        let vq_cfg = VqganConfig::llamagen_vq16();
        if vq_cfg.codebook_size != cfg.gen_vision.image_token_size || vq_cfg.emb_dim != cfg.gen_vision.n_embed {
            return Err(format!("gen_vision_config is a {}-code {}-wide VQ, not LlamaGen's VQ-16", cfg.gen_vision.image_token_size, cfg.gen_vision.n_embed));
        }
        let image_size = cfg.vision.image_size;
        let vq_weights = vqgan::import::load_hf_dir(dir, crate::model::VQ_PREFIX, &vq_cfg)?;
        let vq = Vqgan::new_batched(vq_cfg, &vq_weights.tensors, image_size, image_size, gpu_core::Gpu::new(&vqgan::KERNELS), false, parallel);
        let (gh, gw) = vq.latent_size();
        if gh != gw {
            return Err(format!("a {image_size} px square gives a {gh}x{gw} token grid"));
        }
        let grid = gh;

        let dir_str = dir.to_str().ok_or("checkpoint path is not UTF-8")?;
        let tokenizer = QwenBpe::from_dir(dir_str)?;
        let tok_cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).map_err(|e| format!("tokenizer_config.json: {e}"))?)
            .map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let bos = tok_cfg["bos_token"].as_str().ok_or("tokenizer_config.json: no bos_token")?.to_string();
        let special: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("special_tokens_map.json")).map_err(|e| format!("special_tokens_map.json: {e}"))?)
            .map_err(|e| format!("special_tokens_map.json: {e}"))?;
        let pad = special["pad_token"].as_str().ok_or("special_tokens_map.json: no pad_token")?;
        let pad_id = tokenizer.special_id(pad).ok_or_else(|| format!("the tokenizer has no {pad:?} token"))?;
        tokenizer.special_id(crate::model::IMAGE_START).ok_or("the tokenizer has no <begin_of_image> token")?;
        Ok(TextToImage { engine, heads, vq, tokenizer, bos, image_start: crate::model::IMAGE_START.into(), pad_id, parallel, tokens: (grid * grid) as usize, grid, image_size })
    }

    /// Images per batch.
    pub fn parallel(&self) -> usize {
        self.parallel as usize
    }

    /// Image tokens per image.
    pub fn image_tokens(&self) -> usize {
        self.tokens
    }

    /// The conditional and unconditional prompt ids for `prompt`.
    pub fn prompt_ids(&self, prompt: &str) -> Result<(Vec<u32>, Vec<u32>), String> {
        let turns = [deepseekvl::prompt::Turn { role: deepseekvl::prompt::Role::User, content: prompt.to_string() }];
        let text = deepseekvl::prompt::render(&deepseekvl::prompt::JANUS, "", &turns, "")?;
        let cond = self.tokenizer.encode(&format!("{}{text}{}", self.bos, self.image_start));
        let mut uncond = cond.clone();
        let n = uncond.len();
        if n > 2 {
            uncond[1..n - 1].fill(self.pad_id);
        }
        Ok((cond, uncond))
    }

    /// Run `steps` guided steps from `cond`/`uncond`. At each step `choose`
    /// gets the step index and the blended logits `[parallel, vocab]` and
    /// returns one token per image. Returns each image's tokens.
    pub fn run_tokens(&mut self, cond: &[u32], uncond: &[u32], cfg_weight: f32, steps: usize, choose: &mut dyn FnMut(usize, &[f32]) -> Result<Vec<u32>, String>) -> Result<Vec<Vec<u32>>, String> {
        let rows = 2 * self.parallel as usize;
        if cond.len().max(uncond.len()) + steps > PagedDecoder::max_seq_len(&self.engine) {
            return Err(format!("a {}-token prompt and {steps} image tokens exceed the {}-token context", cond.len(), PagedDecoder::max_seq_len(&self.engine)));
        }
        let mut tables: Vec<BlockTable> = (0..rows).map(|_| BlockTable::new()).collect();
        let result = self.run_rows(&mut tables, cond, uncond, cfg_weight, steps, choose);
        for t in &mut tables {
            PagedDecoder::release_table(&mut self.engine, t);
        }
        result
    }

    fn run_rows(&mut self, tables: &mut [BlockTable], cond: &[u32], uncond: &[u32], cfg_weight: f32, steps: usize, choose: &mut dyn FnMut(usize, &[f32]) -> Result<Vec<u32>, String>) -> Result<Vec<Vec<u32>>, String> {
        let p = self.parallel as usize;
        let v = self.heads.vocab();
        let mut hidden = Vec::new();
        for (r, table) in tables.iter_mut().enumerate() {
            let ids = if r % 2 == 0 { cond } else { uncond };
            let inputs: Vec<PrefillInput> = ids.iter().map(|&t| PrefillInput::Token(t)).collect();
            hidden.extend(self.engine.prefill_mixed(table, &inputs));
        }
        let mut out = vec![Vec::with_capacity(steps); p];
        for step in 0..steps {
            let logits = self.heads.logits(&hidden);
            let blended: Vec<f32> = (0..p).flat_map(|i| model::hostmath::cfg_blend(&logits[2 * i * v..(2 * i + 1) * v], &logits[(2 * i + 1) * v..(2 * i + 2) * v], cfg_weight)).collect();
            let chosen = choose(step, &blended)?;
            if chosen.len() != p {
                return Err(format!("{} tokens chosen for {p} images", chosen.len()));
            }
            for (i, &t) in chosen.iter().enumerate() {
                out[i].push(t);
            }
            if step + 1 == steps {
                break;
            }
            let fed: Vec<u32> = chosen.iter().flat_map(|&t| [t, t]).collect();
            let embeds = self.heads.token_embeds(&fed)?;
            let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
            hidden = self.engine.forward_batched_embed(&mut refs, &embeds);
        }
        Ok(out)
    }

    /// Generate `parallel` images for `req`. `cancelled` is polled every
    /// step; `progress` sees `(step, total)`.
    pub fn generate(&mut self, req: &Request, cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<Rgb8>, String> {
        let (cond, uncond) = self.prompt_ids(req.prompt)?;
        let (v, total) = (self.heads.vocab(), self.tokens);
        let mut rng = runtime::sample::Rng::new(req.seed);
        let temperature = req.temperature;
        let codes = self.run_tokens(&cond, &uncond, req.cfg_weight, total, &mut |step, blended| {
            if cancelled() {
                return Err("cancelled".into());
            }
            progress(step + 1, total);
            Ok(blended.chunks(v).map(|l| runtime::sample::sample_logits(l, temperature, 0, &mut rng)).collect())
        })?;
        Ok(self.decode(&codes))
    }

    /// Decode each image's token grid to RGB8 (`clip((x + 1) / 2 * 255)`).
    pub fn decode(&self, codes: &[Vec<u32>]) -> Vec<Rgb8> {
        assert_eq!(codes.len(), self.parallel as usize, "one token grid per image of the batch");
        let flat: Vec<u32> = codes.iter().flatten().copied().collect();
        let px = self.vq.decode(&flat);
        let side = self.image_size as usize;
        let plane = side * side;
        px.chunks(3 * plane)
            .map(|img| {
                let mut out = vec![0u8; 3 * plane];
                for c in 0..3 {
                    for i in 0..plane {
                        out[i * 3 + c] = ((img[c * plane + i] + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0) as u8;
                    }
                }
                Rgb8 { w: side as u32, h: side as u32, px: out }
            })
            .collect()
    }

    /// The token grid side (24 for VQ-16 at 384 px).
    pub fn grid(&self) -> u32 {
        self.grid
    }
}
