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

/// Paged-cache blocks of 16 tokens; prompts prefill in chunks of 256.
const BLOCK: u32 = 16;
const MAX_PREFILL: u32 = 256;

/// What the generation build holds beside the decoder: the generation heads
/// and the VQ-16 decoder in fp32 with the activations of a 384-pixel decode,
/// measured on the released checkpoint (about 2.5 GiB) with headroom.
pub const GENERATION_EXTRA_BYTES: u64 = 3 << 30;

/// The card and per-sequence context for a generation build of `parallel`
/// images over `cards` (`(index, free bytes)`): the roomiest card, and every
/// sequence's share of the KV cache that fits beside the weights, in whole
/// blocks, up to the checkpoint's position table.
pub fn place(fp: &deepseekvl::model::Footprint, parallel: u32, cards: &[(u32, u64)]) -> Result<(u32, u32), String> {
    let &(card, free) = cards.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).ok_or("no GPU to place the model on")?;
    let rows = 2 * parallel as u64;
    let per_seq = free.saturating_sub(fp.decoder + GENERATION_EXTRA_BYTES) / (rows * fp.kv_per_token.max(1));
    let context = (per_seq.min(fp.max_context as u64) as u32) / BLOCK * BLOCK;
    // The image's own 576 tokens and a prompt.
    if context < 1024 {
        return Err(format!("{parallel} image(s) need {} GiB for the decoder and heads plus {} MiB per context token per sequence; the roomiest card has {} GiB free", (fp.decoder + GENERATION_EXTRA_BYTES) >> 30, fp.kv_per_token * rows >> 20, free >> 30));
    }
    Ok((card, context))
}

/// The begin-of-sequence text and the padding id of the checkpoint in `dir`,
/// checking that its tokenizer has the begin-of-image tag.
pub fn prompt_specials(dir: &Path, tokenizer: &QwenBpe) -> Result<(String, u32), String> {
    let read = |name: &str| -> Result<serde_json::Value, String> {
        serde_json::from_str(&std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"))?).map_err(|e| format!("{name}: {e}"))
    };
    let bos = read("tokenizer_config.json")?["bos_token"].as_str().ok_or("tokenizer_config.json: no bos_token")?.to_string();
    let special = read("special_tokens_map.json")?;
    let pad = special["pad_token"].as_str().ok_or("special_tokens_map.json: no pad_token")?;
    let pad_id = tokenizer.special_id(pad).ok_or_else(|| format!("the tokenizer has no {pad:?} token"))?;
    tokenizer.special_id(crate::model::IMAGE_START).ok_or("the tokenizer has no <begin_of_image> token")?;
    Ok((bos, pad_id))
}

/// The ids the decoder reads before the first image token: BOS, the user's
/// turn as Janus-Pro renders it, and the begin-of-image tag.
pub fn conditional_prompt(tokenizer: &QwenBpe, bos: &str, image_start: &str, prompt: &str) -> Result<Vec<u32>, String> {
    let turns = [deepseekvl::prompt::Turn { role: deepseekvl::prompt::Role::User, content: prompt.to_string() }];
    let text = deepseekvl::prompt::render(&deepseekvl::prompt::JANUS, "", &turns, "")?;
    Ok(tokenizer.encode(&format!("{bos}{text}{image_start}")))
}

/// The unconditional twin of a conditional prompt: everything between BOS and
/// the begin-of-image tag replaced by `pad_id`, which is what classifier-free
/// guidance contrasts against.
pub fn unconditional_prompt(cond: &[u32], pad_id: u32) -> Vec<u32> {
    let mut uncond = cond.to_vec();
    let n = uncond.len();
    if n > 2 {
        uncond[1..n - 1].fill(pad_id);
    }
    uncond
}

impl TextToImage {
    /// Load the generation path from `dir` for batches of `parallel` images,
    /// the decoder's linears stored at `tier` (`BF16`, the checkpoint's own,
    /// keeps its values exactly; see this module's doc before choosing `I8`),
    /// every sequence's KV cache sized for `context` tokens (the prompt and
    /// the image's tokens together).
    pub fn load(dir: &Path, parallel: u32, tier: qwen3::Dtype, context: u32) -> Result<TextToImage, String> {
        Self::load_tuned(dir, parallel, tier, context, None)
    }

    /// [`Self::load`] with the generation fine-tune in `tuned` (what `brain
    /// januspro finetune --mode generation` wrote) applied: its adapter folded
    /// into the decoder's weights and its heads replacing the checkpoint's.
    pub fn load_tuned(dir: &Path, parallel: u32, tier: qwen3::Dtype, context: u32, tuned: Option<&Path>) -> Result<TextToImage, String> {
        if parallel == 0 {
            return Err("at least one image per batch".into());
        }
        let (cfg, rd) = crate::model::open(dir)?;
        let dcfg = qwen3::hf::decoder_config_as(&cfg.language.to_string(), "llama")?;
        let rows = 2 * parallel;
        let weights = {
            let src = qwen3::import::nested_source(&rd, deepseekvl::import::DECODER, &dcfg)?;
            let mut weights = Engine::tensors_from(&dcfg, &src)?;
            if let Some(t) = tuned {
                let adapter = t.join(deepseekvl::train::ADAPTER_FILE);
                qwen3::lora::fold_adapter_into(&mut weights, adapter.to_str().ok_or("path is not UTF-8")?).map_err(|e| format!("{}: {e}", adapter.display()))?;
            }
            weights
        };
        let per_seq = context.div_ceil(BLOCK);
        let engine = Engine::from_map_tier(dcfg, &weights, BLOCK, rows * per_seq, rows, per_seq, MAX_PREFILL, false, tier);
        drop(weights);
        let mut heads = GenHeads::load(&rd, &cfg, rows)?;
        if let Some(t) = tuned {
            let file = t.join(crate::train::GENERATION_FILE);
            let st = checkpoint::st::load_safetensors(file.to_str().ok_or("path is not UTF-8")?).map_err(|e| format!("{}: {e}", file.display()))?;
            heads.apply_tuned(&st.tensors)?;
        }

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
        let (bos, pad_id) = prompt_specials(dir, &tokenizer)?;
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
        let cond = conditional_prompt(&self.tokenizer, &self.bos, &self.image_start, prompt)?;
        let uncond = unconditional_prompt(&cond, self.pad_id);
        Ok((cond, uncond))
    }

    /// Run `steps` guided steps from `cond`/`uncond`, for every image of the
    /// batch alike. At each step `choose` gets the step index and the blended
    /// logits `[parallel, vocab]` and returns one token per image. Returns
    /// each image's tokens.
    pub fn run_tokens(&mut self, cond: &[u32], uncond: &[u32], cfg_weight: f32, steps: usize, choose: &mut dyn FnMut(usize, &[f32]) -> Result<Vec<u32>, String>) -> Result<Vec<Vec<u32>>, String> {
        let p = self.parallel as usize;
        let prompts: Vec<(Vec<u32>, Vec<u32>)> = (0..p).map(|_| (cond.to_vec(), uncond.to_vec())).collect();
        self.run_images(&prompts, &vec![cfg_weight; p], steps, choose)
    }

    /// [`Self::run_tokens`] with a prompt pair and a guidance weight of its own
    /// for each of up to `parallel` images; the images share the decoder's
    /// batch, so a few images cost little more than one. `choose` sees one
    /// blended row per image given.
    pub fn run_images(&mut self, prompts: &[(Vec<u32>, Vec<u32>)], cfg_weights: &[f32], steps: usize, choose: &mut dyn FnMut(usize, &[f32]) -> Result<Vec<u32>, String>) -> Result<Vec<Vec<u32>>, String> {
        let images = prompts.len();
        if images == 0 || images > self.parallel as usize || cfg_weights.len() != images {
            return Err(format!("{images} prompts and {} guidance weights for a build of {} images", cfg_weights.len(), self.parallel));
        }
        let max_seq = PagedDecoder::max_seq_len(&self.engine);
        if let Some((cond, uncond)) = prompts.iter().find(|(c, u)| c.len().max(u.len()) + steps > max_seq) {
            return Err(format!("a {}-token prompt and {steps} image tokens exceed the {max_seq}-token context", cond.len().max(uncond.len())));
        }
        let mut tables: Vec<BlockTable> = (0..2 * images).map(|_| BlockTable::new()).collect();
        let result = self.run_rows(&mut tables, prompts, cfg_weights, steps, choose);
        for t in &mut tables {
            PagedDecoder::release_table(&mut self.engine, t);
        }
        result
    }

    fn run_rows(&mut self, tables: &mut [BlockTable], prompts: &[(Vec<u32>, Vec<u32>)], cfg_weights: &[f32], steps: usize, choose: &mut dyn FnMut(usize, &[f32]) -> Result<Vec<u32>, String>) -> Result<Vec<Vec<u32>>, String> {
        let p = prompts.len();
        let v = self.heads.vocab();
        let mut hidden = Vec::new();
        for (r, table) in tables.iter_mut().enumerate() {
            let (cond, uncond) = &prompts[r / 2];
            let ids = if r % 2 == 0 { cond } else { uncond };
            let inputs: Vec<PrefillInput> = ids.iter().map(|&t| PrefillInput::Token(t)).collect();
            hidden.extend(self.engine.prefill_mixed(table, &inputs));
        }
        let mut out = vec![Vec::with_capacity(steps); p];
        for step in 0..steps {
            let logits = self.heads.logits(&hidden);
            let blended: Vec<f32> = (0..p).flat_map(|i| model::hostmath::cfg_blend(&logits[2 * i * v..(2 * i + 1) * v], &logits[(2 * i + 1) * v..(2 * i + 2) * v], cfg_weights[i])).collect();
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

    /// Draw one image for each of `reqs` (at most `parallel`) as one batch:
    /// each with its own prompt, guidance weight, temperature and seed, so the
    /// image is the one the request would get alone. `cancelled` is polled
    /// every step; `progress` sees `(step, total)`.
    pub fn generate_many(&mut self, reqs: &[Request], cancelled: &dyn Fn() -> bool, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<Rgb8>, String> {
        let prompts = reqs.iter().map(|r| self.prompt_ids(r.prompt)).collect::<Result<Vec<_>, _>>()?;
        let weights: Vec<f32> = reqs.iter().map(|r| r.cfg_weight).collect();
        let (v, total) = (self.heads.vocab(), self.tokens);
        let mut rngs: Vec<runtime::sample::Rng> = reqs.iter().map(|r| runtime::sample::Rng::new(r.seed)).collect();
        let codes = self.run_images(&prompts, &weights, total, &mut |step, blended| {
            if cancelled() {
                return Err("cancelled".into());
            }
            progress(step + 1, total);
            Ok(blended.chunks(v).zip(reqs.iter().zip(rngs.iter_mut())).map(|(l, (r, rng))| runtime::sample::sample_logits(l, r.temperature, 0, rng)).collect())
        })?;
        Ok(self.decode(&codes))
    }

    /// Decode each image's token grid to RGB8 (`clip((x + 1) / 2 * 255)`).
    pub fn decode(&self, codes: &[Vec<u32>]) -> Vec<Rgb8> {
        assert!(!codes.is_empty() && codes.len() <= self.parallel as usize, "{} token grids for a build of {} images", codes.len(), self.parallel);
        // The decoder is built for `parallel` grids: a smaller batch is padded
        // with blank grids whose pixels are dropped.
        let mut flat: Vec<u32> = codes.iter().flatten().copied().collect();
        flat.resize(self.parallel as usize * self.tokens, 0);
        let px = self.vq.decode(&flat);
        let side = self.image_size as usize;
        let plane = side * side;
        px.chunks(3 * plane)
            .take(codes.len())
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
