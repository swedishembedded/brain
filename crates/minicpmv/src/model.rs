// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The composite: the image processor and the vision tower produce visual
//! rows, the prompt layout says where they go, and the Qwen3.5 decoder - a
//! `brain-qwen35` paged engine - is prefilled from the token embeddings with
//! those rows written over the placeholder rows (the reference's
//! `masked_scatter`), then decodes. Tower and decoder share one device.

use std::path::Path;

use data::rng::Rng;
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;
use imaging::Rgb8;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen35::serve::Engine;

use crate::config::{Downsample, MiniCpmVConfig};
use crate::preprocess::{process_image, ImageOptions, ProcessedImage};
use crate::prompt::{view_layout, Formatter, Message, Prompt, IMAGE_PAD};
use crate::vision::{Tower, View, VISION_PIPELINES};

/// How one request is processed and decoded.
#[derive(Clone, Debug)]
pub struct GenOptions {
    pub max_new: usize,
    /// `<= 0` is greedy.
    pub temperature: f32,
    /// `0` disables top-k.
    pub top_k: usize,
    /// Outside `(0, 1)` disables nucleus sampling.
    pub top_p: f32,
    pub seed: u64,
    pub enable_thinking: bool,
    /// The checkpoint's mode when `None`.
    pub downsample: Option<Downsample>,
    /// The processor's `max_slice_nums` when `None`.
    pub max_slices: Option<u32>,
    pub use_image_id: Option<bool>,
}

impl Default for GenOptions {
    fn default() -> GenOptions {
        GenOptions {
            max_new: 256,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: data::rng::random_seed(),
            enable_thinking: false,
            downsample: None,
            max_slices: None,
            use_image_id: None,
        }
    }
}

/// A prepared request: the prompt and the input embeddings the decoder is
/// prefilled from.
pub struct Prepared {
    pub prompt: Prompt,
    pub embeds: Vec<f32>,
}

/// Device bytes a served instance holds beyond its weights for the vision
/// tower's activations: the packed views of a request (a 36-slice image is
/// about 37k patch rows at 4x) through the encoder's scratch and score slab.
pub const TOWER_SCRATCH_BYTES: u64 = 3 << 30;

/// Device bytes of a built composite: fp32 weights, the decoder's KV pool and
/// per-sequence Gated-DeltaNet state, and the tower's scratch.
pub fn footprint(cfg: &MiniCpmVConfig, max_seq_len: u32, max_concurrent: u32) -> u64 {
    let vision: u64 = crate::vision::tensor_manifest(cfg).iter().map(|(_, n)| *n as u64).sum();
    let text: u64 = cfg.text.param_list().iter().map(|(_, n)| *n as u64).sum();
    let t = &cfg.text;
    let full = t.layer_types().iter().filter(|l| **l == qwen35::config::LayerType::Full).count() as u64;
    let kv = full * 2 * (t.n_kv_heads * t.head_dim) as u64 * max_seq_len as u64 * max_concurrent as u64 * 4;
    let gdn = (t.n_layers as u64 - full) * t.gdn_state_len() * max_concurrent as u64 * 4;
    4 * (vision + text) + kv + gdn + TOWER_SCRATCH_BYTES
}

pub struct MiniCpmV {
    pub cfg: MiniCpmVConfig,
    pub tower: Tower,
    pub engine: Engine,
    pub fmt: Formatter,
    /// The processor defaults from `preprocessor_config.json`.
    pub image_opts: ImageOptions,
    /// End-of-generation ids (`generation_config.json`'s `eos_token_id`).
    pub eos: Vec<u32>,
}

/// `generation_config.json`'s end ids, or the template's end of turn.
fn eos_ids(dir: &Path, fmt: &Formatter) -> Vec<u32> {
    let from_file = std::fs::read(dir.join("generation_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| match &v["eos_token_id"] {
            serde_json::Value::Array(a) => Some(a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect::<Vec<_>>()),
            serde_json::Value::Number(n) => n.as_u64().map(|n| vec![n as u32]),
            _ => None,
        });
    from_file.filter(|v| !v.is_empty()).unwrap_or_else(|| fmt.tokenizer.special_id("<|im_end|>").into_iter().collect())
}

impl MiniCpmV {
    /// Load the checkpoint in `dir` onto `gpu`. `max_seq_len` bounds one
    /// sequence's prompt plus generation; `max_concurrent` how many sequences
    /// the decoder's KV pool holds at once.
    pub fn load(dir: &Path, gpu: &Gpu, max_seq_len: u32, max_concurrent: u32, kv: KvTier) -> Result<MiniCpmV, String> {
        let cfg = MiniCpmVConfig::from_dir(dir)?;
        let fmt = Formatter::from_dir(dir)?;
        let image_opts = ImageOptions::from_dir(dir, cfg.vision.patch, cfg.downsample)?;
        let weights = crate::import::vision_weights(dir, &cfg)?;
        let tower = Tower::new_on(gpu.new_like(VISION_PIPELINES), cfg.clone(), &weights);
        drop(weights);
        let src = crate::import::open_decoder(dir, &cfg)?;
        let text = qwen35::config::Qwen35Config { block_size: max_seq_len, ..cfg.text.clone() };
        let engine = Engine::from_source_on(gpu, text, &src, max_seq_len, max_concurrent, kv);
        let eos = eos_ids(dir, &fmt);
        Ok(MiniCpmV { cfg, tower, engine, fmt, image_opts, eos })
    }

    fn options(&self, opts: &GenOptions) -> ImageOptions {
        let mut o = self.image_opts.clone();
        if let Some(m) = opts.downsample {
            o.downsample = m;
        }
        if let Some(n) = opts.max_slices {
            o.max_slices = n;
        }
        if let Some(b) = opts.use_image_id {
            o.use_image_id = b;
        }
        o
    }

    /// Process `images`, encode them, lay out and tokenize the conversation,
    /// and build the decoder's input embeddings.
    pub fn prepare(&self, messages: &[Message], images: &[Rgb8], opts: &GenOptions) -> Result<Prepared, String> {
        let rendered = self.fmt.render(messages, true, opts.enable_thinking)?;
        self.prepare_text(&rendered, images, opts)
    }

    /// [`Self::prepare`] for a conversation already rendered by the chat
    /// template (one `<|image_pad|>` per image, in order).
    pub fn prepare_text(&self, rendered: &str, images: &[Rgb8], opts: &GenOptions) -> Result<Prepared, String> {
        let io = self.options(opts);
        let processed: Vec<ProcessedImage> = images.iter().map(|img| process_image(img, &io)).collect();
        let layouts: Vec<String> = processed.iter().map(|p| view_layout(p, io.downsample, IMAGE_PAD)).collect();
        let prompt = self.fmt.expand(rendered, &layouts, &[], io.use_image_id)?;
        let visual = if processed.is_empty() {
            Vec::new()
        } else {
            let views: Vec<View> = processed.iter().flat_map(|p| p.views.iter().copied()).collect();
            let patches: Vec<f32> = processed.iter().flat_map(|p| p.patches.iter().copied()).collect();
            self.tower.encode(&patches, &views, io.downsample)
        };
        let embeds = self.splice(&prompt, &visual, &[])?;
        Ok(Prepared { prompt, embeds })
    }

    /// The token embeddings of `prompt` with the image pad rows replaced by
    /// `image_rows` and the video pad rows by `video_rows`, each in order.
    pub fn splice(&self, prompt: &Prompt, image_rows: &[f32], video_rows: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.cfg.text.d_model as usize;
        let mut embeds = self.engine.embed_tokens(&prompt.ids);
        let (mut at_image, mut at_video) = (0usize, 0usize);
        for &(row0, n) in &prompt.runs {
            let (rows, at) = if prompt.ids[row0 as usize] == self.fmt.image_pad { (image_rows, &mut at_image) } else { (video_rows, &mut at_video) };
            let len = n as usize * d;
            let src = rows.get(*at..*at + len).ok_or_else(|| format!("minicpmv: the prompt has more placeholder rows than the {} visual rows encoded", rows.len() / d))?;
            embeds[row0 as usize * d..row0 as usize * d + len].copy_from_slice(src);
            *at += len;
        }
        if at_image != image_rows.len() || at_video != video_rows.len() {
            return Err(format!(
                "minicpmv: visual rows and placeholders disagree: {} image rows placed of {}, {} video rows placed of {}",
                at_image / d,
                image_rows.len() / d,
                at_video / d,
                video_rows.len() / d
            ));
        }
        Ok(embeds)
    }

    /// Generate from a prepared request. `on_token` sees every generated
    /// token (end-of-generation ids excluded) and stops generation by
    /// returning `false`.
    pub fn generate_prepared(&mut self, prepared: &Prepared, opts: &GenOptions, on_token: &mut dyn FnMut(u32) -> bool) -> Result<Vec<u32>, String> {
        self.generate_batch(&[(prepared, opts)], &mut |_, t| on_token(t)).pop().expect("one result per request")
    }

    /// Generate for several prepared requests at once: each is prefilled on
    /// its own, then every live sequence decodes in ONE batched step per
    /// token on the shared KV pool, each with its own sampling. `on_token(r,
    /// id)` streams request `r`'s tokens and stops that request alone by
    /// returning `false`. Requests beyond the pool's free sequences wait for
    /// the next wave. A request that does not fit the sequence capacity fails
    /// alone.
    pub fn generate_batch(&mut self, reqs: &[(&Prepared, &GenOptions)], on_token: &mut dyn FnMut(usize, u32) -> bool) -> Vec<Result<Vec<u32>, String>> {
        let mut results: Vec<Result<Vec<u32>, String>> = reqs
            .iter()
            .map(|(p, o)| {
                let n = p.prompt.ids.len();
                if n + o.max_new > self.engine.max_seq_len() {
                    Err(format!("minicpmv: {n} prompt tokens + {} new exceed the {}-token sequence capacity", o.max_new, self.engine.max_seq_len()))
                } else {
                    Ok(Vec::new())
                }
            })
            .collect();
        let pending: Vec<usize> = (0..reqs.len()).filter(|&r| results[r].is_ok()).collect();
        for wave in pending.chunks(self.engine.free_blocks().max(1) as usize) {
            self.decode_wave(reqs, wave, &mut results, on_token);
        }
        results
    }

    /// Prefill and decode the requests `wave` names, together.
    fn decode_wave(&mut self, reqs: &[(&Prepared, &GenOptions)], wave: &[usize], results: &mut [Result<Vec<u32>, String>], on_token: &mut dyn FnMut(usize, u32) -> bool) {
        struct Live {
            r: usize,
            table: BlockTable,
            rng: Rng,
            next: u32,
        }
        let vocab = self.engine.vocab();
        let mut live: Vec<Live> = Vec::with_capacity(wave.len());
        for &r in wave {
            let (p, o) = reqs[r];
            let mut table = BlockTable::new();
            let mut rng = Rng::new(o.seed);
            let hidden = self.engine.prefill_embeds(&mut table, &p.embeds);
            let next = if o.temperature <= 0.0 {
                self.engine.admit_greedy(&hidden)
            } else {
                qwen35::sample::sample_logits(&self.engine.logits(&hidden), o.temperature, o.top_k, o.top_p, &mut rng)
            };
            live.push(Live { r, table, rng, next });
        }
        loop {
            // Accept each sequence's pending token or retire the sequence.
            let mut keep = Vec::with_capacity(live.len());
            for mut l in live.drain(..) {
                let max_new = reqs[l.r].1.max_new;
                let out = results[l.r].as_mut().expect("only admitted requests are live");
                let done = self.eos.contains(&l.next) || out.len() >= max_new || {
                    out.push(l.next);
                    !on_token(l.r, l.next) || out.len() >= max_new
                };
                if done {
                    self.engine.release_table(&mut l.table);
                } else {
                    keep.push(l);
                }
            }
            live = keep;
            if live.is_empty() {
                return;
            }
            let inputs: Vec<u32> = live.iter().map(|l| l.next).collect();
            let greedy = live.iter().all(|l| reqs[l.r].1.temperature <= 0.0);
            let mut tables: Vec<&mut BlockTable> = live.iter_mut().map(|l| &mut l.table).collect();
            let next: Vec<u32> = if greedy {
                self.engine.forward_batched_greedy(&mut tables, &inputs)
            } else {
                let logits = self.engine.forward_batched_logits(&mut tables, &inputs);
                live.iter_mut()
                    .zip(logits.chunks_exact(vocab))
                    .map(|(l, row)| {
                        let o = reqs[l.r].1;
                        qwen35::sample::sample_logits(row, o.temperature, o.top_k, o.top_p, &mut l.rng)
                    })
                    .collect()
            };
            for (l, n) in live.iter_mut().zip(next) {
                l.next = n;
            }
        }
    }

    /// [`Self::prepare`] then [`Self::generate_prepared`].
    pub fn generate(&mut self, messages: &[Message], images: &[Rgb8], opts: &GenOptions, on_token: &mut dyn FnMut(u32) -> bool) -> Result<Vec<u32>, String> {
        let prepared = self.prepare(messages, images, opts)?;
        self.generate_prepared(&prepared, opts, on_token)
    }

    /// The longest prompt plus reply one sequence holds.
    pub fn engine_capacity(&self) -> usize {
        self.engine.max_seq_len()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.fmt.tokenizer.decode(ids)
    }
}
