// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of generative multimodal models
// for its clients. If your team needs expertise in teaching a model to draw
// your domain then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Fine-tuning Janus-Pro.
//!
//! **Understanding** is `brain-deepseekvl`'s trainer over Janus-Pro's tower,
//! roles and image tags ([`finetune_understanding`]).
//!
//! **Generation** teaches the decoder to draw. An image is encoded once by
//! the frozen VQ-16 into 576 codes. A step feeds the decoder the prompt, the
//! begin-of-image tag and the first 575 codes (each a row of the code
//! embedding through the generation aligner, spliced into the residual
//! stream), runs it to its final-norm hidden states
//! ([`qwen3::Qwen::enable_external_head`]), and reads the next code from each
//! of the 576 positions from the tag on through the generation head. The
//! cross-entropy against the image's own codes is differentiated back through
//! the head, the decoder's LoRA, the splice, the aligner and the code
//! embedding. Trainable: the adapter, the head, the aligner and the code
//! embedding; the VQ and the decoder's base stay frozen.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::TensorSource;
use data::qwen_tokenizer::QwenBpe;
use deepseekvl::train::{Hyper, Options, Outcome, StepInfo, TrainableProjector};
use model::projector::ProjectorConfig;
use model::{cosine_lr, grad_multiplier, Adam};
use qwen3::finetune::LoraInit;
use qwen3::{Dtype, Qwen, QwenConfig, IGNORE};
use vqgan::config::VqganConfig;
use vqgan::model::Vqgan;

use crate::gen::{GEN_ALIGNER_PREFIX, GEN_EMBED, GEN_HEAD_PREFIX};
use crate::model::{IMAGE_END, IMAGE_START, TOWER_PREFIX};
use crate::t2i::{conditional_prompt, prompt_specials, unconditional_prompt};

/// Fine-tune Janus-Pro's understanding path on the image-and-reply dataset in
/// `dataset` (see [`deepseekvl::train::read_dataset`]).
pub fn finetune_understanding(dir: &Path, dataset: &Path, opts: &Options, progress: &mut dyn FnMut(&StepInfo)) -> Result<Outcome, String> {
    let (cfg, rd) = crate::model::open(dir)?;
    let examples = deepseekvl::train::load_examples(dataset)?;
    let tower = Box::new(deepseekvl::tower::SiglipTower::load(&rd, TOWER_PREFIX, deepseekvl::import::ALIGNER_PREFIX, &cfg.aligner)?);
    let parts = deepseekvl::model::Parts { tower, style: deepseekvl::prompt::JANUS, wrap: Some((IMAGE_START, IMAGE_END)), language: cfg.language };
    let frontend = deepseekvl::model::Frontend::open(dir, parts)?;
    deepseekvl::train::run(frontend, &rd, deepseekvl::import::ALIGNER_PREFIX, examples, opts, progress)
}

/// The generation heads to train, with their starting weights.
pub struct GenParts {
    /// `gen_aligner`: a code embedding row in, a decoder-width row out.
    pub aligner: (ProjectorConfig, HashMap<String, Vec<f32>>),
    /// `gen_head`: a decoder hidden state in, logits over the codebook out.
    pub head: (ProjectorConfig, HashMap<String, Vec<f32>>),
    /// `gen_embed`, `[codebook, code_dim]`.
    pub embed: Vec<f32>,
    pub code_dim: usize,
    /// Codes per image (576 for VQ-16 at 384 px): the head reads this many
    /// positions, the aligner one fewer.
    pub positions: usize,
}

/// One generation example: the decoder's input tokens and the image's codes.
#[derive(Clone, Debug)]
pub struct GenPrepared {
    /// The prompt ending in the begin-of-image tag, then one placeholder per
    /// fed-back code (`codes.len() - 1`); the splice overwrites those rows.
    pub tokens: Vec<u32>,
    /// The image's VQ codes in raster order.
    pub codes: Vec<u32>,
}

impl GenPrepared {
    /// The residual row of the begin-of-image tag: the first row the head
    /// reads.
    fn tag_row(&self) -> usize {
        self.tokens.len() - self.codes.len()
    }
}

/// The gradients of every generation part.
pub struct GenGrads {
    pub head: HashMap<String, Vec<f32>>,
    pub aligner: HashMap<String, Vec<f32>>,
    /// `[codebook, code_dim]`: nonzero only at the codes that were fed back.
    pub embed: Vec<f32>,
}

/// The generation fine-tune: the decoder, its generation heads and the splice
/// between them.
pub struct GenTrainer {
    decoder: Qwen,
    aligner: TrainableProjector,
    head: TrainableProjector,
    embed: Vec<f32>,
    /// AdamW's first and second moments of the code embedding.
    embed_state: (Vec<f32>, Vec<f32>),
    code_dim: usize,
    positions: usize,
    block: u32,
    splice_at: Option<usize>,
}

/// Mean cross-entropy of `logits` (`[rows, vocab]`) against `targets`, and
/// its gradient.
fn cross_entropy(logits: &[f32], targets: &[u32], vocab: usize) -> (f32, Vec<f32>) {
    let rows = targets.len();
    let mut grad = vec![0.0f32; logits.len()];
    let mut loss = 0.0f64;
    for (r, &t) in targets.iter().enumerate() {
        let row = &logits[r * vocab..(r + 1) * vocab];
        let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let sum: f64 = row.iter().map(|&l| (l as f64 - max).exp()).sum();
        loss -= row[t as usize] as f64 - max - sum.ln();
        for (j, g) in grad[r * vocab..(r + 1) * vocab].iter_mut().enumerate() {
            let p = (row[j] as f64 - max).exp() / sum;
            *g = ((p - if j == t as usize { 1.0 } else { 0.0 }) / rows as f64) as f32;
        }
    }
    ((loss / rows as f64) as f32, grad)
}

impl GenTrainer {
    /// Build the trainer for rows of at most `block` tokens. `cfg` is the
    /// decoder's configuration: with `lora` set, the base (at `dt`) is frozen
    /// under fresh adapters drawn from `seed`; without, every decoder weight
    /// trains. `base` is read while the decoder is built and not after.
    pub fn new(cfg: QwenConfig, base: Box<dyn TensorSource + '_>, dt: Dtype, block: u32, parts: GenParts, seed: u64) -> Result<GenTrainer, String> {
        if parts.aligner.0.out_dim != cfg.d_model || parts.head.0.input_dim != cfg.d_model {
            return Err(format!("the generation heads read and write {}- and {}-wide rows for a {}-wide decoder", parts.head.0.input_dim, parts.aligner.0.out_dim, cfg.d_model));
        }
        if parts.positions < 2 || parts.aligner.0.input_dim as usize != parts.code_dim || parts.embed.len() % parts.code_dim != 0 || parts.embed.len() / parts.code_dim != parts.head.0.out_dim as usize {
            return Err("the code embedding, the aligner's input and the head's vocabulary disagree".to_string());
        }
        let mut decoder = if cfg.lora.is_some() {
            let init = LoraInit::fresh(&cfg, base, seed);
            Qwen::new_lora_dt(cfg, 1, block, &init, dt)
        } else {
            let shard = qwen3::Shard::whole(cfg.n_layers as usize);
            Qwen::new_shard(cfg, 1, block, &*base, true, shard)
        };
        decoder.enable_external_head();
        let n = parts.embed.len();
        Ok(GenTrainer {
            decoder,
            aligner: TrainableProjector::new(parts.aligner.0, parts.aligner.1, parts.positions - 1)?,
            head: TrainableProjector::new(parts.head.0, parts.head.1, parts.positions)?,
            embed_state: (vec![0.0; n], vec![0.0; n]),
            embed: parts.embed,
            code_dim: parts.code_dim,
            positions: parts.positions,
            block,
            splice_at: None,
        })
    }

    /// The decoder, for saving its adapter.
    pub fn decoder(&self) -> &Qwen {
        &self.decoder
    }

    pub fn head_weights(&self) -> HashMap<String, Vec<f32>> {
        self.head.weights()
    }

    pub fn aligner_weights(&self) -> HashMap<String, Vec<f32>> {
        self.aligner.weights()
    }

    /// The code embedding, `[codebook, code_dim]`.
    pub fn embed(&self) -> &[f32] {
        &self.embed
    }

    pub fn set_head_weights(&mut self, weights: &HashMap<String, Vec<f32>>) {
        self.head.set_weights(weights);
    }

    pub fn set_aligner_weights(&mut self, weights: &HashMap<String, Vec<f32>>) {
        self.aligner.set_weights(weights);
    }

    pub fn set_embed(&mut self, embed: &[f32]) {
        assert_eq!(embed.len(), self.embed.len(), "the code embedding keeps its shape");
        self.embed.copy_from_slice(embed);
    }

    /// Run `p` through the aligner, the decoder and the head: the loss, and
    /// (when `backward`) every gradient, the decoder's left accumulated.
    fn run(&mut self, p: &GenPrepared, backward: bool) -> (f32, Option<GenGrads>) {
        assert_eq!(p.codes.len(), self.positions, "an example has {} image codes", self.positions);
        let fed = self.positions - 1;
        assert_eq!(p.tokens.len(), p.tag_row() + self.positions, "the tokens are the prompt and {fed} placeholders");
        assert!(p.tokens.len() <= self.block as usize, "an example of {} tokens does not fit the {}-token block", p.tokens.len(), self.block);
        let (tag, d, vocab) = (p.tag_row(), self.decoder.cfg.d_model as usize, self.head.cfg().out_dim as usize);
        // The begin-of-image tag is the last prompt token and predicts the
        // first code; the fed-back rows (codes 0..) follow it.
        let row0 = tag + 1;
        if self.splice_at != Some(row0) {
            self.decoder.enable_mm_splice(row0 as u32, fed as u32);
            self.splice_at = Some(row0);
        }

        let fed_codes: Vec<f32> = p.codes[..fed].iter().flat_map(|&c| self.embed[c as usize * self.code_dim..(c as usize + 1) * self.code_dim].iter().copied()).collect();
        self.decoder.write_img_embeds(&self.aligner.forward(&[fed_codes]));
        let mut x = p.tokens.clone();
        x.resize(self.block as usize, 0);
        self.decoder.set_batch(&x, &vec![IGNORE; self.block as usize]);
        if backward {
            self.decoder.zero_grads();
            self.aligner.zero_grads();
            self.head.zero_grads();
        }
        let hidden = self.decoder.forward_hidden();
        let selected = hidden[tag * d..(tag + self.positions) * d].to_vec();
        let logits = self.head.forward(&[selected]);
        let (loss, d_logits) = cross_entropy(&logits, &p.codes, vocab);
        if !backward {
            return (loss, None);
        }

        let d_selected = self.head.backward(&d_logits).remove(0);
        let mut d_hidden = vec![0.0f32; self.block as usize * d];
        d_hidden[tag * d..(tag + self.positions) * d].copy_from_slice(&d_selected);
        self.decoder.backward_hidden(&d_hidden);
        let d_fed = self.aligner.backward(&self.decoder.read_d_img_embeds()).remove(0);
        let mut embed = vec![0.0f32; self.embed.len()];
        for (i, &c) in p.codes[..fed].iter().enumerate() {
            for k in 0..self.code_dim {
                embed[c as usize * self.code_dim + k] += d_fed[i * self.code_dim + k];
            }
        }
        (loss, Some(GenGrads { head: self.head.grads(), aligner: self.aligner.grads(), embed }))
    }

    /// The loss on `p` with the current weights.
    pub fn loss(&mut self, p: &GenPrepared) -> f32 {
        self.run(p, false).0
    }

    /// The loss on `p` and the gradient of every generation part (the
    /// decoder's gradients stay accumulated in it).
    pub fn loss_and_grads(&mut self, p: &GenPrepared) -> (f32, GenGrads) {
        let (loss, grads) = self.run(p, true);
        (loss, grads.expect("a backward pass ran"))
    }

    /// One optimiser step (1-based `t`) on `p`; returns its loss.
    pub fn step(&mut self, p: &GenPrepared, t: u32, h: &Hyper) -> f32 {
        let (loss, grads) = self.run(p, true);
        let grads = grads.expect("a backward pass ran");
        let clip = (h.grad_clip > 0.0).then_some(h.grad_clip);
        let scale = grad_multiplier(grads.embed.iter().map(|g| (*g as f64).powi(2)).sum(), clip, 1.0);
        let (m, v) = &mut self.embed_state;
        Adam::default().update_slice(t, h.aligner_lr, h.weight_decay, scale, &mut self.embed, m, v, &grads.embed);
        self.head.step(t, h.aligner_lr, h.weight_decay, h.grad_clip);
        self.aligner.step(t, h.aligner_lr, h.weight_decay, h.grad_clip);
        self.decoder.adamw_step(t, h.lr, h.weight_decay, Adam::default(), clip, 1.0);
        self.decoder.poll_wait();
        loss
    }
}

/// The settings of a generation fine-tune.
#[derive(Clone, Debug)]
pub struct GenOptions {
    /// The adapter, schedule and dtype, as for understanding.
    pub options: Options,
    /// The share of steps whose prompt is replaced by padding, which teaches
    /// the unconditional branch classifier-free guidance contrasts against.
    pub cfg_dropout: f32,
}

/// What a generation fine-tune leaves.
pub struct GenOutcome {
    pub initial_loss: f32,
    pub final_loss: Option<f32>,
    pub examples: usize,
    pub block: u32,
    pub trainer: GenTrainer,
}

/// The file [`GenOutcome::save`] writes the generation parts to, beside
/// [`deepseekvl::train::ADAPTER_FILE`].
pub const GENERATION_FILE: &str = "generation.safetensors";

impl GenOutcome {
    /// Write the trained parameters into `dir`: the decoder's adapter as
    /// `adapter.safetensors` and the head, aligner and code embedding as
    /// `generation.safetensors` (named as the checkpoint names them).
    pub fn save(&self, dir: &Path, card_id: &str, base_id: &str) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
        qwen3::lora::save_adapter(&path(deepseekvl::train::ADAPTER_FILE), self.trainer.decoder(), card_id, base_id, None).map_err(|e| format!("{}: {e}", path(deepseekvl::train::ADAPTER_FILE)))?;
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        for (prefix, projector) in [(GEN_HEAD_PREFIX, &self.trainer.head), (GEN_ALIGNER_PREFIX, &self.trainer.aligner)] {
            let weights = projector.weights();
            let mut names: Vec<&String> = weights.keys().collect();
            names.sort();
            tensors.extend(names.into_iter().map(|n| (format!("{prefix}{n}"), projector.shape(n), weights[n].clone())));
        }
        let rows = self.trainer.embed.len() / self.trainer.code_dim;
        tensors.push((GEN_EMBED.to_string(), vec![rows as u64, self.trainer.code_dim as u64], self.trainer.embed.clone()));
        checkpoint::st::save_safetensors(&path(GENERATION_FILE), &tensors, &serde_json::json!({ "generation": "janus-pro" }), None).map_err(|e| format!("{}: {e}", path(GENERATION_FILE)))
    }
}

/// A generation dataset's examples: each line of `train.jsonl` holds a
/// `prompt` and the `image` (a path relative to `dir`) to draw for it.
pub fn read_generation_dataset(dir: &Path) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    let file = dir.join("train.jsonl");
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let at = |why: &str| format!("{}:{}: {why}", file.display(), n + 1);
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| at(&e.to_string()))?;
        let prompt = v["prompt"].as_str().ok_or_else(|| at("no \"prompt\""))?;
        let image = v["image"].as_str().ok_or_else(|| at("no \"image\" path"))?;
        out.push((prompt.to_string(), dir.join(image)));
    }
    if out.is_empty() {
        return Err(format!("{}: no examples", file.display()));
    }
    Ok(out)
}

/// Fine-tune Janus-Pro to generate, from the checkpoint directory `dir`, on
/// the prompt-and-image dataset in `dataset` ([`read_generation_dataset`]):
/// every image is encoded once by the frozen VQ-16, which is then released,
/// and the decoder's adapter, the generation head, the aligner and the code
/// embedding train on the codes.
pub fn finetune_generation(dir: &Path, dataset: &Path, opts: &GenOptions, progress: &mut dyn FnMut(&StepInfo)) -> Result<GenOutcome, String> {
    let (cfg, rd) = crate::model::open(dir)?;
    let base_cfg = qwen3::hf::decoder_config_as(&cfg.language.to_string(), "llama")?;
    let processor = deepseekvl::preprocess::ImageProcessor::from_dir(dir)?;
    let dir_str = dir.to_str().ok_or("checkpoint path is not UTF-8")?;
    let tokenizer = QwenBpe::from_dir(dir_str)?;
    let (bos, pad_id) = prompt_specials(dir, &tokenizer)?;

    let vq_cfg = VqganConfig::llamagen_vq16();
    if vq_cfg.codebook_size != cfg.gen_vision.image_token_size || vq_cfg.emb_dim != cfg.gen_vision.n_embed {
        return Err(format!("gen_vision_config is a {}-code {}-wide VQ, not LlamaGen's VQ-16", cfg.gen_vision.image_token_size, cfg.gen_vision.n_embed));
    }
    let size = cfg.vision.image_size;
    let mut examples: Vec<(Vec<u32>, Vec<u32>, usize)> = Vec::new();
    {
        let weights = vqgan::import::load_hf_dir(dir, crate::model::VQ_PREFIX, &vq_cfg)?;
        let vq = Vqgan::new_batched(vq_cfg, &weights.tensors, size, size, gpu_core::Gpu::new(&vqgan::KERNELS), false, 1);
        for (prompt, path) in read_generation_dataset(dataset)? {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let image = imaging::codec::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            let (codes, _) = vq.encode(&processor.pixel_values(&image)?);
            let cond = conditional_prompt(&tokenizer, &bos, IMAGE_START, &prompt)?;
            let tag = cond.len();
            let mut tokens = cond;
            tokens.extend(std::iter::repeat(pad_id).take(codes.len() - 1));
            examples.push((tokens, codes, tag));
        }
    }
    let positions = examples[0].1.len();
    if examples.iter().any(|e| e.1.len() != positions) {
        return Err("every image gives the same number of codes".to_string());
    }

    let o = &opts.options;
    let longest = examples.iter().map(|e| e.0.len()).max().expect("a dataset has examples");
    let block = match o.block {
        Some(b) if (b as usize) < longest => return Err(format!("--block {b} is shorter than the longest example ({longest} tokens)")),
        Some(b) => b,
        None => longest as u32,
    };
    let (h, a) = (&cfg.gen_head, &cfg.gen_aligner);
    let head_cfg = ProjectorConfig::from_type("mlp_gelu", 2, h.n_embed, h.image_token_embed)?.with_out_dim(h.image_token_size)?;
    let aligner_cfg = ProjectorConfig::from_type(&a.projector_type, a.depth, a.input_dim, a.n_embed)?;
    let parts = GenParts {
        head: (head_cfg, crate::gen::head_weights(&rd, &head_cfg)?),
        aligner: (aligner_cfg, deepseekvl::import::aligner_weights(&rd, GEN_ALIGNER_PREFIX, &aligner_cfg)?),
        embed: rd.tensor(GEN_EMBED).ok_or_else(|| format!("{GEN_EMBED} is not in the checkpoint"))?,
        code_dim: cfg.gen_vision.n_embed as usize,
        positions,
    };
    let mut decoder_cfg = base_cfg.clone();
    if o.rank > 0 {
        decoder_cfg.lora = Some(qwen3::LoraCfg { rank: o.rank, alpha: o.alpha, targets: o.targets.clone() });
    }
    let base = qwen3::import::nested_source(&rd, deepseekvl::import::DECODER, &base_cfg)?;
    let mut trainer = GenTrainer::new(decoder_cfg, Box::new(base), o.dtype, block, parts, o.seed)?;

    let fit = o.hyper.fit_opts(o.steps, 1, block, o.lr, o.seed);
    let hyper_at = |step: u32| Hyper { lr: cosine_lr(step, &fit), aligner_lr: cosine_lr(step, &fit) * o.aligner_lr.map_or(0.1, |x| x / o.lr), weight_decay: fit.weight_decay, grad_clip: fit.grad_clip };
    let prepared = |i: usize, unconditional: bool| {
        let (tokens, codes, tag) = &examples[i];
        let mut tokens = tokens.clone();
        if unconditional {
            let uncond = unconditional_prompt(&tokens[..*tag], pad_id);
            tokens[..*tag].copy_from_slice(&uncond);
        }
        GenPrepared { tokens, codes: codes.clone() }
    };
    let initial_loss = (0..examples.len()).map(|i| trainer.loss(&prepared(i, false))).sum::<f32>() / examples.len() as f32;
    let mut order = data::rng::Rng::new(o.seed ^ 0x5eed);
    let mut queue: Vec<usize> = Vec::new();
    let mut final_loss = None;
    for step in 0..o.steps {
        if queue.is_empty() {
            queue = (0..examples.len()).collect();
            for i in (1..queue.len()).rev() {
                queue.swap(i, (order.next_u64() % (i as u64 + 1)) as usize);
            }
        }
        let h = hyper_at(step);
        let unconditional = order.next_f32() < opts.cfg_dropout;
        let loss = trainer.step(&prepared(queue.pop().expect("refilled"), unconditional), step + 1, &h);
        final_loss = Some(loss);
        progress(&StepInfo { step: step + 1, steps: o.steps, loss, lr: h.lr });
    }
    Ok(GenOutcome { initial_loss, final_loss, examples: examples.len(), block, trainer })
}
