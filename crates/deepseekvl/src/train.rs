// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language models for
// its clients. If your team needs expertise in adapting multimodal models
// to your images and conversations then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Fine-tuning the composite: the vision towers stay frozen (their features
//! are extracted once per image), the aligner trains from the checkpoint's
//! own weights, and the decoder trains as a LoRA (or whole, when its
//! configuration carries no adapter).
//!
//! A step runs the aligner over an image's cached feature streams, splices
//! its rows into the decoder's residual stream ([`qwen3::Qwen::enable_mm_splice`]),
//! differentiates the reply's cross-entropy, and carries the gradient of the
//! spliced rows back through the aligner. The decoder's optimiser is its own
//! device AdamW; the aligner's is AdamW on the host, which is cheap at its
//! size and keeps its parameters in one place.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;
use data::tokenizer::Tokenizer;
use gpu_core::{DeviceBuffer, Gpu};
use imaging::pixels::Rgb8;
use model::projector::{MlpProjector, ProjectorConfig, PROJECTOR_PIPELINES};
use model::{cosine_lr, grad_multiplier, Adam};
use qwen3::finetune::LoraInit;
use qwen3::{Dtype, Qwen, QwenConfig, IGNORE};

use crate::model::Frontend;
use crate::prompt::{Role, Turn};

/// One training example ready for the decoder: the image's frozen-tower
/// feature streams and the token rows of the conversation with the image's
/// rows marked in place.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// `[rows, width]` per feature stream, as [`crate::tower::Features::streams`].
    pub streams: Vec<Vec<f32>>,
    /// The decoder's input tokens; the image's rows are placeholders the
    /// splice overwrites.
    pub tokens: Vec<u32>,
    /// `targets[i]` is what follows `tokens[i]`, or [`IGNORE`]: only the
    /// reply (and its end-of-sentence) is supervised.
    pub targets: Vec<u32>,
    /// The first residual row of the image's rows.
    pub row0: u32,
}

impl Frontend {
    /// Turn an image and a conversation ending in the reply to learn into a
    /// training example. The prompt is rendered as it is at inference (BOS,
    /// system prompt, the user's turn with its `<image_placeholder>`, the open
    /// `Assistant:`), and the reply follows it closed by the end-of-sentence
    /// token; one image per example.
    pub fn prepare(&self, image: &Rgb8, turns: &[Turn]) -> Result<Prepared, String> {
        let (reply, asked) = match turns.split_last() {
            Some((last, rest)) if last.role == Role::Assistant && !rest.is_empty() => (last, rest),
            _ => return Err("a training conversation ends with the assistant reply to learn, after a user turn".to_string()),
        };
        let prompt = self.prompt_ids(asked)?;
        let placeholders = prompt.iter().filter(|&&t| t == self.splice.image_id).count();
        if placeholders != 1 {
            return Err(format!("a training example has one image, and its prompt has {placeholders} image placeholders"));
        }
        let reply_ids = self.tokenizer.encode(&format!(" {}{}", reply.content.trim(), self.eos));
        if reply_ids.last() != Some(&self.eos_id) {
            return Err("the reply does not end in the end-of-sentence token".to_string());
        }
        let mut sequence = self.splice.expand_ids(&prompt);
        let image_at = sequence.iter().position(|&t| t == self.splice.image_id).expect("the placeholder was expanded");
        let supervised_from = sequence.len();
        sequence.extend(&reply_ids);

        let n = sequence.len() - 1;
        let tokens = sequence[..n].to_vec();
        let targets = (0..n).map(|i| if i + 1 >= supervised_from { sequence[i + 1] } else { IGNORE }).collect();
        let streams = self.tower.encode(&self.processor.pixel_values(image)?).streams;
        Ok(Prepared { streams, tokens, targets, row0: image_at as u32 })
    }
}

/// The schedule-dependent settings of one step.
#[derive(Clone, Copy, Debug)]
pub struct Hyper {
    /// The decoder's learning rate.
    pub lr: f32,
    /// The aligner's, usually smaller: its weights are trained already.
    pub aligner_lr: f32,
    pub weight_decay: f32,
    /// Global gradient-norm clip, per optimiser; `0` disables.
    pub grad_clip: f32,
}

/// The aligner with its host-side AdamW state.
struct Aligner {
    gpu: Gpu,
    projector: MlpProjector,
    inputs: Vec<DeviceBuffer>,
    d_out: DeviceBuffer,
    /// `(master weights, m, v)` by parameter name.
    state: HashMap<String, (Vec<f32>, Vec<f32>, Vec<f32>)>,
    rows: usize,
}

impl Aligner {
    fn new(cfg: ProjectorConfig, weights: HashMap<String, Vec<f32>>, rows: usize) -> Result<Aligner, String> {
        let gpu = Gpu::new(PROJECTOR_PIPELINES);
        let projector = MlpProjector::new(&gpu, cfg, rows as u32, &weights)?;
        let inputs = (0..cfg.inputs()).map(|_| gpu.storage(rows as u64 * cfg.input_dim as u64)).collect();
        let d_out = gpu.storage(rows as u64 * cfg.out_dim as u64);
        let state = weights.into_iter().map(|(n, w)| (n, (w.clone(), vec![0.0; w.len()], vec![0.0; w.len()]))).collect();
        Ok(Aligner { gpu, projector, inputs, d_out, state, rows })
    }

    /// The aligner's rows for `streams`.
    fn forward(&self, streams: &[Vec<f32>]) -> Vec<f32> {
        assert_eq!(streams.len(), self.inputs.len(), "the aligner reads {} feature stream(s)", self.inputs.len());
        for (buf, s) in self.inputs.iter().zip(streams) {
            self.gpu.write_f32(buf, s);
        }
        let refs: Vec<&DeviceBuffer> = self.inputs.iter().collect();
        self.gpu.submit(&[], &self.projector.forward(&self.gpu, &refs));
        self.gpu.read(self.projector.out(), self.rows * self.projector.cfg.out_dim as usize)
    }

    /// Accumulate the parameter gradients for `d_rows`, the loss's gradient
    /// at the aligner's output (after a [`Self::forward`] on the same streams).
    fn backward(&self, d_rows: &[f32]) {
        self.gpu.write_f32(&self.d_out, d_rows);
        let refs: Vec<&DeviceBuffer> = self.inputs.iter().collect();
        self.gpu.submit(&[], &self.projector.backward(&self.gpu, &refs, &self.d_out, None));
    }

    fn grads(&self) -> HashMap<String, Vec<f32>> {
        self.projector.cfg.param_list().into_iter().map(|(n, len)| (n.clone(), self.gpu.read(self.projector.grad(&n), len))).collect()
    }

    fn set_weights(&mut self, weights: &HashMap<String, Vec<f32>>) {
        for (name, w) in weights {
            self.gpu.write_f32(self.projector.param(name), w);
            self.state.get_mut(name).expect("a parameter of the aligner").0 = w.clone();
        }
    }

    fn step(&mut self, t: u32, h: &Hyper) {
        let grads = self.grads();
        let sum_sq: f64 = grads.values().flatten().map(|g| (*g as f64).powi(2)).sum();
        let scale = grad_multiplier(sum_sq, (h.grad_clip > 0.0).then_some(h.grad_clip), 1.0);
        let adam = Adam::default();
        for (name, g) in &grads {
            let (w, m, v) = self.state.get_mut(name).expect("a parameter of the aligner");
            adam.update_slice(t, h.aligner_lr, h.weight_decay, scale, w, m, v, g);
            self.gpu.write_f32(self.projector.param(name), w);
        }
    }
}

/// A fine-tune of the composite: the decoder, the aligner, and the splice
/// between them.
pub struct Trainer {
    decoder: Qwen,
    aligner: Aligner,
    block: u32,
    rows: usize,
    splice_at: Option<u32>,
}

impl Trainer {
    /// Build the trainer for rows of at most `block` tokens, the image taking
    /// `rows` of them. `cfg` is the decoder's configuration: with `lora` set,
    /// the base (at `dt`, see [`Qwen::new_lora_dt`]) is frozen under fresh
    /// adapters drawn from `seed`; without, every decoder weight trains.
    /// `aligner_weights` are the aligner's starting point, named as
    /// [`ProjectorConfig::param_list`] does. `base` is read while the decoder
    /// is built and not after.
    pub fn new(
        cfg: QwenConfig,
        base: Box<dyn TensorSource + '_>,
        dt: Dtype,
        block: u32,
        aligner: ProjectorConfig,
        aligner_weights: HashMap<String, Vec<f32>>,
        rows: usize,
        seed: u64,
    ) -> Result<Trainer, String> {
        if aligner.out_dim != cfg.d_model {
            return Err(format!("the aligner makes {}-wide rows for a {}-wide decoder", aligner.out_dim, cfg.d_model));
        }
        let decoder = if cfg.lora.is_some() {
            let init = LoraInit::fresh(&cfg, base, seed);
            Qwen::new_lora_dt(cfg, 1, block, &init, dt)
        } else {
            let shard = qwen3::Shard::whole(cfg.n_layers as usize);
            Qwen::new_shard(cfg, 1, block, &*base, true, shard)
        };
        Ok(Trainer { decoder, aligner: Aligner::new(aligner, aligner_weights, rows)?, block, rows, splice_at: None })
    }

    /// The row-major shape of aligner parameter `name`: `[out, in]` for a
    /// weight, `[out]` for a bias.
    pub fn aligner_shape(&self, name: &str) -> Vec<u64> {
        let cfg = self.aligner.projector.cfg;
        let len = self.aligner.state[name].0.len() as u64;
        if name.ends_with(".bias") {
            return vec![len];
        }
        let inputs = if name.starts_with("in") { cfg.input_dim } else { cfg.n_embed } as u64;
        vec![len / inputs, inputs]
    }

    /// Which aligner this is, for the file that stores it.
    pub fn aligner_kind(&self) -> String {
        format!("{:?}", self.aligner.projector.cfg)
    }

    /// The decoder, for saving its adapter.
    pub fn decoder(&self) -> &Qwen {
        &self.decoder
    }

    /// The aligner's parameters as trained so far.
    pub fn aligner_weights(&self) -> HashMap<String, Vec<f32>> {
        self.aligner.state.iter().map(|(n, (w, _, _))| (n.clone(), w.clone())).collect()
    }

    /// Replace the aligner's parameters (all of them).
    pub fn set_aligner_weights(&mut self, weights: &HashMap<String, Vec<f32>>) {
        self.aligner.set_weights(weights);
    }

    /// Upload `p` to the decoder: the aligner's rows spliced in, the tokens
    /// padded to the block.
    fn load(&mut self, p: &Prepared) {
        assert!(p.tokens.len() <= self.block as usize, "an example of {} tokens does not fit the {}-token block", p.tokens.len(), self.block);
        if self.splice_at != Some(p.row0) {
            self.decoder.enable_mm_splice(p.row0, self.rows as u32);
            self.splice_at = Some(p.row0);
        }
        self.decoder.write_img_embeds(&self.aligner.forward(&p.streams));
        let mut x = p.tokens.clone();
        let mut y = p.targets.clone();
        x.resize(self.block as usize, 0);
        y.resize(self.block as usize, IGNORE);
        self.decoder.set_batch(&x, &y);
    }

    /// The loss on `p` with the current weights.
    pub fn loss(&mut self, p: &Prepared) -> f32 {
        self.load(p);
        self.decoder.forward()
    }

    /// The loss on `p` and the gradient of every aligner parameter.
    pub fn loss_and_aligner_grads(&mut self, p: &Prepared) -> (f32, HashMap<String, Vec<f32>>) {
        let loss = self.forward_backward(p);
        (loss, self.aligner.grads())
    }

    fn forward_backward(&mut self, p: &Prepared) -> f32 {
        self.load(p);
        self.decoder.zero_grads();
        self.aligner.projector.zero_grads(&self.aligner.gpu);
        let loss = self.decoder.forward();
        self.decoder.backward();
        self.aligner.backward(&self.decoder.read_d_img_embeds());
        loss
    }

    /// One optimiser step (1-based `t`) on `p`; returns its loss.
    pub fn step(&mut self, p: &Prepared, t: u32, h: &Hyper) -> f32 {
        let loss = self.forward_backward(p);
        self.aligner.step(t, h);
        self.decoder.adamw_step(t, h.lr, h.weight_decay, Adam::default(), (h.grad_clip > 0.0).then_some(h.grad_clip), 1.0);
        self.decoder.poll_wait();
        loss
    }
}

/// The settings of a whole fine-tune: the adapter, the schedule, the dtype
/// the frozen decoder is held at.
#[derive(Clone, Debug)]
pub struct Options {
    pub rank: u32,
    pub alpha: f32,
    /// The decoder projections the adapter covers (`wq`, `wk`, ...).
    pub targets: Vec<String>,
    pub steps: u32,
    pub lr: f32,
    /// The aligner's rate; `None` for a tenth of `lr`.
    pub aligner_lr: Option<f32>,
    pub seed: u64,
    /// The rows of a training example; `None` fits the longest one.
    pub block: Option<u32>,
    /// The storage dtype of the frozen decoder.
    pub dtype: Dtype,
    /// Decay, clip, warmup and the floor of the cosine schedule.
    pub hyper: qwen3::finetune::LoraHyper,
}

/// A step's progress.
#[derive(Clone, Copy, Debug)]
pub struct StepInfo {
    pub step: u32,
    pub steps: u32,
    pub loss: f32,
    pub lr: f32,
}

/// What a fine-tune leaves: the trainer holding the trained parameters.
pub struct Outcome {
    pub initial_loss: f32,
    pub final_loss: Option<f32>,
    pub examples: usize,
    pub block: u32,
    pub trainer: Trainer,
}

impl Outcome {
    /// Write the trained parameters into `dir`: the decoder's adapter as
    /// `adapter.safetensors` (never the frozen base) and the aligner as
    /// `aligner.safetensors`.
    pub fn save(&self, dir: &Path, card_id: &str, base_id: &str) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
        qwen3::lora::save_adapter(&path(ADAPTER_FILE), self.trainer.decoder(), card_id, base_id, None).map_err(|e| format!("{}: {e}", path(ADAPTER_FILE)))?;
        let mut names: Vec<String> = self.trainer.aligner_weights().into_keys().collect();
        names.sort();
        let weights = self.trainer.aligner_weights();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = names.into_iter().map(|n| (n.clone(), self.trainer.aligner_shape(&n), weights[&n].clone())).collect();
        checkpoint::st::save_safetensors(&path(ALIGNER_FILE), &tensors, &serde_json::json!({ "aligner": self.trainer.aligner_kind() }), None).map_err(|e| format!("{}: {e}", path(ALIGNER_FILE)))
    }
}

/// The adapter and aligner files [`Outcome::save`] writes.
pub const ADAPTER_FILE: &str = "adapter.safetensors";
pub const ALIGNER_FILE: &str = "aligner.safetensors";

/// A dataset's examples: each line of `train.jsonl` holds an `image` (a
/// path, relative to `dir`) and `messages` (`role` and `content`), ending in
/// the assistant reply to learn.
pub fn read_dataset(dir: &Path) -> Result<Vec<(PathBuf, Vec<Turn>)>, String> {
    let file = dir.join("train.jsonl");
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let at = |why: String| format!("{}:{}: {why}", file.display(), n + 1);
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| at(e.to_string()))?;
        let image = v["image"].as_str().ok_or_else(|| at("no \"image\" path".to_string()))?;
        let turns = v["messages"]
            .as_array()
            .ok_or_else(|| at("no \"messages\"".to_string()))?
            .iter()
            .map(|m| {
                let role = match m["role"].as_str() {
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    other => return Err(at(format!("role {other:?} is not user or assistant"))),
                };
                let content = m["content"].as_str().ok_or_else(|| at("a message has no text \"content\"".to_string()))?.to_string();
                Ok(Turn { role, content })
            })
            .collect::<Result<Vec<_>, String>>()?;
        out.push((dir.join(image), turns));
    }
    if out.is_empty() {
        return Err(format!("{}: no examples", file.display()));
    }
    Ok(out)
}

/// Fine-tune the composite `frontend` opens over the decoder and aligner in
/// `rd` (the aligner under `aligner_prefix`): every example's image goes
/// through the frozen tower once, the tower is then released, and the decoder
/// (an adapter over the frozen base, or whole when `opts.rank` is 0) and the
/// aligner train on the cached features for `opts.steps` steps, the examples
/// taken in a seeded order.
pub fn run(frontend: Frontend, rd: &WeightReader, aligner_prefix: &str, examples: Vec<(Rgb8, Vec<Turn>)>, opts: &Options, progress: &mut dyn FnMut(&StepInfo)) -> Result<Outcome, String> {
    let prepared = examples.iter().map(|(image, turns)| frontend.prepare(image, turns)).collect::<Result<Vec<_>, String>>()?;
    let aligner_cfg = frontend.tower.aligner_config();
    let rows = frontend.splice.rows;
    let base_cfg = frontend.decoder_cfg.clone();
    drop(frontend);
    let mut decoder_cfg = base_cfg.clone();
    if opts.rank > 0 {
        decoder_cfg.lora = Some(qwen3::LoraCfg { rank: opts.rank, alpha: opts.alpha, targets: opts.targets.clone() });
    }

    let longest = prepared.iter().map(|p| p.tokens.len()).max().expect("a dataset has examples");
    let block = match opts.block {
        Some(b) if (b as usize) < longest => return Err(format!("--block {b} is shorter than the longest example ({longest} tokens); it would train on a cut-off reply")),
        Some(b) => b,
        None => longest as u32,
    };
    let base = qwen3::import::nested_source(rd, crate::import::DECODER, &base_cfg)?;
    let aligner_weights = crate::import::aligner_weights(rd, aligner_prefix, &aligner_cfg)?;
    let mut trainer = Trainer::new(decoder_cfg, Box::new(base), opts.dtype, block, aligner_cfg, aligner_weights, rows, opts.seed)?;

    let fit = opts.hyper.fit_opts(opts.steps, 1, block, opts.lr, opts.seed);
    let hyper_at = |step: u32| Hyper { lr: cosine_lr(step, &fit), aligner_lr: cosine_lr(step, &fit) * opts.aligner_lr.map_or(0.1, |a| a / opts.lr), weight_decay: fit.weight_decay, grad_clip: fit.grad_clip };
    let initial_loss = prepared.iter().map(|p| trainer.loss(p)).sum::<f32>() / prepared.len() as f32;
    let mut order = data::rng::Rng::new(opts.seed ^ 0x5eed);
    let mut queue: Vec<usize> = Vec::new();
    let mut final_loss = None;
    for step in 0..opts.steps {
        if queue.is_empty() {
            queue = (0..prepared.len()).collect();
            for i in (1..queue.len()).rev() {
                queue.swap(i, (order.next_u64() % (i as u64 + 1)) as usize);
            }
        }
        let h = hyper_at(step);
        let loss = trainer.step(&prepared[queue.pop().expect("refilled")], step + 1, &h);
        final_loss = Some(loss);
        progress(&StepInfo { step: step + 1, steps: opts.steps, loss, lr: h.lr });
    }
    Ok(Outcome { initial_loss, final_loss, examples: prepared.len(), block, trainer })
}

/// Fine-tune DeepSeek-VL from the checkpoint directory `dir` on the dataset
/// in `dataset` ([`read_dataset`]), on the ambient device: the hybrid tower
/// extracts every image's features, is released, and the decoder and aligner
/// train.
pub fn finetune(dir: &Path, dataset: &Path, opts: &Options, progress: &mut dyn FnMut(&StepInfo)) -> Result<Outcome, String> {
    let cfg = crate::DeepseekVlConfig::from_dir(dir)?;
    let rd = WeightReader::open_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    crate::import::check_coverage(&rd, &[crate::import::SAM_PREFIX, clip::import::siglip::DEEPSEEK_VL_LOW_PREFIX, crate::import::ALIGNER_PREFIX, "language_model."])?;
    let examples = load_examples(dataset)?;
    let tower = Box::new(crate::tower::HybridTower::load(&rd, &cfg)?);
    let frontend = Frontend::open(dir, crate::model::Parts { tower, style: crate::prompt::DEEPSEEK_VL, wrap: None, language: cfg.language })?;
    run(frontend, &rd, crate::import::ALIGNER_PREFIX, examples, opts, progress)
}

/// [`read_dataset`] with every image decoded.
pub fn load_examples(dataset: &Path) -> Result<Vec<(Rgb8, Vec<Turn>)>, String> {
    read_dataset(dataset)?
        .into_iter()
        .map(|(path, turns)| {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok((imaging::codec::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))?, turns))
        })
        .collect()
}
