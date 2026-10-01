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
//! A step runs the aligner over each of an example's images' cached feature
//! streams, splices their rows into the decoder's residual stream
//! ([`qwen3::Qwen::enable_mm_splices`]), differentiates the reply's
//! cross-entropy, and carries the gradient of the spliced rows back through
//! the aligner, image by image. The decoder's optimiser is its own
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
use qwen3::finetune::Trained;
use qwen3::{Dtype, QwenConfig, IGNORE};

use crate::model::Frontend;
use crate::prompt::{Role, Turn};

/// One training example ready for the decoder: each image's frozen-tower
/// feature streams and the token rows of the conversation with the images'
/// rows marked in place.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// Per image, `[rows, width]` per feature stream, as
    /// [`crate::tower::Features::streams`].
    pub images: Vec<Vec<Vec<f32>>>,
    /// The decoder's input tokens; the images' rows are placeholders the
    /// splice overwrites.
    pub tokens: Vec<u32>,
    /// `targets[i]` is what follows `tokens[i]`, or [`IGNORE`]: only the
    /// reply (and its end-of-sentence) is supervised.
    pub targets: Vec<u32>,
    /// The first residual row of each image's rows, in image order.
    pub row0s: Vec<u32>,
}

impl Frontend {
    /// Turn the images and a conversation ending in the reply to learn into a
    /// training example. The prompt is rendered as it is at inference (BOS,
    /// system prompt, the user's turn with its `<image_placeholder>`s, the
    /// open `Assistant:`), and the reply follows it closed by the
    /// end-of-sentence token; the prompt places each image once, in order.
    pub fn prepare(&self, images: &[Rgb8], turns: &[Turn]) -> Result<Prepared, String> {
        let (reply, asked) = match turns.split_last() {
            Some((last, rest)) if last.role == Role::Assistant && !rest.is_empty() => (last, rest),
            _ => return Err("a training conversation ends with the assistant reply to learn, after a user turn".to_string()),
        };
        let prompt = self.prompt_ids(asked)?;
        let placeholders = prompt.iter().filter(|&&t| t == self.splice.image_id).count();
        if images.is_empty() || placeholders != images.len() {
            return Err(format!("a training example with {} image(s) needs one image placeholder per image, and its prompt has {placeholders}", images.len()));
        }
        let reply_ids = self.tokenizer.encode(&format!(" {}{}", reply.content.trim(), self.eos));
        if reply_ids.last() != Some(&self.eos_id) {
            return Err("the reply does not end in the end-of-sentence token".to_string());
        }
        let mut sequence = self.splice.expand_ids(&prompt);
        let row0s: Vec<u32> = self.splice.image_rows(&sequence).into_iter().map(|r| r as u32).collect();
        let supervised_from = sequence.len();
        sequence.extend(&reply_ids);

        let n = sequence.len() - 1;
        let tokens = sequence[..n].to_vec();
        let targets = (0..n).map(|i| if i + 1 >= supervised_from { sequence[i + 1] } else { IGNORE }).collect();
        let images = images.iter().map(|image| Ok(self.tower.encode(&self.processor.pixel_values(image)?).streams)).collect::<Result<Vec<_>, String>>()?;
        Ok(Prepared { images, tokens, targets, row0s })
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

/// A projector (an aligner, a generation head) on the device with its
/// host-side AdamW state: what a trainable `model::projector::MlpProjector`
/// needs around it.
pub struct TrainableProjector {
    gpu: Gpu,
    projector: MlpProjector,
    inputs: Vec<DeviceBuffer>,
    d_out: DeviceBuffer,
    d_inputs: Vec<DeviceBuffer>,
    /// `(master weights, m, v)` by parameter name.
    state: HashMap<String, (Vec<f32>, Vec<f32>, Vec<f32>)>,
    rows: usize,
}

impl TrainableProjector {
    /// `weights` (every [`ProjectorConfig::param_list`] name) for `rows`
    /// rows of feature streams.
    pub fn new(cfg: ProjectorConfig, weights: HashMap<String, Vec<f32>>, rows: usize) -> Result<TrainableProjector, String> {
        let gpu = Gpu::new(PROJECTOR_PIPELINES);
        let projector = MlpProjector::new(&gpu, cfg, rows as u32, &weights)?;
        let stream = |_| gpu.storage(rows as u64 * cfg.input_dim as u64);
        let inputs = (0..cfg.inputs()).map(stream).collect();
        let d_inputs = (0..cfg.inputs()).map(stream).collect();
        let d_out = gpu.storage(rows as u64 * cfg.out_dim as u64);
        let state = weights.into_iter().map(|(n, w)| (n, (w.clone(), vec![0.0; w.len()], vec![0.0; w.len()]))).collect();
        Ok(TrainableProjector { gpu, projector, inputs, d_out, d_inputs, state, rows })
    }

    pub fn cfg(&self) -> ProjectorConfig {
        self.projector.cfg
    }

    /// The projector's `[rows, out_dim]` output for `streams`
    /// (`[rows, input_dim]` each).
    pub fn forward(&self, streams: &[Vec<f32>]) -> Vec<f32> {
        assert_eq!(streams.len(), self.inputs.len(), "the projector reads {} feature stream(s)", self.inputs.len());
        for (buf, s) in self.inputs.iter().zip(streams) {
            self.gpu.write_f32(buf, s);
        }
        let refs: Vec<&DeviceBuffer> = self.inputs.iter().collect();
        self.gpu.submit(&[], &self.projector.forward(&self.gpu, &refs));
        self.gpu.read(self.projector.out(), self.rows * self.projector.cfg.out_dim as usize)
    }

    pub fn zero_grads(&self) {
        self.projector.zero_grads(&self.gpu);
    }

    /// Accumulate the parameter gradients for `d_rows`, the loss's gradient
    /// at the output (after a [`Self::forward`] on the same streams), and
    /// return the gradient at each input stream.
    pub fn backward(&self, d_rows: &[f32]) -> Vec<Vec<f32>> {
        self.gpu.write_f32(&self.d_out, d_rows);
        let refs: Vec<&DeviceBuffer> = self.inputs.iter().collect();
        let d_refs: Vec<&DeviceBuffer> = self.d_inputs.iter().collect();
        self.gpu.submit(&[], &self.projector.backward(&self.gpu, &refs, &self.d_out, Some(&d_refs)));
        let n = self.rows * self.projector.cfg.input_dim as usize;
        self.d_inputs.iter().map(|b| self.gpu.read(b, n)).collect()
    }

    /// The accumulated gradient of every parameter.
    pub fn grads(&self) -> HashMap<String, Vec<f32>> {
        self.projector.cfg.param_list().into_iter().map(|(n, len)| (n.clone(), self.gpu.read(self.projector.grad(&n), len))).collect()
    }

    /// The parameters as trained so far.
    pub fn weights(&self) -> HashMap<String, Vec<f32>> {
        self.state.iter().map(|(n, (w, _, _))| (n.clone(), w.clone())).collect()
    }

    /// Replace the parameters (all of them).
    pub fn set_weights(&mut self, weights: &HashMap<String, Vec<f32>>) {
        for (name, w) in weights {
            self.gpu.write_f32(self.projector.param(name), w);
            self.state.get_mut(name).expect("a parameter of the projector").0 = w.clone();
        }
    }

    /// The row-major shape of parameter `name`: `[out, in]` for a weight,
    /// `[out]` for a bias.
    pub fn shape(&self, name: &str) -> Vec<u64> {
        let cfg = self.projector.cfg;
        let len = self.state[name].0.len() as u64;
        if name.ends_with(".bias") {
            return vec![len];
        }
        let inputs = if name.starts_with("in") { cfg.input_dim } else { cfg.n_embed } as u64;
        vec![len / inputs, inputs]
    }

    /// One AdamW step (1-based `t`) on the accumulated gradients, clipped to
    /// `grad_clip` in global norm when it is positive.
    pub fn step(&mut self, t: u32, lr: f32, weight_decay: f32, grad_clip: f32) {
        self.step_scaled(t, lr, weight_decay, grad_clip, 1.0);
    }

    /// [`Self::step`] on the accumulated gradients multiplied by `mean`
    /// (`1/K` after `K` examples), the clip applied to the scaled norm.
    pub fn step_scaled(&mut self, t: u32, lr: f32, weight_decay: f32, grad_clip: f32, mean: f32) {
        let grads = self.grads();
        let sum_sq: f64 = grads.values().flatten().map(|g| (*g as f64).powi(2)).sum();
        let scale = grad_multiplier(sum_sq, (grad_clip > 0.0).then_some(grad_clip), mean);
        let adam = Adam::default();
        for (name, g) in &grads {
            let (w, m, v) = self.state.get_mut(name).expect("a parameter of the projector");
            adam.update_slice(t, lr, weight_decay, scale, w, m, v, g);
            self.gpu.write_f32(self.projector.param(name), w);
        }
    }
}

/// A fine-tune of the composite: the decoder, the aligner, and the splice
/// between them.
pub struct Trainer {
    decoder: Trained,
    aligner: TrainableProjector,
    block: u32,
    /// Aligner rows per image.
    rows: usize,
    splice_at: Vec<u32>,
}

impl Trainer {
    /// Build the trainer for rows of at most `block` tokens, each image taking
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
        let decoder = qwen3::finetune::build_decoder_trainer(cfg, base, dt, block, seed).map_err(|e| e.to_string())?;
        Trainer::with_decoder(decoder, block, aligner, aligner_weights, rows)
    }

    /// The trainer around a decoder already built (on one card or as a
    /// pipeline), for `block`-token rows of which each image takes `rows`.
    pub fn with_decoder(decoder: Trained, block: u32, aligner: ProjectorConfig, aligner_weights: HashMap<String, Vec<f32>>, rows: usize) -> Result<Trainer, String> {
        Ok(Trainer { decoder, aligner: TrainableProjector::new(aligner, aligner_weights, rows)?, block, rows, splice_at: Vec::new() })
    }

    /// The row-major shape of aligner parameter `name`.
    pub fn aligner_shape(&self, name: &str) -> Vec<u64> {
        self.aligner.shape(name)
    }

    /// Which aligner this is, for the file that stores it.
    pub fn aligner_kind(&self) -> String {
        format!("{:?}", self.aligner.cfg())
    }

    /// The decoder, for saving its adapter.
    pub fn decoder(&self) -> &Trained {
        &self.decoder
    }

    /// The aligner's parameters as trained so far.
    pub fn aligner_weights(&self) -> HashMap<String, Vec<f32>> {
        self.aligner.weights()
    }

    /// Replace the aligner's parameters (all of them).
    pub fn set_aligner_weights(&mut self, weights: &HashMap<String, Vec<f32>>) {
        self.aligner.set_weights(weights);
    }

    /// Upload `p` to the decoder: the aligner's rows for every image spliced
    /// in, the tokens padded to the block.
    fn load(&mut self, p: &Prepared) {
        assert!(p.tokens.len() <= self.block as usize, "an example of {} tokens does not fit the {}-token block", p.tokens.len(), self.block);
        if self.splice_at != p.row0s {
            let regions: Vec<(u32, u32)> = p.row0s.iter().map(|&r| (r, self.rows as u32)).collect();
            self.decoder.enable_mm_splices(&regions);
            self.splice_at = p.row0s.clone();
        }
        let embeds: Vec<f32> = p.images.iter().flat_map(|streams| self.aligner.forward(streams)).collect();
        self.decoder.write_img_embeds(&embeds);
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
        self.decoder.zero_grads();
        self.aligner.zero_grads();
        self.accumulate(p)
    }

    /// The loss on `p`, its gradients added to those already accumulated.
    fn accumulate(&mut self, p: &Prepared) -> f32 {
        self.load(p);
        let loss = self.decoder.forward();
        self.decoder.backward();
        // The aligner's activations hold the last image's: run each image's
        // forward again before its gradient, the parameter gradients adding up.
        let d_images = self.decoder.read_d_img_embeds();
        let per_image = self.rows * self.aligner.cfg().out_dim as usize;
        for (streams, d_rows) in p.images.iter().zip(d_images.chunks(per_image)) {
            self.aligner.forward(streams);
            self.aligner.backward(d_rows);
        }
        loss
    }

    /// One optimiser step (1-based `t`) on `p`; returns its loss.
    pub fn step(&mut self, p: &Prepared, t: u32, h: &Hyper) -> f32 {
        self.step_batch(&[p], t, h)
    }

    /// One optimiser step (1-based `t`) on the mean gradient of `batch`;
    /// returns the mean loss.
    pub fn step_batch(&mut self, batch: &[&Prepared], t: u32, h: &Hyper) -> f32 {
        assert!(!batch.is_empty(), "a step needs at least one example");
        self.decoder.zero_grads();
        self.aligner.zero_grads();
        let mean = 1.0 / batch.len() as f32;
        let loss = batch.iter().map(|p| self.accumulate(p)).sum::<f32>() * mean;
        self.aligner.step_scaled(t, h.aligner_lr, h.weight_decay, h.grad_clip, mean);
        self.decoder.adamw_step(t, h.lr, h.weight_decay, Adam::default(), (h.grad_clip > 0.0).then_some(h.grad_clip), mean);
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
    /// Examples whose gradients are averaged into each step.
    pub batch: u32,
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
        self.trainer.decoder().save_adapter(&path(ADAPTER_FILE), card_id, base_id, None).map_err(|e| format!("{}: {e}", path(ADAPTER_FILE)))?;
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
/// path, relative to `dir`) or `images` (a list of them, in the order the
/// messages place them) and `messages` (`role` and `content`), ending in
/// the assistant reply to learn.
pub fn read_dataset(dir: &Path) -> Result<Vec<(Vec<PathBuf>, Vec<Turn>)>, String> {
    let file = dir.join("train.jsonl");
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let at = |why: String| format!("{}:{}: {why}", file.display(), n + 1);
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| at(e.to_string()))?;
        let images: Vec<&str> = match (v["image"].as_str(), v["images"].as_array()) {
            (Some(one), None) => vec![one],
            (None, Some(many)) if !many.is_empty() => many.iter().map(|p| p.as_str().ok_or_else(|| at("\"images\" holds a path that is not a string".to_string()))).collect::<Result<_, _>>()?,
            _ => return Err(at("an example has an \"image\" path or a non-empty \"images\" list of paths".to_string())),
        };
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
        out.push((images.into_iter().map(|p| dir.join(p)).collect(), turns));
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
pub fn run(frontend: Frontend, rd: &WeightReader, aligner_prefix: &str, examples: Vec<(Vec<Rgb8>, Vec<Turn>)>, opts: &Options, progress: &mut dyn FnMut(&StepInfo)) -> Result<Outcome, String> {
    let prepared = examples.iter().map(|(images, turns)| frontend.prepare(images, turns)).collect::<Result<Vec<_>, String>>()?;
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
        let h = hyper_at(step);
        let mut batch = Vec::with_capacity(opts.batch as usize);
        for _ in 0..opts.batch.max(1) {
            if queue.is_empty() {
                queue = (0..prepared.len()).collect();
                for i in (1..queue.len()).rev() {
                    queue.swap(i, (order.next_u64() % (i as u64 + 1)) as usize);
                }
            }
            batch.push(&prepared[queue.pop().expect("refilled")]);
        }
        let loss = trainer.step_batch(&batch, step + 1, &h);
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
pub fn load_examples(dataset: &Path) -> Result<Vec<(Vec<Rgb8>, Vec<Turn>)>, String> {
    read_dataset(dataset)?
        .into_iter()
        .map(|(paths, turns)| {
            let images = paths
                .iter()
                .map(|path| {
                    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
                    imaging::codec::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok((images, turns))
        })
        .collect()
}
