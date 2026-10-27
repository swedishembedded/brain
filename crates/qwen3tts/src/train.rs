// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of speech synthesizers from feedback
// on how they sound, for its clients. If your team needs expertise in
// reinforcement learning for voice models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The Talker as a policy: the input rows a rollout was generated from.
//!
//! The Talker is not a token model. At every step it reads one embedding row
//! (for a frame already generated: the first codebook's embedding, plus the
//! sum of the residual codebooks' embeddings, plus the next row of the text
//! stream or the pad row) and predicts the first codebook of the next frame. To
//! train it on a rollout, or to score the rollout under another set of weights,
//! those rows have to be rebuilt exactly as generation fed them.
//! [`policy_rows`] does that from the codes alone, and its test is that stepping
//! the rows through the generation path reproduces the rollout.
//!
//! The policy is the first codebook only: the residual codebooks are sampled
//! from the frozen code predictor given it, and are treated as part of the
//! environment.

use crate::gen::TalkerGen;
use crate::mtp::MtpModel;
use crate::prompt::Prompt;

/// The training rows of one rollout and what each position should predict.
#[derive(Clone, Debug)]
pub struct PolicyRows {
    /// Input rows, `[len, d]` row-major: the prompt, then one row per frame.
    pub rows: Vec<f32>,
    /// The width of a row.
    pub d: usize,
    /// For each row, the first-codebook token it should predict, or
    /// [`crate::sft::IGNORE`] where nothing was sampled.
    pub targets: Vec<u32>,
}

impl PolicyRows {
    /// How many rows there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// The positions that have a target.
    pub fn target_positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.targets.iter().enumerate().filter(|(_, t)| **t != crate::sft::IGNORE).map(|(p, _)| p)
    }
}

/// The rows a rollout of `codes` (`[frames, num_q]` row-major) was fed, and the
/// first-codebook token each predicted. `eos` is the end-of-speech token when
/// generation ended on it (the last row then predicts it), `None` when it ended
/// at the length cap.
///
/// The rows mirror `pipeline::generate_codes`: after the prompt, frame `t`'s row
/// is `codec_embed(cb0_t) + sum_k codec_embedding[k-1][cb_k,t]` plus the `t`-th
/// trailing text row, or the pad row once those run out.
#[must_use]
pub fn policy_rows(gen: &TalkerGen, mtp: &MtpModel, prompt: &Prompt, codes: &[u32], num_q: usize, eos: Option<u32>) -> PolicyRows {
    let d = gen.d();
    assert_eq!(codes.len() % num_q, 0, "codes are not whole frames");
    let frames = codes.len() / num_q;
    let prefix = prompt.embeds.len() / d;
    let n_trailing = prompt.trailing.len() / d;

    let mut rows = prompt.embeds.clone();
    rows.reserve(frames * d);
    // One target per row: row `prefix - 1 + t` predicts frame `t`'s first codebook.
    let mut targets = vec![crate::sft::IGNORE; prefix + frames];
    for t in 0..frames {
        let frame = &codes[t * num_q..(t + 1) * num_q];
        // The row that predicts this frame's first codebook is the one before it.
        targets[prefix - 1 + t] = frame[0];
        let mut row = gen.codec_embed(frame[0]).to_vec();
        for (k, code) in frame.iter().enumerate().skip(1) {
            for (a, b) in row.iter_mut().zip(mtp.codec_embed(k - 1, *code)) {
                *a += b;
            }
        }
        let text = if t < n_trailing { &prompt.trailing[t * d..(t + 1) * d] } else { &prompt.tts_pad[..] };
        for (a, b) in row.iter_mut().zip(text) {
            *a += b;
        }
        rows.extend_from_slice(&row);
    }
    // The last row (after the final frame) predicts the end of speech when that
    // is what ended the clip; at the length cap it predicts nothing recorded.
    if let (Some(eos), true) = (eos, frames > 0) {
        targets[prefix + frames - 1] = eos;
    }
    PolicyRows { rows, d, targets }
}

// ---------------------------------------------------------------------------
// The trainer
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// One rollout as a training example: its rows, the advantage it earned in its
/// group, and what the frozen reference policy gave each sampled token.
pub struct Sample<'a> {
    /// The rows of the rollout.
    pub rows: &'a PolicyRows,
    /// How far above its group's mean the rollout's reward stood, in units of the
    /// group's spread (see the speech reward in the learning system).
    pub advantage: f32,
    /// The reference policy's log-probability of each sampled token, one per
    /// target position in order.
    pub reference_logprobs: Vec<f32>,
}

/// Knobs of one policy-gradient update.
#[derive(Clone, Copy, Debug)]
pub struct UpdateOpts {
    /// Learning rate.
    pub lr: f32,
    /// Weight of the k3 KL estimate to the reference policy.
    pub kl_beta: f32,
    /// Global gradient-norm clip.
    pub clip: Option<f32>,
    /// Weight decay.
    pub weight_decay: f32,
}

impl Default for UpdateOpts {
    fn default() -> Self {
        Self { lr: 1e-4, kl_beta: 0.02, clip: Some(1.0), weight_decay: 0.0 }
    }
}

/// What one update saw.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateStats {
    /// The objective: advantage-weighted negative log-likelihood plus the KL term.
    pub loss: f32,
    /// Mean log-probability of the sampled tokens under the policy before the step.
    pub mean_logprob: f32,
    /// Mean k3 KL estimate to the reference before the step.
    pub mean_kl: f32,
    /// How many sampled tokens the update was taken over.
    pub tokens: usize,
}

/// The Talker's trainable form: a LoRA over its attention projections, run on
/// the rows of a rollout, with the codec head on the host so the policy
/// gradient (a per-token weight on the log-likelihood) is applied exactly.
pub struct TalkerTrainer {
    model: qwen3::Qwen,
    d: usize,
    block: usize,
    vocab: usize,
    /// `lm_head.weight`, `[vocab, d]`.
    head: Vec<f32>,
    /// The same transposed, `[d, vocab]`, for the backward product.
    head_t: Vec<f32>,
    /// The base weights of every tensor the LoRA adapts, for folding.
    base: HashMap<String, Vec<f32>>,
    rank: usize,
    lora_scale: f32,
    updates: u32,
}

impl TalkerTrainer {
    /// Build the trainer for the Talker checkpoint at `talker`, with room for
    /// rollouts of up to `block` rows.
    pub fn load(talker: &str, block: u32, rank: u32, alpha: f32, seed: u64) -> Result<Self, String> {
        use qwen3::{LoraCfg, QwenConfig};
        let ckpt = crate::import::load_talker(talker);
        let mut cfg = QwenConfig::from_json_checked(&ckpt.header["config"])?;
        cfg.block_size = block;
        cfg.lora = Some(LoraCfg::attn(rank, alpha));
        let mut init = qwen3::init_weights(&cfg, seed);
        let base_weights = ckpt.by_role("");
        for (k, v) in base_weights {
            init.insert(k, v);
        }
        let d = cfg.d_model as usize;
        let vocab = cfg.vocab as usize;
        let head = init.get("lm_head.weight").cloned().ok_or("the Talker checkpoint has no lm_head.weight")?;
        if head.len() != vocab * d {
            return Err(format!("lm_head.weight holds {} values, expected {vocab} x {d}", head.len()));
        }
        let mut head_t = vec![0.0f32; head.len()];
        for o in 0..vocab {
            for i in 0..d {
                head_t[i * vocab + o] = head[o * d + i];
            }
        }
        let base: HashMap<String, Vec<f32>> = init.iter().filter(|(k, _)| k.contains(".attn.w")).map(|(k, v)| (k.clone(), v.clone())).collect();
        let mut model = qwen3::Qwen::new(cfg, 1, block, &init);
        model.enable_mm_splices(&[(0, block)]);
        model.enable_external_head();
        Ok(Self { model, d, block: block as usize, vocab, head, head_t, base, rank: rank as usize, lora_scale: alpha / rank as f32, updates: 0 })
    }

    /// The final-norm hidden states of `rows` (`[len, d]`), run through the
    /// current weights.
    pub fn hidden(&self, rows: &[f32]) -> Vec<f32> {
        let len = rows.len() / self.d;
        assert!(len <= self.block, "{len} rows exceed the trainer's block of {}", self.block);
        let mut padded = rows.to_vec();
        padded.resize(self.block * self.d, 0.0);
        self.model.set_batch(&vec![0u32; self.block], &vec![crate::sft::IGNORE; self.block]);
        self.model.write_img_embeds(&padded);
        let mut hidden = self.model.forward_hidden();
        hidden.truncate(len * self.d);
        hidden
    }

    /// The first-codebook logits at the target positions of `rows`,
    /// `[targets, vocab]`.
    pub fn logits(&self, rows: &PolicyRows) -> Vec<f32> {
        let hidden = self.hidden(&rows.rows);
        let selected = gather(&hidden, self.d, rows.target_positions());
        model::hostmath::linear_rows(&selected, &self.head, selected.len() / self.d, self.d, self.vocab)
    }

    /// One policy-gradient step over `samples` (a group-relative update when
    /// the advantages are): minimise `-A * log pi(token) + beta * k3(KL)`,
    /// averaged over every sampled token. The reference log-probabilities are the
    /// caller's, taken from the frozen policy.
    pub fn update(&mut self, samples: &[Sample<'_>], opts: &UpdateOpts) -> UpdateStats {
        let tokens: usize = samples.iter().map(|s| s.rows.target_positions().count()).sum();
        assert!(tokens > 0, "an update needs at least one sampled token");
        self.model.zero_grads();
        let mut stats = UpdateStats { tokens, ..UpdateStats::default() };
        for sample in samples {
            let positions: Vec<usize> = sample.rows.target_positions().collect();
            assert_eq!(positions.len(), sample.reference_logprobs.len(), "one reference log-probability per sampled token");
            let hidden = self.hidden(&sample.rows.rows);
            let selected = gather(&hidden, self.d, positions.iter().copied());
            let logits = model::hostmath::linear_rows(&selected, &self.head, positions.len(), self.d, self.vocab);
            let mut dlogits = vec![0.0f32; logits.len()];
            for (i, &position) in positions.iter().enumerate() {
                let row = &logits[i * self.vocab..(i + 1) * self.vocab];
                let target = sample.rows.targets[position] as usize;
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
                let logprob = row[target] - max - sum.ln();
                let u = sample.reference_logprobs[i] - logprob;
                let k3 = u.exp() - u - 1.0;
                // dL/dlogprob for L = -A * logprob + beta * k3, per token and per sampled token averaged.
                let coefficient = (-sample.advantage + opts.kl_beta * (1.0 - u.exp())) / tokens as f32;
                stats.loss += (-sample.advantage * logprob + opts.kl_beta * k3) / tokens as f32;
                stats.mean_logprob += logprob / tokens as f32;
                stats.mean_kl += k3 / tokens as f32;
                let out = &mut dlogits[i * self.vocab..(i + 1) * self.vocab];
                for (o, x) in out.iter_mut().zip(row) {
                    *o = -coefficient * ((x - max).exp() / sum);
                }
                out[target] += coefficient;
            }
            // d(loss)/d(hidden) at the sampled positions, scattered into the block.
            let dselected = model::hostmath::linear_rows(&dlogits, &self.head_t, positions.len(), self.vocab, self.d);
            let mut dhidden = vec![0.0f32; self.block * self.d];
            for (i, &position) in positions.iter().enumerate() {
                dhidden[position * self.d..(position + 1) * self.d].copy_from_slice(&dselected[i * self.d..(i + 1) * self.d]);
            }
            self.model.backward_hidden(&dhidden);
        }
        self.updates += 1;
        self.model.adamw_step(self.updates, opts.lr, opts.weight_decay, model::Adam::default(), opts.clip, 1.0);
        self.model.poll_wait();
        stats
    }

    /// The update's backward always rebuilds each layer from its input boundary: recomputing is accepted as what
    /// it already does, keeping every layer and a narrow saved format are refused by name.
    pub fn set_saved_policy(&self, policy: model::saved::SavedPolicy) -> Result<(), String> {
        model::saved::recompute_only("qwen3tts talker", policy)
    }

    /// The most rows one example can have.
    #[must_use]
    pub fn block(&self) -> usize {
        self.block
    }

    /// Write the trained adapter (`.lora_a`/`.lora_b` tensors, rank and alpha) to `path`
    /// as a safetensors file; [`fold_saved_adapter`] reads it back.
    pub fn save_adapter(&self, path: &std::path::Path) -> Result<(), String> {
        let names: Vec<String> = self.param_names();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = names.iter().map(|n| (n.clone(), vec![self.model.read_weight(n).len() as u64], self.model.read_weight(n))).collect();
        let meta = serde_json::json!({ "rank": self.rank, "alpha": self.lora_scale * self.rank as f32 });
        checkpoint::st::save_safetensors(path.to_str().ok_or("adapter path is not UTF-8")?, &tensors, &meta, None).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The names of the trained parameters.
    pub fn param_names(&self) -> Vec<String> {
        model::Model::param_names(&self.model)
    }

    /// The adapted attention weights with the LoRA folded in:
    /// `W + (alpha / rank) * B A` for every adapted tensor, by the name the
    /// inference Talker knows it under.
    pub fn folded_weights(&self) -> HashMap<String, Vec<f32>> {
        let names: std::collections::HashSet<String> = model::Model::param_names(&self.model).into_iter().collect();
        let mut folded = HashMap::new();
        for (name, weight) in &self.base {
            let (a_name, b_name) = (format!("{name}.lora_a"), format!("{name}.lora_b"));
            if !names.contains(&a_name) {
                continue;
            }
            let a = self.model.read_weight(&a_name);
            let b = self.model.read_weight(&b_name);
            folded.insert(name.clone(), fold_one(weight, &a, &b, self.rank, self.lora_scale));
        }
        folded
    }
}

/// The log-probability of each sampled token given the logits at the target
/// positions (`[targets, vocab]`, as [`TalkerTrainer::logits`] and
/// `ResidentEngine::score` return them).
#[must_use]
pub fn logprobs(logits: &[f32], rows: &PolicyRows) -> Vec<f32> {
    let tokens: Vec<u32> = rows.target_positions().map(|p| rows.targets[p]).collect();
    let vocab = logits.len() / tokens.len().max(1);
    tokens
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            row[t as usize] - max - row.iter().map(|x| (x - max).exp()).sum::<f32>().ln()
        })
        .collect()
}

/// The Talker's adapted attention tensors with the adapter saved at `path`
/// folded in, by name: what [`TalkerTrainer::folded_weights`] gave when it was
/// saved. `talker` is the base Talker checkpoint.
pub fn fold_saved_adapter(talker: &str, path: &std::path::Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let adapter = checkpoint::st::load_safetensors(path.to_str().ok_or("adapter path is not UTF-8")?).map_err(|e| format!("{}: {e}", path.display()))?;
    let meta = adapter.config();
    let rank = meta["rank"].as_u64().ok_or("the adapter records no rank")? as usize;
    let alpha = meta["alpha"].as_f64().ok_or("the adapter records no alpha")? as f32;
    let base = crate::import::load_talker(talker).by_role("");
    let mut folded = HashMap::new();
    for (name, a) in &adapter.tensors {
        let Some(weight_name) = name.strip_suffix(".lora_a") else { continue };
        let b = adapter.tensors.get(&format!("{weight_name}.lora_b")).ok_or_else(|| format!("{weight_name} has an A matrix and no B"))?;
        let weight = base.get(weight_name).ok_or_else(|| format!("the Talker has no tensor {weight_name}"))?;
        folded.insert(weight_name.to_string(), fold_one(weight, a, b, rank, alpha / rank as f32));
    }
    Ok(folded)
}

/// Rows `positions` of the `[*, d]` matrix `hidden`, concatenated.
fn gather(hidden: &[f32], d: usize, positions: impl Iterator<Item = usize>) -> Vec<f32> {
    let mut out = Vec::new();
    for p in positions {
        out.extend_from_slice(&hidden[p * d..(p + 1) * d]);
    }
    out
}

/// `w + scale * b a` with `w: [out, in]`, `a: [rank, in]`, `b: [out, rank]`.
fn fold_one(w: &[f32], a: &[f32], b: &[f32], rank: usize, scale: f32) -> Vec<f32> {
    let inn = a.len() / rank;
    let out = b.len() / rank;
    assert_eq!(w.len(), out * inn, "the adapter's shape does not fit the weight it adapts");
    let mut folded = w.to_vec();
    for o in 0..out {
        for k in 0..rank {
            let bok = scale * b[o * rank + k];
            if bok == 0.0 {
                continue;
            }
            let arow = &a[k * inn..(k + 1) * inn];
            for (x, y) in folded[o * inn..(o + 1) * inn].iter_mut().zip(arow) {
                *x += bok * y;
            }
        }
    }
    folded
}
