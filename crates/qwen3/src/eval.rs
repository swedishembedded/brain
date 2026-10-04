// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Held-out chat-sample scoring: teacher-forced masked loss + token
//! accuracy over [`data::chat::ChatSample`], for a base checkpoint alone or
//! with a named LoRA adapter folded in. This is Gate B of the Definition of
//! Done's "a way to validate that model has learned ideas from the
//! dataset" -- unlike `crates/qwen3/tests/lora_learning_gate.rs` (Gate A,
//! synthetic, no checkpoint needed), this runs against a REAL base
//! checkpoint and a REAL bench-exported dataset, so it lives behind
//! `brain qwen eval` rather than as an always-on test.
//!
//! Loss/accuracy are computed ONLY over positions the sample itself marks
//! trainable (`ChatSample::encode`'s mask) -- prompt/context tokens the
//! model was never asked to predict never count, matching exactly what
//! `qwen3::finetune::finetune` supervises during training.
//!
//! [`preference_margins`] is the preference-pair counterpart: how much more a
//! policy prefers each pair's chosen turn over its rejected one than a
//! reference does, over the same supervised positions.

use std::collections::HashMap;
use std::path::Path;

use data::chat::{ChatSample, RenderOpts};
use data::chat_template::ChatTemplate;
use data::preference::EncodedTurn;
use data::qwen_tokenizer::QwenBpe;

use crate::config::QwenConfig;
use crate::model::Qwen;
use gpu_core::select::Dtype;

/// Aggregate score over a held-out set: `loss` is mean per-token
/// cross-entropy (NaN if every sample was skipped), `token_accuracy` is the
/// fraction of trainable positions where greedy argmax matched the true
/// next token (`None` when no position was scored - unmeasured, not 0), `samples`/`skipped` account for every input sample so a
/// caller can tell "scored 0 out of 0" from "scored 0 out of 40".
#[derive(Debug, Clone, Copy)]
pub struct ChatScore {
    pub loss: f32,
    pub token_accuracy: Option<f64>,
    pub positions: usize,
    pub samples: usize,
    pub skipped: usize,
}

/// Build a servable [`Qwen`] from `weights`, optionally folding a LoRA
/// `adapter` (an adapter-only safetensors file, `qwen3::lora::save_adapter`'s
/// output) into the base tensors first -- the same zero-inference-overhead
/// path a resident uses to serve a named adapter, so scoring an adapter
/// exercises exactly what serving it would do. `dt` is the weight STORAGE
/// tier both builds use; an fp32 master copy is kept nowhere when `dt`
/// differs, so base and adapter scores must pass the SAME tier or the
/// comparison crosses tiers.
/// The base's configuration and every tensor under brain's names as host
/// f32, from whatever `weights` is on disk (a brain checkpoint, a GGUF or a
/// `transformers` directory).
fn base_tensors(weights: &str) -> (QwenConfig, HashMap<String, Vec<f32>>) {
    let (cfg, src) = crate::open_checkpoint(weights).unwrap_or_else(|e| panic!("{e}"));
    let tensors = crate::serve::Engine::tensors_from(&cfg, &*src).unwrap_or_else(|e| panic!("{weights}: {e}"));
    (cfg, tensors)
}

fn load_scored_model(weights: &str, adapter: Option<&str>, t: u32, dt: Dtype) -> Qwen {
    match adapter {
        None => {
            let (cfg, src) = crate::open_checkpoint(weights).unwrap_or_else(|e| panic!("{e}"));
            let shard = crate::Shard::whole(cfg.n_layers as usize);
            Qwen::new_shard_dt(cfg, 1, t, &*src, shard, dt)
        }
        Some(a) => {
            let (mut cfg, mut tensors) = base_tensors(weights);
            crate::lora::fold_adapter_into(&mut tensors, a).expect("fold adapter into base tensors");
            // Folded: the delta is already baked into the base tensors, so
            // this Qwen has no separate lora_a/lora_b params to build.
            cfg.lora = None;
            cfg.block_size = t;
            // Score on an INFERENCE build: the trainable constructor holds
            // grad + optimizer buffers on top of the fp32 master copy that
            // no scorer ever steps - about 7x the weights' device memory
            // for zero benefit, and an OOM on any card too small for a
            // training build.
            let shard = crate::Shard::whole(cfg.n_layers as usize);
            Qwen::new_shard_dt(cfg, 1, t, &tensors, shard, dt)
        }
    }
}

fn argmax(s: &[f32]) -> u32 {
    let mut bi = 0usize;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi as u32
}

/// Score `weights` (optionally with `adapter` folded in) against `samples`,
/// teacher-forced: for each sample's `(ids, mask)`, position `i` predicts
/// `ids[i+1]` from context `ids[..=i]`, counted only where `mask[i+1]` is
/// `true` -- the exact convention `data::loader::TokenDataset` uses during
/// training (`mask[start+1+t]` gates `y[t] = data[start+1+t]`), so this
/// scores exactly what training supervised, nothing else. A sample that
/// fails to encode (see `ChatSample::encode`'s prefix-stability doc) or
/// whose length exceeds `block` is skipped, not silently dropped from the
/// count -- see [`ChatScore::skipped`].
pub fn score_chat(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, samples: &[ChatSample], block: u32) -> ChatScore {
    score_chat_dt(weights, adapter, tok, tmpl, samples, block, Dtype::F32)
}

/// [`score_chat`] at an explicit weight STORAGE tier. Comparing a base
/// score against a tuned score is only meaningful at ONE tier: both builds
/// must land on the same tier or the numbers cross precisions. The tier the
/// linears actually landed on (a device without an f16 path demotes the
/// request) is reported, never silently swallowed.
pub fn score_chat_dt(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, samples: &[ChatSample], block: u32, dt: Dtype) -> ChatScore {
    score_chat_rendered(weights, adapter, tok, tmpl, samples, block, dt, RenderOpts::default())
}

/// [`score_chat_dt`] rendering the samples as `render` says: a run that trained
/// with [`RenderOpts::keep_reasoning`] is scored the way it trained.
#[allow(clippy::too_many_arguments)]
pub fn score_chat_rendered(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, samples: &[ChatSample], block: u32, dt: Dtype, render: RenderOpts) -> ChatScore {
    let model = load_scored_model(weights, adapter, block, dt);
    eprintln!(
        "eval: weight tier requested {dt:?}, linears landed {:?}",
        model.linear_dtype()
    );
    let vocab = model.cfg.vocab as usize;
    let cap = model.ctx_len();

    let mut total_nll = 0.0f64;
    let mut positions = 0usize;
    let mut correct = 0usize;
    let mut skipped = 0usize;

    for (i, s) in samples.iter().enumerate() {
        // A template-encoding failure is a data/config problem, not a
        // too-long sample -- surfaced loudly (the CLI half of this fix
        // already reports them), never silently folded into `skipped`.
        let (ids, mask) = match s.encode_with(tok, tmpl, render) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("eval: sample {i}: chat-template encode failed ({e}); skipping");
                skipped += 1;
                continue;
            }
        };
        if ids.len() < 2 || ids.len() > cap {
            skipped += 1;
            continue;
        }
        let logits = model.logits_all(&ids);
        for i in 0..ids.len() - 1 {
            if !mask[i + 1] {
                continue;
            }
            let target = ids[i + 1] as usize;
            let row = &logits[i * vocab..(i + 1) * vocab];
            let log_prob = model::logprobs::row_logprob(row, target);
            total_nll -= log_prob as f64;
            positions += 1;
            if argmax(row) as usize == target {
                correct += 1;
            }
        }
    }

    ChatScore {
        loss: if positions > 0 { (total_nll / positions as f64) as f32 } else { f32::NAN },
        token_accuracy: (positions > 0).then(|| correct as f64 / positions as f64),
        positions,
        samples: samples.len() - skipped,
        skipped,
    }
}

/// Per-pair preference margins of a policy against a reference over
/// `generic-preference-v1` pairs; see [`preference_margins`].
#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceMargins {
    /// `(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log
    /// ref(rejected))` per scored pair, in nats, each log-probability summed
    /// over that candidate's supervised tokens only.
    pub margins: Vec<f64>,
    /// Pairs not scored because a candidate does not fit the scoring row.
    pub skipped: usize,
}

/// Score preference pairs: each candidate's sequence log-probability (summed
/// over exactly the tokens its mask supervises, the positions DPO training
/// sums) under the policy - `weights` with `policy_adapter` folded in - and
/// under the reference - `weights` with `reference_adapter` folded in, or the
/// base alone - and each pair's margin between them. The two models are
/// built one after the other on the same inference path serving uses, never
/// both at once. A pair with a candidate longer than `block` is skipped and
/// counted.
pub fn preference_margins(weights: &str, reference_adapter: Option<&str>, policy_adapter: Option<&str>, pairs: &[(EncodedTurn, EncodedTurn)], block: u32) -> PreferenceMargins {
    let fits = |t: &EncodedTurn| t.ids.len() >= 2 && t.ids.len() <= block as usize;
    let scored: Vec<&(EncodedTurn, EncodedTurn)> = pairs.iter().filter(|(c, r)| fits(c) && fits(r)).collect();
    let sequence_logprobs = |adapter: Option<&str>| -> Vec<(f64, f64)> {
        let model = load_scored_model(weights, adapter, block, Dtype::F32);
        scored.iter().map(|(c, r)| (supervised_logprob(&model, c), supervised_logprob(&model, r))).collect()
    };
    let reference = sequence_logprobs(reference_adapter);
    let policy = sequence_logprobs(policy_adapter);
    let margins = policy.iter().zip(&reference).map(|((pc, pr), (rc, rr))| (pc - rc) - (pr - rr)).collect();
    PreferenceMargins { margins, skipped: pairs.len() - scored.len() }
}

/// `model`'s log-probability of the supervised tokens of one rendered
/// sequence, teacher-forced: position `i` predicts `ids[i + 1]`, counted
/// where `mask[i + 1]`.
fn supervised_logprob(model: &Qwen, turn: &EncodedTurn) -> f64 {
    let vocab = model.cfg.vocab as usize;
    let logits = model.logits_all(&turn.ids);
    (0..turn.ids.len() - 1)
        .filter(|&i| turn.mask[i + 1])
        .map(|i| model::logprobs::row_logprob(&logits[i * vocab..(i + 1) * vocab], turn.ids[i + 1] as usize) as f64)
        .sum()
}

/// Which KV representation [`score_chat_paged`] scores through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvMode {
    /// The uncalibrated legacy path's own representation — fp32 KV, no
    /// quantization. The gate: `score_chat_paged` at `Fp32` must match
    /// [`score_chat`]'s loss (same math, different engine), or the paged
    /// backend itself is wrong, independent of any int8 question.
    Fp32,
    /// int8 KV, online per-token absmax scales (today's serving default's
    /// candidate — see `crates/qwen3/src/serve.rs`'s `kv_int8`).
    Int8,
    /// int8 KV with a calibrated clip ceiling loaded from `kv_calib.json`
    /// beside `weights` (`model::kvcalib::KvCalib::from_model_dir`). Falls
    /// back to plain `Int8` (uncalibrated) with a printed warning if no
    /// matching calibration file is found — never a hard failure.
    Int8Calib,
}

impl KvMode {
    pub fn parse(s: &str) -> Result<KvMode, String> {
        match s {
            "fp32" => Ok(KvMode::Fp32),
            "int8" => Ok(KvMode::Int8),
            "int8-calib" => Ok(KvMode::Int8Calib),
            other => Err(format!("unknown --kv mode {other:?} (want fp32 | int8 | int8-calib)")),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            KvMode::Fp32 => "fp32",
            KvMode::Int8 => "int8",
            KvMode::Int8Calib => "int8-calib",
        }
    }
}

/// Load the raw decoder tensors for scoring, optionally folding a LoRA
/// adapter in first — the same `c.by_role("")` + `fold_adapter_into` shape
/// [`load_scored_model`]'s adapter branch already uses, shared here so the
/// legacy (`Qwen`) and paged (`Engine`) scoring backends build from
/// identical tensors.
fn load_scored_tensors(weights: &str, adapter: Option<&str>) -> (QwenConfig, HashMap<String, Vec<f32>>) {
    let (mut cfg, mut tensors) = base_tensors(weights);
    if let Some(a) = adapter {
        crate::lora::fold_adapter_into(&mut tensors, a).expect("fold adapter into base tensors");
        cfg.lora = None;
    }
    (cfg, tensors)
}

/// Like [`score_chat`], but scores through the paged serving engine
/// (`crate::serve::Engine`) at a chosen [`KvMode`] instead of the legacy
/// single-sequence `Qwen` — the actual engine `brain serve` runs, so this is
/// the number that can honestly justify (or not) changing what KV
/// representation it defaults to. Same convention as `score_chat`: loss and
/// accuracy count only positions `ChatSample::encode`'s mask marks
/// trainable.
///
/// Each sample gets a fresh `BlockTable`, scored with NO prefix-cache reuse
/// (`Engine::score_positions`) and released before the next sample — held-out
/// samples are independent, so there is nothing to share between them and
/// nothing for prefix reuse to change about the measurement.
pub fn score_chat_paged(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, samples: &[ChatSample], block: u32, kv: KvMode) -> ChatScore {
    let (cfg, tensors) = load_scored_tensors(weights, adapter);
    let block_size = 16u32;
    let max_blocks_per_seq = block.div_ceil(block_size).max(1);
    let num_blocks = max_blocks_per_seq + 1; // one sequence in flight at a time
    let kv_int8 = kv != KvMode::Fp32;
    let mut eng = crate::serve::Engine::from_map(cfg.clone(), &tensors, block_size, num_blocks, 1, max_blocks_per_seq, block.max(1), kv_int8, false);
    if kv == KvMode::Int8Calib {
        let dir = Path::new(weights).parent().unwrap_or_else(|| Path::new("."));
        let calib = model::kvcalib::KvCalib::from_model_dir(dir, cfg.n_layers as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize);
        if calib.is_none() {
            eprintln!("qwen eval --kv int8-calib: no kv_calib.json found beside {weights} (or its shape didn't match); scoring uncalibrated int8 instead");
        }
        eng.set_kv_calib(calib);
    }

    let d = cfg.d_model as usize;
    let mut total_nll = 0.0f64;
    let mut positions = 0usize;
    let mut correct = 0usize;
    let mut skipped = 0usize;

    for (i, s) in samples.iter().enumerate() {
        // Same loud surfacing as `score_chat` -- see the comment there.
        let (ids, mask) = match s.encode(tok, tmpl) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("eval: sample {i}: chat-template encode failed ({e}); skipping");
                skipped += 1;
                continue;
            }
        };
        if ids.len() < 2 || ids.len() as u32 > block {
            skipped += 1;
            continue;
        }
        let mut table = model::paged::BlockTable::new();
        let hidden = eng.score_positions(&mut table, &ids);
        eng.release_table(&mut table);
        for i in 0..ids.len() - 1 {
            if !mask[i + 1] {
                continue;
            }
            let target = ids[i + 1] as usize;
            let row = &hidden[i * d..(i + 1) * d];
            let logits = eng.logits(row);
            let log_prob = model::logprobs::row_logprob(&logits, target);
            total_nll -= log_prob as f64;
            positions += 1;
            if argmax(&logits) as usize == target {
                correct += 1;
            }
        }
    }

    ChatScore {
        loss: if positions > 0 { (total_nll / positions as f64) as f32 } else { f32::NAN },
        token_accuracy: (positions > 0).then(|| correct as f64 / positions as f64),
        positions,
        samples: samples.len() - skipped,
        skipped,
    }
}

#[cfg(test)]
mod paged_scoring_tests {
    use super::*;

    /// GATE: `score_chat_paged` at `KvMode::Fp32` must match `score_chat`'s
    /// loss on the SAME checkpoint + samples — same math (teacher-forced
    /// cross-entropy over the trainable positions), two independently
    /// scheduled compute paths (the legacy single-sequence `Qwen` vs. the
    /// paged `Engine`'s chunked prefill). A real mismatch here means the
    /// paged scoring backend is wrong, independent of any int8 question.
    ///
    /// Needs a real tokenizer + chat template (`QWEN_TOKENIZER=/path/to/
    /// tokenizer.json`, its sibling `tokenizer_config.json` supplies the
    /// template) — self-skips loudly when unset. The
    /// checkpoint's vocab matches the real tokenizer's full range so the
    /// rendered special tokens (`<|im_start|>` etc.) never index outside the
    /// embedding table (the exact class of bug root-caused as a CPU-backend
    /// JIT dispatch segfault in `decode_steps`).
    #[test]
    fn paged_fp32_scoring_matches_the_legacy_backend() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            brain_testutil::skip("set QWEN_TOKENIZER to a real tokenizer.json to run this test");
            return;
        };
        let tok = QwenBpe::from_file(&tok_path).expect("load QWEN_TOKENIZER");
        let base_dir = Path::new(&tok_path).parent().expect("tokenizer path has a parent dir");
        let tmpl = match ChatTemplate::from_model_dir(base_dir) {
            Ok(t) => t,
            Err(e) => {
                brain_testutil::skip(&format!("no chat template beside QWEN_TOKENIZER ({base_dir:?}): {e}"));
                return;
            }
        };

        let cfg = QwenConfig { vocab: 151936, ..QwenConfig::tiny() };
        let init = crate::init::init_weights(&cfg, 13);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
            .collect();
        let dir = std::env::temp_dir().join(format!("qwen-eval-paged-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights_path = dir.join("tiny.safetensors");
        checkpoint::save(weights_path.to_str().unwrap(), cfg.to_json(), &tensors);

        let jsonl = dir.join("samples.jsonl");
        std::fs::write(
            &jsonl,
            concat!(
                r#"{"messages":[{"role":"user","content":"1+1=","train":false},{"role":"assistant","content":"2","train":true}]}"#, "\n",
                r#"{"messages":[{"role":"user","content":"Name a color.","train":false},{"role":"assistant","content":"Blue is a color.","train":true}]}"#, "\n",
            ),
        )
        .unwrap();
        let samples = ChatSample::from_jsonl(&jsonl).expect("parse synthetic samples");
        assert!(!samples.is_empty());

        let legacy = score_chat(weights_path.to_str().unwrap(), None, &tok, &tmpl, &samples, 128);
        let paged = score_chat_paged(weights_path.to_str().unwrap(), None, &tok, &tmpl, &samples, 128, KvMode::Fp32);

        assert_eq!(legacy.samples, paged.samples, "same samples must score, same samples must skip");
        assert_eq!(legacy.positions, paged.positions, "same positions must count");
        // Not bit-exact: independently scheduled chunked-vs-whole-sequence
        // compute rounds differently (the same tolerance/reasoning
        // `random_shared_prefixes_stay_exact` in serve.rs documents for the
        // identical class of comparison).
        assert!(
            (legacy.loss - paged.loss).abs() < 1e-3,
            "paged fp32 loss {} vs legacy loss {} diverges more than rounding should allow",
            paged.loss,
            legacy.loss
        );
        assert_eq!(
            legacy.token_accuracy, paged.token_accuracy,
            "token accuracy is a discrete count over the same positions -- it must match EXACTLY"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `KvMode::parse` must accept exactly the three documented spellings and
    /// reject anything else with a message naming the bad value.
    #[test]
    fn kv_mode_parses_the_three_documented_spellings_and_rejects_others() {
        assert_eq!(KvMode::parse("fp32"), Ok(KvMode::Fp32));
        assert_eq!(KvMode::parse("int8"), Ok(KvMode::Int8));
        assert_eq!(KvMode::parse("int8-calib"), Ok(KvMode::Int8Calib));
        let err = KvMode::parse("bogus").unwrap_err();
        assert!(err.contains("bogus"));
    }
}
