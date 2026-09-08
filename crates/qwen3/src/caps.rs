// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3's capabilities behind the generalized [`capability`] interface — what
//! makes `brain caps qwen` / `brain do qwen generate …` (and the perf suite's
//! `CapabilityTarget`) work with no Qwen-specific plumbing in the CLI.
//!
//! Three actions.
//!
//! `generate`: the same one-shot decode path `brain qwen infer`
//! runs (`Qwen::load_inference` + the KV-cache [`crate::sample`] loop), with a
//! `Progress` emitted **per generated token** so a streaming harness gets a
//! real TTFT/ITL timeline. The manifest is static (no weights needed); the
//! model loads lazily on the first run and stays resident across calls (keyed
//! by weights path + context capacity), mirroring `s3dit::caps`.
//!
//! `lora_train`: the SAME LoRA fine-tuning loop `brain qwen3 finetune --lora`
//! drives (`crate::finetune::finetune` over a `data::chat` masked dataset),
//! published as a capability so anything that reads a manifest - `brain
//! caps`, the event API, D-Bus, a graph editor generating node types - can
//! see that this model is trainable at all. It carries one deliberate
//! improvement over the `flux2`/`s3dit`/`wan` precedent: **its dataset
//! arrives as bytes and its adapter leaves as bytes**. `s3dit::caps`'s
//! `lora_train` asks a caller for a `data` folder and a `save` path, which a
//! scheduler placing work on a machine it has never seen cannot answer;
//! here the only filesystem facts are the base checkpoint and its tokenizer,
//! and both are `host_env` params `Manifest::for_serving` projects out of
//! every off-machine surface. Progress on this action is **stage-level**
//! (prepare → train → save), not per-step: `finetune::finetune` exposes no
//! per-step callback, and claiming a step timeline it cannot produce would
//! be worse than saying so.
//!
//! `lora_gate`: the OTHER half of a continuous-learning cycle - a candidate
//! adapter blob plus a frozen probe set in, a `GateReport` and a
//! promote/reject decision out. Deliberately a second action rather than a
//! flag on `lora_train`: training and judging are separately schedulable,
//! separately priced and separately auditable steps, and a graph that says
//! `lora_train -> lora_gate` is what lets a scheduler place them on two
//! different providers. The decision itself is not written here - it is
//! `promote::gate`'s four bars over `promote::document`'s frozen
//! `{fact, probe_question, expected_answer}` contract, scored by
//! programmatic exact match and never by a model's opinion. This crate
//! supplies only the two things that need a model: what the incumbent
//! decodes, and what the same base with the candidate adapter folded in
//! decodes.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, CancelToken, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::rng::Rng;
use data::tokenizer::Tokenizer;
use promote::document::{document_gate_config, fact_verdicts, train_probe_split, DocumentEnv, DocumentVerifier, FactBatch, FactProbe, FactSplit};
use promote::env::{Environment, Task, Verifier};
use promote::gate::{gate, Cause, Decision, GateInput};
use serde_json::json;

use crate::chat::{parse_request, ParsedRequest, SeqState};
use crate::model::Qwen;

/// The model id used on the CLI (`brain do qwen …`) and the event API.
pub const MODEL: &str = "brain/qwen3";

/// Default generated-token budget.
///
/// **This was a bare `32`, repeated as a literal in two places per crate,
/// with no stated reason in any doc comment, `docs/models/*.md` or roadmap
/// entry, and no measurement behind it.** Thirty-two tokens is half a
/// sentence: it truncated essentially every real answer this decoder was
/// asked for.
///
/// Nothing about this path argues for keeping it that low. The served
/// `generate` action decodes through [`crate::sample`]'s KV-cached loop, so a
/// token is one `O(1)` incremental step, not a recompute of the context; the
/// resident is sized to `prompt + max_new` and grown on demand, so unlike
/// `deepseek2ocr` there is no fixed context ceiling this could overrun; and
/// 128 is already what brain's other KV-cached text actions default to
/// (`llava::caps`, `glmdsa::caps`).
pub const DEFAULT_MAX_NEW: i64 = 128;

/// The longest probe prompt [`gate_lora`] will build a context for.
///
/// A gate run sizes BOTH of its models to the longest prompt it was handed,
/// so an unbounded prompt is an unbounded device allocation requested by a
/// caller who is not standing on this machine. Every other served surface
/// bounds its input for the same reason (a request that names a size must
/// not be able to name any size); this is that bound, for the one dimension
/// this action derives an allocation from. 8192 tokens is orders of
/// magnitude past any real `{fact, probe_question}` phrasing and well inside
/// what a card can hold.
pub const MAX_PROBE_PROMPT: usize = 8192;

/// Default completion budget per PROBE in [`gate_lora`].
///
/// Much smaller than [`DEFAULT_MAX_NEW`] and for a stated reason: a probe's
/// expected answer is a short fact ("13 volts"), the verifier scores the
/// first LINE of the completion, and this budget is paid twice (once per arm)
/// on every probe in the set - 48 of them at minimum. Tokens past the answer
/// cost decode time and change no score.
pub const DEFAULT_GATE_MAX_NEW: i64 = 32;

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let generate = ActionSpec::new("generate", "generate tokens continuing a prompt (KV-cache decode, one Progress per token)")
        .streaming()
        .param(ParamSpec::new("weights", ParamType::Str, "path to a brain-format Qwen checkpoint (.safetensors)").required().host_env("BRAIN_QWEN_WEIGHTS"))
        .param(ParamSpec::new(
            "prompt",
            ParamType::Str,
            "the prompt: text (with a tokenizer) or whitespace/comma-separated token ids (without); ignored when `messages` is set",
        ))
        .param(ParamSpec::new("tokenizer", ParamType::Str, "path to tokenizer.json; omit to feed/return raw token ids").host_env("BRAIN_QWEN_TOKENIZER"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(DEFAULT_MAX_NEW)).min(1.0).max(32768.0).step(1.0))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.0)).min(0.0).max(2.0).step(0.01))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 or negative = disabled)").default(json!(40)).min(0.0).max(1000.0).step(1.0))
        .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)).min(0.0).max(1.0).step(0.01))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)))
        .param(
            ParamSpec::new("precision", ParamType::Str, "model precision: fp32, or int8 (group-wise 32-element weight scales + dynamic activation quant)")
                .default(json!("fp32")),
        )
        .param(ParamSpec::new("eos", ParamType::Int, "stop token id (default: the tokenizer's <|im_end|>/<|endoftext|> when a tokenizer is given; -1 disables)"))
        .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt (needs a tokenizer)").default(json!(false)))
        .param(ParamSpec::new(
            "messages",
            ParamType::Str,
            "JSON array of {role,content,reasoning_content?,tool_calls?,tool_call_id?} chat turns (overrides prompt; needs a tokenizer)",
        ))
        .param(ParamSpec::new("system", ParamType::Str, "optional system prompt prepended to the chat (needs a tokenizer)"))
        .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings (needs a tokenizer)"))
        .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema; needs a tokenizer)"))
        .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{\"type\":\"function\",...}); none withholds tool schemas, required/named are enforced post-generation (finish_reason \"tool_choice_unmet\" when unmet)"))
        .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block (needs a tokenizer)").default(json!(true)))
        .param(ParamSpec::new("reasoning_effort", ParamType::Str, "reasoning effort level: xhigh (default, detailed deliberation), medium (no instruction), or low (brief thinking)").default(json!("xhigh")))
        .param(ParamSpec::new("preserve_thinking", ParamType::Bool, "Qwen3.8 chat-template kwarg: keep <think> blocks from prior assistant turns in the rendered history (inert under this model's Qwen3-era template, whose history framing is positional)").default(json!(true)))
        .output(BlobSpec::new("text", Media::Text, "the generated text (space-separated token ids when no tokenizer is given)"));

    // Every param below is a real per-request knob; the two filesystem facts
    // (base checkpoint, tokenizer) are `host_env`, so `for_serving` drops
    // them and a remote caller is never asked for a path it cannot answer.
    let lora_train = ActionSpec::new("lora_train", "train a LoRA adapter on a masked chat dataset (JSONL blob in, adapter checkpoint blob out)")
        .streaming()
        .param(ParamSpec::new("weights", ParamType::Str, "path to the base brain-format Qwen checkpoint (.safetensors)").required().host_env("BRAIN_QWEN_WEIGHTS"))
        .param(
            ParamSpec::new(
                "tokenizer",
                ParamType::Str,
                "path to the base checkpoint's tokenizer.json; its directory also supplies the chat template. Omit to use the tokenizer.json beside 'weights'",
            )
            .host_env("BRAIN_QWEN_TOKENIZER"),
        )
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank (capacity/size tradeoff)").default(json!(8)).min(1.0).max(256.0).step(1.0))
        .param(ParamSpec::new("alpha", ParamType::Float, "LoRA alpha; omit for 2*rank").min(0.0).max(1024.0))
        .param(ParamSpec::new("steps", ParamType::Int, "training steps").default(json!(500)).min(1.0).max(1_000_000.0).step(1.0))
        .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate (cosine schedule down to lr/10)").default(json!(5e-5)))
        .param(ParamSpec::new("batch", ParamType::Int, "sequences per step").default(json!(4)).min(1.0).max(256.0).step(1.0))
        .param(ParamSpec::new("block", ParamType::Int, "training context length, tokens").default(json!(1024)).min(1.0).max(32768.0).step(1.0))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(1234)))
        .param(ParamSpec::new("dataset_id", ParamType::Str, "provenance id recorded in the adapter's ModelCard"))
        .input(BlobSpec::new("dataset", Media::Bytes, "the training set: data::chat 'generic-messages-v2' JSONL, one packed sample per line").required())
        .input(BlobSpec::new("validation", Media::Bytes, "optional held-out set, same JSONL schema; enables periodic eval"))
        .output(BlobSpec::new("adapter", Media::Bytes, "the trained LoRA adapter checkpoint (safetensors: only the .lora_a/.lora_b tensors)"));

    // No threshold knob anywhere below: `promote::document::
    // document_gate_config` is PRE-REGISTERED, and a request-level
    // `min_effect_size` would let whoever is being gated choose the bar they
    // are gated against. The bars travel back in the report instead, so a
    // reader can see which ones judged the number.
    let lora_gate = ActionSpec::new(
        "lora_gate",
        "gate a candidate LoRA adapter against a frozen probe set (adapter + probes in, a GateReport and a promote/reject decision out)",
    )
    .streaming()
    .param(ParamSpec::new("weights", ParamType::Str, "path to the base brain-format Qwen checkpoint the adapter was trained on").required().host_env("BRAIN_QWEN_WEIGHTS"))
    .param(
        ParamSpec::new("tokenizer", ParamType::Str, "path to the base checkpoint's tokenizer.json; omit to use the tokenizer.json beside 'weights'")
            .host_env("BRAIN_QWEN_TOKENIZER"),
    )
    .param(
        ParamSpec::new("max_new", ParamType::Int, "completion budget per probe; the verifier scores the first LINE of what comes back")
            .default(json!(DEFAULT_GATE_MAX_NEW))
            .min(1.0)
            .max(1024.0)
            .step(1.0),
    )
    .input(BlobSpec::new("adapter", Media::Bytes, "the candidate LoRA adapter checkpoint - exactly `lora_train`'s output blob").required())
    .input(BlobSpec::new(
        "probes",
        Media::Bytes,
        "the frozen probe set: JSONL, one {fact, probe_question, expected_answer} per line, at least 48 of them",
    )
    .required())
    .input(BlobSpec::new(
        "anchor",
        Media::Bytes,
        "the retention suite, same JSONL schema (its `fact` names the behaviour being retained): what training must NOT destroy. Without it the anchor bar has nothing to compare and cannot fire",
    ))
    .output(BlobSpec::new("report", Media::Text, "the full GateReport as JSON: the decision, every number behind it, the bars it was judged against, and a per-fact landed/not-landed row"));

    Manifest::new(
        MODEL,
        "Qwen3 dense decoder - autoregressive text generation with per-token streaming, plus LoRA fine-tuning on a chat dataset and the promote/reject gate over the adapter it produces.",
        vec![generate, lora_train, lora_gate],
    )
}

/// The resident (hot) model: the loaded inference graph plus the key that fixes
/// it. Reused while the weights path matches and the built context capacity
/// covers the request; rebuilt (freeing the old weights first) otherwise.
struct Hot {
    precision: String,
    weights: String,
    cap: u32,
    model: Qwen,
    /// The LM head, read once when this resident is (re)built -- see
    /// `generate_kv_stream_with_head`'s doc comment for the per-request cost
    /// re-reading it every call otherwise pays.
    head: Vec<f32>,
}

/// The executable Qwen model behind the manifest. Construction is free — the
/// checkpoint loads lazily on the first `generate` and stays resident.
#[derive(Default)]
pub struct QwenProvider {
    hot: Arc<Mutex<Option<Hot>>>,
}

impl QwenProvider {
    pub fn new() -> QwenProvider {
        QwenProvider::default()
    }
}

impl Provider for QwenProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "generate" => Some(Arc::new(GenerateAction { hot: self.hot.clone() }) as Arc<dyn Action>),
            // Training holds no resident state: it builds its own trainable
            // graph from the base checkpoint and drops it, so it shares
            // nothing with `generate`'s hot inference model.
            "lora_train" => Some(Arc::new(LoraTrainAction) as Arc<dyn Action>),
            // Gating holds no resident state either, and deliberately does
            // not reuse `generate`'s hot model: the incumbent arm must be
            // the checkpoint on disk at `weights`, not whatever a previous
            // request happened to leave loaded.
            "lora_gate" => Some(Arc::new(LoraGateAction) as Arc<dyn Action>),
            _ => None,
        }
    }
}

struct GenerateAction {
    hot: Arc<Mutex<Option<Hot>>>,
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "generate").expect("known action")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = inv.get_str("weights").ok_or("qwen generate: missing required param 'weights'")?;
        if !Path::new(&weights).exists() {
            return Err(format!("qwen generate: weights not found at '{weights}'"));
        }
        let precision = inv.get_str("precision").unwrap_or_else(|| "fp32".to_string());
        if precision != "fp32" && precision != "int8" {
            return Err(format!("qwen generate: precision must be fp32 or int8, got {precision:?}"));
        }

        // Tokenizer is optional: without one the prompt is raw token ids and the
        // result is returned as ids (the form synthetic/tiny checkpoints use, with
        // no detokenization possible). With one, requests go through the same
        // chat-template/tool-call/stop-string/cancellation logic the HTTP/D-Bus
        // serving path runs (`crate::chat`), so `brain do` and HTTP cannot diverge.
        let tok = match inv.get_str("tokenizer").filter(|p| !p.is_empty()) {
            Some(p) => Some(QwenBpe::from_file(&p)?),
            None => None,
        };
        let plan = match &tok {
            Some(t) => {
                let req = parse_request(t, inv)?;
                // Stop tokens: explicit param wins (-1 disables); else both Qwen3
                // EOS ids (`<|im_end|>` and `<|endoftext|>`) from the tokenizer.
                let eos: Vec<u32> = match inv.get_i64("eos") {
                    Some(e) if e >= 0 => vec![e as u32],
                    Some(_) => Vec::new(),
                    None => ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| t.encode(s).first().copied()).collect(),
                };
                Plan::Chat { req, eos }
            }
            None => {
                let prompt = inv.get_str("prompt").unwrap_or_default();
                let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).max(0) as usize;
                let temp = inv.get_f64("temp").unwrap_or(0.0) as f32;
                let top_k = inv.get_i64("top_k").unwrap_or(40).max(0) as usize;
                let top_p = inv.get_f64("top_p").unwrap_or(1.0) as f32;
                let seed = inv.get_i64("seed").unwrap_or(0).max(0) as u64;
                let ids: Vec<u32> = prompt
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u32>().map_err(|_| format!("qwen generate: without a tokenizer the prompt must be token ids (got '{s}')")))
                    .collect::<Result<_, _>>()?;
                if ids.is_empty() {
                    return Err("qwen generate: empty prompt".to_string());
                }
                let eos: Vec<u32> = match inv.get_i64("eos") {
                    Some(e) if e >= 0 => vec![e as u32],
                    _ => Vec::new(),
                };
                Plan::Raw { ids, max_new, temp, top_k, top_p, seed, eos }
            }
        };

        // Hot path: keep the loaded model resident across calls; rebuild only when
        // the weights change or the built context is too small for this request.
        let need = match &plan {
            Plan::Chat { req, .. } => (req.ids.len() + req.max_new) as u32,
            Plan::Raw { ids, max_new, .. } => (ids.len() + max_new) as u32,
        };
        let mut guard = self.hot.lock().map_err(|_| "qwen: hot model lock poisoned")?;
        let reuse = matches!(&*guard, Some(h) if h.weights == weights && h.cap >= need && h.precision == precision);
        if !reuse {
            *guard = None; // free the old resident weights before loading new
            let cap = need.max(64);
            let model = if precision == "int8" {
                Qwen::load_inference_i8(&weights, 1, cap)
            } else {
                Qwen::load_inference(&weights, 1, cap)
            };
            let head = model.read_weight(model.cfg.head_weight());
            *guard = Some(Hot { precision: precision.clone(), weights: weights.clone(), cap, model, head });
        }
        let hot = guard.as_ref().unwrap();
        let model = &hot.model;

        match plan {
            Plan::Chat { req, eos } => {
                let tok = tok.expect("Plan::Chat is only built when a tokenizer was loaded");
                let mut rng = Rng::new(req.seed);
                let mut seq = SeqState::new(&req, inv.cancel.clone());
                let mut ids_out: Vec<u32> = Vec::with_capacity(req.max_new);
                let gen = crate::sample::generate_kv_stream_with_head(
                    model,
                    &req.ids,
                    req.max_new,
                    req.temp,
                    req.top_k,
                    req.top_p,
                    &eos,
                    &mut rng,
                    &hot.head,
                    &mut |_i, t| {
                        ids_out.push(t);
                        !seq.advance(&tok, &ids_out, progress)
                    },
                );
                Ok(seq.finish(&tok, &gen, progress))
            }
            Plan::Raw { ids, max_new, temp, top_k, top_p, seed, eos } => {
                let mut rng = Rng::new(seed);
                let total = max_new as u32;
                let gen = crate::sample::generate_kv_stream_with_head(model, &ids, max_new, temp, top_k, top_p, &eos, &mut rng, &hot.head, &mut |i, _t| {
                    progress(Progress::step(i as u32 + 1, total, "token"));
                    true
                });
                let text = gen.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
                Ok(Outcome::new()
                    .set("tokens", json!(gen.len()))
                    .set("ids", json!(gen))
                    .set("text", json!(text.clone()))
                    .blob("text", Blob::new(Media::Text, text.into_bytes())))
            }
        }
    }
}

/// `lora_train`: stateless. It builds a trainable graph from the base
/// checkpoint, trains, writes the adapter, and drops everything.
struct LoraTrainAction;

impl Action for LoraTrainAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "lora_train").expect("known action")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        train_lora(inv, progress)
    }
}

/// The `lora_train` action body, exposed the way `flux2::caps::train_action`
/// is: a residency adapter (`crates/cli/src/resident_*.rs`) runs the same
/// code the in-process provider does, so a served run and a `brain do` run
/// cannot train differently.
///
/// A chat-JSONL dataset blob in, a LoRA adapter checkpoint blob out. Every
/// intermediate - the decoded JSONL, the masked token dataset
/// `data::chat::prepare_chat_samples` writes, the full post-training
/// checkpoint - lives in one scratch directory that is removed on **every**
/// exit path, so a served process does not accumulate multi-gigabyte
/// leftovers from cancelled or failed requests.
pub fn train_lora(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let weights = inv.get_str("weights").ok_or("qwen lora_train: missing required param 'weights'")?;
    if !Path::new(&weights).exists() {
        return Err(format!("qwen lora_train: weights not found at '{weights}'"));
    }
    let scratch = scratch_dir("lora-train")?;
    let out = train_in(&scratch, &weights, inv, progress);
    // Best-effort cleanup on success AND on error: a failed removal must not
    // mask the training result (or the error) the caller actually asked for.
    let _ = std::fs::remove_dir_all(&scratch);
    out
}

/// [`train_lora`]'s body, with `scratch` already created and owned by the
/// caller (which removes it however this returns).
fn train_in(scratch: &Path, weights: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let rank = inv.get_i64("rank").unwrap_or(8).max(1) as u32;
    let alpha = inv.get_f64("alpha").map(|a| a as f32).unwrap_or(rank as f32 * 2.0);
    let steps = inv.get_i64("steps").unwrap_or(500).max(1) as u32;
    let lr = inv.get_f64("lr").unwrap_or(5e-5) as f32;
    let batch = inv.get_i64("batch").unwrap_or(4).max(1) as u32;
    let block = inv.get_i64("block").unwrap_or(1024).max(1) as u32;
    let seed = inv.get_i64("seed").unwrap_or(1234).max(0) as u64;
    let dataset_id = inv.get_str("dataset_id").filter(|s| !s.is_empty());

    progress(Progress::step(1, 4, "preparing dataset"));

    // The tokenizer, and from its own directory the checkpoint's chat
    // template, are host facts like the weights are - `brain qwen3 finetune`
    // reads both out of the base model's directory, and this action defaults
    // to exactly that rather than inventing a second convention.
    let tok_path: PathBuf = match inv.get_str("tokenizer").filter(|p| !p.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => Path::new(weights).parent().unwrap_or(Path::new(".")).join("tokenizer.json"),
    };
    let tok = QwenBpe::from_file(&tok_path.to_string_lossy())?;
    let tmpl_dir = tok_path.parent().unwrap_or(Path::new("."));
    let tmpl = ChatTemplate::from_model_dir(tmpl_dir).map_err(|e| format!("qwen lora_train: {e}"))?;

    // The dataset is UNTRUSTED caller input: decoded through the one strict,
    // `deny_unknown_fields` parse (`ChatSample::from_jsonl`) every other
    // caller of this format uses, never a permissive re-read here.
    let train = read_jsonl_blob(inv, "dataset", scratch)?.ok_or("qwen lora_train: missing required input 'dataset'")?;
    if train.is_empty() {
        return Err("qwen lora_train: 'dataset' contains no samples".to_string());
    }
    let val = read_jsonl_blob(inv, "validation", scratch)?.unwrap_or_default();

    // The MODEL's vocab (not the tokenizer's): it must match the checkpoint's
    // own embedding/lm_head row count, which is what `prepare_chat_samples`
    // records in the dataset's meta.json.
    let vocab = crate::config::QwenConfig::from_json(&checkpoint::read_config(weights)).vocab as usize;
    let data_dir = scratch.join("data");
    data::chat::prepare_chat_samples(&train, &val, &tok, &tmpl, vocab, &data_dir).map_err(|e| format!("qwen lora_train: preparing training data: {e}"))?;

    let opts = model::FitOpts {
        steps,
        batch_size: batch,
        block_size: block,
        lr,
        min_lr: lr * 0.1,
        warmup: (steps / 20).max(1),
        decay_iters: steps,
        weight_decay: 0.1,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: if val.is_empty() { 0 } else { (steps / 10).max(1) },
        eval_batches: 20,
        checkpoint_secs: 0,
        // The token mask file `prepare_chat_samples` writes supersedes
        // character-offset masking; `model::load_dataset` prefers it.
        mask_before: None,
        mask_per_line: false,
        align_to_lines: false,
        seed,
    };

    progress(Progress::step(2, 4, format!("training {steps} steps (rank {rank}, alpha {alpha})")));
    let full = scratch.join("full.safetensors");
    let (l0, l1) = crate::finetune::finetune(weights, &data_dir, &opts, &crate::finetune::Mode::Lora { rank, alpha }, &full.to_string_lossy())
        .map_err(|e| format!("qwen lora_train: {e}"))?;

    progress(Progress::step(3, 4, "saving adapter"));
    // Only the adapter tensors travel back - a rank-8 adapter is a few MB
    // against a multi-GB base, and the caller already has the base.
    let base_id = checkpoint::st::read_card(weights).ok().flatten().map(|c| c.id).unwrap_or_else(|| MODEL.to_string());
    // `modelref`'s adapter grammar is `<base>:<owner>:<name>:<tag>`. This
    // action deliberately takes no adapter NAME: it hands back bytes, and
    // deciding where those bytes are published (and under what id) belongs
    // to whoever stores them, not to the training step.
    let card_id = format!("{base_id}:brain:lora:latest");
    let adapter_path = scratch.join("adapter.safetensors");
    // The adapter is read back through a TRAINING-shaped build, never
    // `Qwen::load_inference`: `Model::param_names` reports the OPTIMISED set
    // (`ParamStore::trainable` + `offload`), and an inference build marks
    // every parameter `Role::Frozen`, so `save_adapter` would find no
    // `.lora_a`/`.lora_b` at all and assert. This model exists only to read
    // those tensors out, so it is built at the smallest activation shape
    // there is (b = 1, t = 1) - `param_names`/`read_weight` do not depend on
    // it, and the frozen base is uploaded once either way.
    let ck = checkpoint::load(&full.to_string_lossy());
    let trained_cfg = crate::config::QwenConfig::from_json(&ck.header["config"]);
    let trained = Qwen::new(trained_cfg, 1, 1, &ck.by_role(""));
    crate::lora::save_adapter(&adapter_path.to_string_lossy(), &trained, &card_id, &base_id, dataset_id.as_deref())
        .map_err(|e| format!("qwen lora_train: save_adapter: {e}"))?;
    let bytes = std::fs::read(&adapter_path).map_err(|e| format!("qwen lora_train: read trained adapter: {e}"))?;

    progress(Progress::step(4, 4, "done"));
    Ok(Outcome::new()
        .set("rank", json!(rank))
        .set("alpha", json!(alpha))
        .set("steps", json!(steps))
        .set("train_samples", json!(train.len()))
        .set("val_samples", json!(val.len()))
        .set("initial_loss", json!(l0))
        .set("final_loss", json!(l1))
        .set("base", json!(base_id))
        .blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"id": card_id}))))
}

/// A fresh scratch directory for one request's intermediates, created and
/// owned by the caller (which removes it however the request returns).
///
/// Unique per call: two concurrent requests must not share a scratch tree,
/// and the pid alone does not separate them.
fn scratch_dir(tag: &str) -> Result<PathBuf, String> {
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("brain-qwen3-{tag}-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("qwen {tag}: {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Decode one optional JSONL blob into packed [`ChatSample`]s. The bytes are
/// staged under `scratch` first so parsing goes through the SAME
/// `ChatSample::from_jsonl` the CLI and bench exports use - one strict
/// schema, one set of error messages naming the offending line.
fn read_jsonl_blob(inv: &Invocation, name: &str, scratch: &Path) -> Result<Option<Vec<ChatSample>>, String> {
    let Some(blob) = inv.get_blob(name) else {
        return Ok(None);
    };
    let path = scratch.join(format!("{name}.jsonl"));
    std::fs::write(&path, &blob.bytes).map_err(|e| format!("qwen lora_train: staging '{name}': {e}"))?;
    let samples = ChatSample::from_jsonl(&path).map_err(|e| format!("qwen lora_train: '{name}': {e}"))?;
    Ok(Some(samples))
}

/// `lora_gate`: stateless, like `lora_train`. It builds each arm from bytes
/// on disk, scores it, drops it.
struct LoraGateAction;

impl Action for LoraGateAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == "lora_gate").expect("known action")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        gate_lora(inv, progress)
    }
}

/// The `lora_gate` action body: a candidate adapter blob and a frozen probe
/// set in, `promote::gate`'s four-bar decision out.
///
/// ## What this function is, and what it is NOT
///
/// The decision is `promote::gate::gate` and nothing else - the exact
/// one-sided paired sign test, the pre-registered effect-size floor, the
/// anchor budget and the entropy collapse check, all over
/// `promote::document`'s frozen probe contract. None of that is written
/// here, and none of it may be: it is the same arithmetic `rl::improve`'s
/// training-cycle gate runs, in the leaf crate both can reach (B2b). What
/// this function adds is the only part that needs a model - decoding both
/// arms - plus the untrusted-input handling a served surface owes.
///
/// ## Both arms are built from bytes, never from a resident model
///
/// The incumbent is the checkpoint at `weights` as it sits on disk; the
/// candidate is that same checkpoint with the adapter blob folded in
/// (`lora::fold_adapter_into`), which is exactly what a caller would serve
/// if this gate promoted. `crate::caps`'s own hot `generate` model is
/// deliberately not reused: scoring whatever a previous request happened to
/// leave loaded is the "what is served must be what is scored" mistake
/// `promote::gate`'s module doc records.
///
/// Both arms decode GREEDILY, the same tasks in the same order, through the
/// same KV-cache loop `generate` serves from. A sampled arm would make the
/// comparison noise.
///
/// ## The incumbent is the checkpoint at `weights`, and nothing else
///
/// So a candidate is gated against the BASE, not against some previously
/// promoted adapter. That is the right incumbent for the one-cycle
/// `lora_train -> lora_gate` graph this action exists for; gating cycle k
/// against cycle k-1's promoted adapter means pointing `weights` at a
/// checkpoint that adapter has already been folded into, and the host owns
/// that fact (`BRAIN_QWEN_WEIGHTS`), not the request. Comparing a candidate
/// against a base that is not what is actually being served would make every
/// number here describe a comparison nobody is running.
pub fn gate_lora(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let weights = inv.get_str("weights").ok_or("qwen lora_gate: missing required param 'weights'")?;
    if !Path::new(&weights).exists() {
        return Err(format!("qwen lora_gate: weights not found at '{weights}'"));
    }
    let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_GATE_MAX_NEW).max(1) as usize;
    let adapter_bytes = &inv.get_blob("adapter").ok_or("qwen lora_gate: missing required input 'adapter'")?.bytes;

    // Same convention as `lora_train`: the tokenizer is a host fact, read
    // from beside the base checkpoint unless the caller named another one.
    let tok_path: PathBuf = match inv.get_str("tokenizer").filter(|p| !p.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => Path::new(&weights).parent().unwrap_or(Path::new(".")).join("tokenizer.json"),
    };
    let tok = QwenBpe::from_file(&tok_path.to_string_lossy())?;

    progress(Progress::step(1, 4, "reading the frozen probe set"));
    let probe_rows = FactProbe::from_jsonl(&inv.get_blob("probes").ok_or("qwen lora_gate: missing required input 'probes'")?.bytes)
        .map_err(|e| format!("qwen lora_gate: 'probes': {e}"))?;
    let anchor_rows = match inv.get_blob("anchor") {
        Some(b) => FactProbe::from_jsonl(&b.bytes).map_err(|e| format!("qwen lora_gate: 'anchor': {e}"))?,
        None => Vec::new(),
    };

    let batch = guarded("probes", || FactBatch::new(probe_rows))?;
    // Called for its two structural assertions - the >= 48 held-out floor and
    // the train/probe task-id disjointness - not for the training half, which
    // a gate run never decodes.
    guarded("probes", || {
        let _ = train_probe_split(&batch, &tok);
    })?;
    let probe_tasks: Vec<Task> = tasks_of(&batch, &tok);
    let n_probes = probe_tasks.len();

    let anchor_batch = if anchor_rows.is_empty() { None } else { Some(guarded("anchor", || FactBatch::new(anchor_rows))?) };
    let anchor_tasks: Vec<Task> = anchor_batch.as_ref().map(|b| tasks_of(b, &tok)).unwrap_or_default();
    // A retention suite that IS the primary suite cannot detect forgetting:
    // the same task would be counted as both the thing being improved and the
    // thing being preserved, and the anchor bar would silently track the
    // primary win instead of opposing it.
    let probe_ids: HashSet<&str> = probe_tasks.iter().map(|t| t.id.as_str()).collect();
    let clash: Vec<&str> = anchor_tasks.iter().map(|t| t.id.as_str()).filter(|id| probe_ids.contains(id)).collect();
    if !clash.is_empty() {
        return Err(format!("qwen lora_gate: anchor probe(s) {clash:?} are also frozen probes - a retention suite that is the primary suite cannot detect forgetting"));
    }

    // ONE decode pass per arm over `probes ++ anchor`, exactly as
    // `rl::improve::score_and_gate` orders it, so the two halves are split
    // back out by index rather than decoded twice.
    let all: Vec<Task> = probe_tasks.iter().chain(anchor_tasks.iter()).cloned().collect();
    let longest = all.iter().map(|t| t.prompt.len()).max().unwrap_or(0);
    if longest > MAX_PROBE_PROMPT {
        return Err(format!(
            "qwen lora_gate: a probe prompt is {longest} tokens, past the {MAX_PROBE_PROMPT}-token bound - both arms are sized to the longest probe, so an unbounded prompt is an unbounded allocation"
        ));
    }
    let cap = (longest + max_new).max(64) as u32;
    // `special_id`, never `encode(...).first()`: on a vocabulary that has no
    // such special token, encoding the literal text yields the id of `<`,
    // which would stop every completion at its first angle bracket.
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.special_id(s)).collect();
    let verifier = DocumentVerifier::new(&tok);

    progress(Progress::step(2, 4, format!("scoring the incumbent on {n_probes} probe(s) and {} anchor(s)", anchor_tasks.len())));
    let incumbent = {
        let m = Qwen::load_inference(&weights, 1, cap);
        score_arm(&m, &all, &verifier, max_new, &eos, &inv.cancel)?
    };

    progress(Progress::step(3, 4, "folding the candidate adapter and scoring it"));
    let candidate = {
        let scratch = scratch_dir("lora-gate")?;
        let out = fold_and_score(&scratch, &weights, adapter_bytes, cap, &all, &verifier, max_new, &eos, &inv.cancel);
        // Best-effort cleanup on success AND on error, as `train_lora` does.
        let _ = std::fs::remove_dir_all(&scratch);
        out?
    };

    progress(Progress::step(4, 4, "gating"));
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    // The anchor headline is the retention suite's own mean when there is
    // one, and otherwise the held-out mean - byte-for-byte
    // `rl::improve::score_and_gate`'s rule. With no anchor suite the
    // AnchorRegressed bar cannot fire (the headline is then the very set the
    // effect-size bar just cleared), which is why the report states how many
    // anchor probes there were rather than leaving a reader to assume four
    // bars were live.
    let headline = |a: &Arm| if a.scores.len() > n_probes { mean(&a.scores[n_probes..]) } else { mean(&a.scores[..n_probes]) };
    let cfg = document_gate_config();
    let report = gate(
        &GateInput {
            candidate_scores: &candidate.scores[..n_probes],
            incumbent_scores: &incumbent.scores[..n_probes],
            anchor_candidate: headline(&candidate),
            anchor_incumbent: headline(&incumbent),
            entropy_candidate: candidate.mean_entropy,
            entropy_incumbent: incumbent.mean_entropy,
        },
        &cfg,
    );

    // Per-fact rows from the candidate arm's OWN per-task scores - the same
    // decodes the gate was computed from, never a second pass. A batch
    // verdict of "promote" can still hide facts that did not land, and
    // "reject" can still hide facts that did (continuous-learning B8).
    let by_task: Vec<(String, f64)> = probe_tasks.iter().zip(candidate.scores.iter()).map(|(t, s)| (t.id.clone(), *s)).collect();
    let verdicts = fact_verdicts(&batch, &by_task);

    let (decision, cause) = match report.decision {
        Decision::Promote => ("promote".to_string(), None),
        // The same rendering `checkpoint::st::GateOutcome` records for a
        // training cycle, so a lineage record and a gate response cannot
        // describe one decision two ways. `cause` beside it is the machine
        // handle: a consumer selecting on the reason must not have to parse
        // a Debug rendering.
        Decision::Reject(c) => (format!("reject: {c:?}"), Some(cause_name(c))),
    };
    let promoted = cause.is_none();
    let report_json = json!({
        "decision": &decision,
        "promote": promoted,
        "cause": cause,
        "p_value": report.p_value,
        "effect_size": report.effect_size,
        "anchor_delta": report.anchor_delta,
        "entropy_ratio": report.entropy_ratio,
        "n_discordant": report.n_discordant,
        "k_wins": report.k_wins,
        "probes": n_probes,
        "anchor_probes": anchor_tasks.len(),
        "candidate_pass_rate": mean(&candidate.scores[..n_probes]),
        "incumbent_pass_rate": mean(&incumbent.scores[..n_probes]),
        // The pre-registered bars travel WITH the number they judged: a
        // report that says "promote" without saying what it had to clear is
        // not re-checkable later.
        "gate_config": {
            "alpha": cfg.alpha,
            "min_effect_size": cfg.min_effect_size,
            "anchor_budget": cfg.anchor_budget,
            "min_entropy_ratio": cfg.min_entropy_ratio,
        },
        "facts": verdicts.iter().map(|v| json!({"fact": v.fact, "landed": v.landed})).collect::<Vec<_>>(),
    });

    Ok(Outcome::new()
        .set("decision", json!(decision))
        .set("promote", json!(promoted))
        .set("cause", json!(cause))
        .set("p_value", json!(report.p_value))
        .set("effect_size", json!(report.effect_size))
        .set("anchor_delta", json!(report.anchor_delta))
        .set("entropy_ratio", json!(report.entropy_ratio))
        .set("n_discordant", json!(report.n_discordant))
        .set("k_wins", json!(report.k_wins))
        .set("probes", json!(n_probes))
        .set("anchor_probes", json!(anchor_tasks.len()))
        .set("facts_not_landed", json!(verdicts.iter().filter(|v| !v.landed).map(|v| v.fact.clone()).collect::<Vec<_>>()))
        .blob("report", Blob::new(Media::Text, report_json.to_string().into_bytes())))
}

/// Every one of a batch's frozen probes as a [`Task`], in batch order.
/// `DocumentEnv::tasks` maps a seed onto `seed % len`, so `0..len` enumerates
/// the half exactly once - the same enumeration `promote::document`'s own
/// tests use, rather than a second convention for walking a batch.
fn tasks_of(batch: &FactBatch, tok: &QwenBpe) -> Vec<Task> {
    let env = DocumentEnv::new(batch, FactSplit::Probe, tok);
    (0..env.len()).map(|i| env.tasks(i as u64).remove(0)).collect()
}

/// One arm's scores over the task list the gate was handed.
struct Arm {
    /// Per task, in the `probes ++ anchor` order they were decoded in.
    scores: Vec<f64>,
    /// Mean completion entropy over those tasks - the gate's non-degeneracy
    /// signal.
    mean_entropy: f64,
}

/// Decode every task greedily and score it with `verifier`, keeping the
/// per-task score and the arm's mean completion entropy.
///
/// The entropy is `model::rollout::mean_completion_entropy`, the same
/// function `rl::improve`'s training-cycle gate reads its own
/// `entropy_candidate`/`entropy_incumbent` from - the policy's full
/// per-position distribution, not the logprob of the token it happened to
/// emit. It costs one extra forward per task, which is what buys a
/// degeneracy signal a pass-rate cannot see.
fn score_arm(qwen: &Qwen, tasks: &[Task], verifier: &dyn Verifier, max_new: usize, eos: &[u32], cancel: &CancelToken) -> Result<Arm, String> {
    let head = qwen.read_weight(qwen.cfg.head_weight());
    let mut scores = Vec::with_capacity(tasks.len());
    let mut entropy = 0.0f64;
    for task in tasks {
        // Between tasks, per `CancelToken`'s own contract. A cancelled run
        // returns NO decision: a gate computed from a truncated probe suite
        // would be a promote/reject verdict over a set nobody chose.
        if cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        // Greedy (`temp = 0`), so the rng is never drawn from; it exists
        // because the shared decode loop takes one.
        let mut rng = Rng::new(0);
        let completion = crate::sample::generate_kv_stream_with_head(qwen, &task.prompt, max_new, 0.0, 0, 1.0, eos, &mut rng, &head, &mut |_, _| true);
        scores.push(verifier.verify(task, &[], &completion).value as f64);
        entropy += model::rollout::mean_completion_entropy(qwen, &task.prompt, &completion);
    }
    let mean_entropy = if tasks.is_empty() { 0.0 } else { entropy / tasks.len() as f64 };
    Ok(Arm { scores, mean_entropy })
}

/// Build the candidate arm - the base checkpoint with the adapter blob folded
/// into its weights - and score it. The fold needs the whole base as a host
/// tensor map (`checkpoint::load` + `by_role`), which the streaming incumbent
/// load avoids; that cost is inherent to serving an adapter at all, and is
/// why the incumbent is scored and dropped FIRST rather than held resident
/// alongside this one.
///
/// Built with `new_shard(train = false)` and NOT with the cheaper
/// `Qwen::from_tensors_decode`: a decode-only build sizes its activations for
/// a single token and refuses a batched forward outright, and the gate's
/// entropy signal is exactly that forward (`Model::logits_all` over
/// prompt + completion).
fn fold_and_score(
    scratch: &Path,
    weights: &str,
    adapter_bytes: &[u8],
    cap: u32,
    tasks: &[Task],
    verifier: &dyn Verifier,
    max_new: usize,
    eos: &[u32],
    cancel: &CancelToken,
) -> Result<Arm, String> {
    let adapter_path = scratch.join("candidate.safetensors");
    std::fs::write(&adapter_path, adapter_bytes).map_err(|e| format!("qwen lora_gate: staging the candidate adapter: {e}"))?;
    let ck = checkpoint::load(weights);
    let cfg = crate::config::QwenConfig::from_json(&ck.header["config"]);
    let mut tensors = ck.by_role("");
    // The adapter is caller-supplied bytes: an adapter for a different base
    // (a tensor name the checkpoint does not carry, a rank that does not
    // divide the factors) must fail THIS request with a message, not unwind
    // the thread serving it.
    guarded("adapter", || crate::lora::fold_adapter_into(&mut tensors, &adapter_path.to_string_lossy()))?
        .map_err(|e| format!("qwen lora_gate: folding the candidate adapter: {e}"))?;
    let n_layers = cfg.n_layers as usize;
    let m = Qwen::new_shard(cfg, 1, cap, &tensors, false, crate::model::Shard::whole(n_layers));
    score_arm(&m, tasks, verifier, max_new, eos, cancel)
}

/// The stable, machine-readable name of a rejection cause - what a consumer
/// selects on. `Cause`'s `Debug` rendering carries the deciding numbers and
/// is what a human reads; it is not a wire format.
fn cause_name(cause: Cause) -> &'static str {
    match cause {
        Cause::NotSignificant { .. } => "NotSignificant",
        Cause::EffectTooSmall { .. } => "EffectTooSmall",
        Cause::AnchorRegressed { .. } => "AnchorRegressed",
        Cause::Degenerate { .. } => "Degenerate",
    }
}

/// Run `f`, converting a panic into a clean error naming what failed.
///
/// `promote::document`'s contract checks panic BY DESIGN, and correctly so:
/// every one of them is a defect in whatever produced the batch, and
/// continuing produces a flattering number. But those checks run here against
/// input that arrived over D-Bus or HTTP from a caller who is not standing on
/// this machine, where all request input is hostile - a malformed probe set
/// must fail this ONE request, with the message that names the offending
/// record, rather than unwinding the worker thread that is serving it.
fn guarded<T>(what: &str, f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|e| {
        let msg = e
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown failure".to_string());
        format!("qwen lora_gate: '{what}': {msg}")
    })
}

/// The two request shapes `generate` accepts: with a tokenizer, the shared
/// chat-serving parse (chat template, tool schemas, stop strings); without
/// one, raw token ids in and out, with no detokenization possible.
enum Plan {
    Chat { req: ParsedRequest, eos: Vec<u32> },
    Raw { ids: Vec<u32>, max_new: usize, temp: f32, top_k: usize, top_p: f32, seed: u64, eos: Vec<u32> },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QwenConfig;
    use capability::Registry;
    use promote::document::MIN_HELD_OUT_PROBES;

    /// Write a tiny synthetic Qwen checkpoint at `path`, returning its config.
    fn write_tiny_checkpoint(path: &Path, cfg: &QwenConfig, seed: u64) {
        let init = crate::init::init_weights(cfg, seed);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| {
                let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
                (name, vec![n as u64], v)
            })
            .collect();
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
    }

    /// A `tokenizer.json` + `tokenizer_config.json` pair covering exactly the
    /// characters the fixture below uses, so a LoRA round-trip needs no real
    /// checkpoint and no `QWEN_TOKENIZER`.
    ///
    /// `QwenBpe` encodes each byte through GPT-2's `bytes_to_unicode` map,
    /// which is the IDENTITY on bytes `0x21..=0x7E`; the only two other bytes
    /// this fixture emits are space and newline, at U+0120 and U+010A. With
    /// no merges every byte is its own token, and a byte with no vocab entry
    /// is dropped by `encode_piece` rather than mis-encoded - so this vocab
    /// is enough to round-trip the ASCII fixture text.
    fn write_byte_tokenizer(dir: &Path) {
        let mut vocab = serde_json::Map::new();
        for (i, b) in (0x21u8..=0x7e).enumerate() {
            vocab.insert((b as char).to_string(), json!(i));
        }
        let n = vocab.len();
        vocab.insert('\u{0120}'.to_string(), json!(n)); // space
        vocab.insert('\u{010a}'.to_string(), json!(n + 1)); // newline
        std::fs::write(dir.join("tokenizer.json"), json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();
        // One line per message, role-tagged: prefix-stable, so
        // `render_with_message_boundaries` can attribute every span.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            json!({"chat_template": "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}"}).to_string(),
        )
        .unwrap();
    }

    /// The dataset blob a caller hands `lora_train`: `data::chat`'s
    /// `generic-messages-v2` JSONL, one packed sample per line, the assistant
    /// turn supervised and the user turn not.
    fn dataset_jsonl() -> Vec<u8> {
        let pairs = [
            ("where does the zarnu river end", "it ends in kestrel valley"),
            ("what year was ondrix corp founded", "ondrix corp was founded in 1994"),
            ("where is quenite mined", "quenite is mined on vesper island"),
            ("who charted the belanor strait", "the belanor strait was charted by ilva reso"),
            ("how tall is the murran spire", "the murran spire is 214 metres tall"),
            ("what powers the kessel array", "the kessel array runs on tidal current"),
        ];
        let mut out = String::new();
        for (q, a) in pairs {
            out.push_str(
                &json!({"messages": [
                    {"role": "user", "content": q, "train": false},
                    {"role": "assistant", "content": a, "train": true},
                ]})
                .to_string(),
            );
            out.push('\n');
        }
        out.into_bytes()
    }

    /// `continuous-learning` B3a/B3b: qwen3's LoRA training loop was CLI-only
    /// (`crates/cli/src/qwen_cli.rs::finetune_lora`) and its promote/reject
    /// gate was not reachable from this crate at all, so nothing that reads a
    /// manifest - `brain caps`, the event API, D-Bus, whale's node-type
    /// generation - could see that this model can be trained, let alone that
    /// training it and gating the result are two separate steps. The manifest
    /// is the only place those facts can live, and `lora_train -> lora_gate`
    /// being TWO actions is what makes the whale graph two nodes.
    #[test]
    fn manifest_lists_generate_lora_train_and_lora_gate() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["generate", "lora_train", "lora_gate"]);
    }

    /// `lora_gate`'s I/O shape: the candidate adapter and the frozen probe set
    /// travel as blobs (the adapter blob being exactly `lora_train`'s output
    /// blob is what lets a scheduler wire the two nodes together without a
    /// shared filesystem), and the gate's four bars are NOT caller-tunable.
    /// Pre-registered means fixed before the run - a `min_effect_size` knob on
    /// the request would let whoever is being gated pick the bar they are
    /// gated against, which is the whole failure mode `document_gate_config`
    /// exists to close.
    #[test]
    fn lora_gate_takes_the_adapter_and_the_frozen_probes_as_blobs() {
        let m = manifest();
        let g = m.actions.iter().find(|a| a.name == "lora_gate").expect("lora_gate is declared");
        assert!(g.streaming, "lora_gate must stream stage progress");
        for (name, required) in [("adapter", true), ("probes", true), ("anchor", false)] {
            let b = g.inputs.iter().find(|b| b.name == name).unwrap_or_else(|| panic!("lora_gate declares the '{name}' input"));
            assert_eq!(b.media, Media::Bytes, "'{name}' travels as bytes, never as a server path");
            assert_eq!(b.required, required, "'{name}' required-ness");
        }
        assert!(g.outputs.iter().any(|b| b.name == "report" && b.media == Media::Text), "the GateReport must come back as a retrievable blob");
        for banned in ["alpha", "min_effect_size", "anchor_budget", "min_entropy_ratio", "gate_config"] {
            assert!(!g.params.iter().any(|p| p.name == banned), "the pre-registered gate bars must not be a caller-supplied param ('{banned}')");
        }
    }

    /// The deliberate improvement over the `flux2`/`s3dit` precedent: the
    /// dataset arrives as bytes and the adapter leaves as bytes, so a caller
    /// that does not share this machine's filesystem can still drive a
    /// training node and collect its product.
    #[test]
    fn lora_train_declares_the_adapter_as_a_retrievable_output_blob() {
        let m = manifest();
        let lt = m.actions.iter().find(|a| a.name == "lora_train").expect("lora_train is declared");
        assert!(lt.streaming, "lora_train must stream progress");
        assert!(
            lt.inputs.iter().any(|b| b.name == "dataset" && b.media == Media::Bytes && b.required),
            "the training set must travel as a required Bytes blob, never as a server path"
        );
        assert!(
            lt.outputs.iter().any(|b| b.name == "adapter" && b.media == Media::Bytes),
            "the trained adapter must come back as a retrievable Bytes blob"
        );
        let w = lt.params.iter().find(|p| p.name == "weights").expect("base weights param");
        assert!(w.required);
        assert_eq!(w.host_env.as_deref(), Some("BRAIN_QWEN_WEIGHTS"), "the base checkpoint is the HOST's fact, not a caller's");
    }

    /// REGRESSION GUARD for the BlobSpec discipline above: `s3dit::caps`'s
    /// `lora_train` still asks a caller for a `data` folder and a `save`
    /// path, which a scheduler placing work on a machine it has never seen
    /// cannot answer. Nothing qwen3 publishes off-machine may do that - the
    /// exact served knob list is asserted so a path param added later cannot
    /// slip in unnoticed.
    #[test]
    fn the_served_manifest_carries_no_filesystem_path_param() {
        let served = manifest().for_serving();
        for a in &served.actions {
            let names: Vec<&str> = a.params.iter().map(|p| p.name.as_str()).collect();
            for banned in ["weights", "tokenizer", "data", "save", "dataset", "adapter", "out", "dir", "path"] {
                assert!(!names.contains(&banned), "served action '{}' must not ask an off-machine caller for '{banned}': {names:?}", a.name);
            }
        }
        let lt = served.actions.iter().find(|a| a.name == "lora_train").expect("lora_train survives for_serving");
        let names: Vec<&str> = lt.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["rank", "alpha", "steps", "lr", "batch", "block", "seed", "dataset_id"]);
        assert!(lt.inputs.iter().any(|b| b.name == "dataset"), "the dataset survives as a blob");
        assert!(lt.outputs.iter().any(|b| b.name == "adapter"), "the adapter survives as a blob");
    }

    /// The real round-trip: drive `lora_train` through the Registry on the
    /// tiny synthetic checkpoint, then fold the returned adapter blob into a
    /// fresh copy of that same base and assert it measurably moved a targeted
    /// weight - and left an untargeted one (`tok.weight`) alone. An adapter
    /// whose `lora_b` never left its zero-delta init folds as a no-op, so
    /// this fails if the action returns a plausible file that was not trained.
    #[test]
    fn a_trained_adapter_blob_folds_into_the_base_and_changes_its_weights() {
        // `ChatSample::encode` terminates every sample with `data::chat::
        // ENDOFTEXT` (151643), so the checkpoint's vocab has to cover it -
        // `QwenConfig::tiny`'s own 23 would index outside the embedding.
        let cfg = QwenConfig { vocab: 151936, ..QwenConfig::tiny() };
        let dir = std::env::temp_dir().join(format!("qwen-caps-lora-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("tiny.safetensors");
        write_tiny_checkpoint(&base, &cfg, 3);
        write_byte_tokenizer(&dir);

        let mut reg = Registry::new();
        reg.register(Arc::new(QwenProvider::new()));
        let mut steps_seen = 0u32;
        let inv = Invocation::new()
            // Both host-resolved params are passed explicitly: an ambient
            // BRAIN_QWEN_* on the dev box must not decide what this runs on.
            .set("weights", json!(base.to_str().unwrap()))
            .set("tokenizer", json!(dir.join("tokenizer.json").to_str().unwrap()))
            .set("rank", json!(4))
            .set("steps", json!(8))
            .set("lr", json!(1e-2))
            .set("batch", json!(2))
            .set("block", json!(12))
            .blob("dataset", Blob::new(Media::Bytes, dataset_jsonl()));
        let out = reg.run(MODEL, "lora_train", inv, &mut |_p| steps_seen += 1).unwrap();
        assert!(steps_seen > 0, "lora_train must emit progress");
        assert_eq!(out.outputs["rank"].as_u64(), Some(4));
        assert_eq!(out.outputs["steps"].as_u64(), Some(8));

        let adapter = out.blobs.get("adapter").expect("adapter blob");
        assert_eq!(adapter.media, Media::Bytes);
        let adapter_path = dir.join("adapter.safetensors");
        std::fs::write(&adapter_path, &adapter.bytes).unwrap();

        let base_w = checkpoint::load(base.to_str().unwrap()).by_role("");
        let mut folded = base_w.clone();
        let lora = crate::lora::fold_adapter_into(&mut folded, adapter_path.to_str().unwrap()).unwrap();
        assert_eq!(lora.rank, 4);
        let wq = "blocks.0.attn.wq.weight";
        assert_ne!(folded[wq], base_w[wq], "a trained adapter must move a targeted projection");
        assert_eq!(folded["tok.weight"], base_w["tok.weight"], "the embedding is not a LoRA target and must be untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The context capacity both arms are built at, and the completion budget
    /// the fixture decodes with - the same rule [`gate_lora`] applies, so the
    /// expectations harvested below describe the models the ACTION builds, not
    /// two differently-sized ones.
    const GATE_MAX_NEW: usize = 4;
    const GATE_CAP: u32 = 64;

    /// A deliberately destructive rank-2 LoRA adapter over the projections
    /// that WRITE the residual stream - each layer's attention output and its
    /// whole MLP - written in the on-disk form `lora::fold_adapter_into`
    /// reads. Not trained: the point of the fixture is a candidate whose
    /// behaviour is wholesale different from the incumbent's, which is what an
    /// adapter that has catastrophically forgotten looks like from the gate's
    /// side. The factors come from `data::rng::Lcg` (the sanctioned test PRNG)
    /// at a magnitude far above `init_weights`' own, so the fold cannot be a
    /// numerical no-op.
    ///
    /// Targeting `wq` alone was measured NOT to be enough: at this model's
    /// size both arms still decoded the same fixed-point token for every
    /// prompt, because a perturbed query projection reaches the tied head only
    /// through an attention softmax that a tiny random model has already
    /// saturated. The residual writers reach it directly.
    fn write_destructive_adapter(path: &Path, cfg: &QwenConfig, base: &Path) {
        let w = checkpoint::load(base.to_str().unwrap()).by_role("");
        let mut lcg = data::rng::Lcg::new(9);
        let rank = 2usize;
        let (d, ff, hq) = (cfg.d_model as usize, cfg.d_ff as usize, (cfg.n_heads * cfg.head_dim) as usize);
        // `(leaf, out, inn)` exactly as `QwenConfig::param_list` lays each
        // linear out - asserted against the checkpoint below rather than
        // assumed, since a wrong `inn` would silently fold a transposed delta.
        let targets = [("attn.wo.weight", d, hq), ("mlp.gate.weight", ff, d), ("mlp.up.weight", ff, d), ("mlp.down.weight", d, ff)];
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        for l in 0..cfg.n_layers {
            for (leaf, out, inn) in targets {
                let name = format!("blocks.{l}.{leaf}");
                let base_len = w.get(&name).unwrap_or_else(|| panic!("base has no {name}")).len();
                assert_eq!(base_len, out * inn, "{name} is {base_len} floats, not [{out}, {inn}]");
                tensors.push((format!("{name}.lora_a"), vec![(rank * inn) as u64], lcg.vec_scaled(rank * inn, 1.0)));
                tensors.push((format!("{name}.lora_b"), vec![(out * rank) as u64], lcg.vec_scaled(out * rank, 1.0)));
            }
        }
        let mut card = checkpoint::st::ModelCard::new("test/candidate:brain:lora:latest", "qwen");
        card.variant_of = Some("test/base".to_string());
        card.adapter = Some(checkpoint::st::Adapter {
            kind: "lora".to_string(),
            rank: Some(rank as u32),
            base: Some("test/base".to_string()),
            alpha: Some(rank as f32 * 2.0),
            targets: Some(vec!["wo".to_string(), "gate".to_string(), "up".to_string(), "down".to_string()]),
            dataset_id: None,
            per_target: None,
        });
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &json!({"rank": rank, "alpha": rank * 2}), Some(&card)).unwrap();
    }

    /// Greedily decode each question and keep the FIRST LINE, which is what
    /// `promote::document::DocumentVerifier` scores. Deliberately built from
    /// `crate::sample::generate_kv`'s public API rather than from
    /// [`score_arm`]'s internals: the fixture re-derives what each arm does
    /// independently, so if `lora_gate` ever stopped decoding greedily the
    /// expectations would stop matching and this test would fail.
    fn decode_first_lines(model: &Qwen, tok: &QwenBpe, questions: &[String]) -> Vec<String> {
        questions
            .iter()
            .map(|q| {
                let mut rng = Rng::new(0);
                let ids = crate::sample::generate_kv(model, &tok.encode(q), GATE_MAX_NEW, 0.0, 0, 1.0, None, &mut rng);
                tok.decode(&ids).lines().next().unwrap_or("").to_string()
            })
            .collect()
    }

    fn probe_jsonl(rows: &[(String, String, String)]) -> Vec<u8> {
        let mut out = String::new();
        for (fact, question, answer) in rows {
            out.push_str(&json!({"fact": fact, "probe_question": question, "expected_answer": answer}).to_string());
            out.push('\n');
        }
        out.into_bytes()
    }

    /// `continuous-learning` B3b, the round trip the milestone is specified by:
    /// a candidate adapter that destroys the anchor suite must come back
    /// `reject: AnchorRegressed`, never `promote`.
    ///
    /// The fixture labels each suite with the arm whose behaviour it is meant
    /// to describe, which is what makes this a real round trip rather than a
    /// mock: the frozen probes expect what the CANDIDATE decodes (so the
    /// candidate wins the primary suite outright - the sign test and the
    /// effect-size bar both clear, and the rejection cannot be dismissed as
    /// noise), and the anchor suite expects what the INCUMBENT decodes (a
    /// retention suite is exactly "the behaviours already in production"). The
    /// candidate destroys every one of them, which is the catastrophic
    /// forgetting `Cause::AnchorRegressed` exists to catch, and the only bar
    /// of the four that can catch it.
    #[test]
    fn an_anchor_destroying_candidate_adapter_is_rejected_as_anchor_regressed() {
        // vocab 96 covers exactly `write_byte_tokenizer`'s id range and
        // nothing else: no chat template and no `ENDOFTEXT` are involved
        // here, so the model stays tiny instead of carrying a 151936-row
        // embedding this test would then softmax over twice per task.
        let cfg = QwenConfig { vocab: 96, ..QwenConfig::tiny() };
        let dir = std::env::temp_dir().join(format!("qwen-caps-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("tiny.safetensors");
        write_tiny_checkpoint(&base, &cfg, 5);
        write_byte_tokenizer(&dir);
        let tok_path = dir.join("tokenizer.json");
        let tok = QwenBpe::from_file(tok_path.to_str().unwrap()).unwrap();
        let adapter = dir.join("candidate.safetensors");
        write_destructive_adapter(&adapter, &cfg, &base);

        let probe_q: Vec<String> = (0..MIN_HELD_OUT_PROBES).map(|i| format!("probe {i} reads")).collect();
        let anchor_q: Vec<String> = (0..6).map(|j| format!("anchor {j} holds")).collect();
        for q in probe_q.iter().chain(anchor_q.iter()) {
            assert!(tok.encode(q).len() + GATE_MAX_NEW <= GATE_CAP as usize, "fixture prompt {q:?} must fit the built context");
        }

        // Both arms, decoded exactly as the action decodes them: the incumbent
        // straight off the base checkpoint, the candidate off the base with
        // the adapter folded in - which is what a caller would actually serve.
        let incumbent = Qwen::load_inference(base.to_str().unwrap(), 1, GATE_CAP);
        let inc_probe = decode_first_lines(&incumbent, &tok, &probe_q);
        let inc_anchor = decode_first_lines(&incumbent, &tok, &anchor_q);
        drop(incumbent);
        let mut folded = checkpoint::load(base.to_str().unwrap()).by_role("");
        crate::lora::fold_adapter_into(&mut folded, adapter.to_str().unwrap()).unwrap();
        let candidate = Qwen::new_shard(cfg.clone(), 1, GATE_CAP, &folded, false, crate::model::Shard::whole(cfg.n_layers as usize));
        let cand_probe = decode_first_lines(&candidate, &tok, &probe_q);
        let cand_anchor = decode_first_lines(&candidate, &tok, &anchor_q);
        drop(candidate);

        // Preconditions of the fixture, asserted rather than assumed: a
        // candidate that decoded like the incumbent would make every pair
        // concordant and the gate would (correctly) reject it as
        // NotSignificant, proving nothing about the anchor bar.
        assert_ne!(cand_probe, inc_probe, "the destructive adapter must actually change what the model decodes");
        assert_ne!(cand_anchor, inc_anchor, "the destructive adapter must actually change the anchored behaviours");
        for a in cand_probe.iter().chain(inc_anchor.iter()) {
            assert!(!promote::document::normalize(a).is_empty(), "an expected answer that normalises to nothing cannot be a probe: {a:?}");
        }

        let probes = probe_jsonl(
            &(0..MIN_HELD_OUT_PROBES)
                .map(|i| (format!("record {i} states the reading is {}", 10 + i), probe_q[i].clone(), cand_probe[i].clone()))
                .collect::<Vec<_>>(),
        );
        let anchors = probe_jsonl(
            &(0..anchor_q.len())
                .map(|j| (format!("retained behaviour {j} must survive training"), anchor_q[j].clone(), inc_anchor[j].clone()))
                .collect::<Vec<_>>(),
        );

        let mut reg = Registry::new();
        reg.register(Arc::new(QwenProvider::new()));
        let mut stages = 0u32;
        let inv = Invocation::new()
            .set("weights", json!(base.to_str().unwrap()))
            .set("tokenizer", json!(tok_path.to_str().unwrap()))
            .set("max_new", json!(GATE_MAX_NEW as i64))
            .blob("adapter", Blob::new(Media::Bytes, std::fs::read(&adapter).unwrap()))
            .blob("probes", Blob::new(Media::Bytes, probes))
            .blob("anchor", Blob::new(Media::Bytes, anchors));
        let out = reg.run(MODEL, "lora_gate", inv, &mut |_p| stages += 1).unwrap();
        assert!(stages > 0, "lora_gate must emit progress");

        assert_eq!(out.outputs["promote"], json!(false));
        assert_eq!(out.outputs["cause"], json!("AnchorRegressed"));
        let decision = out.outputs["decision"].as_str().unwrap();
        assert!(decision.starts_with("reject: AnchorRegressed"), "got {decision:?}");
        // The rejection is about the ANCHOR, not about noise on the primary
        // suite: the first two bars are checked before it and both cleared.
        assert!(out.outputs["p_value"].as_f64().unwrap() <= 0.05, "the sign test must have cleared alpha: {out:?}");
        assert!(out.outputs["effect_size"].as_f64().unwrap() >= 0.15, "the effect-size bar must have cleared: {out:?}");
        assert!(out.outputs["anchor_delta"].as_f64().unwrap() > 0.02, "the anchor suite must have regressed past its budget");

        let report: serde_json::Value = serde_json::from_slice(&out.blobs["report"].bytes).unwrap();
        assert_eq!(report["cause"], json!("AnchorRegressed"));
        assert_eq!(report["probes"], json!(MIN_HELD_OUT_PROBES));
        assert_eq!(report["anchor_probes"], json!(anchor_q.len()));
        // The pre-registered bars travel WITH the number they judged.
        assert_eq!(report["gate_config"]["min_effect_size"], json!(0.15));
        // Every fact landed on the candidate arm (its own probes are what the
        // candidate answers), so the per-fact rows must say so - a rejected
        // cycle still reports which facts did and did not take.
        let facts = report["facts"].as_array().unwrap();
        assert_eq!(facts.len(), MIN_HELD_OUT_PROBES);
        assert!(facts.iter().all(|f| f["landed"] == json!(true)), "every fact's own probe was answered by the candidate arm");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_declares_generate() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        // The full action list is `manifest_lists_generate_and_lora_train`'s
        // assertion; this one is about `generate`'s own shape.
        let g = &m.actions[0];
        assert_eq!(g.name, "generate");
        assert!(g.streaming, "generate must stream (one Progress per token)");
        assert!(g.params.iter().any(|p| p.name == "weights" && p.required));
        // `prompt` is NOT required: `messages` (the shared chat-serving parse)
        // can supply the request instead, matching `resident_llm`'s spec.
        assert!(g.params.iter().any(|p| p.name == "prompt" && !p.required));
        assert!(g.params.iter().any(|p| p.name == "messages"));
        assert_eq!(g.params.iter().find(|p| p.name == "max_new").unwrap().default, Some(json!(DEFAULT_MAX_NEW)));
        assert_eq!(g.outputs[0].media, Media::Text);
        // validation: defaults fill, missing required rejected, no weights loaded.
        let inv = g.validate(Invocation::new().set("weights", json!("w")).set("prompt", json!("1 2"))).unwrap();
        assert_eq!(inv.get_i64("max_new"), Some(DEFAULT_MAX_NEW));
        assert!(g.validate(Invocation::new().set("prompt", json!("1"))).is_err());
        assert!(g.validate(Invocation::new().set("weights", json!("w")).set("prompt", json!("1")).set("bogus", json!(1))).is_err());
        // the manifest round-trips to JSON for discovery.
        assert_eq!(manifest().to_json()["actions"][0]["name"], "generate");
    }

    #[test]
    fn sampling_params_carry_ui_ranges() {
        let g = &manifest().actions[0];
        let p = |name: &str| g.params.iter().find(|p| p.name == name).unwrap();
        assert_eq!((p("max_new").min, p("max_new").max, p("max_new").step), (Some(1.0), Some(32768.0), Some(1.0)));
        assert_eq!((p("temp").min, p("temp").max, p("temp").step), (Some(0.0), Some(2.0), Some(0.01)));
        assert_eq!((p("top_k").min, p("top_k").max, p("top_k").step), (Some(0.0), Some(1000.0), Some(1.0)));
        assert_eq!((p("top_p").min, p("top_p").max, p("top_p").step), (Some(0.0), Some(1.0), Some(0.01)));
    }

    #[test]
    fn missing_weights_is_a_clean_error() {
        let reg = {
            let mut r = Registry::new();
            r.register(Arc::new(QwenProvider::new()));
            r
        };
        let err = reg
            .run(MODEL, "generate", Invocation::new().set("weights", json!("/nonexistent/qwen.safetensors")).set("prompt", json!("1 2")), &mut |_| {})
            .unwrap_err();
        assert!(err.contains("not found"), "got: {err}");
    }

    /// End-to-end on a tiny synthetic checkpoint: save `QwenConfig::tiny` +
    /// `init_weights` to disk, then drive `generate` through the Registry and
    /// assert one Progress per generated token and ids/text outputs.
    #[test]
    fn tiny_checkpoint_generates_with_per_token_progress() {
        let cfg = QwenConfig::tiny();
        let init = crate::init::init_weights(&cfg, 7);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| {
                let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
                (name, vec![n as u64], v)
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("qwen-caps-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);

        let mut reg = Registry::new();
        reg.register(Arc::new(QwenProvider::new()));
        let mut steps = 0u32;
        let inv = Invocation::new()
            .set("weights", json!(path.to_str().unwrap()))
            .set("prompt", json!("1 5 3"))
            .set("max_new", json!(4));
        let out = reg.run(MODEL, "generate", inv, &mut |_p| steps += 1).unwrap();
        let n = out.outputs["tokens"].as_u64().unwrap();
        assert_eq!(n, 4, "greedy decode with no eos must emit max_new tokens");
        assert_eq!(steps as u64, n, "one Progress per generated token");
        assert_eq!(out.outputs["ids"].as_array().unwrap().len() as u64, n);
        // ids are within the tiny vocab; the text blob is their rendering.
        for v in out.outputs["ids"].as_array().unwrap() {
            assert!(v.as_u64().unwrap() < cfg.vocab as u64);
        }
        let text = String::from_utf8(out.blobs["text"].bytes.clone()).unwrap();
        assert_eq!(text.split_whitespace().count() as u64, n);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REGRESSION: with a tokenizer, `generate` must go through the SAME
    /// chat-serving parse (`crate::chat::{parse_request, SeqState}`) the
    /// HTTP/D-Bus path runs — a `messages` + `tools` request must validate,
    /// stream, and resolve a `finish_reason` exactly like the old
    /// hand-rolled `apply_chat_template`-only implementation this replaces,
    /// but now with tool-call/stop-string/cancellation parity it never had.
    ///
    /// Needs a real tokenizer (`QWEN_TOKENIZER=/path/to/tokenizer.json`) --
    /// self-skips loudly when unset. The checkpoint's
    /// vocab is sized to the real tokenizer's full range so the rendered
    /// chat-template special tokens (`<|im_start|>` etc., ids up in the
    /// 151000s) never index outside the embedding table.
    #[test]
    fn tokenizer_present_runs_the_shared_chat_parse_with_tools() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            brain_testutil::skip("set QWEN_TOKENIZER to a real tokenizer.json to run this test");
            return;
        };

        let cfg = QwenConfig { vocab: 151936, ..QwenConfig::tiny() };
        let init = crate::init::init_weights(&cfg, 11);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| {
                let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
                (name, vec![n as u64], v)
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("qwen-caps-chat-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);

        let mut reg = Registry::new();
        reg.register(Arc::new(QwenProvider::new()));
        let mut events = 0u32;
        let messages = json!([{"role": "user", "content": "what's the weather in Paris?"}]).to_string();
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]).to_string();
        let inv = Invocation::new()
            .set("weights", json!(path.to_str().unwrap()))
            .set("tokenizer", json!(tok_path))
            .set("messages", json!(messages))
            .set("tools", json!(tools))
            .set("max_new", json!(4));
        let out = reg.run(MODEL, "generate", inv, &mut |_p| events += 1).unwrap();
        assert!(out.outputs.get("finish_reason").is_some(), "shared SeqState::finish must report a finish_reason");
        assert!(out.outputs.get("prompt_tokens").is_some());
        assert!(out.outputs.get("completion_tokens").is_some());
        assert!(events > 0, "must stream at least the final 'done' Progress");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
