// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3's capabilities behind the generalized [`capability`] interface — what
//! makes `brain caps qwen` / `brain do qwen generate …` (and the perf suite's
//! `CapabilityTarget`) work with no Qwen-specific plumbing in the CLI.
//!
//! Two actions.
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

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::rng::Rng;
use data::tokenizer::Tokenizer;
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

    Manifest::new(
        MODEL,
        "Qwen3 dense decoder - autoregressive text generation with per-token streaming, plus LoRA fine-tuning on a chat dataset.",
        vec![generate, lora_train],
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
    // Unique per call: two concurrent training requests must not share a
    // scratch tree, and the pid alone does not separate them.
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let scratch = std::env::temp_dir().join(format!("brain-qwen3-lora-train-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| format!("qwen lora_train: {}: {e}", scratch.display()))?;
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

    /// `continuous-learning` B3a: qwen3's LoRA training loop was CLI-only
    /// (`crates/cli/src/qwen_cli.rs::finetune_lora`), so nothing that reads a
    /// manifest - `brain caps`, the event API, D-Bus, whale's node-type
    /// generation - could see that this model can be trained at all. The
    /// manifest is the only place that fact can live.
    #[test]
    fn manifest_lists_generate_and_lora_train() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<_> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["generate", "lora_train"]);
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
