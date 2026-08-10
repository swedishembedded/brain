// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3's capabilities behind the generalized [`capability`] interface — what
//! makes `brain caps qwen` / `brain do qwen generate …` (and the perf suite's
//! `CapabilityTarget`) work with no Qwen-specific plumbing in the CLI.
//!
//! One action, `generate`: the same one-shot decode path `brain qwen infer`
//! runs (`Qwen::load_inference` + the KV-cache [`crate::sample`] loop), with a
//! `Progress` emitted **per generated token** so a streaming harness gets a
//! real TTFT/ITL timeline. The manifest is static (no weights needed); the
//! model loads lazily on the first run and stays resident across calls (keyed
//! by weights path + context capacity), mirroring `zimage::caps`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::qwen_tokenizer::QwenBpe;
use data::rng::Rng;
use data::tokenizer::Tokenizer;
use serde_json::json;

use crate::chat::{parse_request, ParsedRequest, SeqState};
use crate::model::Qwen;

/// The model id used on the CLI (`brain do qwen …`) and the event API.
pub const MODEL: &str = "brain/qwen";

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let generate = ActionSpec::new("generate", "generate tokens continuing a prompt (KV-cache decode, one Progress per token)")
        .streaming()
        .param(ParamSpec::new("weights", ParamType::Str, "path to a brain-format Qwen checkpoint (.safetensors)").required())
        .param(ParamSpec::new(
            "prompt",
            ParamType::Str,
            "the prompt: text (with a tokenizer) or whitespace/comma-separated token ids (without); ignored when `messages` is set",
        ))
        .param(ParamSpec::new("tokenizer", ParamType::Str, "path to tokenizer.json; omit to feed/return raw token ids"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(32)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.0)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 or negative = disabled)").default(json!(40)))
        .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)))
        .param(
            ParamSpec::new("precision", ParamType::Str, "model precision: fp32, or int8 (per-channel weights + dynamic activation quant)")
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
        .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (accepted, ignored)"))
        .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block (needs a tokenizer)").default(json!(true)))
        .output(BlobSpec::new("text", Media::Text, "the generated text (space-separated token ids when no tokenizer is given)"));
    Manifest::new(MODEL, "Qwen3 dense decoder — autoregressive text generation with per-token streaming.", vec![generate])
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
        (name == "generate").then(|| Arc::new(GenerateAction { hot: self.hot.clone() }) as Arc<dyn Action>)
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
                let max_new = inv.get_i64("max_new").unwrap_or(32).max(0) as usize;
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

    #[test]
    fn manifest_declares_generate() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        let g = &m.actions[0];
        assert_eq!(g.name, "generate");
        assert!(g.streaming, "generate must stream (one Progress per token)");
        assert!(g.params.iter().any(|p| p.name == "weights" && p.required));
        // `prompt` is NOT required: `messages` (the shared chat-serving parse)
        // can supply the request instead, matching `resident_llm`'s spec.
        assert!(g.params.iter().any(|p| p.name == "prompt" && !p.required));
        assert!(g.params.iter().any(|p| p.name == "messages"));
        assert_eq!(g.params.iter().find(|p| p.name == "max_new").unwrap().default, Some(json!(32)));
        assert_eq!(g.outputs[0].media, Media::Text);
        // validation: defaults fill, missing required rejected, no weights loaded.
        let inv = g.validate(Invocation::new().set("weights", json!("w")).set("prompt", json!("1 2"))).unwrap();
        assert_eq!(inv.get_i64("max_new"), Some(32));
        assert!(g.validate(Invocation::new().set("prompt", json!("1"))).is_err());
        assert!(g.validate(Invocation::new().set("weights", json!("w")).set("prompt", json!("1")).set("bogus", json!(1))).is_err());
        // the manifest round-trips to JSON for discovery.
        assert_eq!(manifest().to_json()["actions"][0]["name"], "generate");
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
    /// self-skips loudly when unset, per `docs/testing.md`. The checkpoint's
    /// vocab is sized to the real tokenizer's full range so the rendered
    /// chat-template special tokens (`<|im_start|>` etc., ids up in the
    /// 151000s) never index outside the embedding table.
    #[test]
    fn tokenizer_present_runs_the_shared_chat_parse_with_tools() {
        let Ok(tok_path) = std::env::var("QWEN_TOKENIZER") else {
            eprintln!("SKIP: set QWEN_TOKENIZER to a real tokenizer.json to run this test");
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
