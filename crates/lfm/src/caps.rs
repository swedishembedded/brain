// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder capabilities behind the generalized [`capability`] interface
//! — what makes `brain caps lfm` / `brain do lfm fill_mask …` (and the perf
//! suite's `CapabilityTarget`) work with no LFM-specific plumbing in the CLI.
//!
//! Two one-shot actions (deliberately NOT `.streaming()` — an encoder emits a
//! single artifact, so `ttfa == e2e` and `tpoa` stays null in perf terms):
//! - `fill_mask`: MLM top-k at every `<|mask|>` position.
//! - `embed`: per-token hidden states (LE-f32 bytes + shape meta, per the
//!   `events::bytes` convention) plus a mean-pooled sequence embedding.
//!
//! Both run the chunked long-context path (bounded attention slab), so 8k-token
//! inputs work on any backend. The model stays resident across calls, keyed by
//! weights path + exact context length (bidirectional attention makes unmasked
//! padding unsound — see the note at the build site); the tokenizer is cached
//! alongside.

use std::path::Path;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use serde_json::json;

use crate::model::Lfm;

/// The model id used on the CLI (`brain do lfm …`) and the event API.
pub const MODEL: &str = "lfm";

/// Attention-slab budget for the chunked path (chunk 2048 at T=8192, H=16).
const SLAB_BUDGET: u64 = 512 << 20;
/// Fill-mask probe capacity the resident model is built with.
const PROBE_CAP: u32 = 64;

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor): the
/// checkpoint + tokenizer are service-side configuration (`BRAIN_LFM*` env),
/// so the actions carry only request parameters.
pub fn manifest_resident() -> Manifest {
    let mut m = manifest();
    for a in &mut m.actions {
        a.params.retain(|p| p.name != "weights" && p.name != "tokenizer");
    }
    m
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let common = |a: ActionSpec| {
        a.param(ParamSpec::new("weights", ParamType::Str, "path to a brain-format LFM2.5-Encoder checkpoint (.safetensors)").required())
            .param(ParamSpec::new("tokenizer", ParamType::Str, "path to the checkpoint's tokenizer.json").required())
            .param(ParamSpec::new("text", ParamType::Str, "input text; falls back to the 'text' input blob for long documents"))
            .input(BlobSpec::new("text", Media::Text, "input document (used when the 'text' param is absent)"))
    };
    let fill_mask = common(ActionSpec::new("fill_mask", "MLM top-k predictions at every <|mask|> position"))
        .param(ParamSpec::new("topk", ParamType::Int, "predictions per mask position").default(json!(5)))
        .output(BlobSpec::new("predictions", Media::Text, "JSON: [{row, tokens: [{id, token, logit}]}]"));
    let embed = common(ActionSpec::new("embed", "bidirectional encoding: per-token hidden states + mean-pooled embedding"))
        .param(ParamSpec::new("max_tokens", ParamType::Int, "truncate the input to this many tokens (0 = no limit)").default(json!(0)))
        .output(BlobSpec::new("embeddings", Media::Bytes, "LE-f32 [n_tokens, dim] hidden states; shape in blob meta"));
    Manifest::new(MODEL, "LFM2.5 bidirectional encoder — fill-mask and long-context embeddings (8k).", vec![fill_mask, embed])
}

/// The resident (hot) model + tokenizer and the key that fixes them.
struct Hot {
    weights: String,
    tokenizer_path: String,
    cap: u32,
    tok: QwenBpe,
    model: Lfm,
}

/// The executable LFM model behind the manifest. Construction is free — the
/// checkpoint loads lazily on the first run and stays resident.
#[derive(Default)]
pub struct LfmProvider {
    hot: Arc<Mutex<Option<Hot>>>,
}

impl LfmProvider {
    pub fn new() -> LfmProvider {
        LfmProvider::default()
    }
}

impl Provider for LfmProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "fill_mask" | "embed" => Some(Arc::new(EncoderAction { name: name.to_string(), hot: self.hot.clone() }) as Arc<dyn Action>),
            _ => None,
        }
    }
}

struct EncoderAction {
    name: String,
    hot: Arc<Mutex<Option<Hot>>>,
}

/// Input text: the `text` param, else the `text` blob (long documents).
fn input_text(inv: &Invocation) -> Result<String, String> {
    if let Some(t) = inv.get_str("text").filter(|t| !t.is_empty()) {
        return Ok(t);
    }
    if let Some(b) = inv.blobs.get("text") {
        return String::from_utf8(b.bytes.clone()).map_err(|e| format!("lfm: text blob is not UTF-8: {e}"));
    }
    Err("lfm: provide the 'text' param or a 'text' input blob".to_string())
}

impl Action for EncoderAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = inv.get_str("weights").ok_or("lfm: missing required param 'weights'")?;
        if !Path::new(&weights).exists() {
            return Err(format!("lfm: weights not found at '{weights}'"));
        }
        let tokenizer_path = inv.get_str("tokenizer").ok_or("lfm: missing required param 'tokenizer'")?;
        let text = input_text(inv)?;

        // Tokenize outside the lock-heavy section (needs the cached tokenizer,
        // so take the lock once and do everything hot under it — the encoder is
        // one-shot and the executor batches concurrency above this level).
        let mut guard = self.hot.lock().map_err(|_| "lfm: hot model lock poisoned")?;
        let tok_reusable = matches!(&*guard, Some(h) if h.tokenizer_path == tokenizer_path);
        if !tok_reusable {
            *guard = None;
        }
        let tok_tmp;
        let tok: &QwenBpe = match &*guard {
            Some(h) => &h.tok,
            None => {
                tok_tmp = QwenBpe::from_file(&tokenizer_path)?;
                &tok_tmp
            }
        };

        let mut ids: Vec<u32> = tok.template_prefix().to_vec();
        ids.extend(tok.encode(&text));
        let max_tokens = inv.get_i64("max_tokens").unwrap_or(0);
        if self.name == "embed" && max_tokens > 0 {
            ids.truncate(max_tokens as usize);
        }
        if ids.is_empty() {
            return Err("lfm: empty input".to_string());
        }
        // The graph is built at the EXACT request length: bidirectional
        // attention means pad tokens would attend into (and be attended by)
        // every real token, corrupting the encoding — padding is only sound
        // with zeroed pad states + an additive key mask, which lands with the
        // batched-serving phase. Rebuilds cost one weight upload when the
        // length changes; fixed-length callers (perf) build once.
        let need = ids.len() as u32;

        let reuse = matches!(&*guard, Some(h) if h.weights == weights && h.cap == need);
        if !reuse {
            let old_tok = guard.take().map(|h| h.tok);
            let tok = match old_tok {
                Some(t) if tok_reusable => t,
                _ => QwenBpe::from_file(&tokenizer_path)?,
            };
            let model = Lfm::load_inference_chunked(&weights, 1, need, SLAB_BUDGET, PROBE_CAP);
            *guard = Some(Hot { weights: weights.clone(), tokenizer_path: tokenizer_path.clone(), cap: need, tok, model });
        }
        let hot = guard.as_ref().expect("hot model present");
        let (model, tok) = (&hot.model, &hot.tok);
        let cap = hot.cap;

        let n = ids.len();
        debug_assert_eq!(n as u32, cap);
        model.set_tokens(&ids);
        progress(Progress { step: 1, total: 2, message: "encode".to_string() });

        match self.name.as_str() {
            "fill_mask" => {
                let mask_id = tok.special_id("<|mask|>").ok_or("lfm: tokenizer has no <|mask|>")?;
                let rows: Vec<u32> = ids.iter().enumerate().filter_map(|(i, &t)| (t == mask_id).then_some(i as u32)).collect();
                if rows.is_empty() {
                    return Err("lfm fill_mask: no <|mask|> in the input".to_string());
                }
                if rows.len() as u32 > PROBE_CAP {
                    return Err(format!("lfm fill_mask: {} masks > capacity {PROBE_CAP}", rows.len()));
                }
                model.set_probe_rows(&rows);
                model.forward();
                let logits = model.read_probe_logits();
                let topk = inv.get_i64("topk").unwrap_or(5).clamp(1, 64) as usize;
                let v = model.cfg.vocab as usize;
                let results: Vec<serde_json::Value> = rows
                    .iter()
                    .enumerate()
                    .map(|(i, &row)| {
                        let lrow = &logits[i * v..(i + 1) * v];
                        let mut idx: Vec<u32> = (0..v as u32).collect();
                        idx.sort_unstable_by(|&x, &y| lrow[y as usize].total_cmp(&lrow[x as usize]));
                        let toks: Vec<serde_json::Value> = idx[..topk]
                            .iter()
                            .map(|&id| json!({"id": id, "token": tok.decode(&[id]), "logit": lrow[id as usize]}))
                            .collect();
                        json!({"row": row, "tokens": toks})
                    })
                    .collect();
                let payload = serde_json::to_vec(&results).map_err(|e| e.to_string())?;
                Ok(Outcome::new()
                    .set("masks", json!(rows.len()))
                    .set("predictions", json!(results))
                    .blob("predictions", Blob::new(Media::Text, payload)))
            }
            "embed" => {
                model.forward();
                let d = model.cfg.d_model as usize;
                let hidden = &model.read_hidden()[..n * d];
                let mut mean = vec![0.0f32; d];
                for row in hidden.chunks_exact(d) {
                    for (m, &x) in mean.iter_mut().zip(row) {
                        *m += x;
                    }
                }
                for x in &mut mean {
                    *x /= n as f32;
                }
                let bytes: Vec<u8> = hidden.iter().flat_map(|f| f.to_le_bytes()).collect();
                let blob = Blob::new(Media::Bytes, bytes).with_meta(json!({"shape": [n, d], "dtype": "f32le"}));
                Ok(Outcome::new()
                    .set("tokens", json!(n))
                    .set("dim", json!(d))
                    .set("mean", json!(mean))
                    .blob("embeddings", blob))
            }
            other => Err(format!("lfm: unknown action {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn manifest_declares_one_shot_encoder_actions() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["fill_mask", "embed"]);
        for a in &m.actions {
            assert!(!a.streaming, "{}: encoder actions are one-shot (ttfa == e2e)", a.name);
            assert!(a.params.iter().any(|p| p.name == "weights" && p.required));
            assert!(a.params.iter().any(|p| p.name == "tokenizer" && p.required));
        }
        let fm = &m.actions[0];
        assert_eq!(fm.params.iter().find(|p| p.name == "topk").unwrap().default, Some(json!(5)));
        let inv = fm
            .validate(Invocation::new().set("weights", json!("w")).set("tokenizer", json!("t")).set("text", json!("x")))
            .unwrap();
        assert_eq!(inv.get_i64("topk"), Some(5));
        assert_eq!(manifest().to_json()["actions"][1]["name"], "embed");
    }

    #[test]
    fn missing_weights_is_a_clean_error() {
        let mut r = Registry::new();
        r.register(Arc::new(LfmProvider::new()));
        let err = r
            .run(
                MODEL,
                "embed",
                Invocation::new().set("weights", json!("/nonexistent/lfm.safetensors")).set("tokenizer", json!("t")).set("text", json!("hi")),
                &mut |_| {},
            )
            .unwrap_err();
        assert!(err.contains("not found"), "got: {err}");
    }
}
