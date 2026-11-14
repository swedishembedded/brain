// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The measuring actions of `brain/qwen3`: `eval` (held-out loss and token
//! accuracy of a checkpoint, and of a named adapter beside it), `calib` (the
//! INT8 KV-cache outlier report and clip table), and the tool-call pair
//! `toolcall_gen` (a masked fine-tuning set) and `toolcall_eval` (exact match
//! of a checkpoint on held-out cases).
//!
//! Swedish Embedded AB implements evaluation harnesses that tell its clients
//! whether a fine-tune learned their data. If your team needs expertise in
//! held-out evaluation and KV-cache quantization then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use serde_json::json;

use crate::config::QwenConfig;
use crate::eval::{score_chat, score_chat_paged, ChatScore, KvMode};
use crate::served::{count, guarded, path_param, Resolved, StoredAdapter};

/// Largest sample count a generated or scored tool-call set may ask for.
const MAX_CASES: f64 = 1_000_000.0;
/// Largest vocabulary the tool-call set is rendered for.
const MAX_VOCAB: f64 = 4_194_304.0;
/// Most candidate tools one case presents.
const MAX_TOOLS: f64 = 64.0;
/// Tokens per paged-cache block while calibrating.
const CALIB_BLOCK: u32 = 16;

fn role_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("weights", ParamType::Str, "the checkpoint (.safetensors, .gguf or a Hugging Face release); overrides the model store's own pick when set").host_resolved())
        .param(ParamSpec::new("tokenizer", ParamType::Str, "the tokenizer.json whose directory also supplies the chat template; omit to use the model store's own pick").host_resolved())
}

pub(crate) fn eval_spec() -> ActionSpec {
    role_params(ActionSpec::new(
        "eval",
        "teacher-forced held-out loss and token accuracy of the checkpoint on chat JSONL, and with `adapter_name` of that adapter folded in beside it",
    ))
    .param(ParamSpec::new("block", ParamType::Int, "scoring context, tokens; a sample longer than it is skipped and counted").default(json!(1024)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
    .param(ParamSpec::new("adapter_name", ParamType::Str, "a named adapter of this base in the model store, OWNER/NAME[:TAG]; scored beside the base with the verdict on whether it beats it"))
    .param(ParamSpec::new("kv", ParamType::Str, "also score through the paged serving engine at each of these KV representations, comma-separated: fp32, int8, int8-calib"))
    .input(BlobSpec::new("samples", Media::Bytes, "the held-out set: data::chat 'generic-messages-v2' JSONL, one sample per line").required())
    .output(BlobSpec::new("report", Media::Text, "the scores as lines of text"))
}

pub(crate) fn calib_spec() -> ActionSpec {
    role_params(ActionSpec::new(
        "calib",
        "measure the INT8 KV-cache outlier statistics of the checkpoint on chat JSONL: a per-stream report and the usable clip table",
    ))
    .param(ParamSpec::new("percentile", ParamType::Float, "the quantile of each stream's magnitudes the clip table keeps").default(json!(0.999)).min(0.0).max(1.0))
    .input(BlobSpec::new("samples", Media::Bytes, "the calibration prompts: data::chat 'generic-messages-v2' JSONL, one sample per line").required())
    .output(BlobSpec::new("report", Media::Text, "per (layer, K|V, kv head): absmax, p99, p99.99 and outlier ratio, most quantization-hostile first, as JSON"))
    .output(BlobSpec::new("clip", Media::Text, "the clip table (kv_calib.json) an int8-KV engine loads beside the checkpoint when asked to"))
}

pub(crate) fn toolcall_gen_spec() -> ActionSpec {
    ActionSpec::new("toolcall_gen", "write a masked tool-call fine-tuning set (train and held-out validation) rendered through the checkpoint's chat template")
        .from_scratch()
        .param(ParamSpec::new("tokenizer", ParamType::Str, "the tokenizer.json whose directory also supplies the chat template; omit to use the model store's own pick").host_resolved())
        .param(ParamSpec::new("out", ParamType::Str, "the dataset directory to write").host_resolved().required())
        .param(ParamSpec::new("n", ParamType::Int, "training cases").default(json!(400)).min(1.0).max(MAX_CASES))
        .param(ParamSpec::new("held", ParamType::Int, "held-out validation cases").default(json!(40)).min(1.0).max(MAX_CASES))
        .param(ParamSpec::new("tools", ParamType::Int, "candidate tools presented in each case").default(json!(3)).min(1.0).max(MAX_TOOLS))
        .param(ParamSpec::new("vocab", ParamType::Int, "the model's vocabulary size, recorded with the dataset").default(json!(151_936)).min(1.0).max(MAX_VOCAB))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the cases (omit for random; the outcome echoes the one used)").min(0.0))
}

pub(crate) fn toolcall_eval_spec() -> ActionSpec {
    role_params(ActionSpec::new("toolcall_eval", "score the checkpoint's exact-match and token accuracy on held-out tool-call cases (teacher-forced greedy)"))
        .param(ParamSpec::new("n", ParamType::Int, "held-out cases").default(json!(40)).min(1.0).max(MAX_CASES))
        .param(ParamSpec::new("tools", ParamType::Int, "candidate tools presented in each case").default(json!(3)).min(1.0).max(MAX_TOOLS))
        .param(ParamSpec::new("seq", ParamType::Int, "scoring context, tokens").default(json!(512)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the cases (omit for random; the outcome echoes the one used)").min(0.0))
}

/// The tokenizer and the chat template that ships beside it.
fn tokenizer_and_template(path: &Path, action: &str) -> Result<(QwenBpe, ChatTemplate), String> {
    let tok = QwenBpe::from_file(&path.to_string_lossy()).map_err(|e| format!("qwen {action}: tokenizer {}: {e}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let template = ChatTemplate::from_model_dir(dir).map_err(|e| format!("qwen {action}: chat template beside {}: {e}", path.display()))?;
    Ok((tok, template))
}

/// The input blob `name` as chat samples, parsed by the one strict reader of
/// the format (`ChatSample::from_jsonl`), which names the offending line.
fn samples_blob(inv: &Invocation, name: &str, action: &str) -> Result<Vec<ChatSample>, String> {
    let blob = inv.get_blob(name).ok_or_else(|| format!("qwen {action}: missing required input '{name}'"))?;
    let scratch = capability::scratch::scratch_dir(&format!("qwen3-{action}")).map_err(|e| format!("qwen {action}: a scratch directory: {e}"))?;
    let path = scratch.join(format!("{name}.jsonl"));
    let parsed = std::fs::write(&path, &blob.bytes).map_err(|e| format!("qwen {action}: staging '{name}': {e}")).and_then(|()| ChatSample::from_jsonl(&path).map_err(|e| format!("qwen {action}: '{name}': {e}")));
    let _ = std::fs::remove_dir_all(&scratch);
    let samples = parsed?;
    if samples.is_empty() {
        return Err(format!("qwen {action}: '{name}' holds no samples"));
    }
    Ok(samples)
}

fn percent(accuracy: Option<f64>) -> String {
    accuracy.map_or_else(|| "n/a".to_string(), |a| format!("{:.1}%", a * 100.0))
}

fn score_json(score: &ChatScore) -> serde_json::Value {
    json!({"loss": score.loss, "token_accuracy": score.token_accuracy, "samples": score.samples, "skipped": score.skipped, "positions": score.positions})
}

fn score_line(label: &str, score: &ChatScore, note: &str) -> String {
    format!("{label}: loss {:.4}{note}  token-acc {}  ({}/{} samples scored, {} positions)", score.loss, percent(score.token_accuracy), score.samples, score.samples + score.skipped, score.positions)
}

pub(crate) fn eval(resolved: &Resolved, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let weights = resolved.weights(inv, "eval")?;
    let samples = samples_blob(inv, "samples", "eval")?;
    let (tok, template) = tokenizer_and_template(&resolved.tokenizer(inv, &weights), "eval")?;
    let block = count(inv, "block", 1024).max(1);
    let kv_modes = match inv.get_str("kv").filter(|k| !k.is_empty()) {
        Some(spec) => spec.split(',').map(|m| KvMode::parse(m.trim()).map_err(|e| format!("qwen eval: kv: {e}"))).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let adapter = inv.get_str("adapter_name").filter(|a| !a.is_empty()).map(|spec| StoredAdapter::locate(&weights, &spec, "eval")).transpose()?;
    let adapter_path = match &adapter {
        Some(a) if a.path.is_file() => Some(a.path.to_string_lossy().into_owned()),
        Some(a) => return Err(format!("qwen eval: {} is not in the model store (looked for {})", a.reference, a.path.display())),
        None => None,
    };
    let base_id = crate::served::base_id(&weights);
    let steps = 1 + kv_modes.len() as u32 + u32::from(adapter.is_some());
    let mut lines = Vec::new();

    progress(Progress::step(1, steps, "scoring the base"));
    let base = guarded("eval", || score_chat(&weights, None, &tok, &template, &samples, block))?;
    lines.push(score_line(&format!("base {base_id}"), &base, ""));
    let mut out = Outcome::new().set("base", score_json(&base));

    // The paged engine is the one `brain serve` runs; fp32 is scored only when asked, as the paged-vs-legacy cross-check.
    let mut paged = Vec::new();
    for (i, kv) in kv_modes.iter().enumerate() {
        progress(Progress::step(2 + i as u32, steps, format!("scoring through the paged engine, kv {}", kv.label())));
        let score = guarded("eval", || score_chat_paged(&weights, None, &tok, &template, &samples, block, *kv))?;
        let delta = if base.loss.is_finite() { score.loss - base.loss } else { f32::NAN };
        lines.push(score_line(&format!("base {base_id} [kv={}]", kv.label()), &score, &format!(" ({delta:+.4} vs legacy fp32)")));
        paged.push(json!({"kv": kv.label(), "score": score_json(&score)}));
    }
    if !paged.is_empty() {
        out = out.set("kv", json!(paged));
    }

    if let (Some(adapter), Some(path)) = (&adapter, &adapter_path) {
        progress(Progress::step(steps, steps, "scoring the adapter"));
        let tuned = guarded("eval", || score_chat(&weights, Some(path), &tok, &template, &samples, block))?;
        lines.push(score_line(&adapter.reference, &tuned, ""));
        let beats = tuned.loss < base.loss;
        lines.push(format!("{} {} base on held-out loss ({:.4} vs {:.4})", adapter.reference, if beats { "beats" } else { "does NOT beat" }, tuned.loss, base.loss));
        out = out.set("adapter", json!({"reference": adapter.reference, "score": score_json(&tuned), "beats_base": beats}));
    }
    Ok(out.blob("report", Blob::new(Media::Text, lines.join("\n").into_bytes())))
}

pub(crate) fn calib(resolved: &Resolved, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let weights = resolved.weights(inv, "calib")?;
    let percentile = inv.get_f64("percentile").map_or(0.999, |p| p as f32);
    if !(0.0..=1.0).contains(&percentile) {
        return Err(format!("qwen calib: percentile must be in [0,1], got {percentile}"));
    }
    let samples = samples_blob(inv, "samples", "calib")?;
    let (tok, template) = tokenizer_and_template(&resolved.tokenizer(inv, &weights), "calib")?;
    let cfg = QwenConfig::from_json_checked(&checkpoint::read_config(&weights)).map_err(|e| format!("qwen calib: {weights}: {e}"))?;

    // Keep the first few distinct encode errors: the template engine's diagnostics ("not prefix-stable ...
    // refusing to guess a loss-mask boundary") say why a sample was dropped.
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut encode_errors: Vec<String> = Vec::new();
    for s in &samples {
        match s.encode(&tok, &template) {
            Ok((ids, _mask)) if !ids.is_empty() => prompts.push(ids),
            Ok(_) => {}
            Err(e) => {
                let e = e.to_string();
                if encode_errors.len() < 3 && !encode_errors.contains(&e) {
                    encode_errors.push(e);
                }
            }
        }
    }
    if prompts.is_empty() {
        return Err(format!("qwen calib: no sample encoded to a usable prompt ({})", encode_errors.join("; ")));
    }
    let skipped = samples.len() - prompts.len();

    // Every prompt's table stays resident at once (the pool is read back once per layer after every prefill), so
    // the pool is sized for the sum of the prompts' blocks, not the longest.
    let max_len = prompts.iter().map(Vec::len).max().unwrap_or(1) as u32;
    let blocks_per_seq = max_len.div_ceil(CALIB_BLOCK).max(1);
    let total_blocks: u32 = prompts.iter().map(|p| (p.len() as u32).div_ceil(CALIB_BLOCK).max(1)).sum();
    let num_blocks = total_blocks + prompts.len() as u32;
    let max_batch = prompts.len().max(1) as u32;

    progress(Progress::step(1, 2, format!("running {} prompts through the fp32-KV engine", prompts.len())));
    let collector = guarded("calib", || {
        let mut engine = crate::serve::Engine::load(&weights, CALIB_BLOCK, num_blocks, max_batch, blocks_per_seq, max_len.max(1), false, false);
        engine.calibrate_kv(&prompts)
    })?;
    progress(Progress::step(2, 2, "reducing the statistics"));
    let rows = collector.report();
    let base_id = crate::served::base_id(&weights);

    let streams: Vec<serde_json::Value> = rows.iter().map(|r| json!({"name": r.name, "absmax": r.absmax, "p99": r.p99, "p9999": r.p9999, "outlier_ratio": r.outlier_ratio})).collect();
    let report = serde_json::to_string_pretty(&json!({"model": base_id, "prompts": prompts.len(), "streams": streams})).map_err(|e| format!("qwen calib: {e}"))?;
    let clip = model::kvcalib::KvCalib::from_collector(&base_id, cfg.n_layers as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize, percentile, &collector);
    let clip = serde_json::to_string_pretty(&clip.to_json()).map_err(|e| format!("qwen calib: {e}"))?;
    let median = if rows.is_empty() { 0.0 } else { rows[rows.len() / 2].outlier_ratio };
    Ok(Outcome::new()
        .set("prompts", json!(prompts.len()))
        .set("skipped", json!(skipped))
        .set("encode_errors", json!(encode_errors))
        .set("streams", json!(rows.len()))
        .set("worst_stream", json!(rows.first().map(|r| r.name.clone())))
        .set("worst_ratio", json!(rows.first().map_or(0.0, |r| r.outlier_ratio)))
        .set("median_ratio", json!(median))
        .set("percentile", json!(percentile))
        .blob("report", Blob::new(Media::Text, report.into_bytes()))
        .blob("clip", Blob::new(Media::Text, clip.into_bytes())))
}

pub(crate) fn toolcall_gen(resolved: &Resolved, inv: &Invocation, _: &mut dyn FnMut(Progress)) -> ActionResult {
    let out = path_param(inv, "out", "toolcall_gen")?;
    // It needs no checkpoint, only the tokenizer: named, resolved, or beside the checkpoint the store resolved.
    let tokenizer = match (inv.get_str("tokenizer").filter(|p| !p.is_empty()), &resolved.weights) {
        (Some(named), _) => PathBuf::from(named),
        (None, Some(weights)) => resolved.tokenizer(inv, weights),
        (None, None) => resolved.tokenizer.clone().map(PathBuf::from).ok_or("qwen toolcall_gen: no tokenizer (pass 'tokenizer', or pull a Qwen3 model into the models directory)")?,
    };
    let (tok, template) = tokenizer_and_template(&tokenizer, "toolcall_gen")?;
    let (n, held) = (count(inv, "n", 400) as usize, count(inv, "held", 40) as usize);
    let tools = count(inv, "tools", 3) as usize;
    let vocab = count(inv, "vocab", 151_936) as usize;
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let train: Vec<_> = data::toolcall::generate(n, tools, seed).iter().map(|c| c.to_chat_example()).collect();
    let validation: Vec<_> = data::toolcall::generate(held, tools, seed ^ 0xDEAD).iter().map(|c| c.to_chat_example()).collect();
    data::chat::prepare_chat(&train, &validation, &tok, &template, vocab, &out).map_err(|e| format!("qwen toolcall_gen: {e}"))?;
    Ok(Outcome::new().set("train", json!(n)).set("validation", json!(held)).set("seed", json!(seed)).set("out", json!(out.display().to_string())))
}

pub(crate) fn toolcall_eval(resolved: &Resolved, inv: &Invocation, _: &mut dyn FnMut(Progress)) -> ActionResult {
    let weights = resolved.weights(inv, "toolcall_eval")?;
    let (tok, template) = tokenizer_and_template(&resolved.tokenizer(inv, &weights), "toolcall_eval")?;
    let (n, tools, seq) = (count(inv, "n", 40) as usize, count(inv, "tools", 3) as usize, count(inv, "seq", 512) as usize);
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let (exact, token_accuracy) = guarded("toolcall_eval", || crate::toolcall_eval::score(&weights, &tok, &template, n, tools, seq, seed))?;
    Ok(Outcome::new().set("exact_match", json!(exact)).set("token_accuracy", json!(token_accuracy)).set("cases", json!(n)).set("seed", json!(seed)))
}
