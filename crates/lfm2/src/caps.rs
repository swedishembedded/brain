// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder's served actions: `fill_mask` and `embed` (one-shot, the
//! chunked long-context path, so 8k-token inputs work on any backend), and
//! the masked-language-model workflow around them: `tokenize` a corpus,
//! `finetune` on it, `eval` a checkpoint, `import` the reference release.
//!
//! The inference model stays resident across calls, keyed by the exact
//! context length (bidirectional attention makes unmasked padding unsound -
//! see the note at the build site); the tokenizer is cached alongside.
//!
//! Swedish Embedded AB implements on-device text encoders - embeddings,
//! retrieval and masked-language fine-tuning - for its clients. If your
//! team needs expertise in encoder models on its own hardware then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use capability::{
    hot::Hot, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use serde_json::json;

use crate::model::Lfm;

/// The model id the catalog serves LFM2.5-Encoder under.
pub const MODEL: &str = "brain/lfm2";

/// Attention-slab budget for the chunked path. `Lfm::new_impl` derives
/// `chunk = budget / (H · T · 4)`, so at T=8192 / H=16 this is **1024**, not the
/// 2048 this comment used to claim (512 MiB / (16·8192·4 B) = 1024).
/// Only the GEMM/naive attention paths allocate against it — the fused flash
/// path selected on a cooperative device writes no `[H, chunk, T]` slab at all.
const SLAB_BUDGET: u64 = 512 << 20;
/// Fill-mask probe capacity the resident model is built with.
const PROBE_CAP: u32 = 64;

/// The batched inference actions a serving adapter runs itself; every other
/// action it hands to [`LfmProvider`].
pub const BATCHED: [&str; 2] = ["fill_mask", "embed"];

fn data_param() -> ParamSpec {
    ParamSpec::new("data", ParamType::Str, "a directory `tokenize` wrote (train.u32.bin, val.u32.bin), on the serving machine").required().host_resolved()
}

fn seq_param(default: u32) -> ParamSpec {
    ParamSpec::new("seq", ParamType::Int, "tokens per sequence").default(json!(default)).min(1.0).max(capability::bounds::CONTEXT_TOKENS)
}

fn batch_param() -> ParamSpec {
    ParamSpec::new("batch", ParamType::Int, "sequences per batch").default(json!(4)).min(1.0).max(MAX_BATCH)
}

/// Most sequences one batch may hold.
const MAX_BATCH: f64 = 1_024.0;

/// The full, static capability manifest: safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let text = |a: ActionSpec| {
        a.param(ParamSpec::new("text", ParamType::Str, "input text; falls back to the 'text' input blob for long documents"))
            .input(BlobSpec::new("text", Media::Text, "input document (used when the 'text' param is absent)"))
    };
    let fill_mask = text(ActionSpec::new("fill_mask", "MLM top-k predictions at every <|mask|> position").cancellable())
        .param(ParamSpec::new("topk", ParamType::Int, "predictions per mask position").min(0.0).max(capability::bounds::TOP_K).default(json!(5)))
        .output(BlobSpec::new("predictions", Media::Text, "JSON: [{row, tokens: [{id, token, logit}]}]"));
    let embed = text(ActionSpec::new("embed", "bidirectional encoding: per-token hidden states + mean-pooled embedding").cancellable())
        .param(ParamSpec::new("max_tokens", ParamType::Int, "truncate the input to this many tokens (0 = no limit)").min(0.0).max(capability::bounds::TOKENS).default(json!(0)))
        .param(
            ParamSpec::new("normalize", ParamType::Bool, "L2-normalize the mean-pooled embedding before returning it")
                .default(json!(false)),
        )
        .output(BlobSpec::new("embeddings", Media::Bytes, "LE-f32 [n_tokens, dim] hidden states; shape in blob meta"));
    let tokenize = ActionSpec::new("tokenize", "tokenize a text corpus with the checkpoint's tokenizer into the train and validation splits `finetune` and `eval` read").cancellable()
        .param(ParamSpec::new("corpus", ParamType::Str, "the text file on the serving machine").required().host_resolved())
        .param(ParamSpec::new("dest", ParamType::Str, "the directory to write train.u32.bin, val.u32.bin and meta.json to, on the serving machine").required().host_resolved())
        .param(ParamSpec::new("val_frac", ParamType::Float, "share of the tokens held out for validation").default(json!(0.05)).min(0.0).max(capability::bounds::UNIT));
    let finetune = ActionSpec::new("finetune", "masked-language-model fine-tuning on a tokenized corpus, with periodic validation").cancellable()
        .training()
        .streaming()
        .param(data_param())
        .param(ParamSpec::new("save", ParamType::Str, "where to write the fine-tuned checkpoint (.safetensors), on the serving machine").required().host_resolved())
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(100)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(batch_param())
        .param(seq_param(1024))
        .param(ParamSpec::new("lr", ParamType::Float, "peak AdamW learning rate").default(json!(3e-5)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the corruption and the batch order (omit for random)").min(0.0))
        .param(ParamSpec::new("activations", ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()), "what the backward keeps of the forward: every unit's intermediates, or only each unit's input, rebuilt in the backward for the same gradients at lower memory (default keep)"));
    let eval = ActionSpec::new("eval", "pseudo-perplexity and masked-token accuracy on a tokenized validation split").cancellable()
        .param(data_param())
        .param(ParamSpec::new("batches", ParamType::Int, "validation batches").default(json!(8)).min(1.0).max(capability::bounds::TRAIN_COUNT))
        .param(batch_param())
        .param(seq_param(1024))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the masking, so two checkpoints are scored on the same corruption (omit for random)").min(0.0));
    let import = ActionSpec::new("import", "convert the reference Hugging Face release into brain's checkpoint, every tensor verified").cancellable()
        .from_scratch()
        .param(ParamSpec::new("source", ParamType::Str, "the Hugging Face checkpoint directory, on the serving machine").required().host_resolved())
        .param(ParamSpec::new("dest", ParamType::Str, "where to write the converted .safetensors, on the serving machine").required().host_resolved());
    Manifest::new(MODEL, "LFM2.5 bidirectional encoder: fill-mask and long-context embeddings (8k), and masked-language fine-tuning.", vec![fill_mask, embed, tokenize, finetune, eval, import])
}

/// The resident (hot) model + tokenizer and the key that fixes them.
struct LfmResident {
    /// The parsed tokenizer this build was made with, kept for the special
    /// ids `fill_mask` decodes with. Shared with the tokenizer cache.
    tok: std::sync::Arc<QwenBpe>,
    model: Lfm,
}

/// The executable LFM model behind the manifest, over one checkpoint and its
/// tokenizer (neither for `import`, which writes the checkpoint).
/// Construction is free: the checkpoint loads on the first call that needs it
/// and stays resident.
#[derive(Default)]
pub struct LfmProvider {
    weights: Option<String>,
    tokenizer: Option<String>,
    hot: Arc<Hot<(String, u32), LfmResident>>,
    tok_hot: Arc<Hot<String, std::sync::Arc<QwenBpe>>>,
}

impl LfmProvider {
    pub fn new(weights: Option<String>, tokenizer: Option<String>) -> LfmProvider {
        LfmProvider { weights, tokenizer, ..LfmProvider::default() }
    }

    fn weights(&self, action: &str) -> Result<String, String> {
        let weights = self.weights.clone().ok_or_else(|| format!("lfm2 {action}: no LFM2.5-Encoder checkpoint (pull one, or name it with --weights)"))?;
        if !Path::new(&weights).exists() {
            return Err(format!("lfm2 {action}: weights not found at '{weights}'"));
        }
        Ok(weights)
    }

    fn tokenizer(&self, action: &str) -> Result<std::sync::Arc<QwenBpe>, String> {
        let path = self.tokenizer.clone().ok_or_else(|| format!("lfm2 {action}: no tokenizer.json beside the checkpoint (name it with --tokenizer)"))?;
        self.tok_hot.get_cloned(path.clone(), || Ok(std::sync::Arc::new(QwenBpe::from_file(&path)?)))
    }
}

/// LFM2.5-Encoder over `assembly`'s `weights` and `tokenizer` roles.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(LfmProvider::new(assembly.role_path("weights").ok(), assembly.role_path("tokenizer").ok())))
}

impl Provider for LfmProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.iter().any(|a| a.name == name).then(|| {
            Arc::new(LfmAction {
                name: name.to_string(),
                provider: LfmProvider { weights: self.weights.clone(), tokenizer: self.tokenizer.clone(), hot: self.hot.clone(), tok_hot: self.tok_hot.clone() },
            }) as Arc<dyn Action>
        })
    }
}

/// One action over the provider's checkpoint, tokenizer and caches (parsed
/// tokenizers are cached apart from the model: a request of a different
/// length rebuilds the model but never re-parses the tokenizer).
struct LfmAction {
    name: String,
    provider: LfmProvider,
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

impl Action for LfmAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().find(|a| a.name == self.name).expect("known action")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        match self.name.as_str() {
            "fill_mask" | "embed" => self.encode(inv, progress),
            "tokenize" => self.tokenize(inv),
            "finetune" => self.finetune(inv, progress),
            "eval" => self.eval(inv),
            "import" => import(inv),
            other => Err(format!("lfm2: unknown action {other}")),
        }
    }
}

/// The MLM corruption config for an LFM tokenizer: mask, pad, bos and eos
/// resolved from the tokenizer file, never hardcoded.
fn mlm_config(tok: &QwenBpe) -> Result<data::mlm::MlmConfig, String> {
    let mask = tok.special_id("<|mask|>").ok_or("lfm2: the tokenizer has no <|mask|>")?;
    let mut specials: Vec<u32> = ["<|pad|>", "<|startoftext|>", "<|im_end|>"].iter().filter_map(|s| tok.special_id(s)).collect();
    specials.push(mask);
    Ok(data::mlm::MlmConfig { special_ids: specials, ..data::mlm::MlmConfig::new(mask, tok.vocab_size() as u32) })
}

fn import(inv: &Invocation) -> ActionResult {
    let source = inv.get_str("source").ok_or("lfm2 import: missing required param 'source'")?;
    let dest = inv.get_str("dest").ok_or("lfm2 import: missing required param 'dest'")?;
    if let Some(parent) = Path::new(&dest).parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("lfm2 import: {}: {e}", parent.display()))?;
    }
    crate::import::import(&source, &dest)?;
    Ok(Outcome::new().set("dest", json!(dest)))
}

impl LfmAction {
    fn tokenize(&self, inv: &Invocation) -> ActionResult {
        let tok = self.provider.tokenizer("tokenize")?;
        let corpus = inv.get_str("corpus").ok_or("lfm2 tokenize: missing required param 'corpus'")?;
        let dest = inv.get_str("dest").ok_or("lfm2 tokenize: missing required param 'dest'")?;
        let val_frac = inv.get_f64("val_frac").unwrap_or(0.05);
        let text = std::fs::read_to_string(&corpus).map_err(|e| format!("lfm2 tokenize: {corpus}: {e}"))?;
        let ids = tok.encode(&text);
        let split = ((ids.len() as f64) * (1.0 - val_frac)) as usize;
        let dir = Path::new(&dest);
        let io = |what: &str, r: std::io::Result<()>| r.map_err(|e| format!("lfm2 tokenize: {what}: {e}"));
        io(&dest, std::fs::create_dir_all(dir))?;
        io("train.u32.bin", data::binio::write_u32_bin(&dir.join("train.u32.bin"), &ids[..split]))?;
        io("val.u32.bin", data::binio::write_u32_bin(&dir.join("val.u32.bin"), &ids[split..]))?;
        io("meta.json", std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(tok.vocab_size())))?;
        Ok(Outcome::new().set("train_tokens", json!(split)).set("val_tokens", json!(ids.len() - split)).set("vocab", json!(tok.vocab_size())))
    }

    fn finetune(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = self.provider.weights("finetune")?;
        let mlm = mlm_config(&*self.provider.tokenizer("finetune")?)?;
        let dir = inv.get_str("data").ok_or("lfm2 finetune: missing required param 'data'")?;
        let save = inv.get_str("save").ok_or("lfm2 finetune: missing required param 'save'")?;
        let train = data::binio::read_tokens_u32(&Path::new(&dir).join("train")).map_err(|e| format!("lfm2 finetune: {dir}/train: {e}"))?;
        let val = data::binio::read_tokens_u32(&Path::new(&dir).join("val")).unwrap_or_default();
        let (steps, batch, seq) = (inv.get_i64("steps").unwrap_or(100) as u32, inv.get_i64("batch").unwrap_or(4) as u32, inv.get_i64("seq").unwrap_or(1024) as u32);
        let saved = inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?;
        let defaults = crate::train::MlmTrainOpts::default();
        let opts = crate::train::MlmTrainOpts { steps, lr: inv.get_f64("lr").unwrap_or(3e-5) as f32, seed: inv.get_i64("seed").map_or(defaults.seed, |s| s as u64), ..defaults };
        // Long sequences train on the chunked regime: bounded [H, chunk, T]
        // attention slabs and the gathered supervised-row MLM head (full T x T
        // scores and [n, vocab] logits both exceed the binding budget at 8k).
        let mut m = if seq > 2048 {
            let head_cap = ((batch * seq) as f32 * 0.35) as u32 + 64;
            Lfm::load_train_chunked(&weights, batch, seq, SLAB_BUDGET, head_cap)
        } else {
            Lfm::load_train(&weights, batch, seq)
        };
        if let Some(policy) = saved {
            m.set_saved_policy(policy).map_err(|e| format!("lfm2 finetune: activations {policy:?}: {e}"))?;
        }
        let mut lines = Vec::new();
        let val_loss = crate::train::finetune(&m, &train, &val, &mlm, batch, seq, &opts, &mut |line| {
            progress(Progress::step(0, steps, line.clone()));
            lines.push(line);
        }, &|| inv.cancel.is_cancelled())?;
        if let Some(parent) = Path::new(&save).parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| format!("lfm2 finetune: {}: {e}", parent.display()))?;
        }
        m.save(&save);
        Ok(Outcome::new().set("val_loss", json!(val_loss)).set("pseudo_perplexity", json!(val_loss.exp())).set("seed", json!(opts.seed)).set("log", json!(lines)).set("saved", json!(save)))
    }

    fn eval(&self, inv: &Invocation) -> ActionResult {
        let weights = self.provider.weights("eval")?;
        let mlm = mlm_config(&*self.provider.tokenizer("eval")?)?;
        let dir = inv.get_str("data").ok_or("lfm2 eval: missing required param 'data'")?;
        let val = data::binio::read_tokens_u32(&Path::new(&dir).join("val")).map_err(|e| format!("lfm2 eval: {dir}/val: {e}"))?;
        let (batches, batch, seq) = (inv.get_i64("batches").unwrap_or(8) as u32, inv.get_i64("batch").unwrap_or(4) as u32, inv.get_i64("seq").unwrap_or(1024) as u32);
        let seed = inv.get_i64("seed").map_or_else(data::rng::random_seed, |s| s as u64);
        let model = Lfm::load_inference(&weights, batch, seq);
        let r = crate::train::evaluate(&model, &val, &mlm, batches, batch, seq, seed);
        Ok(Outcome::new()
            .set("pseudo_perplexity", json!(r.pseudo_perplexity))
            .set("masked_accuracy", json!(r.masked_accuracy))
            .set("masked_tokens", json!(r.masked_tokens))
            .set("seed", json!(seed)))
    }

    fn encode(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = self.provider.weights(&self.name)?;
        let tok = self.provider.tokenizer(&self.name)?;
        let text = input_text(inv)?;

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
        // The graph is built at the EXACT request length (see the comment
        // above): the key is the checkpoint plus that length.
        self.provider.hot.try_get_mut(
            (weights.clone(), need),
            || Ok(LfmResident { tok: tok.clone(), model: Lfm::load_inference_chunked(&weights, 1, need, SLAB_BUDGET, PROBE_CAP) }),
            |hot| {
                let (model, tok) = (&hot.model, &hot.tok);
                let cap = need;
                let n = ids.len();
            debug_assert_eq!(n as u32, cap);
            model.set_tokens(&ids);
            progress(Progress::step(1, 2, "encode"));

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
                    // `false` by default: the raw, un-normalized mean stays
                    // byte-identical for every caller that predates this param -
                    // see `EmbeddingPipeline`'s own doc for why the SDK's Qwen3
                    // backbone always normalizes but this one does not by default
                    // (a genuinely different existing-caller compatibility bar).
                    if inv.get_bool("normalize").unwrap_or(false) {
                        let norm = mean.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
                        if norm > 0.0 {
                            for x in &mut mean {
                                *x = (*x as f64 / norm) as f32;
                            }
                        }
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
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    #[test]
    fn manifest_declares_the_encoder_and_its_training_workflow() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["fill_mask", "embed", "tokenize", "finetune", "eval", "import"]);
        for name in BATCHED {
            let a = m.actions.iter().find(|a| a.name == name).unwrap();
            assert!(!a.streaming, "{name}: encoder actions are one-shot (ttfa == e2e)");
        }
        let finetune = m.actions.iter().find(|a| a.name == "finetune").unwrap();
        assert!(finetune.training && finetune.streaming);
        assert!(m.actions.iter().find(|a| a.name == "import").unwrap().from_scratch, "import writes the checkpoint");
        let fm = &m.actions[0];
        let inv = fm.validate(Invocation::new().set("text", json!("x"))).unwrap();
        assert_eq!(inv.get_i64("topk"), Some(5));
    }

    /// `normalize` defaults to `false` - the SAME behavior every caller
    /// before this param existed already gets, byte-identical.
    #[test]
    fn embed_normalize_defaults_to_false_and_leaves_the_raw_mean_unnormalized() {
        let e = manifest().actions.into_iter().find(|a| a.name == "embed").expect("embed action");
        let n = e.params.iter().find(|p| p.name == "normalize").expect("normalize param");
        assert_eq!(n.default, Some(json!(false)));
    }

    /// End to end over a tiny synthetic checkpoint: `normalize: true` must
    /// actually L2-normalize the returned mean, and the default (unset) run
    /// must NOT - the two runs over the identical input must differ exactly
    /// where the norm does.
    #[test]
    fn embed_normalize_true_l2_normalizes_the_mean_and_false_leaves_it_raw() {
        let dir = std::env::temp_dir().join(format!("lfm2-caps-normalize-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // `write_byte_tokenizer`-equivalent below assigns ids across the full
        // printable-ASCII + space/newline range (~96 tokens); `LfmConfig::tiny`'s
        // vocab of 23 is too small to index that.
        let cfg = crate::config::LfmConfig { vocab: 256, ..crate::config::LfmConfig::tiny() };
        let init = crate::init::init_weights(&cfg, 7);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap().clone())).collect();
        let weights = dir.join("tiny.safetensors");
        checkpoint::save(weights.to_str().unwrap(), cfg.to_json(), &tensors);

        let mut vocab = serde_json::Map::new();
        for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
            vocab.insert(c.to_string(), json!(i));
        }
        std::fs::write(dir.join("tokenizer.json"), json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();

        let mut r = Registry::new();
        r.register(Arc::new(LfmProvider::new(Some(weights.to_str().unwrap().to_string()), Some(dir.join("tokenizer.json").to_str().unwrap().to_string()))));
        let run = |normalize: bool| -> Vec<f64> {
            let out = r.run(MODEL, "embed", Invocation::new().set("text", json!("hello world")).set("normalize", json!(normalize)), &mut |_| {}).unwrap();
            out.outputs["mean"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
        };

        let raw = run(false);
        let raw_norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(raw_norm > 0.0 && (raw_norm - 1.0).abs() > 1e-6, "a random-init tiny model's raw mean must not already be unit norm (got {raw_norm})");

        let normalized = run(true);
        let normalized_norm = normalized.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((normalized_norm - 1.0).abs() < 1e-6, "normalize: true must return a unit-norm vector, got norm {normalized_norm}");

        // Same direction, different magnitude: normalized == raw / raw_norm.
        for (n, r) in normalized.iter().zip(&raw) {
            assert!((n - r / raw_norm).abs() < 1e-5, "normalized value should be raw/||raw||");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The masked-language workflow end to end on a tiny checkpoint:
    /// `tokenize` writes the splits, `finetune` trains on them and saves a
    /// checkpoint the provider loads back, and `eval` scores it.
    #[test]
    fn tokenize_finetune_and_eval_round_trip() {
        let dir = std::env::temp_dir().join(format!("lfm2-caps-mlm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = crate::config::LfmConfig { vocab: 260, ..crate::config::LfmConfig::tiny() };
        let init = crate::init::init_weights(&cfg, 7);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap().clone())).collect();
        let weights = dir.join("tiny.safetensors");
        checkpoint::save(weights.to_str().unwrap(), cfg.to_json(), &tensors);
        let mut vocab = serde_json::Map::new();
        for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
            vocab.insert(c.to_string(), json!(i));
        }
        let tokenizer = dir.join("tokenizer.json");
        std::fs::write(&tokenizer, json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": [{"id": 256, "content": "<|mask|>", "special": true}]}).to_string()).unwrap();
        let corpus = dir.join("corpus.txt");
        std::fs::write(&corpus, "the quick brown fox jumps over the lazy dog. ".repeat(40)).unwrap();

        let run = |weights: &std::path::Path, action: &str, inv: Invocation| {
            let mut r = Registry::new();
            r.register(Arc::new(LfmProvider::new(Some(weights.to_str().unwrap().to_string()), Some(tokenizer.to_str().unwrap().to_string()))));
            r.run(MODEL, action, inv, &mut |_| {}).unwrap_or_else(|e| panic!("{action}: {e}"))
        };
        let data = dir.join("data");
        let split = run(&weights, "tokenize", Invocation::new().set("corpus", json!(corpus.to_str().unwrap())).set("dest", json!(data.to_str().unwrap())).set("val_frac", json!(0.25)));
        assert!(split.outputs["train_tokens"].as_u64().unwrap() > split.outputs["val_tokens"].as_u64().unwrap());

        let tuned = dir.join("tuned.safetensors");
        let ft = Invocation::new().set("data", json!(data.to_str().unwrap())).set("save", json!(tuned.to_str().unwrap())).set("steps", json!(2)).set("batch", json!(1)).set("seq", json!(32)).set("seed", json!(3));
        let out = run(&weights, "finetune", ft);
        assert!(out.outputs["val_loss"].as_f64().unwrap().is_finite());
        assert!(tuned.exists());

        let ev = Invocation::new().set("data", json!(data.to_str().unwrap())).set("batches", json!(1)).set("batch", json!(1)).set("seq", json!(32)).set("seed", json!(3));
        let scored = run(&tuned, "eval", ev);
        assert!(scored.outputs["pseudo_perplexity"].as_f64().unwrap() > 0.0);
        assert!(scored.outputs["masked_tokens"].as_u64().unwrap() > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_checkpoint_is_a_clean_error() {
        let run = |weights: Option<&str>| {
            let mut r = Registry::new();
            r.register(Arc::new(LfmProvider::new(weights.map(str::to_string), Some("t".into()))));
            r.run(MODEL, "embed", Invocation::new().set("text", json!("hi")), &mut |_| {}).unwrap_err()
        };
        assert!(run(Some("/nonexistent/lfm.safetensors")).contains("not found"));
        assert!(run(None).contains("no LFM2.5-Encoder checkpoint"));
    }
}
