// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM-5.2's served actions: `train` a decoder from scratch on a tokenized
//! dataset, `finetune` a brain checkpoint, `eval` its validation loss,
//! `generate` text, and `import` a Hugging Face release into a brain
//! checkpoint.
//!
//! The weights are the store's `weights` role ([`crate::spec::SPEC`]): a brain
//! checkpoint (char-level, carrying its own vocabulary, written by `import`,
//! `train` and `finetune` with a `glm` card) or a downloaded Hugging Face
//! directory served as released with its own `tokenizer.json`. `train` and
//! `import` need none. The OpenVINO export and NPU decode need the `npu`
//! crate and so are actions of `samples/models/glmdsa`.
//!
//! Swedish Embedded AB implements training and serving of sparse
//! Mixture-of-Experts language models on its clients' own data and machines.
//! If your team needs expertise in MLA and expert-routed decoders then you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{hot::Hot, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::binio::Meta;
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use serde_json::json;

use crate::config::GlmConfig;
use crate::model::Glm;

/// The model id used by `brain caps`, `brain glmdsa <action>` and the event API.
///
/// `brain/glm`, not `brain/glmdsa`: this is the MODEL ref, and it is already
/// the canonical id everywhere else - `modelref::alias::ROWS` maps the bare
/// `glm` onto it and the checkpoint's own `ModelCard` carries it. `glmdsa` is
/// the *architecture* id (`brain_arch`, from llama.cpp's `LLM_ARCH_GLM_DSA`),
/// a different namespace.
pub const MODEL: &str = "brain/glm";

/// Largest model width, depth, head or expert count a run may ask for.
const MAX_DIM: f64 = 16_384.0;
/// Most sequences one step may hold.
const MAX_BATCH: f64 = 4_096.0;
/// Most windows one evaluation may score.
const MAX_EVAL_BATCHES: f64 = 1_000_000.0;
/// Highest sampling temperature.
const MAX_TEMPERATURE: f64 = 10.0;

/// Steps, batch and block of a run that names none, and the learning rates of a
/// fresh run and of a fine-tune.
const STEPS: i64 = 2_000;
const BATCH: i64 = 8;
const BLOCK: i64 = 128;
const LR: f64 = 3e-4;
const FINETUNE_LR: f64 = 1e-4;
/// Seconds between periodic checkpoint writes of a long run.
const CHECKPOINT_SECS: u64 = 600;

const SIZES: [&str; 3] = ["tiny", "small", "base"];

fn host_path(name: &str, what: &str) -> ParamSpec {
    ParamSpec::new(name, ParamType::Str, what).host_resolved()
}

fn seed_param(what: &str) -> ParamSpec {
    ParamSpec::new("seed", ParamType::Int, what).min(0.0)
}

fn activations_param() -> ParamSpec {
    ParamSpec::new(
        "activations",
        ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
        "what the backward keeps of the forward (default: the model's own)",
    )
}

/// The knobs `train` and `finetune` share.
fn training_params(spec: ActionSpec, lr: f64) -> ActionSpec {
    spec.training()
        .streaming()
        .param(host_path("data", "the dataset directory (`brain data` prepares one)").required())
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(STEPS)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("batch", ParamType::Int, "sequences per step").default(json!(BATCH)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("block", ParamType::Int, "tokens per sequence").default(json!(BLOCK)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate").default(json!(lr)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("mask", ParamType::Str, "train only on what follows this character on each line (the answer after '=' of an LHS=RHS dataset)"))
        .param(ParamSpec::new("align", ParamType::Bool, "start every sequence at a line boundary").default(json!(false)))
        .param(activations_param())
}

pub fn train_spec() -> ActionSpec {
    training_params(ActionSpec::new("train", "train the GLM decoder from scratch on a tokenized dataset (train.bin, val.bin, meta.json)"), LR)
        .from_scratch()
        .param(host_path("save", "where to write the checkpoint (.safetensors); omit to train without keeping it"))
        .param(ParamSpec::new("size", ParamType::Enum(SIZES.map(String::from).to_vec()), "the shape preset the sizes below override").default(json!("small")))
        .param(ParamSpec::new("layers", ParamType::Int, "decoder layers (default: the preset's)").min(1.0).max(MAX_DIM))
        .param(ParamSpec::new("d_model", ParamType::Int, "model width (default: the preset's)").min(1.0).max(MAX_DIM))
        .param(ParamSpec::new("heads", ParamType::Int, "attention heads (default: the preset's)").min(1.0).max(MAX_DIM))
        .param(ParamSpec::new("experts", ParamType::Int, "routed experts (default: the preset's)").min(1.0).max(MAX_DIM))
        .param(seed_param("seeds the initialization and the batches (omit for random; the outcome echoes it)"))
}

pub fn finetune_spec() -> ActionSpec {
    training_params(ActionSpec::new("finetune", "continue training the resolved brain checkpoint on a tokenized dataset, into a new checkpoint"), FINETUNE_LR)
        .param(host_path("save", "where to write the fine-tuned checkpoint (.safetensors); seeded from the base checkpoint when absent, resumed when present").required())
        .param(seed_param("seeds the batches (omit for random; the outcome echoes it)"))
}

pub fn eval_spec() -> ActionSpec {
    ActionSpec::new("eval", "validation cross-entropy and perplexity of the resolved brain checkpoint on a dataset's val split")
        .param(host_path("data", "the dataset directory").required())
        .param(ParamSpec::new("batches", ParamType::Int, "windows scored").default(json!(20)).min(1.0).max(MAX_EVAL_BATCHES))
        .param(ParamSpec::new("block", ParamType::Int, "tokens per window; 0 = the checkpoint's own context").default(json!(0)).min(0.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(seed_param("seeds the windows drawn (omit for random; the outcome echoes it)"))
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("generate", "generate text continuing a prompt (GLM MLA + MoE decoder, KV-cached decoding)")
        .param(host_path("data", "a dataset directory whose meta.json holds the vocabulary, for a checkpoint that does not embed one"))
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (empty: a newline)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").min(0.0).max(capability::bounds::TOKENS).default(json!(DEFAULT_MAX_NEW)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (0 = greedy)").min(0.0).max(MAX_TEMPERATURE).default(json!(DEFAULT_TEMP)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 = disabled)").min(0.0).max(capability::bounds::TOP_K).default(json!(DEFAULT_TOP_K)))
        .param(seed_param("RNG seed (omit for random; the outcome echoes it)"))
        .output(BlobSpec::new("text", Media::Text, "the generated text"))
}

pub fn import_spec() -> ActionSpec {
    ActionSpec::new("import", "convert a Hugging Face GLM-5.2 checkpoint into a brain checkpoint, carded for the model store")
        .from_scratch()
        .param(host_path("source", "the Hugging Face checkpoint directory").required())
        .param(host_path("dest", "where to write the .safetensors").required())
        .param(ParamSpec::new("id", ParamType::Str, "the model id recorded on the card (default: the destination's base name)"))
}

/// The full, static capability manifest: safe to build with nothing on disk.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "GLM-5.2 decoder (MLA + sigmoid noaux_tc MoE + DSA indexer): training, evaluation, conversion and text generation.",
        vec![train_spec(), finetune_spec(), eval_spec(), generate_spec(), import_spec()],
    )
}

const DEFAULT_MAX_NEW: i64 = 128;
const DEFAULT_TEMP: f64 = 0.8;
const DEFAULT_TOP_K: i64 = 40;

/// Sampling parameters, read once so the served and direct paths agree on
/// defaults. Kept next to the [`ParamSpec`] defaults above: a default that
/// lives in only one of the two places is how they drift.
///
/// The seed is the one given, or a fresh draw when none is; a seed that is not
/// a non-negative integer is an error. The caller reports the seed it used.
pub fn sampling(inv: &Invocation) -> Result<(usize, f32, usize, u64), String> {
    Ok((
        inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).max(0) as usize,
        inv.get_f64("temp").unwrap_or(DEFAULT_TEMP) as f32,
        inv.get_i64("top_k").unwrap_or(DEFAULT_TOP_K).max(0) as usize,
        inv.seed()?.unwrap_or_else(data::rng::random_seed),
    ))
}

/// A model resident for generation, and the vocabulary that encodes its prompts.
struct GlmModel {
    tok: Box<dyn Tokenizer + Send>,
    model: Glm,
}

/// What a loaded model is keyed on: the checkpoint and the dataset that
/// supplied its vocabulary.
type ModelKey = (String, Option<String>);

/// A char-level checkpoint's vocabulary: the embedded one, else the dataset's
/// meta.json.
pub fn char_vocabulary(embedded: Option<Vec<char>>, data: Option<&Path>) -> Result<Vec<char>, String> {
    if let Some(itos) = embedded {
        return Ok(itos);
    }
    let dir = data.ok_or("glmdsa: the checkpoint embeds no char vocabulary; give data, a dataset directory whose meta.json holds it")?;
    let meta = dir.join("meta.json");
    let text = std::fs::read_to_string(&meta).map_err(|e| format!("{}: {e}", meta.display()))?;
    Ok(Meta::from_json(&text).map_err(|e| format!("{}: {e}", meta.display()))?.itos)
}

/// Open `weights` for generation: a Hugging Face directory is served as
/// released with its `tokenizer.json`; a brain checkpoint streams from its
/// mmap, in the encoding it is stored in.
fn load(weights: &str, data: Option<&Path>) -> Result<GlmModel, String> {
    let path = Path::new(weights);
    if path.is_dir() {
        let (cfg, src) = crate::import::open_dir(path).map_err(|e| format!("glmdsa: {e}"))?;
        let tokenizer = path.join("tokenizer.json");
        let tok = ::data::qwen_tokenizer::QwenBpe::from_file(tokenizer.to_str().ok_or("glmdsa: non-UTF8 path")?).map_err(|e| format!("glmdsa: {}: {e}", tokenizer.display()))?;
        let block = cfg.block_size;
        return Ok(GlmModel { tok: Box::new(tok), model: Glm::from_source_inference(cfg, &src, 1, block) });
    }
    let reader = checkpoint::weightio::WeightReader::open(weights).map_err(|e| format!("glmdsa: {weights}: {e}"))?;
    let config = reader.config();
    let tok = CharTokenizer::from_itos(char_vocabulary(Glm::itos_from_config(&config), data)?);
    let block = GlmConfig::from_json(&config).block_size;
    Ok(GlmModel { tok: Box::new(tok), model: Glm::from_reader_inference(&reader, 1, block) })
}

/// GLM over the store's `weights` role (none serves only `train` and `import`).
/// Construction is free: the checkpoint loads on the first `generate` and stays
/// resident.
#[derive(Default)]
pub struct GlmProvider {
    weights: Option<String>,
    hot: Arc<Hot<ModelKey, GlmModel>>,
}

impl GlmProvider {
    pub fn new(weights: Option<String>) -> GlmProvider {
        GlmProvider { weights, hot: Arc::default() }
    }
}

/// GLM over `assembly`'s `weights` role.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(GlmProvider::new(assembly.role_path("weights").ok())))
}

impl Provider for GlmProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let kind = match name {
            "train" => Kind::Train,
            "finetune" => Kind::Finetune,
            "eval" => Kind::Eval,
            "generate" => Kind::Generate,
            "import" => Kind::Import,
            _ => return None,
        };
        Some(Arc::new(GlmAction { kind, weights: self.weights.clone(), hot: self.hot.clone() }))
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Train,
    Finetune,
    Eval,
    Generate,
    Import,
}

struct GlmAction {
    kind: Kind,
    weights: Option<String>,
    hot: Arc<Hot<ModelKey, GlmModel>>,
}

impl GlmAction {
    fn need_weights(&self, action: &str) -> Result<&str, String> {
        self.weights.as_deref().ok_or_else(|| format!("glmdsa {action}: no GLM checkpoint (import one, train one, or name it with --weights)"))
    }
}

impl Action for GlmAction {
    fn spec(&self) -> ActionSpec {
        match self.kind {
            Kind::Train => train_spec(),
            Kind::Finetune => finetune_spec(),
            Kind::Eval => eval_spec(),
            Kind::Generate => generate_spec(),
            Kind::Import => import_spec(),
        }
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.kind {
            Kind::Train => train(inv, None, progress),
            Kind::Finetune => train(inv, Some(self.need_weights("finetune")?), progress),
            Kind::Eval => eval(inv, self.need_weights("eval")?),
            Kind::Generate => generate(inv, self.need_weights("generate")?, &self.hot, progress),
            Kind::Import => import(inv),
        }
    }
}

fn required_path(inv: &Invocation, name: &str, action: &str) -> Result<PathBuf, String> {
    inv.get_str(name).map(PathBuf::from).ok_or_else(|| format!("glmdsa {action}: missing required param '{name}'"))
}

/// The fit options both training actions share, from the invocation.
fn fit_opts(inv: &Invocation, action: &str, default_lr: f64) -> Result<model::FitOpts, String> {
    let n = |name: &str, default: i64| inv.get_i64(name).unwrap_or(default).max(1) as u32;
    let steps = n("steps", STEPS);
    let lr = inv.get_f64("lr").unwrap_or(default_lr) as f32;
    let mask = match inv.get_str("mask") {
        None => None,
        Some(m) => {
            let mut chars = m.chars();
            Some(chars.next().filter(|_| chars.next().is_none()).ok_or_else(|| format!("glmdsa {action}: mask {m:?} is not one character"))?)
        }
    };
    Ok(model::FitOpts {
        steps,
        batch_size: n("batch", BATCH),
        block_size: n("block", BLOCK),
        lr,
        // 50 steps, unless the run is shorter than that: a warmup longer than the run never reaches the peak rate.
        warmup: 50.min((steps / 2).max(1)),
        decay_iters: steps,
        min_lr: lr * 0.1,
        weight_decay: 0.1,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: (steps / 10).max(1),
        eval_batches: 20,
        checkpoint_secs: CHECKPOINT_SECS,
        mask_before: mask,
        mask_per_line: mask.is_some(),
        align_to_lines: inv.get_bool("align").unwrap_or(false),
        seed: inv.seed()?.unwrap_or_else(data::rng::random_seed),
        adam: crate::ADAM,
        precision: model::TrainPrecision { activations: inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?, ..Default::default() },
        ..Default::default()
    })
}

/// A from-scratch shape: the `size` preset, then the named overrides.
fn fresh_config(inv: &Invocation, block: u32) -> Result<GlmConfig, String> {
    let size = inv.get_str("size").unwrap_or_else(|| "small".into());
    let mut cfg = GlmConfig::preset(&size, block).ok_or_else(|| format!("glmdsa train: size {size:?} is not one of {SIZES:?}"))?;
    let set = |name: &str, field: &mut u32| {
        if let Some(v) = inv.get_i64(name) {
            *field = v.max(1) as u32;
        }
    };
    set("layers", &mut cfg.n_layers);
    set("d_model", &mut cfg.d_model);
    set("heads", &mut cfg.n_heads);
    set("experts", &mut cfg.n_routed_experts);
    Ok(cfg)
}

/// `train` (`base` none: fresh weights) and `finetune` (`base`: the checkpoint
/// the run continues). A fine-tune's output starts as a copy of the base, so a
/// run that stopped resumes from its own output.
fn train(inv: &Invocation, base: Option<&str>, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let action = if base.is_some() { "finetune" } else { "train" };
    let dir = required_path(inv, "data", action)?;
    let save = if base.is_some() { Some(required_path(inv, "save", action)?) } else { inv.get_str("save").map(PathBuf::from) };
    let opts = fit_opts(inv, action, if base.is_some() { FINETUNE_LR } else { LR })?;
    let cfg = match base {
        Some(path) => {
            if Path::new(path).is_dir() {
                return Err(format!("glmdsa finetune: {path} is a Hugging Face directory; `import` it into a brain checkpoint first"));
            }
            let cfg = GlmConfig::from_json(&checkpoint::read_config(path));
            if let Some(out) = save.as_deref().filter(|out| !out.exists()) {
                if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent).map_err(|e| format!("glmdsa finetune: {}: {e}", parent.display()))?;
                }
                std::fs::copy(path, out).map_err(|e| format!("glmdsa finetune: seed {} from {path}: {e}", out.display()))?;
            }
            cfg
        }
        None => fresh_config(inv, opts.block_size)?,
    };
    let report = model::train::fit_watched::<Glm>(&dir, cfg, &opts, save.as_deref(), &mut |s| {
        progress(Progress::step(s.step, s.steps, format!("loss {:.4}", s.loss)));
        !inv.cancel.is_cancelled()
    })
    .map_err(|e| format!("glmdsa {action}: {e}"))?;
    if report.interrupted {
        return Err(format!("cancelled after step {}", report.steps_completed));
    }
    Ok(Outcome::new()
        .set("initial_loss", json!(report.initial_loss))
        .set("final_loss", json!(report.final_loss))
        .set("steps", json!(report.steps_completed))
        .set("seed", json!(opts.seed))
        .set("saved", json!(save.map(|p| p.display().to_string()))))
}

fn eval(inv: &Invocation, weights: &str) -> ActionResult {
    if Path::new(weights).is_dir() {
        return Err(format!("glmdsa eval: {weights} is a Hugging Face directory; `import` it into a brain checkpoint first"));
    }
    let dir = required_path(inv, "data", "eval")?;
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let batches = inv.get_i64("batches").unwrap_or(20).max(1) as usize;
    let block = inv.get_i64("block").unwrap_or(0).max(0) as u32;
    let ce = crate::eval::val_cross_entropy(weights, &dir, batches, block, seed).map_err(|e| format!("glmdsa eval: {e}"))?;
    Ok(Outcome::new().set("val_ce", json!(ce)).set("val_perplexity", json!(ce.exp())).set("seed", json!(seed)))
}

fn generate(inv: &Invocation, weights: &str, hot: &Hot<ModelKey, GlmModel>, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let (max_new, temp, top_k, seed) = sampling(inv)?;
    let data = inv.get_str("data").map(PathBuf::from);
    let key = (weights.to_string(), data.as_ref().map(|d| d.display().to_string()));
    hot.try_get(
        key,
        || load(weights, data.as_deref()),
        |hot| {
            let prompt = inv.get_str("prompt").filter(|p| !p.is_empty()).unwrap_or_else(|| "\n".into());
            progress(Progress::step(0, max_new as u32, "generating"));
            let mut rng = Rng::new(seed);
            let generated = crate::sample::generate_kv(&hot.model, &hot.tok.encode(&prompt), max_new, temp, top_k, None, &mut rng);
            progress(Progress::step(max_new as u32, max_new as u32, "done"));
            Ok(Outcome::new().set("seed", json!(seed)).blob("text", Blob::new(Media::Text, hot.tok.decode(&generated).into_bytes())))
        },
    )
}

fn import(inv: &Invocation) -> ActionResult {
    let source = required_path(inv, "source", "import")?;
    let dest = required_path(inv, "dest", "import")?;
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("glmdsa import: {}: {e}", parent.display()))?;
    }
    let (source_str, dest_str) = (source.to_string_lossy(), dest.to_string_lossy());
    crate::import::import_as(&source_str, &dest_str, inv.get_str("id").as_deref()).map_err(|e| format!("glmdsa import: {e}"))?;
    Ok(Outcome::new().set("dest", json!(dest_str)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    fn registry(weights: Option<&str>) -> Registry {
        let mut reg = Registry::new();
        reg.register(Arc::new(GlmProvider::new(weights.map(str::to_string))));
        reg
    }

    fn run(reg: &Registry, action: &str, inv: Invocation) -> ActionResult {
        reg.run(MODEL, action, inv, &mut |_| {})
    }

    /// A char dataset: train.bin, val.bin and the vocabulary.
    fn write_dataset(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let text = "1+1=2\n2+2=4\n3+3=6\n".repeat(60);
        let mut itos: Vec<char> = text.chars().collect();
        itos.sort_unstable();
        itos.dedup();
        let ids: Vec<u16> = CharTokenizer::from_itos(itos.clone()).encode(&text).into_iter().map(|t| t as u16).collect();
        let split = ids.len() * 9 / 10;
        data::binio::write_u16_bin(&dir.join("train.bin"), &ids[..split]).unwrap();
        data::binio::write_u16_bin(&dir.join("val.bin"), &ids[split..]).unwrap();
        std::fs::write(dir.join("meta.json"), Meta { vocab_size: itos.len(), itos }.to_json()).unwrap();
    }

    fn finetune_inv(data: &Path, save: &Path) -> Invocation {
        Invocation::new()
            .set("data", json!(data.to_str().unwrap()))
            .set("save", json!(save.to_str().unwrap()))
            .set("steps", json!(3))
            .set("batch", json!(2))
            .set("block", json!(16))
            .set("seed", json!(4))
    }

    fn train_inv(data: &Path, save: &Path) -> Invocation {
        finetune_inv(data, save).set("size", json!("tiny"))
    }

    #[test]
    fn the_manifest_names_the_actions_and_marks_the_training_ones() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["train", "finetune", "eval", "generate", "import"]);
        let get = |n: &str| m.actions.iter().find(|a| a.name == n).unwrap();
        assert!(get("train").training && get("train").streaming && get("train").from_scratch);
        assert!(get("finetune").training && get("finetune").streaming && !get("finetune").from_scratch, "a fine-tune needs its checkpoint");
        assert!(get("import").from_scratch && !get("eval").from_scratch && !get("generate").from_scratch);
    }

    /// `sampling` reads the same defaults the `ParamSpec`s declare. They live in
    /// two places (a spec default is JSON, a read is Rust), so pin them equal:
    /// a drifting default changes generation silently, with nothing failing.
    #[test]
    fn the_read_defaults_match_the_declared_defaults() {
        let (max_new, temp, top_k, _seed) = sampling(&Invocation::default()).unwrap();
        let spec = generate_spec();
        let declared = |name: &str| spec.params.iter().find(|p| p.name == name).and_then(|p| p.default.clone()).unwrap_or_else(|| panic!("'{name}' declares no default"));
        assert_eq!(declared("max_new").as_i64(), Some(max_new as i64));
        assert_eq!(declared("temp").as_f64().map(|v| v as f32), Some(temp));
        assert_eq!(declared("top_k").as_i64(), Some(top_k as i64));
        for action in manifest().actions {
            if let Some(seed) = action.params.iter().find(|p| p.name == "seed") {
                assert!(seed.default.is_none(), "{}: 'seed' must not declare a fixed default", action.name);
            }
        }
    }

    #[test]
    fn a_missing_checkpoint_is_a_named_error_not_a_panic() {
        let reg = registry(None);
        for action in ["generate", "eval"] {
            let inv = Invocation::new().set("data", json!("/nonexistent"));
            let err = run(&reg, action, inv).unwrap_err();
            assert!(err.contains("no GLM checkpoint"), "{action}: {err}");
        }
    }

    #[test]
    fn a_multi_character_mask_and_an_unknown_size_are_refused() {
        let reg = registry(None);
        let err = run(&reg, "train", Invocation::new().set("data", json!("/nonexistent")).set("mask", json!("=>"))).unwrap_err();
        assert!(err.contains("one character"), "{err}");
        let err = run(&reg, "train", Invocation::new().set("data", json!("/nonexistent")).set("size", json!("huge"))).unwrap_err();
        assert!(err.contains("size"), "{err}");
    }

    /// Train a few steps from scratch, then the checkpoint is carded for the
    /// store, scores a validation loss, and generates: an unseeded draw reports
    /// its seed and passing that seed back reproduces the text. A fine-tune of
    /// it continues into a second checkpoint.
    #[test]
    fn train_then_eval_generate_and_finetune_through_the_actions() {
        let dir = std::env::temp_dir().join(format!("glmdsa-caps-{}", std::process::id()));
        let data = dir.join("data");
        write_dataset(&data);
        let save = dir.join("glm.safetensors");

        let trained = run(&registry(None), "train", train_inv(&data, &save)).unwrap();
        assert_eq!(trained.outputs["steps"].as_u64(), Some(3));
        assert_eq!(trained.outputs["seed"].as_u64(), Some(4));
        let card = checkpoint::st::read_card(save.to_str().unwrap()).unwrap().expect("the checkpoint carries a card");
        assert_eq!(card.family, crate::spec::CARD_FAMILY);

        let reg = registry(Some(save.to_str().unwrap()));
        let scored = run(&reg, "eval", Invocation::new().set("data", json!(data.to_str().unwrap())).set("batches", json!(1)).set("seed", json!(1))).unwrap();
        assert!(scored.outputs["val_perplexity"].as_f64().is_some_and(|p| p > 1.0));

        let request = || Invocation::new().set("prompt", json!("1+")).set("max_new", json!(6)).set("temp", json!(1.5)).set("top_k", json!(0));
        let first = run(&reg, "generate", request()).unwrap();
        let drawn = first.outputs["seed"].as_u64().expect("the outcome carries the seed the draw used");
        let replay = run(&reg, "generate", request().set("seed", json!(drawn))).unwrap();
        assert_eq!(replay.blobs["text"].bytes, first.blobs["text"].bytes, "the reported seed reproduces the draw");
        let err = run(&reg, "generate", request().set("seed", json!(-1))).unwrap_err();
        assert!(err.contains("seed"), "{err}");

        let tuned = dir.join("tuned.safetensors");
        let fine = run(&reg, "finetune", finetune_inv(&data, &tuned).set("lr", json!(1e-4))).unwrap();
        assert_eq!(fine.outputs["steps"].as_u64(), Some(3));
        assert!(checkpoint::st::read_card(tuned.to_str().unwrap()).unwrap().is_some_and(|c| c.family == crate::spec::CARD_FAMILY));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cancelled run stops between steps and says how far it got.
    #[test]
    fn a_cancelled_training_run_stops_between_steps() {
        let dir = std::env::temp_dir().join(format!("glmdsa-caps-cancel-{}", std::process::id()));
        let data = dir.join("data");
        write_dataset(&data);
        let mut inv = train_inv(&data, &dir.join("glm.safetensors")).set("steps", json!(50));
        inv.cancel = capability::CancelToken::armed();
        inv.cancel.cancel();
        let err = run(&registry(None), "train", inv).unwrap_err();
        assert!(err.contains("cancelled after step 1"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_hugging_face_directory_is_refused_by_the_actions_that_need_a_brain_checkpoint() {
        let dir = std::env::temp_dir().join(format!("glmdsa-caps-hf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let reg = registry(Some(dir.to_str().unwrap()));
        let err = run(&reg, "eval", Invocation::new().set("data", json!("/nonexistent"))).unwrap_err();
        assert!(err.contains("import"), "{err}");
        let err = run(&reg, "finetune", Invocation::new().set("data", json!("/nonexistent")).set("save", json!("/nonexistent/out.safetensors"))).unwrap_err();
        assert!(err.contains("import"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `import` writes a brain checkpoint the store recognises, carded under the
    /// id the caller gave; a missing source is a named error.
    #[test]
    fn import_converts_a_hugging_face_directory_into_a_carded_checkpoint() {
        let hf = crate::import::tests::build_tiny_hf_dir("caps-import");
        let dest = hf.join("out").join("imported.safetensors");
        let inv = Invocation::new().set("source", json!(hf.to_str().unwrap())).set("dest", json!(dest.to_str().unwrap())).set("id", json!("acme/glm-tiny"));
        let out = run(&registry(None), "import", inv).unwrap();
        assert_eq!(out.outputs["dest"].as_str(), dest.to_str());
        let card = checkpoint::st::read_card(dest.to_str().unwrap()).unwrap().expect("the checkpoint carries a card");
        assert_eq!((card.id.as_str(), card.family.as_str()), ("acme/glm-tiny", crate::spec::CARD_FAMILY));
        let err = run(&registry(None), "import", Invocation::new().set("source", json!("/nonexistent")).set("dest", json!(dest.to_str().unwrap()))).unwrap_err();
        assert!(err.contains("import"), "{err}");
        let _ = std::fs::remove_dir_all(&hf);
    }
}
