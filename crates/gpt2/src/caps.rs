// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The dense GPT baseline's served actions: `train` it on a tokenized
//! dataset (alone, or as one rank of a distributed job), `generate` text
//! from a checkpoint, and `eval` a checkpoint's validation perplexity and
//! exact-match accuracy.
//!
//! The model is brain's own baseline with no released checkpoint and no
//! store spec, so a checkpoint and a dataset are named by path (host side).
//!
//! Swedish Embedded AB implements training small language models on its
//! clients' own data and machines. If your team needs expertise in language
//! model training, from one card to a cluster, then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::binio::Meta;
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use model::job_params::JobRequest;
use serde_json::json;

use crate::{Gpt, GptConfig, TrainOpts};

pub const MODEL: &str = "brain/gpt2";

/// Largest model width, depth or head count a run may ask for.
const MAX_DIM: f64 = 16_384.0;
/// Most sequences one step may hold, or accumulate.
const MAX_BATCH: f64 = 4_096.0;
/// Longest continuation one call may generate.
const MAX_NEW: f64 = 1_000_000.0;
/// Most batches or samples one evaluation may score.
const MAX_EVAL: f64 = 1_000_000.0;
/// Validation batches the perplexity averages over, and held-out lines the
/// exact match decodes, when the caller names none.
const EVAL_BATCHES: i64 = 20;
const EVAL_SAMPLES: i64 = 200;

fn host_path(name: &str, what: &str) -> ParamSpec {
    ParamSpec::new(name, ParamType::Str, what).host_resolved()
}

pub fn train_spec() -> ActionSpec {
    let d = TrainOpts::default();
    ActionSpec::new("train", "train the GPT on a tokenized dataset (train.bin, val.bin, meta.json), alone or as one rank of a distributed job")
        .training()
        .streaming()
        .from_scratch()
        .param(host_path("data", "the dataset directory (`brain data` prepares one)").required())
        .param(host_path("save", "where to write the checkpoint (.safetensors); omit to train without keeping it"))
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(d.steps)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("batch", ParamType::Int, "sequences per micro-step").default(json!(d.batch_size)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("block", ParamType::Int, "tokens per sequence (the model's context)").default(json!(d.block_size)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate").default(json!(d.lr)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("warmup", ParamType::Int, "linear warmup steps").default(json!(d.warmup)).min(0.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("eval_interval", ParamType::Int, "steps between validation losses; 0 never").default(json!(d.eval_interval)).min(0.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("grad_accum", ParamType::Int, "micro-steps accumulated per optimizer step").default(json!(d.grad_accum)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("layers", ParamType::Int, "transformer layers").default(json!(4)).min(1.0).max(MAX_DIM))
        .param(ParamSpec::new("d_model", ParamType::Int, "model width").default(json!(128)).min(1.0).max(MAX_DIM))
        .param(ParamSpec::new("heads", ParamType::Int, "attention heads").default(json!(4)).min(1.0).max(MAX_DIM))
        .param(ParamSpec::new("mask", ParamType::Str, "train only on what follows this character on each line (the answer after '=' of an LHS=RHS dataset)"))
        .param(ParamSpec::new("align", ParamType::Bool, "start every sequence at a line boundary").default(json!(false)))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the initialization and the batches (omit for random, or the job plan's digest in a distributed run)").min(0.0))
        .param(ParamSpec::new(
            "activations",
            ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
            "what the backward keeps of the forward (default: the model's own)",
        ))
        .params(model::job_params::specs())
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("generate", "sample text from a checkpoint (character vocabulary, KV-cached decoding)")
        .param(host_path("weights", "the checkpoint (.safetensors) `train` wrote").required())
        .param(host_path("data", "a dataset directory whose meta.json holds the vocabulary, for a checkpoint that does not embed one"))
        .param(ParamSpec::new("prompt", ParamType::Str, "the text to continue").default(json!("\n")))
        .param(ParamSpec::new("max_new", ParamType::Int, "characters to generate").default(json!(200)).min(0.0).max(MAX_NEW))
        .param(ParamSpec::new("temperature", ParamType::Float, "sampling temperature").default(json!(0.8)).min(0.0).max(10.0))
        .param(ParamSpec::new("top_k", ParamType::Int, "sample among the k most likely characters").default(json!(40)).min(1.0).max(capability::bounds::TOP_K))
        .param(ParamSpec::new("seed", ParamType::Int, "sampling seed (omit for random; the outcome echoes it)").min(0.0))
}

pub fn eval_spec() -> ActionSpec {
    ActionSpec::new("eval", "validation perplexity, and exact-match accuracy on the held-out tail of an LHS=RHS dataset")
        .param(host_path("weights", "the checkpoint (.safetensors)").required())
        .param(host_path("data", "the dataset directory").required())
        .param(ParamSpec::new("batches", ParamType::Int, "validation batches for the perplexity").default(json!(EVAL_BATCHES)).min(1.0).max(MAX_EVAL))
        .param(ParamSpec::new("samples", ParamType::Int, "held-out lines decoded for the exact match").default(json!(EVAL_SAMPLES)).min(1.0).max(MAX_EVAL))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the batches and the lines drawn (omit for random; the outcome echoes it)").min(0.0))
}

pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "Dense GPT baseline (character vocabulary): brain's reference decoder for training and distributed-training work.", vec![train_spec(), generate_spec(), eval_spec()])
}

fn path(inv: &Invocation, name: &str, action: &str) -> Result<PathBuf, String> {
    inv.get_str(name).map(PathBuf::from).ok_or_else(|| format!("gpt2 {action}: missing required param '{name}'"))
}

/// A checkpoint's character vocabulary: embedded, else the dataset's meta.json.
fn vocabulary(weights: &str, data: Option<&Path>) -> Result<Vec<char>, String> {
    if let Some(itos) = Gpt::load_itos(weights) {
        return Ok(itos);
    }
    let dir = data.ok_or("the checkpoint embeds no vocabulary; give data, a dataset directory whose meta.json holds it")?;
    let meta = dir.join("meta.json");
    let text = std::fs::read_to_string(&meta).map_err(|e| format!("{}: {e}", meta.display()))?;
    Ok(Meta::from_json(&text).map_err(|e| format!("{}: {e}", meta.display()))?.itos)
}

fn train(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let dir = path(inv, "data", "train")?;
    let save = inv.get_str("save").map(PathBuf::from);
    let job = JobRequest::from_invocation(inv)?;
    let mut o = TrainOpts::default();
    let n = |name: &str, default: u32| inv.get_i64(name).map_or(default, |v| v.max(0) as u32);
    o.steps = n("steps", o.steps).max(1);
    o.batch_size = n("batch", o.batch_size).max(1);
    o.block_size = n("block", o.block_size).max(1);
    o.lr = inv.get_f64("lr").map_or(o.lr, |v| v as f32);
    o.warmup = n("warmup", o.warmup);
    o.eval_interval = n("eval_interval", o.eval_interval);
    o.grad_accum = n("grad_accum", o.grad_accum).max(1);
    o.decay_iters = o.steps;
    o.precision.activations = inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?;
    if let Some(m) = inv.get_str("mask") {
        let mut chars = m.chars();
        o.mask_before = Some(chars.next().filter(|_| chars.next().is_none()).ok_or_else(|| format!("gpt2 train: mask {m:?} is not one character"))?);
        o.mask_per_line = true;
    }
    o.align_to_lines = inv.get_bool("align").unwrap_or(false);
    // Every rank of a job must start from the same weights and draw the
    // same batch stream: without a seed, the plan every rank holds names it.
    o.seed = match (inv.seed()?, &job) {
        (Some(seed), _) => seed,
        (None, Some(job)) => job.plan_seed()?,
        (None, None) => data::rng::random_seed(),
    };
    let cfg = GptConfig { vocab: 0, block_size: o.block_size, n_layers: n("layers", 4).max(1), d_model: n("d_model", 128).max(1), n_heads: n("heads", 4).max(1), d_ff: 0 };
    let report = match &job {
        // A distributed run stops together or not at all: one rank leaving
        // would stall its peers' next reduction.
        Some(job) => job.run(|sync| crate::train_distributed(&dir, cfg, &o, save.as_deref(), sync).map(|(report, _)| report).map_err(|e| e.to_string()))?,
        None => model::train::fit_watched::<Gpt>(&dir, cfg, &o, save.as_deref(), &mut |s| {
            progress(Progress::step(s.step, s.steps, format!("loss {:.4}", s.loss)));
            !inv.cancel.is_cancelled()
        })
        .map_err(|e| format!("gpt2 train: {e}"))?,
    };
    if report.interrupted {
        return Err(format!("cancelled after step {}", report.steps_completed));
    }
    Ok(Outcome::new()
        .set("initial_loss", json!(report.initial_loss))
        .set("final_loss", json!(report.final_loss))
        .set("steps", json!(report.steps_completed))
        .set("seed", json!(o.seed))
        .set("rank", json!(job.as_ref().map(|j| j.rank)))
        .set("saved", json!(save.map(|p| p.display().to_string()))))
}

fn generate(inv: &Invocation) -> ActionResult {
    let weights = path(inv, "weights", "generate")?.display().to_string();
    let data = inv.get_str("data").map(PathBuf::from);
    let tok = CharTokenizer::from_itos(vocabulary(&weights, data.as_deref())?);
    let block = GptConfig::from_json(&checkpoint::read_config(&weights)).block_size;
    let model = Gpt::load(&weights, 1, block);
    let prompt = inv.get_str("prompt").filter(|p| !p.is_empty()).unwrap_or_else(|| "\n".into());
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let ids = crate::sample::generate_kv(
        &model,
        &tok.encode(&prompt),
        inv.get_i64("max_new").unwrap_or(200).max(0) as usize,
        inv.get_f64("temperature").unwrap_or(0.8) as f32,
        inv.get_i64("top_k").unwrap_or(40).max(1) as usize,
        &mut Rng::new(seed),
    );
    Ok(Outcome::new().set("text", json!(format!("{prompt}{}", tok.decode(&ids)))).set("seed", json!(seed)))
}

fn eval(inv: &Invocation) -> ActionResult {
    let weights = path(inv, "weights", "eval")?.display().to_string();
    let dir = path(inv, "data", "eval")?;
    let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    let ppl = crate::eval::val_perplexity(&weights, &dir, inv.get_i64("batches").unwrap_or(EVAL_BATCHES).max(1) as usize, seed).map_err(|e| format!("gpt2 eval: perplexity: {e}"))?;
    let mut outcome = Outcome::new().set("val_perplexity", json!(ppl)).set("seed", json!(seed));
    if dir.join("meta.json").exists() {
        let (acc, n) = crate::eval::exact_match(&weights, &dir, inv.get_i64("samples").unwrap_or(EVAL_SAMPLES).max(1) as usize, seed).map_err(|e| format!("gpt2 eval: exact match: {e}"))?;
        if n > 0 {
            outcome = outcome.set("exact_match", json!(acc)).set("exact_match_samples", json!(n));
        }
    }
    Ok(outcome)
}

#[derive(Default)]
pub struct Gpt2Provider;

impl Provider for Gpt2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        manifest().actions.into_iter().find(|a| a.name == name).map(|spec| Arc::new(Gpt2Action { spec }) as Arc<dyn Action>)
    }
}

struct Gpt2Action {
    spec: ActionSpec,
}

impl Action for Gpt2Action {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.spec.name.as_str() {
            "train" => train(inv, progress),
            "generate" => generate(inv),
            "eval" => eval(inv),
            other => Err(format!("gpt2: unknown action {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Registry;

    /// A char dataset trains for a few steps; the checkpoint embeds its
    /// vocabulary, generates deterministically from a seed and evaluates.
    #[test]
    fn train_then_generate_and_eval_through_the_actions() {
        let dir = std::env::temp_dir().join(format!("gpt2-caps-{}", std::process::id()));
        let data_dir = dir.join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let text = "1+1=2\n2+2=4\n3+3=6\n".repeat(60);
        std::fs::write(data_dir.join("input.txt"), &text).unwrap();
        let mut itos: Vec<char> = text.chars().collect();
        itos.sort_unstable();
        itos.dedup();
        let tok = CharTokenizer::from_itos(itos.clone());
        let ids: Vec<u16> = tok.encode(&text).into_iter().map(|t| t as u16).collect();
        let split = ids.len() * 9 / 10;
        data::binio::write_u16_bin(&data_dir.join("train.bin"), &ids[..split]).unwrap();
        data::binio::write_u16_bin(&data_dir.join("val.bin"), &ids[split..]).unwrap();
        std::fs::write(data_dir.join("meta.json"), Meta { vocab_size: itos.len(), itos }.to_json()).unwrap();

        let save = dir.join("gpt.safetensors");
        let mut reg = Registry::new();
        reg.register(Arc::new(Gpt2Provider));
        let run = |action: &str, inv: Invocation| reg.run(MODEL, action, inv, &mut |_| {}).unwrap_or_else(|e| panic!("{action}: {e}"));
        let d = json!(data_dir.to_str().unwrap());
        let trained = run(
            "train",
            Invocation::new().set("data", d.clone()).set("save", json!(save.to_str().unwrap())).set("steps", json!(3)).set("batch", json!(2)).set("block", json!(16)).set("layers", json!(1)).set("d_model", json!(16)).set("heads", json!(2)).set("eval_interval", json!(0)).set("warmup", json!(1)).set("seed", json!(2)),
        );
        assert_eq!(trained.outputs["steps"].as_u64(), Some(3));
        let w = json!(save.to_str().unwrap());
        let gen = |seed: u64| run("generate", Invocation::new().set("weights", w.clone()).set("prompt", json!("1+")).set("max_new", json!(5)).set("seed", json!(seed))).outputs["text"].clone();
        assert_eq!(gen(7), gen(7), "the seed reproduces the sample");
        assert!(gen(7).as_str().unwrap().starts_with("1+"));
        let scored = run("eval", Invocation::new().set("weights", w).set("data", d).set("batches", json!(1)).set("samples", json!(3)).set("seed", json!(1)));
        assert!(scored.outputs["val_perplexity"].as_f64().is_some_and(|p| p > 1.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_multi_character_mask_and_a_partial_job_are_refused() {
        let mut reg = Registry::new();
        reg.register(Arc::new(Gpt2Provider));
        let e = reg.run(MODEL, "train", Invocation::new().set("data", json!("/nonexistent")).set("mask", json!("=>")), &mut |_| {}).unwrap_err();
        assert!(e.contains("one character"), "{e}");
        let e = reg.run(MODEL, "train", Invocation::new().set("data", json!("/nonexistent")).set("job_plan", json!("p")), &mut |_| {}).unwrap_err();
        assert!(e.contains("go together"), "{e}");
    }
}
