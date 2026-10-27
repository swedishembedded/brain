// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain gpt …` — train / evaluate / sample the dense GPT baseline. Uses the
//! shared `args` grammar; `infer` is the canonical verb for `gen`.
//!
//!   brain gpt train <data_dir> [--out F --steps N --batch B --block T --lr X
//!                               --layers L --d-model D --heads H --mask = --align
//!                               --job-plan FILE --job-rank N
//!                               --profile-collectives FILE]
//!   brain gpt infer --weights F [--data <dir>] [--prompt "..." --max-new N
//!                               --temp X --top-k K]
//!   brain gpt eval  --weights F --data <dir> [--batches N --samples M]

use std::path::{Path, PathBuf};

use data::binio::Meta;
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use gpt2::model::Gpt;
use gpt2::train::TrainOpts;
use gpt2::GptConfig;

use crate::args::{canon_verb, Args};

pub fn run_gpt(argv: &[String]) {
    match argv.first().map(|s| canon_verb(s)) {
        Some("train") => train(&argv[1..]),
        Some("infer") => gen(&argv[1..]),
        Some("eval") => eval(&argv[1..]),
        other => eprintln!("usage: brain gpt2 <train|infer|eval> ...  (got {other:?})"),
    }
}

fn eval(args: &[String]) {
    let mut a = Args::new(args);
    let weights = a.str_or("--weights", "");
    let data_dir = a.str_or("--data", "");
    let batches = a.usize_or("--batches", 20);
    let samples = a.usize_or("--samples", 200);
    let seed = a.seed("gpt2 eval");
    a.finish();
    if weights.is_empty() || data_dir.is_empty() {
        eprintln!("usage: brain gpt2 eval --weights F --data <dir> [--batches N --samples M]");
        return;
    }
    let dir = Path::new(&data_dir);
    match eval::gpt_val_perplexity(&weights, dir, batches, seed) {
        Ok(ppl) => println!("val_perplexity {ppl:.4}"),
        Err(e) => eprintln!("perplexity error: {e}"),
    }
    if dir.join("meta.json").exists() {
        match eval::gpt_exact_match(&weights, dir, samples, seed) {
            Ok((acc, n)) if n > 0 => println!("exact_match {:.4} ({}/{} held-out)", acc, (acc * n as f32).round() as usize, n),
            Ok(_) => {}
            Err(e) => eprintln!("exact-match error: {e}"),
        }
    }
}

fn train(args: &[String]) {
    let (activations, args) = crate::args::split_activations_or_exit(args, "brain gpt2 train");
    let mut a = Args::new(&args);
    let Some(dir) = a.positional() else {
        eprintln!("usage: brain gpt2 train <data_dir> [flags] {}", crate::args::ACTIVATIONS_USAGE);
        return;
    };
    let mut o = TrainOpts::default();
    let mut cfg = GptConfig { vocab: 0, block_size: o.block_size, n_layers: 4, d_model: 128, n_heads: 4, d_ff: 0 };
    let out = a.take_str("--out").map(PathBuf::from);
    o.precision.activations = activations;
    o.steps = a.u32_or("--steps", o.steps);
    o.batch_size = a.u32_or("--batch", o.batch_size);
    o.block_size = a.u32_or("--block", o.block_size);
    o.lr = a.f32_or("--lr", o.lr);
    o.warmup = a.u32_or("--warmup", o.warmup);
    o.eval_interval = a.u32_or("--eval-interval", o.eval_interval);
    o.grad_accum = a.u32_or("--grad-accum", o.grad_accum);
    cfg.n_layers = a.u32_or("--layers", cfg.n_layers);
    cfg.d_model = a.u32_or("--d-model", cfg.d_model);
    cfg.n_heads = a.u32_or("--heads", cfg.n_heads);
    if let Some(m) = a.char_opt("--mask") {
        o.mask_before = Some(m);
        o.mask_per_line = true;
    }
    o.align_to_lines = a.take_flag("--align");
    // The distributed pair. `--rank` is LoRA rank everywhere else in the CLI,
    // so this process's place in a job is `--job-rank`.
    let job_plan = a.take_str("--job-plan");
    let job_rank = a.take_str("--job-rank").map(|value| match value.parse::<usize>() {
        Ok(rank) => rank,
        Err(_) => {
            eprintln!("--job-rank {value:?} is not a rank number");
            std::process::exit(2);
        }
    });
    let job = match (job_plan, job_rank) {
        (None, None) => None,
        (Some(plan), Some(rank)) => Some((plan, rank)),
        _ => {
            eprintln!("usage: --job-plan FILE and --job-rank N must be given together (the plan every rank shares, and this rank's place in it)");
            std::process::exit(2);
        }
    };
    let profile = match crate::collective_profile::take_flag(&mut a, job.is_some()) {
        Ok(profile) => profile,
        Err(message) => {
            eprintln!("usage: {message}");
            std::process::exit(2);
        }
    };
    // Every rank of a job must start from the same weights and draw the same
    // batch stream, so it must train with the same seed. A rank that drew its
    // own random one would diverge from its peers before the first reduction.
    // The plan file is the value every rank shares, so its digest is the seed
    // when none was given.
    o.seed = match (job.as_ref().map(|(plan, _)| plan), a.seed_opt("--seed")) {
        (_, Some(seed)) => seed,
        (Some(plan), None) => match crate::args::plan_seed(Path::new(plan), "gpt2 train") {
            Ok(seed) => seed,
            Err(e) => {
                eprintln!("error: reading {plan}: {e}");
                std::process::exit(1);
            }
        },
        (None, None) => a.seed("gpt2 train"),
    };
    a.finish();
    o.decay_iters = o.steps;
    println!(
        "training gpt on {dir}: steps={} batch={} block={} layers={} d_model={} heads={}",
        o.steps, o.batch_size, o.block_size, cfg.n_layers, cfg.d_model, cfg.n_heads
    );
    let Some((plan, rank)) = job else {
        match gpt2::train::train(Path::new(&dir), cfg, &o, out.as_deref()) {
            Ok((i0, i1)) => println!("done: train loss {i0:.4} -> {i1:.4}"),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        return;
    };
    let context = match model::JobContext::from_plan_file(Path::new(&plan), rank) {
        Ok(context) => context,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "distributed job {:?} rank {}/{}",
        context.plan().job,
        context.rank(),
        context.plan().world_size
    );
    let collective = match context.collective() {
        Ok(collective) => collective,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    // A requested profile wraps this rank's transport; the wrapper lives until
    // the run returns so its snapshot covers the whole run. Without the flag
    // the group is the constructed collective itself and nothing is written.
    let timed = profile.as_ref().map(|_| model::TimedCollective::new(&collective));
    let group: &dyn model::Collective = match &timed {
        Some(timed) => timed,
        None => &collective,
    };
    let plan_identity = context.plan().identity();
    let sync = model::GradSync {
        collective: group,
        rank: context.rank(),
        checkpoint: Some(model::CheckpointIdentity { job: context.plan().job.as_str(), plan: &plan_identity }),
    };
    let started = std::time::Instant::now();
    let result = gpt2::train::train_distributed(Path::new(&dir), cfg, &o, out.as_deref(), sync);
    let run = started.elapsed();
    if let (Some(base), Some(timed)) = (&profile, &timed) {
        crate::collective_profile::write(base, context.rank(), context.plan().world_size, &context.plan().identity(), run, &timed.snapshot());
    }
    match result {
        Ok((report, _)) => {
            let final_loss = report.final_loss.unwrap_or(report.initial_loss);
            println!("done: train loss {:.4} -> {final_loss:.4}", report.initial_loss);
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn gen(args: &[String]) {
    let mut a = Args::new(args);
    let weights = a.str_or("--weights", "");
    let data_dir = a.str_or("--data", "");
    let prompt = a.str_or("--prompt", "");
    let max_new = a.usize_or("--max-new", 200);
    let temp = a.f32_or("--temp", 0.8);
    let top_k = a.usize_or("--top-k", 40);
    let seed = a.seed("gpt2 infer");
    a.finish();
    if weights.is_empty() {
        eprintln!("usage: brain gpt2 infer --weights F [--data <dir>] [--prompt ... --max-new N --temp X --top-k K]");
        return;
    }
    let itos = match gpt2::model::Gpt::load_itos(&weights) {
        Some(itos) => itos,
        None => {
            if data_dir.is_empty() {
                eprintln!(
                    "this checkpoint has no embedded vocab (trained before vocab embedding, \
                     or a BPE model); pass --data <dir> with its meta.json"
                );
                return;
            }
            let meta_path = Path::new(&data_dir).join("meta.json");
            match std::fs::read_to_string(&meta_path).ok().and_then(|s| Meta::from_json(&s).ok()) {
                Some(m) => m.itos,
                None => {
                    eprintln!("gen needs a char vocab: none embedded and no meta.json at {}", meta_path.display());
                    return;
                }
            }
        }
    };
    let tok = CharTokenizer::from_itos(itos);
    let model = Gpt::load(&weights, 1, model_block(&weights));
    let prompt_text = if prompt.is_empty() { "\n" } else { prompt.as_str() };
    let prompt_ids: Vec<u32> = tok.encode(prompt_text);
    let mut rng = Rng::new(seed);
    let gen = gpt2::sample::generate_kv(&model, &prompt_ids, max_new, temp, top_k, &mut rng);
    print!("{prompt_text}");
    print!("{}", tok.decode(&gen));
    println!();
}

/// Read the block size from a checkpoint header so the decoder is sized right.
fn model_block(weights: &str) -> u32 {
    GptConfig::from_json(&checkpoint::read_config(weights)).block_size
}

#[cfg(test)]
mod tests {
    use crate::args::plan_seed;

    /// The same plan file must give every rank the same seed, or the ranks
    /// start from different weights and train on different batch streams.
    #[test]
    fn the_plan_file_yields_one_seed_and_a_different_plan_a_different_one() {
        let dir = std::env::temp_dir().join(format!("brain-gpt2-plan-seed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.json");
        let b = dir.join("b.json");
        std::fs::write(&a, br#"{"job":"a"}"#).unwrap();
        std::fs::write(&b, br#"{"job":"b"}"#).unwrap();
        assert_eq!(plan_seed(&a, "gpt2 train").unwrap(), plan_seed(&a, "gpt2 train").unwrap(), "one plan, one seed");
        assert_ne!(plan_seed(&a, "gpt2 train").unwrap(), plan_seed(&b, "gpt2 train").unwrap(), "a different plan must not replay the same stream");
        std::fs::remove_dir_all(&dir).ok();
    }
}
