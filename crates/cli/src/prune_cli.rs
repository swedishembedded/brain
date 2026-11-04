// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain prune` - structured pruning of a chat model, over
//! [`brain::ModelPruner`].
//!
//! Swedish Embedded AB implements solutions for shrinking trained models to
//! the compute a deployment has, by removing what they need least, for its
//! clients. If your team needs expertise in structured pruning and model
//! compression then you can procure our services by sending an email to
//! info@swedishembedded.com.

const USAGE: &str = "\
brain prune BASE --dataset FILE --out DIR [options]

  Remove the least important channels of every prunable dimension of a chat
  model (qwen3: each layer's MLP hidden channels), ranked by the estimated
  loss of removing each on the dataset, and write the narrower model to
  DIR/checkpoint, a checkpoint directory brain serves like any other. The
  original and the pruned model are measured on held-out records; DIR/prune.json
  holds the outcome.

  BASE               a checkpoint file or directory, or a vendor/repo in the
                     model store.
  --dataset FILE     chat JSONL representative of what will be served; a
                     quarter of its records is held out unless --held-out is
                     given.
  --out DIR          where the prepared dataset, the checkpoint and prune.json go.
  --keep F           the share of each dimension's channels kept (default 0.75).
  --keep-channels N  keep exactly N channels of each dimension.
  --held-out FILE    chat JSONL to measure on; all of --dataset ranks.
  --arch ID          the base's architecture (default qwen3).
  --models-dir DIR   where a vendor/repo base resolves.
  --seed N           the calibration/held-out split's seed (default random).
  --max-block N      the longest record, in tokens.
  --no-thinking      render records without the reasoning block.
  --recover-steps N  then distill a LoRA adapter over the pruned checkpoint
                     for N steps, the original as the teacher, written to
                     DIR/recovery.safetensors (default 0: none).
  --recover-lr LR    the recovery adapter's learning rate (default 1e-4).
  --recover-rank R   the recovery adapter's rank (default 16).
";

/// How narrow the pruned dimensions become; at most one form.
#[derive(Debug, PartialEq)]
enum Keep {
    Fraction(f64),
    Channels(usize),
}

#[derive(Debug, PartialEq)]
struct Args {
    base: String,
    dataset: Option<String>,
    out: Option<String>,
    held_out: Option<String>,
    keep: Option<Keep>,
    arch: Option<String>,
    models_dir: Option<String>,
    seed: Option<u64>,
    max_block: Option<u32>,
    thinking: bool,
    recover_steps: u32,
    recover_lr: Option<f32>,
    recover_rank: Option<u32>,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args { base: String::new(), dataset: None, out: None, held_out: None, keep: None, arch: None, models_dir: None, seed: None, max_block: None, thinking: true, recover_steps: 0, recover_lr: None, recover_rank: None };
    let mut i = 0;
    let need = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("brain prune: {flag} needs a value"))
    };
    let number = |v: &str, flag: &str| format!("brain prune: {flag} wants a number, got '{v}'");
    while i < argv.len() {
        match argv[i].as_str() {
            "--dataset" => a.dataset = Some(need(&mut i, "--dataset")?),
            "--out" => a.out = Some(need(&mut i, "--out")?),
            "--held-out" => a.held_out = Some(need(&mut i, "--held-out")?),
            "--keep" | "--keep-channels" if a.keep.is_some() => return Err("brain prune: give one of --keep and --keep-channels".to_string()),
            "--keep" => {
                let v = need(&mut i, "--keep")?;
                a.keep = Some(Keep::Fraction(v.parse().map_err(|_| number(&v, "--keep"))?));
            }
            "--keep-channels" => {
                let v = need(&mut i, "--keep-channels")?;
                a.keep = Some(Keep::Channels(v.parse().map_err(|_| number(&v, "--keep-channels"))?));
            }
            "--arch" => a.arch = Some(need(&mut i, "--arch")?),
            "--models-dir" => a.models_dir = Some(need(&mut i, "--models-dir")?),
            "--seed" => {
                let v = need(&mut i, "--seed")?;
                a.seed = Some(v.parse().map_err(|_| number(&v, "--seed"))?);
            }
            "--max-block" => {
                let v = need(&mut i, "--max-block")?;
                a.max_block = Some(v.parse().map_err(|_| number(&v, "--max-block"))?);
            }
            "--no-thinking" => a.thinking = false,
            "--recover-steps" => {
                let v = need(&mut i, "--recover-steps")?;
                a.recover_steps = v.parse().map_err(|_| number(&v, "--recover-steps"))?;
            }
            "--recover-lr" => {
                let v = need(&mut i, "--recover-lr")?;
                a.recover_lr = Some(v.parse().map_err(|_| number(&v, "--recover-lr"))?);
            }
            "--recover-rank" => {
                let v = need(&mut i, "--recover-rank")?;
                a.recover_rank = Some(v.parse().map_err(|_| number(&v, "--recover-rank"))?);
            }
            "-h" | "--help" => return Err(String::new()),
            other if other.starts_with('-') => return Err(format!("brain prune: unknown option '{other}'")),
            other if a.base.is_empty() => a.base = other.to_string(),
            other => return Err(format!("brain prune: unexpected argument '{other}'")),
        }
        i += 1;
    }
    if a.base.is_empty() {
        return Err("brain prune: no base given".to_string());
    }
    Ok(a)
}

fn pruner(a: &Args) -> brain::ModelPruner {
    let mut p = brain::ModelPruner::from_pretrained(a.base.clone()).thinking(a.thinking).recover(a.recover_steps);
    if let Some(d) = &a.dataset {
        p = p.dataset(d);
    }
    if let Some(d) = &a.out {
        p = p.out_dir(d);
    }
    if let Some(d) = &a.held_out {
        p = p.held_out(d);
    }
    p = match a.keep {
        Some(Keep::Fraction(f)) => p.keep(f),
        Some(Keep::Channels(n)) => p.keep_channels(n),
        None => p,
    };
    if let Some(id) = &a.arch {
        p = p.arch(id.clone());
    }
    if let Some(d) = &a.models_dir {
        p = p.models_dir(d.clone());
    }
    if let Some(s) = a.seed {
        p = p.seed(s);
    }
    if let Some(m) = a.max_block {
        p = p.max_block(m);
    }
    if let Some(lr) = a.recover_lr {
        p = p.recover_lr(lr);
    }
    if let Some(r) = a.recover_rank {
        p = p.recover_rank(r);
    }
    p
}

pub fn run_prune(argv: &[String]) -> ! {
    let args = match parse(argv) {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("{msg}\n");
            }
            print!("{USAGE}");
            std::process::exit(if msg.is_empty() { 0 } else { 2 });
        }
    };
    let outcome = match pruner(&args).run_with(crate::optimize_cli::report_progress) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("brain prune: {e}");
            std::process::exit(1);
        }
    };
    let r = &outcome.report;
    println!("calibration records {}, held-out records {}, seed {}", outcome.calibration_records, outcome.held_out_records, outcome.seed);
    println!("{}: {} -> {} channels per dimension; parameters {} -> {}", r.config_key, r.width_before, r.width_after, r.params_before, r.params_after);
    println!("trainer build: held-out loss {:.6} -> {:.6}", r.original_loss, r.pruned_loss);
    println!("measured on the {}:", outcome.measured_on);
    for d in &outcome.deployed {
        println!("  {:<10} {:.6}", d.label, d.loss);
    }
    if let Some(rec) = &outcome.recovery {
        println!("recovery adapter {} ({} steps, rank {}): held-out loss {:.6} -> {:.6} on the trainer build", rec.adapter.display(), rec.steps, rec.rank, rec.plan_loss, rec.recovered_loss);
    }
    println!("checkpoint: {}", outcome.checkpoint.display());
    println!("report: {}", outcome.report_path.display());
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_the_documented_options() {
        let a = parse(&s(&["Qwen/Qwen3-0.6B", "--dataset", "d.jsonl", "--out", "run", "--keep", "0.6", "--seed", "3", "--recover-steps", "20", "--recover-rank", "8"])).unwrap();
        assert_eq!((a.base.as_str(), a.dataset.as_deref(), a.out.as_deref()), ("Qwen/Qwen3-0.6B", Some("d.jsonl"), Some("run")));
        assert_eq!((a.keep, a.seed, a.recover_steps, a.recover_rank), (Some(Keep::Fraction(0.6)), Some(3), 20, Some(8)));
        assert_eq!(parse(&s(&["b", "--keep-channels", "512"])).unwrap().keep, Some(Keep::Channels(512)));
    }

    #[test]
    fn bad_input_is_refused_by_name() {
        assert!(parse(&s(&["b", "--keep", "0.5", "--keep-channels", "8"])).unwrap_err().contains("one of"));
        assert!(parse(&s(&["b", "--keep", "half"])).unwrap_err().contains("--keep"));
        assert!(parse(&s(&["--dataset", "d"])).unwrap_err().contains("no base"));
    }
}
