// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain finetune` - supervised fine-tuning of a chat model through the SDK's generic trainers
//! ([`brain::ChatFineTune`], [`brain::ChatFullFineTune`]); the architecture is a flag, not a token.
//!
//! Swedish Embedded AB implements solutions for adapting open-weight language models to a team's own data on the
//! memory it actually has, for its clients. If your team needs expertise in fine-tuning large language models of
//! any architecture then you can procure our services by sending an email to info@swedishembedded.com.

const USAGE: &str = "\
brain finetune BASE --dataset FILE --out DIR [options]

  Fine-tune a chat model on a chat JSONL dataset: with --lora RANK a LoRA adapter over the frozen base
  (written to DIR/adapter.safetensors, which `brain serve` loads beside the base), without it every
  weight (written as a checkpoint directory under DIR).

  BASE               a checkpoint file or directory, or a vendor/repo in the model store.
  --dataset FILE     chat JSONL: messages with `train` marking the supervised turns.
  --out DIR          where the prepared dataset, the result and its training record go.
  --lora RANK        train an adapter of this rank (default: a full fine-tune).
  --alpha A          the adapter's scale (default: the rank).
  --continue FILE    continue the adapter in FILE, at its own rank.
  --adapter-id ID    the id written on the adapter's card.
  --arch ID          the base's architecture (default: what its config names).
  --models-dir DIR   where a vendor/repo base resolves.
  --steps N          optimizer steps (default 100).
  --lr X             the peak learning rate.
  --weight-decay X   AdamW weight decay.
  --grad-accum N     micro-batches per step.
  --seed N           the seed of the initial adapter and the batch order (default random, reported).
  --max-block N      the longest record, in tokens.
  --base-format F    the frozen base's format for an adapter: f32|bf16|int8|nf4|fp8, as far as the
                     architecture's trainer holds it (default: chosen from the device and what fits).
  --activations P    what the backward keeps of the forward: keep|recompute|bf16|fp8.
  --no-thinking      render records without the reasoning block.

Run on a device with `--device` (default: every device brain finds).
";

#[derive(Debug, Default, PartialEq)]
struct Args {
    base: String,
    dataset: Option<String>,
    out: Option<String>,
    lora: Option<u32>,
    alpha: Option<f32>,
    continue_from: Option<String>,
    adapter_id: Option<String>,
    arch: Option<String>,
    models_dir: Option<String>,
    steps: Option<u32>,
    lr: Option<f32>,
    weight_decay: Option<f32>,
    grad_accum: Option<u32>,
    seed: Option<u64>,
    max_block: Option<u32>,
    base_format: Option<String>,
    activations: Option<model::saved::SavedPolicy>,
    no_thinking: bool,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 0;
    let need = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("brain finetune: {flag} needs a value"))
    };
    fn number<T: std::str::FromStr>(v: &str, flag: &str) -> Result<T, String> {
        v.parse().map_err(|_| format!("brain finetune: {flag} wants a number, got '{v}'"))
    }
    while i < argv.len() {
        let flag = argv[i].clone();
        match flag.as_str() {
            "--dataset" => a.dataset = Some(need(&mut i, &flag)?),
            "--out" => a.out = Some(need(&mut i, &flag)?),
            "--lora" => a.lora = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--alpha" => a.alpha = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--continue" => a.continue_from = Some(need(&mut i, &flag)?),
            "--adapter-id" => a.adapter_id = Some(need(&mut i, &flag)?),
            "--arch" => a.arch = Some(need(&mut i, &flag)?),
            "--models-dir" => a.models_dir = Some(need(&mut i, &flag)?),
            "--steps" => a.steps = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--lr" => a.lr = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--weight-decay" => a.weight_decay = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--grad-accum" => a.grad_accum = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--seed" => a.seed = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--max-block" => a.max_block = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--base-format" => a.base_format = Some(need(&mut i, &flag)?),
            "--activations" => a.activations = Some(crate::args::parse_activations(&need(&mut i, &flag)?).map_err(|e| format!("brain finetune: {e}"))?),
            "--no-thinking" => a.no_thinking = true,
            "-h" | "--help" => return Err(String::new()),
            other if other.starts_with('-') => return Err(format!("brain finetune: unknown option '{other}'")),
            other if a.base.is_empty() => a.base = other.to_string(),
            other => return Err(format!("brain finetune: unexpected argument '{other}'")),
        }
        i += 1;
    }
    if a.base.is_empty() {
        return Err("brain finetune: no base given".to_string());
    }
    if a.lora.is_none() {
        // Each of these shapes an adapter; a full fine-tune that was given one would silently ignore it.
        for (flag, given) in [("--alpha", a.alpha.is_some()), ("--continue", a.continue_from.is_some()), ("--adapter-id", a.adapter_id.is_some()), ("--base-format", a.base_format.is_some())] {
            if given {
                return Err(format!("brain finetune: {flag} applies to an adapter; give --lora RANK"));
            }
        }
    }
    Ok(a)
}

fn lora(a: &Args, rank: u32) -> brain::ChatFineTune {
    let mut t = brain::ChatFineTune::from_pretrained(a.base.clone()).rank(rank).thinking(!a.no_thinking);
    if let Some(v) = &a.dataset {
        t = t.dataset(v);
    }
    if let Some(v) = &a.out {
        t = t.out_dir(v);
    }
    if let Some(v) = a.alpha {
        t = t.alpha(v);
    }
    if let Some(v) = &a.continue_from {
        t = t.continue_from(v);
    }
    if let Some(v) = &a.adapter_id {
        t = t.adapter_id(v.clone());
    }
    if let Some(v) = &a.arch {
        t = t.arch(v.clone());
    }
    if let Some(v) = &a.models_dir {
        t = t.models_dir(v.clone());
    }
    if let Some(v) = a.steps {
        t = t.steps(v);
    }
    if let Some(v) = a.lr {
        t = t.lr(v);
    }
    if let Some(v) = a.weight_decay {
        t = t.weight_decay(v);
    }
    if let Some(v) = a.grad_accum {
        t = t.grad_accum(v);
    }
    if let Some(v) = a.seed {
        t = t.seed(v);
    }
    if let Some(v) = a.max_block {
        t = t.max_block(v);
    }
    if let Some(v) = &a.base_format {
        t = t.base_format(v.clone());
    }
    if let Some(v) = a.activations {
        t = t.activations(v);
    }
    t
}

fn full(a: &Args) -> brain::ChatFullFineTune {
    let mut t = brain::ChatFullFineTune::from_pretrained(a.base.clone()).thinking(!a.no_thinking);
    if let Some(v) = &a.dataset {
        t = t.dataset(v);
    }
    if let Some(v) = &a.out {
        t = t.out_dir(v);
    }
    if let Some(v) = &a.arch {
        t = t.arch(v.clone());
    }
    if let Some(v) = &a.models_dir {
        t = t.models_dir(v.clone());
    }
    if let Some(v) = a.steps {
        t = t.steps(v);
    }
    if let Some(v) = a.lr {
        t = t.lr(v);
    }
    if let Some(v) = a.weight_decay {
        t = t.weight_decay(v);
    }
    if let Some(v) = a.grad_accum {
        t = t.grad_accum(v);
    }
    if let Some(v) = a.seed {
        t = t.seed(v);
    }
    if let Some(v) = a.max_block {
        t = t.max_block(v);
    }
    if let Some(v) = a.activations {
        t = t.train_precision(model::TrainPrecision { activations: Some(v), ..Default::default() });
    }
    t
}

fn losses(initial: Option<f32>, last: Option<f32>) -> String {
    let show = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.4}"));
    format!("loss {} -> {}", show(initial), show(last))
}

pub fn run_finetune(argv: &[String]) -> ! {
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
    let code = match args.lora {
        Some(rank) => match lora(&args, rank).run_with(&brain::CancelToken::default(), |p| eprintln!("step {}: loss {:.4}", p.step, p.loss)) {
            Ok(o) => {
                println!("status: {:?}, {} steps, {}", o.status, o.steps_completed, losses(o.initial_loss, o.final_loss));
                if let Some(p) = &o.adapter {
                    println!("adapter: {}", p.display());
                }
                if let Some(p) = &o.resume_state {
                    println!("resume state: {}", p.display());
                }
                0
            }
            Err(e) => {
                eprintln!("brain finetune: {e}");
                1
            }
        },
        None => match full(&args).run_with(&brain::CancelToken::default(), |p| eprintln!("step {}: loss {:.4}", p.step, p.loss)) {
            Ok(o) => {
                println!("status: {:?}, {} steps, {}", o.status, o.steps_completed, losses(o.initial_loss, o.final_loss));
                if let Some(p) = &o.checkpoint {
                    println!("checkpoint: {}", p.display());
                }
                0
            }
            Err(e) => {
                eprintln!("brain finetune: {e}");
                1
            }
        },
    };
    crate::drain_before_exit(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn a_full_fine_tune_refuses_the_flags_that_shape_an_adapter() {
        for flag in ["--alpha 4", "--continue a.safetensors", "--adapter-id x", "--base-format nf4"] {
            let err = parse(&argv(&format!("base --dataset d --out o {flag}"))).unwrap_err();
            assert!(err.contains("--lora"), "{flag}: {err}");
        }
        assert!(parse(&argv("base --dataset d --out o --lora 4 --alpha 8 --base-format nf4")).is_ok());
    }

    #[test]
    fn a_number_that_does_not_parse_is_an_error_naming_the_flag() {
        let err = parse(&argv("base --steps many")).unwrap_err();
        assert!(err.contains("--steps") && err.contains("many"), "{err}");
    }

    #[test]
    fn the_usage_names_every_flag_the_parser_takes() {
        for flag in ["--dataset", "--out", "--lora", "--alpha", "--continue", "--adapter-id", "--arch", "--models-dir", "--steps", "--lr", "--weight-decay", "--grad-accum", "--seed", "--max-block", "--base-format", "--activations", "--no-thinking"] {
            assert!(USAGE.contains(flag), "{flag} is parsed but absent from the usage");
        }
    }
}
