// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain optimize` - the weight precision search of a chat model, over
//! [`brain::ModelOptimizer`].
//!
//! Swedish Embedded AB implements solutions for fitting open-weight language
//! models into the memory a deployment has, by measurement, for its clients.
//! If your team needs expertise in mixed-precision model compression then you
//! can procure our services by sending an email to info@swedishembedded.com.

const USAGE: &str = "\
brain optimize BASE --dataset FILE --out DIR [options]

  Search the weight format of every linear of a chat model within a byte
  budget, then measure the plan, the original and every uniform format that
  fits the budget on the deployed kernels. Prints the --weight-format policy
  to serve and writes DIR/optimize.json.

  BASE               a checkpoint file or directory, or a vendor/repo in the
                     model store.
  --dataset FILE     chat JSONL representative of what will be served; a
                     quarter of its records is held out unless --held-out is
                     given.
  --out DIR          where the prepared dataset and optimize.json go.
  --held-out FILE    chat JSONL to measure on; all of --dataset calibrates.
  --budget FORMAT    the bytes of every linear at one format (default int8).
  --bits-per-weight B  mean stored bits per weight, scales included.
  --max-bytes N      bytes over every searched linear.
  --formats LIST     candidate formats, comma separated (default: every
                     frozen-base format the architecture's trainer holds that
                     the device holds as itself).
  --arch ID          the base's architecture (default qwen3).
  --models-dir DIR   where a vendor/repo base resolves.
  --seed N           the calibration/held-out split's seed (default random,
                     reported).
  --max-block N      the longest record, in tokens.
  --no-thinking      render records without the reasoning block.

Exit status: 0 when the plan measured better than every uniform format that
fits its budget (accepted), 3 when it did not, 1 or 2 on an error.
";

/// How the budget was given; at most one form.
#[derive(Debug, PartialEq)]
enum BudgetArg {
    Like(String),
    Bits(f64),
    Bytes(u64),
}

#[derive(Debug, PartialEq)]
struct Args {
    base: String,
    dataset: Option<String>,
    out: Option<String>,
    held_out: Option<String>,
    budget: Option<BudgetArg>,
    formats: Option<String>,
    arch: Option<String>,
    models_dir: Option<String>,
    seed: Option<u64>,
    max_block: Option<u32>,
    thinking: bool,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args { base: String::new(), dataset: None, out: None, held_out: None, budget: None, formats: None, arch: None, models_dir: None, seed: None, max_block: None, thinking: true };
    let mut i = 0;
    let need = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("brain optimize: {flag} needs a value"))
    };
    let number = |v: &str, flag: &str| format!("brain optimize: {flag} wants a number, got '{v}'");
    let set_budget = |a: &mut Args, b: BudgetArg| -> Result<(), String> {
        if a.budget.is_some() {
            return Err("brain optimize: give one of --budget, --bits-per-weight and --max-bytes".to_string());
        }
        a.budget = Some(b);
        Ok(())
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--dataset" => a.dataset = Some(need(&mut i, "--dataset")?),
            "--out" => a.out = Some(need(&mut i, "--out")?),
            "--held-out" => a.held_out = Some(need(&mut i, "--held-out")?),
            "--budget" => {
                let v = need(&mut i, "--budget")?;
                set_budget(&mut a, BudgetArg::Like(v))?;
            }
            "--bits-per-weight" => {
                let v = need(&mut i, "--bits-per-weight")?;
                set_budget(&mut a, BudgetArg::Bits(v.parse().map_err(|_| number(&v, "--bits-per-weight"))?))?;
            }
            "--max-bytes" => {
                let v = need(&mut i, "--max-bytes")?;
                set_budget(&mut a, BudgetArg::Bytes(v.parse().map_err(|_| number(&v, "--max-bytes"))?))?;
            }
            "--formats" => a.formats = Some(need(&mut i, "--formats")?),
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
            "-h" | "--help" => return Err(String::new()),
            other if other.starts_with('-') => return Err(format!("brain optimize: unknown option '{other}'")),
            other if a.base.is_empty() => a.base = other.to_string(),
            other => return Err(format!("brain optimize: unexpected argument '{other}'")),
        }
        i += 1;
    }
    if a.base.is_empty() {
        return Err("brain optimize: no base given".to_string());
    }
    Ok(a)
}

fn optimizer(a: &Args) -> brain::ModelOptimizer {
    let mut o = brain::ModelOptimizer::from_pretrained(a.base.clone()).thinking(a.thinking);
    if let Some(d) = &a.dataset {
        o = o.dataset(d);
    }
    if let Some(d) = &a.out {
        o = o.out_dir(d);
    }
    if let Some(d) = &a.held_out {
        o = o.held_out(d);
    }
    o = match &a.budget {
        Some(BudgetArg::Like(f)) => o.budget_like(f.clone()),
        Some(BudgetArg::Bits(b)) => o.budget_bits_per_weight(*b),
        Some(BudgetArg::Bytes(b)) => o.budget_bytes(*b),
        None => o,
    };
    if let Some(f) = &a.formats {
        o = o.tiers(f.clone());
    }
    if let Some(id) = &a.arch {
        o = o.arch(id.clone());
    }
    if let Some(d) = &a.models_dir {
        o = o.models_dir(d.clone());
    }
    if let Some(s) = a.seed {
        o = o.seed(s);
    }
    if let Some(m) = a.max_block {
        o = o.max_block(m);
    }
    o
}

pub fn run_optimize(argv: &[String]) -> ! {
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
    let outcome = match optimizer(&args).run() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("brain optimize: {e}");
            std::process::exit(1);
        }
    };
    let r = &outcome.report;
    println!("calibration records {}, held-out records {}, seed {}", outcome.calibration_records, outcome.held_out_records, outcome.seed);
    for e in &outcome.excluded_tiers {
        println!("left out: {} (this device holds it as {})", e.tier, e.held_as);
    }
    println!("budget {} bytes; plan {} bytes ({:.2} bits/weight) of {} at f32", r.budget_bytes, r.plan_bytes, r.bits_per_weight, r.f32_bytes);
    println!("{:<10} {:>14} {:>16} {:>16}", "", "bytes", "simulated loss", "deployed loss");
    for d in &outcome.deployed {
        let simulated = match d.label.as_str() {
            "plan" => Some(r.plan_loss),
            "original" => Some(r.original_loss),
            tier => r.baselines.iter().find(|b| b.tier == tier).map(|b| b.loss),
        };
        println!("{:<10} {:>14} {:>16} {:>16.6}", d.label, d.bytes, simulated.map_or("-".to_string(), |l| format!("{l:.6}")), d.loss);
    }
    println!("--weight-format {}", r.policy);
    println!("report: {}", outcome.report_path.display());
    if outcome.accepted {
        println!("accepted: the plan beats every uniform format that fits its budget");
        std::process::exit(0);
    }
    println!("not accepted: a uniform format that fits the budget measured as well or better");
    std::process::exit(3);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_the_documented_options() {
        let a = parse(&s(&["Qwen/Qwen3-0.6B", "--dataset", "d.jsonl", "--out", "run", "--held-out", "h.jsonl", "--bits-per-weight", "6.5", "--formats", "bf16,int8,nf4", "--seed", "7", "--max-block", "512", "--no-thinking"])).unwrap();
        assert_eq!(a.base, "Qwen/Qwen3-0.6B");
        assert_eq!((a.dataset.as_deref(), a.out.as_deref(), a.held_out.as_deref()), (Some("d.jsonl"), Some("run"), Some("h.jsonl")));
        assert_eq!(a.budget, Some(BudgetArg::Bits(6.5)));
        assert_eq!(a.formats.as_deref(), Some("bf16,int8,nf4"));
        assert_eq!((a.seed, a.max_block, a.thinking), (Some(7), Some(512), false));
    }

    #[test]
    fn one_budget_form_at_most() {
        let err = parse(&s(&["base", "--budget", "int8", "--max-bytes", "100"])).unwrap_err();
        assert!(err.contains("one of"), "{err}");
        assert_eq!(parse(&s(&["base", "--budget", "nf4"])).unwrap().budget, Some(BudgetArg::Like("nf4".into())));
    }

    #[test]
    fn bad_input_is_refused_by_name() {
        assert!(parse(&s(&["base", "--seed", "x"])).unwrap_err().contains("--seed"));
        assert!(parse(&s(&["base", "--frobnicate"])).unwrap_err().contains("--frobnicate"));
        assert!(parse(&s(&["--dataset", "d"])).unwrap_err().contains("no base"));
    }
}
