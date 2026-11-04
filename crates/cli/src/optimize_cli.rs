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
  --budget FORMAT    the bytes of every linear at one format (default: halfway
                     between the two smallest candidates' uniform sizes).
  --bits-per-weight B  mean stored bits per weight, scales included.
  --max-bytes N      bytes over every searched linear.
  --latency-rows N   the rows each linear's product is timed at on the device
                     (default 1: single-stream decode); every plan's time is
                     reported.
  --max-latency-like FORMAT  no slower, summed over the linears' products,
                     than every linear at FORMAT.
  --max-latency-ms MS  the linears' summed product time, in milliseconds.
  --formats LIST     candidate formats, comma separated (default: every
                     frozen-base format the architecture's trainer holds that
                     the device holds as itself).
  --arch ID          the base's architecture (default qwen3).
  --models-dir DIR   where a vendor/repo base resolves.
  --seed N           the calibration/held-out split's seed (default random,
                     reported).
  --max-block N      the longest record, in tokens.
  --no-thinking      render records without the reasoning block.
  --recover-steps N  then distill a LoRA adapter over the base held at the
                     plan for N steps, the original model as the teacher; it
                     is written to DIR/recovery.safetensors (default 0: none).
  --recover-lr LR    the recovery adapter's learning rate (default 1e-4).
  --recover-rank R   the recovery adapter's rank (default 16).

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
    recover_steps: u32,
    recover_lr: Option<f32>,
    recover_rank: Option<u32>,
    latency_rows: Option<u32>,
    latency: Option<LatencyArg>,
}

/// How the latency limit was given; at most one form.
#[derive(Debug, PartialEq)]
enum LatencyArg {
    Like(String),
    Ms(f64),
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args { base: String::new(), dataset: None, out: None, held_out: None, budget: None, formats: None, arch: None, models_dir: None, seed: None, max_block: None, thinking: true, recover_steps: 0, recover_lr: None, recover_rank: None, latency_rows: None, latency: None };
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
            "--latency-rows" => {
                let v = need(&mut i, "--latency-rows")?;
                a.latency_rows = Some(v.parse().map_err(|_| number(&v, "--latency-rows"))?);
            }
            "--max-latency-like" | "--max-latency-ms" if a.latency.is_some() => return Err("brain optimize: give one of --max-latency-like and --max-latency-ms".to_string()),
            "--max-latency-like" => a.latency = Some(LatencyArg::Like(need(&mut i, "--max-latency-like")?)),
            "--max-latency-ms" => {
                let v = need(&mut i, "--max-latency-ms")?;
                a.latency = Some(LatencyArg::Ms(v.parse().map_err(|_| number(&v, "--max-latency-ms"))?));
            }
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
    if let Some(rows) = a.latency_rows {
        o = o.latency_rows(rows);
    }
    o = match &a.latency {
        Some(LatencyArg::Like(f)) => o.max_latency_like(f.clone()),
        Some(LatencyArg::Ms(ms)) => o.max_latency_seconds(ms / 1000.0),
        None => o,
    };
    o = o.recover(a.recover_steps);
    if let Some(lr) = a.recover_lr {
        o = o.recover_lr(lr);
    }
    if let Some(r) = a.recover_rank {
        o = o.recover_rank(r);
    }
    o
}

/// One line on stderr per stage; calibration and estimation only at every tenth of the way, so a long search stays
/// readable.
fn report_progress(p: &brain::OptimizerProgress) {
    use brain::OptimizerProgress::{Deployed, Recovering, Search, Teacher};
    use brain::OptimizerSearchProgress::{Calibrated, Estimated, Measuring, Timed};
    let tenth = |i: usize, of: usize| i == of || i.is_multiple_of(of.div_ceil(10).max(1));
    match p {
        Search(Calibrated { batch, of }) if tenth(*batch, *of) => eprintln!("calibrating: batch {batch}/{of}"),
        Search(Estimated { site, of }) if tenth(*site, *of) => eprintln!("estimating: site {site}/{of}"),
        Search(Measuring { label }) => eprintln!("measuring {label} (simulated weights)"),
        Search(Timed { shape, of }) => eprintln!("timing: shape {shape}/{of}"),
        Deployed { label } => eprintln!("measuring {label} as deployed"),
        Teacher => eprintln!("recovery: collecting the original model's distributions"),
        Recovering { step, steps, loss } if tenth(*step as usize, *steps as usize) => eprintln!("recovery: step {step}/{steps}, loss {loss:.4}"),
        _ => {}
    }
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
    let outcome = match optimizer(&args).run_with(report_progress) {
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
    if let (Some(rows), Some(limit)) = (r.timed_rows, r.latency_limit_seconds) {
        println!("latency limit {:.3} ms over the linears' products at {rows} row(s)", limit * 1e3);
    }
    println!("deployed loss measured on the {}", outcome.measured_on);
    println!("{:<10} {:>14} {:>12} {:>16} {:>16}", "", "bytes", "product ms", "simulated loss", "deployed loss");
    let ms = |s: Option<f64>| s.map_or("-".to_string(), |s| format!("{:.3}", s * 1e3));
    for d in &outcome.deployed {
        let (simulated, seconds) = match d.label.as_str() {
            "plan" => (Some(r.plan_loss), r.plan_seconds),
            "original" => (Some(r.original_loss), r.f32_seconds),
            tier => r.baselines.iter().find(|b| b.tier == tier).map_or((None, None), |b| (Some(b.loss), b.seconds)),
        };
        println!("{:<10} {:>14} {:>12} {:>16} {:>16.6}", d.label, d.bytes, ms(seconds), simulated.map_or("-".to_string(), |l| format!("{l:.6}")), d.loss);
    }
    println!("--weight-format {}", r.policy);
    if let Some(rec) = &outcome.recovery {
        println!(
            "recovery adapter {} ({} steps, rank {}): held-out loss {:.6} -> {:.6} on the trainer build{}",
            rec.adapter.display(),
            rec.steps,
            rec.rank,
            rec.plan_loss,
            rec.recovered_loss,
            if rec.improves() { "" } else { " (no improvement)" }
        );
    }
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
        let a = parse(&s(&["Qwen/Qwen3-0.6B", "--dataset", "d.jsonl", "--out", "run", "--held-out", "h.jsonl", "--bits-per-weight", "6.5", "--formats", "bf16,int8,nf4", "--seed", "7", "--max-block", "512", "--no-thinking", "--recover-steps", "50", "--recover-lr", "2e-4", "--recover-rank", "8"])).unwrap();
        assert_eq!((a.recover_steps, a.recover_lr, a.recover_rank), (50, Some(2e-4), Some(8)));
        assert_eq!(a.base, "Qwen/Qwen3-0.6B");
        assert_eq!((a.dataset.as_deref(), a.out.as_deref(), a.held_out.as_deref()), (Some("d.jsonl"), Some("run"), Some("h.jsonl")));
        assert_eq!(a.budget, Some(BudgetArg::Bits(6.5)));
        assert_eq!(a.formats.as_deref(), Some("bf16,int8,nf4"));
        assert_eq!((a.seed, a.max_block, a.thinking), (Some(7), Some(512), false));
    }

    #[test]
    fn one_latency_limit_form_at_most() {
        let a = parse(&s(&["base", "--latency-rows", "8", "--max-latency-like", "int8"])).unwrap();
        assert_eq!((a.latency_rows, a.latency), (Some(8), Some(LatencyArg::Like("int8".into()))));
        assert!(parse(&s(&["base", "--max-latency-ms", "2", "--max-latency-like", "int8"])).unwrap_err().contains("one of"));
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
