// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain flops` — FLOP/OPS accounting for a model, OFFLINE (calculated from
//! the recorded dispatch lists, nothing executes) and optionally ONLINE
//! (accumulated at `Gpu::submit` while one forward/backward actually runs).
//!
//!   brain flops --model qwen|gpt|lfm [--weights F] [--batch B] [--block T]
//!               [--train] [--i8] [--stages N] [--run]
//!
//! Without `--weights` the model's tiny test config is built (synthetic init).
//! `--train` also records + costs the backward graph; `--i8` (qwen) builds the
//! int8 inference path, whose linears report integer OPS, not FLOPs. `--stages N`
//! (qwen/gpt) splits the layers over N pipeline stages and reports PER-STAGE
//! (= per-device) numbers. `--run` executes one forward (and backward with
//! `--train`) on a synthetic batch and prints the online counters beside the
//! offline calculation — on a fully covered model they agree exactly.
//!
//! Coverage is reported honestly: a kernel without a cost formula is listed as
//! UNCOVERED and excluded from the totals (never counted as zero-cost work).

use gpu_core::cost::CostReport;
use model::{Shard, Shardable};

use crate::args::Args;

/// The one spelling of this command's grammar - printed on `--help` (to
/// stdout, exit 0) and on a bad invocation (to stderr, exit 2). One string,
/// so the two can never drift apart.
const USAGE: &str = "usage: brain flops --model qwen|gpt|lfm [--weights F] [--batch B] [--block T] \
                     [--train] [--i8] [--stages N] [--run]";

pub fn run_flops(argv: &[String]) {
    // Before the parser: `--help` is not a flag any of the branches below
    // consume, so leaving it to `Args::finish` made asking for help look like
    // a misuse ("ignoring unrecognised args") of a command that then printed
    // its usage anyway and exited non-zero.
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    let mut a = Args::new(argv);
    let model = a.str_or("--model", "");
    let weights = a.take_str("--weights");
    let b = a.u32_or("--batch", 1);
    let block = a.opt_u32("--block");
    let train = a.take_flag("--train");
    let i8 = a.take_flag("--i8");
    let stages = a.usize_or("--stages", 1);
    let run = a.take_flag("--run");
    a.finish();
    match model.as_str() {
        "qwen" => qwen_flops(weights.as_deref(), b, block, train, i8, stages, run),
        "gpt" => gpt_flops(weights.as_deref(), b, block, train, stages, run),
        "lfm" => lfm_flops(weights.as_deref(), b, block, train, run),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}

/// Contiguous even layer split into `n` stages (embed on the first, head on the
/// last) — the same shape `Pipeline::with_shards` runs.
fn even_shards(n_layers: usize, n: usize) -> Vec<Shard> {
    (0..n)
        .map(|s| Shard {
            start: s * n_layers / n,
            end: (s + 1) * n_layers / n,
            embed: s == 0,
            head: s == n - 1,
            // Cost analysis, not placement: every stage builds on the ambient
            // device (`--device`), whatever the stage count.
            gpu_index: Shard::ANY_GPU,
        })
        .collect()
}

fn print_report(title: &str, r: &CostReport) {
    println!("---- {title} ----");
    println!("{r}");
    println!();
}

/// One model instance's offline report (+ online, with `--run`).
fn report_instance(
    label: &str,
    fwd: CostReport,
    bwd: Option<CostReport>,
    online: Option<CostReport>,
) {
    print_report(&format!("{label}: forward (offline)"), &fwd);
    let mut expect = fwd;
    if let Some(bwd) = bwd {
        print_report(&format!("{label}: backward (offline)"), &bwd);
        expect.merge(&bwd);
    }
    if let Some(online) = online {
        print_report(&format!("{label}: online counters (1 executed pass)"), &online);
        let agree = online.total == expect.total && online.steps == expect.steps;
        println!(
            "{label}: online {} offline{}",
            if agree { "==" } else { "!=" },
            if agree { "" } else { " (expected on partially covered / multi-submit paths)" }
        );
        println!();
    }
}

fn synthetic_tokens(n: usize, vocab: u32) -> Vec<u32> {
    (0..n).map(|i| i as u32 % vocab.max(1)).collect()
}

fn qwen_flops(weights: Option<&str>, b: u32, block: Option<u32>, train: bool, i8: bool, stages: usize, run: bool) {
    use qwen3::{init_weights, Qwen, QwenConfig};
    assert!(!(i8 && train), "the int8 path is inference-only");
    let (cfg, init) = match weights {
        Some(path) => {
            let c = checkpoint::load(path);
            (QwenConfig::from_json(&c.header["config"]), c.by_role(""))
        }
        None => {
            let cfg = QwenConfig::tiny();
            (cfg.clone(), init_weights(&cfg, 3))
        }
    };
    let t = block.unwrap_or(cfg.block_size).min(cfg.block_size);
    println!(
        "qwen: layers={} d_model={} b={b} t={t} mode={}{}",
        cfg.n_layers,
        cfg.d_model,
        if i8 { "int8" } else if train { "train" } else { "infer-fp32" },
        if stages > 1 { format!(" stages={stages}") } else { String::new() }
    );
    let shards = even_shards(cfg.n_layers as usize, stages);
    for (si, sh) in shards.into_iter().enumerate() {
        let label = if stages > 1 {
            format!("stage {si} (layers {}..{})", sh.start, sh.end)
        } else {
            "qwen".to_string()
        };
        let m = if i8 {
            Qwen::new_shard_i8(cfg.clone(), b, t, &init, sh)
        } else {
            Qwen::new_shard(cfg.clone(), b, t, &init, train, sh)
        };
        let fwd = m.cost_fwd();
        let bwd = train.then(|| m.cost_bwd());
        let online = run.then(|| {
            let x = synthetic_tokens((b * t) as usize, cfg.vocab);
            m.set_batch(&x, &x);
            m.gpu().reset_ops_counters();
            // The stage entry point: on a partial shard `forward()`'s loss
            // read has no head buffers; zero-filled boundary residuals are
            // fine for counting (cost is shape-, not data-, dependent).
            let _ = m.run_forward_stage();
            if train {
                m.run_backward_stage();
            }
            m.poll_wait();
            m.gpu().ops_counters()
        });
        report_instance(&label, fwd, bwd, online);
    }
}

fn gpt_flops(weights: Option<&str>, b: u32, block: Option<u32>, train: bool, stages: usize, run: bool) {
    use gpt2::{init_weights, Gpt, GptConfig};
    let (cfg, init) = match weights {
        Some(_path) => {
            eprintln!("brain flops --model gpt --weights: use the tiny config (checkpoint costing not wired)");
            std::process::exit(2);
        }
        None => {
            let cfg = GptConfig::tiny();
            let init = init_weights(&cfg, 3);
            (cfg, init)
        }
    };
    let t = block.unwrap_or(cfg.block_size).min(cfg.block_size);
    println!("gpt: layers={} d_model={} b={b} t={t}", cfg.n_layers, cfg.d_model);
    let shards = even_shards(cfg.n_layers as usize, stages);
    for (si, sh) in shards.into_iter().enumerate() {
        let label = if stages > 1 { format!("stage {si} (layers {}..{})", sh.start, sh.end) } else { "gpt".to_string() };
        let m = Gpt::new_shard(cfg.clone(), b, t, &init, sh);
        let fwd = m.cost_fwd();
        let bwd = train.then(|| m.cost_bwd());
        let online = run.then(|| {
            let x = synthetic_tokens((b * t) as usize, cfg.vocab);
            m.set_batch(&x, &x);
            m.gpu.reset_ops_counters();
            let _ = m.run_forward_stage();
            if train {
                m.run_backward_stage();
            }
            m.poll_wait();
            m.gpu.ops_counters()
        });
        report_instance(&label, fwd, bwd, online);
    }
}

fn lfm_flops(weights: Option<&str>, b: u32, block: Option<u32>, train: bool, run: bool) {
    use lfm2::config::LfmConfig;
    use lfm2::init::init_weights;
    use lfm2::model::Lfm;
    let (m, t) = match weights {
        Some(path) => {
            let t = block.unwrap_or(512);
            let m = if train { Lfm::load_train(path, b, t) } else { Lfm::load_inference(path, b, t) };
            (m, t)
        }
        None => {
            let cfg = LfmConfig::tiny();
            let t = block.unwrap_or(cfg.block_size).min(cfg.block_size);
            let init = init_weights(&cfg, 5);
            let m = if train { Lfm::new_train(cfg, b, t, &init) } else { Lfm::new(cfg, b, t, &init) };
            (m, t)
        }
    };
    println!("lfm: b={b} t={t} mode={}", if train { "train" } else { "infer" });
    let fwd = m.cost_fwd();
    let bwd = train.then(|| m.cost_bwd());
    let online = run.then(|| {
        let x = synthetic_tokens((b * t) as usize, m.cfg.vocab);
        m.set_batch(&x, &x);
        m.gpu.reset_ops_counters();
        m.forward();
        if train {
            m.backward();
        }
        m.poll_wait();
        m.gpu.ops_counters()
    });
    report_instance("lfm", fwd, bwd, online);
}
