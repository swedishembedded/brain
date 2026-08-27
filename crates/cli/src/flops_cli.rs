// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain flops` — FLOP/OPS accounting for a model, OFFLINE (calculated from
//! the recorded dispatch lists, nothing executes) and optionally ONLINE
//! (accumulated at `Gpu::submit` while one forward/backward actually runs).
//!
//!   brain flops --model qwen|gpt|lfm [--weights F] [--batch B] [--block T]
//!               [--train] [--i8] [--stages N] [--run]
//!   brain flops --model flux2 [--variant V] [--width W] [--height H]
//!               [--steps N] [--refs "HxW,..."] [--batch B] [--i8] [--run]
//!   brain flops --model ltxv  [--variant V] [--width W] [--height H]
//!               [--frames F] [--fps R] [--steps N] [--run]
//!
//! Without `--weights` the model's tiny test config is built (synthetic init).
//! `--train` also records + costs the backward graph; `--i8` (qwen) builds the
//! int8 inference path, whose linears report integer OPS, not FLOPs. `--stages N`
//! (qwen/gpt) splits the layers over N pipeline stages and reports PER-STAGE
//! (= per-device) numbers. `--run` executes one forward (and backward with
//! `--train`) on a synthetic batch and prints the online counters beside the
//! offline calculation — on a fully covered model they agree exactly.
//!
//! The image/video models are priced by STAGE - text encode, N denoise
//! evaluations, VAE decode - because "which stage dominates" is the question
//! the number exists to answer, and a video is additionally reported per second
//! of output. None of it runs the model: see the stage-costing section below.
//!
//! Coverage is reported honestly: a kernel without a cost formula is listed as
//! UNCOVERED and excluded from the totals (never counted as zero-cost work),
//! and a whole STAGE that is not modelled is listed the same way rather than
//! folded in as free.

use gpu_core::cost::{Cost, CostReport, Recording};
use gpu_core::roof::Roofs;
use model::{Shard, Shardable};

use crate::args::Args;

/// The one spelling of this command's grammar - printed on `--help` (to
/// stdout, exit 0) and on a bad invocation (to stderr, exit 2). One string,
/// so the two can never drift apart.
const USAGE: &str = "usage: brain flops --model qwen|gpt|lfm [--weights F] [--batch B] [--block T] \
                     [--train] [--i8] [--stages N] [--run]\n\
       brain flops --model flux2 [--variant klein-4b|klein-9b|base-4b|base-9b|tiny] \
                     [--width W] [--height H] [--steps N] [--refs HxW,...] [--batch B] [--i8] [--run] [--vae DIR] [--per-kernel]\n\
       brain flops --model ltxv [--variant ltx25-22b|tiny] [--width W] [--height H] \
                     [--frames F] [--fps R] [--steps N] [--run] [--per-kernel]";

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
    let variant = a.str_or("--variant", "");
    let width = a.opt_u32("--width");
    let height = a.opt_u32("--height");
    let steps = a.opt_u32("--steps");
    let frames = a.opt_u32("--frames");
    let fps = a.opt_u32("--fps");
    let refs = a.take_str("--refs");
    let vae = a.take_str("--vae");
    let per_kernel = a.take_flag("--per-kernel");
    a.finish();
    match model.as_str() {
        "qwen" => qwen_flops(weights.as_deref(), b, block, train, i8, stages, run),
        "gpt" => gpt_flops(weights.as_deref(), b, block, train, stages, run),
        "lfm" => lfm_flops(weights.as_deref(), b, block, train, run),
        "flux2" => flux2_flops(&variant, width, height, steps, refs.as_deref(), b, i8, run, vae.as_deref(), per_kernel),
        "ltxv" => ltxv_flops(&variant, width, height, frames, fps, steps, run, per_kernel),
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

// ===================== diffusion pipelines: stage costing =====================
//
// A generation is not one graph. An image is text-encode, then N denoise
// evaluations, then a VAE decode; a video adds a temporal axis to all three.
// "Which stage dominates" is the question the number exists to answer, so the
// stages are reported separately and then totalled.
//
// None of it runs the model. A diffusion transformer builds its dispatches
// inside `forward()`, so the graph is captured with a DRY `cost::Recording`
// (gpu-core): every step is folded into a report and then dropped. And because
// holding a 4B denoiser's weights just to look at its graph defeats the point,
// the full-depth cost is DERIVED from probe builds of the same config at one
// and zero blocks:
//
//   cost(nd, ns) = cost(0,0) + nd*(cost(1,0) - cost(0,0)) + ns*(cost(0,1) - cost(0,0))
//
// which is exact iff the graph is affine in the block counts. That is an
// assertion about the model, not about the arithmetic, so every run CHECKS it
// at a point the basis does not contain - the (1,1) build, where the double ->
// single transition first appears - and refuses to print a total it could not
// verify. `crates/cli/tests/flops_stage_model.rs` gates the same property, plus
// the one that decides whether the model can predict at all: attention comes
// out quadratic in tokens and the projections linear.

/// One stage of a generation pipeline: what it is, how many times it runs in
/// one generation, and the cost of ONE run.
struct Stage {
    name: String,
    runs: u64,
    per_run: CostReport,
    /// Set when the stage is NOT modelled, in which case it contributes
    /// nothing to the totals and says so. Unmeasured is null, never zero.
    missing: Option<String>,
}

impl Stage {
    fn modelled(name: impl Into<String>, runs: u64, per_run: CostReport) -> Stage {
        Stage { name: name.into(), runs, per_run, missing: None }
    }

    fn unmodelled(name: impl Into<String>, runs: u64, why: impl Into<String>) -> Stage {
        Stage { name: name.into(), runs, per_run: CostReport::default(), missing: Some(why.into()) }
    }

    fn total(&self) -> CostReport {
        self.per_run.scaled(self.runs)
    }
}

fn eng(x: f64) -> String {
    for (t, s) in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")] {
        if x >= t {
            return format!("{:7.3} {s}", x / t);
        }
    }
    format!("{x:7.0}  ")
}

/// Arithmetic intensity: total OPS (fp32 flops plus integer ops) per byte of
/// streamed traffic. Against the device's ridge points this is what makes a
/// stage's number actionable - it says whether making the arithmetic faster
/// could help at all. A mixed-precision stage has two ridge points, which is
/// why the `bound` column is decided by [`Roofs::bound_of`] and not by
/// comparing this one number to one of them.
fn intensity(c: &Cost) -> f64 {
    if c.bytes == 0 { f64::INFINITY } else { (c.flops + c.int_ops) as f64 / c.bytes as f64 }
}

/// Seconds this work needs at the device's own measured roofs - the analytic
/// lower bound, against which a real run's time is the check.
fn roof_seconds(c: &Cost, r: &Roofs) -> Option<f64> {
    r.seconds_at_roof(c.flops, c.int_ops, c.bytes)
}

/// The stage table: per stage and then totalled, with the arithmetic intensity
/// and roof classification that say which one is worth attacking.
fn print_stages(stages: &[Stage], roofs: Option<Roofs>, per_unit: Option<(&str, f64)>, per_kernel: bool) {
    if per_kernel {
        for s in stages.iter().filter(|s| s.missing.is_none()) {
            println!();
            println!("---- {} x{} (per run) ----", s.name, s.runs);
            println!("{}", s.per_run);
        }
    }
    println!();
    println!(
        "{:<40} {:>5} {:>11} {:>11} {:>11} {:>9} {:>8} {:>11}",
        "stage", "runs", "flops", "int_ops", "bytes", "ops/B", "bound", "roof-secs"
    );
    let mut total = CostReport::default();
    for s in stages {
        if let Some(why) = &s.missing {
            println!("{:<40} {:>5} {:>11} {:>11} {:>11} {:>9} {:>8} {:>11}", s.name, s.runs, "?", "?", "?", "?", "?", "?");
            println!("{:<40}   NOT MODELLED: {why}", "");
            continue;
        }
        let t = s.total();
        let c = t.total;
        let bound = roofs.map(|r| r.bound_of(c.flops, c.int_ops, c.bytes).as_str()).unwrap_or("?");
        let secs = roofs.and_then(|r| roof_seconds(&c, &r)).map(|v| format!("{v:9.4} s")).unwrap_or_else(|| "?".into());
        println!(
            "{:<40} {:>5} {:>11} {:>11} {:>11} {:>9.1} {:>8} {:>11}",
            s.name,
            s.runs,
            eng(c.flops as f64),
            eng(c.int_ops as f64),
            eng(c.bytes as f64),
            intensity(&c),
            bound,
            secs
        );
        total.merge(&t);
    }
    let c = total.total;
    let secs = roofs.and_then(|r| roof_seconds(&c, &r));
    println!(
        "{:<40} {:>5} {:>11} {:>11} {:>11} {:>9.1} {:>8} {:>11}",
        "TOTAL (modelled stages)",
        "",
        eng(c.flops as f64),
        eng(c.int_ops as f64),
        eng(c.bytes as f64),
        intensity(&c),
        roofs.map(|r| r.bound_of(c.flops, c.int_ops, c.bytes).as_str()).unwrap_or("?"),
        secs.map(|v| format!("{v:9.4} s")).unwrap_or_else(|| "?".into())
    );
    if let Some((unit, n)) = per_unit {
        // The number a user of a video model actually reasons about.
        println!(
            "per {unit}: {} flops, {} int_ops, {} bytes{}",
            eng(c.flops as f64 / n),
            eng(c.int_ops as f64 / n),
            eng(c.bytes as f64 / n),
            secs.map(|v| format!(", {:.4} roof-seconds", v / n)).unwrap_or_default()
        );
    }
    println!("dispatches: {} covered of {} ({:.1}%)", total.covered, total.steps, total.coverage() * 100.0);
    if !total.uncovered.is_empty() {
        println!("UNCOVERED (excluded from every number above):");
        for (k, n) in &total.uncovered {
            println!("  {k:<40} {n} calls");
        }
    }
    match roofs {
        Some(r) => println!(
            "roofs (measured on this device): {:.0} GFLOP/s fp32, {} int8, {:.0} GB/s DRAM; ridge {:.1} flop/B fp32{}",
            r.gflops,
            r.int8_gops.map(|g| format!("{g:.0} GOP/s")).unwrap_or_else(|| "unmeasured".into()),
            r.gbs,
            r.ridge(),
            r.int8_gops
                .filter(|g| *g > 0.0 && r.gbs > 0.0)
                .map(|g| format!(", {:.1} op/B int8", g / r.gbs))
                .unwrap_or_default()
        ),
        None => println!("roofs: unmeasured on this device - flop/B and the totals stand, the seconds do not"),
    }
    println!(
        "roof-secs is a LOWER BOUND: fp32 and int8 work share the SMs so their times add, memory overlaps \
         so it is a max against that sum, and both halves assume perfect utilisation. Nothing achieves it; \
         the ratio of a measured run to it is the number to track."
    );
}

/// Exact per-kernel equality of two reports, or the first difference found.
/// Totals alone are not enough: two graphs can add up the same while
/// dispatching different work, and the derivation this checks is precisely a
/// claim about WHICH dispatches happen.
fn first_difference(a: &CostReport, b: &CostReport) -> Option<String> {
    if a.steps != b.steps {
        return Some(format!("dispatch count {} vs {}", a.steps, b.steps));
    }
    if a.total != b.total {
        return Some(format!("totals {:?} vs {:?}", a.total, b.total));
    }
    for name in a.by_kernel.keys().chain(b.by_kernel.keys()) {
        let x = a.by_kernel.get(name).map(|k| (k.calls, k.cost));
        let y = b.by_kernel.get(name).map(|k| (k.calls, k.cost));
        if x != y {
            return Some(format!("kernel {name}: {x:?} vs {y:?}"));
        }
    }
    if a.uncovered != b.uncovered {
        return Some(format!("uncovered {:?} vs {:?}", a.uncovered, b.uncovered));
    }
    None
}

/// Derive a whole block stack's cost from probes at one and zero blocks.
///
/// `probe(counts)` records the graph of a build with those per-kind block
/// counts. `full` is the real config's counts. The derivation is affine, and
/// it is CHECKED at the all-ones point - which the basis does not contain, and
/// where a per-kind transition first appears - before anything is returned.
fn affine_block_cost(
    kinds: &[&str],
    full: &[usize],
    floor: &[usize],
    probe: &mut dyn FnMut(&[usize]) -> CostReport,
) -> Result<CostReport, String> {
    // `floor` is the smallest depth the model will build at - zero for FLUX.2,
    // one for LTX (`LtxDit::forward` asserts a non-empty block stack).
    let base_at = floor.to_vec();
    let base = probe(&base_at);
    let mut per_block = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        let mut one = base_at.clone();
        one[i] += 1;
        let c = probe(&one);
        per_block.push(
            c.checked_sub(&base)
                .ok_or_else(|| format!("the +1-{kind} graph does not contain the base graph; the cost is not derivable"))?,
        );
    }
    // A check point OUTSIDE the basis: base + 1 of every kind, unless there is
    // only one kind and that point IS the basis, in which case base + 2.
    let bump = if kinds.len() == 1 { 2 } else { 1 };
    let check: Vec<usize> = base_at.iter().map(|b| b + bump).collect();
    let mut predicted = base.clone();
    for p in &per_block {
        predicted.merge(&p.scaled(bump as u64));
    }
    let recorded = probe(&check);
    if let Some(d) = first_difference(&predicted, &recorded) {
        return Err(format!("block-depth linearity check FAILED at {check:?}: {d}"));
    }
    println!(
        "block-depth linearity check: EXACT at {check:?} ({} dispatches predicted and recorded)",
        recorded.steps
    );
    let mut out = base;
    for ((p, &n), &b) in per_block.iter().zip(full).zip(floor) {
        out.merge(&p.scaled((n - b) as u64));
    }
    Ok(out)
}

// ---------------------------------------------------------------- flux2 ----

/// A toy FLUX.2 config for exercising the machinery without a real variant's
/// memory. Deliberately asymmetric, so a confusion between two axes shows up.
fn flux2_tiny() -> flux2::Flux2Config {
    flux2::Flux2Config {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 16,
        n_heads: 2,
        depth_double: 2,
        depth_single: 3,
        axes_dim: [2, 2, 2, 2],
        txt_len: 8,
        ..flux2::Flux2Config::klein_4b()
    }
}

/// Record one FLUX.2 DiT denoise evaluation at the given block depths.
/// `live` executes it; otherwise nothing reaches the device.
fn flux2_probe(
    gpu: &gpu_core::Gpu,
    base: &flux2::Flux2Config,
    nd: usize,
    ns: usize,
    lh: usize,
    lw: usize,
    refs: &[(usize, usize)],
    bsz: u32,
    prec: flux2::Precision,
    live: bool,
) -> CostReport {
    let cfg = flux2::Flux2Config { depth_double: nd, depth_single: ns, ..base.clone() };
    let n_gen = lh * lw;
    let ni = n_gen + refs.iter().map(|(h, w)| h * w).sum::<usize>();
    let mut ts = flux2::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        ts.insert(name, (shape, vec![0.0f32; n]));
    }
    // One device for every probe: a fresh `Gpu::new` per build re-opens the
    // adapter and starts its own memory accounting, which is noise here.
    let m = flux2::Flux2Model::new_batched(&cfg, &ts, gpu.share(), (cfg.txt_len + ni) as u32, bsz, prec);
    drop(ts);
    let ids = flux2::position_ids(cfg.txt_len, lh, lw, refs);
    let img = vec![0.0f32; ni * cfg.in_channels];
    let ctx = vec![0.0f32; cfg.txt_len * cfg.context_in_dim];
    let samples: Vec<flux2::Sample> =
        (0..bsz).map(|_| flux2::Sample { img_tokens: &img, ctx: &ctx, t: 0.5 }).collect();
    let rec = if live { Recording::live() } else { Recording::dry() };
    let _ = m.forward_batch(&samples, &ids, n_gen);
    rec.take()
}

/// `"64x64,32x48"` -> reference image sizes in LATENT tokens.
fn parse_refs(s: Option<&str>) -> Vec<(usize, usize)> {
    s.into_iter()
        .flat_map(|s| s.split(','))
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let (h, w) = p.trim().split_once('x').unwrap_or_else(|| {
                eprintln!("--refs wants HxW pairs in latent tokens, e.g. 64x64");
                std::process::exit(2);
            });
            (h.parse().expect("--refs height"), w.parse().expect("--refs width"))
        })
        .collect()
}

fn flux2_flops(
    variant: &str,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    refs: Option<&str>,
    bsz: u32,
    i8: bool,
    run: bool,
    vae: Option<&str>,
    per_kernel: bool,
) {
    let variant = if variant.is_empty() { "klein-4b" } else { variant };
    let cfg = match variant {
        "tiny" => flux2_tiny(),
        v => flux2::Flux2Config::from_name(v).unwrap_or_else(|e| {
            eprintln!("brain flops --model flux2: {e}");
            std::process::exit(2);
        }),
    };
    // FLUX.2 latent tokens: the VAE's eightfold spatial downsample and the
    // 2x2 pixel-unshuffle compose, so one token spans 16 pixels per axis.
    let (w, h) = (width.unwrap_or(1024), height.unwrap_or(1024));
    assert!(w % 16 == 0 && h % 16 == 0, "flux2 works in 16-pixel latent tokens: --width/--height must be multiples of 16");
    let (lh, lw) = ((h / 16) as usize, (w / 16) as usize);
    let refs = parse_refs(refs);
    let prec = if i8 { flux2::Precision::Int8 } else { flux2::Precision::F32 };
    // A distilled klein is 4 Euler steps and no CFG; a base variant is 50
    // steps, each of which is TWO forwards (conditional + unconditional).
    let n_steps = steps.unwrap_or(if cfg.distilled { 4 } else { 50 }) as u64;
    let per_step = if cfg.distilled { 1 } else { 2 };
    let ni = lh * lw + refs.iter().map(|(a, b)| a * b).sum::<usize>();

    println!(
        "flux2 {variant}: {w}x{h} = {} generated latent tokens{}, {} text tokens, hidden {} ({} double + {} single blocks), {}",
        lh * lw,
        if refs.is_empty() { String::new() } else { format!(" + {} reference tokens", ni - lh * lw) },
        cfg.txt_len,
        cfg.hidden,
        cfg.depth_double,
        cfg.depth_single,
        if i8 { "int8 DP4A" } else { "fp32" }
    );
    println!(
        "sampling: {n_steps} steps x {per_step} forward(s)/step ({}), batch {bsz}",
        if cfg.distilled { "distilled, no CFG" } else { "CFG: conditional + unconditional" }
    );

    let gpu = gpu_core::Gpu::new(flux2::KERNELS);
    let roofs = gpu_core::roof::ensure(&gpu);
    let denoise = affine_block_cost(&["double", "single"], &[cfg.depth_double, cfg.depth_single], &[0, 0], &mut |c| {
        flux2_probe(&gpu, &cfg, c[0], c[1], lh, lw, &refs, bsz, prec, false)
    });
    let denoise = match denoise {
        Ok(d) => d,
        Err(e) => {
            eprintln!("flux2: {e}");
            std::process::exit(2);
        }
    };

    let mut stages = vec![
        flux2_text_encode_stage(&cfg),
        Stage::modelled("denoise (MMDiT forward)", n_steps * per_step, denoise),
        flux2_vae_stage(lh, lw, vae),
    ];
    stages.retain(|s| s.runs > 0);

    print_stages(&stages, roofs, None, per_kernel);

    if run {
        // OFFLINE vs ONLINE on the one build small enough to execute at these
        // dims: the (1,1) probe. They must agree exactly - a dry recording that
        // saw a different graph than the device ran would invalidate every
        // number above.
        let dry = flux2_probe(&gpu, &cfg, 1, 1, lh, lw, &refs, bsz, prec, false);
        let wet = flux2_probe(&gpu, &cfg, 1, 1, lh, lw, &refs, bsz, prec, true);
        match first_difference(&dry, &wet) {
            None => println!("\noffline == online on the executed (1,1) build: {} dispatches, identical per kernel", wet.steps),
            Some(d) => {
                eprintln!("\noffline != online on the executed (1,1) build: {d}");
                std::process::exit(2);
            }
        }
        // ...and PREDICTED vs MEASURED on that same build. A cost model that
        // only agrees with itself has not been tested; the number that tests it
        // is the ratio of a real run's wall clock to the roofline bound the
        // model computes for it. Wall clock, so the host-side conditioning
        // (RoPE tables, modulation, the readback) is inside it - which makes
        // this an upper bound on the ratio, not a flattering one.
        if let Some(r) = roofs {
            let (wall, device) = time_forward(&gpu, &cfg, lh, lw, &refs, bsz, prec, 3);
            match roof_seconds(&wet.total, &r) {
                Some(bound) if bound > 0.0 => {
                    if let Some(d) = device {
                        println!(
                            "predicted vs measured on that build (DEVICE time): {bound:.4} s at the roof, {d:.4} s measured -> {:.2}x the bound ({:.1}% of the roof)",
                            d / bound,
                            100.0 * bound / d
                        );
                    }
                    println!(
                        "...and end to end (wall clock, host conditioning included): {wall:.4} s -> {:.2}x the bound",
                        wall / bound
                    );
                }
                _ => println!("predicted vs measured: the executed build does no measurable work"),
            }
        }
    }
}

/// One executed (1,1) denoise evaluation, timed two ways: best-of-`reps` WALL
/// seconds, and the summed per-kernel DEVICE time of one pass where the backend
/// can report it.
///
/// Both, because they answer different questions and only one of them tests the
/// cost model. Device time is what the roofline bound is a bound on. Wall clock
/// additionally carries this model's host-side conditioning (the RoPE tables,
/// the folded modulation, the readback), which is real time a user waits but is
/// not work any roof describes - reporting only the wall figure would blame the
/// kernels for it.
///
/// A warm-up pass runs first: first-touch allocation and shader specialisation
/// are not what is being measured.
#[allow(clippy::too_many_arguments)]
fn time_forward(
    gpu: &gpu_core::Gpu,
    base: &flux2::Flux2Config,
    lh: usize,
    lw: usize,
    refs: &[(usize, usize)],
    bsz: u32,
    prec: flux2::Precision,
    reps: usize,
) -> (f64, Option<f64>) {
    let cfg = flux2::Flux2Config { depth_double: 1, depth_single: 1, ..base.clone() };
    let n_gen = lh * lw;
    let ni = n_gen + refs.iter().map(|(h, w)| h * w).sum::<usize>();
    let mut ts = flux2::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        ts.insert(name, (shape, vec![0.0f32; n]));
    }
    let m = flux2::Flux2Model::new_batched(&cfg, &ts, gpu.share(), (cfg.txt_len + ni) as u32, bsz, prec);
    drop(ts);
    let ids = flux2::position_ids(cfg.txt_len, lh, lw, refs);
    let img = vec![0.0f32; ni * cfg.in_channels];
    let ctx = vec![0.0f32; cfg.txt_len * cfg.context_in_dim];
    let samples: Vec<flux2::Sample> =
        (0..bsz).map(|_| flux2::Sample { img_tokens: &img, ctx: &ctx, t: 0.5 }).collect();
    let _ = m.forward_batch(&samples, &ids, n_gen);
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t0 = std::time::Instant::now();
        let _ = m.forward_batch(&samples, &ids, n_gen);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let device = gpu.set_kernel_timing(true).then(|| {
        gpu.reset_kernel_times();
        let _ = m.forward_batch(&samples, &ids, n_gen);
        let secs = gpu.kernel_times().map(|v| v.iter().map(|(_, ms, _)| ms).sum::<f64>() / 1e3);
        gpu.set_kernel_timing(false);
        secs
    });
    (best, device.flatten())
}

/// The FLUX.2 text encoder: a Qwen3 prefill of the fixed text window, run only
/// as far as the deepest tapped layer (the conditioning is a concat of three
/// hidden states, so the layers past the last tap and the LM head never run).
fn flux2_text_encode_stage(cfg: &flux2::Flux2Config) -> Stage {
    let deepest = *flux2::pipeline::TAP_LAYERS.iter().max().expect("tap layers");
    let te_cfg = match cfg.context_in_dim {
        12288 => qwen3::QwenConfig::qwen3_8b(),
        7680 => qwen3::QwenConfig::qwen3_4b(),
        // A toy config has no real encoder behind it.
        _ => {
            return Stage::unmodelled(
                "text-encode (qwen3)",
                1,
                "this variant's context width matches no shipped Qwen3 encoder",
            )
        }
    };
    let t = cfg.txt_len as u32;
    let probe = |n: usize| -> CostReport {
        let shard = Shard { start: 0, end: n, embed: true, head: false, gpu_index: Shard::ANY_GPU };
        let w = qwen_shard_zeros(&te_cfg, n);
        let m = qwen3::Qwen::new_shard(te_cfg.clone(), 1, t, &w, false, shard);
        m.cost_fwd()
    };
    let c1 = probe(1);
    let c2 = probe(2);
    match c2.checked_sub(&c1) {
        Some(per_layer) => {
            let mut out = c1;
            out.merge(&per_layer.scaled(deepest as u64 - 1));
            Stage::modelled(format!("text-encode (qwen3, {deepest} of {} layers)", te_cfg.n_layers), 1, out)
        }
        None => Stage::unmodelled("text-encode (qwen3)", 1, "the encoder graph is not affine in layer count"),
    }
}

/// Zero weights for a Qwen3 build of the first `n_layers` layers only - the
/// point of a probe is to see the GRAPH, and holding a 4B encoder to look at
/// two of its layers would defeat it.
fn qwen_shard_zeros(cfg: &qwen3::QwenConfig, n_layers: usize) -> std::collections::HashMap<String, Vec<f32>> {
    cfg.param_list()
        .into_iter()
        .filter(|(name, _)| match name.strip_prefix("blocks.") {
            Some(rest) => rest.split('.').next().and_then(|s| s.parse::<usize>().ok()).is_some_and(|l| l < n_layers),
            None => true,
        })
        .map(|(name, n)| (name, vec![0.0f32; n]))
        .collect()
}

// ----------------------------------------------------------------- ltxv ----

/// Zero weights at the DiT's own manifest shapes - the probe wants the graph,
/// not the numbers.
fn ltxv_zeros(cfg: &ltxv::config::LtxDitConfig) -> vae::blocks::Tensors {
    ltxv::dit::dit_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            (name, (shape, vec![0.0f32; n]))
        })
        .collect()
}

/// Record one LTX-2.5 DiT denoise evaluation over a `(lat_t, lh, lw)` latent
/// grid at the given block depth.
fn ltxv_probe(
    base: &ltxv::config::LtxDitConfig,
    layers: usize,
    connector_layers: usize,
    lat_t: usize,
    lh: usize,
    lw: usize,
    ctx_len: usize,
    fps: f64,
    live: bool,
) -> CostReport {
    let cfg = ltxv::config::LtxDitConfig {
        num_layers: layers as u32,
        connector_num_layers: connector_layers as u32,
        ..*base
    };
    let dit = ltxv::dit::LtxDit::new(cfg, ltxv_zeros(&cfg), None);
    let t = lat_t * lh * lw;
    let latent = vec![0.0f32; t * cfg.in_channels as usize];
    let timesteps = vec![0.5f32; t];
    let keyframes = vec![0.0f32; t];
    let positions = ltxv::pipeline::real_pixel_positions(lat_t, lh, lw, fps);
    let context = vec![0.0f32; ctx_len * cfg.cross_attention_dim as usize];
    let valid = vec![1.0f32; ctx_len];
    let rec = if live { Recording::live() } else { Recording::dry() };
    let _ = dit.forward(&latent, &timesteps, &positions, &keyframes, &context, ctx_len, t, &valid);
    rec.take()
}

fn ltxv_flops(
    variant: &str,
    width: Option<u32>,
    height: Option<u32>,
    frames: Option<u32>,
    fps: Option<u32>,
    steps: Option<u32>,
    run: bool,
    per_kernel: bool,
) {
    use ltxv::config::LtxDitConfig;
    let variant = if variant.is_empty() { "ltx25-22b" } else { variant };
    let cfg = match variant {
        "ltx25-22b" | "ltx25_22b" => LtxDitConfig::ltx25_22b(),
        "tiny" => LtxDitConfig::tiny(),
        v => {
            eprintln!("brain flops --model ltxv: unknown --variant {v} (ltx25-22b|tiny)");
            std::process::exit(2);
        }
    };
    let sp = ltxv::pipeline::VAE_SPATIAL_SCALE as u32;
    let tp = ltxv::pipeline::VAE_TEMPORAL_SCALE as u32;
    let (w, h) = (width.unwrap_or(768), height.unwrap_or(512));
    let n_frames = frames.unwrap_or(121);
    let fps = fps.unwrap_or(24);
    assert!(w % sp == 0 && h % sp == 0, "ltxv latent cells span {sp} pixels: --width/--height must be multiples of {sp}");
    assert!(n_frames % tp == 1, "the causal VAE takes 1 + 8k frames; --frames must be 1 mod {tp}");
    let (lh, lw) = ((h / sp) as usize, (w / sp) as usize);
    let lat_t = ((n_frames - 1) / tp + 1) as usize;
    let t = lat_t * lh * lw;
    // The real checkpoint conditions on the Gemma-4 tokenizer's full fixed
    // width when the embeddings connector is on; the toy config has no real
    // encoder behind it, so its stub width stands in.
    let ctx_len = if cfg.use_embeddings_connector { 1024 } else { 128 };
    let n_steps = steps.unwrap_or(30) as u64;
    let seconds = n_frames as f64 / fps as f64;

    println!(
        "ltxv {variant}: {w}x{h}, {n_frames} frames at {fps} fps = {seconds:.2} s of video",
    );
    println!(
        "latent grid {lat_t} x {lh} x {lw} = {t} tokens, inner_dim {} ({} blocks + {} connector blocks), {ctx_len} text tokens",
        cfg.inner_dim, cfg.num_layers, cfg.connector_num_layers
    );
    println!("sampling: {n_steps} steps x 1 forward/step (distilled, no CFG)");

    // Two independent stacks: the 48 DiT blocks and the caption connector's
    // own 8. Both are derived, because at 4096 channels neither the real block
    // stack NOR the real connector fits a 24 GB card in fp32 - which is
    // exactly the situation an offline cost model is for. `LtxDit::forward`
    // asserts a non-empty block stack, so the basis floor is one of each.
    let denoise = affine_block_cost(
        &["block", "connector"],
        &[cfg.num_layers as usize, cfg.connector_num_layers as usize],
        &[1, 1],
        &mut |c| ltxv_probe(&cfg, c[0], c[1], lat_t, lh, lw, ctx_len, fps as f64, false),
    );
    let denoise = match denoise {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ltxv: {e}");
            std::process::exit(2);
        }
    };

    let stages = vec![
        Stage::unmodelled(
            "text-encode (gemma-4)",
            1,
            "the LTX text encoder is a separate architecture that this path does not build; it runs \
             once per clip against a denoise loop of many steps",
        ),
        Stage::modelled("denoise (DiT forward)", n_steps, denoise),
        ltxv_vae_stage(lat_t, lh, lw, n_frames, h, w),
    ];

    let gpu = gpu_core::Gpu::new(&ltxv::block::KERNELS);
    let roofs = gpu_core::roof::ensure(&gpu);
    print_stages(&stages, roofs, Some(("second of video", seconds)), per_kernel);
    println!(
        "note: the host-side patchify_proj and proj_out linears are not device dispatches and so are \
         not in these numbers; they are two [{t} x {}] GEMMs against the {} block stack.",
        cfg.inner_dim, cfg.num_layers
    );

    if run {
        let dry = ltxv_probe(&cfg, 1, 1, lat_t, lh, lw, ctx_len, fps as f64, false);
        let wet = ltxv_probe(&cfg, 1, 1, lat_t, lh, lw, ctx_len, fps as f64, true);
        match first_difference(&dry, &wet) {
            None => println!("\noffline == online on the executed 1-block build: {} dispatches, identical per kernel", wet.steps),
            Some(d) => {
                eprintln!("\noffline != online on the executed 1-block build: {d}");
                std::process::exit(2);
            }
        }
    }
}

// ------------------------------------------------------------ flux2 VAE ----

/// Zero tensors at the shapes `vae::VaeDecoder::from_diffusers` asks for, so
/// the decode graph can be built - and therefore priced - without a
/// checkpoint.
///
/// The decoder records its whole dispatch sequence at CONSTRUCTION
/// (`VaeDecoder::steps`), so nothing here executes; `Gpu::cost_of` prices that
/// list directly.
///
/// What has to be right, and what does not: every dispatch's shape comes from
/// the builder's OWN arguments (`conv(prefix, cin, cout, k, ...)`), which come
/// from the `VaeConfig`, never from the tensors handed in. So a wrong element
/// count here cannot change the cost - it would only mis-size a binding, which
/// matters to a decode that runs and not to one that is only priced. What CAN
/// change the cost is a wrong NAME (the builder panics, loudly) or a
/// `VaeConfig` that disagrees with the checkpoint's own `config.json` (a
/// different number of resnets per up-block is a different graph). `--vae`
/// gates exactly that second case, by building the same decode from the real
/// checkpoint's config and tensors and requiring the two graphs to match
/// dispatch for dispatch.
fn vae_decoder_zeros(cfg: &vae::VaeConfig) -> vae::blocks::Tensors {
    let mut t = vae::blocks::Tensors::new();
    let mut put = |name: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        t.insert(name, (shape, vec![0.0f32; n]));
    };
    let conv = |t: &mut dyn FnMut(String, Vec<usize>), p: &str, cin: u32, cout: u32, k: u32| {
        t(format!("{p}.weight"), vec![cout as usize, cin as usize, k as usize, k as usize]);
        t(format!("{p}.bias"), vec![cout as usize]);
    };
    let gnorm = |t: &mut dyn FnMut(String, Vec<usize>), p: &str, c: u32| {
        t(format!("{p}.weight"), vec![c as usize]);
        t(format!("{p}.bias"), vec![c as usize]);
    };
    let resnet = |t: &mut dyn FnMut(String, Vec<usize>), p: &str, cin: u32, cout: u32| {
        gnorm(t, &format!("{p}.norm1"), cin);
        conv(t, &format!("{p}.conv1"), cin, cout, 3);
        gnorm(t, &format!("{p}.norm2"), cout);
        conv(t, &format!("{p}.conv2"), cout, cout, 3);
        if cin != cout {
            conv(t, &format!("{p}.conv_shortcut"), cin, cout, 1);
        }
    };
    // diffusers naming: a 1x1 conv per projection, so the element count is c*c
    // whether the checkpoint stores it as [c,c,1,1] or [c,c].
    let attn = |t: &mut dyn FnMut(String, Vec<usize>), p: &str, c: u32| {
        gnorm(t, &format!("{p}.group_norm"), c);
        for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
            conv(t, &format!("{p}.{leaf}"), c, c, 1);
        }
    };

    let zc = cfg.latent_channels;
    let rc = cfg.reversed_channels();
    let mid_c = *cfg.block_out_channels.last().expect("block_out_channels");
    let f = &mut put as &mut dyn FnMut(String, Vec<usize>);
    if cfg.use_post_quant_conv {
        conv(f, "post_quant_conv", zc, zc, 1);
    }
    conv(f, "decoder.conv_in", zc, mid_c, 3);
    resnet(f, "decoder.mid_block.resnets.0", mid_c, mid_c);
    if cfg.mid_block_add_attention {
        attn(f, "decoder.mid_block.attentions.0", mid_c);
    }
    resnet(f, "decoder.mid_block.resnets.1", mid_c, mid_c);
    let mut prev = mid_c;
    for (i, &out_c) in rc.iter().enumerate() {
        for r in 0..cfg.layers_per_block + 1 {
            let cin = if r == 0 { prev } else { out_c };
            resnet(f, &format!("decoder.up_blocks.{i}.resnets.{r}"), cin, out_c);
        }
        if i < rc.len() - 1 {
            conv(f, &format!("decoder.up_blocks.{i}.upsamplers.0.conv"), out_c, out_c, 3);
        }
        prev = out_c;
    }
    gnorm(f, "decoder.conv_norm_out", prev);
    conv(f, "decoder.conv_out", prev, cfg.out_channels, 3);
    t
}

/// The FLUX.2 VAE decode of one image: `[32, lh*2, lw*2]` latent -> RGB.
///
/// With `--vae <dir|file>` the same decode is built a second time from the REAL
/// checkpoint - its own `config.json` and its own tensors - and the two graphs
/// are compared dispatch by dispatch. See [`vae_decoder_zeros`] for what that
/// does and does not gate.
fn flux2_vae_stage(lh: usize, lw: usize, real: Option<&str>) -> Stage {
    let cfg = vae::VaeConfig::flux2();
    let (h8, w8) = ((lh * 2) as u32, (lw * 2) as u32);
    let ts = vae_decoder_zeros(&cfg);
    let dec = vae::VaeDecoder::from_diffusers(cfg.clone(), &ts, h8, w8, None);
    drop(ts);
    let r = dec.gpu().cost_of(dec.steps());
    drop(dec);
    if let Some(path) = real {
        let p = std::path::Path::new(path);
        let file = if p.is_dir() { p.join("diffusion_pytorch_model.safetensors") } else { p.to_path_buf() };
        let cfg_path = file.with_file_name("config.json");
        let real_cfg = match std::fs::read_to_string(&cfg_path) {
            Ok(j) => vae::VaeConfig::from_json(&serde_json::from_str(&j).expect("vae config.json")),
            Err(_) => cfg.clone(),
        };
        let mut map = vae::blocks::Tensors::new();
        for t in checkpoint::safetensors::read(file.to_str().expect("vae path")).unwrap_or_else(|e| {
            eprintln!("brain flops --vae {path}: {e}");
            std::process::exit(2);
        }) {
            map.insert(t.name, (t.shape, t.data));
        }
        let real_dec = vae::VaeDecoder::from_diffusers(real_cfg, &map, h8, w8, None);
        drop(map);
        let rr = real_dec.gpu().cost_of(real_dec.steps());
        match first_difference(&r, &rr) {
            None => println!("vae manifest check: EXACT - the shape-only weights build the same {} dispatches as {path}", rr.steps),
            Some(d) => {
                eprintln!("vae manifest check FAILED against {path}: {d}");
                std::process::exit(2);
            }
        }
    }
    Stage::modelled(format!("vae-decode ({}x{} image)", lw * 16, lh * 16), 1, r)
}

/// The LTX 3D VAE decode of one clip: `[128, lat_t, lh, lw]` latent -> RGB
/// frames.
///
/// `LtxVaeConfig` carries its own tensor manifest, so - unlike the FLUX.2 2D
/// decoder - the shapes here are the model's own statement of them, not a
/// transcription. The decoder records its graph at construction, so pricing it
/// runs nothing.
fn ltxv_vae_stage(lat_t: usize, lh: usize, lw: usize, frames: u32, h: u32, w: u32) -> Stage {
    let cfg = ltxv::vae3d::LtxVaeConfig::conv25();
    let tiled = ltxv::vae3d::should_tile(frames, h, w);
    let ts: vae::blocks::Tensors = cfg
        .tensor_manifest()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            (name, (shape, vec![0.0f32; n]))
        })
        .collect();
    // Above a size threshold the real pipeline decodes in OVERLAPPING TILES,
    // and it does so because the whole-volume graph does not fit a card - so
    // pricing the whole-volume graph here would both misstate the work (tiles
    // overlap, and the overlap is recomputed) and fail outright on exactly the
    // clips worth pricing. Cost what actually runs: the tiled decoder builds
    // one graph per distinct tile shape as it goes, which a dry recording sees
    // in full without executing any of it.
    let (r, label) = if tiled {
        let td = ltxv::vae3d::LtxVaeTiledDecoder::auto(&cfg, &ts, lat_t as u32, lh as u32, lw as u32, None);
        let tiles = td.plan().tiles().len();
        let latent = vec![0.0f32; cfg.latent_channels as usize * lat_t * lh * lw];
        let rec = Recording::dry();
        let _ = td.decode_with(&latent, |_, _| {});
        (rec.take(), format!("vae-decode (3D, {frames} frames, {tiles} tiles)"))
    } else {
        let dec = ltxv::vae3d::LtxVaeDecoder::build(&cfg, &ts, lat_t as u32, lh as u32, lw as u32, None);
        (dec.gpu().cost_of(dec.steps()), format!("vae-decode (3D, {frames} frames)"))
    };
    drop(ts);
    Stage::modelled(label, 1, r)
}
