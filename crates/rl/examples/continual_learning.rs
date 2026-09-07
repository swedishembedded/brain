// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-cycle continual learning, end to end and in the open: N sequential
//! gated cycles (rollout -> verify -> GRPO -> gate -> promote/reject) over the
//! position-copy curriculum, with the fresh-adapter plasticity control (Arm 2)
//! and the joint-training capacity oracle (Arm 3), printing one row per cycle
//! as it completes and then the retention matrix, the pre-registered targets
//! scored PASS/FAIL, and a verdict block that states what the run does and
//! does not establish.
//!
//! **The DEFAULT regime here is not a success story, and it is not supposed to
//! be.** At this model scale, under `--regime grpo`, the loop runs, the gate
//! discriminates, the harness's structural checks hold - and capability does
//! NOT accumulate: the recorded run finished at ACC 0.271, below the untrained
//! base's own 0.354 on the same probes, with 2 of 12 cycles promoted and a
//! joint-training oracle at 0.071. That oracle number is NOT a capacity
//! verdict, and an earlier version of this comment read it as one: it gives
//! each pooled rule the same budget a single cycle gives it, and the same
//! rank-8 adapter holds all 12 rules at 0.910 once that budget is raised.
//!
//! **`--regime sft` is the one configuration measured to accumulate**, and it
//! pays for that with the property the default regime exists to test. Two
//! seeds of the same 12-cycle study ended at ACC 0.932 and 0.815 with the
//! cycle-1 canary at 1.000 and 0.986, 12 and 10 of 12 cycles promoted, and
//! every pre-registered target passing - against a measured 0.310 ceiling for
//! the best fixed cue-independent policy on these probe sets, and last-task-
//! only controls of 0.267 and 0.293 that do not reach that ceiling. It is
//! teacher-forced on the environment's own known-correct completions, so the
//! no-label structural check is vacuous and is not run; see below.
//!
//! Every number this program prints is measured in front of you on this run;
//! nothing is canned. Whatever it prints is the result, including when that
//! result is negative.
//!
//! Run it:
//! ```text
//! cargo run --release -p brain-rl --features qwen3 --example continual_learning -- \
//!     [--regime grpo|sft] [--cycles 12] [--steps N] [--eval-per-cycle N] [--seed 1] \
//!     [--replay 0.0] [--rehearsal N] [--null-gate] [--skip-oracle] [--out DIR] [--base FILE]
//! ```
//! With NO flags this is bit-identical to the recorded run above: `--regime
//! grpo`, 240 steps, 16 probes per cycle, rehearsal 0.
//!
//! `--regime sft` is a DIFFERENT and strictly WEAKER experiment, not a tuning
//! of the default one. It trains each cycle with teacher-forced supervision on
//! the environment's own known-correct completions, mixed 50/50 with a
//! rehearsal pool over every prior study rule plus the background rules the
//! frozen base already solves. That removes both separately-diagnosed causes
//! of the negative result above at once - the cue-independent shortcut, and
//! GRPO's per-rule supervision starvation - at the cost of the label-free
//! property: under `--regime sft` the no-label structural check is vacuous by
//! construction and is deliberately NOT run, which the program says in its own
//! output rather than leaving to a reader to infer. Its per-regime defaults
//! (800 steps, 48 probes per cycle, rehearsal 4) are documented at the
//! constants that carry them.
//!
//! Budget on an Intel Arc MTL box: about 4 minutes to pretrain the frozen base
//! the first time (cached afterwards, and shared with the study test's cache),
//! then roughly 1 minute per cycle with the plasticity control on, plus about
//! 3 minutes for the oracle. `--null-gate` is Arm 4: the real gate still runs
//! and is still reported, but a coin flip decides what carries forward, so the
//! two runs can be diffed by eye. Measured on this box at seed 1, the coin
//! carried 8 of 12 cycles where the gate would have carried 1, and landed at
//! ACC 0.056 against the gated arm's 0.271, with the servable model COLLAPSED
//! onto 9 distinct completions across 192 probes (against the gated arm's
//! worst cycle at 185 of 192). So the gate carries information - which is not
//! the same as the gate measuring anything a human wants.
//!
//! Swedish Embedded AB builds continual-learning harnesses whose verdicts
//! survive contact with a skeptic - retention matrices from the gate's own
//! decodes, plasticity and capacity control arms, and pre-registered
//! thresholds that are scored honestly whichever way they land. If your team
//! needs expertise in measuring whether a model that keeps training is
//! actually getting better, you can procure our services by sending an email
//! to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use model::FitOpts;
use qwen3::config::{LoraCfg, QwenConfig};
use qwen3::model::Qwen;
use rl::continual::{self, GatePolicy, PositionCopy, Regime, SftConfig, StudyConfig, StudyReport, StudySpec};
use rl::curriculum::{self, ContentSplit, PositionCopyEnv, PositionCopyVerifier, Rule};
use rl::env::Environment;
use rl::gate::GateConfig;
use rl::improve::{self, AdapterMeta};

// ---------------------------------------------------------------------------
// The study's frozen configuration.
//
// These mirror the pre-registered constants of the continual-learning study
// test, deliberately duplicated rather than hoisted: those constants ARE the
// pre-registration record and moving them out of the file that documents them
// would make the record indirect. Both keep their base in the same cache and
// under the same fingerprint, so a base built by either is reused by the other
// and a mismatch is detected rather than silently trained over.
// ---------------------------------------------------------------------------

const LORA_RANK: u32 = 8;
const LORA_ALPHA: f32 = 16.0;
const LORA_TARGETS: [&str; 4] = ["wq", "wk", "wv", "wo"];

const GROUP_SIZE: usize = 2;
const LR: f32 = 5e-3;
const MIN_LR: f32 = 5e-4;
const EXPLORE_TEMP: f32 = 1.5;

// ---------------------------------------------------------------------------
// Per-regime defaults. `--regime grpo` (the default) reproduces the recorded
// study exactly; `--regime sft` is a DIFFERENT experiment with its own
// defaults, and mixing the two sets would silently produce a third thing that
// is neither. An explicit flag always wins over both.
// ---------------------------------------------------------------------------

const GRPO_STEPS: u32 = 240;
const GRPO_EVAL_PER_CYCLE: usize = 16;
const GRPO_REHEARSAL: usize = 0;

/// Derived, not picked. This exact architecture reaches 0.995 on 16 rules at
/// 48,000 record presentations per rule, and a pooled 12-rule SFT run reached
/// 0.910 at 64,000. Under the 50/50 mixture, cycle `k`'s NEW rule receives
/// `0.5 * batch * steps` presentations, so 750 steps hits the first anchor and
/// 1000 the second; 800 sits just above the known-sufficient floor at 51,200,
/// which is the conservative end on purpose - a cycle starts from an adapter
/// that already holds `k` rules and needs to add exactly one, whereas both
/// anchors measure learning N rules from a zero-delta start.
const SFT_STEPS: u32 = 800;

/// Raised from the GRPO run's 16, for POWER, and the reason is structural
/// rather than a preference for bigger numbers. The gate's sign test is
/// computed over DISCORDANT pairs only, so at 16 probes a cycle where the two
/// arms tie on 12 has n = 4 - and at n = 4 the exact one-sided binomial cannot
/// reach p <= 0.05 at ANY outcome, not even a clean 4-0 sweep. 48 lowers the
/// win rate a significant result needs from 75.0% to 64.6% AND triples the
/// ceiling on the discordant count, so the test is reachable at all.
///
/// Probe seeds are `base + cycle * 10_000 + i`, so a 48-probe set is a
/// SUPERSET of the same cycle's 16-probe set and still cannot collide across
/// cycles - but the retention cells are not cell-comparable to the recorded
/// run's, and any report of them has to say so.
const SFT_EVAL_PER_CYCLE: usize = 48;

/// Held at the value the escalation ladder's rehearsal rows already used, so
/// the new run's difference is attributable to the OBJECTIVE rather than to a
/// re-tuned pool.
const SFT_REHEARSAL: usize = 4;

/// The gate's entropy arm is DISABLED (`min_entropy_ratio: 0.0`), and that is
/// a measurement rather than a convenience: the position-copy family has
/// exactly one correct completion per prompt, so a policy that has SOLVED a
/// rule decodes it greedily with near-zero entropy and is indistinguishable
/// from a collapsed one by entropy alone. A candidate scoring a perfect 1.000
/// was rejected as degenerate at entropy ratio 0.078 before this was found.
/// Collapse is instead measured directly, by the number of DISTINCT greedy
/// completions across the probes (the `distinct` column below).
const GATE: GateConfig = GateConfig { alpha: 0.05, min_effect_size: 0.05, anchor_budget: 0.10, min_entropy_ratio: 0.0 };

/// The pre-registered targets, sized from a 5-seed single-cycle measurement
/// before any of them was scored. Printed with observed-vs-target and a
/// PASS/FAIL verdict each.
const PREREG_RETENTION_DROP: f64 = 0.15;
const PREREG_PROBE_MARGIN: f64 = 0.20;
const PREREG_MIN_PROMOTIONS: usize = 10;
const PREREG_MIN_DIAGONAL: f64 = 0.60;
const PREREG_MIN_RHO_AT_N: f64 = 0.60;
const PREREG_ACC_OVER_LAST_ONLY: f64 = 0.15;
const PREREG_MIN_ORACLE_ACC: f64 = 0.60;
const PREREG_MIN_DISTINCT_FRAC: f64 = 0.50;

const PRETRAIN_SEQS: usize = 40_000;
const PRETRAIN_STEPS: u32 = 6_000;
const PRETRAIN_BATCH: u32 = 128;
const PRETRAIN_LR: f32 = 3e-3;
const PRETRAIN_RULES: usize = 16;
const PRETRAIN_SEED: u64 = 20;

fn lora_cfg() -> LoraCfg {
    LoraCfg { rank: LORA_RANK, alpha: LORA_ALPHA, targets: LORA_TARGETS.iter().map(|s| s.to_string()).collect() }
}

/// The full-parameter pretraining shape. NOT `QwenConfig::tiny`: at
/// `d_model = 16` a two-layer decoder cannot learn cue-conditioned copying at
/// all, so a study built on it would only ever be measuring noise.
fn pretrain_config() -> QwenConfig {
    QwenConfig {
        vocab: curriculum::VOCAB,
        block_size: curriculum::SEQ_LEN as u32,
        max_position_embeddings: curriculum::SEQ_LEN as u32,
        n_layers: 2,
        d_model: 64,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        d_ff: 256,
        tie_embeddings: true,
        qk_norm: true,
        lora: None,
        ..QwenConfig::tiny()
    }
}

/// The same shape with the study's LoRA adapter attached: the base stays
/// frozen, only `lora_a`/`lora_b` train.
fn study_config() -> QwenConfig {
    QwenConfig { lora: Some(lora_cfg()), ..pretrain_config() }
}

/// Everything the pretrained base depends on, stringified. A base whose
/// fingerprint does not match was trained for a different experiment, and
/// reusing it would silently make every number below measure something else.
fn fixture_fingerprint() -> String {
    format!(
        "v1|{}|{}|rules={PRETRAIN_RULES}|seqs={PRETRAIN_SEQS}|steps={PRETRAIN_STEPS}|bs={PRETRAIN_BATCH}|lr={PRETRAIN_LR}|seed={PRETRAIN_SEED}|rec={}",
        model::ModelConfig::to_json(&pretrain_config()),
        model::ModelConfig::to_json(&study_config()),
        curriculum::RECORD_LEN
    )
}

/// Parsed command line. `steps`, `eval_per_cycle` and `rehearsal` are
/// `Option` on purpose: their defaults depend on `--regime`, and resolving
/// them at parse time would bake the GRPO defaults into an SFT run.
struct Args {
    regime: RegimeArg,
    cycles: usize,
    steps: Option<u32>,
    eval_per_cycle: Option<usize>,
    seed: u64,
    replay: f64,
    rehearsal: Option<usize>,
    null_gate: bool,
    skip_oracle: bool,
    out: PathBuf,
    base: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq)]
enum RegimeArg {
    Grpo,
    Sft,
}

impl Args {
    fn steps(&self) -> u32 {
        self.steps.unwrap_or(match self.regime {
            RegimeArg::Grpo => GRPO_STEPS,
            RegimeArg::Sft => SFT_STEPS,
        })
    }
    fn eval_per_cycle(&self) -> usize {
        self.eval_per_cycle.unwrap_or(match self.regime {
            RegimeArg::Grpo => GRPO_EVAL_PER_CYCLE,
            RegimeArg::Sft => SFT_EVAL_PER_CYCLE,
        })
    }
    fn rehearsal(&self) -> usize {
        self.rehearsal.unwrap_or(match self.regime {
            RegimeArg::Grpo => GRPO_REHEARSAL,
            RegimeArg::Sft => SFT_REHEARSAL,
        })
    }
    fn regime(&self) -> Regime {
        match self.regime {
            RegimeArg::Grpo => Regime::Grpo,
            RegimeArg::Sft => Regime::Sft(SftConfig::default()),
        }
    }
}

impl Default for Args {
    fn default() -> Args {
        Args {
            regime: RegimeArg::Grpo,
            cycles: 12,
            steps: None,
            eval_per_cycle: None,
            seed: 1,
            replay: 0.0,
            rehearsal: None,
            null_gate: false,
            skip_oracle: false,
            out: std::env::temp_dir().join("brain-rl-example-continual"),
            base: None,
        }
    }
}

const USAGE: &str = "usage: continual_learning [--regime grpo|sft] [--cycles N] [--steps N] \
                     [--eval-per-cycle N] [--seed S] [--replay F] [--rehearsal N] [--null-gate] \
                     [--skip-oracle] [--out DIR] [--base FILE]";

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let need = |i: usize, argv: &[String]| -> String {
        argv.get(i + 1).cloned().unwrap_or_else(|| {
            eprintln!("{}: {USAGE}", argv[i]);
            std::process::exit(2)
        })
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--regime" => {
                a.regime = match need(i, &argv).as_str() {
                    "grpo" => RegimeArg::Grpo,
                    "sft" => RegimeArg::Sft,
                    other => {
                        eprintln!("--regime takes grpo or sft, got {other}\n{USAGE}");
                        std::process::exit(2);
                    }
                };
                i += 2;
            }
            "--cycles" => {
                a.cycles = need(i, &argv).parse().expect("--cycles takes an integer");
                i += 2;
            }
            "--steps" => {
                a.steps = Some(need(i, &argv).parse().expect("--steps takes an integer"));
                i += 2;
            }
            "--eval-per-cycle" => {
                a.eval_per_cycle = Some(need(i, &argv).parse().expect("--eval-per-cycle takes an integer"));
                i += 2;
            }
            "--seed" => {
                a.seed = need(i, &argv).parse().expect("--seed takes an integer");
                i += 2;
            }
            "--replay" => {
                a.replay = need(i, &argv).parse().expect("--replay takes a fraction");
                i += 2;
            }
            "--rehearsal" => {
                a.rehearsal = Some(need(i, &argv).parse().expect("--rehearsal takes an integer"));
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(need(i, &argv));
                i += 2;
            }
            "--base" => {
                a.base = Some(PathBuf::from(need(i, &argv)));
                i += 2;
            }
            "--null-gate" => {
                a.null_gate = true;
                i += 1;
            }
            "--skip-oracle" => {
                a.skip_oracle = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    assert!(a.cycles >= 1 && a.cycles <= curriculum::MAX_CUES, "--cycles must be 1..={}", curriculum::MAX_CUES);
    a
}

/// Build (or reuse) the frozen base: full-parameter causal-LM pretraining on
/// PRETRAINING rules only - cues no study cycle ever uses - then a zero-delta
/// LoRA overlay. The pretraining teaches the format and the "copy content out
/// of the prompt" skill and must never teach a study rule, or every later
/// "the loop learned task k" number would be measuring recall of pretraining.
fn prepare_base(args: &Args) -> PathBuf {
    if let Some(explicit) = &args.base {
        println!("base: {} (supplied with --base; its provenance is your problem, not this program's)", explicit.display());
        return explicit.clone();
    }
    let fingerprint = fixture_fingerprint();
    // The study test's cache. Read-only from here: on a fingerprint mismatch
    // this example builds its own base under `--out` rather than retraining
    // over a fixture another run may still be measuring against.
    let shared = std::env::temp_dir().join("brain-rl-continual-base");
    let shared_base = shared.join("base.safetensors");
    if shared_base.exists() && std::fs::read_to_string(shared.join("fingerprint.txt")).map(|s| s == fingerprint).unwrap_or(false) {
        println!("base: reusing the cached pretrained fixture at {} (fingerprint matches)", shared_base.display());
        return shared_base;
    }

    let cache = args.out.join("base");
    let base = cache.join("base.safetensors");
    let stamp = cache.join("fingerprint.txt");
    if base.exists() && std::fs::read_to_string(&stamp).map(|s| s == fingerprint).unwrap_or(false) {
        println!("base: reusing {} (fingerprint matches)", base.display());
        return base;
    }

    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).expect("create base cache");
    let data_dir = cache.join("pretrain-data");
    let rules = Rule::pretrain_rules(PRETRAIN_RULES);
    println!("base: pretraining on {PRETRAIN_RULES} rules whose cues no study cycle ever uses ({PRETRAIN_STEPS} steps, batch {PRETRAIN_BATCH}) - a few minutes, cached afterwards");
    curriculum::write_pretrain_dataset(&rules, PRETRAIN_SEQS, PRETRAIN_SEED, &data_dir).expect("write pretrain dataset");
    let opts = FitOpts {
        steps: PRETRAIN_STEPS,
        batch_size: PRETRAIN_BATCH,
        block_size: curriculum::SEQ_LEN as u32,
        lr: PRETRAIN_LR,
        min_lr: PRETRAIN_LR / 10.0,
        warmup: 100,
        decay_iters: PRETRAIN_STEPS,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: 0,
        eval_batches: 0,
        seed: PRETRAIN_SEED,
        checkpoint_secs: 0,
        // Draw record-aligned windows. Without this the loader draws windows
        // at uniformly random offsets, and a window starting mid-record
        // supervises completion tokens whose own prompt is not inside it -
        // label noise, not supervision.
        align_to_lines: true,
        ..FitOpts::default()
    };
    let started = std::time::Instant::now();
    let pretrained = cache.join("pretrained.safetensors");
    let (before, after) = continual::pretrain_base::<Qwen>(pretrain_config(), &data_dir, &opts, &pretrained).expect("pretrain");
    continual::overlay_adapter::<Qwen>(&pretrained, &study_config(), PRETRAIN_SEED, &base).expect("overlay adapter");
    std::fs::write(&stamp, &fingerprint).expect("write fingerprint");
    println!("base: pretrained in {:.1}s, loss {before:.3} -> {after:.3}", started.elapsed().as_secs_f64());
    base
}

/// Does the base actually have the copy skill, and is it genuinely ignorant of
/// the study rules? Printed before the study runs, because every number below
/// is measured on top of this one.
fn check_base(base: &Path) {
    let greedy = model::rollout::RolloutParams { max_new: curriculum::OUT_LEN, sample: model::serve::SampleParams::greedy(), eos: None };
    let rules = Rule::pretrain_rules(PRETRAIN_RULES);
    let pre_tasks: Vec<_> = rules
        .iter()
        .copied()
        .flat_map(|r| {
            let e = PositionCopyEnv::new(r, ContentSplit::Eval);
            (0..4).map(move |s| e.tasks(700_000 + s).into_iter().next().expect("one task"))
        })
        .collect();
    let (pre, _) = improve::score_checkpoint::<Qwen>(base, &pre_tasks, &PositionCopyVerifier, &greedy);
    let pre_mean = pre.iter().sum::<f64>() / pre.len() as f64;

    let env = PositionCopyEnv::new(Rule::for_cycle(0), ContentSplit::Eval);
    let study_tasks: Vec<_> = (0..32).map(|s| env.tasks(800_000 + s).into_iter().next().expect("one task")).collect();
    let (study, _) = improve::score_checkpoint::<Qwen>(base, &study_tasks, &PositionCopyVerifier, &greedy);
    let study_mean = study.iter().sum::<f64>() / study.len() as f64;

    println!(
        "base check: {pre_mean:.3} held-out on the {} PRETRAINING rules (the copy skill it was taught) | {study_mean:.3} on study rule T1 (never seen) | {:.3} uniform-token analytic chance",
        rules.len(),
        curriculum::uniform_chance()
    );
    assert!(
        pre_mean > 0.60,
        "the pretrained base scored {pre_mean:.3} on rules it was actually trained on - the fixture never learned the copy skill, so nothing measured on top of it would mean anything"
    );
}

/// Printed BEFORE any number under `--regime sft`, and again in the verdict
/// block, because the regime's claim is strictly weaker than the default's and
/// a reader who skims the trajectory would otherwise carry the wrong one away.
fn print_sft_disclaimer(args: &Args) {
    println!(
        "\nno-label property: NOT APPLICABLE on this run. --regime sft is teacher-forced on the environment's own\n\
         known-correct completions (rl::curriculum::target_of), so rl::improve::assert_trained_spans_were_sampled has\n\
         nothing to check and was deliberately NOT run - reaching it would turn a real structural check into a silent\n\
         no-op that still reads like a pass. This regime makes NO label-free claim; --regime grpo is the one that does.\n\
         Read every number below as \"continual learning with an oracle-labeled curriculum and rehearsal\", which is a\n\
         strictly weaker statement than the pure-sequential result this harness was built to test."
    );
    println!(
        "rehearsal pool composition: at cycle k the pool arm draws UNIFORMLY over the k prior study rules plus the {}\n\
         background rules the frozen base already solves - so cycle 1's pool is those {} background rules alone, which\n\
         is what makes cycle 1 unsolvable by a cue-independent policy. The pool is unbounded in k (no window), since\n\
         ACC is scored over ALL probes and a window would be optimizing against the metric.",
        args.rehearsal(),
        args.rehearsal()
    );
    if args.eval_per_cycle() != GRPO_EVAL_PER_CYCLE {
        println!(
            "probe count: {} per cycle, not the recorded run's {}. Probe seeds are base + cycle*10000 + i, so each set\n\
             is a SUPERSET of the recorded set's seeds - but the retention cells below are NOT cell-comparable to the\n\
             recorded run's, and must not be diffed against them as if they were.",
            args.eval_per_cycle(),
            GRPO_EVAL_PER_CYCLE
        );
    }
    println!(
        "pre-registered targets: A1-A8 below are printed UNCHANGED. They were sized on a 5-seed, single-cycle\n\
         measurement of the GRPO regime at 16 probes, and have NOT been re-measured for this regime - so they are the\n\
         honest bar (moving them to make them reachable would void the apparatus) but they are not a bar this regime's\n\
         own seed noise has been checked against. Re-run the single-cycle pre-registration under this regime before\n\
         reading a PASS here as a result."
    );
}

fn min_distinct_frac(report: &StudyReport) -> f64 {
    report.records.iter().map(|r| r.distinct_completions as f64 / r.decoded_tasks as f64).fold(f64::INFINITY, f64::min)
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

/// The pre-registered targets, scored against what actually happened. Printed
/// whichever way they land - a target that is only reported when it passes is
/// not a target. Returns `(failed, scored)`: how many of the targets that
/// could be evaluated on this run actually fired.
fn print_targets(report: &StudyReport, oracle: Option<f64>) -> (usize, usize) {
    let t = report.r_matrix.len();
    let r_n_1 = report.r_matrix[t - 1][0];
    let r_1_1 = report.r_matrix[0][0];
    let diagonal: Vec<f64> = report.r_matrix.iter().enumerate().map(|(k, row)| row[k]).collect();
    let learned = diagonal.iter().filter(|&&d| d >= PREREG_MIN_DIAGONAL).count();
    let rho_n = report.records.last().and_then(|r| r.plasticity_ratio);
    let last_only = report.last_task_only_acc;
    let distinct = min_distinct_frac(report);

    println!("\nPRE-REGISTERED TARGETS, observed vs target (scored whichever way they land):");
    println!(
        "  A1 retention      R[N][1] {r_n_1:.3} >= R[1][1] {r_1_1:.3} - {PREREG_RETENTION_DROP:.2} = {:.3}   {}",
        r_1_1 - PREREG_RETENTION_DROP,
        verdict(r_n_1 >= r_1_1 - PREREG_RETENTION_DROP)
    );
    println!(
        "  A2 canary alive   R[N][1] {r_n_1:.3} >= b_base {:.3} + {PREREG_PROBE_MARGIN:.2} = {:.3}   {}",
        report.b_base,
        report.b_base + PREREG_PROBE_MARGIN,
        verdict(r_n_1 >= report.b_base + PREREG_PROBE_MARGIN)
    );
    println!("  A3 no collapse    min distinct fraction {distinct:.3} >= {PREREG_MIN_DISTINCT_FRAC:.2}   {}", verdict(distinct >= PREREG_MIN_DISTINCT_FRAC));
    println!("  A4 sustained      promotions {} >= {PREREG_MIN_PROMOTIONS}   {}", report.promotions, verdict(report.promotions >= PREREG_MIN_PROMOTIONS));
    println!("  A5 really learned R[k][k] >= {PREREG_MIN_DIAGONAL:.2} on {learned} of {t} cycles (need {PREREG_MIN_PROMOTIONS})   {}", verdict(learned >= PREREG_MIN_PROMOTIONS));
    match rho_n {
        Some(rho) => println!("  A6 plasticity     rho(N) {rho:.3} >= {PREREG_MIN_RHO_AT_N:.2}   {}", verdict(rho >= PREREG_MIN_RHO_AT_N)),
        None => println!("  A6 plasticity     not measured (Arm 2 disabled)"),
    }
    match last_only {
        Some(lo) => println!(
            "  A7 accumulation   ACC {:.3} >= last-task-only {lo:.3} + {PREREG_ACC_OVER_LAST_ONLY:.2} = {:.3}   {}",
            report.acc,
            lo + PREREG_ACC_OVER_LAST_ONLY,
            verdict(report.acc >= lo + PREREG_ACC_OVER_LAST_ONLY)
        ),
        None => println!("  A7 accumulation   not measured (Arm 2 disabled)"),
    }
    match oracle {
        Some(o) => println!("  A8 capacity       joint-training oracle ACC {o:.3} >= {PREREG_MIN_ORACLE_ACC:.2}   {}", verdict(o >= PREREG_MIN_ORACLE_ACC)),
        None => println!("  A8 capacity       not measured (--skip-oracle)"),
    }

    let mut scored: Vec<bool> = vec![
        r_n_1 >= r_1_1 - PREREG_RETENTION_DROP,
        r_n_1 >= report.b_base + PREREG_PROBE_MARGIN,
        distinct >= PREREG_MIN_DISTINCT_FRAC,
        report.promotions >= PREREG_MIN_PROMOTIONS,
        learned >= PREREG_MIN_PROMOTIONS,
    ];
    scored.extend(rho_n.map(|rho| rho >= PREREG_MIN_RHO_AT_N));
    scored.extend(last_only.map(|lo| report.acc >= lo + PREREG_ACC_OVER_LAST_ONLY));
    scored.extend(oracle.map(|o| o >= PREREG_MIN_ORACLE_ACC));
    (scored.iter().filter(|&&ok| !ok).count(), scored.len())
}

/// What this run does and does not establish, in the same block, with the
/// numbers it actually produced. The "does not" list is fixed because it is a
/// property of the EXPERIMENT's design, not of how the numbers landed.
fn print_verdict(report: &StudyReport, oracle: Option<f64>, args: &Args, targets: (usize, usize)) {
    let t = report.r_matrix.len();
    let r_n_1 = report.r_matrix[t - 1][0];
    let r_1_1 = report.r_matrix[0][0];
    let last_only = report.last_task_only_acc;
    let trained: usize = report.records.iter().map(|r| r.trained_spans).sum();
    let sampled: usize = report.records.iter().map(|r| r.sampled_spans).sum();
    let rejects = report.records.iter().filter(|r| !matches!(r.gate_decision, rl::gate::Decision::Promote)).count();
    let accumulated = match last_only {
        Some(lo) => report.acc >= lo + PREREG_ACC_OVER_LAST_ONLY,
        None => false,
    };

    if args.null_gate {
        let carried = report.records.iter().filter(|r| r.applied_promote).count();
        let agreed = report.records.iter().filter(|r| r.applied_promote == matches!(r.gate_decision, rl::gate::Decision::Promote)).count();
        println!(
            "\nArm 4 (null gate): a coin flip carried {carried} of {t} cycles forward; the real gate - which still ran and is\n\
             still reported above - would have carried {}. They agreed on {agreed} of {t}. Diff this run's ACC/BWT against a\n\
             `--null-gate`-free run: if they are not separated beyond seed noise, the gate is decorative.",
            report.promotions
        );
    }
    match args.regime {
        RegimeArg::Grpo => {
            println!("\nlabel-leak check: {trained} trained completion spans are a sub-multiset of the {sampled} spans the policy itself sampled - OK");
            println!("  (rl::improve::assert_trained_spans_were_sampled ran inside the harness on EVERY cycle and on every control-arm");
            println!("   run; it panics on a violation, so reaching this line is the proof rather than a flag anyone had to check)");
        }
        RegimeArg::Sft => print_sft_disclaimer(args),
    }
    println!(
        "split integrity: {} frozen probe ids checked disjoint from {} explore ids and {} rehearsal ids",
        report.probe_ids_checked, report.explore_ids_checked, report.rehearsal_ids_checked
    );

    let pipeline = match args.regime {
        RegimeArg::Grpo => "rollout -> verify -> GRPO -> gate -> promote/reject",
        RegimeArg::Sft => "teacher-forced SFT on new rule + rehearsal pool -> gate -> promote/reject",
    };
    // Only --regime grpo can claim the no-label property. Naming both
    // properties under a teacher-forced regime would be a false statement in
    // the program's own output, which is the failure mode this whole block
    // exists to prevent.
    // Each arm carries its own line breaks through to the trailing "(worst
    // distinct fraction", so the GRPO paragraph is byte-identical to the one
    // that produced the recorded output.
    let structural = match args.regime {
        RegimeArg::Grpo => {
            "both structural properties\n\
             held on every cycle (every trained span was sampled by the policy; every probe id was disjoint from the explore\n\
             split by content-space partition AND by id hash); no cycle collapsed onto one output (worst distinct fraction"
        }
        RegimeArg::Sft => {
            "the ONE structural property this regime can\n\
             claim held on every cycle (every probe id was disjoint from the explore split by content-space partition AND\n\
             by id hash) - the no-label property is not the second one, it was not checked here at all; no cycle collapsed\n\
             onto one output (worst distinct fraction"
        }
    };
    println!("\nWHAT THIS RUN DOES AND DOES NOT PROVE");
    println!(
        "ESTABLISHES. Over {t} sequential real cycles ({pipeline}) on the\n\
         position-copy task family, at this model scale, with ONE seed ({}) and a frozen pretrained base: the harness ran\n\
         the full protocol and the retention matrix above came from the gate's own decodes; {structural}\n\
         {:.3}); {} ({r_1_1:.3} against the untrained base's {:.3} on the same probe); and the gate\n\
         {}.",
        args.seed,
        min_distinct_frac(report),
        if r_1_1 >= report.b_base + PREREG_PROBE_MARGIN {
            "cycle 1 is real learning"
        } else {
            "cycle 1 did NOT clear the untrained base by the pre-registered margin, so this run says nothing about chaining"
        },
        report.b_base,
        if report.promotions > 0 && rejects > 0 {
            format!("discriminated - {} promotions, {rejects} rejects", report.promotions)
        } else if rejects == 0 {
            format!("promoted all {t} cycles, so THIS run cannot tell a working gate from a rubber stamp")
        } else {
            "rejected every cycle, so this run cannot tell a working gate from a broken loop".to_string()
        }
    );
    if accumulated {
        println!(
            "Capability accumulated on this run: final ACC {:.3} against {} for a last-task-only model and {} for a\n\
             joint-training oracle. If that is new, the negative verdict recorded for this experiment is STALE and must\n\
             be re-derived from a fresh, seed-repeated run rather than replaced by this single one.",
            report.acc,
            last_only.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".to_string()),
            oracle.map(|v| format!("{v:.3}")).unwrap_or_else(|| "not run".to_string())
        );
    } else {
        println!(
            "DOES NOT ESTABLISH - and this run says the opposite. Capability did NOT accumulate: final ACC {:.3} (a\n\
             192-probe, {t}-task aggregate) sits below b_base {:.3} - the untrained base's zero-shot score on ONLY\n\
             probe T1's 16 probes (its score on the other {} probe sets was never separately measured, so this is a\n\
             same-task-scale comparison, not an apples-to-apples 192-probe baseline) - and a last-task-only model's {};\n\
             BWT {:+.3}; the cycle-1 canary went {r_1_1:.3} -> {r_n_1:.3}. rho(N) and its OLS slope over cycle index are\n\
             NOT reliable plasticity summary statistics here: rho is a ratio whose denominator (the fresh-adapter\n\
             control) is sometimes itself near zero, so a large or small ratio can say more about the denominator than\n\
             the numerator. This does NOT mean the fresh control sat near the floor throughout - read the per-cycle\n\
             warm and fresh scores individually, not rho's slope alone. OLS slope on rho: {}.",
            report.acc,
            report.b_base,
            t - 1,
            last_only.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".to_string()),
            report.bwt,
            report
                .plasticity_slope
                .map(|s| format!("{:+.4}/cycle, 95% CI [{:+.4}, {:+.4}] - one seed times {t} points is a diagnostic, not a test", s.slope, s.ci_lo, s.ci_hi))
                .unwrap_or_else(|| "not enough points".to_string())
        );
    }
    match oracle {
        Some(o) if o < PREREG_MIN_ORACLE_ACC => println!(
            "The binding constraint is MEASURED, not guessed: the joint-training oracle - one adapter, all {t} rules AT\n\
             ONCE, the whole study's step budget, zero sequential interference - reaches {o:.3}. So \"the loop forgot task\n\
             1\" and \"this adapter at this budget cannot represent {t} cue-conditioned rules at all\" are NOT\n\
             distinguishable by this run, and they have opposite fixes. Read the BWT above as a capacity/optimization\n\
             result, not a forgetting result."
        ),
        Some(o) => println!(
            "The joint-training oracle reaches {o:.3}, clearing {PREREG_MIN_ORACLE_ACC:.2}: capacity is NOT the binding\n\
             constraint on this run, so the BWT above can be read as a forgetting result."
        ),
        None => println!("Capacity was NOT controlled on this run (--skip-oracle), so the BWT above cannot be read as a forgetting result at all."),
    }
    println!(
        "\nALSO NOT PROVEN, whichever way the numbers landed - these are properties of the experiment's design:\n\
         - Not scale. {t} cycles at this size says nothing about 10^3 cycles or 10^9 parameters. Loss of plasticity in\n\
           deep continual learning has needed ~2000 sequential tasks to become unambiguous; this run ends orders of\n\
           magnitude short, so a passing plasticity bound would mean \"not yet detectable\", not \"does not occur\".\n\
         - Not the slope. One seed times {t} points cannot power a slope test. The slope and its CI are a diagnostic;\n\
           only the coarse level bound rho(N) >= {PREREG_MIN_RHO_AT_N:.2} was ever worth asserting.\n\
         - Not generality. The task family is synthetic and difficulty-invariant BY CONSTRUCTION, and that control was\n\
           bought by removing exactly the properties that break real systems: distribution shift, ambiguity, label\n\
           noise, adversarial content.\n\
         - Not real or live data. No non-stationarity, no data-quality drift, no feedback from a changing environment,\n\
           no unattended operation.\n\
         - Not that the gate measures anything anyone wants. The null-gate arm (--null-gate) can only show the gate\n\
           carries INFORMATION. Alignment with a capability a human cares about needs an external criterion this loop\n\
           cannot generate.\n\
         - Not seed robustness. One seed for the study; the 5-seed measurement covers cycle 1 only and exists to SIZE\n\
           the thresholds, not to estimate the effect.\n\
         - Not order independence. One task order was run.\n\
         - Not recursive self-improvement. This is a well-instrumented incremental-learning experiment with a ratchet.\n\
         - Not full-model continual learning. The base is frozen and only a bounded-rank adapter moves, which is itself\n\
           a regularizer; forgetting is measured strictly within that constraint.\n\
         The protocol is designed to CATCH this loop failing, not to certify it working. Its value is that the targets\n\
         above are conditions that could plausibly have fired - and on this run {} of the {} that could be scored did.",
        targets.0,
        targets.1
    );
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all(&args.out).expect("create --out");

    println!("brain / rl :: continual learning, position-copy curriculum");
    let base = prepare_base(&args);
    check_base(&base);

    let gate_policy = if args.null_gate { GatePolicy::CoinFlip { seed: args.seed } } else { GatePolicy::Real };
    let targets = lora_cfg().targets;
    let spec = StudySpec {
        base_checkpoint: &base,
        adapter: AdapterMeta { rank: LORA_RANK, alpha: LORA_ALPHA, targets: &targets, family: "qwen", base_id: "poscopy-pretrained", dataset_id: None },
    };
    let regime = args.regime();
    let cfg = StudyConfig {
        cycles: args.cycles,
        steps_per_cycle: args.steps(),
        group_size: GROUP_SIZE,
        grad_accum: GROUP_SIZE as u32,
        eval_per_cycle: args.eval_per_cycle(),
        lr: LR,
        min_lr: MIN_LR,
        explore_temp: EXPLORE_TEMP,
        replay_frac: args.replay,
        seed: args.seed,
        gate: GATE,
        gate_policy,
        plasticity_control: true,
        work_dir: args.out.join(if args.null_gate { "arm4" } else { "arm1" }),
        verbose: true,
        regime: regime.clone(),
    };
    let curr = PositionCopy::new(args.rehearsal());

    println!(
        "base: pretrained (frozen) + LoRA r{LORA_RANK} a{LORA_ALPHA} on {}   seed {}   gate: {}",
        LORA_TARGETS.join(","),
        args.seed,
        match gate_policy {
            GatePolicy::Real => "real".to_string(),
            GatePolicy::CoinFlip { seed } => format!("coin-flip (Arm 4, null gate, seed {seed}) - the real gate still runs and is still reported"),
        }
    );
    match &regime {
        Regime::Grpo => println!(
            "{} cycles x {} GRPO steps, group {GROUP_SIZE}, temp {EXPLORE_TEMP}, replay {:.2}, rehearsal {}, {} frozen probes per cycle",
            args.cycles,
            args.steps(),
            args.replay,
            args.rehearsal(),
            args.eval_per_cycle()
        ),
        Regime::Sft(s) => println!(
            "{} cycles x {} teacher-forced SFT steps at batch {}, lr {}, mixture {:.2} new rule / {:.2} rehearsal pool,\n\
             rehearsal {} background rules, {} frozen probes per cycle",
            args.cycles,
            args.steps(),
            s.batch,
            s.lr,
            s.new_weight,
            s.rehearsal_weight,
            args.rehearsal(),
            args.eval_per_cycle()
        ),
    }
    if let Regime::Sft(_) = &regime {
        print_sft_disclaimer(&args);
    }
    println!("arms: 1 (the gated loop) + 2 (per-cycle fresh-adapter control){}\n", if args.skip_oracle { "" } else { " + 3 (joint-training capacity oracle)" });

    let started = std::time::Instant::now();
    let mut report = continual::run_study::<Qwen, _>(&spec, &curr, &cfg).expect("run_study");
    println!("\n{}", report.matrix_table());

    let oracle = if args.skip_oracle {
        None
    } else {
        println!("Arm 3: training one fresh adapter on all {} rules POOLED for {} steps - the capacity control", args.cycles, args.steps() * args.cycles as u32);
        let o = continual::joint_oracle::<Qwen, _>(&spec, &curr, &cfg).expect("joint_oracle");
        report.joint_oracle_acc = Some(o);
        Some(o)
    };

    println!("{}", report.summary());
    if let Some(o) = oracle {
        println!("joint-training oracle ACC {o:.3}  (gap to the loop: {:+.3})", o - report.acc);
    }
    let targets = print_targets(&report, oracle);
    print_verdict(&report, oracle, &args, targets);
    println!("\ntotal wall clock {:.1}s   artifacts under {}", started.elapsed().as_secs_f64(), args.out.display());
}
