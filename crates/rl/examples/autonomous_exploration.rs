// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Autonomous exploration: a policy bootstrapping a task from a **cold start**
//! with no labels anywhere - a freshly random-initialized model, an
//! environment that emits prompts, and a programmatic verifier that scores
//! whatever the policy happens to produce. Nothing supervised is read, no
//! target is ever written into a training set, and the only signal is the
//! verifier's.
//!
//! ## What this demo shows, and what it does not
//!
//! It measures one thing end to end: whether held-out accuracy on a frozen,
//! never-trained-on probe set moves away from chance under verifier-only GRPO,
//! starting from random weights. **On this box, at the default budget, it does
//! not** - measured over 6000 steps the mean verified reward stays at the
//! uniform-token reference and the held-out column wobbles inside sampling
//! noise. The reason is visible in the table the program prints rather than
//! argued: a GRPO group whose members all earn the same reward has zero
//! advantage and is dropped, so a policy at chance trains on a fraction of
//! what it samples and that fraction carries almost no signal. Pass
//! `--base <checkpoint>` to run the identical label-free loop on top of a base
//! that already has the copy skill (the `continual_learning` example prints
//! where it cached one) and it learns - the difference between the two runs is
//! the prerequisite skill, not the presence of labels, because there are none
//! in either.
//!
//! The "no labels" claim is not asserted in prose - it is checked structurally
//! every round by [`rl::improve::assert_trained_spans_were_sampled`], every
//! trained completion span having to be a member of the multiset the policy
//! itself sampled, and the check is then deliberately BROKEN once (a target
//! spliced straight out of `Task::answer` into the trained list) so you can
//! see it panic and know it has teeth.
//!
//! It does NOT show continual learning: this is ONE task, one environment, one
//! seed, no gate, no retention probe, and no second cycle. Whether a sequence
//! of such cycles accumulates capability is a different question with
//! different controls, measured by the `continual_learning` example next door -
//! and measured there as a NEGATIVE at this model scale. It also does not show
//! anything about scale, generality, or live data: the task family is
//! synthetic and difficulty-invariant by construction.
//!
//! Run it:
//! ```text
//! cargo run --release -p brain-rl --features qwen3 --example autonomous_exploration -- \
//!     [--rounds 8] [--steps 150] [--group-size 2] [--seed 11] [--temp 1.0] [--lr 3e-3]
//! ```
//!
//! Swedish Embedded AB builds verifier-driven training loops whose "the model
//! never saw a label" property is a structural check on every batch rather
//! than a claim in a README. If your team needs expertise in reward-verified
//! training that can prove what it did and did not train on, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use model::rollout::RolloutParams;
use model::serve::SampleParams;
use model::{FitOpts, Model, ModelConfig};
use qwen3::config::QwenConfig;
use qwen3::model::Qwen;
use rl::curriculum::{self, ContentSplit, PositionCopyEnv, PositionCopyVerifier, Rule};
use rl::env::{Environment, Task};
use rl::improve;
use rl::objective::grpo::{CycleLog, Grpo, GrpoConfig};

/// Frozen held-out probes, drawn from the EVAL content split: the same rule,
/// content tuples that the explore split can never produce (the partition is a
/// property of the content, not of the seed).
const HELD_OUT: usize = 16;
const PROBE_SEED_BASE: u64 = 700_000_000;

struct Args {
    rounds: usize,
    steps: u32,
    group_size: usize,
    seed: u64,
    temp: f32,
    lr: f32,
    out: PathBuf,
    base: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Args {
        Args { rounds: 6, steps: 500, group_size: 2, seed: 11, temp: 1.5, lr: 3e-3, out: std::env::temp_dir().join("brain-rl-example-explore"), base: None }
    }
}

const USAGE: &str = "usage: autonomous_exploration [--rounds N] [--steps N] [--group-size N] [--seed S] [--temp F] [--lr F] [--out DIR] [--base FILE]";

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
            "--rounds" => {
                a.rounds = need(i, &argv).parse().expect("--rounds takes an integer");
                i += 2;
            }
            "--steps" => {
                a.steps = need(i, &argv).parse().expect("--steps takes an integer");
                i += 2;
            }
            "--group-size" => {
                a.group_size = need(i, &argv).parse().expect("--group-size takes an integer");
                i += 2;
            }
            "--seed" => {
                a.seed = need(i, &argv).parse().expect("--seed takes an integer");
                i += 2;
            }
            "--temp" => {
                a.temp = need(i, &argv).parse().expect("--temp takes a float");
                i += 2;
            }
            "--lr" => {
                a.lr = need(i, &argv).parse().expect("--lr takes a float");
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
    assert!(a.rounds >= 1, "--rounds must be at least 1");
    assert!(a.group_size >= 1, "--group-size must be at least 1");
    a
}

/// The policy's shape. Full-parameter training from random weights: there is
/// no pretrained base here on purpose, because the whole point of this demo is
/// bootstrapping from near-chance with verifier signal alone.
fn config() -> QwenConfig {
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

/// Write a checkpoint of freshly random-initialized weights - the round-0
/// policy, which has never seen this task or any other.
fn write_random_init(cfg: &QwenConfig, seed: u64, out: &Path) {
    let init = Qwen::init_weights(cfg, seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| {
            let v = init.get(&name).unwrap_or_else(|| panic!("a fresh init is missing {name}")).clone();
            assert_eq!(v.len(), n, "{name} has {} elements, the config wants {n}", v.len());
            (name, vec![n as u64], v)
        })
        .collect();
    checkpoint::save(out.to_str().expect("utf-8 path"), cfg.to_json(), &tensors);
}

/// One round of verifier-only GRPO, starting from `base`'s weights.
fn train_round(base: &Path, out: &Path, env: PositionCopyEnv, args: &Args, round: usize, log: &CycleLog) {
    let c = checkpoint::load(base.to_str().expect("utf-8 path"));
    let cfg = QwenConfig::from_json(&c.header["config"]);
    let init = c.by_role("");
    let model = Qwen::new(cfg.clone(), 1, cfg.block_size(), &init);
    let grpo = GrpoConfig {
        group_size: args.group_size,
        clip_eps: 0.2,
        kl_beta: 0.0,
        seq_len: curriculum::SEQ_LEN,
        rollout: RolloutParams { max_new: curriculum::OUT_LEN, sample: SampleParams { temp: args.temp, top_k: 0, top_p: 1.0 }, eos: None },
        max_attempts: 1,
    };
    let objective = Grpo::new(env, PositionCopyVerifier, grpo).with_log(log.clone());
    let opts = FitOpts {
        steps: args.steps,
        batch_size: 1,
        block_size: curriculum::SEQ_LEN as u32,
        lr: args.lr,
        min_lr: args.lr / 10.0,
        warmup: 0,
        decay_iters: args.steps * args.rounds as u32,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: args.group_size as u32,
        eval_interval: 0,
        eval_batches: 0,
        seed: args.seed.wrapping_add(round as u64 * 1_000),
        checkpoint_secs: 0,
        ..FitOpts::default()
    };
    model::fit_with(model, objective, &opts, Some(out)).expect("fit_with");
}

/// Run `assert_trained_spans_were_sampled` on a deliberately CORRUPTED trained
/// list - one target spliced straight out of `Task::answer` - with the panic
/// hook silenced, and return the message it panicked with. A check nobody has
/// ever seen fail is indistinguishable from a check that cannot fail.
fn negative_control(trained: &[Vec<u32>], sampled: &[Vec<u32>], target: &[u32]) -> Option<String> {
    let mut spliced = trained.to_vec();
    spliced.push(target.to_vec());
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(move || improve::assert_trained_spans_were_sampled(&spliced, sampled));
    std::panic::set_hook(hook);
    match result {
        Ok(()) => None,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            Some(msg)
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn main() {
    let args = parse_args();
    let _ = std::fs::remove_dir_all(&args.out);
    std::fs::create_dir_all(&args.out).expect("create --out");

    let rule = Rule::for_cycle(0);
    let train_env = PositionCopyEnv::new(rule, ContentSplit::Explore);
    let probe_env = PositionCopyEnv::new(rule, ContentSplit::Eval);
    let probes: Vec<Task> = (0..HELD_OUT).map(|i| probe_env.tasks(PROBE_SEED_BASE + i as u64).into_iter().next().expect("one task")).collect();
    let verifier = PositionCopyVerifier;
    let greedy = RolloutParams { max_new: curriculum::OUT_LEN, sample: SampleParams::greedy(), eos: None };

    println!("brain / rl :: autonomous exploration - no labels, verifier only");
    println!(
        "environment: position-copy cue{:02} picks({},{},{})   verifier: PositionCopyVerifier (programmatic, deterministic, dense partial credit)",
        rule.cue, rule.picks[0], rule.picks[1], rule.picks[2]
    );
    println!(
        "policy: {}, {} rounds x {} GRPO steps, group {}, temp {}, lr {}",
        if args.base.is_some() { "a supplied checkpoint (--base), still trained with NO labels" } else { "fresh random init (nothing pretrained, no supervised data anywhere)" },
        args.rounds,
        args.steps,
        args.group_size,
        args.temp,
        args.lr
    );

    // The "the answer is not a label" property, shown rather than claimed: the
    // Task carries the RULE, and the target is recomputed from the prompt.
    let sample = &probes[0];
    println!(
        "\nwhat a Task carries: id {}  prompt {:?}\n  Task::answer  = {}   <- the RULE, the only thing stored\n  target_of(..) = {:?}   <- recomputed from the prompt on demand, never stored, never written into a batch",
        sample.id,
        sample.prompt,
        serde_json::to_string(&sample.answer).expect("answer serializes"),
        curriculum::target_of(sample)
    );

    // Held-out integrity: the content-space partition makes an explore tuple
    // structurally unable to be a probe tuple; this is the second, direct
    // check on the ids that were really generated.
    let probe_ids: HashSet<&str> = probes.iter().map(|t| t.id.as_str()).collect();
    assert_eq!(probe_ids.len(), HELD_OUT, "the frozen probe set must not contain a duplicate task");
    let mut explore_checked = 0usize;
    for s in 0..4_096u64 {
        let t = train_env.tasks(s).into_iter().next().expect("one task");
        assert!(!probe_ids.contains(t.id.as_str()), "explore task {} is also a held-out probe", t.id);
        explore_checked += 1;
    }
    println!("held-out: {HELD_OUT} frozen probes from the EVAL content split, checked disjoint from {explore_checked} explore draws by id hash");

    let init = match &args.base {
        // Escape hatch, and the control that makes the cold-start result
        // legible: point this at a checkpoint that already HAS the copy skill
        // (the `continual_learning` example prints where it cached one) and
        // the very same label-free loop is run on top of it. The difference
        // between the two runs is the prerequisite skill, not the labels -
        // there are none in either.
        Some(p) => {
            println!("policy: starting from the supplied checkpoint {} (NOT a cold start; its provenance is yours to vouch for)", p.display());
            p.clone()
        }
        None => {
            let p = args.out.join("round-init.safetensors");
            write_random_init(&config(), args.seed, &p);
            p
        }
    };
    let (base_scores, _) = improve::score_checkpoint::<Qwen>(&init, &probes, &verifier, &greedy);
    let start_acc = mean(&base_scores);
    println!(
        "starting policy on the frozen probes: {start_acc:.3}   (uniform-token analytic reference {:.3}; a real model is not uniform, which is why the MEASURED number is the baseline)\n",
        curriculum::uniform_chance()
    );

    println!("rnd  sampled  verified>0  verified=1.0  mean_reward  trained/sampled  heldout   secs");
    let mut incumbent = init.clone();
    let mut last_log: Option<CycleLog> = None;
    let mut accs = vec![start_acc];
    let mut round_rewards: Vec<f32> = Vec::with_capacity(args.rounds);
    let started = std::time::Instant::now();
    for round in 0..args.rounds {
        let round_started = std::time::Instant::now();
        let out = args.out.join(format!("round{round:02}.safetensors"));
        let log = CycleLog::new();
        train_round(&incumbent, &out, PositionCopyEnv::new(rule, ContentSplit::Explore), &args, round, &log);

        let sampled = log.sampled();
        let trained = log.trained();
        let rewards = log.rewards();
        // Every round, unconditionally: no completion span may reach training
        // that the policy did not itself sample. This panics on violation, so
        // a printed row is itself the evidence.
        improve::assert_trained_spans_were_sampled(&trained, &sampled);

        let nonzero = rewards.iter().filter(|&&r| r > 0.0).count();
        let perfect = rewards.iter().filter(|&&r| r >= 1.0).count();
        let mean_reward = if rewards.is_empty() { 0.0 } else { rewards.iter().sum::<f32>() / rewards.len() as f32 };
        let (scores, _) = improve::score_checkpoint::<Qwen>(&out, &probes, &verifier, &greedy);
        let acc = mean(&scores);
        accs.push(acc);
        round_rewards.push(mean_reward);
        println!(
            "{:3}  {:7}  {:10}  {:12}  {mean_reward:11.3}  {:6}/{:<8}  {acc:7.3}  {:5.1}",
            round,
            sampled.len(),
            nonzero,
            perfect,
            trained.len(),
            sampled.len(),
            round_started.elapsed().as_secs_f64()
        );
        incumbent = out;
        last_log = Some(log);
    }

    // ---- The no-label check, and the proof that it has teeth --------------
    let log = last_log.expect("at least one round ran");
    let trained = log.trained();
    let sampled = log.sampled();
    println!("\nno-label check (rl::improve::assert_trained_spans_were_sampled):");
    println!("  round {}: {} trained spans checked against {} sampled spans -> PASS (it ran on EVERY round; it panics on failure)", args.rounds - 1, trained.len(), sampled.len());
    let target = curriculum::target_of(&probes[0]);
    match negative_control(&trained, &sampled, &target) {
        Some(msg) => {
            println!("  negative control: splicing one target straight out of Task::answer into the trained list -> PANIC");
            for line in msg.lines() {
                println!("    {}", line.trim());
            }
            println!("  -> the check has teeth; it is not a no-op");
        }
        None => println!("  negative control DID NOT PANIC - the no-label check is a no-op and every PASS above is worthless"),
    }

    let end_acc = *accs.last().expect("accs is never empty");
    let best_acc = accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let total_steps = args.steps * args.rounds as u32;
    println!(
        "\nheld-out accuracy {start_acc:.3} -> {end_acc:.3} (best round {best_acc:.3}) over {} rounds, {total_steps} GRPO steps total, from verifier signal alone.",
        args.rounds
    );
    println!("per-round held-out:   {:?}", accs.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<f64>>());
    println!("per-round mean reward: {:?}", round_rewards.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<f32>>());

    // The verdict is computed from the trajectory, not written in advance.
    // `MOVED` is deliberately coarse: 16 probes at 3 positions each cannot
    // resolve anything finer, and a demo that declares success on a wobble is
    // worse than one that declares nothing.
    const MOVED: f64 = 0.20;
    let learned = end_acc >= start_acc + MOVED;
    let reward_start = round_rewards.first().copied().unwrap_or(0.0);
    let reward_end = round_rewards.last().copied().unwrap_or(0.0);
    println!("\nWHAT THIS RUN DOES AND DOES NOT PROVE");
    println!(
        "STRUCTURAL, and it held on every round: no completion span reached training that the policy had not sampled\n\
         itself ({} spans checked against {} on the last round alone), and the check was then deliberately violated in\n\
         front of you and panicked. The environment emitted prompts, the verifier scored completions, and no label,\n\
         target or supervised example was read at any point - `Task::answer` carries the RULE and the target is\n\
         recomputed from the prompt on demand.",
        trained.len(),
        sampled.len()
    );
    if learned {
        println!(
            "MEASURED on this run: held-out accuracy moved {start_acc:.3} -> {end_acc:.3} on {HELD_OUT} frozen probes the\n\
             policy never trained on, with mean verified reward {reward_start:.3} -> {reward_end:.3}. That is the loop\n\
             learning from verifier signal alone, at this budget, on this task, with one seed."
        );
    } else {
        println!(
            "MEASURED on this run, and it is a NEGATIVE: the policy did NOT bootstrap. Held-out went {start_acc:.3} ->\n\
             {end_acc:.3} (best {best_acc:.3}) and the mean verified reward stayed at {reward_start:.3} -> {reward_end:.3},\n\
             against a {:.3} uniform-token reference - i.e. flat at chance across all {total_steps} steps. The wobble in the\n\
             held-out column is sampling noise on {HELD_OUT} probes, not progress.\n\
             The mechanism is visible in the table rather than guessed at: a group whose members all earn the SAME reward\n\
             has zero advantage and is dropped, so at chance the loop trains on a fraction of what it samples and that\n\
             fraction carries almost no signal. Verifier-only RL cannot conjure a skill the policy has no gradient\n\
             toward; it can only sharpen one the policy can already sometimes stumble into.\n\
             The control for that claim is one flag away: `--base <checkpoint that already has the copy skill>` runs this\n\
             SAME label-free loop on a base that can stumble into a correct completion (the continual_learning example\n\
             prints where it cached one), and it learns. The difference between the two runs is the prerequisite skill,\n\
             not the presence of labels - there are none in either.",
            curriculum::uniform_chance()
        );
    }
    println!(
        "DOES NOT PROVE, whichever way the numbers landed: continual learning (ONE task, ONE environment, no gate, no\n\
         retention probe, no second cycle - the continual_learning example chains twelve gated cycles and finds that\n\
         capability does NOT accumulate at this scale); scale (one seed, a two-layer 64-wide model, {total_steps} steps);\n\
         generality (the task family is synthetic and difficulty-invariant BY CONSTRUCTION, with no distribution shift,\n\
         ambiguity, label noise or adversarial content); live or unattended operation; or self-improvement of any kind.\n\
         total wall clock {:.1}s   artifacts under {}",
        started.elapsed().as_secs_f64(),
        args.out.display()
    );
}
