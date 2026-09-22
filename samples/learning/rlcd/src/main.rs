// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RLCD end to end: an executable Bayesian world, a model trained directly
//! against its exact posteriors, and an evaluation that reports calibration
//! AND cost-sensitive decision regret - not accuracy alone.
//!
//! ```text
//! RlcdPipeline::from_pretrained(encoder)
//!     .train(spec)
//!     .evaluate()
//!     .save(head)
//!     .finish()?
//! ```
//!
//! Same stage chain every brain pipeline has. What makes this sample RLCD
//! rather than an ordinary classifier demo is what happens around that
//! chain, in order:
//!
//! 1. **The world** ([`world::DiagnosisWorld`]) is executable and exact: a
//!    latent fault at `P = 0.2`, a diagnostic test at `P(+|fault) = 0.8` /
//!    `P(+|healthy) = 0.1`. Its own numbers are checked BEFORE training even
//!    starts, against [`brain::check_information_refinement`] - the identity
//!    `prior = E[posterior | evidence]` a Bayesian world's own probability
//!    model must satisfy. This is the oracle-honesty check, not a formality.
//! 2. **Training** fits the model against the world's EXACT posteriors
//!    (`2/3`, `1/19`, not "faulty"/"healthy" labels) with a proper scoring
//!    rule, over many distinct phrasings of each evidence state - see
//!    `world.rs` for why textual variety is what makes held-out evaluation
//!    mean anything.
//! 3. **Evaluation** reports calibration (ECE, NLL, Brier) AND decision
//!    regret against TWO cost matrices the model never trained under - the
//!    number that tells a learned belief apart from a learned policy.
//! 4. **A witness audit** ([`brain::witness_search`]) exhaustively checks the
//!    trained model against the oracle at the world's own canonical evidence
//!    points, under a safety-critical cost matrix, and reports any case
//!    where the model's induced action disagrees with the Bayes-optimal one.
//!
//! Run it:
//!
//! ```text
//! make samples/learning/rlcd/run ARGS="--steps 400"
//! make samples/learning/rlcd/run ARGS="--head out/rlcd-head.safetensors --ask 'The diagnostic test came back positive.'"
//! ```
//!
//! Swedish Embedded AB builds decision systems that report a calibrated
//! probability AND act on it according to the actual cost of being wrong,
//! not merely the most likely label. If your team needs judgment under
//! uncertainty that is audited rather than assumed, you can procure our
//! services by sending an email to info@swedishembedded.com.

mod world;

use brain::{check_information_refinement, witness_search, CostMatrix, Flow, Learner, LossConfig, Observation, RlcdPipeline, RlcdSpec};

struct Args {
    encoder: String,
    head_in: Option<String>,
    save_to: String,
    steps: usize,
    ask: Option<String>,
}

fn parse_args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        encoder: std::env::var("BRAIN_MINILM_DIR").unwrap_or_else(|_| {
            format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")
        }),
        head_in: None,
        save_to: "out/rlcd-head.safetensors".into(),
        steps: 400,
        ask: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--encoder" => a.encoder = next(),
            "--head" => a.head_in = Some(next()),
            "--save" => a.save_to = next(),
            "--steps" => a.steps = next().parse().unwrap_or(a.steps),
            "--ask" => a.ask = Some(next()),
            "--help" | "-h" => {
                eprintln!(
                    "usage: rlcd [--encoder DIR] [--head FILE] [--save FILE] [--steps N] [--ask STATE]\n\n\
                     Without --head: trains on the device-diagnosis world, evaluates, saves, audits.\n\
                     With --head:    skips training and audits those weights directly.\n\
                     Without --ask:  reads states from stdin until end of input, after the audit."
                );
                std::process::exit(0);
            }
            other => eprintln!("rlcd: ignoring unknown argument {other:?}"),
        }
        i += 2;
    }
    a
}

/// Rule 5 of samples/README.md: say what is missing and leave cleanly.
fn require(path: &std::path::Path, what: &str, remedy: &str) {
    if !path.exists() {
        eprintln!("rlcd: no {what} at {}", path.display());
        eprintln!("rlcd: {remedy}");
        std::process::exit(1);
    }
}

/// This sample trains through `crates/decide`, not Laya - a Laya checkpoint
/// (marked by a root `rl_agent_config.json`, no root `config.json`) has no
/// training support in this SDK yet, so fail with a clear message here
/// rather than a confusing one three layers down.
fn require_decide_shaped_encoder(dir: &str) {
    let base = std::path::Path::new(dir);
    if base.join("rl_agent_config.json").is_file() && !base.join("config.json").is_file() {
        eprintln!("rlcd: {dir} looks like a Laya checkpoint (rl_agent_config.json, no root config.json)");
        eprintln!("rlcd: this sample trains through crates/decide, which Laya does not support yet - pass a MiniLM-shaped --encoder DIR instead");
        std::process::exit(1);
    }
    if !base.join("config.json").is_file() {
        eprintln!("rlcd: no encoder checkpoint at {dir}");
        eprintln!("rlcd: run `brain pull sentence-transformers/all-MiniLM-L6-v2`, or pass --encoder DIR");
        std::process::exit(1);
    }
}

/// Adapts a trained [`RlcdPipeline`] into [`Learner`] for [`witness_search`]:
/// an [`Observation`] carries no text, only its exact posterior, so probing
/// a real model needs this fixed mapping back to one canonical rendering per
/// evidence class (`world::canonical_render`).
struct TrainedLearner<'a> {
    pipeline: &'a mut RlcdPipeline,
}

impl Learner for TrainedLearner<'_> {
    fn predict(&mut self, observation: &Observation) -> Vec<f32> {
        let state = world::canonical_render(&observation.name);
        self.pipeline.probability(state).unwrap_or_else(|e| {
            eprintln!("rlcd: probability({state:?}) failed during witness audit: {e}");
            vec![1.0; observation.posterior.len()] // deliberately non-normalized: never a silent false pass
        })
    }
}

fn main() {
    let args = parse_args();
    require_decide_shaped_encoder(&args.encoder);

    println!("rlcd: the device-diagnosis world - P(fault) = {:.2}, P(+|fault) = {:.2}, P(+|healthy) = {:.2}", world::P_FAULT, world::P_POSITIVE_GIVEN_FAULT, world::P_POSITIVE_GIVEN_HEALTHY);
    match check_information_refinement(&world::DiagnosisWorld, 1e-5) {
        Ok(()) => println!("rlcd: oracle self-consistency: OK (prior = E[posterior | evidence], exact)"),
        Err(e) => {
            eprintln!("rlcd: the world's own oracle failed its consistency check: {e}");
            std::process::exit(1);
        }
    }

    let (train, eval) = world::examples();
    println!("rlcd: {} training phrasings, {} held out for evaluation (unseen wording, same evidence)", train.len(), eval.len());

    let spec = RlcdSpec::default()
        .instructions("is this device faulty".to_string())
        .options(world::OUTCOME_NAMES.iter().map(|s| s.to_string()).collect())
        .train(train)
        .eval(eval)
        .loss(LossConfig::cross_entropy().with_brier(0.5))
        .steps(args.steps)
        .seed(11)
        // Dozens of examples, not BANKING77's thousands - a full encoder
        // fine-tune would memorize rather than generalize, and Decide::
        // save_head refuses to write a head-only checkpoint over a moved
        // encoder anyway (see RlcdSpec::freeze_encoder's own doc).
        .freeze_encoder(true)
        .eval_costs(vec![
            ("symmetric".into(), CostMatrix::binary(1.0, 1.0)),
            ("safety-critical".into(), CostMatrix::binary(1.0, 10.0)),
        ]);

    let mut flow = RlcdPipeline::builder(&args.encoder);
    if let Some(h) = &args.head_in {
        require(std::path::Path::new(h), "head weights", "train first, or drop --head");
        flow = flow.head(h);
    }
    let mut chain = Flow::new(flow.load());

    if args.head_in.is_none() {
        chain = chain.train(spec).evaluate().save(&args.save_to);
    } else {
        println!("rlcd: --head supplied - skipping training, auditing those weights directly");
    }
    chain = chain.report();

    // ---- pull the pipeline out of the chain for the witness audit --------
    // `Flow::stage` (and so every generic stage) is crate-private to the SDK,
    // so a sample composes the chain and its own extra step by finishing the
    // chain, running plain code on the pipeline it hands back, then
    // re-wrapping it to continue - exactly the seam `Flow::new` exists for.
    let mut pipeline = match chain.finish() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("rlcd: {e}");
            std::process::exit(1);
        }
    };

    println!("\n--- witness audit (safety-critical cost matrix: C_FP=1, C_FN=10) ---");
    let audit_costs = CostMatrix::binary(1.0, 10.0);
    let witnesses = {
        let mut learner = TrainedLearner { pipeline: &mut pipeline };
        witness_search(&world::DiagnosisWorld, &mut learner, &audit_costs, &[])
    };
    if witnesses.is_empty() {
        println!("  no decision failures found: the model's induced action agrees with the oracle at every canonical evidence point");
    } else {
        for w in &witnesses {
            println!(
                "  WITNESS at {:?}: oracle action {}, model action {} (regret {:.4})",
                w.failing.name, w.oracle_action, w.learner_action, w.regret
            );
            if let Some(near) = &w.nearby_correct {
                println!("    nearby correct point: {:?}", near.name);
            }
            if let Some(reveal) = &w.resolving_reveal {
                println!("    resolving evidence reveal: {:?}", reveal.name);
            }
        }
    }

    // ---- resume the chain for interactive inference -----------------------
    let chain = Flow::new(Ok(pipeline));
    let chain = match &args.ask {
        Some(state) => chain.ask(state),
        None => {
            println!("\nrlcd: type a state description and press enter (empty line or ^D to stop)");
            chain.tui()
        }
    };
    if let Err(e) = chain.report().finish() {
        eprintln!("rlcd: {e}");
        std::process::exit(1);
    }
}
