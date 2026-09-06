// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The multi-cycle continual-learning study: 12 sequential real cycles
//! (rollout -> verify -> GRPO -> gate -> promote/reject) over the
//! position-copy curriculum, with the three control arms that make its
//! numbers mean something.
//!
//! ## The result, up front
//!
//! At this model scale the loop RUNS and does NOT accumulate capability.
//! Twelve sequential gated cycles produce a real retention matrix, the gate
//! discriminates (2 promotions, 10 rejects, including one for regressing the
//! retention anchor), both structural properties hold on every cycle - and
//! final ACC lands BELOW the untrained base's own score on the same probes.
//! The tests are named for that, and two TRIPWIRE assertions fail if it ever
//! stops being true, so the conclusion cannot silently go stale.
//!
//! ## What this file asserts
//!
//! That the harness ran the full protocol and its matrix is well formed and
//! self-consistent; that no label reached training on any cycle and every
//! frozen probe id was disjoint from the explore split; that no cycle
//! collapsed onto a single output; that the FIRST cycle is real learning well
//! clear of the untrained base; and that the gate both promoted and rejected
//! within one run, with the retention anchor firing at least once.
//!
//! ## What this file deliberately does NOT assert
//!
//! The continual-learning claims themselves (A1-A7 of the pre-registration).
//! They are printed with observed-vs-target and a PASS/FAIL verdict each, and
//! they FAIL. Asserting them would either be a lie or a test that never goes
//! green. The plasticity slope and its confidence interval are likewise
//! reported only - one seed times twelve points cannot power a slope test,
//! and in this run rho(N) is a ratio of two near-floor numbers, i.e. noise.
//!
//! ## What nothing here establishes
//!
//! Scale (twelve cycles at this size says nothing about a thousand),
//! generality (the task family is synthetic and difficulty-invariant BY
//! CONSTRUCTION - that control was bought by removing exactly the properties
//! that break real systems), seed robustness (one seed for the study; the
//! five-seed measurement covers cycle 1 only and exists to SIZE the
//! thresholds, not to estimate the effect), order independence (one task
//! order was run), or anything resembling recursive self-improvement. Nor
//! does the negative result generalize: it is a statement about THIS scale,
//! adapter, budget and regime, and the joint-training oracle says the binding
//! constraint is capacity/optimization rather than forgetting.
//!
//! Run one binary at a time, single-threaded (shared GPU):
//! `cargo test -p brain-rl --features qwen3 --test continual_study --release
//!  -- --test-threads=1 --nocapture`

use std::path::{Path, PathBuf};

use model::FitOpts;
use qwen3::config::{LoraCfg, QwenConfig};
use qwen3::model::Qwen;
use rl::continual::{self, GatePolicy, PositionCopy, Regime, StudyConfig, StudyReport, StudySpec};
use rl::curriculum::{self, ContentSplit, PositionCopyEnv, PositionCopyVerifier, Rule};
use rl::env::Environment;
use rl::gate::{Decision, GateConfig};
use rl::improve::{self, AdapterMeta};

// ---------------------------------------------------------------------------
// PRE-REGISTERED CONSTANTS
//
// Measured on this box before any of the assertions below were written, by
// running `seed_noise_band_stays_within_the_preregistered_bound` (5 seeds x
// cycle 1, one frozen pretrained base). The measurement and the numbers it
// produced are recorded verbatim in the roadmap's P19 entry.
//
// MEASURED: see `PREREG_MEASUREMENT` below.
//
// From the point these were frozen, the probe sets, the gate config and the
// study model shape are frozen too: changing any of them invalidates the
// retention matrix and must be recorded as such rather than silently applied.
// ---------------------------------------------------------------------------

/// The verbatim record of the pre-registration measurement these constants
/// were sized from - printed by every test so a stale threshold is visible in
/// the output rather than buried in a comment.
const PREREG_MEASUREMENT: &str = "5 seeds x cycle 1 on one frozen pretrained base (Intel Arc MTL, release): \
     R[1][1] = [0.938, 0.833, 0.708, 1.000, 0.833], all five PROMOTE, spread s = 0.292, sigma = 0.100, mean 0.863; \
     b_base (untrained base on probe T1) = 0.354; uniform-token analytic chance = 0.031; \
     distinct completions 16/16 on every seed. \
     CAVEAT ON RECORD: s = 0.292 is single-cycle TRAINING-OUTCOME spread and it is LARGER than \
     PREREG_RETENTION_DROP (0.15), so no A1 verdict at one seed - in either direction - is robust to \
     seed at that precision. At 120 steps/cycle the same measurement gave s = 0.604 with one seed of \
     five failing to learn at all; escalation rung 1 (120 -> 240 steps) was taken and recorded.";

const CYCLES: usize = 12;
/// `n = 16` -> the exact one-sided sign test needs >= 12 wins and 0 losses
/// for `p <= 0.05` (`p = 0.0384`). Never reduce this: the gate stops being
/// able to reach significance at all.
const EVAL_PER_CYCLE: usize = 16;
const STEPS_PER_CYCLE: u32 = 240;
const GROUP_SIZE: usize = 2;
const LR: f32 = 5e-3;
const MIN_LR: f32 = 5e-4;
/// Training-rollout sampling temperature - see `StudyConfig::explore_temp`.
const EXPLORE_TEMP: f32 = 1.5;
/// Anchor-replay fraction. `0.0` = PURE SEQUENTIAL, which is the
/// configuration this study was asked to measure and the one whose numbers
/// are on the record. Replay (0.3 / 0.5, with and without rehearsal) was
/// measured too and is reported in the roadmap, not asserted here.
const REPLAY_FRAC: f64 = 0.0;
const STUDY_SEED: u64 = 1;

/// `min_entropy_ratio: 0.0` DISABLES the gate's entropy arm, and that is a
/// measurement, not a convenience.
///
/// The position-copy family has exactly ONE correct completion per prompt, so
/// a policy that has actually solved a rule decodes it greedily with
/// near-zero completion entropy. Measured on this box: a cycle-1 candidate
/// that scored a PERFECT 1.000 on its held-out probe was rejected as
/// `Cause::Degenerate` at `entropy_ratio = 0.078` against an untrained
/// incumbent's high-entropy baseline. On a deterministic verifiable task the
/// entropy ratio cannot separate "collapsed onto one output" from "solved
/// it": both are low-entropy. Leaving the arm at 0.25 would have rejected
/// roughly one in five otherwise-perfect cycles for succeeding.
///
/// The check it is NOT allowed to silently replace - "did the policy collapse
/// onto a single output?" - is instead asserted directly and far more
/// sharply by A3 below, over the distinct greedy completions the servable
/// model produced across its probes. A collapsed policy emits ONE completion
/// for every prompt; a correct one emits a different completion per prompt.
const GATE: GateConfig = GateConfig { alpha: 0.05, min_effect_size: 0.05, anchor_budget: 0.10, min_entropy_ratio: 0.0 };
/// A3's real non-degeneracy floor: at least this fraction of the servable
/// model's decoded probes must produce DISTINCT completions.
const PREREG_MIN_DISTINCT_FRAC: f64 = 0.50;

/// `max(0.15, 2s)` with the measured `s = 0.292` -> `0.583`, rounded to 0.59.
/// Deliberately loose: this constant's job is to fail FIRST when the box or
/// the model drifts far enough that the other thresholds are stale, not to
/// be a tight bound on anything.
const PREREG_NOISE_BAND: f64 = 0.59;
/// `R[N][1] >= R[1][1] - 0.15`.
const PREREG_RETENTION_DROP: f64 = 0.15;
/// `R[N][1] >= b_base + 0.20`.
const PREREG_PROBE_MARGIN: f64 = 0.20;
/// `>= N - 2`.
const PREREG_MIN_PROMOTIONS: usize = 10;
/// `R[k][k] >= 0.60` on at least `PREREG_MIN_PROMOTIONS` of the N cycles.
const PREREG_MIN_DIAGONAL: f64 = 0.60;
/// The one non-vacuous LEVEL bound on plasticity (the slope is reported only).
const PREREG_MIN_RHO_AT_N: f64 = 0.60;
/// F4: accumulated capability must beat a last-task-only model by this much.
const PREREG_ACC_OVER_LAST_ONLY: f64 = 0.15;
/// Arm 3: capacity is not the binding constraint.
const PREREG_MIN_ORACLE_ACC: f64 = 0.60;

// ---------------------------------------------------------------------------
// Fixture: the frozen pretrained base.
// ---------------------------------------------------------------------------

/// LoRA rank/alpha the study's carried-forward adapter uses, and which
/// projections it adapts.
///
/// Escalation rung 2 (rank 8 -> 16, attention-only -> attention + MLP) was
/// TRIED AND REJECTED on measurement, not skipped: at rank 16 on
/// `wq,wk,wv,wo,gate,up,down` with this learning rate, four of five seeds
/// scored 0.00-0.08 on their own held-out probe (mean sampled reward ~0.10,
/// against ~0.65 at rank 8) - the larger adapter at `lr = 5e-3` destroys the
/// pretrained copy prior faster than GRPO can rebuild it. Rung 2 is a
/// capacity lever and capacity was never the binding constraint here.
const LORA_RANK: u32 = 8;
const LORA_ALPHA: f32 = 16.0;
const LORA_TARGETS: [&str; 4] = ["wq", "wk", "wv", "wo"];

fn lora_cfg() -> LoraCfg {
    LoraCfg { rank: LORA_RANK, alpha: LORA_ALPHA, targets: LORA_TARGETS.iter().map(|s| s.to_string()).collect() }
}

/// Sequences of PRETRAINING data (pretrain-only cues; no study rule appears).
const PRETRAIN_SEQS: usize = 40_000;
const PRETRAIN_STEPS: u32 = 6_000;
const PRETRAIN_BATCH: u32 = 128;
const PRETRAIN_LR: f32 = 3e-3;
/// How many of the 16 pretraining cues to teach the base.
const PRETRAIN_RULES: usize = 16;

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-rl-continual-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Everything the pretrained base depends on, stringified. Any change to it
/// invalidates the cached fixture - a stale base would silently make every
/// number in this file measure a different experiment.
fn fixture_fingerprint() -> String {
    format!(
        "v1|{}|{}|rules={PRETRAIN_RULES}|seqs={PRETRAIN_SEQS}|steps={PRETRAIN_STEPS}|bs={PRETRAIN_BATCH}|lr={PRETRAIN_LR}|seed=20|rec={}",
        pretrain_config().to_json(),
        study_config().to_json(),
        curriculum::RECORD_LEN
    )
}

/// The full-parameter pretraining shape. NOT `QwenConfig::tiny`: at
/// `d_model = 16` a two-layer decoder cannot learn cue-conditioned copying at
/// all, so a study built on `tiny` would only ever be measuring noise.
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

/// The same shape with the study's LoRA adapter attached - the frozen base
/// stays frozen, only `lora_a`/`lora_b` train.
fn study_config() -> QwenConfig {
    QwenConfig { lora: Some(lora_cfg()), ..pretrain_config() }
}

fn lora_targets() -> Vec<String> {
    lora_cfg().targets
}

/// Build the study's frozen base: full-parameter causal-LM pretraining on
/// PRETRAINING rules only, then a zero-delta LoRA overlay.
///
/// CACHED across the three tests in this file, keyed by
/// [`fixture_fingerprint`] - the base is a deterministic function of that
/// fingerprint and costs minutes to build, and all three tests must measure
/// against the SAME base anyway or their numbers are not comparable. The
/// sanity checks below re-run on every call, cached or not: a fixture that
/// cannot be re-validated is not a fixture.
fn prepare_base() -> PathBuf {
    let cache = std::env::temp_dir().join("brain-rl-continual-base");
    let stamp = cache.join("fingerprint.txt");
    let base = cache.join("base.safetensors");
    let fingerprint = fixture_fingerprint();
    let cached = std::fs::read_to_string(&stamp).map(|s| s == fingerprint).unwrap_or(false) && base.exists();

    let rules = Rule::pretrain_rules(PRETRAIN_RULES);
    let mut how = "cached".to_string();
    if !cached {
        let _ = std::fs::remove_dir_all(&cache);
        std::fs::create_dir_all(&cache).unwrap();
        let data_dir = cache.join("pretrain-data");
        curriculum::write_pretrain_dataset(&rules, PRETRAIN_SEQS, 20, &data_dir).expect("write pretrain dataset");

        let pretrained = cache.join("pretrained.safetensors");
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
            seed: 20,
            checkpoint_secs: 0,
            // Load record-aligned windows. Without this the loader draws
            // windows at random offsets and most supervised completion tokens
            // have no prompt in their own window - label noise, not
            // supervision.
            align_to_lines: true,
            ..FitOpts::default()
        };
        let started = std::time::Instant::now();
        let (loss_before, loss_after) = continual::pretrain_base::<Qwen>(pretrain_config(), &data_dir, &opts, &pretrained).expect("pretrain");
        continual::overlay_adapter::<Qwen>(&pretrained, &study_config(), 20, &base).expect("overlay adapter");
        std::fs::write(&stamp, &fingerprint).unwrap();
        how = format!("trained in {:.1}s, loss {loss_before:.3} -> {loss_after:.3}", started.elapsed().as_secs_f64());
    }

    // Fixture sanity, printed BEFORE any study runs: does the base actually
    // have the copy skill, and is it genuinely ignorant of the study rules?
    let greedy = model::rollout::RolloutParams { max_new: curriculum::OUT_LEN, sample: model::serve::SampleParams::greedy(), eos: None };
    let pre_tasks: Vec<_> = rules
        .iter()
        .copied()
        .flat_map(|r| {
            let e = PositionCopyEnv::new(r, ContentSplit::Eval);
            (0..4).map(move |s| e.tasks(700_000 + s).into_iter().next().unwrap())
        })
        .collect();
    let (pre_scores, _) = improve::score_checkpoint::<Qwen>(&base, &pre_tasks, &PositionCopyVerifier, &greedy);
    let pre_mean = pre_scores.iter().sum::<f64>() / pre_scores.len() as f64;

    let study_env = PositionCopyEnv::new(Rule::for_cycle(0), ContentSplit::Eval);
    let study_tasks: Vec<_> = (0..32).map(|s| study_env.tasks(800_000 + s).into_iter().next().unwrap()).collect();
    let (study_scores, _) = improve::score_checkpoint::<Qwen>(&base, &study_tasks, &PositionCopyVerifier, &greedy);
    let study_mean = study_scores.iter().sum::<f64>() / study_scores.len() as f64;

    println!(
        "fixture ({how}): held-out score on the {} PRETRAINING rules {pre_mean:.3} (the copy skill it was taught) \
         | on study rule T1 {study_mean:.3} (a rule it has never seen) | uniform-token analytic chance {:.3}",
        rules.len(),
        curriculum::uniform_chance()
    );
    assert!(
        pre_mean > 0.60,
        "the pretrained base scored {pre_mean:.3} on rules it was actually trained on - \
         the fixture never learned the copy skill, so nothing measured on top of it means anything"
    );
    base
}

fn study_config_for(seed: u64, cycles: usize, work_dir: PathBuf, plasticity_control: bool, gate_policy: GatePolicy) -> StudyConfig {
    StudyConfig {
        cycles,
        steps_per_cycle: STEPS_PER_CYCLE,
        group_size: GROUP_SIZE,
        grad_accum: GROUP_SIZE as u32,
        eval_per_cycle: EVAL_PER_CYCLE,
        lr: LR,
        min_lr: MIN_LR,
        explore_temp: EXPLORE_TEMP,
        replay_frac: REPLAY_FRAC,
        seed,
        gate: GATE,
        gate_policy,
        plasticity_control,
        work_dir,
        verbose: true,
        // The recorded study's regime, stated rather than defaulted: this
        // file's pre-registered constants describe a GRPO run and nothing
        // else.
        regime: Regime::Grpo,
    }
}

fn spec<'a>(base: &'a Path, targets: &'a [String]) -> StudySpec<'a> {
    StudySpec {
        base_checkpoint: base,
        adapter: AdapterMeta { rank: LORA_RANK, alpha: LORA_ALPHA, targets, family: "qwen", base_id: "poscopy-pretrained", dataset_id: None },
    }
}

fn print_report(report: &StudyReport) {
    println!("\n{}", report.table());
    println!("{}", report.matrix_table());
    println!("{}", report.summary());
}

// ---------------------------------------------------------------------------
// T1 - the pre-registration measurement, and the guard that keeps every other
// threshold honest.
// ---------------------------------------------------------------------------

/// If the box, the toolchain or the model drifts so that single-cycle seed
/// noise exceeds the band the other thresholds were sized against, this test
/// fails FIRST and says the pre-registered numbers are stale - rather than
/// letting the twelve-cycle study pass or fail for the wrong reason.
#[test]
fn seed_noise_band_stays_within_the_preregistered_bound() {
    if gpu_disabled() {
        return;
    }
    println!("pre-registration on record: {PREREG_MEASUREMENT}");
    let dir = tmp("noise");
    let base = prepare_base();
    let targets = lora_targets();

    let mut diagonals = Vec::new();
    let mut b_bases = Vec::new();
    for seed in 1..=5u64 {
        let cfg = study_config_for(seed, 1, dir.join(format!("seed{seed}")), false, GatePolicy::Real);
        let report = continual::run_study::<Qwen, _>(&spec(&base, &targets), &PositionCopy::new(0), &cfg).expect("run_study");
        diagonals.push(report.r_matrix[0][0]);
        b_bases.push(report.b_base);
        println!("seed {seed}: R[1][1] = {:.3}   b_base = {:.3}   gate = {:?}", report.r_matrix[0][0], report.b_base, report.records[0].gate_decision);
    }

    let hi = diagonals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lo = diagonals.iter().cloned().fold(f64::INFINITY, f64::min);
    let spread = hi - lo;
    let mean = diagonals.iter().sum::<f64>() / diagonals.len() as f64;
    let sigma = (diagonals.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / diagonals.len() as f64).sqrt();
    println!("\nR[1][1] over 5 seeds: {diagonals:?}\n  spread s = {spread:.3}   sigma = {sigma:.3}   mean = {mean:.3}   b_base = {:.3}", b_bases[0]);
    println!("  band the study's thresholds were sized against: {PREREG_NOISE_BAND:.3} = max(0.15, 2s)");

    assert!(
        spread <= PREREG_NOISE_BAND,
        "single-cycle seed noise spread {spread:.3} exceeds the pre-registered band {PREREG_NOISE_BAND:.3}: the thresholds in this file were sized against a quieter box and are now STALE. \
         Re-run the pre-registration and record the new numbers rather than loosening a threshold. Observed: {diagonals:?}"
    );
    // Every b_base must be identical: it is measured on the untrained base
    // and frozen probes, neither of which depends on the study seed.
    for b in &b_bases {
        assert!((b - b_bases[0]).abs() < 1e-12, "b_base must not vary with the study seed, got {b_bases:?}");
    }
}

// ---------------------------------------------------------------------------
// T2 - the study.
//
// NAMED FOR ITS RESULT, not for its hypothesis. The 12-cycle loop RUNS, the
// harness is correct, the gate discriminates - and capability does NOT
// accumulate at this model scale. Every pre-registered target A1-A7 is
// printed with its observed value and a PASS/FAIL verdict; the assertions
// below are the ones that are actually true of this system, plus two
// TRIPWIRES that fail if the picture ever improves, so the "not established"
// conclusion in the roadmap cannot silently go stale.
// ---------------------------------------------------------------------------

#[test]
fn twelve_cycles_run_end_to_end_but_capability_does_not_accumulate_at_this_scale() {
    if gpu_disabled() {
        return;
    }
    println!("pre-registration on record: {PREREG_MEASUREMENT}");
    let dir = tmp("study");
    let base = prepare_base();
    let targets = lora_targets();
    let cfg = study_config_for(STUDY_SEED, CYCLES, dir.join("arm1"), true, GatePolicy::Real);
    let report = continual::run_study::<Qwen, _>(&spec(&base, &targets), &PositionCopy::new(0), &cfg).expect("run_study");

    // Print EVERYTHING before asserting anything: a failing assertion must
    // show the whole trajectory it failed inside of, not just its own number.
    print_report(&report);

    let t = CYCLES - 1;
    let r = &report.r_matrix;
    let r_n_1 = r[t][0];
    let r_1_1 = r[0][0];
    let last = report.records.last().expect("a 12-cycle study has records");
    let last_only = report.last_task_only_acc.expect("Arm 2 was enabled");
    let diagonal: Vec<f64> = r.iter().enumerate().map(|(k, row)| row[k]).collect();
    let learned = diagonal.iter().filter(|&&d| d >= PREREG_MIN_DIAGONAL).count();
    let rho_n = last.plasticity_ratio.expect("Arm 2 was enabled");

    // ---- The pre-registered targets, scored honestly --------------------
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    println!(
        "\nPRE-REGISTERED TARGETS (A1-A7), observed vs target:\n\
         A1 retention      R[N][1] {r_n_1:.3} >= R[1][1] {r_1_1:.3} - {PREREG_RETENTION_DROP:.2} = {:.3}   {}\n\
         A2 canary alive   R[N][1] {r_n_1:.3} >= b_base {:.3} + {PREREG_PROBE_MARGIN:.2} = {:.3}   {}\n\
         A3 no collapse    min distinct fraction {:.3} >= {PREREG_MIN_DISTINCT_FRAC:.2}   {}\n\
         A4 sustained      promotions {} >= {PREREG_MIN_PROMOTIONS}   {}\n\
         A5 really learned R[k][k] >= {PREREG_MIN_DIAGONAL:.2} on {learned} of {CYCLES} cycles (need {PREREG_MIN_PROMOTIONS})   {}\n\
         A6 plasticity     rho(N) {rho_n:.3} >= {PREREG_MIN_RHO_AT_N:.2}   {}\n\
         A7 accumulation   ACC {:.3} >= last-task-only {last_only:.3} + {PREREG_ACC_OVER_LAST_ONLY:.2} = {:.3}   {}",
        r_1_1 - PREREG_RETENTION_DROP,
        verdict(r_n_1 >= r_1_1 - PREREG_RETENTION_DROP),
        report.b_base,
        report.b_base + PREREG_PROBE_MARGIN,
        verdict(r_n_1 >= report.b_base + PREREG_PROBE_MARGIN),
        min_distinct_frac(&report),
        verdict(min_distinct_frac(&report) >= PREREG_MIN_DISTINCT_FRAC),
        report.promotions,
        verdict(report.promotions >= PREREG_MIN_PROMOTIONS),
        verdict(learned >= PREREG_MIN_PROMOTIONS),
        verdict(rho_n >= PREREG_MIN_RHO_AT_N),
        report.acc,
        last_only + PREREG_ACC_OVER_LAST_ONLY,
        verdict(report.acc >= last_only + PREREG_ACC_OVER_LAST_ONLY),
    );

    // ---- What the run DOES establish, asserted --------------------------

    // H1 - the harness ran the full protocol and its matrix is well formed.
    assert_eq!(report.records.len(), CYCLES);
    assert_eq!(r.len(), CYCLES);
    for (i, row) in r.iter().enumerate() {
        assert_eq!(row.len(), i + 1, "the retention matrix must be lower-triangular; row {i} has {} entries", row.len());
        assert!(row.iter().all(|v| (0.0..=1.0).contains(v)), "row {i} carries an out-of-range score: {row:?}");
    }
    // H2 - the reported aggregates really are computed from that matrix.
    assert!((report.acc - continual::acc(r)).abs() < 1e-12);
    assert!((report.bwt - continual::bwt(r)).abs() < 1e-12);

    // H3 - both structural properties held, over a non-trivial amount.
    // `assert_trained_spans_were_sampled` already ran in-harness on EVERY
    // cycle (it panics, so reaching here means it passed on all 12).
    assert_eq!(report.probe_ids_checked, CYCLES * EVAL_PER_CYCLE, "not every frozen probe id was hashed for disjointness");
    assert!(report.explore_ids_checked >= CYCLES * 256, "the explore split was not sampled deeply enough");
    for rec in &report.records {
        assert!(rec.sampled_spans > 0, "cycle {} sampled nothing", rec.cycle + 1);
        if rec.applied_promote {
            assert!(rec.trained_spans > 0, "cycle {} promoted without training on a single sampled span", rec.cycle + 1);
        }
    }

    // H4 - no cycle collapsed onto a single output (the real, direct
    // mode-collapse check; see GATE's doc comment for why the entropy ratio
    // cannot make this call here).
    for rec in &report.records {
        let frac = rec.distinct_completions as f64 / rec.decoded_tasks as f64;
        assert!(
            frac >= PREREG_MIN_DISTINCT_FRAC,
            "after cycle {} the servable model produced only {} distinct completions over {} probes ({frac:.3})",
            rec.cycle + 1,
            rec.distinct_completions,
            rec.decoded_tasks
        );
    }

    // H5 - the FIRST cycle is genuinely real learning, well clear of the
    // untrained base's own score. This is what makes the later cycles'
    // failure a statement about CHAINING rather than about the setup.
    assert!(
        r_1_1 >= report.b_base + PREREG_PROBE_MARGIN,
        "cycle 1 only reached {r_1_1:.3} against the untrained base's {:.3} - without a real first cycle the rest of this run says nothing",
        report.b_base
    );

    // H6 - the gate is a gate, not a rubber stamp: it both promoted and
    // rejected within one run, and the anchor arm (retention) is load-bearing
    // rather than decorative - it fired.
    let rejects: Vec<&rl::continual::CycleRecord> = report.records.iter().filter(|x| matches!(x.gate_decision, Decision::Reject(_))).collect();
    assert!(report.promotions >= 1, "the gate promoted nothing at all - this run cannot distinguish a working gate from a broken loop");
    assert!(!rejects.is_empty(), "the gate promoted every cycle - it is a rubber stamp in this run");
    assert!(
        report.records.iter().any(|x| matches!(x.gate_decision, Decision::Reject(rl::gate::Cause::AnchorRegressed { .. }))),
        "no cycle was ever rejected for regressing the retention anchor, so Cause::AnchorRegressed is still decorative in this run"
    );

    // ---- TRIPWIRES ------------------------------------------------------
    // These pin the NEGATIVE result. If either fires, the loop got better
    // than it was when the roadmap's P19 verdict was written, and that
    // verdict must be re-derived from a fresh run rather than left standing.
    assert!(
        report.promotions < PREREG_MIN_PROMOTIONS,
        "TRIPWIRE (good news): {} of {CYCLES} cycles promoted, at or above the pre-registered {PREREG_MIN_PROMOTIONS}. \
         The roadmap's P19 'capability does not accumulate at this scale' verdict is now STALE - re-run the study, \
         re-derive the claim, and rename this test.",
        report.promotions
    );
    assert!(
        report.acc < last_only + PREREG_ACC_OVER_LAST_ONLY,
        "TRIPWIRE (good news): ACC {:.3} now beats the last-task-only control {last_only:.3} by the pre-registered margin. \
         The roadmap's P19 verdict is STALE - re-derive it.",
        report.acc
    );

    println!(
        "\nWHAT THIS RUN DOES AND DOES NOT ESTABLISH\n\
         ESTABLISHES, over {CYCLES} sequential real cycles (rollout -> verify -> GRPO -> gate -> promote/reject) on the \
         position-copy family, at this model scale, with ONE seed and a frozen pretrained base: the harness runs the full \
         protocol; the retention matrix is real and comes from the gate's own decodes; both structural properties held on \
         every cycle (every trained completion span was a member of the multiset the policy actually sampled, and all {} \
         frozen probe ids were disjoint from {} explore ids); no cycle collapsed onto a single output; the first cycle is \
         real learning ({r_1_1:.3} against the untrained base's own zero-shot score on that SAME 16-probe task, {:.3}); \
         and the gate discriminated - {} promotions, {} rejects, including a retention-anchor rejection.\n\
         DOES NOT ESTABLISH - and the run says the opposite: capability did NOT accumulate. Final ACC {:.3} (a 192-probe, \
         {CYCLES}-task aggregate) sits BELOW b_base {:.3} - the untrained base's zero-shot score on ONLY probe T1's 16 \
         probes, since the base's score on the other {} probe sets was never separately measured, so this is a \
         same-task-scale comparison, not an apples-to-apples 192-probe baseline - and only barely above a last-task-only \
         model's {last_only:.3}; BWT {:+.3}; the cycle-1 canary fell {r_1_1:.3} -> {r_n_1:.3}; only {learned} of {CYCLES} \
         cycles reached R[k][k] >= {PREREG_MIN_DIAGONAL:.2}. rho(N) = {rho_n:.3} and its OLS slope over cycle index are \
         NOT reliable plasticity summary statistics here: rho is a ratio whose denominator (the fresh-adapter control) \
         is sometimes itself near zero, which can produce large or small ratios that say more about the denominator than \
         about the numerator - the OLS slope's 95% CI {:?} reflects exactly that instability rather than a real trend. \
         This does NOT mean the fresh control sat near the floor throughout - per-cycle warm-vs-fresh scores must be read \
         individually, not summarized by rho's slope alone.\n\
         The binding constraint is measured, not guessed: see the joint-training oracle (Arm 3), which cannot hold the 12 \
         rules either even when trained on all of them AT ONCE.",
        report.probe_ids_checked,
        report.explore_ids_checked,
        report.b_base,
        report.promotions,
        rejects.len(),
        report.acc,
        report.b_base,
        CYCLES - 1,
        report.bwt,
        report.plasticity_slope.map(|s| (s.ci_lo, s.ci_hi)),
    );
}

fn min_distinct_frac(report: &StudyReport) -> f64 {
    report
        .records
        .iter()
        .map(|r| r.distinct_completions as f64 / r.decoded_tasks as f64)
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// T3 - the capacity control, and the reason T2's BWT is not a forgetting
// result.
// ---------------------------------------------------------------------------

#[test]
fn joint_training_oracle_shows_capacity_is_the_binding_constraint_not_forgetting() {
    if gpu_disabled() {
        return;
    }
    println!("pre-registration on record: {PREREG_MEASUREMENT}");
    let dir = tmp("oracle");
    let base = prepare_base();
    let targets = lora_targets();
    let cfg = study_config_for(STUDY_SEED, CYCLES, dir.join("arm3"), false, GatePolicy::Real);
    let started = std::time::Instant::now();
    let oracle = continual::joint_oracle::<Qwen, _>(&spec(&base, &targets), &PositionCopy::new(0), &cfg).expect("joint_oracle");
    println!(
        "joint-training oracle: one fresh rank-{LORA_RANK} adapter on all {CYCLES} rules pooled for {} GRPO steps -> \
         ACC {oracle:.3} over all {} frozen probes, against a pre-registered {PREREG_MIN_ORACLE_ACC:.2} ({:.1}s)",
        STEPS_PER_CYCLE * CYCLES as u32,
        CYCLES * EVAL_PER_CYCLE,
        started.elapsed().as_secs_f64()
    );
    assert!((0.0..=1.0).contains(&oracle), "the oracle must report a fraction, got {oracle}");

    // TRIPWIRE, same contract as T2's: this pins the measured NEGATIVE. The
    // oracle is the capacity control - one adapter trained on all 12 rules AT
    // ONCE, with the whole study's step budget, and no sequential interference
    // whatsoever. It lands far below the pre-registered bar and below the
    // untrained base's own score, so "the loop forgot task 1" and "a rank-8
    // adapter driven by GRPO at this budget cannot represent 12 cue-conditioned
    // rules at all" are NOT distinguishable by T2's run, and they have opposite
    // fixes. T2's BWT must therefore be read as a CAPACITY/OPTIMIZATION result,
    // not a forgetting result.
    assert!(
        oracle < PREREG_MIN_ORACLE_ACC,
        "TRIPWIRE (good news): joint-training oracle ACC {oracle:.3} now clears the pre-registered {PREREG_MIN_ORACLE_ACC:.2}. \
         Capacity is no longer the binding constraint, so the roadmap's P19 reading of T2's BWT as a capacity result is \
         STALE and must be re-derived from a fresh run."
    );
}
