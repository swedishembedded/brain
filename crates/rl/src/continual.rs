// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-cycle continual learning: run [`crate::improve::cycle`] N times in a
//! row over a curriculum of genuinely different, deliberately
//! difficulty-invariant tasks, and measure what a single cycle structurally
//! cannot - retention, forward transfer, and plasticity.
//!
//! ## What one cycle can and cannot establish
//!
//! One trained-and-promoted cycle establishes that a gated improvement loop
//! closes once. It says nothing about whether the second cycle destroys the
//! first one's capability, whether the tenth cycle can still learn at all, or
//! whether the gate is carrying information rather than noise. Those are
//! properties of a SEQUENCE, and every one of them needs a control:
//!
//! - **Retention** needs probe sets frozen at introduction and never trained
//!   on again. Here cycle *k*'s 16 evaluation tasks are simultaneously the
//!   gate's held-out set at cycle *k* and task *k*'s retention probe forever
//!   after, so the retention matrix falls out of the gate's own decodes at
//!   zero extra cost and [`crate::gate::Cause::AnchorRegressed`] becomes
//!   load-bearing instead of decorative.
//! - **Plasticity** needs a fresh-adapter control trained on the same task
//!   with the same step budget (Arm 2): "cycle 12 scored lower" is otherwise
//!   indistinguishable from "cycle 12's task was harder", which is why the
//!   task family must be difficulty-invariant by construction.
//! - **Capacity** needs a joint-training oracle ([`joint_oracle`], Arm 3):
//!   without it, "the loop forgot" and "the adapter is too small to hold N
//!   rules" are indistinguishable, and they have opposite fixes.
//! - **Gate informativeness** needs a null gate ([`GatePolicy::CoinFlip`],
//!   Arm 4): if the real loop is not separated from a coin flip beyond seed
//!   noise, the gate is decorative.
//!
//! ## What a passing run does NOT prove
//!
//! Not scale (N cycles at this size says nothing about 10^3 cycles or 10^9
//! parameters - loss of plasticity in deep continual learning has needed
//! ~2000 sequential tasks to become unambiguous). Not the slope (one seed
//! times N points cannot power a slope test; [`StudyReport::
//! plasticity_slope`] is REPORTED as a diagnostic and only a coarse level
//! bound is ever worth asserting). Not generality (the task family is
//! synthetic and difficulty-invariant by construction - that control was
//! bought by removing exactly the properties that break real systems). Not
//! live operation, not seed robustness, not order independence, and not
//! recursive self-improvement: this is a well-instrumented incremental-
//! learning experiment with a ratchet.
//!
//! Swedish Embedded AB builds the measurement harnesses that separate "the
//! model improved" from "the model improved without destroying what it knew,
//! and can still learn the next thing" - retention matrices, plasticity
//! controls, capacity oracles, and null-gate arms wired into the training
//! loop itself. If your team needs expertise in evaluating a continual
//! learning system honestly, you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use model::rollout::RolloutParams;
use model::serve::SampleParams;
use model::{FitOpts, Model, ModelConfig};

use crate::curriculum::{ContentSplit, PositionCopyEnv, PositionCopyVerifier, Rule, OUT_LEN, PROMPT_LEN};
use crate::env::{Environment, Task, Verifier};
use crate::gate::{Decision, GateConfig, GateReport};
use crate::improve::{self, AdapterMeta, ArmScores, CycleArtifacts, Evaluation, ProvenanceInput};
use crate::objective::grpo::{CycleLog, Grpo, GrpoConfig};

/// Base seed for the frozen evaluation probes. Probe seeds are a pure
/// function of `(cycle, index)`, never of [`StudyConfig::seed`]: the probe
/// SETS must be identical across every arm and every seed of the study, or
/// the arms are not scored on the same thing.
const PROBE_SEED_BASE: u64 = 900_000_000;
/// How many explore draws per cycle the pre-loop split-integrity check runs.
const EXPLORE_AUDIT_DRAWS: u64 = 256;

/// A sequence of tasks to learn, one per cycle.
pub trait Curriculum {
    type Env: Environment;
    type Ver: Verifier;

    /// Cycle `k`'s TRAINING environment (explore content split).
    fn env_for(&self, cycle: usize) -> Self::Env;

    /// Cycle `k`'s EVALUATION environment - same rule, disjoint content split
    /// by construction. Its tasks are frozen at introduction and are both the
    /// gate's held-out set at cycle `k` and task `k`'s retention probe
    /// forever after.
    fn eval_env_for(&self, cycle: usize) -> Self::Env;

    fn verifier(&self) -> Self::Ver;

    /// Human-readable name for cycle `k`'s task, e.g. `"cue02 picks(3,0,4)"`.
    fn label(&self, cycle: usize) -> String;

    /// `(prompt_len, completion_len)` - drives [`GrpoConfig::seq_len`] and
    /// [`RolloutParams::max_new`]. Every cycle MUST report the same shape,
    /// and every cycle's task MUST be of identical difficulty by
    /// construction: that invariance is the only thing that makes the
    /// plasticity ratio interpretable, because it is what lets "cycle 12
    /// scored lower" mean "the model learns less well now" rather than
    /// "cycle 12's task was harder".
    fn shape(&self) -> (usize, usize);

    /// Background environments the base ALREADY solves, mixed into EVERY
    /// cycle's training draw (including cycle 1). Default: none.
    ///
    /// This exists because of a measured failure, not as a convenience. With
    /// no rehearsal, cycle 1's training distribution contains exactly ONE
    /// cue, so the cheapest policy that solves cycle 1 ignores the cue
    /// entirely - and an unconditional policy is a hole every later cycle
    /// then has to climb out of. Rehearsal on rules the base already knows
    /// makes the cue-independent shortcut score ZERO on part of the mix, so
    /// the shortcut stops being cheap and the policy has to stay
    /// cue-conditional from cycle 1 onward. See [`ReplayEnv`].
    fn rehearsal_envs(&self) -> Vec<Self::Env> {
        Vec::new()
    }
}

/// The one [`Curriculum`] impl: [`crate::curriculum`]'s position-copy family,
/// one rule per cycle, plus `rehearsal` of the PRETRAINING rules (cues no
/// study cycle uses, which the frozen base already solves).
pub struct PositionCopy {
    pub rehearsal: usize,
}

impl PositionCopy {
    pub fn new(rehearsal: usize) -> PositionCopy {
        PositionCopy { rehearsal }
    }
}

impl Default for PositionCopy {
    fn default() -> Self {
        PositionCopy { rehearsal: 4 }
    }
}

impl Curriculum for PositionCopy {
    type Env = PositionCopyEnv;
    type Ver = PositionCopyVerifier;

    fn env_for(&self, cycle: usize) -> PositionCopyEnv {
        PositionCopyEnv::new(Rule::for_cycle(cycle), ContentSplit::Explore)
    }
    fn eval_env_for(&self, cycle: usize) -> PositionCopyEnv {
        PositionCopyEnv::new(Rule::for_cycle(cycle), ContentSplit::Eval)
    }
    fn verifier(&self) -> PositionCopyVerifier {
        PositionCopyVerifier
    }
    fn label(&self, cycle: usize) -> String {
        let r = Rule::for_cycle(cycle);
        format!("cue{:02} picks({},{},{})", r.cue, r.picks[0], r.picks[1], r.picks[2])
    }
    fn shape(&self) -> (usize, usize) {
        (PROMPT_LEN, OUT_LEN)
    }
    fn rehearsal_envs(&self) -> Vec<PositionCopyEnv> {
        Rule::pretrain_rules(self.rehearsal).into_iter().map(|r| PositionCopyEnv::new(r, ContentSplit::Explore)).collect()
    }
}

/// Cycle `k`'s TRAINING distribution: the new rule with probability
/// `1 - replay_frac`, a uniformly chosen EARLIER rule otherwise. Ordinary
/// experience replay, and the reason it is here is a measurement, not a
/// preference.
///
/// With `replay_frac = 0.0` (pure sequential), this loop provably stalls
/// after one cycle at this scale, and the mechanism is specific and
/// diagnosable: cycle 1 only ever sees ONE cue, so the cheapest policy that
/// solves it is CUE-INDEPENDENT ("always emit picks(0,1,2)"). Measured on
/// this box, cycle 1 reaches 1.000 that way and every later cycle is then
/// stuck at 0.00-0.17, because unlearning an unconditional rule is far harder
/// than learning a conditional one. Raising the exploration temperature
/// (1.0 / 1.5 / 2.0 / 2.5), the group size (2 / 4), the step budget (240 /
/// 960) and the adapter rank (8 / 16, attention-only / attention + MLP) all
/// leave that unchanged - a longer cycle 1 makes it strictly WORSE, because a
/// more converged unconditional policy is a deeper hole.
///
/// Replay removes the degenerate solution rather than papering over it: once
/// a cycle's training distribution contains more than one cue, a
/// cue-independent policy cannot score well on it, so the policy has to learn
/// the discrimination the retention matrix is trying to measure. Any run
/// using it is a "continual learning WITH REPLAY" result and must be reported
/// as such - it is a different, weaker claim than pure sequential learning.
struct ReplayEnv<'a, C: Curriculum> {
    curr: &'a C,
    cycle: usize,
    replay_frac: f64,
    rehearsal: Vec<C::Env>,
}

impl<'a, C: Curriculum> ReplayEnv<'a, C> {
    fn new(curr: &'a C, cycle: usize, replay_frac: f64) -> ReplayEnv<'a, C> {
        ReplayEnv { curr, cycle, replay_frac, rehearsal: curr.rehearsal_envs() }
    }
}

impl<C: Curriculum> Environment for ReplayEnv<'_, C> {
    fn name(&self) -> &str {
        "replay"
    }
    fn tasks(&self, seed: u64) -> Vec<Task> {
        let mut rng = data::rng::Rng::new(seed ^ 0x5EED_5EED);
        if rng.next_f64() >= self.replay_frac {
            return self.curr.env_for(self.cycle).tasks(seed);
        }
        // Replay: an earlier cycle's rule, or - crucially at cycle 1, where
        // there IS no earlier cycle - a rehearsal rule the base already
        // solves. Both are "something a cue-independent policy gets wrong".
        let n_prior = self.cycle;
        let total = n_prior + self.rehearsal.len();
        if total == 0 {
            return self.curr.env_for(self.cycle).tasks(seed);
        }
        let pick = (rng.next_u64() % total as u64) as usize;
        if pick < n_prior {
            self.curr.env_for(pick).tasks(seed)
        } else {
            self.rehearsal[pick - n_prior].tasks(seed)
        }
    }
}

/// Every rule pooled into one environment - [`joint_oracle`]'s task source.
struct PooledEnv<'a, C: Curriculum> {
    curr: &'a C,
    cycles: usize,
}

impl<C: Curriculum> Environment for PooledEnv<'_, C> {
    fn name(&self) -> &str {
        "pooled"
    }
    fn tasks(&self, seed: u64) -> Vec<Task> {
        let k = (seed % self.cycles as u64) as usize;
        self.curr.env_for(k).tasks(seed)
    }
}

/// What decides which checkpoint carries forward into the next cycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GatePolicy {
    /// The real [`crate::gate::gate`] decision decides.
    Real,
    /// Arm 4: the real gate still runs and is still recorded, but a coin flip
    /// decides what carries forward. If Arm 1 is not separated from this
    /// beyond seed noise, the gate is decorative.
    CoinFlip { seed: u64 },
}

impl GatePolicy {
    /// Whether cycle `k`'s candidate actually carries forward. Deterministic
    /// for a fixed seed, so an Arm-4 run is reproducible.
    pub fn applies(&self, decision: Decision, cycle: usize) -> bool {
        match *self {
            GatePolicy::Real => decision == Decision::Promote,
            GatePolicy::CoinFlip { seed } => coin(seed, cycle),
        }
    }
}

/// SplitMix64 finalizer over `(seed, cycle)` - a reproducible coin.
fn coin(seed: u64, cycle: usize) -> bool {
    let mut z = seed.wrapping_add((cycle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) & 1) == 1
}

/// Everything about a study run that is not architecture-specific.
#[derive(Clone, Debug)]
pub struct StudyConfig {
    pub cycles: usize,
    pub steps_per_cycle: u32,
    pub group_size: usize,
    /// Rows per optimizer step. Set equal to `group_size` so one whole GRPO
    /// group's advantages land in one update.
    pub grad_accum: u32,
    /// Frozen evaluation tasks per cycle - simultaneously the gate's held-out
    /// set and that cycle's permanent retention probe.
    pub eval_per_cycle: usize,
    pub lr: f32,
    pub min_lr: f32,
    /// Sampling temperature for TRAINING rollouts (evaluation is always
    /// greedy).
    ///
    /// Not a free knob: it is the exploration control the loop lives or dies
    /// on. `Grpo` learns only from a group whose members earned DIFFERENT
    /// rewards - `group_advantages` drops a zero-variance group outright - so
    /// once a cycle has driven the policy to be confident, sampling at
    /// temperature 1.0 draws the same completion `group_size` times, every
    /// group is dropped, and the next cycle trains on almost nothing. Measured
    /// on this box at temperature 1.0: cycle 1 trained on 136 of 834 sampled
    /// spans and scored 0.938, and by cycle 3 the SAME loop trained on 34 of
    /// 936 and scored 0.083. That is an exploration collapse, not a capacity
    /// or a forgetting result, and reading it as one would have been exactly
    /// the wrong conclusion.
    pub explore_temp: f32,
    /// Fraction of cycle `k`'s training draws that come from an EARLIER
    /// cycle's rule instead of the new one - see [`ReplayEnv`] for the
    /// measurement that made this necessary and for what it costs the claim.
    /// `0.0` is pure sequential learning.
    pub replay_frac: f64,
    pub seed: u64,
    pub gate: GateConfig,
    pub gate_policy: GatePolicy,
    /// Arm 2: train a fresh adapter per cycle at an identical budget, to get
    /// both the transfer-vs-fresh-control baseline and the plasticity
    /// denominator (see [`xfer_vs_fresh_control`] for why this is not FWT).
    pub plasticity_control: bool,
    pub work_dir: PathBuf,
    /// Print one row per cycle as it completes (a 12-cycle run is minutes of
    /// wall clock; silence for all of it is not a usable harness).
    pub verbose: bool,
}

/// Architecture-specific facts [`run_study`] cannot derive.
pub struct StudySpec<'a> {
    /// The FROZEN pretrained base + fresh zero-delta adapter: cycle 1's
    /// incumbent, and also Arm 2's fresh-adapter starting point.
    pub base_checkpoint: &'a Path,
    pub adapter: AdapterMeta<'a>,
}

/// One cycle's full record - what the gate said, what actually carried
/// forward, and every number the study's metrics are computed from.
#[derive(Clone, Debug)]
pub struct CycleRecord {
    pub cycle: usize,
    pub label: String,
    pub regime: &'static str,
    /// What the REAL gate said, always, even under [`GatePolicy::CoinFlip`].
    pub gate_decision: Decision,
    /// What actually carried forward - differs from `gate_decision` only
    /// under [`GatePolicy::CoinFlip`].
    pub applied_promote: bool,
    pub report: GateReport,
    /// The candidate arm on cycle `k`'s probe: `R[k][k]` when promoted.
    pub heldout_candidate: f64,
    /// The incumbent arm on cycle `k`'s probe - i.e. the model servable after
    /// cycle `k-1`, zero-shot on a task it has never seen. Feeds
    /// [`xfer_vs_fresh_control`].
    pub heldout_incumbent: f64,
    /// `R[k][0..=k]` from the servable arm - this cycle's whole matrix row.
    pub retention_row: Vec<f64>,
    /// `retention_row[0]`: the single most legible forgetting number.
    pub canary_t1: f64,
    pub plasticity_warm: f64,
    pub plasticity_fresh: Option<f64>,
    pub plasticity_ratio: Option<f64>,
    /// Mean verified reward over every completion this cycle SAMPLED.
    pub mean_reward: f32,
    /// Distinct greedy completions the SERVABLE arm produced across the
    /// `decoded_tasks` it was scored on. The direct mode-collapse
    /// measurement - see `improve::ArmScores::distinct_completions` for why
    /// the gate's entropy ratio cannot make it on a task family with one
    /// correct completion per prompt.
    pub distinct_completions: usize,
    pub decoded_tasks: usize,
    pub sampled_spans: usize,
    pub trained_spans: usize,
    pub wall_secs: f64,
}

impl CycleRecord {
    /// One line of [`StudyReport::table`] - printed live by [`run_study`] as
    /// the cycle completes.
    pub fn row(&self) -> String {
        let gate = match self.gate_decision {
            Decision::Promote => "PROMOTE".to_string(),
            Decision::Reject(c) => format!("reject/{}", cause_name(c)),
        };
        let applied = if self.applied_promote { "yes" } else { "no " };
        let rho = self.plasticity_ratio.map(|r| format!("{r:5.2}")).unwrap_or_else(|| "    -".to_string());
        let distinct = format!("{}/{}", self.distinct_completions, self.decoded_tasks);
        format!(
            "{:3}  {:<20} {:6}  {:6.3}  {:6.3}  {:6.3}  {rho}  {:6.3}  {:5}/{:<5}  {distinct:>7}  {:<14} {applied}  {:6.1}",
            self.cycle + 1,
            self.label,
            self.trained_spans,
            self.heldout_candidate,
            self.heldout_incumbent,
            self.canary_t1,
            self.mean_reward,
            self.trained_spans,
            self.sampled_spans,
            gate,
            self.wall_secs
        )
    }
}

fn cause_name(c: crate::gate::Cause) -> &'static str {
    match c {
        crate::gate::Cause::NotSignificant { .. } => "notsig",
        crate::gate::Cause::EffectTooSmall { .. } => "effect",
        crate::gate::Cause::AnchorRegressed { .. } => "anchor",
        crate::gate::Cause::Degenerate { .. } => "degen",
    }
}

/// The whole study: per-cycle records, the retention matrix, the control
/// arms, and the aggregate metrics computed from them.
#[derive(Clone, Debug)]
pub struct StudyReport {
    pub records: Vec<CycleRecord>,
    /// Lower-triangular retention matrix: `r_matrix[i][j]` is probe *j*'s
    /// mean score under the model servable after cycle *i* (`j <= i`).
    pub r_matrix: Vec<Vec<f64>>,
    /// Arm 2 per cycle - a fresh adapter's score on the same probe at an
    /// identical budget (empty when `plasticity_control` is off).
    pub b_fresh: Vec<f64>,
    /// The untrained base on cycle 1's probe: the chance baseline every
    /// "is this above chance" statement is measured against. Deliberately the
    /// MODEL's own score, not the analytic `1/vocab`, because a real model is
    /// not uniform.
    pub b_base: f64,
    /// Arm 2's cycle-N model scored on ALL probes - "what a system that only
    /// ever learns the newest task achieves".
    pub last_task_only_acc: Option<f64>,
    /// Arm 3, filled by [`joint_oracle`].
    pub joint_oracle_acc: Option<f64>,
    pub promotions: usize,
    pub acc: f64,
    pub bwt: f64,
    /// See [`xfer_vs_fresh_control`] - NOT the literature's FWT metric.
    pub xfer_vs_fresh: Option<f64>,
    pub plasticity_slope: Option<bench::metrics::Slope>,
    /// How many frozen probe ids the pre-loop split-integrity check hashed.
    pub probe_ids_checked: usize,
    /// How many explore-split task ids it checked those against.
    pub explore_ids_checked: usize,
}

/// The header [`run_study`] prints above its live per-cycle rows, and
/// [`StudyReport::table`] repeats.
const TABLE_HEADER: &str = "cyc  task                 trained  heldout   zero0  probeT1    rho  reward  train/sampled  distinct  gate           kept    secs";

impl StudyReport {
    pub fn table(&self) -> String {
        let mut s = String::from(TABLE_HEADER);
        s.push('\n');
        for r in &self.records {
            s.push_str(&r.row());
            s.push('\n');
        }
        s
    }

    /// The full retention matrix as an ASCII grid. Rows are "after cycle i",
    /// columns are probe j; `.` means task j had not been introduced yet.
    pub fn matrix_table(&self) -> String {
        let t = self.r_matrix.len();
        let mut s = String::from("retention matrix R[i][j]  (rows = model servable after cycle i, cols = frozen probe j)\n     ");
        for j in 0..t {
            s.push_str(&format!(" T{:<3}", j + 1));
        }
        s.push('\n');
        for (i, row) in self.r_matrix.iter().enumerate() {
            s.push_str(&format!("c{:<3} ", i + 1));
            for j in 0..t {
                match row.get(j) {
                    Some(v) => s.push_str(&format!(" {v:.2}")),
                    None => s.push_str("    ."),
                }
            }
            s.push('\n');
        }
        s
    }

    pub fn summary(&self) -> String {
        let t = self.r_matrix.len();
        let xvf = self.xfer_vs_fresh.map(|v| format!("{v:+.3}")).unwrap_or_else(|| "n/a".to_string());
        let rho_n = self.records.last().and_then(|r| r.plasticity_ratio).map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".to_string());
        let slope = match self.plasticity_slope {
            Some(s) => format!("slope {:+.4}/cycle, 95% CI [{:+.4}, {:+.4}]", s.slope, s.ci_lo, s.ci_hi),
            None => "not enough points".to_string(),
        };
        let mut causes: Vec<String> = Vec::new();
        for r in &self.records {
            if let Decision::Reject(c) = r.gate_decision {
                causes.push(format!("c{}:{}", r.cycle + 1, cause_name(c)));
            }
        }
        let causes = if causes.is_empty() { "-".to_string() } else { causes.join(" ") };
        let oracle = self.joint_oracle_acc.map(|v| format!("{v:.3}")).unwrap_or_else(|| "not run".to_string());
        let last_only = self.last_task_only_acc.map(|v| format!("{v:.3}")).unwrap_or_else(|| "not run".to_string());
        format!(
            "ACC {:.3}   BWT {:+.3}   xfer_vs_fresh {xvf}   promotions {}/{t}   rejects: {causes}\n\
             chance baseline b_base (untrained base on probe T1) {:.3}\n\
             plasticity rho(N) {rho_n}   {slope}\n\
             joint-training oracle ACC {oracle}   last-task-only ACC {last_only}\n\
             split integrity: {} frozen probe ids checked disjoint from {} explore ids\n",
            self.acc, self.bwt, self.promotions, self.b_base, self.probe_ids_checked, self.explore_ids_checked
        )
    }
}

/// `ACC = (1/T) Σ_j R[T][j]` - the final model's mean score over every
/// frozen probe (Lopez-Paz & Ranzato's average accuracy).
pub fn acc(r: &[Vec<f64>]) -> f64 {
    let Some(last) = r.last() else { return 0.0 };
    if last.is_empty() {
        return 0.0;
    }
    last.iter().sum::<f64>() / last.len() as f64
}

/// `BWT = (1/(T-1)) Σ_{j<T} (R[T][j] - R[j][j])` - how much every earlier
/// task moved, on average, between the moment it was learned and the end of
/// the run. Negative is forgetting.
pub fn bwt(r: &[Vec<f64>]) -> f64 {
    let t = r.len();
    if t < 2 {
        return 0.0;
    }
    let last = &r[t - 1];
    let mut total = 0.0;
    for j in 0..t - 1 {
        total += last[j] - r[j][j];
    }
    total / (t - 1) as f64
}

/// NOT the literature's forward-transfer (FWT) metric, on purpose - Lopez-
/// Paz & Ranzato's FWT compares against an UNTRAINED random-init reference
/// (`b̄_k`), which this harness never scores: Arm 2 always trains its fresh
/// adapter for the full per-cycle budget (see
/// [`StudyConfig::plasticity_control`]), because the same fresh-adapter
/// score doubles as the plasticity denominator. `zero_shot[k] - b_fresh[k]`
/// therefore compares against a FULLY-TRAINED fresh control and is
/// guaranteed negative whenever training works at all - it answers "does
/// the incumbent's accumulated experience transfer to an unseen task at
/// least as well as a fresh adapter trained on that task alone", not "does
/// it beat having learned nothing". Reading this as FWT invites exactly
/// that confusion (found on adversarial review of this harness's own
/// output), which is why it is named and printed as `xfer_vs_fresh`, not
/// `fwt`.
///
/// `(1/(T-1)) Σ_{k>=2} (zero_shot[k] - b_fresh[k])`. `zero_shot[k]` is
/// `R[k-1][k]` (the incumbent arm's score on cycle `k`'s probe, free from
/// the gate's own decodes); `b_fresh[k]` is Arm 2's same-cycle fresh-adapter
/// score.
pub fn xfer_vs_fresh_control(zero_shot: &[f64], b_fresh: &[f64]) -> f64 {
    assert_eq!(zero_shot.len(), b_fresh.len(), "continual::xfer_vs_fresh_control: zero-shot and fresh-control series must be parallel");
    let t = zero_shot.len();
    if t < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in 1..t {
        total += zero_shot[k] - b_fresh[k];
    }
    total / (t - 1) as f64
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// The frozen probe seeds for cycle `k`. A pure function of `(k, i)` - never
/// of the study seed - so every arm and every seed scores the same tasks.
fn probe_seeds(cycle: usize, n: usize) -> Vec<u64> {
    (0..n).map(|i| PROBE_SEED_BASE + (cycle as u64) * 10_000 + i as u64).collect()
}

fn cycle_opts(cfg: &StudyConfig, cycle: usize, block: u32) -> FitOpts {
    FitOpts {
        steps: cfg.steps_per_cycle,
        batch_size: 1,
        block_size: block,
        lr: cfg.lr,
        min_lr: cfg.min_lr,
        warmup: 0,
        decay_iters: cfg.steps_per_cycle,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: cfg.grad_accum,
        eval_interval: 0,
        eval_batches: 0,
        seed: cfg.seed.wrapping_add(cycle as u64 * 1_000),
        checkpoint_secs: 0,
        ..FitOpts::default()
    }
}

fn grpo_cfg(cfg: &StudyConfig, block: usize, completion_len: usize) -> GrpoConfig {
    GrpoConfig {
        group_size: cfg.group_size,
        clip_eps: 0.2,
        kl_beta: 0.0,
        seq_len: block,
        rollout: RolloutParams { max_new: completion_len, sample: SampleParams { temp: cfg.explore_temp, top_k: 0, top_p: 1.0 }, eos: None },
        max_attempts: 1,
    }
}

fn greedy(completion_len: usize) -> RolloutParams {
    RolloutParams { max_new: completion_len, sample: SampleParams::greedy(), eos: None }
}

/// The `block_size` the base checkpoint was actually written at - what
/// [`crate::improve::cycle`] will construct the model with, and therefore
/// what [`GrpoConfig::seq_len`] must equal.
fn block_of<M: Model>(path: &Path) -> u32 {
    let c = checkpoint::load(path.to_str().expect("utf-8 path"));
    M::Config::from_json(&c.header["config"]).block_size()
}

/// Run the whole study: `cfg.cycles` sequential gated cycles (Arm 1), plus
/// the per-cycle fresh-adapter control (Arm 2) when enabled.
pub fn run_study<M: Model, C: Curriculum>(spec: &StudySpec, curr: &C, cfg: &StudyConfig) -> std::io::Result<StudyReport> {
    assert!(cfg.cycles >= 1, "continual::run_study: a study needs at least one cycle");
    assert!(cfg.eval_per_cycle >= 1, "continual::run_study: a study needs at least one probe per cycle");
    let (prompt_len, completion_len) = curr.shape();
    let block = block_of::<M>(spec.base_checkpoint);
    assert!(
        prompt_len + completion_len <= block as usize,
        "continual::run_study: the curriculum's {prompt_len}+{completion_len} token shape does not fit the base checkpoint's block_size {block}"
    );

    // ---- Pre-loop split integrity (structural, runs every study) ----------
    //
    // Freeze every cycle's probe set FIRST, then assert by id hash that no
    // probe id repeats anywhere in the study and that no probe id is
    // reachable from any cycle's explore split. The content-space partition
    // in `curriculum` already makes the second property structural; this is
    // the belt-and-braces check that it actually held, on the ids that were
    // really generated, rather than on the argument that it must.
    let probes: Vec<Vec<Task>> = (0..cfg.cycles)
        .map(|k| {
            let env = curr.eval_env_for(k);
            probe_seeds(k, cfg.eval_per_cycle).into_iter().map(|s| env.tasks(s).into_iter().next().expect("Curriculum::eval_env_for produced no task")).collect()
        })
        .collect();
    let mut probe_ids: HashSet<String> = HashSet::new();
    for row in &probes {
        for t in row {
            assert!(
                probe_ids.insert(t.id.clone()),
                "continual::run_study: probe task id {} appears in more than one frozen probe set - the retention probes must be pairwise disjoint or a matrix row scores the wrong task",
                t.id
            );
        }
    }
    assert_eq!(probe_ids.len(), cfg.cycles * cfg.eval_per_cycle);
    let mut explore_ids_checked = 0usize;
    for k in 0..cfg.cycles {
        let env = curr.env_for(k);
        for s in 0..EXPLORE_AUDIT_DRAWS {
            let t = env.tasks(s.wrapping_mul(0x9E37_79B9).wrapping_add(k as u64)).into_iter().next().expect("Curriculum::env_for produced no task");
            assert!(
                !probe_ids.contains(&t.id),
                "continual::run_study: explore task {} is also a frozen retention probe - a held-out score is only honest if the policy never trained on the task it is scored against",
                t.id
            );
            explore_ids_checked += 1;
        }
    }

    // ---- The chance baseline: the untrained base on cycle 1's probe ------
    let verifier = curr.verifier();
    let greedy_params = greedy(completion_len);
    let (base_scores, _) = improve::score_checkpoint::<M>(spec.base_checkpoint, &probes[0], &verifier, &greedy_params);
    let b_base = mean(&base_scores);

    std::fs::create_dir_all(&cfg.work_dir)?;
    let adapter_dir = cfg.work_dir.join("adapters");

    if cfg.verbose {
        println!("chance baseline (untrained base on probe T1): {b_base:.3}");
        println!("{TABLE_HEADER}");
    }

    let mut incumbent: PathBuf = spec.base_checkpoint.to_path_buf();
    let mut records: Vec<CycleRecord> = Vec::with_capacity(cfg.cycles);
    let mut r_matrix: Vec<Vec<f64>> = Vec::with_capacity(cfg.cycles);
    let mut b_fresh: Vec<f64> = Vec::new();
    let mut promotions = 0usize;
    let mut last_fresh: Option<PathBuf> = None;

    for k in 0..cfg.cycles {
        let started = Instant::now();
        let cycle_dir = cfg.work_dir.join(format!("cycle{k:02}"));
        std::fs::create_dir_all(&cycle_dir)?;
        let train_out = cycle_dir.join("train.safetensors");

        let held_out = &probes[k];
        let anchor: Vec<Task> = probes[..k].iter().flatten().cloned().collect();
        assert_eq!(
            anchor.len(),
            k * cfg.eval_per_cycle,
            "continual::run_study: the anchor suite must be exactly the earlier cycles' probes in cycle order - slicing it back into matrix cells depends on that order"
        );

        let log = CycleLog::new();
        let train_env = ReplayEnv::new(curr, k, cfg.replay_frac);
        let objective = Grpo::new(train_env, curr.verifier(), grpo_cfg(cfg, block as usize, completion_len)).with_log(log.clone());
        let opts = cycle_opts(cfg, k, block);

        let outcome = improve::cycle::<M, _>(
            &incumbent,
            objective,
            &Evaluation { held_out, anchor: &anchor, verifier: &verifier, rollout: &greedy_params, gate_cfg: &cfg.gate },
            &opts,
            CycleArtifacts {
                train_out: &train_out,
                adapter_out_dir: &adapter_dir,
                adapter: spec.adapter,
                provenance: ProvenanceInput {
                    regime: "grpo".to_string(),
                    seed: opts.seed,
                    hyperparams: serde_json::json!({
                        "group_size": cfg.group_size,
                        "grad_accum": cfg.grad_accum,
                        "steps": cfg.steps_per_cycle,
                        "lr": cfg.lr,
                    }),
                    environment: curr.label(k),
                    cycle: k as u64,
                },
            },
        )?;

        // Structural check #1, unconditionally, EVERY cycle (not only on a
        // promotion): every completion span this cycle trained on is a member
        // of the multiset it actually sampled. This panics in-harness rather
        // than returning a flag - a label reaching training invalidates the
        // whole run, so there is nothing to carry forward and report.
        let trained = log.trained();
        let sampled = log.sampled();
        improve::assert_trained_spans_were_sampled(&trained, &sampled);

        let applied_promote = cfg.gate_policy.applies(outcome.decision, k);
        if outcome.decision == Decision::Promote {
            promotions += 1;
        }

        // The R row comes from whichever arm is actually SERVABLE after this
        // cycle - the candidate if it carried forward, the retained incumbent
        // otherwise - never from "the arm we would have liked".
        let servable: &ArmScores = if applied_promote { &outcome.candidate } else { &outcome.incumbent };
        let mut row: Vec<f64> = (0..k).map(|j| mean(&servable.anchor[j * cfg.eval_per_cycle..(j + 1) * cfg.eval_per_cycle])).collect();
        row.push(mean(&servable.held_out));
        let canary_t1 = row[0];

        let heldout_candidate = mean(&outcome.candidate.held_out);
        let heldout_incumbent = mean(&outcome.incumbent.held_out);

        // ---- Arm 2: the fresh-adapter control -----------------------------
        let plasticity_fresh = if cfg.plasticity_control {
            let fresh_out = cycle_dir.join("fresh.safetensors");
            let fresh_log = CycleLog::new();
            // The SAME training distribution as Arm 1, so the plasticity
            // ratio compares two runs that differ only in where they started.
            let fresh_env = ReplayEnv::new(curr, k, cfg.replay_frac);
            let fresh_obj = Grpo::new(fresh_env, curr.verifier(), grpo_cfg(cfg, block as usize, completion_len)).with_log(fresh_log.clone());
            train_from::<M, _>(spec.base_checkpoint, fresh_obj, &opts, &fresh_out)?;
            improve::assert_trained_spans_were_sampled(&fresh_log.trained(), &fresh_log.sampled());
            let (scores, _) = improve::score_checkpoint::<M>(&fresh_out, held_out, &verifier, &greedy_params);
            last_fresh = Some(fresh_out);
            let v = mean(&scores);
            b_fresh.push(v);
            Some(v)
        } else {
            None
        };

        let record = CycleRecord {
            cycle: k,
            label: curr.label(k),
            regime: "grpo",
            gate_decision: outcome.decision,
            applied_promote,
            report: outcome.report,
            heldout_candidate,
            heldout_incumbent,
            retention_row: row.clone(),
            canary_t1,
            plasticity_warm: heldout_candidate,
            plasticity_fresh,
            plasticity_ratio: plasticity_fresh.map(|f| heldout_candidate / f.max(1e-6)),
            mean_reward: mean_f32(&log.rewards()),
            distinct_completions: servable.distinct_completions,
            decoded_tasks: servable.held_out.len() + servable.anchor.len(),
            sampled_spans: sampled.len(),
            trained_spans: trained.len(),
            wall_secs: started.elapsed().as_secs_f64(),
        };
        if cfg.verbose {
            println!("{}", record.row());
        }
        records.push(record);
        r_matrix.push(row);

        if applied_promote {
            incumbent = train_out;
        }
    }

    // ---- Arm 2's endpoint: what a last-task-only system would score -------
    let last_task_only_acc = match &last_fresh {
        Some(path) => {
            let all: Vec<Task> = probes.iter().flatten().cloned().collect();
            let (scores, _) = improve::score_checkpoint::<M>(path, &all, &verifier, &greedy_params);
            Some(mean(&scores))
        }
        None => None,
    };

    let zero_shot: Vec<f64> = records.iter().map(|r| r.heldout_incumbent).collect();
    let xfer_vs_fresh_value = if b_fresh.len() == records.len() && records.len() >= 2 { Some(xfer_vs_fresh_control(&zero_shot, &b_fresh)) } else { None };
    let rho: Vec<f64> = records.iter().filter_map(|r| r.plasticity_ratio).collect();
    let xs: Vec<f64> = (1..=rho.len()).map(|i| i as f64).collect();
    let plasticity_slope = bench::metrics::ols_slope_ci(&xs, &rho);

    Ok(StudyReport {
        acc: acc(&r_matrix),
        bwt: bwt(&r_matrix),
        xfer_vs_fresh: xfer_vs_fresh_value,
        plasticity_slope,
        promotions,
        records,
        r_matrix,
        b_fresh,
        b_base,
        last_task_only_acc,
        joint_oracle_acc: None,
        probe_ids_checked: probe_ids.len(),
        explore_ids_checked,
    })
}

fn mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// Train `objective` starting from `base`'s weights, writing the result to
/// `out` - [`crate::improve::cycle`] minus the gate and the adapter, for the
/// control arms whose whole point is that they are NOT gated.
fn train_from<M: Model, O: model::Objective<M>>(base: &Path, objective: O, opts: &FitOpts, out: &Path) -> std::io::Result<()> {
    let c = checkpoint::load(base.to_str().expect("utf-8 path"));
    let cfg = M::Config::from_json(&c.header["config"]);
    let init = c.by_role("");
    let model = M::new(cfg.clone(), 1, cfg.block_size(), &init);
    model::fit_with(model, objective, opts, Some(out))?;
    Ok(())
}

/// Arm 3, the capacity control: one fresh adapter trained on ALL
/// `cfg.cycles` rules pooled for `cycles * steps_per_cycle` steps, scored on
/// every frozen probe set.
///
/// Without it, "the loop forgot task 1" and "the adapter is simply too small
/// to hold N rules at once" are indistinguishable, and they have opposite
/// fixes (change the training regime vs. buy more rank). A high oracle ACC
/// says capacity is not the binding constraint, so a low loop BWT is a
/// forgetting result; a low oracle ACC says the opposite and the loop's BWT
/// must be read as a capacity result.
pub fn joint_oracle<M: Model, C: Curriculum>(spec: &StudySpec, curr: &C, cfg: &StudyConfig) -> std::io::Result<f64> {
    let (_, completion_len) = curr.shape();
    let block = block_of::<M>(spec.base_checkpoint);
    std::fs::create_dir_all(&cfg.work_dir)?;
    let out = cfg.work_dir.join("oracle.safetensors");
    let _ = std::fs::remove_file(&out);

    let log = CycleLog::new();
    let env = PooledEnv { curr, cycles: cfg.cycles };
    let objective = Grpo::new(env, curr.verifier(), grpo_cfg(cfg, block as usize, completion_len)).with_log(log.clone());
    let mut opts = cycle_opts(cfg, 0, block);
    opts.steps = cfg.steps_per_cycle * cfg.cycles as u32;
    opts.decay_iters = opts.steps;
    train_from::<M, _>(spec.base_checkpoint, objective, &opts, &out)?;
    improve::assert_trained_spans_were_sampled(&log.trained(), &log.sampled());

    let verifier = curr.verifier();
    let all: Vec<Task> = (0..cfg.cycles)
        .flat_map(|k| {
            let e = curr.eval_env_for(k);
            probe_seeds(k, cfg.eval_per_cycle).into_iter().map(move |s| e.tasks(s).into_iter().next().expect("Curriculum::eval_env_for produced no task"))
        })
        .collect();
    let (scores, _) = improve::score_checkpoint::<M>(&out, &all, &verifier, &greedy(completion_len));
    Ok(mean(&scores))
}

/// Step one of preparing the study's frozen base: full-parameter causal-LM
/// training on a dataset directory of PRETRAINING rules only (cues no study
/// cycle ever uses). This teaches the format and the "copy content out of the
/// prompt" skill; it must not teach any study rule, or every later "the loop
/// learned task k" number would be measuring recall of pretraining.
/// Returns `(initial_loss, final_loss)` - a caller that does not look at
/// those two numbers has no way to tell "the base learned the skill" from
/// "the base learned nothing and every later measurement is noise".
///
/// `opts.align_to_lines` must be `true` for a
/// [`crate::curriculum::write_pretrain_dataset`] directory; see that
/// function's own doc comment for what silently goes wrong otherwise.
pub fn pretrain_base<M: Model>(cfg: M::Config, dataset_dir: &Path, opts: &FitOpts, out: &Path) -> std::io::Result<(f32, f32)> {
    // `model::fit` RESUMES from `out` when it exists; a study must never
    // silently continue from a stale base of unknown provenance.
    let _ = std::fs::remove_file(out);
    model::fit::<M>(dataset_dir, cfg, opts, Some(out))
}

/// Step two: copy every parameter the pretrained checkpoint has into a
/// `study_cfg`-shaped checkpoint and fill the remaining (LoRA) factors from
/// `M::init_weights`, which initializes the adapter to a ZERO delta. The
/// result is behaviorally identical to `pretrained` on day one, which is what
/// makes the fresh-adapter control (Arm 2) exact: Arm 1 and Arm 2 start from
/// literally the same function.
pub fn overlay_adapter<M: Model>(pretrained: &Path, study_cfg: &M::Config, seed: u64, out: &Path) -> std::io::Result<()> {
    let c = checkpoint::load(pretrained.to_str().expect("utf-8 path"));
    let src = c.by_role("");
    let fresh = M::init_weights(study_cfg, seed);
    let mut carried = 0usize;
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = study_cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| {
            let v = match src.get(&name) {
                Some(v) => {
                    carried += 1;
                    v.clone()
                }
                None => fresh.get(&name).unwrap_or_else(|| panic!("continual::overlay_adapter: {name} is in neither the pretrained checkpoint nor a fresh init")).clone(),
            };
            assert_eq!(v.len(), n, "continual::overlay_adapter: {name} has {} elements, the study config wants {n}", v.len());
            (name, vec![n as u64], v)
        })
        .collect();
    assert!(carried > 0, "continual::overlay_adapter: the pretrained checkpoint shares no parameter name with the study config - wrong architecture or wrong shape");
    checkpoint::save(out.to_str().expect("utf-8 path"), study_cfg.to_json(), &tensors);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built 3x3 lower-triangular R matrix with a deliberately
    /// NEGATIVE backward transfer: task 1 was learned at 0.90 and ends at
    /// 0.60, task 2 at 0.80 ending at 0.70.
    fn forgetting_matrix() -> Vec<Vec<f64>> {
        vec![vec![0.90], vec![0.85, 0.80], vec![0.60, 0.70, 0.75]]
    }

    #[test]
    fn acc_is_the_final_rows_mean_over_every_probe() {
        let r = forgetting_matrix();
        assert!((acc(&r) - (0.60 + 0.70 + 0.75) / 3.0).abs() < 1e-12);
        // A one-cycle study's ACC is just its diagonal.
        assert!((acc(&[vec![0.42]]) - 0.42).abs() < 1e-12);
        assert_eq!(acc(&[]), 0.0);
    }

    #[test]
    fn bwt_is_negative_when_earlier_tasks_decayed() {
        let r = forgetting_matrix();
        // ((0.60 - 0.90) + (0.70 - 0.80)) / 2 = -0.20
        assert!((bwt(&r) - -0.20).abs() < 1e-12, "{}", bwt(&r));
        assert!(bwt(&r) < 0.0, "a matrix whose earlier tasks decayed must report negative BWT");
    }

    #[test]
    fn bwt_is_positive_when_later_training_helped_earlier_tasks() {
        let r = vec![vec![0.50], vec![0.55, 0.60], vec![0.70, 0.75, 0.65]];
        // ((0.70 - 0.50) + (0.75 - 0.60)) / 2 = 0.175
        assert!((bwt(&r) - 0.175).abs() < 1e-12, "{}", bwt(&r));
        // A single-cycle study has nothing to have forgotten.
        assert_eq!(bwt(&[vec![0.9]]), 0.0);
    }

    #[test]
    fn xfer_vs_fresh_control_compares_zero_shot_against_the_fresh_adapter_control() {
        // Cycle 0 is excluded by construction (nothing precedes it).
        let zero_shot = [0.05, 0.30, 0.40];
        let b_fresh = [0.60, 0.20, 0.25];
        // ((0.30 - 0.20) + (0.40 - 0.25)) / 2 = 0.125
        assert!((xfer_vs_fresh_control(&zero_shot, &b_fresh) - 0.125).abs() < 1e-12);
        assert_eq!(xfer_vs_fresh_control(&[0.1], &[0.2]), 0.0);
    }

    #[test]
    #[should_panic(expected = "must be parallel")]
    fn xfer_vs_fresh_control_rejects_mismatched_series() {
        let _ = xfer_vs_fresh_control(&[0.1, 0.2], &[0.3]);
    }

    #[test]
    fn coin_flip_is_deterministic_for_a_fixed_seed_and_not_constant() {
        let p = GatePolicy::CoinFlip { seed: 11 };
        let first: Vec<bool> = (0..12).map(|k| p.applies(Decision::Promote, k)).collect();
        let again: Vec<bool> = (0..12).map(|k| p.applies(Decision::Promote, k)).collect();
        assert_eq!(first, again, "an Arm-4 run must be reproducible");
        // And it must genuinely ignore the gate's decision, or it is not a
        // null gate at all.
        let rejected: Vec<bool> = (0..12)
            .map(|k| p.applies(Decision::Reject(crate::gate::Cause::NotSignificant { p_value: 1.0, alpha: 0.05 }), k))
            .collect();
        assert_eq!(first, rejected, "a coin-flip policy must not consult the real decision");
        assert!(first.iter().any(|&b| b) && first.iter().any(|&b| !b), "a coin that always lands the same way is not a coin: {first:?}");
        // A different seed gives a different sequence.
        let other: Vec<bool> = (0..12).map(|k| GatePolicy::CoinFlip { seed: 12 }.applies(Decision::Promote, k)).collect();
        assert_ne!(first, other);
    }

    #[test]
    fn real_policy_carries_forward_exactly_the_gates_decision() {
        let p = GatePolicy::Real;
        assert!(p.applies(Decision::Promote, 3));
        assert!(!p.applies(Decision::Reject(crate::gate::Cause::EffectTooSmall { effect_size: 0.0, min_effect_size: 0.05 }), 3));
    }

    #[test]
    fn position_copy_curriculum_reports_one_shape_and_distinct_labels() {
        let c = PositionCopy::default();
        assert_eq!(c.shape(), (PROMPT_LEN, OUT_LEN));
        let labels: HashSet<String> = (0..12).map(|k| c.label(k)).collect();
        assert_eq!(labels.len(), 12, "each cycle must be identifiable in the printed trajectory");
        // The invariant the plasticity ratio rests on: every cycle's task is
        // the same shape, so a later cycle scoring lower cannot be explained
        // by a harder task.
        for k in 0..12 {
            let t = c.env_for(k).tasks(k as u64).into_iter().next().unwrap();
            assert_eq!(t.prompt.len(), PROMPT_LEN);
            assert_eq!(crate::curriculum::target_of(&t).len(), OUT_LEN);
        }
    }

    #[test]
    fn probe_seeds_are_frozen_per_cycle_and_never_overlap() {
        let all: HashSet<u64> = (0..12).flat_map(|k| probe_seeds(k, 16)).collect();
        assert_eq!(all.len(), 12 * 16, "probe seed ranges must not collide across cycles");
        assert_eq!(probe_seeds(3, 16), probe_seeds(3, 16), "probe seeds must be a pure function of (cycle, index)");
    }

    #[test]
    fn pooled_env_covers_every_rule_in_the_study() {
        let curr = PositionCopy::default();
        let env = PooledEnv { curr: &curr, cycles: 12 };
        let cues: HashSet<u32> = (0..240u64).map(|s| env.tasks(s).into_iter().next().unwrap().prompt[0]).collect();
        assert_eq!(cues.len(), 12, "the joint-training oracle must actually see all 12 rules, got {cues:?}");
    }
}
