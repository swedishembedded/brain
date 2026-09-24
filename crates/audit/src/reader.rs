// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The loop: one episode in, one ledger row out.
//!
//! Everything the other modules in this crate decide is wired together here,
//! in the order that makes the cheap decisions first. Screen, then one
//! forward pass, then the evidence precondition, and only then a training run
//! and two decode passes. What survives all four is promoted, joins the
//! adapter pool, enters the rehearsal reservoir and joins the audit
//! rotation; what does not is recorded with the stage that stopped it.
//!
//! ## Why this is generic over a seam rather than over a model
//!
//! Everything here that touches a model does so through [`Learner`]: a
//! forward pass for the reach test, a training run for the candidate, a
//! decode for each arm, and the joint oracle. That is four methods, and
//! putting them behind a trait keeps the ORCHESTRATION - which is where the
//! mistakes live - in a crate with no device, no weights and a test suite
//! that runs in milliseconds against a fake.
//!
//! It is the seam this workspace already uses elsewhere for the same reason:
//! a scheduler generic over a decoder, a study generic over an environment.
//! The binding to a real model is a thin implementation, written once,
//! above this crate.
//!
//! ## One precondition the diagnosis has that is easy to miss
//!
//! The oracle bounds what a schedule could have achieved over what the
//! reader has LEARNED, so a run that has promoted nothing has nothing for it
//! to bound. Such a run looks exactly like a stalled one by promote rate,
//! and asking would spend a full training run to be told nothing. So the
//! oracle is asked only when the bank is non-empty as well, and a run that
//! never got started is a different problem from one that stopped.
//!
//! ## What one step costs, and why it does not grow
//!
//! The expensive part of a continual learner is not training, it is
//! re-checking what it already knows. Left alone that is quadratic in the
//! length of the run. Here it is bounded by the audit budget, so a step
//! costs the same at episode ten thousand as at episode ten, and what
//! degrades instead is the DETECTION LATENCY, a number the reader reports.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::bank::{Probe, ProbeConfig, ProbeSet, SpanSelector, UniformSelector};
use crate::growth::{Action, Diagnosis, Growth, GrowthConfig};
use crate::pool::{AdapterId, Pool, PoolConfig};
use crate::reservoir::{Reservoir, ReservoirConfig};
use crate::retention::Retention;
use crate::schedule::{block_drop_bar, AuditConfig, Schedule};
use crate::run::ReaderState;
use crate::stream::{Cursor, Episode, EpisodeId};
use crate::triage::{adjudicate, reach, reader_gate_config, screen, TriageConfig, Verdict};

use promote::gate::GateInput;

/// Which side of a comparison a decode is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// What is currently served.
    Incumbent,
    /// What this episode trained.
    Candidate,
}

/// What one decode pass over a set of probes yields.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    /// One score per probe, in the order they were given.
    pub scores: Vec<f64>,
    /// Mean completion entropy over that pass, for the degeneracy bar.
    pub mean_entropy: f64,
    /// What the model actually answered, in the same order.
    ///
    /// The gate needs only the scores, and the loop ignores these. They are
    /// here because a caller scoring a CAPABILITY battery has to be able to
    /// show what the model wrote: a battery that can only say how many
    /// passed cannot say how the failures were wrong, and "it answered with
    /// a flag that does not exist" and "it answered correctly in a different
    /// order" are not the same result.
    pub answers: Vec<String>,
}

/// Everything the loop needs a model for, and nothing else.
pub trait Learner {
    /// Mean loss of `text` under the CURRENT model. One forward pass, and
    /// the only model cost an episode incurs before the gate.
    fn loss(&mut self, text: &str) -> f64;

    /// Train a candidate adapter on `rows`, which already include the
    /// rehearsal mix. The bytes are what the pool stores.
    fn train(&mut self, rows: &[&str], seed: u64) -> Vec<u8>;

    /// Decode `probes` under `arm` once, returning both what the gate needs
    /// from that pass.
    ///
    /// One method rather than two on purpose. A decode yields a score and a
    /// completion entropy together, and asking for them separately makes a
    /// correct implementation decode everything twice - the degeneracy bar
    /// would cost as much as the whole gate. Both arms must be decoded the
    /// same way, from what would actually be served.
    fn score(&mut self, arm: Arm, probes: &[&Probe]) -> Scored;

    /// Train ONE adapter jointly over `rows` - everything the reader has
    /// learned - and score it on `probes`. The upper bound any schedule
    /// could have reached, and the most expensive thing the reader ever
    /// does.
    ///
    /// It takes the rows as well as the probes because it must TRAIN, not
    /// merely evaluate. An oracle handed only the probes could not do the
    /// one thing that makes it an upper bound.
    fn joint_oracle(&mut self, rows: &[&str], probes: &[&Probe]) -> f64;
}

/// What happened to one episode.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Refused before probes could be frozen: an answer would have been
    /// reachable from the half that gets trained on.
    /// Every candidate probe was reproduced in a row that would be trained
    /// on, so nothing in this episode could be verified independently of
    /// its own training half. A SINGLE repeated line is not this: it is
    /// dropped from the probes and the episode carries on.
    Leaked { probes: usize },
    /// Triage or the gate decided.
    Decided(Verdict),
}

impl Outcome {
    pub fn stage(&self) -> &'static str {
        match self {
            Outcome::Leaked { .. } => "ingest",
            Outcome::Decided(v) => v.stage(),
        }
    }

    pub fn promoted(&self) -> bool {
        matches!(self, Outcome::Decided(Verdict::Promoted(_)))
    }
}

/// One row of the run ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub episode: EpisodeId,
    pub source: PathBuf,
    pub outcome: Outcome,
    /// Earlier episodes re-checked this step.
    pub audited: usize,
    /// Probe decodes spent on the audit, for one arm.
    pub audit_decodes: usize,
    /// What actually carried forward, which is the real gate's verdict
    /// under every arm but [`ControlArm::NullGate`], where a coin decides.
    /// Recorded beside `outcome` rather than instead of it: the two arms are
    /// only comparable if the real gate's decision is kept in both.
    pub carried: bool,
    /// What the oracle said, when it was asked.
    pub diagnosis: Option<Diagnosis>,
    /// What was done about it.
    pub action: Option<Action>,
}

/// Which CONTROL arm a run is. One at a time: each answers a different
/// question about whether the real arm's number means anything, and a run
/// that changed two things at once answers neither.
///
/// Both are deliberately the same stream, the same seeds and the same
/// budget as the real arm. A control that differs in anything else is
/// measuring that instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlArm {
    /// The reader as it ships.
    #[default]
    Real,
    /// A coin decides what carries forward. The real gate still runs and its
    /// verdict is still what the ledger records, so the two arms' promote
    /// rates are directly comparable. If they are not separated beyond seed
    /// noise, the gate is decorative.
    NullGate { seed: u64 },
    /// Each episode is gated against ANOTHER episode's frozen probes, while
    /// still being trained on its own rows. The gate should then almost
    /// never promote: training on one document does not make you better at
    /// answering questions about a different one. A promote rate here near
    /// the real arm's means the gate is responding to something other than
    /// what the episode taught - the training run happening at all, say.
    ShuffledLabels { seed: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReaderConfig {
    pub triage: TriageConfig,
    pub probes: ProbeConfig,
    pub audit: AuditConfig,
    pub reservoir: ReservoirConfig,
    pub growth: GrowthConfig,
    pub pool: PoolConfig,
    /// Rehearsal rows drawn per row of the episode itself. The measured
    /// regime mixes half and half, which is 1.0.
    pub rehearsal_ratio: f64,
    pub seed: u64,
    /// Which control arm this run is. [`ControlArm::Real`] by default, so a
    /// caller that says nothing gets the reader rather than a control.
    #[serde(default)]
    pub arm: ControlArm,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        ReaderConfig {
            triage: TriageConfig::default(),
            probes: ProbeConfig::default(),
            audit: AuditConfig::default(),
            reservoir: ReservoirConfig::default(),
            growth: GrowthConfig::default(),
            pool: PoolConfig::default(),
            rehearsal_ratio: 1.0,
            seed: 0,
            arm: ControlArm::Real,
        }
    }
}

/// The reader.
pub struct Reader<L: Learner> {
    cfg: ReaderConfig,
    learner: L,
    pool: Pool,
    bank: BTreeMap<EpisodeId, ProbeSet>,
    schedule: Schedule,
    reservoir: Reservoir,
    growth: Growth,
    /// What earlier episodes still score. Fed from the audit blocks this
    /// step already decodes, so the retention matrix costs no extra decode.
    retention: Retention,
    selector: Box<dyn SpanSelector>,
    episode: u64,
}

impl<L: Learner> Reader<L> {
    /// Open a reader over a pool rooted at `pool`.
    pub fn new(cfg: ReaderConfig, learner: L, pool: Pool) -> Reader<L> {
        Reader {
            learner,
            pool,
            bank: BTreeMap::new(),
            retention: Retention::default(),
            schedule: Schedule::new(cfg.audit),
            reservoir: Reservoir::new(cfg.reservoir),
            growth: Growth::new(cfg.growth),
            selector: Box::new(UniformSelector),
            cfg,
            episode: 0,
        }
    }

    /// Use a different rule for which lines are worth probing. Whatever it
    /// prefers, the blind draw still comes from what it rejected.
    pub fn with_selector(mut self, selector: Box<dyn SpanSelector>) -> Self {
        self.selector = selector;
        self
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub fn reservoir(&self) -> &Reservoir {
        &self.reservoir
    }

    /// Episodes in the audit bank.
    /// What earlier episodes still score, and the backward transfer over
    /// it. Sparse by construction - see [`crate::retention`].
    pub fn retention(&self) -> &Retention {
        &self.retention
    }

    pub fn bank_size(&self) -> usize {
        self.schedule.len()
    }

    /// Ticks to re-check everything the reader has learned. The number it
    /// reports instead of claiming nothing was forgotten.
    pub fn detection_latency(&self) -> u64 {
        self.schedule.detection_latency()
    }

    pub fn learner(&self) -> &L {
        &self.learner
    }

    /// The learner, mutably.
    ///
    /// A promotion is a decision the LOOP makes and the learner has to act
    /// on: what the gate promoted is what must be served from the next
    /// episode onward. The loop does not do it itself because what "serving"
    /// means is the binding's business, not the policy's.
    pub fn learner_mut(&mut self) -> &mut L {
        &mut self.learner
    }

    /// Episodes read so far, including those refused.
    pub fn episode(&self) -> u64 {
        self.episode
    }

    /// Everything that must survive the process, ready to be written.
    ///
    /// `cursor` is where the STREAM got to, which the reader does not own:
    /// it reads episodes it is handed, so the caller holding the stream is
    /// the one that knows. Passing it through here keeps a run's position
    /// and a run's knowledge in one file rather than two that can disagree.
    pub fn state(&self, cursor: Option<Cursor>) -> ReaderState {
        ReaderState {
            schema: crate::run::SCHEMA,
            bank: self.bank.clone(),
            schedule: self.schedule.clone(),
            reservoir: self.reservoir.clone(),
            growth: self.growth.clone(),
            retention: self.retention.clone(),
            episode: self.episode,
            cursor,
        }
    }

    /// Write everything that must survive the process: the pool's index and
    /// the reader's own state, into `run`.
    ///
    /// One call rather than two deliberately. The reader mutates the pool
    /// every step and holds its bank in memory, and those are two halves of
    /// one thing: a state saved without its pool reopens pointing at
    /// adapters whose index was never written. Making it impossible to do
    /// half of it is worth more than the flexibility of separate calls.
    pub fn checkpoint(&self, run: &crate::run::Run, cursor: Option<Cursor>) -> Result<(), crate::run::RunError> {
        self.pool.save().map_err(|e| crate::run::RunError::Io(self.pool.root().to_path_buf(), e.to_string()))?;
        run.save_state(&self.state(cursor))
    }

    /// Carry on from a saved state, as if the process had not stopped.
    pub fn restore(cfg: ReaderConfig, learner: L, pool: Pool, state: ReaderState) -> Reader<L> {
        Reader {
            learner,
            pool,
            bank: state.bank,
            schedule: state.schedule,
            reservoir: state.reservoir,
            growth: state.growth,
            retention: state.retention,
            selector: Box::new(UniformSelector),
            cfg,
            episode: state.episode,
        }
    }

    /// Under [`ControlArm::ShuffledLabels`], another banked episode's frozen
    /// probes: the ones this episode will be gated against instead of its
    /// own.
    ///
    /// `None` for every other arm, and for the first episode of a shuffled
    /// run - there is nothing to borrow from yet, and inventing an empty
    /// probe set would refuse it for a reason the arm is not about.
    fn shuffled_probes(&self, id: &EpisodeId) -> Option<Vec<Probe>> {
        let ControlArm::ShuffledLabels { seed } = self.cfg.arm else { return None };
        let others: Vec<&EpisodeId> = self.bank.keys().filter(|k| *k != id).collect();
        if others.is_empty() {
            return None;
        }
        let pick = (promote::gate::coin(seed, self.episode as usize) as usize
            + (self.episode as usize).wrapping_mul(2))
            % others.len();
        self.bank.get(others[pick]).map(|s| s.probes().to_vec())
    }

    /// Read one episode.
    pub fn step(&mut self, ep: &Episode) -> Row {
        let row = |outcome: Outcome| Row {
            episode: ep.id.clone(),
            source: ep.source.clone(),
            // Nothing that stopped before the gate carried anything, under
            // any arm: the null coin replaces the gate, not the screen.
            carried: false,
            outcome,
            audited: 0,
            audit_decodes: 0,
            diagnosis: None,
            action: None,
        };
        self.episode += 1;
        self.pool.tick();

        // A. Nothing here touches the model.
        if let Some(why) = screen(&ep.text, &self.cfg.triage) {
            return row(Outcome::Decided(Verdict::Unstructured(why)));
        }

        // B. One forward pass, and the only model cost an episode incurs
        // before the gate.
        if let Some(v) = reach(self.learner.loss(&ep.text), &self.cfg.triage) {
            return row(Outcome::Decided(v));
        }

        // Freezing the probes is what makes the episode checkable, and it
        // can refuse: an answer reachable from the half that gets trained on
        // would measure memorisation rather than retention. Refused BEFORE
        // the training run, since the point is not to train on it at all.
        let probes = match ProbeSet::build(ep, &self.cfg.probes, self.selector.as_ref()) {
            Ok(p) => p,
            Err(crate::bank::BankError::EveryProbeInTrainedRows(_)) => return row(Outcome::Leaked { probes: 0 }),
            Err(_) => return row(Outcome::Decided(Verdict::TooSmallToGate { probes: 0, floor: self.cfg.triage.min_probes })),
        };

        // C. An episode that could not have produced a significant result is
        // held back rather than promoted on evidence it never had. Checked
        // before training, because training it would be spending a run on a
        // decision that cannot be made.
        if probes.probes().len() < self.cfg.triage.min_probes {
            return row(Outcome::Decided(Verdict::TooSmallToGate { probes: probes.probes().len(), floor: self.cfg.triage.min_probes }));
        }

        // The audit plan is drawn BEFORE training, so the same earlier
        // episodes are scored under both arms.
        let plan = self.schedule.plan();
        // Owned rather than borrowed: the bank is written to further down
        // when an episode promotes, and a block holding a reference into it
        // would make that write impossible. The clone is bounded by the
        // audit budget, which is the whole point of having one.
        let audit_blocks: Vec<(EpisodeId, Vec<Probe>)> = plan
            .blocks()
            .into_iter()
            .filter_map(|id| {
                let set = self.bank.get(id)?;
                Some((id.clone(), set.probes().iter().take(plan.probes_per_block).cloned().collect()))
            })
            .collect();

        // The shuffled-label arm swaps the probes the GATE sees for another
        // episode's, while the training rows below stay this episode's own.
        // Nothing else about the step changes, so the two arms' promote
        // rates are a comparison of one thing.
        let borrowed = self.shuffled_probes(&ep.id);
        let own: Vec<&Probe> = match &borrowed {
            Some(p) => p.iter().collect(),
            None => probes.probes().iter().collect(),
        };
        let mut all: Vec<&Probe> = own.clone();
        for (_, block) in &audit_blocks {
            all.extend(block.iter());
        }

        // D. Train, then score both arms on exactly the same probes in the
        // same order.
        let mut rows: Vec<&str> = probes.trained_rows().iter().map(String::as_str).collect();
        let rehearsal = (rows.len() as f64 * self.cfg.rehearsal_ratio).round() as usize;
        rows.extend(self.reservoir.draw(rehearsal, self.cfg.seed ^ self.episode));
        let adapter = self.learner.train(&rows, self.cfg.seed ^ self.episode);

        let incumbent = self.learner.score(Arm::Incumbent, &all);
        let candidate = self.learner.score(Arm::Candidate, &all);
        let (incumbent_entropy, candidate_entropy) = (incumbent.mean_entropy, candidate.mean_entropy);
        let (incumbent, candidate) = (incumbent.scores, candidate.scores);
        let n = own.len();
        let mut blocks: Vec<(f64, f64)> = Vec::with_capacity(audit_blocks.len());
        let mut at = n;
        for (id, block) in &audit_blocks {
            let end = at + block.len();
            let (c, i) = (mean(&candidate[at..end]), mean(&incumbent[at..end]));
            blocks.push((c, i));
            // `R[k][j]`, taken from the INCUMBENT arm: what this earlier
            // episode scores against what is currently served. The candidate
            // arm may not be promoted a few lines below, and a retention
            // matrix built from scores that were then thrown away would
            // describe a model nobody ever ran.
            self.retention.observed(self.episode, id, i);
            at = end;
        }

        let mut gate_cfg = reader_gate_config();
        // A sampled block has a larger standard error than the complete one
        // the pre-registered bar was fixed for, so the bar is re-derived from
        // what was actually sampled. It can only loosen.
        gate_cfg.max_block_drop = block_drop_bar(plan.probes_per_block);

        let verdict = adjudicate(
            n,
            &GateInput {
                candidate_scores: &candidate[..n],
                incumbent_scores: &incumbent[..n],
                anchor_candidate: mean_of_pairs(&blocks, |(c, _)| *c),
                anchor_incumbent: mean_of_pairs(&blocks, |(_, i)| *i),
                entropy_candidate: candidate_entropy,
                entropy_incumbent: incumbent_entropy,
                anchor_blocks: &blocks,
            },
            &self.cfg.triage,
            &gate_cfg,
        );

        // What the gate SAID is what the ledger records, under every arm.
        // What carries forward is what the arm's policy allows, so the null
        // arm's promote rate is comparable with the real one's rather than
        // being a different measurement.
        let decided = matches!(verdict, Verdict::Promoted(_));
        let promoted = match self.cfg.arm {
            ControlArm::NullGate { seed } => {
                promote::gate::coin(seed, self.episode as usize)
            }
            _ => decided,
        };
        if promoted {
            let id = AdapterId(ep.id.as_str().to_string());
            // A duplicate id can only mean the same CONTENT was promoted
            // twice, which triage should have caught as already known. Not
            // fatal, and not silently overwritten either.
            if self.pool.admit(&id, &adapter).is_ok() {
                let demand: BTreeMap<AdapterId, f64> = self.pool.live().iter().map(|e| (e.id.clone(), 1.0)).collect();
                self.pool.select(&demand);
            }
            self.reservoir.offer(&ep.id, probes.trained_rows(), &verdict);
            self.schedule.admit(ep.id.clone());
            self.bank.insert(ep.id.clone(), probes);
            // `R[j][j]`: what it scored on its own probes under the arm that
            // was actually promoted, at the moment it was promoted.
            self.retention.learned(&ep.id, mean(&candidate[..n]));
        }

        // The growth question is asked of the promote rate, and only paid
        // for when that rate has fallen.
        self.growth.record(promoted);
        let (diagnosis, action) = if self.growth.oracle_due() && !self.bank.is_empty() {
            let bank_probes: Vec<&Probe> = self.bank.values().flat_map(|s| s.probes().iter()).collect();
            let bank_rows: Vec<&str> = self.bank.values().flat_map(|s| s.trained_rows().iter().map(String::as_str)).collect();
            let joint = self.learner.joint_oracle(&bank_rows, &bank_probes);
            let sequential = mean(&self.learner.score(Arm::Incumbent, &bank_probes).scores);
            let d = self.growth.diagnose(joint, sequential);
            let a = self.growth.act(d, self.cfg.reservoir.cap);
            (Some(d), Some(a))
        } else {
            (None, None)
        };

        Row {
            episode: ep.id.clone(),
            source: ep.source.clone(),
            carried: promoted,
            outcome: Outcome::Decided(verdict),
            audited: audit_blocks.len(),
            audit_decodes: audit_blocks.iter().map(|(_, b)| b.len()).sum(),
            diagnosis,
            action,
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

fn mean_of_pairs(blocks: &[(f64, f64)], pick: impl Fn(&(f64, f64)) -> f64) -> f64 {
    if blocks.is_empty() {
        0.0
    } else {
        blocks.iter().map(pick).sum::<f64>() / blocks.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static N: AtomicUsize = AtomicUsize::new(0);

    struct Dir(PathBuf);

    impl Dir {
        fn new() -> Dir {
            let d = std::env::temp_dir().join(format!("brain-reader-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&d);
            Dir(d)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A learner that touches no model and counts what it was asked to do,
    /// so the loop's control flow can be asserted directly. `wins` is how
    /// many probes the candidate is made to flip.
    #[derive(Default)]
    struct Fake {
        loss: f64,
        wins: usize,
        /// Training runs after which the candidate stops winning anything,
        /// so a run can promote for a while and then stall the way a real
        /// one does. `0` means it never stalls.
        stall_after: usize,
        oracle: f64,
        forwards: usize,
        trainings: usize,
        oracles: usize,
        last_train_rows: usize,
        /// The ids of the probes the gate was last scored against. The
        /// shuffled-label arm is a claim about exactly this.
        last_scored: Vec<String>,
    }

    impl Learner for Fake {
        fn loss(&mut self, _text: &str) -> f64 {
            self.forwards += 1;
            self.loss
        }
        fn train(&mut self, rows: &[&str], _seed: u64) -> Vec<u8> {
            self.trainings += 1;
            self.last_train_rows = rows.len();
            b"adapter".to_vec()
        }
        fn score(&mut self, arm: Arm, probes: &[&Probe]) -> Scored {
            // The incumbent fails everything; the candidate flips `wins` of
            // the episode's own probes and matches on the rest, which is the
            // shape of an episode that taught something without cost.
            let scores = probes
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let stalled = self.stall_after > 0 && self.trainings > self.stall_after;
                    match arm {
                        Arm::Candidate if i < self.wins && !stalled => 1.0,
                        Arm::Candidate => 0.0,
                        Arm::Incumbent => 0.0,
                    }
                })
                .collect();
            let answers = probes.iter().map(|p| p.expected.clone()).collect();
            self.last_scored = probes.iter().map(|p| format!("{:?}", p.id)).collect();
            Scored { scores, mean_entropy: 2.0, answers }
        }
        fn joint_oracle(&mut self, _rows: &[&str], _probes: &[&Probe]) -> f64 {
            self.oracles += 1;
            self.oracle
        }
    }

    fn manual(tag: &str, n: usize) -> String {
        (0..n).map(|i| format!("--{tag}{i:03} VALUE   set the {tag} {i:03} option to VALUE\n")).collect()
    }

    fn episode(tag: &str, n: usize) -> Episode {
        let text = manual(tag, n);
        Episode { id: EpisodeId::of(&text), source: PathBuf::from(format!("{tag}.txt")), ordinal: 0, text }
    }

    fn cfg() -> ReaderConfig {
        ReaderConfig {
            probes: ProbeConfig { probe_permille: 400, blind_permille: 100, ..ProbeConfig::default() },
            audit: AuditConfig { budget: 64, probes_per_block: 8, canary_blocks: 1, canary_refresh: 8 },
            growth: GrowthConfig { window: 4, promote_rate_floor: 0.25, cooldown: 4, ..GrowthConfig::default() },
            pool: PoolConfig { resident: 4, margin: 0.1, dwell: 2, survival: 1000 },
            ..ReaderConfig::default()
        }
    }

    fn reader(dir: &Dir, fake: Fake) -> Reader<Fake> {
        let pool = Pool::create(&dir.0, cfg().pool).expect("pool");
        Reader::new(cfg(), fake, pool)
    }

    /// The whole point of putting the screen first: garbage must not cost a
    /// forward pass, let alone a training run.
    #[test]
    fn an_unstructured_episode_never_reaches_the_model() {
        let d = Dir::new();
        let mut r = reader(&d, Fake { loss: 1.0, wins: 20, ..Default::default() });
        let ep = Episode { id: EpisodeId::of("x"), source: PathBuf::from("t"), ordinal: 0, text: "tiny".to_string() };
        let row = r.step(&ep);
        assert_eq!(row.outcome.stage(), "screen");
        assert_eq!(r.learner().forwards, 0, "the screen must decide before any forward pass");
        assert_eq!(r.learner().trainings, 0);
    }

    /// One forward pass, no training run, nothing entering any structure.
    #[test]
    fn an_already_known_episode_costs_one_forward_pass_and_no_training() {
        let d = Dir::new();
        let mut r = reader(&d, Fake { loss: 0.01, wins: 20, ..Default::default() });
        let row = r.step(&episode("flag", 40));
        assert_eq!(row.outcome.stage(), "reach");
        assert_eq!(r.learner().forwards, 1);
        assert_eq!(r.learner().trainings, 0, "there is nothing to learn, so nothing may be trained");
        assert_eq!(r.bank_size(), 0);
        assert!(r.reservoir().is_empty());
    }

    /// A promotion has to land in all three structures, or later episodes
    /// cannot rehearse it, re-check it, or route through it.
    #[test]
    fn a_promoted_episode_enters_the_pool_the_reservoir_and_the_audit_rotation() {
        let d = Dir::new();
        let mut r = reader(&d, Fake { loss: 1.5, wins: 64, ..Default::default() });
        let row = r.step(&episode("flag", 60));
        assert!(row.outcome.promoted(), "expected a promote, got {:?}", row.outcome);
        assert_eq!(r.pool().live().len(), 1);
        assert_eq!(r.reservoir().len(), 1);
        assert_eq!(r.bank_size(), 1);
    }

    /// And a rejection must land in none of them, or a refused episode
    /// influences every later cycle through the back door.
    #[test]
    fn a_rejected_episode_enters_none_of_them() {
        let d = Dir::new();
        let mut r = reader(&d, Fake { loss: 1.5, wins: 0, ..Default::default() });
        let row = r.step(&episode("flag", 60));
        assert!(!row.outcome.promoted());
        assert_eq!(row.outcome.stage(), "gate");
        assert_eq!(r.pool().live().len(), 0);
        assert!(r.reservoir().is_empty());
        assert_eq!(r.bank_size(), 0);
    }

    /// The headline property. A continual learner that re-checks everything
    /// is quadratic; this one is not, and the cost of a step at episode 60
    /// must look like the cost at episode 2.
    #[test]
    fn the_audit_cost_of_a_step_does_not_grow_with_the_run() {
        let d = Dir::new();
        let mut r = reader(&d, Fake { loss: 1.5, wins: 64, ..Default::default() });
        let mut rows = Vec::new();
        for i in 0..60 {
            rows.push(r.step(&episode(&format!("t{i}"), 60)));
        }
        let budget = cfg().audit.budget;
        for row in &rows {
            assert!(row.audit_decodes <= budget, "a step spent {} of a {budget} budget", row.audit_decodes);
        }
        assert!(r.bank_size() > 40, "the bank must actually have grown, or this proves nothing: {}", r.bank_size());
        assert!(r.detection_latency() > 1, "and the latency must have grown with it, got {}", r.detection_latency());
    }

    /// V1, end to end. A restated line means one copy would be trained and
    /// the other probed, so THAT LINE cannot be a probe. The episode is only
    /// refused when nothing is left that could be verified independently of
    /// its own training half.
    #[test]
    fn a_repeated_line_costs_its_probe_and_only_a_wholly_repeated_document_costs_the_episode() {
        struct Pin;
        impl SpanSelector for Pin {
            fn score(&self, line: &str) -> f64 {
                if line.starts_with("--flag007") {
                    1.0
                } else if line.starts_with("alias") {
                    -1.0
                } else {
                    0.0
                }
            }
        }
        let d = Dir::new();
        let mut c = cfg();
        c.probes.blind_permille = 0;
        let pool = Pool::create(&d.0, c.pool).expect("pool");
        let mut r = Reader::new(c, Fake { loss: 1.5, wins: 64, ..Default::default() }, pool).with_selector(Box::new(Pin));

        // One restated line, in a document that is otherwise full of
        // distinct ones. The episode goes on; the restated line does not
        // become a probe.
        let mut text = manual("flag", 40);
        text.push_str("alias for the above: --flag007 VALUE   set the flag 007 option to VALUE\n");
        let ep = Episode { id: EpisodeId::of(&text), source: PathBuf::from("m.txt"), ordinal: 0, text };
        let row = r.step(&ep);
        assert!(!matches!(row.outcome, Outcome::Leaked { .. }), "one repeated line must not cost the whole episode: {:?}", row.outcome);
        assert_eq!(r.learner().trainings, 1, "the episode reached training");

        // Nothing in this run is a document whose every line repeats, so
        // the half that DOES refuse gets its own episode.
        let all_same = "the same line over and over again\n".repeat(40);
        let ep = Episode { id: EpisodeId::of(&all_same), source: PathBuf::from("same.txt"), ordinal: 0, text: all_same };
        let row = r.step(&ep);
        assert!(
            matches!(row.outcome, Outcome::Leaked { .. }),
            "a document with nothing that could be probed independently is refused at ingest, got {:?}",
            row.outcome
        );
        assert_eq!(row.outcome.stage(), "ingest");
        assert_eq!(r.learner().trainings, 1, "and it is refused BEFORE it is trained on");
    }

    /// The oracle costs a full training run over the history, so a healthy
    /// reader must never pay for it, and a stalled one must.
    #[test]
    fn the_oracle_is_asked_only_when_the_promote_rate_falls() {
        let d = Dir::new();
        let mut healthy = reader(&d, Fake { loss: 1.5, wins: 64, oracle: 0.9, ..Default::default() });
        for i in 0..8 {
            healthy.step(&episode(&format!("h{i}"), 60));
        }
        assert_eq!(healthy.learner().oracles, 0, "a promoting reader has nothing to diagnose");

        // A real stall is a run that promoted for a while and then stopped.
        // A run that never promoted anything cannot be diagnosed at all: the
        // oracle is an upper bound over what was learned, and nothing was.
        let d2 = Dir::new();
        let mut stalled = reader(&d2, Fake { loss: 1.5, wins: 64, stall_after: 6, oracle: 0.9, ..Default::default() });
        let mut diagnoses = Vec::new();
        for i in 0..14 {
            if let Some(dg) = stalled.step(&episode(&format!("s{i}"), 60)).diagnosis {
                diagnoses.push(dg);
            }
        }
        assert!(stalled.learner().oracles > 0, "a reader that stopped promoting must ask why");
        assert!(!diagnoses.is_empty(), "and must record what it was told");

        let d3 = Dir::new();
        let mut barren = reader(&d3, Fake { loss: 1.5, wins: 0, oracle: 0.9, ..Default::default() });
        for i in 0..12 {
            barren.step(&episode(&format!("b{i}"), 60));
        }
        assert_eq!(
            barren.learner().oracles,
            0,
            "a run that has promoted NOTHING has nothing for the oracle to bound, so asking would cost a full training run for a meaningless answer"
        );
    }

    /// The whole premise is a reader left running for days and stopped when
    /// convenient. Stopping and resuming has to be invisible in the ledger.
    #[test]
    fn a_resumed_reader_carries_on_exactly_where_it_stopped() {
        let unbroken = {
            let d = Dir::new();
            let mut r = reader(&d, Fake { loss: 1.5, wins: 30, oracle: 0.9, ..Default::default() });
            (0..10).map(|i| r.step(&episode(&format!("e{i}"), 50))).collect::<Vec<_>>()
        };

        let d = Dir::new();
        let run = crate::run::Run::create(
            &d.0,
            &crate::run::Manifest { schema: crate::run::SCHEMA, model: "fake".to_string(), cfg: cfg() },
        )
        .expect("run");
        let pool = Pool::create(&run.pool_root(), cfg().pool).expect("pool");
        let mut r = Reader::new(cfg(), Fake { loss: 1.5, wins: 30, oracle: 0.9, ..Default::default() }, pool);
        let mut rows: Vec<Row> = (0..4).map(|i| r.step(&episode(&format!("e{i}"), 50))).collect();
        let bank_before = r.bank_size();
        r.checkpoint(&run, None).expect("checkpoint");
        drop(r);

        // A fresh process: nothing carried over but the directory.
        let run = crate::run::Run::open(&d.0).expect("the run must reopen");
        let saved = run.load_state().expect("load").expect("a checkpoint was written");
        let pool = Pool::open(&run.pool_root()).expect("the pool must reopen");
        let mut resumed = Reader::restore(cfg(), Fake { loss: 1.5, wins: 30, oracle: 0.9, ..Default::default() }, pool, saved);
        assert_eq!(resumed.episode(), 4);
        assert_eq!(resumed.bank_size(), bank_before);
        rows.extend((4..10).map(|i| resumed.step(&episode(&format!("e{i}"), 50))));

        let shape = |v: &[Row]| v.iter().map(|w| (w.episode.clone(), w.outcome.stage(), w.audited)).collect::<Vec<_>>();
        assert_eq!(shape(&rows), shape(&unbroken), "a stop and a resume must leave the same ledger as an unbroken run");
    }

    /// The reader is meant to be left running for days; the same seed over
    /// the same corpus has to produce the same ledger.
    #[test]
    fn the_same_seed_over_the_same_corpus_produces_the_same_ledger() {
        let run = || {
            let d = Dir::new();
            let mut r = reader(&d, Fake { loss: 1.5, wins: 30, oracle: 0.9, ..Default::default() });
            let rows: Vec<Row> = (0..12).map(|i| r.step(&episode(&format!("e{i}"), 50))).collect();
            rows.into_iter().map(|w| (w.episode, w.outcome.stage(), w.audited, w.audit_decodes)).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
    /// The null arm's whole job: the real gate still runs and is still
    /// recorded, and a coin decides what carries forward. Without both
    /// halves the two arms are not comparable - one would be measuring a
    /// different quantity than the other.
    #[test]
    fn the_null_gate_arm_carries_the_coin_and_records_the_gate() {
        let arm_cfg = |arm| ReaderConfig { arm, ..cfg() };
        let run = |arm| {
            let d = Dir::new();
            let pool = Pool::create(&d.0, cfg().pool).expect("pool");
            let mut r = Reader::new(arm_cfg(arm), Fake { loss: 1.5, wins: 30, oracle: 0.9, ..Default::default() }, pool);
            (0..12).map(|i| r.step(&episode(&format!("e{i}"), 50))).collect::<Vec<Row>>()
        };

        let real = run(ControlArm::Real);
        let null = run(ControlArm::NullGate { seed: 11 });

        assert_eq!(
            real.iter().map(|w| w.outcome.stage()).collect::<Vec<_>>(),
            null.iter().map(|w| w.outcome.stage()).collect::<Vec<_>>(),
            "the null arm must run the same stream through the same gate; only what CARRIES may differ"
        );
        for w in &real {
            assert_eq!(w.carried, w.outcome.promoted(), "under the real arm the gate is what carries");
        }
        let gated: Vec<bool> = null.iter().map(|w| w.outcome.promoted()).collect();
        let coined: Vec<bool> = null.iter().map(|w| w.carried).collect();
        assert_ne!(gated, coined, "a coin that agreed with the gate on all twelve would not be a control");
        for (i, w) in null.iter().enumerate() {
            let reached_gate = w.outcome.stage() == "gate";
            let expect = reached_gate && promote::gate::coin(11, i + 1);
            assert_eq!(w.carried, expect, "episode {i}: the coin decides what carries, and only past the gate");
        }
    }

    /// The shuffled-label arm gates each episode against ANOTHER episode's
    /// frozen probes while training on its own rows. What it must actually
    /// do is substitute the probes - a run that quietly kept scoring the
    /// episode's own would be the real arm wearing a different name.
    #[test]
    fn the_shuffled_label_arm_gates_against_another_episodes_probes() {
        let d = Dir::new();
        let pool = Pool::create(&d.0, cfg().pool).expect("pool");
        let mut r = Reader::new(
            ReaderConfig { arm: ControlArm::ShuffledLabels { seed: 5 }, ..cfg() },
            Fake { loss: 1.5, wins: 30, oracle: 0.9, ..Default::default() },
            pool,
        );

        // The first episode has nothing to borrow from, so it is scored
        // against its own - stated in the arm's own doc rather than left to
        // be discovered as an inconsistency.
        let first = r.step(&episode("e0", 50));
        assert!(first.outcome.promoted(), "the fixture's first episode promotes, so there is something in the bank to borrow");
        let own_of_first: Vec<String> = r.learner().last_scored.clone();

        let second = r.step(&episode("e1", 50));
        assert_eq!(second.outcome.stage(), "gate", "the second episode must reach the gate for this to be about the gate");
        let scored_for_second = r.learner().last_scored.clone();

        let borrowed: Vec<&String> = scored_for_second.iter().filter(|p| own_of_first.contains(p)).collect();
        assert!(
            !borrowed.is_empty(),
            "the second episode must be gated against the first's probes; scored {scored_for_second:?} against own {own_of_first:?}"
        );
    }
}
