// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Discovery, as a program with no model in it.
//!
//! This is the SEARCH half of `SEARCH -> VERIFY -> SELECT -> COMPRESS`, split
//! off from the training pipeline it used to be a phase inside. The split is
//! not tidiness. Policy improvement is a compression mechanism: it makes a
//! better copy of behaviour already present in its data and has no mechanism
//! for finding behaviour that is not. The sample's own campaign measured that
//! directly - cloning converged on a teacher that dies on E1M1 with 6 of 29
//! kills, and no amount of it produced a run that finished a level. Reaching
//! a strategy nobody demonstrated is a different problem, and this is the
//! program that works on it.
//!
//! It costs no forward pass and loads no encoder, so it runs at engine speed:
//! measured on this box, about 19 ms a decision against roughly 100 ms with a
//! network in the loop.
//!
//! ## The contract that makes the result honest
//!
//! > **Search may use snapshots. The artifact may not.**
//!
//! Returning to a promising position by restoring an engine snapshot is what
//! makes this affordable - a replay of three hundred actions costs three
//! hundred steps every time, a snapshot costs one request. But a trajectory
//! assembled out of restores is not a run anybody could play. So what a
//! campaign PRODUCES is an action list from the level's own start, and it is
//! not in the solution set until that list has been replayed from the start
//! and has reproduced the same kills, secrets, exit and tic count. The engine
//! is deterministic - measured, two separate process launches on one seed
//! agree bit for bit - which is what makes that a real gate rather than a
//! hope.
//!
//! ## What it optimises
//!
//! [`crate::report::Bar::UvMax`]: every monster, every secret, then the exit.
//! There is no time term in that score on purpose (see its own docs); time is
//! the archive's tiebreaker instead, so the campaign drives the tic count
//! down everywhere at once without ever paying an agent to end its own run.

use std::time::{Duration, Instant};

use brain::decision::Rng;
use brain::search::{Admission, Allocator, Archive, Cascade, Gain, Niche, Rung, Verdict, Worth};
use brain::Demonstration;
use brain::Env;

use crate::env::DoomEnv;

/// How close two achievements have to be before the archive calls them equal
/// and lets the tic count decide.
///
/// The UV-Max bar moves in steps of `0.8/total_kills` per monster - about
/// 0.027 on E1M1 and 0.0045 on E1M6 - so a tolerance well below the smallest
/// of those keeps two genuinely different achievements apart while absorbing
/// float noise on the route term.
const SAME_ACHIEVEMENT: f32 = 1e-4;

/// How often an exploring walk repeats what it just did.
///
/// "To help explore in a consistent direction, the probability of repeating
/// the previous action is 95% for Atari and 90% for robotics" - Go-Explore,
/// Methods. Without it, uniform random per step is a random walk: it covers
/// distance like the square root of the steps taken, so a hundred steps go
/// almost nowhere and the archive stops growing.
const REPEAT: f32 = 0.95;

/// One way of producing a candidate.
///
/// Operators differ in only two things - how long they act for, and how often
/// they take the scripted player's choice rather than a random one - and that
/// is deliberate: a table of two numbers is something a campaign can be
/// honest about, where four hand-written walk functions would drift into four
/// slightly different definitions of what a walk is.
///
/// Which of them is worth running is [`Allocator`]'s problem, measured on
/// archive gain per second. It is NOT settled here, and the first campaign
/// showed why it cannot be: over 180 s on E1M1 the three original operators
/// scored 3.75, 3.57 and 3.70 gain/s - indistinguishable, because all three
/// walked 60 steps and 60 steps is not long enough to accomplish anything in
/// a level full of monsters. Measured against that campaign, the scripted
/// player playing straight through killed a monster every 244 decisions while
/// the search managed one every 960: the search was four times LESS
/// productive per step than simply playing. `commit` and `chase` are the
/// response - long stretches of real play resumed from a cell the archive
/// already reached, which is the thing a plain teacher run cannot do.
struct Operator {
    name: &'static str,
    /// Decisions to take after returning to a cell.
    walk: usize,
    /// How often to take the scripted player's choice. Below 1.0 the walk
    /// wanders off what the teacher would do, which is the only way to reach
    /// something no teacher demonstrates; at 1.0 it is the teacher continuing
    /// from somewhere the teacher could never have got to on its own.
    guided: f32,
    /// Set off from the best cell the archive holds rather than from a drawn
    /// one. What makes an operator a LOCAL search on the best thing found so
    /// far instead of a sample of the frontier.
    from_best: bool,
    /// How often to head for the way out ahead of anything else.
    ///
    /// The scripted player will not leave under UV-Max orders while anything
    /// is left to hunt - right for the category, since a Max run leaves last,
    /// and the reason no campaign had ever produced a trajectory that exits.
    /// The verification rung therefore never fired, and the one part of this
    /// machine that turns a search into an artifact anybody can use had never
    /// run at all.
    ///
    /// A search has to be able to reach the ending even from a state the
    /// teacher would not end from. What is KEPT is still decided by the
    /// score, where leaving early is worth less than clearing (0.6 against
    /// 1.3), so an operator that leaves too soon produces cells the archive
    /// files and does not prefer.
    leave: f32,
    /// Sweep walls: alternate pressing use with a sidestep, so the player
    /// runs along a wall testing it rather than pressing on one spot.
    ///
    /// Pressing use finds a secret only when the player happens to be facing
    /// the right wall, and a level has a great many walls. Measured on E1M1,
    /// pressing alone found one secret of three across campaigns totalling
    /// well over an hour, and the archive shows why the rest matter: 54 cells
    /// reached a state with ONE monster left and none ever cleared the level,
    /// so the monsters that are missing are behind the secrets that are
    /// missing.
    sweep: bool,
    /// How often to press use against whatever is in front of the player,
    /// ahead of anything else.
    ///
    /// The mechanism that finds SECRETS, and there is no other one. A DOOM
    /// secret is a wall that opens when pushed and looks very nearly like a
    /// wall that does not; nothing in the observation says which, and saying
    /// which would be handing the agent the answer. What a player does is
    /// push on things, so this does too. Measured before it existed, every
    /// campaign on E1M1 found zero secrets - which makes the Max category
    /// unreachable no matter how well the fighting goes.
    press: f32,
    /// How often to take a draw from the trained policy's distribution.
    ///
    /// The operator that closes the loop. Every other one here proposes with
    /// machinery written by hand, so a campaign is exactly as good as what
    /// was written and gets no better when the policy does. With this arm
    /// the search's proposal distribution improves as the model improves,
    /// and the model improves from what the search finds - which is the only
    /// reason to compress a search into a model at all.
    ///
    /// A DRAW rather than the argmax, deliberately. A deterministic policy
    /// resumed from the same cell walks the same way every time, so as a
    /// search operator it would be worth exactly one evaluation per cell.
    /// Sampled, it is a proposal distribution, which is what a policy is
    /// actually good for here.
    policy: f32,
}

const OPERATORS: [Operator; 7] = [
    // Blind, and blind in a level full of things that shoot back is mostly
    // dead - but it is the only operator that can produce an action no
    // teacher and no policy would ever pick.
    Operator { name: "wander", walk: 60, guided: 0.0, from_best: false, leave: 0.0, sweep: false, press: 0.0, policy: 0.0 },
    // Starts from somewhere plausible and wanders off it.
    Operator { name: "probe", walk: 60, guided: 0.85, from_best: false, leave: 0.0, sweep: false, press: 0.0, policy: 0.0 },
    // A long stretch of real play from a drawn cell. Long enough to finish a
    // firefight, clear a room and walk into the next one.
    Operator { name: "commit", walk: 400, guided: 1.0, from_best: false, leave: 0.0, sweep: false, press: 0.0, policy: 0.0 },
    // The same, from the furthest-along cell there is: pushing the front of
    // the search forward rather than filling in behind it.
    Operator { name: "chase", walk: 400, guided: 1.0, from_best: true, leave: 0.0, sweep: false, press: 0.0, policy: 0.0 },
    // Walk about pushing on walls. Half the decisions are a push, the rest
    // are the teacher moving on, which together is a player running their
    // shoulder along the walls of a room - the only way a secret is ever
    // found by someone who has not been told where one is. What makes it a
    // SEARCH rather than a coin flip is the ledger it pushes against: see
    // `memory::Sweep` and `Campaign::choose`.
    Operator { name: "frisk", walk: 200, guided: 1.0, from_best: false, leave: 0.0, sweep: true, press: 0.5, policy: 0.0 },
    // Play on from the furthest-along cell there is, and take the way out
    // when one is offered. The only operator that can produce a FINISHED
    // level, which is what the verification rung exists to check and what
    // the compression phase is supposed to be fitted to.
    Operator { name: "leave", walk: 400, guided: 1.0, from_best: true, leave: 0.4, sweep: false, press: 0.0, policy: 0.0 },
    // The trained policy playing on from a cell the archive already reached.
    //
    // Only drawn when a campaign was actually given a policy - see
    // `Campaign::arms`. `guided` sits at 1.0 behind it so a proposal the
    // model cannot answer falls back to the scripted player rather than to
    // noise: an arm that silently turns into `wander` when the backend
    // errors files `wander`'s gain under this arm's name, and the allocator
    // then spends its budget on a measurement of something else.
    Operator { name: "pursue", walk: 400, guided: 1.0, from_best: false, leave: 0.0, sweep: false, press: 0.0, policy: 1.0 },
];

/// What a trained policy would do here: a distribution over the options on
/// offer, given the objective they were offered under.
///
/// A closure rather than a model, because the search must not know what a
/// model is. It hosts a proposal distribution; where one comes from is the
/// caller's business, which is what keeps `crates/search` free of both the
/// engine and the GPU.
pub type Proposer<'a> = dyn FnMut(&str, &str, &[String]) -> Option<Vec<f32>> + 'a;

/// What a search campaign found, and what it cost.
pub struct Found {
    pub cells: usize,
    pub best: f32,
    pub best_tics: u64,
    /// Verified UV-Max runs: replayed from the level's own start and confirmed.
    pub solved: Vec<Solution>,
    /// The decisions that advanced the archive, for the compression phase.
    pub learned: Vec<Demonstration>,
}

/// A trajectory that was replayed from the level start and reproduced what it
/// claimed. The durable output of a campaign.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Solution {
    /// `E1M1`, or the generated scenario's name.
    pub level: String,
    pub skill: u32,
    /// The option TEXT chosen at each decision, from the level's own start.
    ///
    /// Text and not an index. On an exact replay the indices would be stable,
    /// because the same state rebuilds the same option list - but only for as
    /// long as nothing upstream changes how options are built, and the day
    /// that changes a stored index silently means a different action. Text is
    /// self-checking: a replay that cannot find the sentence it is looking for
    /// says so and fails, instead of quietly playing something else.
    pub actions: Vec<String>,
    /// Game tics the run took, at 35 to the second. What a speedrun is scored
    /// on, and what the archive minimises.
    pub tics: u64,
    pub kills: u32,
    pub total_kills: u32,
    pub secrets: u32,
    pub total_secrets: u32,
    pub score: f32,
    /// Whether this is the CATEGORY - every monster, every secret, and out -
    /// rather than merely a level finished.
    ///
    /// Recorded on the solution rather than recomputed by a reader, and
    /// reported apart from the count of verified runs. A campaign that
    /// verified two finishes and no Max saying "2 verified UV-Max" is a
    /// claim nobody made; the whole point of a verification rung is that
    /// what it reports is what happened.
    pub uvmax: bool,
}

impl Solution {
    pub fn seconds(&self) -> f64 {
        self.tics as f64 / 35.0
    }

    /// `2:14.83`, the way a runner writes a time.
    pub fn clock(&self) -> String {
        let s = self.seconds();
        format!("{}:{:05.2}", (s / 60.0) as u64, s % 60.0)
    }
}

/// How a cell was reached: the whole way there, from the level's own start.
///
/// ABSOLUTE, not relative, and this is the second design it has had. The first
/// stored a chain - "resume at cell P, then do these things" - because a full
/// list per cell looked like hundreds of megabytes of duplicated prefix. It
/// does not work, and the reason is structural rather than a bug that could be
/// fixed in place: an archive IMPROVES cells and EVICTS them, and a trajectory
/// hanging off P is invalidated by either.
///
/// Measured before it was replaced: the best cell of an 1817-cell archive
/// could not be traced back to the level's start at all, and neither could the
/// best cell of a fresh 443-cell one. Every artifact the campaign existed to
/// produce was unreachable, and nothing said so until an audit was written to
/// ask - the search itself never notices, because it restores snapshots and
/// carries happily on.
///
/// The duplication is paid for by INTERNING instead. An option's text is one
/// of a few thousand distinct sentences, so a trail is a list of `u32` into a
/// vocabulary the campaign owns: a thousand-decision trail is four kilobytes,
/// and an 1800-cell archive holds single-digit megabytes of them. Nothing can
/// go stale, because nothing points at anything that is allowed to change.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Trail {
    /// Indices into the campaign's vocabulary - the INPUTS sent to the game,
    /// from the level's own start. See `DoomEnv::inputs` for why the input
    /// and not the option's sentence.
    pub steps: Vec<u32>,
    /// Where the run actually WAS, every [`WITNESS_EVERY`] decisions.
    ///
    /// A trail that does not replay is useless, and "does not replay" arrives
    /// as a missing option sentence hundreds of decisions in - which says
    /// that the two runs disagree and nothing about where they started to.
    /// These are the witnesses that turn it into a measurement: the first
    /// mark that disagrees is the first place the replay and the search
    /// parted company, and by how far.
    pub marks: Vec<Mark>,
}

/// Where a run stood at one decision.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Mark {
    pub at: u32,
    pub x: i32,
    pub y: i32,
    pub angle: i32,
    pub tic: i64,
    /// A digest of the observation the agent READ at that decision.
    ///
    /// Position, facing and the clock are the simulation; this is everything
    /// else, in one number - what is in sight, what is remembered, what the
    /// route makes of where the player stands. Two runs whose observations
    /// agree will make the same decision, so a digest that disagrees is the
    /// first place they could possibly have parted company, whether the cause
    /// was the world or the agent's own memory of it. Position alone cannot
    /// say that: a run can stand in exactly the right place holding a
    /// different idea of what it has seen.
    #[serde(default)]
    pub read: u64,
    /// The observation itself, for the first few decisions only.
    ///
    /// A digest says THAT two runs read differently and never WHICH LINE, and
    /// which line is the whole diagnosis - the memory of what has been seen,
    /// the history the agent keeps, what the route makes of where it stands,
    /// are three different faults. Kept only near the start, where a
    /// divergence is still traceable, because the text is a hundred times the
    /// size of the digest and a campaign holds thousands of trails.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub said: String,
}

/// How many decisions into a trail the observation text is kept beside its
/// digest. See [`Mark::said`].
const EXPLAIN_FIRST: u32 = 128;

/// Whether a written-down action names what it MEANT as well as what it
/// sent.
///
/// An action is recorded as `tag|tics|commands` (see `DoomEnv::inputs`); it
/// used to be recorded as `tics|commands`, which named two different acts
/// whenever two options sent the same thing - and on E1M1 that was nearly
/// every decision. The two forms are told apart by their first character: a
/// tag name starts with a letter and a tic count with a digit. That is
/// enough, because the only question being asked is whether a file predates
/// the fix.
fn names_an_act(written: &str) -> bool {
    written.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
}

/// FNV-1a over the observation text.
///
/// A digest rather than the text: a trail carries a mark every few decisions
/// and storing the prose would make the witnesses larger than the trail. It
/// is a comparison, never a lookup, so collision resistance past "two
/// different observations rarely collide" buys nothing.
pub fn digest(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// How often a trail writes down where it was.
///
/// Often enough to localise a divergence to a handful of decisions, rare
/// enough that the marks are a few percent of the trail's own size.
const WITNESS_EVERY: usize = 8;

/// How many trails an archive audit replays, spread across their lengths.
///
/// Each one costs a whole episode against the engine, so this is a handful
/// rather than a sweep - enough to say whether the archive's artifacts are
/// trustworthy and where they stop being so.
const AUDIT_SAMPLES: usize = 6;

/// How many cells read back off disk a campaign will pay to make returnable
/// again.
///
/// Rehydrating one means REPLAYING its trail from the level's own start, at
/// engine speed - for a cell deep in a level that is upwards of a thousand
/// decisions, tens of seconds. Unbounded, a reloaded campaign would spend its
/// whole budget rebuilding snapshots it then draws from twice. Bounded, it
/// resumes the most promising handful and re-finds the rest, which it is
/// quite good at: the trails that were not rehydrated are still in the
/// archive and still feed the compression phase.
const REHYDRATIONS: usize = 24;

/// How many times [`Campaign::pick_reachable`] will redraw looking for a cell
/// it can return to before falling back to the level's start.
///
/// Weighted selection already favours the frontier, so a handful of draws
/// finds a live cell whenever a reasonable share of the archive is live. The
/// bound exists for the opposite case - an archive that is almost entirely
/// trails - where looping until one turns up would be an unbounded search
/// inside what is supposed to be one unit of work.
const PICK_TRIES: usize = 32;

/// How many decisions a campaign hands to the compression phase.
///
/// Bounded because a long campaign admits tens of thousands and a training
/// set is not better for being unbounded - only slower to fit, and more
/// heavily weighted toward whatever the search happened to do most of.
const KEEP_DECISIONS: usize = 20_000;

/// What a campaign writes to disk: the archive and the vocabulary its trails
/// are spelled in.
///
/// One file, never two. A trail is a list of indices, so an archive read back
/// against a vocabulary that is not its own names DIFFERENT actions - and
/// nothing downstream could catch it, because every index would still resolve
/// to a perfectly good sentence and the replay would simply do something else.
#[derive(serde::Serialize, serde::Deserialize)]
struct Kept {
    vocab: Vec<String>,
    archive: Archive<Trail>,
}

/// One search campaign against one loaded level.
pub struct Campaign {
    archive: Archive<Trail>,
    alloc: Allocator,
    rng: Rng,
    /// The niche the level's own start sits in - where every chain terminates
    /// and the one cell that may have no parent.
    start: Option<Niche>,
    /// The episode seed every replay has to use. A trail is only a way back
    /// to a cell on the level it was walked on.
    seed: u64,
    /// Every distinct option sentence the campaign has seen, so a trail is a
    /// list of indices rather than a list of strings. See [`Trail`].
    vocab: Vec<String>,
    /// The reverse lookup, which is the only reason interning is cheap.
    spoken: std::collections::HashMap<String, u32>,
    steps: usize,
    restores: usize,
    refused: usize,
    /// Cells whose engine snapshot exists IN THIS PROCESS.
    ///
    /// An archive read back off disk carries trails, not snapshots - a
    /// snapshot lives in the engine, and the engine that held it has exited.
    /// So a reloaded cell is a place the search knows how to reach and cannot
    /// yet return to, and the difference has to be tracked or `resume_from`
    /// silently restores whatever the slot held last, which is some other
    /// cell's state. That is the worst possible failure for a search: it
    /// explores the wrong place and reports the right one.
    live: std::collections::HashSet<Niche>,
    /// How many more reloaded cells may be paid for. See [`Campaign::go_to`].
    rehydrations: usize,
    /// Decisions worth imitating, for the compression phase.
    ///
    /// What a search hands to training, and it is DECISIONS rather than
    /// trajectories on purpose. Behaviour cloning fits state-to-action, so
    /// whole episodes buy it nothing - and requiring them would force this to
    /// solve a problem it does not have, since a walk starts from a restored
    /// snapshot and the way to that snapshot is not something the walk holds.
    ///
    /// Two sources, and they are not the same thing. Decisions that ADVANCED
    /// the archive - a walk that reached somewhere new or somewhere faster -
    /// are the ones whose choices are worth copying; the rest of what a
    /// search does is mostly the walk that did not work, and fitting a
    /// policy to that teaches it to wander. And every decision of a VERIFIED
    /// run, which is a level played start to finish: the fragments teach a
    /// policy what to do in a situation, and only a whole run teaches it
    /// what the situations are in the order they come.
    learned: Vec<Demonstration>,
    /// Which of [`OPERATORS`] this campaign may draw, by index.
    ///
    /// An operator that needs something the campaign was not given - a
    /// policy, today - is left OUT rather than degraded into a different
    /// operator. A disabled arm that quietly behaves like `wander` still
    /// reports its gain under its own name, and the allocator then spends
    /// real budget comparing an operator against a copy of another one.
    arms: Vec<usize>,
    /// The most walls any one trajectory has pushed on, so that "the search
    /// is looking for secrets" is a number rather than a hope. A high-water
    /// mark, because the live figure belongs to whichever cell was restored
    /// last and says nothing about the campaign.
    searched: usize,
    /// Completed categories noticed during a walk, waiting to be replayed.
    ///
    /// Queued rather than verified on the spot because verifying RESTARTS the
    /// level, and doing that in the middle of a walk would throw away the rest
    /// of the walk's budget.
    claims: Vec<Claim>,
}

impl Campaign {
    /// `with_policy` says whether a proposal distribution will be supplied.
    /// Without one the policy arm is not offered at all - see
    /// [`Campaign::arms`].
    pub fn new(slots: usize, seed: u64, with_policy: bool) -> Campaign {
        let arms: Vec<usize> = OPERATORS
            .iter()
            .enumerate()
            .filter(|(_, o)| with_policy || o.policy == 0.0)
            .map(|(i, _)| i)
            .collect();
        let names: Vec<&'static str> = arms.iter().map(|i| OPERATORS[*i].name).collect();
        Campaign {
            archive: Archive::new(slots, SAME_ACHIEVEMENT),
            alloc: Allocator::new(&names),
            rng: Rng::new(seed),
            start: None,
            seed,
            vocab: Vec::new(),
            spoken: std::collections::HashMap::new(),
            steps: 0,
            restores: 0,
            refused: 0,
            live: std::collections::HashSet::new(),
            rehydrations: REHYDRATIONS,
            learned: Vec::new(),
            arms,
            searched: 0,
            claims: Vec::new(),
        }
    }

    /// The operators this campaign is drawing between, for its opening line.
    fn operators(&self) -> String {
        self.arms.iter().map(|i| OPERATORS[*i].name).collect::<Vec<_>>().join(", ")
    }

    /// An index drawn from `weights`, which are the policy's probabilities.
    ///
    /// Drawn with the CAMPAIGN's own generator, so a search seeded the same
    /// way makes the same draws: the policy is a distribution, and where the
    /// randomness that resolves it comes from is a property of the search.
    fn draw(&mut self, weights: &[f32]) -> Option<usize> {
        let total: f32 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
        // `<= 0.0` first so a NaN total falls through to the finite check
        // rather than being compared into a silent `true`.
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let mut point = self.rng.next_f32() * total;
        for (i, w) in weights.iter().enumerate() {
            if !w.is_finite() || *w <= 0.0 {
                continue;
            }
            point -= *w;
            if point <= 0.0 {
                return Some(i);
            }
        }
        // Floating-point arithmetic can leave `point` a hair above zero after
        // the last subtraction. The last positive weight is the answer.
        weights.iter().rposition(|w| w.is_finite() && *w > 0.0)
    }

    /// The index of an option sentence, adding it to the vocabulary if this
    /// is the first time it has come up.
    fn intern(&mut self, text: &str) -> u32 {
        if let Some(i) = self.spoken.get(text) {
            return *i;
        }
        let i = self.vocab.len() as u32;
        self.vocab.push(text.to_string());
        self.spoken.insert(text.to_string(), i);
        i
    }

    /// A trail as the sentences it is made of.
    ///
    /// Infallible, unlike the chain-walking it replaces: a trail holds the
    /// whole way from the level's start and points at nothing that can be
    /// improved or evicted out from under it. An index outside the vocabulary
    /// can only mean a corrupt archive, and is reported as one.
    fn spell(&self, trail: &Trail) -> Option<Vec<String>> {
        trail
            .steps
            .iter()
            .map(|i| self.vocab.get(*i as usize).cloned())
            .collect()
    }

    /// Read an archive back off disk, if there is one.
    ///
    /// Its cells carry trails and no snapshots; see [`Campaign::live`]. What
    /// this buys is a campaign that COMPOUNDS - without it, every run searches
    /// from the level's front door again and a solution found once can be
    /// quietly lost.
    fn carry_in(&mut self, path: &str) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        match serde_json::from_str::<Kept>(&text) {
            Ok(was) => {
                // The vocabulary comes back FIRST and whole. A trail is a list
                // of indices into it, so an archive read back against a
                // different vocabulary would name different actions - which is
                // not a corruption anything downstream could detect, because
                // every index would still resolve to a perfectly good
                // sentence. They are written and read as one file for exactly
                // that reason.
                if let Some(stale) = was.vocab.iter().find(|w| !names_an_act(w)) {
                    // Refused, not repaired and not silently carried. Every
                    // trail in the file is a list of indices into this
                    // vocabulary, so an archive whose actions were written
                    // down before they named what they MEANT is one whose
                    // every trail replays as a different run - and the
                    // symptom of carrying it anyway is not an error, it is a
                    // search reporting cells it cannot return to and
                    // solutions it cannot verify. See `DoomEnv::inputs`.
                    println!(
                        "  {path} was written before an action recorded what it MEANT \
                         ({stale:?} names no act), so none of its trails can be replayed. \
                         Delete it and search again; nothing here can repair it."
                    );
                    return;
                }
                self.spoken = was
                    .vocab
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (w.clone(), i as u32))
                    .collect();
                self.vocab = was.vocab;
                let was = was.archive;
                println!(
                    "  carried in {} cells from {path}, best {:.3} - up to {} of them \
                     will be walked back to, the rest are trails only",
                    was.len(),
                    was.best().map(|e| e.worth.reached).unwrap_or(0.0),
                    REHYDRATIONS
                );
                self.archive = was;
            }
            // Said out loud rather than started fresh. An archive that will
            // not parse is a campaign's whole history, and silently replacing
            // it with an empty one is how a week of searching disappears.
            Err(e) => println!("  {path} will not be read ({e}); searching from nothing"),
        }
    }

    fn carry_out(&self, path: &str) {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let kept = Kept {
            vocab: self.vocab.clone(),
            archive: self.archive.clone(),
        };
        match serde_json::to_string(&kept)
            .map_err(|e| format!("{e}"))
            .and_then(|t| std::fs::write(path, t).map_err(|e| format!("{path}: {e}")))
        {
            Ok(()) => println!(
                "  archive: {} cells and {} distinct options to {path}",
                self.archive.len(),
                self.vocab.len()
            ),
            Err(e) => println!("  the archive could NOT be written: {e}"),
        }
    }

    /// Stand the level up and file the spawn, so the first draw has somewhere
    /// to go. Without this the archive is empty and every operator returns
    /// having done nothing.
    fn seed_start(&mut self, env: &mut DoomEnv, seed: u64, allowed: u32) -> Result<(), String> {
        env.reset(seed);
        if let Some(f) = env.fault() {
            return Err(format!("the engine faulted before the search began: {f}"));
        }
        let Some(cell) = env.cell() else {
            return Err("the engine would not say where the player is".into());
        };
        let worth = Worth::new(env.score(allowed).value(), env.cost());
        if !env.hold_at(0) {
            return Err("the engine would not hold the level's starting state".into());
        }
        self.archive.offer(cell.clone(), worth, Trail::default());
        // The archive assigns the slot; the engine is then told to hold that
        // slot. Doing it the other way round - hold first, file second - is
        // how a cell ends up pointing at another cell's snapshot.
        let slot = self.archive.get(&cell).map(|e| e.slot).unwrap_or(0);
        if !env.hold_at(slot) {
            self.archive.remove(&cell);
            return Err("the engine would not hold the level's starting state".into());
        }
        // It was just held, so it is returnable in THIS process. Saying so
        // matters most on a carried-in archive, where the offer above is
        // refused (the cell is already there, reached at least as well) and
        // the start would otherwise be the one cell the campaign cannot go
        // back to.
        self.live.insert(cell.clone());
        self.start = Some(cell);
        Ok(())
    }

    /// Replay the best trail the archive holds and check that it arrives
    /// where the archive says it does.
    ///
    /// Always, never on request. An archive whose trails do not replay is an
    /// archive whose ARTIFACTS ARE WORTHLESS - every solution it ever
    /// produces will fail the same way - and the failure is invisible from
    /// inside the search, which restores snapshots and carries happily on.
    /// The damage shows up only at the far end, which is exactly the shape of
    /// defect an optional gate is worst at catching: a gate that never runs
    /// is worse than no gate, because it also removes the suspicion that
    /// there was ever anything to check.
    ///
    /// It found a real one. `DoomEnv`'s `trail` - where the player has
    /// actually moved, which the "fall back the way you came" option is
    /// computed from - was not part of a snapshot, so a restored run offered
    /// a different option list than the one that had been held, and every
    /// trajectory that finished a level failed to replay at the same
    /// decision, naming an option the game no longer offered.
    ///
    /// Costs one episode, once per campaign.
    fn audit(&mut self, env: &mut DoomEnv, allowed: u32) {
        // Shortest first, then a middling one, then the best. The order is
        // the diagnosis: a SHORT trail is one walk straight off the level's
        // start with no resuming in it, so if that fails the recording itself
        // is wrong, and if only the long ones fail the fault is in what a
        // resume restores. Reporting one number could not tell those apart.
        let mut picked: Vec<(usize, f32, Trail)> = self
            .archive
            .iter()
            .filter(|e| !e.what.steps.is_empty())
            .map(|e| (e.what.steps.len(), e.worth.reached, e.what.clone()))
            .collect();
        if picked.is_empty() {
            return;
        }
        picked.sort_by_key(|(n, _, _)| *n);
        // Spread across the LENGTHS, because length is what the diagnosis
        // turns on: a short trail is one walk straight off the level's start
        // with no resuming in it, so the shortest one that fails is the
        // boundary between "the recording is wrong" and "what a resume
        // restores is wrong".
        let last = picked.len() - 1;
        let mut sample: Vec<usize> = (0..AUDIT_SAMPLES)
            .map(|i| last * i / (AUDIT_SAMPLES - 1).max(1))
            .collect();
        sample.dedup();

        let (mut good, mut bad) = (0, 0);
        for i in sample {
            let (n, said, trail) = picked[i].clone();
            let Some(actions) = self.spell(&trail) else {
                println!("  archive audit: a trail names options outside the vocabulary");
                bad += 1;
                continue;
            };
            match replay_checked(env, self.seed, &actions, allowed, &trail.marks, None) {
                Ok(got) if (got.value() - said).abs() <= SAME_ACHIEVEMENT => {
                    good += 1;
                    println!("    {n:>5} decisions -> {said:.3}, replays");
                }
                Ok(got) => {
                    bad += 1;
                    println!(
                        "    {n:>5} decisions -> claims {said:.3}, REPLAYS TO {:.3}",
                        got.value()
                    );
                }
                Err(e) => {
                    bad += 1;
                    println!("    {n:>5} decisions -> WILL NOT REPLAY: {e}");
                }
            }
        }
        if bad == 0 {
            println!("  archive audit: every sampled trail replays from the level's own start");
        } else {
            println!(
                "  archive audit FAILED on {bad} of {} sampled trails. Until that is fixed \
                 this archive cannot produce an artifact anybody can use.",
                good + bad
            );
        }
    }

    /// File a state the search has just reached, holding the engine snapshot
    /// in whatever slot the archive gave it.
    ///
    /// A cell the engine will not hold is REMOVED rather than kept: an
    /// archive entry whose snapshot does not exist is one that returns to
    /// some other cell's state and then reports having explored this one.
    fn file(&mut self, env: &mut DoomEnv, cell: Niche, worth: Worth, trail: Trail) -> Admission {
        let verdict = self.archive.offer(cell.clone(), worth, trail);
        if verdict == Admission::Rejected {
            return verdict;
        }
        let Some(slot) = self.archive.get(&cell).map(|e| e.slot) else {
            return Admission::Rejected;
        };
        if !env.hold_at(slot) {
            self.archive.remove(&cell);
            return Admission::Rejected;
        }
        self.live.insert(cell);
        verdict
    }

    /// Draw a cell the campaign can actually set off from.
    ///
    /// On a carried-in archive almost every cell is a trail with no snapshot
    /// behind it, and rebuilding one is bounded (see [`REHYDRATIONS`]). Once
    /// that budget is gone a plain draw returns an unreachable cell nearly
    /// every time, and the campaign does NOTHING for the rest of its run
    /// while reporting an archive full of cells - measured, a reloaded
    /// campaign sat for 150 seconds with zero steps and zero resumes taken,
    /// and would have sat there for the remaining eighty minutes.
    ///
    /// So: draw normally while there is budget to rebuild, and draw only from
    /// what is returnable once there is not. The fallback is the level's own
    /// start, which is always returnable, so the answer is never nothing.
    fn pick_reachable(&mut self) -> Option<(Niche, usize)> {
        let rebuilding = self.rehydrations > 0;
        for _ in 0..PICK_TRIES {
            let drawn = self
                .archive
                .pick(&mut self.rng)
                .map(|e| (e.niche.clone(), e.slot))?;
            if rebuilding || self.live.contains(&drawn.0) {
                return Some(drawn);
            }
        }
        let start = self.start.clone()?;
        let e = self.archive.get(&start)?;
        Some((start.clone(), e.slot))
    }

    /// Return to a cell, rebuilding its snapshot first if this process never
    /// held one.
    ///
    /// A refusal is reported and the cell is DROPPED, never papered over by
    /// starting the level again: a search that quietly restarts is one that
    /// explores the opening a thousand times and says it resumed.
    fn go_to(
        &mut self,
        env: &mut DoomEnv,
        at: &Niche,
        slot: usize,
        allowed: u32,
    ) -> Option<String> {
        if self.live.contains(at) {
            if let Some(seen) = env.resume_from(slot) {
                return Some(seen);
            }
            self.refused += 1;
            self.live.remove(at);
            self.archive.remove(at);
            return None;
        }
        // Read back off disk. The only way to stand where it stands is to
        // walk there again from the level's own start.
        if self.rehydrations == 0 {
            return None;
        }
        self.rehydrations -= 1;
        let Some(trail) = self.archive.get(at).map(|e| e.what.clone()) else {
            self.archive.remove(at);
            return None;
        };
        let Some(actions) = self.spell(&trail) else {
            self.archive.remove(at);
            return None;
        };
        // CHECKED, against the trail's own witnesses. A walk back that ends
        // somewhere other than where the archive says it ends is the worst
        // failure a search can have: it explores one place and reports
        // another, and every cell it then files carries a trail that does
        // not lead to it. Verifying costs a position comparison every eighth
        // decision on a walk that is happening anyway, and a rehydration is
        // bounded to `REHYDRATIONS` a campaign.
        if replay_checked(env, self.seed, &actions, allowed, &trail.marks, None).is_err()
            || !env.hold_at(slot)
        {
            self.archive.remove(at);
            return None;
        }
        self.live.insert(at.clone());
        // Walked back to rather than restored, so the observation is simply
        // the one the replay ended on.
        Some(env.look())
    }

    /// Keep a decision for the compression phase, bounded.
    ///
    /// A long campaign admits tens of thousands of cells, and a training set
    /// is not better for being unbounded - it is just slower to fit and more
    /// heavily weighted toward whatever the search happened to do most of.
    /// Past the cap the oldest go, because the newest come from further along
    /// and are the ones a policy most needs.
    fn keep(&mut self, d: Demonstration) {
        self.learned.push(d);
        if self.learned.len() > KEEP_DECISIONS {
            let drop = self.learned.len() - KEEP_DECISIONS;
            self.learned.drain(0..drop);
        }
    }

    /// Run one operator once and report what it bought.
    ///
    /// Every operator has the same shape - go back somewhere, act for a
    /// while, file what was reached - and differs only in WHERE it returns to
    /// and HOW it chooses each action. Keeping that shape in one place is
    /// what stops three operators drifting into three slightly different
    /// definitions of what a walk is.
    fn operate(
        &mut self,
        env: &mut DoomEnv,
        arm: usize,
        allowed: u32,
        mut propose: Option<&mut Proposer<'_>>,
    ) -> Gain {
        let began = Instant::now();
        let mut gain = Gain::default();
        let op = &OPERATORS[self.arms[arm]];

        let picked = if op.from_best {
            self.archive.best().map(|e| (e.niche.clone(), e.slot))
        } else {
            self.pick_reachable()
        };
        let Some((from, slot)) = picked else {
            gain.seconds = began.elapsed().as_secs_f64();
            return gain;
        };
        let Some(resumed) = self.go_to(env, &from, slot, allowed) else {
            gain.seconds = began.elapsed().as_secs_f64();
            return gain;
        };
        self.restores += 1;
        // The observation the walk is about to decide on, carried forward
        // from each step so a demonstration records what was actually READ
        // rather than what the state looked like afterwards.
        let mut here_and_now = resumed;

        // What it took to get to the cell this walk resumed at. Copied out
        // once, because every cell the walk files carries the whole way from
        // the level's start.
        let prefix = self.archive.get(&from).map(|e| e.what.steps.clone()).unwrap_or_default();
        let marked = self.archive.get(&from).map(|e| e.what.marks.clone()).unwrap_or_default();
        let mut steps: Vec<u32> = Vec::new();
        let mut marks: Vec<Mark> = Vec::new();
        let mut last: Option<String> = None;
        for _ in 0..op.walk {
            let options = env.actions();
            if options.is_empty() {
                break;
            }
            let chose = self.choose(
                env,
                op,
                &options,
                last.as_deref(),
                &here_and_now,
                propose.as_deref_mut(),
            );
            // Captured BEFORE the step, because a decision is a question
            // about the state it was asked in and that state is about to
            // stop existing. Whether it is worth keeping is decided a few
            // lines below, once the archive has said what it bought.
            let asked = Demonstration {
                objective: env.objective(),
                observation: here_and_now.clone(),
                options: options.clone(),
                action: chose,
            };
            last = Some(options[chose].clone());
            steps.push(self.intern(&env.inputs()[chose]));
            let (next, _, done) = env.step(chose);
            here_and_now = next;
            self.steps += 1;
            self.searched = self.searched.max(env.walls_tested());
            if env.fault().is_some() {
                break;
            }
            let here = prefix.len() + steps.len();
            if here % WITNESS_EVERY == 0 {
                if let Some(mut m) = env.mark(here as u32, &here_and_now) {
                    if m.at < EXPLAIN_FIRST {
                        m.said = here_and_now.clone();
                    }
                    marks.push(m);
                }
            }
            if let Some(cell) = env.cell() {
                let worth = Worth::new(env.score(allowed).value(), env.cost());
                // The whole way here from the level's own start: what it
                // took to reach the cell this walk resumed at, plus what the
                // walk has done since.
                let mut whole = prefix.clone();
                whole.extend_from_slice(&steps);
                let mut seen = marked.clone();
                seen.extend_from_slice(&marks);
                let trail = Trail {
                    steps: whole,
                    marks: seen,
                };
                // What the archive's best was BEFORE this admission, so a
                // cell that advances the frontier is credited against the old
                // frontier rather than against itself.
                let top = self.archive.best().map(|e| e.worth.reached).unwrap_or(0.0);
                match self.file(env, cell, worth, trail) {
                    Admission::Fresh => {
                        gain.admitted(true, worth.reached, top);
                        self.keep(asked);
                    }
                    Admission::Improved => {
                        gain.admitted(false, worth.reached, top);
                        self.keep(asked);
                    }
                    Admission::Rejected => {}
                }
            }
            if done {
                let ended = env.score(allowed);
                if ended.finished {
                    // The walk's own whole trajectory, taken from what it was
                    // resumed with rather than read back out of the archive:
                    // the terminal cell may have been refused in favour of an
                    // equally good one, and reading the archive would then
                    // hand back a different trajectory than the one that just
                    // finished the level.
                    let mut whole = prefix.clone();
                    whole.extend_from_slice(&steps);
                    let mut seen = marked.clone();
                    seen.extend_from_slice(&marks);
                    if let Some(actions) = self.spell(&Trail {
                        steps: whole,
                        marks: seen.clone(),
                    }) {
                        self.claims.push(Claim {
                            actions,
                            marks: seen,
                            claimed: ended,
                            claimed_tics: env.cost(),
                        });
                    }
                }
                break;
            }
        }
        gain.seconds = began.elapsed().as_secs_f64();
        gain
    }

    /// One progress line: what the archive holds and what it has verified.
    ///
    /// Coverage per axis rather than a cell count on its own, because a cell
    /// count cannot tell a search that is opening new regions from one stuck
    /// in a room producing a new square of floor every few steps. The axes
    /// are the niche's own: map, x, y, keys, monsters left, secrets found.
    fn say(&self, level: &str, elapsed: Duration, solved: &[Solution]) {
        let walls = self.searched;
        let cover = self
            .archive
            .coverage()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("/");
        let best = self.archive.best();
        println!(
            "  {level} {:>5.0}s: {} cells (cover {}), best {:.3} in {} tics, {walls} walls, {} steps, {} resumes, {} verified",
            elapsed.as_secs_f64(),
            self.archive.len(),
            cover,
            best.map(|e| e.worth.reached).unwrap_or(0.0),
            best.map(|e| e.worth.cost).unwrap_or(0),
            self.steps,
            self.restores,
            solved.len()
        );
    }

    /// Which option this operator takes next.
    fn choose(
        &mut self,
        env: &mut DoomEnv,
        op: &Operator,
        options: &[String],
        last: Option<&str>,
        reading: &str,
        mut propose: Option<&mut Proposer<'_>>,
    ) -> usize {
        // The policy first, when this arm has one: everything below is a
        // fallback for a proposal that could not be made.
        if op.policy > 0.0 && self.rng.next_f32() < op.policy {
            let asked = propose
                .as_mut()
                .and_then(|f| f(&env.objective(), reading, options))
                .filter(|p| p.len() == options.len());
            if let Some(i) = asked.and_then(|p| self.draw(&p)) {
                return i;
            }
        }
        if op.leave > 0.0 && self.rng.next_f32() < op.leave {
            if let Some(i) = env.exit_option().filter(|i| *i < options.len()) {
                return i;
            }
        }
        if op.press > 0.0 && self.rng.next_f32() < op.press {
            // A wall this run has already pushed on teaches nothing by being
            // pushed again. So a sweep that finds one slides along to the
            // next piece of wall, and when there is nowhere left to slide to
            // it sets off for a room that has never been searched at all.
            //
            // This is the whole difference between pressing use and
            // SEARCHING. Measured without the ledger, campaigns totalling
            // well over an hour on E1M1 found one secret of three: an
            // alternation driven by "did I press last time" re-tests the
            // same wall as soon as anything else happens in between, and in
            // a level full of monsters something always does.
            //
            // And there has to be something within arm's length to push ON.
            // DOOM's use range is 64 units, so a push made in the middle of
            // a room reaches no wall, and the decision is better spent
            // getting to one - which is what falling through to the scripted
            // player does. Measured before this test existed: 174 walls
            // "tested" across a campaign on E1M1 and not one secret found,
            // because most of the pushing was done at nothing.
            // Nothing within arm's length to push ON. DOOM's use range is
            // 64 units, so a push made in the middle of a room reaches no
            // wall - the decision is better spent on the walk, which is
            // what falling through to the scripted player does.
            //
            // Steering at the nearest wall instead was tried and is worse:
            // on the same level, seed and budget it scored 0.331 by 93
            // seconds against 1.039, and its count of walls actually tested
            // stopped rising after the first thirty seconds. Walking at the
            // least room also selects the ways BACKWARD - a wall behind the
            // player is near too - so the walk oscillates into a corner
            // rather than coming alongside anything, and the walks ended
            // early enough to triple the number of resumes.
            if !env.wall_in_reach() {
                // Let the walk carry on; it will pass a wall soon enough.
            } else if env.pressed_here() {
                if op.sweep {
                    if let Some(i) = env
                        .sidestep_options()
                        .first()
                        .filter(|i| **i < options.len())
                    {
                        return *i;
                    }
                }
                if let Some(i) = env.frisk_option().filter(|i| *i < options.len()) {
                    return i;
                }
            } else if let Some(i) = env.use_option().filter(|i| *i < options.len()) {
                return i;
            }
        }
        if op.guided > 0.0 && self.rng.next_f32() < op.guided {
            if let Some(i) = env.demo().filter(|i| *i < options.len()) {
                return i;
            }
        }
        // An option list rebuilt every step has no stable index, so "the same
        // action again" is matched on the TEXT up to the number in it:
        // "walk forward, 320 units" and "walk forward, 288 units" are the
        // same intention.
        if let Some(prev) = last.filter(|_| self.rng.next_f32() < REPEAT) {
            if let Some(i) = same_again(prev, options) {
                return i;
            }
        }
        (self.rng.next_u64() as usize) % options.len()
    }
}

/// The option in `options` that means what `prev` meant, if any.
///
/// Matched on the words rather than the whole sentence, because the numbers
/// in an option are recomputed every step: the distance to a monster changes
/// while the intention "attack that monster" does not.
fn same_again(prev: &str, options: &[String]) -> Option<usize> {
    let words = |s: &str| {
        s.split_whitespace()
            .filter(|w| !w.chars().next().is_some_and(|c| c.is_ascii_digit()))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let want = words(prev);
    options.iter().position(|o| words(o) == want)
}

/// Replay an action list from the level's own start, checking the trail's
/// own witnesses as it goes, and report what actually happened.
///
/// The expensive rung of the cascade, and the only one whose answer is worth
/// anything to a human: everything above it reasons about states reached by
/// restoring snapshots, which is not a run anybody could play. This is a
/// whole episode against the engine, from the level's front door, taking only
/// the actions in the list. Search may use snapshots; the artifact may not.
///
/// An option the list names and the game does not offer is a HARD failure,
/// never a skip. The list is a claim about what the level does; a replay that
/// silently walks past a missing option is one that verifies a different
/// trajectory and reports success.
///
/// The witnesses are what make a failure actionable. It reports the FIRST
/// decision at which the replay stood somewhere the search did not, or stood
/// in exactly the right place and READ something different there - the only
/// useful thing to know about a trail that does not reproduce, and two
/// entirely different faults. Without them the failure arrives as a missing
/// option three hundred decisions in, which says the two runs disagree and
/// nothing whatever about where they began to.
///
/// An empty `marks` checks nothing on the way and reports only the outcome.
pub fn replay_checked(
    env: &mut DoomEnv,
    seed: u64,
    actions: &[String],
    allowed: u32,
    marks: &[Mark],
    mut keep: Option<&mut Vec<Demonstration>>,
) -> Result<crate::report::Score, String> {
    env.reset(seed);
    if let Some(f) = env.fault() {
        return Err(format!("the engine faulted at the level's start: {f}"));
    }
    let mut expect = marks.iter().peekable();
    for (i, want) in actions.iter().enumerate() {
        if let Some(m) = expect.peek() {
            if m.at as usize == i {
                let m = expect.next().expect("peeked");
                if let Some(now) = env.mark(i as u32, &env.look()) {
                    if (now.x, now.y, now.angle) != (m.x, m.y, m.angle) {
                        return Err(format!(
                            "at decision {i} the replay stands at ({}, {}) facing {} on tic {}, \
                             where the search stood at ({}, {}) facing {} on tic {} - \
                             {} units apart",
                            now.x, now.y, now.angle, now.tic,
                            m.x, m.y, m.angle, m.tic,
                            (((now.x - m.x) as f64).hypot((now.y - m.y) as f64)).round() as i64
                        ));
                    }
                    // Same place, same clock, different reading. The world
                    // agrees and what the agent makes of it does not, which
                    // is a different fault entirely and worth saying so.
                    if m.read != 0 && now.read != m.read {
                        // The world agrees and the reading does not, so the
                        // difference is in what this side DERIVES from it -
                        // the memory of what has been seen, the history line,
                        // what the route makes of where the player stands.
                        // Printed in full, because which LINE differs is the
                        // whole diagnosis and a digest cannot say.
                        return Err(format!(
                            "at decision {i} the replay stands exactly where the search \
                             stood - ({}, {}) facing {} on tic {} - and READS something \
                             different there.\n--- the replay reads ---\n{}\n\
                             --- the search read ---\n{}",
                            now.x, now.y, now.angle, now.tic,
                            env.look(),
                            if m.said.is_empty() { "(not kept this far in)" } else { &m.said }
                        ));
                    }
                }
            }
        }
        let offered = env.inputs();
        let Some(chose) = offered.iter().position(|o| o == want) else {
            return Err(format!(
                "at decision {i} no option on offer sends {want:?}; {} were offered",
                offered.len()
            ));
        };
        // Kept BEFORE the step, because a decision is a question about the
        // state it was asked in and that state is about to stop existing.
        //
        // This is the one place a whole winning trajectory can be turned
        // into training data. The search's own kept decisions are the ones
        // that ADVANCED the archive, which is most of what is worth
        // imitating and is not the same thing as the run that actually won:
        // a verified solution is a complete level played start to finish,
        // and every decision in it is one a policy asked to play the level
        // start to finish will have to make.
        if let Some(into) = keep.as_deref_mut() {
            into.push(Demonstration {
                objective: env.objective(),
                observation: env.look(),
                options: env.options().iter().map(|o| o.text.clone()).collect(),
                action: chose,
            });
        }
        let (_, _, done) = env.step(chose);
        if let Some(f) = env.fault() {
            return Err(format!("the engine faulted at decision {i}: {f}"));
        }
        if done {
            break;
        }
    }
    Ok(env.score(allowed))
}

/// What a candidate solution claims, for the cascade to check.
pub struct Claim {
    pub actions: Vec<String>,
    /// Where the search stood on the way, every few decisions.
    ///
    /// Carried so that a claim which fails to replay says WHERE it parted
    /// company rather than only that it scored less. A verification that can
    /// only report the final number is a verification nobody can act on -
    /// and this is the rung whose failures are the most expensive to
    /// diagnose, because reaching one costs a whole campaign.
    pub marks: Vec<Mark>,
    /// What the SEARCH believed, reached by restoring snapshots.
    pub claimed: crate::report::Score,
    pub claimed_tics: u64,
}

/// Rung one: is this even a claim worth paying to check?
///
/// Free, and it refuses the overwhelming majority. A campaign files thousands
/// of cells and almost none of them finish the level; replaying every one of
/// them from the level start would be the entire budget.
///
/// A FINISHED level is checked, not only a completed category. That is a
/// deliberate widening: a Max is what the campaign is for, but a run that
/// merely got out is a real artifact - it is the first thing the compression
/// phase can be fitted to, and it is the only way to find out whether the
/// verification path works at all before the category is within reach.
/// Which of the two a verified run turned out to be is recorded on the
/// solution rather than decided here.
struct WorthChecking {
    /// The best verified score so far, and the time that went with it. A
    /// claim is worth an episode when it beats that on the category first
    /// and on the clock second - the same ordering the archive keeps its
    /// elites by, so the cascade and the archive cannot disagree about which
    /// of two runs is better.
    best_score: f32,
    best_tics: u64,
}

impl Rung<Claim> for WorthChecking {
    fn name(&self) -> &str {
        "finished the level, and beat the best verified run"
    }
    fn check(&mut self, c: &Claim) -> Verdict {
        if !c.claimed.finished {
            return Verdict::Reject("did not finish the level".into());
        }
        let scored = c.claimed.value();
        if scored > self.best_score + SAME_ACHIEVEMENT {
            return Verdict::Pass;
        }
        if scored < self.best_score - SAME_ACHIEVEMENT {
            return Verdict::Reject(format!(
                "{scored:.3} is behind the verified {:.3}",
                self.best_score
            ));
        }
        if c.claimed_tics >= self.best_tics {
            return Verdict::Reject(format!(
                "{} tics is no better than the verified {}",
                c.claimed_tics, self.best_tics
            ));
        }
        Verdict::Pass
    }
}

/// Run a campaign against the level the environment is configured for.
///
/// `budget` is wall clock, because that is the resource a search actually
/// spends and the only one the allocator can compare its operators on.
pub fn campaign(
    env: &mut DoomEnv,
    seed: u64,
    budget: Duration,
    allowed: u32,
    every: Duration,
    archive: Option<&str>,
    mut propose: Option<&mut Proposer<'_>>,
) -> Result<Found, String> {
    let mut run = Campaign::new(env.slots(), seed, propose.is_some());
    // The archive comes in BEFORE the level is stood up, so the spawn cell is
    // filed against whatever was carried in rather than into an empty map.
    if let Some(path) = archive {
        run.carry_in(path);
    }
    run.seed_start(env, seed, allowed)?;
    if archive.is_some() {
        run.audit(env, allowed);
        // The audit left the engine at the end of its own replayed episode,
        // so the level has to be stood back up before the search resumes.
        run.seed_start(env, seed, allowed)?;
    }
    let level = env.label().unwrap_or_else(|| "the level".into());
    println!(
        "doom: searching {level} for {:.0}s, {} slots, operators: {}",
        budget.as_secs_f64(),
        env.slots(),
        run.operators()
    );

    let began = Instant::now();
    let mut said = Instant::now();
    let mut solved: Vec<Solution> = Vec::new();
    let mut verified_best = (f32::MIN, u64::MAX);

    while began.elapsed() < budget {
        let arm = run.alloc.choose();
        let gain = run.operate(env, arm, allowed, propose.as_deref_mut());
        run.alloc.credit(arm, gain);

        // A completed category is checked the moment it is claimed, not at
        // the end: the archive keeps one elite per niche, so a slower
        // solution found later can replace the record of a faster one that
        // was never written down.
        for claim in std::mem::take(&mut run.claims) {
            // The cascade's cheap rung reads the running best, which moves as
            // solutions are verified. Rebuilt rather than mutated so the bar
            // a claim is judged against is never stale.
            let mut ladder: Cascade<Claim> = Cascade::new();
            ladder.push(Box::new(WorthChecking {
                best_score: verified_best.0,
                best_tics: verified_best.1,
            }));
            if let Verdict::Reject(_) = ladder.admit(&claim) {
                continue;
            }
            let mut played: Vec<Demonstration> = Vec::new();
            match replay_checked(
                env,
                seed,
                &claim.actions,
                allowed,
                &claim.marks,
                Some(&mut played),
            ) {
                Ok(got) if got.finished => {
                    let l = env.level_counts();
                    let s = Solution {
                        level: level.clone(),
                        skill: env.skill(),
                        actions: claim.actions.clone(),
                        tics: env.cost(),
                        kills: l.0,
                        total_kills: l.1,
                        secrets: l.2,
                        total_secrets: l.3,
                        score: got.value(),
                        uvmax: got.is_uvmax(),
                    };
                    println!(
                        "doom: VERIFIED {} on {level}: {}/{} kills, {}/{} secrets, \
                         {} tics ({}), {} decisions, replayed from the level's own start",
                        if got.is_uvmax() { "a UV-MAX" } else { "a finished level" },
                        s.kills, s.total_kills, s.secrets, s.total_secrets, s.tics, s.clock(),
                        s.actions.len()
                    );
                    verified_best = (s.score, s.tics);
                    solved.push(s);
                    // The complete winning run, as decisions. What the
                    // compression phase is actually supposed to be fitted
                    // to: not the fragments that advanced a search, but a
                    // level played through. Through `keep` like everything
                    // else, so the training set stays bounded - and last,
                    // so that if the cap bites it is the search's older
                    // fragments that go rather than the run that won.
                    for d in played.drain(..) {
                        run.keep(d);
                    }
                }
                // A claim the replay does not reproduce is a DEFECT, not a
                // near miss: the search reached that state by restoring
                // snapshots and the action list is supposed to be another way
                // of reaching the same one. Said out loud, because a campaign
                // whose claims do not replay is producing artifacts nobody
                // can use and every other number it reports is suspect.
                Ok(got) => println!(
                    "doom: a claimed finish did NOT replay - from the level's own start \
                     the same actions scored {:.3} and did not finish. The search reached \
                     that state by restoring snapshots; the action list does not reach it.",
                    got.value()
                ),
                Err(e) => println!("doom: a claimed finish could not be replayed: {e}"),
            }
            // The replay left the engine at the end of its own episode, and
            // every snapshot the archive points at is still valid - they are
            // the engine's, not the episode's - so the search carries on.
        }

        if said.elapsed() >= every {
            said = Instant::now();
            run.say(&level, began.elapsed(), &solved);
        }
    }

    run.say(&level, began.elapsed(), &solved);
    if let Some(path) = archive {
        run.carry_out(path);
    }
    for spent in run.alloc.report() {
        println!(
            "    {:<16} {:>5} draws, {:>7.1}s, {:.3} gain/s",
            spent.name, spent.draws, spent.seconds, spent.rate
        );
    }
    if run.refused > 0 {
        println!(
            "    {} resumes were REFUSED - that part of the archive is not restorable",
            run.refused
        );
    }
    let best = run.archive.best();
    let cells = run.archive.len();
    let (reached, cost) = (
        best.map(|e| e.worth.reached).unwrap_or(0.0),
        best.map(|e| e.worth.cost).unwrap_or(0),
    );
    println!("    {} decisions kept for training", run.learned.len());
    Ok(Found {
        cells,
        best: reached,
        best_tics: cost,
        solved,
        learned: run.learned,
    })
}
