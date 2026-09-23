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
}

const OPERATORS: [Operator; 6] = [
    // Blind, and blind in a level full of things that shoot back is mostly
    // dead - but it is the only operator that can produce an action no
    // teacher and no policy would ever pick.
    Operator { name: "wander", walk: 60, guided: 0.0, from_best: false, leave: 0.0, sweep: false, press: 0.0 },
    // Starts from somewhere plausible and wanders off it.
    Operator { name: "probe", walk: 60, guided: 0.85, from_best: false, leave: 0.0, sweep: false, press: 0.0 },
    // A long stretch of real play from a drawn cell. Long enough to finish a
    // firefight, clear a room and walk into the next one.
    Operator { name: "commit", walk: 400, guided: 1.0, from_best: false, leave: 0.0, sweep: false, press: 0.0 },
    // The same, from the furthest-along cell there is: pushing the front of
    // the search forward rather than filling in behind it.
    Operator { name: "chase", walk: 400, guided: 1.0, from_best: true, leave: 0.0, sweep: false, press: 0.0 },
    // Walk about pressing on everything. Half the decisions are a push, the
    // rest are the teacher moving on, which together is a player running
    // their shoulder along the walls of a room - the only way a secret is
    // ever found by someone who has not been told where it is.
    Operator { name: "frisk", walk: 200, guided: 1.0, from_best: false, leave: 0.0, sweep: true, press: 0.5 },
    // Play on from the furthest-along cell there is, and take the way out
    // when one is offered. The only operator that can produce a FINISHED
    // level, which is what the verification rung exists to check and what
    // the compression phase is supposed to be fitted to.
    Operator { name: "leave", walk: 400, guided: 1.0, from_best: true, leave: 0.4, sweep: false, press: 0.0 },
];

/// What a search campaign found, and what it cost.
pub struct Found {
    pub cells: usize,
    pub best: f32,
    pub best_tics: u64,
    /// Verified UV-Max runs: replayed from the level's own start and confirmed.
    pub solved: Vec<Solution>,
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
    /// Indices into the campaign's vocabulary, from the level's own start.
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
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Mark {
    pub at: u32,
    pub x: i32,
    pub y: i32,
    pub angle: i32,
    pub tic: i64,
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
    /// Completed categories noticed during a walk, waiting to be replayed.
    ///
    /// Queued rather than verified on the spot because verifying RESTARTS the
    /// level, and doing that in the middle of a walk would throw away the rest
    /// of the walk's budget.
    claims: Vec<Claim>,
}

impl Campaign {
    pub fn new(slots: usize, seed: u64) -> Campaign {
        Campaign {
            archive: Archive::new(slots, SAME_ACHIEVEMENT),
            alloc: Allocator::new(&OPERATORS.map(|o| o.name)),
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
            claims: Vec::new(),
        }
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
        let kept = Kept { vocab: self.vocab.clone(), archive: self.archive.clone() };
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
            match replay_checked(env, self.seed, &actions, allowed, &trail.marks) {
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
    fn file(
        &mut self,
        env: &mut DoomEnv,
        cell: Niche,
        worth: Worth,
        trail: Trail,
    ) -> Admission {
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
    fn go_to(&mut self, env: &mut DoomEnv, at: &Niche, slot: usize, allowed: u32) -> bool {
        if self.live.contains(at) {
            if env.resume_from(slot).is_some() {
                return true;
            }
            self.refused += 1;
            self.live.remove(at);
            self.archive.remove(at);
            return false;
        }
        // Read back off disk. The only way to stand where it stands is to
        // walk there again from the level's own start.
        if self.rehydrations == 0 {
            return false;
        }
        self.rehydrations -= 1;
        let Some(actions) = self.archive.get(at).and_then(|e| self.spell(&e.what)) else {
            self.archive.remove(at);
            return false;
        };
        if replay(env, self.seed, &actions, allowed).is_err() || !env.hold_at(slot) {
            self.archive.remove(at);
            return false;
        }
        self.live.insert(at.clone());
        true
    }

    /// Run one operator once and report what it bought.
    ///
    /// Every operator has the same shape - go back somewhere, act for a
    /// while, file what was reached - and differs only in WHERE it returns to
    /// and HOW it chooses each action. Keeping that shape in one place is
    /// what stops three operators drifting into three slightly different
    /// definitions of what a walk is.
    fn operate(&mut self, env: &mut DoomEnv, arm: usize, allowed: u32) -> Gain {
        let began = Instant::now();
        let mut gain = Gain::default();
        let op = &OPERATORS[arm];

        let picked = if op.from_best {
            self.archive.best().map(|e| (e.niche.clone(), e.slot))
        } else {
            self.pick_reachable()
        };
        let Some((from, slot)) = picked else {
            gain.seconds = began.elapsed().as_secs_f64();
            return gain;
        };
        if !self.go_to(env, &from, slot, allowed) {
            gain.seconds = began.elapsed().as_secs_f64();
            return gain;
        }
        self.restores += 1;

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
            let chose = self.choose(env, op, &options, last.as_deref());
            last = Some(options[chose].clone());
            steps.push(self.intern(&options[chose]));
            let (_, _, done) = env.step(chose);
            self.steps += 1;
            if env.fault().is_some() {
                break;
            }
            let here = prefix.len() + steps.len();
            if here % WITNESS_EVERY == 0 {
                if let Some(m) = env.mark(here as u32) {
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
                let trail = Trail { steps: whole, marks: seen };
                // What the archive's best was BEFORE this admission, so a
                // cell that advances the frontier is credited against the old
                // frontier rather than against itself.
                let top = self.archive.best().map(|e| e.worth.reached).unwrap_or(0.0);
                match self.file(env, cell, worth, trail) {
                    Admission::Fresh => gain.admitted(true, worth.reached, top),
                    Admission::Improved => gain.admitted(false, worth.reached, top),
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
                    if let Some(actions) = self.spell(&Trail { steps: whole, marks: Vec::new() }) {
                        self.claims.push(Claim {
                            actions,
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
        let cover = self
            .archive
            .coverage()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("/");
        let best = self.archive.best();
        println!(
            "  {level} {:>5.0}s: {} cells (cover {}), best {:.3} in {} tics,              {} steps, {} resumes, {} verified",
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
    ) -> usize {
        if op.leave > 0.0 && self.rng.next_f32() < op.leave {
            if let Some(i) = env.exit_option().filter(|i| *i < options.len()) {
                return i;
            }
        }
        if op.press > 0.0 && self.rng.next_f32() < op.press {
            // Press, then slide along and press again. `last` is what the
            // walk just did, so alternating on it is what turns a repeated
            // push on one spot into a sweep of the wall.
            let pressed_last = last.is_some_and(|l| Some(l) == env.use_text().as_deref());
            if op.sweep && pressed_last {
                let sides = env.sidestep_options();
                if let Some(i) = sides.first().filter(|i| **i < options.len()) {
                    return *i;
                }
            }
            if let Some(i) = env.use_option().filter(|i| *i < options.len()) {
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

/// Replay an action list from the level's own start and report what actually
/// happened.
///
/// The expensive rung of the cascade, and the only one whose answer is worth
/// anything to a human: everything above it reasons about states reached by
/// restoring snapshots, which is not a run anybody could play. This is a
/// whole episode against the engine, from the level's front door, taking only
/// the actions in the list.
///
/// An option the list names and the game does not offer is a HARD failure,
/// never a skip. The list is a claim about what the level does; a replay that
/// silently walks past a missing option is one that verifies a different
/// trajectory and reports success.
pub fn replay(
    env: &mut DoomEnv,
    seed: u64,
    actions: &[String],
    allowed: u32,
) -> Result<crate::report::Score, String> {
    replay_checked(env, seed, actions, allowed, &[])
}

/// [`replay`], checking the trail's own witnesses as it goes.
///
/// Reports the FIRST decision at which the replay stood somewhere the search
/// did not, which is the only useful thing to know about a trail that does
/// not reproduce. Without it the failure arrives as a missing option sentence
/// three hundred decisions in, which says the two runs disagree and nothing
/// whatever about where they began to.
pub fn replay_checked(
    env: &mut DoomEnv,
    seed: u64,
    actions: &[String],
    allowed: u32,
    marks: &[Mark],
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
                if let Some(now) = env.mark(i as u32) {
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
                }
            }
        }
        let options = env.actions();
        let Some(chose) = options.iter().position(|o| o == want) else {
            return Err(format!(
                "at decision {i} the game did not offer {want:?}; it offered {} options",
                options.len()
            ));
        };
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
) -> Result<Found, String> {
    let mut run = Campaign::new(env.slots(), seed);
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
        OPERATORS.map(|o| o.name).join(", ")
    );


    let began = Instant::now();
    let mut said = Instant::now();
    let mut solved: Vec<Solution> = Vec::new();
    let mut verified_best = (f32::MIN, u64::MAX);

    while began.elapsed() < budget {
        let arm = run.alloc.choose();
        let gain = run.operate(env, arm, allowed);
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
            match replay(env, seed, &claim.actions, allowed) {
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
    Ok(Found {
        cells: run.archive.len(),
        best: best.map(|e| e.worth.reached).unwrap_or(0.0),
        best_tics: best.map(|e| e.worth.cost).unwrap_or(0),
        solved,
    })
}
