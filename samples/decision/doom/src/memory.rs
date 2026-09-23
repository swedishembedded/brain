// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! What the player saw a moment ago, and where it was.
//!
//! Under fair play the engine sends only what is in line of sight right now.
//! Everything else is dropped, so turning away from a medikit deletes it from
//! the agent's world: the option to go and get it disappears in the same
//! decision that the player stops looking at it, and the only way anything is
//! ever picked up is by walking into it.
//!
//! That is not how anyone plays. You look left, see a medikit, look right to
//! check the corridor, and the medikit is still behind your left shoulder.
//!
//! This is not extra information. Nothing enters here that was not in an
//! observation the agent was already shown; it is the same information, kept
//! for a while instead of thrown away one decision later.
//!
//! ## What goes stale, and what does not
//!
//! A remembered thing is held as a POSITION ON THE MAP and reported as a
//! bearing and a distance from wherever the player is standing now. That
//! makes walking away from it correct rather than stale - it becomes "220
//! units behind you", which is true - and it is also why the map coordinates
//! never leave this module. A policy told where it is in map units would be
//! memorising a map, which is the one thing the generated levels exist to
//! make worthless.
//!
//! What genuinely goes stale is a MONSTER, because a monster moves. An item
//! does not. So the two are forgotten on completely different terms:
//!
//! - a monster is forgotten after [`MONSTER_MEMORY`] decisions out of sight,
//!   because by then it could be anywhere;
//! - an item is remembered until there is evidence it has gone - the player
//!   reached the spot and it was not there, which also covers having picked
//!   it up.
//!
//! Swedish Embedded AB builds perception and state-estimation for systems that
//! act on partial views of the world, where what was true a second ago still
//! matters and knowing how long it stays true is the hard part. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::obs::{Player, State, Thing};

/// Decisions a monster stays remembered once it is out of sight. Ten of them
/// is six to ten seconds of game time, by which point a monster that was
/// coming for you is somewhere else entirely.
pub const MONSTER_MEMORY: u32 = 10;

/// How close counts as having got there.
pub const REACHED: i32 = 56;

/// How far off the nose the spot has to be to count as having been LOOKED at.
///
/// Being near something is not evidence about it. A player standing on the
/// other side of a wall from a medikit is within arm's length of it and has
/// learned nothing; so is one who walked past it and turned away. What
/// unseats the memory is having looked where it was and not seen it, and
/// looking is a cone - about ninety degrees for a player at the controls,
/// which is forty-five either side.
pub const LOOKED: i32 = 45;

/// How many remembered things are worth saying. Past a handful this is no
/// longer memory, it is a map.
pub const MAX_RECALLED: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Threat,
    Hazard,
    Pickup,
}

impl Class {
    /// Whether this kind of thing stays where it was left.
    fn stays_put(self) -> bool {
        !matches!(self, Class::Threat)
    }
}

/// The walk to a remembered thing, round whatever is between here and there.
///
/// A bearing and a distance again, so nothing about the map reaches a policy -
/// but they are the bearing to set off on and the length of the WALK, which in
/// a corridor is a different direction and a longer way than the straight line
/// to the same place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Path {
    pub bearing: i32,
    /// How far the walk is. Longer than the straight line whenever there are
    /// corners in it, and that difference is what decides whether going back
    /// is worth the trip.
    pub distance: i32,
    /// Open floor that way, so an option can say whether setting off is a
    /// step or a wall.
    pub clearance: i32,
    /// How far the FIRST leg is. The bearing points at a waypoint a few cells
    /// along rather than at the far end, so walking the whole path on this
    /// heading walks past the corner it turns at.
    pub step: i32,
}

/// Something out of sight, placed from where the player is standing now.
#[derive(Clone, Debug)]
pub struct Recalled {
    /// The map-object id this memory is of.
    ///
    /// Not read when the memory is rendered - the observation names things by
    /// WHAT they are and where, never by id, because an id is a handle into
    /// one level's object table and a policy given one would be memorising a
    /// map. It is carried so that a caller resolving a recalled thing back to
    /// the live object (asking the engine for a route to it) has the handle
    /// to do it with.
    #[allow(dead_code)]
    pub id: i64,
    pub kind: String,
    pub class: Class,
    /// Degrees relative to the player's facing, negative to the left - the
    /// same convention the engine uses for what IS in sight.
    pub bearing: i32,
    pub distance: i32,
    /// What it had when last seen. The comparison against the best it was
    /// ever seen with is made inside the memory (that is what "the one you
    /// have been hitting" is derived from), so the raw figure does not reach
    /// the observation.
    #[allow(dead_code)]
    pub health: Option<i32>,
    /// Decisions since it was last seen. Staleness is applied inside the
    /// memory - a monster is forgotten after ten decisions out of sight -
    /// rather than reported, because "I last saw it nine decisions ago" is
    /// not something a player knows as a number.
    #[allow(dead_code)]
    pub ago: u32,
    /// The way back to it, when someone has asked the engine. `None` when
    /// nobody asked, or when there is no walkable way there.
    pub path: Option<Path>,
}

#[derive(Clone, Debug)]
struct Held {
    id: i64,
    kind: String,
    class: Class,
    x: f64,
    y: f64,
    health: Option<i32>,
    /// The most health it has ever been seen with. A monster below its own
    /// best is one that has been shot, which is how a player knows which of
    /// two identical sergeants they have been fighting.
    best_health: Option<i32>,
    ago: u32,
    path: Option<Path>,
}

/// Somewhere a live monster was seen and has not been accounted for since.
///
/// A SECOND and much coarser memory than [`Held`], and the reason it exists
/// is a distinction the sharp one gets right for the wrong purpose. A
/// monster's exact position goes stale in ten decisions, correctly: it has
/// moved, so the way to where it was is the way to where it is not, and
/// aiming at the remembered spot aims at nothing.
///
/// But "there was a monster over there" does NOT go stale. A monster in a
/// room is still in that room; even one that wandered is somewhere in the
/// region it was seen in. For "reach the exit" that fact is worthless, which
/// is why nothing here ever kept it. For UV-Max it is most of the task: the
/// last few monsters of a level are the ones that were seen once, from a
/// doorway, and never gone back for.
///
/// So this is kept at ROOM scale rather than at the monster's spot, and it is
/// forgotten on the same evidence an item is - the player went there and
/// looked - rather than on a timer.
#[derive(Clone, Debug)]
struct Haunt {
    kind: String,
    /// The centre of the region it was seen in, in map units.
    x: f64,
    y: f64,
    path: Option<Path>,
}

/// Side of the region a haunt is filed under, in map units.
///
/// Room scale, not monster scale. Two sightings of the same monster from
/// different doorways are one piece of unfinished business, and filing them
/// separately would send the search to the same room twice.
const HAUNT_UNITS: f64 = 256.0;

/// Side of the region wall-testing is tallied over, in map units. Room
/// scale, the same as a haunt's, because "go and search that room" is the
/// grain at which the question is worth asking.
const SWEEP_UNITS: f64 = 256.0;

/// Distinct walls pushed on inside one region before it counts as searched.
///
/// A number, not a proof: nothing here knows how many walls a room has, and
/// nothing can without being handed the level. What it buys is that the
/// frontier DRAINS - a room pushed on this many times stops being offered,
/// so the search moves on instead of sweeping the first room forever.
const SWEPT: u32 = 12;

/// How finely one push is remembered, in map units. A DOOM linedef is 8 to
/// 128 units long; at 64 the ledger distinguishes the pieces of wall a
/// sidestep actually moves between without filing every step as new.
const PRESS_UNITS: f64 = 64.0;

/// Facings one spot is divided into, so that standing in a corner and
/// turning is testing two walls rather than pushing one twice.
const FACINGS: i32 = 8;

/// How close a wall has to be for pushing on it to reach it, in map units.
///
/// DOOM's own `USERANGE`, shared with the option builder rather than spelled
/// twice. It matters because a push that reaches nothing tests nothing:
/// pressing use in the middle of a room is a decision spent, and counting it
/// as a wall searched retires rooms that were never searched at all - it
/// tells the archive a run has been looking when it has not.
const IN_REACH: i32 = crate::action::USE_RANGE;

/// How much rarer than the commonest wall face a face has to be before it
/// counts as looking out of place.
///
/// A level is built from a handful of textures repeated over and over, so
/// the ones a run keeps seeing are the ordinary ones. Eight times is a
/// judgement, not a measurement: low enough that a genuinely unusual face
/// stands out after a room or two, high enough that the second-commonest
/// wall in a level does not.
const ODD: u32 = 8;

/// How much wall a run has to have looked at before it is entitled to call
/// any of it unusual. Everything is unusual when you have seen three things.
///
/// Counted as LOOKS, not as distinct faces. A corridor built from one
/// repeated texture offers a single distinct face however far you walk down
/// it, so a run there would never form an opinion at all - and a run that
/// has stared at the same wall forty times knows perfectly well what
/// ordinary looks like around here.
const ENOUGH_TO_JUDGE: u32 = 40;

/// A region of the level and how much of its walls this run has pushed on.
///
/// The counterpart of [`Haunt`] for secrets, and a separate ledger for a
/// concrete reason: a haunt retires on EVIDENCE (went there, looked, saw
/// nothing), and there is no such evidence about a wall. A wall that has
/// not been pushed looks exactly like a wall that has. So this one retires
/// on WORK DONE instead, which is the only thing the agent can actually
/// observe about its own searching.
#[derive(Clone)]
struct Sweep {
    x: f64,
    y: f64,
    /// Distinct walls pushed on here. Distinct, because pressing use two
    /// hundred times against one spot tests one wall.
    tried: u32,
    path: Option<Path>,
}

#[derive(Clone, Default)]
pub struct Memory {
    held: Vec<Held>,
    /// Places live monsters were seen, at room scale. See [`Haunt`].
    haunts: Vec<Haunt>,
    /// Regions stood in and how thoroughly searched. See [`Sweep`].
    swept: Vec<Sweep>,
    /// Wall faces this run has LOOKED at, and how often each.
    ///
    /// Keyed by what is drawn and where it is drawn - a texture together with
    /// its alignment - so that both cues a player actually uses land in one
    /// tally. An unusual texture is a rare key; an ordinary texture shoved
    /// out of alignment with its neighbours is also a rare key, and that
    /// second kind is most of how DOOM marks a secret door.
    ///
    /// The texture is stored as a hash rather than a name: nothing here ever
    /// needs to know WHICH texture it is, only whether it has seen this one
    /// before, and a collision would merely mean two textures look alike to
    /// the agent. That keeps this cheap to clone into a snapshot.
    looks: std::collections::HashMap<(u32, i32), u32>,
    /// The wall pushed on this decision, waiting to hear what came of it.
    ///
    /// A push and its outcome arrive at different moments: where the player
    /// was standing is only true BEFORE the step, and what the push came to
    /// is only known after it. Holding the spot until the answer lands is
    /// what lets the ledger record the two together.
    pending: Option<((i32, i32), u8)>,
    /// Walls this run has actually pushed ON - as opposed to spots it has
    /// pushed AT, which `pressed` counts and which include the pushes that
    /// met nothing.
    ///
    /// Kept apart because the two answer different questions. "Have I tried
    /// here?" must say yes either way or the sweep loops; "have I searched
    /// this room?" must say no when the pushes touched nothing, or a room
    /// retires on the strength of the agent pressing air in it.
    walls: u32,
    /// Doors, switches and walls this run has actually made move.
    ///
    /// Capability, in the same sense as a key or a shotgun: a run that has
    /// opened something can reach ground a run that has not, so the two are
    /// not in the same situation even standing in the same place. An axis of
    /// the archive's name for a situation - see `DoomEnv::cell`.
    opened: u32,
    /// Which walls have been pushed on: a bit per facing, per spot.
    ///
    /// A mask rather than a set of `(spot, facing)` because this is cloned
    /// into every engine snapshot the search holds, and a search holds
    /// thousands. Eight facings fit in a byte, so a run that has pushed on
    /// a thousand walls costs a few kilobytes a snapshot instead of tens.
    pressed: std::collections::HashMap<(i32, i32), u8>,
    /// Whether something was picked up in the observation being folded in.
    took: bool,
}

/// Where a thing at this bearing and distance is standing.
fn place(p: &Player, bearing: i32, distance: i32) -> Option<(f64, f64)> {
    let (x, y, a) = (p.x?, p.y?, p.angle?);
    let world = ((a + bearing) as f64).to_radians();
    Some((
        x as f64 + distance as f64 * world.cos(),
        y as f64 + distance as f64 * world.sin(),
    ))
}

/// Which way and how far that place is from here.
fn from_here(p: &Player, x: f64, y: f64) -> Option<(i32, i32)> {
    let (px, py, a) = (p.x?, p.y?, p.angle?);
    let dx = x - px as f64;
    let dy = y - py as f64;
    let world = dy.atan2(dx).to_degrees();
    let mut rel = world - a as f64;
    while rel > 180.0 {
        rel -= 360.0;
    }
    while rel <= -180.0 {
        rel += 360.0;
    }
    Some((rel.round() as i32, dx.hypot(dy).round() as i32))
}

impl Memory {
    pub fn new() -> Memory {
        Memory::default()
    }

    /// A new episode is a new world.
    ///
    /// ALL of it, and the exhaustive destructure is the point: this cleared
    /// only what was in sight, so every episode after the first in a process
    /// began holding the last one's monsters, the last one's rooms and the
    /// last one's pushed-on walls. An agent that remembers a level it has
    /// not played yet is the smaller half of that. The larger half is that a
    /// trajectory replayed after another one starts with the other's memory,
    /// so the same actions are offered different options and the replay
    /// diverges - which is a trajectory that cannot be verified, for a
    /// reason nothing in the trajectory could reveal.
    ///
    /// No `..`: adding a ledger to this type fails to compile until somebody
    /// says whether a new episode inherits it.
    pub fn clear(&mut self) {
        let Memory { held, haunts, swept, pressed, looks, pending, walls, opened, took } = self;
        held.clear();
        haunts.clear();
        swept.clear();
        pressed.clear();
        looks.clear();
        *pending = None;
        *walls = 0;
        *opened = 0;
        *took = false;
    }

    /// Fold an observation in: refresh what is in view, age what is not, and
    /// forget what there is evidence has gone.
    pub fn observe(&mut self, state: &State) {
        let p = &state.player;
        // Something was collected this step, so whatever is being stood on is
        // the thing that went. The engine says one was taken without saying
        // which, and the nearest remembered one is the answer that is right
        // whenever the question is asked at all.
        self.took = state.events.iter().any(|e| e.kind == "item");
        if p.x.is_none() || p.y.is_none() || p.angle.is_none() {
            return;
        }
        for h in self.held.iter_mut() {
            h.ago += 1;
        }
        let seen = [
            (Class::Threat, &state.threats),
            (Class::Hazard, &state.hazards),
            (Class::Pickup, &state.pickups),
        ];
        for (class, things) in seen {
            for t in things.iter() {
                self.note(p, class, t);
                if class == Class::Threat && t.health.is_none_or(|h| h > 0) {
                    self.haunt(p, t);
                }
            }
        }
        self.forget(p);
        self.settle(p);
        self.stood_in(p);
        self.look_at_the_wall(state);
        // What the last push came to, from the engine rather than guessed
        // at. See `Memory::pushed`.
        if let Some(came_to) = state
            .events
            .iter()
            .find(|e| e.kind == "push")
            .map(|e| e.what.clone().unwrap_or_default())
        {
            self.pushed(state, &came_to);
        }
    }

    /// File the region a live monster was seen in as unfinished business.
    fn haunt(&mut self, p: &Player, t: &Thing) {
        let Some((x, y)) = place(p, t.bearing, t.distance) else {
            return;
        };
        let region = |a: f64, b: f64| (a / HAUNT_UNITS).floor() == (b / HAUNT_UNITS).floor();
        if let Some(h) = self
            .haunts
            .iter_mut()
            .find(|h| region(h.x, x) && region(h.y, y))
        {
            // The freshest sighting wins the position, so a monster that is
            // walking toward the player does not leave the marker behind it.
            h.x = x;
            h.y = y;
            h.kind = t.kind.clone();
            return;
        }
        self.haunts.push(Haunt {
            kind: t.kind.clone(),
            x,
            y,
            path: None,
        });
    }

    /// Drop the regions the player has been to and looked at.
    ///
    /// The same evidence rule an item is forgotten on, and deliberately so:
    /// being NEAR a place teaches nothing (a player on the far side of a wall
    /// is within arm's length), having gone there and looked teaches that
    /// whatever was there is not there now. A monster that is still about
    /// will be seen again on the way, and files a fresh region when it is.
    fn settle(&mut self, p: &Player) {
        self.haunts.retain(|h| match from_here(p, h.x, h.y) {
            Some((bearing, d)) if d <= HAUNT_UNITS as i32 => bearing.abs() > LOOKED,
            _ => true,
        });
    }

    /// Which piece of wall the player is standing in front of: a spot and a
    /// facing, both coarse enough that shuffling does not invent new walls.
    fn wall(p: &Player) -> Option<((i32, i32), u8)> {
        let (x, y, a) = (p.x?, p.y?, p.angle?);
        Some(Memory::wall_at(x, y, a))
    }

    /// The same, for a place and a facing given rather than stood in.
    fn wall_at(x: i32, y: i32, angle: i32) -> ((i32, i32), u8) {
        let spot = (
            (x as f64 / PRESS_UNITS).floor() as i32,
            (y as f64 / PRESS_UNITS).floor() as i32,
        );
        (spot, 1u8 << (angle.rem_euclid(360) * FACINGS / 360))
    }

    /// File the region the player is standing in as somewhere whose walls
    /// could be searched.
    ///
    /// Unconditionally, including regions already filed - a region is only
    /// ever added here, never retired by being walked through, because
    /// walking through a room is not evidence about its walls.
    fn stood_in(&mut self, p: &Player) {
        let (Some(x), Some(y)) = (p.x, p.y) else {
            return;
        };
        let (x, y) = (x as f64, y as f64);
        let region = |a: f64, b: f64| (a / SWEEP_UNITS).floor() == (b / SWEEP_UNITS).floor();
        if self.swept.iter().any(|s| region(s.x, x) && region(s.y, y)) {
            return;
        }
        self.swept.push(Sweep {
            x,
            y,
            tried: 0,
            path: None,
        });
    }

    /// One wall face, as the pair "what is drawn and where".
    fn face(w: &crate::obs::Wall) -> (u32, i32) {
        // FNV-1a over the three surfaces together: a two-sided line shows its
        // upper or lower where a solid one shows its middle, and which of the
        // three is doing the showing is not something worth distinguishing.
        let mut h: u32 = 0x811c_9dc5;
        for b in w.texture.bytes().chain(w.above.bytes()).chain(w.below.bytes()) {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        (h, w.offset)
    }

    /// Note what the player is looking at, so that "unlike the others" is
    /// something this run can work out for itself later.
    fn look_at_the_wall(&mut self, state: &State) {
        if let Some(w) = &state.facing_wall {
            *self.looks.entry(Memory::face(w)).or_insert(0) += 1;
        }
    }

    /// Does the wall in front look unlike the ones this run has been seeing?
    ///
    /// The cue a player actually uses, and the whole reason the wall face is
    /// in the observation at all. A level is built from a few textures
    /// repeated everywhere; a wall that is not one of them, or is one of them
    /// visibly out of step, is where a player pushes.
    ///
    /// It is a guess and it is allowed to be wrong. Nothing here knows what
    /// is behind the wall, and plenty of odd-looking walls are just walls -
    /// what this buys is an ORDER to search in, not an answer. The sweep
    /// still tests everything it can reach.
    pub fn odd_wall(&self, state: &State) -> bool {
        let Some(w) = &state.facing_wall else {
            return false;
        };
        if self.looks.values().copied().sum::<u32>() < ENOUGH_TO_JUDGE {
            return false;
        }
        let seen = self.looks.get(&Memory::face(w)).copied().unwrap_or(0);
        let usual = self.looks.values().copied().max().unwrap_or(0);
        seen.saturating_mul(ODD) <= usual
    }

    /// Is there a wall close enough in front of the player for a push to
    /// reach it? See [`IN_REACH`].
    pub fn wall_in_reach(state: &State) -> bool {
        state.clearance.ahead <= IN_REACH
    }

    /// Has the wall in front of the player already been pushed on?
    ///
    /// What turns pressing use from a coin flip into a search: a sweep that
    /// can tell moves along to a wall it has not tested, and one that cannot
    /// re-tests the wall it is standing at for as long as its budget lasts.
    pub fn pressed_here(&self, state: &State) -> bool {
        let p = &state.player;
        match (p.x, p.y, p.angle) {
            (Some(x), Some(y), Some(a)) => self.pressed_at(x, y, a),
            _ => false,
        }
    }

    /// The same question about a place and a facing the player is not in
    /// yet - so that "turn and push on that" can be offered only when there
    /// is something there this run has not already pushed on.
    pub fn pressed_at(&self, x: i32, y: i32, angle: i32) -> bool {
        let (spot, bit) = Memory::wall_at(x, y, angle);
        self.pressed.get(&spot).is_some_and(|m| m & bit != 0)
    }

    /// Which way to turn to face a wall this run has not pushed on, within
    /// reach of a push once turned.
    ///
    /// The half of searching for a hidden door that was missing. `use`
    /// reaches only what the player FACES, and a player walks corridors
    /// facing along them - so the walls are beside them, an arm's length
    /// away, the whole time and never in front. Measured with only the
    /// facing half: nineteen walls tested in seven minutes of campaign.
    ///
    /// Nothing here is level knowledge. The clearance readings are the
    /// agent's own, the ledger is its own, and neither says a secret is
    /// there - only that there is a wall it has not tried.
    pub fn untried_walls(&self, state: &State) -> Vec<i32> {
        let c = &state.clearance;
        let p = &state.player;
        let (Some(x), Some(y), Some(a)) = (p.x, p.y, p.angle) else {
            return Vec::new();
        };
        // Already facing one: pushing is the act, not turning.
        if c.ahead <= IN_REACH {
            return Vec::new();
        }
        [(90, c.left), (-90, c.right), (180, c.behind)]
            .into_iter()
            .filter(|(_, gap)| *gap <= IN_REACH)
            .map(|(bearing, _)| bearing)
            .filter(|bearing| !self.pressed_at(x, y, a + bearing))
            .collect()
    }

    /// Record that the wall in front of the player has been pushed on.
    ///
    /// Only a push that REACHED a wall counts. DOOM's use range is 64 units,
    /// so a push made in the middle of a room touches nothing, and counting
    /// it retires the room as searched on the strength of the agent having
    /// walked about in it pressing air. Measured before this: 174 walls
    /// "tested" on E1M1 and not one secret found.
    ///
    /// Only a wall never pushed before counts toward its region's tally, so
    /// a region retires when its walls have been SEARCHED rather than when
    /// the use key has been pressed enough times.
    pub fn press(&mut self, state: &State) {
        self.pending = Memory::wall(&state.player);
    }

    /// Fold in what the engine said the push came to.
    ///
    /// The clearance reading was only ever a GUESS at whether a push would
    /// reach anything: it is measured from the player's centre along their
    /// facing, and `P_UseLines` traces its own line and can miss where the
    /// probe hit. The engine knows which happened and the player hears it,
    /// so the ledger takes the answer instead of the guess. Measured over
    /// one short campaign: of 56 pushes the agent made, 21 reached nothing
    /// at all, and every one of those was being filed as a wall tested.
    fn pushed(&mut self, state: &State, came_to: &str) {
        let Some((spot, bit)) = self.pending.take() else {
            return;
        };
        if came_to == "worked" {
            self.opened += 1;
        }
        // TRIED and TESTED are different, and conflating them cost a whole
        // measured campaign. A push that reached nothing tested no wall - it
        // must not retire a room as searched - but it is still a thing this
        // run has now tried from this spot and facing, and forgetting that
        // makes the spot eligible forever: `pressed_here` stays false, so
        // the sweep pushes there again, and `untried_walls` keeps offering
        // the turn to it. Measured with only the first half of this rule, a
        // 420-second campaign tested seven walls against the baseline's
        // thirty-eight and found no secret where the baseline found one.
        let mask = self.pressed.entry(spot).or_insert(0);
        if *mask & bit != 0 {
            return;
        }
        *mask |= bit;
        if came_to == "nothing there" {
            return;
        }
        self.walls += 1;
        let p = &state.player;
        self.stood_in(p);
        let (Some(x), Some(y)) = (p.x, p.y) else {
            return;
        };
        let (x, y) = (x as f64, y as f64);
        let region = |a: f64, b: f64| (a / SWEEP_UNITS).floor() == (b / SWEEP_UNITS).floor();
        if let Some(s) = self
            .swept
            .iter_mut()
            .find(|s| region(s.x, x) && region(s.y, y))
        {
            s.tried += 1;
        }
    }

    /// How many things this run has made move by pushing on them.
    pub fn opened(&self) -> u32 {
        self.opened
    }

    /// Distinct walls this run has pushed on.
    ///
    /// The archive's measure of how thoroughly a trajectory has SEARCHED, as
    /// opposed to how far it has got. Two runs standing in the same room
    /// having killed the same monsters are not in the same situation when
    /// one of them has tested forty walls, and an archive that cannot see
    /// the difference keeps whichever arrived sooner - which is exactly the
    /// one that did no searching.
    pub fn tested(&self) -> usize {
        self.walls as usize
    }

    /// The nearest regions stood in whose walls have not been searched, in
    /// map units, for whoever can ask the engine the way. The index stands
    /// in for an id, as it does for a haunt, because a region is a place
    /// rather than a thing.
    ///
    /// NEAREST, and no more of them than an option list can hold. This
    /// ledger grows with every room the player walks into and never shrinks
    /// except by being searched, so asking the engine to route to all of it
    /// costs more every decision - measured, routing the whole frontier cut
    /// the search's throughput roughly in half, which is a far larger loss
    /// than any secret it could have found. The far ones are also the ones
    /// `unfrisked` would drop anyway.
    pub fn unswept(&self, state: &State) -> Vec<(i64, f64, f64)> {
        let p = &state.player;
        let mut open: Vec<(i64, f64, f64, i32)> = self
            .swept
            .iter()
            .enumerate()
            .filter(|(_, s)| s.tried < SWEPT)
            .filter_map(|(i, s)| {
                let (_, distance) = from_here(p, s.x, s.y)?;
                (distance >= SWEEP_UNITS as i32).then_some((i as i64, s.x, s.y, distance))
            })
            .collect();
        open.sort_by_key(|r| r.3);
        open.truncate(MAX_RECALLED);
        open.into_iter().map(|(i, x, y, _)| (i, x, y)).collect()
    }

    /// Fold routes to the unsearched regions back in, by the index
    /// `unswept` handed out.
    pub fn routed_sweeps(&mut self, paths: &[(i64, Option<Path>)]) {
        for (i, path) in paths {
            if let Some(s) = self.swept.get_mut(*i as usize) {
                s.path = *path;
            }
        }
    }

    /// Somewhere else whose walls are worth searching, placed from where the
    /// player is standing now, nearest first.
    ///
    /// Not the region being stood in: the way to push on the walls here is
    /// to push on them, and offering a walk to where the player already is
    /// is an option that spends a decision going nowhere.
    pub fn unfrisked(&self, state: &State) -> Vec<Recalled> {
        let p = &state.player;
        let mut out: Vec<Recalled> = self
            .swept
            .iter()
            .filter(|s| s.tried < SWEPT)
            .filter_map(|s| {
                let (bearing, distance) = from_here(p, s.x, s.y)?;
                if distance < SWEEP_UNITS as i32 {
                    return None;
                }
                Some(Recalled {
                    id: 0,
                    kind: "unsearched walls".to_string(),
                    class: Class::Pickup,
                    bearing,
                    distance,
                    health: None,
                    ago: 0,
                    path: s.path,
                })
            })
            .collect();
        out.sort_by_key(|r| r.distance);
        out.truncate(MAX_RECALLED);
        out
    }

    /// Where the unfinished business is, in map units, for whoever can ask
    /// the engine the way.
    ///
    /// The index into [`Memory::haunts`] stands in for an object id, because
    /// a haunt is a PLACE rather than a thing - the monster that made it may
    /// be dead, may have moved, or may never have had a stable id at all.
    pub fn hunts(&self) -> Vec<(i64, f64, f64)> {
        self.haunts
            .iter()
            .enumerate()
            .map(|(i, h)| (i as i64, h.x, h.y))
            .collect()
    }

    /// Fold routes to the haunts back in, by the index `hunts` handed out.
    pub fn routed_hunts(&mut self, paths: &[(i64, Option<Path>)]) {
        for (i, path) in paths {
            if let Some(h) = self.haunts.get_mut(*i as usize) {
                h.path = *path;
            }
        }
    }

    /// Unfinished business, placed from where the player is standing now.
    ///
    /// Nearest first, because the cheapest monster to go back for is the one
    /// closest to hand, and a Max run is a sequence of exactly that decision.
    pub fn unfinished(&self, state: &State) -> Vec<Recalled> {
        let p = &state.player;
        let mut out: Vec<Recalled> = self
            .haunts
            .iter()
            .enumerate()
            .filter_map(|(i, h)| {
                let (bearing, distance) = from_here(p, h.x, h.y)?;
                Some(Recalled {
                    id: i as i64,
                    kind: h.kind.clone(),
                    class: Class::Threat,
                    bearing,
                    distance,
                    health: None,
                    ago: 0,
                    path: h.path,
                })
            })
            .collect();
        out.sort_by_key(|r| r.distance);
        out.truncate(MAX_RECALLED);
        out
    }

    fn note(&mut self, p: &Player, class: Class, t: &Thing) {
        let Some((x, y)) = place(p, t.bearing, t.distance) else {
            return;
        };
        if let Some(h) = self.held.iter_mut().find(|h| h.id == t.id) {
            h.x = x;
            h.y = y;
            h.health = t.health;
            if let Some(n) = t.health {
                h.best_health = Some(h.best_health.map_or(n, |b| b.max(n)));
            }
            h.kind.clone_from(&t.kind);
            h.ago = 0;
            return;
        }
        self.held.push(Held {
            id: t.id,
            kind: t.kind.clone(),
            class,
            x,
            y,
            health: t.health,
            best_health: t.health,
            ago: 0,
            path: None,
        });
    }

    fn forget(&mut self, p: &Player) {
        let took = self.took;
        self.held.retain(|h| {
            if h.ago == 0 {
                return true;
            }
            if !h.class.stays_put() {
                return h.ago <= MONSTER_MEMORY;
            }
            // It stays put, so what unseats the memory is EVIDENCE that it
            // has gone, and being nearby is not evidence: a player on the far
            // side of a wall from a medikit is within arm's length of it and
            // has learned nothing, and so is one who walked past and turned
            // away. Two things count - having picked something up while
            // standing on it, and having looked where it was and not seen it.
            match from_here(p, h.x, h.y) {
                Some((bearing, d)) if d <= REACHED => !(took || bearing.abs() <= LOOKED),
                Some(_) => true,
                // Nothing to place it against, so nothing was learned.
                None => true,
            }
        });
    }

    /// Where the things worth walking back to are, in map units, for whoever
    /// can ask the engine the way.
    ///
    /// The only place coordinates leave this module, and they go to the
    /// ENGINE rather than into an observation. A policy told where it is on
    /// the map would be memorising a map, which is the one thing the
    /// generated levels exist to make worthless.
    ///
    /// Only things that stay put: a monster has moved since, so the way to
    /// where it was is the way to where it is not.
    pub fn goals(&self) -> Vec<(i64, f64, f64)> {
        self.held
            .iter()
            .filter(|h| h.ago > 0 && h.class.stays_put())
            .map(|h| (h.id, h.x, h.y))
            .collect()
    }

    /// Fold the answers back in, by id. Anything not answered for loses
    /// whatever path it had, because a route computed from somewhere the
    /// player is no longer standing is worse than none.
    pub fn routed(&mut self, paths: &[(i64, Option<Path>)]) {
        for h in self.held.iter_mut() {
            h.path = paths.iter().find(|(id, _)| *id == h.id).and_then(|(_, p)| *p);
        }
    }

    /// Everything seen this episode that is now carrying less health than the
    /// most it was ever seen with.
    ///
    /// This is what tells two identical sergeants apart. The policy is
    /// memoryless and the option text is the same for both, so with nothing
    /// to separate them it has no reason to finish one before starting on the
    /// other - and flipping between them is the correct response to a state
    /// in which they are indistinguishable. A player is never in that
    /// position: they can see which one they have been hitting.
    pub fn wounded(&self) -> Vec<i64> {
        self.held
            .iter()
            .filter(|h| match (h.health, h.best_health) {
                (Some(now), Some(best)) => now < best,
                _ => false,
            })
            .map(|h| h.id)
            .collect()
    }

    /// What is worth saying, nearest first. Only things out of sight: what
    /// can be seen is in the observation already, and saying it twice is
    /// noise in the place a decision is made.
    pub fn recall(&self, state: &State) -> Vec<Recalled> {
        let p = &state.player;
        let mut out: Vec<Recalled> = self
            .held
            .iter()
            .filter(|h| h.ago > 0)
            .filter_map(|h| {
                let (bearing, distance) = from_here(p, h.x, h.y)?;
                Some(Recalled {
                    id: h.id,
                    kind: h.kind.clone(),
                    class: h.class,
                    bearing,
                    distance,
                    health: h.health,
                    ago: h.ago,
                    path: h.path,
                })
            })
            .collect();
        out.sort_by_key(|r| r.distance);
        out.truncate(MAX_RECALLED);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::State;

    /// A state with one medikit dead ahead at 200 units, and one sergeant to
    /// the right at 300, from a player at the origin facing east.
    /// The distinction this ledger exists for. A monster's exact position
    /// goes stale in ten decisions, correctly - it has moved. That it WAS
    /// over there does not, and for "kill everything" that is most of the
    /// task: the last monsters of a level are the ones seen once from a
    /// doorway and never gone back for.
    #[test]
    fn a_monster_seen_once_stays_unfinished_business_long_after_it_goes_stale() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());
        assert_eq!(m.unfinished(&looking_at_both()).len(), 1, "the sighting was not filed");

        // Look away for far longer than a monster's own memory lasts.
        let away = State::parse(&build(0, 0, 0, NOTHING)).unwrap();
        for _ in 0..(MONSTER_MEMORY * 4) {
            m.observe(&away);
        }
        assert!(
            m.recall(&away).iter().all(|r| r.class != Class::Threat),
            "the sharp memory of the monster should have gone stale"
        );
        assert_eq!(
            m.unfinished(&away).len(),
            1,
            "but the fact that one was over there should not have"
        );
    }

    /// Forgotten on evidence, not on a timer: the player went there and
    /// looked. Being NEAR it teaches nothing - a player on the far side of a
    /// wall is within arm's length - which is the same rule an item is
    /// forgotten on, deliberately.
    #[test]
    fn unfinished_business_is_settled_by_going_there_and_looking() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());
        assert_eq!(m.unfinished(&looking_at_both()).len(), 1);

        // Bearing -90 from a player at the origin facing east is world -90,
        // so the sergeant is at roughly (0, -300).
        //
        // Walk to within fifty units of it but face the other way. That is
        // the case the rule exists for: being near something teaches nothing
        // about it.
        let near_but_turned = State::parse(&build(0, -250, 90, NOTHING)).unwrap();
        m.observe(&near_but_turned);
        assert_eq!(
            m.unfinished(&near_but_turned).len(),
            1,
            "standing near it with its back turned counted as having checked"
        );

        // Now look at where it was, and find nothing.
        let looked = State::parse(&build(0, -250, -90, NOTHING)).unwrap();
        m.observe(&looked);
        assert!(m.unfinished(&looked).is_empty(), "going there and looking did not settle it");
    }

    /// Two sightings of one monster from two doorways are ONE piece of
    /// unfinished business. Filed per sighting, the search would be sent to
    /// the same room twice and the ledger would grow without bound while the
    /// player stands still watching something pace about.
    #[test]
    fn sightings_in_one_region_are_one_piece_of_unfinished_business() {
        let mut m = Memory::new();
        let seen = |d: i32| {
            State::parse(&build(
                0,
                0,
                0,
                &format!(
                    r#""pickups":[],"threats":[{{"id":9,"type":"IMP","distance":{d},"bearing":0,"visible":true,"health":60,"targetingMe":false}}],"hazards":[]"#
                ),
            ))
            .unwrap()
        };
        m.observe(&seen(300));
        m.observe(&seen(320));
        m.observe(&seen(280));
        assert_eq!(m.unfinished(&seen(300)).len(), 1);
    }

    /// A corpse is not unfinished business. Filing one would send the search
    /// back to a room it has already cleared, forever.
    #[test]
    fn a_dead_monster_is_not_filed() {
        let mut m = Memory::new();
        let corpse = State::parse(&build(
            0,
            0,
            0,
            r#""pickups":[],"threats":[{"id":9,"type":"IMP","distance":300,"bearing":0,"visible":true,"health":0,"targetingMe":false}],"hazards":[]"#,
        ))
        .unwrap();
        m.observe(&corpse);
        assert!(m.unfinished(&corpse).is_empty());
    }

    fn looking_at_both() -> State {
        State::parse(&build(0, 0, 0, r#""pickups":[{"id":7,"type":"Medikit","distance":200,"bearing":0,"visible":true}],"threats":[{"id":9,"type":"FORMER HUMAN SERGEANT","distance":300,"bearing":-90,"visible":true,"health":30,"targetingMe":false}],"hazards":[]"#)).unwrap()
    }

    pub fn build(x: i32, y: i32, angle: i32, things: &str) -> String {
        format!(
            r#"{{"tic":1,"episodeTic":1,
            "level":{{"episode":1,"map":1,"skill":2,"tic":1,"kills":0,"totalKills":0,
                      "items":0,"totalItems":0,"secrets":0,"totalSecrets":0}},
            "player":{{"id":0,"standingInDamage":false,"health":100,"armor":0,
                      "x":{x},"y":{y},"angle":{angle},"weapon":"pistol","ammo":50,"keys":[]}},
            {things},
            "clearance":{{"ahead":320,"right":320,"behind":320,"left":320,
                         "aheadRight":320,"aheadLeft":320}},
            "events":[],"done":false,"outcome":"alive"}}"#
        )
    }

    pub const NOTHING: &str = r#""pickups":[],"threats":[],"hazards":[]"#;

    #[test]
    fn what_was_seen_is_still_there_after_turning_away_from_it() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());

        // Turn 90 degrees to the left without moving. The medikit was dead
        // ahead; it is now off the right shoulder, at the same distance.
        let turned = State::parse(&build(0, 0, 90, NOTHING)).unwrap();
        m.observe(&turned);
        let r = m.recall(&turned);

        let medikit = r.iter().find(|r| r.id == 7).expect("the medikit is remembered");
        assert_eq!(medikit.distance, 200, "it did not move");
        assert_eq!(medikit.bearing, -90, "it is off the right shoulder now");
        assert_eq!(medikit.ago, 1);
    }

    #[test]
    fn walking_away_makes_it_further_off_rather_than_wrong() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());

        // Walk 500 units the other way. The medikit is now behind, and the
        // distance is the distance it really is.
        let away = State::parse(&build(-500, 0, 0, NOTHING)).unwrap();
        m.observe(&away);
        let r = m.recall(&away);

        let medikit = r.iter().find(|r| r.id == 7).expect("still remembered");
        assert_eq!(medikit.distance, 700);
        assert_eq!(medikit.bearing.abs(), 0, "still straight ahead, just further");
    }

    #[test]
    fn a_monster_is_forgotten_long_before_an_item_is() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());

        // Stand still and look at nothing for a long time.
        let blind = State::parse(&build(0, 0, 0, NOTHING)).unwrap();
        for _ in 0..MONSTER_MEMORY {
            m.observe(&blind);
        }
        assert!(
            m.recall(&blind).iter().any(|r| r.id == 9),
            "a monster is still worth remembering this soon"
        );

        m.observe(&blind);
        assert!(
            !m.recall(&blind).iter().any(|r| r.id == 9),
            "a monster out of sight this long could be anywhere"
        );
        assert!(
            m.recall(&blind).iter().any(|r| r.id == 7),
            "an item does not walk off"
        );
    }

    #[test]
    fn getting_there_and_finding_nothing_is_how_an_item_is_forgotten() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());

        // Walk onto the spot. Nothing is in sight there, so it has gone -
        // which is also what picking it up looks like from here.
        let arrived = State::parse(&build(200, 0, 0, NOTHING)).unwrap();
        m.observe(&arrived);
        assert!(
            !m.recall(&arrived).iter().any(|r| r.id == 7),
            "it was not where it was"
        );
    }

    #[test]
    fn seeing_it_again_replaces_what_was_remembered_about_it() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());

        // The sergeant has moved and been hurt since.
        let again = State::parse(&build(0, 0, 0, r#""pickups":[],"hazards":[],"threats":[{"id":9,"type":"FORMER HUMAN SERGEANT","distance":100,"bearing":0,"visible":true,"health":8,"targetingMe":true}]"#)).unwrap();
        m.observe(&again);

        let blind = State::parse(&build(0, 0, 0, NOTHING)).unwrap();
        m.observe(&blind);
        let sergeant = m
            .recall(&blind)
            .into_iter()
            .find(|r| r.id == 9)
            .expect("remembered");
        assert_eq!(sergeant.distance, 100, "where it was last, not where it was first");
        assert_eq!(sergeant.health, Some(8));
    }

    #[test]
    fn the_one_you_have_been_shooting_is_the_one_carrying_less_than_it_had() {
        let mut m = Memory::new();
        let two = |a: i32, b: i32| {
            State::parse(&build(
                0,
                0,
                0,
                &format!(
                    r#""pickups":[],"hazards":[],"threats":[
                    {{"id":1,"type":"FORMER HUMAN SERGEANT","distance":300,"bearing":-30,
                      "visible":true,"health":{a},"targetingMe":true}},
                    {{"id":2,"type":"FORMER HUMAN SERGEANT","distance":300,"bearing":30,
                      "visible":true,"health":{b},"targetingMe":true}}]"#
                ),
            ))
            .unwrap()
        };

        m.observe(&two(30, 30));
        assert!(m.wounded().is_empty(), "nothing has been hit yet");

        m.observe(&two(12, 30));
        assert_eq!(m.wounded(), vec![1], "the left one has been hit");

        // Healing is not a thing DOOM monsters do, but a monster spawning
        // later with more health than this one must not make this one read as
        // unhurt again.
        m.observe(&two(12, 30));
        assert_eq!(m.wounded(), vec![1]);
    }

    #[test]
    fn what_can_be_seen_is_not_also_recalled() {
        let mut m = Memory::new();
        let s = looking_at_both();
        m.observe(&s);
        assert!(
            m.recall(&s).is_empty(),
            "both are in the observation already; saying them twice is noise"
        );
    }

    #[test]
    fn an_episode_does_not_inherit_the_last_one_s_memory() {
        let mut m = Memory::new();
        m.observe(&looking_at_both());
        m.clear();
        let blind = State::parse(&build(0, 0, 0, NOTHING)).unwrap();
        m.observe(&blind);
        assert!(m.recall(&blind).is_empty());
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;

    fn seen_then_turned_away() -> Memory {
        let mut m = Memory::new();
        m.observe(&State::parse(&super::tests::build(
            0,
            0,
            0,
            r#""pickups":[{"id":7,"type":"Medikit","distance":200,"bearing":0,"visible":true}],"threats":[{"id":9,"type":"IMP","distance":300,"bearing":0,"visible":true,"health":60,"targetingMe":false}],"hazards":[]"#,
        )).unwrap());
        m.observe(&State::parse(&super::tests::build(0, 0, 180, super::tests::NOTHING)).unwrap());
        m
    }

    /// The engine has to be asked the way to a place, so the place has to
    /// leave here. Items only: the way to where a monster was is the way to
    /// where it is not.
    #[test]
    fn the_things_worth_walking_back_to_are_offered_for_routing() {
        let m = seen_then_turned_away();
        let goals = m.goals();
        assert_eq!(goals.len(), 1, "expected the medikit and only the medikit");
        assert_eq!(goals[0].0, 7);
        assert!((goals[0].1 - 200.0).abs() < 1.0, "the medikit was 200 units east");
    }

    /// The answer comes back on the recalled thing, beside the straight line
    /// to it, so an option can offer the walk instead of the wall.
    #[test]
    fn a_route_answered_for_reaches_the_recalled_thing() {
        let mut m = seen_then_turned_away();
        m.routed(&[(
            7,
            Some(Path {
                bearing: -40,
                distance: 330,
                clearance: 256,
                step: 128,
            }),
        )]);
        let state = State::parse(&super::tests::build(0, 0, 180, super::tests::NOTHING)).unwrap();
        let r = m.recall(&state);
        let kit = r
            .iter()
            .find(|r| r.id == 7)
            .expect("the medikit is remembered");
        assert_eq!(
            kit.path,
            Some(Path {
                bearing: -40,
                distance: 330,
                clearance: 256,
                step: 128
            })
        );
        // And the straight line is still there and still different: 180 off
        // the nose at 200 units, against a 330-unit walk starting 40 to the
        // left. Offering only one of the two would be losing information.
        assert_eq!(kit.distance, 200);
    }

    /// A stale route is worse than none: it is a heading, and a heading
    /// computed from somewhere the player is no longer standing points
    /// somewhere they never wanted to go.
    #[test]
    fn a_thing_not_answered_for_loses_the_route_it_had() {
        let mut m = seen_then_turned_away();
        m.routed(&[(
            7,
            Some(Path {
                bearing: -40,
                distance: 330,
                clearance: 256,
                step: 128,
            }),
        )]);
        m.routed(&[]);
        let state = State::parse(&super::tests::build(0, 0, 180, super::tests::NOTHING)).unwrap();
        assert_eq!(
            m.recall(&state).iter().find(|r| r.id == 7).unwrap().path,
            None
        );
    }
}

#[cfg(test)]
mod expiry_tests {
    use super::*;

    /// Saw a medikit 200 units east, then walked onto the spot.
    fn walked_onto_it(angle: i32, events: &str) -> Memory {
        let mut m = Memory::new();
        m.observe(&State::parse(&super::tests::build(
            0,
            0,
            0,
            r#""pickups":[{"id":7,"type":"Medikit","distance":200,"bearing":0,"visible":true}],"threats":[],"hazards":[]"#,
        )).unwrap());
        let on_top = super::tests::build(200, 0, angle, super::tests::NOTHING)
            .replace(r#""events":[]"#, &format!(r#""events":[{events}]"#));
        m.observe(&State::parse(&on_top).unwrap());
        m
    }

    fn still_there(m: &Memory) -> bool {
        m.held.iter().any(|h| h.id == 7)
    }

    /// The one that was wrong. Standing near where something was, while
    /// facing away from it, is not evidence that it has gone - and it was
    /// deleting remembered items on nothing but proximity.
    #[test]
    fn being_near_it_while_looking_elsewhere_is_not_evidence() {
        assert!(still_there(&walked_onto_it(180, "")), "forgotten without looking");
    }

    /// Looked where it was and did not see it: it has gone.
    #[test]
    fn looking_at_the_spot_and_seeing_nothing_is_evidence() {
        assert!(!still_there(&walked_onto_it(0, "")));
    }

    /// And picking something up while standing on it settles it whichever
    /// way the player happens to be facing.
    #[test]
    fn collecting_something_there_is_evidence() {
        let taken = r#"{"tic":2,"type":"item","what":"Medikit","amount":25}"#;
        assert!(!still_there(&walked_onto_it(180, taken)));
    }

    /// Far away and facing it is not evidence either: seeing nothing at 400
    /// units says nothing about what is on the floor there.
    #[test]
    fn looking_from_a_distance_is_not_evidence() {
        let mut m = Memory::new();
        m.observe(&State::parse(&super::tests::build(
            0, 0, 0,
            r#""pickups":[{"id":7,"type":"Medikit","distance":200,"bearing":0,"visible":true}],"threats":[],"hazards":[]"#,
        )).unwrap());
        m.observe(&State::parse(&super::tests::build(-200, 0, 0, super::tests::NOTHING)).unwrap());
        assert!(still_there(&m));
    }
}

#[cfg(test)]
mod frisk_tests {
    use super::*;

    /// Standing with a wall an arm's length in front, which is what a push
    /// has to be for it to test anything. See `IN_REACH`.
    fn at(x: i32, y: i32, angle: i32) -> State {
        let mut s =
            State::parse(&super::tests::build(x, y, angle, super::tests::NOTHING)).unwrap();
        s.clearance.ahead = 32;
        s
    }

    /// The same spot with open floor in front of it.
    fn in_the_open(x: i32, y: i32, angle: i32) -> State {
        State::parse(&super::tests::build(x, y, angle, super::tests::NOTHING)).unwrap()
    }

    /// Push on what is in front, and hear what came of it.
    ///
    /// Two calls, because that is how the two arrive: where the player was
    /// standing is only true BEFORE the step, and what the push did is only
    /// known after it.
    fn push(m: &mut Memory, at: &State, came_to: &str) {
        m.press(at);
        m.pushed(at, came_to);
    }

    /// The whole point of the ledger. A wall pushed on once is a wall this
    /// run knows about, and a search that cannot tell the difference spends
    /// its budget pushing the same wall over and over.
    #[test]
    fn a_wall_that_has_been_pushed_on_is_known_to_have_been_pushed_on() {
        let mut m = Memory::new();
        let here = at(0, 0, 0);
        assert!(!m.pressed_here(&here), "nothing has been pushed yet");
        push(&mut m, &here, "solid");
        assert!(m.pressed_here(&here));
        // The same spot facing the other way is a DIFFERENT wall, and the
        // one a sweep is about to move on to.
        assert!(!m.pressed_here(&at(0, 0, 180)));
    }

    /// Far enough along a wall is a different piece of wall. A ledger that
    /// could not tell would retire a whole room after a single push.
    #[test]
    fn sliding_along_a_wall_reaches_a_piece_of_it_that_is_untested() {
        let mut m = Memory::new();
        push(&mut m, &at(0, 0, 0), "solid");
        assert!(!m.pressed_here(&at(200, 0, 0)));
    }

    /// A push that reached nothing tested nothing, and the ENGINE says which
    /// it was.
    ///
    /// Clearance was only ever a guess at this: it is measured from the
    /// player's centre along their facing, while `P_UseLines` traces its own
    /// line and can miss where the probe hit. Measured over one short
    /// campaign, of 56 pushes the agent made 21 reached nothing at all - and
    /// counting those retires rooms as searched on the strength of the agent
    /// having walked about in them pressing air.
    #[test]
    fn a_push_that_reached_nothing_tested_nothing() {
        let mut m = Memory::new();
        m.observe(&in_the_open(0, 0, 0));
        push(&mut m, &in_the_open(0, 0, 0), "nothing there");
        assert_eq!(m.tested(), 0, "the push met no line at all");
        assert_eq!(m.unswept(&at(600, 0, 0)).len(), 1, "the room is still unsearched");
        // A push elsewhere that DID meet a wall tested one, wherever
        // clearance thought the wall was.
        push(&mut m, &in_the_open(200, 0, 0), "solid");
        assert_eq!(m.tested(), 1);
    }

    /// A push that reached nothing is still a push that was MADE, and
    /// forgetting that is a loop.
    ///
    /// The bug this guards cost a whole measured campaign. Treating "reached
    /// nothing" as "never tried" leaves `pressed_here` false, so the sweep
    /// pushes at the same spot again and `untried_walls` keeps offering the
    /// turn to it - measured, a 420-second campaign tested seven walls
    /// against the baseline's thirty-eight, and found no secret where the
    /// baseline found one.
    ///
    /// Tried and tested are different questions and the ledger has to answer
    /// both: yes it has been tried, no it tested nothing.
    #[test]
    fn a_push_that_reached_nothing_was_still_tried_there() {
        let mut m = Memory::new();
        let spot = in_the_open(0, 0, 0);
        push(&mut m, &spot, "nothing there");
        assert!(m.pressed_here(&spot), "the sweep would push here forever");
        assert_eq!(m.tested(), 0, "and it still searched no wall");
    }

    /// And a push that MADE SOMETHING MOVE is the discovery the whole sweep
    /// exists for, so it is counted apart from the walls that did nothing.
    #[test]
    fn a_push_that_worked_is_remembered_as_a_thing_opened() {
        let mut m = Memory::new();
        assert_eq!(m.opened(), 0);
        push(&mut m, &at(0, 0, 0), "solid");
        assert_eq!(m.opened(), 0, "a solid wall opened nothing");
        push(&mut m, &at(0, 0, 90), "worked");
        assert_eq!(m.opened(), 1);
        assert_eq!(m.tested(), 2, "both were walls the run has now tried");
    }

    /// The rule a haunt does NOT follow, and the reason this is a second
    /// ledger rather than a field on the first: looking at where a monster
    /// was is evidence it has gone, and looking at a wall is no evidence at
    /// all about what is behind it.
    #[test]
    fn walking_through_a_room_does_not_test_its_walls() {
        let mut m = Memory::new();
        m.observe(&at(0, 0, 0));
        m.observe(&at(64, 0, 90));
        assert_eq!(m.unswept(&at(600, 0, 0)).len(), 1, "still worth coming back to push on");
    }

    /// And pushing on enough of them does.
    #[test]
    fn pushing_on_enough_of_a_rooms_walls_retires_it() {
        let mut m = Memory::new();
        m.observe(&at(0, 0, 0));
        for i in 0..SWEPT as i32 {
            // A different piece of wall each time, which is what a sweep
            // actually is.
            push(&mut m, &at((i % 4) * 64, (i / 4) * 64, 0), "solid");
        }
        assert!(m.unswept(&at(600, 0, 0)).is_empty(), "its walls have been tested");
    }

    /// Pushing the same wall again is one wall tested, not two.
    #[test]
    fn pushing_the_same_wall_again_tests_nothing_new() {
        let mut m = Memory::new();
        m.observe(&at(0, 0, 0));
        for _ in 0..SWEPT * 2 {
            push(&mut m, &at(0, 0, 0), "solid");
        }
        assert_eq!(m.tested(), 1);
        assert_eq!(m.unswept(&at(600, 0, 0)).len(), 1, "one wall is not a swept room");
    }

    /// The frontier has to leave here to be routed to, exactly as unfinished
    /// business does: the way to a room is not the heading to it.
    #[test]
    fn untested_rooms_are_offered_for_routing_and_take_their_answers() {
        let mut m = Memory::new();
        m.observe(&at(0, 0, 0));
        m.observe(&at(600, 0, 0));
        let open = m.unswept(&at(600, 0, 0));
        assert_eq!(open.len(), 1, "the room being stood in is not routed to");
        let path = Path {
            bearing: -40,
            distance: 700,
            clearance: 256,
            step: 128,
        };
        assert_eq!(open[0].0, 0, "the far one");
        m.routed_sweeps(&[(open[0].0, Some(path))]);

        let back = m.unfrisked(&at(600, 0, 0));
        assert_eq!(
            back.len(),
            1,
            "the room being stood in is not somewhere to go"
        );
        assert_eq!(back[0].path, Some(path));
    }

    /// Every episode after the first in a process began holding the last
    /// one's memory, and the ledgers this file added made it worse rather
    /// than causing it. The visible half is an agent that remembers a level
    /// it has not played; the expensive half is that a trajectory replayed
    /// after another one is offered different options and diverges, so it
    /// cannot be verified - and nothing in the trajectory says why.
    #[test]
    fn a_new_episode_remembers_nothing_from_the_last_one() {
        let mut m = Memory::new();
        m.observe(&State::parse(&super::tests::build(
            0,
            0,
            0,
            r#""pickups":[{"id":7,"type":"Medikit","distance":200,"bearing":0,"visible":true}],"threats":[{"id":9,"type":"IMP","distance":300,"bearing":0,"visible":true,"health":60,"targetingMe":false}],"hazards":[]"#,
        )).unwrap());
        push(&mut m, &at(0, 0, 0), "solid");
        m.observe(&at(600, 0, 0));
        let far = at(2000, 2000, 0);
        assert!(m.tested() > 0 && !m.unswept(&far).is_empty() && !m.recall(&far).is_empty());

        m.clear();

        assert_eq!(m.tested(), 0, "walls pushed on in the last episode");
        assert!(m.unswept(&far).is_empty(), "rooms stood in during the last episode");
        assert!(m.recall(&far).is_empty(), "things seen in the last episode");
        assert!(m.unfinished(&far).is_empty(), "monsters from the last episode");
    }

    /// The half that was missing, and the reason the ledger was honest and
    /// nearly inert: `use` reaches only what the player FACES, and a player
    /// walks corridors facing along them, so the walls are beside them an
    /// arm's length away the whole time and never in front.
    #[test]
    fn a_wall_beside_the_player_is_somewhere_to_turn_and_push() {
        let m = Memory::new();
        let mut s = in_the_open(0, 0, 0);
        s.clearance.left = 40;
        s.clearance.right = 300;
        s.clearance.behind = 300;
        assert_eq!(m.untried_walls(&s), vec![90], "the wall on the left");
    }

    /// One already pushed on is not somewhere to turn: it has been tried,
    /// and turning to try it again is the waste the ledger exists to stop.
    #[test]
    fn a_wall_already_pushed_on_is_not_somewhere_to_turn() {
        let mut m = Memory::new();
        push(&mut m, &at(0, 0, 90), "solid");
        let mut s = in_the_open(0, 0, 0);
        s.clearance.left = 40;
        s.clearance.right = 300;
        s.clearance.behind = 300;
        assert!(m.untried_walls(&s).is_empty());
    }

    /// And with a wall already in front of them, pushing is the act rather
    /// than turning away to a different one.
    #[test]
    fn nothing_to_turn_to_while_facing_a_wall_within_reach() {
        let m = Memory::new();
        let mut s = in_the_open(0, 0, 0);
        s.clearance.ahead = 40;
        s.clearance.left = 40;
        assert!(m.untried_walls(&s).is_empty());
    }

    /// What the archive counts. Two runs standing in the same place having
    /// killed the same monsters are not in the same situation when one has
    /// tested forty walls and the other none - and an archive that cannot
    /// see the difference keeps the faster one, which deletes the search.
    #[test]
    fn walls_tested_counts_distinct_walls() {
        let mut m = Memory::new();
        push(&mut m, &at(0, 0, 0), "solid");
        push(&mut m, &at(0, 0, 90), "solid");
        push(&mut m, &at(0, 0, 0), "solid");
        assert_eq!(m.tested(), 2);
    }
}

#[cfg(test)]
mod wall_tests {
    use super::*;

    fn looking_at(texture: &str, offset: i32) -> State {
        let mut s =
            State::parse(&super::tests::build(0, 0, 0, super::tests::NOTHING)).unwrap();
        s.clearance.ahead = 32;
        s.facing_wall = Some(crate::obs::Wall {
            texture: texture.into(),
            above: String::new(),
            below: String::new(),
            offset,
            distance: 32,
        });
        s
    }

    /// The cue a player actually uses. A level is built from a handful of
    /// textures repeated everywhere; one that is not among them is where a
    /// player pushes.
    #[test]
    fn a_wall_unlike_the_ones_around_it_looks_odd() {
        let mut m = Memory::new();
        // Enough ordinary wall to have an opinion about what ordinary is.
        for i in 0..40 {
            m.observe(&looking_at("BROWN96", i % 2));
        }
        assert!(!m.odd_wall(&looking_at("BROWN96", 0)), "the usual wall is not odd");
        assert!(m.odd_wall(&looking_at("SW1STRTN", 0)), "a wall seen nowhere else");
    }

    /// And the same texture shoved out of alignment is odd too, which is most
    /// of how DOOM marks a secret door - a name-only signal would miss it.
    #[test]
    fn a_usual_texture_out_of_alignment_looks_odd() {
        let mut m = Memory::new();
        for _ in 0..40 {
            m.observe(&looking_at("BROWN96", 0));
        }
        for i in 1..6 {
            m.observe(&looking_at(&format!("OTHER{i}"), 0));
        }
        assert!(m.odd_wall(&looking_at("BROWN96", 37)), "the same wall, out of step");
    }

    /// Everything is unusual when you have seen three things, so a run that
    /// has barely looked at anything keeps its opinions to itself.
    #[test]
    fn a_run_that_has_seen_almost_nothing_calls_nothing_odd() {
        let mut m = Memory::new();
        m.observe(&looking_at("BROWN96", 0));
        assert!(!m.odd_wall(&looking_at("SW1STRTN", 0)));
    }

    /// And a new episode has no opinions at all.
    #[test]
    fn what_walls_looked_like_does_not_survive_the_episode() {
        let mut m = Memory::new();
        for _ in 0..40 {
            m.observe(&looking_at("BROWN96", 0));
        }
        for i in 1..6 {
            m.observe(&looking_at(&format!("OTHER{i}"), 0));
        }
        assert!(m.odd_wall(&looking_at("SW1STRTN", 0)));
        m.clear();
        assert!(!m.odd_wall(&looking_at("SW1STRTN", 0)), "last episode's walls");
    }
}
