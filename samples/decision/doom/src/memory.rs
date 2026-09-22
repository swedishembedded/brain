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

#[derive(Clone, Default)]
pub struct Memory {
    held: Vec<Held>,
    /// Places live monsters were seen, at room scale. See [`Haunt`].
    haunts: Vec<Haunt>,
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
    pub fn clear(&mut self) {
        self.held.clear();
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
    }

    /// File the region a live monster was seen in as unfinished business.
    fn haunt(&mut self, p: &Player, t: &Thing) {
        let Some((x, y)) = place(p, t.bearing, t.distance) else {
            return;
        };
        let region = |a: f64, b: f64| {
            (a / HAUNT_UNITS).floor() == (b / HAUNT_UNITS).floor()
        };
        if let Some(h) = self.haunts.iter_mut().find(|h| region(h.x, x) && region(h.y, y)) {
            // The freshest sighting wins the position, so a monster that is
            // walking toward the player does not leave the marker behind it.
            h.x = x;
            h.y = y;
            h.kind = t.kind.clone();
            return;
        }
        self.haunts.push(Haunt { kind: t.kind.clone(), x, y, path: None });
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
                Some((bearing, d)) if d <= REACHED => {
                    !(took || bearing.abs() <= LOOKED)
                }
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
        m.routed(&[(7, Some(Path { bearing: -40, distance: 330, clearance: 256, step: 128 }))]);
        let state = State::parse(&super::tests::build(0, 0, 180, super::tests::NOTHING)).unwrap();
        let r = m.recall(&state);
        let kit = r.iter().find(|r| r.id == 7).expect("the medikit is remembered");
        assert_eq!(kit.path, Some(Path { bearing: -40, distance: 330, clearance: 256, step: 128 }));
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
        m.routed(&[(7, Some(Path { bearing: -40, distance: 330, clearance: 256, step: 128 }))]);
        m.routed(&[]);
        let state = State::parse(&super::tests::build(0, 0, 180, super::tests::NOTHING)).unwrap();
        assert_eq!(m.recall(&state).iter().find(|r| r.id == 7).unwrap().path, None);
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
