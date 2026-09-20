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

/// How close counts as having got there. A player who walks over the spot an
/// item was on and does not pick anything up has learned that it has gone.
pub const REACHED: i32 = 56;

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

/// Something out of sight, placed from where the player is standing now.
#[derive(Clone, Debug)]
pub struct Recalled {
    pub id: i64,
    pub kind: String,
    pub class: Class,
    /// Degrees relative to the player's facing, negative to the left - the
    /// same convention the engine uses for what IS in sight.
    pub bearing: i32,
    pub distance: i32,
    pub health: Option<i32>,
    /// Decisions since it was last seen.
    pub ago: u32,
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
}

#[derive(Clone, Default)]
pub struct Memory {
    held: Vec<Held>,
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
            }
        }
        self.forget(p);
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
        });
    }

    fn forget(&mut self, p: &Player) {
        self.held.retain(|h| {
            if h.ago == 0 {
                return true;
            }
            if !h.class.stays_put() {
                return h.ago <= MONSTER_MEMORY;
            }
            // It stays put, so the only thing that unseats the memory is
            // having been there. Out of sight AND within arm's length of
            // where it was means it is not there any more - which is also
            // what picking it up looks like from here.
            match from_here(p, h.x, h.y) {
                Some((_, d)) => d > REACHED,
                None => true,
            }
        });
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
    fn looking_at_both() -> State {
        State::parse(&build(0, 0, 0, r#""pickups":[{"id":7,"type":"Medikit","distance":200,"bearing":0,"visible":true}],"threats":[{"id":9,"type":"FORMER HUMAN SERGEANT","distance":300,"bearing":-90,"visible":true,"health":30,"targetingMe":false}],"hazards":[]"#)).unwrap()
    }

    fn build(x: i32, y: i32, angle: i32, things: &str) -> String {
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

    const NOTHING: &str = r#""pickups":[],"threats":[],"hazards":[]"#;

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
