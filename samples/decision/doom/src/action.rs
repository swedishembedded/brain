// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What can be done RIGHT NOW - rebuilt from the state on every decision.
//!
//! This is the part a fixed-head policy network cannot express. The options
//! below are not a fixed enumeration with a stable index: "attack the imp 12
//! degrees left" exists only while that imp is alive and in sight, "press the
//! door in front of you" only while something is in front of you, and the same
//! slot holds a different meaning one step later. A network whose output layer
//! IS the action space has to be retrained to learn a new action; a model that
//! reads the option text can be handed one it has never seen.
//!
//! Each option carries the primitive engine commands it expands to, so the
//! policy chooses among *intentions* ("kill that thing") while the engine
//! still receives key presses.
//!
//! Swedish Embedded AB designs the action vocabularies that make a control
//! problem learnable at all - the layer between "what the machine can be told"
//! and "what a decision is about". If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::obs::State;

/// One thing the agent may do this step.
#[derive(Clone, Debug)]
pub struct Option_ {
    /// What the model reads. The only thing it gets.
    pub text: String,
    /// The engine commands, as the JSON array `/api/step` takes.
    pub commands: String,
    /// How many tics to run while they are held.
    pub tics: u32,
    /// What this option is trying to do, for the inspector and for the
    /// scripted teacher. Never shown to the model.
    pub tag: Tag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Attack,
    Grab,
    Advance,
    Explore,
    Retreat,
    Use,
    Exit,
}

/// A decision every ~4-8 tics is about 5-9 per second of game time, which is
/// the rate a human plays at and slow enough that a decision model with a
/// millisecond of latency is never the thing holding the game up.
const FIGHT_TICS: u32 = 4;
const MOVE_TICS: u32 = 6;
/// Less floor than this is not somewhere to walk: the player is 32 units wide
/// and covers about 40 in one decision, so under 64 is a step into a wall.
const MIN_ROOM: i32 = 64;

fn json_turn(angle: i32) -> String {
    format!("{{\"type\":\"turn-to\",\"angle\":{}}}", angle.rem_euclid(360))
}

/// Build the option list for this state.
///
/// Always non-empty: the environment contract requires it while an episode is
/// live, and a state with nothing to do would end a rollout for a reason that
/// is not the game's.
pub fn options(state: &State) -> Vec<Option_> {
    let mut out: Vec<Option_> = Vec::new();
    let c = &state.clearance;

    // --- fight ------------------------------------------------------------
    for t in state.visible_threats().take(3) {
        let facing = state.facing(t.bearing);
        out.push(Option_ {
            text: format!(
                "attack the {} {} units away, {}",
                t.kind.to_lowercase(),
                t.distance,
                bearing_phrase(t.bearing)
            ),
            // Turn onto it and fire in the same step: the turn servo closes
            // the angle over the step's tics, so firing after it is aimed at
            // where the thing is now rather than where it was.
            commands: format!("[{},{{\"type\":\"shoot\"}}]", json_turn(facing)),
            tics: FIGHT_TICS,
            tag: Tag::Attack,
        });
    }

    // --- take something ---------------------------------------------------
    if let Some(p) = state.pickups.iter().find(|p| p.visible && p.distance < 700) {
        out.push(Option_ {
            text: format!(
                "go and pick up the {} {} units away, {}",
                p.kind.to_lowercase(),
                p.distance,
                bearing_phrase(p.bearing)
            ),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":8}}]",
                json_turn(state.facing(p.bearing))
            ),
            tics: MOVE_TICS,
            tag: Tag::Grab,
        });
    }

    // --- move -------------------------------------------------------------
    if c.ahead >= MIN_ROOM {
        out.push(Option_ {
            text: format!("walk forward, {} units of open floor ahead", c.ahead),
            commands: "[{\"type\":\"forward\",\"amount\":8}]".into(),
            tics: MOVE_TICS,
            tag: Tag::Advance,
        });
    }
    if c.ahead_left >= MIN_ROOM {
        out.push(Option_ {
            text: format!("turn left and go that way, {} units of room", c.ahead_left),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":8}}]",
                json_turn(state.facing(-45))
            ),
            tics: MOVE_TICS,
            tag: Tag::Explore,
        });
    }
    if c.ahead_right >= MIN_ROOM {
        out.push(Option_ {
            text: format!("turn right and go that way, {} units of room", c.ahead_right),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":8}}]",
                json_turn(state.facing(45))
            ),
            tics: MOVE_TICS,
            tag: Tag::Explore,
        });
    }
    if c.behind >= MIN_ROOM {
        out.push(Option_ {
            text: "back away from whatever is in front of you".into(),
            commands: "[{\"type\":\"backward\",\"amount\":8}]".into(),
            tics: MOVE_TICS,
            tag: Tag::Retreat,
        });
    }

    // --- the exit ---------------------------------------------------------
    //
    // Offered only when there is somewhere to go. The exit is usually behind
    // a wall, and an option that walks into it is worse than no option: it is
    // the most attractive-looking thing on the list and it does nothing, which
    // is exactly the trap the scripted player fell into for whole episodes.
    if let Some(e) = state.exit.as_ref().filter(|e| e.clearance >= MIN_ROOM) {
        out.push(Option_ {
            text: format!(
                "head for the level exit, {} units away, {}",
                e.distance,
                bearing_phrase(e.bearing)
            ),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":8}}]",
                json_turn(state.facing(e.bearing))
            ),
            tics: MOVE_TICS,
            tag: Tag::Exit,
        });
    }

    // --- open what is in the way ------------------------------------------
    //
    // Always offered. A door reads as "no clearance ahead", which is exactly
    // the situation where every movement option has been filtered out, so this
    // is also what keeps the list non-empty in a dead end.
    out.push(Option_ {
        text: "push on the wall or door directly in front of you".into(),
        commands: "[{\"type\":\"use\"},{\"type\":\"forward\",\"amount\":4}]".into(),
        tics: MOVE_TICS,
        tag: Tag::Use,
    });

    // --- look around ------------------------------------------------------
    //
    // Also always offered, and it is not filler: a level is mostly not in
    // front of you, and a policy with no way to turn without moving cannot
    // find a door it is standing beside.
    out.push(Option_ {
        text: "turn around to see what is behind you".into(),
        commands: format!("[{}]", json_turn(state.facing(180))),
        tics: FIGHT_TICS,
        tag: Tag::Explore,
    });

    out
}

fn bearing_phrase(b: i32) -> String {
    match b {
        b if b.abs() <= 10 => "straight ahead".to_string(),
        b if b < 0 => format!("{} degrees to your left", -b),
        b => format!("{b} degrees to your right"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(json_patch: &str) -> State {
        let base = format!(
            r#"{{"tic":1,"episodeTic":1,"level":{{"episode":1,"map":1,"skill":2,"tic":1,
            "kills":0,"totalKills":4,"items":0,"totalItems":3,"secrets":0,"totalSecrets":1}},
            "player":{{"health":100,"armor":0,"x":0,"y":0,"angle":90,"weapon":"pistol",
            "ammo":50,"keys":[]}},{json_patch},"events":[],"done":false,"outcome":"alive"}}"#
        );
        State::parse(&base).expect("test state parses")
    }

    const NOTHING: &str = r#""threats":[],"hazards":[],"pickups":[],
        "clearance":{"ahead":0,"right":0,"behind":0,"left":0,"aheadRight":0,"aheadLeft":0}"#;

    #[test]
    fn the_list_is_never_empty_even_walled_in() {
        // Boxed in with nothing alive and nowhere to walk. The environment
        // contract says the option list must be non-empty while the episode is
        // live; an empty one ends the rollout for a reason that is not the
        // game's, and the failure would look like the policy giving up.
        let opts = options(&state(NOTHING));
        assert!(!opts.is_empty());
        assert!(opts.iter().any(|o| o.tag == Tag::Use));
    }

    #[test]
    fn an_option_names_the_thing_it_acts_on() {
        let s = state(
            r#""threats":[{"id":9,"type":"IMP","distance":150,"bearing":-12,"visible":true,
            "health":60,"targetingMe":true}],"hazards":[],"pickups":[],
            "clearance":{"ahead":320,"right":0,"behind":0,"left":0,"aheadRight":0,"aheadLeft":0}"#,
        );
        let opts = options(&s);
        let attack = opts.iter().find(|o| o.tag == Tag::Attack).expect("an attack option");
        // The MEANING is in the text - this is the whole claim of a decision
        // model over a fixed-head policy, so it is worth a test rather than a
        // comment.
        assert!(attack.text.contains("imp"), "{}", attack.text);
        assert!(attack.text.contains("12 degrees to your left"), "{}", attack.text);
        // 90 + (-12): the turn is absolute because the engine's turn keys
        // cannot name a direction.
        assert!(attack.commands.contains("\"angle\":78"), "{}", attack.commands);
    }

    #[test]
    fn options_that_would_walk_into_a_wall_are_not_offered() {
        let blocked = options(&state(NOTHING));
        assert!(!blocked.iter().any(|o| o.tag == Tag::Advance));
        let open = options(&state(
            r#""threats":[],"hazards":[],"pickups":[],
            "clearance":{"ahead":320,"right":0,"behind":0,"left":0,"aheadRight":0,"aheadLeft":0}"#,
        ));
        assert!(open.iter().any(|o| o.tag == Tag::Advance));
    }
}
