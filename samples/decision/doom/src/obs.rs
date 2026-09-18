// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The game's JSON, typed - and then rendered as the text a decision model
//! reads.
//!
//! Two deliberate choices here.
//!
//! **Typed, with `deny_unknown_fields`.** This JSON crosses a process boundary,
//! so it is external input whatever produced it (see brain's own rule: file
//! input is exactly as hostile as network input, it just fails later and
//! quieter). Required fields are plain types, so serde itself refuses a
//! response that has lost one rather than substituting a plausible default -
//! and an unknown field is an error rather than a silent no-op, which is what
//! catches the engine side being renamed underneath this one.
//!
//! **Text, not a feature vector.** The model is a decision model over language,
//! so the observation is prose with the numbers in it. That is not a
//! presentation detail: it is what lets the action set change every tick and
//! still be understood, because an option like "attack the IMP 12 degrees to
//! your right" carries its own meaning instead of being index 3.
//!
//! Swedish Embedded AB turns machine state into the representation a model can
//! actually decide from - the step most teams skip and then blame the model
//! for. If your team needs that, you can procure our services by sending an
//! email to info@swedishembedded.com.

use serde::Deserialize;

// `deny_unknown_fields` means every field the engine sends has to be named
// here, and a few of them are named for that reason alone: they pin the
// contract rather than feed a decision. Removing one would not remove the
// field from the wire, it would make the response stop parsing - so these are
// load-bearing precisely by existing. Rust's dead-code lint cannot see that,
// which is what the allow is for.
#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub tic: i64,
    #[serde(rename = "episodeTic")]
    pub episode_tic: i64,
    pub level: Level,
    pub player: Player,
    pub threats: Vec<Thing>,
    pub hazards: Vec<Thing>,
    pub pickups: Vec<Thing>,
    pub clearance: Clearance,
    /// Absent on a map with no exit linedef at all.
    pub exit: Option<Exit>,
    /// The nearest place the player has not walked, and the way to it.
    ///
    /// Served by the engine but NOT shown to the model. With a walkable route
    /// to the exit available there is nothing to explore for, and a sensor
    /// that is in the observation because it might help is one nobody can
    /// measure the value of.
    pub unexplored: Option<Frontier>,
    pub events: Vec<Event>,
    /// Present only when the engine's per-step event buffer overflowed.
    #[serde(rename = "eventsDropped", default)]
    pub events_dropped: u32,
    pub done: bool,
    pub outcome: String,
}

// `deny_unknown_fields` means every field the engine sends has to be named
// here, and a few of them are named for that reason alone: they pin the
// contract rather than feed a decision. Removing one would not remove the
// field from the wire, it would make the response stop parsing - so these are
// load-bearing precisely by existing. Rust's dead-code lint cannot see that,
// which is what the allow is for.
#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Level {
    pub episode: u32,
    pub map: u32,
    pub skill: u32,
    pub tic: i64,
    pub kills: u32,
    #[serde(rename = "totalKills")]
    pub total_kills: u32,
    pub items: u32,
    #[serde(rename = "totalItems")]
    pub total_items: u32,
    pub secrets: u32,
    #[serde(rename = "totalSecrets")]
    pub total_secrets: u32,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Player {
    /// The player's own map-object id. Part of the schema; the sample places
    /// episodes through the episode endpoint rather than by object id.
    pub id: i64,
    // x/y are not part of any decision - a policy navigating by coordinates
    // would be memorising one map - but they are how the environment notices
    // that it has stopped moving. See `DoomEnv::stuck`.
    pub health: i32,
    pub armor: i32,
    /// Absent only if the player has no map object, which happens between a
    /// level ending and the next one loading.
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub angle: Option<i32>,
    pub weapon: Option<String>,
    pub ammo: Option<i32>,
    pub keys: Vec<String>,
    /// The floor underfoot is damaging. A player sees the screen flash and
    /// their health tick down; without it an agent crosses a nukage pool
    /// wondering why it is dying.
    #[serde(rename = "standingInDamage", default)]
    pub standing_in_damage: bool,
}

// `deny_unknown_fields` means every field the engine sends has to be named
// here, and a few of them are named for that reason alone: they pin the
// contract rather than feed a decision. Removing one would not remove the
// field from the wire, it would make the response stop parsing - so these are
// load-bearing precisely by existing. Rust's dead-code lint cannot see that,
// which is what the allow is for.
#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Thing {
    pub id: i64,
    #[serde(rename = "type")]
    pub kind: String,
    pub distance: i32,
    /// Degrees relative to the player's facing, negative to the left.
    pub bearing: i32,
    pub visible: bool,
    pub health: Option<i32>,
    #[serde(rename = "targetingMe")]
    pub targeting_me: Option<bool>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clearance {
    pub ahead: i32,
    pub right: i32,
    pub behind: i32,
    pub left: i32,
    #[serde(rename = "aheadRight")]
    pub ahead_right: i32,
    #[serde(rename = "aheadLeft")]
    pub ahead_left: i32,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exit {
    pub distance: i32,
    pub bearing: i32,
    pub kind: String,
    /// How far the player would get setting off toward it right now. The exit
    /// is usually behind a wall, and without this an agent cannot tell "walk
    /// that way" from "walk into that".
    pub clearance: i32,
    /// A walkable place to stand to use this exit, when the engine found one.
    ///
    /// Never shown to the model - it is how a CURRICULUM places an episode
    /// near the goal, which is a property of the experiment and not of the
    /// situation the agent is deciding in.
    pub spot: Option<Spot>,
    /// How far the exit is ALONG WALKABLE GROUND, and which way to set off.
    ///
    /// This is the honest version of `bearing` and `distance` above, which
    /// measure a straight line that goes through walls. Absent on a level
    /// whose exit the engine could not route to.
    #[serde(rename = "pathDistance")]
    pub path_distance: Option<i32>,
    /// What the route actually leads to: absent for the exit itself, or the
    /// colour of the key the exit is locked behind.
    ///
    /// A level whose exit needs a key gives the agent two jobs in sequence.
    /// The engine does the first one - it routes to the key - but calling a
    /// key "the exit" would be a lie the agent has no way to catch.
    pub goal: Option<String>,
    #[serde(rename = "routeBearing")]
    pub route_bearing: Option<i32>,
    /// How far the next waypoint on the route is. Part of the schema; the
    /// decision is made on the bearing and the total, not on this.
    #[serde(rename = "routeDistance")]
    pub route_distance: Option<i32>,
    #[serde(rename = "routeClearance")]
    pub route_clearance: Option<i32>,
    /// What stands between the player and the next step of the route.
    ///
    /// The route runs through shut doors on purpose, because a player opens
    /// them - so "the way there starts left and the player cannot walk left"
    /// is the normal state of affairs at every door in the game, not an error.
    /// Without this the agent has a bearing and no idea why walking it does
    /// nothing.
    #[serde(rename = "blockedBy")]
    pub blocked_by: Option<Blocker>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Blocker {
    /// `door` when pressing use against it opens it, `wall` when it does not.
    pub kind: String,
    pub bearing: i32,
    pub distance: i32,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frontier {
    pub distance: i32,
    pub bearing: i32,
    pub clearance: i32,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spot {
    pub x: f32,
    pub y: f32,
}

// `deny_unknown_fields` means every field the engine sends has to be named
// here, and a few of them are named for that reason alone: they pin the
// contract rather than feed a decision. Removing one would not remove the
// field from the wire, it would make the response stop parsing - so these are
// load-bearing precisely by existing. Rust's dead-code lint cannot see that,
// which is what the allow is for.
#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub tic: i64,
    #[serde(rename = "type")]
    pub kind: String,
    pub what: Option<String>,
    pub amount: i32,
}

impl State {
    pub fn parse(json: &str) -> Result<State, String> {
        serde_json::from_str(json).map_err(|e| format!("the game sent something unreadable: {e}"))
    }

    pub fn angle(&self) -> i32 {
        self.player.angle.unwrap_or(0)
    }

    /// Absolute map angle that faces `bearing` degrees off the player's nose.
    pub fn facing(&self, bearing: i32) -> i32 {
        (self.angle() + bearing).rem_euclid(360)
    }

    /// Threats worth reacting to: in line of sight, nearest first. Something
    /// behind a wall cannot be shot and does not belong in a decision.
    pub fn visible_threats(&self) -> impl Iterator<Item = &Thing> {
        self.threats.iter().filter(|t| t.visible)
    }
}

/// How far is "close enough to describe as close", in map units. The player is
/// 56 units tall and a typical corridor is 128 wide, so these are room-scale,
/// corridor-scale and across-the-map.
const NEAR: i32 = 200;
const MID: i32 = 500;

fn range_word(d: i32) -> &'static str {
    if d < NEAR {
        "close"
    } else if d < MID {
        "nearby"
    } else {
        "far"
    }
}

fn side_word(bearing: i32) -> String {
    match bearing {
        b if b.abs() <= 10 => "straight ahead".to_string(),
        b if b < 0 => format!("{} degrees left", -b),
        b => format!("{b} degrees right"),
    }
}

/// What the agent has done recently, which the game itself does not report.
///
/// This is in the observation because a DECISION DEPENDS ON IT and the policy
/// is memoryless. Each decision is an independent forward pass with no
/// recurrence, so anything the agent needs to remember has to be in what it
/// reads - and "am I going in circles" is exactly that.
///
/// Leaving it out was a real, measured defect rather than an omission. The
/// scripted teacher uses both of these fields, so cloning it taught the policy
/// a mapping that does not exist: the same observation, with the teacher
/// choosing differently depending on history the policy could not see. And a
/// GREEDY policy without them cannot leave a loop at all - it is deterministic,
/// so if the best action in a state returns it to that state, it takes the
/// same action forever. Measured: sampled rollouts averaged +7.19 while the
/// greedy evaluation of the same weights scored +3.20, which is that loop.
#[derive(Clone, Copy, Debug, Default)]
pub struct History {
    /// Decisions in a row that moved the player nowhere.
    pub stuck: u32,
    /// Times this patch of floor has been entered this episode, including now.
    pub visits_here: u32,
    /// Distinct patches entered this episode.
    pub patches: usize,
}

/// The observation as the model sees it.
///
/// Kept SHORT on purpose. The encoder reads a bounded span, so every line that
/// is always the same is a line that crowds out one that varies - and what
/// varies here is the threats, the clearances and the recent events.
pub fn render(state: &State, history: History) -> String {
    let mut out = String::new();
    let p = &state.player;

    out.push_str(&format!(
        "health {} armor {}, {} with {} rounds",
        p.health,
        p.armor,
        p.weapon.as_deref().unwrap_or("nothing"),
        p.ammo.unwrap_or(0)
    ));
    if !p.keys.is_empty() {
        out.push_str(&format!(", carrying {}", p.keys.join(" and ")));
    }
    out.push_str(&format!(
        ". Killed {} of {} enemies, {} of {} items, {} of {} secrets.\n",
        state.level.kills,
        state.level.total_kills,
        state.level.items,
        state.level.total_items,
        state.level.secrets,
        state.level.total_secrets
    ));

    if state.player.standing_in_damage {
        out.push_str("THE FLOOR HERE IS BURNING YOU - get off it.\n");
    }

    let vis: Vec<&Thing> = state.visible_threats().take(3).collect();
    if vis.is_empty() {
        let lurking = state.threats.len();
        if lurking > 0 {
            out.push_str(&format!("No enemy in sight, {lurking} somewhere beyond the walls.\n"));
        } else {
            out.push_str("No enemy anywhere near.\n");
        }
    } else {
        out.push_str("In sight: ");
        for (i, t) in vis.iter().enumerate() {
            if i > 0 {
                out.push_str("; ");
            }
            out.push_str(&format!(
                "{} {} at {} units {}",
                t.kind.to_lowercase(),
                range_word(t.distance),
                t.distance,
                side_word(t.bearing)
            ));
            // Whether a target is nearly dead changes which one to shoot next,
            // and it is the kind of thing that is obvious on screen and absent
            // from a state that reports only positions.
            if let Some(h) = t.health {
                if h <= 20 {
                    out.push_str(", nearly dead");
                }
            }
            if t.targeting_me == Some(true) {
                out.push_str(", coming for you");
            }
        }
        out.push_str(".\n");
    }

    if let Some(b) = state.hazards.iter().find(|b| b.visible && b.distance < 400) {
        // A barrel beside a monster is a free kill and beside the player is a
        // way to die, so where they are belongs in the state.
        out.push_str(&format!(
            "An explosive {} sits {} units {}.\n",
            b.kind.to_lowercase(),
            b.distance,
            side_word(b.bearing)
        ));
    }

    if let Some(pick) = state.pickups.iter().find(|i| i.visible) {
        out.push_str(&format!(
            "A {} lies {} units {}.\n",
            pick.kind.to_lowercase(),
            pick.distance,
            side_word(pick.bearing)
        ));
    }

    let c = &state.clearance;
    out.push_str(&format!(
        "Room to move: {} ahead, {} left, {} right, {} behind.\n",
        c.ahead, c.left, c.right, c.behind
    ));

    if let Some(e) = &state.exit {
        // The ROUTE, when the engine could compute one - how far the exit is
        // along ground a player can walk, and which way to set off. The
        // straight-line bearing is deliberately not shown alongside it: it
        // points through walls, and an agent given both has to learn which of
        // two contradictory numbers to believe.
        match (e.path_distance, e.route_bearing) {
            (Some(path), Some(bearing)) => out.push_str(&match &e.goal {
                Some(key) => format!(
                    "The way out is LOCKED and needs the {} key. The key is {} units of \
                     walking away, and the way there starts {}.\n",
                    key.to_uppercase(),
                    path,
                    side_word(bearing)
                ),
                None => format!(
                    "The exit {} is {} units of walking away, and the way there starts {}.\n",
                    if e.kind == "switch" { "switch" } else { "line" },
                    path,
                    side_word(bearing)
                ),
            }),
            _ => out.push_str(&format!(
                "The exit {} is {} units away as the crow flies, {}, with {} units of clear \
                 floor that way.\n",
                if e.kind == "switch" { "switch" } else { "line" },
                e.distance,
                side_word(e.bearing),
                e.clearance
            )),
        }
        if let Some(b) = &e.blocked_by {
            out.push_str(&match b.kind.as_str() {
                "door" => format!(
                    "A SHUT DOOR is in the way, {} units {} - open it.\n",
                    b.distance,
                    side_word(b.bearing)
                ),
                _ => format!(
                    "A wall is in the way {} units {} - the route goes round it.\n",
                    b.distance,
                    side_word(b.bearing)
                ),
            });
        }
    }

    // Where the agent has been. See `History` for why this is not optional.
    out.push_str(&match history.visits_here {
        0 | 1 => "This is new ground.".to_string(),
        2..=4 => format!("You have been through here {} times.", history.visits_here),
        n => format!("You keep coming back here - {n} times now, and it is wearing thin."),
    });
    if history.stuck >= 2 {
        out.push_str(&format!(
            " You have not actually moved for {} decisions.",
            history.stuck
        ));
    }
    out.push_str(&format!(" {} patches of this level explored.\n", history.patches));

    if !state.events.is_empty() {
        let mut parts: Vec<String> = Vec::new();
        for e in state.events.iter().take(4) {
            parts.push(match e.kind.as_str() {
                "hurt" => format!("took {} damage", e.amount),
                "heal" => format!("healed {}", e.amount),
                "kill" => format!("killed {}", e.amount),
                "item" => "picked something up".to_string(),
                "armor" => format!("gained {} armor", e.amount),
                "ammo" => format!("found {}", e.what.as_deref().unwrap_or("ammo")),
                "weapon" => format!("found a {}", e.what.as_deref().unwrap_or("weapon")),
                "key" => format!("found the {}", e.what.as_deref().unwrap_or("key")),
                "secret" => "found a secret".to_string(),
                "death" => "died".to_string(),
                "exit" => "left the level".to_string(),
                other => other.to_string(),
            });
        }
        out.push_str(&format!("Just now: {}.", parts.join(", ")));
    } else {
        out.push_str("Nothing happened in the last moment.");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"tic":100,"episodeTic":40,"level":{"episode":1,"map":1,"skill":2,
      "tic":40,"kills":1,"totalKills":4,"items":0,"totalItems":37,"secrets":0,"totalSecrets":3},
      "player":{"id":0,"health":80,"armor":0,"x":10,"y":20,"angle":90,"weapon":"pistol",
      "ammo":42,"keys":[]},"threats":[{"id":1,"type":"IMP","distance":150,"bearing":-12,"visible":true,
      "health":60,"targetingMe":true}],"hazards":[],"pickups":[],"clearance":{"ahead":320,
      "right":64,"behind":0,"left":128,"aheadRight":320,"aheadLeft":64},
      "exit":{"distance":900,"bearing":30,"kind":"switch","clearance":128,"spot":null,
      "pathDistance":1400,"routeBearing":-40,"routeDistance":90,"routeClearance":200},
      "unexplored":null,"events":[{"tic":39,"type":"hurt",
      "what":null,"amount":15}],"done":false,"outcome":"alive"}"#;

    #[test]
    fn a_state_round_trips_and_renders_what_matters() {
        let s = State::parse(SAMPLE).expect("parses");
        assert_eq!(s.player.health, 80);
        assert_eq!(s.visible_threats().count(), 1);
        // Bearing is relative, so facing the imp is the player's own angle plus
        // its bearing - the arithmetic the observation exists to avoid making
        // the policy learn.
        assert_eq!(s.facing(-12), 78);

        let text = render(&s, History::default());
        for needle in ["health 80", "imp", "12 degrees left", "coming for you", "took 15 damage"] {
            assert!(text.contains(needle), "rendered text is missing {needle:?}:\n{text}");
        }
    }

    #[test]
    fn a_field_the_engine_stopped_sending_is_an_error_not_a_default() {
        // The failure this guards is silent: if `health` became Option<i32>
        // with a default, an engine that stopped reporting it would train a
        // policy on a player who is permanently at zero health, and every
        // number in the run would still look plausible.
        let missing = SAMPLE.replace("\"health\":80,", "");
        assert!(State::parse(&missing).is_err(), "a missing required field must not parse");

        let extra = SAMPLE.replace("\"tic\":100,", "\"tic\":100,\"newField\":7,");
        assert!(State::parse(&extra).is_err(), "an unknown field must not be ignored");
    }
}
