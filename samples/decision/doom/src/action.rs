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
    /// How far there is to walk the way this option goes, for the options that
    /// go somewhere; 0 for the ones that do not.
    ///
    /// Carried on the option rather than looked up again by whoever needs it.
    /// The scripted player used to recover it by scanning the option's own
    /// TEXT for the first integer, which worked only because no monster in
    /// DOOM has a digit in its name.
    pub room: i32,
    /// How far along this option's direction the floor starts burning, when
    /// it does, and `None` when the way is clear.
    ///
    /// The model reads this in the text. The teacher cannot read text without
    /// scraping it - which is exactly what `room` exists to stop - and floor
    /// damage is the single commonest way a run ends: three of nine levels
    /// killed the scripted player in nukage or hellslime, one of them after
    /// choosing "get off the burning floor" 189 times and still drowning in
    /// it. Getting OUT is a poor second to not walking in.
    pub fire: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tag {
    Attack,
    /// Move sideways without changing facing.
    Sidestep,
    Grab,
    Advance,
    Explore,
    Retreat,
    Use,
    /// Open the specific thing the route is blocked by.
    Open,
    /// Go to, and press, the switch that opens what the route is blocked by.
    Switch,
    /// Put a different weapon in hand.
    Arm,
    /// Fire while moving across the line of fire.
    Circle,
    /// Remove whatever is standing in the way of the route.
    Clear,
    /// Get off floor that is burning the player.
    Escape,
    /// Go back the way you came, along ground already walked.
    Fallback,
    /// Go back for a monster seen earlier and never dealt with.
    ///
    /// Distinct from [`Tag::Attack`], which fights what is in front of the
    /// player right now. A level is not cleared by fighting what comes to
    /// you: the last monsters of a Max run are the ones glimpsed once from a
    /// doorway, and going back for them is a decision nothing here could
    /// express before.
    Hunt,
    /// Turn to face a wall an arm's length away that has not been tried.
    ///
    /// The half of searching for a hidden door that was missing. `use`
    /// reaches 64 units and only what the player FACES, and a player walks
    /// corridors facing along them - so the walls are beside them the whole
    /// time and never in front. Measured with only the pushing half: a
    /// campaign tested nineteen walls in seven minutes.
    Face,
    /// Go and push on the walls of a room this run has never searched.
    ///
    /// The only mechanism that finds a DOOM secret without being told where
    /// one is, made into a DESTINATION rather than a reflex. Pressing use
    /// wherever the player happens to be standing is a coin flip against a
    /// level's worth of wall; going to a room whose walls this run has not
    /// tested, and testing them, is a search. See `memory::Sweep`.
    Frisk,
    /// Operate the lift the route runs through, and wait for it.
    Ride,
    Exit,
}

/// A decision every ~4-8 tics is about 5-9 per second of game time, which is
/// the rate a human plays at and slow enough that a decision model with a
/// millisecond of latency is never the thing holding the game up.
/// How long the player waits on a lift after calling it.
///
/// A DOOM lift takes a little over three seconds to come down, wait and go
/// back up. Deciding again every few tics while standing on one is a decision
/// about nothing; this is long enough for the platform to actually move.
const RIDE_TICS: u32 = 64;
/// How long an attack decision lasts.
///
/// Measured against the engine's own weapon timing rather than chosen: the
/// pistol's cycle is about sixteen tics, so a four-tic attack asked the game
/// to fire four times as often as any weapon can and got one shot in four.
/// An option called "attack" that usually does not attack is an option the
/// policy cannot learn the meaning of.
///
/// Faster weapons fire more than once inside it, which is correct - the
/// trigger is held and the weapon's own state machine decides.
const FIGHT_TICS: u32 = 16;
/// Tics to hold while a weapon is lowered and the next one raised.
const ARM_TICS: u32 = 18;
const MOVE_TICS: u32 = 6;
/// Less floor than this is not somewhere to walk: the player is 32 units wide
/// and covers about 40 in one decision, so under 64 is a step into a wall.
const MIN_ROOM: i32 = 64;
/// How much open floor makes running worth the loss of control. Two corridor
/// widths: enough that the extra ground covered is not immediately spent
/// overshooting whatever is at the end of it.
const RUN_ROOM: i32 = 256;
const CIRCLE_TARGETS: usize = 2;
/// How far `use` reaches, in map units - DOOM's own USERANGE.
///
/// Shared with the wall ledger, which counts a push as having tested
/// something only when there was something within this distance to test.
/// See `memory::Memory::wall_in_reach`.
pub const USE_RANGE: i32 = 64;
/// Close enough to "facing it" that turning again would waste a decision.
const FACING_TOL: i32 = 20;
/// The same, for anything that is going to be pressed rather than walked at.
///
/// `use` is a RAY sixty-four units long cast along the player's facing, so a
/// few degrees is a miss: twenty degrees off a door fifty-six units away is
/// eleven units to one side, which is past the end of a door frame the player
/// is already standing beside. Measured on E1M1, whose exit door the scripted
/// player pressed four hundred and forty-five times in a row from ten units
/// off its west end, with the bearing reading eleven degrees the whole time.
const AIM_TOL: i32 = 4;
/// How long to stand still while a door rises.
///
/// A DOOM door opens at 2 units a tic, so a head-high one needs about thirty.
/// Pressing use again before it is up re-triggers it and sends it back DOWN,
/// which is what a follower did four times in a row at E1M1's first door while
/// reporting the ceiling at 0, 42, 0, 42.
const DOOR_TICS: u32 = 30;

/// How long to hold `forward` to cover a given distance, in tics.
///
/// A standing start covers about 22 units in eight tics and more than that
/// once the player is already moving. A fixed stride past a waypoint 21 units
/// away lands the player in the next cell along, whose route points back at
/// the one just left - measured on E1M1, a follower bouncing between two cells
/// 45 units apart for the rest of the episode. Walking the distance that is
/// actually there is what a player does and it does not oscillate.
fn walk_tics(distance: i32) -> u32 {
    (distance / 8).clamp(2, MOVE_TICS as i32) as u32
}

fn json_turn(angle: i32) -> String {
    format!(
        "{{\"type\":\"turn-to\",\"angle\":{}}}",
        angle.rem_euclid(360)
    )
}

/// Build the option list for this state.
///
/// Always non-empty: the environment contract requires it while an episode is
/// live, and a state with nothing to do would end a rollout for a reason that
/// is not the game's.
pub fn options(state: &State) -> Vec<Option_> {
    let mut out: Vec<Option_> = Vec::new();
    let c = &state.clearance;

    // --- arm --------------------------------------------------------------
    //
    // A player picks the gun for the fight. Without this the agent fights
    // whatever the last pickup left in its hands: walking over a shotgun
    // switches to it, and running the shells out strands it on a pistol with
    // no way back. Measured before this existed, three of nine levels ended
    // with a former human sergeant - a shotgunner - killing it.
    //
    // Only weapons it is not already holding, and only ones that can fire.
    // Offering a switch to the empty gun you are already carrying is an
    // option that does nothing, and every one of those costs a decision.
    for w in &state.player.weapons {
        if Some(w.name.as_str()) == state.player.weapon.as_deref() || !w.loaded() {
            continue;
        }
        out.push(Option_ {
            text: format!("draw the {}", w.name),
            commands: format!(
                "[{{\"type\":\"switch-weapon\",\"amount\":{}}}]",
                w.slot
            ),
            // Long enough for the lowering and raising animation to finish,
            // or the next decision is taken mid-swap with nothing in hand.
            tics: ARM_TICS,
            tag: Tag::Arm,
            room: 0,
            fire: None,
        });
    }

    // --- fight ------------------------------------------------------------
    // How many of the things shooting at you get a circle-strafe pair. Every
    // option costs a decision its share of attention and the encoder its
    // share of time, and a room with six shooters in it would otherwise put
    // twelve near-identical sentences in front of the model.
    let mut circling = 0usize;
    for t in state.threats_in_view() {
        let facing = state.facing(t.bearing);
        // Naming the one already being fought is what makes finishing it
        // choosable. The two options are otherwise the same sentence with a
        // different bearing in it.
        let already = if state.wounded.contains(&t.id) {
            ", which you have wounded"
        } else {
            ""
        };
        out.push(Option_ {
            text: format!(
                "attack the {} {} units away, {}{}",
                t.kind.to_lowercase(),
                t.distance,
                bearing_phrase(t.bearing),
                already
            ),
            // Turn onto it and fire in the same step: the turn servo closes
            // the angle over the step's tics, so firing after it is aimed at
            // where the thing is now rather than where it was.
            commands: format!("[{},{{\"type\":\"shoot\"}}]", json_turn(facing)),
            tics: FIGHT_TICS,
            tag: Tag::Attack,
            room: 0,
            fire: None,
        });

        // Circle-strafing: keep firing while crossing the line of fire.
        //
        // The technique every DOOM player uses and the one the option list
        // had no way to express - `sidestep` exists but stops shooting, and
        // `attack` stands still while a hitscanner shoots back. An imp's
        // fireball travels, so moving across its line beats it; a sergeant
        // hits instantly, so the value is being somewhere else by the time
        // the next shot comes. Offered only for the things actually shooting
        // at you, and only to a side with room to move, because a circle
        // strafe into a wall is just standing still while being shot.
        if t.targeting_me == Some(true) && circling < CIRCLE_TARGETS {
            circling += 1;
            for (name, key, room) in [
                ("left", "strafe-left", c.left),
                ("right", "strafe-right", c.right),
            ] {
                if room < MIN_ROOM {
                    continue;
                }
                out.push(Option_ {
                    text: format!(
                        "attack the {} {} units away, {}, sidestepping {name} as you fire",
                        t.kind.to_lowercase(),
                        t.distance,
                        bearing_phrase(t.bearing)
                    ),
                    commands: format!(
                        "[{},{{\"type\":\"{key}\",\"amount\":{FIGHT_TICS}}},{{\"type\":\"shoot\"}}]",
                        json_turn(facing)
                    ),
                    tics: FIGHT_TICS,
                    tag: Tag::Circle,
                    fire: None,
                    room,
                });
            }
        }
    }

    // --- take something ---------------------------------------------------
    // The nearest one WORTH taking, not simply the nearest. An item the
    // engine will refuse to hand over never leaves the floor, so an option
    // chosen because it is there stays choosable for ever - see
    // `State::worth_taking`.
    if let Some(p) = state
        .pickups
        .iter()
        .find(|p| p.visible && p.distance < 700 && state.worth_taking(p))
    {
        // FACE FIRST, THEN WALK - the same split the exit option makes, and
        // for the same reason. Turning while walking walks the ARC of the
        // turn, so a pickup 100 degrees off the nose is approached along a
        // curve that ends against the wall on the inside of it. Measured on
        // E1M3, where the level starts with three pickups around the player:
        // the scripted player spent its whole episode within 60 units of its
        // own spawn, revisiting the same patch fourteen times, and killed
        // nothing in eight hundred decisions.
        let far_off = p.bearing.abs() > FACING_TOL;
        out.push(Option_ {
            text: if far_off {
                format!(
                    "turn toward the {} {} units away, {}",
                    p.kind.to_lowercase(),
                    p.distance,
                    bearing_phrase(p.bearing)
                )
            } else {
                format!(
                    "go and pick up the {} {} units away, straight ahead",
                    p.kind.to_lowercase(),
                    p.distance
                )
            },
            commands: if far_off {
                format!("[{}]", json_turn(state.facing(p.bearing)))
            } else {
                "[{\"type\":\"forward\",\"amount\":8}]".into()
            },
            tics: if far_off { FIGHT_TICS } else { MOVE_TICS },
            tag: Tag::Grab,
            room: p.distance,
            fire: None,
        });
    }

    // Something seen a moment ago and no longer in view. Without this the
    // only way anything is ever collected is by walking into it: the option
    // to go and get a medikit disappears in the same decision the player
    // stops looking at it. Items only - a monster has moved since, and
    // "go to where it was" is an invitation to walk into where it is not.
    //
    // Routed when the engine could find a way there, which is the difference
    // between going back for something and walking at the wall between here
    // and it. A route is a path over ground the player can walk, so it needs
    // no clearance gate - the same reasoning as `route_usable`, and for the
    // same reason: the straight ray is shortest exactly at the corner where
    // the route is most worth following.
    //
    // Without one there is only the straight line, and that one IS gated:
    // measured on health-gathering-supreme, offering it unguarded cost the
    // scripted player three extra deaths in twenty-four episodes.
    for r in state
        .recalled
        .iter()
        .filter(|r| r.class != crate::memory::Class::Threat)
        .filter(|r| state.worth_taking_kind(&r.kind))
        .filter(|r| r.path.is_some() || c.toward(r.bearing) >= r.distance.min(300))
        .take(2)
    {
        let (bearing, room, leg, walk) = match r.path {
            // As far as the next waypoint, not the whole way: the bearing
            // points a few cells along and the path turns after that.
            Some(p) => (
                p.bearing,
                p.clearance,
                p.step.max(1),
                format!(", {} units of walking", p.distance),
            ),
            None => (r.bearing, r.distance, r.distance, String::new()),
        };
        // Face first, then walk, as everything else that goes somewhere
        // does. Turning while walking walks the ARC of the turn, and this is
        // the option that turns furthest: a thing remembered behind you is a
        // hundred and forty degrees off the nose, so the arc is a half circle
        // into whatever is beside you. Measured on E1M1 at the easiest
        // difficulty with four thousand decisions, the player alternated
        // between this option and the way out for the whole episode, arriving
        // at neither and ending where it started.
        let far_off = bearing.abs() > FACING_TOL;
        out.push(Option_ {
            text: if far_off {
                format!(
                    "turn back toward the {} you saw, {} units {}{walk}",
                    r.kind.to_lowercase(),
                    r.distance,
                    bearing_phrase(r.bearing)
                )
            } else {
                format!(
                    "go back for the {} you saw, {} units ahead{walk}",
                    r.kind.to_lowercase(),
                    r.distance
                )
            },
            commands: if far_off {
                format!("[{}]", json_turn(state.facing(bearing)))
            } else {
                format!("[{{\"type\":\"forward\",\"amount\":{}}}]", walk_tics(leg))
            },
            tics: if far_off { FIGHT_TICS } else { walk_tics(leg) },
            tag: Tag::Grab,
            fire: None,
            room,
        });
    }

    // --- hunt -------------------------------------------------------------
    //
    // Go back for a monster seen earlier and never dealt with. The option
    // that "kill everything" needs and that nothing here could express: every
    // other way of fighting reacts to what is in front of the player, and a
    // level is not cleared by what walks into you. The last few monsters of a
    // Max run are the ones glimpsed once from a doorway.
    //
    // Fair play holds. Nothing here is a monster the agent has not already
    // been shown - the ledger is built from observations it was given, the
    // same way `recalled` is - and it is a ROOM rather than a position,
    // because a monster has moved since and claiming otherwise would be the
    // lie the item memory is careful not to tell.
    for r in state.unfinished.iter().take(2) {
        let (bearing, room, leg, walk) = match r.path {
            Some(p) => (
                p.bearing,
                p.clearance,
                p.step.max(1),
                format!(", {} units of walking", p.distance),
            ),
            None => (r.bearing, r.distance, r.distance, String::new()),
        };
        // Withheld when there is no route and no clear line either, for the
        // same reason the recall options are: an option aimed through a wall
        // is one the agent takes and does not move.
        if r.path.is_none() && c.toward(r.bearing) < r.distance.min(300) {
            continue;
        }
        let far_off = bearing.abs() > FACING_TOL;
        out.push(Option_ {
            text: if far_off {
                format!(
                    "go back and hunt down the {} you saw earlier, {} units {}{walk}",
                    r.kind.to_lowercase(),
                    r.distance,
                    bearing_phrase(r.bearing)
                )
            } else {
                format!(
                    "go back and hunt down the {} you saw earlier, {} units ahead{walk}",
                    r.kind.to_lowercase(),
                    r.distance
                )
            },
            commands: if far_off {
                format!("[{}]", json_turn(state.facing(bearing)))
            } else {
                format!("[{{\"type\":\"forward\",\"amount\":{}}}]", walk_tics(leg))
            },
            tics: if far_off { FIGHT_TICS } else { walk_tics(leg) },
            tag: Tag::Hunt,
            fire: None,
            room,
        });
    }

    // A wall an arm's length away that this run has not pushed on, and the
    // turn that would put it in front of the player.
    //
    // `use` reaches 64 units and only what the player faces. A player walks
    // corridors facing along them, so the walls are beside them the whole
    // way and never in front - which is why a search that could only push at
    // what it happened to face tested nineteen walls in seven minutes.
    //
    // Fair play holds, and the line is the same one the room frontier is on:
    // the clearance readings are the agent's own and the ledger is its own.
    // Neither says a secret is behind any of these. They say there is a wall
    // there and this run has not tried it.
    let secrets_left = state.level.secrets < state.level.total_secrets;
    for bearing in state.untried_walls.iter().take(2).filter(|_| secrets_left) {
        out.push(Option_ {
            text: format!(
                "turn to face the wall {} you and try pushing on it",
                bearing_phrase(*bearing)
            ),
            commands: format!("[{}]", json_turn(state.facing(*bearing))),
            tics: FIGHT_TICS,
            tag: Tag::Face,
            room: c.toward(*bearing),
            fire: None,
        });
    }

    // Somewhere to go and search for a secret.
    //
    // Fair play, and the line is worth being exact about: this is built from
    // rooms the player has STOOD IN and a tally of pushes the player itself
    // made. Nothing in it comes from the level's secret sectors, and nothing
    // in it says a secret is there - only that this run has not looked. A
    // level has a great many walls and no observation distinguishes the one
    // that opens, so what a player does is go somewhere they have not tried
    // and try it, which is what this offers.
    //
    // Only when there is a secret left to find: the counter is on the
    // player's own status bar, so knowing "0 of 3" is not being told
    // anything, and offering a wall search on a level with none left is
    // spending decisions on nothing.
    for r in state.unfrisked.iter().take(1).filter(|_| secrets_left) {
        let (bearing, room, leg, walk) = match r.path {
            Some(p) => (
                p.bearing,
                p.clearance,
                p.step.max(1),
                format!(", {} units of walking", p.distance),
            ),
            None => (r.bearing, r.distance, r.distance, String::new()),
        };
        // Withheld with no route and no clear line, for the same reason the
        // hunt and recall options are: a heading through a wall is an option
        // the agent takes and does not move.
        if r.path.is_none() && c.toward(r.bearing) < r.distance.min(300) {
            continue;
        }
        let far_off = bearing.abs() > FACING_TOL;
        out.push(Option_ {
            text: if far_off {
                format!(
                    "go and search the walls of a room you have not searched yet, {} units {}{walk}",
                    r.distance,
                    bearing_phrase(r.bearing)
                )
            } else {
                format!(
                    "go and search the walls of a room you have not searched yet, {} units ahead{walk}",
                    r.distance
                )
            },
            commands: if far_off {
                format!("[{}]", json_turn(state.facing(bearing)))
            } else {
                format!("[{{\"type\":\"forward\",\"amount\":{}}}]", walk_tics(leg))
            },
            tics: if far_off { FIGHT_TICS } else { walk_tics(leg) },
            tag: Tag::Frisk,
            fire: None,
            room,
        });
    }

    // --- move -------------------------------------------------------------
    //
    // Every way of moving says whether it walks into burning floor, and from
    // how far. The observation reports that per direction, but an agent then
    // has to work out for itself which option goes which way - and it is the
    // one thing on a level like E1M3 that reliably ends a run, so it belongs
    // in the option rather than beside it. A player does not deduce that the
    // corridor ahead is nukage; they look at it.
    // Whether walking a given way, for a given distance, goes into any patch
    // of burning floor the player can see.
    let fire = |bearing: i32, distance: i32| {
        state
            .burning_floor
            .iter()
            .filter(|b| b.in_the_way(bearing, distance))
            .min_by_key(|b| b.distance)
            .map(|b| b.distance)
    };
    let hot = |bearing: i32, distance: i32| match fire(bearing, distance) {
        Some(d) => format!(", and the floor starts burning {d} units along"),
        None => String::new(),
    };
    if c.ahead >= MIN_ROOM {
        out.push(Option_ {
            text: format!(
                "walk forward, {} units of open floor ahead{}",
                c.ahead,
                hot(0, c.ahead)
            ),
            commands: "[{\"type\":\"forward\",\"amount\":8}]".into(),
            tics: MOVE_TICS,
            tag: Tag::Advance,
            room: c.ahead,
            fire: fire(0, c.ahead),
        });
    }
    if c.ahead_left >= MIN_ROOM {
        out.push(Option_ {
            text: format!(
                "turn left and go that way, {} units of room{}",
                c.ahead_left,
                hot(-45, c.ahead_left)
            ),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":8}}]",
                json_turn(state.facing(-45))
            ),
            tics: MOVE_TICS,
            tag: Tag::Explore,
            room: c.ahead_left,
            fire: fire(-45, c.ahead_left),
        });
    }
    if c.ahead_right >= MIN_ROOM {
        out.push(Option_ {
            text: format!(
                "turn right and go that way, {} units of room{}",
                c.ahead_right,
                hot(45, c.ahead_right)
            ),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":8}}]",
                json_turn(state.facing(45))
            ),
            tics: MOVE_TICS,
            tag: Tag::Explore,
            room: c.ahead_right,
            fire: fire(45, c.ahead_right),
        });
    }
    if c.behind >= MIN_ROOM {
        out.push(Option_ {
            text: format!(
                "back away from whatever is in front of you{}",
                hot(180, c.behind)
            ),
            commands: "[{\"type\":\"backward\",\"amount\":8}]".into(),
            tics: MOVE_TICS,
            tag: Tag::Retreat,
            room: c.behind,
            fire: fire(180, c.behind),
        });
    }

    // --- get off the burning floor ----------------------------------------
    //
    // First in the list because nothing else matters while health is draining
    // for free. Aimed at the edge of the pool, which is a thing the player can
    // see and the route cannot help with: the route is pointed wherever the
    // run is going, and on a level like E1M3 that is across more of it.
    if let Some(d) = state
        .player
        .dry_land
        .as_ref()
        .filter(|_| state.player.standing_in_damage)
    {
        out.push(Option_ {
            text: format!(
                "get off the burning floor, {} units {}",
                d.distance,
                bearing_phrase(d.bearing)
            ),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":{}}}]",
                json_turn(state.facing(d.bearing)),
                walk_tics(d.distance)
            ),
            tics: walk_tics(d.distance),
            tag: Tag::Escape,
            room: d.distance,
            fire: None,
        });
    }

    // --- the exit ---------------------------------------------------------
    //
    // Offered when there is enough room to COVER THE DISTANCE, which is not
    // the same as a fixed amount of room. Requiring MIN_ROOM unconditionally
    // hid this option exactly when the player was standing next to the exit -
    // 32 units away with 40 units of clearance - so an agent placed at the
    // goal was not offered the goal.
    if let Some(e) = state.exit.as_ref().filter(|e| route_usable(e)) {
        // Prefer the ROUTE bearing where the engine could compute one: it
        // points down the corridor rather than at the wall the exit is behind.
        let Some((bearing, away)) = (match (e.route_bearing, e.path_distance) {
            (Some(b), Some(d)) => Some((b, d)),
            _ => e.bearing.zip(e.distance),
        }) else {
            return out;
        };

        // The way on is a LIFT: the floor beyond is higher than a player
        // climbs and something in the level moves it. Pressing use calls it
        // down, and then the thing to do is wait - which is the one situation
        // where standing still is progress, and the one an agent offered only
        // "walk that way" cannot express. It holds forward against a wall
        // that was about to come down for it, and from outside that is
        // indistinguishable from being stuck.
        if e.route_is_lift == Some(true) {
            out.push(Option_ {
                text: format!(
                    "operate the lift {bearing_text} and wait for it - the way on is \
                     up it, not past it",
                    bearing_text = bearing_phrase(bearing)
                ),
                commands: format!("[{},{{\"type\":\"use\"}}]", json_turn(state.facing(bearing))),
                tics: RIDE_TICS,
                tag: Tag::Ride,
                room: e.route_clearance.or(e.clearance).unwrap_or(0),
                fire: None,
            });
        }

        // TURNING AND WALKING ARE SEPARATE DECISIONS.
        //
        // Doing both in one step is what wedged a follower on every corner: it
        // turns over the step's tics while walking, so it walks along the
        // ARC of the turn and into the inside of the corner, and the engine
        // slides it back along the wall it is pressed against. Measured, that
        // is 13% of the way to the exit and then four hundred decisions going
        // nowhere. Facing first and moving second costs one extra decision at
        // each corner and actually gets round it.
        // When the way out is locked, the option that matters is the KEY, and
        // it is a different act from pressing a switch: you walk onto a key,
        // you do not use it. Saying so in the option text is the whole point -
        // the model reads what an option MEANS.
        if let Some(goal) = &e.goal {
            // What the route leads to, said plainly. The model reads what an
            // option MEANS, so an option that says "the exit" while pointing
            // at a keycard, a switch or a corridor nobody has walked down is
            // a lie it has no way to catch.
            let what = match goal.as_str() {
                "unexplored" => "ground nobody has looked at yet".to_string(),
                "switch" => "the switch that opens the way on".to_string(),
                key => format!("the {key} key that unlocks the way out"),
            };
            let text = if bearing.abs() > FACING_TOL {
                format!(
                    "turn toward {what}, {away} units of walking {}",
                    bearing_phrase(bearing)
                )
            } else {
                format!("go to {what}, {away} units of walking ahead")
            };
            let commands = if bearing.abs() > FACING_TOL {
                format!("[{}]", json_turn(state.facing(bearing)))
            } else {
                "[{\"type\":\"forward\",\"amount\":8}]".into()
            };
            out.push(Option_ {
                text,
                commands,
                tics: if bearing.abs() > FACING_TOL {
                    FIGHT_TICS
                } else {
                    walk_tics(away)
                },
                tag: Tag::Exit,
                room: e.route_clearance.or(e.clearance).unwrap_or(0),
                fire: fire(bearing, away),
            });
        } else if e.distance.is_some_and(|d| d <= USE_RANGE) {
            // Close enough to press. Face the exit ITSELF, not the route,
            // which has already delivered the player here.
            out.push(Option_ {
                text: format!(
                    "press the level exit, {} units {}",
                    e.distance.unwrap_or(0),
                    bearing_phrase(e.bearing.unwrap_or(0))
                ),
                commands: format!(
                    "[{},{{\"type\":\"use\"}}]",
                    json_turn(state.facing(e.bearing.unwrap_or(0)))
                ),
                tics: FIGHT_TICS,
                tag: Tag::Exit,
                room: e.route_clearance.or(e.clearance).unwrap_or(0),
                fire: fire(bearing, away),
            });
        } else if bearing.abs() > FACING_TOL {
            out.push(Option_ {
                text: format!(
                    "turn to face the way out, the exit is {away} units of walking {}",
                    bearing_phrase(bearing)
                ),
                commands: format!("[{}]", json_turn(state.facing(bearing))),
                tics: FIGHT_TICS,
                tag: Tag::Exit,
                room: e.route_clearance.or(e.clearance).unwrap_or(0),
                fire: fire(bearing, away),
            });
        } else {
            // As far as the next waypoint, not a fixed stride.
            let step = e.route_distance.unwrap_or(away);
            out.push(Option_ {
                text: format!("head for the level exit, {away} units of walking ahead"),
                commands: "[{\"type\":\"forward\",\"amount\":8},{\"type\":\"use\"}]".into(),
                tics: walk_tics(step),
                tag: Tag::Exit,
                room: e.route_clearance.or(e.clearance).unwrap_or(0),
                fire: fire(bearing, away),
            });
        }
    }

    // --- somewhere to go when the route has nothing ------------------------
    //
    // The engine reports the nearest ground nobody has looked at whether or
    // not it can route to a goal, and when it cannot route at all that is the
    // only navigation anything has. It was parsed and read by nothing, so on
    // a level whose route field cannot be built the agent was handed no goal
    // of any kind and had to find its way by clearance alone.
    if state.exit.is_none() {
        if let Some(f) = state.unexplored.filter(|f| f.distance > 0) {
            let far_off = f.bearing.abs() > FACING_TOL;
            out.push(Option_ {
                text: if far_off {
                    format!(
                        "turn toward ground nobody has looked at, {} units {}",
                        f.distance,
                        bearing_phrase(f.bearing)
                    )
                } else {
                    format!(
                        "go to ground nobody has looked at, {} units ahead",
                        f.distance
                    )
                },
                commands: if far_off {
                    format!("[{}]", json_turn(state.facing(f.bearing)))
                } else {
                    "[{\"type\":\"forward\",\"amount\":8}]".into()
                },
                tics: if far_off { FIGHT_TICS } else { MOVE_TICS },
                tag: Tag::Explore,
                room: f.clearance,
                fire: fire(f.bearing, f.distance),
            });
        }
    }

    // --- fall back the way you came ---------------------------------------
    //
    // Backing AWAY aims at nothing: it walks opposite whatever is in front,
    // which in a room with four monsters converging is as likely to be a wall
    // or a fifth monster as it is a way out. This aims at floor the player
    // has stood on, so it is the one direction it is certain it can go - and
    // in a level built of rooms joined by corridors that is the corridor,
    // where things arrive one at a time instead of all at once.
    if let Some((bearing, away)) = state.came_from {
        let far_off = bearing.abs() > FACING_TOL;
        out.push(Option_ {
            text: if far_off {
                format!(
                    "turn back the way you came, {away} units {}",
                    bearing_phrase(bearing)
                )
            } else {
                format!("fall back the way you came, {away} units behind you")
            },
            commands: if far_off {
                format!("[{}]", json_turn(state.facing(bearing)))
            } else {
                "[{\"type\":\"forward\",\"amount\":8}]".into()
            },
            tics: if far_off { FIGHT_TICS } else { MOVE_TICS },
            tag: Tag::Fallback,
            room: away,
            fire: None,
        });
    }

    // --- sidestep ----------------------------------------------------------
    //
    // Strafing moves without turning, which is the only way off a corner the
    // player is pressed into: turning to face the open side and walking turns
    // back INTO the corner on the way round.
    if c.left >= MIN_ROOM {
        out.push(Option_ {
            text: format!(
                "sidestep left without turning, {} units of room{}",
                c.left,
                hot(-90, c.left)
            ),
            commands: "[{\"type\":\"strafe-left\",\"amount\":8}]".into(),
            tics: MOVE_TICS,
            tag: Tag::Sidestep,
            room: c.left,
            fire: fire(-90, c.left),
        });
    }
    if c.right >= MIN_ROOM {
        out.push(Option_ {
            text: format!(
                "sidestep right without turning, {} units of room{}",
                c.right,
                hot(90, c.right)
            ),
            commands: "[{\"type\":\"strafe-right\",\"amount\":8}]".into(),
            tics: MOVE_TICS,
            tag: Tag::Sidestep,
            room: c.right,
            fire: fire(90, c.right),
        });
    }

    // --- the door on the route --------------------------------------------
    //
    // The route runs THROUGH shut doors, because a player opens them. So the
    // engine saying "the way out starts 7 degrees to your right" and the
    // player being unable to walk there is the normal state of affairs at
    // every door in the game, and without an option aimed at the door the
    // agent presses forward against it until the episode ends.
    //
    // Aimed, and in two phases, because `use` reaches 64 units in the
    // direction the player is FACING: the untargeted option below is useless
    // against a door off to one side, which is how a follower stood at
    // E1M1's first door facing 74 degrees away from it.
    if let Some(b) = state.exit.as_ref().and_then(|e| e.blocked_by.as_ref()) {
        // What is in the way opens from somewhere else. Pushing on it does
        // nothing at all - the player has to walk to the switch and press
        // that - so this is a different job from opening a door, and the only
        // one that makes progress. E1M2's way on from the red key is a sector
        // of zero height that a switch a hundred units to one side raises;
        // told it was a wall, a follower shoved at it for six hundred
        // decisions with the switch in plain view.
        // Something standing in the way is shot, not walked round: a barrel
        // explodes, a monster dies, and a decoration is the one case where
        // going round is the answer - which the movement options already
        // offer. Aimed, because a shot goes where the player is facing.
        if b.kind == "thing" && b.alive != Some(true) {
            // A barrel or a lamp. Shooting a barrel forty units away is how
            // the player dies, not how it gets past, so the answer is to
            // step round - sideways, because turning and walking would aim
            // straight back at the thing.
            let what = b.what.as_deref().unwrap_or("it").to_lowercase();
            let (dir, word) = if b.bearing >= 0 {
                ("left", "left")
            } else {
                ("right", "right")
            };
            out.push(Option_ {
                text: format!("step {word} round the {what} in your way"),
                commands: format!("[{{\"type\":\"strafe-{dir}\",\"amount\":8}}]"),
                tics: MOVE_TICS,
                tag: Tag::Clear,
                room: 0,
                fire: None,
            });
        } else if b.kind == "thing" {
            let what = b.what.as_deref().unwrap_or("it").to_lowercase();
            if b.bearing.abs() > AIM_TOL {
                out.push(Option_ {
                    text: format!(
                        "turn to face the {what} blocking your way, {} units {}",
                        b.distance,
                        bearing_phrase(b.bearing)
                    ),
                    commands: format!("[{}]", json_turn(state.facing(b.bearing))),
                    tics: FIGHT_TICS,
                    tag: Tag::Clear,
                    room: 0,
                    fire: None,
                });
            } else {
                out.push(Option_ {
                    text: format!(
                        "shoot the {what} blocking your way, {} units {}",
                        b.distance,
                        bearing_phrase(b.bearing)
                    ),
                    commands: "[{\"type\":\"shoot\"}]".into(),
                    tics: FIGHT_TICS,
                    tag: Tag::Clear,
                    room: 0,
                    fire: None,
                });
            }
        }
        if let Some(sw) = &b.switch {
            if sw.distance > USE_RANGE {
                out.push(Option_ {
                    text: format!(
                        "go to the switch that opens the way out, {} units {}",
                        sw.distance,
                        bearing_phrase(sw.bearing)
                    ),
                    commands: format!(
                        "[{},{{\"type\":\"forward\",\"amount\":{}}}]",
                        json_turn(state.facing(sw.bearing)),
                        walk_tics(sw.distance)
                    ),
                    tics: walk_tics(sw.distance),
                    tag: Tag::Switch,
                    room: sw.distance,
                    fire: None,
                });
            } else if sw.bearing.abs() > AIM_TOL {
                out.push(Option_ {
                    text: format!(
                        "turn to face the switch that opens the way out, {} units {}",
                        sw.distance,
                        bearing_phrase(sw.bearing)
                    ),
                    commands: format!("[{}]", json_turn(state.facing(sw.bearing))),
                    tics: FIGHT_TICS,
                    tag: Tag::Switch,
                    room: 0,
                    fire: None,
                });
            } else {
                out.push(Option_ {
                    text: format!(
                        "press the switch that opens the way out, {} units {}",
                        sw.distance,
                        bearing_phrase(sw.bearing)
                    ),
                    commands: "[{\"type\":\"use\"}]".into(),
                    tics: DOOR_TICS,
                    tag: Tag::Switch,
                    room: 0,
                    fire: None,
                });
            }
        }
        if b.kind == "door" && b.distance <= USE_RANGE {
            if b.bearing.abs() > AIM_TOL {
                out.push(Option_ {
                    text: format!(
                        "turn to face the shut door on the way out, {} units {}",
                        b.distance,
                        bearing_phrase(b.bearing)
                    ),
                    commands: format!("[{}]", json_turn(state.facing(b.bearing))),
                    tics: FIGHT_TICS,
                    tag: Tag::Open,
                    room: 0,
                    fire: None,
                });
            } else {
                out.push(Option_ {
                    text: format!(
                        "open the shut door blocking the way out, {} units {}",
                        b.distance,
                        bearing_phrase(b.bearing)
                    ),
                    commands: "[{\"type\":\"use\"}]".into(),
                    tics: DOOR_TICS,
                    tag: Tag::Open,
                    room: 0,
                    fire: None,
                });
            }
        }
    }

    // --- open what is in the way ------------------------------------------
    //
    // Always offered. A door reads as "no clearance ahead", which is exactly
    // the situation where every movement option has been filtered out, so this
    // is also what keeps the list non-empty in a dead end.
    out.push(Option_ {
        // The sentence says whether the wall looks out of place, because the
        // sentence is what a policy READS. A ledger the search consults is
        // no use to a cloned model: it chooses among these words and nothing
        // else, so a cue it cannot see is a cue it cannot learn. The search
        // reads the same words, so both halves of the loop are looking at
        // the same thing.
        //
        // It is the agent's own judgement from wall faces it has looked at -
        // see `memory::Memory::odd_wall` - not a fact about the level, and
        // it is wrong often. What it buys is an order to search in.
        text: if state.odd_wall {
            "push on the wall directly in front of you - it does not look like the others around here"
                .into()
        } else {
            "push on the wall or door directly in front of you".into()
        },
        commands: "[{\"type\":\"use\"},{\"type\":\"forward\",\"amount\":4}]".into(),
        tics: MOVE_TICS,
        tag: Tag::Use,
        room: 0,
        fire: None,
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
        room: 0,
        fire: None,
    });

    // --- run --------------------------------------------------------------
    //
    // A twin at a run for every option that walks somewhere far enough for it
    // to be worth the loss of control.
    //
    // DOOM's speed key selects the second entry of `forwardmove` - 0x32
    // against 0x19, twice the ground per tic - and every human plays holding
    // it. Without it the agent crossed every level at half pace, and at a
    // fixed decision budget half pace is half the level: running out of
    // decisions is how most of these episodes ended. It stays a CHOICE rather
    // than being folded into walking, because overshooting the doorway you
    // meant to take is exactly why a player lets go of it.
    //
    // Built here, over the finished list, rather than at each of the five
    // places that emit a movement - a modifier added per site is one that
    // gets forgotten at the sixth.
    let sprints: Vec<Option_> = out
        .iter()
        .filter(|o| o.room >= RUN_ROOM && o.commands.contains("\"forward\""))
        .map(|o| Option_ {
            text: format!("{}, at a run", o.text),
            commands: format!("[{{\"type\":\"run\",\"amount\":8}},{}", &o.commands[1..]),
            tics: o.tics,
            tag: o.tag,
            room: o.room,
            fire: None,
        })
        .collect();
    out.extend(sprints);

    out
}

/// Whether heading for the exit is worth offering.
///
/// When the engine computed a ROUTE, always: the route is a path over ground
/// the player can walk, so there is by construction somewhere to go. Gating it
/// on a clearance probe hid the option exactly where the corridor turns - the
/// straight ray is short at a corner, which is the one moment the route
/// bearing is most worth following.
///
/// Without a route there is only the straight line, which does point through
/// walls, so that one is gated.
fn route_usable(e: &crate::obs::Exit) -> bool {
    match (e.route_bearing, e.path_distance) {
        (Some(_), Some(_)) => true,
        _ => match (e.clearance, e.distance) {
            (Some(c), Some(d)) => c >= MIN_ROOM.min(d),
            _ => false,
        },
    }
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

    pub fn state(json_patch: &str) -> State {
        let base = format!(
            r#"{{"tic":1,"episodeTic":1,"level":{{"episode":1,"map":1,"skill":2,"tic":1,
            "kills":0,"totalKills":4,"items":0,"totalItems":3,"secrets":0,"totalSecrets":1}},
            "player":{{"id":0,"health":100,"armor":0,"x":0,"y":0,"angle":90,
            "weapon":"pistol","ammo":50,"keys":[]}},{json_patch},"events":[],"done":false,"outcome":"alive"}}"#
        );
        State::parse(&base).expect("test state parses")
    }

    const NOTHING: &str = r#""threats":[],"hazards":[],"pickups":[],
        "clearance":{"ahead":0,"right":0,"behind":0,"left":0,"aheadRight":0,"aheadLeft":0}"#;

    /// Something behind the wall in front of you.
    ///
    /// A shotgun rather than a medikit: this is about the ROUTE, and a
    /// medikit at full health is refused by the engine and so is correctly
    /// never offered at all.
    ///
    /// The straight line to it is 180 degrees off the nose and there is no
    /// floor that way. A player who walked past it knows the way back is out
    /// through the door on the left, and so does the route.
    fn recalled_round_a_corner(path: Option<crate::memory::Path>) -> State {
        let mut s = state(NOTHING);
        s.recalled = vec![crate::memory::Recalled {
            id: 7,
            kind: "Shotgun".into(),
            class: crate::memory::Class::Pickup,
            bearing: 180,
            distance: 220,
            health: None,
            ago: 3,
            path,
        }];
        s
    }

    /// The half that was missing. `use` reaches 64 units and only what the
    /// player FACES, and a player walks corridors facing along them - so the
    /// walls are beside them the whole way and never in front.
    #[test]
    fn a_wall_beside_the_player_can_be_turned_to_and_tried() {
        let mut s = state(NOTHING);
        assert!(!options(&s).iter().any(|o| o.tag == Tag::Face));
        s.untried_walls = vec![90];
        assert!(
            options(&s).iter().any(|o| o.tag == Tag::Face),
            "no way to turn and look at a wall an arm's length away"
        );
    }

    /// With every secret already found there is nothing to turn for.
    #[test]
    fn no_wall_is_turned_to_on_a_level_with_no_secrets_left() {
        let mut s = state(NOTHING);
        s.untried_walls = vec![90];
        s.level.secrets = s.level.total_secrets;
        assert!(!options(&s).iter().any(|o| o.tag == Tag::Face));
    }

    /// A room this run has stood in and never pushed on the walls of, with
    /// a route to it.
    fn somewhere_unsearched(path: Option<crate::memory::Path>) -> State {
        let mut s = state(NOTHING);
        s.unfrisked = vec![crate::memory::Recalled {
            id: 0,
            kind: "unsearched walls".into(),
            class: crate::memory::Class::Pickup,
            bearing: 180,
            distance: 600,
            health: None,
            ago: 0,
            path,
        }];
        s
    }

    /// What turns pressing use from a coin flip into a search.
    ///
    /// The ledger is only worth keeping if the agent can act on it, and
    /// acting on it means being able to say "go to a room you have not
    /// searched" - a destination derived from the run's own history, not
    /// from where the level's secrets actually are.
    #[test]
    fn a_room_whose_walls_were_never_searched_is_somewhere_to_be_sent() {
        assert!(
            !options(&state(NOTHING)).iter().any(|o| o.tag == Tag::Frisk),
            "offered a wall search with nowhere unsearched to go"
        );
        let routed = options(&somewhere_unsearched(Some(crate::memory::Path {
            bearing: -90,
            distance: 700,
            clearance: 256,
            step: 128,
        })));
        assert!(
            routed.iter().any(|o| o.tag == Tag::Frisk),
            "nowhere to go and search"
        );
    }

    /// And a heading through a wall is not a way to get there, exactly as
    /// for going back to a remembered thing.
    #[test]
    fn an_unsearched_room_behind_a_wall_is_not_offered_as_a_straight_line() {
        assert!(
            !options(&somewhere_unsearched(None))
                .iter()
                .any(|o| o.tag == Tag::Frisk),
            "a straight line into a wall was offered as a way to go searching"
        );
    }

    /// With every secret already found there is nothing to search for, and
    /// the decisions are better spent.
    #[test]
    fn nothing_is_searched_for_on_a_level_with_no_secrets_left() {
        let mut s = somewhere_unsearched(Some(crate::memory::Path {
            bearing: -90,
            distance: 700,
            clearance: 256,
            step: 128,
        }));
        s.level.secrets = s.level.total_secrets;
        assert!(!options(&s).iter().any(|o| o.tag == Tag::Frisk));
    }

    /// The gap this closes: with no way to ask for a route, going back for
    /// something was a straight line at where it was, so it had to be
    /// withheld wherever a wall stood in the way - which is most of a level,
    /// and is why items were only ever collected by walking into them.
    #[test]
    fn going_back_for_a_remembered_thing_follows_the_route_and_not_the_wall() {
        let unrouted = options(&recalled_round_a_corner(None));
        assert!(
            !unrouted.iter().any(|o| o.text.contains("you saw")),
            "a straight line into a wall was offered as a way back"
        );

        let routed = options(&recalled_round_a_corner(Some(crate::memory::Path {
            bearing: -90,
            distance: 480,
            clearance: 256,
            step: 128,
        })));
        let back = routed
            .iter()
            .find(|o| o.text.contains("you saw"))
            .expect("a way back, now that there is one");
        // Turned onto the ROUTE - 90 + (-90) - and not onto the straight
        // line at 180 + 90, which is the wall. Turning only: a quarter turn
        // walked while turning walks the arc of it.
        assert!(back.commands.contains("\"angle\":0"), "{}", back.commands);
        assert!(!back.commands.contains("forward"), "{}", back.commands);
        assert!(back.text.contains("220 units"), "{}", back.text);
        assert!(back.text.contains("480 units of walking"), "{}", back.text);
    }

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
        let attack = opts
            .iter()
            .find(|o| o.tag == Tag::Attack)
            .expect("an attack option");
        // The MEANING is in the text - this is the whole claim of a decision
        // model over a fixed-head policy, so it is worth a test rather than a
        // comment.
        assert!(attack.text.contains("imp"), "{}", attack.text);
        assert!(
            attack.text.contains("12 degrees to your left"),
            "{}",
            attack.text
        );
        // 90 + (-12): the turn is absolute because the engine's turn keys
        // cannot name a direction.
        assert!(
            attack.commands.contains("\"angle\":78"),
            "{}",
            attack.commands
        );
    }

    #[test]
    fn a_shut_door_on_the_route_can_be_faced_and_then_opened() {
        // The route runs THROUGH doors, so this is the state at every door in
        // the game: a bearing to walk, and a body that cannot walk it. Both
        // phases have to exist or the agent stands at the door pressing use
        // into the wall beside it.
        let walled = r#""threats":[],"hazards":[],"pickups":[],
            "clearance":{"ahead":0,"right":0,"behind":0,"left":0,"aheadRight":0,"aheadLeft":0},
            "exit":{"distance":900,"bearing":40,"kind":"switch","clearance":0,
                    "pathDistance":1200,"routeBearing":40,"routeDistance":32,
                    "routeClearance":0,
                    "blockedBy":{"kind":"door","bearing":%B,"distance":24}}"#;

        let aside = options(&state(&walled.replace("%B", "40")));
        let turn = aside
            .iter()
            .find(|o| o.tag == Tag::Open)
            .expect("a way to face the door");
        assert!(turn.commands.contains("turn-to"), "{}", turn.commands);
        assert!(turn.text.contains("shut door"), "{}", turn.text);

        let facing = options(&state(&walled.replace("%B", "3")));
        let open = facing
            .iter()
            .find(|o| o.tag == Tag::Open)
            .expect("a way to open the door");
        assert!(open.commands.contains("use"), "{}", open.commands);
        // Long enough for the door to rise. Pressing use again while it is
        // moving sends it back down.
        assert!(open.tics >= DOOR_TICS, "{} tics", open.tics);
    }

    #[test]
    fn a_locked_way_out_offers_the_key_and_not_the_exit() {
        // A level whose exit needs a key gives the agent two jobs in sequence.
        // The route does the first one, and the option has to SAY so - "head
        // for the level exit" pointed at a keycard is a lie the model has no
        // way to catch, and the whole claim here is that it reads what an
        // option means.
        let locked = r#""threats":[],"hazards":[],"pickups":[],
            "clearance":{"ahead":320,"right":0,"behind":0,"left":0,"aheadRight":0,"aheadLeft":0},
            "exit":{"distance":2688,"bearing":4,"kind":"switch","clearance":320,
                    "pathDistance":2112,"goal":"red","routeBearing":6,
                    "routeDistance":184,"routeClearance":320}"#;
        let opts = options(&state(locked));
        let way = opts
            .iter()
            .find(|o| o.tag == Tag::Exit)
            .expect("a way onward");
        assert!(way.text.contains("red key"), "{}", way.text);
        // You WALK ONTO a key. Pressing use on one does nothing.
        assert!(way.commands.contains("forward"), "{}", way.commands);
        assert!(!way.commands.contains("use"), "{}", way.commands);
    }

    #[test]
    fn a_way_out_a_switch_opens_offers_the_switch_and_not_the_wall() {
        // DOOM's walls are full of sectors some switch elsewhere raises, and
        // from in front of one there is nothing to push: the agent has to walk
        // to the switch and press THAT. Offering "push on the wall" here is
        // what had a player shoving at E1M2's for six hundred decisions with
        // the switch a hundred units off to its left.
        let walled = r#""threats":[],"hazards":[],"pickups":[],
            "clearance":{"ahead":0,"right":200,"behind":200,"left":200,
                         "aheadRight":0,"aheadLeft":0},
            "exit":{"distance":1302,"bearing":-77,"kind":"switch","clearance":26,
                    "pathDistance":1248,"routeBearing":7,"routeDistance":32,
                    "routeClearance":0,
                    "blockedBy":{"kind":"switch","bearing":7,"distance":25,
                                 "switch":{"bearing":%B,"distance":%D}}}"#;

        // Far off: walk to it.
        let away = options(&state(&walled.replace("%B", "-40").replace("%D", "108")));
        let go = away
            .iter()
            .find(|o| o.tag == Tag::Switch)
            .expect("a way to the switch");
        assert!(go.text.contains("switch"), "{}", go.text);
        assert!(go.commands.contains("forward"), "{}", go.commands);
        assert!(!go.commands.contains("\"use\""), "{}", go.commands);

        // Standing at it and facing it: press it.
        let here = options(&state(&walled.replace("%B", "2").replace("%D", "40")));
        let press = here
            .iter()
            .find(|o| o.tag == Tag::Switch)
            .expect("a way to press it");
        assert!(press.commands.contains("use"), "{}", press.commands);

        // And the observation says so, rather than calling it a wall.
        let text = crate::obs::render(
            &state(&walled.replace("%B", "-40").replace("%D", "108")),
            crate::obs::History::default(),
        );
        assert!(text.contains("switch"), "{text}");
    }

    #[test]
    fn a_way_that_walks_into_the_fire_says_so() {
        // The one thing on a level like E1M3 that reliably ends a run, and
        // the agent can only weigh it if the option it is weighing carries
        // it. A patch ahead is in the way of walking forward; the same patch
        // is not in the way of walking away from it.
        let seen = r#""threats":[],"hazards":[],"pickups":[],
            "clearance":{"ahead":320,"right":0,"behind":320,"left":0,
                         "aheadRight":0,"aheadLeft":0},
            "burningFloor":[{"bearing":0,"width":20,"distance":96}]"#;
        let opts = options(&state(seen));
        let fwd = opts
            .iter()
            .find(|o| o.tag == Tag::Advance)
            .expect("somewhere to walk");
        assert!(fwd.text.contains("burning 96 units along"), "{}", fwd.text);
        let back = opts
            .iter()
            .find(|o| o.tag == Tag::Retreat)
            .expect("somewhere to back off to");
        assert!(!back.text.contains("burning"), "{}", back.text);
        // And the same fact as a NUMBER, because the teacher cannot read the
        // sentence. Floor damage ends three of the nine levels' runs, and a
        // fact only the model can see is one the teacher walks into.
        assert_eq!(fwd.fire, Some(96), "forward walks into it, 96 units along");
        assert_eq!(back.fire, None, "backing away does not");

        let text = crate::obs::render(&state(seen), crate::obs::History::default());
        assert!(text.contains("BURNING FLOOR in sight"), "{text}");
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

#[cfg(test)]
mod lift_tests {
    use super::tests::state;
    use super::*;

    fn on_a_lift(is_lift: bool) -> State {
        let lift = if is_lift { r#","routeIsLift":true"# } else { "" };
        state(&format!(
            r#""threats":[],"hazards":[],"pickups":[],
            "clearance":{{"ahead":64,"right":0,"behind":320,"left":0,"aheadRight":0,"aheadLeft":0}},
            "exit":{{"distance":900,"bearing":30,"kind":"switch","clearance":64,
                    "pathDistance":1200,"routeBearing":30,"routeDistance":32,
                    "routeClearance":64{lift}}}"#
        ))
    }

    /// The gap this closes. The floor the route runs up is higher than a
    /// player can climb, so every option that says "walk" walks into it. The
    /// act is to press use and WAIT, and waiting is the thing no other option
    /// in the list can express.
    #[test]
    fn a_lift_on_the_route_can_be_operated_and_waited_on() {
        assert!(
            !options(&on_a_lift(false)).iter().any(|o| o.tag == Tag::Ride),
            "a lift was offered where the route said there was none"
        );
        let opts = options(&on_a_lift(true));
        let ride = opts
            .iter()
            .find(|o| o.tag == Tag::Ride)
            .expect("a way to operate the lift");
        // Faces the route and presses use: 90 + 30.
        assert!(ride.commands.contains("\"angle\":120"), "{}", ride.commands);
        assert!(ride.commands.contains("use"), "{}", ride.commands);
        // And holds still long enough for the platform to actually move.
        assert_eq!(ride.tics, RIDE_TICS);
        assert!(ride.text.contains("lift"), "{}", ride.text);
    }
}

#[cfg(test)]
mod approach_tests {
    use super::tests::state;
    use super::*;

    /// An ammo clip rather than a stimpak: this is about the GEOMETRY of the
    /// approach, and a stimpak at full health is refused by the engine and so
    /// is correctly never offered.
    fn pickup_at(bearing: i32) -> State {
        state(&format!(
            r#""threats":[],"hazards":[],
            "pickups":[{{"id":7,"type":"Ammo clip","distance":104,"bearing":{bearing},"visible":true}}],
            "clearance":{{"ahead":182,"right":79,"behind":145,"left":205,"aheadRight":90,"aheadLeft":90}}"#
        ))
    }

    /// Turning and walking in one step walks the ARC of the turn, into the
    /// inside of whatever corner is there. The exit option was split for this
    /// reason and going to a pickup was not, so a level that starts the
    /// player beside three of them kept it there: on E1M3 the scripted player
    /// stayed within 60 units of its spawn for eight hundred decisions.
    #[test]
    fn a_pickup_off_to_the_side_is_turned_toward_before_it_is_walked_to() {
        let opts = options(&pickup_at(108));
        let grab = opts.iter().find(|o| o.tag == Tag::Grab).expect("a way to it");
        assert!(grab.commands.contains("turn-to"), "{}", grab.commands);
        assert!(!grab.commands.contains("forward"), "turned and walked at once: {}", grab.commands);
        assert!(grab.text.starts_with("turn toward"), "{}", grab.text);
    }

    /// And once it is ahead, walking is the whole act - no turn, so no arc.
    #[test]
    fn a_pickup_already_ahead_is_walked_to() {
        let opts = options(&pickup_at(2));
        let grab = opts.iter().find(|o| o.tag == Tag::Grab).expect("a way to it");
        assert!(grab.commands.contains("forward"), "{}", grab.commands);
        assert!(!grab.commands.contains("turn-to"), "{}", grab.commands);
    }
}

#[cfg(test)]
mod saturated_tests {
    use super::tests::state;
    use super::*;

    fn a_stimpak_and_a_clip(health: i32) -> State {
        let mut s = state(
            r#""threats":[],"hazards":[],
            "pickups":[{"id":7,"type":"Stimpak","distance":112,"bearing":0,"visible":true},
                       {"id":8,"type":"Ammo clip","distance":160,"bearing":0,"visible":true}],
            "clearance":{"ahead":182,"right":79,"behind":145,"left":205,"aheadRight":90,"aheadLeft":90}"#,
        );
        s.player.health = health;
        s
    }

    /// The one that held two of the nine levels for their whole episode.
    ///
    /// DOOM refuses a stimpak at full health and leaves it lying there, so an
    /// option chosen because the item is present stays choosable for ever:
    /// the player paces over it until the decisions run out. Twelve hundred
    /// decisions, no kills, full health, inside 80 units of one item.
    #[test]
    fn an_item_the_engine_will_refuse_is_not_offered() {
        let opts = options(&a_stimpak_and_a_clip(100));
        let grab = opts.iter().find(|o| o.tag == Tag::Grab).expect("something to take");
        assert!(
            grab.text.contains("ammo clip"),
            "went for a stimpak at full health: {}",
            grab.text
        );
    }

    /// And it IS offered the moment it would do something.
    #[test]
    fn the_same_item_is_offered_once_it_would_heal() {
        let opts = options(&a_stimpak_and_a_clip(99));
        let grab = opts.iter().find(|o| o.tag == Tag::Grab).expect("something to take");
        assert!(grab.text.contains("stimpak"), "{}", grab.text);
    }

    /// A potion goes past 100, so full health is no reason to leave one.
    #[test]
    fn a_pickup_that_exceeds_full_health_is_still_worth_taking() {
        let mut s = a_stimpak_and_a_clip(100);
        s.pickups[0].kind = "Health Potion".into();
        let opts = options(&s);
        let grab = opts.iter().find(|o| o.tag == Tag::Grab).expect("something to take");
        assert!(grab.text.contains("health potion"), "{}", grab.text);
    }
}

#[cfg(test)]
mod fallback_tests {
    use super::tests::state;
    use super::*;

    fn cornered(came_from: Option<(i32, i32)>) -> State {
        let mut s = state(
            r#""threats":[{"id":1,"type":"IMP","distance":91,"bearing":0,"visible":true,
                          "health":60,"targetingMe":true}],"hazards":[],"pickups":[],
            "clearance":{"ahead":90,"right":40,"behind":30,"left":40,"aheadRight":40,"aheadLeft":40}"#,
        );
        s.came_from = came_from;
        s
    }

    /// Backing AWAY aims at nothing - it walks opposite whatever is in front,
    /// which in a room with four monsters converging is as likely to be a
    /// wall or a fifth monster as it is a way out. Falling back aims at floor
    /// the player has stood on.
    #[test]
    fn the_way_back_is_offered_when_there_is_one() {
        assert!(
            !options(&cornered(None)).iter().any(|o| o.tag == Tag::Fallback),
            "offered a way back without a trail to follow"
        );
        let opts = options(&cornered(Some((150, 480))));
        let back = opts
            .iter()
            .find(|o| o.tag == Tag::Fallback)
            .expect("a way back");
        // Off the nose, so it turns first and does not walk the arc.
        assert!(back.commands.contains("turn-to"), "{}", back.commands);
        assert!(!back.commands.contains("forward"), "{}", back.commands);
        assert!(back.text.contains("480 units"), "{}", back.text);
    }

    /// Once it is behind you, walking is the whole act.
    #[test]
    fn once_facing_the_way_back_it_walks() {
        let opts = options(&cornered(Some((3, 480))));
        let back = opts.iter().find(|o| o.tag == Tag::Fallback).expect("a way back");
        assert!(back.commands.contains("forward"), "{}", back.commands);
        assert!(!back.commands.contains("turn-to"), "{}", back.commands);
    }
}

#[cfg(test)]
mod frontier_tests {
    use super::tests::state;
    use super::*;

    /// The engine reports the nearest ground nobody has looked at whether or
    /// not it can route anywhere, and when it cannot route at all that is the
    /// only navigation there is. It was parsed and read by nothing, so on a
    /// level whose route field cannot be built the agent got no goal at all
    /// and had to find its way by clearance alone.
    #[test]
    fn with_no_route_the_frontier_is_still_somewhere_to_go() {
        let s = state(
            r#""threats":[],"hazards":[],"pickups":[],
            "unexplored":{"distance":512,"bearing":26,"clearance":200},
            "clearance":{"ahead":320,"right":80,"behind":80,"left":80,"aheadRight":80,"aheadLeft":80}"#,
        );
        let go = options(&s)
            .into_iter()
            .find(|o| o.tag == Tag::Explore && o.text.contains("nobody has looked at"))
            .expect("somewhere to go");
        // A quarter turn off the nose, so it faces first rather than walking
        // the arc.
        assert!(go.commands.contains("turn-to"), "{}", go.commands);
        assert!(go.text.contains("512 units"), "{}", go.text);
    }

    /// And it is not offered alongside a real route, which already leads
    /// there and says what it leads to.
    #[test]
    fn a_real_route_is_not_second_guessed() {
        let s = state(
            r#""threats":[],"hazards":[],"pickups":[],
            "unexplored":{"distance":512,"bearing":26,"clearance":200},
            "exit":{"distance":900,"bearing":30,"kind":"switch","clearance":64,
                    "pathDistance":1200,"routeBearing":30,"routeDistance":32,
                    "routeClearance":64,"goal":"unexplored"},
            "clearance":{"ahead":320,"right":80,"behind":80,"left":80,"aheadRight":80,"aheadLeft":80}"#,
        );
        assert!(
            !options(&s).iter().any(|o| o.text.contains("nobody has looked at,")),
            "offered a second frontier beside the route's own"
        );
    }
}
