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
    /// Stand still and let the moment pass.
    Hold,
    /// Remove whatever is standing in the way of the route.
    Clear,
    /// Get off floor that is burning the player.
    Escape,
    Exit,
}

/// A decision every ~4-8 tics is about 5-9 per second of game time, which is
/// the rate a human plays at and slow enough that a decision model with a
/// millisecond of latency is never the thing holding the game up.
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
const USE_RANGE: i32 = 64;
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
            for (name, key, room) in [("left", "strafe-left", c.left), ("right", "strafe-right", c.right)] {
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
                    room,
                });
            }
        }
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
            room: p.distance,
        });
    }

    // Something seen a moment ago and no longer in view. Without this the
    // only way anything is ever collected is by walking into it: the option
    // to go and get a medikit disappears in the same decision the player
    // stops looking at it. Items only - a monster has moved since, and
    // "go to where it was" is an invitation to walk into where it is not.
    //
    // Only when there is floor that way. The option walks a STRAIGHT LINE at
    // where the thing was, and a remembered medikit two rooms away through a
    // wall turns "go back for it" into "walk at that wall" - measured on
    // health-gathering-supreme, where offering it unguarded cost the scripted
    // player three extra deaths in twenty-four episodes. The route is what
    // knows the way round corners, and it does not yet take a remembered
    // thing as a goal.
    for r in state
        .recalled
        .iter()
        .filter(|r| r.class != crate::memory::Class::Threat)
        .filter(|r| c.toward(r.bearing) >= r.distance.min(300))
        .take(2)
    {
        out.push(Option_ {
            text: format!(
                "go back for the {} you saw, {} units {}",
                r.kind.to_lowercase(),
                r.distance,
                bearing_phrase(r.bearing)
            ),
            commands: format!(
                "[{},{{\"type\":\"forward\",\"amount\":{}}}]",
                json_turn(state.facing(r.bearing)),
                walk_tics(r.distance)
            ),
            tics: walk_tics(r.distance),
            tag: Tag::Grab,
            room: r.distance,
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
    let hot = |bearing: i32, distance: i32| match state
        .burning_floor
        .iter()
        .filter(|b| b.in_the_way(bearing, distance))
        .min_by_key(|b| b.distance)
    {
        Some(b) => format!(", and the floor starts burning {} units along", b.distance),
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
            });
        }
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
        text: "push on the wall or door directly in front of you".into(),
        commands: "[{\"type\":\"use\"},{\"type\":\"forward\",\"amount\":4}]".into(),
        tics: MOVE_TICS,
        tag: Tag::Use,
        room: 0,
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

    fn state(json_patch: &str) -> State {
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
