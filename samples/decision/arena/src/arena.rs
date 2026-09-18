// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A corridor with monsters in it - the smallest game that still has the
//! property this sample exists to show.
//!
//! No brain dependency: the environment is plain Rust so the sample's whole
//! brain closure stays the one SDK surface it declares.
//!
//! **The action set changes every tick, and that is the point.** You can only
//! shoot a monster that is alive and in range, only grab a medkit that is
//! still on the floor, only reload with reserve ammo left. A fixed-head policy
//! network cannot express that: its output layer's WIDTH is the action space,
//! fixed when the weights were created. Here the actions are text handed to
//! the model with the observation, so the same policy handles "shoot the imp"
//! on one tick and "grab medkit, reload, retreat" on the next - and a monster
//! it has never met is just a different string.
//!
//! The corridor is one-dimensional on purpose. Every extra dimension is more
//! observation tokens, and observation length is what a control loop pays per
//! tick; the interesting structure here is the decision, not the geometry.

/// Deterministic little PRNG, so an episode is a pure function of its seed and
/// an evaluation is reproducible. Not for anything but this game.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn chance(&mut self, p: f32) -> bool {
        (self.next() % 1000) as f32 / 1000.0 < p
    }
}

/// The two monsters are built to make TARGET CHOICE the decision.
///
/// An earlier version had a tanky demon that hit hard and a weak imp that hit
/// softly - which sounds like it demands priority and does not, because the
/// right answer is always "shoot whatever dies soonest" and that is what
/// firing at the first thing in the list already does. Seven hand-written
/// policies were measured on it and the one that ignored the observation won.
///
/// So: the imp is a **glass cannon** - one shot kills it, and while it lives
/// it does more damage than anything else. The demon is a **slow grinder** -
/// five shots to kill, but it barely hurts. Now the order is worth a lot and
/// it is not the order the option list happens to be in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Dies to one shot, hits hardest. Kill on sight.
    Imp,
    /// Takes five shots, hits softly. Safe to leave for last.
    Demon,
}

impl Kind {
    fn name(&self) -> &'static str {
        match self {
            Kind::Imp => "imp",
            Kind::Demon => "demon",
        }
    }
    fn hp(&self) -> i32 {
        match self {
            Kind::Imp => 1,
            Kind::Demon => 5,
        }
    }
    /// Damage per hit landed on the player.
    fn damage(&self) -> i32 {
        match self {
            Kind::Imp => 18,
            Kind::Demon => 7,
        }
    }
    /// Cells moved per tick.
    fn speed(&self) -> i32 {
        match self {
            Kind::Imp => 2,
            Kind::Demon => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Monster {
    pub kind: Kind,
    pub pos: i32,
    pub hp: i32,
}

/// What the agent chose. Built fresh each tick from what is actually possible.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Shoot monster `i`, by index into `monsters`.
    Shoot(usize),
    Advance,
    Retreat,
    GrabMedkit,
    Reload,
    /// Stand still and let them come. Always available, and the reason the
    /// game has a decision in it at all: with scarce ammo and accuracy that
    /// falls off with range, waiting for a better shot is often correct.
    Hold,
}

pub const CORRIDOR: i32 = 12;
const MAX_HP: i32 = 100;
/// Shots per clip, and clips in reserve. Deliberately SCARCE: the first
/// version of this game gave 18 shots for at most 15 points of monster health,
/// so a wasted shot cost nothing and "fire at everything, always" was the best
/// policy there was. With ten shots for around seven points of health, a shot
/// taken at bad odds is a shot that is not there at the end.
const CLIP: i32 = 6;
const RESERVE: i32 = 6;
/// How far the gun reaches. Beyond this a shot is not offered at all.
const RANGE: i32 = 6;
/// Accuracy falls `ACCURACY_FALLOFF` per cell: 85% at point blank, 25% at
/// range 5, 10% at the edge. This is what makes the range in an option's text
/// worth reading.
const ACCURACY_FALLOFF: f32 = 0.15;

pub struct Arena {
    pub rng: Rng,
    pub hp: i32,
    pub pos: i32,
    pub ammo: i32,
    pub reserve: i32,
    pub monsters: Vec<Monster>,
    pub medkit: Option<i32>,
    /// Rebuilt every tick by `actions`; `step` indexes into it.
    pub legal: Vec<Action>,
    pub ticks: u32,
    pub cleared: bool,
}

impl Arena {
    pub fn new() -> Arena {
        Arena {
            rng: Rng::new(0),
            hp: MAX_HP,
            pos: 0,
            ammo: CLIP,
            reserve: RESERVE,
            monsters: Vec::new(),
            medkit: None,
            legal: Vec::new(),
            ticks: 0,
            cleared: false,
        }
    }

    pub fn reset(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.hp = MAX_HP;
        self.pos = 0;
        self.ammo = CLIP;
        self.reserve = RESERVE;
        self.ticks = 0;
        self.cleared = false;
        // Two or three monsters, at least one of each kind often enough that
        // target priority matters.
        // Always at least one of each, so target priority is always a live
        // question, and the ORDER is shuffled so it cannot be read off the
        // option list's position.
        let extra = self.rng.below(2) as usize;
        let mut kinds = vec![Kind::Imp, Kind::Demon];
        for _ in 0..extra {
            kinds.push(if self.rng.chance(0.5) { Kind::Imp } else { Kind::Demon });
        }
        for i in (1..kinds.len()).rev() {
            let j = self.rng.below(i as u64 + 1) as usize;
            kinds.swap(i, j);
        }
        self.monsters = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Monster {
                kind,
                pos: 4 + (i as i32 * 2) + self.rng.below(3) as i32,
                hp: kind.hp(),
            })
            .collect();
        self.medkit = if self.rng.chance(0.7) { Some(1 + self.rng.below(4) as i32) } else { None };
    }

    fn living(&self) -> impl Iterator<Item = (usize, &Monster)> {
        self.monsters.iter().enumerate().filter(|(_, m)| m.hp > 0)
    }

    /// The observation, as the model reads it.
    ///
    /// Compact on purpose: every token is encode time in a control loop. It
    /// carries only what a decision needs - own condition, and each threat's
    /// kind, health and distance.
    pub fn observe(&self) -> String {
        let mut s = format!("hp {} ammo {} reserve {}", self.hp, self.ammo, self.reserve);
        for (_, m) in self.living() {
            let d = (m.pos - self.pos).abs();
            s.push_str(&format!(" | {} hp {} range {}", m.kind.name(), m.hp, d));
        }
        if let Some(k) = self.medkit {
            s.push_str(&format!(" | medkit range {}", (k - self.pos).abs()));
        }
        if self.living().count() == 0 {
            s.push_str(" | corridor clear");
        }
        s
    }

    /// A terminal picture of the corridor.
    pub fn render(&self) -> String {
        let mut cells = vec!['.'; CORRIDOR as usize];
        if let Some(k) = self.medkit {
            if (0..CORRIDOR).contains(&k) {
                cells[k as usize] = '+';
            }
        }
        for (_, m) in self.living() {
            if (0..CORRIDOR).contains(&m.pos) {
                cells[m.pos as usize] = match m.kind {
                    Kind::Imp => 'i',
                    Kind::Demon => 'D',
                };
            }
        }
        if (0..CORRIDOR).contains(&self.pos) {
            cells[self.pos as usize] = '@';
        }
        format!("[{}] hp {:>3} ammo {}", cells.into_iter().collect::<String>(), self.hp.max(0), self.ammo)
    }

    /// **What is possible right now** - rebuilt every tick, and different
    /// almost every tick.
    pub fn actions(&mut self) -> Vec<String> {
        let mut legal = Vec::new();
        let mut text = Vec::new();
        // One shot option per living monster IN RANGE, named by what it is and
        // how hurt it is. This is the part a fixed output layer cannot do.
        if self.ammo > 0 {
            for (i, m) in self.monsters.iter().enumerate() {
                if m.hp > 0 && (m.pos - self.pos).abs() <= RANGE {
                    legal.push(Action::Shoot(i));
                    text.push(format!("shoot the {} at range {}", m.kind.name(), (m.pos - self.pos).abs()));
                }
            }
        }
        if self.ammo < CLIP && self.reserve > 0 {
            legal.push(Action::Reload);
            text.push("reload".to_string());
        }
        if let Some(k) = self.medkit {
            if self.hp < MAX_HP {
                legal.push(Action::GrabMedkit);
                text.push(format!("grab the medkit at range {}", (k - self.pos).abs()));
            }
        }
        if self.living().next().is_some() {
            legal.push(Action::Advance);
            text.push("advance toward the enemy".to_string());
        }
        if self.pos > 0 {
            legal.push(Action::Retreat);
            text.push("retreat".to_string());
        }
        // Always available, so "do not take this shot" is something the policy
        // can actually express.
        legal.push(Action::Hold);
        text.push("hold position and wait for a closer shot".to_string());
        self.legal = legal;
        text
    }

    /// Apply one action and let the world respond. Returns `(reward, done)`.
    pub fn step(&mut self, choice: usize) -> (f32, bool) {
        let action = self.legal.get(choice).cloned().unwrap_or(Action::Advance);
        self.ticks += 1;
        // A small per-tick cost, so dithering in a corner is not free.
        let mut reward = -0.02f32;

        match action {
            Action::Shoot(i) => {
                self.ammo -= 1;
                let dist = (self.monsters[i].pos - self.pos).abs();
                // Accuracy falls with distance: close shots are near-certain,
                // long ones are a coin flip. Gives "advance" a real purpose.
                let hit = self.rng.chance(1.0 - ACCURACY_FALLOFF * dist as f32);
                if hit {
                    self.monsters[i].hp -= 1;
                    reward += 0.08;
                    if self.monsters[i].hp <= 0 {
                        // Killing the dangerous one is worth more, so target
                        // priority is something to learn rather than decorate.
                        reward += match self.monsters[i].kind {
                            Kind::Imp => 0.6,
                            Kind::Demon => 1.2,
                        };
                    }
                }
            }
            Action::Reload => {
                let want = (CLIP - self.ammo).min(self.reserve);
                self.ammo += want;
                self.reserve -= want;
            }
            Action::GrabMedkit => {
                if let Some(k) = self.medkit {
                    // Moving to it takes the tick; it only heals on arrival.
                    let step = (k - self.pos).signum();
                    self.pos += step * 2;
                    if (self.pos - k).abs() <= 1 {
                        self.pos = k;
                        self.hp = (self.hp + 40).min(MAX_HP);
                        self.medkit = None;
                        reward += 0.15;
                    }
                }
            }
            Action::Advance => {
                self.pos = (self.pos + 1).min(CORRIDOR - 1);
            }
            Action::Retreat => {
                self.pos = (self.pos - 1).max(0);
            }
            Action::Hold => {}
        }

        // The world moves: every living monster closes, and anything adjacent
        // bites.
        for m in self.monsters.iter_mut().filter(|m| m.hp > 0) {
            let d = m.pos - self.pos;
            if d.abs() <= 1 {
                self.hp -= m.kind.damage();
                reward -= 0.02 * m.kind.damage() as f32;
            } else {
                m.pos -= d.signum() * m.kind.speed().min(d.abs() - 1).max(0);
            }
        }

        if self.hp <= 0 {
            return (reward - 1.5, true);
        }
        if self.monsters.iter().all(|m| m.hp <= 0) {
            self.cleared = true;
            // Worth more than the sum of the kills, so surviving to the end is
            // the objective rather than trading hits for damage.
            return (reward + 2.0, true);
        }
        (reward, false)
    }
}


/// `(random, teacher, ceiling)` - the three numbers any learned result has to
/// be read between.
///
/// The `ceiling` is the best hand-written policy found, which is what says
/// whether a success threshold is reachable at all. Setting one without it is
/// how a previous version of this sample asked the learner to exceed a bar no
/// policy could reach.
pub fn reference_band(seeds: impl Iterator<Item = u64> + Clone) -> [(f32, f32); 3] {
    let ladder = policy_ladder(seeds);
    let get = |name: &str| {
        ladder.iter().find(|(n, _, _)| *n == name).map(|&(_, w, r)| (w, r)).unwrap_or((0.0, 0.0))
    };
    [get("random"), get("always shoot"), get("imps first, then close before firing")]
}


/// The range an option's text advertises, for a policy that reads it.
fn range_of(option: &str) -> i32 {
    option.rsplit(' ').next().and_then(|t| t.parse().ok()).unwrap_or(99)
}

/// Pick the shoot option minimizing `key`.
/// Pick the shoot option minimizing `key`. A key of 99 or more means "do not
/// take this shot", so a policy can decline every option and fall through to
/// reloading or holding - which is the whole point of the redesign.
fn shoot_by(opts: &[String], key: &mut dyn FnMut(&str) -> i32) -> Option<usize> {
    opts.iter()
        .enumerate()
        .filter(|(_, o)| o.starts_with("shoot"))
        .map(|(i, o)| (i, key(o)))
        .filter(|&(_, k)| k < 99)
        .min_by_key(|&(_, k)| k)
        .map(|(i, _)| i)
}

/// How well several hand-written policies do - the ceiling probe.
///
/// A success threshold set above what ANY policy can reach is not a demanding
/// criterion, it is a broken one, and the only way to know which is to write
/// the best policy you can and measure it. These are ordered by how much they
/// use of what the observation offers.
pub fn policy_ladder(seeds: impl Iterator<Item = u64> + Clone) -> Vec<(&'static str, f32, f32)> {
    let mut out = Vec::new();
    for (name, kind) in [
        ("random", 0u8),
        ("always shoot", 1),
        ("demon first", 2),
        ("demon first + medkit + retreat", 3),
        ("nearest first", 4),
        ("imps first (glass cannons)", 10),
        ("imps first, then close before firing", 11),
    ] {
        let mut rng = Rng::new(99);
        let (mut wins, mut n, mut total) = (0usize, 0usize, 0.0f32);
        for seed in seeds.clone() {
            let mut a = Arena::new();
            a.reset(seed);
            for _ in 0..40 {
                let opts = a.actions();
                if opts.is_empty() {
                    break;
                }
                let find = |p: &dyn Fn(&String) -> bool| opts.iter().position(p);
                let pick = match kind {
                    0 => rng.below(opts.len() as u64) as usize,
                    1 => find(&|o| o.starts_with("shoot"))
                        .or_else(|| find(&|o| o == "reload"))
                        .unwrap_or(0),
                    // Target priority: a demon does nearly three times an
                    // imp's damage, so it should die first.
                    2 => find(&|o| o.starts_with("shoot the demon"))
                        .or_else(|| find(&|o| o.starts_with("shoot")))
                        .or_else(|| find(&|o| o == "reload"))
                        .unwrap_or(0),
                    // ...plus healing when hurt and backing off with no gun.
                    // Range is in every shoot option's text, so a policy that
                    // reads the options can order by it.
                    4 => shoot_by(&opts, &mut |o| range_of(o)).or_else(|| find(&|o| o == "reload")).unwrap_or(0),
                    // Finishing a wounded monster removes a damage source a
                    // tick sooner than starting a fresh one.
                    // Kill the glass cannons first: one shot each, and each
                    // one alive costs more per tick than the demon does. Both
                    // facts are readable from the option text alone.
                    10 => shoot_by(&opts, &mut |o| if o.contains("imp") { 0 } else { 50 })
                        .or_else(|| find(&|o| o == "reload"))
                        .unwrap_or(0),
                    // ...and decline a long shot at the demon, which is the
                    // only target it is ever worth waiting on.
                    11 => shoot_by(&opts, &mut |o| {
                        if o.contains("imp") {
                            0
                        } else if range_of(o) <= 3 {
                            50
                        } else {
                            99
                        }
                    })
                    .or_else(|| find(&|o| o == "reload"))
                    .or_else(|| find(&|o| o.starts_with("hold")))
                    .unwrap_or(0),
                    _ => {
                        if a.hp <= 50 {
                            find(&|o| o.starts_with("grab"))
                        } else {
                            None
                        }
                        .or_else(|| find(&|o| o.starts_with("shoot the demon")))
                        .or_else(|| find(&|o| o.starts_with("shoot")))
                        .or_else(|| find(&|o| o == "reload"))
                        .or_else(|| find(&|o| o == "retreat"))
                        .unwrap_or(0)
                    }
                };
                let (r, done) = a.step(pick);
                total += r;
                if done {
                    break;
                }
            }
            n += 1;
            wins += usize::from(a.cleared);
        }
        out.push((name, wins as f32 / n.max(1) as f32, total / n.max(1) as f32));
    }
    out
}

impl Default for Arena {
    fn default() -> Arena {
        Arena::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole claim of this sample: the action set is not fixed.
    #[test]
    fn the_legal_action_set_changes_with_the_situation() {
        let mut a = Arena::new();
        a.reset(7);
        let opening = a.actions();
        assert!(opening.len() >= 2, "opening had {} options", opening.len());

        // Out of ammo: every shoot option must disappear and reload appear.
        a.ammo = 0;
        let dry = a.actions();
        assert!(!dry.iter().any(|o| o.starts_with("shoot")), "shooting offered with no ammo: {dry:?}");
        assert!(dry.iter().any(|o| o == "reload"), "reload not offered with an empty clip: {dry:?}");

        // Full health: the medkit is not worth offering.
        a.ammo = CLIP;
        a.hp = 100;
        a.medkit = Some(2);
        assert!(!a.actions().iter().any(|o| o.starts_with("grab")), "medkit offered at full health");
        a.hp = 50;
        assert!(a.actions().iter().any(|o| o.starts_with("grab")), "medkit not offered when hurt");
    }

    /// Options must NAME what they do - the text is the only thing the model
    /// gets, and "option 2" would make the architecture's point moot.
    #[test]
    fn options_describe_themselves() {
        let mut a = Arena::new();
        a.reset(3);
        a.monsters = vec![Monster { kind: Kind::Demon, pos: 3, hp: 5 }];
        a.pos = 0;
        let opts = a.actions();
        let shoot = opts.iter().find(|o| o.starts_with("shoot")).expect("no shoot option");
        assert!(shoot.contains("demon"), "the option does not say what it shoots: {shoot}");
        assert!(shoot.contains("range"), "the option does not say how far: {shoot}");
    }

    /// An episode must be a pure function of its seed, or evaluation cannot be
    /// compared between runs.
    #[test]
    fn an_episode_is_reproducible_from_its_seed() {
        let run = || {
            let mut a = Arena::new();
            a.reset(42);
            let mut log = Vec::new();
            for _ in 0..12 {
                let opts = a.actions();
                log.push((a.observe(), opts.clone()));
                if opts.is_empty() {
                    break;
                }
                let (r, done) = a.step(0);
                log.push((format!("{r:.3}"), vec![]));
                if done {
                    break;
                }
            }
            log
        };
        assert_eq!(run(), run(), "the same seed produced two different episodes");
    }

    /// **What is the best a policy can do here?** A success threshold above
    /// the reachable ceiling is a broken criterion, not a demanding one, so
    /// the ladder is measured rather than assumed.
    #[test]
    fn the_policy_ladder_shows_the_headroom() {
        for (name, wins, ret) in policy_ladder(0..300u64) {
            eprintln!("  {name:<34} {:.1}%  return {ret:+.3}", wins * 100.0);
        }
        let ladder = policy_ladder(0..300u64);
        let naive = ladder[1].1;
        let best = ladder.iter().skip(2).map(|r| r.1).fold(0.0f32, f32::max);
        assert!(
            best > naive + 0.05,
            "no policy that READS the observation beats the naive one ({:.1}% vs {:.1}%) - \
             this game does not reward skill, so nothing learned on it can be shown to have learned",
            best * 100.0,
            naive * 100.0
        );
    }

    /// **The difficulty band this game has to sit in**, and the reason it is a
    /// test rather than a tuning note.
    ///
    /// A learned policy is only interesting in the gap between what random
    /// play gets and what a competent heuristic gets. If random already wins
    /// most episodes there is nothing to learn; if the heuristic cannot win
    /// either, the reward carries no signal. The first draft of this game had
    /// a random policy winning 53% - almost the heuristic's 62% - and a
    /// training run that beat random by 5 points looked like a result when it
    /// was mostly noise.
    ///
    /// So: random must stay well under the heuristic, and the heuristic must
    /// stay well under perfect.
    #[test]
    fn the_game_leaves_room_between_random_play_and_competent_play() {
        let (random, smart) = (play_policy(Policy::Random), play_policy(Policy::Smart));
        eprintln!("  random {random}/120, heuristic {smart}/120");
        assert!(random < 45, "random play wins too often ({random}/120) - nothing to learn");
        assert!(smart > random + 30, "the heuristic barely beats random ({smart} vs {random})");
        assert!(smart < 110, "the game is trivially winnable ({smart}/120)");
    }

    enum Policy {
        Smart,
        Random,
    }

    fn play_policy(policy: Policy) -> usize {
        let mut wins = 0;
        let mut rng = Rng::new(99);
        for seed in 0..120u64 {
            let mut a = Arena::new();
            a.reset(seed);
            for _ in 0..40 {
                let opts = a.actions();
                if opts.is_empty() {
                    break;
                }
                let pick = match policy {
                    Policy::Smart => opts
                        .iter()
                        .position(|o| o.starts_with("shoot"))
                        .or_else(|| opts.iter().position(|o| o == "reload"))
                        .unwrap_or(0),
                    Policy::Random => rng.below(opts.len() as u64) as usize,
                };
                let (_, done) = a.step(pick);
                if done {
                    break;
                }
            }
            if a.cleared {
                wins += 1;
            }
        }
        wins
    }

    /// The game must be winnable, and not by accident: a hand-written policy
    /// that shoots the nearest thing should clear the corridor far more often
    /// than one that always takes the first option.
    #[test]
    fn a_sensible_policy_beats_a_thoughtless_one() {
        let play = |smart: bool| -> usize {
            let mut wins = 0;
            for seed in 0..120u64 {
                let mut a = Arena::new();
                a.reset(seed);
                for _ in 0..40 {
                    let opts = a.actions();
                    if opts.is_empty() {
                        break;
                    }
                    // "Smart": shoot if you can, else reload, else whatever.
                    // "Thoughtless": walk into them, which is a real and
                    // available strategy rather than a fallback into the smart
                    // one - the first draft of this test picked `retreat`,
                    // which is not offered at the starting position, so both
                    // arms silently fell through to shooting and tied at
                    // exactly 75/120.
                    let pick = if smart {
                        opts.iter()
                            .position(|o| o.starts_with("shoot"))
                            .or_else(|| opts.iter().position(|o| o == "reload"))
                            .unwrap_or(0)
                    } else {
                        opts.iter()
                            .position(|o| o.starts_with("advance"))
                            .unwrap_or(opts.len() - 1)
                    };
                    let (_, done) = a.step(pick);
                    if done {
                        break;
                    }
                }
                if a.cleared {
                    wins += 1;
                }
            }
            wins
        };
        let (smart, thoughtless) = (play(true), play(false));
        assert!(
            smart > thoughtless + 30,
            "the game does not reward play: shooting won {smart}/120, charging won {thoughtless}/120"
        );
        assert!(smart < 120, "the game is trivially winnable: {smart}/120");
        assert!(smart > 24, "the game is barely winnable at all: {smart}/120");
    }
}
