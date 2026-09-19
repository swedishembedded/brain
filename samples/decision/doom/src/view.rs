// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Watching the agent decide.
//!
//! The panel is drawn INTO the canvas, next to the game's own framebuffer, by
//! [`draw`] - one function, used by the window, by the PNG dump and by the
//! training watcher. That is deliberate: a headless run writes exactly the
//! image a human would have been looking at, so a screenshot in the README is
//! evidence and a server-side run loses nothing.
//!
//! What it shows, and why each part earns its space:
//!
//! - **The frame**, so a reader can see the situation the numbers describe.
//! - **The observation text**, which is literally all the model gets. Reading
//!   it next to the frame is how you find out that the state is missing
//!   something the picture makes obvious.
//! - **Every option with its probability**, sorted as offered. This is the
//!   decision: not "it moved forward" but "it gave forward 0.62 and attacking
//!   0.31", which is the difference between a policy that is confident and one
//!   that is guessing.
//! - **The reward trace**, so a decision can be read against what it earned.
//!
//! Swedish Embedded AB builds operator-facing views of systems that decide on
//! their own - the thing you need before anybody will let such a system near
//! production. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use brain::viewport::{Canvas, Viewport};
use brain::ControlPipeline;

use crate::env::{DoomEnv, Inspect};
use crate::{obs, Args};

const WIDTH: u32 = 1120;
const HEIGHT: u32 = 640;
const BG: [u8; 3] = [14, 14, 18];
const PANEL: [u8; 3] = [24, 24, 32];
const INK: [u8; 3] = [220, 220, 230];
const DIM: [u8; 3] = [130, 130, 145];
const GOOD: [u8; 3] = [90, 200, 120];
const BAD: [u8; 3] = [220, 90, 80];
const PICK: [u8; 3] = [250, 190, 60];

/// Where the wall clock goes, per decision.
///
/// Kept in the sample rather than measured once and written into a comment:
/// the split between "the model deciding" and "the game advancing" is the
/// number that says whether a training run is worth starting, and it moves
/// with the machine, the backend and the option count.
#[derive(Clone, Copy, Default)]
pub struct Timing {
    pub decisions: u64,
    pub policy_ns: u128,
    pub step_ns: u128,
}

impl Timing {
    pub fn print(&self, who: &str) {
        if self.decisions == 0 {
            return;
        }
        let n = self.decisions as f64;
        let policy = self.policy_ns as f64 / n / 1e6;
        let step = self.step_ns as f64 / n / 1e6;
        println!(
            "  {who} timing: {:.1} ms per decision = {policy:.1} ms policy + {step:.1} ms game \
             ({:.1} decisions/s)",
            policy + step,
            1000.0 / (policy + step)
        );
    }
}

/// One episode's worth of outcome, for both the policy and the script.
pub struct Score {
    pub episodes: usize,
    /// Mean health lost to burning floor per episode. The answer to "has it
    /// learned that slime is bad", which no other column here contains.
    pub floor_damage: f32,
    pub mean_return: f32,
    /// The mean of what the GAME scored, with the exploration bonus removed.
    pub mean_game: f32,
    pub kills: f32,
    pub items: f32,
    pub exits: usize,
    pub deaths: usize,
    pub steps: f32,
}

impl Score {
    pub fn print(&self, who: &str) {
        println!(
            "  {who}: {} episodes, return {:+.2} ({:+.2} from the game itself), \
             {:.1} kills, {:.1} items, {} exits, {} deaths, {:.0} steps, \
             {:.1} health burned off by the floor",
            self.episodes,
            self.mean_return,
            self.mean_game,
            self.kills,
            self.items,
            self.exits,
            self.deaths,
            self.steps,
            self.floor_damage
        );
    }

    pub fn row(&self, who: &str) {
        println!(
            "{:<10} {:>8.2} {:>8.2} {:>8.1} {:>8.1} {:>8} {:>8} {:>8.1}",
            who,
            self.mean_return,
            self.mean_game,
            self.kills,
            self.items,
            self.exits,
            self.deaths,
            self.floor_damage
        );
    }
}

struct Tally {
    /// Who is playing, for the per-episode line. An episode's SCORE says
    /// nothing about why it ended, and at this horizon most of them end for a
    /// reason worth reading - see [`crate::report`].
    who: &'static str,
    ret: f32,
    game: f32,
    kills: f32,
    items: f32,
    exits: usize,
    deaths: usize,
    steps: f32,
    floor: f32,
    n: usize,
}

impl Tally {
    fn new(who: &'static str) -> Tally {
        Tally {
            who,
            ret: 0.0,
            game: 0.0,
            kills: 0.0,
            items: 0.0,
            exits: 0,
            deaths: 0,
            steps: 0.0,
            floor: 0.0,
            n: 0,
        }
    }

    fn add(&mut self, env: &mut DoomEnv, total: f32, steps: usize) {
        let s = env.state();
        self.ret += total;
        self.game += env.extrinsic();
        self.kills += s.level.kills as f32;
        self.items += s.level.items as f32;
        self.exits += usize::from(s.outcome == "exited");
        self.deaths += usize::from(s.outcome == "dead");
        self.steps += steps as f32;
        self.floor += env.floor_damage() as f32;
        self.n += 1;
        println!(
            "  {} ep {:<2} {total:+7.2}  {}",
            self.who,
            self.n,
            env.report()
        );
        // An episode that ended by going nowhere also says what the route made
        // of the spot it stopped in. See `DoomEnv::stall_detail`.
        if let Some(route) = env.stall_detail() {
            println!("      route: {route}");
        }
    }

    fn finish(self) -> Score {
        let n = self.n.max(1) as f32;
        Score {
            episodes: self.n,
            mean_return: self.ret / n,
            mean_game: self.game / n,
            kills: self.kills / n,
            items: self.items / n,
            exits: self.exits,
            deaths: self.deaths,
            steps: self.steps / n,
            floor_damage: self.floor / n,
        }
    }
}

/// Run the scripted player over `seeds` and score it.
pub fn score_scripted(
    env: &mut DoomEnv,
    seeds: &[u64],
    max_steps: usize,
    timing: &mut Timing,
) -> Result<Score, String> {
    let mut tally = Tally::new("scripted");
    for &seed in seeds {
        env.start(seed);
        let (mut total, mut steps) = (0.0f32, 0usize);
        for _ in 0..max_steps {
            let Some(a) = env.scripted() else { break };
            let t0 = std::time::Instant::now();
            let (r, done) = env.apply(a, Vec::new());
            timing.step_ns += t0.elapsed().as_nanos();
            timing.decisions += 1;
            if let Some(f) = &env.fault {
                return Err(f.clone());
            }
            total += r;
            steps += 1;
            if done {
                break;
            }
        }
        tally.add(env, total, steps);
    }
    Ok(tally.finish())
}

/// Run the policy over `seeds` greedily and score it, optionally drawing.
/// Run the policy over `seeds` and score it, optionally drawing.
///
/// The policy is SAMPLED, not argmaxed, because a policy is a distribution and
/// its argmax is a different policy - one nobody trained and nobody measured
/// the objective of. That distinction is usually academic and here it is not:
/// the levels are deterministic, so a greedy policy that walks into a cycle
/// stays in it for the rest of the episode, while the same weights sampled
/// shake loose on the first decision where two options are close. Measured on
/// E1M1 with one set of weights: six episodes out of six finished when
/// sampled, and zero out of four when argmaxed, from the same file.
pub fn score_policy(
    pipe: &mut ControlPipeline<DoomEnv>,
    seeds: &[u64],
    max_steps: usize,
    mut viewer: Option<&mut Viewer>,
    timing: &mut Timing,
) -> Result<Score, String> {
    let mut tally = Tally::new("policy");
    // Seeded from the episode list, so a score is reproducible even though
    // the decisions it is made of are drawn.
    let mut rng = brain::decision::Rng::new(0x5eed_d00d ^ seeds.first().copied().unwrap_or(0));
    for &seed in seeds {
        let mut observation = pipe.env_mut().start(seed);
        let (mut total, mut steps) = (0.0f32, 0usize);
        for _ in 0..max_steps {
            let options: Vec<String> = pipe
                .env()
                .options()
                .iter()
                .map(|o| o.text.clone())
                .collect();
            if options.is_empty() {
                break;
            }
            // The distribution, not just the argmax: this is what the inspector
            // shows and what makes a decision readable.
            let t0 = std::time::Instant::now();
            let probs = pipe
                .policy(&observation, &options)
                .map_err(|e| format!("{e}"))?;
            timing.policy_ns += t0.elapsed().as_nanos();
            let mut u = rng.next_f32();
            let mut chosen = probs.len() - 1;
            for (i, p) in probs.iter().enumerate() {
                if u < *p {
                    chosen = i;
                    break;
                }
                u -= *p;
            }
            let t1 = std::time::Instant::now();
            let (r, done) = pipe.env_mut().apply(chosen, probs);
            timing.step_ns += t1.elapsed().as_nanos();
            timing.decisions += 1;
            if let Some(f) = &pipe.env().fault {
                return Err(f.clone());
            }
            total += r;
            steps += 1;
            observation = obs::render(pipe.env().state(), pipe.env().history());
            if let Some(v) = viewer.as_deref_mut() {
                if !v.tick(&pipe.env().inspect)? {
                    return Ok(tally.finish());
                }
            }
            if done {
                break;
            }
        }
        tally.add(pipe.env_mut(), total, steps);
        if let Some(v) = viewer.as_deref_mut() {
            v.episode_done();
        }
    }
    Ok(tally.finish())
}

/// A window (or not), paced for a human, that draws whatever the environment
/// last published.
pub struct Viewer {
    vp: Viewport,
    recording: bool,
    frame_dir: Option<std::path::PathBuf>,
    saved: u32,
    min_frame: Duration,
    last: Instant,
    paused: bool,
}

impl Viewer {
    pub fn new(args: &Args) -> Result<Viewer, String> {
        let mut vp = if args.view.window {
            let v = Viewport::open("brain: doom decisions", WIDTH, HEIGHT)?;
            if let Some(why) = &v.headless_because {
                // Said out loud. A run that quietly fell back to headless after
                // being asked for a window is a run whose artifacts nobody
                // knows to go looking for.
                eprintln!("doom: no window ({why}); drawing headless instead");
            }
            v
        } else {
            Viewport::headless(WIDTH, HEIGHT)
        };
        let mut recording = false;
        if let Some(path) = &args.record {
            // A failed recording is reported and the run carries on: ffmpeg is
            // an optional tool and losing a video is not worth losing a run.
            match vp.record(path, args.view.fps) {
                Ok(()) => {
                    recording = true;
                    println!("doom: recording to {path} at {} fps", args.view.fps);
                }
                Err(e) => eprintln!("doom: not recording ({e})"),
            }
        }
        Ok(Viewer {
            vp,
            recording,
            frame_dir: args.view.frames.as_ref().map(std::path::PathBuf::from),
            saved: 0,
            min_frame: Duration::from_millis(1000 / args.view.fps.clamp(1, 240) as u64),
            last: Instant::now(),
            paused: false,
        })
    }

    /// Whether anything is going to LOOK at the game's framebuffer.
    ///
    /// Fetching it costs a round trip and ~85KB a step, so it is only paid for
    /// when somebody is watching - and recording counts. Leaving `recording`
    /// out of this is not a small bug: a `--record` run with no window
    /// produced a 48-second video of a black rectangle captioned NO FRAME
    /// CAPTURED, with the decision panel beside it working perfectly.
    pub fn wants_frames(&self) -> bool {
        self.vp.has_window() || self.frame_dir.is_some() || self.recording
    }

    /// Draw one decision. Returns false when the user asked to quit.
    pub fn tick(&mut self, inspect: &Arc<Mutex<Inspect>>) -> Result<bool, String> {
        let snapshot = inspect
            .lock()
            .map_err(|_| "the inspector lock broke")?
            .clone();

        // Recording gets every frame of the decision; the window and the PNG
        // dump get the last, which is the state the decision ended in.
        if self.recording && snapshot.frames.len() > 1 {
            for f in &snapshot.frames[..snapshot.frames.len() - 1] {
                draw_at(self.vp.canvas(), &snapshot, Some(f));
                self.vp.present();
            }
        }
        draw(self.vp.canvas(), &snapshot);

        if let Some(dir) = &self.frame_dir {
            let path = dir.join(format!("decision-{:05}.png", self.saved));
            self.vp.save(&path)?;
            self.saved += 1;
        }
        if !self.vp.has_window() {
            // present() is what feeds the recorder, so a headless recording
            // still has to go through it.
            if self.recording {
                self.vp.present();
            }
            return Ok(true);
        }
        self.vp.present();

        // Pace to something a human can read. Only when there IS a window:
        // a headless dump has no reason to run slower than the game.
        loop {
            let input = self.vp.input();
            if input.quit {
                return Ok(false);
            }
            if input.pause {
                self.paused = !self.paused;
            }
            if input.screenshot {
                let path = std::path::Path::new("out").join(format!("doom-{:05}.png", self.saved));
                self.vp.save(&path)?;
                println!("doom: wrote {}", path.display());
                self.saved += 1;
            }
            if !self.paused && self.last.elapsed() >= self.min_frame {
                break;
            }
            if self.paused && input.next {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        self.last = Instant::now();
        Ok(true)
    }

    fn episode_done(&mut self) {}

    /// Close any recording and say where it went.
    pub fn finish(&mut self) {
        match self.vp.finish_recording() {
            Some(Ok((frames, path))) => {
                println!("doom: wrote {} ({frames} frames)", path.display())
            }
            Some(Err(e)) => eprintln!("doom: the recording did not finish cleanly: {e}"),
            None => {}
        }
    }
}

/// Lay out one decision on the canvas.
///
/// Pure: canvas in, canvas out, no window and no I/O, so the whole panel is
/// exercised by a headless run and by the test at the bottom of this file.
pub fn draw(c: &mut Canvas, i: &Inspect) {
    draw_at(c, i, i.frames.last())
}

/// Lay out one decision, showing `frame` as the game.
///
/// Split from [`draw`] so a recording can walk the frames of ONE decision -
/// the panel is the same for all of them, because they are all the same
/// decision, and only the picture moves.
pub fn draw_at(c: &mut Canvas, i: &Inspect, frame: Option<&crate::frame::Frame>) {
    c.clear(BG);

    // ---- the game ---------------------------------------------------------
    let empty = crate::frame::Frame::default();
    let f = frame.unwrap_or(&empty);
    let (fw, fh) = (f.width.max(320), f.height.max(200));
    let scale = Canvas::fit_scale(fw, fh, 660, 420);
    let (gw, gh) = (fw * scale, fh * scale);
    if f.is_empty() {
        c.fill(8, 8, gw, gh, [0, 0, 0]);
        c.text(16, 16, "NO FRAME CAPTURED", 2, DIM);
    } else {
        c.blit_indexed(8, 8, &f.pixels, fw, fh, &f.palette, scale);
    }
    c.outline(8, 8, gw, gh, [60, 60, 75]);

    // ---- the line that says where we are ----------------------------------
    // Two lines at scale 2, each sized to the frame it sits under. One long
    // line ran off the picture and into the option panel, where it was drawn
    // over and became unreadable at exactly the width most windows are.
    let head = format!(
        "E{}M{} SEED {} STEP {} TIC {} - {}",
        1,
        i.map,
        i.episode,
        i.step,
        i.game_tic,
        i.mission.to_uppercase()
    );
    let stats = format!(
        "HP {} KILLS {}/{} ITEMS {} SECRETS {}{}",
        i.health,
        i.kills,
        i.total_kills,
        i.items,
        i.secrets,
        if i.stuck >= 3 {
            format!("  STUCK x{}", i.stuck)
        } else {
            String::new()
        }
    );
    let stats = if i.back_steps > 0 {
        format!("{stats}  START -{}", i.back_steps)
    } else {
        stats
    };
    c.text(8, (gh + 16) as i32, &head, 2, DIM);
    c.text(8, (gh + 16 + Canvas::line_height(2)) as i32, &stats, 2, INK);

    let hp = (i.health.max(0) as f32 / 100.0).min(1.0);
    let bar_y = (gh + 20 + 2 * Canvas::line_height(2)) as i32;
    c.bar(
        8,
        bar_y,
        gw,
        8,
        hp,
        if hp > 0.34 { GOOD } else { BAD },
        PANEL,
    );

    // ---- what the model was given -----------------------------------------
    let obs_y = bar_y + 14;
    c.shade(8, obs_y, gw, 120, PANEL, 235);
    c.text(14, obs_y + 4, "WHAT THE MODEL READS", 1, DIM);
    let mut y = obs_y + 18;
    for line in i.observation.lines() {
        for chunk in wrap(&line.to_uppercase(), (gw / 6 - 2) as usize) {
            if y > obs_y + 108 {
                break;
            }
            c.text(14, y, &chunk, 1, INK);
            y += Canvas::line_height(1) as i32;
        }
    }

    // ---- the decision -----------------------------------------------------
    let px = (gw + 20) as i32;
    let pw = WIDTH - gw - 28;
    c.shade(px, 8, pw, HEIGHT - 120, PANEL, 235);
    c.text(px + 6, 12, "OPTIONS AND WHAT THE POLICY GAVE THEM", 1, DIM);

    let mut y = 30;
    for (n, opt) in i.options.iter().enumerate() {
        if y > (HEIGHT - 150) as i32 {
            break;
        }
        let p = i.probs.get(n).copied();
        let chosen = n == i.chosen;
        let label = match p {
            Some(p) => format!("{:>3.0}% {}", p * 100.0, opt),
            None => format!("     {opt}"),
        };
        let lines = wrap(&label, (pw / 6 - 4) as usize);
        let block = (lines.len() as u32 * Canvas::line_height(1)) as i32;

        // The bar is the point: a flat set of bars is a policy that has not
        // made up its mind and a single full one is a policy that has
        // collapsed, and both read instantly where four decimal places do
        // not. It sits BEHIND the whole option, and the text is always drawn
        // bright. Dark text on the highlighted bar was the first
        // attempt and it is invisible: draw_text darkens whatever is under it
        // for contrast, so the bright bar it was meant to read against is not
        // there by the time the glyphs land.
        if let Some(p) = p {
            c.bar(
                px + 6,
                y,
                pw - 12,
                block as u32 + 4,
                p,
                if chosen { [110, 80, 20] } else { [40, 46, 64] },
                [28, 28, 36],
            );
        }
        if chosen {
            // A solid edge, which reads at a glance and does not depend on
            // the text colour at all.
            c.fill(px + 2, y, 3, block as u32 + 4, PICK);
        }
        for (k, chunk) in lines.iter().enumerate() {
            c.text(
                px + 10,
                y + 2 + (k as u32 * Canvas::line_height(1)) as i32,
                &chunk.to_uppercase(),
                1,
                if chosen { PICK } else { INK },
            );
        }
        y += block + 8;
    }

    // ---- what the agent knows about the level -----------------------------
    draw_map(c, i, px, (HEIGHT - 120 - MAP_H) as i32, pw - 12, MAP_H);

    // ---- what it earned ---------------------------------------------------
    let plot_y = (HEIGHT - 104) as i32;
    c.shade(px, plot_y, pw, 96, PANEL, 235);
    c.text(
        px + 6,
        plot_y + 4,
        "REWARD PER DECISION, THIS EPISODE",
        1,
        DIM,
    );
    c.plot(px + 6, plot_y + 20, pw - 12, 50, &i.history, GOOD);
    c.text(
        px + 6,
        plot_y + 74,
        &format!(
            "LAST {:+.3}   EPISODE {:+.2}   {}",
            i.reward,
            i.total,
            i.outcome.to_uppercase()
        ),
        1,
        if i.total >= 0.0 { GOOD } else { BAD },
    );
}

/// Height of the minimap panel.
const MAP_H: u32 = 330;

/// The level as the agent knows it: where it can go, where it has been, and
/// which way it is facing.
///
/// Deliberately the SAME grid the route and frontier searches read, rather
/// than a prettier drawing of the level - a picture that came from somewhere
/// else would eventually disagree with what the agent decided on, and the
/// whole point of showing it is to be able to trust the correspondence.
fn draw_map(c: &mut Canvas, i: &Inspect, x: i32, y: i32, w: u32, h: u32) {
    c.shade(x, y, w, h, PANEL, 235);
    c.text(
        x + 6,
        y + 4,
        "WHERE IT CAN GO, AND WHERE IT HAS BEEN",
        1,
        DIM,
    );
    let m = &i.known;
    if m.is_empty() {
        c.text(x + 6, y + 20, "NO MAP", 1, DIM);
        return;
    }

    let top = y + 18;
    let avail_h = h.saturating_sub(24);
    // Whole pixels per cell, so a cell is a crisp block rather than a smear.
    let scale = Canvas::fit_scale(m.width, m.height, w - 12, avail_h);
    let ox = x + 6;
    // The grid's y grows north; the screen's grows down, so it is flipped
    // here. Drawing it unflipped puts the player at the wrong end of the
    // level, which reads as a broken sensor rather than a broken viewer.
    let oy = top;

    for cy in 0..m.height {
        for cx in 0..m.width {
            let v = m.cells[(cy * m.width + cx) as usize];
            let colour = match v {
                // Walked, and merely reachable. The contrast between them is
                // the useful thing on this panel, so it is a large one.
                2 => [90, 200, 120],
                1 => [64, 68, 86],
                _ => continue,
            };
            let sx = ox + (cx * scale) as i32;
            let sy = oy + ((m.height - 1 - cy) * scale) as i32;
            c.fill(sx, sy, scale, scale, colour);
        }
    }

    if let Some((pcx, pcy, angle)) = m.player {
        let sx = ox + (pcx * scale) as i32 + scale as i32 / 2;
        let sy = oy + ((m.height - 1 - pcy) * scale) as i32 + scale as i32 / 2;
        // A stub in the direction of travel. Map angles grow anticlockwise and
        // screen y grows down, hence the negated sine.
        let r = (angle as f32).to_radians();
        let len = (scale * 4).max(6) as f32;
        let ex = sx + (r.cos() * len) as i32;
        let ey = sy - (r.sin() * len) as i32;
        let steps = len as i32;
        for t in 0..=steps {
            let px2 = sx + (ex - sx) * t / steps.max(1);
            let py2 = sy + (ey - sy) * t / steps.max(1);
            c.fill(px2, py2, 1, 1, PICK);
        }
        c.fill(sx - 1, sy - 1, 3, 3, [255, 255, 255]);
    }
}

/// Break `s` into lines of at most `cols` characters, on spaces where it can.
fn wrap(s: &str, cols: usize) -> Vec<String> {
    let cols = cols.max(8);
    let mut out = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > cols {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        // A single word longer than the column is cut rather than allowed to
        // run off the panel and overwrite whatever is beside it.
        if word.chars().count() > cols {
            line.push_str(&word.chars().take(cols).collect::<String>());
        } else {
            line.push_str(word);
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// A window that follows training from another thread.
///
/// Training owns the main thread, so this is the only way to watch it. SDL's
/// own rule is that video lives on the thread that initialised it, which this
/// one does; it must therefore not be combined with a window on the main
/// thread, and nothing here does.
pub struct Watcher {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watcher {
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub fn watch(inspect: Arc<Mutex<Inspect>>, args: &Args) -> Watcher {
    let stop = Arc::new(AtomicBool::new(false));
    if !args.view.window {
        return Watcher { stop, handle: None };
    }
    let flag = stop.clone();
    let fps = args.view.fps.clamp(1, 60);
    let handle = std::thread::spawn(move || {
        let mut vp = match Viewport::open("brain: doom training", WIDTH, HEIGHT) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("doom: no training window ({e})");
                return;
            }
        };
        if let Some(why) = &vp.headless_because {
            eprintln!("doom: no training window ({why})");
            return;
        }
        while !flag.load(Ordering::Relaxed) {
            let snapshot = match inspect.lock() {
                Ok(g) => g.clone(),
                Err(_) => break,
            };
            draw(vp.canvas(), &snapshot);
            vp.present();
            if vp.input().quit {
                break;
            }
            std::thread::sleep(Duration::from_millis(1000 / fps as u64));
        }
    });
    Watcher {
        stop,
        handle: Some(handle),
    }
}

/// Time one decision, against the two things its cost could scale with.
///
/// The question this answers is which lever is worth pulling. If the cost
/// tracks the number of TOKENS, a shorter observation is free speed. If it is
/// flat in tokens and tracks nothing but the call itself, the cost is the
/// fixed per-dispatch overhead of running a six-layer encoder and no amount of
/// editing the prose will touch it - and the honest report is the measurement,
/// not a guess.
pub fn bench(env: DoomEnv, args: &Args) -> Result<(), String> {
    let mut pipe = ControlPipeline::builder(args.encoder(), env)
        .seed(args.seed())
        .device(args.device())
        .load()
        .map_err(|e| format!("{e}"))?;

    let word = "corridor imp shotgun ";
    println!(
        "\n  {:>7} {:>8} {:>10} {:>12}",
        "state", "options", "ms/call", "calls/s"
    );
    for &(state_words, n_opts) in &[
        (4usize, 1usize),
        (4, 8),
        (4, 24),
        (40, 1),
        (40, 8),
        (40, 24),
        (200, 8),
    ] {
        let state = word.repeat(state_words);
        let options: Vec<String> = (0..n_opts)
            .map(|i| format!("attack the imp {i} degrees to your left"))
            .collect();
        // Warm: the first call of a run pays for pipeline creation on the
        // device, which is not what a decision costs.
        for _ in 0..3 {
            pipe.policy(&state, &options).map_err(|e| format!("{e}"))?;
        }
        let n = 30;
        let t = std::time::Instant::now();
        for _ in 0..n {
            pipe.policy(&state, &options).map_err(|e| format!("{e}"))?;
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!(
            "  {state_words:>7} {n_opts:>8} {ms:>10.2} {:>12.0}",
            1000.0 / ms
        );
    }
    println!(
        "\n  (state is in words; the observation this sample builds is about 60-90 \
         words and offers 5-8 options)"
    );
    Ok(())
}

/// Play episodes with the policy, showing every decision.
pub fn play(env: DoomEnv, args: &Args) -> Result<(), String> {
    let head = args.head().ok_or(
        "play needs a trained policy: pass --head FILE (train writes one), or run `doom probe` \
         to watch the scripted player instead",
    )?;
    let mut viewer = Viewer::new(args)?;
    let mut pipe = ControlPipeline::builder(args.encoder(), env)
        .head(head)
        .seed(args.seed())
        .max_steps(args.max_steps())
        .device(args.device())
        .load()
        .map_err(|e| format!("{e}"))?;
    pipe.env_mut().capture_frames(viewer.wants_frames());
    pipe.env_mut().frames_per_tic(args.smooth_video());

    let seeds: Vec<u64> = (0..args.play as u64).map(|i| 9_000_000 + i).collect();
    let mut timing = Timing::default();
    let score = score_policy(
        &mut pipe,
        &seeds,
        args.max_steps(),
        Some(&mut viewer),
        &mut timing,
    )?;
    viewer.finish();
    score.print("policy");
    timing.print("policy");
    Ok(())
}

/// One scripted episode, with every artifact turned on.
///
/// This is what to run first on a new machine: it exercises the whole path -
/// process, socket, observation, action, reward, frame - without needing a
/// trained policy or even an encoder to be any good, and it leaves behind the
/// PNGs and the JSON transcript to look at when something is wrong.
pub fn probe(mut env: DoomEnv, args: &Args) -> Result<(), String> {
    let mut viewer = Viewer::new(args)?;
    env.capture_frames(viewer.wants_frames());
    env.frames_per_tic(args.smooth_video());
    env.start(args.seed());

    println!("doom: scripted probe, {} decisions", args.max_steps());
    let mut total = 0.0f32;
    for step in 0..args.max_steps() {
        let Some(a) = env.scripted() else { break };
        let (r, done) = env.apply(a, Vec::new());
        if let Some(f) = &env.fault {
            return Err(f.clone());
        }
        total += r;
        if !viewer.tick(&env.inspect)? {
            break;
        }
        if step % 20 == 0 || done {
            let s = env.state();
            println!(
                "  step {step:>4}  hp {:>3}  kills {}/{}  items {}  return {total:+.2}  {}",
                s.player.health, s.level.kills, s.level.total_kills, s.level.items, s.outcome
            );
        }
        if done {
            break;
        }
    }
    viewer.finish();
    println!("doom: probe finished, return {total:+.2}");
    println!("doom: {}", env.report());
    if let Some(route) = env.stall_detail() {
        println!("doom: the route, where it stopped: {route}");
    }
    if let Some(d) = &args.view.frames {
        println!("doom: wrote decision PNGs to {d}");
    }
    if let Some(t) = &args.transcript {
        println!("doom: wrote the JSON transcript to {t}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_draws_with_nothing_to_show() {
        // The first frame of every run has no frame, no options and no probs.
        // Drawing it must produce a picture that says so rather than panicking
        // on an empty slice - this used to be reached before the first step of
        // every single run.
        let mut c = Canvas::new(WIDTH, HEIGHT);
        draw(&mut c, &Inspect::default());
        assert!(c.pixels().iter().any(|&p| p != 0), "something was drawn");
    }

    #[test]
    fn the_panel_draws_a_full_decision() {
        let mut c = Canvas::new(WIDTH, HEIGHT);
        let i = Inspect {
            observation: "health 80 armor 0\nin sight: imp close".into(),
            options: vec!["attack the imp".into(), "walk forward".into()],
            probs: vec![0.7, 0.3],
            chosen: 0,
            reward: 0.5,
            total: 2.0,
            step: 4,
            mission: "clear",
            outcome: "alive".into(),
            health: 80,
            history: vec![0.1, -0.2, 0.5],
            ..Inspect::default()
        };
        draw(&mut c, &i);
        assert!(c.pixels().iter().any(|&p| p != 0));
    }

    #[test]
    fn wrapping_never_exceeds_the_column() {
        // A label that runs off its panel overwrites whatever is beside it,
        // and the option text is built from the game's own type names, which
        // are not length-bounded.
        let long = "SUPERCALIFRAGILISTIC ".repeat(6);
        for line in wrap(&long, 20) {
            assert!(line.chars().count() <= 20, "{line:?}");
        }
        assert_eq!(wrap("", 20), vec![String::new()]);
    }
}
