// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pong-like synthetic environment: the world-model substrate task.
//!
//! A deterministic, GPU-free two-paddle pong at 64x64x3 `u8`: the AGENT is
//! the LEFT paddle (3 actions: 0 noop, 1 up, 2 down); the right paddle is a
//! simple ball-tracking opponent. The ball bounces off walls and paddles; a
//! point ends with a serve from the center. Physics is INTEGER fixed-point
//! only (positions/velocities in 1/8-pixel units) — no float math anywhere,
//! so trajectories are bit-identical across platforms and builds. All
//! randomness (serve direction/angle) comes from [`crate::rng::Rng`].
//!
//! Rendering (CHW `u8`, black background, white sprites):
//!   * left paddle 2x8 at x = 2, right paddle 2x8 at x = 61
//!   * 2x2 ball
//!   * dashed 2px center line
//!   * 3px score pips along the top (left group = agent, right = opponent)
//!
//! [`generate`] rolls episodes under a policy and writes an
//! [`crate::episode`] dataset (actions consumed + rewards observed per step).

use std::path::Path;

use crate::episode::EpisodeWriter;
use crate::rng::Rng;

/// Frame channels / height / width.
pub const C: u32 = 3;
pub const H: u32 = 64;
pub const W: u32 = 64;
/// Discrete actions: 0 noop, 1 up, 2 down.
pub const NUM_ACTIONS: u32 = 3;

/// Fixed-point scale: positions/velocities are in 1/8-pixel units.
const S: i32 = 8;
const PADDLE_W: i32 = 2;
const PADDLE_H: i32 = 8;
const BALL: i32 = 2;
/// Left (agent) paddle column and right (opponent) paddle column.
const LEFT_X: i32 = 2;
const RIGHT_X: i32 = 61;
/// Paddle speed, pixels per step (agent); the opponent moves 1 px/step, so
/// a well-angled ball can beat it.
const AGENT_SPEED: i32 = 2;
/// Ball horizontal speed in fixed units (1.25 px/step).
const BALL_VX: i32 = 10;
/// |vy| cap in fixed units.
const BALL_VY_MAX: i32 = 10;

/// The pong environment. All state is integer; stepping never touches floats.
pub struct PongEnv {
    /// Ball top-left in fixed units, in [0, (W-BALL)*S] x [0, (H-BALL)*S].
    bx: i32,
    by: i32,
    /// Ball velocity in fixed units per step.
    vx: i32,
    vy: i32,
    /// Paddle tops in whole pixels, in [0, H-PADDLE_H].
    left_y: i32,
    right_y: i32,
    score_left: u32,
    score_right: u32,
    rng: Rng,
}

impl PongEnv {
    /// A fresh environment; the first serve's direction/angle comes from
    /// `seed`.
    pub fn new(seed: u64) -> PongEnv {
        let mut env = PongEnv {
            bx: 0,
            by: 0,
            vx: 0,
            vy: 0,
            left_y: 0,
            right_y: 0,
            score_left: 0,
            score_right: 0,
            rng: Rng::new(seed),
        };
        env.reset();
        env
    }

    /// Reset scores and paddles and serve. The rng stream is NOT reseeded, so
    /// consecutive resets serve in varied directions while the whole
    /// trajectory stays a pure function of `(seed, actions)`.
    pub fn reset(&mut self) {
        self.score_left = 0;
        self.score_right = 0;
        self.left_y = (H as i32 - PADDLE_H) / 2;
        self.right_y = (H as i32 - PADDLE_H) / 2;
        self.serve();
    }

    /// Center the ball and pick a serve direction/angle from the rng.
    fn serve(&mut self) {
        self.bx = (W as i32 - BALL) / 2 * S;
        self.by = (H as i32 - BALL) / 2 * S;
        self.vx = if self.rng.gen_range_inclusive(0, 1) == 0 { -BALL_VX } else { BALL_VX };
        self.vy = self.rng.gen_range_inclusive(-(BALL_VY_MAX as i64) / 2, BALL_VY_MAX as i64 / 2) as i32;
    }

    /// Ball top-left in pixels (for tests / policies).
    fn ball_px(&self) -> (i32, i32) {
        (self.bx / S, self.by / S)
    }

    /// Advance one step: move the agent paddle per `action` (asserts
    /// `action < 3`), move the opponent and ball, resolve bounces and
    /// scoring, and render. Returns `(frame CHW u8, reward)` with reward
    /// +1 when the opponent misses, -1 when the agent misses, else 0.
    pub fn step(&mut self, action: u32) -> (Vec<u8>, i32) {
        assert!(action < NUM_ACTIONS, "PongEnv::step: action {action} out of range (num_actions = {NUM_ACTIONS})");
        // Agent paddle.
        match action {
            1 => self.left_y -= AGENT_SPEED,
            2 => self.left_y += AGENT_SPEED,
            _ => {}
        }
        self.left_y = self.left_y.clamp(0, H as i32 - PADDLE_H);

        // Opponent: track the ball center at 1 px/step.
        let ball_cy = self.by / S + BALL / 2;
        let opp_cy = self.right_y + PADDLE_H / 2;
        self.right_y += (ball_cy - opp_cy).signum();
        self.right_y = self.right_y.clamp(0, H as i32 - PADDLE_H);

        // Ball motion.
        self.bx += self.vx;
        self.by += self.vy;

        // Top/bottom wall bounce (reflect inside [0, (H-BALL)*S]).
        let by_max = (H as i32 - BALL) * S;
        if self.by < 0 {
            self.by = -self.by;
            self.vy = -self.vy;
        } else if self.by > by_max {
            self.by = 2 * by_max - self.by;
            self.vy = -self.vy;
        }

        // Paddle collisions. The hit offset (ball center vs paddle center,
        // in pixels, -4..=4) steers vy — angle control for the agent.
        let ball_top = self.by / S;
        let overlap = |py: i32| ball_top + BALL > py && ball_top < py + PADDLE_H;
        let left_face = (LEFT_X + PADDLE_W) * S; // right edge of the left paddle
        if self.vx < 0 && self.bx <= left_face && self.bx >= LEFT_X * S && overlap(self.left_y) {
            self.bx = 2 * left_face - self.bx;
            self.vx = -self.vx;
            let off = (self.by / S + BALL / 2) - (self.left_y + PADDLE_H / 2);
            self.vy = (off * 2).clamp(-BALL_VY_MAX, BALL_VY_MAX);
        }
        let right_face = RIGHT_X * S - BALL * S; // where the ball's LEFT edge sits on contact
        if self.vx > 0 && self.bx >= right_face && self.bx <= (RIGHT_X + PADDLE_W) * S && overlap(self.right_y) {
            self.bx = 2 * right_face - self.bx;
            self.vx = -self.vx;
            let off = (self.by / S + BALL / 2) - (self.right_y + PADDLE_H / 2);
            self.vy = (off * 2).clamp(-BALL_VY_MAX, BALL_VY_MAX);
        }

        // Scoring: ball fully past a goal line -> point + fresh serve.
        let mut reward = 0i32;
        if self.bx < 0 {
            reward = -1;
            self.score_right += 1;
            self.serve();
        } else if self.bx > (W as i32 - BALL) * S {
            reward = 1;
            self.score_left += 1;
            self.serve();
        }

        (self.render(), reward)
    }

    /// Render the current state as a CHW `u8` frame (white on black; the
    /// three channels are identical).
    fn render(&self) -> Vec<u8> {
        let plane = (H * W) as usize;
        let mut f = vec![0u8; (C as usize) * plane];
        let mut rect = |x0: i32, y0: i32, w: i32, h: i32| {
            for y in y0.max(0)..(y0 + h).min(H as i32) {
                for x in x0.max(0)..(x0 + w).min(W as i32) {
                    let i = (y * W as i32 + x) as usize;
                    for c in 0..C as usize {
                        f[c * plane + i] = 255;
                    }
                }
            }
        };
        // Center dashed line (2px wide, 3-on / 3-off).
        for seg in (0..H as i32).step_by(6) {
            rect(W as i32 / 2 - 1, seg, 2, 3);
        }
        // Score pips (3x3, y = 1): agent's grow rightward from x = 6,
        // opponent's grow leftward from x = 54. Capped at 5 pips each.
        for k in 0..self.score_left.min(5) as i32 {
            rect(6 + 4 * k, 1, 3, 3);
        }
        for k in 0..self.score_right.min(5) as i32 {
            rect(54 - 4 * k, 1, 3, 3);
        }
        // Paddles + ball on top.
        rect(LEFT_X, self.left_y, PADDLE_W, PADDLE_H);
        rect(RIGHT_X, self.right_y, PADDLE_W, PADDLE_H);
        let (bpx, bpy) = self.ball_px();
        rect(bpx, bpy, BALL, BALL);
        f
    }
}

/// Data-collection policy for [`generate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Uniform random actions.
    Random,
    /// Track the ball: move the agent paddle center toward the ball center.
    Chase,
}

impl Policy {
    pub fn from_name(name: &str) -> Option<Policy> {
        Some(match name {
            "random" => Policy::Random,
            "chase" => Policy::Chase,
            _ => return None,
        })
    }
}

/// Frames per second recorded in the dataset meta (matches the default
/// `brain wm play` pacing).
const FPS: u32 = 15;

/// Roll `episodes` episodes of `steps_per_episode` steps under `policy` and
/// write them as an episode dataset at `dir` (atomic; see
/// [`crate::episode::EpisodeWriter`]). Actions are the policy's choices,
/// rewards the environment's {-1,0,1} per step.
pub fn generate(
    dir: &Path,
    episodes: usize,
    steps_per_episode: usize,
    seed: u64,
    policy: Policy,
) -> Result<(), String> {
    if episodes == 0 || steps_per_episode == 0 {
        return Err("gen_pong: episodes and steps must be > 0".into());
    }
    let mut writer = EpisodeWriter::create(dir, C, H, W, NUM_ACTIONS, FPS)?;
    let mut env = PongEnv::new(seed);
    // A separate stream for the policy so its draws never perturb the
    // environment's serve stream.
    let mut policy_rng = Rng::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    for _ in 0..episodes {
        env.reset();
        for _ in 0..steps_per_episode {
            let action = match policy {
                Policy::Random => policy_rng.gen_range_inclusive(0, NUM_ACTIONS as i64 - 1) as u32,
                Policy::Chase => {
                    let ball_cy = env.by / S + BALL / 2;
                    let pad_cy = env.left_y + PADDLE_H / 2;
                    match (ball_cy - pad_cy).signum() {
                        -1 => 1, // ball above -> up
                        1 => 2,  // ball below -> down
                        _ => 0,
                    }
                }
            };
            let (frame, reward) = env.step(action);
            writer.push(&frame, action, Some(reward as f32))?;
        }
        writer.end_episode();
    }
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::EpisodeDataset;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("brain_gen_pong_{name}_{}", std::process::id()))
    }

    #[test]
    fn pong_deterministic_same_seed_and_actions() {
        let run = || -> Vec<(Vec<u8>, i32)> {
            let mut env = PongEnv::new(1234);
            let mut arng = Rng::new(99);
            (0..200)
                .map(|_| env.step(arng.gen_range_inclusive(0, 2) as u32))
                .collect()
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "same seed + actions must be byte-identical");
        // And a different seed diverges (different serve).
        let mut env = PongEnv::new(4321);
        let mut arng = Rng::new(99);
        let c: Vec<(Vec<u8>, i32)> =
            (0..200).map(|_| env.step(arng.gen_range_inclusive(0, 2) as u32)).collect();
        assert_ne!(a, c, "different seeds should produce different trajectories");
    }

    #[test]
    fn pong_up_vs_down_diverge_within_5_steps() {
        let mut up = PongEnv::new(7);
        let mut down = PongEnv::new(7);
        let mut diverged_at = None;
        for i in 0..5 {
            let (fu, _) = up.step(1);
            let (fd, _) = down.step(2);
            if fu != fd {
                diverged_at = Some(i);
                break;
            }
        }
        assert!(diverged_at.is_some(), "up vs down must diverge within 5 steps");
        assert_ne!(up.left_y, down.left_y);
    }

    #[test]
    fn pong_ball_stays_in_bounds_over_2000_random_steps() {
        let mut env = PongEnv::new(2024);
        let mut arng = Rng::new(5);
        let mut rewards = [0usize; 3]; // counts of -1, 0, +1
        for i in 0..2000 {
            let (_, r) = env.step(arng.gen_range_inclusive(0, 2) as u32);
            rewards[(r + 1) as usize] += 1;
            let (bx, by) = env.ball_px();
            assert!(
                (0..=W as i32 - BALL).contains(&bx) && (0..=H as i32 - BALL).contains(&by),
                "step {i}: ball at ({bx},{by}) out of bounds"
            );
            assert!((0..=H as i32 - PADDLE_H).contains(&env.left_y));
            assert!((0..=H as i32 - PADDLE_H).contains(&env.right_y));
        }
        // A random agent misses constantly: points (and serves) must occur.
        assert!(rewards[0] > 0, "expected the random agent to concede at least once");
    }

    #[test]
    fn pong_frame_is_valid_chw_with_visible_paddles() {
        let mut env = PongEnv::new(3);
        let (frame, _) = env.step(0);
        assert_eq!(frame.len(), (C * H * W) as usize);
        let plane = (H * W) as usize;
        // Count white pixels in each paddle's columns; the ball serves from
        // the center so it cannot overlap either paddle here.
        let count_cols = |x0: i32, x1: i32| -> usize {
            (0..H as i32)
                .flat_map(|y| (x0..=x1).map(move |x| (y * W as i32 + x) as usize))
                .filter(|&i| frame[i] == 255)
                .count()
        };
        assert_eq!(count_cols(LEFT_X, LEFT_X + PADDLE_W - 1), (PADDLE_W * PADDLE_H) as usize);
        assert_eq!(count_cols(RIGHT_X, RIGHT_X + PADDLE_W - 1), (PADDLE_W * PADDLE_H) as usize);
        // Channels identical (white-on-black) and every byte 0 or 255.
        for i in 0..plane {
            assert_eq!(frame[i], frame[plane + i]);
            assert_eq!(frame[i], frame[2 * plane + i]);
            assert!(frame[i] == 0 || frame[i] == 255);
        }
        // Ball visible somewhere.
        let (bpx, bpy) = env.ball_px();
        assert_eq!(frame[(bpy * W as i32 + bpx) as usize], 255);
    }

    #[test]
    fn pong_generate_writes_valid_episode_dataset() {
        let dir = tmp("gen");
        let _ = std::fs::remove_dir_all(&dir);
        generate(&dir, 3, 40, 11, Policy::Random).unwrap();
        let ds = EpisodeDataset::open(&dir).unwrap();
        assert_eq!((ds.n, ds.c, ds.h, ds.w), (120, C, H, W));
        assert_eq!(ds.num_actions, NUM_ACTIONS);
        assert_eq!(ds.episodes.len(), 3);
        assert!(ds.episodes.iter().all(|e| e.len == 40));
        assert!(ds.actions().iter().all(|&a| a < NUM_ACTIONS));
        let rewards = ds.rewards().expect("generate records rewards");
        assert!(rewards.iter().all(|&r| r == -1.0 || r == 0.0 || r == 1.0));
        // Window sampling respects episode boundaries.
        let mut rng = Rng::new(0);
        for _ in 0..100 {
            let w = ds.sample_window(&mut rng, 8).unwrap();
            let ep = w.start_index / 40;
            assert_eq!((w.start_index + 7) / 40, ep, "window crosses an episode boundary");
        }
        // Determinism through disk: regenerating gives identical frame bytes.
        let dir2 = tmp("gen2");
        let _ = std::fs::remove_dir_all(&dir2);
        generate(&dir2, 3, 40, 11, Policy::Random).unwrap();
        let ds2 = EpisodeDataset::open(&dir2).unwrap();
        assert_eq!(ds.frame(0).unwrap(), ds2.frame(0).unwrap());
        assert_eq!(ds.frame(119).unwrap(), ds2.frame(119).unwrap());
        assert_eq!(ds.actions(), ds2.actions());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn pong_chase_policy_generates_valid_dataset() {
        let dir = tmp("chase");
        let _ = std::fs::remove_dir_all(&dir);
        generate(&dir, 2, 100, 8, Policy::Chase).unwrap();
        let ds = EpisodeDataset::open(&dir).unwrap();
        assert_eq!(ds.n, 200);
        assert_eq!(ds.episodes.len(), 2);
        assert!(ds.actions().iter().all(|&a| a < NUM_ACTIONS));
        // The chase policy actually moves the paddle (not all-noop).
        assert!(ds.actions().iter().any(|&a| a != 0));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
