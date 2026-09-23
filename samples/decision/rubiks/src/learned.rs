// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The learned solver: train a policy on retraced walks, then roll it out.
//!
//! No planner is involved at any point. Labels come from walking away from a
//! solved cube, and solving is one forward pass per move. The exact planner
//! elsewhere in this sample is used here only as a MEASURING instrument, and
//! only where it can be: it can price a solution's optimality up to eight
//! moves and says nothing at all beyond that.

use brain::solve::data::{batch, Rng, Walk};
use brain::solve::{rollout, Config, Net, Rollout, StateSpace};

use crate::cube::{self, Cube};
use crate::space::CubeSpace;

pub struct TrainArgs {
    pub steps: usize,
    pub rows: u32,
    pub depth: usize,
    pub d_model: u32,
    pub d_ff: u32,
    pub blocks: u32,
    pub lr: f32,
    pub seed: u64,
    pub eval_every: usize,
    pub eval_cubes: usize,
    pub eval_scramble: usize,
}

pub fn config(a: &TrainArgs, space: &CubeSpace) -> Config {
    Config {
        in_dim: space.feature_len() as u32,
        d_model: a.d_model,
        d_ff: a.d_ff,
        blocks: a.blocks,
        moves: space.moves() as u32,
    }
}

/// Cosine decay after a short linear warmup. The warmup is not decoration:
/// the first steps of a freshly initialised residual stack produce large,
/// badly-conditioned gradients, and entering at full rate is what turns them
/// into a loss that climbs.
fn lr_at(step: usize, steps: usize, peak: f32) -> f32 {
    let warm = (steps / 50).max(100);
    if step < warm {
        return peak * (step as f32 + 1.0) / warm as f32;
    }
    let t = (step - warm) as f32 / (steps - warm).max(1) as f32;
    let cos = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    peak * (0.05 + 0.95 * cos)
}

pub fn train(a: &TrainArgs) -> Net {
    let space = CubeSpace::new();
    let cfg = config(a, &space);
    println!(
        "cubenet: {} parameters, {} wide, {} blocks - batch {}, walks to depth {}",
        cfg.params(),
        cfg.d_model,
        cfg.blocks,
        a.rows,
        a.depth
    );
    let net = Net::new(cfg, a.rows, a.seed);
    let walk = Walk { depth: a.depth, seed: a.seed ^ 0x5EED };
    let mut rng = Rng::new(walk.seed);

    let started = std::time::Instant::now();
    let mut mark = started;
    let every = 100usize;
    let mut window = 0.0f32;

    for step in 0..a.steps {
        let b = batch(&space, &walk, a.rows as usize, &mut rng);
        net.zero_grads();
        net.load_batch(&b.features, &b.labels);
        net.accumulate();
        net.adamw(step as u32 + 1, lr_at(step, a.steps, a.lr), 0.0);

        if (step + 1) % every == 0 {
            window = net.loss();
            let per = mark.elapsed().as_secs_f32() / every as f32;
            mark = std::time::Instant::now();
            println!(
                "  step {:>6}  loss {:.4}  {:.0} ms/step  {:.0} states/s",
                step + 1,
                window,
                per * 1000.0,
                a.rows as f32 / per
            );
        }
        if a.eval_every > 0 && (step + 1) % a.eval_every == 0 {
            let r = measure(&net, a.eval_cubes, a.eval_scramble, 0xE0A1 ^ step as u64);
            println!(
                "  eval  scramble {}: solved {}/{} ({:.0}%), mean {:.1} moves",
                a.eval_scramble, r.solved, r.total, 100.0 * r.rate(), r.mean_moves
            );
        }
    }
    println!(
        "cubenet: {} steps in {:.1}s",
        a.steps,
        started.elapsed().as_secs_f32()
    );
    net
}

pub struct Measured {
    pub solved: usize,
    pub total: usize,
    pub mean_moves: f32,
    pub outcomes: Vec<rollout::Outcome>,
    pub starts: Vec<Cube>,
    pub elapsed: std::time::Duration,
}

impl Measured {
    pub fn rate(&self) -> f32 {
        self.solved as f32 / self.total.max(1) as f32
    }
}

/// Scramble `n` cubes by `scramble` random moves and let the policy drive.
pub fn measure(net: &Net, n: usize, scramble: usize, seed: u64) -> Measured {
    measure_with(net, n, scramble, seed, Rollout { max_steps: 64, forbid_redundant: true, temperature: 0.0, seed })
}

pub fn measure_with(net: &Net, n: usize, scramble: usize, seed: u64, how: Rollout) -> Measured {
    let space = CubeSpace::new();
    let starts: Vec<Cube> = (0..n).map(|i| cube::scramble(scramble, seed ^ (i as u64 * 0x9E37)).0).collect();

    let t0 = std::time::Instant::now();
    let outcomes = rollout::solve_batch(&space, net, &starts, how);
    let elapsed = t0.elapsed();

    let solved = outcomes.iter().filter(|o| o.solved()).count();
    let mean_moves = if solved > 0 {
        outcomes.iter().filter(|o| o.solved()).map(|o| o.moves().len() as f32).sum::<f32>() / solved as f32
    } else {
        0.0
    };
    Measured { solved, total: n, mean_moves, outcomes, starts, elapsed }
}

/// One cube's worth of policy output, for the renderer.
pub struct Turn {
    pub probabilities: Vec<f32>,
    pub picked: usize,
}

/// Ask the policy about a single state. One forward pass; the rest of the
/// batch is padding and is ignored.
pub fn ask(net: &Net, space: &CubeSpace, cube: &Cube, forbid_after: Option<usize>) -> Turn {
    let width = space.feature_len();
    let mut features = vec![0.0f32; net.rows as usize * width];
    space.write_features(cube, &mut features[..width]);
    let probs = net.policy(&features);
    let row = probs[..space.moves()].to_vec();

    let mut picked = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (m, &p) in row.iter().enumerate() {
        if forbid_after.is_some_and(|l| space.redundant(l, m)) {
            continue;
        }
        if p > best {
            best = p;
            picked = m;
        }
    }
    Turn { probabilities: row, picked }
}
