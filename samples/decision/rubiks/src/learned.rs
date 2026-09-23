// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The learned solver: train a policy on retraced walks, then roll it out.
//!
//! No planner is involved at any point. Labels come from walking away from a
//! solved cube, and solving is one forward pass per move. The exact planner
//! elsewhere in this sample is used here only as a MEASURING instrument, and
//! only where it can be: it can price a solution's optimality up to eight
//! moves and says nothing at all beyond that.

use brain::solve::data::Rng;
use brain::solve::{rollout, Config, Net, Rollout, StateSpace};

use crate::cube::{self, Cube};
use crate::space::CubeSpace;

/// One walk's worth of training material, with the states kept.
///
/// `data::batch` returns features and labels only, which is all a policy
/// needs. Bootstrapping a value needs the STATES as well, because the target
/// is formed by expanding each one's successors.
struct Walked {
    features: Vec<f32>,
    labels: Vec<u32>,
    states: Vec<Cube>,
}

/// Draw `rows` states by walking away from solved, balanced across depths,
/// keeping the state alongside its retraced label.
fn walk_batch(space: &CubeSpace, rows: usize, depth: usize, rng: &mut Rng) -> Walked {
    let width = space.feature_len();
    let n_moves = space.moves();
    let quota = rows.div_ceil(depth);
    let mut filled = vec![0usize; depth + 1];
    let mut features = vec![0.0f32; rows * width];
    let mut labels = Vec::with_capacity(rows);
    let mut states = Vec::with_capacity(rows);

    while states.len() < rows {
        let mut s = space.goal();
        let mut last: Option<usize> = None;
        for step in 1..=depth {
            let mut m = rng.below(n_moves);
            let mut guard = 0;
            while last.is_some_and(|l| space.redundant(l, m)) && guard < 64 {
                m = rng.below(n_moves);
                guard += 1;
            }
            s = space.apply(&s, m);
            last = Some(m);
            if filled[step] < quota && states.len() < rows {
                let r = states.len();
                space.write_features(&s, &mut features[r * width..(r + 1) * width]);
                labels.push(space.inverse(m) as u32);
                states.push(s);
                filled[step] += 1;
            }
        }
    }
    Walked { features, labels, states }
}

/// Cost-to-go targets by one step of value iteration.
///
/// `V(s) = min over successors of (1 + V_target(s'))`, anchored at zero on
/// the goal. This is the step that makes the value EXACT rather than merely
/// valid: regressing on walk length learns how far the generator happened to
/// wander, which is an upper bound and carries almost no signal between two
/// states one move apart. Bootstrapping instead asks "which of my successors
/// is nearest", which is precisely the question a rollout asks.
///
/// Evaluated under a FROZEN copy of the network. Using the live weights
/// would make the target move with every step, and the regression would chase
/// its own output.
fn davi_targets(space: &CubeSpace, target: &Net, states: &[Cube]) -> Vec<f32> {
    let width = space.feature_len();
    let n_moves = space.moves();
    let rows = target.rows as usize;

    let mut children: Vec<Cube> = Vec::with_capacity(states.len() * n_moves);
    for s in states {
        for m in 0..n_moves {
            children.push(space.apply(s, m));
        }
    }

    let mut vals = vec![0.0f32; children.len()];
    let mut features = vec![0.0f32; rows * width];
    let mut i = 0;
    while i < children.len() {
        let take = (children.len() - i).min(rows);
        features.iter_mut().for_each(|f| *f = 0.0);
        for j in 0..take {
            space.write_features(&children[i + j], &mut features[j * width..(j + 1) * width]);
        }
        let v = target.value_of(&features);
        vals[i..i + take].copy_from_slice(&v[..take]);
        i += take;
    }

    states
        .iter()
        .enumerate()
        .map(|(r, s)| {
            if space.is_goal(s) {
                return 0.0;
            }
            let mut best = f32::INFINITY;
            for m in 0..n_moves {
                let c = &children[r * n_moves + m];
                // A successor that IS the goal costs exactly one move, whatever
                // the frozen network currently predicts for it.
                let v = if space.is_goal(c) { 0.0 } else { vals[r * n_moves + m].max(0.0) };
                best = best.min(1.0 + v);
            }
            best
        })
        .collect()
}

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
    /// Written at every eval, so a long run is inspectable while it runs and
    /// survives being interrupted.
    pub checkpoint: Option<String>,
    /// How many steps between refreshes of the bootstrap's frozen copy.
    pub refresh: usize,
    /// Weight on the cost-to-go term. Zero trains the move policy alone.
    pub value_weight: f32,
}

/// How heavily cost-to-go is weighted against the move policy.
///
/// Below one because the squared error over targets that run to the walk
/// depth is numerically much larger than a cross-entropy over eighteen
/// options, and an unweighted sum would let the value term set the trunk's
/// gradients almost alone.
pub const VALUE_WEIGHT: f32 = 0.05;


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

/// The deepest walk to draw at this point in training.
///
/// Uniform over `1..=depth` throughout - what ramps is the CEILING, not the
/// distribution below it, so the shallow end is never starved. A policy has
/// to be right near the goal before being right far from it is worth
/// anything: every solve ends in the shallow states, and a run that spends
/// its early capacity on states it cannot yet make progress from learns the
/// average of many wrong answers.
fn depth_at(step: usize, steps: usize, full: usize) -> usize {
    let ramp = steps / 2;
    if step >= ramp {
        return full;
    }
    let t = step as f32 / ramp.max(1) as f32;
    (2.0 + t * (full as f32 - 2.0)).round() as usize
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
    let net = Net::new(cfg.clone(), a.rows, a.seed);
    // The frozen copy the bootstrap reads. Sized to hold one batch's worth of
    // successors, since that is what it is asked to evaluate.
    let target_rows = (a.rows * space.moves() as u32).min(16384).max(a.rows);
    let target = Net::from_weights(cfg, target_rows, &net.weights());
    let refresh = a.refresh.max(1);
    let mut rng = Rng::new(a.seed ^ 0x5EED);

    let started = std::time::Instant::now();
    let mut mark = started;
    let every = 100usize;
    let mut window = 0.0f32;

    for step in 0..a.steps {
        let depth = depth_at(step, a.steps, a.depth);
        let b = walk_batch(&space, a.rows as usize, depth, &mut rng);
        // Refresh the frozen copy on a schedule. Too often and the target
        // chases the estimate; too rarely and the estimate fits a stale one.
        let targets = if a.value_weight > 0.0 {
            if step % refresh == 0 {
                for (n, v) in net.weights() {
                    target.set_weight(&n, &v);
                }
            }
            davi_targets(&space, &target, &b.states)
        } else {
            Vec::new()
        };
        net.zero_grads();
        net.load_batch(&b.features, &b.labels);
        net.accumulate_with_value(&targets, a.value_weight);
        net.adamw(step as u32 + 1, lr_at(step, a.steps, a.lr), 0.0);

        if (step + 1) % every == 0 {
            window = net.loss();
            let per = mark.elapsed().as_secs_f32() / every as f32;
            mark = std::time::Instant::now();
            println!(
                "  step {:>6}  loss {:.4}  depth {:>2}  {:.0} ms/step  {:.0} states/s",
                step + 1,
                window,
                depth_at(step, a.steps, a.depth),
                per * 1000.0,
                a.rows as f32 / per
            );
        }
        if a.eval_every > 0 && (step + 1) % a.eval_every == 0 {
            // Several depths, not one. A single deep number reads as a flat
            // zero for most of a run while the frontier is in fact moving
            // steadily outwards underneath it, which is the difference
            // between "not working" and "not there yet".
            let mut line = String::new();
            for d in [3usize, 6, 9, 12, 16, 20, a.eval_scramble] {
                let r = measure(&net, a.eval_cubes, d, 0xE0A1);
                line.push_str(&format!(" {d}:{:.0}%", 100.0 * r.rate()));
            }
            println!("  eval solved by scramble depth -{line}");
            if let Some(path) = &a.checkpoint {
                let fit = serde_json::json!({
                    "task": "cube-policy", "steps_done": step + 1, "steps": a.steps,
                    "batch": a.rows, "walk_depth": a.depth, "d_model": a.d_model,
                    "d_ff": a.d_ff, "blocks": a.blocks, "lr": a.lr, "seed": a.seed,
                });
                match net.save(path, &fit) {
                    Ok(()) => println!("  saved {path}"),
                    Err(e) => eprintln!("  checkpoint failed: {e}"),
                }
            }
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
    measure_with(net, n, scramble, seed, Rollout { max_steps: 64, forbid_redundant: true, temperature: 0.0, seed, attempts: 1 })
}

pub fn measure_with(net: &Net, n: usize, scramble: usize, seed: u64, how: Rollout) -> Measured {
    measure_how(net, n, scramble, seed, how, false)
}

/// `by_value` drives with the cost-to-go head instead of the move policy.
pub fn measure_how(
    net: &Net,
    n: usize,
    scramble: usize,
    seed: u64,
    how: Rollout,
    by_value: bool,
) -> Measured {
    let space = CubeSpace::new();
    let starts: Vec<Cube> = (0..n).map(|i| cube::scramble(scramble, seed ^ (i as u64 * 0x9E37)).0).collect();

    let t0 = std::time::Instant::now();
    let outcomes = if by_value {
        rollout::solve_batch_by_value(&space, net, &starts, how)
    } else {
        rollout::solve_batch(&space, net, &starts, how)
    };
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
