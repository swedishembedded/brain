// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training the PID controller transformer on simulated plant trajectories,
//! and the closed-loop report that judges a checkpoint against each plant's
//! own oracle controller.
//!
//! Swedish Embedded AB implements learned controllers for real plants on
//! embedded targets for its clients. If your team needs expertise in
//! control and sequence models on its own hardware then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use crate::data::{self as pid_data, PlantSpec, Rng};
use crate::{Pid, PidConfig, BOS, DECIDE};

/// One training run.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainCfg {
    pub steps: u32,
    /// Sequences averaged into each optimizer step.
    pub eff_batch: u32,
    /// Event/effect records per training window.
    pub seq_steps: u32,
    /// Tokens per window; raised to fit `seq_steps` records when smaller.
    pub t_pad: u32,
    pub d_model: u32,
    pub layers: u32,
    pub heads: u32,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    /// Simulated trajectories per training plant, and their length.
    pub n_traj: usize,
    pub traj_len: usize,
    /// Device bytes the microbatch planner may use.
    pub mem_budget: u64,
    pub out: String,
    /// Steps between checkpoint writes; 0 writes only at the end.
    pub ckpt_every: u32,
    /// What the backward keeps of the forward; `None` keeps the model's own default.
    pub activations: Option<model::saved::SavedPolicy>,
}

impl Default for TrainCfg {
    fn default() -> TrainCfg {
        TrainCfg {
            steps: 2000,
            eff_batch: 32,
            seq_steps: 12,
            t_pad: 256,
            d_model: 64,
            layers: 2,
            heads: 4,
            lr: 1e-3,
            wd: 0.01,
            seed: data::rng::random_seed(),
            n_traj: 180,
            traj_len: 80,
            mem_budget: 4u64 << 30,
            out: String::new(),
            ckpt_every: 100,
            activations: None,
        }
    }
}

/// The microbatch and accumulation count that fit `budget` bytes for an
/// effective batch of `eff_batch` sequences of `t` tokens: the largest
/// microbatch that fits and divides the effective batch.
pub fn plan_batch(cfg: &PidConfig, t: u32, eff_batch: u32, budget: u64) -> (u32, u32) {
    // batch-independent: weights + grad + 2 adam moments (4x params).
    let total_params: u64 = cfg.param_list().iter().map(|(_, n)| *n as u64).sum();
    let fixed = 4 * total_params * 4;
    // per-sample activation + backward scratch bytes (rough closed form).
    let d = cfg.d_model as u64;
    let ff = cfg.d_ff as u64;
    let u = cfg.u_bins as u64;
    let h = cfg.n_heads as u64;
    let tt = t as u64;
    let per_sample = 4 * ((cfg.n_layers as u64 + 1) * d * 2 + cfg.n_layers as u64 * (2 * d + 3 * d + d + d + d + 3 * ff + 3 * ff) + h * tt * 3 + d * 6 + u * 2 + ff * 3);
    let avail = budget.saturating_sub(fixed).max(per_sample);
    let mut b_micro = ((avail / per_sample) as u32).clamp(1, eff_batch.max(1));
    while !eff_batch.is_multiple_of(b_micro) {
        b_micro -= 1;
    }
    (b_micro, eff_batch / b_micro)
}

/// What a run did, and the configuration it trained.
pub struct Trained {
    pub cfg: PidConfig,
    pub microbatch: u32,
    pub accumulation: u32,
    pub steps: u32,
    pub final_loss: f32,
    pub model: Pid,
}

/// Train per `tc`, writing the checkpoint to `tc.out` every `ckpt_every`
/// steps and at the end (atomically: the file on disk is always loadable).
/// `on_step(step, loss)` sees every logged step and stops the run by
/// returning false; the checkpoint is written with the steps that ran.
pub fn train(tc: &TrainCfg, on_step: &mut dyn FnMut(u32, Option<f32>) -> bool) -> Result<Trained, String> {
    // Each event/effect step is at most ~17 tokens (variable-length CBOR),
    // plus the leading BOS.
    let t_pad = tc.t_pad.max(1 + 18 * tc.seq_steps);
    let cfg = PidConfig { block_size: t_pad, n_layers: tc.layers, d_model: tc.d_model, n_heads: tc.heads, d_ff: 4 * tc.d_model, ..PidConfig::default_small() };
    let (b_micro, n_accum) = plan_batch(&cfg, t_pad, tc.eff_batch, tc.mem_budget);
    let plants = pid_data::training_plants();
    let ds = pid_data::PidDataset::new(&plants, tc.n_traj, tc.traj_len, tc.seq_steps as usize, t_pad as usize, tc.seed);
    let init = pid_data::init_weights(&cfg, tc.seed);
    let mut model = Pid::new(cfg.clone(), b_micro, t_pad, &init);
    if let Some(policy) = tc.activations {
        model::Model::set_saved_policy(&mut model, policy)?;
    }
    let mut rng = Rng::new(tc.seed ^ 0xABCD_1234);
    let mut steps = 0;
    for step in 1..=tc.steps {
        model.zero_grads();
        for _ in 0..n_accum {
            let (xs, ys) = ds.batch(b_micro as usize, &mut rng);
            model.set_batch(&xs, &ys);
            // forward/backward stay on the device: no per-microbatch readback
            model.forward_submit();
            model.backward();
        }
        model.adamw_step(step, tc.lr, tc.wd, Default::default(), Some(1.0), 1.0 / n_accum as f32);
        // Wait for the step and reclaim its transient submit memory, or the
        // per-submit staging buffers accumulate until an allocation fails.
        model.poll_wait();
        steps = step;
        // The loss is read back only on a logged step: one readback per
        // interval, not per step.
        let logged = step == 1 || step % 100 == 0 || step == tc.steps;
        if !on_step(step, logged.then(|| model.loss())) {
            break;
        }
        if tc.ckpt_every > 0 && step % tc.ckpt_every == 0 && step != tc.steps {
            model.save(&tc.out);
        }
    }
    model.save(&tc.out);
    let final_loss = model.loss();
    Ok(Trained { cfg, microbatch: b_micro, accumulation: n_accum, steps, final_loss, model })
}

/// A checkpoint's weights, for a single-sequence decoder.
pub fn weights_of(model: &Pid) -> HashMap<String, Vec<f32>> {
    model.ps.params.iter().map(|(n, _)| (n.clone(), model.read_weight(n))).collect()
}

/// The closed-loop result on one plant: the model's tracking error against
/// the plant's oracle controller.
#[derive(Clone, Debug, PartialEq)]
pub struct PlantReport {
    pub tau: f32,
    pub gain: f32,
    pub model_mse: f32,
    pub oracle_mse: f32,
    /// The model's steady-state absolute error.
    pub steady_state_error: f32,
}

/// The closed-loop run of `dec` on every plant in `plants`, `steps` long.
pub fn closed_loop(dec: &Pid, cfg: &PidConfig, plants: &[PlantSpec], steps: usize) -> Vec<PlantReport> {
    plants
        .iter()
        .map(|spec| {
            let (model_mse, steady_state_error) = pid_data::rollout_on_plant(dec, cfg, spec, steps);
            let oracle = pid_data::run_oracle_closed_loop(spec, steps);
            let oracle_mse = oracle.iter().map(|&(_, sp, y)| (sp - y).powi(2)).sum::<f32>() / oracle.len() as f32;
            PlantReport { tau: spec.tau, gain: spec.gain, model_mse, oracle_mse, steady_state_error }
        })
        .collect()
}

/// A single-sequence decoder over the checkpoint at `path`.
pub fn load(path: &str) -> Result<(Pid, PidConfig), String> {
    if !std::path::Path::new(path).is_file() {
        return Err(format!("{path}: no such checkpoint"));
    }
    let c = checkpoint::load(path);
    let cfg = PidConfig::from_json(&c.header["config"]);
    let dec = Pid::new(cfg.clone(), 1, cfg.block_size, &c.by_role(""));
    Ok((dec, cfg))
}

/// A representative context of exactly `len` tokens (BOS, then repeated
/// event, DECIDE and effect records): forward cost depends on the token
/// count, not the values.
pub fn profiling_context(len: usize) -> Vec<u32> {
    let mut ctx = vec![BOS];
    while ctx.len() < len {
        ctx.extend(pid_data::encode_event(0.3, 0.1, 0.2));
        ctx.push(DECIDE);
        ctx.extend(pid_data::encode_effect_bin(40));
    }
    ctx.truncate(len);
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_planner_budget_and_divisibility() {
        let cfg = PidConfig::default_small();
        // generous budget -> single microbatch, no accumulation
        assert_eq!(plan_batch(&cfg, 64, 16, 8u64 << 30), (16, 1));
        // tiny budget -> microbatch shrinks but still divides the effective batch
        let (bm, acc) = plan_batch(&cfg, 256, 12, 1 << 20);
        assert!((1..=12).contains(&bm));
        assert_eq!(12 % bm, 0);
        assert_eq!(bm * acc, 12);
    }

    #[test]
    fn a_profiling_context_has_the_asked_length_and_starts_at_bos() {
        let ctx = profiling_context(50);
        assert_eq!((ctx.len(), ctx[0]), (50, BOS));
    }
}
