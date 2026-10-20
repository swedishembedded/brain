// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven fine-tuning that
// can be trusted, for its clients. If your team needs expertise in
// reinforcement learning for language models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Environment, verifier and configuration shared by the GRPO integration
//! tests: one fixed prompt, a reward by a function of the completion, and a
//! record of what the verifier was shown.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use model::rollout::RolloutParams;
use model::serve::SampleParams;
use rl::env::{Environment, Reward, Step, Task, Verifier};
use data::rng::Rng;
use model::{Model, Objective, StepVerdict};
use rl::objective::grpo::{Grpo, GrpoConfig, GrpoOptions};

pub const T: usize = 8;
pub const PROMPT: [u32; 3] = [1, 2, 3];

/// One fixed prompt; `variable(seed)` decides whether the task's reward varies
/// across completions or is the same for all of them.
pub struct Env {
    pub variable: fn(u64) -> bool,
}

impl Environment for Env {
    fn name(&self) -> &str {
        "fixes"
    }
    fn tasks(&self, seed: u64) -> Vec<Task> {
        vec![Task { id: format!("t{seed}"), prompt: PROMPT.to_vec(), answer: serde_json::json!((self.variable)(seed)) }]
    }
}

/// Reward by a closure over the completion the verifier is shown, and a record of what it saw.
pub struct Judge {
    pub reward: fn(&[u32], bool) -> f32,
    pub seen: Rc<RefCell<Vec<Vec<u32>>>>,
}

impl Judge {
    pub fn new(reward: fn(&[u32], bool) -> f32) -> (Judge, Rc<RefCell<Vec<Vec<u32>>>>) {
        let seen = Rc::new(RefCell::new(Vec::new()));
        (Judge { reward, seen: seen.clone() }, seen)
    }
}

impl Verifier for Judge {
    fn verify(&self, task: &Task, _t: &[Step], completion: &[u32]) -> Reward {
        self.seen.borrow_mut().push(completion.to_vec());
        let variable: bool = serde_json::from_value(task.answer.clone()).unwrap();
        Reward { value: (self.reward)(completion, variable), parts: Default::default() }
    }
}

/// Informative on variable tasks (parity of the first token), constant otherwise.
pub fn parity_reward(c: &[u32], variable: bool) -> f32 {
    if variable {
        (c[0] % 2) as f32
    } else {
        1.0
    }
}

pub fn config(group: usize, temp: f32, options: GrpoOptions) -> GrpoConfig {
    GrpoConfig {
        group_size: group,
        clip_eps: 0.2,
        kl_beta: 0.0,
        seq_len: T,
        rollout: RolloutParams { max_new: 4, sample: SampleParams { temp, top_k: 0, top_p: 1.0, ..SampleParams::greedy() }, eos: None },
        max_attempts: 1,
        options,
    }
}

pub fn always_variable(_: u64) -> bool {
    true
}

/// R1 on `m`: the first update of fresh rows sees a ratio of one, and the
/// sampler's reported probability is the trainer's, at sampling temperature `temp`.
pub fn assert_trained_distribution_is_sampled<M: Model>(m: &mut M, temp: f32, seed: u64) {
    let (judge, _) = Judge::new(parity_reward);
    let mut obj = Grpo::new(Env { variable: always_variable }, judge, config(4, temp, GrpoOptions::default()));
    Objective::<M>::prepare(&mut obj, m).unwrap();
    obj.micro_steps(m, &mut Rng::new(seed), 4);
    let st = obj.stats();
    assert!(st.rows_trained > 0, "T={temp}: nothing was trained");
    assert!(st.max_onpolicy_ratio_dev < 1e-4, "T={temp}: first-update |ratio - 1| = {}", st.max_onpolicy_ratio_dev);
    assert!(st.max_sampler_gap < 1e-3, "T={temp}: the sampler's logprob differs from the trainer's by {}", st.max_sampler_gap);
}

/// R3 on `m`: about 80% of tasks give every completion the same reward; every
/// step holds `k` informative rows or is logged short, and an optimiser step
/// is never allowed on zero rows.
pub fn assert_no_step_on_an_empty_batch<M: Model>(m: &mut M, seed: u64) {
    let (judge, _) = Judge::new(parity_reward);
    let mut obj = Grpo::new(Env { variable: |seed| seed % 10 >= 8 }, judge, config(4, 1.0, GrpoOptions::default()));
    Objective::<M>::prepare(&mut obj, m).unwrap();
    let mut rng = Rng::new(seed);
    let k = 4;
    for step in 0..6 {
        let before = obj.stats().clone();
        obj.micro_steps(m, &mut rng, k);
        let trained = obj.stats().rows_trained - before.rows_trained;
        let short = obj.stats().steps_short - before.steps_short;
        assert!(trained == k as usize || short == 1, "step {step}: {trained} rows and no shortfall logged");
        match Objective::<M>::step_verdict(&mut obj) {
            StepVerdict::Apply => assert!(trained > 0, "step {step}: an optimiser step was allowed on zero rows"),
            StepVerdict::Skip(_) => assert_eq!(trained, 0, "step {step}: skipped with {trained} rows"),
            StepVerdict::Abort(e) => panic!("{e}"),
        }
    }
    assert!(obj.stats().groups_zero_variance > 0, "the environment was meant to give zero-variance groups");
}
