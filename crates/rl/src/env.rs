// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `Environment` / `Verifier` seam: a programmatic, deterministic
//! reward path any task family plugs into, and the one every other reward
//! source in this crate (starting with [`crate::atif`]'s trajectory
//! ingestion) is a special case of, rather than a parallel system living
//! next to it.
//!
//! ## Why programmatic-only
//!
//! [`Verifier::verify`] must be re-runnable byte-for-byte against a stored
//! run artifact - no model-as-judge, no network call, no nondeterministic
//! source anywhere in an impl - so a promote/reject decision made from a
//! reward is always re-derivable later from the same artifact. This is
//! also why [`Task::answer`] carries only what a verifier needs to
//! *recompute* the correct answer, never the answer itself: an answer
//! sitting in `Task` would be indistinguishable from a label, and nothing
//! that looks like a label may reach training data.
//!
//! ## Single-turn by default, multi-step supported
//!
//! [`Environment::step`] defaults to ending the episode after exactly one
//! action - the common case (answer a prompt, get scored) needs no
//! override. An environment with real turn structure (a stateful protocol,
//! a tool-use loop) overrides `step` to fold `action` into its own state
//! and report an observation plus whether the episode continues; nothing
//! about the trait shape changes between the two cases.
//!
//! ## `legal_actions` is not an afterthought
//!
//! An unconstrained random policy's cold-start hit rate on even a modest
//! action space can be too thin to bootstrap rejection sampling at all.
//! [`Environment::legal_actions`] lets an environment declare, per task and
//! per step, the action subset a policy should be constrained to explore -
//! applied identically to every arm of every comparison so nothing is
//! hidden, while an unconstrained score can still be reported alongside it.
//! The default (`None`) means "no constraint", so an environment with a
//! genuinely open action space (e.g. free-text generation) need not
//! override it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One task instance: a prompt to present to a policy, plus the material a
/// [`Verifier`] needs to recompute whether a completion is correct.
/// `answer` is deliberately named for what it enables (recomputation), not
/// what it contains - it never carries the answer itself, only e.g. a
/// target value, a reference program, or a checksum a verifier evaluates
/// against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// Stable identifier for this task instance, e.g. for re-scoring a
    /// stored run artifact against the same task later.
    pub id: String,
    /// Token ids presented to the policy as the prompt.
    pub prompt: Vec<u32>,
    /// Whatever a [`Verifier`] needs to recompute correctness - never the
    /// answer itself. See the module doc comment.
    pub answer: serde_json::Value,
}

/// One completed turn of a (possibly multi-step) episode: the action a
/// policy took, and the observation the environment returned for it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// Token ids the policy emitted this turn.
    pub action: Vec<u32>,
    /// Token ids the environment fed back after `action`, empty for a
    /// single-turn environment (nothing to feed back before the episode
    /// ends).
    pub observation: Vec<u32>,
}

/// The result of one [`Environment::step`] call: what to feed back to the
/// policy, and whether the episode is over.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepOutcome {
    /// Token ids to append as the next observation. Empty when there is
    /// nothing to say back (the common single-turn case).
    pub observation: Vec<u32>,
    /// Whether the episode has ended - no further `step` calls follow.
    pub done: bool,
}

/// A task family: produces tasks, optionally runs multi-step episodes, and
/// optionally constrains exploration. See the module doc comment for the
/// design intent behind each default.
pub trait Environment {
    /// Short, stable name for this environment (e.g. for logging/reporting
    /// which family a run artifact came from).
    fn name(&self) -> &str;

    /// Generate this environment's task instances for a given seed -
    /// deterministic in `seed` so a run is reproducible.
    fn tasks(&self, seed: u64) -> Vec<Task>;

    /// Advance one turn of the episode given the transcript so far and the
    /// policy's latest action. The default is single-turn: any action ends
    /// the episode immediately with no observation - the right default for
    /// a plain prompt-in/completion-out task, overridden by environments
    /// with real turn structure.
    fn step(&self, task: &Task, transcript: &[Step], action: &[u32]) -> StepOutcome {
        let _ = (task, transcript, action);
        StepOutcome { observation: Vec::new(), done: true }
    }

    /// The action subset a policy should be constrained to at a given step
    /// of a given task, or `None` for no constraint (the default). See the
    /// module doc comment on why this lives on the trait rather than being
    /// left to caller convention.
    fn legal_actions(&self, task: &Task, step: usize) -> Option<&[u32]> {
        let _ = (task, step);
        None
    }
}

/// A deterministic, programmatic reward: a scalar plus its breakdown.
/// `parts` is a `BTreeMap` (not a `HashMap`) so re-serializing a stored
/// reward is byte-stable - part of what makes a run artifact re-scorable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Reward {
    /// The scalar reward a training objective consumes.
    pub value: f32,
    /// Named contributions to `value` (e.g. `"exact_match"`,
    /// `"format_ok"`) - kept alongside the scalar so a reward is
    /// inspectable and re-derivable, not just a bare number.
    pub parts: BTreeMap<String, f32>,
}

/// Scores a completion against a task, deterministically and
/// programmatically - never by asking a model to judge. `transcript` is
/// the full sequence of turns taken (empty for a single-turn episode
/// scored purely from `completion`); `completion` is the policy's final
/// output.
pub trait Verifier {
    fn verify(&self, task: &Task, transcript: &[Step], completion: &[u32]) -> Reward;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// A tiny deterministic task family: the answer is a token sequence the
    /// completion must match exactly. No model-as-judge anywhere - the
    /// verifier recomputes correctness from `Task.answer` alone.
    struct EchoEnv;

    impl Environment for EchoEnv {
        fn name(&self) -> &str {
            "toy-echo"
        }

        fn tasks(&self, seed: u64) -> Vec<Task> {
            vec![Task { id: format!("echo-{seed}"), prompt: vec![1, 2, 3], answer: serde_json::json!([4, 5]) }]
        }
    }

    struct ExactMatchVerifier;

    impl Verifier for ExactMatchVerifier {
        fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
            let expected: Vec<u32> = serde_json::from_value(task.answer.clone()).expect("answer is a token array");
            let value = if expected == completion { 1.0 } else { 0.0 };
            Reward { value, parts: BTreeMap::from([("exact_match".to_string(), value)]) }
        }
    }

    #[test]
    fn exact_match_verifier_recomputes_correctness_from_task_answer_deterministically() {
        let env = EchoEnv;
        let task = env.tasks(0).into_iter().next().unwrap();
        let verifier = ExactMatchVerifier;

        assert_eq!(verifier.verify(&task, &[], &[4, 5]).value, 1.0);
        assert_eq!(verifier.verify(&task, &[], &[4, 6]).value, 0.0);
        // Determinism: identical inputs must produce an identical reward,
        // not merely an identical value - the whole breakdown re-derives.
        assert_eq!(verifier.verify(&task, &[], &[4, 5]), verifier.verify(&task, &[], &[4, 5]));
    }

    #[test]
    fn default_step_is_single_turn_done_immediately() {
        struct SingleTurn;
        impl Environment for SingleTurn {
            fn name(&self) -> &str {
                "toy-single-turn"
            }
            fn tasks(&self, _seed: u64) -> Vec<Task> {
                vec![Task { id: "t".to_string(), prompt: vec![], answer: serde_json::Value::Null }]
            }
        }
        let env = SingleTurn;
        let task = env.tasks(0).into_iter().next().unwrap();
        let outcome = env.step(&task, &[], &[7]);
        assert!(outcome.done, "an Environment that does not override step() must end after one turn");
        assert!(outcome.observation.is_empty());
        assert!(env.legal_actions(&task, 0).is_none(), "default legal_actions is unconstrained");
    }

    /// A tiny multi-step counting environment: each turn plays one of the
    /// declared legal actions (+1, +2, or stop); the episode ends once the
    /// running sum reaches the task's target or the agent plays stop. The
    /// target is recomputable from `Task.answer` alone, never leaked into
    /// `prompt`.
    struct CounterEnv;
    const INC1: u32 = 1;
    const INC2: u32 = 2;
    const STOP: u32 = 0;
    const LEGAL: [u32; 3] = [INC1, INC2, STOP];

    impl Environment for CounterEnv {
        fn name(&self) -> &str {
            "toy-counter"
        }

        fn tasks(&self, seed: u64) -> Vec<Task> {
            vec![Task { id: format!("counter-{seed}"), prompt: vec![], answer: serde_json::json!({"target": 5}) }]
        }

        fn step(&self, task: &Task, transcript: &[Step], action: &[u32]) -> StepOutcome {
            let target = task.answer["target"].as_u64().expect("counter task answer has a target");
            let sum: u64 =
                transcript.iter().flat_map(|s| s.action.iter()).chain(action.iter()).map(|&a| a as u64).sum();
            let done = action.contains(&STOP) || sum >= target;
            StepOutcome { observation: vec![sum as u32], done }
        }

        fn legal_actions(&self, _task: &Task, _step: usize) -> Option<&[u32]> {
            Some(&LEGAL)
        }
    }

    struct CounterVerifier;
    impl Verifier for CounterVerifier {
        fn verify(&self, task: &Task, transcript: &[Step], completion: &[u32]) -> Reward {
            let target = task.answer["target"].as_u64().expect("counter task answer has a target");
            let sum: u64 =
                transcript.iter().flat_map(|s| s.action.iter()).chain(completion.iter()).map(|&a| a as u64).sum();
            let value = if sum == target { 1.0 } else { 0.0 };
            Reward { value, parts: BTreeMap::from([("reached_target".to_string(), value)]) }
        }
    }

    #[test]
    fn multi_step_step_runs_a_bounded_episode_constrained_to_legal_actions() {
        let env = CounterEnv;
        let task = env.tasks(7).into_iter().next().unwrap();

        let mut transcript: Vec<Step> = Vec::new();
        let mut turns = 0;
        loop {
            let legal = env.legal_actions(&task, turns).expect("counter env always declares legal actions");
            assert_eq!(legal, LEGAL, "legal_actions must constrain exploration identically every turn");
            let action = vec![INC2];
            assert!(legal.contains(&action[0]));
            let outcome = env.step(&task, &transcript, &action);
            transcript.push(Step { action, observation: outcome.observation });
            turns += 1;
            if outcome.done {
                break;
            }
            assert!(turns < 100, "runaway episode - step() never reported done");
        }

        // +2 each turn against target 5: 2, 4, 6 -> done on the 3rd turn.
        assert_eq!(turns, 3);

        let verifier = CounterVerifier;
        let reward = verifier.verify(&task, &transcript, &[]);
        assert_eq!(reward.value, 0.0, "overshooting the target (6 vs 5) must not verify as reached");
        assert_eq!(reward.parts.get("reached_target"), Some(&0.0));

        // A transcript that lands exactly on target does verify.
        let exact_transcript = vec![
            Step { action: vec![INC2], observation: vec![] },
            Step { action: vec![INC2], observation: vec![] },
            Step { action: vec![INC1], observation: vec![] },
        ];
        let reward = verifier.verify(&task, &exact_transcript, &[]);
        assert_eq!(reward.value, 1.0);
    }
}
