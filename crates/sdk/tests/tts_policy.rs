// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "audio")]

//! A synthesizer that can be trained from scores of its own takes: a group of
//! takes of a sentence is drawn, scored by the caller, and the policy moves
//! toward the takes that scored above their group. Needs the Qwen3-TTS
//! checkpoint in the model store and a device; skips without them.

use brain::{PolicyOptions, PolicyUpdate, TtsOptions, TtsPipeline};

const MODEL: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
const TEXT: &str = "The people must be taught to value their liberty.";

#[test]
fn takes_scored_above_their_group_become_likelier_and_the_next_take_changes() {
    let _serial = brain_testutil::env_lock();
    let Ok(pipe) = TtsPipeline::from_pretrained(MODEL) else {
        brain_testutil::skip("Qwen3-TTS checkpoint is not in the model store");
        return;
    };
    let Ok(mut policy) = pipe.policy(PolicyOptions { block: 160, ..PolicyOptions::default() }) else {
        brain_testutil::skip("no device for the trainable synthesizer");
        return;
    };
    let opts = |seed| TtsOptions::new().seed(seed).max_frames(40);
    let before = policy.rollout(TEXT, &opts(21)).unwrap();
    let group = [policy.rollout(TEXT, &opts(22)).unwrap(), policy.rollout(TEXT, &opts(23)).unwrap()];

    let mut last = None;
    for _ in 0..4 {
        let stats = policy.update(&[(&group[0], 1.0), (&group[1], -1.0)], &PolicyUpdate { lr: 3e-4, kl_beta: 0.0, clip: 1.0 }).unwrap();
        last = Some(stats);
    }
    let stats = last.unwrap();
    assert!(stats.tokens > 0 && stats.loss.is_finite() && stats.mean_logprob.is_finite(), "{stats:?}");

    let after = policy.rollout(TEXT, &opts(21)).unwrap();
    assert_ne!(before.audio().samples(), after.audio().samples(), "the same seed gives another take once the policy has moved");
}
