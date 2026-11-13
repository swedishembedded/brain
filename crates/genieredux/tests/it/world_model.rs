// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning generative video models
// into interactive, replayable simulators for its clients. If your team needs
// expertise in action-conditioned world models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! `GenieReduxWorldModel` as a `wm_core::WorldModel`: the trait's contracts
//! (frame shape and range, deterministic `reset`, `reset_initial` replaying
//! the same sequence, action range) and equality with the uncached pipeline
//! (tokenizer encode, `maskgit_sample` over the whole window, decode of the
//! whole window), which is the reference loop the cached session must not
//! change.

use crate::tiny;
use gpu_core::Gpu;
use genieredux::{
    decode_indices, kernel_sources, maskgit_sample, tokenizer_forward, GenieReduxWorldModel,
    MaskSchedule, MaskgitSettings,
};
use wm_core::WorldModel;

const HW: usize = tiny::HW;

fn greedy() -> MaskgitSettings {
    MaskgitSettings { inference_steps: 1, schedule: MaskSchedule::Cosine, sample_temperature: 1.0 }
}

fn model(max_context: usize, seed: u64) -> GenieReduxWorldModel {
    let tw = tiny::tokenizer();
    let dw = tiny::dynamics(&tw);
    GenieReduxWorldModel::new(
        Gpu::new_cpu(&kernel_sources()), tw, tiny::tokenizer_config(), dw, tiny::dynamics_config(),
        tiny::GRID, max_context, greedy(), seed,
    )
    .expect("tiny model is valid")
}

fn play(m: &mut GenieReduxWorldModel, actions: &[u32]) -> Vec<Vec<f32>> {
    actions.iter().map(|&a| m.step(a)).collect()
}

fn bits(frames: &[Vec<f32>]) -> Vec<Vec<u32>> {
    frames.iter().map(|f| f.iter().map(|x| x.to_bits()).collect()).collect()
}

#[test]
fn trait_geometry_and_frame_contract() {
    let mut m = model(6, 1);
    assert_eq!(m.frame_shape(), (3, 8, 8));
    assert_eq!(m.num_actions(), 3);
    m.reset(&tiny::frames(5, 2), &[]);
    for f in play(&mut m, &[0, 1, 2]) {
        assert_eq!(f.len(), tiny::FRAME_LEN);
        assert!(f.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    }
}

#[test]
fn reset_is_a_deterministic_function_of_the_context() {
    let (ctx, acts) = (tiny::frames(5, 2), [2u32]);
    let mut a = model(6, 9);
    let mut b = model(6, 9);
    a.reset(&ctx, &acts);
    b.reset(&tiny::frames(77, 3), &[1, 1]); // a different life first
    play(&mut b, &[1, 0]);
    b.reset(&ctx, &acts);
    let seq = [1u32, 2, 0, 2];
    assert_eq!(bits(&play(&mut a, &seq)), bits(&play(&mut b, &seq)));
}

#[test]
fn reset_initial_replays_the_same_sequence() {
    let mut m = model(6, 3);
    m.reset(&tiny::frames(8, 2), &[1]);
    let seq = [0u32, 2, 1, 1];
    let first = bits(&play(&mut m, &seq));
    m.reset_initial();
    assert_eq!(first, bits(&play(&mut m, &seq)));
    // And it returns to the START, not to the state after the replay.
    m.reset_initial();
    assert_eq!(first[0], bits(&play(&mut m, &seq[..1]))[0]);
}

#[test]
fn empty_context_is_accepted_and_deterministic() {
    let mut m = model(6, 4);
    m.reset(&[], &[]);
    let a = bits(&play(&mut m, &[1, 2]));
    m.reset(&[], &[]);
    assert_eq!(a, bits(&play(&mut m, &[1, 2])));
}

#[test]
fn out_of_range_action_clamps_to_the_last_action() {
    let ctx = tiny::frames(5, 2);
    let (mut a, mut b) = (model(6, 2), model(6, 2));
    a.reset(&ctx, &[]);
    b.reset(&ctx, &[]);
    assert_eq!(bits(&play(&mut a, &[0, 99])), bits(&play(&mut b, &[0, 2])));
}

/// The uncached reference loop over a sliding window of `max_context` frames.
fn reference_run(ctx_n: usize, ctx_actions: &[u32], max_context: usize, steps: &[u32]) -> Vec<Vec<f32>> {
    let gpu = Gpu::new_cpu(&kernel_sources());
    let (tw, tc, dc) = (tiny::tokenizer(), tiny::tokenizer_config(), tiny::dynamics_config());
    let dw = tiny::dynamics(&tw);
    let ctx = tiny::frames(5, ctx_n);
    let (_, mut tokens) = tokenizer_forward(
        &gpu, &tiny::to_video(&ctx, ctx_n), &tw, 1, 3, ctx_n as u32, 8, 8, 4, tc.dim, tc.heads,
        tc.head_dim, tc.code_dim, tc.n_codes,
    );
    let mut actions: Vec<u32> = (0..ctx_n).map(|f| ctx_actions.get(f).copied().unwrap_or(0)).collect();
    let mut out = Vec::new();
    for &a in steps {
        *actions.last_mut().unwrap() = a.min(2);
        let t = tokens.len() / HW;
        let new = maskgit_sample(
            &gpu, &tokens, &tiny::one_hot(&actions, 3), HW, 2, 2, &dw, &dc, &greedy(), 0,
        );
        tokens.extend(&new);
        let video = decode_indices(&gpu, &tokens, &tw, 1, (t + 1) as u32, 2, 2, tc.dim, tc.heads, tc.head_dim, tc.code_dim);
        let n = t + 1;
        let frame: Vec<f32> = (0..3)
            .flat_map(|c| video[(c * n + t) * 64..(c * n + t + 1) * 64].iter().map(|v| v.clamp(0.0, 1.0)))
            .collect();
        out.push(frame);
        actions.push(0);
        if tokens.len() / HW > max_context {
            tokens.drain(..HW);
            actions.remove(0);
        }
    }
    out
}

#[test]
fn cached_session_equals_the_uncached_reference_loop() {
    // Window 6 never slides in 4 steps (cache grows); window 3 slides from step 2.
    for max_context in [6usize, 3] {
        let steps = [1u32, 2, 0, 2];
        let mut m = model(max_context, 5);
        m.reset(&tiny::frames(5, 2), &[2]);
        let got = play(&mut m, &steps);
        let want = reference_run(2, &[2], max_context, &steps);
        assert_eq!(bits(&got), bits(&want), "max_context {max_context}");
    }
}

#[test]
fn construction_rejects_inconsistent_models() {
    let tw = tiny::tokenizer();
    let gpu = || Gpu::new_cpu(&kernel_sources());
    let build = |max_context: usize, dc: genieredux::import::DynamicsConfig| {
        GenieReduxWorldModel::new(
            gpu(), tiny::tokenizer(), tiny::tokenizer_config(), tiny::dynamics(&tw), dc, tiny::GRID,
            max_context, greedy(), 1,
        )
        .err()
    };
    assert!(build(0, tiny::dynamics_config()).is_some(), "zero context frames");
    // 64 positions / 4 tokens per frame: 17 frames do not fit the position table.
    assert!(build(17, tiny::dynamics_config()).is_some(), "window longer than max_seq_len");
    let mut dc = tiny::dynamics_config();
    dc.n_codes = 32;
    assert!(build(4, dc).is_some(), "codebook size disagrees with the tokenizer");
}
