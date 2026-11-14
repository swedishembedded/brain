// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying iterative token
// samplers against their reference for its clients. If your team needs
// expertise in generative world models then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The guided-MaskGIT sampler's spec, from the reference `Dynamics.sample`
//! (GenieRedux `models/dynamics.py`): the re-mask schedule, the overwrite rule,
//! confidence selection, temperature decay and seeding. The loop is driven
//! with a synthetic forward so each rule is observed directly; two tests then
//! tie it to the real dynamics forward on a tiny random model.

use crate::tiny;
use gpu_core::Gpu;
use genieredux::{
    dynamics_forward, kernel_sources, maskgit_sample, maskgit_sample_with, MaskSchedule,
    MaskgitSettings,
};

fn settings(steps: usize, schedule: MaskSchedule, temperature: f32) -> MaskgitSettings {
    MaskgitSettings { inference_steps: steps, schedule, sample_temperature: temperature }
}

/// Logits `[n, nc]` where token `i` strongly prefers class `(i + pass) % nc`.
fn shifting_logits(n: usize, nc: usize, pass: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n * nc];
    for i in 0..n {
        l[i * nc + (i + pass) % nc] = 10.0;
    }
    l
}

fn masked_count(vids: &[u32], mask_id: u32) -> usize {
    vids.iter().filter(|&&v| v == mask_id).count()
}

#[test]
fn schedule_parses_the_reference_names() {
    assert_eq!(MaskSchedule::parse("cosine"), Ok(MaskSchedule::Cosine));
    assert_eq!(MaskSchedule::parse("uniform"), Ok(MaskSchedule::Uniform));
    assert_eq!(MaskSchedule::parse("pow2"), Ok(MaskSchedule::Pow(2.0)));
    assert_eq!(MaskSchedule::parse("log"), Ok(MaskSchedule::Log));
    assert_eq!(MaskSchedule::parse("exp"), Ok(MaskSchedule::Exp));
    assert!(MaskSchedule::parse("bogus").is_err());
}

/// Number of positions masked going into each pass, per schedule.
fn masked_per_pass(schedule: MaskSchedule, steps: usize, n: usize) -> Vec<usize> {
    let nc = 4usize;
    let mut seen = Vec::new();
    maskgit_sample_with(n, nc as u32, &settings(steps, schedule, 0.0), 1, |vids| {
        seen.push(masked_count(vids, nc as u32));
        shifting_logits(n, nc, seen.len())
    });
    seen
}

#[test]
fn cosine_schedule_masks_int_of_cos_ratio_times_n() {
    // pass 0 masks everything; pass k masks int(cos(((k+1)/steps) * pi/2) * n),
    // the last pass clamping to 1e-6 -> int(1e-4) = 0.
    assert_eq!(masked_per_pass(MaskSchedule::Cosine, 4, 100), vec![100, 70, 38, 0]);
}

#[test]
fn uniform_and_pow_schedules_follow_their_formulas() {
    assert_eq!(masked_per_pass(MaskSchedule::Uniform, 4, 100), vec![100, 50, 25, 0]);
    assert_eq!(masked_per_pass(MaskSchedule::Pow(2.0), 4, 100), vec![100, 75, 43, 0]);
}

#[test]
fn every_pass_overwrites_every_position_with_its_own_prediction() {
    // The reference's `torch.where(mask, pred, video_token_ids)` is commented
    // out: video_token_ids = pred_video_ids. So the final tokens are the LAST
    // pass's predictions everywhere, including positions it did not re-mask.
    let (n, nc, steps) = (12usize, 8usize, 3usize);
    let mut inputs: Vec<Vec<u32>> = Vec::new();
    let out = maskgit_sample_with(n, nc as u32, &settings(steps, MaskSchedule::Cosine, 0.0), 1, |vids| {
        inputs.push(vids.to_vec());
        shifting_logits(n, nc, inputs.len() - 1)
    });
    let last = steps - 1;
    assert_eq!(out, (0..n).map(|i| ((i + last) % nc) as u32).collect::<Vec<_>>());
    // Unmasked positions entering pass k carry pass k-1's prediction.
    assert_eq!(inputs.len(), steps, "one recorded input per pass");
    for (k, pass) in inputs.iter().enumerate().skip(1) {
        for (j, &v) in pass.iter().enumerate() {
            if v != nc as u32 {
                assert_eq!(v as usize, (j + k - 1) % nc, "pass {k} position {j}");
            }
        }
    }
}

#[test]
fn least_confident_positions_are_the_ones_remasked() {
    // Token i predicts class 0 with logit x_i = 0.3 i against 0 elsewhere, so
    // confidence rises with i. Pass 1 of 3 re-masks int(cos(pi/3) * 11) = int(5.5) = 5.
    let (n, nc) = (11usize, 4usize);
    let mut second: Vec<u32> = Vec::new();
    let mut pass = 0;
    maskgit_sample_with(n, nc as u32, &settings(3, MaskSchedule::Cosine, 0.0), 1, |vids| {
        if pass == 1 {
            second = vids.to_vec();
        }
        pass += 1;
        let mut l = vec![0.0f32; n * nc];
        for i in 0..n {
            l[i * nc] = 0.3 * i as f32;
        }
        l
    });
    let masked: Vec<usize> = (0..n).filter(|&i| second[i] == nc as u32).collect();
    assert_eq!(masked, vec![0, 1, 2, 3, 4]);
}

#[test]
fn equal_confidence_ties_remask_the_lowest_positions_first() {
    // torch.topk leaves ties unspecified; the port breaks them by position.
    let (n, nc) = (11usize, 4usize);
    let mut second: Vec<u32> = Vec::new();
    let mut pass = 0;
    maskgit_sample_with(n, nc as u32, &settings(3, MaskSchedule::Cosine, 0.0), 1, |vids| {
        if pass == 1 {
            second = vids.to_vec();
        }
        pass += 1;
        vec![0.0f32; n * nc]
    });
    let masked: Vec<usize> = (0..n).filter(|&i| second[i] == nc as u32).collect();
    assert_eq!(masked, vec![0, 1, 2, 3, 4]);
}

#[test]
fn final_pass_is_greedy_whatever_the_temperature() {
    // temperature_k = T * steps_til_x0 / steps, zero on the last pass.
    let (n, nc, steps) = (16usize, 8usize, 3usize);
    let out = maskgit_sample_with(n, nc as u32, &settings(steps, MaskSchedule::Cosine, 1.0e6), 7, |_| {
        shifting_logits(n, nc, 5)
    });
    assert_eq!(out, (0..n).map(|i| ((i + 5) % nc) as u32).collect::<Vec<_>>());
}

#[test]
fn seeded_runs_reproduce_and_the_seed_matters() {
    let (n, nc, steps) = (64usize, 16usize, 4usize);
    // Weak random logits plus a boost on the successor of each kept token, so
    // the noise drawn in early passes reaches the (greedy) last pass through the
    // tokens it is conditioned on.
    let run = |seed: u64| {
        let mut pass = 0;
        maskgit_sample_with(n, nc as u32, &settings(steps, MaskSchedule::Cosine, 1.0), seed, |vids| {
            pass += 1;
            let mut l: Vec<f32> = tiny::rand(100 + pass as u64, n * nc).iter().map(|v| v * 0.05).collect();
            for (i, &v) in vids.iter().enumerate() {
                if v != nc as u32 {
                    l[i * nc + (v as usize + 1) % nc] += 0.3;
                }
            }
            l
        })
    };
    assert_eq!(run(11), run(11));
    assert_ne!(run(11), run(12));
}

/// Reference-path oracle on the tiny dynamics: the last `num_tokens` rows of
/// one forward over `cat(prime, mask frames)[:-1]`, argmaxed.
fn one_pass_oracle(gpu: &Gpu, prime: &[u32], actions: &[f32], num_tokens: usize) -> Vec<u32> {
    let tok = tiny::tokenizer();
    let dw = tiny::dynamics(&tok);
    let dc = tiny::dynamics_config();
    let nc = dc.n_codes as usize;
    let total = prime.len() + num_tokens;
    let t_fwd = total / tiny::HW - 1;
    let mut input = prime.to_vec();
    input.extend(std::iter::repeat_n(dc.n_codes, num_tokens));
    let logits = dynamics_forward(
        gpu, &input[..t_fwd * tiny::HW], actions, &dw, 1, t_fwd as u32, 2, 2,
        dc.dim, dc.heads, dc.head_dim, dc.n_codes, dc.code_dim, dc.num_actions,
    );
    let start = (t_fwd * tiny::HW - num_tokens) * nc;
    (0..num_tokens)
        .map(|t| {
            let row = &logits[start + t * nc..start + (t + 1) * nc];
            row.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
        })
        .collect()
}

#[test]
fn one_pass_equals_the_single_step_path() {
    let gpu = Gpu::new_cpu(&kernel_sources());
    let tok = tiny::tokenizer();
    let dw = tiny::dynamics(&tok);
    let dc = tiny::dynamics_config();
    let prime: Vec<u32> = (0..3 * tiny::HW as u32).map(|i| (i * 5 + 1) % dc.n_codes).collect();
    for frames_out in [1usize, 2] {
        let num_tokens = frames_out * tiny::HW;
        let t_fwd = 3 + frames_out - 1;
        let actions = tiny::one_hot(&(0..t_fwd as u32).map(|a| a % 3).collect::<Vec<_>>(), 3);
        let got = maskgit_sample(
            &gpu, &prime, &actions, num_tokens, 2, 2, &dw, &dc,
            &settings(1, MaskSchedule::Cosine, 1.0), 1,
        );
        assert_eq!(got, one_pass_oracle(&gpu, &prime, &actions, num_tokens), "{frames_out} generated frame(s)");
    }
}
