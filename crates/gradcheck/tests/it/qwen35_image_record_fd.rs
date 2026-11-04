// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying that fine-tuning a vision-language model trains correctly on
// the images it reads, for its clients. If your team needs expertise in gradient verification for multimodal training
// then you can procure our services by sending an email to info@swedishembedded.com.

//! The LoRA adapters of the hybrid decoder, trained on a record whose user turn holds an image, match central finite
//! differences: with the vision tower's rows spliced over the placeholder tokens and the three-axis rotary positions
//! of the prompt in the tables. The rows are inputs of a frozen tower (no gradient is kept for them), so the check is
//! on the adapters alone, through the attention and the recurrence that read the image.

use data::rng::Rng;
use gpu_core::Gpu;
use gradcheck::directional_check;
use model::serve::PromptEmbeds;
use model::IGNORE;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

const IMAGE: u32 = 27;

#[test]
fn the_adapters_match_finite_differences_on_a_record_with_an_image() {
    let cfg = Qwen35Config { lora: Some(lora_cfg(2, 4.0)), ..Qwen35Config::tiny() };
    let (t, d) = (12usize, cfg.d_model as usize);
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, t as u32, &init);

    // "text, a 2 x 2 image, text": positions 3..7 hold the image's rows, whose rotary positions are the 2 x 2 grid's.
    let tokens: Vec<u32> = vec![1, 2, 3, IMAGE, IMAGE, IMAGE, IMAGE, 4, 5, 6, 7, 8];
    let positions: Vec<u32> = (3..7).collect();
    // seed: fixed rows, so the differences are taken around the same frozen-tower output every run
    let mut rng = Rng::new(0x1A6E);
    let rows: Vec<f32> = (0..positions.len() * d).map(|_| rng.next_f32() - 0.5).collect();
    let rotary = qwen3vl::mrope::get_rope_index(&tokens, IMAGE, &[(1, 2, 2)]);
    assert_ne!(rotary[7], [7, 7, 7], "the image spans two positions, not four: the text after it is shifted");
    let embeds = PromptEmbeds::new(d, positions, rows).unwrap().with_rotary_positions(rotary);
    // Supervised: the text after the image, never a placeholder row or the position that predicts the first of them.
    let targets: Vec<u32> = (0..t).map(|i| if (2..7).contains(&i) { IGNORE } else { tokens.get(i + 1).copied().unwrap_or(9) }).collect();
    m.set_batch_embeds(&tokens, &targets, &[Some(embeds)]);

    // Off the invisible start, so every tensor carries gradient.
    for step in 1..=5 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let report = directional_check(&m, 1.25e-3, 4, 7 ^ 0x1234);
    let worst = report.checks.iter().fold((0.0f32, 0.0f32), |(a, r), c| (a.max(c.abs_err), r.max(c.rel_err)));
    eprintln!("{} adapter tensors checked on the image record: worst abs error {:e}, worst rel error {:e}", report.checks.len(), worst.0, worst.1);
    let fails = report.failures(4e-3, 8e-2);
    assert!(fails.is_empty(), "{:?}", fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>());
    assert!(report.dead_gradients().is_empty(), "dead gradients");
    assert!(!report.checks.is_empty());
}
