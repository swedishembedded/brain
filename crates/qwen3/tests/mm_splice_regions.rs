// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements vision-language fine-tuning on consumer
// GPUs for its clients. If your team needs expertise in training multimodal
// models within a fixed memory budget then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A training example may carry several images: the decoder splices each
//! image's rows over its own run of the sequence, and the gradient of every
//! run comes back in image order. The gradient of the second image is checked
//! against a finite difference of the loss, and the placeholder ids under the
//! images are shown not to matter.

use data::rng::Rng;
use qwen3::{Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const REGIONS: [(u32, u32); 2] = [(1, 2), (5, 3)];

fn model(cfg: &QwenConfig) -> Qwen {
    let init = qwen3::init_weights(cfg, 17);
    let mut m = Qwen::new(cfg.clone(), 1, 10, &init);
    m.enable_mm_splices(&REGIONS);
    m
}

fn loss(m: &Qwen, tokens: &[u32], targets: &[u32], embeds: &[f32]) -> f32 {
    m.write_img_embeds(embeds);
    m.set_batch(tokens, targets);
    m.forward()
}

#[test]
fn every_image_of_an_example_is_spliced_and_gets_its_own_gradient() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig { block_size: 10, ..QwenConfig::tiny() };
    let d = cfg.d_model as usize;
    let m = model(&cfg);
    let mut rng = Rng::new(3);
    let rows: usize = REGIONS.iter().map(|r| r.1 as usize).sum();
    let embeds: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian() as f32 * 0.3).collect();
    let tokens = [3u32, 9, 9, 4, 6, 9, 9, 9, 7, 2];
    let targets = [9u32, 4, 6, 7, 2, 5, 8, 3, 1, 6];

    // The ids under the images are overwritten, so changing them changes nothing.
    let base = loss(&m, &tokens, &targets, &embeds);
    let mut other = tokens;
    for (row0, n) in REGIONS {
        for t in &mut other[row0 as usize..(row0 + n) as usize] {
            *t = (*t + 5) % cfg.vocab;
        }
    }
    assert_eq!(loss(&m, &other, &targets, &embeds), base, "the placeholder ids under the images do not reach the loss");

    // The gradient of the second image's rows against a finite difference.
    m.set_batch(&tokens, &targets);
    m.write_img_embeds(&embeds);
    m.zero_grads();
    m.forward();
    m.backward();
    let grad = m.read_d_img_embeds();
    assert_eq!(grad.len(), embeds.len(), "one gradient row per spliced image row, in image order");
    let h = 1e-2f32;
    for at in [2 * d + 3, 3 * d + 1, rows * d - 2] {
        let (mut plus, mut minus) = (embeds.clone(), embeds.clone());
        plus[at] += h;
        minus[at] -= h;
        let fd = (loss(&m, &tokens, &targets, &plus) - loss(&m, &tokens, &targets, &minus)) / (2.0 * h);
        assert!((fd - grad[at]).abs() < 2e-3 + 0.02 * fd.abs(), "d loss / d embed[{at}]: finite difference {fd}, backward {}", grad[at]);
    }
    assert!(grad[..2 * d].iter().any(|g| g.abs() > 1e-6), "the first image receives gradient too");
}
