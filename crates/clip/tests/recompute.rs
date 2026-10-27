// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of vision transformer encoders for its clients. If your
// team needs expertise in training CLIP and SigLIP image towers within a fixed device budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Activation recompute for the CLIP / SigLIP image tower, held to the contract `model::saved::check_recompute`
//! states: a tower whose blocks share one set of intermediates and rebuild a block's from its input before its
//! backward gives the output and every gradient (the parameters' and the injected patch tokens') of the tower that
//! keeps every block, bit for bit, holding fewer device bytes, for both stems and both patch sources. Narrow formats
//! and going back to keeping are refused by name. One test in its own binary, so nothing else allocates while the
//! device bytes are compared.

use clip::config::{ClipVisionConfig, TextAct, VisionStem};
use clip::model::{ClipVision, PatchSource, CLIP_VISION_PIPELINES};
use data::rng::Rng;
use model::saved::{SavedFormat, SavedPolicy};

/// Pairwise-distinct dimensions with five blocks; `native` is the learned position grid's side.
fn config(stem: VisionStem) -> ClipVisionConfig {
    let native_patches = 25;
    ClipVisionConfig {
        shape: gguf::deepseek_ocr_vision::ClipConfig {
            d_model: 32,
            n_layers: 5,
            n_heads: 4,
            ffn_hidden: 20,
            patch_size: 2,
            image_size: 10,
            n_positions: native_patches + stem.prefix_tokens(),
            layer_norm_eps: 1e-5,
        },
        act: TextAct::GeluErf,
        stem,
    }
}

/// The output bits, every parameter's gradient bits, the patch tokens' gradient bits and the device bytes the tower
/// holds after the policy is set.
type Measured = (Vec<u32>, Vec<(String, Vec<u32>)>, Vec<u32>, u64);

fn step(cfg: &ClipVisionConfig, src: PatchSource, init: &std::collections::HashMap<String, Vec<f32>>, seed: u64, policy: SavedPolicy) -> Measured {
    let gpu = gpu_core::testgpu::dev(CLIP_VISION_PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let mut m = ClipVision::new_train_on(gpu.share(), cfg.clone(), 1, src, init);
    m.set_saved_policy(policy).unwrap();
    let held = gpu.memory_usage().unwrap().live - before;
    match src {
        PatchSource::Pixels => m.set_pixels(&clip::init::fixed_pixels(cfg, 1, seed)),
        PatchSource::Tokens { grid } => m.set_tokens(&clip::init::fixed_tokens_grid(cfg, 1, grid, seed)),
    }
    let mut rng = Rng::new(seed ^ 1);
    let seed: Vec<f32> = (0..m.out_len()).map(|_| rng.next_f32() - 0.5).collect();
    m.zero_grads();
    m.forward();
    m.backward(&seed);
    m.poll_wait();
    let mut names = m.dispatched_params();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), m.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    let tokens = m.read_token_grad().into_iter().map(f32::to_bits).collect();
    (m.read_output().into_iter().map(f32::to_bits).collect(), grads, tokens, held)
}

fn check(tag: &str, cfg: &ClipVisionConfig, src: PatchSource) {
    let seed = data::rng::random_seed();
    let init = clip::init::init_vision_weights(cfg, seed);
    let (want, want_grads, want_tokens, keep) = step(cfg, src, &init, seed, SavedPolicy::Keep);
    let (got, got_grads, got_tokens, recompute) = step(cfg, src, &init, seed, SavedPolicy::Recompute);
    assert!(got == want, "{tag}: the output differs recomputing");
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "{tag}: no parameter received a gradient");
    for ((n, a), (_, b)) in want_grads.iter().zip(&got_grads) {
        assert!(a == b, "{tag}: {n}: gradient differs recomputing");
    }
    assert!(got_tokens == want_tokens, "{tag}: the patch-token gradient differs recomputing");
    assert!(recompute < keep, "{tag}: recomputing holds {recompute} device bytes, keeping {keep}");
}

#[test]
fn the_image_tower_recomputes_its_blocks_exactly() {
    check("clip pixels", &config(VisionStem::Clip), PatchSource::Pixels);
    check("clip tokens", &config(VisionStem::Clip), PatchSource::Tokens { grid: (3, 7) });
    check("siglip pixels", &config(VisionStem::Siglip), PatchSource::Pixels);

    // Narrow saved formats do not apply, and a recomputing tower does not go back to keeping.
    let cfg = config(VisionStem::Clip);
    let init = clip::init::init_vision_weights(&cfg, data::rng::random_seed());
    let mut m = ClipVision::new_train_on(gpu_core::testgpu::dev(CLIP_VISION_PIPELINES), cfg, 1, PatchSource::Pixels, &init);
    assert!(m.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("clip"));
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(m.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("clip"));
}
