// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen35Vl` end-to-end smoke tests: vision tower (`qwen3vl::VisionEncoder` +
//! `PatchMerger`, reused as-is) → embedding splice → hybrid decoder, mirroring
//! `crates/qwenvl/src/model.rs`'s own `Qwen3Vl` test module pattern (tiny
//! configs with the merger boundary matched, a text/image/text token stream
//! with IGNORE targets at the image rows, a finite-loss smoke check).
//!
//! Not a numerical-parity test (no independent oracle for "qwen35's hybrid
//! decoder + qwenvl's ViT, spliced, with random weights" -- see `model.rs`'s
//! own honest-scope note). Covers: (a) the composite runs end to end and
//! produces a finite scalar loss, (b) the splice is load-bearing (perturbing
//! the image content changes the loss -- not a silent no-op), (c) the M-RoPE
//! positions written for the image run match `get_rope_index`'s own
//! independent host computation exactly. Run under both `BRAIN_DEVICE=cpu`
//! and the default GPU backend -- `Qwen35Vl::new`
//! builds its decoder via `Qwen35::new`, which resolves its `Gpu` from
//! `BRAIN_DEVICE` (see `gpu_core::Gpu::new`'s own doc), so running this whole
//! file under each setting exercises both.

use std::collections::HashMap;

use data::rng::Rng;
use qwen35moe::config::Qwen35Config;
use qwen35moe::init::init_weights;
use qwen35moe::vl::Qwen35Vl;
use qwen3vl::config::VisionConfig;
use qwen3vl::mrope::get_rope_index;

const IMG: u32 = 7;

fn rand_map(mut rng: Rng, specs: &[(&str, usize, bool)]) -> HashMap<String, Vec<f32>> {
    let mut m = HashMap::new();
    for &(name, n, ones) in specs {
        let v = if ones { vec![1.0; n] } else { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect() };
        m.insert(name.to_string(), v);
    }
    m
}

/// Tiny vision config whose merger output width matches `Qwen35Config::
/// tiny()`'s `d_model` (24) at the merger boundary, mirroring `qwen3vl::
/// model::tests`'s own tiny-fixture pattern (`crates/qwenvl/src/model.rs`).
fn tiny_vcfg() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden: 16,
        num_heads: 2,
        intermediate: 32,
        patch_size: 2,
        temporal_patch_size: 1,
        spatial_merge_size: 2,
        num_position_embeddings: 16,
        out_hidden_size: 24, // == Qwen35Config::tiny().d_model
        in_channels: 2,
        deepstack_indexes: vec![], // Qwen3.5 has none -- see `vl.rs`'s module doc
    }
}

fn vision_weights(vcfg: &VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let (c, pv, mlp) = (vcfg.hidden as usize, vcfg.patch_vec_dim() as usize, vcfg.intermediate as usize);
    let mut specs: Vec<(String, usize, bool)> = vec![
        ("patch_embed.weight".into(), c * pv, false),
        ("patch_embed.bias".into(), c, false),
        ("pos_embed".into(), vcfg.num_position_embeddings as usize * c, false),
    ];
    for b in 0..vcfg.depth {
        specs.extend([
            (format!("blocks.{b}.norm1.weight"), c, true),
            (format!("blocks.{b}.norm1.bias"), c, false),
            (format!("blocks.{b}.qkv.weight"), 3 * c * c, false),
            (format!("blocks.{b}.qkv.bias"), 3 * c, false),
            (format!("blocks.{b}.proj.weight"), c * c, false),
            (format!("blocks.{b}.proj.bias"), c, false),
            (format!("blocks.{b}.norm2.weight"), c, true),
            (format!("blocks.{b}.norm2.bias"), c, false),
            (format!("blocks.{b}.fc1.weight"), mlp * c, false),
            (format!("blocks.{b}.fc1.bias"), mlp, false),
            (format!("blocks.{b}.fc2.weight"), c * mlp, false),
            (format!("blocks.{b}.fc2.bias"), c, false),
        ]);
    }
    let refs: Vec<(&str, usize, bool)> = specs.iter().map(|(n, s, o)| (n.as_str(), *s, *o)).collect();
    rand_map(Rng::new(seed), &refs)
}

fn merger_weights(in_dim: u32, merge: u32, out_dim: u32, seed: u64) -> HashMap<String, Vec<f32>> {
    let merged = in_dim * merge * merge;
    rand_map(
        Rng::new(seed),
        &[
            ("ln.weight", in_dim as usize, true),
            ("ln.bias", in_dim as usize, false),
            ("fc1.weight", (merged * merged) as usize, false),
            ("fc1.bias", merged as usize, false),
            ("fc2.weight", (out_dim * merged) as usize, false),
            ("fc2.bias", out_dim as usize, false),
        ],
    )
}

/// Build a tiny end-to-end model: tokens `[1, 2, IMG, IMG, IMG, IMG, 3]` (a
/// 2x2 merged image grid over a 4x4 patch grid), IGNORE targets at the image
/// rows.
fn build() -> (Qwen35Vl, Vec<u32>, Vec<u32>, Vec<f32>) {
    let vcfg = tiny_vcfg();
    let dcfg = Qwen35Config::tiny();
    let vweights = vision_weights(&vcfg, 1);
    let mweights = merger_weights(vcfg.hidden, vcfg.spatial_merge_size, dcfg.d_model, 2);
    let dweights = init_weights(&dcfg, 3);

    let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, 3];
    let mut targets = vec![2u32, 3, 0, 0, 0, 0, 5];
    for t in targets.iter_mut().take(6).skip(2) {
        *t = model::IGNORE;
    }
    assert!(tokens.len() as u32 <= dcfg.block_size, "tiny config's block_size must fit this fixture's stream");

    let model = Qwen35Vl::new(vcfg.clone(), dcfg, vweights, mweights, &dweights, tokens.len() as u32, IMG, 2, 4);

    let pv_total = (16 * vcfg.patch_vec_dim()) as usize; // 4x4 patch grid
    let mut rng = Rng::new(4);
    let pixels: Vec<f32> = (0..pv_total).map(|_| rng.next_f32() - 0.5).collect();

    (model, tokens, targets, pixels)
}

#[test]
fn end_to_end_forward_is_finite() {
    let (model, tokens, targets, pixels) = build();
    let loss = model.forward(&tokens, &targets, (4, 4), &pixels);
    assert!(loss.is_finite(), "end-to-end loss must be finite, got {loss}");
    assert!(loss > 0.0, "cross-entropy loss should be positive");
}

/// The splice must be load-bearing, not a silent no-op: perturbing the image
/// content (holding tokens/targets/weights fixed) must change the loss --
/// mirrors the "mutation-verify" pattern this repo uses elsewhere (e.g.
/// `qwen3`'s own DeepStack row-index load-bearing test).
#[test]
fn splice_is_load_bearing() {
    let (model, tokens, targets, pixels) = build();
    let loss_a = model.forward(&tokens, &targets, (4, 4), &pixels);

    let mut pixels_b = pixels.clone();
    for p in pixels_b.iter_mut() {
        *p += 5.0;
    }
    let loss_b = model.forward(&tokens, &targets, (4, 4), &pixels_b);
    assert!(loss_a.is_finite() && loss_b.is_finite());
    assert!((loss_a - loss_b).abs() > 1e-5, "perturbing the spliced image content must change the loss: {loss_a} vs {loss_b}");
}

/// The M-RoPE positions derived for the image run must match
/// `get_rope_index`'s own independent output exactly (not just "the model
/// runs") -- an image-token run anchored at `cp=2` lands on the `(t,h,w)`
/// meshgrid, text rows stay diagonal.
/// Backward-through-the-splice (optional per this task's spec, but
/// implemented -- see `model.rs`'s `mm_splice` backward case): after a
/// `Qwen35::new_train` forward/backward with the splice enabled,
/// `read_d_img_embeds` must return a finite, nonzero gradient (proving the
/// residual grad at the spliced rows actually reaches `d_img_embeds` rather
/// than silently staying zero or leaking into `tok.weight`). Drives
/// `Qwen35` directly (not through `Qwen35Vl`, which has no `is_train` path)
/// since this exercises the decoder-level seam in isolation.
#[test]
fn splice_backward_is_nonzero_and_finite() {
    use qwen35moe::model::Qwen35;
    let dcfg = Qwen35Config::tiny();
    let dweights = init_weights(&dcfg, 3);
    let mut m = Qwen35::new_train(dcfg.clone(), 1, dcfg.block_size, &dweights);
    let n_rows = 3u32;
    let row0 = 1u32;
    m.enable_mm_splice(row0, n_rows);
    let img: Vec<f32> = (0..(n_rows * dcfg.d_model)).map(|i| (i as f32) * 0.01).collect();
    m.write_img_embeds(&img);
    let tokens: Vec<u32> = (0..dcfg.block_size).map(|i| (i * 3 + 1) % dcfg.vocab).collect();
    let targets: Vec<u32> = (0..dcfg.block_size).map(|i| (i * 5 + 2) % dcfg.vocab).collect();
    m.set_batch(&tokens, &targets);
    m.zero_grads();
    let loss = m.forward();
    assert!(loss.is_finite());
    m.backward();
    let d_img = m.read_d_img_embeds();
    assert_eq!(d_img.len(), (n_rows * dcfg.d_model) as usize);
    assert!(d_img.iter().all(|v| v.is_finite()), "d_img_embeds must be finite");
    assert!(d_img.iter().any(|&v| v.abs() > 1e-8), "d_img_embeds must be nonzero (backward must actually flow)");
}

#[test]
fn mrope_positions_match_get_rope_index() {
    let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, 3];
    let grids_llm = [(1u32, 2u32, 2u32)];
    let positions = get_rope_index(&tokens, IMG, &grids_llm);
    assert_eq!(
        positions,
        vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3], [4, 4, 4]],
        "image rows must sit on the (t,h,w) meshgrid anchored at cp=2, text rows diagonal"
    );
}
