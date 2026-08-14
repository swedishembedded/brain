// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cross-check `qwen3omnimoe::config::OmniConfig` against the REAL released
//! `config.json`, not just the inline sample in `config.rs`'s unit tests.
//!
//! Real-weight-adjacent, so it follows the engine's standard opt-in-env-var
//! pattern: skips (never panics) when the checkpoint dir
//! is not present.
//!
//! usage: `BRAIN_QWEN3OMNIMOE_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test real_config -- --ignored`

use std::path::PathBuf;

fn hf_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("BRAIN_QWEN3OMNIMOE_HF_DIR").ok()?);
    d.join("config.json").exists().then_some(d)
}

#[test]
#[ignore]
fn matches_the_released_checkpoint() {
    let Some(dir) = hf_dir() else {
        eprintln!("skip: BRAIN_QWEN3OMNIMOE_HF_DIR unset or config.json missing");
        return;
    };
    let json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let c = qwen3omnimoe::config::OmniConfig::parse(&json).expect("parse");

    // Every number here is dumped straight from the released config.json —
    // this test just proves
    // the PARSER reproduces them from the real file, not a hand-copied one.
    assert_eq!(c.thinker.audio.n_layers, 32);
    assert_eq!(c.thinker.audio.d_model, 1280);
    assert_eq!(c.thinker.audio.n_heads, 20);
    assert_eq!(c.thinker.audio.ffn_dim, 5120);
    assert_eq!(c.thinker.audio.num_mel_bins, 128);
    assert_eq!(c.thinker.audio.n_window_infer, 800);
    assert_eq!(c.thinker.audio.output_dim, 2048);

    assert_eq!(c.thinker.vision.depth, 27);
    assert_eq!(c.thinker.vision.hidden, 1152);
    assert_eq!(c.thinker.vision.num_heads, 16);
    assert_eq!(c.thinker.vision.intermediate, 4304);
    assert_eq!(c.thinker.vision.deepstack_indexes, vec![8, 16, 24]);
    assert_eq!(c.thinker.vision.out_hidden_size, 2048);
    assert!(c.thinker.vision.apply_vit_abs_pos_embed);

    assert_eq!(c.thinker.text.n_layers, 48);
    assert_eq!(c.thinker.text.hidden, 2048);
    assert_eq!(c.thinker.text.n_heads, 32);
    assert_eq!(c.thinker.text.n_kv_heads, 4);
    assert_eq!(c.thinker.text.n_experts, 128);
    assert_eq!(c.thinker.text.top_k, 8);
    assert!(!c.thinker.text.has_shared_expert());
    assert!(c.thinker.text.use_qk_norm);
    assert_eq!(c.thinker.text.vocab, 152064);
    assert_eq!(c.thinker.text.mrope_section, vec![24, 20, 20]);

    assert_eq!(c.talker.text.n_layers, 20);
    assert_eq!(c.talker.text.hidden, 1024);
    assert_eq!(c.talker.text.n_experts, 128);
    assert_eq!(c.talker.text.top_k, 6);
    assert_eq!(c.talker.text.shared_expert_intermediate, 768);
    assert_eq!(c.talker.text.vocab, 3072);
    assert_eq!(c.talker.accept_hidden_layer, 24);
    assert_eq!(c.talker.speaker_id.get("chelsie"), Some(&2301));
    assert_eq!(c.talker.speaker_id.get("ethan"), Some(&2302));
    assert_eq!(c.talker.speaker_id.get("aiden"), Some(&2303));

    assert_eq!(c.talker.code_predictor.n_layers, 5);
    assert_eq!(c.talker.code_predictor.num_code_groups, 16);
    assert_eq!(c.talker.code_predictor.vocab, 2048);

    assert_eq!(c.code2wav.num_quantizers, 16);
    assert_eq!(c.code2wav.num_semantic_quantizers, 1);
    assert_eq!(c.code2wav.codebook_size, 2048);
    assert_eq!(c.code2wav.semantic_codebook_size, 4096);
    assert_eq!(c.code2wav.hidden_size, 1024);
    assert_eq!(c.code2wav.intermediate_size, 3072);
    assert_eq!(c.code2wav.sliding_window, 72);
    assert_eq!(c.code2wav.decoder_dim, 1536);
    assert_eq!(c.code2wav.upsample_rates, vec![8, 5, 4, 3]);
    assert_eq!(c.code2wav.upsampling_ratios, vec![2, 2]);
    assert_eq!(c.code2wav.total_upsample(), 1920);

    println!("qwen3omnimoe::config::OmniConfig parses the real checkpoint's config.json exactly.");
}
