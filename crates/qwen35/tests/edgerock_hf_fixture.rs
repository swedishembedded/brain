// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for loading HuggingFace model releases as published for its clients. If
// your team needs expertise in keeping a model zoo loadable from upstream layouts then you can procure our services
// by sending an email to info@swedishembedded.com.

// hf-fixture: edgerock
//! A tiny EdgeRock checkpoint in the release's own layout (`model.layers.{i}.*`
//! with no `language_model` level, `memory.<leaf>` tensors on the
//! full-attention layers, no `lm_head.weight`, a top-level `config.json` with
//! the `fixed_memory_*` fields) read through every loader this crate has: the
//! config, the lazy source's two-way coverage, the eager import, and a forward
//! on the imported weights.

mod common;

use common::hf_fixture::write_text_only_checkpoint;
use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn fixture(tag: &str) -> (std::path::PathBuf, Qwen35Config, std::collections::HashMap<String, Vec<f32>>) {
    let cfg = Qwen35Config::edgerock_tiny();
    let values = qwen35::init::init_weights(&cfg, 5);
    (write_text_only_checkpoint(&cfg, tag, &values), cfg, values)
}

#[test]
fn the_release_layout_loads_end_to_end() {
    let (dir, cfg, values) = fixture("edgerock_loads");

    // The config.json reads to the preset the fixture was written from.
    let read = qwen35::fp8_source::hf_config(&dir).expect("config.json parses");
    assert_eq!(format!("{read:?}"), format!("{cfg:?}"));

    // Two-way coverage through the lazy source (shapes only), memory tensors included.
    qwen35::fp8_source::open_dir(&dir, &cfg, 128).expect("every parameter is covered and nothing is left over");

    // The eager import equals the values the files were written from, up to bf16 storage; the three memory
    // RMSNorms come back as their applied `1 + w`, the other memory tensors unchanged.
    let imported = qwen35::import::import_dir(&dir, &cfg, 128).expect("import");
    assert_eq!(imported.len(), cfg.param_list().len());
    assert!(!imported.contains_key("lm_head.weight"), "a tied head has no lm_head tensor");
    for (name, want) in &values {
        let got = &imported[name];
        assert_eq!(got.len(), want.len(), "{name}");
        // bf16 storage keeps 8 bits: a plain norm is stored as `w - 1` of magnitude ~1, so the error floor is absolute.
        let worst = got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs() - 0.01 * b.abs()));
        assert!(worst < 1e-2, "{name}: imported values differ from the written ones by {worst}");
    }
    assert!(imported.contains_key("blocks.3.memory.admission_up.weight") && imported.contains_key("blocks.7.memory.read_gate_bias"));

    // The imported weights run the forward.
    let tokens: Vec<u32> = (0..cfg.block_size).map(|i| i % cfg.vocab).collect();
    let logits = Qwen35::new_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, cfg.block_size, &imported).logits_all(&tokens);
    assert_eq!(logits.len(), tokens.len() * cfg.vocab as usize);
    assert!(logits.iter().all(|x| x.is_finite()));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_checkpoint_missing_a_memory_tensor_is_refused_by_name() {
    let (dir, cfg, _) = fixture("edgerock_refused");
    let mut plain = cfg.clone();
    plain.anchor_memory = None;
    plain.swa_window = None;
    // The checkpoint carries memory tensors a plain config does not list.
    let err = qwen35::fp8_source::open_dir(&dir, &plain, 128).err().expect("memory tensors without an anchor_memory config");
    assert!(err.contains(".memory."), "{err}");
    std::fs::remove_dir_all(dir).ok();
}
