// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rung-3 single-forward parity: `Qwen35::logits_all` at `Qwen35Config::tiny()`
//! against the real reference goldens dumped by
//! `tools/goldens/qwen35_dump_reference.py` (`transformers.models.qwen3_5`,
//! not a secondhand description). Weights come from the golden's own saved
//! `qwen35_tiny_text_weights.safetensors` (already renamed to brain's
//! `blocks.{l}.*` convention, no import step needed - see that dumper's
//! module doc), so this test needs no checkpoint and no import machinery.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::safetensors::StTensor;
use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{Qwen35, PIPELINES};
use brain_testutil::parity::Table;

fn to_map(tensors: Vec<StTensor>) -> HashMap<String, Vec<f32>> {
    tensors.into_iter().map(|t| (t.name, t.data)).collect()
}

struct Golden {
    weights: HashMap<String, Vec<f32>>,
    taps: HashMap<String, Vec<f32>>,
}

fn load() -> Option<Golden> {
    let dir = brain_testutil::testdata("golden/qwen35/tiny_text");
    let w_path = format!("{dir}/qwen35_tiny_text_weights.safetensors");
    let t_path = format!("{dir}/qwen35_tiny_text.safetensors");
    if !Path::new(&w_path).exists() || !Path::new(&t_path).exists() {
        brain_testutil::skip(&format!("fixture {w_path} absent - run tools/goldens/qwen35_dump_reference.py"));
        return None;
    }
    let mut weights = to_map(checkpoint::safetensors::read(&w_path).expect("read golden weights"));
    let taps = to_map(checkpoint::safetensors::read(&t_path).expect("read golden taps"));
    // The dumper saves RAW HF weights (renamed, but not folded) - apply the
    // same (1+w) fold `crate::import`'s real checkpoint path applies, or
    // every plain RMSNorm computes with a weight off by exactly 1.0 (see
    // qwen35::import's module doc, "The (1+w) RMSNorm fold").
    qwen35::import::fold_plain_rmsnorm_weights(&mut weights);
    Some(Golden { weights, taps })
}

fn run(gpu: Gpu) {
    let Some(golden) = load() else { return };
    let cfg = Qwen35Config::tiny();
    let b = 1;
    let t = cfg.block_size;

    let tokens: Vec<u32> = golden.taps["tokens"].iter().map(|&v| v.round() as u32).collect();
    assert_eq!(tokens.len(), t as usize);

    let m = Qwen35::new_on(gpu, cfg.clone(), b, t, &golden.weights);
    let logits = m.logits_all(&tokens);

    // Achieved in practice: cosine 1.0000000000, rel_l2/max_abs ~1e-7 at
    // every stage (fp32 float-op-order noise between this host replay and
    // the reference's own torch ops) - the floor stays well above that so a
    // real regression (not just fp32 noise) trips it.
    let mut table = Table::new(0.999999, 1e-4);
    table.check("embed", &m.debug_res(0), &golden.taps["embed"]);
    for l in 0..cfg.n_layers as usize {
        table.check(&format!("layer{l}.out"), &m.debug_res(l + 1), &golden.taps[&format!("layer{l}.out")]);
    }
    table.check("logits", &logits, &golden.taps["logits"]);
    table.print();
    assert!(table.failures.is_empty(), "parity failures: {:#?}", table.failures);
}

#[test]
fn tiny_text_logits_match_the_reference_cpu() {
    run(Gpu::new_cpu(PIPELINES));
}

#[test]
fn tiny_text_logits_match_the_reference_default_backend() {
    run(Gpu::new(PIPELINES));
}
