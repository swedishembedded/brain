// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! EdgeRock single-forward parity: `Qwen35::logits_all` at
//! `Qwen35Config::edgerock_tiny()` against the goldens
//! `tools/goldens/edgerock_dump_reference.py` writes (an independent torch
//! implementation of the sliding window and the anchor memory on the
//! `transformers.models.qwen3_5` modules).
//!
//! Two variants: `tiny_text` (random weights, distinct admission scores, the
//! bank evicts) and `tiny_ties` (`admission_up` zeroed: every score is 0, so
//! the tie rule alone decides which records stay). Per stage the table checks
//! the residual after every layer, on the window/memory layers the attention
//! output, every memory intermediate and the logits; the bank itself
//! (`layer{l}.mem.refs`, the token index of every slot of every commit block)
//! must be EXACTLY equal, not merely close, because a selection error is a
//! discrete difference that continuous tolerances can hide.
//!
//! The dumper saves raw HF weights under brain's `blocks.{l}.*` names, so the
//! `(1+w)` RMSNorm fold `crate::import` applies is applied here by hand.

use std::collections::HashMap;
use std::path::Path;

use brain_testutil::parity::Table;
use checkpoint::safetensors::StTensor;
use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn to_map(tensors: Vec<StTensor>) -> HashMap<String, Vec<f32>> {
    tensors.into_iter().map(|t| (t.name, t.data)).collect()
}

struct Golden {
    weights: HashMap<String, Vec<f32>>,
    taps: HashMap<String, Vec<f32>>,
}

fn load(variant: &str) -> Option<Golden> {
    let dir = brain_testutil::testdata(&format!("golden/edgerock/{variant}"));
    let w_path = format!("{dir}/edgerock_tiny_weights.safetensors");
    let t_path = format!("{dir}/edgerock_tiny.safetensors");
    if !Path::new(&w_path).exists() || !Path::new(&t_path).exists() {
        brain_testutil::skip(&format!("fixture {w_path} absent - run tools/goldens/edgerock_dump_reference.py --seed 7"));
        return None;
    }
    let mut weights = to_map(checkpoint::safetensors::read(&w_path).expect("read golden weights"));
    let taps = to_map(checkpoint::safetensors::read(&t_path).expect("read golden taps"));
    qwen35::import::fold_plain_rmsnorm_weights(&mut weights);
    Some(Golden { weights, taps })
}

fn run(gpu: Gpu, variant: &str) {
    let Some(golden) = load(variant) else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    let t = cfg.block_size;

    let tokens: Vec<u32> = golden.taps["tokens"].iter().map(|&v| v.round() as u32).collect();
    assert_eq!(tokens.len(), t as usize);

    let m = Qwen35::new_on(gpu, cfg.clone(), 1, t, &golden.weights);
    m.enable_memory_taps();
    let logits = m.logits_all(&tokens);
    let taps: HashMap<String, Vec<f32>> = m.take_memory_taps().into_iter().collect();

    // fp32 float-op-order noise between this replay and torch is ~1e-6 at
    // every stage; the floor sits well above it, so a real error trips it.
    let mut table = Table::new(0.999999, 1e-4);
    table.check("embed", &m.debug_res(0), &golden.taps["embed"]);
    for l in 0..cfg.n_layers as usize {
        if cfg.layer_has_memory(l) {
            for stage in ["swa.out", "mem.scores", "mem.gate", "mem.query", "mem.key", "mem.value", "mem.read", "mem.out"] {
                let name = format!("layer{l}.{stage}");
                if golden.taps[&name].iter().all(|&x| x == 0.0) {
                    // A cosine against the zero vector is undefined (the ties variant's admission scores).
                    assert!(taps[&name].iter().all(|&x| x == 0.0), "{variant}: {name} must be exactly zero");
                    continue;
                }
                table.check(&name, &taps[&name], &golden.taps[&name]);
            }
            let name = format!("layer{l}.mem.refs");
            assert_eq!(taps[&name], golden.taps[&name], "{variant}: the bank of {name} differs from the reference");
        }
        table.check(&format!("layer{l}.out"), &m.debug_res(l + 1), &golden.taps[&format!("layer{l}.out")]);
    }
    table.check("logits", &logits, &golden.taps["logits"]);
    table.print();
    assert!(table.failures.is_empty(), "{variant}: parity failures: {:#?}", table.failures);
    assert!(taps.contains_key("layer3.mem.refs") && taps.contains_key("layer7.mem.refs"), "both memory layers were tapped");
}

#[test]
fn tiny_text_matches_the_reference_cpu() {
    run(Gpu::new_cpu(pipelines()), "tiny_text");
}

#[test]
fn tiny_text_matches_the_reference_default_backend() {
    println!("default backend: {} / {:?}", gpu_core::backend_name(), gpu_core::adapter_info());
    run(Gpu::new(pipelines()), "tiny_text");
}

#[test]
fn tiny_ties_matches_the_reference_cpu() {
    run(Gpu::new_cpu(pipelines()), "tiny_ties");
}

#[test]
fn tiny_ties_matches_the_reference_default_backend() {
    println!("default backend: {} / {:?}", gpu_core::backend_name(), gpu_core::adapter_info());
    run(Gpu::new(pipelines()), "tiny_ties");
}

/// A decode or chunked-prefill step of a windowed config must refuse, not run
/// plain full attention.
#[test]
#[should_panic(expected = "does not implement this config's sliding window")]
fn the_decode_step_refuses_a_windowed_config() {
    let cfg = Qwen35Config::edgerock_tiny();
    let w = qwen35::init::init_weights(&cfg, 3);
    Qwen35::new_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, cfg.block_size, &w).step(1);
}
