// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of fine-tuned language models straight
// from their training checkpoints for its clients. If your team needs expertise
// in loading large trained models onto a card within a fixed host memory budget
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! `generate` on the checkpoint a parameter-sharded full fine-tune exports (bf16 layer matrices and tables, f32
//! norms, the family's card): the load streams the file onto the device one tensor at a time, so its host peak is
//! bounded by the file, not by a whole-model f32 copy (which, cloned once more, is what a 27B export used to need
//! more than 200 GB of host memory for); the default precision keeps the weights as stored; and the tokens are the
//! ones the eagerly loaded model decodes.

use std::collections::HashMap;
use std::sync::Arc;

use capability::{Invocation, Provider, Registry};
use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use qwen35::caps::{Qwen35Provider, MODEL};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use serde_json::json;

use crate::common::exported::write_exported;

const PROMPT: [u32; 5] = [3, 17, 250, 9, 1021];
const MAX_NEW: usize = 6;

/// A model whose bf16 checkpoint is tens of MB, large enough for a whole-model f32 copy to stand out of the
/// process's own working set, small enough to build in seconds on the CPU backend; its bytes are in the layers, as a
/// real model's are, so the f32 head copy decoding keeps on the host is small beside them. Int8-legal (every
/// contraction a multiple of 32).
fn sized() -> Qwen35Config {
    Qwen35Config { vocab: 4096, n_layers: 16, d_model: 512, intermediate_size: 3072, ..Qwen35Config::tiny_i8() }
}

/// A random initialisation rounded to bf16, so the exported file holds exactly these values.
fn weights(cfg: &Qwen35Config) -> HashMap<String, Vec<f32>> {
    let mut init = qwen35::init::init_weights(cfg, data::rng::random_seed());
    for values in init.values_mut() {
        values.iter_mut().for_each(|v| *v = crate::common::round_bf16(*v));
    }
    init
}

fn generate(path: &std::path::Path, precision: &str) -> Vec<u64> {
    let mut reg = Registry::new();
    reg.register(Arc::new(Qwen35Provider::new()) as Arc<dyn Provider>);
    let prompt = PROMPT.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
    let inv = Invocation::new().set("weights", json!(path.to_str().unwrap())).set("prompt", json!(prompt)).set("max_new", json!(MAX_NEW)).set("temp", json!(0.0)).set("precision", json!(precision));
    let out = reg.run(MODEL, "generate", inv, &mut |_| {}).unwrap_or_else(|e| panic!("generate at {precision}: {e}"));
    out.outputs["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()
}

/// The eager reference: the model built from the whole map in host memory, decoded greedily.
fn eager(cfg: &Qwen35Config, init: &HashMap<String, Vec<f32>>, tier: Dtype) -> Vec<u64> {
    let cap = (PROMPT.len() + MAX_NEW).max(64) as u32;
    let m = Qwen35::new_on_dt(gpu_core::Gpu::new(pipelines()), cfg.clone(), 1, cap, init, &TierPolicy::uniform(tier));
    let mut rng = data::rng::Rng::new(0); // greedy: the seed draws nothing
    qwen35::sample::generate_kv(&m, &PROMPT, MAX_NEW, 0.0, 0, 1.0, &[], &mut rng).into_iter().map(u64::from).collect()
}

/// The streamed load decodes the tokens the eagerly loaded model decodes, at fp32 and at the stored bf16 (the
/// default), and the default holds the stored bf16 rather than widening to fp32.
#[test]
fn an_exported_checkpoint_generates_what_the_eager_model_generates() {
    let dir = brain_testutil::Scratch::dir("qwen35-exported-tokens");
    let cfg = sized();
    let init = weights(&cfg);
    let path = dir.path().join("model.safetensors");
    write_exported(&path, &cfg, &init);
    let base = qwen35::base_source::open_base(&path).unwrap();
    for name in ["blocks.0.mlp.gate.weight", "blocks.3.self_attn.q_proj.weight", "tok.weight", "lm_head.weight"] {
        assert_eq!(model::ops::stored_tier(&*base.src, name), Some(Dtype::BF16), "{name} is read as the bf16 it is stored in");
    }
    assert_eq!(generate(&path, "fp32"), eager(&cfg, &init, Dtype::F32), "fp32");
    assert_eq!(generate(&path, "stored"), eager(&cfg, &init, Dtype::BF16), "stored bf16");
}

/// Every parameter of a config, drawn chunk by chunk from a generator and never held whole: the peak test's
/// checkpoint is written without the process ever holding a model's worth of values, so the allocator holds no
/// freed model-sized heap that a later load could reuse unseen.
struct Synthetic {
    numel: HashMap<String, usize>,
    seed: u64,
}

impl checkpoint::TensorSource for Synthetic {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let mut whole = Vec::new();
        let found = self.with_tensor_chunks(name, usize::MAX, &mut |_, part| whole.extend_from_slice(part));
        if found {
            f(&whole);
        }
        found
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        let Some(&n) = self.numel.get(name) else { return false };
        let mut rng = data::rng::Lcg::new(self.seed ^ name.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b))));
        let mut chunk = Vec::new();
        for off in (0..n).step_by(max_elems.min(n).max(1)) {
            chunk.clear();
            chunk.extend((off..n.min(off + max_elems)).map(|_| rng.scaled(0.02)));
            f(off as u64, &chunk);
        }
        true
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.numel.get(name).copied()
    }
}

/// Host peak of loading and decoding an exported checkpoint at the default precision, measured alone in a child
/// process (`VmHWM` is process-wide): the growth over the process before the call stays below one and a half times the file
/// (the file's worth being the weights themselves on a CPU device).
/// Before the load streamed, a whole-model f32 copy cloned once more came on top: about four times the file.
#[test]
fn loading_an_exported_checkpoint_holds_no_whole_model_host_copy() {
    brain_testutil::run_alone(module_path!(), "host_peak_of_an_exported_load");
}

#[test]
#[ignore = "run alone by loading_an_exported_checkpoint_holds_no_whole_model_host_copy"]
fn host_peak_of_an_exported_load() {
    let dir = brain_testutil::Scratch::dir("qwen35-exported-peak");
    let cfg = sized();
    let path = dir.path().join("model.safetensors");
    let synthetic = |cfg: &Qwen35Config| Synthetic { numel: cfg.param_list().into_iter().collect(), seed: data::rng::random_seed() };
    let file = write_exported(&path, &cfg, &synthetic(&cfg));
    // The device, its driver libraries and its kernels are made first, by a load of a model of the same layout a
    // few KB in size: what is measured is the load of the checkpoint, not the process's first device.
    let warm = dir.path().join("warm.safetensors");
    let small = Qwen35Config { vocab: 2048, ..Qwen35Config::tiny_i8() };
    write_exported(&warm, &small, &synthetic(&small));
    generate(&warm, "stored");
    brain_testutil::reset_peak_rss();
    let before = brain_testutil::rss_bytes();
    generate(&path, "stored");
    let growth = brain_testutil::peak_rss_bytes().saturating_sub(before);
    println!("file {:.1} MB, host peak growth {:.1} MB", file as f64 / 1e6, growth as f64 / 1e6);
    // On a CPU device the weights themselves are host memory: at the stored tier, the file's size. A whole-model f32
    // copy adds twice the file on top; the streamed load adds one tile of staging at a time.
    assert!(growth < file + file / 2, "host peak grew {growth} bytes loading a {file}-byte checkpoint");
}
