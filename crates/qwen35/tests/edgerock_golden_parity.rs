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
use gpu_core::provider::library::{GemmPath, GemmPrecision};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

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
    let model = Qwen35::new_on(gpu, cfg.clone(), 1, cfg.block_size, &golden.weights);
    // The tier the device assigned decides the arithmetic: tensor-core products where the library takes them
    // (CUDA by default), the fp32 kernels everywhere else.
    let tol = match model.gemm_path() {
        GemmPath::Bf16TensorCore => Tolerance::tf32(),
        GemmPath::Fp32Kernels { .. } => Tolerance::exact_fp32(),
    };
    check(&model, &golden, variant, tol);
}

/// How closely a tier must reproduce the reference, and how much of the selection it must keep.
struct Tolerance {
    cosine: f64,
    rel_l2: f64,
    /// Fraction of bank slots that must equal the reference: 1.0 demands the exact bank.
    bank: f64,
}

impl Tolerance {
    /// fp32 arithmetic throughout: float-op-order noise only, the bank exactly the reference's.
    fn exact_fp32() -> Tolerance {
        Tolerance { cosine: 0.999999, rel_l2: 1e-4, bank: 1.0 }
    }
    /// TF32 tensor-core products in the attention and MLP (the CUDA default tier): the convention of
    /// `lora_gemm_precision` (cosine 0.999). Near-tied admission scores can then swap rank, so a bank is held to
    /// agreeing in nearly every slot, not all.
    fn tf32() -> Tolerance {
        Tolerance { cosine: 0.999, rel_l2: 2e-2, bank: 0.95 }
    }
}

fn check(m: &Qwen35, golden: &Golden, variant: &str, tol: Tolerance) {
    let cfg = Qwen35Config::edgerock_tiny();
    let t = cfg.block_size;

    let tokens: Vec<u32> = golden.taps["tokens"].iter().map(|&v| v.round() as u32).collect();
    assert_eq!(tokens.len(), t as usize);

    m.enable_memory_taps();
    let logits = m.logits_all(&tokens);
    let taps: HashMap<String, Vec<f32>> = m.take_memory_taps().into_iter().collect();

    // fp32 float-op-order noise between this replay and torch is ~1e-6 at
    // every stage; the floor sits well above it, so a real error trips it.
    let mut table = Table::new(tol.cosine, tol.rel_l2);
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
            let equal = taps[&name].iter().zip(&golden.taps[&name]).filter(|(a, b)| a == b).count() as f64 / golden.taps[&name].len() as f64;
            assert!(equal >= tol.bank, "{variant}: only {:.1}% of the slots of the bank {name} equal the reference (need {:.1}%)", equal * 100.0, tol.bank * 100.0);
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

/// A CUDA device, or a named skip: this machine cannot run the test, which is not a missing fixture.
fn cuda() -> Option<Gpu> {
    match Gpu::try_new_cuda(pipelines()) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}"));
            None
        }
    }
}

/// CUDA with every GEMM pinned to the fp32 kernels (`GemmPrecision::Fp32Reference`, the tier the cross-backend
/// parity gates are stated against): the stages agree to float-op-order noise and the banks are EXACTLY the
/// reference's. Built as a training model, the one build that takes a precision.
fn cuda_reference(variant: &str) {
    let Some(gpu) = cuda() else { return };
    let Some(golden) = load(variant) else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    let opts = TrainOpts { gemm_precision: GemmPrecision::Fp32Reference, ..TrainOpts::default() };
    let m = Qwen35::new_train_opts_on(gpu, cfg.clone(), 1, cfg.block_size, &golden.weights, &opts);
    check(&m, &golden, variant, Tolerance::exact_fp32());
}

/// CUDA at its default GEMM precision (TF32 tensor-core products where the library takes them): the tolerance
/// the plain qwen35 tiers state, and the banks agreeing wherever no two admission scores are within TF32 noise.
fn cuda_default(variant: &str) {
    let Some(gpu) = cuda() else { return };
    let Some(golden) = load(variant) else { return };
    let cfg = Qwen35Config::edgerock_tiny();
    let opts = TrainOpts { gemm_precision: GemmPrecision::Auto, ..TrainOpts::default() };
    let m = Qwen35::new_train_opts_on(gpu, cfg.clone(), 1, cfg.block_size, &golden.weights, &opts);
    println!("CUDA default tier GEMM path: {:?}", m.gemm_path());
    check(&m, &golden, variant, Tolerance::tf32());
}

#[test]
fn tiny_text_matches_the_reference_cuda_fp32() {
    cuda_reference("tiny_text");
}

#[test]
fn tiny_ties_matches_the_reference_cuda_fp32() {
    cuda_reference("tiny_ties");
}

#[test]
fn tiny_text_matches_the_reference_cuda_default_precision() {
    cuda_default("tiny_text");
}

#[test]
fn tiny_ties_matches_the_reference_cuda_default_precision() {
    cuda_default("tiny_ties");
}
