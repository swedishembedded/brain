// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The tensor-core GEMM tier agrees with the fp32 reference tier on the same
//! tiny model: the LoRA products run on the library with TF32 operands and fp32
//! accumulation, and a bf16 or NF4 base additionally rounds its activation operand
//! to bf16. On a device without the library the
//! automatic tier is the reference tier, so the comparison is trivially exact
//! and the reported path says why.

mod common;

use common::*;
use gpu_core::provider::library::{GemmPath, GemmPrecision};
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn build(gpu: &Gpu, init: &std::collections::HashMap<String, Vec<f32>>, base: Dtype, precision: GemmPrecision) -> Qwen35 {
    let cfg = lora_config();
    let opts = TrainOpts { base: TierPolicy::uniform(base), recompute: true, gemm_precision: precision, ..TrainOpts::default() };
    Qwen35::new_train_opts_on(gpu.share(), cfg, 1, T as u32, init, &opts)
}

#[test]
fn the_tensor_core_tier_trains_like_the_fp32_reference_and_reports_its_path() {
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        eprintln!("lora_gemm_precision: no CUDA device on this box - nothing to run");
        return;
    };
    // One set of weights for both tiers: the comparison is about arithmetic only.
    let init = weights(&lora_config(), 0.3);
    for (base, loss_tol, min_cos) in [(Dtype::F32, 1e-3, 0.999), (Dtype::BF16, 2e-2, 0.999)] {
        let reference = build(&gpu, &init, base, GemmPrecision::Fp32Reference);
        assert!(matches!(reference.gemm_path(), GemmPath::Fp32Kernels { .. }), "the pinned tier must run the fp32 kernels");
        let auto = build(&gpu, &init, base, GemmPrecision::Auto);
        match auto.gemm_path() {
            GemmPath::Bf16TensorCore => eprintln!("lora_gemm_precision: {base:?} base on the tensor-core tier"),
            GemmPath::Fp32Kernels { reason } => eprintln!("lora_gemm_precision: {base:?} base: the device runs the fp32 kernels ({reason})"),
        }
        assert_same_training_signal(loss_and_grads(&auto), loss_and_grads(&reference), loss_tol, min_cos);
    }
}
