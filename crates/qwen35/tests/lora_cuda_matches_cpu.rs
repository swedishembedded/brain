// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The CUDA trainer on the fp32 reference tier computes what the CPU trainer
//! computes: the cooperative RMSNorm kernels (forward and backward), the native
//! copy and reduction kernels and the native Gated DeltaNet kernels all reorder
//! or fuse work the CPU runs serially, so the loss and every adapter gradient
//! are held to the fp32 summation bound against the CPU backend, which is the
//! backend the gradient checks are stated on.

mod common;

use common::*;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::saved::SavedPolicy;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

#[test]
fn the_cuda_fp32_tier_trains_like_the_cpu_backend() {
    let Ok(cuda) = Gpu::try_new_cuda(pipelines()) else {
        eprintln!("lora_cuda_matches_cpu: no CUDA device on this box - nothing to run");
        return;
    };
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    for recompute in [false, true] {
        let opts = TrainOpts { base: TierPolicy::uniform(Dtype::F32), saved: SavedPolicy::recompute_if(recompute), gemm_precision: GemmPrecision::Fp32Reference, ..TrainOpts::default() };
        let on_cpu = Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, T as u32, &init, &opts);
        let on_cuda = Qwen35::new_train_opts_on(cuda.share(), cfg.clone(), 1, T as u32, &init, &opts);
        assert_same_training_signal(loss_and_grads(&on_cuda), loss_and_grads(&on_cpu), 1e-4, 0.99999);
    }
}
