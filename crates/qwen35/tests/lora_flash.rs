// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The training attention at a long sequence: where the device can run the
//! flash-attention kernels the trainer uses them (memory linear in the
//! sequence, not `heads * T * T`), and the loss and every adapter gradient
//! agree with the materialised chain.

mod common;

use common::*;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const LONG: usize = 256;

#[test]
#[ignore = "needs a device that runs workgroup-cooperative kernels (a GPU); the CPU backend keeps the chain"]
fn flash_attention_trains_the_same_adapters_as_the_materialised_chain() {
    let cfg = Qwen35Config { block_size: LONG as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let init = weights(&cfg, 0.3);
    let (x, y) = batch_of(LONG);
    let build = |flash_min_t: Option<u32>| {
        Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg.clone(), 1, LONG as u32, &init, &TrainOpts { flash_min_t, ..TrainOpts::default() })
    };
    let chain = build(Some(u32::MAX));
    let flash = build(None);
    assert!(!chain.attention_is_flash());
    assert!(flash.attention_is_flash(), "this device should run the flash kernels at {LONG} tokens");
    let want = loss_and_grads_on(&chain, &x, &y);
    assert_same_training_signal(loss_and_grads_on(&flash, &x, &y), want, 1e-4, 0.9999);
}
