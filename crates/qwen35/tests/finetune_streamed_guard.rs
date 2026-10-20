// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A streamed-base run refuses a batch too wide for one card before it opens
//! the checkpoint or claims a device, and names the way to the same effective
//! batch. (A real run of 32 rows of 1024 tokens used to run out of device
//! memory minutes into the load.)

use qwen35::base_source::BaseDtype;
use qwen35::finetune::{finetune_streamed, LongSeqOverrides, Mode, StreamedRun};
use std::io::ErrorKind;
use std::path::Path;

fn nf4_recompute() -> StreamedRun {
    StreamedRun { dtype: BaseDtype::Nf4, recompute: true, long: LongSeqOverrides::default() }
}

fn opts(batch: u32, block: u32, accum: u32) -> model::FitOpts {
    model::FitOpts { batch_size: batch, block_size: block, grad_accum: accum, ..Default::default() }
}

#[test]
fn a_wide_batch_of_long_rows_is_refused_with_the_equivalent_accumulation() {
    let mode = Mode::Lora { rank: 16, alpha: 32.0, tuning: Default::default() };
    let err = finetune_streamed(Path::new("no-such-checkpoint"), Path::new("no-such-data"), &opts(32, 1024, 1), &mode, "out.adapter", false, &nf4_recompute()).expect_err("32 x 1024 tokens per micro-step must be refused");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    let msg = err.to_string();
    assert!(msg.contains("--batch 1") && msg.contains("--grad-accum 32"), "{msg}");
}

#[test]
fn a_batch_of_one_long_row_is_not_refused_by_the_guard() {
    let mode = Mode::Lora { rank: 16, alpha: 32.0, tuning: Default::default() };
    let err = finetune_streamed(Path::new("no-such-checkpoint"), Path::new("no-such-data"), &opts(1, 32768, 1), &mode, "out.adapter", false, &nf4_recompute()).expect_err("the checkpoint does not exist");
    // Past the guard: the failure is the missing checkpoint, not the batch.
    assert!(!err.to_string().contains("--grad-accum"), "{err}");
}
