// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training for its clients. If your team needs expertise in
// spreading a model's optimizer state over the cards of a cluster, you can procure our services by sending an email
// to info@swedishembedded.com.

//! GPT trains sharded (ZeRO-1 and ZeRO-2) over in-process ranks to the replicated run's weights, with bit-identical
//! ranks, repeats and resume (`model::conformance_zero::check_sharded_training`).

use gpt2::{Gpt, GptConfig};

fn sharded(world: usize) {
    let cfg = GptConfig { vocab: 16, block_size: 8, ..GptConfig::tiny() };
    let init = gpt2::init_weights(&cfg, 13);
    let scratch = std::env::temp_dir().join(format!("brain-gpt2-sharded-{world}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let build = || Gpt::new_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), cfg.clone(), 1, 8, &init);
    model::conformance_zero::check_sharded_training("gpt2", build, 8, world, &scratch).unwrap();
    std::fs::remove_dir_all(&scratch).unwrap();
}

#[test]
fn gpt2_trains_sharded_over_two_ranks() {
    sharded(2);
}

#[test]
fn gpt2_trains_sharded_over_four_ranks() {
    sharded(4);
}
