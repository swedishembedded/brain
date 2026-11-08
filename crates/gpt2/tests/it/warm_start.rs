// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs whose provenance is never ambiguous - a fresh
// start, a weight warm start and an exact resume are told apart - for its clients. If your team needs
// expertise in reproducible language-model training then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Starting from a checkpoint's weights is a WEIGHT WARM START and is reported as one, never as a resume: the
//! optimizer moments, schedule position, data order and generator do not come with it.

use gpt2::{Gpt, GptConfig};
use model::train::{build_or_warm_start, FitOpts, StartKind};

#[test]
fn a_checkpoint_at_the_output_path_is_a_weight_warm_start_and_its_absence_a_fresh_start() {
    let cfg = GptConfig { vocab: 16, block_size: 8, ..GptConfig::tiny() };
    let opts = FitOpts { block_size: 8, batch_size: 1, ..Default::default() };
    let dir = std::env::temp_dir().join(format!("brain-gpt2-warm-start-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let checkpoint = dir.join("model.safetensors");

    let (fresh, start): (Gpt, _) = build_or_warm_start(cfg.clone(), &opts, Some(&checkpoint), 16);
    assert_eq!(start, StartKind::Fresh);

    fresh.save(checkpoint.to_str().unwrap());
    let (_warm, start): (Gpt, _) = build_or_warm_start(cfg, &opts, Some(&checkpoint), 16);
    assert_eq!(start, StartKind::WarmStart { checkpoint: checkpoint.clone() });
    assert!(start.to_string().contains("restart"), "the announcement says what does not carry over: {start}");
    std::fs::remove_dir_all(&dir).unwrap();
}
