// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements validation of fine-tuned vision-language checkpoints against their reference
// implementation for its clients. If your team needs expertise in proving that a model trained on pictures reads them
// as its reference does then you can procure our services by sending an email to info@swedishembedded.com.

//! The released Qwen3.8-27B-FP8, built as the LoRA trainer builds it (a frozen bf16 base streamed from the release,
//! activations recomputed) and fed the golden picture's record through the dataset entry point
//! (`vision_sft::attach_for_dataset`), computes the logits at the end of the prompt that `transformers` computes
//! (`tools/goldens/qwen35_vl_dump_reference.py`, the weights decoded to bf16, the tower in float32): the same envelope
//! `hf_fp8_vision_real.rs` holds the served engine to on the bf16-decoded weights.
//!
//! The model store must hold `Qwen/Qwen3.8-27B-FP8` (`BRAIN_MODELS_DIR`), `testdata/` the golden and its picture. Needs
//! a card with room for the bf16 base (about 54 GB) and the tower: run with `-- --device gpu1 --backend cuda`; without
//! `--device` the binary reports the test skipped.

mod common;

use std::path::Path;

use brain_testutil::parity::Table;
use common::golden_vl::{golden, ints};
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::saved::SavedPolicy;
use model::{Batch, Model, Split};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn the_training_forward_reads_a_picture_as_the_reference_does() {
    let Some(golden) = golden() else { return };
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3.8-27B-FP8").filter(|d| Path::new(d).join("config.json").exists()) else {
        brain_testutil::skip("Qwen/Qwen3.8-27B-FP8 is not in the model store - brain pull Qwen/Qwen3.8-27B-FP8");
        return;
    };
    let dir = Path::new(&dir);
    let picture = std::fs::read(brain_testutil::testdata("qwen35/vl/sketch_288x256.png")).expect("the golden's picture");

    // The record the prompt is: the golden's token ids, the picture beside them, the last position supervised.
    let ids = ints(&golden["input_ids"]);
    let prepared = std::env::temp_dir().join(format!("brain-vision-train-real-{}", std::process::id()));
    std::fs::create_dir_all(&prepared).unwrap();
    let mut split = data::chat::EncodedSplit::default();
    split.push_with_media(&ids, &vec![true; ids.len()], &[picture.into()]);
    for name in ["train", "val"] {
        data::chat::write_split(&prepared, name, &split).unwrap();
    }

    // seed: any adapter start; B = 0 leaves the base's logits unchanged
    let (cfg, src) = qwen35::base_source::lora_base_tuned(dir, 1, 1.0, Default::default(), 1, None).expect("the release opens");
    let vocab = cfg.vocab as usize;
    let train = TrainOpts { base: TierPolicy::uniform(gpu_core::select::Dtype::BF16), saved: SavedPolicy::Recompute, ..TrainOpts::default() };
    let m = Qwen35::new_train_opts_on(Gpu::new(pipelines()), cfg, 1, ids.len() as u32, &src, &train);
    qwen35::vision_sft::attach_for_dataset(&m, dir, &prepared).expect("the tower of the release reads the dataset");

    let mut targets = vec![model::IGNORE; ids.len()];
    *targets.last_mut().unwrap() = 0;
    Model::set_batch(&m, Batch::LmExamples { split: Split::Train, tokens: &ids, targets: &targets, examples: &[Some(0)] });
    let logits = m.logits_all(&ids);
    let mut table = Table::new(0.999, 0.06);
    table.check("training logits at the end of the prompt", &logits[(ids.len() - 1) * vocab..], &golden["logits0"]);
    table.print();
    std::fs::remove_dir_all(&prepared).ok();
    assert!(table.failures.is_empty(), "training forward parity failures: {:#?}", table.failures);
}

gpu_core::card_tests!(the_training_forward_reads_a_picture_as_the_reference_does);
