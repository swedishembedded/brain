// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language agents on a client's own verified trajectories. If
// your team needs expertise in training a model on the images it will be served, then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The entry point a 27B LoRA fine-tune is driven through (`finetune_streamed`: a frozen bf16 base streamed from its
//! checkpoint, activations recomputed in the backward) trains on a prepared dataset whose examples hold images: the
//! vision tower is read from the checkpoint's release files, its rows are spliced where serving splices them, and the
//! picture decides the loss. A dataset with images over a base that has no vision tower is refused, not trained as text.

mod common;

use common::vision_fixture::{add_vision_release, picture, ppm, TOKENS};
use gpu_core::select::Dtype;
use qwen35::config::Qwen35Config;
use qwen35::finetune::{finetune_streamed, LongSeqOverrides, Mode, StreamedRun};
use qwen35::model::{pipelines, Qwen35};

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-vision-streamed-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One example, "three tokens, a picture of four rows, three tokens", the text after the picture supervised, in both splits.
fn write_dataset(dir: &std::path::Path, vocab: u32, picture_seed: u64) {
    let mut ids = vec![1u32, 2, 3, TOKENS.start];
    ids.extend([TOKENS.image; 4]);
    ids.extend([TOKENS.end, 4, 5, 6]);
    let mask: Vec<bool> = (0..ids.len()).map(|i| i >= 9).collect();
    let mut split = data::chat::EncodedSplit::default();
    split.push_with_media(&ids, &mask, &[ppm(&picture(8, 8, picture_seed)).into()]);
    for name in ["train", "val"] {
        data::chat::write_split(dir, name, &split).unwrap();
    }
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(vocab as usize)).unwrap();
}

fn run(base: &std::path::Path, data: &std::path::Path, out: &std::path::Path) -> std::io::Result<(f32, f32)> {
    run_as(base, data, out, Dtype::BF16)
}

fn run_as(base: &std::path::Path, data: &std::path::Path, out: &std::path::Path, dtype: Dtype) -> std::io::Result<(f32, f32)> {
    // seed: the runs compared start from the same adapter
    let opts = model::FitOpts { steps: 10, batch_size: 1, block_size: 12, lr: 5e-2, min_lr: 5e-2, warmup: 0, checkpoint_secs: 0, eval_interval: 0, eval_batches: 0, seed: 3, ..Default::default() };
    let mode = Mode::Lora { rank: 2, alpha: 4.0, tuning: Default::default() };
    finetune_streamed(base, data, &opts, &mode, out.to_str().unwrap(), false, &StreamedRun::uniform(dtype, dtype != Dtype::F32, LongSeqOverrides::default()))
}

#[test]
fn the_streamed_lora_entry_point_trains_on_images_and_the_picture_decides_the_loss() {
    let dir = scratch("train");
    let cfg = Qwen35Config::tiny();
    // seed: any base weights; the test compares runs over the same base
    let init = qwen35::init::init_weights(&cfg, 7);
    let base = dir.join("base.safetensors");
    Qwen35::new_on(gpu_core::Gpu::new(pipelines()), cfg.clone(), 1, 12, &init).save(base.to_str().unwrap());
    add_vision_release(&dir, 3);

    let (data_a, data_b) = (dir.join("data-a"), dir.join("data-b"));
    for (data, seed) in [(&data_a, 5), (&data_b, 6)] {
        std::fs::create_dir_all(data).unwrap();
        write_dataset(data, cfg.vocab, seed);
    }
    // Whatever the tier the base is held at, and whether the activations are kept or recomputed, the picture reaches
    // the loss measured before the first step.
    for dtype in [Dtype::F32, Dtype::BF16] {
        let a = run_as(&base, &data_a, &dir.join("x.adapter.safetensors"), dtype).unwrap().0;
        let b = run_as(&base, &data_b, &dir.join("y.adapter.safetensors"), dtype).unwrap().0;
        assert_ne!(a, b, "{dtype:?}: the picture did not reach the decoder");
    }
    let (start_a, end_a) = run(&base, &data_a, &dir.join("a.adapter.safetensors")).expect("training on a record with an image");
    eprintln!("loss with picture A {start_a} -> {end_a}");
    assert!(start_a.is_finite() && end_a < 0.9 * start_a, "ten steps should lower the loss: {start_a} -> {end_a}");
    assert!(dir.join("a.adapter.safetensors").exists());
}

#[test]
fn a_dataset_with_images_over_a_base_with_no_vision_tower_is_refused_not_trained_as_text() {
    let dir = scratch("refuse");
    let cfg = Qwen35Config::tiny();
    let init = qwen35::init::init_weights(&cfg, 7);
    let base = dir.join("base.safetensors");
    Qwen35::new_on(gpu_core::Gpu::new(pipelines()), cfg.clone(), 1, 12, &init).save(base.to_str().unwrap());
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_dataset(&data, cfg.vocab, 5);
    let err = run(&base, &data, &dir.join("adapter.safetensors")).unwrap_err().to_string();
    assert!(err.contains("holds images") && err.contains("no vision tower") && err.contains("base.safetensors"), "{err}");
    assert!(!dir.join("adapter.safetensors").exists(), "nothing is written for a refused run");
}
