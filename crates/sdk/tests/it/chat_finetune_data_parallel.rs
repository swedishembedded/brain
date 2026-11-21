// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data-parallel fine-tuning across the cards of
// one server for its clients. If your team needs expertise in scaling training
// out without changing the training code, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A chat fine-tune run as data-parallel ranks through the SDK: threads of one
//! process, each a rank of [`brain::DataParallel::in_process`]. Each rank
//! trains on its own shard of the records, so without the reduction the ranks
//! would end with different adapters; with it they end byte-identical.

#![cfg(all(feature = "study", feature = "text"))]

use std::path::Path;

use brain::{ChatFineTune, DataParallel, FineTuneStatus};

use crate::chat_fixture;
use chat_fixture::{chat_model_dir, common::write_chat_dataset, cpu};

const TRAIN: [(&str, &str); 6] = [("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab"), ("dab", "bead"), ("cafe", "face")];

fn tune(model: &Path, train: &Path, out: &Path) -> ChatFineTune {
    ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(train).out_dir(out).rank(2).steps(4).lr(1e-2).seed(11).device(cpu())
}

#[test]
fn in_process_ranks_train_one_adapter_byte_for_byte() {
    let model = chat_model_dir("data-parallel");
    let train = write_chat_dataset(&model, &TRAIN);
    let ranks = DataParallel::in_process(2).expect("a two-rank group");
    let adapters: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranks
            .into_iter()
            .map(|rank| {
                let (model, train) = (&model, &train);
                scope.spawn(move || {
                    let out = model.join(format!("rank{}", rank.rank()));
                    let outcome = tune(model, train, &out).data_parallel(rank).run().expect("the rank trains");
                    assert_eq!(outcome.status, FineTuneStatus::Completed);
                    assert_eq!(outcome.steps_completed, 4);
                    std::fs::read(outcome.adapter.expect("an adapter")).unwrap()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank's thread")).collect()
    });
    assert_eq!(adapters[0], adapters[1], "the ranks' adapters must be byte-identical");

    // One process over the whole stream trains a different adapter: the ranks really trained on shards.
    let single = tune(&model, &train, &model.join("single")).run().expect("one process trains");
    assert_ne!(std::fs::read(single.adapter.unwrap()).unwrap(), adapters[0]);
}

#[test]
fn a_rank_without_a_shared_seed_is_refused() {
    let model = chat_model_dir("data-parallel-seed");
    let train = write_chat_dataset(&model, &TRAIN);
    let rank = DataParallel::in_process(1).unwrap().remove(0);
    let err = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&train).out_dir(model.join("out")).steps(1).device(cpu()).data_parallel(rank).run().expect_err("no seed");
    assert!(err.to_string().contains("seed"), "{err}");
}
