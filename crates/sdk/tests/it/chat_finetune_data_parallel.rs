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

/// Runs `body` for every rank of a fresh in-process group of `world` on its own thread and returns what each returned,
/// in rank order. A group that has not finished within `limit` is aborted (so no thread is left waiting) and the test
/// fails: the ranks did not agree.
fn ranks_within<T: Send + 'static>(world: usize, limit: std::time::Duration, body: impl Fn(DataParallel) -> T + Send + Sync + 'static) -> Vec<T> {
    let ranks = DataParallel::in_process(world).expect("a group");
    let breaker = ranks[0].clone();
    let body = std::sync::Arc::new(body);
    let (tx, rx) = std::sync::mpsc::channel();
    for rank in ranks {
        let (tx, body) = (tx.clone(), std::sync::Arc::clone(&body));
        std::thread::spawn(move || {
            let r = rank.rank();
            let _ = tx.send((r, body(rank)));
        });
    }
    drop(tx);
    let deadline = std::time::Instant::now() + limit;
    let mut out: Vec<Option<T>> = (0..world).map(|_| None).collect();
    for _ in 0..world {
        match rx.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now())) {
            Ok((r, value)) => out[r] = Some(value),
            Err(_) => {
                breaker.abort("the test's time limit");
                panic!("the ranks did not finish within {limit:?}: they disagreed on where to stop");
            }
        }
    }
    out.into_iter().map(|v| v.expect("every rank answered")).collect()
}

#[test]
fn a_cancellation_one_rank_sees_first_stops_every_rank_after_the_same_step() {
    let model = chat_model_dir("data-parallel-cancel");
    let train = write_chat_dataset(&model, &TRAIN);
    let cancel = brain::CancelToken::armed();
    let shared = (model.to_path_buf(), train.clone(), cancel.clone());
    let outcomes = ranks_within(2, std::time::Duration::from_secs(120), move |rank| {
        let (model, train, cancel) = &shared;
        let r = rank.rank();
        let out = model.join(format!("cancel-rank{r}"));
        let fine_tune = tune(model, train, &out).steps(200).data_parallel(rank);
        // Rank 0 cancels late in its step 2, after rank 1 has looked and gone on into step 3's reduction.
        fine_tune
            .run_with(cancel, |p| {
                if r == 0 && p.step == 2 {
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    cancel.cancel();
                }
            })
            .expect("a cancelled rank answers")
    });
    for o in &outcomes {
        assert_eq!(o.status, FineTuneStatus::Cancelled);
    }
    assert_eq!(outcomes[0].steps_completed, outcomes[1].steps_completed, "both ranks stopped after the same step");
    assert!(outcomes[0].steps_completed <= 3, "{}", outcomes[0].steps_completed);
}

#[test]
fn a_rank_that_fails_breaks_the_group_and_the_others_fail_instead_of_waiting() {
    let model = chat_model_dir("data-parallel-abort");
    let train = write_chat_dataset(&model, &TRAIN);
    let shared = (model.to_path_buf(), train.clone());
    let started = std::time::Instant::now();
    let results = ranks_within(2, std::time::Duration::from_secs(120), move |rank| {
        let (model, train) = &shared;
        if rank.rank() == 1 {
            rank.abort("its dataset could not be read");
            return Err("rank 1 never trained".to_string());
        }
        tune(model, train, &model.join("abort-rank0")).data_parallel(rank).run().map(drop).map_err(|e| e.to_string())
    });
    let why = results[0].as_ref().expect_err("rank 0 cannot train without its peer");
    assert!(why.contains("rank 1 failed: its dataset could not be read"), "{why}");
    assert!(started.elapsed() < std::time::Duration::from_secs(60));
}
