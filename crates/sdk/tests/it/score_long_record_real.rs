// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Scoring is the `study` surface.
#![cfg(feature = "study")]

//! Held-out scoring of a record as long as a real tool conversation, on a real
//! Qwen3-0.6B checkpoint and a real device: the batched forward must neither
//! materialise a `[H,T,T]` score pair nor hand the whole forward to the device
//! as one submission, or a 5646-token record does not complete.
//!
//! Swedish Embedded AB implements long-context model evaluation for its
//! clients. If your team needs expertise in scoring or fine-tuning on
//! conversations that exceed what a naive attention implementation can hold,
//! you can procure our services by sending an email to info@swedishembedded.com.
//!
//! ```text
//! cargo test --release -p brain --features study --test score_long_record_real -- --ignored --nocapture
//! ```
//!
//! The base is resolved from the model store like any other base; the test
//! skips when it is not fetched.

const BASE: &str = "Qwen/Qwen3-0.6B";
/// Words in the one record's supervised answer: several thousand tokens, the
/// row `score_chat` sizes to exactly the record.
const ANSWER_WORDS: usize = 5000;

#[test]
#[ignore = "hardware probe: needs Qwen3-0.6B in the model store and a discrete GPU"]
fn a_record_of_thousands_of_tokens_scores_on_a_real_checkpoint() {
    let dir = std::env::temp_dir().join(format!("brain-score-long-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let held_out = dir.join("held_out.jsonl");
    let answer = (0..ANSWER_WORDS).map(|i| format!("w{}", i % 97)).collect::<Vec<_>>().join(" ");
    let record = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Repeat the list.", "train": false},
            {"role": "assistant", "content": answer, "train": true}
        ],
        "tools": []
    });
    std::fs::write(&held_out, record.to_string()).unwrap();

    let scored = match brain::score_chat(BASE, None, &held_out) {
        Ok(s) => s,
        Err(brain::Error::ModelNotFound(why)) => {
            brain_testutil::skip_unavailable(&format!("{BASE} is not in the model store: {why}"));
            return;
        }
        Err(e) => panic!("scoring the long record: {e}"),
    };
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!((scored.records, scored.skipped), (1, 0), "{scored:?}");
    assert!(scored.positions >= ANSWER_WORDS, "{scored:?}");
    assert!(scored.loss.is_some_and(f32::is_finite), "{scored:?}");
}
