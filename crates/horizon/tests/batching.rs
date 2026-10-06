// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving stacks that answer many concurrent
// prediction requests from one resident model. If your team needs expertise
// in batched inference for survival models you can procure our services by
// sending an email to info@swedishembedded.com.

//! Batched serving is the same answer, faster: several `predict` requests
//! through one forward pass equal each request alone, a failing request does
//! not fail the others, and a subject's prediction depends on nothing but the
//! subject - not its batch neighbours, not the padding of a part-full batch.

use capability::{Blob, Invocation, Media};
use horizon::caps::{predict, predict_batch};
use horizon::saved::Saved;
use horizon::synthetic::{population, CODES};
use horizon::timeline::Subject;
use horizon::train::predict_log_hazards;
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use serde_json::{json, Value};

/// Subjects per device batch of the test model: small, so requests span
/// several batches and the last one is part-full.
const BATCH: u32 = 8;

fn saved(subjects: &[Subject]) -> Saved {
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(subjects, &codes, &codes[..2], &FitOptions::default()).unwrap();
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 8;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.d_ff = 32;
    cfg.rank = 8;
    cfg.knots = vec![0.0, 2.0, 5.0, 10.0];
    let model = Horizon::new(cfg.clone(), BATCH, &horizon::init_weights(&cfg, 3));
    Saved::new(model, vocab)
}

fn request(subjects: &[Subject], times: &str) -> Invocation {
    let jsonl: String = subjects
        .iter()
        .map(|s| serde_json::to_string(s).unwrap() + "\n")
        .collect();
    Invocation::new()
        .set("times", json!(times))
        .blob("subjects", Blob::new(Media::Text, jsonl.into_bytes()))
}

fn lines(out: &capability::Outcome) -> Vec<Value> {
    let text = std::str::from_utf8(&out.blobs["predictions"].bytes).unwrap();
    text.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn assert_same(a: &Value, b: &Value, what: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!((x - y).abs() < 1e-6, "{what}: {x} vs {y}");
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{what}");
            x.iter().zip(y).for_each(|(p, q)| assert_same(p, q, what));
        }
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(x.len(), y.len(), "{what}");
            for (k, v) in x {
                assert_same(v, &y[k], &format!("{what}.{k}"));
            }
        }
        _ => assert_eq!(a, b, "{what}"),
    }
}

#[test]
fn batched_requests_equal_each_request_alone_and_fail_alone() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (subjects, _) = population(60, 4);
    let saved = saved(&subjects);
    // Mixed sizes (one more than a device batch, an empty file, a single
    // subject), different times per request, a bad time and a bad file.
    let invs = vec![
        request(&subjects[..11], "1,4"),
        request(&[], "1"),
        request(&subjects[11..12], "9"),
        request(&subjects[12..13], "11"), // past the last knot
        Invocation::new().blob("subjects", Blob::new(Media::Text, b"{not json}\n".to_vec())),
        Invocation::new().set("times", json!("1")), // no subjects blob
        request(&subjects[13..30], "2.5,5,10"),
    ];
    let batched = predict_batch(&saved, &invs);
    assert_eq!(batched.len(), invs.len());
    for (i, (inv, got)) in invs.iter().zip(&batched).enumerate() {
        let alone = predict(&saved, inv);
        match (got, &alone) {
            (Ok(g), Ok(a)) => {
                assert_eq!(g.outputs["subjects"], a.outputs["subjects"], "request {i}");
                let (g, a) = (lines(g), lines(a));
                assert_eq!(g.len(), a.len(), "request {i}");
                g.iter().zip(&a).for_each(|(x, y)| assert_same(x, y, &format!("request {i}")));
            }
            (Err(g), Err(a)) => assert_eq!(g, a, "request {i}: same refusal"),
            other => panic!("request {i}: batched and alone disagree: {other:?}"),
        }
    }
    assert_eq!(lines(batched[0].as_ref().unwrap()).len(), 11);
    assert_eq!(lines(batched[1].as_ref().unwrap()).len(), 0, "an empty file is an empty answer");
    assert_eq!(lines(batched[6].as_ref().unwrap()).len(), 17);
    assert!(batched[3].as_ref().unwrap_err().contains("outside the model's range"));
    assert!(batched[4].is_err() && batched[5].is_err());
    assert!(batched[0].is_ok() && batched[2].is_ok() && batched[6].is_ok(), "bad neighbours fail nobody else");
}

#[test]
fn a_subjects_prediction_ignores_its_neighbours_and_padding() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (subjects, _) = population(40, 5);
    let saved = saved(&subjects);
    let enc = saved.encode(&subjects).unwrap();
    let together = predict_log_hazards(&saved.model, &enc);
    let spread = together[0]
        .iter()
        .zip(&together[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(spread > 1e-4, "the test model must tell subjects apart ({spread})");
    for i in [0usize, 7, 8, 23, 39] {
        // Alone: the other seven rows of the batch are padding.
        let alone = predict_log_hazards(&saved.model, &enc[i..=i]);
        // Reversed neighbourhood: same subject, different slot, other company.
        let mut rotated = enc.clone();
        rotated.rotate_left(i % 5 + 1);
        let at = (i + enc.len() - (i % 5 + 1)) % enc.len();
        let rot = predict_log_hazards(&saved.model, &rotated);
        for k in 0..alone[0].len() {
            assert!(
                (alone[0][k] - together[i][k]).abs() < 1e-5,
                "subject {i} alone vs with neighbours, hazard {k}"
            );
            assert!(
                (rot[at][k] - together[i][k]).abs() < 1e-5,
                "subject {i} in another slot, hazard {k}"
            );
        }
    }
}
