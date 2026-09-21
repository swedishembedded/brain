// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-weight end-to-end parity: `modernbert::import_dir` loads the REAL
//! `convaiinnovations/laya` checkpoint, `build_sequence` builds a real
//! `(state, question)` call, `ModernBert` + `LayaHead` run it forward on
//! device, and the result is compared BEHAVIORALLY (argmax option agreement,
//! argmax act-decision agreement) against the real Python `DecisionModel`
//! forward (`scripts/parity-dump/laya_real.py`, run on the same real
//! weights, the same tokenizer, the same three `(state, question)` cases).
//!
//! **Why behavioral, not a numeric tolerance** (same reasoning the Laya
//! plan's own "parity tolerance is two different bars" note states, and
//! M2/M3 both cite for their own tiny-fixture tests, which correctly DO use
//! a tight tolerance): the tiny-fp32-fixture parity tests compare
//! FRESHLY-INITIALIZED fp32 weights end to end and can meaningfully assert
//! `1e-4`. This checkpoint is F16 on disk (the model card says bf16; the
//! REAL bytes are F16 - confirmed by this crate's own importer this
//! session), and F16/bf16-to-fp32 rounding error compounds across 28
//! pre-LN layers of a real, trained model; a tensor-for-tensor numeric bound
//! against it would be either meaninglessly loose or flaky depending on
//! which layer's rounding happens to tip a near-tie. Argmax/decision
//! agreement is the strong, meaningful, actually-achievable signal: does the
//! real forward pass in `crates/modernbert` land on the SAME answer the real
//! PyTorch model does, end to end, through the real tokenizer, the real
//! `build_sequence`, and the real weights.
//!
//! Real finding (see the M4 commit message for the definitive statement):
//! run this test yourself to confirm, since whether it passed is exactly the
//! kind of claim that should not be taken on faith from a comment.
//!
//! Gated on both the checkpoint (`brain pull convaiinnovations/laya`) and
//! the fixture (`scripts/parity-dump/laya_real.py`'s output, not committed -
//! same convention as every other `scripts/parity-dump/*.py` fixture in this
//! crate) being present; skips cleanly otherwise.

use gpu_core::DeviceBuffer;
use modernbert::laya::LayaHead;
use modernbert::model::ModernBert;
use modernbert::{OrderedJson, Question, State};

fn fixture_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("MODERNBERT_REAL_FIXTURE_DIR") {
        return std::path::PathBuf::from(d);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya_real")
}

/// The three `(state, question)` cases, transcribed BY HAND from
/// `scripts/parity-dump/laya_real.py`'s own `CASES` - kept in the exact same
/// key order (see `modernbert::sequence`'s module doc on why order matters
/// for a multi-key state) since this test builds them as Rust values
/// directly rather than round-tripping through JSON.
fn cases() -> Vec<(&'static str, State, Question)> {
    vec![
        (
            "triage_choice",
            State::Json(OrderedJson::object(vec![
                ("channel", OrderedJson::str("chat")),
                ("turns", OrderedJson::int(4)),
                ("last_message", OrderedJson::str("My card was charged twice, please refund one of them.")),
                ("tags", OrderedJson::array(vec![OrderedJson::str("billing"), OrderedJson::str("duplicate-charge")])),
            ])),
            Question::Choice {
                ins: "What is the customer's primary intent?".to_string(),
                options: vec![
                    ("refund".to_string(), Some("wants money back".to_string())),
                    ("complaint".to_string(), Some("expressing dissatisfaction".to_string())),
                    ("question".to_string(), Some("asking for information".to_string())),
                    ("other".to_string(), Some(String::new())),
                ],
            },
        ),
        (
            "escalation_noul",
            State::Str(
                "Customer: This is the third time I've contacted support about this. I want to speak to a manager immediately."
                    .to_string(),
            ),
            Question::Noul { ins: "Is the customer asking to escalate to a human or manager?".to_string(), false_text: None, true_text: None },
        ),
        (
            "satisfaction_score",
            State::Json(OrderedJson::object(vec![(
                "summary",
                OrderedJson::str("Agent resolved the billing dispute within one message and offered a discount."),
            )])),
            Question::Score {
                ins: "Rate how satisfied the customer likely is, from 0 (very unhappy) to 3 (delighted).".to_string(),
                options: vec!["very unhappy".to_string(), "neutral".to_string(), "satisfied".to_string(), "delighted".to_string()],
            },
        ),
    ]
}

#[test]
fn real_weight_forward_matches_the_real_python_reference_behaviorally() {
    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull convaiinnovations/laya`"));
        return;
    }
    let manifest_path = fixture_dir().join("manifest.json");
    if !manifest_path.exists() {
        brain_testutil::skip(&format!(
            "{} absent - run `python3 scripts/parity-dump/laya_real.py --dir {dir} --out <scratch>` \
             and copy manifest.json into {}",
            manifest_path.display(),
            fixture_dir().display()
        ));
        return;
    }
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();

    brain_testutil::mem("before import_dir");
    let ckpt = modernbert::import_dir(&dir).expect("import_dir");
    brain_testutil::mem("after import_dir");

    let tok_path = format!("{dir}/tokenizer/tokenizer.json");
    let tok = data::qwen_tokenizer::QwenBpe::from_file(&tok_path).expect("load tokenizer");

    let max_len = ckpt.rl.max_len;
    let head_max_len = ckpt.rl.head_max_len;

    let gpu = gpu_core::testgpu::dev(modernbert::kern::PIPELINES);
    let mut enc = ModernBert::new_on(gpu.share(), ckpt.cfg.clone(), max_len, max_len, &ckpt.encoder_init);
    let cap_markers = 8u32;
    let mut head = LayaHead::new_on(gpu.share(), ckpt.laya_cfg.clone(), max_len, max_len, cap_markers, 1, &ckpt.head_init);
    brain_testutil::mem("after building ModernBert + LayaHead on device");

    let fixture_cases = manifest["cases"].as_array().expect("cases");
    let mut checked = 0;
    for (name, state, q) in cases() {
        let fx = fixture_cases.iter().find(|c| c["name"].as_str() == Some(name)).unwrap_or_else(|| panic!("fixture missing case {name}"));
        let want_ids: Vec<u32> = fx["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want_markers: Vec<usize> = fx["markers"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let want_argmax_option = fx["argmax_option"].as_u64().unwrap() as usize;
        let want_argmax_act = fx["argmax_act"].as_u64().unwrap() as usize;

        let (ids, markers) = modernbert::build_sequence(&tok, &ckpt.cfg, &state, &q, max_len, head_max_len, None, false);
        assert_eq!(ids, want_ids, "case {name}: build_sequence ids diverged from the Python reference");
        assert_eq!(markers, want_markers, "case {name}: build_sequence markers diverged from the Python reference");

        let rows = ids.len() as u32;
        let spans = [(0u32, rows)];
        enc.set_batch(&ids, &spans);
        enc.forward();
        enc.gpu.poll_wait();

        let marker_rows: Vec<u32> = markers.iter().map(|&m| m as u32).collect();
        let qtype = [q.qtype().index()];
        let arity = [markers.len()];
        let hidden_buf: &DeviceBuffer = enc.hidden_buf();
        head.set_call(hidden_buf, &spans, &qtype, &marker_rows, &arity);
        let (logits, act_logits) = head.forward();
        head.poll_wait();

        let got_argmax_option = logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;
        let got_argmax_act = act_logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;

        eprintln!(
            "case {name}: rust logits={logits:?} argmax={got_argmax_option} (want {want_argmax_option}) \
             act_logits={act_logits:?} argmax_act={got_argmax_act} (want {want_argmax_act})"
        );
        assert_eq!(got_argmax_option, want_argmax_option, "case {name}: option argmax disagrees with the real Python reference");
        assert_eq!(got_argmax_act, want_argmax_act, "case {name}: act-head argmax disagrees with the real Python reference");
        checked += 1;
    }
    assert_eq!(checked, cases().len());
}
