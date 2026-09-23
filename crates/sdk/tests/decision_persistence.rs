// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A trained decision model must survive the process that trained it.
//!
//! `save_head` writes the head alone and names the encoder it attaches to,
//! which is the right artifact exactly while the encoder has not moved. The
//! `Decide` arm trains the encoder too (`ENCODER_LR` on every step), so that
//! arm could train and then had nothing it was allowed to write.
//!
//! Of the three ways out, two change what this arm learns - freeze the
//! encoder during training, or make the caller choose between a saveable run
//! and a trained one. The third changes nothing about training: when the
//! encoder HAS moved, the artifact grows to carry it, so a run's checkpoint
//! always reproduces the run. That is the one taken here. `save_head` keeps
//! its refusal, because a head-only file genuinely cannot.
//!
//! Swedish Embedded AB implements the checkpoint boundary - the part that
//! decides whether what you measured is what you can ship - for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use brain::DecisionPipeline;

const INSTRUCTIONS: &str = "which banking intent does this message express";
const PROBE: &str = "my replacement card has still not arrived";

fn options() -> Vec<String> {
    ["card arrival", "exchange rate", "pin blocked", "top up failed"].iter().map(|s| s.to_string()).collect()
}

fn examples() -> Vec<(String, usize)> {
    [
        ("when will my new card get here", 0),
        ("my replacement card has not arrived", 0),
        ("what rate do you use for euros", 1),
        ("how is the exchange rate decided", 1),
        ("my pin is blocked after three tries", 2),
        ("the machine blocked my pin", 2),
        ("my top up did not go through", 3),
        ("topping up my account failed again", 3),
    ]
    .iter()
    .map(|(t, l)| (t.to_string(), *l))
    .collect()
}

fn checkpoint() -> Option<String> {
    let dir = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2")?;
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!(
            "{dir}/model.safetensors absent - run `brain pull sentence-transformers/all-MiniLM-L6-v2`"
        ));
        return None;
    }
    Some(dir)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// THE property: train (which moves the encoder), save, reload in a fresh
/// pipeline, and get the SAME answer.
///
/// Same answer, not merely a loadable file. A checkpoint that loads and
/// answers differently is the failure `save_head`'s refusal exists to
/// prevent, and it would be invisible to any test that only checked the
/// write succeeded.
#[test]
fn a_trained_model_reloads_and_answers_identically() {
    let Some(dir) = checkpoint() else { return };
    let out = scratch("decision-save");

    let opts = options();
    let refs: Vec<&str> = opts.iter().map(String::as_str).collect();
    let mut pipe = DecisionPipeline::builder(&dir).load().expect("load");
    let data = examples();
    let ex: Vec<(&str, usize)> = data.iter().map(|(t, l)| (t.as_str(), *l)).collect();
    pipe.train_choices(&ex, &opts, INSTRUCTIONS, 8, 4, 0x5A_1E, &mut |_, _| {}).expect("train");

    let before = pipe.choose(PROBE, INSTRUCTIONS, &refs).expect("choose");
    pipe.save_model(out.to_str().unwrap()).expect("a trained model must be writable");
    drop(pipe);

    let mut back = DecisionPipeline::builder(out.to_str().unwrap())
        .load()
        .expect("a saved decision model must load through the same builder");
    let after = back.choose(PROBE, INSTRUCTIONS, &refs).expect("choose");

    assert_eq!(after.choice, before.choice, "the reloaded model picked a different option");
    assert_eq!(after.probabilities.len(), before.probabilities.len());
    for ((na, pa), (nb, pb)) in after.probabilities.iter().zip(&before.probabilities) {
        assert_eq!(na, nb, "option order moved across the save");
        assert!(
            (pa - pb).abs() < 1e-4,
            "{na}: reloaded probability {pa} differs from the saved model's {pb}"
        );
    }
    let _ = std::fs::remove_dir_all(&out);
}

/// The artifact is self-contained: a reader who was not there gets the
/// config, the weights and the tokenizer, and needs nothing out of band.
#[test]
fn a_saved_model_is_a_complete_checkpoint_directory() {
    let Some(dir) = checkpoint() else { return };
    let out = scratch("decision-complete");

    let pipe = DecisionPipeline::builder(&dir).load().expect("load");
    pipe.save_model(out.to_str().unwrap()).expect("save");

    for f in ["config.json", "model.safetensors", "tokenizer.json"] {
        assert!(out.join(f).is_file(), "a saved decision model is missing {f}");
    }
    let _ = std::fs::remove_dir_all(&out);
}

/// `save_head`'s refusal is the reason this exists, so it must still refuse -
/// a head-only file really cannot reproduce a model whose encoder moved, and
/// weakening that would trade a loud failure for a silent one.
#[test]
fn save_head_still_refuses_a_moved_encoder() {
    let Some(dir) = checkpoint() else { return };
    let mut pipe = DecisionPipeline::builder(&dir).load().expect("load");
    let data = examples();
    let ex: Vec<(&str, usize)> = data.iter().map(|(t, l)| (t.as_str(), *l)).collect();
    pipe.train_choices(&ex, &options(), INSTRUCTIONS, 4, 2, 1, &mut |_, _| {}).expect("train");

    let path = std::env::temp_dir().join(format!("brain-head-refused-{}.safetensors", std::process::id()));
    let err = pipe.save_head(path.to_str().unwrap()).expect_err("a moved encoder must still be refused");
    assert!(format!("{err}").contains("encoder was trained"), "unhelpful refusal: {err}");
    let _ = std::fs::remove_file(&path);
}

/// A saved model records the fit that made it, not only the architecture it
/// shares with every sibling.
///
/// This is the defect knowledge #151 is about: a published solve-rate table
/// whose training command was never written down anywhere, so the run could
/// not be reproduced, defended or discarded. Architecture is not provenance -
/// every model in this family has the same architecture, so a `config.json`
/// carrying only `n_layers` and `d_model` cannot distinguish two runs that
/// differ in every way that mattered.
#[test]
fn a_saved_model_records_the_fit_that_made_it() {
    let Some(dir) = checkpoint() else { return };
    let out = scratch("decision-fit");

    let opts = options();
    let mut pipe = DecisionPipeline::builder(&dir).load().expect("load");
    let data = examples();
    let ex: Vec<(&str, usize)> = data.iter().map(|(t, l)| (t.as_str(), *l)).collect();
    pipe.train_choices(&ex, &opts, INSTRUCTIONS, 8, 4, 0x5A_1E, &mut |_, _| {}).expect("train");
    pipe.save_model(out.to_str().unwrap()).expect("save");

    let text = std::fs::read_to_string(out.join("config.json")).expect("config.json");
    let cfg: serde_json::Value = serde_json::from_str(&text).expect("config.json parses");
    let fit = cfg
        .get("trained_for")
        .unwrap_or_else(|| panic!("a trained model saved no record of its fit: {text}"));

    // The settings that change the result, and the seed that makes the run
    // repeatable. Without these the checkpoint cannot answer "which run?".
    for key in ["steps", "batch", "examples", "seed"] {
        assert!(fit.get(key).is_some(), "the saved fit does not record {key:?}: {fit}");
    }
    assert_eq!(fit["steps"], 8, "recorded a step count the run did not use");
    assert_eq!(fit["batch"], 4, "recorded a batch the run did not use");
    assert_eq!(fit["examples"], data.len(), "recorded an example count the run did not use");

    let _ = std::fs::remove_dir_all(&out);
}
