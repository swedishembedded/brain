// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end coverage of `DecisionPipeline`'s M6 dual-backend dispatch:
//! `crates/decide`'s MiniLM/BERT-family encoder (the pre-existing path,
//! byte-identical `load_decide`) and `convaiinnovations/laya`'s
//! ModernBERT-large trunk + decision head (`crates/modernbert`, new this
//! milestone).
//!
//! Swedish Embedded AB implements multi-architecture SDK surfaces - one
//! typed API dispatching to whichever real backbone a checkpoint directory
//! names - for clients who need to swap or compare model families without
//! rewriting the calling code. If your team needs a model brought up this
//! way, you can procure our services by emailing info@swedishembedded.com.
//!
//! ## What each test proves
//!
//! - [`a_bert_shaped_directory_with_no_laya_marker_routes_to_the_decide_arm`] /
//!   [`a_laya_shaped_directory_routes_to_the_laya_arm_not_decide`]: the new
//!   `resolve_decision_backend` sniffer classifies a directory by its own
//!   STRUCTURE (root `config.json` vs. `rl_agent_config.json` +
//!   `encoder/config.json`), cheaply, with no real weights needed - the
//!   dispatch logic M6 actually adds.
//! - [`laya_synthetic_fixture_choose_and_probability_produce_real_numbers`]:
//!   a small, fully synthetic (tiny random weights) Laya-shaped checkpoint
//!   directory loads through the Laya arm and answers real `choose`/
//!   `probability` calls - the fast, no-network structural proof that the
//!   whole chain (sniff -> `modernbert::import_dir` -> `build_sequence` ->
//!   trunk forward -> head forward -> host softmax) actually works, endpoint
//!   to endpoint, without needing the real 843 MB checkpoint.
//! - [`laya_backed_pipeline_refuses_training_with_a_typed_error`]: the
//!   deliberate M6 gap - `train_choices`/`save_head` on a Laya-backed
//!   pipeline are a clean `Err`, never a panic or a silent no-op.
//! - [`real_minilm_checkpoint_still_answers_through_the_decide_arm`]: THE
//!   PRIMARY SAFETY BAR - `load_decide` is byte-identical after this
//!   milestone, so a real `sentence-transformers/all-MiniLM-L6-v2` checkpoint
//!   must answer exactly as it did before. Skips cleanly when the checkpoint
//!   is not present in this checkout (same `brain_testutil::skip` convention
//!   every other real-weight test in this plan uses).
//! - [`real_laya_checkpoint_choose_and_probability_produce_real_numbers`]: the
//!   same real-weight proof for the NEW arm, against the real downloaded
//!   `convaiinnovations/laya` checkpoint. Skips cleanly when absent.
//! - [`typed_questions_answer_one_per_question_on_the_synthetic_laya_fixture`]
//!   and [`a_question_outside_the_published_limits_is_refused_by_name`]:
//!   `DecisionPipeline::decide`, the typed multi-question request (a
//!   `Choice` with descriptions, a `Score`, a `Noul`, over structured
//!   `State::Json`) - one answer per question, in order, of the type asked,
//!   and a request outside the published limits refused before any model
//!   runs. `samples/decision/json` is the JSON endpoint built on it.
//! - [`real_minilm_checkpoint_answers_a_typed_request_through_the_decide_arm`]
//!   and [`real_laya_checkpoint_answers_a_typed_multi_question_request`]: the
//!   same typed request through BOTH real checkpoints - one API, two
//!   architectures. Skip cleanly when absent.

use std::path::{Path, PathBuf};

/// A fixture directory that deletes itself when the test ends.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

fn scratch_root(tag: &str) -> Scratch {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-sdk-decision-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

/// A minimal root `config.json` naming `model_type: "bert"` - enough for
/// `resolve_decision_backend`'s sniffer to route to the `Decide` arm, not
/// enough (deliberately - no `model.safetensors`) for `load_decide` to
/// actually finish loading. Proves ROUTING without paying for a real
/// tokenizer/weights fixture.
fn write_bert_shaped_config_only(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "model_type": "bert", "vocab_size": 30522, "hidden_size": 384, "num_hidden_layers": 6,
            "num_attention_heads": 12, "intermediate_size": 1536, "max_position_embeddings": 512,
        })
        .to_string(),
    )
    .unwrap();
}

/// The Laya-specific marker shape (`rl_agent_config.json` at the root plus a
/// nested `encoder/config.json` naming `model_type: "modernbert"`) with NO
/// `model.safetensors` - enough for the sniffer to route to the Laya arm,
/// not enough for `modernbert::import_dir` to finish loading. The mirror
/// image of [`write_bert_shaped_config_only`].
fn write_laya_shaped_markers_only(dir: &Path) {
    std::fs::create_dir_all(dir.join("encoder")).unwrap();
    std::fs::write(
        dir.join("encoder").join("config.json"),
        serde_json::json!({
            "model_type": "modernbert", "vocab_size": 300, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "intermediate_size": 19, "max_position_embeddings": 64,
            "global_attn_every_n_layers": 2, "local_attention": 6,
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("rl_agent_config.json"),
        serde_json::json!({
            "head_layers": 2, "max_len": 48, "head_max_len": 32, "max_prefixes": 6,
            "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
        })
        .to_string(),
    )
    .unwrap();
}

#[test]
fn a_bert_shaped_directory_with_no_laya_marker_routes_to_the_decide_arm() {
    let root = scratch_root("bert-routing");
    write_bert_shaped_config_only(&root);

    let err = brain::DecisionPipeline::builder(root.to_str().unwrap()).load().unwrap_err();
    match err {
        // `load_decide`'s own next read after `config.json` parses is
        // `model.safetensors` - this exact failure point proves the
        // sniffer routed here (not into `modernbert::import_dir`, whose
        // own next-missing-file error names `encoder/config.json`/
        // `rl_agent_config.json` instead) and that `load_decide` itself
        // (untouched by this milestone) still runs unmodified.
        brain::Error::Backend(msg) => assert!(msg.contains("model.safetensors"), "expected the decide arm's own missing-weights message, got: {msg}"),
        other => panic!("expected Error::Backend naming model.safetensors, got {other:?}"),
    }
}

#[test]
fn a_laya_shaped_directory_routes_to_the_laya_arm_not_decide() {
    let root = scratch_root("laya-routing");
    write_laya_shaped_markers_only(&root);

    let err = brain::DecisionPipeline::builder(root.to_str().unwrap()).load().unwrap_err();
    match err {
        brain::Error::Backend(msg) => {
            assert!(!msg.contains("bert"), "the decide arm's own model_type check must never run on a Laya-shaped directory: {msg}");
            // `modernbert::import_dir` reads encoder/config.json, then
            // rl_agent_config.json, then tokenizer/tokenizer.json, then
            // model.safetensors - this fixture supplies only the first two,
            // so the next file it names is the tokenizer, proving
            // `modernbert::import_dir` (not `load_decide`) ran.
            assert!(msg.contains("tokenizer"), "expected modernbert::import_dir's own next-missing-file message, got: {msg}");
        }
        other => panic!("expected a clean Error::Backend, got {other:?}"),
    }
}

/// Build a raw HuggingFace-format `.safetensors` file (`[u64 header_len]
/// [JSON header][blob]`, F32 throughout) - the same shape
/// `checkpoint::weightio::WeightReader::open_hf_dir` (what
/// `modernbert::import_dir` reads) expects, hand-assembled since this crate
/// writes brain's own carded container format, not a foreign HF one.
fn write_hf_safetensors_f32(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, data) in tensors {
        let start = blob.len();
        for v in data {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let end = blob.len();
        header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
    }
    let hbytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&hbytes);
    buf.extend_from_slice(&blob);
    std::fs::write(path, buf).unwrap();
}

/// Map a `ModernBertConfig::tensor_manifest`/`laya::tensor_manifest` brain
/// param name to the real checkpoint's own HF dotted name -
/// `modernbert::import.rs`'s `hf_to_encoder`/`hf_to_head` reversed by hand
/// (those functions are private to that crate; this reproduces their own
/// documented name table, not a re-derivation - see
/// `crates/modernbert/src/import.rs`'s module doc and
/// `hf_to_encoder_name_mapping`/`hf_to_head_name_mapping` unit tests for the
/// ground truth this mirrors).
fn encoder_brain_name_to_hf(name: &str) -> String {
    match name {
        "tok.weight" => return "encoder.embeddings.tok_embeddings.weight".to_string(),
        "emb_norm.weight" => return "encoder.embeddings.norm.weight".to_string(),
        "final_norm.weight" => return "encoder.final_norm.weight".to_string(),
        _ => {}
    }
    let rest = name.strip_prefix("blocks.").unwrap_or_else(|| panic!("unexpected encoder param {name}"));
    let (l, leaf) = rest.split_once('.').unwrap();
    let hf_leaf = match leaf {
        "attn_norm.weight" => "attn_norm.weight",
        "qkv.weight" => "attn.Wqkv.weight",
        "proj.weight" => "attn.Wo.weight",
        "mlp_norm.weight" => "mlp_norm.weight",
        "mlp.wi.weight" => "mlp.Wi.weight",
        "mlp.wo.weight" => "mlp.Wo.weight",
        other => panic!("fixture: unmapped encoder param leaf {other}"),
    };
    format!("encoder.layers.{l}.{hf_leaf}")
}

fn head_brain_name_to_hf(name: &str) -> String {
    if name == "type_emb.weight" {
        return "type_emb.weight".to_string();
    }
    if let Some(rest) = name.strip_prefix("head.") {
        let (l, leaf) = rest.split_once('.').unwrap();
        let hf_leaf = match leaf {
            "attn.in_proj.weight" => "self_attn.in_proj_weight",
            "attn.in_proj.bias" => "self_attn.in_proj_bias",
            "attn.out_proj.weight" => "self_attn.out_proj.weight",
            "attn.out_proj.bias" => "self_attn.out_proj.bias",
            "ff1.weight" => "linear1.weight",
            "ff1.bias" => "linear1.bias",
            "ff2.weight" => "linear2.weight",
            "ff2.bias" => "linear2.bias",
            "norm1.weight" => "norm1.weight",
            "norm1.bias" => "norm1.bias",
            "norm2.weight" => "norm2.weight",
            "norm2.bias" => "norm2.bias",
            other => panic!("fixture: unmapped head param leaf {other}"),
        };
        return format!("head.layers.{l}.{hf_leaf}");
    }
    match name {
        "scorer.norm.weight" => "scorer.0.weight",
        "scorer.norm.bias" => "scorer.0.bias",
        "scorer.fc1.weight" => "scorer.1.weight",
        "scorer.fc1.bias" => "scorer.1.bias",
        "scorer.fc2.weight" => "scorer.3.weight",
        "scorer.fc2.bias" => "scorer.3.bias",
        "act.fc1.weight" => "act_head.0.weight",
        "act.fc1.bias" => "act_head.0.bias",
        "act.fc2.weight" => "act_head.2.weight",
        "act.fc2.bias" => "act_head.2.bias",
        other => panic!("fixture: unmapped head param {other}"),
    }
    .to_string()
}

/// A full, fully synthetic (tiny random weights) `convaiinnovations/laya`
/// checkpoint directory: `encoder/config.json`
/// (`ModernBertConfig::tiny()`-shaped, but with a wider vocab so a
/// merge-free byte-level tokenizer's ~261 ids fit), `rl_agent_config.json`,
/// `tokenizer/tokenizer.json` (the same merge-free byte-level fixture shape
/// `crates/sdk/tests/embedding_pipeline.rs::write_qwen3_embed_base` uses),
/// and `model.safetensors` (every encoder+head tensor `modernbert::
/// import_dir`'s coverage check requires, named the real checkpoint's own
/// way via [`encoder_brain_name_to_hf`]/[`head_brain_name_to_hf`]).
fn write_laya_fixture(dir: &Path) {
    std::fs::create_dir_all(dir.join("encoder")).unwrap();
    std::fs::create_dir_all(dir.join("tokenizer")).unwrap();

    let encoder_cfg = serde_json::json!({
        "model_type": "modernbert", "vocab_size": 300, "hidden_size": 64, "num_hidden_layers": 4,
        "num_attention_heads": 4, "intermediate_size": 19, "max_position_embeddings": 64,
        "global_attn_every_n_layers": 2, "local_attention": 6,
    });
    std::fs::write(dir.join("encoder").join("config.json"), encoder_cfg.to_string()).unwrap();
    std::fs::write(
        dir.join("rl_agent_config.json"),
        serde_json::json!({
            "head_layers": 2, "max_len": 48, "head_max_len": 32, "max_prefixes": 6,
            "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
        })
        .to_string(),
    )
    .unwrap();

    let mut vocab = serde_json::Map::new();
    for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::json!(i));
    }
    let specials = [("[UNK]", 256u32), ("[CLS]", 257), ("[SEP]", 258), ("[PAD]", 259), ("[MASK]", 260)];
    let added_tokens: Vec<serde_json::Value> = specials.iter().map(|(c, id)| serde_json::json!({"content": c, "id": id})).collect();
    std::fs::write(
        dir.join("tokenizer").join("tokenizer.json"),
        serde_json::json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added_tokens}).to_string(),
    )
    .unwrap();

    let cfg = modernbert::ModernBertConfig::from_hf_json(&encoder_cfg).unwrap();
    let laya_cfg = modernbert::LayaConfig::new(cfg.d_model);
    let enc_init = modernbert::init::init_weights(&cfg, 7);
    let head_init = modernbert::init::init_weights_laya(&laya_cfg, 8);

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for (name, shape) in cfg.tensor_manifest() {
        let data = enc_init.get(&name).unwrap_or_else(|| panic!("fixture: init missing {name}")).clone();
        tensors.push((encoder_brain_name_to_hf(&name), shape, data));
    }
    for (name, shape) in modernbert::laya::tensor_manifest(&laya_cfg) {
        let data = head_init.get(&name).unwrap_or_else(|| panic!("fixture: init missing {name}")).clone();
        tensors.push((head_brain_name_to_hf(&name), shape, data));
    }
    tensors.push(("temperature".to_string(), vec![3], vec![1.0, 1.0, 1.0]));

    write_hf_safetensors_f32(&dir.join("model.safetensors"), &tensors);
}

#[test]
fn laya_synthetic_fixture_choose_and_probability_produce_real_numbers() {
    let root = scratch_root("laya-synthetic");
    write_laya_fixture(&root);

    let mut pipe = brain::DecisionPipeline::builder(root.to_str().unwrap()).load().expect("a synthetic Laya-shaped directory must load through the Laya arm");

    // `inner()` has no `Decide` to hand back on this arm - the M6 breaking
    // change (`Option<&mut Decide>`, not `&mut Decide`).
    assert!(pipe.inner().is_none(), "a Laya-backed pipeline must return None from inner()");

    let choice = pipe.choose("a short state string", "which option applies", &["alpha", "beta", "gamma"]).expect("laya choose");
    assert_eq!(choice.probabilities.len(), 3);
    let sum: f32 = choice.probabilities.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-3, "option probabilities must sum to 1, got {sum}: {:?}", choice.probabilities);
    assert!(choice.probabilities.iter().any(|(name, _)| name == &choice.choice), "choice must name one of the supplied options");
    assert!((0.0..=1.0).contains(&choice.confidence), "confidence out of range: {}", choice.confidence);

    let p = pipe.probability("a short state string", "does the proposition hold").expect("laya probability");
    assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
}

#[test]
fn laya_backed_pipeline_refuses_training_with_a_typed_error() {
    let root = scratch_root("laya-training-gap");
    write_laya_fixture(&root);
    let mut pipe = brain::DecisionPipeline::builder(root.to_str().unwrap()).load().unwrap();

    let err = pipe.train_choices(&[("hello", 0)], &["a".to_string(), "b".to_string()], "pick one", 1, 0, &mut |_, _| {}).unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(msg.to_lowercase().contains("not yet implemented"), "expected a clear not-yet-implemented message, got: {msg}"),
        other => panic!("expected Error::Backend, got {other:?}"),
    }

    let save_path = root.join("should-not-be-written.safetensors");
    let err = pipe.save_head(save_path.to_str().unwrap()).unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(msg.to_lowercase().contains("not yet implemented"), "expected a clear not-yet-implemented message, got: {msg}"),
        other => panic!("expected Error::Backend, got {other:?}"),
    }
    assert!(!save_path.exists(), "save_head must not write anything on the Laya arm");
}

/// THE PRIMARY SAFETY BAR: `load_decide` is byte-identical after this
/// milestone, so a real `sentence-transformers/all-MiniLM-L6-v2` checkpoint
/// must answer exactly as before. Skips cleanly (does not fail) when the
/// checkpoint is not present in this checkout - the same convention every
/// other real-weight test in this plan follows.
#[test]
fn real_minilm_checkpoint_still_answers_through_the_decide_arm() {
    let Some(dir) = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull sentence-transformers/all-MiniLM-L6-v2`"));
        return;
    }

    let mut pipe = brain::DecisionPipeline::builder(&dir).load().expect("a real MiniLM directory must still load through the decide arm");
    assert!(pipe.inner().is_some(), "a Decide-backed pipeline must return Some(&mut Decide) from inner()");

    let choice = pipe.choose("I am still waiting on my replacement card", "which banking intent does this message express", &["card arrival", "exchange rate", "pin blocked"]).expect("decide choose");
    assert_eq!(choice.probabilities.len(), 3);
    let sum: f32 = choice.probabilities.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-3, "option probabilities must sum to 1, got {sum}");
}

/// The same real-weight proof for the NEW arm, against the real
/// `convaiinnovations/laya` checkpoint M4 fetched. Skips cleanly when the
/// checkpoint is absent from this checkout.
#[test]
fn real_laya_checkpoint_choose_and_probability_produce_real_numbers() {
    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull convaiinnovations/laya`"));
        return;
    }

    let mut pipe = brain::DecisionPipeline::builder(&dir).load().expect("the real Laya checkpoint directory must load through the laya arm");
    assert!(pipe.inner().is_none(), "a Laya-backed pipeline must return None from inner()");

    let choice = pipe
        .choose(
            "Customer: This is the third time I've contacted support about this billing issue.",
            "What is the customer's primary intent?",
            &["refund", "complaint", "question", "other"],
        )
        .expect("laya choose against the real checkpoint");
    assert_eq!(choice.probabilities.len(), 4);
    let sum: f32 = choice.probabilities.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-3, "option probabilities must sum to 1, got {sum}: {:?}", choice.probabilities);

    let p = pipe.probability("Customer: I want to speak to a manager immediately.", "Is the customer asking to escalate to a human or manager?").expect("laya probability against the real checkpoint");
    assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
}

/// One state, three typed questions, one answer each - the request shape a
/// caller with a JSON front end (`samples/decision/json`) needs, and the one
/// `choose`/`probability` alone cannot express: a `Score` has no surface at
/// all, a `Choice`'s options cannot carry the descriptions the model is
/// supposed to read, and a caller holding structured state has nowhere to put
/// it but a string of its own devising.
#[test]
fn typed_questions_answer_one_per_question_on_the_synthetic_laya_fixture() {
    use brain::decision::{Answer, OrderedJson, Question, State};

    let root = scratch_root("laya-typed");
    write_laya_fixture(&root);
    let mut pipe = brain::DecisionPipeline::builder(root.to_str().unwrap()).load().unwrap();

    let state = State::Json(OrderedJson::parse(r#"{"ticket": "payments keep failing", "days": 3}"#).unwrap());
    let questions = vec![
        Question::Noul { instructions: "does this convey urgency".into(), yes: None, no: None },
        Question::Choice {
            instructions: "which team should handle this".into(),
            options: vec![
                brain::decision::Opt::described("billing", "payments, invoicing, refunds"),
                brain::decision::Opt::new("technical"),
                brain::decision::Opt::new("sales"),
            ],
        },
        Question::Score {
            instructions: "how frustrated is the customer".into(),
            levels: vec!["calm".into(), "frustrated".into(), "very angry".into()],
        },
    ];

    let answers = pipe.decide(&state, &questions).expect("a typed request must answer");
    assert_eq!(answers.len(), 3, "one answer per question, no more and no fewer");

    // In the order asked, and of the type asked - a caller keys its own
    // results off position, so a reordered or retyped answer is silently
    // wrong rather than loudly broken.
    match &answers[0] {
        Answer::Noul { noul } => assert!((0.0..=1.0).contains(noul), "noul out of range: {noul}"),
        other => panic!("question 0 was a noul, got {other:?}"),
    }
    match &answers[1] {
        Answer::Choice { choice, probabilities, confidence } => {
            assert_eq!(probabilities.len(), 3);
            let sum: f32 = probabilities.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-3, "choice probabilities must sum to 1, got {sum}");
            assert!(["billing", "technical", "sales"].contains(&choice.as_str()), "the choice must name a supplied option, got {choice}");
            assert!(probabilities.iter().all(|(n, _)| ["billing", "technical", "sales"].contains(&n.as_str())), "probabilities are keyed by the option NAME, never by the description the model read");
            assert!((0.0..=1.0).contains(confidence));
        }
        other => panic!("question 1 was a choice, got {other:?}"),
    }
    match &answers[2] {
        Answer::Score { score, legend, probabilities, confidence } => {
            assert_eq!(legend, &["calm".to_string(), "frustrated".to_string(), "very angry".to_string()]);
            assert_eq!(probabilities.len(), 3);
            // 0-based expectation over the levels, so it can never leave them.
            assert!((0.0..=2.0).contains(score), "score must lie on the level scale, got {score}");
            assert!((0.0..=1.0).contains(confidence));
        }
        other => panic!("question 2 was a score, got {other:?}"),
    }
}

/// A question outside the published limits is refused with the limit in the
/// message, on both arms, BEFORE any model runs - a malformed request from a
/// pipe must not reach the tokenizer.
#[test]
fn a_question_outside_the_published_limits_is_refused_by_name() {
    use brain::decision::{Question, State};

    let root = scratch_root("laya-limits");
    write_laya_fixture(&root);
    let mut pipe = brain::DecisionPipeline::builder(root.to_str().unwrap()).load().unwrap();

    let one_level = Question::Score { instructions: "how bad".into(), levels: vec!["only".into()] };
    let err = pipe.decide(&State::Str("anything".into()), &[one_level]).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("levels"), "the error must name what is wrong: {msg}");

    let no_options = Question::Choice { instructions: "which".into(), options: vec![] };
    let err = pipe.decide(&State::Str("anything".into()), &[no_options]).unwrap_err();
    assert!(format!("{err}").contains("option"), "expected a message about the empty option list, got {err}");
}

/// The same typed request through the OTHER backbone: one API, two
/// architectures, answered the same way - the property `DecisionPipeline`
/// exists for. Skips cleanly when the checkpoint is absent.
#[test]
fn real_minilm_checkpoint_answers_a_typed_request_through_the_decide_arm() {
    use brain::decision::{Answer, OrderedJson, Question, State};

    let Some(dir) = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull sentence-transformers/all-MiniLM-L6-v2`"));
        return;
    }

    let mut pipe = brain::DecisionPipeline::builder(&dir).load().unwrap();
    let state = State::Json(OrderedJson::parse(r#"{"message": "my replacement card has not arrived"}"#).unwrap());
    let answers = pipe
        .decide(
            &state,
            &[
                Question::Choice {
                    instructions: "which banking intent does this message express".into(),
                    options: vec![brain::decision::Opt::new("card arrival"), brain::decision::Opt::new("exchange rate")],
                },
                Question::Noul { instructions: "is the customer waiting on something".into(), yes: None, no: None },
            ],
        )
        .expect("the decide arm must answer a typed request");
    assert_eq!(answers.len(), 2);
    assert!(matches!(answers[0], Answer::Choice { .. }));
    assert!(matches!(answers[1], Answer::Noul { .. }));
}

/// And against the real Laya checkpoint, which is what the JSON sample runs
/// by default. Skips cleanly when absent.
#[test]
fn real_laya_checkpoint_answers_a_typed_multi_question_request() {
    use brain::decision::{Answer, Question, State};

    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull convaiinnovations/laya`"));
        return;
    }

    let mut pipe = brain::DecisionPipeline::builder(&dir).load().unwrap();
    let state = State::Str("Help! My payments have been failing for 3 days and nobody answers support.".into());
    let answers = pipe
        .decide(
            &state,
            &[
                Question::Choice {
                    instructions: "Which team should handle this?".into(),
                    options: vec![
                        brain::decision::Opt::described("billing", "payments, invoicing, refunds"),
                        brain::decision::Opt::described("technical", "bugs, outages, integrations"),
                        brain::decision::Opt::described("sales", "pricing, upgrades, new accounts"),
                    ],
                },
                Question::Score {
                    instructions: "How frustrated is the customer?".into(),
                    levels: vec!["calm".into(), "frustrated".into(), "very angry".into()],
                },
            ],
        )
        .expect("the laya arm must answer a typed request");
    assert_eq!(answers.len(), 2);
    match &answers[0] {
        // Not a calibration claim about this checkpoint - only that the
        // routing question reaches it and comes back keyed by option name.
        Answer::Choice { choice, .. } => assert!(["billing", "technical", "sales"].contains(&choice.as_str()), "got {choice}"),
        other => panic!("expected a choice, got {other:?}"),
    }
    match &answers[1] {
        Answer::Score { score, legend, .. } => {
            assert_eq!(legend.len(), 3);
            assert!((0.0..=2.0).contains(score), "score off the level scale: {score}");
        }
        other => panic!("expected a score, got {other:?}"),
    }
}

/// The Laya arm's published probabilities must be the REFERENCE
/// implementation's, not merely the right argmax.
///
/// `rl_agent_config.json` carries two calibration tables: a per-qtype
/// `temperature` and a finer `temperature_by_options` keyed by
/// (qtype, option-count bucket), and `rl_agent_api.py` consults the finer one
/// FIRST. They disagree on every `choice` the released checkpoint can be
/// asked - 1.9064 at two options, 1.7602 at 3-5, 1.0000 at 6-10 and 0.1006
/// past ten, against a per-qtype 1.6369 - so reading only the coarse scalar
/// hands a caller a distribution the reference never produces. Argmax is
/// unaffected by any positive temperature, which is exactly why this needs
/// its own test: the answer looks right while the number a caller thresholds
/// on is wrong.
///
/// The expected values here are what the real `rl_agent_api.py` printed for
/// this exact request on this exact checkpoint (its own 4-decimal rounding),
/// not a re-derivation. Skips cleanly when the checkpoint is absent.
#[test]
fn real_laya_choice_probabilities_match_the_reference_serving_calibration() {
    use brain::decision::{Answer, Opt, Question, State};

    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull convaiinnovations/laya`"));
        return;
    }

    let mut pipe = brain::DecisionPipeline::builder(&dir).load().unwrap();
    let answers = pipe
        .decide(
            &State::Str("My card was stolen yesterday.".into()),
            &[Question::Choice {
                instructions: "Which banking topic is this?".into(),
                options: vec![
                    Opt::described("report stolen card", "the customer's card was lost or stolen"),
                    Opt::described("exchange rate", "questions about currency conversion rates"),
                    Opt::described("close account", "the customer wants to close their account"),
                ],
            }],
        )
        .expect("the laya arm must answer");

    match &answers[0] {
        Answer::Choice { choice, probabilities, confidence } => {
            assert_eq!(choice, "report stolen card");
            let want = [0.9688f32, 0.0177, 0.0135];
            for ((name, got), expect) in probabilities.iter().zip(want) {
                assert!(
                    (got - expect).abs() < 1e-3,
                    "{name}: {got} is not the reference's {expect} (a 3-option choice is calibrated at \
                     temperature_by_options[\"choice:3-5\"] = 1.7602, not temperature[0] = 1.6369)"
                );
            }
            assert!(
                (confidence - 0.8543).abs() < 1e-3,
                "confidence {confidence} is not the reference's 0.8543 - the coarse per-qtype temperature \
                 reads 0.8858 here, which is what this test exists to catch"
            );
        }
        other => panic!("expected a choice, got {other:?}"),
    }
}
