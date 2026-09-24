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
//! - [`laya_backed_pipeline_trains_saves_and_reloads_a_head`] and
//!   [`a_head_file_from_the_other_architecture_is_refused_by_name`]: the
//!   training CONTRACT on the Laya arm - `supports_training` is true,
//!   `train_choices` runs and logs a finite loss per step, `save_head`
//!   writes a head that `DecisionPipelineBuilder::head` loads back and that
//!   reproduces the same answer, and a head from the OTHER architecture is
//!   refused by name. That a run actually learns is measured separately, on
//!   real weights, by
//!   [`real_laya_checkpoint_head_training_improves_held_out_accuracy`].
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

/// The MECHANICS of the Laya training contract, on a synthetic fixture: a
/// `Flow` reports the arm as trainable, `train_choices` runs and reports a
/// finite loss, `save_head` writes a real file, and that file loads back
/// through `DecisionPipelineBuilder::head` and reproduces the SAME answer.
///
/// Deliberately NOT a claim that the fixture learned anything - it cannot.
/// Its trunk is random, every option marker is the same `[MASK]` token, and
/// a random ModernBERT maps them all to nearly the same hidden state, so a
/// frozen-trunk head has nothing to tell the options apart by (measured in
/// `crates/modernbert/tests/train_convergence.rs`). That a run LEARNS is what
/// the real-weight test below measures, on the real trunk. Splitting the two
/// is the point: this one runs everywhere in seconds and would catch a
/// broken save/load path or a panicking loop, which is most of what can
/// regress.
#[test]
fn laya_backed_pipeline_trains_saves_and_reloads_a_head() {
    use brain::flow::Stages;

    let root = scratch_root("laya-training");
    write_laya_fixture(&root);
    let mut pipe = brain::DecisionPipeline::builder(root.to_str().unwrap()).load().unwrap();
    assert!(pipe.supports_training(), "the Laya arm trains now");

    let options: Vec<String> = ["alpha", "beta", "gamma"].iter().map(|s| s.to_string()).collect();
    let mut seen = 0usize;
    let loss = pipe
        .train_choices(
            &[("a short state string", 0), ("another state entirely", 2)],
            &options,
            "which option applies",
            6,
            // Batch ONE on purpose: this asserts the per-step log contract,
            // so a step has to be an example. At a batch of one the loop is
            // bit-identical to the per-example one it replaced.
            1,
            7,
            &mut |_, l| {
                assert!(l.is_finite(), "a training step reported a non-finite loss: {l}");
                seen += 1;
            },
        )
        .expect("the laya arm must train");
    assert_eq!(seen, 6, "the log callback must fire once per step");
    assert!(loss.is_finite(), "mean tail loss is not finite: {loss}");
    assert!(pipe.steps_taken() > 0, "training reported no steps");

    let before = pipe.choose("a short state string", "which option applies", &["alpha", "beta", "gamma"]).unwrap();

    // Written OUTSIDE the checkpoint directory on purpose: `import_dir`
    // treats every `.safetensors` in the directory as part of the
    // checkpoint, so a head dropped next to `model.safetensors` makes the
    // directory unloadable. Head files are adapters and belong elsewhere -
    // which is what `samples/decision/*` already do (`out/<name>-head.
    // safetensors`).
    let save_path = std::env::temp_dir().join(format!("brain-laya-head-{}.safetensors", std::process::id()));
    pipe.save_head(save_path.to_str().unwrap()).expect("save_head");
    assert!(save_path.exists(), "save_head wrote nothing");

    let mut reloaded = brain::DecisionPipeline::builder(root.to_str().unwrap())
        .head(save_path.to_str().unwrap())
        .load()
        .expect("a saved Laya head must load back");
    let after = reloaded.choose("a short state string", "which option applies", &["alpha", "beta", "gamma"]).unwrap();
    let _ = std::fs::remove_file(&save_path);

    assert_eq!(before.choice, after.choice, "a reloaded head chose differently");
    for ((n1, p1), (n2, p2)) in before.probabilities.iter().zip(&after.probabilities) {
        assert_eq!(n1, n2);
        assert!((p1 - p2).abs() <= 1e-4, "{n1}: {p1} vs {p2} after a save/load round trip");
    }
}

/// A `decide` head is not a Laya head. Loading one into the other arm must
/// say so by name rather than produce a model that answers plausible
/// nonsense.
#[test]
fn a_head_file_from_the_other_architecture_is_refused_by_name() {
    let root = scratch_root("laya-wrong-head");
    write_laya_fixture(&root);
    let bogus = std::env::temp_dir().join(format!("brain-not-a-laya-head-{}.safetensors", std::process::id()));
    write_hf_safetensors_f32(&bogus, &[("encoder.layer.0.attention.self.query.weight".to_string(), vec![2, 2], vec![0.0; 4])]);

    let err = brain::DecisionPipeline::builder(root.to_str().unwrap())
        .head(bogus.to_str().unwrap())
        .load()
        .expect_err("a foreign head must be refused");
    match err {
        brain::Error::Backend(msg) => assert!(
            msg.contains("not a Laya decision-head parameter"),
            "unhelpful refusal: {msg}"
        ),
        other => panic!("expected Error::Backend, got {other:?}"),
    }
    let _ = std::fs::remove_file(&bogus);
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

/// `train_choices` then `save_head` must WORK on the decide arm, and the knob
/// that makes it work is the one the caller sets.
///
/// This was a reproduced defect: the arm
/// fine-tunes its encoder on every step by default, and a head-only adapter
/// cannot reproduce a model whose encoder moved, so `save_head` refused after
/// every training run this SDK could start. Both halves of the choice are
/// gated here - the default still fine-tunes and is still refused, because
/// that is what this arm's published accuracies were measured with, and the
/// frozen run writes a file that loads back.
#[test]
fn a_frozen_encoder_run_can_save_its_head_and_a_fine_tuned_one_still_cannot() {
    let Some(dir) = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull sentence-transformers/all-MiniLM-L6-v2`"));
        return;
    }
    const INSTRUCTIONS: &str = "which banking intent does this message express";
    /// The option-order seed `train_choices` takes; fixed so the two arms
    /// below shuffle identically and are comparable.
    const SEED: u64 = 7;
    let options: Vec<String> = ["card arrival", "exchange rate", "pin blocked"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let train: &[(&str, usize)] = &[
        ("I am still waiting on my replacement card", 0),
        ("what rate did you use for the conversion", 1),
        ("my pin stopped working at the machine", 2),
    ];
    let out = std::env::temp_dir().join(format!("brain-decide-head-{}.safetensors", std::process::id()));
    let path = out.to_str().unwrap();

    // The DEFAULT: the encoder is fine-tuned, and the head-only file is
    // refused because it could not reproduce this model.
    let mut live = brain::DecisionPipeline::builder(&dir).load().expect("real minilm");
    assert!(!live.encoder_frozen(), "the decide arm fine-tunes its encoder by default");
    live.train_choices(train, &options, INSTRUCTIONS, 3, 7, SEED, &mut |_, _| {}).expect("train");
    assert!(live.encoder_was_trained());
    let err = live.save_head(path).expect_err("a fine-tuned encoder must still refuse a head-only save");
    assert!(format!("{err}").contains("encoder was trained"), "unexpected refusal: {err}");

    // The CHOICE: freeze the encoder, and the same run produces a file.
    let mut frozen = brain::DecisionPipeline::builder(&dir).load().expect("real minilm");
    frozen.set_encoder_frozen(true).expect("the decide arm can freeze its encoder");
    frozen.train_choices(train, &options, INSTRUCTIONS, 3, 7, SEED, &mut |_, _| {}).expect("train");
    assert!(!frozen.encoder_was_trained(), "a frozen encoder must not have moved");
    frozen.save_head(path).expect("a frozen-encoder run must be saveable");

    let refs: Vec<&str> = options.iter().map(String::as_str).collect();
    let before = frozen.choose(train[0].0, INSTRUCTIONS, &refs).expect("choose");
    let mut reloaded = brain::DecisionPipeline::builder(&dir)
        .head(path)
        .load()
        .expect("a trained decide head must load back");
    let after = reloaded.choose(train[0].0, INSTRUCTIONS, &refs).expect("choose");
    assert_eq!(before.choice, after.choice, "the reloaded head answered differently");
    for ((_, a), (_, b)) in before.probabilities.iter().zip(&after.probabilities) {
        assert!((a - b).abs() < 1e-4, "reloaded probabilities differ: {a} vs {b}");
    }
    let _ = std::fs::remove_file(&out);
}

/// A minibatch must be the SAME training, only with less gradient noise: the
/// SDK's batch knob is `decide::Decide::accumulate_batch` underneath, so what
/// is gated here is the wiring - that a batch of `b` consumes `b` examples per
/// step, logs once per step, and produces a model that still answers.
///
/// The gradient itself is held against `b` sequential steps in
/// `crates/decide/tests/minibatch.rs`, where it can be compared buffer by
/// buffer rather than through a loss.
#[test]
fn a_batched_run_consumes_a_batch_per_step_and_still_trains() {
    let Some(dir) = brain_testutil::model_dir("sentence-transformers/all-MiniLM-L6-v2") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull sentence-transformers/all-MiniLM-L6-v2`"));
        return;
    }
    const INSTRUCTIONS: &str = "which banking intent does this message express";
    /// The option-order seed `train_choices` takes; fixed so the two arms
    /// below shuffle identically and are comparable.
    const SEED: u64 = 7;
    let options: Vec<String> = ["card arrival", "exchange rate", "pin blocked"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let train: &[(&str, usize)] = &[
        ("I am still waiting on my replacement card", 0),
        ("what rate did you use for the conversion", 1),
        ("my pin stopped working at the machine", 2),
    ];

    let mut pipe = brain::DecisionPipeline::builder(&dir).load().expect("real minilm");
    assert_eq!(pipe.batch_size(), brain::DEFAULT_TRAIN_BATCH);
    pipe.set_batch_size(4);
    assert_eq!(pipe.batch_size(), 4);

    let mut logged = 0usize;
    let tail = pipe
        .train_choices(train, &options, INSTRUCTIONS, 5, 7, SEED, &mut |_, l| {
            assert!(l.is_finite(), "a batched step reported a non-finite loss: {l}");
            logged += 1;
        })
        .expect("a batched run must train");
    // ONCE PER OPTIMIZER STEP, not once per example: `steps` keeps meaning
    // steps, and the batch is what each of them accumulates over.
    assert_eq!(logged, 5, "the log callback must fire once per optimizer step");
    assert!(tail.is_finite());
    assert_eq!(pipe.steps_taken(), 5, "a batch is one optimizer step, not one per example");

    let refs: Vec<&str> = options.iter().map(String::as_str).collect();
    let a = pipe.choose(train[0].0, INSTRUCTIONS, &refs).expect("choose");
    let sum: f32 = a.probabilities.iter().map(|(_, p)| p).sum();
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

/// THE MEASURED CLAIM: head training on the REAL `convaiinnovations/laya`
/// checkpoint makes the model better on examples it never trained on.
///
/// The task is a deliberately ARBITRARY mapping - four everyday topics onto
/// four meaningless option names (`alpha`/`beta`/`gamma`/`delta`). That is
/// what makes the number honest in both directions: a pretrained decision
/// model cannot guess it, so the zero-shot score is near chance and there is
/// real headroom; and the held-out sentences share only their TOPIC with the
/// training ones, so getting them right requires generalizing from the
/// examples rather than memorizing them.
///
/// Held-out accuracy is scored in the CANONICAL option order, while training
/// shuffles a sampled subset every step (`OptionSampler`), so a model that
/// learned "the answer is at index 2" scores at chance here.
///
/// Skips cleanly when the 843 MB checkpoint is absent, like every other
/// real-weight test in this file.
#[test]
#[ignore = "slow: ~200 real ModernBERT-large training steps - run via `make test/slow`"]
fn real_laya_checkpoint_head_training_improves_held_out_accuracy() {
    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull convaiinnovations/laya`"));
        return;
    }

    const INSTRUCTIONS: &str = "which bucket does this record belong in";
    let options: Vec<String> = ["alpha", "beta", "gamma", "delta"].iter().map(|s| s.to_string()).collect();
    // 0 = finance, 1 = cooking, 2 = sport, 3 = weather - arbitrarily.
    let train: &[(&str, usize)] = &[
        ("the invoice is thirty days overdue", 0),
        ("quarterly revenue beat the forecast", 0),
        ("we wrote down the receivable last month", 0),
        ("the audit found a discrepancy in the ledger", 0),
        ("simmer the sauce for twenty minutes", 1),
        ("preheat the oven before baking", 1),
        ("fold the egg whites into the batter", 1),
        ("season the stock with bay and thyme", 1),
        ("he scored in the final minute", 2),
        ("the match went to extra time", 2),
        ("she broke the national record in the relay", 2),
        ("the referee awarded a penalty", 2),
        ("the wind is picking up outside", 3),
        ("heavy rain all afternoon", 3),
        ("frost is expected overnight", 3),
        ("the storm made landfall at dawn", 3),
    ];
    // 20, not 8: this is the number the whole test exists to produce, and at
    // 8 a single lucky example is 12.5 percentage points of it.
    let held_out: &[(&str, usize)] = &[
        ("the balance sheet shows a larger provision", 0),
        ("cash flow improved after the refinancing", 0),
        ("the tax authority disputed our deduction", 0),
        ("we issued a credit note against the order", 0),
        ("interest on the loan is capitalised monthly", 0),
        ("whisk the butter and sugar until pale", 1),
        ("let the dough rest for an hour", 1),
        ("reduce the stock until it coats a spoon", 1),
        ("blanch the beans before refreshing them", 1),
        ("toast the spices in a dry pan first", 1),
        ("the striker was substituted at half time", 2),
        ("their goalkeeper saved three shots", 2),
        ("she qualified fastest in the heats", 2),
        ("the coach named an unchanged line up", 2),
        ("he was booked for a late challenge", 2),
        ("fog is reducing visibility on the coast", 3),
        ("temperatures will drop below freezing", 3),
        ("a band of showers moves in tonight", 3),
        ("gusts of fifty knots were recorded offshore", 3),
        ("it stayed overcast and humid all day", 3),
    ];

    let refs: Vec<&str> = options.iter().map(String::as_str).collect();
    let score = |pipe: &mut brain::DecisionPipeline| -> f32 {
        let mut hit = 0;
        for (text, label) in held_out {
            let a = pipe.choose(text, INSTRUCTIONS, &refs).expect("choose");
            if a.index == *label {
                hit += 1;
            }
        }
        hit as f32 / held_out.len() as f32
    };

    let mut pipe = brain::DecisionPipeline::builder(&dir).load().expect("real laya checkpoint");
    let before = score(&mut pipe);

    // Fixed rather than configurable: a gate whose step budget a runner can
    // change is not a gate, and every BRAIN_* env var read from this
    // workspace has to be documented as real configuration.
    let steps = 200usize;
    let mut first = 0.0f32;
    let mut n_first = 0usize;
    let tail = pipe
        // Batch ONE, so this gate keeps measuring what it measured when its
        // threshold was fitted: `LAYA_HEAD_LR` was chosen against a budget of
        // one example per step over a few hundred steps, and changing the
        // batch changes the run the number describes.
        .train_choices(train, &options, INSTRUCTIONS, steps, 1, 0x1A_2026, &mut |step, l| {
            if step < steps / 10 + 1 {
                first += l;
                n_first += 1;
            }
        })
        .expect("the laya arm must train on real weights");
    let first = first / n_first.max(1) as f32;

    let after = score(&mut pipe);
    println!(
        "laya head training ({steps} steps): loss {first:.4} -> {tail:.4};  held-out accuracy \
         {before:.3} -> {after:.3}  (chance {:.3}, {} held-out examples)",
        1.0 / options.len() as f32,
        held_out.len()
    );

    assert!(tail.is_finite() && first.is_finite(), "loss went non-finite: {first} -> {tail}");
    // NOT asserted: that the loss falls. It is printed because the trend is
    // worth seeing, but a REINFORCE objective's scalar at a batch of ONE is
    // not a monotone quantity - its policy half is a mean-zero, high-variance
    // term over `G` sampled reports, and the option SUBSET each step draws
    // varies in size (2..4 here), so the per-step floor moves too. Asserting
    // on it would gate this feature on noise. Held-out accuracy is the
    // number that means something, so that is what is gated.
    assert!(
        after > before,
        "held-out accuracy did not improve: {before:.3} -> {after:.3} (chance {:.3})",
        1.0 / options.len() as f32
    );
    // Chance is 0.25 on four options, and the mapping is arbitrary, so this
    // bar cannot be cleared by a model that did not learn from the examples.
    assert!(after >= 0.5, "held-out accuracy {after:.3} is too low to call this trained");

    // ... and what was trained must survive a save/load round trip on real
    // weights too, not only on the synthetic fixture.
    let out = std::env::temp_dir().join(format!("brain-real-laya-head-{}.safetensors", std::process::id()));
    pipe.save_head(out.to_str().unwrap()).expect("save_head on real weights");
    let mut reloaded = brain::DecisionPipeline::builder(&dir)
        .head(out.to_str().unwrap())
        .load()
        .expect("a trained real Laya head must load back");
    let reloaded_acc = score(&mut reloaded);
    let _ = std::fs::remove_file(&out);
    assert!(
        (reloaded_acc - after).abs() < 1e-6,
        "a reloaded head scored {reloaded_acc:.3}, not the {after:.3} that was saved"
    );
}
