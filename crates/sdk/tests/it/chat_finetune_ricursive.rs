// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning very large dense
// decoders on the memory a team actually has, from full-weight training to
// adapters over a 4-bit base. If your team needs expertise in training
// hybrid-attention language models then you can procure our services by
// sending an email to info@swedishembedded.com.

// Fine-tuning is the `study` surface.
#![cfg(feature = "study")]

//! The generic chat fine-tuners on a Ripple base (toy width, CPU backend): the supervised LoRA fine-tune
//! (`ChatFineTune`), the full-parameter fine-tune and the preference fine-tune all run the same `model::finetune`
//! entry points the other architectures do.
//!

use std::path::{Path, PathBuf};

use brain::{ChatFineTune, ChatFullFineTune, FineTuneStatus, PolicyGradientConfig, PreferenceFineTune};

use crate::chat_fixture::{cpu, TempDir};

const TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}{{ message.content }}<|im_end|>{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{% endif %}";

/// The fixture tokenizer with its special tokens just above the letters instead of at the 151k mark, so the toy model's
/// vocabulary is 32 rather than 151,936 (the CPU backend's backward through a head that wide costs seconds a step).
fn write_small_tokenizer(dir: &Path) {
    let byte_encoder = data::bpe::bytes_to_unicode();
    let vocab: serde_json::Map<String, serde_json::Value> = (b'a'..=b'w').enumerate().map(|(id, b)| (byte_encoder[b as usize].to_string(), serde_json::json!(id as u32))).collect();
    let added = serde_json::json!([{"content": "<|endoftext|>", "id": 28}, {"content": "<|im_start|>", "id": 29}, {"content": "<|im_end|>", "id": 30}]);
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({ "model": { "vocab": vocab, "merges": [] }, "added_tokens": added }).to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({ "chat_template": TEMPLATE }).to_string()).unwrap();
}

/// A tiny Ripple checkpoint (brain format) over the small vocabulary.
fn ripple_model_dir(tag: &str) -> TempDir {
    let dir = TempDir::new(tag);
    let mut cfg = ricursive::TrainConfig::tiny();
    cfg.block_size = 128;
    let init = cfg.tiny_weights(11); // seed: any weights serve, the test reads the training, not the model
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, numel)| (name.clone(), vec![numel as u64], init[&name].clone())).collect();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
    write_small_tokenizer(&dir);
    dir
}

/// A chat dataset of one-line question/answer pairs, the answer supervised and the question not.
fn write_dataset(dir: &Path) -> PathBuf {
    let path = dir.join("chat.jsonl");
    let rows: Vec<String> = [("cab", "bad"), ("fed", "deaf"), ("ace", "face"), ("bag", "gab")]
        .iter()
        .map(|(u, a)| serde_json::json!({ "messages": [{"role": "user", "content": u, "train": false}, {"role": "assistant", "content": a, "train": true}], "tools": [] }).to_string())
        .collect();
    std::fs::write(&path, rows.join("\n")).unwrap();
    path
}

/// Spec: a supervised LoRA fine-tune trains an adapter over the frozen base, exports it with its card, and the
/// adapter holds adapter tensors only, at the rank asked for.
#[test]
fn a_ricursive_lora_fine_tune_exports_an_adapter_over_the_frozen_base() {
    let model = ripple_model_dir("ripple-sft");
    let outcome = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .rank(2)
        .steps(6)
        .lr(5e-2)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .run()
        .expect("the LoRA fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let card = checkpoint::st::read_card(adapter.to_str().unwrap()).unwrap().expect("the adapter carries a card");
    assert_eq!(card.family, "ricursive");
    assert_eq!(card.training.expect("the card records its run").regime, "sft_lora");
    let st = checkpoint::st::load_safetensors(adapter.to_str().unwrap()).expect("the adapter opens");
    assert!(!st.tensors.is_empty() && st.tensors.keys().all(|n| model::adapter::device::is_adapter_param(n)), "an adapter file holds adapter tensors only");
    let read = model::lora::device_adapter::read_adapter(adapter.to_str().unwrap()).expect("the adapter validates");
    assert_eq!(read.rank, 2);
    assert_eq!(read.sites.len(), 3 * 8, "q, k, v, o, the gate and the feed-forward trio of each of the three layers");
}

/// Spec: a variant of the adapter this architecture does not train is refused by name, not trained as the plain one.
#[test]
fn a_ricursive_lora_variant_it_does_not_train_is_refused_by_name() {
    let model = ripple_model_dir("ripple-variant");
    let err = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .rank_stabilized(true)
        .steps(1)
        .device(cpu())
        .run()
        .expect_err("a rank-stabilised adapter is not wired");
    assert!(err.to_string().contains("ricursive"), "{err}");
}

/// Spec: every weight trains in a full run and the result is a base the architecture opens again.
#[test]
fn a_ricursive_full_fine_tune_trains_every_weight_and_writes_a_base() {
    let model = ripple_model_dir("ripple-full");
    let outcome = ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(write_dataset(&model))
        .out_dir(model.join("run"))
        .steps(3)
        .lr(1e-2)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .run()
        .expect("the full fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    assert!(outcome.final_loss.zip(outcome.initial_loss).is_some_and(|(last, first)| last < first), "{outcome:?}");
    let checkpoint = outcome.checkpoint.expect("a completed run writes its checkpoint directory");
    assert!(checkpoint.join("tokenizer.json").is_file());
    let trained = checkpoint::weightio::WeightReader::open(checkpoint.join("model.safetensors").to_str().unwrap()).expect("the checkpoint opens");
    let cfg = ricursive::TrainConfig::from_json(&trained.config()).expect("the header carries the training configuration");
    assert_eq!(trained.names().count(), cfg.param_list().len(), "every tensor of the architecture is written");
}

/// Spec: a preference run trains adapters over the frozen base, and the margin starts at zero against the model it
/// starts from.
#[test]
fn a_ricursive_preference_fine_tune_trains_adapters_over_the_frozen_base() {
    let model = ripple_model_dir("ripple-dpo");
    let pairs = model.join("pairs.jsonl");
    let lines: Vec<String> = [("cab", "bad", "wig"), ("fed", "deaf", "vow"), ("ace", "face", "jug"), ("bag", "gab", "mop")]
        .iter()
        .map(|(u, c, r)| {
            serde_json::json!({
                "prompt": [{"role": "user", "content": u}],
                "chosen": {"role": "assistant", "content": c},
                "rejected": {"role": "assistant", "content": r},
                "metadata": {"source": "test"}
            })
            .to_string()
        })
        .collect();
    std::fs::write(&pairs, lines.join("\n")).unwrap();
    let outcome = PreferenceFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(&pairs)
        .out_dir(model.join("run"))
        .rank(2)
        .steps(3)
        .lr(1e-1)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .run()
        .expect("the preference fine-tune runs");

    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let initial = outcome.initial_loss.expect("measured");
    assert!((initial - std::f32::consts::LN_2).abs() < 1e-3, "the margin starts at zero against the starting model: {initial}");
    assert!(outcome.final_loss.is_some_and(|last| last < initial), "{outcome:?}");
    let adapter = outcome.adapter.expect("a completed run exports its adapter");
    let st = checkpoint::st::load_safetensors(adapter.to_str().unwrap()).expect("the adapter opens");
    assert!(st.tensors.keys().all(|n| model::adapter::device::is_adapter_param(n)), "an adapter file holds adapter tensors only");
}

/// Spec: a run cut into two pipeline stages trains what the one-stage run does (the loss before and after, and every
/// adapter factor), through the SDK's trainer on the CPU backend; so does a full fine-tune.
#[test]
fn a_two_stage_run_trains_what_the_one_stage_run_does() {
    let model = ripple_model_dir("ripple-stages");
    let run = |stages: u32, tag: &str| {
        ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
            .arch("ricursive")
            .dataset(write_dataset(&model))
            .out_dir(model.join(tag))
            .rank(2)
            .steps(4)
            .lr(5e-2)
            .stages(stages)
            .seed(7) // seed: the runs compared need the same factors and batches
            .device(cpu())
            .run()
            .expect("the LoRA fine-tune runs")
    };
    let (one, two) = (run(0, "one"), run(2, "two"));
    let (first, last) = ((one.initial_loss.unwrap(), one.final_loss.unwrap()), (two.initial_loss.unwrap(), two.final_loss.unwrap()));
    assert!((first.0 - last.0).abs() < 1e-5 * first.0 && (first.1 - last.1).abs() < 1e-4 * first.1.max(1e-3), "losses {first:?} vs {last:?}");
    let read = |o: &brain::ChatFineTuneOutcome| model::lora::device_adapter::read_adapter(o.adapter.as_ref().unwrap().to_str().unwrap()).unwrap();
    let (a, b) = (read(&one), read(&two));
    let worst = a.sites.iter().zip(&b.sites).flat_map(|(x, y)| x.a.iter().zip(&y.a).chain(x.b.iter().zip(&y.b))).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
    assert!(worst < 1e-4, "the factors differ by {worst}");

    let full = |stages: u32, tag: &str| {
        ChatFullFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).arch("ricursive").dataset(write_dataset(&model)).out_dir(model.join(tag)).steps(3).lr(1e-2).stages(stages).seed(7).device(cpu()).run().expect("the full fine-tune runs")
    };
    let (one, two) = (full(0, "full-one"), full(2, "full-two"));
    assert!((one.final_loss.unwrap() - two.final_loss.unwrap()).abs() < 1e-4 * one.final_loss.unwrap().max(1e-3), "{:?} vs {:?}", one.final_loss, two.final_loss);
}

/// Spec: an architecture whose trainer takes no stage count refuses one by name, before reading anything.
#[test]
fn a_stage_count_is_refused_by_an_architecture_that_lays_itself_out() {
    let err = ChatFullFineTune::from_pretrained("none").arch("qwen3").dataset("d.jsonl").out_dir(std::env::temp_dir()).stages(2).run().expect_err("refused");
    assert!(err.to_string().contains("stages") && err.to_string().contains("lays itself out"), "{err}");
}

fn lsa(top_k: usize, window: usize) -> brain::LsaConfig {
    brain::LsaConfig { index_dim: 0, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 8, top_p: top_k, top_k, sink: 2, window, early: None }
}

/// Spec: native sparse attention is a trainer option any model that declares support takes: at a budget past the row
/// it trains what dense attention does (the same losses before and after), here on the software Vulkan adapter, the
/// device of this test run that has the workgroup-barrier kernels.
#[test]
fn native_sparse_attention_at_a_full_budget_trains_what_dense_attention_does() {
    if !gpu_core::testgpu::dev(&[]).caps().workgroup_reductions {
        brain_testutil::skip_unavailable("N-LSA: this backend does not run workgroup-barrier kernels (set BRAIN_DEVICE=software)");
        return;
    }
    let model = ripple_model_dir("ripple-nlsa");
    let software = brain::Device::parse("software").unwrap();
    let run = |sparse: bool, tag: &str| {
        let t = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).arch("ricursive").dataset(write_dataset(&model)).out_dir(model.join(tag)).rank(2).steps(4).lr(5e-2).seed(7).device(software.clone());
        (if sparse { t.native_sparse_attention(lsa(1024, 16), true) } else { t }).run().expect("the LoRA fine-tune runs")
    };
    let (dense, sparse) = (run(false, "dense"), run(true, "sparse"));
    let (d, s) = ((dense.initial_loss.unwrap(), dense.final_loss.unwrap()), (sparse.initial_loss.unwrap(), sparse.final_loss.unwrap()));
    eprintln!("dense {d:?}, native sparse at a full budget {s:?}");
    assert!((d.0 - s.0).abs() < 1e-4 * d.0 && (d.1 - s.1).abs() < 1e-3 * d.1.max(1e-2), "dense {d:?} vs sparse {s:?}");
}

/// Spec: the option is refused by name where it cannot run: on a device without the barrier kernels, and by an
/// architecture whose model does not declare it.
#[test]
fn native_sparse_attention_is_refused_where_it_cannot_run() {
    if !gpu_core::testgpu::dev(&[]).caps().workgroup_reductions {
        let model = ripple_model_dir("ripple-nlsa-cpu");
        let err = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
            .arch("ricursive")
            .dataset(write_dataset(&model))
            .out_dir(model.join("run"))
            .native_sparse_attention(lsa(1024, 16), true)
            .steps(1)
            .device(cpu())
            .run()
            .expect_err("the CPU backend runs no barrier kernels");
        assert!(err.to_string().contains("workgroup-barrier"), "{err}");
    }
    let err = ChatFullFineTune::from_pretrained("none").arch("qwen35").dataset("d.jsonl").out_dir(std::env::temp_dir()).native_sparse_attention(lsa(1024, 16), true).run().expect_err("qwen35 does not declare it");
    assert!(err.to_string().contains("native sparse attention"), "{err}");
}

/// (prompt, answer, advantage): two actions of well-rewarded rollouts and two of badly rewarded ones.
const ROLLOUTS: [(&str, &str, f64); 4] = [("cab", "bad", 1.0), ("fed", "dad", 1.0), ("ace", "cab", -1.0), ("bag", "egg", -1.0)];

fn chat_record(prompt: &str, answer: &str, advantage: Option<f64>) -> String {
    let mut record = serde_json::json!({"messages": [{"role": "user", "content": prompt, "train": false}, {"role": "assistant", "content": answer, "train": true}], "tools": []});
    if let Some(a) = advantage {
        record["advantage"] = serde_json::json!(a);
    }
    record.to_string()
}

fn write_records(path: PathBuf, rows: &[(&str, &str, Option<f64>)]) -> PathBuf {
    let lines: Vec<String> = rows.iter().map(|(p, a, adv)| chat_record(p, a, *adv)).collect();
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

/// Spec: a held-out set is scored on the model in the training process before the first step and after the last,
/// through the generic trainer: the tuned loss on records like the training ones is lower than the base's.
#[test]
fn a_ricursive_run_scores_held_out_records_before_and_after() {
    let model = ripple_model_dir("ripple-held-out");
    let held_out = write_records(model.join("held.jsonl"), &[("cab", "bad", None), ("fed", "deaf", None)]);
    let outcome = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(write_dataset(&model))
        .held_out(&held_out)
        .out_dir(model.join("run"))
        .rank(2)
        .steps(8)
        .lr(5e-2)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .run()
        .expect("the LoRA fine-tune runs");
    let (base, tuned) = (outcome.base_score.expect("the base is scored"), outcome.tuned_score.expect("the tuned model is scored"));
    eprintln!("held-out loss {:?} -> {:?} over {} positions", base.loss, tuned.loss, base.positions);
    assert_eq!((base.records, tuned.records), (2, 2));
    assert!(base.positions > 0 && base.positions == tuned.positions);
    assert!(tuned.loss.unwrap() < base.loss.unwrap(), "{base:?} -> {tuned:?}");
}

/// Spec: rollouts that carry an advantage train a Ripple adapter that makes the actions of well-rewarded rollouts
/// likelier and those of badly rewarded ones less likely, measured on held-out copies of both sets; the run reports
/// the objective's statistics and records its regime on the adapter's card.
#[test]
fn a_ricursive_policy_gradient_run_raises_rewarded_actions_and_lowers_penalised_ones() {
    let model = ripple_model_dir("ripple-pg");
    let rollouts = write_records(model.join("rollouts.jsonl"), &ROLLOUTS.map(|(p, a, adv)| (p, a, Some(adv))));
    let rewarded = write_records(model.join("rewarded.jsonl"), &[(ROLLOUTS[0].0, ROLLOUTS[0].1, None), (ROLLOUTS[1].0, ROLLOUTS[1].1, None)]);
    let penalised = write_records(model.join("penalised.jsonl"), &[(ROLLOUTS[2].0, ROLLOUTS[2].1, None), (ROLLOUTS[3].0, ROLLOUTS[3].1, None)]);
    let outcome = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .arch("ricursive")
        .dataset(&rollouts)
        .held_out(&rewarded)
        .held_out_text(&penalised)
        .out_dir(model.join("run"))
        .rank(2)
        .steps(10)
        .lr(2e-2)
        .grad_accum(4)
        .seed(7) // seed: runs compared with each other need the same stream
        .device(cpu())
        .policy_gradient(PolicyGradientConfig::default())
        .run()
        .expect("the policy-gradient run");
    assert_eq!(outcome.status, FineTuneStatus::Completed);
    let loss = |s: Option<brain::HeldOutScore>| s.expect("scored").loss.expect("a loss");
    let (good, bad) = ((loss(outcome.base_score), loss(outcome.tuned_score)), (loss(outcome.base_text_score), loss(outcome.tuned_text_score)));
    eprintln!("rewarded actions' loss {:.4} -> {:.4}, penalised actions' loss {:.4} -> {:.4}", good.0, good.1, bad.0, bad.1);
    assert!(good.1 < good.0 - 1e-3, "rewarded actions got no likelier: {good:?}");
    assert!(bad.1 > bad.0 + 1e-3, "penalised actions got no less likely: {bad:?}");

    let pg = outcome.policy_gradient.as_ref().expect("a policy-gradient run reports its statistics");
    assert_eq!((pg.stats.steps.len(), pg.stats.sequences_with_advantage, pg.stats.sequences_imitation), (10, 4, 0));
    assert!((pg.stats.steps[0].ratio_mean - 1.0).abs() < 1e-4, "the first step is on-policy: {:?}", pg.stats.steps[0]);
    let card = checkpoint::st::read_card(outcome.adapter.unwrap().to_str().unwrap()).unwrap().expect("a card");
    assert_eq!(card.family, "ricursive");
    assert_eq!(card.training.expect("the card records its run").regime, "policy_gradient_lora");
}

/// Spec: an architecture whose trainer declares neither refuses a policy gradient by name, before a device is claimed.
#[test]
fn a_policy_gradient_is_refused_by_an_architecture_that_does_not_declare_it() {
    let model = ripple_model_dir("ripple-pg-refused");
    let err = ChatFineTune::from_pretrained(model.join("model.safetensors").to_str().unwrap()).arch("qwen3").dataset(write_dataset(&model)).out_dir(model.join("run")).policy_gradient(PolicyGradientConfig::default()).device(cpu()).run().expect_err("refused");
    assert!(err.to_string().contains("qwen3") && err.to_string().contains("policy_gradient"), "{err}");
}
