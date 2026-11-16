// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `pretrain` through the executor, on a model `init_model` created: every weight trains on raw text with held-out
//! evaluations on the step frames, the checkpoint is published and installed, and a chat full fine-tune trains on it;
//! what pretraining has no use for is refused by name before any work.

use serde_json::{json, Value};

use crate::create::{corpus, init_params};
use crate::support::{assert_reservation_returned, data_digest, rig_qwen3, run_action, work_is_clean, Frames, Rig};

/// A tiny ouro created by `init_model` and installed as `local/<name>`.
pub fn created(rig: &Rig, name: &str) {
    let digest = data_digest(rig, &corpus(400)).hex();
    run_action(rig, trainjob::create::ACTION, &trainjob::create::spec(), init_params(&digest, name), &[], &Frames::default()).expect("the model is created");
}

/// Tool-call chat records in the generic-messages-v2 shape (OpenAI tool calls, arguments as JSON text).
pub fn tool_records() -> Vec<u8> {
    let tools = json!([{"type": "function", "function": {"name": "read_file", "description": "read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]);
    (0..3)
        .map(|i| {
            json!({"tools": tools, "messages": [
                {"role": "user", "content": format!("what is in {i}.txt"), "train": false},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": format!("{{\"path\": \"{i}.txt\"}}")}}], "train": true},
                {"role": "tool", "content": format!("{i}"), "tool_call_id": "call_0", "train": false},
                {"role": "assistant", "content": format!("the number is {i}"), "train": true}
            ]})
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes()
}

pub fn pretrain_params(rig: &Rig, base: &str, install_as: &str) -> Value {
    let train = data_digest(rig, &corpus(300)).hex();
    let held = data_digest(rig, &corpus(40)).hex();
    json!({"base": base, "data": json!({"train": {"digest": train}, "held_out": {"digest": held}}).to_string(), "steps": 6, "batch": 2, "block": 32, "lr": 5e-3, "eval_every": 3, "eval_batches": 2, "seed": 3, "install_as": install_as})
}

#[test]
fn a_created_model_pretrains_with_evaluations_installs_and_takes_a_chat_fine_tune() {
    use trainjob::pretrain::{spec, ACTION};
    let rig = rig_qwen3("pretrain", 16 << 30);
    created(&rig, "scratch");
    let frames = Frames::default();
    let out = run_action(&rig, ACTION, &spec(), pretrain_params(&rig, "local/scratch", "pre-1"), &[], &frames).expect("the run").outputs;
    assert_eq!((out["status"].as_str(), out["objective"].as_str(), out["steps_completed"].as_u64(), out["separator"].as_str()), (Some("completed"), Some("pretrain"), Some(6), Some("<|endoftext|>")), "{out}");
    assert_eq!(out["evaluations"].as_array().unwrap().iter().map(|e| e["step"].as_u64().unwrap()).collect::<Vec<_>>(), [3, 6]);
    let evaluated: Vec<u64> = frames.steps().iter().filter(|f| f["eval_loss"].is_number()).map(|f| f["step"].as_u64().unwrap()).collect();
    assert_eq!(evaluated, [3, 6], "the step frames of the evaluating steps carry the held-out loss");
    assert_eq!(out["installed_as"], "local/pre-1");
    assert!(out["checkpoint_digest"].is_string() && out["documents"] == 300 && out["held_out_documents"] == 40);

    // The pretrained model is a chat base: every weight trains on tool-call records rendered by its own template.
    let chat = data_digest(&rig, &tool_records()).hex();
    let full = json!({"base": "local/pre-1", "steps": 2, "lr": 5e-3, "seed": 3, "max_block": 512, "thinking": false, "install_as": "chat-1", "data": json!({"train": {"digest": chat}}).to_string()});
    let tuned = run_action(&rig, trainjob::consolidate::FULL_ACTION, &trainjob::consolidate::full_spec(), full, &[], &Frames::default()).expect("the chat fine-tune").outputs;
    assert_eq!((tuned["status"].as_str(), tuned["installed_as"].as_str()), (Some("completed"), Some("local/chat-1")), "{tuned}");
    assert!(work_is_clean(&rig.work_root));
    assert_reservation_returned(&rig);
}

#[test]
fn pretrain_refuses_what_it_has_no_use_for_before_any_work() {
    use trainjob::pretrain::{spec, ACTION};
    let rig = rig_qwen3("pretrain-door", 16 << 30);
    let frames = Frames::default();
    let train = data_digest(&rig, &corpus(10)).hex();
    let ask = |patch: Value, blobs: &[(&str, Vec<u8>)]| {
        let mut p = json!({"base": crate::support::BASE, "data": json!({"train": {"digest": train}}).to_string()});
        p.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
        run_action(&rig, ACTION, &spec(), p, blobs, &frames).unwrap_err()
    };
    for (patch, needle) in [
        (json!({"eval_every": 2}), "held_out"),
        (json!({"data": json!({"train": {"digest": train}, "monitor": {"digest": train}}).to_string()}), "not corpora of pretraining"),
        (json!({"data": json!({"train": {"path": "/etc/passwd"}}).to_string()}), "unknown field"),
        (json!({"data": json!({"train": {"blob": "dataset"}}).to_string()}), "not attached"),
        (json!({"base": "nobody/nothing"}), "not in the model store"),
        (json!({"base": "../../etc"}), "base"),
        (json!({"install_as": "a/b"}), "install_as"),
        (json!({"separator": ""}), "separator"),
        (json!({"lr": 1e-4, "min_lr": 1e-3}), "min_lr"),
        (json!({"block": 1}), "block"),
        (json!({"batch": 100_000}), "batch"),
        (json!({"out": "elsewhere"}), "unknown param"),
        (json!({"save": "elsewhere"}), "unknown param"),
    ] {
        let e = ask(patch.clone(), &[]);
        assert!(e.contains(needle), "{patch}: {e}");
    }
    assert!(ask(json!({}), &[("held_out", corpus(5))]).contains("does not use"));
    assert!(frames.events().is_empty() && work_is_clean(&rig.work_root));
    assert!(rig.exec.residency().placements.is_empty());
}
