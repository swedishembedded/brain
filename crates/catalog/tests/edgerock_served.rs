// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context language
// models at a fixed memory budget for its clients. If your team needs
// expertise in bounded-state inference and continuous batching then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A pulled EdgeRock directory (tiny dims, the real layout: text-only `model.layers.*` names, bf16 tensors, the
//! anchor-memory tensors, a tied head) served through the residency adapter: batched requests answer as the same
//! requests do one by one, an exact repeat of a prompt is answered from the prefix cache, and the footprint the
//! scheduler places is the weights plus a fixed number of fixed-size slots.

#[allow(dead_code)]
#[path = "../../qwen35/tests/common/hf_fixture.rs"]
mod hf_fixture;

use capability::Invocation;
use catalog::resident_edgerock::EdgeRockResident;
use catalog::resident_qwen35::Qwen35Sizing;
use qwen35::config::Qwen35Config;
use residency::{Device, Instance, InstanceKey, ResidentModel};

/// The tiny EdgeRock with a vocabulary that holds every byte, so a byte-level tokenizer covers any ASCII prompt.
fn cfg() -> Qwen35Config {
    Qwen35Config { vocab: 264, max_position_embeddings: 96, block_size: 96, ..Qwen35Config::edgerock_tiny() }
}

fn pulled_dir(tag: &str) -> std::path::PathBuf {
    let cfg = cfg();
    let dir = hf_fixture::write_text_only_checkpoint(&cfg, tag, &qwen35::init::init_weights(&cfg, 29));
    let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), serde_json::json!(i))).collect();
    let tokenizer = serde_json::json!({"model": {"vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}});
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    dir
}

fn ask(prompt: &str) -> Invocation {
    Invocation::new().set("prompt", serde_json::json!(prompt)).set("chat", serde_json::json!(false)).set("max_new", serde_json::json!(4)).set("temp", serde_json::json!(0.0)).set("seed", serde_json::json!(1))
}

fn text(result: capability::ActionResult) -> String {
    result.expect("the request is answered").outputs.get("text").and_then(|t| t.as_str().map(str::to_string)).expect("text output")
}

fn metric(instance: &dyn Instance, name: &str) -> u64 {
    instance.metrics().into_iter().find(|(k, _)| k == name).and_then(|(_, v)| v.as_u64()).unwrap_or_else(|| panic!("metric {name}"))
}

#[test]
fn a_pulled_directory_is_served_batched_with_prefix_reuse() {
    let dir = pulled_dir("served");
    let sizing = Qwen35Sizing { max_batch: 3, ..Qwen35Sizing::default() };
    let resident = EdgeRockResident::from_hf_dir(dir.to_str().unwrap(), "rockmatrix-ai/EdgeRock", sizing).expect("a windowed EdgeRock directory");
    let key = InstanceKey::new("rockmatrix-ai/EdgeRock", "default");
    assert_eq!(resident.manifest().max_context_tokens, Some(96), "the advertised context is the model's position limit, not a memory figure");
    assert_eq!(resident.batch_limit(), Some(3));

    let prompts = ["the vault code is 4817, remember it", "a second and different conversation", "and a third one, shorter than both"];
    gpu_core::devices::with_host_tier(|| {
        let mut instance = resident.activate(&key, Device::Cpu).expect("the resident activates");
        let alone: Vec<String> = prompts.iter().map(|p| text(instance.run("generate", &ask(p), &mut |_| {}))).collect();
        assert!(alone.iter().all(|t| !t.is_empty()));

        let invs: Vec<Invocation> = prompts.iter().map(|p| ask(p)).collect();
        let together: Vec<String> = instance.run_batch("generate", &invs, &mut |_, _| {}).into_iter().map(text).collect();
        assert_eq!(together, alone, "a batch answers each request as it is answered alone");
        assert!(metric(&*instance, "kv_prefix_hit_tokens") >= prompts[0].len() as u64, "the repeated prompts were answered from the prefix cache");
    });
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_placed_footprint_is_the_weights_plus_fixed_size_slots() {
    let dir = pulled_dir("footprint");
    let one = EdgeRockResident::from_hf_dir(dir.to_str().unwrap(), "x/EdgeRock", Qwen35Sizing { max_batch: 1, ..Qwen35Sizing::default() }).unwrap();
    let four = EdgeRockResident::from_hf_dir(dir.to_str().unwrap(), "x/EdgeRock", Qwen35Sizing { max_batch: 4, ..Qwen35Sizing::default() }).unwrap();
    let key = InstanceKey::new("x/EdgeRock", "default");
    let (slots_one, slots_four) = (one.slots(), four.slots());
    let per_slot = qwen35::model::slot_state_bytes(&cfg());
    let extra = u64::from(slots_four.0 + slots_four.1 - slots_one.0 - slots_one.1) * per_slot;
    assert_eq!(four.estimate(&key).vram - one.estimate(&key).vram, extra, "a slot costs the same fixed bytes whatever the context");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_plain_qwen35_directory_is_not_taken_for_edgerock() {
    let plain = Qwen35Config::tiny();
    let dir = hf_fixture::write_text_only_checkpoint(&plain, "plain", &qwen35::init::init_weights(&plain, 3));
    let err = EdgeRockResident::from_hf_dir(dir.to_str().unwrap(), "x/Qwen", Qwen35Sizing::default()).err().expect("a plain decoder is refused");
    assert!(err.contains("plain Qwen3.5"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}
