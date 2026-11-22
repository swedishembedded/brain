// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of looped mixture-of-experts language models for its clients. If your team
// needs expertise in putting a new architecture behind a served chat contract with bounded, cancellable requests then
// you can procure our services by sending an email to info@swedishembedded.com.

//! A pulled Inkling 2 directory served by its resident: the tiny reference checkpoint with a byte-level vocabulary and a
//! typed-message chat template of the checkpoint's own shape. The resident serves a chat request on the CPU within its
//! context through the checkpoint's template (with `reasoning_effort` as a template variable, an out-of-range value
//! refused by name, an image refused until the image path is served), the request observes cancellation before admission and between decode tokens, and the footprint it
//! reports holds the weights and the whole visit-state pool.

use std::path::PathBuf;

use capability::Invocation;
use catalog::resident_inkling2::Inkling2Resident;
use catalog::resident_qwen35::Qwen35Sizing;
use capability_mock::cancellation::{certify, Bound};
use data::rng::Lcg;
use residency::{Device, InstanceKey, ResidentModel};
use serde_json::{json, Value};

const SPECIALS: [&str; 7] = ["<|message_system|>", "<|message_user|>", "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>"];
/// Stored unembedding rows and the valid ones (bytes and the specials, then padding).
const VOCAB: usize = 272;
const VALID: usize = 256 + SPECIALS.len();

/// The checkpoint's template shape: a system message with the effort, typed user and model messages, the reply ended
/// by the sampling eos.
const TEMPLATE: &str = "<|message_system|><|content_text|>effort {{ reasoning_effort }}<|end_message|>{% for m in messages %}{% if m.role == 'user' %}<|message_user|><|content_text|>{{ m.content }}<|end_message|>{% else %}{% if m.reasoning_content %}<|message_model|><|content_thinking|>{{ m.reasoning_content }}<|end_message|>{% endif %}<|message_model|><|content_text|>{{ m.content }}<|end_message|><|content_model_end_sampling|>{% endif %}{% endfor %}";

/// The tiny reference checkpoint widened to a byte-level vocabulary, with its tokenizer, template and generation
/// config, as a pull leaves a release directory.
fn pulled_dir(tag: &str) -> Option<PathBuf> {
    let tiny = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    if !tiny.join("model.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("inkling2-served-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let mut config: Value = serde_json::from_str(&std::fs::read_to_string(tiny.join("config.json")).unwrap()).unwrap();
    config["text_config"]["vocab_size"] = json!(VOCAB);
    config["text_config"]["unpadded_vocab_size"] = json!(VALID);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    // seed: any rows serve; the test reads renderings, bounds and cancellation, never the model's quality.
    let mut rng = Lcg::new(11);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = checkpoint::safetensors::read(&tiny.join("model.safetensors").to_string_lossy())
        .unwrap()
        .into_iter()
        .map(|t| {
            if t.name == inkling2::names::embed() || t.name == inkling2::names::unembed() {
                let d = t.shape[1];
                let mut data = t.data;
                data.extend(rng.vec_scaled((VOCAB - t.shape[0]) * d, 0.5));
                (t.name, vec![VOCAB as u64, d as u64], data)
            } else {
                (t.name, t.shape.iter().map(|&s| s as u64).collect(), t.data)
            }
        })
        .collect();
    checkpoint::st::save_safetensors_with_metadata(&dir.join("model.safetensors").to_string_lossy(), &tensors, Default::default()).unwrap();
    let vocab: serde_json::Map<String, Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
    let added: Vec<Value> = SPECIALS.iter().enumerate().map(|(i, c)| json!({"content": c, "id": 256 + i, "special": true})).collect();
    std::fs::write(dir.join("tokenizer.json"), json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added}).to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), json!({"chat_template": TEMPLATE}).to_string()).unwrap();
    std::fs::write(dir.join("generation_config.json"), json!({"eos_token_id": 256 + SPECIALS.len() - 1}).to_string()).unwrap();
    Some(dir)
}

fn resident(dir: &std::path::Path) -> Inkling2Resident {
    let sizing = Qwen35Sizing { ctx: 256, ctx_explicit: true, max_batch: 2, ..Qwen35Sizing::default() };
    Inkling2Resident::from_hf_dir(dir.to_str().unwrap(), "local/inkling2", sizing).expect("the pulled directory is served")
}

fn ask(text: &str) -> Invocation {
    Invocation::new().set("messages", json!(json!([{"role": "user", "content": text}]).to_string())).set("temperature", json!(0.0))
}

#[test]
fn the_resident_serves_a_typed_message_chat_within_its_context() {
    let Some(dir) = pulled_dir("serve") else { return };
    let r = resident(&dir);
    let m = r.manifest();
    assert_eq!(m.max_context_tokens, Some(256), "the context is the asked cap, inside the model's positions");
    let generate = m.actions.iter().find(|a| a.name == "generate").expect("generate");
    assert!(generate.cancellable, "the generate action declares cancellation");
    assert!(generate.params.iter().any(|p| p.name == "reasoning_effort"), "reasoning_effort is a served param");
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let format = qwen3::chat::ChatFormat::own_template(&dir).unwrap();
    let inv = ask("hi").set("reasoning_effort", json!("high")).set("max_new", json!(3));
    let rendered = qwen3::chat::render_prompt_as(&format, &inv).unwrap();
    assert!(rendered.text.starts_with("<|message_system|><|content_text|>effort high<|end_message|>"), "{}", rendered.text);
    gpu_core::devices::with_host_tier(|| {
        let mut instance = r.activate(&InstanceKey::new("local/inkling2", "default"), Device::Cpu).expect("the resident activates");
        let out = instance.run("generate", &inv, &mut |_| {}).expect("served").outputs;
        assert_eq!(out["prompt_tokens"].as_u64(), Some(data::tokenizer::Tokenizer::encode(&tok, &rendered.text).len() as u64), "{out}");
        let e = instance.run("generate", &ask("hi").set("reasoning_effort", json!("1.5")), &mut |_| {}).unwrap_err();
        assert!(e.contains("reasoning_effort"), "an effort outside 0..1 is refused by name: {e}");
        let picture = json!([{"role": "user", "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]);
        let e = instance.run("generate", &Invocation::new().set("messages", json!(picture.to_string())), &mut |_| {}).unwrap_err();
        assert!(e.contains("image input is not served"), "an image is refused, never read as its placeholder token: {e}");
        let long = || ask("count").set("max_new", json!(24)).set("ignore_eos", json!(true));
        let instance = std::cell::RefCell::new(instance);
        // The token fires on the first event (the request's admission); the prefill already queued with it still
        // answers its first token, then the next scheduler step sees the token: two events at most.
        certify("inkling2 generate", &long, &|inv, p| instance.borrow_mut().run("generate", inv, p), Bound { fire_after: 1, within: 2 });
    });
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_footprint_holds_the_weights_and_the_whole_visit_pool() {
    let Some(dir) = pulled_dir("estimate") else { return };
    let r = resident(&dir);
    let cfg = inkling2::Inkling2Config::from_dir(&dir).unwrap();
    let layout = inkling2::serve::Engine::layout(&cfg, model::kv_tier::KvTier::F32).unwrap();
    let s = r.options().sizing;
    assert_eq!((s.live_rows, s.snapshot_rows, s.num_blocks), (2, 2, 2 * 256 / s.block_size), "two sequences of the context, as many cached prompts");
    let pool = u64::from(s.num_blocks) * layout.block_bytes(s.block_size) + 4 * layout.row_bytes() + layout.zero_row_bytes();
    let weights = std::fs::metadata(dir.join("model.safetensors")).unwrap().len();
    let reserved = r.estimate(&InstanceKey::new("local/inkling2", "default")).vram;
    assert!(reserved > weights + pool, "the estimate holds the weights ({weights}), the pool ({pool}) and the step buffers: {reserved}");
    std::fs::remove_dir_all(&dir).ok();
}
