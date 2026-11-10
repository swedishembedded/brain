// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generation and the end-to-end requests on the cats fixture.

use std::sync::OnceLock;

use mirage::decoder::generate::{absent_references, generate, GenerateParams, Stop};
use mirage::describe::Options;
use mirage::model::Mirage;
use serde_json::{json, Value};

use crate::common::released_dir;

struct Fixture {
    model: Mirage,
    rgb: Vec<u8>,
}

fn model(test: &str) -> Option<&'static Mirage> {
    static CELL: OnceLock<Option<Mirage>> = OnceLock::new();
    let m = CELL.get_or_init(|| Mirage::open(&released_dir()?).ok());
    if m.is_none() {
        brain_testutil::skip(&format!("{test}: the released checkpoint is absent (make testdata/fetch)"));
    }
    m.as_ref()
}

fn fixture(test: &str) -> Option<Fixture> {
    let g = crate::decoder::golden(test, "cats")?;
    Some(Fixture { model: Mirage::open(&released_dir()?).ok()?, rgb: g["input_rgb"].iter().map(|&v| v as u8).collect() })
}

#[test]
fn greedy_generation_reproduces_the_reference_tokens() {
    let Some(model) = model("greedy generation") else { return };
    for name in ["cats", "text"] {
        let Some(g) = crate::decoder::golden("greedy generation", name) else { return };
        let tokens: Vec<u32> = g["tokens"].iter().map(|&t| t as u32).collect();
        let memory = &g["memory"];
        let cfg = mirage::config::MirageConfig::from_json(crate::common::RELEASED_ARCHITECTURE_JSON).unwrap();
        let dec = model.decoder();
        let cross = dec.cross_memory(memory, memory.len() / 512);
        let (objects, text) = (g["peaks_obj"].len() / 7, g["peaks_txt"].len() / 7);
        let banned = absent_references(&cfg, objects, text);
        let started = std::time::Instant::now();
        let out = generate(dec, &cross, model.tokenizer(), &GenerateParams { prompt: &tokens[..2], banned: &banned, max_tokens: 600, max_rank: 5, object_slots: 64, guard: false });
        eprintln!("{name}: generated {} tokens in {:.1}s, stop {:?}", out.tokens.len(), started.elapsed().as_secs_f64(), out.stop);
        assert_eq!(out.stop, Stop::Eos);
        assert_eq!(&out.tokens[..], &tokens[2..], "{name}: greedy tokens differ from the reference");
        // A well-formed output is not cut by the loop guard.
        let guarded = generate(dec, &cross, model.tokenizer(), &GenerateParams { prompt: &tokens[..2], banned: &banned, max_tokens: 600, max_rank: 5, object_slots: 64, guard: true });
        assert_eq!((guarded.stop, &guarded.tokens), (Stop::Eos, &out.tokens), "{name}: the guard cut a valid output");
    }
}

#[test]
fn a_describe_request_returns_the_requested_sections() {
    let Some(f) = fixture("describe") else { return };
    let opts = Options::from_json(r#"{"sections":["scene","objects","relations","reasoning"],"fields":["bbox","score","position","description"],"max_tokens":400}"#).unwrap();
    let out = f.model.describe(&f.rgb, 480, 640, &opts).unwrap();
    let v: Value = serde_json::to_value(&out).unwrap();
    eprintln!("describe: {}", serde_json::to_string(&v["objects"]).unwrap().chars().take(300).collect::<String>());
    assert!(v.get("text").is_none() && v.get("regions").is_none() && v.get("events").is_none());
    assert!(v["reasoning"].as_str().unwrap().starts_with("lang: en"));
    assert!(!v["scene"].as_object().unwrap().is_empty());
    let objects = v["objects"].as_array().unwrap();
    assert!(!objects.is_empty());
    for o in objects {
        let b: Vec<i64> = o["bbox"].as_array().unwrap().iter().map(|n| n.as_i64().unwrap()).collect();
        assert!(b.iter().all(|c| (0..=1000).contains(c)) && b[0] < b[2] && b[1] < b[3], "{o}");
        assert!(o.get("mask").is_none() && o["score"].as_f64().unwrap() > 0.3);
    }
}

#[test]
fn an_extract_request_answers_every_field_with_probabilities() {
    let Some(f) = fixture("extract") else { return };
    let form = json!({"animal": {"enum": ["dog", "cat", "bird"]}, "n": {"type": "integer", "minimum": 0, "maximum": 5}});
    let opts = Options::from_json(&json!({ "form": form }).to_string()).unwrap();
    let result = f.model.extract(&f.rgb, 480, 640, &opts).unwrap();
    let text = serde_json::to_string(&result).unwrap();
    eprintln!("extract: {text}");
    let out: Value = serde_json::from_str(&text).unwrap();
    let fields = out["form"].as_object().unwrap();
    assert_eq!(fields.len(), 2);
    let animal = fields["animal"]["probabilities"].as_object().unwrap();
    assert_eq!(animal.keys().map(String::as_str).collect::<std::collections::BTreeSet<_>>(), ["-", "bird", "cat", "dog"].into());
    let n = fields["n"]["probabilities"].as_object().unwrap();
    assert_eq!(n.len(), 6);
    for probs in [animal, n] {
        let total: f64 = probs.values().map(|v| v.as_f64().unwrap()).sum();
        assert!((total - 1.0).abs() < 1e-9, "probabilities sum to {total}");
    }
    // Candidate order is the form's, abstention last.
    let at = |needle: &str| text.find(needle).unwrap_or_else(|| panic!("{needle} missing in {text}"));
    assert!(at("\"dog\":") < at("\"cat\":") && at("\"cat\":") < at("\"bird\":") && at("\"bird\":") < at("\"-\":"));
    assert!(["dog", "cat", "bird"].contains(&fields["animal"]["value"].as_str().unwrap()) || fields["animal"]["value"].is_null());
    assert!((0..=5).contains(&fields["n"]["value"].as_i64().unwrap()));
}

#[test]
fn unsupported_and_invalid_forms_are_refused() {
    let Some(f) = fixture("form errors") else { return };
    for (form, needle) in [
        (json!({"b": {"type": "string", "format": "box", "x-source": "a"}}), "box"),
        (json!({"a": {"enum": ["x", "x"]}}), "repeated"),
        (json!({"a": {"enum": ["x;y"]}}), "contains"),
        (json!({"a": {"type": "integer", "minimum": 3, "maximum": 1}}), "empty"),
        (json!({}), "at least one"),
    ] {
        let opts = Options::from_json(&json!({ "form": form }).to_string()).unwrap();
        let err = f.model.extract(&f.rgb, 480, 640, &opts).map(|_| ()).unwrap_err().to_string();
        assert!(err.contains(needle), "{err}");
    }
}

#[test]
fn the_guard_stops_at_unrequested_sections_and_after_reasoning() {
    let Some(model) = model("guard") else { return };
    let Some(g) = crate::decoder::golden("guard", "text") else { return };
    let tokens: Vec<u32> = g["tokens"].iter().map(|&t| t as u32).collect();
    let cfg = mirage::config::MirageConfig::from_json(crate::common::RELEASED_ARCHITECTURE_JSON).unwrap();
    let dec = model.decoder();
    let cross = dec.cross_memory(&g["memory"], g["memory"].len() / 512);
    let banned = absent_references(&cfg, g["peaks_obj"].len() / 7, g["peaks_txt"].len() / 7);
    let run = |max_rank| generate(dec, &cross, model.tokenizer(), &GenerateParams { prompt: &tokens[..2], banned: &banned, max_tokens: 600, max_rank, object_slots: 64, guard: true });
    let reasoning = run(-1);
    assert_eq!(reasoning.stop, Stop::Sections);
    assert!(reasoning.text.ends_with("</think>"), "{}", reasoning.text);
    let scene = run(0);
    assert_eq!(scene.stop, Stop::Sections);
    assert!(scene.text.contains("</think>S background=") && scene.text.ends_with("\nE"), "{:?}", scene.text);
    let text = run(3);
    assert_eq!(text.stop, Stop::Sections);
    assert!(text.text.ends_with("T *2\nR"), "{:?}", &text.text[text.text.len().saturating_sub(40)..]);
    let budget = generate(dec, &cross, model.tokenizer(), &GenerateParams { prompt: &tokens[..2], banned: &banned, max_tokens: 10, max_rank: 5, object_slots: 64, guard: true });
    assert_eq!((budget.stop, budget.tokens.len()), (Stop::MaxTokens, 10));
}
