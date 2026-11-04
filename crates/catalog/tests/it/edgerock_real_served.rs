// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context language
// models at a fixed memory budget for its clients. If your team needs
// expertise in bounded-state inference and continuous batching then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The released EdgeRock weights, served: the model the store holds (`brain pull rockmatrix-ai/EdgeRock`) is
//! resolved through the catalog, built behind the scheduler at the bf16 weight tier, and asked the question of
//! `tools/goldens/edgerock_dump_real_reference.py`: a fact stated at the start of a prompt longer than the
//! attention window, then "what is the vault code?". Greedy decoding must answer 4817.
//!
//! Run with `BRAIN_DEVICE=cpu` for the host tier (about ten minutes); on a card it needs the free memory of the
//! weights and the slots (`EdgeRockResident::estimate`).

use capability::Invocation;

const FILLER: &str = "The weather report for the coastal towns repeats every hour: light wind from the west, scattered clouds, a calm sea and visibility above ten kilometres. ";

#[test]
#[ignore = "slow: the released 9 GB checkpoint and a prompt of about 1300 tokens"]
fn the_released_model_recalls_the_vault_code_through_the_catalog() {
    let Some(dir) = brain_testutil::model_dir("rockmatrix-ai/EdgeRock").filter(|d| std::path::Path::new(d).join("config.json").exists()) else {
        brain_testutil::skip("rockmatrix-ai/EdgeRock is not in the model store - brain pull rockmatrix-ai/EdgeRock");
        return;
    };
    // The roles the resolver assigns the pulled directory (`qwen35::spec::EdgeRockSpec`), without scanning the
    // store: a scan writes its inventory cache into the store.
    let assembly = capability::Assembly {
        id: "rockmatrix-ai/EdgeRock".to_string(),
        arch: "edgerock".to_string(),
        variant: None,
        roles: [("weights", dir.clone()), ("tokenizer", format!("{dir}/tokenizer.json"))].into_iter().map(|(k, v)| (k.to_string(), std::path::PathBuf::from(v))).collect(),
        provenance: Vec::new(),
    };
    let provider = catalog::provider_from_assembly(catalog::resident_edgerock::MODEL, &assembly).expect("EdgeRock is in the catalog");
    let prompt = format!("Remember this: the vault code is 4817 and the courier's name is Ilse Varga.\n{}\nQuestion: what is the vault code? Answer: the vault code is", FILLER.repeat(40));
    let inv = Invocation::new().set("prompt", serde_json::json!(prompt)).set("chat", serde_json::json!(false)).set("max_new", serde_json::json!(8)).set("temp", serde_json::json!(0.0));
    let outcome = provider.action("generate").expect("a generate action").run(&inv, &mut |_| {}).expect("the released model answers");
    let text = outcome.outputs.get("text").and_then(|t| t.as_str()).expect("text output");
    assert!(text.contains("4817"), "the greedy answer was {text:?}");
}
