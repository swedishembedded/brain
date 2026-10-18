// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served `generate` action on the real checkpoint, through the
//! capability contract the HTTP and D-Bus surfaces use: image blobs, OpenAI
//! style messages with `image_url` parts, a batch decoded together, and a
//! cancelled request that stops alone.

mod common;

use capability::{CancelToken, Invocation};
use serde_json::json;

fn image_inv(fx: &common::Fixture, name: &str, key: &str, inv: Invocation) -> Invocation {
    let img = imaging::codec::load(fx.golden.join(format!("inputs/{name}.png"))).unwrap();
    let hwc: Vec<f32> = img.px.iter().map(|&b| b as f32 / 255.0).collect();
    inv.blob(key, capability::blob::image_blob(&hwc, img.w, img.h, 3))
}

#[test]
fn a_served_batch_reproduces_the_reference_and_a_cancel_stops_alone() {
    let Some(fx) = common::fixture() else { return };
    let manifest = fx.manifest();
    let mut session = minicpmv::caps::load_session(fx.dir.to_str().unwrap(), None).expect("load the served session");

    // The document case as OpenAI messages with the image part where it stands.
    let doc = &manifest["model"]["doc_16x"];
    let doc_want = fx.tensors("model/doc_16x.safetensors");
    let messages = json!([{"role": "user", "content": [{"type": "image_url"}, {"type": "text", "text": doc["question"]}]}]);
    let a = image_inv(&fx, "doc", "image", Invocation::new().set("messages", json!(messages.to_string())).set("temp", json!(0.0)).set("max_new", json!(24)));
    // The sliced-image case as a bare prompt with an attached image.
    let room = &manifest["model"]["room_16x"];
    let room_want = fx.tensors("model/room_16x.safetensors");
    let b = image_inv(&fx, "room", "image", Invocation::new().set("prompt", json!(room["question"])).set("temp", json!(0.0)).set("max_new", json!(24)));
    // A third request cancelled before it starts.
    let gone = CancelToken::armed();
    gone.cancel();
    let mut c = Invocation::new().set("prompt", json!("Count to ten.")).set("temp", json!(0.0));
    c.cancel = gone;

    let mut streamed = vec![String::new(); 3];
    let results = session.generate_batch(&[a, b, c], &mut |i, p| {
        if let Some(d) = p.delta {
            streamed[i].push_str(&d);
        }
    });
    let text = |o: &capability::Outcome| o.outputs["text"].as_str().unwrap().to_string();
    let want = |t: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>| {
        let ids: Vec<u32> = common::ids(&t["greedy"].1).into_iter().filter(|&i| i != 248046 && i != 248044).collect();
        session.model.decode(&ids)
    };
    let (ra, rb) = (results[0].as_ref().expect("doc request"), results[1].as_ref().expect("room request"));
    assert_eq!(text(ra), want(&doc_want).trim_end(), "doc: served text");
    assert!(want(&room_want).starts_with(&text(rb)), "room: served text {:?} is not the reference's prefix", text(rb));
    assert_eq!(streamed[0], text(ra), "the stream adds up to the outcome");
    assert_eq!(ra.outputs["prompt_tokens"], json!(doc_want["input_ids"].1.len()));
    assert_eq!(results[2].as_ref().unwrap_err(), "cancelled");
    assert!(streamed[2].is_empty(), "a cancelled request streams nothing");
}
