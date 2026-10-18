// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End to end on the real checkpoint, from the golden's own input images: the
//! processor, the tower, the splice and the paged decoder reproduce the
//! reference's next-token logits after the prompt and its whole greedy
//! continuation, for a sliced image at 16x and 4x, a document, and two images
//! in one prompt.

mod common;

use minicpmv::prompt::{Message, Part};
use minicpmv::{Downsample, GenOptions, MiniCpmV};
use model::serve::PagedDecoder;

#[test]
fn greedy_generation_matches_the_reference_for_every_image_case() {
    let Some(fx) = common::fixture() else { return };
    let manifest = fx.manifest();
    let gpu = gpu_core::testgpu::dev(qwen35::model::pipelines());
    let mut m = MiniCpmV::load(&fx.dir, &gpu, 2048, 1, model::kv_tier::KvTier::F32).expect("load MiniCPM-V");
    let mut table = brain_testutil::parity::Table::new(0.9999, 2e-3);
    for case in ["room_16x", "room_4x", "doc_16x", "two_images"] {
        let entry = &manifest["model"][case];
        let golden = fx.tensors(&format!("model/{case}.safetensors"));
        let names: Vec<&str> = entry["images"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let images: Vec<imaging::Rgb8> = names.iter().map(|n| imaging::codec::load(fx.golden.join(format!("inputs/{n}.png"))).unwrap()).collect();
        let mut parts: Vec<Part> = names.iter().map(|_| Part::Image).collect();
        parts.push(Part::Text(entry["question"].as_str().unwrap().into()));
        let want = common::ids(&golden["greedy"].1);
        let opts = GenOptions {
            max_new: want.len(),
            downsample: entry["options"].get("downsample_mode").and_then(|v| v.as_str()).map(|s| Downsample::parse(s).unwrap()),
            ..GenOptions::default()
        };
        let prepared = m.prepare(&[Message::user(parts)], &images, &opts).expect("prepare");
        assert_eq!(prepared.prompt.ids, common::ids(&golden["input_ids"].1), "{case}: prompt ids");

        // The next-token logits after the prompt: the reference's last row.
        let mut tbl = model::paged::BlockTable::new();
        let hidden = m.engine.prefill_embeds(&mut tbl, &prepared.embeds);
        let logits = m.engine.logits(&hidden);
        m.engine.release_table(&mut tbl);
        let rows = golden["logits_last"].0[0];
        let v = logits.len();
        table.check(&format!("{case} logits"), &logits, &golden["logits_last"].1[(rows - 1) * v..]);

        let got = m.generate_prepared(&prepared, &opts, &mut |_| true).expect("generate");
        // The reference stops at its end-of-turn token; ours excludes it.
        let want_text: Vec<u32> = want.iter().copied().take_while(|t| !m.eos.contains(t)).collect();
        assert_eq!(got, want_text, "{case}: greedy continuation\n got: {:?}\nwant: {:?}", m.decode(&got), m.decode(&want_text));
    }
    table.print();
    assert!(table.failures.is_empty(), "{:#?}", table.failures);
}
