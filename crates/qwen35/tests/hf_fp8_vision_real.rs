// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements validation of served vision-language checkpoints against their reference
// implementation for its clients. If your team needs expertise in proving that a served multimodal model reads a
// picture as its reference does then you can procure our services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! The released Qwen3.8-27B-FP8 reading one picture, against the reference
//! (`tools/goldens/qwen35_vl_dump_reference.py`: `transformers`' `Qwen3_5ForConditionalGeneration` on the same weights
//! decoded to bf16, the tower in float32, greedy generation):
//!
//!   * the image's patches (normalization and packing) are the processor's;
//!   * the vision tower's patch features and the merger's rows, from the weights as released;
//!   * the prompt a chat request with the image becomes - its token ids and the three-axis rotary position of every
//!     token - through the front end the server runs;
//!   * the decoder's logits at the end of that prompt and at each token of the continuation, FP8 against the bf16
//!     reference: the argmax is the reference's, and the distributions agree to the distance the FP8 tier keeps from
//!     bf16 on text.
//!
//! The model store must hold `Qwen/Qwen3.8-27B-FP8` (`BRAIN_MODELS_DIR`), `testdata/` the golden and its picture.
//! Run with `-- --device gpu1 --backend cuda`; without `--device` the binary reports the test skipped.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use brain_testutil::parity::Table;
use capability::Invocation;
use llmserve::{Frontend, TextFrontend};
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillProgress};
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, parse_request, Qwen35GgufResident};
use serde_json::json;

const CAP: u32 = 512;
const QUESTION: &str = "Describe the part and list every dimension written on it.";

fn golden() -> Option<HashMap<String, Vec<f32>>> {
    let path = brain_testutil::testdata("golden/qwen35/vl/qwen35_vl.safetensors");
    if !Path::new(&path).exists() {
        brain_testutil::skip(&format!("fixture {path} absent - run tools/goldens/qwen35_vl_dump_reference.py"));
        return None;
    }
    Some(checkpoint::safetensors::read(&path).expect("read the golden").into_iter().map(|t| (t.name, t.data)).collect())
}

fn ints(v: &[f32]) -> Vec<u32> {
    v.iter().map(|&x| x.round() as u32).collect()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap()
}

/// The weights as released (FP8 linears, native kernels): the distance the FP8 tier keeps from the bf16 reference on text
/// (`qwen35_fp8_parity`: mean relative L2 0.034, worst 0.10 over prompts of ~120 tokens), which a picture's prompt shares.
fn fp8_release_reads_a_picture_as_the_reference_does() {
    reads_a_picture(None, 0.985, 0.2);
}

/// The same weights decoded to bf16 (the arithmetic the reference runs): what is left of the distance is the splice and
/// the arithmetic of the engine, not the FP8 tier.
fn bf16_decoded_release_reads_a_picture_as_the_reference_does() {
    reads_a_picture(Some(gpu_core::select::Dtype::BF16), 0.999, 0.06);
}

fn reads_a_picture(tier: Option<gpu_core::select::Dtype>, cosine: f64, rel_l2: f64) {
    let Some(golden) = golden() else { return };
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3.8-27B-FP8").filter(|d| Path::new(d).join("config.json").exists()) else {
        brain_testutil::skip("Qwen/Qwen3.8-27B-FP8 is not in the model store - brain pull Qwen/Qwen3.8-27B-FP8");
        return;
    };
    let image_path = brain_testutil::testdata("qwen35/vl/sketch_288x256.png");
    let image = imaging::codec::load(&image_path).expect("the golden's picture");

    let tier = match tier {
        Some(dtype) => model::ops::TierPolicy::uniform(dtype),
        None => held_tiers_at(&dir).expect("a servable checkpoint"),
    };
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the test needs a GPU").0;
    let resident = Qwen35GgufResident::new(dir.clone(), devices, CAP, tier).with_max_batch(2).with_kv_block_rows(64).with_kv_tier(model::kv_tier::KvTier::BF16);
    let (mut sched, tok, stop) = resident.activate_scheduler(&[device]).expect("the FP8 release loads");
    let text = TextFrontend::new(tok, stop, "qwen35", Arc::new(parse_request));
    let front = resident.vision_front_end(&mut sched, text).expect("the tower loads").unwrap_or_else(|_| panic!("the release has a vision tower"));
    let tower = front.tower();

    // Preprocessing: the checkpoint's own processor settings give the reference's patches.
    let patches = tower.spec().geometry.patches(&image);
    let mut table = Table::new(0.999999, 1e-5);
    table.check("patches", &patches.pixels, &golden["patches"]);
    let (gh, gw) = (patches.h_bar / 16, patches.w_bar / 16);
    assert_eq!([1, gh, gw].map(|v| v as f32), [golden["grid_thw"][0], golden["grid_thw"][1], golden["grid_thw"][2]], "the patch grid");

    // The tower, from the weights as released, against the reference in float32.
    let features = tower.features(gh, gw, &patches.pixels);
    table.check("tower features (pre-merger)", &features, &golden["hidden"]);
    let rows = tower.encode(&image);
    table.check("merger rows", &rows.rows, &golden["merged"]);
    table.print();
    assert!(table.failures.is_empty(), "vision tower parity failures: {:#?}", table.failures);

    // The prompt a chat request becomes, through the front end the server runs.
    let messages = json!([{"role": "user", "content": [{"type": "image_url", "image_url": {}}, {"type": "text", "text": QUESTION}]}]);
    let mut inv = Invocation::new().set("messages", json!(messages.to_string())).set("enable_thinking", json!(false)).set("max_new", json!(4));
    inv = inv.blob("image", capability::blob::image_blob(&image.to_hwc_unit(), image.w, image.h, 3));
    let (req, embeds) = front.parse_embedded(&inv).expect("the request builds");
    let embeds = embeds.expect("an image request carries rows");
    assert_eq!(req.ids, ints(&golden["input_ids"]), "the prompt's token ids are the reference template's");
    let l = req.ids.len();
    let want_rope: Vec<[u32; 3]> = (0..l).map(|i| [0, 1, 2].map(|a| golden["position_ids"][a * l + i].round() as u32)).collect();
    assert_eq!(embeds.rotary_positions(), Some(&want_rope[..]), "the three-axis rotary position of every prompt token");
    assert_eq!(embeds.len(), golden["merged"].len() / 5120);

    // The decoder: the logits at the end of the prompt, and at each token of the reference's continuation.
    let engine = sched.decoder_mut();
    let mut blocks = BlockTable::new();
    let hidden = match engine.prefill_chunk_embeds(&mut blocks, &req.ids, &embeds, usize::MAX) {
        PrefillProgress::Done(h) => h,
        PrefillProgress::Partial { .. } => panic!("the whole prompt in one slice"),
    };
    let new_tokens = ints(&golden["new_tokens"]);
    let mut table = Table::new(cosine, rel_l2);
    let first = engine.logits(&hidden);
    table.check("logits at the end of the prompt", &first, &golden["logits0"]);
    assert_eq!(argmax(&first), new_tokens[0], "the first generated token is the reference's");
    // Each later step feeds the reference's token and scores the engine's best candidate against the reference's
    // logits: the same token, and a logit within what the FP8 tier moves it.
    for step in 1..new_tokens.len() {
        let cands = engine.forward_batched_topk(&mut [&mut blocks], &[new_tokens[step - 1]], 8).remove(0);
        assert_eq!(cands[0].0, new_tokens[step], "step {step}: the engine's best token is the reference's");
        // The candidates carry the head's logits.
        let reference = golden[&format!("logits{step}")][cands[0].0 as usize];
        println!("step {step}: logit {:.3} against the reference's {reference:.3}", cands[0].1);
        assert!((cands[0].1 - reference).abs() < 0.5 + 0.05 * reference.abs(), "step {step}: logit {} against the reference's {reference}", cands[0].1);
    }
    table.print();
    assert!(table.failures.is_empty(), "decoder parity failures: {:#?}", table.failures);
    engine.release_table(&mut blocks);
}

gpu_core::card_tests!(fp8_release_reads_a_picture_as_the_reference_does, bf16_decoded_release_reads_a_picture_as_the_reference_does);
