// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image input for the vision-language models its clients serve. If your team needs
// expertise in proving that a cached engine reads pictures exactly as its reference forward does then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec of the engine's image rows (`PagedDecoder::prefill_chunk_embeds`):
//!
//! - on the tiny reference checkpoint, a prompt whose placeholder positions carry embedding rows is served with the
//!   logits of the uncached [`Inkling2::forward_embedded`] over the same rows at every position, whether the prompt is
//!   prefilled whole or in chunks that cut through a run of rows, and a cached prompt with other rows is not reused;
//! - on the released weights and the released tower, one image prompt (the reference processor's own pixels) is
//!   greedy-decoded through the cache to the uncached forward's greedy tokens.

use std::path::PathBuf;

use data::rng::Lcg;
use imaging::RegionImage;
use inkling2::serve::{Engine, EngineOptions};
use inkling2::vision::{expand_placeholders, splice_embeds, ImageProcessorSpec, VisionTower, VisionWeights};
use inkling2::Inkling2;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::visit_state::VisitPoolSizing;
use model::serve::{PagedDecoder, PrefillProgress, PromptEmbeds};

const REAL: &str = "thinkingmachines/Inkling-2-Mini";
/// The f32 reference's own spread between its cached decode and its uncached forward, relative to the tensor's scale
/// (as in the engine's text gates).
const TOL: f32 = 5e-5;

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

/// Prefill `prompt` with its rows in slices of `slice`, then decode `rest` teacher-forced: the logits after the prompt
/// and after each decoded token.
fn served(e: &mut Engine, table: &mut BlockTable, prompt: &[u32], embeds: &PromptEmbeds, slice: usize, rest: &[u32]) -> Vec<Vec<f32>> {
    let hidden = loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk_embeds(table, prompt, embeds, slice) {
            break h;
        }
    };
    let mut out = vec![e.logits(&hidden)];
    for &t in rest {
        out.push(e.forward_batched_logit_rows(&mut [table], &[t]).remove(0));
    }
    out
}

#[test]
fn rows_at_placeholder_positions_are_served_as_the_uncached_forward_reads_them() {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    if !dir.join("model.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let sizing = VisitPoolSizing { block_size: 8, num_blocks: 64, live_rows: 4, snapshot_rows: 4, undo_rows: false };
    let opts = EngineOptions { kv_tier: KvTier::F32, sizing, max_seq_len: 128, prefill_chunk: 5 };
    let mut e = Engine::open(crate::tiny_parity::ops(), &dir, opts).unwrap();
    let (ph, width) = (9u32, e.model().config().hidden as usize);
    // Two runs of rows (an image of three regions, one of four) among text, longer than a cached prompt needs.
    let seq: Vec<u32> = (0..44u32).map(|i| if (5..8).contains(&i) || (20..24).contains(&i) { ph } else { (i * 13 + 2) % 60 + 10 }).collect();
    let prompt_len = 36;
    let positions: Vec<u32> = seq.iter().enumerate().filter(|(_, &t)| t == ph).map(|(i, _)| i as u32).collect();
    // seed: any rows of the embedding's scale; the property is the splice, not their meaning.
    let rows = Lcg::new(17).vec_scaled(positions.len() * width, 1.0);
    let embeds = PromptEmbeds::new(width, positions.clone(), rows).unwrap();
    let whole = e.model().forward_embedded(&seq, &embeds).unwrap();
    let v = e.vocab();
    for slice in [prompt_len, 6, 1] {
        let mut table = BlockTable::new();
        let got = served(&mut e, &mut table, &seq[..prompt_len], &embeds, slice, &seq[prompt_len..seq.len() - 1]);
        for (i, row) in got.iter().enumerate() {
            let p = prompt_len - 1 + i;
            let err = rel_err(row, &whole[p * v..(p + 1) * v]);
            assert!(err <= TOL, "slices of {slice}: position {p} is {err:.3e} from the uncached forward over the same rows");
        }
        e.release_table(&mut table);
    }
    // The same tokens with other rows are another prompt: the cached entry of the first is not reused for it.
    let other = PromptEmbeds::new(width, positions, Lcg::new(18).vec_scaled(embeds.rows().len(), 1.0)).unwrap();
    assert_eq!(e.cached_prefix_blocks_embedded(&seq[..prompt_len], None, &other), 0, "rows are part of the prefix key");
    assert!(e.cached_prefix_blocks_embedded(&seq[..prompt_len], None, &embeds) > 0, "the same rows reuse the cached prompt");
    let mut table = BlockTable::new();
    let got = served(&mut e, &mut table, &seq[..prompt_len], &other, prompt_len, &[]);
    let want = e.model().forward_embedded(&seq[..prompt_len], &other).unwrap();
    let err = rel_err(&got[0], &want[(prompt_len - 1) * v..]);
    assert!(err <= TOL, "other rows are served from their own forward: {err:.3e}");
    e.release_table(&mut table);
    let plain = e.model().forward(&seq).unwrap();
    assert!(rel_err(&plain[(prompt_len - 1) * v..prompt_len * v], &whole[(prompt_len - 1) * v..prompt_len * v]) > 1e-3, "the rows change the logits");
}

/// The released weights and tower: the reference processor's pixels of the real vision golden's first image (a 1x2
/// grid of regions), the checkpoint's chat text around it, greedy-decoded through the cache; every served token is the
/// argmax of the uncached forward over the prompt, the tower's rows and the tokens served before it. About 27 GB
/// resident.
#[test]
fn an_image_prompt_on_the_released_weights_decodes_to_the_uncached_forwards_greedy_tokens() {
    let Some(dir) = brain_testutil::model_dir(REAL).map(PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REAL} is not in the model store"));
        return;
    };
    let gold_dir = brain_testutil::testdata_path("golden/inkling2/vision/real");
    if !gold_dir.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/vision/real absent (make testdata/fetch, or run tools/goldens/inkling2_vision_dump.py real)");
        return;
    }
    let gold: std::collections::HashMap<String, Vec<f32>> = checkpoint::safetensors::read(&gold_dir.join("golden.safetensors").to_string_lossy()).unwrap().into_iter().map(|t| (t.name, t.data)).collect();
    let grid = &gold["image_grid_thw"][..3];
    let (rows, cols) = (grid[1] as u32, grid[2] as u32);
    let regions = (rows * cols) as usize;
    let image = RegionImage { rows, cols, pixels: gold["pixel_values"][..regions * 40 * 40 * 3].to_vec() };

    let cfg = inkling2::Inkling2Config::from_dir(&dir).unwrap();
    let vision = cfg.vision.clone().expect("the release has a vision tower");
    let spec = ImageProcessorSpec::from_json(&std::fs::read_to_string(dir.join("processor_config.json")).unwrap()).unwrap();
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let ph = tok.special_id(&spec.image_token).expect("the placeholder is one token");
    // The checkpoint's template for one user message holding the image and a question, at the default effort.
    let text = format!(
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_image|>{}<|end_message|><|message_user|><|content_text|>Describe this picture in one word.<|end_message|>",
        spec.image_token
    );
    let ids = data::tokenizer::Tokenizer::encode(&tok, &text);
    let prompt = expand_placeholders(&ids, ph, &[regions]).unwrap();

    let sizing = VisitPoolSizing { block_size: 16, num_blocks: 8, live_rows: 1, snapshot_rows: 0, undo_rows: false };
    let mut e = Engine::open(crate::tiny_parity::ops(), &dir, EngineOptions { kv_tier: KvTier::F32, sizing, max_seq_len: 128, prefill_chunk: 16 }).unwrap();
    let tower_gpu = e.model().ops().gpu().new_like(inkling2::vision::pipelines());
    let tower = VisionTower::new(&tower_gpu, &vision, &VisionWeights::open(&dir, &vision).unwrap()).unwrap();
    let embeds = splice_embeds(&prompt, ph, &tower.encode(&tower_gpu, &[image]).unwrap()).unwrap();
    drop(tower);

    const NEW: usize = 6;
    let mut table = BlockTable::new();
    let hidden = loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk_embeds(&mut table, &prompt, &embeds, usize::MAX) {
            break h;
        }
    };
    let mut out = vec![e.admit_greedy(&hidden)];
    while out.len() < NEW {
        let next = e.forward_batched_greedy(&mut [&mut table], &[*out.last().unwrap()]);
        out.push(next[0]);
    }
    e.release_table(&mut table);
    let seq: Vec<u32> = prompt.iter().chain(&out[..NEW - 1]).copied().collect();
    let model: &Inkling2 = e.model();
    let logits = model.forward_embedded(&seq, &embeds).unwrap();
    let v = e.vocab();
    let want: Vec<u32> = (0..NEW).map(|i| argmax(&logits[(prompt.len() - 1 + i) * v..(prompt.len() + i) * v])).collect();
    eprintln!("image prompt: {} tokens ({regions} image rows), served {out:?}: {:?}", prompt.len(), data::tokenizer::Tokenizer::decode(&tok, &out));
    assert_eq!(out, want, "cached greedy tokens of the image prompt against the uncached forward's");
}
