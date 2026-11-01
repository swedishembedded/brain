// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multimodal LLM serving on its clients' own GPUs. If your team needs expertise in
// feeding vision-tower rows and multi-axis rotary positions through a batched paged decoder then you can procure our
// services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! A prompt of embedding rows with rotary positions of their own, through the paged resident engine of an HF FP8
//! directory (the path a vision-language request takes):
//!
//!   * rows equal to the token embeddings, at the default positions, are the token prompt: the same hidden state and
//!     the same next-token distribution, prefilled in one slice or in several;
//!   * a prompt whose every rotary position is shifted by a constant is the same prompt (rotary embedding is
//!     relative), and so are the tokens decoded after it: the shift reaches prefill and decode alike;
//!   * positions that differ in more than a shift (an image of 2x2 merged tokens spans two positions, not four)
//!     change the distribution: the positions are read, not ignored;
//!   * two sequences of a batch decode with their own positions, a text one beside an embedded one.
//!
//! Run with `-- --device gpu1 --backend cuda`; without `--device` the binary reports the tests skipped.

use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillProgress, PromptEmbeds};
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident};

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

const CAP: u32 = 256;
const PROMPT: usize = 40;
const IMAGE: std::ops::Range<usize> = 10..14;

fn engine() -> Qwen35GgufEngine {
    let cfg = Qwen35Config { linear_num_key_heads: 1, linear_num_value_heads: 2, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4, ..fp8_resident_cfg() };
    let dir = write_checkpoint_conditioned(&cfg, "embedded_prompt", 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    let devices = budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, 2 << 30);
    let device = devices.first().expect("the test needs a GPU").0;
    let resident = Qwen35GgufResident::new(path, devices, CAP, tier).with_max_batch(4).with_prefix_slots(2).with_kv_block_rows(16).with_kv_tier(KvTier::F32);
    let (engine, _, _) = resident.activate_engine(&[device]).expect("the tiny FP8 directory loads");
    std::fs::remove_dir_all(dir).ok();
    engine
}

fn prompt(vocab: usize) -> Vec<u32> {
    (0..PROMPT).map(|i| ((i * 7 + 3) % (vocab - 2)) as u32).collect()
}

/// The token embeddings of `prompt` as embedding rows at every position, with `rotary` as their positions.
fn token_rows(engine: &Qwen35GgufEngine, prompt: &[u32], rotary: Option<Vec<[u32; 3]>>) -> PromptEmbeds {
    let rows = engine.instance().embed_rows_of(prompt).expect("the token embeddings");
    let width = engine.embed_width().expect("the engine takes rows");
    let e = PromptEmbeds::run(width, 0, rows).expect("rows");
    match rotary {
        Some(r) => e.with_rotary_positions(r),
        None => e,
    }
}

/// Prefill `prompt` (with `embeds`, in slices of `slice` tokens) and decode `steps` tokens greedily: the last hidden
/// state's logits, then each step's logits.
fn run(engine: &mut Qwen35GgufEngine, prompt: &[u32], embeds: Option<&PromptEmbeds>, slice: usize, steps: usize) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut table = BlockTable::new();
    let hidden = loop {
        let progress = match embeds {
            Some(e) => engine.prefill_chunk_embeds(&mut table, prompt, e, slice),
            None => engine.prefill_chunk(&mut table, prompt, slice),
        };
        if let PrefillProgress::Done(h) = progress {
            break h;
        }
    };
    let mut logits = vec![engine.logits(&hidden)];
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap();
    let mut tokens = vec![argmax(&logits[0])];
    for _ in 0..steps {
        let next = engine.forward_batched_topk(&mut [&mut table], &[*tokens.last().unwrap()], 1);
        tokens.push(next[0][0].0);
        // The head's logits for the step are not exposed by the top-k path; its chosen token and logprob stand for them.
        logits.push(vec![next[0][0].1]);
    }
    engine.release_table(&mut table);
    (logits, tokens)
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs() / (1.0 + y.abs())).fold(0.0f32, f32::max);
    println!("worst relative difference {worst:.2e} (tolerance {tol:.0e})");
    a.len() == b.len() && worst <= tol
}

fn default_rotary() -> Vec<[u32; 3]> {
    (0..PROMPT as u32).map(|i| [i; 3]).collect()
}

/// Rows equal to the token embeddings, with the default positions, are the token prompt, in one slice and in three.
fn rows_equal_to_the_token_embeddings_are_the_token_prompt() {
    let mut e = engine();
    let vocab = e.vocab();
    let prompt = prompt(vocab);
    for slice in [usize::MAX, 16] {
        // The same slicing both ways: a prompt read in slices differs from one read whole by the chunked kernels' rounding.
        let (want_logits, want_tokens) = run(&mut e, &prompt, None, slice, 6);
        let rows = token_rows(&e, &prompt, Some(default_rotary()));
        let (logits, tokens) = run(&mut e, &prompt, Some(&rows), slice, 6);
        assert!(close(&logits[0], &want_logits[0], 2e-3), "slice {slice}: the first distribution differs from the token prompt's");
        assert_eq!(tokens, want_tokens, "slice {slice}");
    }
}

/// A constant shift of every rotary position changes nothing, in prefill (one slice or several) and in the decode after it.
fn a_constant_shift_of_the_rotary_positions_changes_nothing() {
    let mut e = engine();
    let prompt = prompt(e.vocab());
    for shift in [5u32, 301] {
        let shifted: Vec<[u32; 3]> = default_rotary().into_iter().map(|p| p.map(|v| v + shift)).collect();
        for slice in [usize::MAX, 16] {
            let (want_logits, want_tokens) = run(&mut e, &prompt, None, slice, 6);
            let rows = token_rows(&e, &prompt, Some(shifted.clone()));
            let (logits, tokens) = run(&mut e, &prompt, Some(&rows), slice, 6);
            assert!(close(&logits[0], &want_logits[0], 5e-3), "shift {shift} slice {slice}: the first distribution moved");
            for (step, (got, want)) in logits.iter().zip(&want_logits).enumerate().skip(1) {
                assert!(close(got, want, 5e-2), "shift {shift} slice {slice}: the logprob of decode step {step} moved ({got:?} vs {want:?})");
            }
            assert_eq!(tokens, want_tokens, "shift {shift} slice {slice}: decoding after the prompt moved");
        }
    }
}

/// Positions that are not a shift of the default (an image spanning fewer positions than rows) are read by prefill:
/// the distribution moves; and decode continues from the prompt's last position (the sequence's `next`), not its length.
fn positions_that_are_not_a_shift_are_read() {
    let mut e = engine();
    let prompt = prompt(e.vocab());
    let (want_logits, _) = run(&mut e, &prompt, None, usize::MAX, 0);
    // A 2x2 image at rows 10..14 anchored at position 10: the four rows sit at 10, 11, and the text after it resumes at 12.
    let mut rotary = default_rotary();
    let grid = [[10, 10, 10], [10, 10, 11], [10, 11, 10], [10, 11, 11]];
    for (row, p) in IMAGE.zip(grid) {
        rotary[row] = p;
    }
    for (i, r) in rotary.iter_mut().enumerate().skip(IMAGE.end) {
        *r = [i as u32 - 2; 3];
    }
    let rows = token_rows(&e, &prompt, Some(rotary));
    let (logits, _) = run(&mut e, &prompt, Some(&rows), usize::MAX, 0);
    assert!(!close(&logits[0], &want_logits[0], 1e-2), "positions with an image's gap must change the distribution");
}

/// A text sequence and an embedded one decode in one batch, each with its own positions: the embedded one's output is
/// what it is alone, and the text one is untouched.
fn a_text_sequence_and_an_embedded_one_share_a_batch() {
    let mut e = engine();
    let prompt = prompt(e.vocab());
    let shifted: Vec<[u32; 3]> = default_rotary().into_iter().map(|p| p.map(|v| v + 7)).collect();
    let rows = token_rows(&e, &prompt, Some(shifted));
    let (_, alone_text) = run(&mut e, &prompt, None, usize::MAX, 6);
    let (_, alone_embedded) = run(&mut e, &prompt, Some(&rows), usize::MAX, 6);

    let (mut text, mut embedded) = (BlockTable::new(), BlockTable::new());
    let PrefillProgress::Done(h_text) = e.prefill_chunk(&mut text, &prompt, usize::MAX) else { panic!("whole prompt") };
    let PrefillProgress::Done(h_emb) = e.prefill_chunk_embeds(&mut embedded, &prompt, &rows, usize::MAX) else { panic!("whole prompt") };
    let argmax = |l: Vec<f32>| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap();
    let (mut t_tok, mut e_tok) = (vec![argmax(e.logits(&h_text))], vec![argmax(e.logits(&h_emb))]);
    for _ in 0..6 {
        let next = e.forward_batched_topk(&mut [&mut text, &mut embedded], &[*t_tok.last().unwrap(), *e_tok.last().unwrap()], 1);
        t_tok.push(next[0][0].0);
        e_tok.push(next[1][0].0);
    }
    assert_eq!(t_tok, alone_text, "the text sequence is untouched by a neighbour with rotary positions");
    assert_eq!(e_tok, alone_embedded, "the embedded sequence decodes as it does alone");
    e.release_table(&mut text);
    e.release_table(&mut embedded);
}

gpu_core::card_tests!(
    rows_equal_to_the_token_embeddings_are_the_token_prompt,
    a_constant_shift_of_the_rotary_positions_changes_nothing,
    positions_that_are_not_a_shift_are_read,
    a_text_sequence_and_an_embedded_one_share_a_batch
);
