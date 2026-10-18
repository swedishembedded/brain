// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decoder alone, on the real checkpoint: `brain-qwen35` built from the
//! MiniCPM-V directory (vision tensors declared foreign) reproduces the
//! reference's logits on a text-only prompt, and its incremental decode
//! reproduces the reference's greedy continuation.

mod common;

use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use qwen35::model::{pipelines, Qwen35};

#[test]
fn text_only_logits_and_greedy_decode_match_the_reference() {
    let Some(fx) = common::fixture() else { return };
    let golden = fx.tensors("model/text.safetensors");
    let ids = common::ids(&golden["input_ids"].1);
    let greedy = common::ids(&golden["greedy"].1);
    let want_logits = &golden["logits_last"].1;
    let rows = golden["logits_last"].0[0];

    let t = (ids.len() + greedy.len()) as u32;
    let text = qwen35::config::Qwen35Config { block_size: t, ..fx.cfg.text.clone() };
    let src = minicpmv::import::open_decoder(&fx.dir, &fx.cfg).expect("open the decoder");
    let gpu = gpu_core::testgpu::dev(pipelines());
    let m = Qwen35::new_on_dt(gpu, text.clone(), 1, t, &src, &TierPolicy::uniform(Dtype::F32));

    let v = text.vocab as usize;
    let logits = m.logits_all(&ids);
    let got = &logits[(ids.len() - rows) * v..ids.len() * v];
    let mut table = brain_testutil::parity::Table::new(0.9999, 1e-3);
    table.check("logits (last rows)", got, want_logits);
    table.print();
    assert!(table.failures.is_empty(), "{:#?}", table.failures);

    // Incremental decode: chunked prefill, then one token at a time, the head
    // applied on the host - the reference's greedy continuation, token for
    // token, through its end-of-turn token.
    let head = m.read_weight(text.head_weight());
    m.reset_decode_cache();
    let mut hidden = m.prefill_chunked(&ids, 16);
    let mut out = Vec::new();
    for _ in 0..greedy.len() {
        let tok = common::argmax(&common::head_logits(&head, &hidden));
        out.push(tok);
        if out.len() == greedy.len() {
            break;
        }
        hidden = m.step(tok);
    }
    assert_eq!(out, greedy, "greedy continuation differs from the reference");
}
