// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching serving of looped mixture-of-experts models for its clients. If
// your team needs expertise in proving a cached engine against its reference forward then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec of the served engine (`inkling2::serve::Engine`):
//!
//! - on the tiny reference checkpoint, a cached prefill leaves every visit's four convolution states as the
//!   reference's cached prefill does (`prefill.visit{v}.conv{j}`), and each token-by-token decode step serves the
//!   reference's cached decode logits (`decode{s}.logits`);
//! - its logits equal the uncached [`Inkling2::forward`] at every position, whether the prompt is prefilled whole, in
//!   chunks or a token at a time, alone or batched beside another sequence;
//! - it passes the generic serving conformance suite;
//! - a sequence served with routing recorded carries, for every computed position and every one of the plan's MoE
//!   calls, the experts the uncached forward's router picks there;
//! - on the released weights, cached greedy decoding of the reference prompts is the uncached forward's greedy tokens.

use std::collections::HashMap;
use std::path::PathBuf;

use data::rng::Rng;
use inkling2::serve::{Engine, EngineOptions};
use inkling2::{Inkling2, Probe};
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::rollout::{Rollout, RolloutParams, SchedulerRollout};
use model::serve::conformance::{self, Conformance};
use model::serve::visit_state::VisitPoolSizing;
use model::serve::{PagedDecoder, PrefillProgress, SampleParams};

const REAL: &str = "thinkingmachines/Inkling-2-Mini";
/// The f32 reference's own spread between its cached decode and its uncached forward (`decode_vs_full_max_abs`, at
/// most 1.7e-5 on these logits), relative to the tensor's scale, as in the forward's parity gate.
const TOL: f32 = 5e-5;

fn tiny_dir() -> Option<PathBuf> {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if dir.join("golden.safetensors").is_file() {
        Some(dir)
    } else {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        None
    }
}

fn golden(dir: &std::path::Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    checkpoint::safetensors::read(&dir.join("golden.safetensors").to_string_lossy()).expect("read golden").into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale
}

fn tiny_engine(chunk: u32) -> Engine {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    let sizing = VisitPoolSizing { block_size: 8, num_blocks: 64, live_rows: 4, snapshot_rows: 4 };
    Engine::open(crate::tiny_parity::ops(), &dir, EngineOptions { kv_tier: KvTier::F32, sizing, max_seq_len: 128, prefill_chunk: chunk }).expect("the tiny engine")
}

/// Prefill `prompt` in slices of `slice`, then decode `rest` teacher-forced: the logits after the prompt and after each
/// decoded token.
fn served(e: &mut Engine, table: &mut BlockTable, prompt: &[u32], slice: usize, rest: &[u32]) -> Vec<Vec<f32>> {
    let hidden = loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk(table, prompt, slice) {
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
fn the_cached_engine_reproduces_the_references_prefill_state_and_decode() {
    let Some(dir) = tiny_dir() else { return };
    let gold = golden(&dir);
    let tokens: Vec<u32> = gold["tokens"].1.iter().map(|&v| v as u32).collect();
    let steps: Vec<u32> = gold["decode.tokens"].1.iter().map(|&v| v as u32).collect();
    for slice in [tokens.len(), 3] {
        let mut e = tiny_engine(4);
        let mut table = BlockTable::new();
        let got = served(&mut e, &mut table, &tokens, slice, &[]);
        let v = e.vocab();
        let prefill = &gold["prefill.logits"].1;
        let e0 = rel_err(&got[0], &prefill[(tokens.len() - 1) * v..]);
        assert!(e0 <= TOL, "slices of {slice}: the prefill's last logits are {e0:.3e} from the reference");
        let g = e.model().ops().gpu();
        for visit in e.model().plan().visits() {
            for j in 0..4 {
                let (shape, want) = &gold[&format!("prefill.visit{}.conv{j}", visit.index)];
                let (c, k) = (shape[0], shape[1]);
                // The reference keeps the last `taps` inputs; the next step reads the last `taps - 1`.
                let tail: Vec<f32> = (0..c).flat_map(|ch| want[ch * k + 1..(ch + 1) * k].iter().copied()).collect();
                let state = g.read(&e.pool().conv_rows(g, visit.index, j, &[&table]), c * (k - 1));
                let err = rel_err(&state, &tail);
                assert!(err <= TOL, "slices of {slice}: visit {} conv state {j} is {err:.3e} from the reference's", visit.index);
            }
        }
        for (s, &t) in steps.iter().enumerate() {
            let row = e.forward_batched_logit_rows(&mut [&mut table], &[t]).remove(0);
            let err = rel_err(&row, &gold[&format!("decode{s}.logits")].1);
            assert!(err <= TOL, "slices of {slice}: decode step {s} is {err:.3e} from the reference's cached decode");
        }
        e.release_table(&mut table);
    }
}

#[test]
fn served_logits_equal_the_uncached_forward_at_every_position() {
    let Some(dir) = tiny_dir() else { return };
    let gold = golden(&dir);
    let seq: Vec<u32> = gold["tokens"].1.iter().chain(&gold["decode.tokens"].1).map(|&v| v as u32).collect();
    let (prompt_len, n) = (6, seq.len());
    for slice in [prompt_len, 4, 1] {
        let mut e = tiny_engine(4);
        let whole = e.model().forward(&seq).unwrap();
        let v = e.vocab();
        // Another sequence decodes beside it throughout, on the batch's second row.
        let other: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % 60 + 1).collect();
        let (mut a, mut b) = (BlockTable::new(), BlockTable::new());
        let mut got = vec![served(&mut e, &mut a, &seq[..prompt_len], slice, &[]).remove(0)];
        served(&mut e, &mut b, &other[..3], 3, &[]);
        for i in prompt_len..n {
            let rows = e.forward_batched_logit_rows(&mut [&mut a, &mut b], &[seq[i], other[3 + (i - prompt_len) % (n - 3)]]);
            got.push(rows[0].clone());
        }
        for (i, row) in got.iter().take(n - prompt_len + 1).enumerate() {
            let p = prompt_len - 1 + i;
            let err = rel_err(row, &whole[p * v..(p + 1) * v]);
            assert!(err <= TOL, "slices of {slice}: position {p} is {err:.3e} from the uncached forward");
        }
        let full = &gold["decode.full_logits"].1;
        let err = rel_err(&got[got.len() - 2], &full[(n - 2) * v..(n - 1) * v]);
        assert!(err <= TOL, "slices of {slice}: the reference's uncached forward differs by {err:.3e}");
        e.release_table(&mut a);
        e.release_table(&mut b);
    }
}

fn conformance_engine() -> Engine {
    tiny_engine(7)
}

#[test]
fn the_engine_passes_the_serving_conformance_suite() {
    if tiny_dir().is_none() {
        return;
    }
    conformance::run_all(&Conformance { make: conformance_engine, prefix_reuse: true, prefix_prompt_tokens: 20, load_adapter: None });
}

struct Routes(HashMap<String, Vec<f32>>);

impl Probe for Routes {
    fn wants(&self, key: &str) -> bool {
        key.ends_with(".router_indices")
    }
    fn record(&mut self, key: &str, values: Vec<f32>) {
        self.0.insert(key.to_string(), values);
    }
}

#[test]
fn a_recorded_sequence_carries_the_routers_choices_at_every_computed_position() {
    let Some(dir) = tiny_dir() else { return };
    let gold = golden(&dir);
    let prompt: Vec<u32> = gold["tokens"].1.iter().map(|&v| v as u32).collect();
    let mut rollout = SchedulerRollout::new(tiny_engine(4), 4);
    rollout.record_routing(true).expect("the engine records routing");
    let params = RolloutParams { max_new: 5, sample: SampleParams::greedy(), eos: None };
    let completions = rollout.sample_n(&prompt, 2, &params, &mut Rng::new(7));
    let reference = Inkling2::load(crate::tiny_parity::ops(), &dir.join("checkpoint")).unwrap();
    let calls = reference.plan().visits().len();
    for c in completions {
        let record = c.routing.expect("a recording rollout carries the routing");
        let seq: Vec<u32> = prompt.iter().chain(&c.tokens).copied().collect();
        assert_eq!(record.positions as usize, seq.len());
        let mut want = vec![true; seq.len()];
        *want.last_mut().unwrap() = false;
        assert_eq!(record.recorded, want, "every computed position is recorded; the last token is never fed back");
        assert_eq!(record.calls.len(), calls, "one MoE call per logical visit");
        let mut probe = Routes(HashMap::new());
        reference.forward_probed(&seq, &mut probe).unwrap();
        for (call, routed) in record.calls.iter().enumerate() {
            let ids: Vec<u32> = probe.0[&format!("visit{call}.router_indices")].iter().map(|&x| x as u32).collect();
            let k = routed.top_k as usize;
            assert_eq!(&routed.experts[..(seq.len() - 1) * k], &ids[..(seq.len() - 1) * k], "call {call}: the served routing is the router's");
        }
    }
}

/// The real golden's prompts, greedy-decoded through the cache, are the uncached forward's greedy tokens: each served
/// token is the argmax of the uncached logits at its position over the prompt and the tokens served before it.
#[test]
fn cached_greedy_decode_of_the_released_weights_is_the_uncached_forwards() {
    let Some(dir) = brain_testutil::model_dir(REAL).map(PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REAL} is not in the model store"));
        return;
    };
    let gold_dir = brain_testutil::testdata_path("golden/inkling2/real-f32");
    if !gold_dir.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/real-f32 absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py real-f32)");
        return;
    }
    let gold = golden(&gold_dir);
    let sizing = VisitPoolSizing { block_size: 16, num_blocks: 8, live_rows: 1, snapshot_rows: 0 };
    let mut e = Engine::open(crate::tiny_parity::ops(), &dir, EngineOptions { kv_tier: KvTier::F32, sizing, max_seq_len: 128, prefill_chunk: 16 }).unwrap();
    let v = e.vocab();
    const NEW: usize = 6;
    for p in 0..3 {
        let prompt: Vec<u32> = gold[&format!("p{p}.tokens")].1.iter().map(|&t| t as u32).collect();
        let mut table = BlockTable::new();
        let hidden = e.prefill(&mut table, &prompt);
        let mut out = vec![e.admit_greedy(&hidden)];
        while out.len() < NEW {
            let next = e.forward_batched_greedy(&mut [&mut table], &[*out.last().unwrap()]);
            out.push(next[0]);
        }
        e.release_table(&mut table);
        let seq: Vec<u32> = prompt.iter().chain(&out[..NEW - 1]).copied().collect();
        let logits = e.model().forward(&seq).unwrap();
        let argmax = |row: &[f32]| row.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;
        let want: Vec<u32> = (0..NEW).map(|i| argmax(&logits[(prompt.len() - 1 + i) * v..(prompt.len() + i) * v])).collect();
        assert_eq!(out, want, "prompt {p}: cached greedy tokens against the uncached forward's");
        eprintln!("prompt {p}: {} prompt tokens, served {out:?}", prompt.len());
    }
}
