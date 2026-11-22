// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full fine-tuning of mixture-of-experts language models larger than the accelerator
// for its clients. If your team needs expertise in training every expert of such a model through a fixed device
// window, you can procure our services by sending an email to info@swedishembedded.com.

//! Spec: Inkling 2's layer-streamed full fine-tune (`model::streamed` over the expert banks, the rest of the model in
//! f32 on the device) on the tiny looped configuration, whose core banks are visited once per pass:
//! - the shared layer-streaming contracts: a one-layer window trains exactly what the build holding every bank trains
//!   (every step's loss and every final value, bit for bit, the clip off and active) and what the plain f32 build trains
//!   (`check_layer_streaming`); building straight into the window trains exactly what streaming after construction
//!   trains (`check_streamed_construction`);
//! - bf16 banks under an f32 master (`weights=bf16,window=1`): the forward reads the bf16 rounding of the master (the
//!   logits of a plain f32 build over the rounded banks, bit for bit); steps lower the loss; the saved checkpoint
//!   holds the banks as bf16 tensors equal to the master's rounding, reloads into the trainer's function, and the
//!   family's serving engine serves it (its bf16 banks held as bf16); a run stopped and resumed through the fit loop ends where the uninterrupted one
//!   does (`check_exact_resume`).

use inkling2::train::{init_weights, tiny_config, Inkling2Train};
use checkpoint::packed::PackedTensorReader;
use model::serve::PagedDecoder;
use model::Model;

fn batch(b: usize, t: usize, vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let x: Vec<u32> = (0..b * t).map(|i| (i as u32 * 7 + 3) % vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| if i % t == t - 1 { model::IGNORE } else { (i as u32 * 7 + 10) % vocab }).collect();
    (x, y)
}

fn bank_names(cfg: &inkling2::config::Inkling2Config) -> Vec<String> {
    (0..cfg.n_layers).flat_map(|l| {
        let n = inkling2::names::LayerNames::new(l);
        [n.w13(), n.w2()]
    }).collect()
}

fn round_bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
}

fn resident(spec: &str) -> model::resident_ft::ResidentFtConfig {
    model::TrainPrecision::parse(spec).unwrap().resident_config(Some(data::rng::random_seed())).unwrap().expect("a resident precision")
}

#[test]
fn the_expert_banks_stream_through_a_one_layer_window_as_every_bank_held_trains() {
    let tc = tiny_config();
    let init = init_weights(&tc.cfg, data::rng::random_seed());
    let (x, y) = batch(2, 6, tc.cfg.valid_vocab);
    let step = |m: &Inkling2Train, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("inkling2", || Inkling2Train::new(tc.clone(), 2, 6, &init), step).unwrap();
    let streamed = || Inkling2Train::new_resident(tc.clone(), 2, 6, &init, &resident("weights=f32,window=1"));
    model::streamed::check_streamed_construction("inkling2", || Inkling2Train::new(tc.clone(), 2, 6, &init), streamed, step).unwrap();
}

#[test]
fn bf16_banks_under_an_f32_master_train_save_serve_and_resume() {
    let tc = tiny_config();
    let init = init_weights(&tc.cfg, data::rng::random_seed());
    let banks = bank_names(&tc.cfg);
    let (b, t) = (2usize, 6usize);
    let (x, y) = batch(b, t, tc.cfg.valid_vocab);
    let m = Inkling2Train::new_resident(tc.clone(), b as u32, t as u32, &init, &resident("weights=bf16,window=1")).unwrap();

    // The forward reads the stored bf16 value of each bank.
    let mut rounded = init.clone();
    for name in &banks {
        rounded.get_mut(name).unwrap().iter_mut().for_each(|v| *v = round_bf16(*v));
    }
    let plain = Inkling2Train::new(tc.clone(), b as u32, t as u32, &rounded);
    let row: Vec<u32> = x[..t].to_vec();
    let (got, want) = (m.logits(&row).unwrap(), plain.logits(&row).unwrap());
    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "the streamed bf16 banks are not what the forward reads");
    drop(plain);

    let mut losses = Vec::new();
    for step in 1..=4 {
        m.zero_grads();
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        losses.push(m.forward());
        m.backward();
        m.adamw_step(step, 1e-2, 0.0, model::Adam::default(), Some(1.0), 1.0);
    }
    assert!(losses[3] < losses[0], "losses {losses:?}");

    // The saved checkpoint: bf16 banks equal to the master's rounding, everything else as trained.
    let scratch = brain_testutil::Scratch::dir("inkling2-streamed-bf16");
    let ckpt = scratch.path().join("checkpoint");
    std::fs::create_dir_all(&ckpt).unwrap();
    let file = ckpt.join(inkling2::finetune::WEIGHTS_FILE);
    m.save(file.to_str().unwrap());
    std::fs::write(ckpt.join("config.json"), serde_json::to_string(&tc.json).unwrap()).unwrap();
    let reader = checkpoint::packed::PackedReader::open(&ckpt).unwrap();
    let stored = checkpoint::safetensors::read(file.to_str().unwrap()).unwrap();
    let shapes: std::collections::HashMap<String, Vec<u64>> = inkling2::names::text_manifest(&tc.cfg).into_iter().map(|e| (e.name, e.shape)).collect();
    for name in &banks {
        let desc = reader.descriptor(name).unwrap();
        assert_eq!(desc.encoding, checkpoint::packed::QuantEncoding::bf16(), "{name} is not saved in its stored encoding");
        assert_eq!(desc.shape, shapes[name], "{name} is not saved in its published shape");
        let master = m.read_weight(name);
        let saved = &stored.iter().find(|s| &s.name == name).unwrap().data;
        assert!(master.iter().zip(saved).all(|(a, b)| round_bf16(*a).to_bits() == b.to_bits()), "{name}: the saved bank is not the master's bf16 rounding");
    }
    let reloaded = Inkling2Train::load(file.to_str().unwrap(), 1, t as u32).unwrap();
    let (trained, again) = (m.logits(&row).unwrap(), reloaded.logits(&row).unwrap());
    assert!(trained.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "the saved checkpoint does not reload into the trained function");
    drop(reloaded);

    // The family's serving engine serves the exported checkpoint.
    // The engine at the exact products the in-memory trainer runs (`Fp32Reference`): one function on every device.
    let sizing = model::serve::visit_state::VisitPoolSizing { block_size: 16, num_blocks: 2, live_rows: 1, snapshot_rows: 0, undo_rows: false };
    let options = inkling2::serve::EngineOptions { kv_tier: model::kv_tier::KvTier::F32, sizing, max_seq_len: t, prefill_chunk: t as u32 };
    let mut engine = inkling2::serve::Engine::open(crate::tiny_parity::ops(), &ckpt, options).unwrap();
    let valid = tc.cfg.valid_vocab as usize;
    let positions: Vec<usize> = (0..t).collect();
    for (p, hidden) in engine.hidden_at(&row, &positions).into_iter().enumerate() {
        let served = engine.logits(&hidden);
        let want = &trained[p * valid..(p + 1) * valid];
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let err = served.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        assert!(err <= 5e-5, "position {p}: served logits differ by {err:.3e} of their scale");
    }
    drop(engine);

    let resume = scratch.path().join("resume");
    std::fs::create_dir_all(&resume).unwrap();
    let small = tc.clone();
    let start = init_weights(&small.cfg, data::rng::random_seed());
    model::trainprec::check_exact_resume("inkling2", || Inkling2Train::new(small.clone(), 1, 8, &start), 8, &resume, "weights=bf16,window=1").unwrap();
}

/// `--optimizer-state-dir` (`TrainPrecision::state_dir`): a bf16-bank run through the fit loop keeps the banks' master
/// and moments in the named directory, not in host memory.
#[test]
fn the_streamed_banks_keep_their_optimizer_state_in_the_named_directory() {
    let tc = tiny_config();
    let init = init_weights(&tc.cfg, data::rng::random_seed());
    let streamed: u64 = bank_names(&tc.cfg).iter().map(|n| init[n].len() as u64).sum();
    let scratch = brain_testutil::Scratch::dir("inkling2-state-dir");
    model::trainprec::check_state_in_dir("inkling2", || Inkling2Train::new(tc.clone(), 1, 8, &init), 8, scratch.path(), "weights=bf16,window=1", streamed).unwrap();
}
