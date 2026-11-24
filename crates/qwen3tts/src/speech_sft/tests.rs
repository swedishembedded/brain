// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The speech objective on tiny random checkpoints (CPU backend): gradients of
//! every parameter family against finite differences, masking, exact resume
//! and an overfit.

use std::collections::HashMap;

use data::rng::Lcg;
use model::Model;

use super::model::{residual_head, residual_table, Example, SftShape, SpeechSftModel, PROJECTION, TALKER_HEAD, TALKER_TABLE};
use super::objective::{example, rows_needed, RowMap, SpeechObjective};
use crate::config::{MtpConfig, TalkerConfig};
use crate::prompt::{TextLayout, TtsSpecials};
use crate::shard::{Record, RecordId, Shard, ShardMeta, ShardWriter, FORMAT, NO_VOICE};

/// Codec specials above the 16-code codebook, as a real checkpoint places them.
pub(crate) fn specials() -> TtsSpecials {
    TtsSpecials {
        tts_bos: 0,
        tts_eos: 1,
        tts_pad: 2,
        codec_nothink: 20,
        codec_think: 21,
        codec_think_bos: 22,
        codec_think_eos: 23,
        codec_pad: 24,
        codec_bos: 25,
        codec_eos: 26,
        lang: HashMap::new(),
        spk_id: HashMap::new(),
        checkpoint: None,
    }
}

pub(crate) const LANGUAGES: [u32; 2] = [30, 31];
pub(crate) const VOICE: u32 = 40;
const CODEBOOK: u32 = 16;
const GROUPS: u32 = 4;

pub(crate) fn rows() -> RowMap {
    RowMap { languages: LANGUAGES.iter().map(|&l| Some(l)).collect(), voices: vec![VOICE] }
}

/// A tiny Talker (2 layers, `d` wide, 48 codec rows) and code predictor (2
/// layers, 16 wide, 4 codebooks of 16): `d == 16` shares the width, `d == 24`
/// needs the projection.
pub(crate) fn shape(d: u32, batch: u32, block: u32, frames: u32) -> SftShape {
    let talker = TalkerConfig { vocab: 48, d_model: d, num_code_groups: GROUPS, ..TalkerConfig::tiny() };
    let mtp = MtpConfig { vocab: CODEBOOK, embedding_dim: d, num_code_groups: GROUPS, ..MtpConfig::tiny() };
    SftShape { talker, mtp, specials: specials(), batch, block, frames, mtp_weight: 0.3, lr: Default::default(), new_rows: Vec::new(), text_inter: 20 }
}

/// `(text tokens, frames, voice, language, complete)` per record.
type Spec = (&'static [u32], usize, bool, u32, bool);

pub(crate) const RECORDS: [Spec; 3] = [(&[11, 12, 13], 4, true, 0, true), (&[14, 15], 3, false, 1, false), (&[16, 17, 18, 19], 5, true, 1, true)];

/// A shard of `specs`, codes and x-vectors drawn from `seed`.
pub(crate) fn shard(d: u32, specs: &[Spec], seed: u64) -> Shard {
    let meta = ShardMeta { format: FORMAT.into(), codebooks: GROUPS, codebook_size: CODEBOOK, xvector_dim: d, frame_rate: 12.5, languages: vec!["alpha".into(), "beta".into()], voices: vec!["v1".into()], records: 0, producers: Default::default() };
    let mut w = ShardWriter::new(meta);
    let mut rng = Lcg::new(seed);
    for (i, &(text, frames, voice, language, complete)) in specs.iter().enumerate() {
        let ids: Vec<u32> = [3, 4, 5].iter().chain(text).chain(&[6, 7, 8, 9, 10]).copied().collect();
        let codes: Vec<u16> = (0..frames * GROUPS as usize).map(|_| (rng.next_u64() % u64::from(CODEBOOK)) as u16).collect();
        let xvector = rng.vec_scaled(d as usize, 0.5);
        let id = RecordId { id: format!("r{i}"), speaker: "s".into(), source: "test".into(), text: String::new() };
        w.push(&Record { id, text_ids: &ids, codes: &codes, xvector: &xvector, voice: if voice { 0 } else { NO_VOICE }, language, complete }).unwrap();
    }
    Shard::from_bytes(&w.to_bytes()).unwrap()
}

/// A model over `shape` with weights drawn from `seed`.
fn build(shape: &SftShape, seed: u64) -> SpeechSftModel {
    SpeechSftModel::build(shape.clone(), &SpeechSftModel::init_weights(shape, seed), seed).unwrap()
}

fn examples(s: &Shard, picks: &[(usize, TextLayout)]) -> Vec<Example> {
    picks.iter().map(|&(r, layout)| example(&specials(), &rows(), s, r, layout).unwrap()).collect()
}

/// The fewest Talker rows that hold every record of `s`.
fn block_for(s: &Shard) -> u32 {
    rows_needed(&specials(), &rows(), s).unwrap() as u32
}

/// Every entry `indices` names of every family, checked one at a time against
/// central differences of the composite loss.
fn gradcheck(d: u32) {
    let s = shard(d, &RECORDS, 0x51); // seed: fixed fixture
    let block = block_for(&s);
    let m = build(&shape(d, 2, block, 10), 0x7a1); // seed: fixed fixture
    // A named voice and a language in the non-streaming layout beside an x-vector utterance that ends mid-speech.
    m.load(examples(&s, &[(0, TextLayout::NonStreaming), (1, TextLayout::Streaming)])).unwrap();
    let ex0 = examples(&s, &[(0, TextLayout::NonStreaming)]).remove(0);
    let ex1 = examples(&s, &[(1, TextLayout::Streaming)]).remove(0);
    let (dd, cb0, cb1, cb3) = (d as usize, ex0.codes[0] as usize, ex1.codes[1] as usize, ex0.codes[3] as usize);
    let row = |r: usize| (r * dd..r * dd + 3).collect::<Vec<_>>();
    let sp = specials();
    let mut checks: Vec<(String, Vec<usize>)> = vec![
        ("talker.model.layers.0.self_attn.q_proj.weight".into(), vec![0, 17, 101]),
        ("talker.model.layers.1.mlp.down_proj.weight".into(), vec![3, 40, 200]),
        ("talker.model.norm.weight".into(), vec![0, 5]),
        (TALKER_HEAD.into(), row(cb0).into_iter().chain(row(sp.codec_eos as usize)).collect()),
        // The table's rows: a first-codebook code (Talker input AND code-predictor position 1), a special, a voice, a language.
        (TALKER_TABLE.into(), [row(cb0), row(sp.codec_think_bos as usize), row(VOICE as usize), row(LANGUAGES[1] as usize)].concat()),
        ("talker.code_predictor.model.layers.0.self_attn.k_proj.weight".into(), vec![1, 33, 90]),
        ("talker.code_predictor.model.layers.1.mlp.up_proj.weight".into(), vec![2, 77]),
        (residual_head(0), vec![cb1 * 16, cb1 * 16 + 5]),
        (residual_head(2), vec![7, 100]),
        // Table 0 embeds codebook 1: reached by the Talker's frame rows and by the code predictor's position 2.
        (residual_table(0), row(cb1)),
        // The last table embeds codebook 3: reached only by the Talker's frame rows.
        (residual_table(2), row(cb3)),
    ];
    if d != 16 {
        checks.push((PROJECTION[0].into(), vec![0, 50, 300]));
        checks.push((PROJECTION[1].into(), vec![0, 7]));
    }
    let mut failures = Vec::new();
    for (name, indices) in &checks {
        let report = gradcheck::elementwise_check_at(&m, name, 2e-3, indices);
        assert!(report.checks.iter().any(|c| c.analytic.abs() > 1e-7), "{name}: every checked gradient is zero");
        failures.extend(report.failures(2e-4, 3e-2).into_iter().map(|c| format!("{} analytic {:e} numeric {:e}", c.param, c.analytic, c.numeric)));
    }
    assert!(failures.is_empty(), "gradient check failures:\n{}", failures.join("\n"));
}

#[test]
fn every_parameter_family_matches_finite_differences_at_a_shared_width() {
    gradcheck(16);
}

#[test]
fn every_parameter_family_matches_finite_differences_through_the_projection() {
    gradcheck(24);
}

/// Padding rows, an empty batch slot and prompt rows carry no target; an
/// incomplete utterance's end of speech is not supervised: one utterance has
/// the same loss and gradients in a larger, padded build.
#[test]
fn padding_prompt_rows_and_an_open_end_contribute_nothing() {
    let d = 16;
    let s = shard(d, &RECORDS, 0x52); // seed: fixed fixture
    let block = block_for(&s);
    let tight = shape(d, 1, block, 5);
    let padded = shape(d, 3, block + 9, 15);
    let init = SpeechSftModel::init_weights(&tight, 0x7a2); // seed: fixed fixture
    let a = SpeechSftModel::build(tight, &init, 1).unwrap();
    let b = SpeechSftModel::build(padded, &init, 2).unwrap();
    for (r, layout) in [(1, TextLayout::Streaming), (2, TextLayout::NonStreaming)] {
        a.load(examples(&s, &[(r, layout)])).unwrap();
        b.load(examples(&s, &[(r, layout)])).unwrap();
        a.zero_grads();
        b.zero_grads();
        let (la, lb) = (a.forward(), b.forward());
        assert!((la - lb).abs() <= 1e-5 * la.abs(), "record {r}: loss {la} alone, {lb} padded");
        let st = a.stats();
        let frames = s.frames(r) as u32;
        assert_eq!(st.talker_count, frames + u32::from(s.record(r).complete), "record {r}: one target per frame, plus the end of speech only when complete");
        assert_eq!((st.cp_count, st.eos_count), (frames * (GROUPS - 1), u32::from(s.record(r).complete)));
        a.backward();
        b.backward();
        for name in [TALKER_TABLE.to_string(), "talker.model.layers.0.mlp.gate_proj.weight".to_string(), residual_table(1)] {
            let (ga, gb) = (a.read_grad(&name), b.read_grad(&name));
            let scale = ga.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let worst = ga.iter().zip(&gb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(worst <= 1e-4 * scale, "record {r}: {name} gradient differs by {worst:e} (scale {scale:e}) under padding");
        }
    }
}

fn fit_opts(steps: u32, batch: u32, lr: f32, seed: u64) -> model::train::FitOpts {
    model::train::FitOpts {
        steps,
        batch_size: batch,
        block_size: 1,
        lr,
        min_lr: lr,
        warmup: 0,
        decay_iters: 0,
        weight_decay: 0.01,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: 0,
        checkpoint_secs: 0,
        initial_probe_batches: 0,
        seed,
        ..model::train::FitOpts::default()
    }
}

/// N steps straight equal k steps, a restart from the saved state and N - k
/// more: the shared loop's exact resume reaches every group's weights and moments.
#[test]
fn a_resumed_run_equals_the_straight_run() {
    let d = 16;
    let s = shard(d, &RECORDS, 0x53); // seed: fixed fixture
    let shape = shape(d, 2, block_for(&s), 10);
    let init = SpeechSftModel::init_weights(&shape, 0x7a3); // seed: fixed fixture
    let opts = fit_opts(4, 2, 1e-2, 0x5eed); // seed: fixed so both runs draw the same stream
    let objective = || SpeechObjective::new(&s, None, rows(), specials(), 2, 0.5, 0.3).unwrap();
    let run = |stop_at: Option<u32>, state: &std::path::Path| {
        let model = SpeechSftModel::build(shape.clone(), &init, 3).unwrap();
        let mut stop = |r: &model::train::StepReport| stop_at != Some(r.step);
        let control = model::train::FitControl { on_step: Some(&mut stop), state: Some(state), state_every: 0, ..Default::default() };
        model::train::fit_controlled(model, objective(), &opts, None, control).unwrap()
    };
    let dir = brain_testutil::Scratch::dir("speech-sft-resume");
    let (straight_report, straight) = run(None, &dir.path().join("straight.state"));
    assert_eq!(straight_report.steps_completed, 4);
    let state = dir.path().join("resumed.state");
    let (first, _) = run(Some(2), &state);
    assert!(first.interrupted && first.steps_completed == 2);
    let (second, resumed) = run(None, &state);
    assert_eq!((second.resumed_at, second.steps_completed), (Some(2), 4));
    for name in straight.param_names() {
        assert_eq!(straight.read_weight(&name), resumed.read_weight(&name), "{name} differs after the resume");
    }
}

/// One utterance, one layout, many steps: the composite loss falls by a large factor.
#[test]
fn one_batch_overfits() {
    let d = 16;
    let s = shard(d, &RECORDS[..1], 0x54); // seed: fixed fixture
    let shape = shape(d, 1, block_for(&s), 5);
    let model = build(&shape, 0x7a4); // seed: fixed fixture
    let mut losses = Vec::new();
    let mut on_step = |r: &model::train::StepReport| {
        losses.push(r.loss);
        true
    };
    let control = model::train::FitControl { on_step: Some(&mut on_step), ..Default::default() };
    let objective = SpeechObjective::new(&s, None, rows(), specials(), 1, 1.0, 0.3).unwrap();
    let opts = model::train::FitOpts { weight_decay: 0.0, ..fit_opts(80, 1, 2e-2, data::rng::random_seed()) };
    model::train::fit_controlled(model, objective, &opts, None, control).unwrap();
    let (first, last) = (losses[0], *losses.last().unwrap());
    assert!(last < first / 10.0, "the loss fell from {first} to {last} only");
}

#[test]
fn the_heads_device_registers_every_kernel_its_ops_needs() {
    model::ops::assert_training_kernel_list_complete(&super::model::aux_kernels());
}

/// Each group steps at its own rate: with the new rows' multiplier at zero a
/// new voice row stays where it started while the rows beside it train.
#[test]
fn the_new_rows_take_their_own_learning_rate() {
    let d = 16;
    let s = shard(d, &RECORDS[..1], 0x55); // seed: fixed fixture
    let mut shape = shape(d, 1, block_for(&s), 5);
    shape.new_rows = vec![VOICE];
    shape.lr.new_rows = 0.0;
    let m = build(&shape, 0x7a5); // seed: fixed fixture
    m.load(examples(&s, &[(0, TextLayout::Streaming)])).unwrap();
    let before = m.read_weight(TALKER_TABLE);
    m.zero_grads();
    m.forward();
    m.backward();
    assert!(m.read_grad(TALKER_TABLE)[VOICE as usize * 16..][..16].iter().any(|g| *g != 0.0), "the voice row has a gradient");
    m.adamw_step(1, 1e-2, 0.0, model::Adam::default(), Some(1.0), 1.0);
    let after = m.read_weight(TALKER_TABLE);
    let row = |t: &[f32], r: u32| t[r as usize * 16..(r as usize + 1) * 16].to_vec();
    assert_eq!(row(&before, VOICE), row(&after, VOICE), "the new row moved at a zero rate");
    assert_ne!(row(&before, LANGUAGES[0]), row(&after, LANGUAGES[0]), "a base row did not train");
    assert!(m.last_grad_norm().is_some_and(|n| n > 0.0));
}
