// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The speech objective on tiny random checkpoints (CPU backend): gradients of
//! every parameter family against finite differences, masking, exact resume
//! and an overfit.

use std::collections::HashMap;

use data::rng::Lcg;
use data::tokenizer::Tokenizer;
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
    SftShape { talker, mtp, specials: specials(), batch, block, frames, mtp_weight: 0.3, lr: Default::default(), new_rows: Vec::new(), text_inter: 20, gemm: gpu_core::provider::library::GemmPrecision::Fp32Reference }
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
    SpeechSftModel::build(shape.clone(), SpeechSftModel::init_weights(shape, seed), seed).unwrap()
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
    let a = SpeechSftModel::build(tight, init.clone(), 1).unwrap();
    let b = SpeechSftModel::build(padded, init, 2).unwrap();
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
        let model = SpeechSftModel::build(shape.clone(), init.clone(), 3).unwrap();
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

/// A tiny release on disk, as a training run reads one: `config.json` naming
/// the specials and one language, `model.safetensors` under upstream names.
/// The Talker table's named rows are trained (norm ~1), the rest of its
/// special band untrained (norm ~0.002).
fn tiny_release(dir: &std::path::Path, d: u32) {
    let s = shape(d, 1, 1, 1);
    let mut tensors = SpeechSftModel::init_weights(&s, 0x7a6); // seed: fixed fixture
    let table = tensors.get_mut(TALKER_TABLE).unwrap();
    let sp = specials();
    let named: Vec<u32> = [sp.codec_nothink, sp.codec_think, sp.codec_think_bos, sp.codec_think_eos, sp.codec_pad, sp.codec_bos, sp.codec_eos, LANGUAGES[0]].into();
    for id in CODEBOOK..48 {
        let scale = if named.contains(&id) { 1.0 } else { 0.002 };
        table[id as usize * d as usize..(id as usize + 1) * d as usize].fill(scale);
    }
    let config = serde_json::json!({
        "tts_bos_token_id": sp.tts_bos, "tts_eos_token_id": sp.tts_eos, "tts_pad_token_id": sp.tts_pad,
        "tts_model_type": "base", "tts_model_size": "tiny",
        "talker_config": {
            "num_hidden_layers": s.talker.n_layers, "hidden_size": d, "head_dim": s.talker.head_dim,
            "num_attention_heads": s.talker.n_heads, "num_key_value_heads": s.talker.n_kv_heads,
            "intermediate_size": s.talker.d_ff, "vocab_size": 48, "num_code_groups": GROUPS,
            "text_hidden_size": s.talker.text_hidden_size, "text_vocab_size": s.talker.text_vocab_size,
            "codec_nothink_id": sp.codec_nothink, "codec_think_id": sp.codec_think, "codec_think_bos_id": sp.codec_think_bos,
            "codec_think_eos_id": sp.codec_think_eos, "codec_pad_id": sp.codec_pad, "codec_bos_id": sp.codec_bos,
            "codec_eos_token_id": sp.codec_eos, "codec_language_id": {"alpha": LANGUAGES[0]}, "spk_id": {},
            "code_predictor_config": {
                "num_hidden_layers": s.mtp.n_layers, "hidden_size": s.mtp.d_model, "head_dim": s.mtp.head_dim,
                "num_attention_heads": s.mtp.n_heads, "num_key_value_heads": s.mtp.n_kv_heads,
                "intermediate_size": s.mtp.d_ff, "vocab_size": CODEBOOK, "num_code_groups": GROUPS,
            },
        },
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let plan: Vec<(String, Vec<u64>)> = names.iter().map(|n| ((*n).clone(), vec![tensors[*n].len() as u64])).collect();
    let path = dir.join("model.safetensors");
    let mut w = checkpoint::weightio::StWriter::create(path.to_str().unwrap(), &plan, &serde_json::Value::Null, None).unwrap();
    for n in names {
        w.write(n, &tensors[n]).unwrap();
    }
    w.finish().unwrap();
}

/// The whole run from a release on disk: a base language, a new language and a
/// new voice; every trained tensor handed back under its upstream name and
/// shape; the step and held-out reports; the new rows trained.
#[test]
fn a_run_from_a_release_trains_new_rows_and_hands_back_upstream_tensors() {
    let d = 16;
    let dir = brain_testutil::Scratch::dir("speech-sft-release");
    tiny_release(dir.path(), d);
    let s = shard(d, &RECORDS, 0x56); // seed: fixed fixture
    let cfg: super::SpeechSftConfig = serde_json::from_value(serde_json::json!({
        "base": dir.path().to_str().unwrap(),
        "steps": 3, "lr": 1e-2, "batch": 2, "max_frames": 5, "eval_interval": 1,
        "lr_multipliers": {"new_rows": 2.0},
        "rows": {"languages": {"alpha": "base", "beta": "new"}, "voices": {"v1": "new"}},
    }))
    .unwrap();
    let base = super::Base::open(dir.path()).unwrap();
    let r = super::speech_sft(&cfg, &s, Some(&s), model::train::FitControl::default()).unwrap();
    assert_eq!(r.languages["alpha"], LANGUAGES[0]);
    let (beta, v1) = (r.languages["beta"], r.voices["v1"]);
    assert!(beta > LANGUAGES[0] && beta < 48 && v1 < 48 && v1 != beta, "new rows {beta}, {v1}");
    assert_eq!(r.report.steps.len(), 3);
    for step in &r.report.steps {
        assert!(step.loss.is_finite() && step.talker_ce > 0.0 && step.code_predictor_ce > 0.0 && step.frames_per_s > 0.0 && step.grad_norm.is_some(), "{step:?}");
        let held = step.held_out.expect("an evaluation every step");
        assert!(held.loss.is_finite() && held.talker_ce > 0.0);
    }
    let names: Vec<&str> = r.tensors.iter().map(|(n, _, _)| n.as_str()).collect();
    assert!(names.contains(&"talker.model.layers.1.mlp.down_proj.weight") && names.contains(&TALKER_HEAD) && names.contains(&"talker.code_predictor.lm_head.2.weight"));
    for (name, shape, values) in &r.tensors {
        assert_eq!(shape, &base.shapes[name], "{name}");
        assert_eq!(values.len() as u64, shape.iter().product::<u64>(), "{name}");
    }
    let table = &r.tensors.iter().find(|(n, _, _)| n == TALKER_TABLE).unwrap().2;
    let row = |t: &[f32], id: u32| t[id as usize * 16..(id as usize + 1) * 16].to_vec();
    let start = &base.tensors[TALKER_TABLE];
    assert_ne!(row(table, v1), row(start, v1), "the new voice row started at its centroid and trained");
    assert!(row(table, v1).iter().map(|x| x * x).sum::<f32>().sqrt() > 0.1, "the new voice row is not an untrained row");
    let bad = serde_json::from_value::<super::SpeechSftConfig>(serde_json::json!({"base": "x", "steps": 1, "rows": {}, "learning_rate": 1.0}));
    assert!(bad.unwrap_err().to_string().contains("unknown field"));
}

/// The rows a run trains on are the rows generation feeds, bit for bit, in both
/// layouts and with a named voice in the speaker slot, on the real checkpoint.
#[test]
fn training_rows_are_inference_rows_on_the_real_checkpoint() {
    let Some(paths) = crate::testsupport::store_paths() else {
        brain_testutil::skip("no Qwen3-TTS checkpoint in the model store (brain pull Qwen/Qwen3-TTS-12Hz-0.6B-Base)");
        return;
    };
    let base = super::Base::open(std::path::Path::new(&paths.ckpt_dir)).unwrap();
    let shape = SftShape { talker: base.talker.clone(), mtp: base.mtp.clone(), specials: base.specials.clone(), batch: 1, block: 1, frames: 1, mtp_weight: 0.3, lr: Default::default(), new_rows: Vec::new(), text_inter: base.text_inter, gemm: gpu_core::provider::library::GemmPrecision::Fp32Reference };
    let train = super::model::host_tables(&shape, &mut base.tensors.clone()).unwrap();
    let tables = crate::npu_gen::TalkerTables::load(&paths.talker);
    let mtp = crate::gen_kv_mtp::CpuMtp::load(&paths.mtp);
    let sp = &base.specials;
    let tok = crate::prompt::load_tokenizer(&paths.ckpt_dir).unwrap();
    let (role, text) = crate::pipeline::split_input_ids(&tok.encode(&crate::pipeline::assistant_text("Every row the trainer builds is the row generation feeds."))).unwrap();
    let mut rng = Lcg::new(0x5eed_0001); // seed: fixed fixture
    let xvec = rng.vec_scaled(base.talker.d_model as usize, 0.5);
    let codes: Vec<u32> = (0..6 * 16).map(|_| (rng.next_u64() % 2048) as u32).collect();
    let language = sp.language_id("english").unwrap();
    let voice = sp.lang["chinese"]; // any trained row serves as a named voice's slot here
    for layout in [TextLayout::Streaming, TextLayout::NonStreaming] {
        for speaker in [crate::prompt::SpeakerSlot::XVector, crate::prompt::SpeakerSlot::Codec(voice)] {
            let plan = crate::prompt::PromptPlan::build(sp, &crate::prompt::PromptSpec { role_ids: &role, text_ids: &text, instruct_ids: &[], speaker, language_id: language, layout }).unwrap();
            let trained = crate::train::policy_rows(&train, &train, &plan.materialise(&train, Some(&xvec)), &codes, 16, Some(sp.codec_eos));
            let inference = match speaker {
                crate::prompt::SpeakerSlot::Codec(id) => crate::prompt::build_instruct_prompt_in(&tables, sp, &role, &text, &[], Some(id), language, layout),
                _ => crate::prompt::build_xvector_prompt_in(&tables, sp, &role, &text, Some(&xvec), language, layout),
            };
            let served = crate::train::policy_rows(&tables, &mtp, &inference, &codes, 16, Some(sp.codec_eos));
            assert_eq!(trained.targets, served.targets, "{layout:?} {speaker:?}");
            assert!(trained.rows == served.rows, "{layout:?} {speaker:?}: the training rows differ from the rows generation feeds");
        }
    }
}

/// One real utterance overfits and the step runs at a measured frame rate:
/// written for a card, refused on the CPU backend by name.
#[test]
#[ignore = "slow: trains the real checkpoint; run on a GPU"]
fn the_real_checkpoint_overfits_one_utterance_and_reports_its_frame_rate() {
    if brain_testutil::gpu_unavailable(gpu_core::testgpu::missing_gpu()) {
        return;
    }
    let Some(paths) = crate::testsupport::store_paths() else {
        brain_testutil::skip("no Qwen3-TTS checkpoint in the model store (brain pull Qwen/Qwen3-TTS-12Hz-0.6B-Base)");
        return;
    };
    let base = super::Base::open(std::path::Path::new(&paths.ckpt_dir)).unwrap();
    let d = base.talker.d_model;
    let tok = crate::prompt::load_tokenizer(&paths.ckpt_dir).unwrap();
    let ids = tok.encode(&crate::pipeline::assistant_text("One utterance, learned by heart."));
    let mut rng = Lcg::new(0x5eed_0002); // seed: fixed fixture
    let codes: Vec<u16> = (0..60 * 16).map(|_| (rng.next_u64() % 2048) as u16).collect();
    let xvector = rng.vec_scaled(d as usize, 0.5);
    let meta = ShardMeta { format: FORMAT.into(), codebooks: 16, codebook_size: 2048, xvector_dim: d, frame_rate: 12.5, languages: vec!["english".into()], voices: Vec::new(), records: 0, producers: Default::default() };
    let mut w = ShardWriter::new(meta);
    w.push(&Record { id: RecordId { id: "u".into(), speaker: "s".into(), source: "test".into(), text: String::new() }, text_ids: &ids, codes: &codes, xvector: &xvector, voice: NO_VOICE, language: 0, complete: true }).unwrap();
    let s = Shard::from_bytes(&w.to_bytes()).unwrap();
    let cfg: super::SpeechSftConfig = serde_json::from_value(serde_json::json!({
        "base": paths.ckpt_dir, "steps": 40, "lr": 1e-4, "batch": 1, "max_frames": 60, "streaming_fraction": 1.0,
        "rows": {"languages": {"english": "base"}},
    }))
    .unwrap();
    let r = super::speech_sft(&cfg, &s, None, model::train::FitControl::default()).unwrap();
    let (first, last) = (r.report.steps[0].loss, r.report.steps.last().unwrap().loss);
    assert!(last < first / 3.0, "the loss fell from {first} to {last} only");
    let rate = r.report.steps.last().unwrap().frames_per_s;
    eprintln!("speech sft on the real checkpoint: {rate:.0} frames/s at one 60-frame utterance per step");
}
