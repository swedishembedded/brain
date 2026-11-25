// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Distillation of the speech objective on tiny random checkpoints (CPU backend): a student that is its teacher
//! learns nothing from it, the composite gradient against finite differences, the teacher frozen, weights of zero
//! reducing to the plain objective bit for bit, the text vocabulary map, and a whole run from a narrower student
//! release whose rows are all its base's.

use std::collections::HashMap;

use model::Model;

use super::distill::{projection, KdShape, SpeechTeacher, TeacherMap, TextVocabMap};
use super::model::{residual_head, residual_table, Example, SftShape, SpeechSftModel, TALKER_HEAD, TALKER_TABLE};
use super::objective::{example, rows_needed};
use super::tests::{rows, shape, shard, specials, LANGUAGES, RECORDS, VOICE};
use crate::prompt::TextLayout;
use crate::shard::Shard;

/// Every value rounded to bf16 (truncated), as a released checkpoint stores it: the teacher's bf16 tier holds it
/// exactly.
fn bf16(mut init: HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
    for v in init.values_mut().flatten() {
        *v = f32::from_bits(v.to_bits() & 0xffff_0000);
    }
    init
}

/// The identity map over a text vocabulary of `n` tokens.
fn identity(n: u32) -> TeacherMap {
    let tokens: HashMap<String, u32> = (0..n).map(|i| (format!("t{i}"), i)).collect();
    TeacherMap::text_only(TextVocabMap::from_tokens(&tokens, &tokens, n, n).unwrap())
}

fn kd(talker: f32, mtp: f32, hidden: f32, temperature: f32, layers: Vec<(u32, u32)>, teacher_d: u32) -> KdShape {
    KdShape { talker, mtp, hidden, temperature, layers, teacher_d }
}

fn block_for(s: &Shard) -> u32 {
    rows_needed(&specials(), &rows(), s).unwrap() as u32
}

fn examples(s: &Shard, picks: &[(usize, TextLayout)]) -> Vec<Example> {
    picks.iter().map(|&(r, layout)| example(&specials(), &rows(), s, r, layout).unwrap()).collect()
}

/// A student of `student` (distilling under `k`) and its frozen teacher of `teacher`, sized alike.
fn pair(student: SftShape, student_init: HashMap<String, Vec<f32>>, teacher: SftShape, teacher_init: HashMap<String, Vec<f32>>, k: KdShape) -> (SpeechSftModel, SpeechTeacher) {
    let text = student.talker.text_vocab_size;
    let s = SpeechSftModel::build(SftShape { kd: Some(k), ..student }, student_init, 3).unwrap();
    let t = SpeechTeacher::new(&s, teacher, teacher_init, identity(text)).unwrap();
    (s, t)
}

fn load(s: &SpeechSftModel, t: &SpeechTeacher, exs: Vec<Example>) {
    s.load(exs.clone()).unwrap();
    t.teach(s, &exs).unwrap();
}

/// A student holding its teacher's weights is at zero distance from it: every KL and the hidden-state error through
/// the identity projection vanish, while the ground-truth terms do not.
#[test]
fn a_student_that_is_its_teacher_is_at_zero_kl() {
    let d = 16;
    let s = shard(d, &RECORDS, 0x61); // seed: fixed fixture
    let sh = shape(d, 2, block_for(&s), 10);
    let init = bf16(SpeechSftModel::init_weights(&sh, 0x7b1)); // seed: fixed fixture
    let (student, teacher) = pair(sh.clone(), init.clone(), sh, init, kd(1.0, 1.0, 1.0, 2.0, vec![(1, 1), (2, 2)], d));
    load(&student, &teacher, examples(&s, &[(0, TextLayout::NonStreaming), (2, TextLayout::Streaming)]));
    student.forward();
    let st = student.stats();
    assert!(st.kd.talker_count > 0 && st.kd.cp_frames > 0 && st.kd.hidden_count > 0, "{st:?}");
    assert!(st.kd.talker_kl().abs() < 1e-6 && st.kd.code_predictor_kl().abs() < 1e-6 && st.kd.hidden_mse() < 1e-9, "a student that is its teacher: {:?}", st.kd);
    assert!(st.talker_sum > 0.0 && st.cp_sum > 0.0);
    // Another teacher is not at zero distance.
    let other = bf16(SpeechSftModel::init_weights(&shape(d, 2, block_for(&s), 10), 0x7b2)); // seed: fixed fixture
    let (student, teacher) = pair(shape(d, 2, block_for(&s), 10), other, shape(d, 2, block_for(&s), 10), bf16(SpeechSftModel::init_weights(&shape(d, 2, block_for(&s), 10), 0x7b1)), kd(1.0, 1.0, 0.0, 1.0, vec![], d)); // seed: fixed fixture
    load(&student, &teacher, examples(&s, &[(0, TextLayout::NonStreaming)]));
    student.forward();
    assert!(student.stats().kd.talker_kl() > 1e-4 && student.stats().kd.code_predictor_kl() > 1e-4, "{:?}", student.stats().kd);
}

/// The composite loss (ground truth, both KLs at a temperature, a hidden-state pair through its projection, with a
/// teacher wider and deeper than the student) against central differences, entry by entry of every family the
/// distillation reaches.
#[test]
fn the_distilled_loss_matches_finite_differences() {
    let d = 16;
    let s = shard(d, &RECORDS, 0x62); // seed: fixed fixture
    let block = block_for(&s);
    let sh = shape(d, 2, block, 10);
    let mut th = shape(24, 2, block, 10);
    th.talker.n_layers = 3;
    let (student, teacher) = pair(sh.clone(), SpeechSftModel::init_weights(&sh, 0x7b3), th.clone(), SpeechSftModel::init_weights(&th, 0x7b4), kd(0.7, 0.5, 0.3, 2.0, vec![(1, 2)], 24)); // seed: fixed fixture
    // Named voices: the teacher, of another width, cannot read the student's x-vectors.
    let exs = examples(&s, &[(0, TextLayout::NonStreaming), (2, TextLayout::Streaming)]);
    let (cb0, cb1) = (exs[0].codes[0] as usize, exs[1].codes[1] as usize);
    load(&student, &teacher, exs);
    let row = |r: usize| (r * 16..r * 16 + 3).collect::<Vec<_>>();
    let checks: Vec<(String, Vec<usize>)> = vec![
        ("talker.model.layers.0.self_attn.q_proj.weight".into(), vec![0, 17, 101]),
        ("talker.model.layers.1.mlp.down_proj.weight".into(), vec![3, 40]),
        (TALKER_HEAD.into(), [row(cb0), row(7)].concat()),
        (TALKER_TABLE.into(), [row(cb0), row(VOICE as usize), row(LANGUAGES[1] as usize)].concat()),
        ("talker.code_predictor.model.layers.1.mlp.up_proj.weight".into(), vec![2, 77]),
        (residual_head(0), vec![cb1 * 16, cb1 * 16 + 5, 3]),
        (residual_head(2), vec![7, 100]),
        (residual_table(0), row(cb1)),
        (projection(0), vec![0, 17, 200, 383]),
    ];
    let mut failures = Vec::new();
    for (name, indices) in &checks {
        let report = gradcheck::elementwise_check_at(&student, name, 2e-3, indices);
        assert!(report.checks.iter().any(|c| c.analytic.abs() > 1e-7), "{name}: every checked gradient is zero");
        failures.extend(report.failures(2e-4, 3e-2).into_iter().map(|c| format!("{} analytic {:e} numeric {:e}", c.param, c.analytic, c.numeric)));
    }
    assert!(failures.is_empty(), "gradient check failures:\n{}", failures.join("\n"));
    let st = student.stats();
    assert!(st.kd.talker_kl() > 0.0 && st.kd.code_predictor_kl() > 0.0 && st.kd.hidden_mse() > 0.0, "{:?}", st.kd);
}

/// The teacher holds no gradient and no optimizer state, and the student's updates leave it as it was: it teaches the
/// same logits before and after.
#[test]
fn the_teacher_is_frozen() {
    let d = 16;
    let s = shard(d, &RECORDS, 0x63); // seed: fixed fixture
    let sh = shape(d, 2, block_for(&s), 10);
    let (student, teacher) = pair(sh.clone(), SpeechSftModel::init_weights(&sh, 0x7b5), sh.clone(), SpeechSftModel::init_weights(&sh, 0x7b6), kd(1.0, 1.0, 0.0, 1.0, vec![], d)); // seed: fixed fixture
    assert_eq!(teacher.trained_params(), 0, "the teacher trains nothing");
    assert!(student.param_names().iter().all(|n| !n.starts_with("teacher")));
    let exs = examples(&s, &[(0, TextLayout::NonStreaming), (1, TextLayout::Streaming)]);
    load(&student, &teacher, exs.clone());
    student.zero_grads();
    let before = student.forward();
    student.backward();
    student.adamw_step(1, 5e-2, 0.0, model::Adam::default(), Some(1.0), 1.0);
    load(&student, &teacher, exs);
    let after = student.forward();
    assert!(after < before, "a step toward the teacher lowers the loss: {before} -> {after}");
    // The teacher's side is unchanged: a fresh student of the starting weights sees the starting loss.
    let (fresh, again) = pair(sh.clone(), SpeechSftModel::init_weights(&sh, 0x7b5), sh.clone(), SpeechSftModel::init_weights(&sh, 0x7b6), kd(1.0, 1.0, 0.0, 1.0, vec![], d)); // seed: fixed fixture
    load(&fresh, &again, examples(&s, &[(0, TextLayout::NonStreaming), (1, TextLayout::Streaming)]));
    assert_eq!(fresh.forward().to_bits(), before.to_bits());
}

/// Distillation weighted zero is the plain objective bit for bit: the same loss and every gradient identical.
#[test]
fn zero_distillation_weights_are_the_plain_objective() {
    let d = 16;
    let s = shard(d, &RECORDS, 0x64); // seed: fixed fixture
    let sh = shape(d, 2, block_for(&s), 10);
    let init = SpeechSftModel::init_weights(&sh, 0x7b7); // seed: fixed fixture
    let plain = SpeechSftModel::build(sh.clone(), init.clone(), 3).unwrap();
    let (student, teacher) = pair(sh.clone(), init, sh.clone(), SpeechSftModel::init_weights(&sh, 0x7b8), kd(0.0, 0.0, 0.0, 1.0, vec![], d)); // seed: fixed fixture
    let exs = examples(&s, &[(0, TextLayout::NonStreaming), (2, TextLayout::Streaming)]);
    plain.load(exs.clone()).unwrap();
    load(&student, &teacher, exs);
    for m in [&plain, &student] {
        m.zero_grads();
    }
    let (a, b) = (plain.forward(), student.forward());
    assert_eq!(a.to_bits(), b.to_bits(), "loss {a} plain, {b} with zero distillation weights");
    plain.backward();
    student.backward();
    for name in plain.param_names() {
        let (ga, gb) = (plain.read_grad(&name), student.read_grad(&name));
        assert!(ga.iter().zip(&gb).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: the gradient differs");
    }
}

/// The map takes each student token to the teacher's token of the same string, and refuses a student token the
/// teacher does not have, by name.
#[test]
fn the_text_vocabulary_map_refuses_a_token_the_teacher_lacks() {
    let teacher: HashMap<String, u32> = [("a", 0), ("b", 1), ("hej", 2), ("<tts_pad>", 5)].into_iter().map(|(t, i)| (t.to_string(), i)).collect();
    let student: HashMap<String, u32> = [("hej", 0), ("a", 1), ("<tts_pad>", 2)].into_iter().map(|(t, i)| (t.to_string(), i)).collect();
    let map = TextVocabMap::from_tokens(&student, &teacher, 4, 8).unwrap();
    assert_eq!((map.get(0), map.get(1), map.get(2)), (Ok(2), Ok(0), Ok(5)));
    assert!(map.get(3).is_err(), "a student row that is no token maps nowhere");
    let mut lacking = student.clone();
    lacking.insert("tjena".into(), 3);
    let err = TextVocabMap::from_tokens(&lacking, &teacher, 4, 8).unwrap_err();
    assert!(err.contains("\"tjena\"") && err.contains("not in the teacher's vocabulary"), "{err}");
    let err = TextVocabMap::from_tokens(&student, &teacher, 4, 4).unwrap_err();
    assert!(err.contains("past the teacher's 4"), "{err}");
}

/// A Qwen3-TTS release on disk in the upstream layout, of `s`'s shapes: its named languages and voices at the test
/// rows and its voice at `voice` (trained, norm ~1), the tokenizer files of a text vocabulary whose token `t{i}` is id `i + token_offset`.
pub(crate) fn release(dir: &std::path::Path, s: &SftShape, seed: u64, token_offset: u32, voice: u32) {
    let mut tensors = SpeechSftModel::init_weights(s, seed);
    let d = s.d();
    let sp = specials();
    let table = tensors.get_mut(TALKER_TABLE).unwrap();
    for id in [sp.codec_nothink, sp.codec_think, sp.codec_think_bos, sp.codec_think_eos, sp.codec_pad, sp.codec_bos, sp.codec_eos, LANGUAGES[0], LANGUAGES[1], voice] {
        table[id as usize * d..(id as usize + 1) * d].fill(1.0);
    }
    let config = serde_json::json!({
        "architectures": ["Qwen3TTSForConditionalGeneration"],
        "tts_bos_token_id": sp.tts_bos + token_offset, "tts_eos_token_id": sp.tts_eos + token_offset, "tts_pad_token_id": sp.tts_pad + token_offset,
        "tts_model_type": "custom_voice", "tts_model_size": "tiny",
        "talker_config": {
            "num_hidden_layers": s.talker.n_layers, "hidden_size": d, "head_dim": s.talker.head_dim,
            "num_attention_heads": s.talker.n_heads, "num_key_value_heads": s.talker.n_kv_heads,
            "intermediate_size": s.talker.d_ff, "vocab_size": s.talker.vocab, "num_code_groups": s.mtp.num_code_groups,
            "text_hidden_size": s.talker.text_hidden_size, "text_vocab_size": s.talker.text_vocab_size,
            "codec_nothink_id": sp.codec_nothink, "codec_think_id": sp.codec_think, "codec_think_bos_id": sp.codec_think_bos,
            "codec_think_eos_id": sp.codec_think_eos, "codec_pad_id": sp.codec_pad, "codec_bos_id": sp.codec_bos,
            "codec_eos_token_id": sp.codec_eos, "codec_language_id": {"alpha": LANGUAGES[0], "beta": LANGUAGES[1]}, "spk_id": {"v1": voice}, "spk_is_dialect": {"v1": false},
            "code_predictor_config": {
                "num_hidden_layers": s.mtp.n_layers, "hidden_size": s.mtp.d_model, "head_dim": s.mtp.head_dim,
                "num_attention_heads": s.mtp.n_heads, "num_key_value_heads": s.mtp.n_kv_heads,
                "intermediate_size": s.mtp.d_ff, "vocab_size": s.mtp.vocab, "num_code_groups": s.mtp.num_code_groups,
            },
        },
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    // The last two text rows are no token: an embedding padded past its tokenizer, as the releases are.
    let vocab: serde_json::Map<String, serde_json::Value> = (0..s.talker.text_vocab_size - 2 - token_offset).map(|i| (format!("t{i}"), serde_json::json!(i + token_offset))).collect();
    std::fs::write(dir.join("vocab.json"), serde_json::to_vec(&vocab).unwrap()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), b"{\"added_tokens_decoder\": {}}").unwrap();
    for f in ["generation_config.json", "preprocessor_config.json"] {
        std::fs::write(dir.join(f), b"{}").unwrap();
    }
    std::fs::write(dir.join("merges.txt"), b"#version: 0.2\n").unwrap();
    std::fs::create_dir_all(dir.join("speech_tokenizer")).unwrap();
    std::fs::write(dir.join("speech_tokenizer/config.json"), b"{}").unwrap();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let plan: Vec<(String, Vec<u64>)> = names.iter().map(|n| ((*n).clone(), vec![tensors[*n].len() as u64])).collect();
    let mut w = checkpoint::weightio::StWriter::create(dir.join("model.safetensors").to_str().unwrap(), &plan, &serde_json::Value::Null, None).unwrap();
    for n in names {
        w.write(n, &tensors[n]).unwrap();
    }
    w.finish().unwrap();
}

/// A whole distilling run from a student release narrower than its teacher, with a larger, renumbered text
/// vocabulary on the teacher's side: every language and voice the student's own (`base`), the shard's x-vectors of the
/// teacher's width (named voices never read them), each term reported per step
/// and held out, the held-out distance to the teacher the kept best's metric, every tensor handed back at the
/// student's shapes, and the export writing them. A teacher that lacks a student token is refused before training.
#[test]
fn a_narrow_student_release_distils_from_its_teacher_and_exports_its_own_shapes() {
    let dir = brain_testutil::Scratch::dir("speech-sft-distill");
    let (student_dir, teacher_dir) = (dir.path().join("student"), dir.path().join("teacher"));
    std::fs::create_dir_all(&student_dir).unwrap();
    std::fs::create_dir_all(&teacher_dir).unwrap();
    let mut student = shape(16, 1, 1, 1);
    student.talker.n_layers = 1;
    student.mtp.n_layers = 1;
    let mut teacher = shape(24, 1, 1, 1);
    teacher.talker.text_vocab_size = 40;
    release(&student_dir, &student, 0x7c1, 0, VOICE); // seed: fixed fixture
    // The teacher holds the voice at another row: it is taught on its own.
    release(&teacher_dir, &teacher, 0x7c2, 3, VOICE + 1); // seed: fixed fixture
    // Named voices only, their x-vectors (never read) of the teacher's speaker encoder's width.
    let s = shard(24, &[RECORDS[0], RECORDS[2]], 0x65); // seed: fixed fixture
    let cfg = |extra: serde_json::Value| -> super::SpeechSftConfig {
        let mut v = serde_json::json!({
            "base": student_dir.to_str().unwrap(), "teacher": teacher_dir.to_str().unwrap(),
            "steps": 3, "lr": 1e-2, "batch": 2, "max_frames": 5, "eval_interval": 1, "keep_best": true,
            "talker_kd_weight": 1.0, "mtp_kd_weight": 0.5, "hidden_kd_weight": 0.1, "hidden_kd_layers": [[1, 2]], "kd_temperature": 2.0,
            "rows": {"languages": {"alpha": "base", "beta": "base"}, "voices": {"v1": "base"}},
        });
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        serde_json::from_value(v).unwrap()
    };
    let r = super::speech_sft(&cfg(serde_json::json!({})), &s, Some(&s), model::train::FitControl::default()).unwrap();
    assert_eq!((r.languages["alpha"], r.languages["beta"], r.voices["v1"]), (LANGUAGES[0], LANGUAGES[1], VOICE));
    for step in &r.report.steps {
        let k = step.distilled.expect("a distilling step reports its terms");
        assert!(k.talker_kl > 0.0 && k.code_predictor_kl > 0.0 && k.hidden_mse > 0.0 && step.talker_ce > 0.0, "{step:?}");
        let held = step.held_out.unwrap();
        let hk = held.distilled.expect("held out too");
        let selection = 4.0 * (hk.talker_kl + 0.5 * hk.code_predictor_kl) + 0.1 * hk.hidden_mse;
        assert!((held.loss - selection).abs() <= 1e-4 * selection, "the held-out score is the distance to the teacher: {} vs {selection}", held.loss);
    }
    assert!(r.report.fit.kept_best.is_some());
    let base = super::Base::open(&student_dir).unwrap();
    assert!(r.tensors.iter().all(|(n, _, _)| !n.starts_with(super::distill::PRIVATE_PREFIX)), "the projections are not checkpoint tensors");
    assert_eq!(r.tensors.len(), base.tensors.len() - 5, "every trained tensor of the student, the frozen text path aside");
    for (name, shape, _) in &r.tensors {
        assert_eq!(shape, &base.shapes[name], "{name}");
    }
    let out = dir.path().join("release");
    let trained = crate::export::Trained { tensors: &r.tensors, languages: &r.languages, voices: &r.voices };
    crate::export::export(&student_dir, &out, "local/student", &trained, &mut || true).unwrap();
    let exported = super::Base::open(&out).unwrap();
    assert_eq!(exported.talker.d_model, 16);
    assert_eq!(exported.shapes[TALKER_HEAD], base.shapes[TALKER_HEAD]);

    // A record in the x-vector slot cannot be taught by a teacher of another width.
    let err = super::speech_sft(&cfg(serde_json::json!({})), &shard(16, &RECORDS, 0x66), None, model::train::FitControl::default()).unwrap_err(); // seed: fixed fixture
    assert!(err.contains("x-vector slot"), "{err}");
    // A student token the teacher's vocabulary lacks is refused at the start, by name.
    let vocab: serde_json::Map<String, serde_json::Value> = (0..26u32).map(|i| (if i == 12 { "only-student".to_string() } else { format!("t{i}") }, serde_json::json!(i))).collect();
    std::fs::write(student_dir.join("vocab.json"), serde_json::to_vec(&vocab).unwrap()).unwrap();
    let err = super::speech_sft(&cfg(serde_json::json!({})), &s, None, model::train::FitControl::default()).unwrap_err();
    assert!(err.contains("only-student") && err.contains("teacher"), "{err}");
    // Distillation settings without a teacher are refused.
    let mut no_teacher = cfg(serde_json::json!({}));
    no_teacher.teacher = None;
    assert!(no_teacher.validate().unwrap_err().contains("name a teacher"));
}
