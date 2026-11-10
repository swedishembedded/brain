// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the record schema refuses what it must; the objective's targets never turn a missing label into a
//! negative; the objective's backward equals finite differences; and the trainer fine-tunes in every mode,
//! resumes exactly, and writes checkpoints and adapters that load again.

use data::rng::Lcg;
use mirage::config::MirageConfig;
use mirage::train::adapter;
use mirage::train::data::Record;
use mirage::train::net::{params_from_checkpoint, Net, Params};
use mirage::train::objective::{sample_loss, text_target, GtObject, GtText, ReaderLine, Sample, TrainWeights};
use mirage::train::trainer::{Head, Mode, Prepared, TrainConfig, Trainer};

use crate::common::tiny_checkpoint;
use crate::fd::model_check;

#[test]
fn the_schema_accepts_a_record_and_refuses_what_cannot_be_trusted() {
    let ok = r#"{"image":"images/a.jpg","scene":{"caption":"c"},"objects":[{"id":1,"name":"clock","bbox":[10,10,200,300],"mask_polygons":[[10,10,200,10,100,300]]}],"text":[],"license":"x"}"#;
    let rec = Record::parse(ok).unwrap();
    assert_eq!(rec.version, 1);
    assert_eq!(rec.language(), "en");
    assert!(rec.relations.is_none() && rec.text.as_ref().is_some_and(Vec::is_empty), "an empty list is labelled, a missing key is not");
    for (bad, why) in [
        (r#"{"version":2,"image":"a.jpg"}"#, "version"),
        (r#"{"image":"../a.jpg"}"#, "path"),
        (r#"{"image":"/etc/passwd"}"#, "absolute path"),
        (r#"{"image":"a.jpg","objects":[{"id":1,"name":"x","bbox":[300,10,200,300]}]}"#, "unordered box"),
        (r#"{"image":"a.jpg","objects":[{"id":1,"name":"x","bbox":[0,0,1001,5]}]}"#, "box outside the image"),
        (r#"{"image":"a.jpg","objects":[{"id":1,"name":"x","bbox":[0,0,5,5]},{"id":1,"name":"y","bbox":[0,0,5,5]}]}"#, "duplicate id"),
        (r#"{"image":"a.jpg","relations":[{"subject":1,"predicate":"on","object":2}]}"#, "relations without objects"),
        (r#"{"image":"a.jpg","objects":[],"relations":[{"subject":1,"predicate":"on","object":2}]}"#, "relation to a missing object"),
        (r#"{"image":"a.jpg","mystery":1}"#, "unknown field"),
        (r#"{"image":"a.jpg","objects":[{"id":1,"name":"x","bbox":[0,0,5,5],"mask_polygons":[[1,2,3]]}]}"#, "short polygon"),
    ] {
        assert!(Record::parse(bad).is_err(), "{why} must be refused");
    }
}

fn encode(s: &str) -> Vec<u32> {
    s.bytes().map(|b| 3 + u32::from(b) % 20).collect()
}

fn base_sample() -> Sample {
    Sample { canvas: vec![0.0; 3], rh: 400, rw: 300, caption: None, objects: None, text: None, relations: None, closed: false, language: "en".into() }
}

fn object(id: u32, name: &str) -> GtObject {
    GtObject { id, name: name.into(), bbox: [0.1, 0.1, 0.5, 0.6], polygons: None, position: None, attributes: Vec::new() }
}

#[test]
fn a_missing_label_family_is_never_a_negative() {
    let cfg = MirageConfig::tiny();
    let vocab = (cfg.text + mirage::config::LANGUAGE_TAGS) as u32;
    let target = |s: &Sample| text_target(&cfg, s, &encode, cfg.text as u32, vocab);

    // Nothing labelled: no target at all.
    assert!(target(&base_sample()).is_none());

    // Objects labelled, scene not: the first token after the prompt (the choice between a scene line and an
    // entity line) is not supervised; the entity's own tokens and its reference token are.
    let mut s = base_sample();
    s.objects = Some(vec![object(1, "cat")]);
    let t = target(&s).unwrap();
    assert_eq!(&t.tokens[..2], &[cfg.bos, cfg.text as u32], "prompt: bos then the language tag");
    assert_eq!(t.weights[0], 0.0, "the language tag is part of the prompt");
    assert_eq!(t.weights[1], 0.0, "the first line's first token follows an unlabelled section");
    assert!(t.weights[2..].iter().all(|w| *w == 1.0));
    assert!(t.tokens.contains(&vocab), "the object's reference token is a target");
    assert!(!t.tokens.contains(&cfg.eos), "an open record never supervises the stop token");

    // A closed record also supervises the stop token.
    s.closed = true;
    let t = target(&s).unwrap();
    assert_eq!(*t.tokens.last().unwrap(), cfg.eos);
    assert_eq!(*t.weights.last().unwrap(), 1.0);
    assert_eq!(t.weights.len(), t.tokens.len() - 1);

    // Scene labelled, objects not, text labelled: the T line follows a gap, so its first token is masked.
    let mut s = base_sample();
    s.caption = Some("a".into());
    s.text = Some(vec![GtText { bbox: [0.1, 0.1, 0.3, 0.2], value: "ab".into(), line: None }]);
    let t = target(&s).unwrap();
    let scene_len = encode("S caption=a").len();
    let newline = encode("\n").len();
    let first_t = 1 + scene_len + newline; // weights[i] is the prediction of tokens[i + 1]; the prompt is two tokens
    assert_eq!(t.weights[first_t], 0.0, "the first token of T after an unlabelled E section is not a target");
    assert_eq!(t.weights[first_t + 1], 1.0);
}

fn tiny_sample(cfg: &MirageConfig) -> Sample {
    let mut rng = Lcg::new(31); // seed: fixed filler for a tiny training sample
    let mut s = base_sample();
    s.canvas = (0..cfg.canvas * cfg.canvas * 3).map(|_| rng.signed()).collect();
    s.caption = Some("a b".into());
    let mut o1 = object(1, "cat");
    o1.polygons = Some(vec![vec![0.1, 0.1, 0.5, 0.15, 0.45, 0.6, 0.12, 0.55]]);
    let mut o2 = object(2, "dog");
    o2.bbox = [0.2, 0.3, 0.7, 0.7];
    o2.polygons = Some(vec![vec![0.25, 0.35, 0.65, 0.35, 0.65, 0.65, 0.25, 0.65]]);
    s.objects = Some(vec![o1, o2]);
    let window: Vec<u8> = (0..48 * 320 * 3).map(|i| ((i * 29 + 7) % 251) as u8).collect();
    s.text = Some(vec![GtText { bbox: [0.05, 0.7, 0.45, 0.8], value: "ab".into(), line: Some(ReaderLine { windows: vec![window], ranges: vec![(0, 80)], labels: Some(vec![3, 5, 5]) }) }]);
    s.relations = Some(vec![(0, "near".into(), 1)]);
    s.closed = true;
    s
}

fn prepared(cfg: &MirageConfig) -> Prepared {
    let sample = tiny_sample(cfg);
    let vocab = (cfg.text + mirage::config::LANGUAGE_TAGS) as u32;
    let target = text_target(cfg, &sample, &encode, cfg.text as u32, vocab);
    assert!(target.is_some());
    Prepared { sample, target }
}

fn scaled_params(ck: &mirage::import::Checkpoint, factor: f64) -> Params<f64> {
    let mut p: Params<f64> = params_from_checkpoint(ck);
    p.values_mut().flatten().for_each(|v| *v *= factor);
    p
}

#[test]
fn the_objective_matches_finite_differences_and_every_term_is_live() {
    crate::common::cap_threads();
    let ck = tiny_checkpoint();
    let mut cfg = ck.config.clone();
    cfg.head_grad = 1.0;
    let p = prepared(&cfg);
    let params = scaled_params(&ck, 0.5);
    let weights = TrainWeights::default();
    let check = model_check(&cfg, &params, &|net| sample_loss(net, &p.sample, p.target.as_ref(), &weights).expect("the sample has labels").total);
    eprintln!("objective finite-difference check: worst {:.3e} at `{}`; unresolvable {:?}; unreached {:?}", check.worst, check.worst_name, check.unresolvable, check.unreached);
    assert!(check.worst < 1e-3, "objective backward deviates by {:e} at `{}`", check.worst, check.worst_name);
    // The loss here is large (an untrained fixture), so the smallest decoder-internal derivatives are below what a
    // difference quotient resolves (the whole-model check covers them at a smaller loss); every encoder, head,
    // reader and slot tensor is resolved and agrees.
    assert!(check.unresolvable.iter().all(|n| n.starts_with("stack/")), "unresolvable outside the decoder stack: {:?}", check.unresolvable);

    let mut net = Net::<f64>::new(&cfg, &params, None);
    let l = sample_loss(&mut net, &p.sample, p.target.as_ref(), &weights).unwrap();
    for term in ["heat_object", "heat_text", "reg_object", "reg_text", "mask", "text", "ctc"] {
        assert!(l.terms.get(term).is_some_and(|v| v.is_finite() && *v > 0.0), "term `{term}` is missing or zero: {:?}", l.terms);
    }
}

fn cfg_for(mode: Mode) -> TrainConfig {
    let mut c = TrainConfig::new(mode);
    c.lr = 2e-3;
    c
}

fn tiny_trainer(mode: Mode) -> (Trainer, Prepared) {
    crate::common::cap_threads();
    let ck = tiny_checkpoint();
    let p = prepared(&ck.config);
    (Trainer::new(ck, cfg_for(mode)).unwrap(), p)
}

#[test]
fn a_full_fine_tune_overfits_a_fixed_sample() {
    let (mut t, p) = tiny_trainer(Mode::Full);
    let batch = std::slice::from_ref(&p);
    let before = t.eval_loss(batch);
    let mut last = before;
    for _ in 0..30 {
        last = t.step(batch).unwrap().loss;
    }
    let after = t.eval_loss(batch);
    eprintln!("full fine-tune on one tiny sample: loss {before:.4} -> {after:.4} (last step {last:.4})");
    assert!(after < 0.7 * before, "held loss fell only from {before} to {after}");
}

#[test]
fn a_head_only_run_changes_only_its_heads() {
    let (mut t, p) = tiny_trainer(Mode::Heads(vec![Head::Reader]));
    let base = tiny_checkpoint();
    for _ in 0..3 {
        t.step(std::slice::from_ref(&p)).unwrap();
    }
    let trained = t.checkpoint();
    let mut changed = Vec::new();
    for (name, tensor) in &base.tensors {
        if trained.tensors[name].data != tensor.data {
            changed.push(name.clone());
        }
    }
    assert!(!changed.is_empty() && changed.iter().all(|n| n.starts_with("reader/")), "head-only run changed {changed:?}");
    assert!(t.trainable().iter().all(|n| n.starts_with("reader/")));
}

#[test]
fn a_lora_adapter_is_a_no_op_at_init_trains_alone_and_merges_exactly() {
    let ck = tiny_checkpoint();
    let (mut t, p) = tiny_trainer(Mode::Lora { rank: 2, alpha: 4.0, seed: 5 });
    t.set_lr(2e-2);
    let batch = std::slice::from_ref(&p);
    let base_trainer = Trainer::new(ck.clone(), cfg_for(Mode::Full)).unwrap();
    assert_eq!(t.eval_loss(batch), base_trainer.eval_loss(batch), "B starts at zero, so a fresh adapter changes nothing");
    let before = t.eval_loss(batch);
    for _ in 0..20 {
        t.step(batch).unwrap();
    }
    let after = t.eval_loss(batch);
    eprintln!("lora on one tiny sample: loss {before:.4} -> {after:.4}");
    assert!(after < 0.97 * before, "the adapter did not learn: {before} -> {after}");
    assert_eq!(t.checkpoint().tensors, ck.tensors, "the base stays untouched");
    // Merge == live apply: a trainer over the folded weights computes the live adapter's loss.
    let merged = Trainer::new(t.merged().unwrap(), cfg_for(Mode::Full)).unwrap();
    let (live, folded) = (t.eval_loss(batch), merged.eval_loss(batch));
    assert!((live - folded).abs() <= 1e-4 * live.abs().max(1.0), "live {live} vs merged {folded}");
}

#[test]
fn the_adapter_file_round_trips_and_a_foreign_file_is_refused() {
    let cfg = MirageConfig::tiny();
    let mut set = adapter::fresh(&cfg, 2, 4.0, 9);
    set.pairs.values_mut().for_each(|(_, b)| b.iter_mut().enumerate().for_each(|(i, v)| *v = (i % 7) as f32 * 0.01));
    let dir = scratch("adapter");
    let path = dir.join("a.safetensors");
    adapter::save(&path, &set, &cfg).unwrap();
    let back = adapter::load(&path, &cfg).unwrap();
    assert_eq!(back.pairs, set.pairs);
    assert_eq!((back.rank, back.scale), (set.rank, set.scale));
    let ck_path = dir.join("weights.safetensors");
    mirage::import::export(&tiny_checkpoint(), &ck_path).unwrap();
    assert!(adapter::load(&ck_path, &cfg).is_err(), "a weights file is not an adapter");
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mirage-train-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn exact_resume(mode: Mode, label: &str) {
    let (mut straight, p) = tiny_trainer(mode.clone());
    let batch = std::slice::from_ref(&p);
    for _ in 0..4 {
        straight.step(batch).unwrap();
    }
    let (mut first, _) = tiny_trainer(mode.clone());
    for _ in 0..2 {
        first.step(batch).unwrap();
    }
    let dir = scratch(label);
    first.save(&dir).unwrap();
    let mut resumed = Trainer::resume(&dir, cfg_for(mode)).unwrap();
    assert_eq!(resumed.steps(), 2);
    for _ in 0..2 {
        resumed.step(batch).unwrap();
    }
    assert_eq!(resumed.checkpoint().tensors, straight.checkpoint().tensors, "weights after resume differ from the uninterrupted run");
    assert_eq!(resumed.adapter().map(|a| &a.pairs), straight.adapter().map(|a| &a.pairs), "adapter after resume differs");
    assert_eq!(resumed.eval_loss(batch), straight.eval_loss(batch));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_resumed_full_run_ends_exactly_where_the_uninterrupted_run_does() {
    exact_resume(Mode::Full, "resume-full");
}

#[test]
fn a_resumed_lora_run_ends_exactly_where_the_uninterrupted_run_does() {
    exact_resume(Mode::Lora { rank: 2, alpha: 4.0, seed: 5 }, "resume-lora");
}

#[test]
fn a_saved_checkpoint_loads_through_the_importer_unchanged() {
    let (mut t, p) = tiny_trainer(Mode::Heads(vec![Head::Dense]));
    t.step(std::slice::from_ref(&p)).unwrap();
    let dir = scratch("export");
    t.save(&dir).unwrap();
    let loaded = mirage::import::load(&dir.join("mirage.safetensors"), &mirage::import::ImportOptions::default()).unwrap();
    assert_eq!(loaded.tensors, t.checkpoint().tensors);
    assert_eq!(loaded.config, MirageConfig::tiny());
    std::fs::remove_dir_all(&dir).unwrap();
}
