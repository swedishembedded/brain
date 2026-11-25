// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight surgery on tiny Qwen3-TTS checkpoints (upstream names, its own
//! declared graph), judged by what the speech objective's teacher-forced
//! forward predicts on the CPU backend: a physically narrower checkpoint
//! predicts what the masked original predicts, a full-width rotation predicts
//! what the original predicts, a trimmed vocabulary predicts the same for
//! sequences of kept tokens, and an export reads back exactly.

use std::collections::BTreeMap;

use model::Model;
use modelopt::surgery::{self, basis_from_covariance, BasisKind, Checkpoint, Tensor};
use qwen3tts::prompt::TextLayout;
use qwen3tts::shard::{Record, RecordId, Shard, ShardMeta, ShardWriter, FORMAT, NO_VOICE};
use qwen3tts::speech_sft::{objective::example, Base, Example, Logits, RowMap, SftShape, SpeechSftModel};
use serde_json::{json, Value};

const CODEBOOK: usize = 16;
const GROUPS: usize = 4;
const CODEC_VOCAB: usize = 48;
const HEAD_DIM: usize = 8;
/// Regular text tokens: the 256 byte tokens and four merges.
const REGULAR: usize = 260;
/// The added tokens, after the regular ones.
const ADDED: [&str; 6] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|tts_pad|>", "<|tts_text_bos|>", "<|tts_text_eod|>"];
const TEXT_VOCAB: usize = REGULAR + ADDED.len();
const IM_START: u32 = REGULAR as u32 + 1;
const IM_END: u32 = REGULAR as u32 + 2;
/// `assistant` is a regular token, as in the real vocabulary.
const ASSISTANT: u32 = 256;
const NEWLINE: u32 = 10;

/// A config in the upstream layout: talker `d` wide with `layers`, code predictor `dc` wide with `cp_layers`.
fn config(d: usize, dc: usize, layers: usize, cp_layers: usize) -> Value {
    let types = |n: usize| vec!["full_attention"; n];
    let code_predictor = json!({
        "num_hidden_layers": cp_layers, "hidden_size": dc, "head_dim": HEAD_DIM, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 24, "vocab_size": CODEBOOK, "num_code_groups": GROUPS, "rope_theta": 1000000, "rms_norm_eps": 1e-6, "layer_types": types(cp_layers),
    });
    let codec = json!({
        "codec_nothink_id": 20, "codec_think_id": 21, "codec_think_bos_id": 22, "codec_think_eos_id": 23, "codec_pad_id": 24, "codec_bos_id": 25, "codec_eos_token_id": 26,
        "codec_language_id": {"alpha": 30, "beta": 31}, "spk_id": {"v1": 40}, "spk_is_dialect": {"v1": false},
    });
    let mut talker = json!({
        "num_hidden_layers": layers, "hidden_size": d, "head_dim": HEAD_DIM, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 32, "vocab_size": CODEC_VOCAB, "num_code_groups": GROUPS, "text_hidden_size": 20, "text_vocab_size": TEXT_VOCAB,
        "rope_theta": 1000000, "rms_norm_eps": 1e-6, "rope_scaling": {"interleaved": true, "mrope_section": [2, 1, 1]},
        "code_predictor_config": code_predictor,
    });
    for (k, v) in codec.as_object().unwrap() {
        talker[k] = v.clone();
    }
    json!({
        "architectures": ["Qwen3TTSForConditionalGeneration"],
        "tts_model_type": "custom_voice",
        "im_start_token_id": IM_START, "im_end_token_id": IM_END, "assistant_token_id": ASSISTANT,
        "tts_pad_token_id": REGULAR + 3, "tts_bos_token_id": REGULAR + 4, "tts_eos_token_id": REGULAR + 5,
        "speaker_encoder_config": {"enc_dim": d, "sample_rate": 24000},
        "talker_config": talker,
    })
}

/// The input width (columns) of an upstream tensor of the tiny config, for its `[rows, cols]` shape.
fn cols(name: &str, c: &Value) -> usize {
    let t = &c["talker_config"];
    let (d, th, dc) = (t["hidden_size"].as_u64().unwrap() as usize, t["text_hidden_size"].as_u64().unwrap() as usize, t["code_predictor_config"]["hidden_size"].as_u64().unwrap() as usize);
    let cp = name.starts_with("talker.code_predictor.model.layers.");
    let width = if cp { dc } else { d };
    let ff = |s: &str| t[s]["intermediate_size"].as_u64().unwrap_or_else(|| t["intermediate_size"].as_u64().unwrap()) as usize;
    let attn = |section: &Value| section["num_attention_heads"].as_u64().unwrap() as usize * HEAD_DIM;
    match () {
        _ if name.ends_with("norm.weight") || name.ends_with(".bias") => 1,
        _ if name.ends_with("o_proj.weight") => attn(if cp { &t["code_predictor_config"] } else { t }),
        _ if name.ends_with("down_proj.weight") => ff(if cp { "code_predictor_config" } else { "" }),
        _ if name.contains("text_embedding") || name.contains("linear_fc") => th,
        _ if name.contains("code_predictor.lm_head") => dc,
        _ => width,
    }
}

/// A tiny random checkpoint of `config` in upstream names, its tensors drawn from `seed`.
fn tiny(config: Value, seed: u64) -> Checkpoint {
    let has_projection = config["talker_config"]["hidden_size"] != config["talker_config"]["code_predictor_config"]["hidden_size"];
    let talker = qwen3tts::config::TalkerConfig::from_json(&config);
    let mtp = qwen3tts::config::MtpConfig::from_json(&config);
    let specials = qwen3tts::prompt::TtsSpecials::from_config(&config).unwrap();
    let shape = SftShape { talker, mtp, specials, batch: 1, block: 1, frames: 1, mtp_weight: 0.3, lr: Default::default(), new_rows: Vec::new(), text_inter: 20, gemm: gpu_core::provider::library::GemmPrecision::Fp32Reference, gt_weight: 1.0, kd: None };
    let init = SpeechSftModel::init_weights(&shape, seed);
    assert_eq!(init.contains_key("talker.code_predictor.small_to_mtp_projection.weight"), has_projection);
    let tensors = init
        .into_iter()
        .map(|(name, data)| {
            let c = cols(&name, &config);
            let t = if c == 1 { Tensor::new(vec![data.len()], data) } else { Tensor::new(vec![data.len() / c, c], data) };
            (name, t.unwrap())
        })
        .collect();
    Checkpoint { config, tensors }
}

fn rows() -> RowMap {
    RowMap { languages: vec![Some(30), Some(31)], voices: vec![40] }
}

/// Three utterances of `texts` (text token ids between the chat framing), codes from `seed`.
fn shard(d: usize, texts: &[&[u32]], seed: u64) -> Shard {
    let meta = ShardMeta { format: FORMAT.into(), codebooks: GROUPS as u32, codebook_size: CODEBOOK as u32, xvector_dim: d as u32, frame_rate: 12.5, languages: vec!["alpha".into(), "beta".into()], voices: vec!["v1".into()], records: 0, producers: Default::default() };
    let mut w = ShardWriter::new(meta);
    let mut rng = data::rng::Lcg::new(seed);
    for (i, text) in texts.iter().enumerate() {
        let ids: Vec<u32> = [IM_START, ASSISTANT, NEWLINE].iter().chain(text.iter()).chain(&[IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]).copied().collect();
        let frames = 3 + i;
        let codes: Vec<u16> = (0..frames * GROUPS).map(|_| (rng.next_u64() % CODEBOOK as u64) as u16).collect();
        let xvector = rng.vec_scaled(d, 0.5);
        let id = RecordId { id: format!("r{i}"), speaker: "s".into(), source: "test".into(), text: String::new() };
        w.push(&Record { id, text_ids: &ids, codes: &codes, xvector: &xvector, voice: if i % 2 == 0 { 0 } else { NO_VOICE }, language: (i % 2) as u32, complete: i != 1 }).unwrap();
    }
    Shard::from_bytes(&w.to_bytes()).unwrap()
}

const TEXTS: [&[u32]; 3] = [&[104, 105, 257, 33], &[258, 32, 120], &[259, 101, 102, 103, 32]];

fn examples(ck: &Checkpoint, texts: &[&[u32]]) -> Vec<Example> {
    let s = shard(ck.config["talker_config"]["hidden_size"].as_u64().unwrap() as usize, texts, 0x5a); // seed: fixed fixture
    let sp = qwen3tts::prompt::TtsSpecials::from_config(&ck.config).unwrap();
    (0..s.len()).map(|r| example(&sp, &rows(), &s, r, if r == 1 { TextLayout::Streaming } else { TextLayout::NonStreaming }).unwrap()).collect()
}

/// What the speech objective's teacher-forced forward predicts for `examples` on checkpoint `ck`.
fn predict(ck: &Checkpoint, examples: &[Example]) -> (Logits, f32) {
    let base = Base::from_parts(&ck.config, &|n| ck.get(n).ok().map(|t| t.shape.iter().map(|&d| d as u64).collect()), &|n| ck.get(n).ok().map(|t| t.data.clone())).unwrap();
    let g = base.mtp.num_code_groups as usize;
    let block = examples.iter().map(|e| e.plan.len() + e.codes.len() / g).max().unwrap() as u32;
    let frames = examples.iter().map(|e| e.codes.len() / g).sum::<usize>() as u32;
    let shape = SftShape { talker: base.talker, mtp: base.mtp, specials: base.specials, batch: examples.len() as u32, block, frames, mtp_weight: 0.3, lr: Default::default(), new_rows: Vec::new(), text_inter: base.text_inter, gemm: gpu_core::provider::library::GemmPrecision::Fp32Reference, gt_weight: 1.0, kd: None };
    let m = SpeechSftModel::build(shape, base.tensors, 1).unwrap(); // seed: the unread token tables only
    m.load(examples.to_vec()).unwrap();
    let stats = m.score();
    (m.logits(), stats.loss(&qwen3tts::speech_sft::LossWeights::sft(0.3)))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn assert_same(what: &str, a: &Logits, b: &Logits, tol: f32) {
    assert_eq!((&a.talker_targets, &a.cp_targets), (&b.talker_targets, &b.cp_targets), "{what}: the same supervised positions");
    let (t, c) = (max_diff(&a.talker, &b.talker), max_diff(&a.cp, &b.cp));
    assert!(t <= tol && c <= tol, "{what}: talker logits differ by {t}, code predictor logits by {c}");
}

fn graph(ck: &Checkpoint) -> model::surgery::SurgeryGraph {
    qwen3tts::surgery::graph(&ck.config, &|n| ck.tensors.contains_key(n)).unwrap()
}

#[test]
fn narrower_channels_and_fewer_layers_predict_what_the_masked_checkpoint_predicts() {
    let ck = tiny(config(24, 16, 3, 2), 0x7a1); // seed: fixed fixture
    let ex = examples(&ck, &TEXTS);
    let g = graph(&ck);
    let mlp = g.dims_matching("talker.layers.*.mlp");
    let cp_mlp = g.dims_matching("code_predictor.layers.*.mlp");
    let kv = g.dims_matching("talker.layers.*.kv_heads");
    let keep_mlp: Vec<Vec<usize>> = (0..mlp.len()).map(|l| (0..32).filter(|c| (c + l) % 4 != 0).collect()).collect();
    let keep_cp: Vec<Vec<usize>> = (0..cp_mlp.len()).map(|l| (0..24).filter(|c| (c * 7 + l) % 4 != 1).collect()).collect();
    let keep_kv: Vec<Vec<usize>> = (0..kv.len()).map(|l| vec![l % 2]).collect();

    let mut masked = ck.clone();
    let mut cut = ck.clone();
    for (dims, keep) in [(&mlp, &keep_mlp), (&cp_mlp, &keep_cp), (&kv, &keep_kv)] {
        surgery::mask_channels(&mut masked, dims, keep).unwrap();
        surgery::select_channels(&mut cut, dims, keep).unwrap();
    }
    surgery::mask_layers(&mut masked, g.stack("talker.layers").unwrap(), &[1]).unwrap();
    surgery::mask_layers(&mut masked, g.stack("code_predictor.layers").unwrap(), &[0]).unwrap();
    // Each edit reads the graph of the checkpoint as it now stands.
    let now = graph(&cut);
    let change = surgery::remove_layers(&mut cut, now.stack("talker.layers").unwrap(), &[1]).unwrap();
    let now = graph(&cut);
    surgery::remove_layers(&mut cut, now.stack("code_predictor.layers").unwrap(), &[0]).unwrap();
    assert_eq!(change.config["talker_config.num_hidden_layers"][1], json!(2));
    assert_eq!(cut.config["talker_config"]["num_attention_heads"], json!(2), "a KV group takes its query heads");
    assert_eq!(cut.config["talker_config"]["code_predictor_config"]["layer_types"], json!(["full_attention"]));
    assert!(cut.params() < ck.params());
    qwen3tts::surgery::check_upstream(&cut.config, &|n| cut.tensors.contains_key(n)).unwrap();
    assert_same("channels and layers", &predict(&masked, &ex).0, &predict(&cut, &ex).0, 1e-4);
}

#[test]
fn a_folded_text_path_predicts_what_the_masked_projection_predicts() {
    let ck = tiny(config(24, 16, 2, 2), 0xf01); // seed: fixed fixture
    let ex = examples(&ck, &TEXTS);
    let g = graph(&ck);
    let fold = g.fold("talker.text").unwrap();
    let keep: Vec<usize> = (0..20).filter(|c| c % 3 != 2).collect();
    let mut masked = ck.clone();
    surgery::mask_channels(&mut masked, &[&fold.hidden], std::slice::from_ref(&keep)).unwrap();
    let mut folded = ck.clone();
    surgery::fold_embedding(&mut folded, fold, &keep).unwrap();
    assert_eq!(folded.config["talker_config"]["text_hidden_size"], json!(keep.len()));
    assert_same("text fold", &predict(&masked, &ex).0, &predict(&folded, &ex).0, 1e-4);
}

#[test]
fn a_trimmed_vocabulary_predicts_the_same_for_sequences_of_kept_tokens() {
    let ck = tiny(config(24, 16, 2, 2), 0xb0c); // seed: fixed fixture
    let g = graph(&ck);
    let used: Vec<u32> = TEXTS.iter().flat_map(|t| t.iter().copied()).chain([IM_START, IM_END, ASSISTANT, NEWLINE]).collect();
    let mut keep: Vec<usize> = (0..256).chain(used.iter().map(|&t| t as usize)).chain(REGULAR..TEXT_VOCAB).collect();
    keep.sort_unstable();
    keep.dedup();
    let mut cut = ck.clone();
    surgery::trim_vocabulary(&mut cut, g.vocabulary("text").unwrap(), &keep).unwrap();
    let new = |t: u32| keep.binary_search(&(t as usize)).unwrap() as u32;
    assert_eq!(cut.config["im_end_token_id"], json!(new(IM_END)));
    assert_eq!(cut.config["tts_eos_token_id"], json!(keep.len() - 1));
    let renumbered: Vec<Vec<u32>> = TEXTS.iter().map(|t| t.iter().map(|&x| new(x)).collect()).collect();
    let refs: Vec<&[u32]> = renumbered.iter().map(Vec::as_slice).collect();
    // The framing ids move too: build the examples of the trimmed checkpoint from its own specials.
    let before = predict(&ck, &examples(&ck, &TEXTS)).0;
    let after = predict(&cut, &examples_renumbered(&cut, &refs, &new)).0;
    assert_same("vocabulary trim", &before, &after, 1e-6);
}

/// [`examples`] with the chat framing ids renumbered by `new`.
fn examples_renumbered(ck: &Checkpoint, texts: &[&[u32]], new: &dyn Fn(u32) -> u32) -> Vec<Example> {
    let s = shard(ck.config["talker_config"]["hidden_size"].as_u64().unwrap() as usize, texts, 0x5a); // seed: fixed fixture
    let sp = qwen3tts::prompt::TtsSpecials::from_config(&ck.config).unwrap();
    let mut out: Vec<Example> = (0..s.len()).map(|r| example(&sp, &rows(), &s, r, if r == 1 { TextLayout::Streaming } else { TextLayout::NonStreaming }).unwrap()).collect();
    // The shard holds the original framing ids; the trimmed checkpoint reads their new numbers.
    for e in &mut out {
        for parts in &mut e.plan.rows {
            for p in parts.iter_mut() {
                if let qwen3tts::prompt::Part::Text(id) = p {
                    if [IM_START, IM_END, ASSISTANT, NEWLINE].contains(id) {
                        *id = new(*id);
                    }
                }
            }
        }
        for id in &mut e.plan.trailing {
            if [IM_START, IM_END, ASSISTANT, NEWLINE].contains(id) {
                *id = new(*id);
            }
        }
    }
    out
}

/// A second moment of the talker's states, from the checkpoint's own forward (any symmetric positive matrix would do).
fn covariance(seed: u64, d: usize) -> Vec<f64> {
    let mut rng = data::rng::Lcg::new(seed);
    let x = rng.vec_scaled(3 * d * d, 1.0);
    let mut c = vec![0f64; d * d];
    for row in x.chunks(d) {
        for i in 0..d {
            for j in 0..d {
                c[i * d + j] += f64::from(row[i]) * f64::from(row[j]);
            }
        }
    }
    c
}

/// `examples` with their speaker x-vectors (computed outside the checkpoint) written into the rotated stream.
fn in_basis(examples: &[Example], basis: &surgery::Basis, k: usize) -> Vec<Example> {
    examples.iter().map(|e| Example { xvector: basis.project(&e.xvector, k), ..e.clone() }).collect()
}

#[test]
fn a_full_width_rotation_predicts_what_the_original_predicts() {
    let ck = tiny(config(24, 16, 2, 2), 0x0ca); // seed: fixed fixture
    let ex = examples(&ck, &TEXTS);
    let want = predict(&ck, &ex).0;
    let g = graph(&ck);
    let (talker, cp) = (g.stream("talker").unwrap(), g.stream("code_predictor").unwrap());

    // The code predictor's stream has only folded norms: exact in any basis.
    let mut rotated = ck.clone();
    surgery::rotate_stream(&mut rotated, cp, &basis_from_covariance(&covariance(3, 16), 16, BasisKind::Pca).unwrap(), 16, &[]).unwrap(); // seed: fixed fixture
    // The talker's final norm is shared with the raw codec embeddings: a permutation keeps it exact.
    let perm = basis_from_covariance(&covariance(4, 24), 24, BasisKind::Identity).unwrap(); // seed: fixed fixture
    surgery::rotate_stream(&mut rotated, talker, &perm, 24, &[]).unwrap();
    assert_same("rotation", &want, &predict(&rotated, &in_basis(&ex, &perm, 24)).0, 1e-4);

    // A general rotation of the talker keeps its own head exact (the gains here are all one, so the code predictor too).
    let mut pca = ck.clone();
    let basis = basis_from_covariance(&covariance(5, 24), 24, BasisKind::Pca).unwrap(); // seed: fixed fixture
    surgery::rotate_stream(&mut pca, talker, &basis, 24, &[]).unwrap();
    let got = predict(&pca, &in_basis(&ex, &basis, 24)).0;
    assert!(max_diff(&want.talker, &got.talker) <= 1e-4, "the talker's head is exact under any rotation");
}

#[test]
fn a_narrower_stream_is_a_working_checkpoint_of_the_new_width() {
    let ck = tiny(config(24, 16, 2, 2), 0x0cb); // seed: fixed fixture
    let mut cut = ck.clone();
    let g = graph(&ck);
    surgery::rotate_stream(&mut cut, g.stream("talker").unwrap(), &basis_from_covariance(&covariance(6, 24), 24, BasisKind::Pca).unwrap(), 20, &[]).unwrap(); // seed: fixed fixture
    let now = graph(&cut);
    surgery::rotate_stream(&mut cut, now.stream("code_predictor").unwrap(), &basis_from_covariance(&covariance(7, 16), 16, BasisKind::Pca).unwrap(), 12, &[]).unwrap(); // seed: fixed fixture
    assert_eq!((cut.config["talker_config"]["hidden_size"].clone(), cut.config["speaker_encoder_config"]["enc_dim"].clone()), (json!(20), json!(20)));
    assert_eq!(cut.get("talker.code_predictor.model.codec_embedding.2.weight").unwrap().shape, vec![CODEBOOK, 20]);
    assert_eq!(cut.get("talker.code_predictor.small_to_mtp_projection.weight").unwrap().shape, vec![12, 20]);
    let narrow = examples(&cut, &TEXTS);
    let (logits, loss) = predict(&cut, &narrow);
    assert!(loss.is_finite() && logits.talker.iter().all(|v| v.is_finite()));
    // Equal widths would make upstream drop the projection this checkpoint needs.
    let mut equal = ck.clone();
    surgery::rotate_stream(&mut equal, graph(&ck).stream("talker").unwrap(), &basis_from_covariance(&covariance(6, 24), 24, BasisKind::Pca).unwrap(), 16, &[]).unwrap(); // seed: fixed fixture
    assert!(qwen3tts::surgery::check_upstream(&equal.config, &|n| equal.tensors.contains_key(n)).unwrap_err().contains("both 16 wide"));
}

/// The files of an upstream release around `ck`, F32 so an export of it is exact.
fn write_release(ck: &Checkpoint, dir: &std::path::Path) {
    std::fs::create_dir_all(dir.join("speech_tokenizer")).unwrap();
    std::fs::write(dir.join("speech_tokenizer/config.json"), "{}").unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&ck.config).unwrap()).unwrap();
    for f in ["generation_config.json", "preprocessor_config.json"] {
        std::fs::write(dir.join(f), "{}").unwrap();
    }
    let mut vocab: BTreeMap<String, usize> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), i)).collect();
    let merges = [("a", "s"), ("as", "s"), ("ass", "i"), ("t", "a")];
    for (a, b) in merges {
        let n = vocab.len();
        vocab.insert(format!("{a}{b}"), n);
    }
    std::fs::write(dir.join("vocab.json"), serde_json::to_vec(&vocab).unwrap()).unwrap();
    std::fs::write(dir.join("merges.txt"), std::iter::once("#version: 0.2".to_string()).chain(merges.iter().map(|(a, b)| format!("{a} {b}"))).map(|l| l + "\n").collect::<String>()).unwrap();
    let added: serde_json::Map<String, Value> = ADDED.iter().enumerate().map(|(i, c)| ((REGULAR + i).to_string(), json!({"content": c, "special": true}))).collect();
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::to_vec(&json!({"added_tokens_decoder": added})).unwrap()).unwrap();
    let list = ck.to_list();
    let plan: Vec<(String, Vec<u64>)> = list.iter().map(|(n, s, _)| (n.clone(), s.clone())).collect();
    let mut w = checkpoint::weightio::StWriter::create(dir.join("model.safetensors").to_str().unwrap(), &plan, &Value::Null, None).unwrap();
    for (n, _, v) in &list {
        w.write(n, v).unwrap();
    }
    w.finish().unwrap();
}

#[test]
fn an_edited_export_reads_back_exactly_and_brain_loads_it() {
    let scratch = brain_testutil::Scratch::dir("modelopt-surgery-export");
    let ck = tiny(config(24, 16, 3, 2), 0xe11); // seed: fixed fixture
    let base = scratch.path().join("base");
    write_release(&ck, &base);
    assert_eq!(Checkpoint::read(&base).unwrap(), ck, "the base reads back as written");

    let mut cut = ck.clone();
    let g = graph(&ck);
    let mlp = g.dims_matching("talker.layers.*.mlp");
    surgery::select_channels(&mut cut, &mlp, &vec![(0..20).collect(); mlp.len()]).unwrap();
    let now = graph(&cut);
    surgery::remove_layers(&mut cut, now.stack("talker.layers").unwrap(), &[2]).unwrap();
    let out = scratch.path().join("edited");
    let list = cut.to_list();
    let files = vec![("merges.txt".to_string(), b"#version: 0.2\n".to_vec())];
    qwen3tts::export::export_edited(&base, &out, "local/edited", &qwen3tts::export::Edited { config: &cut.config, tensors: &list, files: &files }, &mut || true).unwrap();
    assert_eq!(Checkpoint::read(&out).unwrap(), cut, "every edited tensor and the config read back exactly");
    assert_eq!(std::fs::read(out.join("merges.txt")).unwrap(), b"#version: 0.2\n");
    assert_eq!(std::fs::read(out.join("vocab.json")).unwrap(), std::fs::read(base.join("vocab.json")).unwrap());
    // Brain's serving loaders take the narrower release as it is.
    let view = qwen3tts::import::talker_view(&out).unwrap();
    assert_eq!(view.shape("blocks.0.mlp.gate.weight").unwrap(), &[20, 24]);
    let cfg = qwen3tts::config::TalkerConfig::from_json(&cut.config);
    assert_eq!((cfg.n_layers, cfg.d_ff), (2, 20));

    // A tensor the config does not describe is refused, and nothing is left behind.
    let mut wrong = cut.clone();
    wrong.config["talker_config"]["intermediate_size"] = json!(21);
    let bad = scratch.path().join("bad");
    let err = qwen3tts::export::export_edited(&base, &bad, "local/bad", &qwen3tts::export::Edited { config: &wrong.config, tensors: &list, files: &[] }, &mut || true).unwrap_err();
    assert!(err.contains("does not read the edited checkpoint back"), "{err}");
    assert!(!bad.exists());
}

/// The speech objective's base over the tensors of `ck`.
fn base_of(ck: &Checkpoint) -> Base {
    Base::from_parts(&ck.config, &|n| ck.get(n).ok().map(|t| t.shape.iter().map(|&d| d as u64).collect()), &|n| ck.get(n).ok().map(|t| t.data.clone())).unwrap()
}

#[test]
fn a_calibrated_plan_edits_the_checkpoint_and_its_measurement_sees_the_cost() {
    use qwen3tts::speech_sft::calibrate::{calibrate, CalibrationOpts, Inputs};
    let ck = tiny(config(24, 16, 3, 2), 0xca1); // seed: fixed fixture
    let texts: Vec<&[u32]> = (0..6).map(|i| TEXTS[i % 3]).collect();
    let s = shard(24, &texts, 0x5b); // seed: fixed fixture
    let opts = CalibrationOpts { records: 0, batch: 2, mtp_weight: 0.3, layers: true, sample_rows: 64, gemm: gpu_core::provider::library::GemmPrecision::Fp32Reference };
    let (ev, cal) = calibrate(base_of(&ck), &s, "sha256:test", &opts, &mut |_| {}).unwrap();
    assert_eq!((cal.records, cal.layers["talker.layers"].len(), cal.layers["code_predictor.layers"].len()), (6, 3, 2));
    for a in cal.layers.values().flatten() {
        assert!(a.kl.unwrap() >= 0.0 && a.loss.unwrap().is_finite() && (-1.0..=1.0).contains(&a.cosine), "{a:?}");
    }
    assert_eq!((cal.streams["talker"].width, cal.streams["code_predictor"].width), (24, 16));
    assert!(cal.streams["talker"].shared.contains_key("talker.model.norm.weight"));
    assert_eq!((cal.channels["talker.layers.2.mlp"].sum_sq.len(), cal.channels["talker.text"].sum_sq.len()), (32, 20));

    // The calibration file reads back as written, to f32.
    let scratch = brain_testutil::Scratch::dir("modelopt-calibration");
    let path = scratch.path().join("calibration.safetensors");
    surgery::calibration::save(&path, &cal).unwrap();
    let back = surgery::calibration::load(&path).unwrap();
    assert_eq!((back.records, back.source.as_str()), (cal.records, "sha256:test"));
    for (stack, layers) in &cal.layers {
        for (a, b) in layers.iter().zip(&back.layers[stack]) {
            assert!((a.kl.unwrap() - b.kl.unwrap()).abs() <= 1e-12 && (a.cosine - b.cosine).abs() <= 1e-12, "{stack}: {a:?} vs {b:?}");
        }
    }
    let (a, b) = (&cal.streams["talker"].covariance, &back.streams["talker"].covariance);
    assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-6 * x.abs().max(1.0)));

    // A plan that keeps everything measures what the unedited checkpoint predicts.
    let m = surgery::plan::Model { graph: &|c: &Checkpoint| qwen3tts::surgery::graph(&c.config, &|n| c.tensors.contains_key(n)), check: &|c: &Checkpoint| qwen3tts::surgery::check_upstream(&c.config, &|n| c.tensors.contains_key(n)) };
    let all: Vec<surgery::plan::Edit> = serde_json::from_str(r#"[{"select_channels": {"dims": "talker.layers.*.mlp", "keep": 32, "score": "wanda"}}]"#).unwrap();
    let mut same = ck.clone();
    surgery::plan::apply(&mut same, &all, &back, &BTreeMap::new(), &m).unwrap();
    let unchanged = ev.measure(base_of(&same), &Inputs::default()).unwrap();
    assert!(unchanged.talker_kl < 1e-9 && unchanged.code_predictor_kl < 1e-9, "{unchanged:?}");

    let plan: Vec<surgery::plan::Edit> = serde_json::from_str(
        r#"[{"remove_layers": {"stack": "talker.layers", "count": 1, "by": "kl"}},
            {"select_channels": {"dims": "talker.layers.*.mlp", "keep": 16, "score": "wanda"}},
            {"rotate_stream": {"stream": "code_predictor", "width": 12, "basis": "pca"}}]"#,
    )
    .unwrap();
    let mut edited = ck.clone();
    let applied = surgery::plan::apply(&mut edited, &plan, &back, &BTreeMap::new(), &m).unwrap();
    let order: Vec<&str> = applied.changes.iter().map(|(c, _)| c.edit.as_str()).collect();
    assert_eq!(order, ["select_channels", "rotate_stream", "remove_layers"], "a plan applies its edits in a fixed order");
    let cheapest = cal.layers["talker.layers"].iter().enumerate().min_by(|a, b| a.1.kl.unwrap().total_cmp(&b.1.kl.unwrap())).unwrap().0;
    assert!(!applied.changes[2].0.kept["talker.layers"].contains(&cheapest), "the layer whose removal costs least goes");
    let cost = ev.measure(base_of(&edited), &Inputs::default()).unwrap();
    assert!(cost.talker_kl > 1e-6 && cost.loss.is_finite(), "{cost:?}");
    assert!(edited.params() < ck.params());
}

/// The talker's final norm is shared with the raw codec embeddings (both enter `small_to_mtp_projection`): a general
/// rotation cannot keep both inputs of that projection exact, a rotation within channels of equal gain does.
#[test]
fn a_pca_rotation_within_equal_gains_keeps_the_shared_norm_exact() {
    let mut ck = tiny(config(24, 16, 2, 2), 0x9a1); // seed: fixed fixture
    let gains = [1.0f32, 2.5, -0.5, 4.0];
    ck.get_mut("talker.model.norm.weight").unwrap().data = (0..24).map(|i| gains[i % 4]).collect();
    let ex = examples(&ck, &TEXTS);
    let want = predict(&ck, &ex).0;
    let cov = covariance(8, 24); // seed: fixed fixture
    let stats = model::surgery::StreamStats { width: 24, covariance: cov.clone(), rows: 1, shared: BTreeMap::new() };
    let cal = model::surgery::Calibration { streams: BTreeMap::from([("talker".to_string(), stats)]), ..Default::default() };
    let m = surgery::plan::Model { graph: &|c: &Checkpoint| qwen3tts::surgery::graph(&c.config, &|n| c.tensors.contains_key(n)), check: &|_| Ok(()) };
    let plan: Vec<surgery::plan::Edit> = serde_json::from_str(r#"[{"rotate_stream": {"stream": "talker", "width": 24, "basis": "pca"}}]"#).unwrap();
    let mut grouped = ck.clone();
    let applied = surgery::plan::apply(&mut grouped, &plan, &cal, &BTreeMap::new(), &m).unwrap();
    let (basis, k) = &applied.inputs.streams["talker"];
    assert_same("pca within equal gains", &want, &predict(&grouped, &in_basis(&ex, basis, *k)).0, 1e-4);

    let mut general = ck.clone();
    let b = basis_from_covariance(&cov, 24, BasisKind::Pca).unwrap();
    surgery::rotate_stream(&mut general, graph(&ck).stream("talker").unwrap(), &b, 24, &[]).unwrap();
    let got = predict(&general, &in_basis(&ex, &b, 24)).0;
    assert!(max_diff(&want.talker, &got.talker) <= 1e-4 && max_diff(&want.cp, &got.cp) > 1e-3, "a general rotation keeps the head exact and moves the code predictor's input");
}
