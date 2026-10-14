// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tiny random-weight checkpoints of both decision architectures.
//!
//! They exist so everything that sits on a checkpoint - the loaders, the
//! capability, the residency adapter, the agreement gate between the SDK and
//! the served path - can be exercised on every machine, with no downloaded
//! weights. They are structurally real (the same files, names, tokenizer
//! shape and coverage checks a pulled checkpoint goes through) and
//! semantically meaningless: the answers are noise, which is why a gate over
//! them compares two paths to each other, never to a human judgement.
//!
//! Swedish Embedded AB implements verification harnesses that keep a serving
//! contract honest on every machine, weights or not. If your team needs
//! expertise in testing model-serving stacks, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A scratch directory that deletes itself when dropped.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("brain-decisions-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch directory");
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

/// The tiny `decide` encoder shape: real tier-independent dimensions, a
/// vocabulary just wide enough for [`write_decide`]'s tokenizer.
pub fn decide_config() -> decide::config::EncoderConfig {
    decide::config::EncoderConfig { vocab: 128, max_positions: 256, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 64, type_vocab: 2, eps: 1e-12 }
}

/// A WordPiece tokenizer over lowercase letters, digits and punctuation, with
/// every character also available as a `##` continuation, so any ASCII text
/// tokenizes into ids below [`decide_config`]'s vocabulary.
fn decide_tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    for (i, t) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]"].iter().enumerate() {
        vocab.insert((*t).to_string(), serde_json::json!(i));
    }
    let mut next = 4u32;
    for c in "abcdefghijklmnopqrstuvwxyz0123456789".chars() {
        vocab.insert(c.to_string(), serde_json::json!(next));
        vocab.insert(format!("##{c}"), serde_json::json!(next + 36));
        next += 1;
    }
    serde_json::json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 2, "content": "[CLS]", "special": true},
            {"id": 3, "content": "[SEP]", "special": true}
        ],
        "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
            ],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
            }
        },
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": vocab}
    })
    .to_string()
}

/// Write a complete `decide` decision model (encoder AND head, in this
/// architecture's own naming) into `dir`, with weights drawn from `seed`.
pub fn write_decide(dir: &Path, seed: u64) {
    std::fs::create_dir_all(dir).expect("create decide fixture directory");
    let cfg = decide_config();
    let mut config = cfg.to_json();
    config.as_object_mut().expect("config is an object").insert("head".into(), serde_json::json!("option-head"));

    let shapes: std::collections::HashMap<String, Vec<usize>> =
        cfg.tensor_manifest().into_iter().chain(decide::head::tensor_manifest(&cfg)).collect();
    let enc = decide::init::init_weights(&cfg, seed);
    let head = decide::init::init_head(&cfg, seed + 1);
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (name, shape) in cfg.tensor_manifest() {
        let data = enc.get(&name).unwrap_or_else(|| panic!("fixture: encoder init missing {name}")).clone();
        tensors.push((name, shape.iter().map(|&d| d as u64).collect(), data));
    }
    for (name, _) in decide::head::tensor_manifest(&cfg) {
        let data = head.get(&name).unwrap_or_else(|| panic!("fixture: head init missing {name}")).clone();
        let shape = shapes[&name].iter().map(|&d| d as u64).collect();
        tensors.push((name, shape, data));
    }
    let card = checkpoint::st::ModelCard {
        schema_version: 1,
        id: "decision-fixture".into(),
        family: "decide".into(),
        architecture: Some("minilm-l6-decision".into()),
        ..Default::default()
    };
    checkpoint::save_carded(dir.join("model.safetensors").to_str().expect("utf-8 path"), config.clone(), &tensors, &card);
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config).expect("config json")).expect("write config.json");
    std::fs::write(dir.join("tokenizer.json"), decide_tokenizer_json()).expect("write tokenizer.json");
}

/// Write a Laya-shaped checkpoint into `dir`: `encoder/config.json`,
/// `rl_agent_config.json`, `tokenizer/tokenizer.json` (a merge-free byte-level
/// BPE) and `model.safetensors` named the way the real release names its
/// tensors, so `modernbert::import_dir`'s two-way coverage check runs for
/// real. `max_len` is the trained context the checkpoint declares.
pub fn write_laya(dir: &Path, max_len: u32) {
    write_laya_trunk(dir, max_len, false);
}

/// [`write_laya`] with a trunk whose attention carries information between
/// tokens: every block's value and output projections are the identity instead
/// of weights near zero. At the usual random init the trunk's contribution is a
/// rounding error beside the token embeddings, so every option marker (all the
/// same `[MASK]` token) leaves it as the same vector and no head, however
/// trained, can tell the options apart. A pretrained trunk does not have that
/// problem, and this is the fixture standing in for one when a test needs a
/// head to be able to learn through a frozen trunk.
pub fn write_laya_mixing(dir: &Path, max_len: u32) {
    write_laya_trunk(dir, max_len, true);
}

fn write_laya_trunk(dir: &Path, max_len: u32, mixing: bool) {
    std::fs::create_dir_all(dir.join("encoder")).expect("create encoder directory");
    std::fs::create_dir_all(dir.join("tokenizer")).expect("create tokenizer directory");

    let encoder_cfg = serde_json::json!({
        "model_type": "modernbert", "vocab_size": 300, "hidden_size": 64, "num_hidden_layers": 4,
        "num_attention_heads": 4, "intermediate_size": 19, "max_position_embeddings": 64.max(max_len),
        "global_attn_every_n_layers": 2, "local_attention": 6,
    });
    std::fs::write(dir.join("encoder").join("config.json"), encoder_cfg.to_string()).expect("write encoder config");
    std::fs::write(
        dir.join("rl_agent_config.json"),
        serde_json::json!({
            "head_layers": 2, "max_len": max_len, "head_max_len": 32, "max_prefixes": 6,
            "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
        })
        .to_string(),
    )
    .expect("write rl_agent_config.json");

    let mut vocab = serde_json::Map::new();
    for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::json!(i));
    }
    let specials = [("[UNK]", 256u32), ("[CLS]", 257), ("[SEP]", 258), ("[PAD]", 259), ("[MASK]", 260)];
    let added_tokens: Vec<serde_json::Value> = specials.iter().map(|(c, id)| serde_json::json!({"content": c, "id": id})).collect();
    std::fs::write(
        dir.join("tokenizer").join("tokenizer.json"),
        serde_json::json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added_tokens}).to_string(),
    )
    .expect("write tokenizer.json");

    let cfg = modernbert::ModernBertConfig::from_hf_json(&encoder_cfg).expect("fixture encoder config");
    let laya_cfg = modernbert::LayaConfig::new(cfg.d_model);
    let mut enc_init = modernbert::init::init_weights(&cfg, 7);
    if mixing {
        let h = cfg.d_model as usize;
        for l in 0..cfg.n_layers as usize {
            let identity = |w: &mut Vec<f32>, row0: usize| (0..h).for_each(|i| w[(row0 + i) * h + i] = 1.0);
            let qkv = enc_init.get_mut(&format!("blocks.{l}.qkv.weight")).expect("fixture: qkv");
            identity(qkv, 2 * h);
            let mut proj = vec![0.0; h * h];
            identity(&mut proj, 0);
            enc_init.insert(format!("blocks.{l}.proj.weight"), proj);
        }
    }
    let head_init = modernbert::init::init_weights_laya(&laya_cfg, 8);

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for (name, shape) in cfg.tensor_manifest() {
        let data = enc_init.get(&name).unwrap_or_else(|| panic!("fixture: init missing {name}")).clone();
        tensors.push((encoder_brain_name_to_hf(&name), shape, data));
    }
    for (name, shape) in modernbert::laya::tensor_manifest(&laya_cfg) {
        let data = head_init.get(&name).unwrap_or_else(|| panic!("fixture: init missing {name}")).clone();
        tensors.push((head_brain_name_to_hf(&name), shape, data));
    }
    tensors.push(("temperature".to_string(), vec![3], vec![1.0, 1.0, 1.0]));
    write_hf_safetensors_f32(&dir.join("model.safetensors"), &tensors);
}

/// A trained-head directory in the layout of [`crate::laya_head`], attached to
/// the stock Laya checkpoint `base` (addressed as `base_id`): a head drawn from
/// `head_seed` standing in for a trained one, the base reference and the files
/// copied from the base. Carries no manifest; a caller that needs one writes it.
pub fn write_laya_head_only(dir: &Path, base: &Path, base_id: &str, head_seed: u64) {
    std::fs::create_dir_all(dir).expect("create head-only fixture directory");
    let laya_cfg = modernbert::LayaConfig::new(64);
    let head = modernbert::init::init_weights_laya(&laya_cfg, head_seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = modernbert::laya::tensor_manifest(&laya_cfg)
        .into_iter()
        .map(|(name, shape)| {
            let data = head.get(&name).unwrap_or_else(|| panic!("fixture: init missing {name}")).clone();
            (name, shape.iter().map(|&d| d as u64).collect(), data)
        })
        .collect();
    checkpoint::save_carded(dir.join(crate::laya_head::HEAD_FILE).to_str().expect("utf-8 path"), serde_json::json!({}), &tensors, &checkpoint::st::ModelCard::default());
    let base_ref = crate::laya_head::BaseRef::of(base_id, base).expect("fixture base reference");
    crate::laya_head::write_layout(dir, base, &base_ref).expect("fixture head-only layout");
}

/// A raw Hugging Face `.safetensors` file (`[u64 header_len][JSON header]
/// [blob]`, F32 throughout), hand-assembled because the workspace writer
/// produces brain's own carded container, not a foreign one.
pub fn write_hf_safetensors_f32(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, data) in tensors {
        let start = blob.len();
        for v in data {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, blob.len()]}));
    }
    let hbytes = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header json");
    let mut buf = Vec::new();
    buf.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&hbytes);
    buf.extend_from_slice(&blob);
    std::fs::write(path, buf).expect("write safetensors");
}

/// A brain encoder parameter name mapped to the real checkpoint's own dotted
/// name - `modernbert::import`'s name table reversed.
fn encoder_brain_name_to_hf(name: &str) -> String {
    match name {
        "tok.weight" => return "encoder.embeddings.tok_embeddings.weight".to_string(),
        "emb_norm.weight" => return "encoder.embeddings.norm.weight".to_string(),
        "final_norm.weight" => return "encoder.final_norm.weight".to_string(),
        _ => {}
    }
    let rest = name.strip_prefix("blocks.").unwrap_or_else(|| panic!("unexpected encoder param {name}"));
    let (l, leaf) = rest.split_once('.').expect("block leaf");
    let hf_leaf = match leaf {
        "attn_norm.weight" => "attn_norm.weight",
        "qkv.weight" => "attn.Wqkv.weight",
        "proj.weight" => "attn.Wo.weight",
        "mlp_norm.weight" => "mlp_norm.weight",
        "mlp.wi.weight" => "mlp.Wi.weight",
        "mlp.wo.weight" => "mlp.Wo.weight",
        other => panic!("fixture: unmapped encoder param leaf {other}"),
    };
    format!("encoder.layers.{l}.{hf_leaf}")
}

fn head_brain_name_to_hf(name: &str) -> String {
    if name == "type_emb.weight" {
        return "type_emb.weight".to_string();
    }
    if let Some(rest) = name.strip_prefix("head.") {
        let (l, leaf) = rest.split_once('.').expect("head leaf");
        let hf_leaf = match leaf {
            "attn.in_proj.weight" => "self_attn.in_proj_weight",
            "attn.in_proj.bias" => "self_attn.in_proj_bias",
            "attn.out_proj.weight" => "self_attn.out_proj.weight",
            "attn.out_proj.bias" => "self_attn.out_proj.bias",
            "ff1.weight" => "linear1.weight",
            "ff1.bias" => "linear1.bias",
            "ff2.weight" => "linear2.weight",
            "ff2.bias" => "linear2.bias",
            "norm1.weight" => "norm1.weight",
            "norm1.bias" => "norm1.bias",
            "norm2.weight" => "norm2.weight",
            "norm2.bias" => "norm2.bias",
            other => panic!("fixture: unmapped head param leaf {other}"),
        };
        return format!("head.layers.{l}.{hf_leaf}");
    }
    match name {
        "scorer.norm.weight" => "scorer.0.weight",
        "scorer.norm.bias" => "scorer.0.bias",
        "scorer.fc1.weight" => "scorer.1.weight",
        "scorer.fc1.bias" => "scorer.1.bias",
        "scorer.fc2.weight" => "scorer.3.weight",
        "scorer.fc2.bias" => "scorer.3.bias",
        "act.fc1.weight" => "act_head.0.weight",
        "act.fc1.bias" => "act_head.0.bias",
        "act.fc2.weight" => "act_head.2.weight",
        "act.fc2.bias" => "act_head.2.bias",
        other => panic!("fixture: unmapped head param {other}"),
    }
    .to_string()
}

/// A synthetic decision dataset whose right answer is known: the state names
/// one of four problems and the correct option is the desk that handles it.
/// Every option set is the same four desks in a fresh order, so position
/// tells nothing, and every state is unique, so no example repeats across
/// splits.
#[derive(Clone, Debug)]
pub struct SyntheticTask {
    pub train: usize,
    pub validation: usize,
    pub test: usize,
    /// Seeds the problem drawn per example and the option order.
    pub seed: u64,
    /// Replace the TRAIN labels with random desks, leaving validation and test
    /// truthful: the negative control for "did the model learn the task".
    pub shuffle_train_labels: bool,
    /// The family the examples report; give two tasks different families to
    /// hold one family out of another's data.
    pub family: &'static str,
    /// Rotate which desk handles which problem by this many places, so a model
    /// that learned another task's mapping is wrong on this one.
    pub label_shift: usize,
    /// The number the first example's task, run and seed are counted from;
    /// give two tasks that share a file different ranges.
    pub first_id: u64,
}

impl SyntheticTask {
    pub fn new(train: usize, validation: usize, test: usize) -> SyntheticTask {
        SyntheticTask { train, validation, test, seed: 1, shuffle_train_labels: false, family: "desks", label_shift: 0, first_id: 1 }
    }
}

const PROBLEMS: [(&str, &str); 4] = [("invoice", "billing"), ("router", "network"), ("backup", "storage"), ("mailbox", "email")];

/// The JSONL of a [`SyntheticTask`], in the exporter's line format.
pub fn synthetic_jsonl(task: &SyntheticTask) -> String {
    let mut rng = data::rng::Rng::new(task.seed);
    let mut lines = Vec::new();
    let mut next = task.first_id - 1;
    for (split, count) in [("train", task.train), ("validation", task.validation), ("test", task.test)] {
        for _ in 0..count {
            next += 1;
            let problem = (rng.next_u64() % PROBLEMS.len() as u64) as usize;
            let truth = (problem + task.label_shift) % PROBLEMS.len();
            let label = if task.shuffle_train_labels && split == "train" { (rng.next_u64() % PROBLEMS.len() as u64) as usize } else { truth };
            let mut order: Vec<usize> = (0..PROBLEMS.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
            }
            let options: Vec<serde_json::Value> = order.iter().map(|&i| serde_json::json!({"id": PROBLEMS[i].1})).collect();
            lines.push(
                serde_json::json!({
                    "decision_type": "desk_routing",
                    "state_ref": {"run": format!("run-{next}"), "task": format!("task-{next}"), "suite": task.family, "seed": next, "question_id": "q"},
                    "state": format!("ticket {next} about the {}", PROBLEMS[problem].0),
                    "question": {"id": "q", "type": "choice", "instruction": "which desk handles this ticket"},
                    "options": options,
                    "label": PROBLEMS[label].1,
                    "label_source": "oracle",
                    "outcome": {"task_passed": true, "llm_tokens_saved": 0, "recovery_calls": 0},
                    "split": split,
                    "family": task.family,
                    "seed": next,
                })
                .to_string(),
            );
        }
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Backend, DecideWeights, LayaWeights};

    #[test]
    fn the_decide_fixture_sniffs_and_reads_as_a_decision_model() {
        let d = Scratch::new("fx-decide");
        write_decide(&d, 3);
        assert_eq!(Backend::sniff(&d), Some(Backend::Decide));
        DecideWeights::read(d.to_str().unwrap(), None, 0).expect("a complete decide fixture must read");
    }

    #[test]
    fn the_laya_fixture_sniffs_and_reads_with_its_declared_context() {
        let d = Scratch::new("fx-laya");
        write_laya(&d, 48);
        assert_eq!(Backend::sniff(&d), Some(Backend::Laya));
        let w = LayaWeights::read(d.to_str().unwrap(), None).expect("a complete laya fixture must read");
        assert_eq!(w.max_len(), 48);
    }
}
