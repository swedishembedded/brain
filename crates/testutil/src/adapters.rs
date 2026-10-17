// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixtures for tests of served adapter releases: a stand-in base checkpoint
//! and a Qwen3 LoRA adapter whose card names it. Shared by the crate that
//! verifies a release and the one that swaps it in under load.

use std::path::{Path, PathBuf};

use checkpoint::st::{Adapter, ModelCard, TrainingProvenance};

/// A fresh, empty scratch directory unique to this process and `name`.
pub fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-adapter-release-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A stand-in base file: verification reads only its bytes.
pub fn write_base(path: &Path, fill: f32) {
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("embed.weight".to_string(), vec![4], vec![fill; 4])], &serde_json::json!({}), None).unwrap();
}

/// A Qwen3 LoRA adapter file as `qwen3::lora::save_adapter_with_lineage`
/// writes it: a card naming `base_id` and, when given, the base digest
/// it was trained against.
pub fn write_adapter(path: &Path, id: &str, base_id: &str, base_digest: Option<String>, fill: f32) {
    let mut card = ModelCard::new(id, "qwen");
    card.adapter = Some(Adapter { kind: "lora".to_string(), rank: Some(2), base: Some(base_id.to_string()), alpha: Some(4.0), ..Default::default() });
    card.training = Some(TrainingProvenance {
        code_revision: "test".to_string(),
        regime: "sft_lora".to_string(),
        seed: 1,
        hyperparams: serde_json::Value::Null,
        environment: "cpu".to_string(),
        gate: None,
        trained_from: None,
        base_digest,
        cycle: 0,
    });
    let tensors = vec![
        ("blocks.0.attn.wq.weight.lora_a".to_string(), vec![2, 4], vec![fill; 8]),
        ("blocks.0.attn.wq.weight.lora_b".to_string(), vec![4, 2], vec![fill; 8]),
    ];
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({"rank": 2, "alpha": 4.0}), Some(&card)).unwrap();
}

pub fn file_digest(path: &Path) -> String {
    brain_modelstore::fetch::file_digest(path).unwrap()
}

/// One linear an adapter corrects: the base weight's name and its
/// `[out, in]` shape.
pub struct Linear {
    /// The base weight (`blocks.0.attn.wq.weight`).
    pub base: String,
    /// Output width.
    pub out: usize,
    /// Input width.
    pub inn: usize,
}

/// A Qwen3 LoRA adapter file over `linears` with deterministic pseudo-random
/// factors (`seed`), big enough to move a tiny model's greedy decode. The card
/// names `base_id` and carries no base digest, so the file is bound by id.
pub fn write_lora_adapter(path: &Path, id: &str, base_id: &str, rank: u32, linears: &[Linear], seed: u64) {
    write_lora_adapter_of("qwen", path, id, base_id, rank, linears, seed);
}

/// [`write_lora_adapter`] for a card of `family` (`qwen35` for a Qwen3.8
/// adapter). The card lists the linears' leaf names as its targets, as a
/// trainer writes them.
pub fn write_lora_adapter_of(family: &str, path: &Path, id: &str, base_id: &str, rank: u32, linears: &[Linear], seed: u64) {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let r = rank as usize;
    let mut tensors = Vec::new();
    for l in linears {
        tensors.push((format!("{}.lora_a", l.base), vec![r as u64, l.inn as u64], (0..r * l.inn).map(|_| next()).collect::<Vec<f32>>()));
        tensors.push((format!("{}.lora_b", l.base), vec![l.out as u64, r as u64], (0..l.out * r).map(|_| next()).collect::<Vec<f32>>()));
    }
    let alpha = 2.0 * rank as f32;
    let mut card = ModelCard::new(id, family);
    card.variant_of = Some(base_id.to_string());
    let mut targets: Vec<String> = Vec::new();
    for l in linears {
        let leaf = l.base.rsplit('.').nth(1).unwrap_or_default().to_string();
        if !targets.contains(&leaf) {
            targets.push(leaf);
        }
    }
    card.adapter = Some(Adapter { kind: "lora".to_string(), rank: Some(rank), base: Some(base_id.to_string()), alpha: Some(alpha), targets: (family != "qwen").then_some(targets), ..Default::default() });
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({"rank": rank, "alpha": alpha}), Some(&card)).unwrap();
}

/// A `tokenizer.json` with one token per printable ASCII byte (plus space and
/// newline) and no merges, with a ChatML template beside it, written into
/// `dir`; returns the tokenizer's path. `QwenBpe` maps each byte through
/// GPT-2's `bytes_to_unicode`, which is the IDENTITY on `0x21..=0x7E`, so every
/// byte is its own token: enough to round-trip a test's text and the chat
/// template's own `<|im_start|>`/`<|im_end|>` markers with no real tokenizer
/// on the box.
pub fn write_byte_tokenizer(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let mut vocab = serde_json::Map::new();
    for (i, b) in (0x21u8..=0x7e).enumerate() {
        vocab.insert((b as char).to_string(), serde_json::json!(i));
    }
    let n = vocab.len();
    vocab.insert('\u{0120}'.to_string(), serde_json::json!(n)); // space
    vocab.insert('\u{010a}'.to_string(), serde_json::json!(n + 1)); // newline
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, serde_json::json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();
    // A checkpoint with no template serves raw prompts only, and this
    // vocabulary has no `<|im_start|>` special to infer ChatML from.
    let chatml = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::json!({ "chat_template": chatml }).to_string()).unwrap();
    path
}
