// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Acceptance coverage for FLUX.2's cut-over to the model-store resolver:
//! `resolve()` + [`Flux2Spec`] against a synthetic store mirroring a real,
//! hand-verified disk layout - a `black-forest-labs/FLUX.2-klein-4B`
//! diffusers pipeline with an interrupted DiT download and a text encoder of
//! the WRONG size sitting right next to a real `unsloth` GGUF release and a
//! real `Qwen/Qwen3-8B` checkpoint. This is the exact shape a hand-mixed
//! store takes in practice, and the exact shape the original bug report was
//! about: env vars (or a naive "just pick the biggest/first thing") could
//! silently wire the wrong-size `black-forest-labs` text encoder in.
//!
//! Swedish Embedded AB implements model-resolution layers like this one for
//! clients running mixed fleets of hand-placed and fetched checkpoints. If
//! your team needs deterministic, never-silently-guessing model assembly, you
//! can procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use brain_modelstore::resolve::{resolve, ArchSpec, Question, Resolution};
use checkpoint::gguf::GgufValue;
use checkpoint::gguf_write::TensorOut;
use flux2::spec::Flux2Spec;
use flux2::Flux2Config;

fn scratch_root() -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-flux2-resolve-layout-{}-{n}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn f32_tensor(name: &str, shape: Vec<usize>) -> TensorOut {
    let n: usize = shape.iter().product();
    TensorOut { name: name.to_string(), shape, ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; n * 4] }
}

/// Only the five tensors [`flux2::dit_config_from_shapes`] actually reads
/// (mirrors `crates/flux2/src/import.rs`'s own `minimal_dit_entries`) - not
/// the full manifest, which at real klein-9b dimensions would write tens of
/// gigabytes of zeros for a fixture that only needs its own shape read back.
fn write_dit_gguf(path: &Path, cfg: &Flux2Config) {
    let tensors = vec![
        f32_tensor("img_in.weight", vec![cfg.hidden, cfg.in_channels]),
        f32_tensor("txt_in.weight", vec![cfg.hidden, cfg.context_in_dim]),
        f32_tensor("double_blocks.0.img_attn.norm.query_norm.scale", vec![cfg.head_dim()]),
        f32_tensor(&format!("double_blocks.{}.marker", cfg.depth_double - 1), vec![1]),
        f32_tensor(&format!("single_blocks.{}.marker", cfg.depth_single - 1), vec![1]),
    ];
    checkpoint::gguf_write::write(path.to_str().unwrap(), &[("general.architecture".to_string(), GgufValue::String("flux".to_string()))], &tensors, 32).unwrap();
}

/// Toy vocab size every fixture writer in this file agrees on, so
/// `Flux2Spec::classify`'s vocab-compatibility check finds a real match
/// between a synthetic tokenizer and its intended text encoder.
const TOY_VOCAB: usize = 100;

/// A GGUF text encoder sharing `general.architecture = "qwen3"` with a real
/// llama.cpp-quantized Qwen3 - unambiguous for `text_encoder` on its own
/// (see `Flux2Spec::classify`'s doc: no shape check needed).
fn write_qwen3_gguf(path: &Path, hidden: u64) {
    let tokens = GgufValue::Array((0..TOY_VOCAB).map(|i| GgufValue::String(format!("t{i}"))).collect());
    checkpoint::gguf_write::write(
        path.to_str().unwrap(),
        &[
            ("general.architecture".to_string(), GgufValue::String("qwen3".to_string())),
            ("qwen3.embedding_length".to_string(), GgufValue::U64(hidden)),
            ("tokenizer.ggml.tokens".to_string(), tokens),
        ],
        &[TensorOut { name: "dummy".to_string(), shape: vec![1], ty: checkpoint::gguf::GgmlType::F32.id(), data: vec![0u8; 4] }],
        32,
    )
    .unwrap();
}

/// A real vocab table at [`TOY_VOCAB`] entries.
fn write_tokenizer_json(path: &Path) {
    let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
    std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
}

fn write_shard(path: &Path) {
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
}

fn write_index(dir: &Path, shard_name: &str) {
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": {"w": shard_name}})).unwrap(),
    )
    .unwrap();
}

/// A canonical `<vendor>/<repo>` HF checkpoint directory: `config.json` +
/// one shard + its index, the shape `brain_modelstore::inventory::scan`
/// collapses to a single `HfDir` record.
fn write_hf_checkpoint(dir: &Path, architectures: &[&str], hidden_size: u64) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": architectures, "hidden_size": hidden_size, "vocab_size": TOY_VOCAB})).unwrap()).unwrap();
    write_shard(&dir.join("model-00001-of-00001.safetensors"));
    write_index(dir, "model-00001-of-00001.safetensors");
}

/// Real FLUX.2 VAE shape rank: a 2D conv, `[out_ch, in_ch, kh, kw]` (4 dims).
/// `Flux2Spec::classify` uses this to tell a real image VAE apart from an
/// unrelated architecture's causal 3D video VAE, which shares the same
/// tensor names at a 5-dim shape.
fn write_vae_safetensors_flat(path: &Path) {
    checkpoint::st::save_safetensors(
        path.to_str().unwrap(),
        &[
            ("decoder.conv_in.weight".to_string(), vec![512, 32, 3, 3], vec![0.0f32; 512 * 32 * 3 * 3]),
            ("encoder.conv_in.weight".to_string(), vec![128, 3, 3, 3], vec![0.0f32; 128 * 3 * 3 * 3]),
        ],
        &serde_json::json!({}),
        None,
    )
    .unwrap();
}

/// Build the whole store: a `black-forest-labs/FLUX.2-klein-4B` diffusers
/// pipeline snapshot (interrupted DiT download, wrong-size text encoder) next
/// to a real `unsloth` GGUF release (vendor-flat DiT, VAE, and a second
/// candidate text encoder) and a real canonical `Qwen/Qwen3-8B` checkpoint.
fn build_store() -> PathBuf {
    let root = scratch_root();

    // --- black-forest-labs/FLUX.2-klein-4B: a real diffusers pipeline
    // snapshot, mid-download and wrong-sized where it matters. ---
    let bfl_dir = root.join("black-forest-labs").join("FLUX.2-klein-4B");
    std::fs::create_dir_all(&bfl_dir).unwrap();
    std::fs::write(bfl_dir.join("model_index.json"), serde_json::to_vec(&serde_json::json!({"_class_name": "Flux2KleinPipeline"})).unwrap()).unwrap();
    let transformer_dir = bfl_dir.join("transformer");
    std::fs::create_dir_all(&transformer_dir).unwrap();
    // An interrupted download: a `.part` sibling, never the real weights -
    // must contribute nothing (an unusable, incomplete record).
    std::fs::write(transformer_dir.join("diffusion_pytorch_model.safetensors.part"), b"interrupted download, not real weights").unwrap();
    // The pipeline's OWN bundled text encoder: real, complete, and the
    // WRONG shape for a 9b assembly (klein-4b's own 2560-hidden Qwen3-4B) -
    // declared "Qwen3Model" (the bare encoder, no LM head), not
    // "Qwen3ForCausalLM", which is exactly why `Flux2Spec::classify` never
    // treats it as a text_encoder candidate at all, whatever variant is
    // asked for.
    write_hf_checkpoint(&bfl_dir.join("text_encoder"), &["Qwen3Model"], 2560);

    // --- unsloth: a real GGUF release, vendor-flat (loose files directly
    // under the vendor directory - a user's hand-placed download, not a
    // <vendor>/<repo> fetch). ---
    let unsloth_dir = root.join("unsloth");
    std::fs::create_dir_all(&unsloth_dir).unwrap();
    write_dit_gguf(&unsloth_dir.join("flux-2-klein-9b-Q8_0.gguf"), &Flux2Config::klein_9b());
    write_vae_safetensors_flat(&unsloth_dir.join("flux2-vae.safetensors"));
    // A second, equally valid text_encoder candidate - same qwen3
    // architecture and hidden size as the canonical Qwen3-8B checkpoint
    // below, so nothing can pick between them on confidence alone.
    write_qwen3_gguf(&unsloth_dir.join("flux2-klein-9b-uncensored-q8_0.gguf"), 4096);
    // The tokenizer role: a real, standalone `tokenizer.json` - vendor-flat
    // loose files are never scanned for one (only a `<vendor>/<repo>` walk
    // recognizes the filename), so it sits in its own small repo-shaped
    // directory the way a real fetched tokenizer would.
    let unsloth_tokenizer_dir = unsloth_dir.join("flux2-klein-9b-tokenizer");
    std::fs::create_dir_all(&unsloth_tokenizer_dir).unwrap();
    write_tokenizer_json(&unsloth_tokenizer_dir.join("tokenizer.json"));

    // --- Qwen/Qwen3-8B: a real, canonical <vendor>/<repo> checkpoint - the
    // FIRST valid text_encoder candidate. ---
    write_hf_checkpoint(&root.join("Qwen").join("Qwen3-8B"), &["Qwen3ForCausalLM"], 4096);

    root
}

fn override_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// The full acceptance sequence against one store: no override at all, then
/// the variant stated, then the variant AND the text encoder both stated.
#[test]
fn flux2_resolves_against_a_real_mixed_store_only_once_every_role_is_unambiguously_stated() {
    let root = build_store();
    let records = brain_modelstore::inventory::scan(&root);
    let spec = Flux2Spec;
    let specs: [&dyn ArchSpec; 1] = [&spec];

    // Step 1: no overrides at all. Never Missing (every role classifies to
    // at least one candidate) and never Resolved (nothing may be silently
    // picked): `text_encoder` has two equally valid candidates
    // (Qwen/Qwen3-8B and the uncensored GGUF) and `resolve()` reports the
    // FIRST ambiguity it finds role-by-role, before ever reaching the
    // dit-shape-vs-klein/base variant question - so the concrete question
    // here is which text encoder, not which variant. Either way: a real
    // question, never a guess.
    match resolve("flux2", &records, &specs, &BTreeMap::new()) {
        Resolution::Ambiguous(a) => {
            assert_eq!(a.question, Question::Role { role: "text_encoder".to_string() }, "{:?}", a.question);
        }
        other => panic!("step 1: expected Ambiguous, got {other:?}"),
    }

    // Step 2: state the text encoder explicitly - the dit-shape/variant
    // question is real too (klein-vs-base is not recoverable from any
    // weight's shape - see `Flux2Spec::assemble`'s doc).
    let te_override = override_map(&[("text_encoder", root.join("Qwen").join("Qwen3-8B").to_str().unwrap())]);
    match resolve("flux2", &records, &specs, &te_override) {
        Resolution::Ambiguous(a) => {
            assert_eq!(a.question, Question::Variant { shape_class: "9b".to_string() }, "{:?}", a.question);
            let selectors: Vec<(String, String)> = a.choices.iter().flat_map(|c| c.selector.clone()).collect();
            assert!(selectors.contains(&("--variant".to_string(), "klein-9b".to_string())), "{selectors:?}");
            assert!(selectors.contains(&("--variant".to_string(), "base-9b".to_string())), "{selectors:?}");
        }
        other => panic!("step 2: expected Ambiguous, got {other:?}"),
    }

    // Step 3: state the variant alone - text_encoder is still genuinely
    // ambiguous (both candidates unaffected by which klein/base variant was
    // named), so this is Ambiguous{Role:"text_encoder"} again, not Resolved.
    let variant_override = override_map(&[("variant", "klein-9b")]);
    match resolve("flux2", &records, &specs, &variant_override) {
        Resolution::Ambiguous(a) => {
            assert_eq!(a.question, Question::Role { role: "text_encoder".to_string() }, "{:?}", a.question);
            assert_eq!(a.choices.len(), 2, "{:?}", a.choices);
        }
        other => panic!("step 3: expected Ambiguous, got {other:?}"),
    }

    // Step 4: state BOTH - the actual regression this whole cut-over is
    // about. Resolves cleanly, and every role comes from the unsloth/Qwen3-8B
    // files - NONE of them from the incomplete/wrong-size
    // black-forest-labs/FLUX.2-klein-4B directory.
    let qwen3_8b = root.join("Qwen").join("Qwen3-8B");
    let both_override = override_map(&[("variant", "klein-9b"), ("text_encoder", qwen3_8b.to_str().unwrap())]);
    match resolve("flux2", &records, &specs, &both_override) {
        Resolution::Resolved(assembly) => {
            assert_eq!(assembly.variant.as_deref(), Some("klein-9b"));
            let bfl_dir = root.join("black-forest-labs");
            for (role, path) in &assembly.roles {
                assert!(!path.starts_with(&bfl_dir), "role '{role}' resolved to {} under black-forest-labs", path.display());
            }
            assert_eq!(assembly.roles.get("dit").map(PathBuf::as_path), Some(root.join("unsloth").join("flux-2-klein-9b-Q8_0.gguf").as_path()));
            assert_eq!(assembly.roles.get("vae").map(PathBuf::as_path), Some(root.join("unsloth").join("flux2-vae.safetensors").as_path()));
            assert_eq!(assembly.roles.get("text_encoder").map(PathBuf::as_path), Some(qwen3_8b.as_path()));
            assert_eq!(
                assembly.roles.get("tokenizer").map(PathBuf::as_path),
                Some(root.join("unsloth").join("flux2-klein-9b-tokenizer").join("tokenizer.json").as_path())
            );

            // validate() itself already gated Resolved (12288 == 3*4096) -
            // re-run it explicitly so a future change to that gate that
            // stops being honored here fails loudly.
            spec.validate(&assembly).unwrap();

            // And the resolver's own output feeds straight into the real
            // pipeline paths with no further path-picking of any kind.
            let paths = flux2::Paths::from_assembly(&assembly).unwrap();
            assert_eq!(paths.dit, root.join("unsloth").join("flux-2-klein-9b-Q8_0.gguf").to_str().unwrap());
            assert_eq!(paths.te, qwen3_8b.to_str().unwrap());
        }
        other => panic!("step 4: expected Resolved, got {other:?}"),
    }

    std::fs::remove_dir_all(&root).ok();
}
