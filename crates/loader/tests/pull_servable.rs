// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fetching model checkpoints that
// are servable the moment the download ends for its clients. If your team needs
// expertise in model supply chains then you can procure our services by sending
// an email to info@swedishembedded.com.

//! `brain pull` must leave a store directory its architecture can resolve.
//! Each test plans a pull against a mock hub (nothing is downloaded from the
//! network, the "files" are tiny fixtures), runs it through the loader's
//! `execute_plan_opt`, scans the store the way the resolver does and classifies
//! the result through the architecture's own `ArchSpec`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use brain_modelref::ModelRef;
use brain_modelstore::inventory;
use brain_modelstore::resolve::{resolve_under, ArchSpec, Resolution};
use brain_modelstore::{plan, FakeHub, PlanError, Step, Store};
use checkpoint::torchpt_write::{write as write_pt, TensorOut};

fn store(tag: &str) -> Store {
    let dir = std::env::temp_dir().join(format!("loader-pull-servable-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    Store::new(dir)
}

fn downloads(p: &brain_modelstore::Plan) -> Vec<String> {
    p.steps
        .iter()
        .filter_map(|s| match s {
            Step::Download { file, .. } => Some(file.clone()),
            _ => None,
        })
        .collect()
}

fn t(name: &str, shape: Vec<usize>) -> TensorOut {
    let n: usize = shape.iter().product::<usize>().max(1);
    TensorOut { name: name.to_string(), shape, data: vec![0.0; n] }
}

/// The bytes of a tiny `torch.save` file holding `tensors`.
fn pt_bytes(tag: &str, tensors: &[TensorOut]) -> Vec<u8> {
    let path: PathBuf = std::env::temp_dir().join(format!("loader-pull-servable-{tag}-{}.pt", std::process::id()));
    write_pt(path.to_str().unwrap(), tensors).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    bytes
}

const TOY_TEXT_VOCAB: usize = 200;

/// `FunAudioLLM/CosyVoice2-0.5B` as the hub lists it: the three torch
/// checkpoints, both front-end ONNX graphs, the nested `CosyVoice-BlankEN`
/// tokenizer files, and the extras a pull must NOT fetch.
fn cosyvoice_hub() -> FakeHub {
    let mut hub = FakeHub::new();
    let mut add = |file: &str, bytes: Vec<u8>| {
        hub.add_file("FunAudioLLM", "CosyVoice2-0.5B", "main", file, bytes);
    };
    add(
        "llm.pt",
        pt_bytes(
            "llm",
            &[
                t("llm.model.model.embed_tokens.weight", vec![TOY_TEXT_VOCAB, 32]),
                t("speech_embedding.weight", vec![6561, 32]),
                t("llm_decoder.weight", vec![6561, 32]),
                t("llm_embedding.weight", vec![2, 32]),
            ],
        ),
    );
    add("flow.pt", pt_bytes("flow", &[t("input_embedding.weight", vec![6561, 80]), t("spk_embed_affine_layer.weight", vec![80, 192]), t("decoder.estimator.down_blocks.0.0.mlp.1.weight", vec![256, 256])]));
    add(
        "hift.pt",
        pt_bytes(
            "hift",
            &[
                t("f0_predictor.classifier.weight", vec![1, 512]),
                t("m_source.l_linear.weight", vec![1, 8]),
                t("conv_pre.parametrizations.weight.original0", vec![512, 1, 1]),
                t("conv_pre.parametrizations.weight.original1", vec![512, 80, 7]),
            ],
        ),
    );
    add("speech_tokenizer_v2.onnx", s3tokenizer::import::structural_fixture(s3tokenizer::config::S3TokenizerConfig::v2().n_audio_layer as usize));
    add("campplus.onnx", campplus::import::structural_fixture());
    let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_TEXT_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
    add("CosyVoice-BlankEN/vocab.json", serde_json::to_vec(&vocab).unwrap());
    add("CosyVoice-BlankEN/merges.txt", b"#version: 0.2\nt 0\n".to_vec());
    add("CosyVoice-BlankEN/tokenizer_config.json", b"{}".to_vec());
    add("CosyVoice-BlankEN/config.json", b"{}".to_vec());
    // Not read by the cosyvoice pipeline: must stay on the hub.
    add("CosyVoice-BlankEN/model.safetensors", vec![0u8; 8]);
    add("CosyVoice-BlankEN/generation_config.json", b"{}".to_vec());
    add("flow.decoder.estimator.fp32.onnx", vec![0u8; 8]);
    add("speech_tokenizer_v2.batch.onnx", vec![0u8; 8]);
    add("cosyvoice2.yaml", b"x".to_vec());
    add("config.json", b"{}".to_vec());
    add("configuration.json", b"{}".to_vec());
    add("README.md", b"x".to_vec());
    add("asset/dingding.png", vec![0u8; 8]);
    hub
}

#[test]
fn cosyvoice_pull_plans_exactly_the_files_the_spec_resolves() {
    let hub = cosyvoice_hub();
    let p = plan(&ModelRef::new("FunAudioLLM", "CosyVoice2-0.5B", None), &store("cosy-plan"), &hub).unwrap();
    let mut got = downloads(&p);
    got.sort();
    assert_eq!(
        got,
        vec![
            "CosyVoice-BlankEN/config.json",
            "CosyVoice-BlankEN/merges.txt",
            "CosyVoice-BlankEN/tokenizer_config.json",
            "CosyVoice-BlankEN/vocab.json",
            "campplus.onnx",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "speech_tokenizer_v2.onnx",
        ]
    );
}

#[test]
fn a_pulled_cosyvoice_checkpoint_resolves_through_its_spec() {
    let st = store("cosy-resolve");
    let hub = cosyvoice_hub();
    let reference = ModelRef::new("FunAudioLLM", "CosyVoice2-0.5B", None);
    let p = plan(&reference, &st, &hub).unwrap();
    loader::supply::execute_plan_opt(&st, &hub, &p, "cosyvoice", &mut |_, _, _| {}).unwrap();

    let records = inventory::scan(st.root());
    let spec = cosyvoice::spec::CosyVoiceSpec;
    let specs: Vec<&dyn ArchSpec> = vec![&spec];
    let resolution = resolve_under(st.root(), "cosyvoice", &records, &specs, &BTreeMap::new());
    let Resolution::Resolved(assembly) = resolution else { panic!("a pulled cosyvoice checkpoint must resolve, got {resolution:?}") };
    let paths = cosyvoice::pipeline::CosyVoicePaths::from_assembly(&assembly).unwrap();
    let repo = st.repo_dir(&reference);
    assert_eq!(PathBuf::from(&paths.llm), repo);
    assert_eq!(PathBuf::from(&paths.tokenizer), repo.join("CosyVoice-BlankEN"), "the tokenizer role is the directory holding vocab.json and merges.txt");
}

/// A valid one-tensor safetensors file: the inventory checks a shard's
/// declared data extent before it calls a directory complete.
fn tiny_safetensors() -> Vec<u8> {
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0u8; 4]);
    bytes
}

const LLAVA_CONFIG: &[u8] = br#"{"architectures": ["LlavaLlamaForCausalLM"]}"#;

/// `liuhaotian/llava-v1.5-13b` as the hub lists it: PyTorch `.bin` shards, a
/// sentencepiece `tokenizer.model`, and no vision tower.
fn llava_upstream_hub() -> FakeHub {
    let mut hub = FakeHub::new();
    for f in [
        "pytorch_model-00001-of-00003.bin",
        "pytorch_model-00002-of-00003.bin",
        "pytorch_model-00003-of-00003.bin",
        "pytorch_model.bin.index.json",
        "mm_projector.bin",
        "tokenizer.model",
        "tokenizer_config.json",
    ] {
        hub.add_file("liuhaotian", "llava-v1.5-13b", "main", f, vec![0u8; 8]);
    }
    hub.add_file("liuhaotian", "llava-v1.5-13b", "main", "config.json", LLAVA_CONFIG.to_vec());
    hub
}

#[test]
fn an_upstream_llava_layout_is_refused_before_any_download() {
    let hub = llava_upstream_hub();
    let err = plan(&ModelRef::new("liuhaotian", "llava-v1.5-13b", None), &store("llava-refuse"), &hub).unwrap_err();
    let PlanError::NoUpstreamArtifact(_, why) = *err else { panic!("expected a named refusal, got {err}") };
    assert!(why.contains("model.safetensors") && why.contains("tokenizer.json"), "the refusal must name what the llava loader reads: {why}");
}

#[test]
fn a_llava_repo_in_the_layout_the_loader_reads_pulls_and_resolves() {
    let st = store("llava-ready");
    let mut hub = FakeHub::new();
    hub.add_file("someone", "llava-v1.5-13b-st", "main", "config.json", LLAVA_CONFIG.to_vec());
    hub.add_file("someone", "llava-v1.5-13b-st", "main", "model.safetensors", tiny_safetensors());
    hub.add_file("someone", "llava-v1.5-13b-st", "main", "tokenizer.json", b"{}".to_vec());
    let reference = ModelRef::new("someone", "llava-v1.5-13b-st", None);
    let p = plan(&reference, &st, &hub).unwrap();
    loader::supply::execute_plan_opt(&st, &hub, &p, "llava", &mut |_, _, _| {}).unwrap();

    let records = inventory::scan(st.root());
    let spec = llava::spec::LlavaSpec;
    let specs: Vec<&dyn ArchSpec> = vec![&spec];
    let Resolution::Resolved(assembly) = resolve_under(st.root(), "llava", &records, &specs, &BTreeMap::new()) else { panic!("a pulled llava checkpoint must resolve") };
    assert_eq!(assembly.roles["weights"], st.repo_dir(&reference));
}
