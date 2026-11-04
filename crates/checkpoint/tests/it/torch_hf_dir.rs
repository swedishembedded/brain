// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A Hugging Face checkpoint that ships only `pytorch_model*.bin` (a single
//! file, or shards named by `pytorch_model.bin.index.json`) opens through
//! `WeightReader::open_hf_dir` like a safetensors one: every tensor, by its
//! name, decoded one at a time and equal to what the eager reader returns.

use checkpoint::torchpt_write::{write, TensorOut};
use checkpoint::weightio::WeightReader;

fn tensor(name: &str, shape: &[usize], base: f32) -> TensorOut {
    let n: usize = shape.iter().product();
    TensorOut { name: name.into(), shape: shape.to_vec(), data: (0..n).map(|i| base + i as f32 * 0.5).collect() }
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain-torch-hf-dir-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn assert_reads(r: &WeightReader, want: &[TensorOut]) {
    let mut names: Vec<&str> = r.names().collect();
    names.sort();
    let mut expect: Vec<&str> = want.iter().map(|t| t.name.as_str()).collect();
    expect.sort();
    assert_eq!(names, expect);
    for t in want {
        assert_eq!(r.shape(&t.name).unwrap(), t.shape.iter().map(|&d| d as u64).collect::<Vec<_>>().as_slice(), "{}", t.name);
        assert_eq!(r.dtype(&t.name), Some("F32"), "{}", t.name);
        assert_eq!(r.tensor(&t.name).unwrap(), t.data, "{}", t.name);
    }
}

#[test]
fn a_sharded_bin_checkpoint_reads_every_tensor_by_name() {
    let dir = scratch("sharded");
    let a = [tensor("model.embed_tokens.weight", &[4, 3], 1.0), tensor("model.norm.weight", &[3], 9.0)];
    let b = [tensor("lm_head.weight", &[4, 3], -2.0)];
    write(dir.join("pytorch_model-00001-of-00002.bin").to_str().unwrap(), &a).unwrap();
    write(dir.join("pytorch_model-00002-of-00002.bin").to_str().unwrap(), &b).unwrap();
    let map: serde_json::Map<String, serde_json::Value> = a
        .iter()
        .map(|t| (t.name.clone(), "pytorch_model-00001-of-00002.bin".into()))
        .chain(b.iter().map(|t| (t.name.clone(), "pytorch_model-00002-of-00002.bin".into())))
        .collect();
    std::fs::write(dir.join("pytorch_model.bin.index.json"), serde_json::json!({"weight_map": map}).to_string()).unwrap();

    let r = WeightReader::open_hf_dir(&dir).unwrap();
    let all: Vec<TensorOut> = a.into_iter().chain(b).collect();
    assert_reads(&r, &all);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_single_bin_checkpoint_opens_and_safetensors_wins_when_both_ship() {
    let dir = scratch("single");
    let t = [tensor("model.norm.weight", &[5], 3.0)];
    write(dir.join("pytorch_model.bin").to_str().unwrap(), &t).unwrap();
    assert_reads(&WeightReader::open_hf_dir(&dir).unwrap(), &t);

    let st = [("model.norm.weight".to_string(), vec![5u64], vec![7.0f32; 5])];
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &st, &serde_json::Value::Null, None).unwrap();
    assert_eq!(WeightReader::open_hf_dir(&dir).unwrap().tensor("model.norm.weight").unwrap(), vec![7.0; 5]);

    // A brain-format file beside the download (sorting first) is not it.
    let brain = [("model.norm.weight".to_string(), vec![5u64], vec![-1.0f32; 5])];
    checkpoint::st::save_safetensors(dir.join("model.brain.safetensors").to_str().unwrap(), &brain, &serde_json::Value::Null, None).unwrap();
    assert_eq!(WeightReader::open_hf_dir(&dir).unwrap().tensor("model.norm.weight").unwrap(), vec![7.0; 5]);
    let _ = std::fs::remove_dir_all(&dir);
}
