// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The strict safetensors importer: the header is inspected before any data
//! is read, every expected tensor must be present with its exact shape and
//! dtype, and nothing is ever filled in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mirage::import::{self, Checkpoint, ImportError, ImportOptions, TensorProblem};

use crate::common::{released_or_skip, tiny_checkpoint};

/// The SHA-256 of the released `mirage.safetensors` this suite was written against.
const RELEASED_SAFETENSORS_SHA256: &str = "6ba97c5c1bef661c9b0c80782135682b38b4312fb33e0b645c93c10604bb16cd";

/// A path inside a private temporary directory that is removed when it drops.
struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Scratch {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Some(dir) = self.0.parent() {
            std::fs::remove_dir_all(dir).ok();
        }
    }
}

fn scratch(name: &str) -> Scratch {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-mirage-import-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir.join(name))
}

fn raw_file(path: &Path, tensors: &[(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)], metadata: &[(String, String)]) {
    let views: Vec<(String, safetensors::tensor::TensorView)> =
        tensors.iter().map(|(n, d, s, b)| (n.to_string(), safetensors::tensor::TensorView::new(*d, s.clone(), b).unwrap())).collect();
    let meta: std::collections::HashMap<String, String> = metadata.iter().cloned().collect();
    std::fs::write(path, safetensors::serialize(views, Some(meta)).unwrap()).unwrap();
}

/// Write `ck` with one tensor removed, added or reshaped through the raw writer.
fn write_tampered(ck: &Checkpoint, tamper: impl Fn(&mut BTreeMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)>)) -> Scratch {
    let mut raw: BTreeMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)> = ck
        .tensors
        .iter()
        .map(|(n, t)| (n.clone(), (safetensors::Dtype::F32, t.shape.clone(), t.data.iter().flat_map(|v| v.to_le_bytes()).collect())))
        .collect();
    tamper(&mut raw);
    let list: Vec<(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)> = raw.iter().map(|(n, (d, s, b))| (n.as_str(), *d, s.clone(), b.clone())).collect();
    let path = scratch("tampered.safetensors");
    raw_file(
        &path,
        &list,
        &[
            ("config".to_string(), ck.config.to_json()),
            ("source_sha256".to_string(), ck.metadata.source_sha256.clone()),
            ("step".to_string(), ck.metadata.step.to_string()),
            ("format".to_string(), "mirage".to_string()),
        ],
    );
    path
}

fn problems(err: ImportError) -> Vec<TensorProblem> {
    match err {
        ImportError::Tensors(p) => p,
        other => panic!("expected tensor problems, got {other}"),
    }
}

#[test]
fn an_exported_checkpoint_round_trips_exactly() {
    let ck = tiny_checkpoint();
    let path = scratch("tiny.safetensors");
    import::export(&ck, &path).unwrap();
    let back = import::load(&path, &ImportOptions::default()).unwrap();
    assert_eq!(back.config, ck.config);
    assert_eq!(back.metadata, ck.metadata, "source hash, step and format survive");
    assert_eq!(back.tensors.len(), ck.tensors.len());
    for (name, t) in &ck.tensors {
        let b = &back.tensors[name];
        assert_eq!(b.shape, t.shape, "{name}");
        assert!(b.data.iter().zip(&t.data).all(|(x, y)| x.to_bits() == y.to_bits()), "{name} differs bit for bit");
    }
}

#[test]
fn export_is_deterministic() {
    let ck = tiny_checkpoint();
    let (a, b) = (scratch("a.safetensors"), scratch("b.safetensors"));
    import::export(&ck, &a).unwrap();
    import::export(&ck, &b).unwrap();
    assert_eq!(std::fs::read(a).unwrap(), std::fs::read(b).unwrap());
}

#[test]
fn export_refuses_a_checkpoint_that_does_not_match_its_manifest() {
    let mut ck = tiny_checkpoint();
    ck.tensors.remove("pe_gate");
    let err = import::export(&ck, &scratch("bad.safetensors")).unwrap_err();
    assert!(err.to_string().contains("pe_gate"), "{err}");

    let mut ck = tiny_checkpoint();
    ck.tensors.get_mut("registers").unwrap().data.pop();
    let err = import::export(&ck, &scratch("bad.safetensors")).unwrap_err();
    assert!(err.to_string().contains("registers"), "{err}");
}

#[test]
fn a_missing_tensor_is_named_and_nothing_is_zero_filled() {
    let ck = tiny_checkpoint();
    let path = write_tampered(&ck, |t| {
        t.remove("body_1/grn/gamma");
    });
    let p = problems(import::load(&path, &ImportOptions::default()).unwrap_err());
    assert_eq!(p, vec![TensorProblem::Missing { name: "body_1/grn/gamma".to_string() }]);
}

#[test]
fn an_unexpected_tensor_is_named() {
    let ck = tiny_checkpoint();
    let path = write_tampered(&ck, |t| {
        t.insert("body_9/extra".to_string(), (safetensors::Dtype::F32, vec![1], vec![0; 4]));
    });
    let p = problems(import::load(&path, &ImportOptions::default()).unwrap_err());
    assert_eq!(p, vec![TensorProblem::Unexpected { name: "body_9/extra".to_string() }]);
}

#[test]
fn a_wrong_shape_or_dtype_is_named() {
    let ck = tiny_checkpoint();
    let path = write_tampered(&ck, |t| {
        t.insert("registers".to_string(), (safetensors::Dtype::F32, vec![2, 16, 2], vec![0; 4 * 64]));
        t.insert("pe_gate".to_string(), (safetensors::Dtype::F16, vec![1], vec![0; 2]));
    });
    let mut p = problems(import::load(&path, &ImportOptions::default()).unwrap_err());
    p.sort_by_key(|x| x.name().to_string());
    assert_eq!(p.len(), 2);
    assert!(matches!(&p[0], TensorProblem::Dtype { name, found } if name == "pe_gate" && found == "F16"), "{:?}", p[0]);
    assert!(matches!(&p[1], TensorProblem::Shape { name, expected, found } if name == "registers" && expected == &[2, 32] && found == &[2, 16, 2]), "{:?}", p[1]);
}

#[test]
fn every_problem_is_reported_together() {
    let ck = tiny_checkpoint();
    let path = write_tampered(&ck, |t| {
        t.remove("pe_gate");
        t.remove("enc_norm/scale");
        t.insert("zzz".to_string(), (safetensors::Dtype::F32, vec![1], vec![0; 4]));
    });
    let err = import::load(&path, &ImportOptions::default()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("pe_gate") && text.contains("enc_norm/scale") && text.contains("zzz"), "{text}");
}

#[test]
fn a_header_problem_is_reported_before_any_tensor_data_is_read() {
    // A header that claims more bytes than the file holds must fail on its own,
    // not by reading past the end.
    let ck = tiny_checkpoint();
    let path = scratch("truncated.safetensors");
    import::export(&ck, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 100]).unwrap();
    let err = import::load(&path, &ImportOptions::default()).unwrap_err();
    assert!(matches!(err, ImportError::Header(_)), "{err}");
    let err = import::inspect(&path).unwrap_err();
    assert!(matches!(err, ImportError::Header(_)), "{err}");
}

#[test]
fn metadata_is_validated() {
    let ck = tiny_checkpoint();
    // Missing key, unknown key, bad step, wrong format, config that does not parse.
    let t = &ck.tensors;
    let tensors: Vec<(&str, safetensors::Dtype, Vec<usize>, Vec<u8>)> =
        t.iter().map(|(n, x)| (n.as_str(), safetensors::Dtype::F32, x.shape.clone(), x.data.iter().flat_map(|v| v.to_le_bytes()).collect())).collect();
    let cfg = ck.config.to_json();
    let full = |extra: &[(&str, &str)], drop: &str| -> Vec<(String, String)> {
        let mut m: Vec<(String, String)> =
            vec![("config".into(), cfg.clone()), ("source_sha256".into(), "ab".repeat(32)), ("step".into(), "5".into()), ("format".into(), "mirage".into())];
        m.retain(|(k, _)| k != drop);
        for (k, v) in extra {
            m.retain(|(kk, _)| kk != k);
            m.push((k.to_string(), v.to_string()));
        }
        m
    };
    let cases: Vec<(Vec<(String, String)>, &str)> = vec![
        (full(&[], "step"), "step"),
        (full(&[], "config"), "config"),
        (full(&[("extra", "1")], ""), "extra"),
        (full(&[("step", "many")], ""), "step"),
        (full(&[("format", "other")], ""), "format"),
        (full(&[("source_sha256", "xyz")], ""), "source_sha256"),
        (full(&[("config", "{\"dim\": 3}")], ""), "config"),
    ];
    for (meta, field) in cases {
        let path = scratch("meta.safetensors");
        raw_file(&path, &tensors, &meta);
        let err = import::load(&path, &ImportOptions::default()).unwrap_err();
        assert!(err.to_string().contains(field), "{field}: {err}");
    }
}

#[test]
fn a_pinned_digest_must_match_the_file() {
    let ck = tiny_checkpoint();
    let path = scratch("pinned.safetensors");
    import::export(&ck, &path).unwrap();
    let digest = import::file_sha256(&path).unwrap();
    assert_eq!(digest.len(), 64);
    import::load(&path, &ImportOptions { expected_sha256: Some(digest.clone()) }).unwrap();
    let err = import::load(&path, &ImportOptions { expected_sha256: Some("00".repeat(32)) }).unwrap_err();
    assert!(matches!(&err, ImportError::Digest { found, .. } if *found == digest), "{err}");
}

#[test]
fn a_model_directory_must_agree_with_its_config_json() {
    let ck = tiny_checkpoint();
    let holder = scratch("dir");
    let dir = holder.parent().unwrap().join("modeldir");
    std::fs::create_dir_all(&dir).unwrap();
    import::export(&ck, &dir.join("mirage.safetensors")).unwrap();
    let envelope = |step: u64, sha: &str| {
        serde_json::json!({
            "model_type": "mirage",
            "architecture": serde_json::from_str::<serde_json::Value>(&ck.config.to_json()).unwrap(),
            "training_step": step,
            "source_checkpoint_sha256": sha,
            "cact_bits": 2,
            "cact_bits_map": {"embedding": 4, "mhc": 4, "default": 2}
        })
        .to_string()
    };
    std::fs::write(dir.join("config.json"), envelope(1234, &"ab".repeat(32))).unwrap();
    import::load_dir(&dir, &ImportOptions::default()).expect("agreeing files load");

    std::fs::write(dir.join("config.json"), envelope(99, &"ab".repeat(32))).unwrap();
    let err = import::load_dir(&dir, &ImportOptions::default()).unwrap_err();
    assert!(err.to_string().contains("training_step"), "{err}");

    std::fs::write(dir.join("config.json"), envelope(1234, &"cd".repeat(32))).unwrap();
    let err = import::load_dir(&dir, &ImportOptions::default()).unwrap_err();
    assert!(err.to_string().contains("source_checkpoint_sha256"), "{err}");
}

#[test]
fn the_released_checkpoint_matches_the_manifest_exactly() {
    let Some(dir) = released_or_skip("released safetensors import") else { return };
    let path = dir.join("mirage.safetensors");
    let header = import::inspect(&path).expect("header inspection of the released file");
    assert_eq!(header.tensors.len(), 583);
    assert_eq!(header.metadata.step, 1_100_000);
    assert_eq!(header.metadata.format, "mirage");
    assert_eq!(header.metadata.source_sha256, "dc8db0c4f1a9f516a9080224514de7b6b85193a8c6648a35b0f0486c5bc25224");

    let ck = import::load_dir(&dir, &ImportOptions { expected_sha256: Some(RELEASED_SAFETENSORS_SHA256.to_string()) }).expect("released checkpoint imports");
    assert_eq!(ck.tensors.values().map(|t| t.data.len()).sum::<usize>(), 54_078_804);
    assert!(ck.tensors.values().all(|t| t.data.iter().all(|v| v.is_finite())), "released weights are finite");
}

#[test]
fn the_released_checkpoint_exports_to_the_same_values() {
    let Some(dir) = released_or_skip("released safetensors round trip") else { return };
    let ck = import::load_dir(&dir, &ImportOptions::default()).unwrap();
    let path = scratch("released-copy.safetensors");
    import::export(&ck, &path).unwrap();
    let back = import::load(&path, &ImportOptions::default()).unwrap();
    assert_eq!(back.metadata, ck.metadata);
    for (name, t) in &ck.tensors {
        assert!(back.tensors[name].data.iter().zip(&t.data).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}");
    }
}
