// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A checkpoint read under other names (`weightio::Renamed`) is a whole
//! checkpoint: its tensors, their shapes and its config are what a loader
//! sees, eagerly (`checkpoint::load_reader`) or written out
//! (`WeightReader::save`), and a mapping naming a tensor the source lacks is
//! refused when the view is built.

use checkpoint::weightio::{Renamed, WeightReader};
use serde_json::json;

fn source(tag: &str) -> (std::path::PathBuf, WeightReader) {
    let dir = std::env::temp_dir().join(format!("brain-renamed-view-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let tensors = [
        ("up.a.weight".to_string(), vec![2u64, 3], (0..6).map(|i| i as f32).collect()),
        ("up.b.bias".to_string(), vec![2u64], vec![-1.0f32, 1.0]),
        ("other.weight".to_string(), vec![1u64], vec![9.0f32]),
    ];
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &json!({"ignored": true}), None).unwrap();
    let r = WeightReader::open(path.to_str().unwrap()).unwrap();
    (dir, r)
}

#[test]
fn a_renamed_view_reads_saves_and_loads_as_its_own_checkpoint() {
    let (dir, src) = source("reads");
    let map = vec![("a.weight".to_string(), "up.a.weight".to_string()), ("b.bias".to_string(), "up.b.bias".to_string())];
    let view = WeightReader::derived(Box::new(Renamed::new(src, map, json!({"d_model": 3})).unwrap()));

    let mut names: Vec<&str> = view.names().collect();
    names.sort();
    assert_eq!(names, ["a.weight", "b.bias"]);
    assert_eq!(view.shape("a.weight").unwrap(), [2, 3]);
    assert_eq!(view.config(), json!({"d_model": 3}));

    let c = checkpoint::load_reader(&view).unwrap();
    assert_eq!(c.header["config"], json!({"d_model": 3}));
    assert_eq!(c.find("a.weight", "").unwrap(), &(0..6).map(|i| i as f32).collect::<Vec<_>>());
    assert_eq!(c.find("b.bias", "").unwrap(), &vec![-1.0, 1.0]);
    assert_eq!(c.tensors.len(), 2);

    let out = dir.join("view.safetensors");
    view.save(out.to_str().unwrap(), None).unwrap();
    let back = checkpoint::load(out.to_str().unwrap());
    assert_eq!(back.header, c.header);
    assert_eq!(back.by_role(""), c.by_role(""));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_mapping_to_a_missing_source_tensor_is_refused() {
    let (dir, src) = source("missing");
    let err = Renamed::new(src, vec![("a.weight".to_string(), "up.nope.weight".to_string())], json!({})).err().unwrap();
    assert!(err.contains("up.nope.weight"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
