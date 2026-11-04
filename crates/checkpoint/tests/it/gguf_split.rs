// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for `MmapGguf::open`'s split-file path: a `<base>-NNNNN-of-
//! MMMMM.gguf` set must open exactly like a single file, and a malformed or
//! incomplete split must be refused BEFORE any tensor is read, by name, not
//! discovered thirty tensors into a forward pass.
//!
//! Swedish Embedded AB implements checkpoint container tooling for its
//! clients. If your team needs expertise in loading multi-file model
//! releases then you can procure our services by sending an email to
//! info@swedishembedded.com.

use checkpoint::gguf::{GgmlType, GgufValue, MmapGguf};
use checkpoint::gguf_write::{self, TensorOut};

fn scratch_dir(name: &str) -> String {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn f32_tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> TensorOut {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorOut { name: name.to_string(), shape, ty: GgmlType::F32.id(), data }
}

fn base_kv() -> Vec<(String, GgufValue)> {
    vec![("general.architecture".to_string(), GgufValue::String("testarch".to_string()))]
}

/// Three parts, two tensors each, distinct values per tensor so a wrong
/// part-index read (not just a wrong name) would show up as wrong data.
fn three_parts() -> Vec<Vec<TensorOut>> {
    vec![
        vec![f32_tensor("layers.0.weight", vec![4], vec![1.0, 2.0, 3.0, 4.0]), f32_tensor("layers.1.weight", vec![4], vec![5.0, 6.0, 7.0, 8.0])],
        vec![f32_tensor("layers.2.weight", vec![4], vec![9.0, 10.0, 11.0, 12.0]), f32_tensor("layers.3.weight", vec![4], vec![13.0, 14.0, 15.0, 16.0])],
        vec![f32_tensor("layers.4.weight", vec![4], vec![17.0, 18.0, 19.0, 20.0]), f32_tensor("layers.5.weight", vec![4], vec![21.0, 22.0, 23.0, 24.0])],
    ]
}

#[test]
fn a_three_part_split_opens_from_any_part_path_and_reads_every_tensor_correctly() {
    let dir = scratch_dir("checkpoint-gguf-split-basic");
    let part1 = gguf_write::write_split(&dir, "model", &base_kv(), &three_parts(), 32).unwrap();
    assert!(part1.ends_with("model-00001-of-00003.gguf"), "{part1}");

    let g = MmapGguf::open(&part1).unwrap();
    let mut names = g.names().to_vec();
    names.sort();
    assert_eq!(names, vec!["layers.0.weight", "layers.1.weight", "layers.2.weight", "layers.3.weight", "layers.4.weight", "layers.5.weight"]);

    // One tensor from each part - proves cross-part indexing, not just part 1.
    assert_eq!(g.tensor("layers.0.weight").unwrap().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(g.tensor("layers.2.weight").unwrap().unwrap(), vec![9.0, 10.0, 11.0, 12.0]);
    assert_eq!(g.tensor("layers.5.weight").unwrap().unwrap(), vec![21.0, 22.0, 23.0, 24.0]);

    // Opening from a NON-first part must find the same complete set.
    let part2 = format!("{dir}/model-00002-of-00003.gguf");
    let g2 = MmapGguf::open(&part2).unwrap();
    assert_eq!(g2.names().len(), 6);
    assert_eq!(g2.tensor("layers.4.weight").unwrap().unwrap(), vec![17.0, 18.0, 19.0, 20.0]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_missing_part_errors_by_the_missing_filename() {
    let dir = scratch_dir("checkpoint-gguf-split-missing");
    let part1 = gguf_write::write_split(&dir, "model", &base_kv(), &three_parts(), 32).unwrap();
    std::fs::remove_file(format!("{dir}/model-00002-of-00003.gguf")).unwrap();

    let err = match MmapGguf::open(&part1) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.contains("model-00002-of-00003.gguf"), "error should name the missing part: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn split_tensors_count_disagreeing_with_the_real_total_is_refused() {
    let dir = scratch_dir("checkpoint-gguf-split-tensorcount");
    let parts = three_parts();
    for (i, part_tensors) in parts.iter().enumerate() {
        let mut kv = base_kv();
        kv.push(("split.no".to_string(), GgufValue::U32(i as u32)));
        kv.push(("split.count".to_string(), GgufValue::U32(3)));
        // Wrong on purpose: the real total is 6, every part claims 7.
        kv.push(("split.tensors.count".to_string(), GgufValue::U64(7)));
        let path = format!("{dir}/model-{:05}-of-00003.gguf", i + 1);
        gguf_write::write(&path, &kv, part_tensors, 32).unwrap();
    }

    let err = match MmapGguf::open(&format!("{dir}/model-00001-of-00003.gguf")) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.contains("split.tensors.count"), "{err}");
    assert!(err.contains('6') && err.contains('7'), "should name both the real and declared counts: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn split_no_off_by_one_against_the_filename_is_refused() {
    let dir = scratch_dir("checkpoint-gguf-split-splitno");
    let parts = three_parts();
    for (i, part_tensors) in parts.iter().enumerate() {
        let mut kv = base_kv();
        // Wrong on purpose: 1-based (as the FILENAME is), not the 0-based
        // index MmapGguf::open requires (llama.cpp's own convention).
        kv.push(("split.no".to_string(), GgufValue::U32(i as u32 + 1)));
        kv.push(("split.count".to_string(), GgufValue::U32(3)));
        kv.push(("split.tensors.count".to_string(), GgufValue::U64(6)));
        let path = format!("{dir}/model-{:05}-of-00003.gguf", i + 1);
        gguf_write::write(&path, &kv, part_tensors, 32).unwrap();
    }

    let err = match MmapGguf::open(&format!("{dir}/model-00001-of-00003.gguf")) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.contains("split.no"), "{err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn mismatched_architecture_across_parts_is_refused() {
    let dir = scratch_dir("checkpoint-gguf-split-arch");
    let parts = three_parts();
    for (i, part_tensors) in parts.iter().enumerate() {
        let arch = if i == 1 { "otherarch" } else { "testarch" };
        let mut kv = vec![("general.architecture".to_string(), GgufValue::String(arch.to_string()))];
        kv.push(("split.no".to_string(), GgufValue::U32(i as u32)));
        kv.push(("split.count".to_string(), GgufValue::U32(3)));
        kv.push(("split.tensors.count".to_string(), GgufValue::U64(6)));
        let path = format!("{dir}/model-{:05}-of-00003.gguf", i + 1);
        gguf_write::write(&path, &kv, part_tensors, 32).unwrap();
    }

    let err = match MmapGguf::open(&format!("{dir}/model-00001-of-00003.gguf")) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.contains("general.architecture"), "{err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_split_part_missing_split_no_entirely_is_refused() {
    let dir = scratch_dir("checkpoint-gguf-split-nosplitno");
    let parts = three_parts();
    for (i, part_tensors) in parts.iter().enumerate() {
        let mut kv = base_kv();
        // Only part 1 carries split.no/count - parts 2 and 3 carry none,
        // as if a non-split-aware tool wrote them.
        if i == 0 {
            kv.push(("split.no".to_string(), GgufValue::U32(0)));
            kv.push(("split.count".to_string(), GgufValue::U32(3)));
        }
        let path = format!("{dir}/model-{:05}-of-00003.gguf", i + 1);
        gguf_write::write(&path, &kv, part_tensors, 32).unwrap();
    }

    let err = match MmapGguf::open(&format!("{dir}/model-00001-of-00003.gguf")) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.contains("split.no"), "{err}");

    std::fs::remove_dir_all(&dir).ok();
}

/// A plain, unsplit file must be completely unaffected by any of this -
/// `split::split_name` declines and `open` takes the single-file path it
/// always has.
#[test]
fn a_plain_unsplit_file_still_opens_normally() {
    let dir = scratch_dir("checkpoint-gguf-split-plain");
    let path = format!("{dir}/model.gguf");
    let tensors = vec![f32_tensor("w", vec![4], vec![1.0, 2.0, 3.0, 4.0])];
    gguf_write::write(&path, &base_kv(), &tensors, 32).unwrap();

    let g = MmapGguf::open(&path).unwrap();
    assert_eq!(g.names(), &["w"]);
    assert_eq!(g.tensor("w").unwrap().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    std::fs::remove_dir_all(&dir).ok();
}
