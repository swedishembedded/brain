// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen35Config::tiny().param_list()` two-way coverage against the golden
//! dumper's own saved weights (`tools/goldens/qwen35_dump_reference.py`'s
//! `collect_weights`) - the practical, available-NOW version of "param_list
//! matches the real checkpoint header manifest" (M3's gate), since the real
//! 27B FP8 checkpoint is not fetched until M10. Every name `param_list()`
//! expects must exist in the golden with the same element count, and the
//! golden must carry no tensor `param_list()` doesn't expect (a real
//! importer's two-way-coverage discipline applied here to config vs. dump,
//! not config vs. checkpoint).

use std::collections::HashSet;

#[test]
fn tiny_param_list_matches_the_golden_weights_two_way() {
    let path = brain_testutil::testdata("golden/qwen35/tiny_text/qwen35_tiny_text_weights.safetensors");
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip(&format!("fixture {path} absent - run tools/goldens/qwen35_dump_reference.py"));
        return;
    }
    let golden = checkpoint::safetensors::read(&path).expect("read golden weights");
    let golden_numel: std::collections::HashMap<String, usize> =
        golden.iter().map(|t| (t.name.clone(), t.shape.iter().product())).collect();

    let cfg = qwen35::Qwen35Config::tiny();
    let expected = cfg.param_list();

    let mut missing = Vec::new();
    let mut wrong_shape = Vec::new();
    for (name, numel) in &expected {
        match golden_numel.get(name) {
            None => missing.push(name.clone()),
            Some(&n) if n != *numel => wrong_shape.push((name.clone(), *numel, n)),
            _ => {}
        }
    }
    assert!(missing.is_empty(), "param_list() names absent from the golden: {missing:?}");
    assert!(wrong_shape.is_empty(), "param_list()/golden numel mismatch (name, expected, golden): {wrong_shape:?}");

    let expected_names: HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    let extra: Vec<&String> = golden_numel.keys().filter(|n| !expected_names.contains(n.as_str())).collect();
    assert!(extra.is_empty(), "golden carries tensors param_list() does not expect: {extra:?}");

    assert_eq!(expected.len(), golden_numel.len());
    println!("two-way coverage: {} tensors match exactly", expected.len());
}
