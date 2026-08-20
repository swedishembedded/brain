// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for `checkpoint::quantize`: a source in, a readable GGUF out,
//! with every tensor accounted for and each one's fidelity matching what its
//! decision promised.
//!
//! Swedish Embedded AB implements checkpoint conversion and quantization
//! tooling for its clients. If your team needs expertise in on-device model
//! quantization then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Three properties, each of which a real conversion has to have and none of
//! which the unit tests inside the module can see (they stop at the
//! decision - these go through the encoder, the writer AND the reader):
//!
//! 1. **Two-way coverage.** Every source tensor appears in the output, and
//!    the output has nothing the source did not. This is the export mirror of
//!    `ltxv::import::validate_manifest`, and it is the property that catches
//!    a silently dropped tensor - the failure mode an import-side check
//!    cannot see because the tensor is gone before the importer runs.
//! 2. **Kept means untouched.** A tensor the policy kept must come back
//!    BIT-identical, not merely close. Anything less means the skip list is
//!    not actually protecting what it names.
//! 3. **Quantized means quantized well.** Cosine AND rel_l2. Cosine is
//!    scale-invariant, so `got = 1.05 * want` scores 1.0 - it cannot see a
//!    dropped or doubled scale factor at all, and a dropped or doubled block
//!    scale is precisely the error this encoder is able to make.

use std::collections::HashMap;

use checkpoint::gguf::{GgufValue, MmapGguf};
use checkpoint::quantize::{convert, plan, Decision, Kept, Policy, Tier};

/// Deterministic, non-degenerate filler: distinct per (tensor, index), with
/// both signs and a per-row scale spread so a per-block scale has real work
/// to do. `data::rng` is not a dependency of this crate, so this is a local
/// LCG rather than a copy of a shared helper.
fn filler(seed: u64, n: usize, row: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 32) as u32 as f32) / (u32::MAX as f32); // [0,1)
            // Scale by the row index so different rows span different
            // magnitudes - a per-tensor scale would fit this poorly and a
            // per-block one must not.
            (u - 0.5) * 2.0 * (1.0 + (i / row.max(1)) as f32 * 0.05)
        })
        .collect()
}

/// A source exercising every [`Decision`] branch at once.
fn source() -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let mut m = HashMap::new();
    m.insert("blocks.0.mlp.weight".to_string(), (vec![8, 64], filler(1, 8 * 64, 64)));
    m.insert("blocks.1.mlp.weight".to_string(), (vec![96, 32], filler(2, 96 * 32, 32)));
    m.insert("blocks.0.norm.weight".to_string(), (vec![64], filler(3, 64, 64)));
    m.insert("odd_row.weight".to_string(), (vec![4, 20], filler(4, 80, 20)));
    m.insert("pos_embedding".to_string(), (vec![2, 3, 64], filler(5, 384, 64)));
    m.insert("scale_shift_table".to_string(), (vec![9, 64], filler(6, 9 * 64, 64)));
    m.insert("tiny.weight".to_string(), (vec![1, 32], filler(7, 32, 32)));
    m
}

fn policy() -> Policy {
    Policy::new().never_quantize(&["scale_shift_table"]).min_elems(64)
}

fn scratch(name: &str) -> String {
    std::env::temp_dir().join(name).to_string_lossy().into_owned()
}

fn cosine_and_rel_l2(a: &[f32], b: &[f32]) -> (f64, f64) {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum::<f64>().sqrt();
    let diff: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb), diff / na)
}

#[test]
fn every_source_tensor_is_accounted_for_and_none_is_invented() {
    let src = source();
    let rows = plan(&src, Tier::Q8_0, &policy()).unwrap();
    assert_eq!(rows.len(), src.len(), "the plan must have exactly one row per source tensor");

    let by_name: HashMap<&str, &Decision> = rows.iter().map(|r| (r.name.as_str(), &r.decision)).collect();
    assert_eq!(by_name["blocks.0.mlp.weight"], &Decision::Quantize);
    assert_eq!(by_name["blocks.1.mlp.weight"], &Decision::Quantize);
    assert_eq!(by_name["blocks.0.norm.weight"], &Decision::Keep(Kept::NotRank2 { rank: 1 }));
    assert_eq!(by_name["pos_embedding"], &Decision::Keep(Kept::NotRank2 { rank: 3 }));
    assert_eq!(by_name["odd_row.weight"], &Decision::Keep(Kept::RowNotBlockAligned { row: 20, block: 32 }));
    assert_eq!(by_name["scale_shift_table"], &Decision::Keep(Kept::NeverQuantize { pattern: "scale_shift_table".to_string() }));
    assert_eq!(by_name["tiny.weight"], &Decision::Keep(Kept::TooSmall { numel: 32, min: 64 }));
}

#[test]
fn a_converted_source_reads_back_with_kept_tensors_bit_identical() {
    let src = source();
    let path = scratch("checkpoint-quantize-roundtrip.gguf");
    let kv = vec![
        ("general.architecture".to_string(), GgufValue::String("testarch".to_string())),
        ("general.file_type".to_string(), GgufValue::U32(7)),
    ];
    let mut seen = 0usize;
    let report = convert(&src, Tier::Q8_0, &policy(), &kv, &path, &mut |_, _| seen += 1).unwrap();
    assert_eq!(seen, src.len(), "the progress callback must fire once per tensor");
    assert_eq!(report.quantized(), 2);
    assert_eq!(report.kept(), 5);

    let mg = MmapGguf::open(&path).unwrap();

    // (1) Two-way coverage, through the reader rather than through the plan.
    let mut out_names: Vec<String> = mg.names().to_vec();
    out_names.sort();
    let mut in_names: Vec<String> = src.keys().cloned().collect();
    in_names.sort();
    assert_eq!(out_names, in_names, "output tensor set must equal the source's exactly");
    assert_eq!(mg.kv()["general.architecture"].as_str(), Some("testarch"));

    for (name, (shape, want)) in &src {
        assert_eq!(mg.shape(name).unwrap(), shape.as_slice(), "{name}: shape must survive the round trip");
        let got = mg.tensor(name).unwrap().unwrap();
        assert_eq!(got.len(), want.len(), "{name}: element count");
        let row = report.rows.iter().find(|r| &r.name == name).unwrap();
        if row.quantized() {
            assert_eq!(mg.dtype(name), Some("Q8_0"), "{name} was planned quantized and must be stored as Q8_0");
            let (cos, rel) = cosine_and_rel_l2(want, &got);
            // Q8_0 is 8 bits over a 32-element block. `quant.rs`'s own
            // fidelity gate for this type asserts cosine > 0.9999 and rmse <
            // 0.02 on comparable data; these are that floor restated per
            // tensor, with rel_l2 alongside because cosine cannot see a
            // scale error at all.
            assert!(cos > 0.9999, "{name}: Q8_0 cosine {cos:.9}");
            assert!(rel < 0.01, "{name}: Q8_0 rel_l2 {rel:.6}");
        } else {
            // (2) Kept means BIT-identical, not close.
            assert_eq!(mg.dtype(name), Some("F32"), "{name} was kept and must be stored as F32");
            assert_eq!(&got, want, "{name}: a kept tensor must round-trip bit-identically");
        }
    }

    // The saving is real and is measured here, not asserted from the block
    // geometry: 1.0625 B/weight against 4, on the quantized rows only.
    let q_params = report.quantized_params();
    assert!(q_params > 0);
    assert!(report.output_bytes() < report.f32_bytes(), "a Q8_0 conversion must be smaller than f32");
    std::fs::remove_file(&path).ok();
}

#[test]
fn the_same_tool_converts_a_gguf_source_with_a_different_policy() {
    // Reusability, demonstrated rather than claimed: convert once, then feed
    // the RESULT back in as the source under a different policy. Nothing in
    // `quantize` knows which format it is reading.
    let src = source();
    let first = scratch("checkpoint-quantize-chain-1.gguf");
    let second = scratch("checkpoint-quantize-chain-2.gguf");
    convert(&src, Tier::Q8_0, &Policy::new(), &[], &first, &mut |_, _| {}).unwrap();

    let mg = MmapGguf::open(&first).unwrap();
    // A policy that protects a DIFFERENT tensor than the first pass did.
    let report = convert(&mg, Tier::Q8_0, &Policy::new().never_quantize(&["blocks.0."]), &[], &second, &mut |_, _| {}).unwrap();
    let kept: Vec<&str> = report.rows.iter().filter(|r| !r.quantized()).map(|r| r.name.as_str()).collect();
    assert!(kept.contains(&"blocks.0.mlp.weight"), "the second pass's own policy must apply: {kept:?}");
    assert!(!kept.contains(&"blocks.1.mlp.weight"));

    let out = MmapGguf::open(&second).unwrap();
    assert_eq!(out.names().len(), src.len());
    std::fs::remove_file(&first).ok();
    std::fs::remove_file(&second).ok();
}

#[test]
fn a_source_that_cannot_produce_an_enumerated_tensor_is_a_hard_error() {
    // The failure this refuses to paper over: a source that lists a tensor
    // and then cannot hand it over. Zero-filling it would produce a
    // structurally valid GGUF that is silently wrong.
    struct Liar;
    impl checkpoint::TensorSource for Liar {
        fn with_tensor(&self, _name: &str, _f: &mut dyn FnMut(&[f32])) -> bool {
            false
        }
    }
    impl checkpoint::quantize::TensorManifest for Liar {
        fn tensor_names(&self) -> Vec<String> {
            vec!["ghost.weight".to_string()]
        }
        fn tensor_shape(&self, _name: &str) -> Option<Vec<usize>> {
            Some(vec![4, 32])
        }
    }
    let path = scratch("checkpoint-quantize-liar.gguf");
    let err = convert(&Liar, Tier::Q8_0, &Policy::new(), &[], &path, &mut |_, _| {}).unwrap_err();
    assert!(err.contains("ghost.weight"), "the error must name the tensor: {err}");
    std::fs::remove_file(format!("{path}.tmp")).ok();
}
