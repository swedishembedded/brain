// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec gates for loading a THIRD-PARTY (ai-toolkit / ComfyUI) LoRA adapter
//! and folding it into the FLUX.2 inference tensors.
//!
//! The single most dangerous failure mode for an external adapter loader is a
//! SILENT no-op: a key that does not match any base tensor gets skipped, the
//! run completes, and the output is the base model wearing the adapter's name.
//! That looks exactly like success. So the loud-failure cases below are the
//! point of this file, not an afterthought -- every one of them asserts that
//! the error names the offending tensor.
//!
//! Fold semantics are pinned against the reference implementations (see
//! `flux2::lora::fold_external_adapter`'s doc): `W += strength·(alpha/r)·B·A`
//! with `alpha/r == 1.0` when the file carries no `.alpha` tensor.

use std::collections::HashMap;

use flux2::lora::fold_external_adapter;

/// Write a minimal F32 safetensors file: 8-byte LE header length, JSON header,
/// then the tensor payloads back to back. Enough to stand in for an adapter.
fn write_st(path: &std::path::Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, data) in tensors {
        let start = blob.len();
        for v in data {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, blob.len()],
            }),
        );
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&blob);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain-flux2-lora-external-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// A 2x3 base tensor with a rank-1 adapter over it, so the expected fold is
/// small enough to write out by hand. `tag` keeps concurrently running tests
/// off each other's files.
fn tiny_case(tag: &str) -> (std::path::PathBuf, flux2::Tensors, Vec<f32>) {
    // A [r=1, in=3], B [out=2, r=1]  ->  B*A = [[1,2,3],[2,4,6]]
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0];
    let p = tmp(&format!("tiny-{tag}.safetensors"));
    write_st(
        &p,
        &[
            ("diffusion_model.img_in.lora_A.weight", vec![1, 3], a),
            ("diffusion_model.img_in.lora_B.weight", vec![2, 1], b),
        ],
    );
    let mut ts: flux2::Tensors = HashMap::new();
    ts.insert("img_in.weight".into(), (vec![2, 3], vec![10.0; 6]));
    let want = vec![11.0, 12.0, 13.0, 12.0, 14.0, 16.0];
    (p, ts, want)
}

/// The fold is `W += strength·(alpha/r)·B·A`, and with no `.alpha` tensor in
/// the file both ai-toolkit and ComfyUI resolve the alpha multiplier to 1.0.
#[test]
fn folds_b_times_a_into_the_named_base_tensor() {
    let (p, mut ts, want) = tiny_case("fold");
    let info = fold_external_adapter(p.to_str().unwrap(), &mut ts, 1.0).expect("folds");
    assert_eq!(info.pairs, 1, "one adapted linear");
    assert_eq!(info.rank, 1);
    assert_eq!(ts["img_in.weight"].1, want, "W + B*A, exactly");
}

/// `strength` (ComfyUI's `strength_model`, brain's `--lora-scale`) multiplies
/// the whole delta -- 0.0 must reproduce the base weights bit-for-bit, which
/// is what makes a with/without comparison meaningful.
#[test]
fn strength_scales_the_delta_and_zero_is_a_no_op() {
    let (p, mut ts, want) = tiny_case("zero");
    let base = ts["img_in.weight"].1.clone();
    fold_external_adapter(p.to_str().unwrap(), &mut ts, 0.0).expect("folds");
    assert_eq!(ts["img_in.weight"].1, base, "strength 0 leaves the base alone");

    let (p, mut ts, _) = tiny_case("half");
    fold_external_adapter(p.to_str().unwrap(), &mut ts, 0.5).expect("folds");
    let half: Vec<f32> =
        want.iter().zip(&base).map(|(w, b)| b + 0.5 * (w - b)).collect();
    assert_eq!(ts["img_in.weight"].1, half, "strength 0.5 halves the delta");
}

/// THE critical gate. An adapter key that matches no base tensor must be a
/// hard error naming that tensor -- never a quiet skip that returns the base
/// model and looks like a successful run.
#[test]
fn an_unmatched_adapter_key_fails_loudly_by_name() {
    let p = tmp("unmatched.safetensors");
    write_st(
        &p,
        &[
            ("diffusion_model.nope.lora_A.weight", vec![1, 3], vec![1.0; 3]),
            ("diffusion_model.nope.lora_B.weight", vec![2, 1], vec![1.0; 2]),
        ],
    );
    let mut ts: flux2::Tensors = HashMap::new();
    ts.insert("img_in.weight".into(), (vec![2, 3], vec![10.0; 6]));
    let err = fold_external_adapter(p.to_str().unwrap(), &mut ts, 1.0)
        .expect_err("an unmatched key must NOT be silently skipped");
    assert!(err.contains("nope.weight"), "error must name the missing tensor: {err}");
    assert_eq!(ts["img_in.weight"].1, vec![10.0; 6], "base untouched on failure");
}

/// A pair whose shapes disagree with the base tensor is a wrong-model adapter
/// (e.g. a 9B LoRA against 4B weights). Fail by name rather than corrupting.
#[test]
fn a_shape_mismatch_fails_by_name_and_leaves_the_base_untouched() {
    let p = tmp("badshape.safetensors");
    write_st(
        &p,
        &[
            ("diffusion_model.img_in.lora_A.weight", vec![1, 5], vec![1.0; 5]),
            ("diffusion_model.img_in.lora_B.weight", vec![2, 1], vec![1.0; 2]),
        ],
    );
    let mut ts: flux2::Tensors = HashMap::new();
    ts.insert("img_in.weight".into(), (vec![2, 3], vec![10.0; 6]));
    let err = fold_external_adapter(p.to_str().unwrap(), &mut ts, 1.0)
        .expect_err("a 5-wide adapter over a 3-wide base must fail");
    assert!(err.contains("img_in.weight"), "error must name the tensor: {err}");
    assert_eq!(ts["img_in.weight"].1, vec![10.0; 6], "base untouched on failure");
}

/// Half a pair means the file is not what we think it is. Refuse it.
#[test]
fn an_unpaired_lora_a_fails_by_name() {
    let p = tmp("unpaired.safetensors");
    write_st(&p, &[("diffusion_model.img_in.lora_A.weight", vec![1, 3], vec![1.0; 3])]);
    let mut ts: flux2::Tensors = HashMap::new();
    ts.insert("img_in.weight".into(), (vec![2, 3], vec![10.0; 6]));
    let err = fold_external_adapter(p.to_str().unwrap(), &mut ts, 1.0)
        .expect_err("a lora_A with no lora_B must fail");
    assert!(err.contains("img_in"), "error must name the stem: {err}");
}

/// A file with no recognisable adapter keys at all is a user error worth
/// naming -- silently folding nothing is the no-op trap again.
#[test]
fn a_file_with_no_adapter_pairs_fails() {
    let p = tmp("empty.safetensors");
    write_st(&p, &[("some.random.tensor", vec![2], vec![1.0, 2.0])]);
    let mut ts: flux2::Tensors = HashMap::new();
    ts.insert("img_in.weight".into(), (vec![2, 3], vec![10.0; 6]));
    let err = fold_external_adapter(p.to_str().unwrap(), &mut ts, 1.0)
        .expect_err("a file with no lora pairs must fail");
    assert!(!err.is_empty(), "error must say something: {err}");
}

/// The real third-party adapter against the real klein-9b tensor manifest:
/// every one of its pairs must land on a tensor the variant actually has.
/// Weight-free -- the manifest is built from the config, so this needs only
/// the adapter file (`BRAIN_FLUX2_LORA`), not the 9B checkpoint.
#[test]
fn the_real_vsp_adapter_covers_only_klein_9b_tensors() {
    let Ok(path) = std::env::var("BRAIN_FLUX2_LORA") else {
        brain_testutil::skip("BRAIN_FLUX2_LORA unset");
        return;
    };
    let cfg = flux2::Flux2Config::klein_9b();
    let mut ts: flux2::Tensors = cfg
        .tensor_manifest()
        .into_iter()
        .map(|(n, s)| {
            let numel = s.iter().product::<usize>();
            (n, (s, vec![0.0f32; numel]))
        })
        .collect();
    let info = fold_external_adapter(&path, &mut ts, 1.0).expect("real adapter folds");
    assert_eq!(info.pairs, 112, "8 double blocks x 2 streams x 4 + 24 single x 2");
    assert_eq!(info.rank, 32, "the file's own rank");
    // Folded onto an all-zero base, every adapted tensor must now be nonzero:
    // proof the delta actually reached the weights.
    let touched = ts.values().filter(|(_, d)| d.iter().any(|v| *v != 0.0)).count();
    assert_eq!(touched, 112, "every adapted tensor changed");
}
