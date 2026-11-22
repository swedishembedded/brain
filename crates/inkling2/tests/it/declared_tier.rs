// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving a model with the low-precision activation numerics it was
// trained with for its clients. If your team needs expertise in proving a cached engine against its reference forward
// on quantized expert banks then you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of the declared activation tier in the served engine (`--linear-activations declared`):
//!
//! - on a tiny release with published NVFP4 expert banks (each declaring the
//!   vendor's activation quantization as the released `hf_quant_config.json` does; see [`nvfp4`]), the engine's served logits equal
//!   the uncached [`Inkling2::forward`] under the same option at every position: the prompt's grouped products and the
//!   decode steps' products gathered per slot read the same round-tripped activations;
//! - the option changes the served numerics (it is not silently ignored), and the engine passes the serving
//!   conformance suite under it.

use std::path::{Path, PathBuf};

use data::rng::Lcg;
use inkling2::names::LayerNames;
use inkling2::serve::{Engine, EngineOptions};
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::policy::LinearActivations;
use model::serve::conformance::{self, Conformance};
use model::serve::visit_state::VisitPoolSizing;
use model::serve::{PagedDecoder, PrefillProgress};

/// The f32 reference's own spread, relative to the tensor's scale (as in the engine's other gates).
const TOL: f32 = 5e-5;

fn write_safetensors(path: &Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut off = 0;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (.., bytes) in tensors {
        out.extend(bytes);
    }
    std::fs::write(path, out).unwrap();
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The tiny reference checkpoint with its expert banks replaced by published-format banks of random codes (the
/// property is the engine against the forward, not the weights' meaning), under a `parameter_groups` declaration of
/// the released shape: every other tensor a float32 cast, each layer's banks with the activation quantization the
/// release requires.
fn declared_release(tag: &str) -> Option<PathBuf> {
    let tiny = brain_testutil::testdata_path("golden/inkling2/tiny/checkpoint");
    if !tiny.join("model.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("inkling2-declared-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(tiny.join("config.json"), dir.join("config.json")).unwrap();
    let cfg = inkling2::Inkling2Config::from_dir(&tiny).unwrap();
    let banks: Vec<String> = (0..cfg.n_layers).flat_map(|l| [LayerNames::new(l).w13(), LayerNames::new(l).w2()]).collect();
    // seed: any codes serve; both sides read the same file.
    let mut rng = Lcg::new(29);
    let mut tensors = Vec::new();
    let mut cast = Vec::new();
    for t in checkpoint::safetensors::read(&tiny.join("model.safetensors").to_string_lossy()).unwrap() {
        if !banks.contains(&t.name) {
            cast.push(t.name.clone());
            tensors.push((t.name, "F32", t.shape, f32_bytes(&t.data)));
            continue;
        }
        let (e, n, k) = (t.shape[0], t.shape[1], t.shape[2]);
        let layer: u32 = t.name.split('.').find_map(|p| p.parse().ok()).expect("a layer index in the bank's name");
        if nvfp4(layer) {
            let codes: Vec<u8> = (0..e * n * k / 2).map(|_| rng.next_u32() as u8).collect();
            let scales: Vec<u8> = (0..e * n * k / 16).map(|_| 0x28 + (rng.next_u32() % 0x10) as u8).collect();
            // A per-expert global of order one (amax / 2688), so the banks' products are of the activations' scale.
            let amax: Vec<f32> = (0..e).map(|i| 2688.0 * (0.3 + 0.05 * i as f32)).collect();
            tensors.push((t.name.clone(), "U8", vec![e, n, k / 2], codes));
            tensors.push((format!("{}_scale", t.name), "F8_E4M3", vec![e, n, k / 16], scales));
            tensors.push((format!("{}_amax", t.name), "F32", vec![e], f32_bytes(&amax)));
            tensors.push((format!("{}_outer_scale_denominator", t.name), "F32", vec![1], f32_bytes(&[2688.0])));
        } else {
            let values: Vec<u8> = (0..e * n * k).map(|_| (rng.next_u32() % 0x70) as u8 | if rng.next_u32().is_multiple_of(2) { 0x80 } else { 0 }).collect();
            let scales: Vec<u8> = (0..e * n * k / 32).map(|_| 116 + (rng.next_u32() % 4) as u8).collect();
            tensors.push((t.name.clone(), "F8_E4M3", vec![e, n, k], values));
            tensors.push((format!("{}_scale", t.name), "U8", vec![e, n, k / 32], scales));
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors);
    let mut groups = vec![serde_json::json!({"compute_dtype": "float32", "output_dtype": "float32", "required_activation_quantization": null,
        "tensors": cast, "weight_block_shape": [], "weight_transform": "cast"})];
    for l in 0..cfg.n_layers {
        let n = LayerNames::new(l);
        groups.push(if nvfp4(l) {
            serde_json::json!({"compute_dtype": "nvfp4", "output_dtype": "uint8", "required_activation_quantization": {"amax_type": "per_token", "block_size": 16,
                "format": "nvfp4_e2m1", "rounding": "nearest", "scale_dtype": "float8_e4m3fn", "scale_selection_rule": "fouroversix_mae"},
                "tensors": [n.w13(), n.w2()], "weight_block_shape": [1, 16], "weight_scale_format": "amax_denominator", "weight_transform": "nvfp4_nearest"})
        } else {
            serde_json::json!({"compute_dtype": "mxfp8", "output_dtype": "float8_e4m3fn", "required_activation_quantization": {"block_size": 32, "format": "mxfp8_e4m3",
                "rounding": "nearest", "scale_dtype": "ue8m0"}, "tensors": [n.w13(), n.w2()], "weight_block_shape": [32, 32], "weight_transform": "mxfp8_nearest"})
        });
    }
    std::fs::write(dir.join("hf_quant_config.json"), serde_json::json!({"model": {"parameter_groups": groups}, "quant_method": "inkling2"}).to_string()).unwrap();
    Some(dir)
}

/// Which layers' banks are NVFP4; the rest are MXFP8. An MXFP8 bank needs rows of a multiple of 128 inputs, wider than
/// the tiny model's 64 and 32, so every tiny bank is NVFP4 (the MXFP8 activation round trip is measured on the released
/// weights, `activation_tier`).
fn nvfp4(_layer: u32) -> bool {
    true
}

fn ops(choice: LinearActivations) -> model::ops::Ops {
    let mut ops = crate::tiny_parity::ops();
    ops.set_linear_activations(choice);
    ops
}

fn options() -> EngineOptions {
    EngineOptions { kv_tier: KvTier::F32, sizing: VisitPoolSizing { block_size: 8, num_blocks: 64, live_rows: 4, snapshot_rows: 4 }, max_seq_len: 128, prefill_chunk: 5 }
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale
}

/// The served logit rows of `seq`: its first `prompt` tokens prefilled in slices of `slice` (the grouped products),
/// every further token decoded teacher-forced (the gathered products).
fn served_rows(e: &mut Engine, seq: &[u32], prompt: usize, slice: usize) -> Vec<Vec<f32>> {
    let mut table = BlockTable::new();
    let hidden = loop {
        if let PrefillProgress::Done(h) = e.prefill_chunk(&mut table, &seq[..prompt], slice) {
            break h;
        }
    };
    let mut rows = vec![e.logits(&hidden)];
    for &t in &seq[prompt..seq.len() - 1] {
        rows.push(e.forward_batched_logit_rows(&mut [&mut table], &[t]).remove(0));
    }
    e.release_table(&mut table);
    rows
}

#[test]
fn the_declared_tier_is_served_as_the_uncached_forward_computes_it_in_prefill_and_decode() {
    let Some(dir) = declared_release("serve") else { return };
    let seq: Vec<u32> = (0..24u32).map(|i| (i * 29 + 5) % 60 + 3).collect();
    let prompt = 9;
    let mut e = Engine::open(ops(LinearActivations::Declared), &dir, options()).expect("the declared tier opens on a release that declares it");
    assert!(e.model().ops().linear_activations().is_declared());
    let whole = e.model().forward(&seq).unwrap();
    let v = e.vocab();
    for slice in [prompt, 4] {
        for (i, row) in served_rows(&mut e, &seq, prompt, slice).iter().enumerate() {
            let p = prompt - 1 + i;
            let err = rel_err(row, &whole[p * v..(p + 1) * v]);
            assert!(err <= TOL, "slices of {slice}: position {p} ({}) is {err:.3e} from the uncached forward under the declared tier", if i == 0 { "prefill" } else { "decode" });
        }
    }
    // The option reaches the products: the reference numerics serve other logits.
    let declared = served_rows(&mut e, &seq, prompt, prompt);
    let mut auto = Engine::open(ops(LinearActivations::Auto), &dir, options()).unwrap();
    let plain = served_rows(&mut auto, &seq, prompt, prompt);
    let moved = plain.iter().zip(&declared).map(|(a, b)| rel_err(b, a)).fold(0.0f32, f32::max);
    assert!(moved > 1e-3, "the declared tier changes the served numerics (at most {moved:.3e})");
    std::fs::remove_dir_all(&dir).ok();
}

fn conformance_engine() -> Engine {
    let dir = DIR.get().expect("written by the test").clone();
    // The chunk the engine's own conformance run uses (`serve.rs`): the suite's cancellation property needs the doomed
    // request decoding into a block of its own within two steps.
    Engine::open(ops(LinearActivations::Declared), &dir, EngineOptions { prefill_chunk: 7, ..options() }).expect("the declared engine")
}

static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

#[test]
fn the_declared_tier_passes_the_serving_conformance_suite() {
    let Some(dir) = declared_release("conformance") else { return };
    DIR.set(dir.clone()).expect("set once");
    conformance::run_all(&Conformance { make: conformance_engine, prefix_reuse: true, prefix_prompt_tokens: 20, load_adapter: None });
    std::fs::remove_dir_all(&dir).ok();
}
