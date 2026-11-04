// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The GGUF name map, gated against the safetensors route on real weights.
//!
//! Swedish Embedded AB implements checkpoint-format parity gating for its
//! clients. If your team needs expertise in proving that two loaders of one
//! model really produce the same weights then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # What this catches that loading does not
//!
//! Every mistake a tensor-name map can make is shape-compatible, and therefore
//! invisible to the loader:
//!
//! * The vision tower's `qkv` is ONE fused `[3H, H]` matrix in both formats, so
//!   a wrong q/k/v order inside it loads cleanly and produces attention over
//!   the wrong projections.
//! * `v.post_ln` and the merger's `norm` are both `[hidden]`, so swapping them
//!   with any other width-sized norm loads cleanly.
//! * The patch-embed Conv3d's two temporal slices hold the same element count
//!   whether they are interleaved (correct) or concatenated (wrong).
//! * On every GQA layer of the decoder, `k` and `v` have identical shapes.
//!
//! So the gate is elementwise, against the same checkpoint read through the
//! already-parity-gated safetensors path, for every tensor of every stage.
//!
//! # Tolerances, and why they are what they are
//!
//! The comparison is between two different numeric encodings of the same
//! weights, so exactness is not the bar and neither is a loose one:
//!
//! * The released safetensors are bf16 (8 mantissa bits) and the projector
//!   ships as F16 (10 mantissa bits), so every vision value round-trips
//!   exactly except where bf16's wider exponent range holds a value f16 must
//!   flush. A relative L2 of 1e-3 is orders of magnitude tighter than any
//!   misnaming and still leaves room for those.
//! * The language half ships as Q8_0: 8-bit values with one shared scale per
//!   32-element block, whose relative error is a few tenths of a percent.
//!   2e-2 admits that and nothing else. A swapped pair of tensors lands at
//!   cosine near zero, orders of magnitude outside either bound.
//!
//! # Running it
//!
//! Both checkpoints must be on the box; the test skips (loudly) otherwise.
//! The safetensors side is found in the model store, and the GGUF side is
//! named by `BRAIN_QWEN3VL_GGUF` (the language half, or the directory holding
//! it and its `mmproj-*.gguf`).

use std::collections::HashMap;

use brain_testutil::model_dir;

/// Relative L2 of `got` against `want`, and their cosine similarity.
fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut dot, mut ga, mut wa) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        num += (g - w) * (g - w);
        den += w * w;
        dot += g * w;
        ga += g * g;
        wa += w * w;
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
    let cos = if ga > 0.0 && wa > 0.0 { dot / (ga.sqrt() * wa.sqrt()) } else { 1.0 };
    (rel, cos)
}

fn check(stage: &str, got: &HashMap<String, Vec<f32>>, want: &HashMap<String, Vec<f32>>, rel_max: f64, cos_min: f64) {
    let mut names: Vec<&String> = want.keys().collect();
    names.sort();
    assert!(!names.is_empty(), "{stage}: the safetensors route produced no tensors");
    let (mut worst_rel, mut worst_name) = (0f64, String::new());
    for name in &names {
        let w = &want[*name];
        let g = got.get(*name).unwrap_or_else(|| panic!("{stage}: the GGUF route is missing {name}"));
        assert_eq!(g.len(), w.len(), "{stage}: {name} length {} vs {}", g.len(), w.len());
        let (rel, cos) = compare(g, w);
        assert!(cos >= cos_min, "{stage}: {name} cosine {cos:.9} < {cos_min} (a name map error, not a quantization one)");
        assert!(rel <= rel_max, "{stage}: {name} rel_l2 {rel:.6} > {rel_max}");
        if rel > worst_rel {
            worst_rel = rel;
            worst_name = (*name).clone();
        }
    }
    // Two-way: a tensor the GGUF route produced that the reference does not
    // have is a name this map invented, which would sit unused in the model.
    for name in got.keys() {
        assert!(want.contains_key(name), "{stage}: the GGUF route produced {name}, which the reference route does not have");
    }
    println!("  {stage:<12} {:>3} tensors, worst rel_l2 {worst_rel:.6} on {worst_name}", names.len());
}

#[test]
fn the_gguf_route_reproduces_the_safetensors_weights_tensor_for_tensor() {
    let Some(gguf) = std::env::var("BRAIN_QWEN3VL_GGUF").ok().filter(|s| !s.is_empty()) else {
        println!("skipped: set BRAIN_QWEN3VL_GGUF to the Qwen3-VL language-half GGUF (or its directory)");
        return;
    };
    let Some(hf) = model_dir("Qwen/Qwen3-VL-4B-Instruct").filter(|d| std::path::Path::new(d).join("config.json").exists()) else {
        println!("skipped: the Qwen/Qwen3-VL-4B-Instruct safetensors checkpoint is not in the model store");
        return;
    };

    let files = qwen3vl::gguf_import::GgufFiles::locate(std::path::Path::new(&gguf)).expect("locate the two GGUF halves");
    let tok = qwen3vl::gguf_import::tokenizer(&files).expect("the embedded tokenizer");
    let lm = checkpoint::gguf::MmapGguf::open(files.lm.to_str().unwrap()).unwrap();
    let mmproj = checkpoint::gguf::MmapGguf::open(files.mmproj.to_str().unwrap()).unwrap();
    let gcfg = qwen3vl::gguf_import::config(&lm, &mmproj, &tok).expect("config from the two files' own metadata");
    drop(lm);
    drop(mmproj);

    // The config derived from GGUF metadata must be the config the released
    // `config.json` states. Everything below compares weights; if the two
    // routes disagreed about the shape of the model, that comparison would be
    // meaningless.
    let hf_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{hf}/config.json")).unwrap()).unwrap();
    let hcfg = qwen3vl::Qwen3VlConfig::from_hf(&hf_json);
    assert_eq!(gcfg.vision, hcfg.vision, "vision config derived from GGUF KV differs from the released config.json");
    assert_eq!(gcfg.mrope_section, hcfg.mrope_section);
    assert_eq!(gcfg.image_token_id, hcfg.image_token_id);
    assert_eq!(gcfg.video_token_id, hcfg.video_token_id);
    assert_eq!(gcfg.vision_start_token_id, hcfg.vision_start_token_id);
    assert_eq!(gcfg.vision_end_token_id, hcfg.vision_end_token_id);
    for (name, g, h) in [
        ("n_layers", gcfg.text.n_layers, hcfg.text.n_layers),
        ("d_model", gcfg.text.d_model, hcfg.text.d_model),
        ("n_heads", gcfg.text.n_heads, hcfg.text.n_heads),
        ("n_kv_heads", gcfg.text.n_kv_heads, hcfg.text.n_kv_heads),
        ("head_dim", gcfg.text.head_dim, hcfg.text.head_dim),
        ("d_ff", gcfg.text.d_ff, hcfg.text.d_ff),
        ("vocab", gcfg.text.vocab, hcfg.text.vocab),
    ] {
        assert_eq!(g, h, "text config {name} from GGUF KV differs from config.json");
    }

    let got = qwen3vl::gguf_import::weights(&files, &gcfg).expect("read both GGUF halves");
    let hf_map: HashMap<String, Vec<f32>> = checkpoint::safetensors::read_model_dir(std::path::Path::new(&hf))
        .expect("read the safetensors checkpoint")
        .into_iter()
        .map(|t| (t.name, t.data))
        .collect();
    let want = qwen3vl::import::partition(hf_map, hcfg.vision.deepstack_indexes.len());

    // F16 projector against bf16 safetensors.
    check("vision", &got.vision, &want.vision, 1e-3, 0.999999);
    check("merger", &got.main_merger, &want.main_merger, 1e-3, 0.999999);
    for (k, (g, w)) in got.deepstack.iter().zip(&want.deepstack).enumerate() {
        check(&format!("deepstack{k}"), g, w, 1e-3, 0.999999);
    }
    // Q8_0 language half.
    check("decoder", &got.decoder, &want.decoder, 2e-2, 0.9995);
}
