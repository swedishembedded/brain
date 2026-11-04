// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAGNOSTIC real-weight, real-width parity check for Gemma-4 - never
//! attempted before at any scale beyond the tiny random-dim config.
//! `parity.rs::gemma4_tiny_matches_reference` proves the op sequence at
//! tiny random dims with every structural flag set to the real value; this
//! proves the SAME op sequence at real width (hidden=3840, head_dim=256/512,
//! GQA/MQA, k_eq_v) on the first 6 real layers (5 sliding + 1 full, the
//! real 5:1 `sliding_window_pattern`'s minimal instance) loaded straight
//! from the real 26 GB bf16 checkpoint.
//!
//! Golden: `tools/goldens/gemma4_real_dump_reference.py`.

use std::path::Path;

use checkpoint::mmap::MmapSafetensors;
use gemma4::block::Tensors;
use gemma4::{Gemma4Config, Gemma4Model};

const REPO: &str = "Lightricks/LTX-2.5";
const LAYERS: u32 = 6;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn report(label: &str, got: &[f32], want: &[f32], min_cos: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, m) = (cosine(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.9}  max_abs={m:.3e}  n={}", got.len());
    assert!(c >= min_cos, "{label}: cosine {c:.9} < {min_cos}");
}

/// Golden dumps `rotary_emb`'s own full-duplicated-width return; brain
/// stores only the compact active-frequency prefix - `parity.rs::row_prefix`'s
/// doc has the full reasoning, reproduced here since integration tests each
/// compile as their own crate.
fn row_prefix(golden_flat: &[f32], t: usize, full_width: usize, half: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(t * half);
    for row in 0..t {
        out.extend_from_slice(&golden_flat[row * full_width..row * full_width + half]);
    }
    out
}

fn real_gemma4_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_TEXT_ENCODER") {
        if !p.is_empty() && p.ends_with(".safetensors") {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("gemma4") && n.ends_with(".safetensors")))
        .map(|p| p.to_string_lossy().into_owned())
}

fn load_real_weights(mt: &MmapSafetensors, layers: u32) -> Tensors {
    let mut out = Tensors::new();
    let mut want = |name: &str| {
        let t = mt.tensor(name).unwrap_or_else(|| panic!("real gemma4 checkpoint: missing tensor {name}"));
        // `import_gemma4`'s own renaming: strip the `model.` prefix, keep
        // `text_embedding_projection.*` verbatim - see `crate::import::
        // classify`'s doc.
        let canonical = name.strip_prefix("model.").unwrap_or(name).to_string();
        out.insert(canonical, (t.shape, t.data));
    };
    want("model.embed_tokens.weight");
    want("model.norm.weight");
    for l in 0..layers {
        let p = format!("model.layers.{l}");
        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            want(&format!("{p}.{suffix}"));
        }
        // Full-attention layers (`attention_k_eq_v=True`) have no separate
        // v_proj in the real checkpoint - see the dumper's own comment.
        let is_full = (l + 1) % 6 == 0;
        if !is_full {
            want(&format!("{p}.self_attn.v_proj.weight"));
        }
    }
    out
}

struct Fixture {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
}

#[test]
fn gemma4_real_reduced_layers_matches_reference() {
    let fx_path = brain_testutil::testdata("golden/gemma4/gemma4_real_reduced.safetensors");
    if !Path::new(&fx_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/gemma4_real_dump_reference.py"));
        return;
    }
    let Some(weights_path) = real_gemma4_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_TEXT_ENCODER to a real {REPO} Gemma-4 bf16 safetensors (none in the model store)"));
        return;
    };

    let mt = MmapSafetensors::open(&weights_path).unwrap_or_else(|e| panic!("opening {weights_path}: {e}"));
    let cfg = Gemma4Config { num_hidden_layers: LAYERS, ..Gemma4Config::gemma4_12b() };
    let n = cfg.num_hidden_layers as usize;
    let hidden = cfg.hidden_size as usize;

    let t0 = std::time::Instant::now();
    let w = load_real_weights(&mt, LAYERS);
    eprintln!("real weight subset loaded ({} tensors) in {:.2}s", w.len(), t0.elapsed().as_secs_f64());

    let fx = Fixture { t: checkpoint::safetensors::read(&fx_path).expect("read golden") };
    let input_ids: Vec<u32> = fx.get("input_ids").iter().map(|&v| v.round() as u32).collect();
    let t = input_ids.len();

    let model = Gemma4Model::new(cfg, w, None);
    let t1 = std::time::Instant::now();
    let out = model.forward(&input_ids);
    eprintln!("real-weight forward ({LAYERS} layers, {t} tokens, hidden={hidden}) ran in {:.2}s", t1.elapsed().as_secs_f64());

    let sliding_half = out.rope_sliding_cos.len() / t;
    let full_half = out.rope_full_cos.len() / t;
    let sliding_full_width = cfg.head_dim as usize;
    let full_full_width = cfg.global_head_dim as usize;
    report("rope_sliding_cos", &out.rope_sliding_cos, &row_prefix(fx.get("rope_sliding_cos"), t, sliding_full_width, sliding_half), 0.999999);
    report("rope_sliding_sin", &out.rope_sliding_sin, &row_prefix(fx.get("rope_sliding_sin"), t, sliding_full_width, sliding_half), 0.999999);
    report("rope_full_cos", &out.rope_full_cos, &row_prefix(fx.get("rope_full_cos"), t, full_full_width, full_half), 0.999999);
    report("rope_full_sin", &out.rope_full_sin, &row_prefix(fx.get("rope_full_sin"), t, full_full_width, full_half), 0.999999);

    report("layer0_self_attn_out (sliding, GQA)", &out.layer0_self_attn_out, fx.get("layer0_self_attn_out"), 0.999);
    report(&format!("layer{}_self_attn_out (full, MQA + k_eq_v)", n - 1), &out.layer_last_self_attn_out, fx.get(&format!("layer{}_self_attn_out", n - 1)), 0.999);

    assert_eq!(out.hidden_states.len(), n + 1);
    for (k, hs) in out.hidden_states.iter().enumerate() {
        report(&format!("hidden_states.{k}"), hs, fx.get(&format!("hidden_states.{k}")), 0.999);
    }
    report("last_hidden_state", &out.last_hidden_state, fx.get("last_hidden_state"), 0.999);

    // The real checkpoint's aggregate-embed weight is sized for the real
    // model's full 49 hidden states (`hidden*49` in_features) - out of
    // scope for this reduced `n+1=7`-state test (`AggregateEmbed` itself
    // is already parity-proven at tiny scale, `parity.rs::gemma4_tiny_
    // matches_reference`; this test's own job is the 48-layer TOWER, not
    // the aggregate projection on top of it).
}
