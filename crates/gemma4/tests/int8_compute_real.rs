// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The int8 tier's correctness gate, on the REAL 12B checkpoint.
//!
//! Swedish Embedded AB implements and validates quantized inference paths for
//! its clients. If your team needs expertise in proving a low-precision
//! compute path agrees with its full-precision reference then you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! # Why this test and not a synthetic one
//!
//! An int8 text encoder that produces different embeddings does not fail
//! loudly - it silently changes the conditioning of every generation
//! downstream, and the output still looks like a video. So the thing that has
//! to be checked is the real checkpoint's own weight distributions, which is
//! where per-output-channel int8 is actually stressed: i.i.d. Gaussian test
//! weights have a per-row dynamic range that flatters the quantizer and
//! predicts nothing about a trained one.
//!
//! # Why BOTH layer indices
//!
//! Gemma-4's two layer types are not the same graph with a different mask.
//! A `sliding_attention` layer has `head_dim` 256, 8 KV heads and a real
//! `v_proj`; a `full_attention` layer has `head_dim` 512, ONE KV head (MQA)
//! and no `v_proj` at all - its values are the pre-norm keys. Gating only a
//! sliding layer would leave the entire `attention_k_eq_v` path unmeasured,
//! and it is the path where a quantization error has the least redundancy to
//! hide in (one KV head, not eight). Layer 0 is sliding and layer 5 is the
//! first full layer, so this covers one of each.
//!
//! # Why cosine AND rel_l2
//!
//! Cosine is scale-invariant: a systematically mis-scaled output scores
//! 1.0. The one error a per-channel scale can make is a scale error, so the
//! magnitude has to be checked separately or the check cannot see its own
//! most likely failure.

use gemma4::block::{open_device, Gemma4Layer, Precision};
use gemma4::config::Gemma4Config;
use gemma4::rope::{full_table, sliding_table, upload_rope};
use gemma4::gguf_src::Gemma4GgufSource;

const REPO: &str = "Lightricks/LTX-2.5";

/// The real Q8_0 GGUF, from `$BRAIN_GEMMA4_GGUF` or the model store. Produced
/// by `brain quantize <bf16 safetensors> --out <path> --arch gemma4`.
fn gguf_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_GEMMA4_GGUF") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("gemma4") && n.contains("Q8_0") && n.ends_with(".gguf")))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

/// f64-accumulated, so the comparison is not itself the thing losing
/// precision at these widths.
fn cosine_and_rel_l2(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum::<f64>().sqrt();
    let diff: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb), diff / na)
}

/// Deterministic, non-degenerate `[t, hidden]` activation.
fn activation(t: usize, hidden: usize) -> Vec<f32> {
    (0..t * hidden).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 1.3).collect()
}

#[test]
fn real_q8_0_layer_int8_compute_matches_fp32_on_both_attention_types() {
    let Some(path) = gguf_path() else {
        brain_testutil::skip(&format!(
            "set BRAIN_GEMMA4_GGUF to a real {REPO} Gemma-4 Q8_0 GGUF (none in the model store); \
             produce one with: brain quantize <gemma4 bf16 safetensors> --out <path> --arch gemma4"
        ));
        return;
    };

    let cfg = Gemma4Config::gemma4_12b();
    let src = Gemma4GgufSource::open(&path, &cfg).unwrap_or_else(|e| panic!("opening {path}: {e}"));

    let gpu = open_device(None);
    if !gpu.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("this device exposes no packed-int8 dot path, so there is no int8 tier to compare");
        return;
    }

    // Smallest shape that still exercises the sliding window's own boundary
    // is far larger than needed to catch a quantization error; the error
    // this gate is looking for is per-weight, not per-position, so a short
    // sequence is the honest choice for a real-weight test.
    let (t, hidden) = (8usize, cfg.hidden_size as usize);
    let x = activation(t, hidden);

    let sliding_tbl = sliding_table(cfg.head_dim, cfg.rope_theta_sliding, t);
    let full_tbl = full_table(cfg.global_head_dim, cfg.rope_theta_full, cfg.partial_rotary_factor, t);
    let sliding_rope = upload_rope(&gpu, &sliding_tbl);
    let full_rope = upload_rope(&gpu, &full_tbl);

    let mut checked = Vec::new();
    for l in [0u32, 5] {
        let lt = cfg.layer_type(l);
        let tensors = gemma4::load_layer_tensors(&src, &cfg, l).unwrap_or_else(|e| panic!("layer {l}: {e}"));
        let rope = match lt {
            gemma4::LayerType::Sliding => &sliding_rope,
            gemma4::LayerType::Full => &full_rope,
        };

        let f32_layer = Gemma4Layer::on(gpu.share(), &cfg, &tensors, l, Precision::Fp32);
        let (out_f32, attn_f32) = f32_layer.forward(&x, rope, t as u32);
        drop(f32_layer);

        let i8_layer = Gemma4Layer::on(gpu.share(), &cfg, &tensors, l, Precision::Int8);
        let (out_i8, attn_i8) = i8_layer.forward(&x, rope, t as u32);
        drop(i8_layer);

        let (cos_out, rel_out) = cosine_and_rel_l2(&out_f32, &out_i8);
        let (cos_attn, rel_attn) = cosine_and_rel_l2(&attn_f32, &attn_i8);
        println!("real Q8_0 layer {l} ({lt:?}) int8 vs fp32: out cosine {cos_out:.9} rel_l2 {rel_out:.6}, self_attn cosine {cos_attn:.9} rel_l2 {rel_attn:.6}");

        // The floor, not the measurement. `ltxv`'s own real-weight int8
        // block gate sets 0.99 against a measured 0.9963 for the same
        // reason: int8 is a lossy tier, so the assertion only has to catch a
        // BROKEN port (a wrong scale, a mis-packed lane, a transposed
        // operand), never to reproduce one particular run's digits. The
        // printed numbers above are the record of what was actually
        // measured.
        assert!(cos_out >= 0.99, "layer {l} ({lt:?}) int8 output diverged from fp32: cosine {cos_out:.9}");
        assert!(rel_out <= 0.15, "layer {l} ({lt:?}) int8 output magnitude diverged from fp32: rel_l2 {rel_out:.6}");
        assert!(cos_attn >= 0.99, "layer {l} ({lt:?}) int8 self-attention diverged from fp32: cosine {cos_attn:.9}");
        // The load-bearing assertion, and the one this test did not have
        // until a deliberate mutation showed why it needed it. Halving every
        // per-output-channel weight scale - the single most likely packing
        // bug in this tier - left `cos_attn` BIT-IDENTICAL (cosine cannot see
        // a scale factor at all) and left the layer OUTPUT inside every
        // tolerance above, because `post_attention_layernorm` renormalizes
        // the attention branch and the residual dominates what survives.
        // Only the attention tap's own magnitude sees it, and it sees it
        // enormously: rel_l2 0.061 -> 0.503.
        //
        // Generalizes past this model: a tap measured DOWNSTREAM of a
        // normalization cannot gate a scale error introduced upstream of it,
        // whatever metric it uses.
        assert!(rel_attn <= 0.15, "layer {l} ({lt:?}) int8 self-attention magnitude diverged from fp32: rel_l2 {rel_attn:.6}");
        checked.push(lt);
    }

    // The point of the loop, asserted rather than assumed: a re-indexing
    // that accidentally picked two layers of the same type would silently
    // stop covering one of the two attention paths.
    assert!(checked.contains(&gemma4::LayerType::Sliding), "no sliding_attention layer was checked");
    assert!(checked.contains(&gemma4::LayerType::Full), "no full_attention layer was checked");
}

/// The int8 tier must be a REQUEST, resolved against the device, never a
/// hardcoded choice - so a device whose fast path is fp32 is not dragged
/// through a quantize/dequantize detour for nothing.
#[test]
fn int8_is_requested_and_resolved_against_the_device_never_assumed() {
    let gpu = open_device(None);
    let resolved = Precision::for_device(&gpu, Precision::Int8);
    if gpu.caps().numeric.int8_dot {
        assert_eq!(resolved, Precision::Int8, "a device with a packed-int8 path must honour the request");
    } else {
        assert_eq!(resolved, Precision::Fp32, "a device without a packed-int8 path must fall back, not quantize for nothing");
    }
    // fp32 is never promoted: asking for the portable tier always gets it.
    assert_eq!(Precision::for_device(&gpu, Precision::Fp32), Precision::Fp32);
}

/// The whole encoder, both tiers, same file: 48 layers end to end rather than
/// one layer in isolation.
///
/// Opt-in (`BRAIN_GEMMA4_FULL_PARITY=1`) because it reads the real 13 GiB
/// checkpoint twice and would dominate the default suite's budget. The
/// per-layer gate above is what runs by default, matching `ltxv`'s own
/// precedent of gating one real block rather than a whole real forward. This
/// exists so the end-to-end number is reproducible by command rather than
/// being a one-off someone wrote in a document.
///
/// Both arms read the SAME quantized file, so the only difference is the
/// arithmetic - which is exactly the comparison wanted, and is why
/// `BRAIN_LTXV_TEXT_PRECISION` exists on the pipeline side too. Comparing an
/// int8 GGUF forward against a bf16 safetensors forward would confound the
/// tier with the storage format.
#[test]
fn real_q8_0_whole_encoder_int8_matches_fp32() {
    if std::env::var("BRAIN_GEMMA4_FULL_PARITY").as_deref() != Ok("1") {
        brain_testutil::skip("set BRAIN_GEMMA4_FULL_PARITY=1 to run the whole-encoder int8-vs-fp32 comparison (reads the real checkpoint twice)");
        return;
    }
    let Some(path) = gguf_path() else {
        brain_testutil::skip("set BRAIN_GEMMA4_GGUF to a real Gemma-4 Q8_0 GGUF");
        return;
    };
    let cfg = Gemma4Config::gemma4_12b();
    let src = Gemma4GgufSource::open(&path, &cfg).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let ids: Vec<u32> = vec![2, 476, 5479, 12817, 611, 496, 8698, 1];

    let f32_out = gemma4::forward_streamed(&cfg, &src, None, Precision::Fp32, &ids).expect("fp32 forward");
    let i8_out = gemma4::forward_streamed(&cfg, &src, None, Precision::Int8, &ids).expect("int8 forward");

    let (cos, rel) = cosine_and_rel_l2(&f32_out.last_hidden_state, &i8_out.last_hidden_state);
    println!("real Q8_0 WHOLE encoder ({} layers) int8 vs fp32: last_hidden_state cosine {cos:.9} rel_l2 {rel:.6}", cfg.num_hidden_layers);
    assert_eq!(f32_out.hidden_states.len(), i8_out.hidden_states.len());
    for (i, (a, b)) in f32_out.hidden_states.iter().zip(&i8_out.hidden_states).enumerate() {
        let (c, r) = cosine_and_rel_l2(a, b);
        assert!(c >= 0.98, "hidden_states[{i}] diverged: cosine {c:.9}");
        assert!(r <= 0.30, "hidden_states[{i}] magnitude diverged: rel_l2 {r:.6}");
    }
    assert!(cos >= 0.98, "whole-encoder int8 output diverged from fp32: cosine {cos:.9}");
    assert!(rel <= 0.30, "whole-encoder int8 magnitude diverged from fp32: rel_l2 {rel:.6}");
}
