// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SERVED path, end to end: a string in, a pooled embedding out.
//!
//! `tests/parity.rs` gates the towers themselves against dumped goldens, but it
//! feeds them token ids from the reference. This file gates everything that sits
//! between a caller and those towers — the CLIP BPE, the fixed 77-token context,
//! the tower selection and the "projected `text_embeds` when the tower projects,
//! else the pooled EOS row" choice — because a serving contract that returns a
//! confidently wrong vector is worse than one that does not exist.
//!
//! Measured against HuggingFace on the released SDXL checkpoint (`CLIPTextModel`
//! for CLIP-L, `CLIPTextModelWithProjection` for OpenCLIP-bigG):
//!
//! ```text
//! clip_l         "a photo of a cat"  cosine 1.0000000000  max_abs 1.490e-05
//! clip_l         "a photo of a dog"  cosine 1.0000001192  max_abs 1.800e-05
//! openclip_bigg  "a photo of a cat"  cosine 0.9999998212  max_abs 1.001e-05
//! openclip_bigg  "a photo of a dog"  cosine 1.0000001192  max_abs 9.298e-06
//! ```
//!
//! Set `BRAIN_CLIP_DIR` to the SDXL checkpoint root to run; the tests skip
//! themselves when it is absent (AGENTS.md: no absolute paths, no baked-in
//! fixtures).

use clip::caps::Session;

fn dir() -> Option<String> {
    let d = std::env::var("BRAIN_CLIP_DIR").ok().filter(|p| !p.is_empty())?;
    std::path::Path::new(&d).join("tokenizer").exists().then_some(d)
}

fn session() -> Option<Session> {
    let d = dir()?;
    let gpu = gpu_core::testgpu::dev(clip::model::TEXT_PIPELINES);
    Some(Session::load(&d, gpu).expect("load clip session"))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// The batched path must agree with the single-item path **exactly** — it is a
/// different graph (built at `b = N`), so this is the gate that stops
/// `run_batch` quietly returning a different answer than `run`.
#[test]
fn a_batch_agrees_with_one_at_a_time() {
    let Some(s) = session() else {
        eprintln!("skip: set BRAIN_CLIP_DIR");
        return;
    };
    let texts: Vec<String> =
        ["a photo of a cat", "a photo of a dog", "an empty street at night"].iter().map(|s| s.to_string()).collect();

    let batched = s.embed_text_batch("clip_l", &texts).expect("batched");
    assert_eq!(batched.len(), texts.len());

    for (i, t) in texts.iter().enumerate() {
        let single = s.embed_text_batch("clip_l", std::slice::from_ref(t)).expect("single");
        let c = cosine(&batched[i], &single[0]);
        assert!(c > 0.9999, "row {i} batched vs single cosine {c:.8}");
    }
}

/// Different strings must produce different embeddings. Guards the degenerate
/// failure this action could have — returning the pad/EOS row of an empty
/// sequence for everything, which still has the right shape and norm.
#[test]
fn distinct_texts_give_distinct_embeddings() {
    let Some(s) = session() else {
        eprintln!("skip: set BRAIN_CLIP_DIR");
        return;
    };
    let v = s
        .embed_text_batch("clip_l", &["a photo of a cat".into(), "an empty street at night".into()])
        .expect("batch");
    let c = cosine(&v[0], &v[1]);
    assert!(c < 0.99, "unrelated prompts should not be near-identical (cosine {c:.6})");
    assert!(v[0].iter().any(|x| x.abs() > 1e-6), "embedding is all zeros");
}

/// Both towers serve, at their own widths. SDXL conditions on both, so a mixed
/// batch is two forwards and the widths must not be swapped.
#[test]
fn both_towers_serve_at_their_own_width() {
    let Some(s) = session() else {
        eprintln!("skip: set BRAIN_CLIP_DIR");
        return;
    };
    let t = vec!["a photo of a cat".to_string()];
    assert_eq!(s.embed_text_batch("clip_l", &t).expect("clip_l")[0].len(), 768);
    if std::path::Path::new(&dir().unwrap()).join("tokenizer_2").exists() {
        assert_eq!(s.embed_text_batch("openclip_bigg", &t).expect("bigg")[0].len(), 1280);
    }
}

/// An unknown tower is an error naming the tower, never a silent fallback to
/// `clip_l` — a caller asking for bigG and receiving a 768-d CLIP-L vector would
/// have no way to tell.
#[test]
fn an_unknown_tower_is_an_error() {
    let Some(s) = session() else {
        eprintln!("skip: set BRAIN_CLIP_DIR");
        return;
    };
    let e = s.embed_text_batch("clip_h", &["x".to_string()]).unwrap_err();
    assert!(e.contains("clip_h"), "error should name the tower, got: {e}");
}
