// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The surgery edits on toy networks evaluated here on the host: what an
//! edit promises (the narrower network computes what the masked one does, a
//! full-width rotation computes what the original does) checked on outputs.

use std::collections::BTreeMap;

use data::rng::Lcg;
use model::hostmath::{linear_rows, silu};
use model::surgery::{Axis, ChannelDim, ConfigWidth, EmbeddingFold, LayerStack, Lines, Stream, StreamNorm, StreamReader, Vocabulary};
use serde_json::json;

use super::*;

const D: usize = 8;
const F: usize = 6;
const V: usize = 5;
const L: usize = 3;

fn filled(rng: &mut Lcg, shape: &[usize], scale: f32) -> Tensor {
    let n = shape.iter().product();
    Tensor::new(shape.to_vec(), rng.vec_scaled(n, scale)).unwrap()
}

/// A toy residual network: `h = E[tok]`, per layer `h += b_l + Wo_l silu(Wi_l (rms(h) ⊙ n_l))`, a final norm `f`,
/// logits `C (rms(h) ⊙ f)` and a side projection `S` that reads the final norm's output and the raw embedding.
fn toy(seed: u64) -> Checkpoint {
    let mut rng = Lcg::new(seed);
    let mut t = BTreeMap::new();
    t.insert("emb".to_string(), filled(&mut rng, &[V, D], 1.0));
    for l in 0..L {
        t.insert(format!("layers.{l}.norm"), Tensor::new(vec![D], (0..D).map(|i| 0.5 + 0.1 * i as f32 + 0.05 * l as f32).collect()).unwrap());
        t.insert(format!("layers.{l}.wi"), filled(&mut rng, &[F, D], 0.5));
        t.insert(format!("layers.{l}.wo"), filled(&mut rng, &[D, F], 0.5));
        t.insert(format!("layers.{l}.bias"), filled(&mut rng, &[D], 0.2));
    }
    t.insert("final".to_string(), Tensor::new(vec![D], (0..D).map(|i| 2.0 - 0.15 * i as f32).collect()).unwrap());
    t.insert("head".to_string(), filled(&mut rng, &[V, D], 0.5));
    t.insert("side".to_string(), filled(&mut rng, &[4, D], 0.5));
    Checkpoint { config: json!({"model": {"width": D, "ff": F, "layers": L}, "vocab": V, "bos": 3, "types": ["a", "b", "c"]}), tensors: t }
}

fn rms(x: &[f32], g: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + 1e-6).sqrt();
    x.iter().zip(g).map(|(v, g)| v * inv * g).collect()
}

/// `(logits, side(final norm), side(raw embedding))` per token.
fn run(ck: &Checkpoint, tokens: &[usize]) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let w = |n: &str| &ck.get(n).unwrap().data;
    let d = ck.get("final").unwrap().data.len();
    let layers = ck.config_u64("model.layers").unwrap() as usize;
    let f = ck.get("layers.0.wi").unwrap().shape[0];
    tokens
        .iter()
        .map(|&tok| {
            let e = w("emb")[tok * d..(tok + 1) * d].to_vec();
            let mut h = e.clone();
            for l in 0..layers {
                let a = rms(&h, w(&format!("layers.{l}.norm")));
                let mid: Vec<f32> = linear_rows(&a, w(&format!("layers.{l}.wi")), 1, d, f).into_iter().map(silu).collect();
                let out = linear_rows(&mid, w(&format!("layers.{l}.wo")), 1, f, d);
                h.iter_mut().zip(out.iter().zip(w(&format!("layers.{l}.bias")))).for_each(|(h, (o, b))| *h += o + b);
            }
            let o = rms(&h, w("final"));
            let head_rows = ck.get("head").unwrap().shape[0];
            (linear_rows(&o, w("head"), 1, d, head_rows), linear_rows(&o, w("side"), 1, d, 4), linear_rows(&e, w("side"), 1, d, 4))
        })
        .collect()
}

fn stream(ck: &Checkpoint, shared_final: bool, with_side: bool) -> Stream {
    let layers = ck.config_u64("model.layers").unwrap() as usize;
    let mut writers = vec![("emb".to_string(), Axis::Cols)];
    let mut readers = Vec::new();
    let mut norms = Vec::new();
    for l in 0..layers {
        writers.push((format!("layers.{l}.wo"), Axis::Rows));
        writers.push((format!("layers.{l}.bias"), Axis::Rows));
        readers.push(StreamReader { tensor: format!("layers.{l}.wi"), norm: Some(format!("layers.{l}.norm")) });
        norms.push(StreamNorm { tensor: format!("layers.{l}.norm"), shared: false });
    }
    readers.push(StreamReader { tensor: "head".into(), norm: Some("final".into()) });
    if with_side {
        readers.push(StreamReader { tensor: "side".into(), norm: None });
    }
    norms.push(StreamNorm { tensor: "final".into(), shared: shared_final });
    Stream { name: "residual".into(), width: ck.get("final").unwrap().data.len(), keys: vec!["model.width".into()], writers, readers, norms }
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + x.abs()))
}

/// A random symmetric positive matrix, as a calibration second moment.
fn covariance(seed: u64, d: usize) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let x = rng.vec_scaled(4 * d * d, 1.0);
    let mut c = vec![0f64; d * d];
    for row in x.chunks(d) {
        for i in 0..d {
            for j in 0..d {
                c[i * d + j] += f64::from(row[i]) * f64::from(row[j]);
            }
        }
    }
    c
}

const TOKENS: [usize; 5] = [0, 1, 2, 3, 4];

#[test]
fn a_full_width_rotation_computes_the_same_network() {
    let ck = toy(0x5eed); // seed: fixed fixture
    let want = run(&ck, &TOKENS);
    let basis = basis_from_covariance(&covariance(7, D), D, BasisKind::Pca).unwrap(); // seed: fixed fixture
    let mut rotated = ck.clone();
    let change = rotate_stream(&mut rotated, &stream(&ck, false, false), &basis, D, &[]).unwrap();
    assert_eq!(change.params_before, change.params_after);
    for ((a, _, _), (b, _, _)) in want.iter().zip(run(&rotated, &TOKENS)) {
        assert!(close(a, &b, 1e-4), "{a:?} vs {b:?}");
    }
}

/// A stream whose every writer lies in a `k`-dimensional subspace loses nothing when cut to that subspace.
#[test]
fn a_stream_inside_k_directions_is_cut_to_k_without_loss() {
    let mut ck = toy(0xa11); // seed: fixed fixture
    let k = 5;
    // Zero the last D - k coordinates of every writer: the stream lives in the first k, so the identity basis
    // ordered by energy keeps exactly them.
    for name in ck.tensors.keys().cloned().collect::<Vec<_>>() {
        let t = ck.tensors.get_mut(&name).unwrap();
        if name == "emb" {
            t.data.chunks_mut(D).for_each(|r| r[k..].fill(0.0));
        } else if name.ends_with(".wo") || name.ends_with(".bias") {
            let cols = t.rows_cols().1;
            t.data[k * cols..].fill(0.0);
        }
    }
    let want = run(&ck, &TOKENS);
    let mut cov = vec![0f64; D * D];
    (0..k).for_each(|i| cov[i * D + i] = 1.0 + i as f64);
    for kind in [BasisKind::Identity, BasisKind::Pca] {
        let basis = basis_from_covariance(&cov, D, kind).unwrap();
        let mut cut = ck.clone();
        let change = rotate_stream(&mut cut, &stream(&ck, false, false), &basis, k, &[]).unwrap();
        assert_eq!(change.config["model.width"], [json!(D), json!(k)]);
        assert_eq!(cut.get("emb").unwrap().shape, vec![V, k]);
        assert_eq!(cut.get("layers.1.wo").unwrap().shape, vec![k, F]);
        for ((a, _, _), (b, _, _)) in want.iter().zip(run(&cut, &TOKENS)) {
            assert!(close(a, &b, 1e-4), "{kind:?}: {a:?} vs {b:?}");
        }
    }
}

/// A shared final norm keeps the head exact under any rotation and the shared projection exact under a
/// permutation; under a general rotation only the raw-embedding input of the projection stays exact.
#[test]
fn a_shared_norm_keeps_its_exclusive_readers_exact() {
    let ck = toy(0xbee); // seed: fixed fixture
    let want = run(&ck, &TOKENS);
    let decl = stream(&ck, true, true);
    let cov = covariance(9, D); // seed: fixed fixture
    let fit = SharedNormFit { norm: "final".into(), covariance: cov.clone() };
    let mut pca = ck.clone();
    rotate_stream(&mut pca, &decl, &basis_from_covariance(&cov, D, BasisKind::Pca).unwrap(), D, std::slice::from_ref(&fit)).unwrap();
    let mut perm = ck.clone();
    rotate_stream(&mut perm, &decl, &basis_from_covariance(&cov, D, BasisKind::Identity).unwrap(), D, &[fit]).unwrap();
    for (w, (p, q)) in want.iter().zip(run(&pca, &TOKENS).into_iter().zip(run(&perm, &TOKENS))) {
        assert!(close(&w.0, &p.0, 1e-4) && close(&w.0, &q.0, 1e-4), "head");
        assert!(close(&w.2, &p.2, 1e-4) && close(&w.2, &q.2, 1e-4), "side on the raw embedding");
        assert!(close(&w.1, &q.1, 1e-4), "side on the norm, permuted: {:?} vs {:?}", w.1, q.1);
    }
}

#[test]
fn a_rotation_refuses_a_misdeclared_stream_and_touches_nothing() {
    let ck = toy(3); // seed: fixed fixture
    let basis = basis_from_covariance(&covariance(1, D), D, BasisKind::Pca).unwrap(); // seed: fixed fixture
    let mut twice = stream(&ck, false, false);
    twice.writers.push(("emb".into(), Axis::Cols));
    let mut c = ck.clone();
    assert!(rotate_stream(&mut c, &twice, &basis, 4, &[]).unwrap_err().contains("declared twice"));
    let mut wrong = stream(&ck, false, false);
    wrong.readers.push(StreamReader { tensor: "layers.0.wo".into(), norm: None });
    wrong.writers.retain(|(n, _)| n != "layers.0.wo");
    assert!(rotate_stream(&mut c, &wrong, &basis, 4, &[]).unwrap_err().contains("not [_, 8]"));
    assert!(rotate_stream(&mut c, &stream(&ck, false, false), &basis, 0, &[]).is_err());
    assert_eq!(c, ck);
}

fn mlp_dims(ck: &Checkpoint) -> Vec<ChannelDim> {
    (0..ck.config_u64("model.layers").unwrap() as usize)
        .map(|l| ChannelDim {
            name: format!("layers.{l}.mlp"),
            width: F,
            keys: vec![ConfigWidth::new("model.ff", 1)],
            lines: vec![Lines::new(format!("layers.{l}.wi"), Axis::Rows, 1), Lines::new(format!("layers.{l}.wo"), Axis::Cols, 1)],
            masks: vec![format!("layers.{l}.wo")],
        })
        .collect()
}

#[test]
fn selected_channels_compute_what_the_masked_network_does() {
    let ck = toy(0xc0); // seed: fixed fixture
    let dims = mlp_dims(&ck);
    let refs: Vec<&ChannelDim> = dims.iter().collect();
    let keep = vec![vec![0, 2, 5], vec![1, 2, 3], vec![0, 4, 5]];
    let mut masked = ck.clone();
    mask_channels(&mut masked, &refs, &keep).unwrap();
    let mut cut = ck.clone();
    let change = select_channels(&mut cut, &refs, &keep).unwrap();
    assert_eq!(change.config["model.ff"], [json!(F), json!(3)]);
    assert_eq!(cut.get("layers.2.wi").unwrap().shape, vec![3, D]);
    assert_eq!(change.params_before - change.params_after, (L * 3 * 2 * D) as u64);
    for (a, b) in run(&masked, &TOKENS).iter().zip(run(&cut, &TOKENS)) {
        assert!(close(&a.0, &b.0, 1e-5));
    }
    let mut uneven = ck.clone();
    let err = select_channels(&mut uneven, &refs, &[vec![0, 1], vec![1, 2, 3], vec![0, 4, 5]]).unwrap_err();
    assert!(err.contains("model.ff"), "{err}");
    assert_eq!(uneven, ck, "a refused selection touches nothing");
}

#[test]
fn removed_layers_compute_what_the_masked_network_does_and_renumber() {
    let ck = toy(0xd0); // seed: fixed fixture
    let stack = LayerStack { name: "layers".into(), prefix: "layers.".into(), count: L, key: "model.layers".into(), outputs: vec!["wo".into(), "bias".into()], lists: vec!["types".into()] };
    let mut masked = ck.clone();
    mask_layers(&mut masked, &stack, &[1]).unwrap();
    let mut cut = ck.clone();
    let change = remove_layers(&mut cut, &stack, &[1]).unwrap();
    assert_eq!(change.kept["layers"], vec![0, 2]);
    assert_eq!(cut.config["types"], json!(["a", "c"]));
    assert_eq!(cut.get("layers.1.wi").unwrap(), ck.get("layers.2.wi").unwrap());
    assert!(cut.get("layers.2.wi").is_err());
    for (a, b) in run(&masked, &TOKENS).iter().zip(run(&cut, &TOKENS)) {
        assert!(close(&a.0, &b.0, 1e-6));
    }
    assert!(remove_layers(&mut cut.clone(), &stack, &[0, 0]).is_err());
}

/// `out = W2 silu(W1 E[t] + b1) + b2`, as a text projection computes it.
fn projected(ck: &Checkpoint, tok: usize) -> Vec<f32> {
    let w = |n: &str| ck.get(n).unwrap();
    let (h, out) = (w("w1").shape[0], w("w2").shape[0]);
    let e = &w("e").data[tok * h..(tok + 1) * h];
    let mut mid = linear_rows(e, &w("w1").data, 1, h, h);
    mid.iter_mut().zip(&w("b1").data).for_each(|(m, b)| *m = silu(*m + b));
    let mut y = linear_rows(&mid, &w("w2").data, 1, h, out);
    y.iter_mut().zip(&w("b2").data).for_each(|(y, b)| *y += b);
    y
}

#[test]
fn a_folded_embedding_computes_its_kept_channels_exactly() {
    let mut rng = Lcg::new(0xf0); // seed: fixed fixture
    let h = 7;
    let mut t = BTreeMap::new();
    for (n, s) in [("e", vec![V, h]), ("w1", vec![h, h]), ("b1", vec![h]), ("w2", vec![4, h]), ("b2", vec![4])] {
        t.insert(n.to_string(), filled(&mut rng, &s, 0.7));
    }
    let ck = Checkpoint { config: json!({"text": h}), tensors: t };
    let hidden = ChannelDim { name: "text".into(), width: h, keys: vec![ConfigWidth::new("text", 1)], lines: vec![Lines::new("w1", Axis::Rows, 1), Lines::new("b1", Axis::Rows, 1), Lines::new("w2", Axis::Cols, 1)], masks: vec!["w2".into()] };
    let fold = EmbeddingFold { name: "text".into(), embedding: "e".into(), linear: "w1".into(), bias: Some("b1".into()), hidden: hidden.clone() };
    let keep = vec![1, 2, 4, 6];
    let mut masked = ck.clone();
    mask_channels(&mut masked, &[&hidden], std::slice::from_ref(&keep)).unwrap();
    let mut folded = ck.clone();
    let change = fold_embedding(&mut folded, &fold, &keep).unwrap();
    assert_eq!(change.config["text"], [json!(h), json!(4)]);
    assert_eq!(folded.get("e").unwrap().shape, vec![V, 4]);
    assert_eq!(folded.get("w1").unwrap().shape, vec![4, 4]);
    for tok in 0..V {
        assert!(close(&projected(&masked, tok), &projected(&folded, tok), 1e-5));
    }
}

#[test]
fn a_trimmed_vocabulary_keeps_its_rows_and_renumbers_its_ids() {
    let ck = toy(0xe0); // seed: fixed fixture
    let vocab = Vocabulary { name: "text".into(), size: V, key: "vocab".into(), tables: vec!["emb".into(), "head".into()], ids: vec!["bos".into()] };
    let mut cut = ck.clone();
    let change = trim_vocabulary(&mut cut, &vocab, &[0, 3, 4]).unwrap();
    assert_eq!(change.config["bos"], [json!(3), json!(1)]);
    assert_eq!(cut.get("emb").unwrap().data[D..2 * D], ck.get("emb").unwrap().data[3 * D..4 * D]);
    let err = trim_vocabulary(&mut ck.clone(), &vocab, &[0, 1]).unwrap_err();
    assert!(err.contains("bos is token 3"), "{err}");
}

#[test]
fn top_k_keeps_the_highest_scores_in_index_order() {
    assert_eq!(top_k(&[0.1, 5.0, 0.3, 5.0, 2.0], 3), vec![1, 3, 4]);
}

/// A byte-level vocabulary: the 256 byte tokens, five merges and two added tokens.
fn bpe_files() -> bpe::TokenizerFiles {
    let mut vocab: BTreeMap<String, usize> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), i)).collect();
    let merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"), ("w", "o"), ("Ġ", "w")];
    for (a, b) in merges {
        let n = vocab.len();
        vocab.insert(format!("{a}{b}"), n);
    }
    let merges_txt: String = std::iter::once("#version: 0.2".to_string()).chain(merges.iter().map(|(a, b)| format!("{a} {b}"))).map(|l| l + "\n").collect();
    let config = json!({"added_tokens_decoder": {"262": {"content": "<|im_start|>", "special": true}, "263": {"content": "<|im_end|>", "special": true}}, "model_max_length": 10});
    bpe::TokenizerFiles { vocab_json: serde_json::to_string(&vocab).unwrap(), merges_txt, tokenizer_config: config }
}

#[test]
fn a_trimmed_bpe_keeps_bytes_specials_the_corpus_and_its_merge_closure() {
    let trim = bpe::trim(&bpe_files(), &["<|im_start|>hello<|im_end|>", "hello hello"], &["wow"]).unwrap();
    let r = &trim.report;
    assert_eq!((r.kept, r.of, r.bytes, r.added), (262, 264, 256, 2));
    assert_eq!((r.corpus, r.closure, r.merges_kept, r.merges_of), (1, 3, 4, 6), "{r:?}");
    assert_eq!((r.corpus_changed, r.probe_changed), (0, 1), "the corpus encodes as before; the probe loses its merge");
    assert_eq!(&trim.keep[256..], &[256, 257, 258, 259, 262, 263]);
    let vocab: BTreeMap<String, usize> = serde_json::from_str(&trim.files.vocab_json).unwrap();
    assert_eq!((vocab["hello"], vocab.len()), (259, 260));
    assert!(!trim.files.merges_txt.contains("w o") && trim.files.merges_txt.starts_with("#version: 0.2\nh e\n"));
    assert_eq!(trim.files.tokenizer_config["added_tokens_decoder"]["261"]["content"], "<|im_end|>");
    assert_eq!(trim.files.tokenizer_config["model_max_length"], 10);
}
