// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native latent sparse attention (N-LSA) on the Qwen3 training model: the
//! training path's exactness anchor, and (run by hand) a model trained from
//! scratch with sparse attention on long-range recall, then served sparsely.
//!
//! Swedish Embedded AB implements models that are trained to read a bounded
//! set of keys per token for its clients. If your team needs expertise in
//! sparse-attention training and long-context evaluation, you can procure
//! our services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use model::lsa::indexer::{IndexGeometry, Indexer, LayerProjection};
use model::lsa::LsaConfig;
use qwen3::{Qwen, QwenConfig, IGNORE};

fn small(vocab: u32, t: u32, d_model: u32, head_dim: u32) -> QwenConfig {
    QwenConfig { vocab, block_size: t, n_layers: 2, d_model, n_heads: 4, n_kv_heads: 2, head_dim, d_ff: 2 * d_model, max_position_embeddings: t, rope_theta: 10000.0, ..QwenConfig::tiny() }
}

fn lsa(top_k: usize, top_p: usize) -> LsaConfig {
    LsaConfig { index_dim: 0, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 16, top_p, top_k, sink: 2, window: 16, early: None }
}

fn grads(m: &Qwen, cfg: &QwenConfig) -> Vec<f32> {
    let mut all = Vec::new();
    for l in 0..cfg.n_layers {
        for leaf in ["attn.wq.weight", "attn.wk.weight", "attn.wv.weight", "attn.wo.weight"] {
            all.extend(m.read_grad(&format!("blocks.{l}.{leaf}")));
        }
    }
    all
}

/// With a budget past the context every query selects every causal key, so
/// the sparse training step - masks filled mid-forward, reused by the
/// backward's recompute - must give dense attention's loss and gradients; a
/// window-only selection must not.
#[test]
fn a_budget_past_the_context_trains_like_dense_attention() {
    if !crate::lsa_engine::lsa_runs_here() {
        return;
    }
    let t = 64u32;
    let cfg = small(97, t, 64, 16);
    let mut rng = Lcg::random();
    let init = qwen3::init_weights(&cfg, rng.next_u64());
    let x: Vec<u32> = (0..2 * t).map(|_| rng.next_u32() % cfg.vocab).collect();
    let y: Vec<u32> = (0..2 * t).map(|_| rng.next_u32() % cfg.vocab).collect();
    let step = |m: &Qwen| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        (loss, grads(m, &cfg))
    };
    let (dense_loss, dense_grads) = step(&Qwen::new(cfg.clone(), 2, t, &init));
    let mut full = Qwen::new(cfg.clone(), 2, t, &init);
    full.enable_native_lsa(lsa(1024, 1024), true).unwrap();
    let (full_loss, full_grads) = step(&full);
    let mut window = Qwen::new(cfg.clone(), 2, t, &init);
    window.enable_native_lsa(lsa(1024, 1024), false).unwrap();
    let (window_loss, _) = step(&window);
    let worst = dense_grads.iter().zip(&full_grads).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let scale = dense_grads.iter().fold(0f32, |m, a| m.max(a.abs()));
    println!("loss dense {dense_loss:.6} full-budget sparse {full_loss:.6} window-only {window_loss:.6}; gradient maxabs {worst:.2e} of {scale:.2e}");
    assert!((dense_loss - full_loss).abs() < 1e-4, "full-budget sparse loss {full_loss} vs dense {dense_loss}");
    assert!(worst < 1e-4 * scale.max(1e-6) + 1e-7, "gradients differ by {worst} (largest {scale})");
    assert!((dense_loss - window_loss).abs() > 1e-4, "a window-only selection changed nothing: the masks are not applied");
}

/// Key/value pairs each recall document holds.
const PAIRS: usize = 4;

/// One associative-recall document: `PAIRS` key/value pairs, filler far past
/// the window, then the keys again, each to be answered with its value.
fn recall(rng: &mut Lcg, t: usize) -> (Vec<u32>, Vec<u32>, Vec<usize>) {
    let (keys, values, filler) = (0..32u32, 32..64u32, 64..96u32);
    let mut chosen: Vec<u32> = Vec::new();
    while chosen.len() < PAIRS {
        let k = keys.start + rng.next_u32() % (keys.end - keys.start);
        if !chosen.contains(&k) {
            chosen.push(k);
        }
    }
    let vals: Vec<u32> = chosen.iter().map(|_| values.start + rng.next_u32() % (values.end - values.start)).collect();
    let mut x: Vec<u32> = chosen.iter().zip(&vals).flat_map(|(&k, &v)| [k, v]).collect();
    let query_at = t - 3 * PAIRS;
    while x.len() < query_at {
        x.push(filler.start + rng.next_u32() % (filler.end - filler.start));
    }
    let mut answers = Vec::new();
    let mut order: Vec<usize> = (0..PAIRS).collect();
    for i in (1..PAIRS).rev() {
        order.swap(i, rng.next_u32() as usize % (i + 1));
    }
    for &i in &order {
        answers.push(x.len() + 1);
        x.push(chosen[i]);
        x.push(vals[i]);
    }
    while x.len() < t {
        x.push(filler.start + rng.next_u32() % (filler.end - filler.start));
    }
    let mut y = vec![IGNORE; t];
    for &a in &answers {
        y[a - 1] = x[a];
    }
    (x, y, answers)
}

/// Trained from scratch on recall whose pairs sit over 200 tokens before
/// their queries, past a 16-key window: native sparse attention learns to
/// retrieve them as dense attention does, far below the loss a window-only
/// model is stuck at; then the sparse model, served by the engine reading
/// sparsely through the identity indexer (the index space it trained in),
/// gives every answer the log-probability the training model gives it -
/// the training masks and the serving kernels select and attend alike.
#[test]
#[ignore = "experiment: trains three small models (tens of minutes on a GPU)"]
fn native_sparse_attention_learns_long_range_recall_and_serves_sparsely() {
    let (t, b, steps) = (256u32, 16u32, 3000u32);
    let cfg = small(100, t, 128, 32);
    let mut rng = Lcg::random();
    let init = qwen3::init_weights(&cfg, rng.next_u64());
    let cfg_lsa = lsa(16, 8);
    let batch = |rng: &mut Lcg| {
        let (mut x, mut y, mut answers) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..b as usize {
            let (xs, ys, a) = recall(rng, t as usize);
            answers.extend(a.into_iter().map(|p| (r, p)));
            x.extend(xs);
            y.extend(ys);
        }
        (x, y, answers)
    };
    let train = |mode: Option<bool>, rng: &mut Lcg| -> Qwen {
        let mut m = Qwen::new(cfg.clone(), b, t, &init);
        if let Some(long_range) = mode {
            m.enable_native_lsa(cfg_lsa, long_range).unwrap();
        }
        for s in 1..=steps {
            let (x, y, _) = batch(rng);
            m.zero_grads();
            m.set_batch(&x, &y);
            m.forward();
            m.backward();
            m.adamw_step(s, 2e-3, 0.0, Default::default(), Some(1.0), 1.0);
        }
        m
    };
    // Mean answer loss over held-out batches, from the training model's own forward.
    let held_out: Vec<_> = (0..4).map(|_| batch(&mut rng)).collect();
    let loss = |m: &Qwen| -> f64 {
        let mut total = 0.0;
        for (x, y, _) in &held_out {
            m.set_batch(x, y);
            total += m.forward() as f64;
        }
        total / held_out.len() as f64
    };
    let mut models = Vec::new();
    for (name, mode) in [("dense", None), ("native sparse", Some(true)), ("window only", Some(false))] {
        let m = train(mode, &mut rng);
        let l = loss(&m);
        println!("{name:<14} held-out answer loss {l:.3} (chance {:.3})", (32f64).ln());
        models.push((l, m));
    }
    let (dense, sparse, window) = (models[0].0, models[1].0, models[2].0);
    assert!(window > 2.5, "a window-only model reached {window}: the task does not need long range");
    assert!(dense < window - 1.0 && sparse < window - 1.0, "dense {dense} / native sparse {sparse} did not learn what a window cannot ({window})");

    // Serve the sparse model through the engine reading sparsely.
    let dir = std::env::temp_dir().join(format!("brain-nlsa-{}", rng.next_u64()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("nlsa.safetensors");
    let sparse_model = &models[1].1;
    sparse_model.save(path.to_str().unwrap());
    let dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
    let identity: Vec<f32> = (0..dim * dim).map(|i| f32::from(u8::from(i / dim == i % dim))).collect();
    let geometry = IndexGeometry { n_layers: cfg.n_layers as usize, n_heads: cfg.n_heads as usize, n_kv_heads: cfg.n_kv_heads as usize, head_dim: cfg.head_dim as usize, rope_base: cfg.rope_theta };
    let indexer = Indexer { cfg: LsaConfig { index_dim: dim, ..cfg_lsa }, geometry, layers: vec![Some(LayerProjection { wq: identity.clone(), wk: identity }); cfg.n_layers as usize], origin: "native".into(), tree: None };
    let mut engine = qwen3::serve::Engine::load(path.to_str().unwrap(), 16, 64, 1, t / 16, 64, false, false);
    engine.set_lsa(&indexer).unwrap();
    let (x, y, answers) = &held_out[0];
    sparse_model.set_batch(x, y);
    sparse_model.forward();
    let trained = sparse_model.batch_token_logprobs();
    let (mut served_nll, mut trained_nll, mut worst, mut n) = (0f64, 0f64, 0f32, 0usize);
    for r in 0..4usize {
        let doc = &x[r * t as usize..][..t as usize];
        let served = engine.position_logprobs(doc, 0);
        for &(row, p) in answers.iter().filter(|(row, _)| *row == r) {
            let (s_lp, t_lp) = (served[p - 1], trained[row * t as usize + p - 1]);
            served_nll -= s_lp as f64;
            trained_nll -= t_lp as f64;
            worst = worst.max((s_lp - t_lp).abs());
            n += 1;
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    let (served_nll, trained_nll) = (served_nll / n as f64, trained_nll / n as f64);
    println!("served sparsely: answer loss {served_nll:.4} against the training model's {trained_nll:.4} over {n} answers (largest single difference {worst:.2e})");
    // The selection is discrete and the two paths compute it in different
    // arithmetic (device f32 index vectors and scores against host f64), so a
    // near-tie bucket can flip and move one answer slightly; the answers'
    // loss must agree.
    assert!((served_nll - trained_nll).abs() < 0.02, "served sparse answer loss {served_nll} vs training {trained_nll}");
}

/// One training step (forward and backward) over `t` tokens timed dense
/// (the flash kernels) and with native sparse attention at a fixed budget:
/// dense grows quadratically with the context, sparse linearly, so past a
/// few thousand tokens sparse training is faster and the gap widens.
#[test]
#[ignore = "measurement: trains a small model at contexts up to 16k tokens"]
fn native_sparse_training_scales_linearly_with_the_context() {
    let lsa = LsaConfig { index_dim: 0, pool_size: 4, bucket_size: 8, bucket_cap: 16, cluster_block: 512, top_p: 64, top_k: 512, sink: 4, window: 512, early: None };
    let mut rng = Lcg::random();
    let mut ratios = Vec::new();
    for t in [2048u32, 4096, 8192, 16384, 32768] {
        let cfg = QwenConfig { n_layers: 4, n_heads: 8, n_kv_heads: 2, ..small(1000, t, 512, 64) };
        let init = qwen3::init_weights(&cfg, rng.next_u64());
        let x: Vec<u32> = (0..t).map(|_| rng.next_u32() % cfg.vocab).collect();
        let y: Vec<u32> = (0..t).map(|_| rng.next_u32() % cfg.vocab).collect();
        let time = |m: &Qwen| {
            let step = || {
                m.zero_grads();
                m.set_batch(&x, &y);
                m.forward();
                m.backward();
                m.read_grad("blocks.0.attn.wq.weight");
            };
            step();
            let start = std::time::Instant::now();
            for _ in 0..3 {
                step();
            }
            start.elapsed().as_secs_f64() / 3.0
        };
        let dense = time(&Qwen::new(cfg.clone(), 1, t, &init));
        let mut sparse_model = Qwen::new(cfg.clone(), 1, t, &init);
        sparse_model.enable_native_lsa(lsa, true).unwrap();
        let sparse = time(&sparse_model);
        println!("t {t:>6}: dense {:>8.1} ms  native sparse {:>8.1} ms  speedup {:.2}x", 1e3 * dense, 1e3 * sparse, dense / sparse);
        ratios.push(dense / sparse);
    }
    assert!(ratios.windows(2).all(|w| w[1] > w[0]), "sparse training's advantage must grow with the context: {ratios:?}");
    assert!(*ratios.last().unwrap() > 1.0, "at 16k tokens sparse training must be faster than dense: {ratios:?}");
}
