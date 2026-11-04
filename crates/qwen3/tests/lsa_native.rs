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

fn small(vocab: u32, t: u32) -> QwenConfig {
    QwenConfig { vocab, block_size: t, n_layers: 2, d_model: 64, n_heads: 4, n_kv_heads: 2, head_dim: 16, d_ff: 128, max_position_embeddings: t, rope_theta: 10000.0, ..QwenConfig::tiny() }
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
    let t = 64u32;
    let cfg = small(97, t);
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

/// One associative-recall document: eight key/value pairs, filler far past
/// the window, then the eight keys again, each to be answered with its value.
fn recall(rng: &mut Lcg, t: usize) -> (Vec<u32>, Vec<u32>, Vec<usize>) {
    let (keys, values, filler) = (0..64u32, 64..128u32, 128..192u32);
    let mut chosen: Vec<u32> = Vec::new();
    while chosen.len() < 8 {
        let k = keys.start + rng.next_u32() % (keys.end - keys.start);
        if !chosen.contains(&k) {
            chosen.push(k);
        }
    }
    let vals: Vec<u32> = chosen.iter().map(|_| values.start + rng.next_u32() % (values.end - values.start)).collect();
    let mut x: Vec<u32> = chosen.iter().zip(&vals).flat_map(|(&k, &v)| [k, v]).collect();
    let query_at = t - 24;
    while x.len() < query_at {
        x.push(filler.start + rng.next_u32() % (filler.end - filler.start));
    }
    let mut answers = Vec::new();
    let mut order: Vec<usize> = (0..8).collect();
    for i in (1..8).rev() {
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

/// Trained from scratch on recall whose pairs sit 200 tokens before their
/// queries, past a 16-key window: dense attention and native sparse attention
/// learn it, a window-only model cannot; the sparse model then answers through
/// the serving engine reading sparsely, as it was trained.
#[test]
#[ignore = "experiment: trains three small models (minutes on a GPU)"]
fn native_sparse_attention_learns_long_range_recall_and_serves_sparsely() {
    let (t, b, steps) = (256u32, 16u32, 3000u32);
    let cfg = small(200, t);
    let mut rng = Lcg::random();
    let init = qwen3::init_weights(&cfg, rng.next_u64());
    let cfg_lsa = lsa(16, 8);
    let train = |mode: Option<bool>, rng: &mut Lcg| -> (Qwen, f32) {
        let mut m = Qwen::new(cfg.clone(), b, t, &init);
        if let Some(long_range) = mode {
            m.enable_native_lsa(cfg_lsa, long_range).unwrap();
        }
        let mut last = 0.0;
        for s in 1..=steps {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for _ in 0..b {
                let (xs, ys, _) = recall(rng, t as usize);
                x.extend(xs);
                y.extend(ys);
            }
            m.zero_grads();
            m.set_batch(&x, &y);
            let loss = m.forward();
            m.backward();
            m.adamw_step(s, 3e-3, 0.0, Default::default(), Some(1.0), 1.0);
            last = 0.95 * last + 0.05 * loss;
            if s % 500 == 0 {
                println!("  {:?} step {s}: loss {last:.3}", mode);
            }
        }
        (m, last)
    };
    let accuracy = |m: &Qwen, rng: &mut Lcg| -> f64 {
        let (mut hit, mut total) = (0usize, 0usize);
        for _ in 0..4 {
            let (mut x, mut y, mut ans) = (Vec::new(), Vec::new(), Vec::new());
            for r in 0..b as usize {
                let (xs, ys, a) = recall(rng, t as usize);
                ans.extend(a.into_iter().map(|p| (r, p, xs[p])));
                x.extend(xs);
                y.extend(ys);
            }
            m.set_batch(&x, &y);
            m.forward();
            let lp = m.batch_token_logprobs();
            for (r, p, _) in ans {
                hit += usize::from(lp[r * t as usize + p - 1] > (0.5f32).ln());
                total += 1;
            }
        }
        hit as f64 / total as f64
    };
    let mut results = Vec::new();
    for (name, mode) in [("dense", None), ("native sparse", Some(true)), ("window only", Some(false))] {
        let (m, loss) = train(mode, &mut rng);
        let acc = accuracy(&m, &mut rng);
        println!("{name:<14} final loss {loss:.3}, answers held with p > 0.5: {:.1}%", 100.0 * acc);
        results.push((name, loss, acc, m));
    }
    let (dense_acc, sparse_acc, window_acc) = (results[0].2, results[1].2, results[2].2);
    assert!(sparse_acc > 0.9 && dense_acc > 0.9, "dense {dense_acc} / native sparse {sparse_acc} did not learn the task");
    assert!(window_acc < 0.3, "a window-only model answered {window_acc}: the task does not need long range");

    // Serve the sparse model through the engine reading sparsely: the index
    // space it was trained in is the identity map over its un-rotated heads.
    let dir = std::env::temp_dir().join(format!("brain-nlsa-{}", rng.next_u64()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("nlsa.safetensors");
    results[1].3.save(path.to_str().unwrap());
    let dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
    let identity: Vec<f32> = (0..dim * dim).map(|i| f32::from(u8::from(i / dim == i % dim))).collect();
    let geometry = IndexGeometry { n_layers: cfg.n_layers as usize, n_heads: cfg.n_heads as usize, n_kv_heads: cfg.n_kv_heads as usize, head_dim: cfg.head_dim as usize, rope_base: cfg.rope_theta };
    let indexer = Indexer { cfg: LsaConfig { index_dim: dim, ..cfg_lsa }, geometry, layers: vec![Some(LayerProjection { wq: identity.clone(), wk: identity }); cfg.n_layers as usize], origin: "native".into() };
    let mut engine = qwen3::serve::Engine::load(path.to_str().unwrap(), 16, 64, 1, t / 16, 64, false, false);
    engine.set_lsa(&indexer).unwrap();
    let (mut hit, mut total) = (0, 0);
    for _ in 0..16 {
        let (x, _, answers) = recall(&mut rng, t as usize);
        for &a in answers.iter().take(4) {
            let out = engine.generate_greedy(&[x[..a].to_vec()], 1, None);
            hit += usize::from(out[0][0] == x[a]);
            total += 1;
        }
    }
    let served = hit as f64 / total as f64;
    println!("native sparse model served sparsely: {:.1}% of answers", 100.0 * served);
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(served > 0.9, "served sparsely the trained model answered {served}");
}
