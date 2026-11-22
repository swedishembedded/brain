// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of looped mixture-of-experts models for its clients. If your team needs
// expertise in fine-tuning such a model then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the training build computes the inference forward's function and learns.

use inkling2::train::{init_weights, tiny_config, Inkling2Train};
use model::Model;

fn batch(b: usize, t: usize, vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let x: Vec<u32> = (0..b * t).map(|i| (i as u32 * 7 + 3) % vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| if i % t == t - 1 { model::IGNORE } else { (i as u32 * 7 + 10) % vocab }).collect();
    (x, y)
}

/// A full fine-tune of the tiny looped config overfits a fixed batch.
#[test]
fn a_full_fine_tune_overfits_a_fixed_batch() {
    let tc = tiny_config();
    let seed = data::rng::Lcg::random().next_u32() as u64;
    let init = init_weights(&tc.cfg, seed);
    let m = Inkling2Train::new(tc.clone(), 2, 8, &init);
    let (x, y) = batch(2, 8, tc.cfg.valid_vocab);
    m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    let first = m.forward();
    assert!(first.is_finite(), "loss {first}");
    let mut last = first;
    for step in 1..=30 {
        m.zero_grads();
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        last = m.forward();
        m.backward();
        m.adamw_step(step, 1e-2, 0.0, model::Adam::default(), Some(1.0), 1.0);
    }
    assert!(last < 0.5 * first, "loss {first} -> {last}");
}

/// The training forward reproduces the reference's logits on the tiny golden (its own recipe, compacted experts and
/// chunked head), and its loss is the cross-entropy of those logits.
#[test]
fn the_training_forward_reproduces_the_reference_logits_and_its_loss() {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if !dir.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let gold: std::collections::HashMap<String, Vec<f32>> = checkpoint::safetensors::read(&dir.join("golden.safetensors").to_string_lossy()).unwrap().into_iter().map(|t| (t.name, t.data)).collect();
    let tokens: Vec<u32> = gold["tokens"].iter().map(|&v| v as u32).collect();
    let t = tokens.len();
    let m = Inkling2Train::from_dir(&dir.join("checkpoint"), 1, t as u32).unwrap();
    let cfg = m.config().clone();
    let logits = m.logits(&tokens).unwrap();
    let valid = cfg.valid_vocab as usize;
    let trimmed = logits;
    let want = &gold["logits"];
    let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    let err = trimmed.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
    assert!(err <= 5e-5, "logits relative error {err:.3e}");
    // The loss of a batch is the mean cross-entropy of those logits over its supervised positions.
    let targets: Vec<u32> = (0..t).map(|i| if i + 1 < t { tokens[i + 1] } else { model::IGNORE }).collect();
    m.set_batch(model::Batch::Lm { tokens: &tokens, targets: &targets });
    let loss = m.forward();
    let mut want_loss = 0.0f64;
    for i in 0..t - 1 {
        let row = &trimmed[i * valid..(i + 1) * valid];
        let mx = row.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
        let lse = mx + row.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
        want_loss += lse - row[targets[i] as usize] as f64;
    }
    want_loss /= (t - 1) as f64;
    assert!((f64::from(loss) - want_loss).abs() <= 1e-4 * want_loss.abs().max(1.0), "loss {loss}, cross-entropy of the logits {want_loss}");
}


fn lora_config(micro_targets: bool) -> inkling2::TrainConfig {
    tiny_config().with_lora(model::adapter::device::LoraCfg::new(2, 4.0, inkling2::train::targets(micro_targets))).unwrap()
}

fn random_adapters(m: &Inkling2Train, seed: u64) {
    let mut rng = data::rng::Lcg::new(seed);
    for name in m.param_names() {
        let n = m.read_weight(&name).len();
        m.write_weight(&name, &rng.vec_scaled(n, 0.2));
    }
}

/// A bf16 base trains the adapters an f32 build over its rounded values does: the frozen linears held as bf16 are read
/// exactly as the bf16 values they hold (the routing pinned to the f32 build's, though with identical values it does
/// not move), so the loss and every adapter gradient agree to f32 rounding.
#[test]
fn a_bf16_base_trains_the_adapters_the_f32_base_does() {
    let tc = lora_config(false);
    let seed = data::rng::Lcg::random().next_u32() as u64;
    let init = <Inkling2Train as Model>::init_weights(&tc, seed);
    let rounded: std::collections::HashMap<String, Vec<f32>> = init
        .iter()
        .map(|(n, v)| {
            // Every linear but the router, which stays binary32 at every tier.
            let rounded = inkling2::train::is_linear(&tc.cfg, n) && !n.ends_with("mlp.gate.weight");
            let v = if rounded { model::ops::Weight::round_trip(v, 1, v.len(), gpu_core::select::Dtype::BF16) } else { v.clone() };
            (n.clone(), v)
        })
        .collect();
    let build = |tier, init: &std::collections::HashMap<String, Vec<f32>>| Inkling2Train::from_tensors(tc.clone(), init, 2, 6, &model::ops::TierPolicy::uniform(tier), seed).unwrap();
    let (f32m, bf16m) = (build(gpu_core::select::Dtype::F32, &rounded), build(gpu_core::select::Dtype::BF16, &init));
    random_adapters(&f32m, seed);
    random_adapters(&bf16m, seed);
    let (x, y) = batch(2, 6, tc.cfg.valid_vocab);
    f32m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    f32m.forward();
    let pinned = std::sync::Arc::new(f32m.taken_routing().unwrap());
    let grads = |m: &Inkling2Train| {
        model::training::TokenObjectives::set_routing_replay(m, Some(pinned.clone())).unwrap();
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        m.zero_grads();
        let loss = m.forward();
        m.backward();
        (loss, m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect::<Vec<_>>())
    };
    let ((lw, want), (lg, got)) = (grads(&f32m), grads(&bf16m));
    assert!((lw - lg).abs() <= 1e-5 * lw.abs().max(1.0), "loss {lg} over the bf16 base, {lw} over its rounded values");
    let mut compared = 0;
    for ((name, w), (_, g)) in want.iter().zip(&got) {
        let norm = w.iter().map(|v| v * v).sum::<f32>().sqrt();
        let diff = w.iter().zip(g).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        // The two builds sum their products in different orders (the bf16 tier decodes in tiles): f32 rounding,
        // measured up to 2.7e-4 of a tensor's norm; a bf16 rounding of the weights moves them by percents.
        assert!(diff <= 1e-3 * norm + 1e-7, "{name}: the bf16 base's gradient differs by {diff} at a norm of {norm}");
        compared += 1;
    }
    assert!(compared > 10, "only {compared} adapter tensors");
}

/// A LoRA run over the frozen base trains only the adapter, lowers the loss on a fixed batch, and its saved adapter,
/// read back into a fresh build of the same base, computes the trained model's loss.
#[test]
fn a_lora_run_trains_only_the_adapter_and_its_saved_adapter_reloads() {
    let tc = lora_config(false);
    let seed = data::rng::Lcg::random().next_u32() as u64;
    let init = <Inkling2Train as Model>::init_weights(&tc, seed);
    let base_only: std::collections::HashMap<String, Vec<f32>> = init.iter().filter(|(n, _)| !model::adapter::device::is_adapter_param(n)).map(|(n, v)| (n.clone(), v.clone())).collect();
    let m = Inkling2Train::new(tc.clone(), 2, 8, &init);
    assert!(m.param_names().iter().all(|n| model::adapter::device::is_adapter_param(n)), "only adapter tensors train");
    let (x, y) = batch(2, 8, tc.cfg.valid_vocab);
    m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    let first = m.forward();
    let mut last = first;
    for step in 1..=40 {
        m.zero_grads();
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        last = m.forward();
        m.backward();
        m.adamw_step(step, 1e-2, 0.0, model::Adam::default(), Some(1.0), 1.0);
    }
    m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    last = last.min(m.forward());
    assert!(last < 0.85 * first, "loss {first} -> {last}");
    let trained = m.forward();
    let dir = brain_testutil::Scratch::dir("inkling2-adapter");
    let path = dir.path().join("adapter.safetensors");
    inkling2::finetune::save_adapter_file(&m, &path).unwrap();
    // A fresh build of the same base with the saved adapter's tensors.
    let mut reloaded_init = base_only;
    for t in checkpoint::safetensors::read(&path.to_string_lossy()).unwrap() {
        reloaded_init.insert(t.name, t.data);
    }
    let r = Inkling2Train::new(tc, 2, 8, &reloaded_init);
    r.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
    let again = r.forward();
    assert!((again - trained).abs() <= 1e-5 * trained.abs(), "reloaded adapter loss {again} vs trained {trained}");
}

/// Packed rows: a row holding two documents trains exactly as the documents do alone - the loss and every gradient of
/// a full fine-tune equal the sum over the documents' own batches (attention, the convolution states and the
/// relative positions restart at the document boundary).
#[test]
fn a_packed_row_trains_as_its_documents_alone() {
    let tc = tiny_config();
    let seed = data::rng::Lcg::random().next_u32() as u64;
    let init = init_weights(&tc.cfg, seed);
    let v = tc.cfg.valid_vocab;
    let (a, b) = (5usize, 7usize);
    let tokens: Vec<u32> = (0..a + b).map(|i| (i as u32 * 11 + 2) % v).collect();
    // Each document's own next-token targets; the last token of each document is unsupervised.
    let targets: Vec<u32> = (0..a + b).map(|i| if i + 1 == a || i + 1 == a + b { model::IGNORE } else { tokens[i + 1] }).collect();
    let segments: Vec<u32> = (0..a + b).map(|i| u32::from(i >= a)).collect();
    let packed = Inkling2Train::new(tc.clone(), 1, (a + b) as u32, &init);
    let replay = |m: &Inkling2Train| {
        // Pin each build's own routing so the comparison is of one function.
        let r = std::sync::Arc::new(m.taken_routing().unwrap());
        model::training::TokenObjectives::set_routing_replay(m, Some(r)).unwrap();
    };
    packed.set_batch(model::Batch::LmPacked { tokens: &tokens, targets: &targets, segments: &segments });
    packed.forward();
    replay(&packed);
    packed.set_batch(model::Batch::LmPacked { tokens: &tokens, targets: &targets, segments: &segments });
    packed.zero_grads();
    let loss = packed.forward() * (a + b - 2) as f32;
    packed.backward();
    let mut sum = 0.0;
    let mut grads: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for (lo, hi) in [(0, a), (a, a + b)] {
        let alone = Inkling2Train::new(tc.clone(), 1, (hi - lo) as u32, &init);
        alone.set_batch(model::Batch::Lm { tokens: &tokens[lo..hi], targets: &targets[lo..hi] });
        alone.forward();
        replay(&alone);
        alone.set_batch(model::Batch::Lm { tokens: &tokens[lo..hi], targets: &targets[lo..hi] });
        alone.zero_grads();
        let n = (hi - lo - 1) as f32;
        sum += alone.forward() * n;
        alone.backward();
        for name in alone.param_names() {
            let g: Vec<f32> = alone.read_grad(&name).iter().map(|x| x * n).collect();
            let e = grads.entry(name).or_insert_with(|| vec![0.0; g.len()]);
            e.iter_mut().zip(&g).for_each(|(a, b)| *a += b);
        }
    }
    assert!((loss - sum).abs() <= 1e-4 * sum.abs(), "packed loss {loss} vs documents alone {sum}");
    for (name, want) in grads {
        let got: Vec<f32> = packed.read_grad(&name).iter().map(|x| x * (a + b - 2) as f32).collect();
        let norm = want.iter().map(|v| v * v).sum::<f32>().sqrt();
        let diff = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        assert!(diff <= 1e-3 * norm + 1e-6, "{name}: packed gradient differs by {diff} at a norm of {norm}");
    }
}

/// The family's serving engine (what `served_loss` and a precision search score through) serves the function the
/// trainer trains: its logits at every position of a row equal the training build's over the same checkpoint, with no
/// adapter and with an adapter the trainer saved (applied beside the base, not folded).
#[test]
fn the_serving_engine_serves_what_the_trainer_trains() {
    use model::finetune::ChatTrainer;
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if !dir.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let ckpt = dir.join("checkpoint");
    let tokens: Vec<u32> = vec![3, 17, 5, 40, 9, 22, 61, 8, 12];
    let trainer = Inkling2Train::from_dir(&ckpt, 1, tokens.len() as u32).unwrap();
    let want = trainer.logits(&tokens).unwrap();
    let valid = trainer.config().valid_vocab as usize;
    let tiers = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32);
    let mut engine = inkling2::finetune::Inkling2Trainer::serving_engine(&ckpt, &tiers, None, tokens.len() as u32).expect("the family serves").unwrap();
    let positions: Vec<usize> = (0..tokens.len()).collect();
    for (p, hidden) in engine.hidden_at(&tokens, &positions).into_iter().enumerate() {
        let got = engine.logits(&hidden);
        assert_eq!(got.len(), valid, "a served row is the valid vocabulary");
        let row = &want[p * valid..(p + 1) * valid];
        let scale = row.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let err = got.iter().zip(row).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        assert!(err <= 5e-5, "position {p}: served logits differ by {err:.3e} of their scale");
    }
    // With an adapter the trainer saved (every site of a checkpoint with dense experts), the engine serves the trainer's
    // adapted logits: the adapter rides beside the base, every sequence under it.
    let lora = model::adapter::device::LoraCfg::new(2, 4.0, inkling2::train::targets(false));
    let base_tier = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32);
    // seed: any factors; the engine reads the file the trainer saves.
    let adapted = Inkling2Train::from_dir_lora(&ckpt, lora, &inkling2::train::BuildOptions { batch: 1, seq: tokens.len() as u32, base: &base_tier, seed: 3, continue_from: None }).unwrap();
    random_adapters(&adapted, 3);
    let scratch = brain_testutil::Scratch::dir("inkling2-scoring-adapter");
    let path = scratch.path().join("adapter.safetensors");
    inkling2::finetune::save_adapter_file(&adapted, &path).unwrap();
    let want_adapted = adapted.logits(&tokens).unwrap();
    let mut engine = inkling2::finetune::Inkling2Trainer::serving_engine(&ckpt, &tiers, Some(&path), tokens.len() as u32).expect("the family serves").expect("an adapter is served");
    for (p, hidden) in engine.hidden_at(&tokens, &positions).into_iter().enumerate() {
        let got = engine.logits(&hidden);
        let row = &want_adapted[p * valid..(p + 1) * valid];
        let scale = row.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let err = got.iter().zip(row).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        assert!(err <= 1e-4, "position {p}: the adapter's served logits differ by {err:.3e} of their scale");
        let moved = got.iter().zip(&want[p * valid..(p + 1) * valid]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        assert!(p == 0 || moved > 1e-3, "position {p}: the adapter moves the logits ({moved:.3e})");
    }
}

/// The backward recomputes every visit from its saved input: it accepts recompute and refuses keeping every visit by
/// name.
#[test]
fn the_backward_always_recomputes_each_visit() {
    let tc = tiny_config();
    let init = init_weights(&tc.cfg, data::rng::random_seed());
    let mut m = Inkling2Train::new(tc, 1, 4, &init);
    model::saved::check_always_recomputes("inkling2", &mut m).unwrap();
}

/// The router's gradient is a declared choice: detached, the routers take no gradient (and the loss is unchanged), while
/// the experts and everything below them still train.
#[test]
fn a_detached_router_takes_no_gradient_and_leaves_the_loss() {
    let tc = tiny_config();
    let init = init_weights(&tc.cfg, data::rng::random_seed());
    let m = Inkling2Train::new(tc.clone(), 2, 6, &init);
    let (x, y) = batch(2, 6, tc.cfg.valid_vocab);
    // The last layer: nothing after it routes, so its experts' gradient is the same with every router detached.
    let router = "model.llm.layers.7.mlp.gate.weight";
    let expert = "model.llm.layers.7.mlp.experts.w2_weight";
    let run = |detached: bool| {
        m.set_detached_router(detached);
        m.zero_grads();
        m.set_batch(model::Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        (loss, m.read_grad(router), m.read_grad(expert))
    };
    let (full_loss, full_router, full_expert) = run(false);
    let (loss, router_grad, expert_grad) = run(true);
    assert_eq!(loss, full_loss, "detaching the router changes no forward");
    assert!(full_router.iter().any(|g| *g != 0.0), "the differentiated router has a gradient");
    assert!(router_grad.iter().all(|g| *g == 0.0), "a detached router takes none");
    assert_eq!(expert_grad, full_expert, "the last layer's experts' gradient does not depend on the routers'");
}
