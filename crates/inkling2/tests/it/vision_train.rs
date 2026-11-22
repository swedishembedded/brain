// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image-text fine-tuning for its clients. If your team needs expertise in training a
// multimodal model from its published release then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the vision tower trains. On a small random tower in the published naming, the analytic gradient of every
//! parameter class (patch embedding, class token, position table, `ln_pre`, every block tensor including the layer
//! scales, the merger) equals central finite differences of the serving forward; keeping every layer's activations and
//! recomputing them give the same gradients bit for bit; a tower whose weights are frozen learns through adapters
//! (their gradients are exact too, and a step on a toy loss lowers it); the splice's backward feeds the tower the rows
//! the decoder's gradient holds at the placeholder positions.

use std::collections::HashMap;

use data::rng::{random_seed, Lcg};
use gpu_core::Gpu;
use imaging::RegionImage;
use inkling2::config::VisionConfig;
use inkling2::names::vision_manifest_of;
use inkling2::vision::{splice_embeds, train_pipelines, LoraTraining, VisionTower, VisionWeights};
use model::adapter::device::LoraCfg;
use model::saved::{SavedFormat, SavedPolicy};

/// Tiny but complete: two layers, four heads of width 16, a 4x4 position table, regions of 4x4 pixels. A width that is
/// a multiple of 64 keeps every image's rows on the 256-byte binding boundary the cached attention backward needs.
fn tiny_config() -> VisionConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "inkling2_vision", "text_hidden_size": 12, "temporal_patch_size": 2, "native_patch_size": 2,
        "adapter_merge_size": 2, "projection_hidden_dim": 20, "projector_bias": false, "hidden_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "mlp_ratio": 2.0, "posemb_grid_size": 4, "layer_norm_eps": 1e-5
    }))
    .unwrap()
}

/// Random weights in the published naming: norm gains near one, layer scales of order one (so the blocks matter),
/// linears scaled to keep attention scores of order one.
fn random_weights(cfg: &VisionConfig, rng: &mut Lcg) -> VisionWeights {
    let tensors: HashMap<String, Vec<f32>> = vision_manifest_of(cfg)
        .into_iter()
        .map(|e| {
            let n: u64 = e.shape.iter().product();
            let mut v = rng.vec_scaled(n as usize, 1.0);
            let scale = if e.name.ends_with("ln_1.weight") || e.name.ends_with("ln_2.weight") || e.name.ends_with("ln_pre.weight") || e.name.ends_with("gamma") {
                for x in &mut v {
                    *x = 1.0 + 0.3 * *x;
                }
                1.0
            } else if e.name.ends_with(".weight") && e.shape.len() == 2 && !e.name.contains("embedding") {
                0.35
            } else {
                0.5
            };
            (e.name, v.into_iter().map(|x| x * scale).collect())
        })
        .collect();
    VisionWeights::from_tensors(cfg, tensors).unwrap()
}

/// A 2x1-region image and a 1x1 one, so attention spans differ and the class rows are not adjacent.
fn images(cfg: &VisionConfig, rng: &mut Lcg) -> Vec<RegionImage> {
    let region = (cfg.native_patch_size * cfg.adapter_merge_size) as usize;
    [(2u32, 1u32), (1, 1)]
        .into_iter()
        .map(|(rows, cols)| RegionImage { rows, cols, pixels: rng.vec_scaled((rows * cols) as usize * region * region * 3, 1.0) })
        .collect()
}

fn gpu() -> Gpu {
    gpu_core::testgpu::dev(train_pipelines())
}

/// The toy loss: half the squared distance of the features to a fixed target. Its gradient is a sum of coherent
/// terms, so finite differences in f32 are not drowned by cancellation the way a loss linear in the features is.
fn toy_loss(rows: &[f32], target: &[f32]) -> f64 {
    rows.iter().zip(target).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / 2.0
}

/// The loss's gradient with respect to the features.
fn toy_gradient(rows: &[f32], target: &[f32]) -> Vec<f32> {
    rows.iter().zip(target).map(|(a, b)| a - b).collect()
}

fn all_trainable(_: &str) -> bool {
    true
}

struct Fixture {
    cfg: VisionConfig,
    weights: VisionWeights,
    images: Vec<RegionImage>,
    /// The toy loss pulls the features towards this target.
    dir: Vec<f32>,
}

fn fixture() -> Fixture {
    let cfg = tiny_config();
    let mut rng = Lcg::new(random_seed());
    let weights = random_weights(&cfg, &mut rng);
    let images = images(&cfg, &mut rng);
    let regions: usize = images.iter().map(RegionImage::regions).sum();
    let dir = rng.vec_scaled(regions * cfg.text_hidden_size as usize, 1.0);
    Fixture { cfg, weights, images, dir }
}

fn grads_of(g: &Gpu, tower: &VisionTower) -> Vec<(String, Vec<f32>)> {
    let store = tower.store();
    store.params.iter().filter(|(n, _)| store.grad.contains_key(n)).map(|(n, _)| (n.clone(), store.read_grad(g, n))).collect()
}

/// One training pass on the fixture, from zeroed gradients.
fn train_step(g: &Gpu, tower: &VisionTower, f: &Fixture) -> Vec<f32> {
    tower.zero_grads(g);
    let (features, pass) = tower.encode_train(g, &f.images).unwrap();
    tower.backward(g, pass, &toy_gradient(features.rows(), &f.dir)).unwrap();
    features.rows().to_vec()
}

/// Central differences of `loss` over up to `samples` elements of a tensor (spread over it, plus its largest analytic
/// gradient), each against the analytic gradient. `write` stores a perturbed tensor; the original is restored.
fn check_tensor(label: &str, analytic: &[f32], original: &[f32], write: &dyn Fn(&[f32]), loss: &dyn Fn() -> f64, samples: usize) {
    let eps = 4e-3f32;
    let mut at: Vec<usize> = (0..samples.min(original.len())).map(|i| i * original.len() / samples.min(original.len())).collect();
    let peak = (0..analytic.len()).max_by(|&a, &b| analytic[a].abs().total_cmp(&analytic[b].abs())).expect("a non-empty tensor");
    assert!(analytic[peak].abs() > 1e-7, "{label}: the gradient is zero everywhere, nothing reached this tensor");
    at.push(peak);
    for j in at {
        let mut p = original.to_vec();
        p[j] += eps;
        write(&p);
        let up = loss();
        p[j] = original[j] - eps;
        write(&p);
        let down = loss();
        write(original);
        let numeric = (up - down) / (2.0 * f64::from(eps));
        let err = (f64::from(analytic[j]) - numeric).abs();
        // The f32 forward limits the difference quotient to a few parts in a thousand of the tensor's largest
        // gradient; a missing term or a wrong sign is off by far more than that.
        assert!(err <= 3e-3 * f64::from(analytic[peak].abs()) + 3e-2 * numeric.abs(), "{label}[{j}]: analytic {} numeric {numeric} (tensor peak {})", analytic[j], analytic[peak]);
    }
}

#[test]
fn every_parameter_class_of_the_tower_has_the_gradient_of_its_forward() {
    let g = gpu();
    let f = fixture();
    let tower = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &all_trainable, None).unwrap();
    let features = train_step(&g, &tower, &f);
    let served = tower.encode(&g, &f.images).unwrap();
    let worst = features.iter().zip(served.rows()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let peak = features.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(worst < 1e-5 * (1.0 + peak), "the training forward differs from the serving forward by {worst} (largest feature {peak})");

    let store = tower.store();
    let loss = || toy_loss(tower.encode(&g, &f.images).unwrap().rows(), &f.dir);
    let mut classes = 0;
    for (name, grad) in grads_of(&g, &tower) {
        let original = store.read_weight(&g, &name);
        check_tensor(&name, &grad, &original, &|p| g.write_f32(store.w(&name), p), &loss, 6);
        classes += 1;
    }
    // 6 embedding tensors, 14 per layer, the merger's two.
    assert_eq!(classes, 6 + 14 * 2 + 2, "every tensor of the tower trains");
}

#[test]
fn keeping_every_layer_and_recomputing_give_the_same_gradients_bit_for_bit() {
    let g = gpu();
    let f = fixture();
    let mut tower = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &all_trainable, None).unwrap();
    let bits = |v: Vec<(String, Vec<f32>)>| -> Vec<(String, Vec<u32>)> { v.into_iter().map(|(n, x)| (n, x.into_iter().map(f32::to_bits).collect())).collect() };
    tower.set_saved_policy(SavedPolicy::Keep).unwrap();
    train_step(&g, &tower, &f);
    let kept = bits(grads_of(&g, &tower));
    tower.set_saved_policy(SavedPolicy::Recompute).unwrap();
    train_step(&g, &tower, &f);
    assert_eq!(bits(grads_of(&g, &tower)), kept, "recomputing a layer's cache is the cache the forward made");
    let e = tower.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err();
    assert!(e.contains("inkling2 vision tower"), "{e}");
}

#[test]
fn a_tower_trains_any_subset_and_stops_the_backward_where_nothing_needs_a_gradient() {
    let g = gpu();
    let f = fixture();
    let full = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &all_trainable, None).unwrap();
    train_step(&g, &full, &f);
    let want: HashMap<String, Vec<f32>> = grads_of(&g, &full).into_iter().collect();

    // The merger alone: no tower tensor has a gradient buffer, the merger's equal the full tower's.
    let merger_only = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &|n: &str| n.starts_with("model.visual.proj_"), None).unwrap();
    train_step(&g, &merger_only, &f);
    let got = grads_of(&g, &merger_only);
    assert_eq!(got.len(), 2);
    for (name, grad) in got {
        assert_eq!(grad, want[&name], "{name}");
    }

    // The last layer and the merger: the lower layer and the embeddings are never differentiated, the gradients of
    // what trains are the full tower's.
    let top = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &|n: &str| n.contains("layers.1.") || n.starts_with("model.visual.proj_"), None).unwrap();
    train_step(&g, &top, &f);
    for (name, grad) in grads_of(&g, &top) {
        assert_eq!(grad, want[&name], "{name}");
    }
}

#[test]
fn the_splice_backward_hands_the_tower_the_rows_the_decoder_gradient_holds_at_the_placeholders() {
    let g = gpu();
    let f = fixture();
    let tower = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &all_trainable, None).unwrap();
    let image_token = 7u32;
    let width = f.cfg.text_hidden_size as usize;
    let (features, pass) = tower.encode_train(&g, &f.images).unwrap();
    // A prompt: text, two images' placeholders (2 + 1 regions), text.
    let prompt = [1, image_token, image_token, image_token, 2, 3];
    let embeds = splice_embeds(&prompt, image_token, &features).unwrap();
    assert_eq!(embeds.positions(), [1, 2, 3]);

    // The decoder's gradient with respect to its input embeddings: the loss's gradient at the image rows, something else
    // at the text rows.
    let upstream = toy_gradient(features.rows(), &f.dir);
    let mut d_embeddings = vec![0.25f32; prompt.len() * width];
    d_embeddings[width..4 * width].copy_from_slice(&upstream);
    let d_rows = embeds.take_row_gradients(&mut d_embeddings).unwrap();
    assert_eq!(d_rows, upstream);
    assert!(d_embeddings[width..4 * width].iter().all(|v| *v == 0.0), "the placeholder rows no longer reach the token embedding");
    tower.zero_grads(&g);
    tower.backward(&g, pass, &d_rows).unwrap();
    let spliced: HashMap<String, Vec<f32>> = grads_of(&g, &tower).into_iter().collect();

    train_step(&g, &tower, &f);
    for (name, grad) in grads_of(&g, &tower) {
        assert_eq!(grad, spliced[&name], "{name}");
    }

    let e = tower.encode_train(&g, &f.images).and_then(|(_, pass)| tower.backward(&g, pass, &d_rows[1..])).unwrap_err();
    assert!(e.contains("gradient values"), "{e}");
}

fn all_leaves() -> Vec<String> {
    ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj", "proj_in", "proj_out"].iter().map(|s| s.to_string()).collect()
}

#[test]
fn a_frozen_tower_learns_through_adapters_with_exact_gradients() {
    let g = gpu();
    let f = fixture();
    let lc = LoraCfg::new(2, 4.0, all_leaves());
    let mut rng = Lcg::new(random_seed());
    let mut draw = || rng.scaled(0.3);
    let tower = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &|_| false, Some(LoraTraining { cfg: &lc, init: &mut draw })).unwrap();
    let base = VisionTower::new(&g, &f.cfg, &f.weights).unwrap();
    let lora = tower.lora_store().expect("adapters");
    // Eight leaves: six per layer, two in the merger.
    assert_eq!(lora.params.len(), 2 * (6 * 2 + 2), "A and B for every adapted linear");

    // B starts at zero: the adapted tower is the base, bit for bit, and the base weights are not what trains.
    assert_eq!(tower.encode(&g, &f.images).unwrap(), base.encode(&g, &f.images).unwrap());
    assert!(tower.store().grad.is_empty(), "no tower weight has a gradient buffer");

    // Give B values so the gradient reaches A too, then check every factor of a few sites against differences.
    let mut rng = Lcg::new(random_seed());
    for (name, len) in lora.params.clone() {
        if name.ends_with("lora_b") {
            g.write_f32(lora.w(&name), &rng.vec_scaled(len, 0.3));
        }
    }
    train_step(&g, &tower, &f);
    let loss = || toy_loss(tower.encode(&g, &f.images).unwrap().rows(), &f.dir);
    let names: Vec<String> = lora.params.iter().map(|(n, _)| n.clone()).filter(|n| ["layers.0.self_attn.q_proj", "layers.1.self_attn.v_proj", "layers.1.mlp.c_fc", "layers.0.mlp.c_proj", "layers.1.self_attn.out_proj", "layers.0.self_attn.k_proj", "proj_in", "proj_out"].iter().any(|s| n.contains(s))).collect();
    assert_eq!(names.len(), 16);
    for name in names {
        let analytic = lora.read_grad(&g, &name);
        check_tensor(&name, &analytic, &lora.read_weight(&g, &name), &|p| g.write_f32(lora.w(&name), p), &loss, 5);
    }
    // Everything the adapters do not touch is as it was loaded: a fold writes only the adapted weights.
    let untouched = "model.visual.backbone.layers.0.ln_1.weight";
    assert_eq!(tower.store().read_weight(&g, untouched), base.store().read_weight(&g, untouched));
}

#[test]
fn an_adapter_step_on_a_frozen_tower_lowers_a_toy_loss() {
    let g = gpu();
    let f = fixture();
    let lc = LoraCfg::new(2, 4.0, all_leaves());
    let mut rng = Lcg::new(random_seed());
    let mut draw = || rng.scaled(0.3);
    let tower = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &|_| false, Some(LoraTraining { cfg: &lc, init: &mut draw })).unwrap();
    let lora = tower.lora_store().expect("adapters");
    // Normalised gradient steps of a fixed length over all factors: no learning rate to tune against the loss scale.
    let step = 0.02f32;
    let mut first = None;
    let mut last = 0.0;
    for _ in 0..40 {
        tower.zero_grads(&g);
        let (features, pass) = tower.encode_train(&g, &f.images).unwrap();
        let loss = toy_loss(features.rows(), &f.dir);
        first.get_or_insert(loss);
        last = loss;
        tower.backward(&g, pass, &toy_gradient(features.rows(), &f.dir)).unwrap();
        let grads: Vec<(String, Vec<f32>)> = lora.trained_params().map(|(n, _)| (n.clone(), lora.read_grad(&g, n))).collect();
        let norm = grads.iter().flat_map(|(_, v)| v).map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt() as f32;
        assert!(norm.is_finite() && norm > 0.0, "the adapters receive a gradient");
        for (name, grad) in grads {
            let stepped: Vec<f32> = lora.read_weight(&g, &name).iter().zip(&grad).map(|(w, d)| w - step * d / norm).collect();
            g.write_f32(lora.w(&name), &stepped);
        }
    }
    let first = first.expect("a step ran");
    assert!(last < 0.9 * first, "the loss went from {first} to {last}");
}

#[test]
fn a_tower_refuses_what_it_cannot_do_by_name() {
    let g = gpu();
    let f = fixture();
    let mut rng = Lcg::new(random_seed());
    let mut draw = || rng.scaled(0.3);

    // An adapter over a weight that is trained in full.
    let lc = LoraCfg::new(2, 4.0, vec!["c_fc".to_string()]);
    let e = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &all_trainable, Some(LoraTraining { cfg: &lc, init: &mut draw })).err().expect("trained and adapted");
    assert!(e.contains("c_fc.weight") && e.contains("trained in full"), "{e}");
    // Targets that match no site of the tower.
    let lc = LoraCfg::new(2, 4.0, vec!["wq".to_string()]);
    let e = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &|_| false, Some(LoraTraining { cfg: &lc, init: &mut draw })).err().expect("no such site");
    assert!(e.contains("match no site"), "{e}");
    // A different adapter method.
    let lc = LoraCfg::new(2, 4.0, all_leaves()).with_kind(model::adapter::DeviceKind::Dora);
    let e = VisionTower::new_trainable(&g, &f.cfg, &f.weights, &|_| false, Some(LoraTraining { cfg: &lc, init: &mut draw })).err().expect("plain LoRA only");
    assert!(e.contains("plain LoRA"), "{e}");

    // An inference tower has no backward.
    let served = VisionTower::new(&g, &f.cfg, &f.weights).unwrap();
    let e = served.encode_train(&g, &f.images).err().expect("built for inference");
    assert!(e.contains("inference"), "{e}");
}

#[test]
fn the_resident_projector_serves_groups_of_any_size_and_grows_without_changing_a_feature() {
    let g = gpu();
    let f = fixture();
    let roomy = VisionTower::new(&g, &f.cfg, &f.weights).unwrap();
    // Room for one region: the three regions of the fixture make it grow; features are those of the roomy one.
    let tight = VisionTower::new(&g, &f.cfg, &f.weights).unwrap().with_projector_capacity(&g, 1);
    let want = roomy.encode(&g, &f.images).unwrap();
    assert_eq!(tight.encode(&g, &f.images).unwrap(), want);
    // A smaller group afterwards reads the grown projector's first rows.
    assert_eq!(tight.encode(&g, &f.images[1..]).unwrap().rows(), want.image(1));
}
