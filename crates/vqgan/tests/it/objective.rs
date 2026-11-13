// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the full VQGAN training objective for its clients. If your team needs expertise in
// training an image tokenizer with perceptual and adversarial losses, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The full VQGAN objective on the trainer: what each term contributes to the loss and to the gradient, the adaptive
//! adversarial weight, the discriminator's schedule, and a short overfit run with every term on.
//!
//! The terms' own derivatives are gradient-checked where they are defined (`lpips`'s `backward` test, `entropy`,
//! `patchgan`); what is checked here is the WIRING, exactly, without finite differences. A bias adds to every
//! position of its channel, so a layer's bias gradient is the sum over positions of the gradient at its output:
//! switching one term on changes the generator's last convolution bias gradient by the sum of THAT term's
//! image-space gradient, and the encoder's last convolution bias gradient by the sum of the entropy term's row
//! gradient. Everything else of the two backward passes is identical, so the difference isolates the term.
//!
//! Tolerances are Higham's (`brain_testutil::rounding`): a sum of `k` terms with absolute sum `S` is within
//! `gamma_k S` of exact, two evaluations of it within twice that.

use brain_testutil::rounding::agreement_bound;
use optim::Adam;
use vae::blocks::{Activations, Tensors};
use vqgan::train::{VqganTrainer, TRAIN_PIPELINES};
use vqgan::{BetaTerm, Objective, ReconstructionLoss, StepOptions, VqganConfig};

use crate::recompute::{fill, weights};

const SIDE: u32 = 32;

/// A VQGAN small enough for seconds-long steps: 3 channels (LPIPS compares RGB), 32x32 (above LPIPS' minimum side),
/// a 16x16 latent of 4-dimensional codes.
fn config() -> VqganConfig {
    VqganConfig {
        in_channels: 3,
        out_channels: 3,
        nf: 4,
        ch_mult: vec![1, 2],
        res_blocks: 1,
        dec_res_blocks: 1,
        attn_resolutions: vec![16],
        img_size: SIDE,
        codebook_size: 6,
        emb_dim: 4,
        z_channels: None,
        head_act: false,
        codebook_l2_norm: false,
        beta: 0.25,
        beta_on: BetaTerm::Codebook,
        norm_groups: 2,
        norm_eps: 1e-6,
    }
}

/// [`config`] with LlamaGen's knobs: a wider latent behind `quant_conv`, the L2-normalised codebook.
fn llamagen_config() -> VqganConfig {
    VqganConfig { dec_res_blocks: 2, z_channels: Some(6), head_act: true, codebook_l2_norm: true, beta_on: BetaTerm::Commitment, ..config() }
}

fn trainer_for(cfg: &VqganConfig, obj: Objective, lpips: Option<&lpips::import::Tensors>) -> Result<VqganTrainer, String> {
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let tensors: Tensors = weights(cfg);
    VqganTrainer::with_objective(cfg.clone(), obj, lpips, &tensors, SIDE, SIDE, gpu.share(), Activations::Keep)
}

fn trainer(obj: Objective, lpips: Option<&lpips::import::Tensors>) -> VqganTrainer {
    trainer_for(&config(), obj, lpips).expect("a trainer for a valid objective")
}

fn batch() -> (Vec<f32>, Vec<f32>) {
    let n = (3 * SIDE * SIDE) as usize;
    (fill(n, 1, 1.0), fill(n, 2, 1.0))
}

fn lpips_weights() -> lpips::import::Tensors {
    lpips::import::random(lpips::Trunk::Vgg16, data::rng::random_seed())
}

fn opts() -> StepOptions {
    StepOptions { lr_generator: 1e-2, lr_discriminator: 1e-2, weight_decay: 0.0, adam: Adam::default(), clip: None }
}

/// The generator's last convolution bias: the sum over positions of the gradient at the output image, per channel.
fn last_bias(m: &VqganTrainer) -> String {
    let last = m.config().generator_blocks().len() - 1;
    format!("generator.blocks.{last}.bias")
}

/// The encoder's last convolution bias, when it emits the codes directly: the sum over tokens of the gradient wrt the
/// quantiser's rows.
fn encoder_bias(m: &VqganTrainer) -> String {
    let last = m.config().encoder_blocks().len() - 1;
    format!("encoder.blocks.{last}.bias")
}

/// One forward and one backward of [`VqganTrainer::loss`]'s objective on the fixed batch, gradients left in place.
fn forward_backward(m: &VqganTrainer) -> f32 {
    let (image, target) = batch();
    m.set_batch(&image, &target);
    m.zero_grads();
    let loss = m.loss();
    m.backward();
    m.gpu().poll_wait();
    loss
}

fn l1_norm(v: &[f32]) -> f64 {
    v.iter().map(|x| x.abs() as f64).sum()
}

/// Spec: a perceptual weight without the LPIPS weights, an invalid objective, and a perceptual term on an image the
/// metric cannot take are refused at construction, each naming its cause.
#[test]
fn an_unusable_objective_is_refused_by_name() {
    let obj = Objective { perceptual_weight: 1.0, ..Objective::reconstruction_only() };
    let err = trainer_for(&config(), obj.clone(), None).map(|_| ()).unwrap_err();
    assert!(err.contains("LPIPS v0.1 vgg weights") && err.contains("perceptual_weight"), "{err}");
    let w = lpips_weights();
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let small = VqganTrainer::with_objective(config(), obj, Some(&w), &weights(&config()), 12, 12, gpu.share(), Activations::Keep).map(|_| ()).unwrap_err();
    assert!(small.contains("at least 16x16") && small.contains("vgg"), "{small}");
    // Weights of the other trunk are refused by the metric, naming the missing tensor's trunk.
    let alex = lpips::import::random(lpips::Trunk::Alex, 1);
    let wrong = trainer_for(&config(), Objective { perceptual_weight: 1.0, ..Objective::reconstruction_only() }, Some(&alex)).map(|_| ()).unwrap_err();
    assert!(wrong.contains("vgg"), "{wrong}");
    let negative = trainer_for(&config(), Objective { disc_weight: -1.0, ..Objective::reconstruction_only() }, None).map(|_| ()).unwrap_err();
    assert!(negative.contains("disc_weight"), "{negative}");
}

/// Spec: the perceptual term adds `weight * LPIPS(x_hat, x)` to the loss and `weight * dLPIPS/dx_hat` to the gradient at
/// the output image, and nothing else changes.
#[test]
fn the_perceptual_term_enters_the_loss_and_the_output_gradient() {
    let w = lpips_weights();
    let weight = 0.7f32;
    let on = trainer(Objective { perceptual_weight: weight, ..Objective::reconstruction_only() }, Some(&w));
    let off = trainer(Objective::reconstruction_only(), None);
    let (l_on, l_off) = (forward_backward(&on), forward_backward(&off));
    let (out, (_, target)) = (on.output(), batch());
    assert_eq!(out, off.output(), "the two trainers differ before the perceptual term");

    // LPIPS(x_hat, x) and its gradient wrt x_hat, from the metric on its own device.
    let px = (SIDE * SIDE) as usize;
    let mut metric = lpips::Lpips::with_trunk(gpu_core::testgpu::dev(lpips::GRAD_PIPELINES), lpips::Trunk::Vgg16, &w).expect("lpips");
    let mut pair = out.clone();
    pair.extend(&target);
    let metric_gpu = metric.gpu().share();
    let pair = metric_gpu.storage_init("pair", &pair);
    let (dist, grad) = metric.distance_grad(&pair, SIDE, SIDE).expect("distance_grad");
    let d_image = metric_gpu.read(grad, 3 * px);
    let chain = lpips::Trunk::Vgg16.convs().iter().map(|c| (c.cin * c.k * c.k + c.cout) as usize).sum::<usize>() + 5;

    let value = weight as f64 * dist.total;
    let tol = agreement_bound(chain, value) + 2.0 * brain_testutil::rounding::U * (l_on as f64).abs();
    assert!(((l_on - l_off) as f64 - value).abs() <= tol, "loss rose by {} for a perceptual term of {value} (tolerance {tol:e})", l_on - l_off);

    let name = last_bias(&on);
    let (g_on, g_off) = (on.read_grad(&name), off.read_grad(&name));
    for c in 0..3 {
        let want: f64 = weight as f64 * d_image[c * px..(c + 1) * px].iter().map(|v| *v as f64).sum::<f64>();
        // L1's gradient is +-1/n at every pixel, so the reconstruction's own share of a channel's sum is at most px/n.
        let s = weight as f64 * l1_norm(&d_image[c * px..(c + 1) * px]) + (px as f64) / (3 * px) as f64;
        let tol = agreement_bound(chain + px, s);
        let got = (g_on[c] - g_off[c]) as f64;
        assert!((got - want).abs() <= tol, "channel {c}: the output bias gradient moved by {got}, the perceptual gradient sums to {want} (tolerance {tol:e})");
    }
    assert!(g_on.iter().zip(&g_off).any(|(a, b)| a != b), "the perceptual term changed no gradient");
}

/// The accumulated-operation count `k` of the device entropy kernels against the f64 oracle, for `agreement_bound`:
/// a logit `-|z - e|^2 / tau` is `3 dim + 2` operations deep and its absolute error grows with its size, so the
/// softmax probabilities inherit a relative error of `2 |a|max` times that depth (`exp` amplifies it once, the
/// normalising sum once more); the sums over the codes (3 operations each) and over the tokens (4) add their own.
fn entropy_depth(rows: &[f32], codebook: &[f32], dim: usize, temperature: f32) -> usize {
    let a_max = rows
        .chunks(dim)
        .flat_map(|z| codebook.chunks(dim).map(move |e| z.iter().zip(e).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / temperature as f64))
        .fold(0.0f64, f64::max);
    ((3 * dim + 2) as f64 * 2.0 * (1.0 + a_max)).ceil() as usize + 3 * codebook.len() / dim + 4 * rows.len() / dim
}

/// Spec: the entropy term adds `ratio * L_ent` to the loss, `ratio * dL_ent/drows` to the quantiser's row gradient
/// (summed over tokens, the encoder's last bias) and `ratio * dL_ent/dcodebook` to the codebook's.
#[test]
fn the_entropy_term_enters_the_loss_and_both_gradients() {
    let ratio = 0.3f32;
    let temperature = 0.5f32;
    let obj = Objective { entropy_loss_ratio: ratio, entropy_temperature: temperature, ..Objective::reconstruction_only() };
    let (on, off) = (trainer(obj, None), trainer(Objective::reconstruction_only(), None));
    let (l_on, l_off) = (forward_backward(&on), forward_backward(&off));
    let (rows, codebook) = on.quantizer_inputs();
    let emb = config().emb_dim as usize;
    let ent = vqgan::entropy::entropy_loss(&rows, &codebook, emb, temperature);
    assert!(ent.loss != 0.0 && ent.d_rows.iter().any(|g| g.abs() > 1e-4), "a vanishing entropy term proves nothing: {}", ent.loss);

    let tol = 4.0 * brain_testutil::rounding::U * ((l_on as f64).abs() + ratio as f64 * ent.loss.abs());
    assert!(((l_on - l_off) as f64 - ratio as f64 * ent.loss).abs() <= tol, "loss rose by {} for an entropy term of {}", l_on - l_off, ratio as f64 * ent.loss);

    let (bias_on, bias_off) = (on.read_grad(&encoder_bias(&on)), off.read_grad(&encoder_bias(&off)));
    let t = rows.len() / emb;
    let depth = entropy_depth(&rows, &codebook, emb, temperature);
    for c in 0..emb {
        let want: f64 = (0..t).map(|i| ratio as f64 * ent.d_rows[i * emb + c] as f64).sum();
        let s = (0..t).map(|i| ratio as f64 * (ent.d_rows[i * emb + c] as f64).abs()).sum::<f64>() + bias_off[c].abs() as f64 + bias_on[c].abs() as f64;
        let tol = agreement_bound(t + depth, s);
        let got = (bias_on[c] - bias_off[c]) as f64;
        assert!((got - want).abs() <= tol, "row gradient, channel {c}: encoder bias gradient moved by {got}, the entropy row gradient sums to {want} (tolerance {tol:e})");
    }
    let (cb_on, cb_off) = (on.read_grad("quantize.embedding.weight"), off.read_grad("quantize.embedding.weight"));
    for (j, g) in ent.d_codebook.iter().enumerate() {
        let want = ratio as f64 * *g as f64;
        let got = (cb_on[j] - cb_off[j]) as f64;
        let tol = agreement_bound(depth, want.abs() + cb_on[j].abs() as f64 + cb_off[j].abs() as f64);
        assert!((got - want).abs() <= tol, "codebook gradient entry {j}: moved by {got}, the entropy gradient is {want} (tolerance {tol:e})");
    }
}

/// With the L2-normalised codebook the entropy gradient reaches the parameter through the normalisation: each raw row
/// gets the normalisation's Jacobian applied, `(g - n (n . g)) / |e|`.
#[test]
fn the_entropy_codebook_gradient_goes_through_the_normalisation() {
    let (ratio, temperature) = (0.3f32, 0.5f32);
    let cfg = llamagen_config();
    let obj = Objective { entropy_loss_ratio: ratio, entropy_temperature: temperature, ..Objective::reconstruction_only() };
    let (on, off) = (trainer_for(&cfg, obj, None).unwrap(), trainer_for(&cfg, Objective::reconstruction_only(), None).unwrap());
    forward_backward(&on);
    forward_backward(&off);
    let (rows, normalised) = on.quantizer_inputs();
    let emb = cfg.emb_dim as usize;
    let ent = vqgan::entropy::entropy_loss(&rows, &normalised, emb, temperature);
    let depth = entropy_depth(&rows, &normalised, emb, temperature);
    let raw = on.read_weight("quantize.embedding.weight");
    let (g_on, g_off) = (on.read_grad("quantize.embedding.weight"), off.read_grad("quantize.embedding.weight"));
    for k in 0..cfg.codebook_size as usize {
        let (e, n, g) = (&raw[k * emb..(k + 1) * emb], &normalised[k * emb..(k + 1) * emb], &ent.d_codebook[k * emb..(k + 1) * emb]);
        let norm = e.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let dot: f64 = n.iter().zip(g).map(|(a, b)| *a as f64 * *b as f64).sum();
        for d in 0..emb {
            let want = ratio as f64 * (g[d] as f64 - n[d] as f64 * dot) / norm;
            let got = (g_on[k * emb + d] - g_off[k * emb + d]) as f64;
            let tol = agreement_bound(depth + 64, want.abs() + g_on[k * emb + d].abs() as f64 + g_off[k * emb + d].abs() as f64);
            assert!((got - want).abs() <= tol, "code {k} dim {d}: raw gradient moved by {got}, the normalised entropy gradient maps to {want} (tolerance {tol:e})");
        }
    }
}

/// Spec: the adaptive weight balances the adversarial gradient at the generator's last layer to the reconstruction's:
/// `||g_total - g_nll|| = disc_weight * (||g_nll|| - 1e-4 lambda)`, which also fixes that `disc_weight` and `lambda`
/// both scale the adversarial seed.
#[test]
fn the_adaptive_weight_balances_the_adversarial_gradient_at_the_last_layer() {
    let weight = 0.5f32;
    let adversarial = Objective { disc_weight: weight, disc_start: 0, disc_seed: Some(7), ..Objective::reconstruction_only() };
    let (a, b) = (trainer(adversarial, None), trainer(Objective::reconstruction_only(), None));
    let (image, target) = batch();
    a.set_batch(&image, &target);
    let losses = a.train_step(0, &opts());
    a.gpu().poll_wait();
    forward_backward(&b);
    let lambda = losses.adaptive_weight.expect("the adversarial term is live at step 0");
    assert!(lambda.is_finite() && lambda > 0.0, "lambda {lambda}");
    assert!(losses.adversarial.is_some() && losses.discriminator.is_some());

    let name = {
        let last = a.config().generator_blocks().len() - 1;
        format!("generator.blocks.{last}.weight")
    };
    let (g_a, g_b) = (a.read_grad(&name), b.read_grad(&name));
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let diff: Vec<f32> = g_a.iter().zip(&g_b).map(|(x, y)| x - y).collect();
    let (adv, nll) = (norm(&diff), norm(&g_b));
    let want = weight as f64 * (nll - 1e-4 * lambda as f64);
    // The gradients are sums over the output positions (and the chain behind them), differenced: round-off of `k`
    // terms of absolute sum S per entry, the norm of a vector of such errors bounded by their sum.
    let tol = agreement_bound(4 * (SIDE * SIDE) as usize, l1_norm(&g_a) + l1_norm(&g_b));
    assert!((adv - want).abs() <= tol, "the adversarial gradient at the last layer has norm {adv}, the reconstruction's is {nll}: expected {want} (tolerance {tol:e})");
    assert!(adv > 1e-3 * nll, "an adversarial gradient of {adv} against {nll} balances nothing");
}

/// Spec: a recipe that does not adapt the weight (LlamaGen) applies the adversarial gradient at `disc_weight` alone:
/// the last layer's adversarial gradient is the adaptive run's divided by its `lambda`, and the reported adversarial
/// loss is `disc_weight * (-mean D)` with no `lambda` in it.
#[test]
fn without_adaptation_the_adversarial_term_is_applied_at_weight_one() {
    let weight = 0.5f32;
    let base = Objective { disc_weight: weight, disc_start: 0, disc_seed: Some(7), ..Objective::reconstruction_only() };
    let (adaptive, plain, none) = (
        trainer(Objective { adaptive_weight: true, ..base.clone() }, None),
        trainer(Objective { adaptive_weight: false, ..base }, None),
        trainer(Objective::reconstruction_only(), None),
    );
    let (image, target) = batch();
    for m in [&adaptive, &plain] {
        m.set_batch(&image, &target);
    }
    let (la, lp) = (adaptive.train_step(0, &opts()), plain.train_step(0, &opts()));
    adaptive.gpu().poll_wait();
    plain.gpu().poll_wait();
    forward_backward(&none);
    let lambda = la.adaptive_weight.expect("adaptive recipes report lambda") as f64;
    assert_eq!(lp.adaptive_weight, None, "a recipe without adaptation has no lambda to report");
    // Same discriminator, same generator: the raw generator loss is the same, and only its factor differs.
    let (raw_a, raw_p) = (la.adversarial.unwrap() / (weight as f64 * lambda), lp.adversarial.unwrap() / weight as f64);
    assert!((raw_a - raw_p).abs() <= 1e-6 * raw_p.abs().max(1.0), "raw -mean D differs: {raw_a} vs {raw_p}");

    let last = adaptive.config().generator_blocks().len() - 1;
    let name = format!("generator.blocks.{last}.weight");
    let (g_a, g_p, g_n) = (adaptive.read_grad(&name), plain.read_grad(&name), none.read_grad(&name));
    let norm_diff = |g: &[f32]| g.iter().zip(&g_n).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
    let (adv_a, adv_p) = (norm_diff(&g_a), norm_diff(&g_p));
    let tol = agreement_bound(4 * (SIDE * SIDE) as usize, l1_norm(&g_a) + l1_norm(&g_p) + l1_norm(&g_n));
    assert!((adv_p - adv_a / lambda).abs() <= tol / lambda.min(1.0), "adversarial gradient norms: adaptive {adv_a} (lambda {lambda}), plain {adv_p} (tolerance {tol:e})");
}

/// Spec: LlamaGen's discriminator loss is `disc_weight * hinge`; CodeFormer's is the bare hinge. The same
/// discriminator on the same pair differs by exactly the factor.
#[test]
fn the_discriminator_loss_is_scaled_only_where_the_recipe_scales_it() {
    let weight = 0.5f32;
    let make = |scale_disc_loss| trainer(Objective { disc_weight: weight, scale_disc_loss, disc_seed: Some(5), ..Objective::reconstruction_only() }, None);
    let (scaled, bare) = (make(true), make(false));
    let (image, target) = batch();
    let first = |m: &VqganTrainer| {
        m.set_batch(&image, &target);
        m.loss();
        m.discriminator_step(&opts())
    };
    let (s, b) = (first(&scaled), first(&bare));
    assert!(b > 0.0 && (s - weight as f64 * b).abs() <= 1e-6 * b, "scaled {s} vs bare {b}");
}

/// Spec: the discriminator and the adversarial term begin at `disc_start` and not before.
#[test]
fn the_adversarial_term_begins_at_disc_start() {
    let obj = Objective { disc_weight: 0.5, disc_start: 2, disc_seed: Some(3), ..Objective::reconstruction_only() };
    let m = trainer(obj, None);
    let (image, target) = batch();
    m.set_batch(&image, &target);
    for step in 0..4 {
        let l = m.train_step(step, &opts());
        let live = step >= 2;
        assert_eq!((l.adversarial.is_some(), l.adaptive_weight.is_some(), l.discriminator.is_some()), (live, live, live), "step {step}");
    }
}

/// Spec: trained alone against a fixed real and fake batch, the discriminator's hinge loss falls.
#[test]
fn the_discriminator_loss_falls_when_only_it_trains() {
    let m = trainer(Objective { disc_weight: 0.5, disc_seed: Some(5), ..Objective::reconstruction_only() }, None);
    let (image, target) = batch();
    m.set_batch(&image, &target);
    m.loss();
    let losses: Vec<f64> = (0..25).map(|_| m.discriminator_step(&opts())).collect();
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    assert!(losses[24] < 0.7 * losses[0], "the hinge loss went {} -> {}: {losses:?}", losses[0], losses[24]);
}

/// Spec: with every term on (random LPIPS weights, the discriminator starting after a few steps, a nonzero entropy
/// ratio) every loss component is finite throughout, and the objective falls over an overfit run on one batch.
#[test]
fn the_full_objective_overfits_one_batch() {
    let w = lpips_weights();
    let obj = Objective {
        reconstruction: ReconstructionLoss::L2,
        perceptual_weight: 0.5,
        disc_weight: 0.2,
        disc_start: 4,
        disc_ndf: 4,
        disc_layers: 2,
        disc_seed: Some(11),
        entropy_loss_ratio: 0.05,
        entropy_temperature: 0.5,
        ..Objective::reconstruction_only()
    };
    let m = trainer(obj, Some(&w));
    let (image, target) = batch();
    m.set_batch(&image, &target);
    let mut gen = Vec::new();
    for step in 0..24 {
        let l = m.train_step(step, &opts());
        let parts = [l.reconstruction, l.perceptual, l.codebook, l.commitment, l.entropy, l.generator()];
        assert!(parts.iter().all(|v| v.is_finite()), "step {step}: {l:?}");
        assert!(l.perceptual > 0.0 && l.entropy != 0.0, "step {step}: an inactive term {l:?}");
        if let Some(d) = l.discriminator {
            assert!(d.is_finite(), "step {step}: discriminator {d}");
        }
        // The adversarial term moves with the discriminator; the rest is a function of the parameters alone.
        gen.push(parts[..5].iter().sum::<f64>());
    }
    assert!(gen[23] < 0.8 * gen[0], "the generator objective (without the adversarial term) went {} -> {}: {gen:?}", gen[0], gen[23]);
}
