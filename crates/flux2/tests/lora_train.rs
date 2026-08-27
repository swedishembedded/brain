// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA training mechanism for FLUX.2, tiny-config end-to-end on the host f32
//! trainer (no VAE/text-encoder — random latents + ctx): with the base
//! **frozen**, training only the low-rank A,B via the gradchecked
//! [`flux2::modelgrad::grads`] path drives the flow-matching loss down over a
//! fixed 2-sample synthetic batch; the adapter round-trips through
//! checkpoint save/load; and `fold_into_tensors` reproduces `apply` exactly
//! through the fused inference layout (proving the row/column offsets match
//! the build-time split) while changing the forward output. No GPU.

use flux2::lora::{load_adapter, save_adapter, LoraAdapter, LoraCfg};
use flux2::modelgrad::{forward, grads, init_model, make_flow_batch, Batch, Cfg, ModelWeights};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 2.0
    }
}

fn batch(c: &Cfg, sigma: f64, r: &mut impl FnMut() -> f64) -> Batch<f32> {
    let x0: Vec<f32> = (0..c.n_img() * c.in_channels).map(|_| r() as f32).collect();
    let ctx: Vec<f32> = (0..c.txt_len * c.context_in_dim).map(|_| r() as f32).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|_| r() as f32).collect();
    make_flow_batch(c, &x0, &ctx, sigma, &noise)
}

/// The tiny [`flux2::Flux2Config`] whose manifest matches [`Cfg::tiny`]
/// (mlp_ratio 0.75 → mlp_hidden 12).
fn tiny_fc() -> flux2::Flux2Config {
    flux2::Flux2Config {
        in_channels: 4,
        context_in_dim: 6,
        hidden: 16,
        n_heads: 2,
        depth_double: 2,
        depth_single: 2,
        mlp_ratio: 0.75,
        axes_dim: [2, 2, 2, 2],
        txt_len: 3,
        ..flux2::Flux2Config::klein_4b()
    }
}

fn manifest_tensors(fc: &flux2::Flux2Config, seed: u64) -> flux2::Tensors {
    let mut r = rng(seed);
    let mut ts = flux2::Tensors::new();
    for (name, shape) in fc.tensor_manifest() {
        let n: usize = shape.iter().product();
        // qk-norm scales centred at 1, everything else small around 0
        let (base, scale) = if name.ends_with("norm.scale") { (1.0, 0.1) } else { (0.0, 0.2) };
        let data: Vec<f32> = (0..n).map(|_| (base + r() * scale) as f32).collect();
        ts.insert(name, (shape, data));
    }
    ts
}

#[test]
fn lora_only_descends_with_base_frozen_and_roundtrips() {
    let c = Cfg::tiny();
    let base = init_model::<f32>(&c, 0x51a7); // FROZEN — never mutated below
    let mut r = rng(0xfeed_0001);
    let batches = [batch(&c, 0.35, &mut r), batch(&c, 0.7, &mut r)]; // fixed 2-sample set
    let lc = LoraCfg::new(4);
    let mut ad = LoraAdapter::new(&c, lc);

    // B=0 at init → adapter is a no-op → same loss as the bare base.
    let (l_base, _) = grads(&c, &base, &batches[0]);
    let (l0, _) = grads(&c, &ad.apply(&base), &batches[0]);
    assert!(
        (l_base - l0).abs() / l_base.max(1e-9) < 1e-6,
        "fresh adapter must be a no-op ({l_base} vs {l0})"
    );

    // ~50 steps on the fixed pair; loss must clearly descend and hold.
    let mean0: f64 = batches.iter().map(|b| grads(&c, &ad.apply(&base), b).0).sum::<f64>() / 2.0;
    let mut last = mean0;
    let mut lmin = mean0;
    for i in 0..50 {
        let b = &batches[i % 2];
        let (_, g) = grads(&c, &ad.apply(&base), b);
        ad.step(&g, 0.02);
        let mean: f64 = batches.iter().map(|b| grads(&c, &ad.apply(&base), b).0).sum::<f64>() / 2.0;
        lmin = lmin.min(mean);
        last = mean;
    }
    eprintln!("FLUX.2 LoRA-only training: loss {mean0:.5} -> {last:.5} (min {lmin:.5})");
    assert!(lmin < 0.75 * mean0, "LoRA-only training barely moved the loss (min {lmin:.4} vs {mean0:.4})");
    assert!(last < 0.85 * mean0, "final loss did not hold near the floor ({last:.4} vs {mean0:.4})");

    // Save/load round-trip through the checkpoint container: reloaded adapter
    // reproduces the same effective weights.
    let path = std::env::temp_dir().join(format!("flux2_lora_test_{}.safetensors", std::process::id()));
    let path = path.to_str().unwrap().to_string();
    save_adapter(&path, &ad);
    let re = load_adapter(&path, &c).expect("reload");
    std::fs::remove_file(&path).ok();
    assert_eq!(re.rank(), ad.rank());
    assert!((re.alpha() - ad.alpha()).abs() < 1e-6);
    let (wa, wb) = (ad.apply(&base), re.apply(&base));
    let diff = wa.dbl[0].img.wq.iter().zip(&wb.dbl[0].img.wq).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let diff2 = wa.sgl[0].wo_b.iter().zip(&wb.sgl[0].wo_b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(diff.max(diff2) < 1e-7, "adapter save/load changed the weights (max diff {diff:.2e}/{diff2:.2e})");
    eprintln!("FLUX.2 LoRA adapter save/load round-trips ({} tensors).", ad.to_tensors().len());

    // fold_into_tensors: folding the trained adapter into the fused inference
    // tensors (a) changes the forward output and (b) matches `apply` exactly —
    // the fold's fused row/column offsets equal the build-time split.
    let fc = tiny_fc();
    let mut ts = manifest_tensors(&fc, 0xD00D);
    // `from_tensors` consumes the map, and this test needs `ts` again after
    // the fold, so the pre-fold extraction gets a copy (tiny config).
    let base_ts = ModelWeights::from_tensors(&c, &mut ts.clone()).unwrap();
    let b = &batches[0];
    let (out_before, _) = forward(&c, &base_ts, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
    ad.fold_into_tensors(&mut ts).expect("fold");
    let folded = ModelWeights::from_tensors(&c, &mut ts).unwrap();
    let (out_after, _) = forward(&c, &folded, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
    let max_change = out_before.iter().zip(&out_after).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_change > 1e-4, "folding a trained adapter did not change the forward output");
    let applied = ad.apply(&base_ts);
    let (out_applied, _) = forward(&c, &applied, &b.img, &b.ctx, b.t, &b.cos, &b.sin);
    let max_dev = out_applied.iter().zip(&out_after).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_dev < 1e-5, "fold_into_tensors diverges from apply (max dev {max_dev:.2e})");
    eprintln!("fold_into_tensors changes the forward (max Δ {max_change:.3e}) and matches apply (dev {max_dev:.1e}).");
}
