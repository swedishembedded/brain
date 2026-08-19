// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-resident LoRA training: with the base **frozen**, training only the
//! low-rank `(A, B)` pairs through the gradchecked device trainer drives the
//! flow-matching loss down, and the adapter is an exact no-op at init.
//!
//! Runs on the CPU backend by default and on the real GPU with
//! `BRAIN_DEV_GPU=1`. The second test is the real-weight version of the same
//! comparison `tests/lora_train.rs` runs on the host: a concept-only adapter
//! must lower a HELD-OUT concept clip's loss more than a distractor's. It needs
//! `BRAIN_WAN_{DIT,VAE,T5,TOKENIZER}` and skips loudly without them.

use std::collections::HashMap;
use std::path::Path;

use wan::config::WanConfig;
use wan::import::dit_manifest;
use wan::lora::{LoraAdapter, LoraCfg};
use wan::model::Tensors;
use wan::modelgrad::{make_flow_batch, Batch, Cfg, ModelWeights};
use wan::train::DeviceTrainer;

fn device() -> &'static str {
    if std::env::var("BRAIN_DEV_GPU").as_deref() == Ok("1") {
        "gpu"
    } else {
        "cpu"
    }
}

fn tiny_wan(c: &Cfg) -> WanConfig {
    WanConfig {
        name: "tiny-dev-lora",
        dim: c.dim,
        ffn_dim: c.ffn_dim,
        num_heads: c.n_heads,
        num_layers: c.n_layers,
        in_channels: c.in_channels,
        out_channels: c.out_channels,
        text_dim: c.text_dim,
        text_len: c.text_len,
        freq_dim: c.freq_dim,
        ..WanConfig::t2v_1_3b()
    }
}

fn synthetic_weights(cfg: &WanConfig) -> Tensors {
    let mut t: Tensors = HashMap::new();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for (name, shape) in dit_manifest(cfg) {
        let n: usize = shape.iter().product();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(0.2 * (((state >> 33) as u32) as f32 / (1u64 << 31) as f32 - 0.5));
        }
        if name.contains("norm_q") || name.contains("norm_k") || name.ends_with("norm3.weight") {
            for x in v.iter_mut() {
                *x += 1.0;
            }
        }
        t.insert(name, (shape, v));
    }
    t
}

fn fixed_batch(cfg: &Cfg) -> Batch<f32> {
    let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let rows = cfg.text_len - 1;
    let ctx: Vec<f32> = (0..rows * cfg.text_dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    make_flow_batch(cfg, &x0, &ctx, rows, 0.5, &noise)
}

#[test]
fn device_lora_is_a_no_op_at_init_then_descends_with_the_base_frozen() {
    let cfg = Cfg::tiny();
    let ts = synthetic_weights(&tiny_wan(&cfg));
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let tr = DeviceTrainer::on_device(&cfg, Some(device()));
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(4));

    // B = 0 at init, so the adapter is an exact no-op and the device loss is
    // the bare base's.
    let (l_base, _) = tr.grads(&base, &fixed_batch(&cfg));
    let b = fixed_batch(&cfg);
    let (l0, _) = tr.grads(&ad.apply(&base), &b);
    assert!((l_base - l0).abs() / l_base.max(1e-9) < 1e-9, "a fresh adapter must be a no-op ({l_base} vs {l0})");

    let mut last = l0;
    for step in 0..40 {
        let (l, g) = tr.grads(&ad.apply(&base), &b);
        ad.step(&g, 3e-3);
        if step % 10 == 0 {
            eprintln!("  device lora step {step:>3}  loss {l:.6}");
        }
        last = l;
    }
    eprintln!("device lora ({}): loss {l0:.6} -> {last:.6} over 40 steps (rank 4, lr 3e-3)", device());
    assert!(last < l0 * 0.9, "device LoRA training must descend: {l0} -> {last}");

    // The base is frozen: `apply` clones, so the original is untouched.
    let base_again = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    assert!(base == base_again, "the base weights must not move during LoRA training");
}

/// [`fixed_batch`] at both scalar types, for the f64 oracle comparison.
fn batches(cfg: &Cfg) -> (Batch<f64>, Batch<f32>) {
    let x0: Vec<f64> = (0..cfg.latent_len()).map(|i| ((i % 23) as f64 / 23.0 - 0.5) * 1.1).collect();
    let noise: Vec<f64> = (0..x0.len()).map(|i| ((i % 13) as f64 / 13.0 - 0.5) * 0.8).collect();
    let rows = cfg.text_len - 1;
    let ctx: Vec<f64> = (0..rows * cfg.text_dim).map(|i| ((i % 7) as f64 / 7.0 - 0.5) * 1.4).collect();
    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    (
        make_flow_batch(cfg, &x0, &ctx, rows, 0.5, &noise),
        make_flow_batch(cfg, &f32v(&x0), &f32v(&ctx), rows, 0.5, &f32v(&noise)),
    )
}

/// Relative L2 of the f32 `dev` against the f64 reference `want`.
fn rel_l2(want: &[f64], dev: &[f32]) -> f64 {
    assert_eq!(want.len(), dev.len(), "rel_l2: length");
    let n = want.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff = want.iter().zip(dev).map(|(&a, &b)| (a - b as f64).powi(2)).sum::<f64>().sqrt();
    diff / n.max(1e-9)
}

/// `(out, in)` of the ten targeted linears of one block, in `LEAVES` order.
fn target_shapes(cfg: &Cfg) -> [(usize, usize); 10] {
    let (d, f) = (cfg.dim, cfg.ffn_dim);
    [(d, d), (d, d), (d, d), (d, d), (d, d), (d, d), (d, d), (d, d), (f, d), (d, f)]
}

/// `blocks[l]`'s ten targeted weight matrices, mutably, in `LEAVES` order.
fn targets_mut(w: &mut ModelWeights<f64>, l: usize) -> [&mut Vec<f64>; 10] {
    let b = &mut w.blocks[l];
    [
        &mut b.sq.w, &mut b.sk.w, &mut b.sv.w, &mut b.so.w, &mut b.cq.w, &mut b.ck.w, &mut b.cv.w, &mut b.co.w, &mut b.ff1.w, &mut b.ff2.w,
    ]
}

/// The same ten, from a gradient set.
fn target_grads(g: &wan::modelgrad::ModelGrads<f64>, l: usize) -> [&Vec<f64>; 10] {
    let b = &g.blocks[l];
    [&b.sq.w, &b.sk.w, &b.sv.w, &b.so.w, &b.cq.w, &b.ck.w, &b.cv.w, &b.co.w, &b.ff1.w, &b.ff2.w]
}

/// One block's f64 adapter grads: `(dA, dB)` per targeted linear.
type OracleBlock = Vec<(Vec<f64>, Vec<f64>)>;

/// The f64 oracle for one LoRA step: fold `W_eff = W + scale·B·A` in f64, run
/// the FD-gradchecked host reference at f64, and project `dL/dW_eff` onto
/// `(dA, dB)` in f64. Neither the device path nor the f32 host path is exact
/// here; this is what both are measured against.
fn oracle(cfg: &Cfg, w64: &ModelWeights<f64>, ad: &LoraAdapter, b64: &Batch<f64>) -> (f64, Vec<OracleBlock>) {
    let (r, scale) = (ad.rank(), ad.scale() as f64);
    let shapes = target_shapes(cfg);
    let mut w = w64.clone();
    for l in 0..cfg.n_layers {
        let ab = ad.block_ab(l);
        for (t, ((a, b), (out, inn))) in targets_mut(&mut w, l).into_iter().zip(ab.iter().zip(shapes)) {
            for o in 0..out {
                for k in 0..r {
                    let bok = b[o * r + k] as f64 * scale;
                    for i in 0..inn {
                        t[o * inn + i] += bok * a[k * inn + i] as f64;
                    }
                }
            }
        }
    }
    let (loss, g) = wan::modelgrad::grads(cfg, &w, b64);
    let blocks = (0..cfg.n_layers)
        .map(|l| {
            let ab = ad.block_ab(l);
            target_grads(&g, l)
                .into_iter()
                .zip(ab.iter().zip(shapes))
                .map(|(dw, ((a, b), (out, inn)))| {
                    let mut da = vec![0f64; r * inn];
                    let mut db = vec![0f64; out * r];
                    for o in 0..out {
                        for k in 0..r {
                            let bok = b[o * r + k] as f64 * scale;
                            let mut acc = 0f64;
                            for i in 0..inn {
                                da[k * inn + i] += bok * dw[o * inn + i];
                                acc += dw[o * inn + i] * a[k * inn + i] as f64;
                            }
                            db[o * r + k] = acc * scale;
                        }
                    }
                    (da, db)
                })
                .collect()
        })
        .collect();
    (loss, blocks)
}

/// The on-device fold (`W_eff = base + scale·B·A`) and the on-device
/// projection (`dA = scale·Bᵀ·dW`, `dB = scale·dW·Aᵀ`) against the f64 oracle,
/// alongside the f32 host path they replace.
///
/// The comparison is taken with `B != 0`: at init `B = 0` makes the fold
/// trivially exact and would check nothing. Both f32 paths are held to the
/// same bar, and the device one additionally to the host one's own error -
/// two `cross_attn` pairs of this synthetic config carry adapter grads three
/// orders of magnitude below the rest, so their RELATIVE error is set by
/// cancellation in f32 and neither f32 path resolves them better than the
/// other.
#[test]
fn the_on_device_lora_fold_and_projection_match_the_f64_oracle() {
    let cfg = Cfg::tiny();
    let base = wan::modelgrad::init_model::<f32>(&cfg, 0x9911);
    let base64 = wan::modelgrad::init_model::<f64>(&cfg, 0x9911);
    let rank = 4;
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(rank));
    let (b64, b) = batches(&cfg);
    let mut tr = DeviceTrainer::on_device(&cfg, Some(device()));

    // Two steps to move B off zero.
    for _ in 0..2 {
        let (_, g) = tr.grads(&ad.apply(&base), &b);
        ad.step(&g, 1e-2);
    }

    let (l_want, want) = oracle(&cfg, &base64, &ad, &b64);
    // Host route: apply on the host, full dW back, host projection.
    let (l_host, g) = tr.grads(&ad.apply(&base), &b);
    let hg = ad.project(&g);
    // Device route: the base resident, everything else on-device.
    assert!(tr.begin_lora(&base, rank), "the tiny stack must fit the resident-base budget");
    let (l_dev, dg) = tr.lora_grads(&base, &ad, &b);

    let (rh, rd) = (
        (l_want - l_host).abs() / l_want.abs().max(1e-9),
        (l_want - l_dev).abs() / l_want.abs().max(1e-9),
    );
    eprintln!("lora fold ({}): loss oracle {l_want:.9}  host f32 {l_host:.9} ({rh:.2e})  device {l_dev:.9} ({rd:.2e})", device());
    assert!(rd < 1e-6, "the on-device fold must reproduce the oracle loss: {l_want} vs {l_dev}");

    let (mut worst_d, mut worst_h, mut name) = (0f64, 0f64, String::new());
    assert_eq!(dg.blocks.len(), want.len(), "block count");
    for (l, (wb, (hb, db))) in want.iter().zip(hg.blocks.iter().zip(&dg.blocks)).enumerate() {
        for (i, ((wa, wbv), ((hda, hdb), (dda, ddb)))) in wb.iter().zip(hb.iter().zip(db)).enumerate() {
            for (what, w, h, d) in [("dA", wa, hda, dda), ("dB", wbv, hdb, ddb)] {
                let (rd, rh) = (rel_l2(w, d), rel_l2(w, h));
                if rd > worst_d {
                    (worst_d, worst_h, name) = (rd, rh, format!("blocks.{l} pair {i} {what}"));
                }
            }
        }
    }
    eprintln!("lora project ({}) vs f64 oracle: worst device rel_l2 {worst_d:.3e} at {name} (host f32 there: {worst_h:.3e})", device());
    assert!(worst_d < 1e-4, "on-device adapter grads vs the f64 oracle: rel_l2 {worst_d:.3e} at {name}");
    assert!(worst_d < 4.0 * worst_h.max(1e-7), "the device path must be no less accurate than the host one at {name}: {worst_d:.3e} vs {worst_h:.3e}");
}

/// The on-device route must train the SAME adapter the host route does: same
/// losses step by step, same descent.
#[test]
fn the_on_device_lora_route_tracks_the_host_route_over_a_run() {
    let cfg = Cfg::tiny();
    let ts = synthetic_weights(&tiny_wan(&cfg));
    let base = ModelWeights::from_tensors(&cfg, &ts).expect("host weights");
    let (rank, lr, steps) = (4, 3e-3, 8);
    let b = fixed_batch(&cfg);

    let host: Vec<f64> = {
        let tr = DeviceTrainer::on_device(&cfg, Some(device()));
        let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(rank));
        (0..steps)
            .map(|_| {
                let (l, g) = tr.grads(&ad.apply(&base), &b);
                ad.step(&g, lr);
                l
            })
            .collect()
    };

    let mut tr = DeviceTrainer::on_device(&cfg, Some(device()));
    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(rank));
    assert!(tr.begin_lora(&base, rank), "the tiny stack must fit the resident-base budget");
    let dev: Vec<f64> = (0..steps)
        .map(|_| {
            let (l, g) = tr.lora_grads(&base, &ad, &b);
            ad.step_projected(&g, lr);
            l
        })
        .collect();

    for (i, (h, d)) in host.iter().zip(&dev).enumerate() {
        let r = (h - d).abs() / h.abs().max(1e-9);
        eprintln!("  step {i:>2}  host {h:.9}  device {d:.9}  rel {r:.2e}");
        assert!(r < 1e-5, "step {i}: host {h} vs on-device {d}");
    }
    assert!(dev[steps - 1] < dev[0] * 0.98, "the on-device LoRA route must descend: {} -> {}", dev[0], dev[steps - 1]);
}

// ============ real weights, real data, GPU (smoke scale) ============

/// `BRAIN_WAN_{DIT,VAE,T5,TOKENIZER}` or a loud skip.
fn real_paths() -> Option<wan::Paths> {
    match wan::Paths::from_env() {
        Ok(p) => Some(p),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("set BRAIN_WAN_{{DIT,VAE,T5,TOKENIZER}} for the real-weight device LoRA gate: {e}"));
            None
        }
    }
}

fn read_pth(path: &str) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    Ok(checkpoint::torchpt::read(path)?
        .into_iter()
        .map(|t| checkpoint::safetensors::StTensor { name: t.name, shape: t.shape, data: t.data })
        .collect())
}

/// Mean flow-matching loss over `(latent, ctx)` pairs at FIXED per-sample
/// `(sigma, noise)` draws, so a before/after comparison never confounds "the
/// adapter changed" with "a different noise draw landed".
/// `on_device` picks the route: the resident-base one (`W_eff` folded on the
/// GPU) or the host-apply one it replaces. Both compute the same quantity.
fn mean_loss(
    tr: &DeviceTrainer,
    base: &ModelWeights<f32>,
    adapter: &LoraAdapter,
    samples: &[(Vec<f32>, Vec<f32>)],
    sigmas: &[f64],
    noises: &[Vec<f32>],
    on_device: bool,
) -> f64 {
    let cfg = *tr.cfg();
    let w = if on_device { None } else { Some(adapter.apply(base)) };
    let total: f64 = samples
        .iter()
        .enumerate()
        .map(|(i, (latent, ctx))| {
            let rows = ctx.len() / cfg.text_dim;
            let b = make_flow_batch(&cfg, latent, ctx, rows, sigmas[i], &noises[i]);
            match &w {
                Some(w) => tr.grads(w, &b).0,
                None => tr.lora_grads(base, adapter, &b).0,
            }
        })
        .sum();
    total / samples.len() as f64
}

#[test]
fn a_gpu_trained_concept_lora_lowers_held_out_concept_loss_more_than_distractor_loss() {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the real-weight device LoRA gate");
        return;
    }
    let Some(paths) = real_paths() else { return };
    let cfg = WanConfig::t2v_1_3b();
    let (frames, size) = (5usize, 64u32);
    let (n_train, n_eval) = (2usize, 1usize);

    let base_dir = std::env::temp_dir().join(format!("wan-dev-g1-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base_dir);
    let concept = data::gen_clips::generate_concept_set(n_train + n_eval, frames, size, size, 101);
    let distractor = data::gen_clips::generate_distractor_set(n_eval, frames, size, size, 202);
    let (train_c, heldout_c) = concept.split_at(n_train);
    let (train_dir, heldout_dir, distractor_dir) = (base_dir.join("train"), base_dir.join("heldout"), base_dir.join("distractor"));
    data::videoset::write_clipset(&train_dir, train_c, size, size, 8).expect("train clips");
    data::videoset::write_clipset(&heldout_dir, heldout_c, size, size, 8).expect("heldout clips");
    data::videoset::write_clipset(&distractor_dir, &distractor, size, size, 8).expect("distractor clips");

    let train_set = wan::finetune::ClipSet::load_dir(&train_dir).expect("train set");
    let heldout_set = wan::finetune::ClipSet::load_dir(&heldout_dir).expect("heldout set");
    let distractor_set = wan::finetune::ClipSet::load_dir(&distractor_dir).expect("distractor set");
    let mut drng = data::rng::Rng::new(777);
    let train_clips: Vec<wan::finetune::Clip> = (0..n_train).map(|_| train_set.sample(&mut drng, frames).expect("sample")).collect();
    let heldout_clips: Vec<wan::finetune::Clip> = (0..n_eval).map(|_| heldout_set.sample(&mut drng, frames).expect("sample")).collect();
    let distractor_clips: Vec<wan::finetune::Clip> = (0..n_eval).map(|_| distractor_set.sample(&mut drng, frames).expect("sample")).collect();

    let (lf, lh, lw) = cfg.latent_shape(frames, size as usize, size as usize).expect("latent shape");
    let tcfg = Cfg::from_wan(&cfg, lf, lh, lw);
    let t0 = std::time::Instant::now();

    // ---- ONE umT5 session: train + held-out-concept + distractor captions ----
    let ctxs: Vec<Vec<f32>> = {
        let tok = if Path::new(&paths.tokenizer).is_dir() {
            data::unigram::UnigramTokenizer::from_dir(&paths.tokenizer)
        } else {
            data::unigram::UnigramTokenizer::from_file(&paths.tokenizer)
        }
        .expect("tokenizer");
        let t5cfg = t5encoder::config::T5Config::umt5_xxl();
        let imported = t5encoder::import::import_wan(read_pth(&paths.t5).expect("read t5"), &t5cfg).expect("import t5");
        let gpu = gpu_core::Gpu::new_cpu(t5encoder::model::PIPELINES);
        let enc = t5encoder::model::T5Encoder::new_on(gpu, t5cfg, 1, cfg.text_len as u32, &t5encoder::import::to_init(imported));
        train_clips
            .iter()
            .chain(&heldout_clips)
            .chain(&distractor_clips)
            .map(|c| {
                let (ids, mask) = tok.encode_padded(&c.caption, cfg.text_len);
                enc.set_tokens(&ids);
                enc.set_mask(&mask);
                enc.forward();
                enc.poll_wait();
                enc.read_context()
            })
            .collect()
    };

    // ---- ONE Wan-VAE session ----
    let latents: Vec<Vec<f32>> = {
        let vcfg = wan::vae3d::WanVaeConfig::wan21();
        let vweights = wan::import::import_vae(read_pth(&paths.vae).expect("read vae"), &vcfg).expect("import vae");
        let enc = wan::vae3d::WanVaeEncoder::build(&vcfg, &vweights, &vcfg.encode_chunks(frames as u32), size, size, None);
        train_clips.iter().chain(&heldout_clips).chain(&distractor_clips).map(|c| enc.encode(&c.video)).collect()
    };
    eprintln!("device G1: umT5 + VAE encode ({} clips) in {:.1}s", latents.len(), t0.elapsed().as_secs_f32());

    let samples: Vec<(Vec<f32>, Vec<f32>)> = latents.into_iter().zip(ctxs).collect();
    let (train_s, rest) = samples.split_at(n_train);
    let (heldout_s, distractor_s) = rest.split_at(n_eval);

    let mut ern = data::rng::Rng::new(555);
    let sigmas: Vec<f64> = (0..2 * n_eval).map(|_| (ern.next_f64()).clamp(1e-3, 1.0)).collect();
    let noises: Vec<Vec<f32>> = heldout_s
        .iter()
        .chain(distractor_s)
        .map(|(latent, _)| (0..latent.len()).map(|_| ern.next_gaussian() as f32).collect())
        .collect();
    let (c_sigmas, d_sigmas) = sigmas.split_at(n_eval);
    let (c_noises, d_noises) = noises.split_at(n_eval);

    let t1 = std::time::Instant::now();
    let raw = checkpoint::safetensors::read(&paths.dit).expect("read DiT safetensors");
    let tensors = wan::import::import_dit(raw, &cfg).expect("import DiT");
    let base = ModelWeights::from_tensors(&tcfg, &tensors).expect("host weights");
    drop(tensors);
    eprintln!("device G1: DiT import in {:.1}s", t1.elapsed().as_secs_f32());

    let mut tr = DeviceTrainer::on_device(&tcfg, Some("gpu"));
    let rank = 8;
    let mut adapter = LoraAdapter::new(&tcfg, LoraCfg::new(rank));
    let fresh = LoraAdapter::new(&tcfg, LoraCfg::new(rank));

    // The starting losses through the route this replaces, then through the
    // resident-base one. A fresh adapter has `B = 0`, so the two folds are the
    // same weights and the two numbers must agree to the last digits.
    let host_concept = mean_loss(&tr, &base, &fresh, heldout_s, c_sigmas, c_noises, false);
    let host_distractor = mean_loss(&tr, &base, &fresh, distractor_s, d_sigmas, d_noises, false);

    let t_up = std::time::Instant::now();
    assert!(tr.begin_lora(&base, rank), "the 1.3B stack must fit the resident-base budget on this card");
    eprintln!("device G1: frozen base resident in {:.1}s", t_up.elapsed().as_secs_f32());

    let before_concept = mean_loss(&tr, &base, &fresh, heldout_s, c_sigmas, c_noises, true);
    let before_distractor = mean_loss(&tr, &base, &fresh, distractor_s, d_sigmas, d_noises, true);
    eprintln!("device G1: start concept    host-apply {host_concept:.6} vs on-device {before_concept:.6}");
    eprintln!("device G1: start distractor host-apply {host_distractor:.6} vs on-device {before_distractor:.6}");
    for (what, h, d) in [("concept", host_concept, before_concept), ("distractor", host_distractor, before_distractor)] {
        let r = (h - d).abs() / h.abs().max(1e-9);
        assert!(r < 1e-6, "{what}: the two LoRA routes disagree at init: {h} vs {d} (rel {r:.2e})");
    }

    let steps = 15u32;
    let mut trng = data::rng::Rng::new(4242);
    let t2 = std::time::Instant::now();
    for _ in 0..steps {
        let idx = trng.gen_range_inclusive(0, train_s.len() as i64 - 1) as usize;
        let (latent, ctx) = &train_s[idx];
        let sigma = (trng.next_f64()).clamp(1e-3, 1.0);
        let noise: Vec<f32> = (0..latent.len()).map(|_| trng.next_gaussian() as f32).collect();
        let b = make_flow_batch(&tcfg, latent, ctx, cfg.text_len, sigma, &noise);
        let (_l, g) = tr.lora_grads(&base, &adapter, &b);
        adapter.step_projected(&g, 3e-3);
    }
    let train_s_secs = t2.elapsed().as_secs_f32();
    eprintln!("device G1: trained {steps} steps in {train_s_secs:.1}s ({:.2}s/step)", train_s_secs / steps as f32);

    let after_concept = mean_loss(&tr, &base, &adapter, heldout_s, c_sigmas, c_noises, true);
    let after_distractor = mean_loss(&tr, &base, &adapter, distractor_s, d_sigmas, d_noises, true);

    let concept_drop = before_concept - after_concept;
    let distractor_drop = before_distractor - after_distractor;
    eprintln!("device G1: held-out concept loss {before_concept:.6} -> {after_concept:.6} (drop {concept_drop:.6})");
    eprintln!("device G1: distractor loss       {before_distractor:.6} -> {after_distractor:.6} (drop {distractor_drop:.6})");
    eprintln!("device G1: total wall time {:.1}s", t0.elapsed().as_secs_f32());

    assert!(concept_drop > 0.0, "held-out concept loss must fall during training: {before_concept} -> {after_concept}");
    assert!(
        concept_drop > distractor_drop,
        "held-out concept loss must fall MORE than the distractor's: concept drop {concept_drop:.6} vs distractor drop {distractor_drop:.6}"
    );

    let _ = std::fs::remove_dir_all(&base_dir);
}
