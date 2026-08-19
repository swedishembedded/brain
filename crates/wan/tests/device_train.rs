// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device full-model training for the Wan DiT: (1) the GPU trainer's gradients
//! match the finite-difference-gradchecked host reference (`modelgrad`) at
//! fp32, tensor by tensor over the whole checkpoint manifest, and (2) it
//! overfits one batch.
//!
//! This is the end-to-end device training loop - the block stack on the GPU
//! through the persistent [`wan::BlockDev`] engine, wrapped by the host
//! patch embedding, text MLP, timestep path, modulated head and flow-matching
//! loss. Runs on the CPU backend by default and on the real GPU with
//! `BRAIN_DEV_GPU=1`.

use wan::modelgrad::{self, grad_views, grads, init_model, make_flow_batch, params_mut, Batch, Cfg};
use wan::train::DeviceTrainer;

fn device() -> &'static str {
    if std::env::var("BRAIN_DEV_GPU").as_deref() == Ok("1") {
        "gpu"
    } else {
        "cpu"
    }
}

fn rel_l2(host: &[f64], dev: &[f64]) -> f64 {
    let n = host.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff = host.iter().zip(dev).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    diff / n.max(1e-9)
}

/// One deterministic training example, at both scalar types.
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

#[test]
fn device_grads_match_the_gradchecked_host_reference() {
    let cfg = Cfg::tiny();
    let w64 = init_model::<f64>(&cfg, 0x9911);
    let w32 = init_model::<f32>(&cfg, 0x9911);
    let (b64, b32) = batches(&cfg);

    let (hl, hg) = grads(&cfg, &w64, &b64);
    let tr = DeviceTrainer::on_device(&cfg, Some(device()));
    let (dl, dg) = tr.grads(&w32, &b32);

    eprintln!("loss host(f64)={hl:.8} device({})={dl:.8}", device());
    assert!((hl - dl).abs() / hl.abs().max(1e-9) < 1e-4, "loss mismatch {hl} vs {dl}");

    let hv = grad_views(&hg);
    let dv = grad_views(&dg);
    assert_eq!(hv.len(), dv.len());
    let (mut worst, mut worst_name) = (0.0f64, String::new());
    for ((hn, h), (dn, d)) in hv.iter().zip(&dv) {
        assert_eq!(hn, dn, "grad order");
        let d64: Vec<f64> = d.iter().map(|&x| x as f64).collect();
        let rel = rel_l2(h, &d64);
        assert!(rel.is_finite(), "{hn}: non-finite grad");
        if rel > worst {
            worst = rel;
            worst_name = hn.clone();
        }
    }
    eprintln!("device vs host grads over {} tensors: worst rel_l2 = {worst:.3e} ({worst_name})", hv.len());
    assert!(worst < 1e-4, "device grad rel_l2 {worst:.3e} too high ({worst_name})");
}

#[test]
fn the_device_trainer_overfits_one_batch() {
    let cfg = Cfg::tiny();
    let mut w = init_model::<f32>(&cfg, 0x4242);
    let (_b64, b) = batches(&cfg);
    let tr = DeviceTrainer::on_device(&cfg, Some(device()));

    let nparams: usize = {
        let mut c = w.clone();
        params_mut(&mut c).iter().map(|(_, p)| p.len()).sum()
    };
    let mut m = vec![0f32; nparams];
    let mut v = vec![0f32; nparams];
    let (lr, b1, b2, eps) = (3e-3f32, 0.9f32, 0.999f32, 1e-8f32);

    let (l0, _) = tr.grads(&w, &b);
    let mut l = l0;
    for step in 1..=60 {
        let (loss, g) = tr.grads(&w, &b);
        l = loss;
        let gv: Vec<Vec<f32>> = grad_views(&g).into_iter().map(|(_, x)| x.clone()).collect();
        let bc1 = 1.0 - b1.powi(step);
        let bc2 = 1.0 - b2.powi(step);
        let mut off = 0;
        for (i, (_, p)) in params_mut(&mut w).into_iter().enumerate() {
            for j in 0..p.len() {
                let gj = gv[i][j];
                m[off] = b1 * m[off] + (1.0 - b1) * gj;
                v[off] = b2 * v[off] + (1.0 - b2) * gj * gj;
                p[j] -= lr * (m[off] / bc1) / ((v[off] / bc2).sqrt() + eps);
                off += 1;
            }
        }
        if step % 20 == 0 {
            eprintln!("  step {step:3}: loss = {l:.4e}");
        }
    }
    eprintln!("device overfit ({}): loss {l0:.4e} -> {l:.4e}", device());
    assert!(l < l0 * 0.05, "the device trainer did not overfit: {l0:.4e} -> {l:.4e}");
}

/// A device step and a host step must move the loss the same way from the same
/// start - the property a trainer swap has to preserve.
#[test]
fn a_device_step_and_a_host_step_agree_on_the_loss_trajectory() {
    let cfg = Cfg::tiny();
    let w0 = init_model::<f32>(&cfg, 0x1357);
    let (_b64, b) = batches(&cfg);
    let tr = DeviceTrainer::on_device(&cfg, Some(device()));

    let mut hw = w0.clone();
    let mut dw = w0;
    let (mut hl, mut dl) = (0.0, 0.0);
    for _ in 0..5 {
        let (l, g) = modelgrad::grads(&cfg, &hw, &b);
        hl = l;
        sgd(&mut hw, &g, 1e-2);
        let (l, g) = tr.grads(&dw, &b);
        dl = l;
        sgd(&mut dw, &g, 1e-2);
    }
    eprintln!("5 SGD steps: host loss {hl:.6}, device loss {dl:.6}");
    assert!((hl - dl).abs() / hl.abs().max(1e-9) < 5e-3, "trajectories diverged: {hl} vs {dl}");
}

fn sgd(w: &mut wan::modelgrad::ModelWeights<f32>, g: &wan::modelgrad::ModelGrads<f32>, lr: f32) {
    let gv: Vec<Vec<f32>> = grad_views(g).into_iter().map(|(_, x)| x.clone()).collect();
    for (i, (_, p)) in params_mut(w).into_iter().enumerate() {
        for (a, &b) in p.iter_mut().zip(&gv[i]) {
            *a -= lr * b;
        }
    }
}

/// Wall-clock of one full-scale forward+backward, device against host, at the
/// real 1.3B topology and a small latent extent - the number that decides
/// whether a LoRA run is minutes or hours. Synthetic weights, so it needs no
/// checkpoint; opt in with `BRAIN_WAN_DEV_BENCH=1` (it allocates ~5.6 GB of
/// host weights and runs one host step, which is slow by construction).
#[test]
fn device_and_host_step_wall_clock_at_the_real_topology() {
    if std::env::var("BRAIN_WAN_DEV_BENCH").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_WAN_DEV_BENCH=1 for the full-scale device/host step timing");
        return;
    }
    let wc = wan::WanConfig::t2v_1_3b();
    let (frames, size) = (5usize, 64usize);
    let (lf, lh, lw) = wc.latent_shape(frames, size, size).expect("latent shape");
    let cfg = Cfg::from_wan(&wc, lf, lh, lw);
    eprintln!(
        "bench: dim {} ffn {} heads {} layers {} | {} tokens, {} text rows",
        cfg.dim, cfg.ffn_dim, cfg.n_heads, cfg.n_layers, cfg.n_tokens(), cfg.text_len
    );

    let t0 = std::time::Instant::now();
    let w = init_model::<f32>(&cfg, 7);
    eprintln!("bench: synthetic weights in {:.1}s", t0.elapsed().as_secs_f32());
    let x0: Vec<f32> = (0..cfg.latent_len()).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let ctx: Vec<f32> = (0..cfg.text_len * cfg.text_dim).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 0.4).collect();
    let b = make_flow_batch(&cfg, &x0, &ctx, cfg.text_len, 0.5, &noise);

    let mut tr = DeviceTrainer::on_device(&cfg, Some("gpu"));
    let t1 = std::time::Instant::now();
    let (dl, _dg) = tr.grads(&w, &b);
    let warm = t1.elapsed().as_secs_f64();
    let t2 = std::time::Instant::now();
    let (dl2, _dg) = tr.grads(&w, &b);
    let dev = t2.elapsed().as_secs_f64();
    eprintln!("bench: device step {dev:.2}s (first {warm:.2}s), loss {dl:.6}/{dl2:.6}");

    // Phase split: one block's forward (weight upload + graph) and one block's
    // backward (upload + forward recompute + backward + grad readback), each
    // repeated over the stack's depth, so the trainer's cost is attributable
    // rather than a single number.
    let d = cfg.dims();
    let bw = &w.blocks[0];
    let (e0, ctxe) = (vec![0.01f32; 6 * cfg.dim], vec![0.02f32; cfg.text_len * cfg.dim]);
    let x = vec![0.03f32; d.t * cfg.dim];
    let t4 = std::time::Instant::now();
    for _ in 0..cfg.n_layers {
        let _ = tr.engine().forward(d, bw, &x, &e0, &ctxe, &b.cos, &b.sin);
    }
    let fwd = t4.elapsed().as_secs_f64();
    let t5 = std::time::Instant::now();
    for _ in 0..cfg.n_layers {
        let _ = tr.engine().backward(d, bw, &x, &e0, &ctxe, &b.cos, &b.sin, &x);
    }
    let bwd = t5.elapsed().as_secs_f64();
    eprintln!("bench: {} block forwards {fwd:.2}s, {} block backwards {bwd:.2}s (host wrapper {:.2}s)", cfg.n_layers, cfg.n_layers, (dev - fwd - bwd).max(0.0));

    // The LoRA bookkeeping a finetune step pays on top of the trainer: rebuild
    // the effective weights, then project every base-weight grad onto the
    // low-rank pairs and Adam-step them.
    let mut ad = wan::lora::LoraAdapter::new(&cfg, wan::lora::LoraCfg::new(8));
    let t6 = std::time::Instant::now();
    let weff = ad.apply(&w);
    let apply = t6.elapsed().as_secs_f64();
    let (_l, g) = tr.grads(&weff, &b);
    let t7 = std::time::Instant::now();
    ad.step(&g, 1e-4);
    let proj = t7.elapsed().as_secs_f64();
    eprintln!("bench: lora apply {apply:.2}s, lora project+adam {proj:.2}s (rank 8)");
    drop((weff, g));

    // A WHOLE LoRA step by each route. The host-apply one rebuilds the
    // effective weights on the host, uploads them and reads every full weight
    // grad back; the on-device one keeps the frozen base resident and moves
    // only the rank-sized adapter in either direction.
    let t8 = std::time::Instant::now();
    let (hostl, g) = tr.grads(&ad.apply(&w), &b);
    ad.step(&g, 1e-4);
    drop(g);
    let host_lora = t8.elapsed().as_secs_f64();

    let t9 = std::time::Instant::now();
    let resident = tr.begin_lora(&w, 8);
    let upload = t9.elapsed().as_secs_f64();
    assert!(resident, "the resident-base budget must hold the whole 1.3B stack");
    let mut devl = 0.0;
    let mut dev_lora = 0.0;
    for i in 0..2 {
        let t10 = std::time::Instant::now();
        let (l, lg) = tr.lora_grads(&w, &ad, &b);
        ad.step_projected(&lg, 1e-4);
        dev_lora = t10.elapsed().as_secs_f64();
        devl = l;
        eprintln!("bench: on-device lora step {i} {dev_lora:.2}s, loss {l:.6}");
    }
    eprintln!(
        "bench: lora step - host-apply route {host_lora:.2}s (loss {hostl:.6}), on-device route {dev_lora:.2}s (loss {devl:.6}), one-time base upload {upload:.1}s, {:.1}x",
        host_lora / dev_lora.max(1e-9)
    );

    let t3 = std::time::Instant::now();
    let (hl, _hg) = modelgrad::grads(&cfg, &w, &b);
    let host = t3.elapsed().as_secs_f64();
    eprintln!("bench: host step   {host:.2}s, loss {hl:.6}");
    eprintln!("bench: speedup {:.1}x", host / dev.max(1e-9));
    assert!((hl - dl2).abs() / hl.abs().max(1e-9) < 1e-3, "loss mismatch host {hl} vs device {dl2}");
}
