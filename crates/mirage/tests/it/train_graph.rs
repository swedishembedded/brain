// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the training graph computes the inference operators' values (f32 against the host oracle, stage by
//! stage on the tiny configuration), and its backward equals central finite differences of the whole model in
//! `f64`, with a nonzero gradient reaching the encoder from every head.

use mirage::decoder::{Decoder, Positions};
use mirage::train::net::{params_from_checkpoint, Net, Params};
use mirage::train::tape::Id;
use mirage::vision::memory::Slot;
use mirage::vision::ops::valid_cells;
use mirage::vision::weights::Weights;
use mirage::vision::VisionModel;

use crate::common::tiny_checkpoint;
use crate::fd::noise;
use crate::vision::assert_stage;

const SIZE: (usize, usize) = (400, 300);

fn slots(cfg: &mirage::config::MirageConfig) -> Vec<Slot> {
    let base = cfg.text + mirage::config::LANGUAGE_TAGS;
    vec![
        Slot { bbox: [0.1, 0.1, 0.6, 0.5], token: base },
        Slot { bbox: [0.3, 0.2, 0.9, 0.7], token: base + 1 },
        Slot { bbox: [0.05, 0.6, 0.45, 0.95], token: base + cfg.detections },
    ]
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[test]
fn the_f32_graph_equals_the_host_operators_stage_by_stage() {
    crate::common::cap_threads();
    let ck = tiny_checkpoint();
    let cfg = ck.config.clone();
    let vm = VisionModel::new(&ck);
    let params: Params<f32> = params_from_checkpoint(&ck);
    let canvas = f32s(&noise(cfg.canvas * cfg.canvas * 3, 71, 1.0));
    let (h, w) = SIZE;

    let host_enc = vm.encoder.forward(&canvas, h, w);
    let host_dense = vm.dense.forward(&host_enc, h, w);
    let (host_protos, host_coefs) = vm.dense.mask_maps(&host_dense.head_in, &host_enc, h, w);

    let mut net = Net::<f32>::new(&cfg, &params, None);
    let canvas_id = net.constant(canvas);
    let enc = net.encoder(canvas_id, h, w);
    for (k, (_, id)) in enc.stages.iter().enumerate() {
        assert_stage(&format!("stage{k}"), net.tape.val(*id), &host_enc.stages[k].1);
    }
    assert_stage("top", net.tape.val(enc.top), &host_enc.top);
    assert_stage("encoder", net.tape.val(enc.memory), &host_enc.encoder);
    let dense = net.dense(&enc, h, w);
    assert_stage("head_in", net.tape.val(dense.head_in), &host_dense.head_in);
    assert_stage("head", net.tape.val(dense.head), &host_dense.head);
    let (protos, coefs) = net.mask_maps(dense.head_in, &enc, h, w);
    assert_stage("protos", net.tape.val(protos), &host_protos);
    assert_stage("coefs", net.tape.val(coefs), &host_coefs);

    let grid = cfg.grid();
    let valid = valid_cells(h, w, grid);
    let slot_list = slots(&cfg);
    let host_mem = vm.memory.build(&host_enc.encoder, &valid, &host_dense.head_in, &slot_list);
    let mem = net.decoder_memory(&enc, dense.head_in, &valid, &slot_list);
    assert_stage("memory", net.tape.val(mem), &host_mem);

    let window: Vec<u8> = (0..48 * 320 * 3).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    let host_read = vm.reader.forward(&window);
    let win = net.constant(window.iter().map(|&p| f32::from(p) / 127.5 - 1.0).collect());
    let logits = net.reader(win);
    assert_stage("reader", net.tape.val(logits), &host_read);

    let dec = Decoder::new(&cfg, &Weights::new(&ck));
    let tokens: Vec<u32> = vec![2, 24, 5, 17, 36, 9, 9, 30, 1];
    let rows = host_mem.len() / cfg.dim;
    let cross = dec.cross_memory(&host_mem, rows);
    let host_logits = dec.forward(&mut dec.new_state(), &cross, &tokens, Positions::All);
    let got = net.decoder_logits(&tokens, mem, rows);
    assert_stage("decoder logits", net.tape.val(got), &host_logits);
}

/// A scalar of every head's output: a fixed random projection of the detection map, the mask maps, the reader
/// logits and the decoder logits, so one backward reaches the encoder from each of them.
fn all_heads_loss(net: &mut Net<'_, f64>, canvas: Id) -> Id {
    let cfg = net.cfg;
    let (h, w) = SIZE;
    let enc = net.encoder(canvas, h, w);
    let dense = net.dense(&enc, h, w);
    let (protos, coefs) = net.mask_maps(dense.head_in, &enc, h, w);
    let valid = valid_cells(h, w, cfg.grid());
    let mem = net.decoder_memory(&enc, dense.head_in, &valid, &slots(cfg));
    let rows = net.tape.len(mem) / cfg.dim;
    let window = net.constant(noise(48 * 320 * 3, 5, 1.0));
    let read = net.reader(window);
    let logits = net.decoder_logits(&[2, 24, 5, 17, 36, 9, 30, 1], mem, rows);
    let mut parts = Vec::new();
    for (i, id) in [dense.head, protos, coefs, read, logits].into_iter().enumerate() {
        let proj = net.constant(noise(net.tape.len(id), 100 + i as u64, 1.0));
        let prod = net.tape.mul(id, proj);
        parts.push(net.tape.sum_weighted(prod, 1.0));
    }
    let mut total = parts[0];
    for p in &parts[1..] {
        total = net.tape.add(total, *p);
    }
    total
}

fn loss_and_grads(cfg: &mirage::config::MirageConfig, params: &Params<f64>) -> (f64, std::collections::BTreeMap<String, Vec<f64>>) {
    let mut net = Net::<f64>::new(cfg, params, None);
    let canvas = net.constant(noise(cfg.canvas * cfg.canvas * 3, 71, 1.0));
    let loss = all_heads_loss(&mut net, canvas);
    let value = net.tape.val(loss)[0];
    let mut grads = net.tape.backward(loss);
    (value, net.param_grads(&mut grads))
}

#[test]
fn the_whole_model_matches_finite_differences_and_every_head_reaches_the_encoder() {
    crate::common::cap_threads();
    let ck = tiny_checkpoint();
    let mut cfg = ck.config.clone();
    // The oracle differentiates the true derivative; the head-gradient scale is a deliberate departure from it.
    cfg.head_grad = 1.0;
    let mut params: Params<f64> = params_from_checkpoint(&ck);
    // Trained-model scale rather than the noise fixture's: keeps the 64-token sequence away from saturation.
    for v in params.values_mut().flatten() {
        *v *= 0.5;
    }
    let (_, grads) = loss_and_grads(&cfg, &params);
    let manifest = mirage::spec::Manifest::new(&cfg);
    let mut worst = 0f64;
    let mut worst_name = String::new();
    let mut unreached = Vec::new();
    let mut unresolvable = Vec::new();
    for spec in manifest.iter().filter(|s| !s.training_only) {
        let g = &grads[&spec.name];
        if g.iter().all(|v| *v == 0.0) {
            unreached.push(spec.name.clone());
            continue;
        }
        // One random direction per tensor.
        let dir = noise(g.len(), 900 + spec.numel() as u64, 1.0);
        let analytic: f64 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        // Richardson extrapolation of two central differences (step h and h/2): rounding stays at the h/2 level
        // (|loss| * 1e-16 / h, far below the smallest gradients) and the truncation error drops to O(h^4).
        let h = 2e-4;
        let shifted = |sign: f64, step: f64| {
            let mut p = params.clone();
            p.get_mut(&spec.name).unwrap().iter_mut().zip(&dir).for_each(|(v, d)| *v += sign * step * d);
            loss_and_grads(&cfg, &p).0
        };
        let central = |step: f64| (shifted(1.0, step) - shifted(-1.0, step)) / (2.0 * step);
        let fd = (4.0 * central(h / 2.0) - central(h)) / 3.0;
        // Below this a finite difference of the loss cannot resolve a derivative at all.
        if analytic.abs().max(fd.abs()) < 1e-9 {
            unresolvable.push(spec.name.clone());
            continue;
        }
        let err = (analytic - fd).abs() / analytic.abs().max(fd.abs());
        if err > 1e-4 {
            eprintln!("  {}: analytic {analytic:.6e} fd {fd:.6e} err {err:.2e}", spec.name);
        }
        if err > worst {
            worst = err;
            worst_name = spec.name.clone();
        }
    }
    eprintln!("model finite-difference check: worst relative error {worst:.3e} at `{worst_name}`; unresolvable: {unresolvable:?}");
    // The residual routing parameters' gradient vanishes after 20 Sinkhorn rounds (a doubly stochastic projection absorbs
    // the perturbation); the Sinkhorn primitive itself is held by its own finite-difference check.
    assert_eq!(unresolvable, ["stack/mhc_a_res", "stack/mhc_b_res", "stack/mhc_phi_res"], "only the residual routing parameters are below the finite-difference floor");
    assert!(worst < 1e-3, "the model backward deviates by {worst:e} at `{worst_name}`");
    // Every encoder tensor receives gradient from the heads; the tensors left out are exactly the ones the
    // composed loss cannot reach (the mask coefficient and reader tensors are reached, MTP heads are training-only).
    assert!(unreached.is_empty(), "tensors that received no gradient: {unreached:?}");
}
