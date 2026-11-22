// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-pass (looped) decoders for its clients. If your team needs expertise in
// training the gates that carry one pass's messages into the next then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec of the message gate `x' = (1 - tanh(Wf h)) x + tanh(Wg h) m`, `h = Wd [rms(x); rms(m)]` (`model::message_gate`):
//!
//! * every input and weight gradient equals per-element central differences of an f64 forward that shares no code
//!   with the composed kernels, and the forward equals that oracle;
//! * the weight gradients accumulate over calls (a looped layer's gate runs once per pass);
//! * against the Inkling 2 reference golden, every recorded q/k/v/mlp gate output of every looped visit is
//!   reproduced from the recorded inputs and messages with the checkpoint's own `controllers.*` weights.

use std::collections::HashMap;

use data::rng::{random_seed, Lcg};
use gpu_core::provider::library::GemmPrecision;
use gpu_core::{DeviceBuffer, Gpu};
use model::message_gate::{MessageGate, MessageGateIds, MessageGateWeights};
use model::ops::Ops;

const EPS: f32 = 1e-6;

fn pipes() -> &'static [(&'static str, &'static str)] {
    static P: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut v: Vec<(&str, &str)> = model::ops::kernel_list().to_vec();
        for k in model::message_gate::kernels().iter().chain(model::adapter::merged::KERNELS) {
            if !v.iter().any(|(n, _)| *n == k.0) {
                v.push(*k);
            }
        }
        v
    })
}

fn ops(g: &Gpu) -> Ops {
    let mut ops = Ops::new(g.share()).expect("Ops::new");
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    ops
}

/// The f64 oracle: `[rows, d]` output of the gate.
fn gate64(x: &[f64], m: &[f64], wd: &[f64], wf: &[f64], wg: &[f64], rows: usize, d: usize, k: usize) -> Vec<f64> {
    let norm = |v: &[f64]| {
        let inv = 1.0 / (v.iter().map(|a| a * a).sum::<f64>() / d as f64 + EPS as f64).sqrt();
        v.iter().map(|a| a * inv).collect::<Vec<f64>>()
    };
    let mut y = vec![0.0; rows * d];
    for r in 0..rows {
        let (xr, mr) = (&x[r * d..(r + 1) * d], &m[r * d..(r + 1) * d]);
        let cat: Vec<f64> = norm(xr).into_iter().chain(norm(mr)).collect();
        let h: Vec<f64> = (0..k).map(|i| (0..2 * d).map(|j| wd[i * 2 * d + j] * cat[j]).sum()).collect();
        for c in 0..d {
            let f = (0..k).map(|i| wf[c * k + i] * h[i]).sum::<f64>().tanh();
            let gg = (0..k).map(|i| wg[c * k + i] * h[i]).sum::<f64>().tanh();
            y[r * d + c] = (1.0 - f) * xr[c] + gg * mr[c];
        }
    }
    y
}

struct Run {
    y: Vec<f32>,
    dx: Vec<f32>,
    dm: Vec<f32>,
    dwd: Vec<f32>,
    dwf: Vec<f32>,
    dwg: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run(g: &Gpu, gate: &MessageGate, x: &[f32], m: &[f32], wd: &[f32], wf: &[f32], wg: &[f32], dy: &[f32], calls: usize) -> Run {
    let ids = MessageGateIds::resolve(g).unwrap();
    let ops = ops(g);
    let b = |name: &str, v: &[f32]| g.storage_init(name, v);
    let (xb, mb, dyb) = (b("x", x), b("m", m), b("dy", dy));
    let (wdb, wfb, wgb) = (b("wd", wd), b("wf", wf), b("wg", wg));
    let (gd, gf, gg) = (b("gd", &vec![0.0; wd.len()]), b("gf", &vec![0.0; wf.len()]), b("gg", &vec![0.0; wg.len()]));
    let n = x.len();
    let (y, dx, dm): (DeviceBuffer, DeviceBuffer, DeviceBuffer) = (g.storage(n as u64), g.storage(n as u64), g.storage(n as u64));
    let w = MessageGateWeights { down: &wdb, forget: &wfb, gain: &wgb };
    let grads = MessageGateWeights { down: &gd, forget: &gf, gain: &gg };
    let (acts, scratch) = (gate.acts(g), gate.scratch(g));
    for _ in 0..calls {
        let mut s = Vec::new();
        gate.forward(&mut s, g, &ids, &ops, &w, &xb, &mb, &acts, &y);
        gate.backward(&mut s, g, &ids, &ops, &w, &xb, &mb, &acts, &dyb, &dx, &dm, &grads, &scratch);
        g.submit(&[], &s);
    }
    Run { y: g.read(&y, n), dx: g.read(&dx, n), dm: g.read(&dm, n), dwd: g.read(&gd, wd.len()), dwf: g.read(&gf, wf.len()), dwg: g.read(&gg, wg.len()) }
}

#[test]
fn message_gate_gradients_match_finite_differences_and_accumulate_over_calls() {
    let g = gpu_core::testgpu::dev(pipes());
    let mut rng = Lcg::new(random_seed());
    let (rows, d, k) = (3usize, 6usize, 3usize);
    let gate = MessageGate::new(rows as u32, d as u32, k as u32, EPS).unwrap();
    let x = rng.vec_scaled(rows * d, 2.0);
    let m = rng.vec_scaled(rows * d, 2.0);
    let wd = rng.vec_scaled(k * 2 * d, 0.8);
    let wf = rng.vec_scaled(d * k, 0.8);
    let wg = rng.vec_scaled(d * k, 0.8);
    let dy = rng.vec_scaled(rows * d, 1.0);
    let got = run(&g, &gate, &x, &m, &wd, &wf, &wg, &dy, 1);

    let to64 = |v: &[f32]| v.iter().map(|a| *a as f64).collect::<Vec<f64>>();
    let mut ins = [to64(&x), to64(&m), to64(&wd), to64(&wf), to64(&wg)];
    let dy64 = to64(&dy);
    let loss = |ins: &[Vec<f64>; 5]| gate64(&ins[0], &ins[1], &ins[2], &ins[3], &ins[4], rows, d, k).iter().zip(&dy64).map(|(a, b)| a * b).sum::<f64>();
    let y64 = gate64(&ins[0], &ins[1], &ins[2], &ins[3], &ins[4], rows, d, k);
    for (i, (a, b)) in got.y.iter().zip(&y64).enumerate() {
        assert!((*a as f64 - b).abs() < 1e-5, "y[{i}] {a} vs oracle {b}");
    }
    let names = ["dx", "dm", "d down", "d forget", "d gain"];
    let grads = [&got.dx, &got.dm, &got.dwd, &got.dwf, &got.dwg];
    for t in 0..5 {
        for e in 0..ins[t].len() {
            let h = 1e-5;
            let keep = ins[t][e];
            ins[t][e] = keep + h;
            let plus = loss(&ins);
            ins[t][e] = keep - h;
            let minus = loss(&ins);
            ins[t][e] = keep;
            let fd = (plus - minus) / (2.0 * h);
            let a = grads[t][e] as f64;
            assert!((a - fd).abs() < 2e-4 * (1.0 + fd.abs()), "{}[{e}]: composed {a} vs finite difference {fd}", names[t]);
        }
    }

    let twice = run(&g, &gate, &x, &m, &wd, &wf, &wg, &dy, 2);
    for (once, two, what) in [(&got.dwd, &twice.dwd, "down"), (&got.dwf, &twice.dwf, "forget"), (&got.dwg, &twice.dwg, "gain")] {
        for (a, b) in once.iter().zip(two.iter()) {
            assert!((2.0 * a - b).abs() < 1e-5 * (1.0 + a.abs()), "{what}: two calls give {b}, one {a}: the weight gradient must accumulate");
        }
    }
    assert_eq!(got.dx, twice.dx, "the input gradient is written, not accumulated");
}

fn golden(path: &std::path::Path) -> HashMap<String, Vec<f32>> {
    checkpoint::safetensors::read(&path.to_string_lossy()).expect("read golden").into_iter().map(|t| (t.name, t.data)).collect()
}

/// The reference's `Inkling2LoopMessageGate` at every looped visit: q, k, v and mlp gates from the recorded
/// attention / MLP inputs and messages.
#[test]
fn inkling2_tiny_golden_message_gates_are_reproduced() {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if !dir.join("golden.safetensors").exists() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let gold = golden(&dir.join("golden.safetensors"));
    let ckpt = golden(&dir.join("checkpoint/model.safetensors"));
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let rows = gold["tokens"].len();
    let d = gold["embed"].len() / rows;
    let g = gpu_core::testgpu::dev(pipes());
    let ids = MessageGateIds::resolve(&g).unwrap();
    let ops = ops(&g);
    let mut checked = 0;
    for v in manifest["visits"].as_array().unwrap() {
        if v["segment"] != "core" || v["pass"].as_u64().unwrap() == 0 {
            continue;
        }
        let (visit, layer, full) = (v["visit"].as_u64().unwrap(), v["layer"].as_u64().unwrap(), v["kind"] == "full");
        let attn_in = &gold[&format!("visit{visit}.attn_in")];
        let messages = &gold[&format!("visit{visit}.messages")];
        let slice = |t: &[f32], i: usize| t[i * rows * d..(i + 1) * rows * d].to_vec();
        // (controller, gated tensor, input, message)
        let cases = [
            ("attn", "q", slice(attn_in, 0), slice(messages, 0)),
            ("k", "k", slice(attn_in, if full { 1 } else { 0 }), slice(messages, 1)),
            ("v", "v", slice(attn_in, if full { 2 } else { 0 }), slice(messages, 2)),
            ("mlp", "mlp", gold[&format!("visit{visit}.mlp_in")].clone(), slice(messages, 3)),
        ];
        for (ctl, tag, x, m) in cases {
            let w = |leaf: &str| g.storage_init(leaf, &ckpt[&format!("model.llm.layers.{layer}.controllers.{ctl}.{leaf}")]);
            let (down, forget, gain) = (w("down_weight"), w("forget_weight"), w("gain_weight"));
            let rank = (ckpt[&format!("model.llm.layers.{layer}.controllers.{ctl}.down_weight")].len() / (2 * d)) as u32;
            let gate = MessageGate::new(rows as u32, d as u32, rank, EPS).unwrap();
            let (xb, mb, y) = (g.storage_init("x", &x), g.storage_init("m", &m), g.storage((rows * d) as u64));
            let acts = gate.acts(&g);
            let mut s = Vec::new();
            gate.forward(&mut s, &g, &ids, &ops, &MessageGateWeights { down: &down, forget: &forget, gain: &gain }, &xb, &mb, &acts, &y);
            g.submit(&[], &s);
            let got = g.read(&y, rows * d);
            let want = &gold[&format!("visit{visit}.gated_{tag}")];
            let err = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-5, "visit {visit} (layer {layer}) {tag} gate: max |d| {err} against the reference");
            checked += 1;
        }
    }
    assert!(checked >= 16, "only {checked} gates checked");
}

/// Spec: `forward_corrected` hands each product's input and output to the caller right after the product, so a caller
/// that adds `x dW^T` to each (an adapter's correction of `Wd`, `Wf` and `Wg`) gets the gate of the corrected weights:
/// the f64 oracle over `W + dW`, each of the three corrected.
#[test]
fn a_corrected_forward_is_the_gate_of_the_corrected_weights() {
    use model::message_gate::GateProduct;
    use model::ops::Weight;
    let g = gpu_core::testgpu::dev(pipes());
    let mut rng = Lcg::new(random_seed());
    let (rows, d, k) = (3usize, 8usize, 4usize);
    let gate = MessageGate::new(rows as u32, d as u32, k as u32, EPS).unwrap();
    let (x, m) = (rng.vec_scaled(rows * d, 1.0), rng.vec_scaled(rows * d, 1.0));
    let (wd, wf, wg) = (rng.vec_scaled(k * 2 * d, 0.3), rng.vec_scaled(d * k, 0.3), rng.vec_scaled(d * k, 0.3));
    let (dd, df, dg) = (rng.vec_scaled(k * 2 * d, 0.2), rng.vec_scaled(d * k, 0.2), rng.vec_scaled(d * k, 0.2));
    let ops = ops(&g);
    let ids = MessageGateIds::resolve(&g).unwrap();
    let axpy = g.kernel_index("axpy").expect("axpy is registered");
    let b = |name: &str, v: &[f32]| g.storage_init(name, v);
    let (xb, mb, wdb, wfb, wgb) = (b("x", &x), b("m", &m), b("wd", &wd), b("wf", &wf), b("wg", &wg));
    let deltas = [(GateProduct::Down, Weight::dense_f32(b("dd", &dd), k as u32, 2 * d as u32)), (GateProduct::Forget, Weight::dense_f32(b("df", &df), d as u32, k as u32)), (GateProduct::Gain, Weight::dense_f32(b("dg", &dg), d as u32, k as u32))];
    let acts = gate.acts(&g);
    let y = g.storage((rows * d) as u64);
    let mut s = Vec::new();
    let mut seen = Vec::new();
    gate.forward_corrected(&mut s, &g, &ids, &ops, &MessageGateWeights { down: &wdb, forget: &wfb, gain: &wgb }, &xb, &mb, &acts, &y, &mut |s, product, input, output| {
        seen.push(product);
        let w = &deltas.iter().find(|(p, _)| *p == product).unwrap().1;
        let n = rows as u32 * w.n();
        let tmp = g.storage(u64::from(n));
        ops.matmul(s, w, &ops.act_f32(input, 0, rows as u32, w.k()), &tmp, 0);
        s.push(g.step(axpy, &[output, &tmp], &[n, 1.0f32.to_bits()], n));
    });
    g.submit(&[], &s);
    assert_eq!(seen, [GateProduct::Down, GateProduct::Forget, GateProduct::Gain], "each product once, in the forward's order");
    let to64 = |v: &[f32]| v.iter().map(|a| *a as f64).collect::<Vec<f64>>();
    let sum64 = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| *p as f64 + *q as f64).collect::<Vec<f64>>();
    let want = gate64(&to64(&x), &to64(&m), &sum64(&wd, &dd), &sum64(&wf, &df), &sum64(&wg, &dg), rows, d, k);
    for (i, (a, w)) in g.read(&y, rows * d).iter().zip(&want).enumerate() {
        assert!((*a as f64 - w).abs() < 1e-5, "y[{i}] {a} vs oracle {w}");
    }
}
