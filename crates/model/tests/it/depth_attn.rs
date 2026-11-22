// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements depth-attention residual streams for its clients. If your team needs expertise in
// training decoders that attend over their own earlier layers then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec of depth attention over a retained-state store (`model::depth_attn`, `model::saved::RetainedStates`):
//!
//! * the forward equals the host reference (a softmax over RMS-normed states, mixing the normed states);
//! * the backward equals per-element central differences of an f64 forward that shares no code with the kernels,
//!   for every retained state, the partial and every query;
//! * the gradient of a retained state read by two consumers is the sum of what each contributes, and the producer
//!   collects it with `grad_add_step`;
//! * against the Inkling 2 reference golden, every recorded attention input and the final readout are reproduced
//!   from the recorded block states and the checkpoint's queries.

use std::collections::HashMap;

use data::rng::{random_seed, Lcg};
use gpu_core::{DeviceBuffer, Gpu};
use model::depth_attn::{DepthAttn, DepthAttnIds};
use model::saved::RetainedStates;

fn pipes() -> &'static [(&'static str, &'static str)] {
    static P: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    P.get_or_init(|| model::depth_attn::kernels().iter().chain(model::saved::retained_kernels()).copied().collect())
}

const EPS: f32 = 1e-6;

/// Perturbs one input of the f64 oracle (the stack or the queries) by the given step.
type Perturb<'a> = &'a mut dyn FnMut(&mut Vec<Vec<f64>>, &mut Vec<f64>, f64);

/// f64 oracle: the loss `sum(g * out)` of depth attention over `stack` (each `[rows, d]`) and `queries` (`[nq, d]`).
fn loss64(stack: &[Vec<f64>], queries: &[f64], g: &[f64], rows: usize, d: usize, nq: usize) -> f64 {
    let scale = (d as f64).powf(-0.5);
    let normed: Vec<Vec<f64>> = stack
        .iter()
        .map(|s| {
            let mut n = s.clone();
            for r in 0..rows {
                let row = &mut n[r * d..(r + 1) * d];
                let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / d as f64 + EPS as f64).sqrt();
                row.iter_mut().for_each(|v| *v *= inv);
            }
            n
        })
        .collect();
    let mut loss = 0.0;
    for q in 0..nq {
        for r in 0..rows {
            let z: Vec<f64> = normed.iter().map(|n| scale * (0..d).map(|j| n[r * d + j] * queries[q * d + j]).sum::<f64>()).collect();
            let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
            let sum: f64 = e.iter().sum();
            for j in 0..d {
                let o: f64 = normed.iter().zip(&e).map(|(n, ei)| ei / sum * n[r * d + j]).sum();
                loss += g[(q * rows + r) * d + j] * o;
            }
        }
    }
    loss
}

struct Case {
    rows: usize,
    d: usize,
    capacity: usize,
    store: Vec<f32>,
    partial: Vec<f32>,
}

impl Case {
    fn new(rng: &mut Lcg, rows: usize, d: usize, capacity: usize) -> Case {
        // Per-row offsets so the norms differ between states and rows.
        let mut store = rng.vec_scaled(capacity * rows * d, 1.5);
        for (i, v) in store.iter_mut().enumerate() {
            *v += 0.3 * ((i / d) % 5) as f32 - 0.6;
        }
        Case { rows, d, capacity, store, partial: rng.vec_scaled(rows * d, 2.0) }
    }

    fn state(&self, slot: usize) -> &[f32] {
        &self.store[slot * self.rows * self.d..(slot + 1) * self.rows * self.d]
    }

    fn upload(&self, g: &Gpu) -> (RetainedStates, DeviceBuffer) {
        let store = RetainedStates::new(g, self.capacity as u32, self.rows as u32, self.d as u32, true).expect("store");
        g.write_f32(store.values(), &self.store);
        g.write_f32(store.grads().unwrap(), &vec![0.0; self.store.len()]);
        (store, g.storage_init("partial", &self.partial))
    }
}

/// Run one consumer's forward and backward; returns (out, dpartial, dquery).
#[allow(clippy::too_many_arguments)]
fn run(g: &Gpu, store: &RetainedStates, partial: &DeviceBuffer, slots: &[u32], queries: &[f32], nq: u32, dout: &[f32], dpartial: &DeviceBuffer) -> (Vec<f32>, Vec<f32>) {
    let ids = DepthAttnIds::resolve(g).unwrap();
    let da = DepthAttn::new(store, slots, nq, EPS).unwrap();
    let query = g.storage_init("query", queries);
    let out = g.storage(da.out_len());
    let stats = g.storage(da.stats_len());
    let dz = g.storage(da.dz_len());
    let doutb = g.storage_init("dout", dout);
    let dquery = g.storage_init("dquery", &vec![0.0; queries.len()]);
    let mut steps = vec![da.forward(g, &ids, store, partial, &query, &out, &stats)];
    steps.extend(da.backward(g, &ids, store, partial, &query, &stats, &doutb, dpartial, &dz, &dquery));
    g.submit(&[], &steps);
    (g.read(&out, da.out_len() as usize), g.read(&dquery, queries.len()))
}

#[test]
fn depth_attention_matches_the_host_reference_and_its_backward_matches_finite_differences() {
    let g = gpu_core::testgpu::dev(pipes());
    let mut rng = Lcg::new(random_seed());
    let (rows, d, nq) = (3usize, 8usize, 3usize);
    let case = Case::new(&mut rng, rows, d, 5);
    let slots = [3u32, 0, 2];
    let queries = rng.vec_scaled(nq * d, 2.0);
    let dout = rng.vec_scaled(nq * rows * d, 1.0);
    let (store, partial) = case.upload(&g);
    let dpartial = g.storage_init("dpartial", &vec![0.0; rows * d]);
    let (out, dquery) = run(&g, &store, &partial, &slots, &queries, nq as u32, &dout, &dpartial);

    let stack: Vec<&[f32]> = slots.iter().map(|s| case.state(*s as usize)).chain([case.partial.as_slice()]).collect();
    let want = model::hostmath::depth_attention(&stack, rows, d, &queries, nq, EPS);
    for (i, (a, b)) in out.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-5, "out[{i}] {a} vs host {b}");
    }

    let to64 = |v: &[f32]| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
    let stack64: Vec<Vec<f64>> = stack.iter().map(|s| to64(s)).collect();
    let (q64, g64) = (to64(&queries), to64(&dout));
    let fd = |stack: &[Vec<f64>], q: &[f64], which: Perturb| -> f64 {
        let h = 1e-5;
        let (mut sp, mut qp) = (stack.to_vec(), q.to_vec());
        which(&mut sp, &mut qp, h);
        let plus = loss64(&sp, &qp, &g64, rows, d, nq);
        let (mut sm, mut qm) = (stack.to_vec(), q.to_vec());
        which(&mut sm, &mut qm, -h);
        (plus - loss64(&sm, &qm, &g64, rows, d, nq)) / (2.0 * h)
    };
    let close = |got: f32, want: f64, what: &str| assert!((got as f64 - want).abs() < 2e-4 * (1.0 + want.abs()), "{what}: kernel {got} vs finite difference {want}");
    for (k, slot) in slots.iter().enumerate() {
        let grad = store.read_grad(&g, *slot);
        for e in 0..rows * d {
            close(grad[e], fd(&stack64, &q64, &mut |s, _, h| s[k][e] += h), &format!("d store slot {slot} [{e}]"));
        }
    }
    let untouched = store.read_grad(&g, 1);
    assert!(untouched.iter().all(|v| *v == 0.0), "a slot outside the stack received a gradient");
    let dpart = g.read(&dpartial, rows * d);
    for e in 0..rows * d {
        close(dpart[e], fd(&stack64, &q64, &mut |s, _, h| s[slots.len()][e] += h), &format!("d partial [{e}]"));
    }
    for e in 0..nq * d {
        close(dquery[e], fd(&stack64, &q64, &mut |_, q, h| q[e] += h), &format!("d query [{e}]"));
    }
}

/// Spec: with per-row queries (`[nq, rows, width]`, query `q` of row `r` at `(q * rows + r) * width`), every row is
/// mixed by its own queries: row `r` of the output is the host reference's row `r` under row `r`'s query set, as when
/// every row of a batch runs under its own adapter's queries. The backward takes shared queries only and refuses.
#[test]
fn per_row_queries_mix_each_row_with_its_own_queries() {
    let g = gpu_core::testgpu::dev(pipes());
    let mut rng = Lcg::new(random_seed());
    let (rows, d, nq) = (4usize, 8usize, 3usize);
    let case = Case::new(&mut rng, rows, d, 4);
    let slots = [2u32, 0];
    let per_row = rng.vec_scaled(nq * rows * d, 2.0);
    let (store, partial) = case.upload(&g);
    let ids = DepthAttnIds::resolve(&g).unwrap();
    let da = DepthAttn::new(&store, &slots, nq as u32, EPS).unwrap().with_row_queries();
    let query = g.storage_init("query", &per_row);
    let (out, stats) = (g.storage(da.out_len()), g.storage(da.stats_len()));
    g.submit(&[], &[da.forward(&g, &ids, &store, &partial, &query, &out, &stats)]);
    let out = g.read(&out, da.out_len() as usize);
    let stack: Vec<&[f32]> = slots.iter().map(|s| case.state(*s as usize)).chain([case.partial.as_slice()]).collect();
    for r in 0..rows {
        let own: Vec<f32> = (0..nq).flat_map(|q| per_row[(q * rows + r) * d..(q * rows + r + 1) * d].iter().copied()).collect();
        let want = model::hostmath::depth_attention(&stack, rows, d, &own, nq, EPS);
        for q in 0..nq {
            for j in 0..d {
                let (a, b) = (out[(q * rows + r) * d + j], want[(q * rows + r) * d + j]);
                assert!((a - b).abs() < 1e-5, "row {r} query {q} [{j}]: {a} vs host {b}");
            }
        }
    }
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dz = g.storage(da.dz_len());
        da.backward(&g, &ids, &store, &partial, &query, &stats, &out_buf(&g, &da), &g.storage((rows * d) as u64), &dz, &g.storage((nq * d) as u64))
    }));
    assert!(refused.is_err(), "the backward refuses per-row queries");
}

fn out_buf(g: &Gpu, da: &DepthAttn) -> DeviceBuffer {
    g.storage(da.out_len())
}

#[test]
fn a_retained_state_read_by_two_consumers_receives_the_sum_of_their_gradients() {
    let g = gpu_core::testgpu::dev(pipes());
    let mut rng = Lcg::new(random_seed());
    let (rows, d) = (2usize, 12usize);
    let case = Case::new(&mut rng, rows, d, 4);
    let (qa, qb) = (rng.vec_scaled(2 * d, 1.5), rng.vec_scaled(d, 1.5));
    let (da, db) = (rng.vec_scaled(2 * rows * d, 1.0), rng.vec_scaled(rows * d, 1.0));
    let shared = 2u32;
    let grad_of = |consumers: &[bool; 2]| {
        let (store, partial) = case.upload(&g);
        let dpartial = g.storage_init("dpartial", &vec![0.0; rows * d]);
        if consumers[0] {
            run(&g, &store, &partial, &[0, shared], &qa, 2, &da, &dpartial);
        }
        if consumers[1] {
            run(&g, &store, &partial, &[shared, 3], &qb, 1, &db, &dpartial);
        }
        // The producer collects the accumulated gradient into its own (here: a buffer of ones).
        let dst = g.storage_init("dst", &vec![1.0; rows * d]);
        g.submit(&[], &[store.grad_add_step(&g, shared, &dst)]);
        (store.read_grad(&g, shared), g.read(&dst, rows * d))
    };
    let (a, _) = grad_of(&[true, false]);
    let (b, _) = grad_of(&[false, true]);
    let (both, collected) = grad_of(&[true, true]);
    assert!(a.iter().any(|v| *v != 0.0) && b.iter().any(|v| *v != 0.0), "each consumer contributes a gradient");
    for i in 0..rows * d {
        assert!((both[i] - (a[i] + b[i])).abs() < 1e-6, "slot gradient [{i}] {} is not the sum {} + {}", both[i], a[i], b[i]);
        assert!((collected[i] - (1.0 + both[i])).abs() < 1e-6, "grad_add_step [{i}]");
    }
}

#[test]
fn a_stack_is_refused_when_it_names_a_slot_twice_or_one_the_store_lacks() {
    let g = gpu_core::testgpu::dev(pipes());
    let store = RetainedStates::new(&g, 3, 2, 4, false).unwrap();
    assert!(DepthAttn::new(&store, &[0, 0], 1, EPS).unwrap_err().contains("twice"));
    assert!(DepthAttn::new(&store, &[3], 1, EPS).unwrap_err().contains("slot 3"));
    assert!(DepthAttn::new(&store, &[0], 5, EPS).unwrap_err().contains("queries"));
    assert!(DepthAttn::new(&store, &[], 1, EPS).is_ok(), "a stack of the partial alone is a plain residual");
}

fn golden(dir: &std::path::Path, file: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let path = dir.join(file);
    checkpoint::safetensors::read(&path.to_string_lossy()).expect("read golden").into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

/// The reference's `Inkling2BlockAttentionResidual`: every visit's attention input and the final readout, reproduced
/// from the recorded block states and partials with the checkpoint's own (published-name) queries.
#[test]
fn inkling2_tiny_golden_attention_inputs_are_reproduced() {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if !dir.join("golden.safetensors").exists() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let gold = golden(&dir, "golden.safetensors");
    let ckpt = golden(&dir.join("checkpoint"), "model.safetensors");
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let g = gpu_core::testgpu::dev(pipes());
    let ids = DepthAttnIds::resolve(&g).unwrap();
    let rows = gold["tokens"].1.len();
    let d = gold["embed"].0[1];
    let check = |blocks: &[f32], partial: &[f32], query: &[f32], want: &[f32], what: &str| {
        let n_blocks = blocks.len() / (rows * d);
        let nq = query.len() / d;
        let store = RetainedStates::new(&g, n_blocks.max(1) as u32, rows as u32, d as u32, false).unwrap();
        if n_blocks > 0 {
            g.write_f32(store.values(), blocks);
        }
        let slots: Vec<u32> = (0..n_blocks as u32).collect();
        let da = DepthAttn::new(&store, &slots, nq as u32, EPS).unwrap();
        let (pb, qb) = (g.storage_init("partial", partial), g.storage_init("query", query));
        let (out, stats) = (g.storage(da.out_len()), g.storage(da.stats_len()));
        g.submit(&[], &[da.forward(&g, &ids, &store, &pb, &qb, &out, &stats)]);
        let got = g.read(&out, da.out_len() as usize);
        let err = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-5, "{what}: max |d| {err} against the reference");
    };
    let mut checked = 0;
    for v in manifest["visits"].as_array().unwrap() {
        let (visit, layer) = (v["visit"].as_u64().unwrap(), v["layer"].as_u64().unwrap());
        if layer == 0 {
            continue;
        }
        let query = &ckpt[&format!("model.llm.attn_res.attn_queries.{}", layer - 1)].1;
        let want = &gold[&format!("visit{visit}.attn_in")].1;
        check(&gold[&format!("visit{visit}.block_states")].1, &gold[&format!("visit{visit}.in")].1, query, want, &format!("visit {visit} (layer {layer}) attention input"));
        checked += 1;
    }
    let last = manifest["visits"].as_array().unwrap().len() - 1;
    check(&gold[&format!("visit{last}.block_states")].1, &gold[&format!("visit{last}.out")].1, &ckpt["model.llm.attn_res.output_query"].1, &gold["final_in"].1, "final readout");
    assert!(checked >= 8, "only {checked} visits checked");
}
