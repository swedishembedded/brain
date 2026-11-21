// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The routing-replay gate: a tiny mixture-of-experts language model served through the generic scheduler records the
//! expert selection of every token it computes, the completions carry it, and a trainer that computes the same model
//! with a different reduction order reproduces the served log-probabilities only when it replays that routing.
//!
//! The serving engine sums the router's dot products last-to-first, the trainer first-to-last. The first router's
//! weights for one token are a near-tie between two experts, found so that the two orders select different experts
//! there. Without replay the trainer runs other experts than the engine did at that token and its log-probability of
//! the next token is far from the recorded one; with replay every token's log-probability equals the engine's to
//! rounding (the weights still come from the trainer's own logits, which differ from the engine's in the last bits).
//!
//! Swedish Embedded AB implements solutions for on-policy reinforcement learning on mixture-of-experts models served by
//! a separate inference engine for its clients. If your team needs expertise in serve/train numerics then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::sync::Arc;

use data::rng::{random_seed, Lcg, Rng};
use gpu_core::Gpu;
use model::moe::{router_fwd_kind, router_fwd_replay, MoeIds, MoeShape, RouterKind, RouterOverride, ROUTER_REPLAY_KERNEL};
use model::paged::{BlockAllocator, BlockTable};
use model::rollout::{Rollout, RolloutParams, SchedulerRollout};
use model::routing::{RouterCallShape, RoutingReplay};
use model::serve::{PagedDecoder, SampleParams};

const PIPES: &[(&str, &str)] = &[("router_gate", kernels::ROUTER_GATE), ("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID), ROUTER_REPLAY_KERNEL];

const V: usize = 8;
const D: usize = 8;
const E: usize = 5;
const K: usize = 2;
const CALLS: usize = 2;
/// The token whose embedding is all ones: the near-tie of the first router is for it.
const TIE_TOKEN: u32 = 3;
/// Experts of the first router: `LEAD` wins clearly, `A` and `B` tie for the second slot.
const LEAD: usize = 0;
const A: usize = 1;
const B: usize = 2;

#[derive(Clone, Copy, PartialEq)]
enum Order {
    FirstToLast,
    LastToFirst,
}

fn dot(a: &[f32], b: &[f32], order: Order) -> f32 {
    let mut s = 0.0f32;
    match order {
        Order::FirstToLast => a.iter().zip(b).for_each(|(x, y)| s += x * y),
        Order::LastToFirst => a.iter().zip(b).rev().for_each(|(x, y)| s += x * y),
    }
    s
}

struct Weights {
    embed: Vec<f32>,
    router: Vec<Vec<f32>>,
    experts: Vec<Vec<f32>>,
    head: Vec<f32>,
}

/// The tiny model: per token, `h = embed[tok]`, then per MoE call `h += sum_e gate_e(h) * W_e h` with the gate from
/// the device router, then `log_softmax(head h)`. Position-wise (no attention), which is all a routing gate needs.
struct TinyMoe {
    g: Gpu,
    kind: RouterKind,
    w: Arc<Weights>,
    order: Order,
}

/// What one forward over `tokens` produced: the next-token log-probabilities per position, and per call the experts
/// each position selected (`[positions * K]`, ascending within a position).
struct Forward {
    logprobs: Vec<Vec<f32>>,
    selected: Vec<Vec<u32>>,
}

impl TinyMoe {
    fn shapes() -> Vec<RouterCallShape> {
        vec![RouterCallShape { n_experts: E as u32, top_k: K as u32 }; CALLS]
    }

    fn router_ids(&self) -> MoeIds {
        let gate = if matches!(self.kind, RouterKind::SigmoidNoAuxTc { .. }) { "router_gate_sigmoid" } else { "router_gate" };
        MoeIds { router_gate: self.g.kernel_index(gate).expect("registered"), linear_gated: 0, silu_mul: 0, scale_add: 0 }
    }

    /// The device gate of `logits` (`[rows, E]`), forced to `replay`'s call `call` when given.
    fn gate(&self, logits: &[f32], call: usize, replay: Option<&RoutingReplay>) -> Vec<f32> {
        let g = &self.g;
        let rows = logits.len() / E;
        let shape = MoeShape { rows: rows as u32, d_model: D as u32, moe_ff: D as u32, n_experts: E as u32, top_k: K as u32 };
        let (lb, bias) = (g.storage_init("router_logits", logits), g.storage_init("bias", &[0.0; E]));
        let (gate, probs) = (g.storage((rows * E) as u64), g.storage((rows * E) as u64));
        let mut steps = vec![router_fwd_kind(g, &self.router_ids(), self.kind, &shape, &lb, Some(&bias), &gate, Some(&probs))];
        let forced = replay.map(|r| r.upload_call(g, call, K as u32));
        if let Some(experts) = &forced {
            let over = RouterOverride { kernel: g.kernel_index(ROUTER_REPLAY_KERNEL.0).expect("registered"), experts };
            steps.push(router_fwd_replay(g, over, self.kind, &shape, &lb, Some(&bias), &gate));
        }
        g.submit(&[], &steps);
        g.read(&gate, rows * E)
    }

    fn forward(&self, tokens: &[u32], replay: Option<&RoutingReplay>) -> Forward {
        let w = &self.w;
        let mut h: Vec<Vec<f32>> = tokens.iter().map(|&t| w.embed[t as usize * D..(t as usize + 1) * D].to_vec()).collect();
        let mut selected = Vec::new();
        for call in 0..CALLS {
            let logits: Vec<f32> = h.iter().flat_map(|x| (0..E).map(move |e| dot(x, &w.router[call][e * D..(e + 1) * D], self.order))).collect();
            let gate = self.gate(&logits, call, replay);
            let gr = &gate;
            selected.push((0..h.len()).flat_map(|r| (0..E).filter(move |&e| gr[r * E + e] != 0.0).map(|e| e as u32)).collect::<Vec<u32>>());
            for (r, x) in h.iter_mut().enumerate() {
                let mut y = vec![0.0f32; D];
                for e in (0..E).filter(|&e| gate[r * E + e] != 0.0) {
                    let m = &w.experts[call][e * D * D..(e + 1) * D * D];
                    for (i, yi) in y.iter_mut().enumerate() {
                        *yi += gate[r * E + e] * dot(&m[i * D..(i + 1) * D], x, Order::FirstToLast);
                    }
                }
                x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            }
        }
        let logprobs = h
            .iter()
            .map(|x| {
                let z: Vec<f32> = (0..V).map(|v| dot(&w.head[v * D..(v + 1) * D], x, Order::FirstToLast)).collect();
                let mx = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = mx + z.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
                z.iter().map(|v| v - lse).collect()
            })
            .collect();
        Forward { logprobs, selected }
    }
}

/// The serving engine over [`TinyMoe`], summing its router dot products last to first, recording its routing into
/// every table that carries a log.
struct Engine {
    m: TinyMoe,
    alloc: BlockAllocator,
    /// Whether the engine declares its MoE calls (records routing).
    records: bool,
}

impl Engine {
    /// Run `tokens` at positions `table.len()..` (the table is already extended) and record their routing.
    fn run(&self, table: &mut BlockTable, first: u32, tokens: &[u32]) -> Vec<Vec<f32>> {
        let f = self.m.forward(tokens, None);
        if let Some(log) = table.routing_log_mut() {
            for (call, ids) in f.selected.iter().enumerate() {
                for (i, row) in ids.as_chunks::<K>().0.iter().enumerate() {
                    log.record(call, first + i as u32, row);
                }
            }
        }
        f.logprobs
    }
}

impl PagedDecoder for Engine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        64
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        256
    }
    fn vocab(&self) -> usize {
        V
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(8)
    }
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        let first = table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        self.run(table, first, prompt).pop().unwrap()
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        hidden.to_vec()
    }
    fn forward_batched_greedy(&mut self, _: &mut [&mut BlockTable], _: &[u32]) -> Vec<u32> {
        unreachable!("every request samples")
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("every request samples")
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        None
    }
    fn kv_pool_bytes(&self) -> u64 {
        0
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        4096
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
        unreachable!("every request reads the whole row")
    }
    fn topk_capacity(&self) -> usize {
        4
    }
    fn full_row_readback(&self) -> bool {
        true
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        tables
            .iter_mut()
            .zip(inputs)
            .map(|(t, &tok)| {
                let (_, _) = t.append(&mut self.alloc).unwrap();
                let first = t.len() - 1;
                self.run(t, first, &[tok]).pop().unwrap()
            })
            .collect()
    }
    fn routing_calls(&self) -> Option<Vec<RouterCallShape>> {
        self.records.then(TinyMoe::shapes)
    }
}

/// Weights whose first router puts [`TIE_TOKEN`] on a near-tie between experts [`A`] and [`B`] that the two summation
/// orders break differently on the device router (`kind`): `B`'s router row is a permutation of `A`'s, so the two
/// logits are equal in exact arithmetic. Candidates are drawn until one flips; experts `A` and `B` map `h` to `h` and
/// `-h`, so the flip moves the next token's distribution.
fn weights_with_a_flipping_tie(g: &Gpu, kind: RouterKind, rng: &mut Lcg) -> Weights {
    const CANDIDATES: usize = 4096;
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(CANDIDATES);
    for _ in 0..CANDIDATES {
        // mixed magnitudes that cancel to a moderate logit, so the two orders round differently while the router's
        // scores stay away from saturation
        let mut a: Vec<f32> = (0..D).map(|_| rng.signed() * 10f32.powf(rng.scaled(1.0))).collect();
        let head: f32 = a[..D - 1].iter().sum();
        a[D - 1] = rng.scaled(2.0) - head;
        let mut perm: Vec<usize> = (0..D).collect();
        for i in (1..D).rev() {
            perm.swap(i, (rng.next_u32() as usize) % (i + 1));
        }
        let mut r = vec![0.0f32; E * D];
        let la: f32 = a.iter().sum();
        for i in 0..D {
            r[A * D + i] = a[i];
            r[B * D + i] = a[perm[i]];
            r[LEAD * D + i] = (la + 0.5) / D as f32;
            for e in (0..E).filter(|e| ![LEAD, A, B].contains(e)) {
                r[e * D + i] = (la - 3.0) / D as f32;
            }
        }
        rows.push(r);
    }
    let ones = vec![1.0f32; D];
    let probe = |order: Order| -> Vec<f32> {
        let ones = &ones;
        let logits: Vec<f32> = rows.iter().flat_map(|r| (0..E).map(move |e| dot(ones, &r[e * D..(e + 1) * D], order))).collect();
        let m = TinyMoe { g: g.share(), kind, w: Arc::new(Weights { embed: vec![], router: vec![], experts: vec![], head: vec![] }), order };
        m.gate(&logits, 0, None)
    };
    let (fwd, rev) = (probe(Order::FirstToLast), probe(Order::LastToFirst));
    let picks = |gate: &[f32], n: usize| (gate[n * E + A] != 0.0, gate[n * E + B] != 0.0);
    let n = (0..CANDIDATES).find(|&n| picks(&fwd, n) != picks(&rev, n)).expect("no candidate near-tie flips between the two summation orders");

    let mut embed = rng.vec_scaled(V * D, 1.0);
    embed[TIE_TOKEN as usize * D..(TIE_TOKEN as usize + 1) * D].fill(1.0);
    let mut experts = vec![rng.vec_scaled(E * D * D, 0.1), rng.vec_scaled(E * D * D, 0.1)];
    for (e, s) in [(A, 1.0f32), (B, -1.0)] {
        let m = &mut experts[0][e * D * D..(e + 1) * D * D];
        m.fill(0.0);
        (0..D).for_each(|i| m[i * D + i] = s);
    }
    Weights { embed, router: vec![rows.swap_remove(n), rng.vec_scaled(E * D, 1.0)], experts, head: rng.vec_scaled(V * D, 0.5) }
}

#[test]
fn replayed_routing_makes_the_trainer_reproduce_the_served_logprobs() {
    let mut rng = Lcg::new(random_seed());
    for kind in [
        RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: true, routed_scaling: 1.0 },
        RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: 2.5 },
    ] {
        let g = gpu_core::testgpu::dev(PIPES);
        let w = Arc::new(weights_with_a_flipping_tie(&g, kind, &mut rng));
        let engine = Engine { m: TinyMoe { g: g.share(), kind, w: w.clone(), order: Order::LastToFirst }, alloc: BlockAllocator::new(256, 8), records: true };
        let trainer = TinyMoe { g: g.share(), kind, w, order: Order::FirstToLast };

        let mut rollout = SchedulerRollout::new(engine, 4);
        rollout.record_routing(true).expect("the engine records routing");
        let prompt = [1, 5, TIE_TOKEN, 6, TIE_TOKEN];
        let params = RolloutParams { max_new: 6, sample: SampleParams { temp: 1.0, ..SampleParams::greedy() }, eos: None };
        let completions = rollout.sample_n(&prompt, 4, &params, &mut Rng::new(rng.next_u64()));

        let (mut gap_own, mut gap_replayed) = (0.0f32, 0.0f32);
        for c in &completions {
            let record = c.routing.clone().expect("a recording rollout carries the routing");
            let seq: Vec<u32> = prompt.iter().chain(&c.tokens).copied().collect();
            assert_eq!(record.positions as usize, seq.len());
            assert_eq!(record.policy_version, c.policy_version);
            let mut want = vec![true; seq.len()];
            *want.last_mut().unwrap() = false;
            assert_eq!(record.recorded, want, "every computed position is recorded; the last token is never fed back");
            record.validate(&TinyMoe::shapes(), Some(c.policy_version)).expect("the record fits the trainer");

            let replay = RoutingReplay { batch: 1, seq: seq.len() as u32, rows: vec![Some((Arc::new(record), 0))] };
            replay.validate(&TinyMoe::shapes()).unwrap();
            let own = trainer.forward(&seq, None);
            let replayed = trainer.forward(&seq, Some(&replay));
            for (i, (&tok, &served)) in c.tokens.iter().zip(&c.logprobs).enumerate() {
                let pos = prompt.len() - 1 + i;
                gap_own = gap_own.max((own.logprobs[pos][tok as usize] - served).abs());
                gap_replayed = gap_replayed.max((replayed.logprobs[pos][tok as usize] - served).abs());
            }
        }
        // The replayed gap is the rounding of a router whose dot products cancel from magnitudes up to 10 (logits differ
        // by ~1e-6 between the orders); the own-routing gap is a different expert's output.
        let name = if matches!(kind, RouterKind::SigmoidNoAuxTc { .. }) { "sigmoid" } else { "softmax" };
        assert!(gap_own > 5e-2, "{name}: the trainer's own routing must differ from the engine's at the near-tie (largest gap {gap_own})");
        assert!(gap_replayed < 1e-3, "{name}: with replayed routing the trainer reproduces the served log-probabilities (largest gap {gap_replayed})");
    }
}

#[test]
fn a_scheduler_refuses_to_record_routing_an_engine_does_not_record() {
    let g = gpu_core::testgpu::dev(PIPES);
    let kind = RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: true, routed_scaling: 1.0 };
    let w = Arc::new(Weights { embed: vec![], router: vec![], experts: vec![], head: vec![] });
    let engine = Engine { m: TinyMoe { g, kind, w, order: Order::FirstToLast }, alloc: BlockAllocator::new(16, 8), records: false };
    let mut rollout = SchedulerRollout::new(engine, 1);
    let err = rollout.record_routing(true).unwrap_err();
    assert!(err.contains("does not record its MoE routing"), "{err}");
    assert!(rollout.record_routing(false).is_ok(), "turning recording off is always possible");
}
