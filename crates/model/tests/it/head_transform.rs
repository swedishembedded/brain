// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the declared head transform (`model::head_transform`) on the serving side: a padded column of the head can
//! never be picked, drawn or given probability, however large its raw logit, and a soft cap bounds every served logit.
//! Covered on the device (greedy pick, top-k extraction, keyed draws, full rows) and on the host (served rows, log
//! probabilities, a decoder that declares the transform through the scheduler and the conformance property).
//!
//! Swedish Embedded AB implements solutions for serving and training language models whose output head is padded or
//! soft-capped for its clients. If your team needs expertise in LM-head numerics then you can procure our services by
//! sending an email to info@swedishembedded.com.

use data::rng::{random_seed, Lcg};
use model::head_transform::{HeadTransform, MASKED_LOGIT};
use model::paged::{BlockAllocator, BlockTable};
use model::readout::{argmax_rows, draw_rows, topk_rows, ReadoutKernels, DRAW_KERNELS, READOUT_KERNELS};
use model::serve::conformance::{a_declared_head_transform_is_honoured, Conformance};
use model::serve::{draw_key, draw_row, Drawn, PagedDecoder, Request, RowDraw, SampleParams, Scheduler};

const ROWS: usize = 6;
const VOCAB: usize = 40; // the head's rows, padding included
const VALID: u32 = 25;
const CAP: f32 = 5.0;

fn transform() -> HeadTransform {
    HeadTransform { softcap: Some(CAP), valid_vocab: Some(VALID) }
}

/// Logits whose padded columns are far above everything valid: unmasked, they win every readout.
fn raw_logits(rng: &mut Lcg) -> Vec<f32> {
    let mut v = rng.vec_scaled(ROWS * VOCAB, 4.0);
    for row in v.chunks_mut(VOCAB) {
        row[VALID as usize..].iter_mut().for_each(|x| *x += 60.0);
    }
    v
}

/// The registered kernels must outlive the device (`testgpu::dev` borrows them for `'static`).
fn kernels() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| READOUT_KERNELS.iter().chain(DRAW_KERNELS.iter()).chain(model::head_transform::KERNELS.iter()).copied().collect())
}

#[test]
fn padded_columns_are_never_picked_drawn_or_returned_on_the_device() {
    let g = gpu_core::testgpu::dev(kernels());
    let (t, rk) = (transform(), ReadoutKernels::resolve(&g).expect("readout kernels"));
    let transform_kernel = g.kernel_index("logit_cap_mask").expect("registered");
    let mut rng = Lcg::new(random_seed());
    let raw = raw_logits(&mut rng);
    let fresh = |apply: bool| {
        let b = g.storage_init("logits", &raw);
        if apply {
            g.submit(&[], &[t.step(&g, transform_kernel, &b, ROWS as u32, VOCAB as u32).expect("a transform that does something")]);
        }
        b
    };
    // The control: untransformed, the padding wins everywhere.
    assert!(argmax_rows(&g, rk, &fresh(false), ROWS as u32, VOCAB as u32).iter().all(|&i| i >= VALID), "the fixture's padding must dominate the raw logits");

    let logits = fresh(true);
    assert!(argmax_rows(&g, rk, &logits, ROWS as u32, VOCAB as u32).iter().all(|&i| i < VALID), "the greedy pick chose a padded column");
    let top = topk_rows(&g, rk, &fresh(true), ROWS as u32, VOCAB as u32, 8);
    for (r, row) in top.iter().enumerate() {
        assert!(row.iter().all(|&(i, v)| i < VALID && v.abs() <= CAP * (1.0 + 1e-5)), "row {r}: candidates {row:?} leave the valid vocabulary or exceed the cap");
        assert!(row.windows(2).all(|w| w[0].1 >= w[1].1), "row {r}: candidates are not best first");
    }
    // Many keyed draws at a high temperature (where a leaked column would show), whole rows, nuclei and top-k alike.
    for key in 0..60u32 {
        let draws: Vec<Option<RowDraw>> = (0..ROWS)
            .map(|r| match r % 3 {
                0 => Some(RowDraw { temp: 3.0, min_p: 0.0, top_p: 1.0, top_k: 0, top: 0, support: 0, key: draw_key(7, 100 + u64::from(key), r as u32) }),
                1 => Some(RowDraw { temp: 2.0, min_p: 0.01, top_p: 0.95, top_k: 0, top: 0, support: 0, key: draw_key(8, u64::from(key), r as u32) }),
                _ => Some(RowDraw { temp: 1.5, min_p: 0.0, top_p: 1.0, top_k: 6, top: 0, support: 0, key: draw_key(9, u64::from(key), r as u32) }),
            })
            .collect();
        for (r, d) in draw_rows(&g, &fresh(true), &draws, VOCAB as u32).into_iter().enumerate() {
            let Drawn::Token(token, lp, _) = d else { panic!("row {r} was keyed") };
            assert!(token < VALID && lp.is_finite(), "row {r}, key {key}: drew {token} (log-probability {lp})");
        }
    }
    // A full row read back: padding holds the floor, every valid entry is capped, and `trim` cuts the padding without
    // capping a second time.
    let rows = draw_rows(&g, &fresh(true), &[None; ROWS], VOCAB as u32);
    for (r, d) in rows.into_iter().enumerate() {
        let Drawn::Row(row) = d else { panic!("row {r} was not keyed") };
        assert!(row[VALID as usize..].iter().all(|&v| v == MASKED_LOGIT), "row {r}: padding is not at the floor");
        let trimmed = t.trim(row.clone());
        assert_eq!(trimmed.len(), VALID as usize);
        assert_eq!(trimmed, row[..VALID as usize], "trim must not change the served entries");
        assert!(trimmed.iter().all(|v| v.abs() <= CAP * (1.0 + 1e-5)));
    }
}

#[test]
fn the_device_transform_equals_the_host_one() {
    let g = gpu_core::testgpu::dev(kernels());
    let t = transform();
    let kernel = g.kernel_index("logit_cap_mask").expect("registered");
    let raw = raw_logits(&mut Lcg::new(random_seed()));
    let b = g.storage_init("logits", &raw);
    g.submit(&[], &[t.step(&g, kernel, &b, ROWS as u32, VOCAB as u32).unwrap()]);
    let dev = g.read(&b, ROWS * VOCAB);
    for r in 0..ROWS {
        let host = t.served(raw[r * VOCAB..(r + 1) * VOCAB].to_vec());
        assert_eq!(host.len(), VALID as usize);
        for (i, h) in host.iter().enumerate() {
            assert!((dev[r * VOCAB + i] - h).abs() <= 1e-5, "row {r} col {i}: device {} vs host {h}", dev[r * VOCAB + i]);
        }
    }
    assert!(HeadTransform::NONE.step(&g, kernel, &b, 1, 1).is_none(), "the identity dispatches nothing");
}

#[test]
fn host_rows_are_cut_capped_and_a_distribution_over_the_valid_vocabulary() {
    let t = transform();
    let raw = raw_logits(&mut Lcg::new(random_seed()))[..VOCAB].to_vec();
    let served = t.served(raw.clone());
    assert_eq!(served.len(), VALID as usize);
    assert!(served.iter().zip(&raw).all(|(s, r)| (s - model::hostmath::softcap(*r, CAP)).abs() <= 1e-6));
    let total: f64 = (0..VALID as usize).map(|i| f64::from(t.logprob_at(raw.clone(), i, 1.0)).exp()).sum();
    assert!((total - 1.0).abs() <= 1e-4, "probabilities over the valid vocabulary sum to {total}");
    for key in 0..200u32 {
        let (token, _) = draw_row(&served, RowDraw { temp: 3.0, min_p: 0.0, top_p: 1.0, top_k: 0, top: 0, support: 0, key: draw_key(1, u64::from(key), 0) });
        assert!(token < VALID, "drew {token}");
    }
}

/// A decoder over a padded head that declares its transform and serves transformed rows, as the contract says.
struct Padded {
    alloc: BlockAllocator,
}

impl Padded {
    fn raw_row() -> Vec<f32> {
        let mut r: Vec<f32> = (0..VOCAB).map(|i| (i % 7) as f32 - 3.0).collect();
        r[VALID as usize..].iter_mut().for_each(|x| *x = 80.0);
        r
    }
}

impl PagedDecoder for Padded {
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
        1024
    }
    fn vocab(&self) -> usize {
        transform().served_vocab(VOCAB)
    }
    fn head_transform(&self) -> HeadTransform {
        transform()
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
        table.reserve(prompt.len() as u32, &mut self.alloc).unwrap();
        vec![0.0]
    }
    fn logits(&self, _hidden: &[f32]) -> Vec<f32> {
        transform().served(Self::raw_row())
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        let best = model::hostmath::argmax_row(&self.logits(&[]));
        inputs.iter().map(|_| best).collect()
    }
    fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
        unreachable!("window capacity is 1")
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
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        let row = self.logits(&[]);
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        inputs.iter().map(|_| order[..k.min(8)].iter().map(|&i| (i as u32, row[i])).collect()).collect()
    }
    fn topk_capacity(&self) -> usize {
        8
    }
    fn full_row_readback(&self) -> bool {
        true
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|_| self.logits(&[])).collect()
    }
}

fn padded() -> Padded {
    Padded { alloc: BlockAllocator::new(256, 8) }
}

#[test]
fn a_decoder_that_declares_the_transform_is_served_inside_the_valid_vocabulary() {
    let mut sched = Scheduler::new(padded(), 4);
    assert_eq!(sched.decoder_mut().vocab(), VALID as usize, "vocab() reports the valid size");
    let ids: Vec<u64> = (0..8u64)
        .map(|seed| sched.submit_sampled(Request { prompt: vec![1, 2, 3], max_new: 12, stop: Vec::new() }, SampleParams { temp: 3.0, ..SampleParams::greedy() }, seed + 1))
        .collect();
    let done = sched.run();
    for id in ids {
        assert!(done[&id].iter().all(|&t| t < VALID), "request {id} emitted a padded token: {:?}", done[&id]);
    }
    // A prompt that names a padded token is refused at admission, as any token at or above vocab().
    let id = sched.submit(Request { prompt: vec![1, VALID, 3], max_new: 2, stop: Vec::new() });
    let report = sched.step_report();
    assert!(report.rejected.iter().any(|(got, _)| *got == id), "a padded token in a prompt must be rejected: {:?}", report.rejected);
}

#[test]
fn the_conformance_property_accepts_a_decoder_that_honours_its_declaration() {
    a_declared_head_transform_is_honoured(&Conformance { make: padded, prefix_reuse: false, prefix_prompt_tokens: 36, load_adapter: None });
}

/// A decoder that declares a cap but serves raw rows breaks the property: it is the check that has teeth.
#[test]
#[should_panic(expected = "exceeds the soft cap")]
fn the_conformance_property_refuses_a_decoder_that_serves_raw_rows() {
    // A thin lie: the same decoder, but its `logits` forgets the cap.
    struct Liar(Padded);
    impl PagedDecoder for Liar {
        fn alloc_mut(&mut self) -> &mut BlockAllocator {
            self.0.alloc_mut()
        }
        fn max_prefill_tokens(&self) -> u32 {
            64
        }
        fn free_blocks(&self) -> u32 {
            self.0.free_blocks()
        }
        fn max_seq_len(&self) -> usize {
            1024
        }
        fn vocab(&self) -> usize {
            self.0.vocab()
        }
        fn head_transform(&self) -> HeadTransform {
            transform()
        }
        fn blocks_for(&self, tokens: u32) -> u32 {
            self.0.blocks_for(tokens)
        }
        fn reclaim_prefix(&mut self, w: u32) -> u32 {
            self.0.reclaim_prefix(w)
        }
        fn release_table(&mut self, t: &mut BlockTable) {
            self.0.release_table(t)
        }
        fn prefill(&mut self, t: &mut BlockTable, p: &[u32]) -> Vec<f32> {
            self.0.prefill(t, p)
        }
        fn logits(&self, _h: &[f32]) -> Vec<f32> {
            Padded::raw_row()[..VALID as usize].iter().map(|v| v * 10.0).collect()
        }
        fn forward_batched_greedy(&mut self, t: &mut [&mut BlockTable], i: &[u32]) -> Vec<u32> {
            self.0.forward_batched_greedy(t, i)
        }
        fn forward_batched_greedy_window(&mut self, t: &mut [&mut BlockTable], i: &[u32], k: usize) -> Vec<Vec<u32>> {
            self.0.forward_batched_greedy_window(t, i, k)
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
        fn forward_batched_topk(&mut self, t: &mut [&mut BlockTable], i: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
            self.0.forward_batched_topk(t, i, k)
        }
        fn topk_capacity(&self) -> usize {
            8
        }
    }
    a_declared_head_transform_is_honoured(&Conformance { make: || Liar(padded()), prefix_reuse: false, prefix_prompt_tokens: 36, load_adapter: None });
}
