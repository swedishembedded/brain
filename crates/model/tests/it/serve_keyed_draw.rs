// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements exact, reproducible sampled decoding at
// serving batch sizes for its clients. If your team needs expertise in
// sampling a served language model without one host core bounding it then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The keyed whole-row draw (`model::serve::draw_row`) and the scheduler's use of it: the draw follows the
//! distribution the row sampler defines for temperature, min-p, top-p and top-k, reports that distribution's
//! log-probability, and a seed draws the same tokens whatever else shares the batch; a request the keyed draw does
//! not cover (a logit processor) is still sampled from its own row on the host.

use model::paged::{BlockAllocator, BlockTable};
use model::serve::{draw_key, draw_row, Drawn, PagedDecoder, Request, RowDraw, SampleParams, Scheduler};

/// The distribution the row sampler defines for a temperature-and-min-p request: the softmax of `logit / temp`
/// over the tokens of positive weight whose weight relative to the best is at least `min_p`.
fn exact(row: &[f32], temp: f32, min_p: f32) -> Vec<f64> {
    exact_truncated(row, temp, min_p, 1.0, 0)
}

/// The distribution `sample_logits_row` defines, written out from its contract: rank the row best first (equal
/// logits lowest id first, NaN never), keep the best `top_k` (all when 0), take the softmax weights of `logit / temp`,
/// cut the nucleus at the first rank whose cumulative weight reaches `top_p` of the normaliser (the whole row's
/// without `top_k`, the kept list's with it), then keep the ranks whose weight is at least `min_p` of the best.
fn exact_truncated(row: &[f32], temp: f32, min_p: f32, top_p: f32, top_k: usize) -> Vec<f64> {
    let mut order: Vec<usize> = (0..row.len()).filter(|&i| !row[i].is_nan()).collect();
    order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    let best = row[order[0]] as f64;
    let weight = |i: usize| ((row[i] as f64 - best) / temp as f64).exp();
    let whole: f64 = if top_k > 0 { order.iter().take(top_k).map(|&i| weight(i)).sum() } else { order.iter().map(|&i| weight(i)).sum() };
    let kept: Vec<usize> = if top_k > 0 { order[..top_k.min(order.len())].to_vec() } else { order.clone() };
    let mut cut = kept.len();
    if top_p > 0.0 && top_p < 1.0 {
        let mut acc = 0.0;
        for (r, &i) in kept.iter().enumerate() {
            acc += weight(i);
            if acc / whole >= top_p as f64 {
                cut = r + 1;
                break;
            }
        }
    }
    let survivors: Vec<usize> = kept[..cut].iter().copied().filter(|&i| weight(i) > 0.0 && weight(i) >= min_p as f64).collect();
    let z: f64 = survivors.iter().map(|&i| weight(i)).sum();
    let mut p = vec![0.0; row.len()];
    for i in survivors {
        p[i] = weight(i) / z;
    }
    p
}

/// Pearson's statistic of `counts` against `probs` over `n` draws, and whether any draw fell outside the support.
fn chi_square(counts: &[u64], probs: &[f64], n: u64) -> (f64, bool) {
    let mut stat = 0.0;
    let mut outside = false;
    for (&c, &p) in counts.iter().zip(probs) {
        if p == 0.0 {
            outside |= c > 0;
            continue;
        }
        let e = p * n as f64;
        stat += (c as f64 - e).powi(2) / e;
    }
    (stat, outside)
}

/// A row with ties, a NaN, a `-inf` and a long low tail, at two temperatures and with and without min-p: the
/// keyed draw lands on each token as often as the exact distribution says (Pearson's test at a one-in-a-million
/// false-alarm level for 11 degrees of freedom), never outside the survivors, and reports each token's exact
/// log-probability.
#[test]
fn the_keyed_draw_follows_the_row_samplers_distribution_and_reports_its_logprob() {
    let row = [2.0f32, 1.5, 1.5, 0.3, -1.0, f32::NAN, f32::NEG_INFINITY, -3.0, 0.0, 1.9, -0.5, 0.7, -2.0];
    let n = 60_000u64;
    let seed = 0x1234_5678_9abc_def0; // seed: fixed fixture so a statistical failure reproduces on the same draws
    for (temp, min_p) in [(1.0f32, 0.0f32), (0.6, 0.0), (1.4, 0.15), (1.0, 0.5)] {
        let probs = exact(&row, temp, min_p);
        let mut counts = vec![0u64; row.len()];
        for pos in 0..n {
            let (t, lp) = draw_row(&row, RowDraw { temp, min_p, top_p: 1.0, top_k: 0, key: draw_key(seed, pos, 0) });
            counts[t as usize] += 1;
            assert!((lp as f64 - probs[t as usize].ln()).abs() < 1e-4, "temp {temp} min_p {min_p}: token {t} logprob {lp} vs {}", probs[t as usize].ln());
        }
        let (stat, outside) = chi_square(&counts, &probs, n);
        assert!(!outside, "temp {temp} min_p {min_p}: a draw left the survivors: {counts:?}");
        let dof = probs.iter().filter(|&&p| p > 0.0).count() - 1;
        // The chi-square quantile at 1 - 1e-6 is below 3 * dof + 30 for every dof this test reaches.
        assert!(stat < (3 * dof + 30) as f64, "temp {temp} min_p {min_p}: chi-square {stat:.1} over {dof} dof, counts {counts:?} vs {probs:?}");
    }
}

/// A nucleus, a `top_k` and both together, with and without min-p, over a row whose ties straddle the cuts: the
/// keyed draw follows the row sampler's truncated distribution and reports its log-probability.
#[test]
fn the_keyed_draw_follows_the_truncated_distributions() {
    let row = [2.0f32, 1.5, 1.5, 0.3, -1.0, f32::NAN, f32::NEG_INFINITY, -3.0, 0.0, 1.9, -0.5, 0.7, -2.0, 1.5, 0.3];
    let n = 60_000u64;
    let seed = 0x0bad_5eed_1234_0001; // seed: fixed fixture so a statistical failure reproduces on the same draws
    for (temp, min_p, top_p, top_k) in [(1.0f32, 0.0f32, 0.8f32, 0usize), (0.7, 0.0, 0.5, 0), (1.3, 0.05, 0.9, 0), (1.0, 0.0, 1.0, 3), (1.0, 0.0, 1.0, 5), (1.2, 0.0, 0.7, 6), (1.0, 0.2, 0.95, 8)] {
        let probs = exact_truncated(&row, temp, min_p, top_p, top_k);
        let mut counts = vec![0u64; row.len()];
        for pos in 0..n {
            let (t, lp) = draw_row(&row, RowDraw { temp, min_p, top_p, top_k, key: draw_key(seed, pos, 0) });
            counts[t as usize] += 1;
            assert!((lp as f64 - probs[t as usize].ln()).abs() < 1e-4, "{temp} {min_p} {top_p} {top_k}: token {t} logprob {lp} vs {}", probs[t as usize].ln());
        }
        let (stat, outside) = chi_square(&counts, &probs, n);
        assert!(!outside, "{temp} {min_p} {top_p} {top_k}: a draw left the survivors: {counts:?} {probs:?}");
        let dof = probs.iter().filter(|&&p| p > 0.0).count().max(2) - 1;
        assert!(stat < (3 * dof + 30) as f64, "{temp} {min_p} {top_p} {top_k}: chi-square {stat:.1} over {dof} dof, counts {counts:?} vs {probs:?}");
    }
}

/// Which requests the keyed draw covers: real sampling with no logit processor.
#[test]
fn the_keyed_draw_covers_every_sampled_request_without_a_processor() {
    let t = SampleParams { temp: 1.0, ..SampleParams::greedy() };
    assert!(t.row_drawable());
    assert!(SampleParams { min_p: 0.05, ..t }.row_drawable());
    assert!(SampleParams { top_k: 40, ..t }.row_drawable());
    assert!(SampleParams { top_p: 0.9, ..t }.row_drawable());
    assert!(!SampleParams::greedy().row_drawable());
    assert!(!SampleParams { presence_penalty: 0.5, ..t }.row_drawable());
}

const VOCAB: usize = 48;

/// A decoder whose logits row depends only on the token fed in, so a sequence's rows are the same in any batch.
/// `on_device` answers keyed rows itself (as a device would), counting the rows it drew and returned.
struct Echo {
    alloc: BlockAllocator,
    on_device: bool,
    drawn: usize,
    returned: usize,
}

fn row_for(input: u32) -> Vec<f32> {
    (0..VOCAB).map(|t| ((t as u32 * 7 + input * 13) % 17) as f32 * 0.25).collect()
}

impl PagedDecoder for Echo {
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
        VOCAB
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
        vec![*prompt.last().unwrap() as f32]
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        row_for(hidden[0] as u32)
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
        unreachable!("every request here needs the whole row")
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
        inputs.iter().map(|&i| row_for(i)).collect()
    }
    fn forward_batched_drawn(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], draws: &[Option<RowDraw>]) -> Vec<Drawn> {
        if !self.on_device {
            let rows = self.forward_batched_logit_rows(tables, inputs);
            return model::serve::draw_rows_host(rows, draws);
        }
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs
            .iter()
            .zip(draws)
            .map(|(&i, d)| match d {
                Some(d) => {
                    self.drawn += 1;
                    let (t, lp) = draw_row(&row_for(i), *d);
                    Drawn::Token(t, lp)
                }
                None => {
                    self.returned += 1;
                    Drawn::Row(row_for(i))
                }
            })
            .collect()
    }
}

fn echo(on_device: bool) -> Scheduler<Echo> {
    Scheduler::new(Echo { alloc: BlockAllocator::new(512, 8), on_device, drawn: 0, returned: 0 }, 8)
}

/// One request alone, then the same request beside others that arrive before and after it, on a decoder that
/// draws on its own and on one that reads its rows back: the same seed draws the same tokens every time.
#[test]
fn a_seed_draws_the_same_tokens_in_any_batch_and_wherever_the_row_is_drawn() {
    let params = SampleParams { temp: 1.0, ..SampleParams::greedy() };
    let seed = 0xfeed_beef; // seed: fixed fixture, the request whose stream is compared
    let mine = || Request { prompt: vec![3, 4, 5], max_new: 24, stop: Vec::new() };
    let other = |p: u32| Request { prompt: vec![p, p + 1], max_new: 30, stop: Vec::new() };
    let alone = {
        let mut s = echo(true);
        let id = s.submit_sampled(mine(), params, seed);
        s.run().remove(&id).unwrap()
    };
    for on_device in [true, false] {
        let mut s = echo(on_device);
        s.submit_sampled(other(9), params, 7); // seed: test-double, a neighbour's stream is not compared
        s.submit_sampled(other(11), SampleParams { min_p: 0.2, ..params }, 8); // seed: test-double, as above
        let id = s.submit_sampled(mine(), params, seed);
        s.submit_sampled(other(2), params, 9); // seed: test-double, as above
        assert_eq!(s.run().remove(&id).unwrap(), alone, "on_device {on_device}");
    }
}

/// A penalised request in the same step as keyed ones (a nucleus among them): the decoder draws the keyed rows and
/// hands the penalised row back, which the host samples (its argmax moves off the repeated token).
#[test]
fn rows_the_keyed_draw_does_not_cover_are_sampled_on_the_host_beside_drawn_ones() {
    let keyed = SampleParams { temp: 1.0, ..SampleParams::greedy() };
    let mut s = echo(true);
    // seed: test-double, three keyed neighbours
    for seed in 1..=3 {
        s.submit_sampled(Request { prompt: vec![seed as u32], max_new: 6, stop: Vec::new() }, keyed, seed);
    }
    let nucleus = s.submit_sampled(Request { prompt: vec![4], max_new: 6, stop: Vec::new() }, SampleParams { top_p: 0.5, ..keyed }, 4); // seed: test-double
    let penalised = s.submit_sampled(Request { prompt: vec![5], max_new: 6, stop: Vec::new() }, SampleParams { frequency_penalty: 50.0, ..SampleParams::greedy() }, 5); // seed: test-double
    let out = s.run();
    let mut seen = out[&penalised].clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 6, "a frequency penalty of 50 never repeats a token: {:?}", out[&penalised]);
    assert_eq!(out[&nucleus].len(), 6);
    let dec = s.decoder();
    // The first token of each request comes from its prefill row; the five after it from decode steps.
    assert_eq!((dec.drawn, dec.returned), (4 * 5, 5), "keyed rows drawn by the decoder, the others returned");
}
