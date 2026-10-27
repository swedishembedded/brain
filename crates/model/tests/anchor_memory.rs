// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `model::anchor_memory`'s two device stages against independent host
//! references: the admission selection (`anchor_select`) and the indexed read
//! with a sink (`anchor_read_*` and the row softmax). The host side restates
//! the DEFINITION of the bank (sort all earlier tokens by score descending,
//! position ascending) rather than the kernel's rank-by-counting scan, so a
//! shared bug cannot make both agree.

use data::rng::{random_seed, Lcg};
use gpu_core::{DeviceBuffer, Gpu};
use model::anchor_memory::{anchor_read_bwd, anchor_read_steps, anchor_select_steps, decode_refs, AnchorMemoryKernels, AnchorMemoryShape, ANCHOR_MEMORY_PIPELINES};

fn dev() -> (Gpu, AnchorMemoryKernels) {
    let g = gpu_core::testgpu::dev(&ANCHOR_MEMORY_PIPELINES);
    let k = AnchorMemoryKernels::resolve(&g).expect("the pipelines were registered");
    (g, k)
}

fn shape(n_seq: u32, t: u32, groups: u32, heads: u32, head_dim: u32, capacity: u32, commit_block: u32) -> AnchorMemoryShape {
    AnchorMemoryShape {
        n_seq,
        t,
        d_model: 1,
        n_heads: heads,
        n_kv_heads: groups,
        head_dim,
        rotary_half: 0,
        rms_eps: 1e-6,
        capacity,
        commit_block,
        admission_rank: 1,
        writer_intermediate: 1,
    }
}

/// `refs[s][b][g][c]` by definition: the `capacity` best of the tokens of blocks `< b`, `-1` padded.
fn bank_by_definition(scores: &[f32], s: &AnchorMemoryShape) -> Vec<i64> {
    let (t, groups, cap) = (s.t as usize, s.n_kv_heads as usize, s.capacity as usize);
    let mut refs = Vec::new();
    for seq in 0..s.n_seq as usize {
        for b in 0..s.n_blocks() as usize {
            for g in 0..groups {
                let committed = (b * s.commit_block as usize).min(t);
                let score = |j: usize| scores[(seq * t + j) * groups + g];
                let mut cands: Vec<usize> = (0..committed).collect();
                cands.sort_by(|&x, &y| score(y).partial_cmp(&score(x)).unwrap().then(x.cmp(&y)));
                cands.truncate(cap);
                refs.extend(cands.iter().map(|&j| j as i64));
                refs.extend(std::iter::repeat_n(-1, cap - cands.len()));
            }
        }
    }
    refs
}

/// Scores drawn from `levels` distinct values, so ties are the common case when `levels` is small.
fn scores_with_ties(r: &mut Lcg, n: usize, levels: u32) -> Vec<f32> {
    (0..n).map(|_| (r.next_u32() % levels) as f32 * 0.25 - 0.5).collect()
}

fn run_select(g: &Gpu, k: &AnchorMemoryKernels, s: &AnchorMemoryShape, scores: &[f32]) -> Vec<i64> {
    let buf = g.storage_init("scores", scores);
    // Poison the buffer: the kernel must not depend on an initialised bank.
    let refs = g.storage_init("refs", &vec![f32::from_bits(0xDEAD_BEEF); s.refs_len() as usize]);
    g.submit(&[], &anchor_select_steps(g, k, s, &buf, &refs));
    decode_refs(&g.read(&refs, s.refs_len() as usize))
}

#[test]
fn selection_matches_the_definition_with_ties_eviction_and_empty_slots() {
    let (g, k) = dev();
    let mut r = Lcg::new(random_seed());
    // (n_seq, t, groups, capacity, commit_block): eviction, a ragged last block, capacity above the
    // candidate count (empty slots), one block only, a block exactly filling the sequence.
    for (n_seq, t, groups, cap, chunk) in [(2, 19, 3, 5, 4), (1, 48, 2, 6, 4), (2, 13, 1, 40, 4), (1, 3, 2, 4, 4), (2, 8, 2, 3, 8), (1, 1, 1, 2, 1)] {
        for levels in [3, 1000] {
            let s = shape(n_seq, t, groups, groups, 2, cap, chunk);
            let scores = scores_with_ties(&mut r, (n_seq * t * groups) as usize, levels);
            let got = run_select(&g, &k, &s, &scores);
            let want = bank_by_definition(&scores, &s);
            assert_eq!(got, want, "n_seq {n_seq} t {t} groups {groups} cap {cap} chunk {chunk} levels {levels}");
        }
    }
}

#[test]
fn equal_scores_keep_the_earliest_positions() {
    let (g, k) = dev();
    let s = shape(1, 24, 2, 2, 2, 5, 4);
    let got = run_select(&g, &k, &s, &vec![0.0; (s.t * s.n_kv_heads) as usize]);
    let cap = s.capacity as usize;
    let last_block_group0 = &got[((s.n_blocks() as usize - 1) * 2) * cap..][..cap];
    assert_eq!(last_block_group0, &[0, 1, 2, 3, 4], "a tie goes to the earlier position");
    let first_block = &got[..2 * cap];
    assert!(first_block.iter().all(|&r| r == -1), "block 0 reads an empty bank");
}

/// The read against a bank given by `refs`, in f64: softmax over the bank's logits plus one zero logit, the zero
/// logit carrying a zero value.
fn host_read(q: &[f32], key: &[f32], value: &[f32], refs: &[i64], s: &AnchorMemoryShape) -> Vec<f32> {
    let (t, h, groups, hd, cap) = (s.t as usize, s.n_heads as usize, s.n_kv_heads as usize, s.head_dim as usize, s.capacity as usize);
    let per_group = h / groups;
    let mut out = vec![0f32; s.rows() as usize * h * hd];
    for row in 0..s.rows() as usize {
        let (seq, pos) = (row / t, row % t);
        let block = pos / s.commit_block as usize;
        for head in 0..h {
            let g = head / per_group;
            let bank = &refs[((seq * s.n_blocks() as usize + block) * groups + g) * cap..][..cap];
            let live: Vec<usize> = bank.iter().filter(|&&r| r >= 0).map(|&r| r as usize).collect();
            let logit = |j: usize| -> f64 {
                let dot: f64 = (0..hd).map(|d| f64::from(q[(row * h + head) * hd + d]) * f64::from(key[((seq * t + j) * groups + g) * hd + d])).sum();
                dot / (hd as f64).sqrt()
            };
            let logits: Vec<f64> = live.iter().map(|&j| logit(j)).collect();
            let m = logits.iter().cloned().fold(0.0, f64::max); // the sink's zero logit is in the max
            let denom: f64 = (-m).exp() + logits.iter().map(|l| (l - m).exp()).sum::<f64>();
            for d in 0..hd {
                let acc: f64 = live.iter().zip(&logits).map(|(&j, l)| (l - m).exp() / denom * f64::from(value[((seq * t + j) * groups + g) * hd + d])).sum();
                out[(row * h + head) * hd + d] = acc as f32;
            }
        }
    }
    out
}

fn upload_refs(g: &Gpu, refs: &[i64]) -> DeviceBuffer {
    let words: Vec<u32> = refs.iter().map(|&r| (r + 1) as u32).collect();
    let buf = g.storage(words.len() as u64);
    g.write(&buf, &words);
    buf
}

#[test]
fn indexed_read_with_sink_matches_the_host_reference() {
    let (g, k) = dev();
    let mut r = Lcg::new(random_seed());
    // Two heads per group; a bank that is partly empty (cap 40 over <= 13 tokens) and one that is full.
    for (n_seq, t, groups, heads, hd, cap, chunk) in [(2, 13, 2, 4, 6, 40, 4), (1, 30, 3, 6, 5, 4, 3)] {
        let s = shape(n_seq, t, groups, heads, hd, cap, chunk);
        let rows = (n_seq * t) as usize;
        let scores = scores_with_ties(&mut r, rows * groups as usize, 7);
        let refs = bank_by_definition(&scores, &s);
        let q = r.vec_scaled(rows * (heads * hd) as usize, 1.5);
        let key = r.vec_scaled(rows * (groups * hd) as usize, 1.5);
        let value = r.vec_scaled(rows * (groups * hd) as usize, 1.0);

        let (qb, kb, vb) = (g.storage_init("q", &q), g.storage_init("k", &key), g.storage_init("v", &value));
        let refs_buf = upload_refs(&g, &refs);
        let logits = g.storage(rows as u64 * u64::from(heads) * u64::from(cap + 1));
        let probs = g.storage(rows as u64 * u64::from(heads) * u64::from(cap + 1));
        let read = g.storage(rows as u64 * u64::from(heads * hd));
        g.submit(&[], &anchor_read_steps(&g, &k, &s, &qb, &kb, &vb, &refs_buf, &logits, &probs, &read));
        let got = g.read(&read, rows * (heads * hd) as usize);
        let want = host_read(&q, &key, &value, &refs, &s);

        let worst = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-5, "t {t} cap {cap}: read differs from the host reference by {worst}");
        // Block 0 sees an empty bank: only the sink, so the read is exactly zero.
        let block0 = rows_of_block0(&s);
        assert!(block0.iter().all(|&row| got[row * (heads * hd) as usize..][..(heads * hd) as usize].iter().all(|&x| x == 0.0)), "an empty bank must read zero");
        assert!(got.iter().any(|&x| x != 0.0), "the read is not vacuous");
    }
}

fn rows_of_block0(s: &AnchorMemoryShape) -> Vec<usize> {
    (0..s.n_seq as usize).flat_map(|seq| (0..(s.commit_block.min(s.t)) as usize).map(move |p| seq * s.t as usize + p)).collect()
}

/// The read's backward by explicit scatter, in f64: softmax over the live slots plus the sink, `dP_c = d_read . v_c`,
/// `dl_c = p_c (dP_c - sum_i p_i dP_i)`, then each slot adds into its record's key and value gradient.
fn host_read_bwd(q: &[f32], key: &[f32], value: &[f32], refs: &[i64], d_read: &[f32], s: &AnchorMemoryShape) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (t, h, groups, hd, cap) = (s.t as usize, s.n_heads as usize, s.n_kv_heads as usize, s.head_dim as usize, s.capacity as usize);
    let per_group = h / groups;
    let (mut dq, mut dk, mut dv) = (vec![0f64; q.len()], vec![0f64; key.len()], vec![0f64; value.len()]);
    let scale = 1.0 / (hd as f64).sqrt();
    for row in 0..s.rows() as usize {
        let (seq, pos) = (row / t, row % t);
        let block = pos / s.commit_block as usize;
        for head in 0..h {
            let g = head / per_group;
            let bank = &refs[((seq * s.n_blocks() as usize + block) * groups + g) * cap..][..cap];
            let live: Vec<usize> = bank.iter().filter(|&&r| r >= 0).map(|&r| r as usize).collect();
            let qv = |d: usize| f64::from(q[(row * h + head) * hd + d]);
            let kv = |j: usize, d: usize| f64::from(key[((seq * t + j) * groups + g) * hd + d]);
            let vv = |j: usize, d: usize| f64::from(value[((seq * t + j) * groups + g) * hd + d]);
            let dr = |d: usize| f64::from(d_read[(row * h + head) * hd + d]);
            let logits: Vec<f64> = live.iter().map(|&j| (0..hd).map(|d| qv(d) * kv(j, d)).sum::<f64>() * scale).collect();
            let m = logits.iter().cloned().fold(0.0, f64::max);
            let denom: f64 = (-m).exp() + logits.iter().map(|l| (l - m).exp()).sum::<f64>();
            let p: Vec<f64> = logits.iter().map(|l| (l - m).exp() / denom).collect();
            let dp: Vec<f64> = live.iter().map(|&j| (0..hd).map(|d| dr(d) * vv(j, d)).sum()).collect();
            let sum: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
            for (c, &j) in live.iter().enumerate() {
                let dl = p[c] * (dp[c] - sum);
                for d in 0..hd {
                    dq[(row * h + head) * hd + d] += dl * kv(j, d) * scale;
                    dk[((seq * t + j) * groups + g) * hd + d] += dl * qv(d) * scale;
                    dv[((seq * t + j) * groups + g) * hd + d] += p[c] * dr(d);
                }
            }
        }
    }
    (dq, dk, dv)
}

#[test]
fn read_backward_gathers_the_same_gradients_a_scatter_would() {
    let (g, k) = dev();
    let mut r = Lcg::new(random_seed());
    // Several records read by many rows (cap close to the candidates), empty slots, two sequences.
    for (n_seq, t, groups, heads, hd, cap, chunk) in [(2, 13, 2, 4, 6, 40, 4), (2, 30, 3, 6, 5, 4, 3), (1, 17, 1, 3, 4, 5, 2)] {
        let s = shape(n_seq, t, groups, heads, hd, cap, chunk);
        let rows = (n_seq * t) as usize;
        let scores = scores_with_ties(&mut r, rows * groups as usize, 7);
        let refs = bank_by_definition(&scores, &s);
        let q = r.vec_scaled(rows * (heads * hd) as usize, 1.5);
        let key = r.vec_scaled(rows * (groups * hd) as usize, 1.5);
        let value = r.vec_scaled(rows * (groups * hd) as usize, 1.0);
        let d_read = r.vec_scaled(rows * (heads * hd) as usize, 1.0);

        let (qb, kb, vb, db) = (g.storage_init("q", &q), g.storage_init("k", &key), g.storage_init("v", &value), g.storage_init("d", &d_read));
        let refs_buf = upload_refs(&g, &refs);
        let logits = g.storage(rows as u64 * u64::from(heads) * u64::from(cap + 1));
        let probs = g.storage(rows as u64 * u64::from(heads) * u64::from(cap + 1));
        let read = g.storage(rows as u64 * u64::from(heads * hd));
        g.submit(&[], &anchor_read_steps(&g, &k, &s, &qb, &kb, &vb, &refs_buf, &logits, &probs, &read));
        let (dq, dk, dv) = anchor_read_bwd(&g, &k, &s, &qb, &kb, &vb, &refs_buf, &probs, &db);

        let (want_q, want_k, want_v) = host_read_bwd(&q, &key, &value, &refs, &d_read, &s);
        for (name, got, want) in [("dq", g.read(&dq, q.len()), want_q), ("dkey", g.read(&dk, key.len()), want_k), ("dvalue", g.read(&dv, value.len()), want_v)] {
            let worst = got.iter().zip(&want).fold(0f64, |m, (a, b)| m.max((f64::from(*a) - b).abs()));
            let scale = want.iter().fold(0f64, |m, b| m.max(b.abs())).max(1e-3);
            assert!(worst < 1e-4 * scale.max(1.0), "t {t} cap {cap} {name}: differs from the scatter reference by {worst} (scale {scale})");
            assert!(want.iter().any(|b| b.abs() > 1e-3), "{name} reference is vacuous");
        }
        // A record no row reads (the last block's tokens) gets exactly zero.
        let last_block_start = ((s.n_blocks() - 1) * s.commit_block) as usize;
        let dkey = g.read(&dk, key.len());
        for row in last_block_start..t as usize {
            assert!(dkey[(row * (groups * hd) as usize)..][..(groups * hd) as usize].iter().all(|&x| x == 0.0), "a record of the last block is read by nobody");
        }
    }
}
