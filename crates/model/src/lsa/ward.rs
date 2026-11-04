// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ward merging of one cluster block's pools into buckets.
//!
//! A bucket is the sum of its keys; its *alignment* is that sum's norm, and
//! its centroid is the sum over its alignment. Merging two buckets costs the
//! alignment the union loses: `a(b) + a(b') - a(b ∪ b')`. Each round every
//! live bucket finds its cheapest partner among those the size cap allows (a
//! tie to the lower index), every pair that is each other's cheapest merges
//! into its lower member, and in the round that would overshoot the target
//! count only the cheapest such pairs merge. There is no seed and no
//! assignment pass: the merges are the partition.
//!
//! The round structure, tie rules and the cheapest-first cut follow the
//! reference implementation (provenance in `/NOTICE.md`).

/// Rounds of mutual-cheapest merges before a block is declared stuck.
///
/// Each round at least halves the surplus of a block that can still merge, so
/// a block of `P` pools needs about `log2(P)` rounds; this is a hard ceiling
/// against a defect, not a tuning knob.
const MAX_ROUNDS: usize = 256;

/// Sum each run of `pool_size` consecutive keys of a block into a pool key.
///
/// `keys` is `[n, dim]`, `n` a multiple of `pool_size`; the result is
/// `[n / pool_size, dim]`.
pub fn pool_keys(keys: &[f32], dim: usize, pool_size: usize) -> Vec<f32> {
    let pools = keys.len() / dim / pool_size;
    let mut out = vec![0f32; pools * dim];
    for (p, pool) in out.chunks_exact_mut(dim).enumerate() {
        for k in 0..pool_size {
            let row = &keys[(p * pool_size + k) * dim..][..dim];
            for (o, x) in pool.iter_mut().zip(row) {
                *o += *x;
            }
        }
    }
    out
}

/// Merge `pools` (`[P, dim]`, each holding `sizes[i]` keys) down to exactly
/// `bucket_count` buckets no larger than `bucket_cap` keys.
///
/// Returns each pool's bucket, numbered from 0 in the order of the bucket's
/// first pool. Fails when the cap leaves a block with no allowed pair before
/// it reaches its count, which a validated [`super::LsaConfig`] rules out.
pub fn ward_labels(pools: &[f32], sizes: &[u32], dim: usize, bucket_count: usize, bucket_cap: u32) -> Result<Vec<u32>, String> {
    let n = sizes.len();
    if pools.len() != n * dim {
        return Err(format!("ward: {} pool values for {n} pools of width {dim}", pools.len()));
    }
    if bucket_count == 0 || bucket_count > n {
        return Err(format!("ward: cannot merge {n} pools into {bucket_count} buckets"));
    }
    let mut sums: Vec<f64> = pools.iter().map(|x| *x as f64).collect();
    let mut size: Vec<u32> = sizes.to_vec();
    let mut alive = vec![true; n];
    let mut parent: Vec<usize> = (0..n).collect();
    let mut live = n;

    for _ in 0..MAX_ROUNDS {
        if live <= bucket_count {
            break;
        }
        let merges_left = live - bucket_count;
        let gram = live_gram(&sums, &alive, dim);
        let alignment: Vec<f64> = (0..n).map(|i| gram[i * n + i].max(0.0).sqrt()).collect();

        // Each live bucket's cheapest allowed partner; `None` when the cap
        // allows none.
        let mut best: Vec<Option<(usize, f64)>> = vec![None; n];
        for i in (0..n).filter(|&i| alive[i]) {
            for j in (0..n).filter(|&j| alive[j] && j != i) {
                if size[i] + size[j] > bucket_cap {
                    continue;
                }
                let merged = (gram[i * n + i] + gram[j * n + j] + 2.0 * gram[i * n + j]).max(0.0).sqrt();
                let cost = alignment[i] + alignment[j] - merged;
                if best[i].is_none_or(|(_, c)| cost < c) {
                    best[i] = Some((j, cost));
                }
            }
        }

        // Pairs that are each other's cheapest, listed once by their lower
        // member, cheapest first (a tie to the lower member).
        let mut mutual: Vec<(f64, usize, usize)> = (0..n)
            .filter_map(|i| {
                let (j, cost) = best[i]?;
                (i < j && best[j].is_some_and(|(back, _)| back == i)).then_some((cost, i, j))
            })
            .collect();
        if mutual.is_empty() {
            break;
        }
        mutual.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        mutual.truncate(merges_left);

        for &(_, kept, absorbed) in &mutual {
            for d in 0..dim {
                sums[kept * dim + d] += sums[absorbed * dim + d];
            }
            size[kept] += size[absorbed];
            alive[absorbed] = false;
            parent[absorbed] = kept;
        }
        live -= mutual.len();
    }
    if live != bucket_count {
        return Err(format!(
            "ward: stopped at {live} buckets, short of {bucket_count}, under a cap of {bucket_cap} keys"
        ));
    }

    // A survivor is the lowest pool of its bucket (a pair merges into its
    // lower member), so numbering survivors in index order numbers buckets by
    // their first pool.
    let mut number = vec![0u32; n];
    let mut next = 0u32;
    for i in 0..n {
        if alive[i] {
            number[i] = next;
            next += 1;
        }
    }
    Ok((0..n)
        .map(|mut i| {
            while parent[i] != i {
                i = parent[i];
            }
            number[i]
        })
        .collect())
}

/// Gram matrix `[n, n]` of the live bucket sums; dead rows are left zero.
fn live_gram(sums: &[f64], alive: &[bool], dim: usize) -> Vec<f64> {
    let n = alive.len();
    let mut gram = vec![0f64; n * n];
    for i in (0..n).filter(|&i| alive[i]) {
        let a = &sums[i * dim..][..dim];
        for j in (i..n).filter(|&j| alive[j]) {
            let b = &sums[j * dim..][..dim];
            let g: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            gram[i * n + j] = g;
            gram[j * n + i] = g;
        }
    }
    gram
}
