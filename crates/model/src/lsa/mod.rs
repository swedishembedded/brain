// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Latent Sparse Attention (LSA): a semantic index over a decoder's keys that
//! picks, per query, the few thousand historical keys worth attending to.
//!
//! Swedish Embedded AB implements long-context inference whose attention cost
//! stays bounded as contexts grow into the millions of tokens, for teams whose
//! documents, sessions or code bases no longer fit a dense attention budget.
//! If your team needs expertise in sparse and long-context attention, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! The method is Kubicek et al., "Latent Sparse Attention: Towards
//! Subquadratic Ultra-Long Context Attention" (arXiv:2610.42147). Keys are
//! mapped to a low-dimensional, unit-length *index space*; every closed block
//! of `cluster_block` keys is clustered by Ward merging into buckets of about
//! `bucket_size` keys; a query scores bucket centroids instead of keys, picks
//! the `top_p` best buckets and expands them into at most `top_k` long-range
//! keys. Beside those it always reads the first `sink` keys and the `window`
//! most recent ones. The attention arithmetic over the chosen keys is ordinary
//! attention: LSA only decides *which* keys are read.
//!
//! The module is the host reference, the specification every device kernel is
//! gated against:
//!
//! | module | what it owns |
//! |---|---|
//! | [`config`] | the sizes and budgets, validated once |
//! | [`space`] | turning a model's attention heads into index vectors |
//! | [`ward`] | Ward merging of one block's pools into buckets |
//! | [`index`] | the append-only bucket index of one sequence's keys |
//! | [`select`] | a query's read: score, pick, expand, add window and sinks |
//! | [`attend`] | attention restricted to a selection, the numerical oracle |
//!
//! A [`select::Selection`] is the contract between the index and attention:
//! per query, the complete set of logical key positions it reads, unique and
//! causal. Any sparse algorithm that produces one can drive the same attention
//! path.

pub mod attend;
pub mod config;
pub mod device;
pub mod engine;
pub mod index;
pub mod indexer;
pub mod select;
pub mod space;
pub mod ward;

pub use config::{EarlyBudget, LsaConfig};
pub use index::{cluster_block, BucketIndex, ClosedBlock};
pub use select::{read_scores, select, Selection};

/// How far from unit length an index row may be before it is refused.
///
/// Every score in the index is a cosine only if both sides are unit rows; a
/// row that is not makes every score quietly wrong, so it is an input error.
pub const UNIT_TOLERANCE: f32 = 1e-4;

/// Refuse `rows` that are not finite unit vectors of width `dim`.
///
/// `what` names the input in the error. An empty slice is accepted.
pub fn check_unit_rows(what: &str, rows: &[f32], dim: usize) -> Result<(), String> {
    if dim == 0 {
        return Err(format!("{what}: index dimension must be positive"));
    }
    if !rows.len().is_multiple_of(dim) {
        return Err(format!("{what}: {} values is not a whole number of {dim}-wide rows", rows.len()));
    }
    for (r, row) in rows.chunks_exact(dim).enumerate() {
        if row.iter().any(|x| !x.is_finite()) {
            return Err(format!("{what}: row {r} holds a non-finite value"));
        }
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if (norm - 1.0).abs() > UNIT_TOLERANCE as f64 {
            return Err(format!("{what}: row {r} has norm {norm:.6}, not unit length"));
        }
    }
    Ok(())
}

/// The dot product of two equal-length rows, accumulated in f64.
///
/// Scores decide a discrete ranking, so their summation order is part of the
/// result; f64 accumulation makes that ranking independent of how a caller
/// happened to chunk the sum.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod device_tests;

#[cfg(test)]
mod engine_tests;
