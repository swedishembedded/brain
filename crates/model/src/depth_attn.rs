// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements depth-attention residual streams for its clients. If your team needs expertise in
// decoders whose layers attend over the model's own earlier states then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Depth attention: a softmax over a stack of retained states, mixed by learned queries (block attention residuals).
//!
//! Per row (token), the stack is `[s_0, .., s_{n-2}, partial]`: states a forward retained in a
//! [`crate::saved::RetainedStates`] store (selected by slot, in the order given) and the current partial state. Each is
//! RMS-normed without a gain in f32, `n_i = s_i * rsqrt(mean(s_i^2) + eps)`, and each of `nq` learned queries (shared
//! by every row) reads
//!
//! ```text
//! z[q,i] = scale * <query[q], n_i>,  w[q,:] = softmax_i z[q,:],  out[q] = sum_i w[q,i] n_i
//! ```
//!
//! With [`DepthAttn::with_row_queries`] each row has its own queries instead (`[nq, rows, width]`): a served batch whose
//! rows run under different adapters, each with its own effective query matrix. That form is forward-only.
//!
//! The output mixes the NORMED states. A layer reads its input this way instead of from the plain residual stream;
//! a plain residual is the degenerate stack whose softmax puts all mass on the partial. The backward fans the
//! gradient into every state of the stack: into the store's per-slot gradient, which every consumer of a retained
//! state adds into, and into the partial's gradient (also added, since the partial is a residual the caller's own
//! backward writes too), and adds the queries' gradient (a looped layer's queries are used once per visit).
//!
//! Kernels: `depth_attn_fwd` (norm, scores, softmax and mix in one pass per row, saving per-row RMS and weights),
//! `depth_attn_bwd` (state gradients and score gradients) and `depth_attn_bwd_query` (the query gradient, reduced over
//! rows). The host reference is [`crate::hostmath::depth_attention`].

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::saved::RetainedStates;

/// The most retained states one stack reads (the partial comes on top).
pub const MAX_RETAINED: usize = 32;

/// The most queries one consumer mixes with.
pub const MAX_QUERIES: usize = 4;

/// Kernels depth attention dispatches; append to the model's kernel list (with
/// [`crate::saved::retained_kernels`] for the store).
#[must_use]
pub fn kernels() -> &'static [(&'static str, &'static str)] {
    &[("depth_attn_fwd", kernels::DEPTH_ATTN_FWD), ("depth_attn_bwd", kernels::DEPTH_ATTN_BWD), ("depth_attn_bwd_query", kernels::DEPTH_ATTN_BWD_QUERY)]
}

/// The kernel slots of [`kernels`] on one device.
#[derive(Clone, Copy, Debug)]
pub struct DepthAttnIds {
    pub fwd: usize,
    pub bwd: usize,
    pub bwd_query: usize,
}

impl DepthAttnIds {
    pub fn resolve(gpu: &Gpu) -> Result<DepthAttnIds, String> {
        let k = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("depth attention: kernel {name} is not registered (append model::depth_attn::kernels() to the kernel list)"));
        Ok(DepthAttnIds { fwd: k("depth_attn_fwd")?, bwd: k("depth_attn_bwd")?, bwd_query: k("depth_attn_bwd_query")? })
    }
}

/// One consumer of a [`RetainedStates`] store: which slots its stack reads (in stack order), how many queries mix it,
/// and the norm's epsilon. The softmax scale is `width^-0.5`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAttn {
    rows: u32,
    dim: u32,
    capacity: u32,
    slots: Vec<u32>,
    nq: u32,
    eps: f32,
    /// Queries per row (`[nq, rows, width]`) instead of shared (`[nq, width]`).
    row_queries: bool,
}

impl DepthAttn {
    /// A consumer reading `slots` of `store` (distinct, each below the store's capacity, at most [`MAX_RETAINED`])
    /// plus the partial, mixed by `nq` queries (1 to [`MAX_QUERIES`]).
    pub fn new(store: &RetainedStates, slots: &[u32], nq: u32, eps: f32) -> Result<DepthAttn, String> {
        if slots.len() > MAX_RETAINED {
            return Err(format!("depth attention: {} retained states exceed {MAX_RETAINED}", slots.len()));
        }
        if !(1..=MAX_QUERIES as u32).contains(&nq) {
            return Err(format!("depth attention: {nq} queries, want 1..={MAX_QUERIES}"));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(format!("depth attention: eps {eps} must be positive"));
        }
        for (i, s) in slots.iter().enumerate() {
            if *s >= store.capacity() {
                return Err(format!("depth attention: slot {s} out of the store's 0..{}", store.capacity()));
            }
            if slots[..i].contains(s) {
                return Err(format!("depth attention: slot {s} appears twice in one stack"));
            }
        }
        Ok(DepthAttn { rows: store.rows(), dim: store.width(), capacity: store.capacity(), slots: slots.to_vec(), nq, eps, row_queries: false })
    }

    /// This consumer with a query set per row: [`Self::forward`] reads `query` as `[nq, rows, width]`, row `r`'s query
    /// `q` at `(q * rows + r) * width` (the layout of the output). Forward only: [`Self::backward`] refuses it.
    #[must_use]
    pub fn with_row_queries(mut self) -> DepthAttn {
        self.row_queries = true;
        self
    }

    /// Elements of the `query` buffer [`Self::forward`] reads: `nq * width`, times the rows with per-row queries.
    pub fn query_len(&self) -> u64 {
        self.nq as u64 * self.dim as u64 * if self.row_queries { self.rows as u64 } else { 1 }
    }

    /// States in the stack: the retained slots plus the partial.
    pub fn states(&self) -> usize {
        self.slots.len() + 1
    }

    pub fn queries(&self) -> u32 {
        self.nq
    }

    /// Elements of the per-row statistics the forward saves for the backward (`rows * n * (1 + nq)`).
    pub fn stats_len(&self) -> u64 {
        self.rows as u64 * self.states() as u64 * (1 + self.nq as u64)
    }

    /// Elements of the score-gradient scratch the backward writes (`rows * nq * n`).
    pub fn dz_len(&self) -> u64 {
        self.rows as u64 * self.nq as u64 * self.states() as u64
    }

    /// Elements of the output (`nq * rows * width`): query `q`'s mix is rows `q * rows ..`.
    pub fn out_len(&self) -> u64 {
        self.nq as u64 * self.rows as u64 * self.dim as u64
    }

    fn params(&self) -> Vec<u32> {
        let scale = (self.dim as f32).powf(-0.5);
        let mut p = vec![self.rows, self.dim, self.slots.len() as u32, self.nq, self.eps.to_bits(), scale.to_bits(), u32::from(self.row_queries), 0];
        p.extend(self.slots.iter().copied().chain(std::iter::repeat(0)).take(MAX_RETAINED));
        p
    }

    fn check_store(&self, store: &RetainedStates) {
        assert!(
            store.rows() == self.rows && store.width() == self.dim && store.capacity() == self.capacity,
            "depth attention: built for a {} x {} x {} store, given {} x {} x {}",
            self.capacity,
            self.rows,
            self.dim,
            store.capacity(),
            store.rows(),
            store.width()
        );
    }

    /// The forward: `out` (`[nq, rows, width]`) and the saved `stats` ([`Self::stats_len`]) from the stack of `store`'s
    /// slots plus `partial` (`[rows, width]`), mixed by `query` (`[nq, width]`, or `[nq, rows, width]` with
    /// [`Self::with_row_queries`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, gpu: &Gpu, ids: &DepthAttnIds, store: &RetainedStates, partial: &DeviceBuffer, query: &DeviceBuffer, out: &DeviceBuffer, stats: &DeviceBuffer) -> Step {
        self.check_store(store);
        gpu.step(ids.fwd, &[store.values(), partial, query, out, stats], &self.params(), self.rows)
    }

    /// The backward of [`Self::forward`] given `dout` (`[nq, rows, width]`): ADDS each retained state's gradient into
    /// the store's slot gradient, the partial's into `dpartial`, and the queries' into `dquery` (`[nq, width]`);
    /// `dz` ([`Self::dz_len`]) is scratch. `stats` is the forward's.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        gpu: &Gpu,
        ids: &DepthAttnIds,
        store: &RetainedStates,
        partial: &DeviceBuffer,
        query: &DeviceBuffer,
        stats: &DeviceBuffer,
        dout: &DeviceBuffer,
        dpartial: &DeviceBuffer,
        dz: &DeviceBuffer,
        dquery: &DeviceBuffer,
    ) -> Vec<Step> {
        self.check_store(store);
        assert!(!self.row_queries, "depth attention backward: per-row queries are a serving form; a trained query matrix is shared by every row");
        let grads = store.grads().expect("depth attention backward: an inference store has no gradients");
        let params = self.params();
        vec![
            gpu.step(ids.bwd, &[store.values(), partial, query, stats, dout, grads, dpartial, dz], &params, self.rows),
            gpu.step(ids.bwd_query, &[store.values(), partial, stats, dz, dquery], &params, self.nq * self.dim),
        ]
    }
}
