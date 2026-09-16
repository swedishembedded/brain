// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The connectome, stored the way the engine can actually run it.

/// A connectome in compressed-sparse-column form: every neuron owns a
/// contiguous range of its INCOMING edges.
///
/// The transposition is the point. Spike propagation reads naturally as "for
/// each neuron that fired, add its weight to every target", which is a
/// scatter-add and wants an atomic this engine does not have. Stored by
/// postsynaptic neuron instead, each neuron reads its own edges and writes its
/// own output: no two workgroups touch the same address, so nothing needs to
/// be locked. See `crates/kernels/wgsl/syn_gather_csc.wgsl`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Csc {
    /// Neuron count.
    pub n: u32,
    /// Column starts, length `n + 1`. Neuron `i`'s edges are
    /// `indptr[i]..indptr[i + 1]`.
    pub indptr: Vec<u32>,
    /// Presynaptic neuron of each edge, length `nnz`.
    pub pre: Vec<u32>,
    /// Signed synaptic weight of each edge, length `nnz`.
    pub w: Vec<f32>,
}

impl Csc {
    pub fn nnz(&self) -> usize {
        self.pre.len()
    }

    /// Multiply every edge by a factor belonging to its POSTSYNAPTIC neuron.
    ///
    /// The graph is stored by column, so a per-postsynaptic factor is a pure
    /// row scaling and costs one pass. It is how per-neuron excitability is
    /// expressed without a per-neuron threshold in the kernel: scaling every
    /// input a neuron receives by `k` is exactly equivalent to dividing its
    /// threshold by `k`, and it keeps `lif_step`'s uniform-parameter form.
    ///
    /// A caller applying this must scale any EXTERNAL current it injects into
    /// that neuron by the same factor, or the two input paths end up on
    /// different scales.
    pub fn scale_by_post(&mut self, factor: &[f32]) -> Result<(), String> {
        if factor.len() != self.n as usize {
            return Err(format!("scale_by_post needs one factor per neuron: {} for {}", factor.len(), self.n));
        }
        for (post, f) in factor.iter().enumerate() {
            let (a, b) = (self.indptr[post] as usize, self.indptr[post + 1] as usize);
            for w in &mut self.w[a..b] {
                *w *= f;
            }
        }
        Ok(())
    }

    /// Build from an edge list of `(pre, post, weight)`, counting-sorted into
    /// columns. Edge order within a column follows the input order, which is
    /// what makes the fp32 accumulation in `syn_gather_csc` reproducible:
    /// the same edge list always produces the same summation order, so a
    /// restored snapshot replays bit-for-bit.
    pub fn from_edges(n: u32, edges: &[(u32, u32, f32)]) -> Result<Csc, String> {
        let nn = n as usize;
        let mut counts = vec![0u32; nn + 1];
        for &(pre, post, _) in edges {
            if pre >= n || post >= n {
                return Err(format!("edge ({pre} -> {post}) is out of range for {n} neurons"));
            }
            counts[post as usize + 1] += 1;
        }
        for i in 0..nn {
            counts[i + 1] += counts[i];
        }
        let indptr = counts.clone();
        let mut cursor = counts;
        let mut pre_idx = vec![0u32; edges.len()];
        let mut w = vec![0.0f32; edges.len()];
        for &(pre, post, weight) in edges {
            let slot = cursor[post as usize] as usize;
            pre_idx[slot] = pre;
            w[slot] = weight;
            cursor[post as usize] += 1;
        }
        let csc = Csc { n, indptr, pre: pre_idx, w };
        csc.validate()?;
        Ok(csc)
    }

    /// Structural checks, run at construction and again at load: a malformed
    /// `indptr` does not fail a kernel, it silently reads another neuron's
    /// edges, so this is the one place it can be caught at all.
    pub fn validate(&self) -> Result<(), String> {
        if self.indptr.len() != self.n as usize + 1 {
            return Err(format!("indptr has {} entries, expected n + 1 = {}", self.indptr.len(), self.n as usize + 1));
        }
        if self.pre.len() != self.w.len() {
            return Err(format!("pre has {} entries but w has {}", self.pre.len(), self.w.len()));
        }
        if self.indptr.first() != Some(&0) {
            return Err("indptr must start at 0".to_string());
        }
        if self.indptr.last() != Some(&(self.nnz() as u32)) {
            return Err(format!("indptr ends at {:?}, expected nnz = {}", self.indptr.last(), self.nnz()));
        }
        for i in 1..self.indptr.len() {
            if self.indptr[i] < self.indptr[i - 1] {
                return Err(format!("indptr is not monotonic at {i}"));
            }
        }
        if let Some(bad) = self.pre.iter().find(|&&p| p >= self.n) {
            return Err(format!("presynaptic index {bad} is out of range for {} neurons", self.n));
        }
        Ok(())
    }

    /// The same graph with every edge's SOURCE randomly reassigned, keeping
    /// each neuron's in-degree and the multiset of weights exactly.
    ///
    /// The structural control. If a creature learns as well on this as on the
    /// real connectome, then nothing about the published wiring mattered and
    /// the result is about the learning rule, the body, or the reward - which
    /// is a finding, but not the one anybody wants to claim.
    ///
    /// Only `pre` is permuted, so `indptr` is untouched and in-degree is
    /// preserved neuron for neuron. Out-degree is NOT preserved: it becomes
    /// binomial where the real graph is heavy-tailed. That asymmetry is
    /// unavoidable - a shuffle preserving both degree sequences is a different
    /// and much more expensive object (an edge-swap walk) - and it is the
    /// honest limitation of this control rather than a detail to omit. What it
    /// does destroy is every correlation between who a neuron listens to and
    /// what it is, which is the thing being tested.
    pub fn shuffled_sources(&self, seed: u64) -> Csc {
        let mut out = self.clone();
        // Fisher-Yates over the `pre` array, with this workspace's
        // deterministic test PRNG shape so a shuffled run is reproducible.
        let mut state = seed | 1;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for i in (1..out.pre.len()).rev() {
            let j = next() % (i + 1);
            out.pre.swap(i, j);
        }
        out
    }

    /// In-degree of every neuron, straight out of `indptr`. The degree
    /// distribution is what a connectome import gates itself on, and it costs
    /// nothing to read here rather than recomputing it from an edge list.
    pub fn in_degrees(&self) -> Vec<u32> {
        (0..self.n as usize).map(|i| self.indptr[i + 1] - self.indptr[i]).collect()
    }
}
