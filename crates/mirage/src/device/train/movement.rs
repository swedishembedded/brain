// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Data movement on the device tape: weighted segment plans, concatenation and slices.
//!
//! A plan's segments may overlap in the destination (a mean pools several sources into one row) and, for its
//! transpose, in the source (an embedding row used by many tokens). Both directions run as the same owner-side
//! gather ([`PlanTables`]): the owner axis is cut at every segment boundary into elementary intervals, each with the
//! ordered list of segments covering it, and one thread sums one owned element. No element has two writers, and the
//! summation order is the segment order, so the result is deterministic.

use gpu_core::DeviceBuffer;

use super::DevTape;
use crate::train::ops::Seg;
use crate::train::tape::Id;

/// Host-built tables of one plan direction, uploaded for `mirage_t_plan`.
struct PlanTables {
    ivals: DeviceBuffer,
    list: DeviceBuffer,
    covered: usize,
    n_ivals: usize,
}

impl PlanTables {
    /// Tables for `segs` given as `(owner offset, other offset, len, weight)`.
    fn build(dev: &super::Device, segs: &[(usize, usize, usize, f32)]) -> PlanTables {
        let live: Vec<&(usize, usize, usize, f32)> = segs.iter().filter(|s| s.2 > 0).collect();
        let mut cuts: Vec<usize> = live.iter().flat_map(|s| [s.0, s.0 + s.2]).collect();
        cuts.sort_unstable();
        cuts.dedup();
        // `cover[i]` lists the segments (in segment order) covering the elementary interval `cuts[i]..cuts[i + 1]`.
        let mut cover: Vec<Vec<usize>> = vec![Vec::new(); cuts.len().saturating_sub(1)];
        for (si, s) in live.iter().enumerate() {
            let first = cuts.partition_point(|&c| c < s.0);
            let last = cuts.partition_point(|&c| c < s.0 + s.2);
            cover[first..last].iter_mut().for_each(|c| c.push(si));
        }
        let (mut ivals, mut list) = (Vec::new(), Vec::new());
        let mut covered = 0usize;
        for (i, segs_here) in cover.iter().enumerate().filter(|(_, c)| !c.is_empty()) {
            ivals.extend([cuts[i], covered, list.len() / 3, list.len() / 3 + segs_here.len()].map(super::launch::word));
            covered += cuts[i + 1] - cuts[i];
            for &si in segs_here {
                let s = live[si];
                list.extend([super::launch::word(s.0), super::launch::word(s.1), s.3.to_bits()]);
            }
        }
        let n_ivals = ivals.len() / 4;
        let upload = |words: &[u32]| {
            let buf = dev.gpu().storage(words.len().max(1) as u64);
            if !words.is_empty() {
                dev.gpu().write(&buf, words);
            }
            buf
        };
        PlanTables { ivals: upload(&ivals), list: upload(&list), covered, n_ivals }
    }
}

impl DevTape<'_> {
    /// `out = sum of weighted segment copies of src`, `out_len` elements.
    pub(super) fn op_plan(&mut self, src: Id, out_len: usize, segs: Vec<Seg<f32>>) -> Id {
        let n = self.len_of(src);
        for s in &segs {
            assert!(s.dst + s.len <= out_len && s.src + s.len <= n, "plan: a segment leaves its buffer");
        }
        let dev = self.dev;
        let fwd: Vec<_> = segs.iter().map(|s| (s.dst, s.src, s.len, s.w)).collect();
        let t = PlanTables::build(dev, &fwd);
        let sb = self.buf(src);
        // Uncovered outputs stay zero; covered ones are accumulated onto zero.
        let out = self.rec.alloc_zeroed(out_len);
        if t.covered > 0 {
            self.rec.plan(&t.ivals, &t.list, &sb, &out, t.covered, t.n_ivals);
        }
        self.op(out, out_len, &[src], move |cx, gy| {
            let Some(gx) = cx.target(src) else { return };
            let bwd: Vec<_> = segs.iter().map(|s| (s.src, s.dst, s.len, s.w)).collect();
            let t = PlanTables::build(cx.rec.dev, &bwd);
            if t.covered > 0 {
                cx.rec.plan(&t.ivals, &t.list, gy, &gx, t.covered, t.n_ivals);
            }
        })
    }

    /// `a` then `b` along the row axis.
    pub(super) fn op_concat(&mut self, a: Id, b: Id) -> Id {
        let (na, nb) = (self.len_of(a), self.len_of(b));
        let (ab, bb) = (self.buf(a), self.buf(b));
        let out = self.rec.alloc(na + nb);
        self.rec.span(&ab, &out, na, 0, 0, false);
        self.rec.span(&bb, &out, nb, 0, na, false);
        self.op(out, na + nb, &[a, b], move |cx, gy| {
            if let Some(g) = cx.target(a) {
                cx.rec.span(gy, &g, na, 0, 0, true);
            }
            if let Some(g) = cx.target(b) {
                cx.rec.span(gy, &g, nb, na, 0, true);
            }
        })
    }

    /// Elements `[start, start + len)` of `x`.
    pub(super) fn op_slice(&mut self, x: Id, start: usize, len: usize) -> Id {
        let n = self.len_of(x);
        assert!(start + len <= n, "slice: the range leaves the buffer");
        let xb = self.buf(x);
        let out = self.rec.alloc(len);
        self.rec.span(&xb, &out, len, start, 0, false);
        self.op(out, len, &[x], move |cx, gy| {
            if let Some(g) = cx.target(x) {
                cx.rec.span(gy, &g, len, 0, start, true);
            }
        })
    }
}
