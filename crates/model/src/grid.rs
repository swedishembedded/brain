// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3D process grid — the rank ↔ (tensor, pipeline, data) coordinate mapping that
//! composes the three parallelism dimensions.
//!
//! A world of `tp·pp·dp` ranks is laid out so **tensor-parallel peers are
//! adjacent** (ranks `0..tp` are one TP group), then pipeline, then data:
//!
//! ```text
//! rank = (dp_rank · pp + pp_rank) · tp + tp_rank
//! ```
//!
//! This is the Megatron PTD-P placement: TP needs the tightest coupling (an
//! all-reduce *per layer*) so its peers should share a node / NVLink and get the
//! lowest global-rank stride; pipeline crosses nodes (a residual per stage
//! boundary); data-parallel is outermost (one grad all-reduce per step). The same
//! mapping works whether ranks are threads on one box or processes across a
//! cluster — each dimension's peer set becomes a [`crate::Collective`] group, so
//! the transport (host-staged today, network later) is orthogonal to the layout.

/// A rank's coordinate in the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Coord {
    pub tp: usize,
    pub pp: usize,
    pub dp: usize,
}

/// A `tp × pp × dp` process grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub tp: usize,
    pub pp: usize,
    pub dp: usize,
}

impl Grid {
    pub fn new(tp: usize, pp: usize, dp: usize) -> Grid {
        assert!(tp >= 1 && pp >= 1 && dp >= 1, "grid dims must be >= 1");
        Grid { tp, pp, dp }
    }

    pub fn world_size(&self) -> usize {
        self.tp * self.pp * self.dp
    }

    /// Coordinate of a global rank (TP fastest, then PP, then DP).
    pub fn coord(&self, rank: usize) -> Coord {
        debug_assert!(rank < self.world_size(), "rank out of range");
        Coord { tp: rank % self.tp, pp: (rank / self.tp) % self.pp, dp: rank / (self.tp * self.pp) }
    }

    /// Global rank of a coordinate (inverse of [`Self::coord`]).
    pub fn rank(&self, c: Coord) -> usize {
        debug_assert!(c.tp < self.tp && c.pp < self.pp && c.dp < self.dp, "coord out of range");
        (c.dp * self.pp + c.pp) * self.tp + c.tp
    }

    /// The ranks in this rank's **tensor-parallel** group (same pp+dp, all tp),
    /// in ascending tp order. Length `tp`; `rank` sits at index `coord(rank).tp`.
    pub fn tp_group(&self, rank: usize) -> Vec<usize> {
        let c = self.coord(rank);
        (0..self.tp).map(|tp| self.rank(Coord { tp, ..c })).collect()
    }

    /// The ranks in this rank's **pipeline** group (same tp+dp, all pp), in stage
    /// order. Length `pp`.
    pub fn pp_group(&self, rank: usize) -> Vec<usize> {
        let c = self.coord(rank);
        (0..self.pp).map(|pp| self.rank(Coord { pp, ..c })).collect()
    }

    /// The ranks in this rank's **data-parallel** group (same tp+pp, all dp), in
    /// replica order. Length `dp`.
    pub fn dp_group(&self, rank: usize) -> Vec<usize> {
        let c = self.coord(rank);
        (0..self.dp).map(|dp| self.rank(Coord { dp, ..c })).collect()
    }
}

// ---- in-process realisation: one collective per group in each dimension --------

use crate::collective::HostCollective;
use std::sync::Arc;

/// In-process realisation of a [`Grid`]: one [`HostCollective`] per group in each
/// dimension, so rank `r` can reach its tensor / pipeline / data peers. A future
/// networked realisation builds the same per-group communicators over sockets —
/// callers use the returned [`Collective`](crate::Collective) the same way.
pub struct LocalGroups {
    grid: Grid,
    tp: Vec<(Arc<HostCollective>, usize)>, // per rank: (its TP collective, local index in group)
    pp: Vec<(Arc<HostCollective>, usize)>,
    dp: Vec<(Arc<HostCollective>, usize)>,
}

impl LocalGroups {
    pub fn new(grid: Grid) -> LocalGroups {
        let world = grid.world_size();
        let build = |group_of: &dyn Fn(usize) -> Vec<usize>, size: usize| {
            let mut out: Vec<Option<(Arc<HostCollective>, usize)>> = vec![None; world];
            for r in 0..world {
                if out[r].is_some() {
                    continue;
                }
                let members = group_of(r);
                let coll = HostCollective::new(size);
                for (i, &m) in members.iter().enumerate() {
                    out[m] = Some((coll.clone(), i));
                }
            }
            out.into_iter().map(Option::unwrap).collect::<Vec<_>>()
        };
        let tp = build(&|r| grid.tp_group(r), grid.tp);
        let pp = build(&|r| grid.pp_group(r), grid.pp);
        let dp = build(&|r| grid.dp_group(r), grid.dp);
        LocalGroups { grid, tp, pp, dp }
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }
    /// Rank `r`'s tensor-parallel collective + its local index within the group.
    pub fn tp(&self, r: usize) -> (Arc<HostCollective>, usize) {
        self.tp[r].clone()
    }
    /// Rank `r`'s pipeline collective + its local (stage) index.
    pub fn pp(&self, r: usize) -> (Arc<HostCollective>, usize) {
        self.pp[r].clone()
    }
    /// Rank `r`'s data-parallel collective + its local (replica) index.
    pub fn dp(&self, r: usize) -> (Arc<HostCollective>, usize) {
        self.dp[r].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Collective;

    #[test]
    fn coord_rank_roundtrip_over_whole_world() {
        for &(tp, pp, dp) in &[(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2), (4, 3, 2), (2, 1, 3)] {
            let g = Grid::new(tp, pp, dp);
            let mut seen = vec![false; g.world_size()];
            for (r, slot) in seen.iter_mut().enumerate() {
                let c = g.coord(r);
                assert_eq!(g.rank(c), r, "roundtrip {r} in {g:?}");
                assert!(c.tp < tp && c.pp < pp && c.dp < dp);
                assert!(!*slot, "coord not unique");
                *slot = true;
            }
            assert!(seen.into_iter().all(|x| x), "grid covers every rank exactly once");
        }
    }

    #[test]
    fn tp_peers_are_adjacent() {
        // TP fastest => ranks 0..tp are one TP group (tightest coupling).
        let g = Grid::new(4, 2, 2); // world 16
        assert_eq!(g.tp_group(0), vec![0, 1, 2, 3]);
        assert_eq!(g.tp_group(5), vec![4, 5, 6, 7]);
        // this rank's index within its TP group equals its tp coord
        for r in 0..g.world_size() {
            assert_eq!(g.tp_group(r)[g.coord(r).tp], r);
        }
    }

    #[test]
    fn pipeline_group_strides_by_tp() {
        let g = Grid::new(4, 2, 2);
        // pp peers of rank 1 (tp=1,pp=0,dp=0): pp=0 -> rank 1, pp=1 -> rank 1+4=5
        assert_eq!(g.pp_group(1), vec![1, 5]);
    }

    #[test]
    fn data_group_strides_by_tp_times_pp() {
        let g = Grid::new(4, 2, 2);
        // dp peers of rank 1: stride tp*pp = 8 -> ranks 1 and 9
        assert_eq!(g.dp_group(1), vec![1, 9]);
    }

    #[test]
    fn groups_have_the_right_sizes_and_shared_coords() {
        let g = Grid::new(2, 3, 2); // world 12
        for r in 0..g.world_size() {
            let c = g.coord(r);
            let tpg = g.tp_group(r);
            let ppg = g.pp_group(r);
            let dpg = g.dp_group(r);
            assert_eq!(tpg.len(), 2);
            assert_eq!(ppg.len(), 3);
            assert_eq!(dpg.len(), 2);
            // every TP peer shares this rank's (pp,dp); etc.
            for &p in &tpg {
                assert_eq!((g.coord(p).pp, g.coord(p).dp), (c.pp, c.dp));
            }
            for &p in &ppg {
                assert_eq!((g.coord(p).tp, g.coord(p).dp), (c.tp, c.dp));
            }
            for &p in &dpg {
                assert_eq!((g.coord(p).tp, g.coord(p).pp), (c.tp, c.pp));
            }
        }
    }

    #[test]
    fn degenerate_1d_grids() {
        let g = Grid::new(1, 1, 1);
        assert_eq!(g.world_size(), 1);
        assert_eq!(g.tp_group(0), vec![0]);
        assert_eq!(g.pp_group(0), vec![0]);
        assert_eq!(g.dp_group(0), vec![0]);
    }

    // ---- LocalGroups: grid -> per-group collectives ----

    #[test]
    fn peers_share_a_collective_non_peers_do_not() {
        let lg = LocalGroups::new(Grid::new(2, 2, 2)); // world 8
        // TP peers 0 and 1 share their TP collective (local idx 0,1); rank 2 is a
        // different TP group.
        let (c0, i0) = lg.tp(0);
        let (c1, i1) = lg.tp(1);
        let (c2, _) = lg.tp(2);
        assert!(Arc::ptr_eq(&c0, &c1), "TP peers share the collective");
        assert!(!Arc::ptr_eq(&c0, &c2), "different TP group => different collective");
        assert_eq!((i0, i1), (0, 1), "local ranks within the TP group");
        // DP peers of rank 0 are 0 and 4 (stride tp*pp=4); they share the DP collective.
        let (d0, _) = lg.dp(0);
        let (d4, l4) = lg.dp(4);
        assert!(Arc::ptr_eq(&d0, &d4));
        assert_eq!(l4, 1);
    }

    #[test]
    fn tp_collective_all_reduces_within_the_group() {
        // (2,1,1): one TP group of 2. Drive an all-reduce through the grid-derived
        // collective from two threads.
        let lg = LocalGroups::new(Grid::new(2, 1, 1));
        let out: Vec<std::sync::Mutex<Vec<f32>>> = (0..2).map(|_| std::sync::Mutex::new(Vec::new())).collect();
        std::thread::scope(|s| {
            for r in 0..2usize {
                let (lg, out) = (&lg, &out);
                s.spawn(move || {
                    let (coll, local) = lg.tp(r);
                    let z = coll.all_reduce(local, vec![(r + 1) as f32, 10.0]);
                    *out[r].lock().unwrap() = z;
                });
            }
        });
        // ranks contribute [1,10] and [2,10] => sum [3,20] on both.
        for m in out {
            assert_eq!(m.into_inner().unwrap(), vec![3.0, 20.0]);
        }
    }

    #[test]
    fn independent_tp_groups_reduce_independently() {
        // (2,2,1): two TP groups {0,1} and {2,3}. Each reduces only within itself.
        let lg = LocalGroups::new(Grid::new(2, 2, 1));
        let out: Vec<std::sync::Mutex<Vec<f32>>> = (0..4).map(|_| std::sync::Mutex::new(Vec::new())).collect();
        std::thread::scope(|s| {
            for r in 0..4usize {
                let (lg, out) = (&lg, &out);
                s.spawn(move || {
                    let (coll, local) = lg.tp(r);
                    *out[r].lock().unwrap() = coll.all_reduce(local, vec![r as f32]);
                });
            }
        });
        let r: Vec<Vec<f32>> = out.into_iter().map(|m| m.into_inner().unwrap()).collect();
        assert_eq!(r[0], vec![1.0]); // 0+1
        assert_eq!(r[1], vec![1.0]);
        assert_eq!(r[2], vec![5.0]); // 2+3
        assert_eq!(r[3], vec![5.0]);
    }
}
