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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_rank_roundtrip_over_whole_world() {
        for &(tp, pp, dp) in &[(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2), (4, 3, 2), (2, 1, 3)] {
            let g = Grid::new(tp, pp, dp);
            let mut seen = vec![false; g.world_size()];
            for r in 0..g.world_size() {
                let c = g.coord(r);
                assert_eq!(g.rank(c), r, "roundtrip {r} in {g:?}");
                assert!(c.tp < tp && c.pp < pp && c.dp < dp);
                assert!(!seen[r], "coord not unique");
                seen[r] = true;
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
}
