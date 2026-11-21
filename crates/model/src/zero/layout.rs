// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training for its clients. If your team needs expertise in
// partitioning a model's training state over the cards of a cluster, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The flat, deterministic partition of the trained tensors into units sharded over the ranks ([`super`]).

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// One unit of a [`ShardLayout`]: members `(tensor index, offset, count)` in its flat vector, padded to `padded`
/// elements (a multiple of `world * align`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Unit {
    pub members: Vec<(usize, u64, u64)>,
    pub padded: u64,
}

/// The part of a tensor a rank owns within one unit: `len` elements from `tensor_offset` in the tensor, at
/// `shard_offset` in the rank's slice of the unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub tensor: usize,
    pub tensor_offset: u64,
    pub shard_offset: u64,
    pub len: u64,
}

/// A flat, deterministic partition of tensors into units sharded evenly over `world` ranks (module documentation).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardLayout {
    pub names: Vec<String>,
    pub numels: Vec<u64>,
    pub world: usize,
    pub align: u64,
    pub units: Vec<Unit>,
}

impl ShardLayout {
    /// Pack tensors `names`/`numels` in backward order into units of at most `cap` elements (a larger tensor is a
    /// unit of its own); only tensors with the same `group` share a unit (element type, learning-rate multiplier).
    pub fn plan(names: &[String], numels: &[u64], groups: &[u32], cap: u64, world: usize, align: u64) -> Result<ShardLayout, String> {
        if world == 0 {
            return Err("a sharded layout needs at least one rank".into());
        }
        if names.len() != numels.len() || names.len() != groups.len() {
            return Err(format!("{} names, {} sizes and {} groups do not describe one tensor set", names.len(), numels.len(), groups.len()));
        }
        let align = align.max(1);
        let quantum = world as u64 * align;
        let mut units = Vec::new();
        let mut open: Vec<(usize, u64, u64)> = Vec::new();
        let mut open_group = None;
        let mut filled = 0u64;
        let close = |open: &mut Vec<(usize, u64, u64)>, filled: &mut u64, units: &mut Vec<Unit>| {
            if !open.is_empty() {
                units.push(Unit { members: std::mem::take(open), padded: filled.div_ceil(quantum).max(1) * quantum });
                *filled = 0;
            }
        };
        for index in (0..names.len()).rev() {
            let count = numels[index];
            if count == 0 {
                continue;
            }
            let start = filled.div_ceil(align) * align;
            if open_group != Some(groups[index]) || (filled > 0 && start + count > cap) {
                close(&mut open, &mut filled, &mut units);
                open_group = Some(groups[index]);
            }
            let start = filled.div_ceil(align) * align;
            open.push((index, start, count));
            filled = start + count;
        }
        close(&mut open, &mut filled, &mut units);
        Ok(ShardLayout { names: names.to_vec(), numels: numels.to_vec(), world, align, units })
    }

    /// Elements of unit `u` each rank owns.
    pub fn shard_len(&self, u: usize) -> u64 {
        self.units[u].padded / self.world as u64
    }

    /// The elements of unit `u`'s flat vector rank `rank` owns.
    pub fn slice(&self, u: usize, rank: usize) -> Range<u64> {
        let n = self.shard_len(u);
        rank as u64 * n..(rank as u64 + 1) * n
    }

    /// The tensors' parts rank `rank` owns in unit `u`, in member order.
    pub fn pieces(&self, u: usize, rank: usize) -> Vec<Piece> {
        let slice = self.slice(u, rank);
        self.units[u]
            .members
            .iter()
            .filter_map(|&(tensor, offset, count)| {
                let start = slice.start.max(offset);
                let end = slice.end.min(offset + count);
                (start < end).then(|| Piece { tensor, tensor_offset: start - offset, shard_offset: start - slice.start, len: end - start })
            })
            .collect()
    }

    /// Elements one rank's shards add up to.
    pub fn shard_elems(&self) -> u64 {
        (0..self.units.len()).map(|u| self.shard_len(u)).sum()
    }

    /// The largest unit, in elements: what the staging buffer holds.
    pub fn max_unit(&self) -> u64 {
        self.units.iter().map(|u| u.padded).max().unwrap_or(0)
    }

    /// Reassemble whole tensors from every rank's unit shards (`shards[rank][unit]`).
    pub fn gather(&self, shards: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>, String> {
        if shards.len() != self.world {
            return Err(format!("{} ranks' shards for a layout of {} ranks", shards.len(), self.world));
        }
        let mut tensors: Vec<Vec<f32>> = self.numels.iter().map(|&n| vec![0.0; n as usize]).collect();
        for (rank, units) in shards.iter().enumerate() {
            if units.len() != self.units.len() {
                return Err(format!("rank {rank} holds {} unit shards, the layout has {}", units.len(), self.units.len()));
            }
            for (u, shard) in units.iter().enumerate() {
                if shard.len() as u64 != self.shard_len(u) {
                    return Err(format!("rank {rank} unit {u}: {} elements, the layout's shard holds {}", shard.len(), self.shard_len(u)));
                }
                for p in self.pieces(u, rank) {
                    let (t, s) = (p.tensor_offset as usize, p.shard_offset as usize);
                    tensors[p.tensor][t..t + p.len as usize].copy_from_slice(&shard[s..s + p.len as usize]);
                }
            }
        }
        Ok(tensors)
    }

    /// Rank `rank`'s unit shards of whole `tensors` (padding zero).
    pub fn split(&self, tensors: &[Vec<f32>], rank: usize) -> Result<Vec<Vec<f32>>, String> {
        for (i, t) in tensors.iter().enumerate() {
            if t.len() as u64 != self.numels[i] {
                return Err(format!("{}: {} elements, the layout's tensor has {}", self.names[i], t.len(), self.numels[i]));
            }
        }
        Ok((0..self.units.len())
            .map(|u| {
                let mut shard = vec![0.0; self.shard_len(u) as usize];
                for p in self.pieces(u, rank) {
                    let (t, s) = (p.tensor_offset as usize, p.shard_offset as usize);
                    shard[s..s + p.len as usize].copy_from_slice(&tensors[p.tensor][t..t + p.len as usize]);
                }
                shard
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zero::Sharding;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("t{i}")).collect()
    }

    /// Every element of every tensor is owned by exactly one rank, units respect the cap (unless one tensor is
    /// larger), are padded to the rank quantum, and members start on the alignment.
    #[test]
    fn a_layout_partitions_every_element_once() {
        let numels = [10u64, 3, 0, 64, 7, 100, 2];
        for (world, align, cap) in [(1, 1, 16), (2, 2, 16), (3, 1, 1000), (4, 2, 20)] {
            let numels: Vec<u64> = numels.iter().map(|n| n.div_ceil(align) * align).collect();
            let layout = ShardLayout::plan(&names(numels.len()), &numels, &vec![0; numels.len()], cap, world, align).unwrap();
            let mut owned: Vec<Vec<u32>> = numels.iter().map(|&n| vec![0; n as usize]).collect();
            for (u, unit) in layout.units.iter().enumerate() {
                assert_eq!(unit.padded % (world as u64 * align), 0);
                assert!(unit.members.len() == 1 || unit.members.last().map(|m| m.1 + m.2).unwrap() <= cap, "unit {u} over the cap");
                for &(_, offset, _) in &unit.members {
                    assert_eq!(offset % align, 0);
                }
                for rank in 0..world {
                    for p in layout.pieces(u, rank) {
                        for i in p.tensor_offset..p.tensor_offset + p.len {
                            owned[p.tensor][i as usize] += 1;
                        }
                    }
                }
            }
            assert!(owned.iter().all(|t| t.iter().all(|&c| c == 1)), "world {world}: {owned:?}");
            assert_eq!(layout, ShardLayout::plan(&names(numels.len()), &numels, &vec![0; numels.len()], cap, world, align).unwrap());
        }
    }

    /// Splitting whole tensors by one layout and gathering them by it is the identity; through another world size
    /// too.
    #[test]
    fn split_then_gather_is_the_identity_across_world_sizes() {
        let numels = vec![6u64, 9, 31, 4];
        let tensors: Vec<Vec<f32>> = numels.iter().enumerate().map(|(t, &n)| (0..n).map(|i| (t * 100) as f32 + i as f32).collect()).collect();
        let mut prev: Option<Vec<Vec<f32>>> = None;
        for world in [1, 2, 3, 4] {
            let layout = ShardLayout::plan(&names(4), &numels, &[0, 0, 1, 0], 16, world, 1).unwrap();
            let shards: Vec<Vec<Vec<f32>>> = (0..world).map(|r| layout.split(&tensors, r).unwrap()).collect();
            let back = layout.gather(&shards).unwrap();
            assert_eq!(back, tensors);
            if let Some(p) = &prev {
                assert_eq!(p, &back);
            }
            prev = Some(back);
        }
    }

    /// Tensors of different groups never share a unit.
    #[test]
    fn groups_never_share_a_unit() {
        let layout = ShardLayout::plan(&names(4), &[2, 2, 2, 2], &[0, 1, 0, 1], 1000, 2, 1).unwrap();
        for unit in &layout.units {
            let g: Vec<usize> = unit.members.iter().map(|m| m.0 % 2).collect();
            assert!(g.windows(2).all(|w| w[0] == w[1]), "{unit:?}");
        }
    }

    #[test]
    fn stages_parse_by_number_and_name() {
        assert_eq!(Sharding::parse("2").unwrap(), Sharding::Gradients);
        assert_eq!(Sharding::parse("optimizer").unwrap(), Sharding::Optimizer);
        assert!(Sharding::parse("4").is_err());
    }
}
