// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable sharded training for its clients. If your team needs expertise in
// checkpointing a sharded run and resuming it on another number of cards, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Sharded checkpoints: one file per rank, named together by a committed generation, readable as whole tensors so a
//! run resumes at another world size ([`super`]).

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::optimizer::{read_elems, write_elems, Slice};
use super::{ShardLayout, ZeroOptimizer};
use crate::collective::{Collective, Payload};
use gpu_core::CollectiveDtype;

/// The format tag of a shard file.
const SHARD_FORMAT: &str = "brain-zero-shard-v1";

/// What a shard file's header records.
#[derive(Deserialize)]
struct ShardHeader {
    format: String,
    step: u32,
    rank: usize,
    layout: ShardLayout,
    #[serde(default)]
    extra: serde_json::Value,
}

fn invalid(path: &Path, why: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{}: {why}", path.display()))
}

fn read_shard(path: &Path) -> io::Result<(ShardHeader, HashMap<String, Vec<f32>>)> {
    let p = path.to_str().ok_or_else(|| invalid(path, "not a UTF-8 path".into()))?;
    let st = checkpoint::st::load_safetensors(p)?;
    let header: ShardHeader = serde_json::from_value(st.config().clone()).map_err(|e| invalid(path, format!("not a shard header: {e}")))?;
    if header.format != SHARD_FORMAT {
        return Err(invalid(path, format!("format {:?}, expected {SHARD_FORMAT}", header.format)));
    }
    Ok((header, st.tensors))
}

const KINDS: [&str; 3] = ["master", "m", "v"];

impl ZeroOptimizer {
    /// Unit `u`'s master and moments, wherever they live.
    fn slice_state(&self, u: usize) -> [Vec<f32>; 3] {
        let n = self.layout.shard_len(u);
        match &self.units[u].slice {
            Slice::Host(h) => [h.master.clone(), h.m.clone(), h.v.clone()],
            Slice::Device(d) => {
                self.gpu.poll_wait();
                [&d.w, &d.m, &d.v].map(|b| read_elems(&self.gpu, b, 0, n, CollectiveDtype::F32))
            }
        }
    }

    /// Replace unit `u`'s master and moments.
    fn set_slice_state(&mut self, u: usize, [master, m, v]: [Vec<f32>; 3]) {
        match &mut self.units[u].slice {
            Slice::Host(h) => (h.master, h.m, h.v) = (master, m, v),
            Slice::Device(d) => {
                for (buf, data) in [(&d.w, &master), (&d.m, &m), (&d.v, &v)] {
                    write_elems(&self.gpu, buf, 0, data, CollectiveDtype::F32);
                }
            }
        }
    }

    /// This rank's checkpoint file for `step` in `dir`.
    pub fn shard_path(dir: &Path, step: u32, rank: usize, world: usize) -> PathBuf {
        dir.join(format!("shard-{step:08}-rank{rank:05}-of{world:05}.safetensors"))
    }

    /// Write this rank's shard (master, moments, the layout and `extra`: the trainer's per-rank state such as its
    /// RNG) durably to `dir` as of the steps applied so far; returns the file. Every rank writes its own;
    /// [`commit_generation`] names them together.
    pub fn save(&self, dir: &Path, extra: serde_json::Value) -> io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = Self::shard_path(dir, self.step, self.rank, self.layout.world);
        let mut tensors = Vec::with_capacity(3 * self.units.len());
        for u in 0..self.units.len() {
            for (kind, data) in KINDS.iter().zip(self.slice_state(u)) {
                tensors.push((format!("{kind}:{u}"), vec![data.len() as u64], data));
            }
        }
        let header = serde_json::json!({
            "format": SHARD_FORMAT,
            "step": self.step,
            "rank": self.rank,
            "sharding": self.sharding,
            "layout": self.layout,
            "extra": extra,
        });
        let p = path.to_str().ok_or_else(|| io::Error::other(format!("{}: not a UTF-8 path", path.display())))?;
        checkpoint::st::save_safetensors(p, &tensors, &header, None)?;
        Ok(path)
    }

    /// Remove this rank's shard files in `dir` but the newest `keep` steps.
    pub fn prune(&self, dir: &Path, keep: usize) -> io::Result<()> {
        let suffix = format!("-rank{:05}-of{:05}.safetensors", self.rank, self.layout.world);
        let mut steps: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let step = name.strip_prefix("shard-")?.strip_suffix(&suffix)?.parse().ok()?;
                Some((step, e.path()))
            })
            .collect();
        steps.sort_by_key(|(step, _)| std::cmp::Reverse(*step));
        for (_, path) in steps.into_iter().skip(keep) {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Restore the state saved at `step` in `dir` and push the restored values into the model's parameters; returns
    /// this rank's `extra`. When the saved layout is this one, this rank reads its own file; otherwise (another
    /// world size or unit cap) every rank's file of that step is read and the logical tensors are sliced again
    /// ([`read_logical`]), and the `extra` is rank 0's. Every rank calls it.
    pub fn restore(&mut self, coll: &dyn Collective, dir: &Path, step: u32) -> io::Result<serde_json::Value> {
        let own = Self::shard_path(dir, step, self.rank, self.layout.world);
        let saved = if own.exists() { Some(read_shard(&own)?).filter(|(h, _)| h.layout == self.layout) } else { None };
        let extra = if let Some((header, mut tensors)) = saved {
            for u in 0..self.units.len() {
                let mut take = |kind: &str| tensors.remove(&format!("{kind}:{u}")).ok_or_else(|| invalid(&own, format!("no {kind} for unit {u}")));
                let state = [take("master")?, take("m")?, take("v")?];
                self.set_slice_state(u, state);
            }
            header.extra
        } else {
            let logical = read_logical(dir, step)?;
            if logical.names != self.layout.names || logical.numels != self.layout.numels {
                return Err(io::Error::other(format!("{}: the checkpoint at step {step} holds other tensors than this model trains", dir.display())));
            }
            let split = |what: &[Vec<f32>]| self.layout.split(what, self.rank).map_err(io::Error::other);
            let (master, m, v) = (split(&logical.master)?, split(&logical.m)?, split(&logical.v)?);
            for (u, ((master, m), v)) in master.into_iter().zip(m).zip(v).enumerate() {
                self.set_slice_state(u, [master, m, v]);
            }
            logical.extra
        };
        self.step = step;
        self.publish(coll).map_err(io::Error::other)?;
        Ok(extra)
    }
}

/// The whole tensors a sharded checkpoint holds, in its layout's order.
pub struct LogicalState {
    pub names: Vec<String>,
    pub numels: Vec<u64>,
    pub step: u32,
    pub master: Vec<Vec<f32>>,
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    /// Rank 0's `extra`.
    pub extra: serde_json::Value,
}

/// Reassemble the logical tensors of the shards every rank saved at `step` in `dir`: the path a run with another
/// layout resumes through. Every rank's file must be there, agree on the layout and the step, and cover its slice.
pub fn read_logical(dir: &Path, step: u32) -> io::Result<LogicalState> {
    let prefix = format!("shard-{step:08}-rank");
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".safetensors"))).collect();
    files.sort();
    let first = files.first().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{}: no shard saved at step {step}", dir.display())))?;
    let (head, _) = read_shard(first)?;
    let layout = head.layout;
    let world = layout.world;
    let mut by_rank: BTreeMap<usize, (ShardHeader, HashMap<String, Vec<f32>>)> = BTreeMap::new();
    for f in &files {
        let (h, t) = read_shard(f)?;
        if h.layout != layout || h.step != step {
            return Err(invalid(f, "its layout or step differs from the other shards of this step".into()));
        }
        if by_rank.insert(h.rank, (h, t)).is_some() {
            return Err(invalid(f, "a second shard of the same rank".into()));
        }
    }
    if by_rank.len() != world || by_rank.keys().copied().ne(0..world) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: step {step} has shards of ranks {:?}, its layout has {world}", dir.display(), by_rank.keys().collect::<Vec<_>>())));
    }
    let mut gathered = Vec::new();
    for kind in KINDS {
        let shards: Vec<Vec<Vec<f32>>> = by_rank
            .values_mut()
            .map(|(h, t)| (0..layout.units.len()).map(|u| t.remove(&format!("{kind}:{u}")).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("rank {} has no {kind} for unit {u}", h.rank)))).collect::<io::Result<Vec<_>>>())
            .collect::<io::Result<_>>()?;
        gathered.push(layout.gather(&shards).map_err(io::Error::other)?);
    }
    let v = gathered.pop().unwrap_or_default();
    let m = gathered.pop().unwrap_or_default();
    let master = gathered.pop().unwrap_or_default();
    let extra = by_rank.remove(&0).map(|(h, _)| h.extra).unwrap_or_default();
    Ok(LogicalState { names: layout.names, numels: layout.numels, step, master, m, v, extra })
}

/// Rank 0 names the shard files every rank saved at `step` in `dir` in one committed generation
/// ([`crate::CheckpointCommit`]), after every rank's file is durable (the call is a barrier over `coll`). Returns the
/// manifest's path on rank 0, `None` elsewhere.
pub fn commit_generation(coll: &dyn Collective, rank: usize, dir: &Path, step: u32, job: &str, plan: &str) -> io::Result<Option<PathBuf>> {
    pollster::block_on(coll.all_reduce(rank, Payload::f32(vec![1.0]))).map_err(io::Error::other)?;
    if rank != 0 {
        return Ok(None);
    }
    let world = coll.world_size();
    let paths: Vec<(String, PathBuf)> = (0..world).map(|r| (format!("shard-{r:05}"), ZeroOptimizer::shard_path(dir, step, r, world))).collect();
    let files: Vec<(&str, &Path)> = paths.iter().map(|(role, p)| (role.as_str(), p.as_path())).collect();
    let commit = crate::CheckpointCommit::build(job, plan, step, world, dir, &files)?;
    commit.write(dir).map(Some)
}
