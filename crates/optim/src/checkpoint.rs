// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in crash-safe, resumable training runs
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Transactional training checkpoint: values, optimiser state (moments and
//! masters), step, schedule and rounding seed, committed atomically.
//!
//! # Protocol
//!
//! A checkpoint is a directory `gen-<step>` under the checkpoint root:
//!
//! 1. write `tmp-<step>-<pid>/{values.bin,state.bin,meta.json}` (`meta.json`
//!    last, carrying the length and FNV-1a 64 hash of the two data files);
//! 2. fsync every file, then the temp directory;
//! 3. `rename` it to `gen-<step>` (atomic on POSIX) and fsync the root.
//!
//! A directory named `gen-*` therefore always came from a fully durable
//! write. Recovery nonetheless re-verifies lengths and hashes (a disk can
//! lie) and falls back to the newest generation that verifies; stale `tmp-*`
//! directories from a crash are ignored and removed by the next commit. The
//! two newest valid generations are kept so that a bad newest one never leaves
//! the run without a checkpoint.
//!
//! Everything is streamed through bounded buffers; no stage holds a whole
//! tensor. Resume parity is possible because the rounding randomness is a pure
//! function of `(seed, step, tensor, slot, element)` (see `streamed`).

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use paramstore::precision::ValueDtype;
use serde_json::{json, Value};

use crate::codec::{Rounding, SlotKind, StateCodec};
use crate::storage::OptimizerStateStorage;
use crate::streamed::{ParamTiles, StreamedOptimizer};
use paramstore::precision::StateFormat;

/// What the trainer owns besides weights and optimiser state.
#[derive(Clone, Debug, PartialEq)]
pub struct RunState {
    pub step: u32,
    /// Seed of the stochastic-rounding stream.
    pub seed: u64,
    /// Draw counter of the trainer's own data/sampling RNG.
    pub rng_counter: u64,
    /// Named schedule scalars (learning rate position, warmup progress, ...).
    pub schedule: Vec<(String, f64)>,
}

/// Points of the commit protocol, for crash-injection tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    DataWritten,
    MetaWritten,
    Renamed,
}

const KEEP: usize = 2;
const IO_CHUNK: usize = 1 << 20;

struct Fnv(u64);
impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn update(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Writer that hashes and counts what passes through.
struct HashWriter<W: Write> {
    inner: W,
    hash: Fnv,
    len: u64,
}
impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(b)?;
        self.hash.update(&b[..n]);
        self.len += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn fsync_dir(p: &Path) -> io::Result<()> {
    File::open(p)?.sync_all()
}

fn hash_file(p: &Path) -> io::Result<(u64, u64)> {
    let mut f = File::open(p)?;
    let (mut h, mut len, mut buf) = (Fnv::new(), 0u64, vec![0u8; IO_CHUNK]);
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            return Ok((len, h.0));
        }
        h.update(&buf[..n]);
        len += n as u64;
    }
}

fn gen_name(step: u32) -> String {
    format!("gen-{step:010}")
}

fn value_codec(v: ValueDtype) -> StateCodec {
    let format = match v {
        ValueDtype::F32 => StateFormat::F32,
        ValueDtype::Bf16 => StateFormat::Bf16,
    };
    StateCodec { format, kind: SlotKind::Signed }
}

/// A checkpoint root directory.
pub struct CheckpointDir {
    root: PathBuf,
}

impl CheckpointDir {
    pub fn new(root: impl Into<PathBuf>) -> io::Result<CheckpointDir> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(CheckpointDir { root })
    }

    /// The part of the manifest that identifies the run's tensors and rule.
    fn tensor_table<S: OptimizerStateStorage>(opt: &StreamedOptimizer<S>) -> Value {
        Value::Array(
            opt.specs()
                .iter()
                .map(|s| json!({"name": s.name, "numel": s.numel, "precision": format!("{:?}", s.precision)}))
                .collect(),
        )
    }

    fn manifest<S: OptimizerStateStorage>(opt: &StreamedOptimizer<S>, run: &RunState, files: Value) -> Value {
        let tensors = Self::tensor_table(opt);
        json!({
            "format": 1,
            "rule": opt.rule().name(),
            "step": run.step,
            "seed": run.seed,
            "rng_counter": run.rng_counter,
            "schedule": run.schedule,
            "tensors": tensors,
            "files": files,
        })
    }

    /// Commit a checkpoint of the optimiser at `run.step`. Returns the
    /// generation directory. Re-committing an existing valid step is a no-op.
    pub fn commit<S: OptimizerStateStorage>(
        &self,
        opt: &mut StreamedOptimizer<S>,
        params: &mut dyn ParamTiles,
        run: &RunState,
    ) -> io::Result<PathBuf> {
        self.commit_until(opt, params, run, Stage::Renamed)
    }

    /// [`Self::commit`] that stops (leaving the on-disk state a crash at that
    /// point would leave) after `stop`.
    pub fn commit_until<S: OptimizerStateStorage>(
        &self,
        opt: &mut StreamedOptimizer<S>,
        params: &mut dyn ParamTiles,
        run: &RunState,
        stop: Stage,
    ) -> io::Result<PathBuf> {
        self.remove_stale_tmp()?;
        let final_dir = self.root.join(gen_name(run.step));
        if final_dir.exists() && self.verify(&final_dir, opt).is_ok() {
            return Ok(final_dir);
        }
        let tmp = self.root.join(format!("tmp-{:010}-{}", run.step, std::process::id()));
        if tmp.exists() {
            fs::remove_dir_all(&tmp)?;
        }
        fs::create_dir(&tmp)?;

        let write = |name: &str, f: &mut dyn FnMut(&mut dyn Write) -> io::Result<()>| -> io::Result<(u64, u64)> {
            let file = OpenOptions::new().write(true).create_new(true).open(tmp.join(name))?;
            let mut hw = HashWriter { inner: BufWriter::with_capacity(IO_CHUNK, file), hash: Fnv::new(), len: 0 };
            f(&mut hw)?;
            hw.flush()?;
            hw.inner.get_ref().sync_all()?;
            Ok((hw.len, hw.hash.0))
        };
        let tile = opt.tile_elems();
        let specs = opt.specs().to_vec();
        let (vlen, vhash) = write("values.bin", &mut |w| {
            let mut vals = vec![0f32; tile];
            for (ti, s) in specs.iter().enumerate() {
                let c = value_codec(s.precision.value);
                let mut bytes = vec![0u8; c.data_bytes(tile)];
                for start in (0..s.numel).step_by(tile) {
                    let n = tile.min(s.numel - start);
                    params.read_value(ti, start, &mut vals[..n])?;
                    let nb = c.data_bytes(n);
                    c.encode(&vals[..n], &mut bytes[..nb], &mut [], Rounding::NEAREST, start as u64);
                    w.write_all(&bytes[..nb])?;
                }
            }
            Ok(())
        })?;
        let (slen, shash) = write("state.bin", &mut |w| opt.state_mut().export(w))?;
        let files = json!({
            "values.bin": {"len": vlen, "fnv": format!("{vhash:016x}")},
            "state.bin": {"len": slen, "fnv": format!("{shash:016x}")},
        });
        if stop < Stage::MetaWritten {
            return Ok(tmp);
        }
        let meta = serde_json::to_vec_pretty(&Self::manifest(opt, run, files)).map_err(|e| bad(e.to_string()))?;
        write("meta.json", &mut |w| w.write_all(&meta))?;
        fsync_dir(&tmp)?;
        if stop < Stage::Renamed {
            return Ok(tmp);
        }
        if final_dir.exists() {
            fs::remove_dir_all(&final_dir)?; // an invalid leftover of the same step
        }
        fs::rename(&tmp, &final_dir)?;
        fsync_dir(&self.root)?;
        self.prune(opt)?;
        Ok(final_dir)
    }

    fn entries(&self, prefix: &str) -> io::Result<Vec<PathBuf>> {
        let mut v: Vec<PathBuf> = fs::read_dir(&self.root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(prefix)))
            .collect();
        v.sort();
        Ok(v)
    }

    fn remove_stale_tmp(&self) -> io::Result<()> {
        for p in self.entries("tmp-")? {
            fs::remove_dir_all(p)?;
        }
        Ok(())
    }

    fn prune<S: OptimizerStateStorage>(&self, opt: &StreamedOptimizer<S>) -> io::Result<()> {
        let mut kept = 0;
        for p in self.entries("gen-")?.into_iter().rev() {
            if kept < KEEP && self.verify(&p, opt).is_ok() {
                kept += 1;
            } else if kept >= KEEP {
                fs::remove_dir_all(p)?;
            }
        }
        Ok(())
    }

    /// Check one generation against this run's tensors: manifest parses, the
    /// tensor table and rule match, file lengths and hashes match.
    fn verify<S: OptimizerStateStorage>(&self, dir: &Path, opt: &StreamedOptimizer<S>) -> io::Result<Value> {
        let meta: Value = serde_json::from_slice(&fs::read(dir.join("meta.json"))?).map_err(|e| bad(format!("meta.json: {e}")))?;
        if meta["format"] != 1 || meta["rule"] != opt.rule().name() || meta["tensors"] != Self::tensor_table(opt) {
            return Err(bad(format!("{}: rule or tensor table differs from this run", dir.display())));
        }
        for name in ["values.bin", "state.bin"] {
            let (len, hash) = hash_file(&dir.join(name))?;
            let f = &meta["files"][name];
            if f["len"] != len || f["fnv"] != format!("{hash:016x}") {
                return Err(bad(format!("{}/{name}: length or hash mismatch", dir.display())));
            }
        }
        Ok(meta)
    }

    /// Restore the newest generation that verifies into `params` and the
    /// optimiser (state, step, seed). `None` when there is no checkpoint at
    /// all; an error when generations exist but none is usable.
    pub fn restore<S: OptimizerStateStorage>(
        &self,
        opt: &mut StreamedOptimizer<S>,
        params: &mut dyn ParamTiles,
    ) -> io::Result<Option<RunState>> {
        let gens = self.entries("gen-")?;
        if gens.is_empty() {
            return Ok(None);
        }
        let mut last_err = None;
        for dir in gens.into_iter().rev() {
            let meta = match self.verify(&dir, opt) {
                Ok(m) => m,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };
            let tile = opt.tile_elems();
            let specs = opt.specs().to_vec();
            let mut r = BufReader::with_capacity(IO_CHUNK, File::open(dir.join("values.bin"))?);
            let mut vals = vec![0f32; tile];
            for (ti, s) in specs.iter().enumerate() {
                let c = value_codec(s.precision.value);
                let mut bytes = vec![0u8; c.data_bytes(tile)];
                for start in (0..s.numel).step_by(tile) {
                    let n = tile.min(s.numel - start);
                    let nb = c.data_bytes(n);
                    r.read_exact(&mut bytes[..nb])?;
                    c.decode(&bytes[..nb], &[], &mut vals[..n]);
                    params.write_value(ti, start, &vals[..n], Rounding::NEAREST)?;
                }
            }
            opt.state_mut().import(&mut BufReader::with_capacity(IO_CHUNK, File::open(dir.join("state.bin"))?))?;
            let run = RunState {
                step: meta["step"].as_u64().ok_or_else(|| bad("step".into()))? as u32,
                seed: meta["seed"].as_u64().ok_or_else(|| bad("seed".into()))?,
                rng_counter: meta["rng_counter"].as_u64().ok_or_else(|| bad("rng_counter".into()))?,
                schedule: serde_json::from_value(meta["schedule"].clone()).map_err(|e| bad(format!("schedule: {e}")))?,
            };
            opt.set_position(run.step, run.seed);
            return Ok(Some(run));
        }
        Err(last_err.unwrap_or_else(|| bad("no usable checkpoint generation".into())))
    }
}
