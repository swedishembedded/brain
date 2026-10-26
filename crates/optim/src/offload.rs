// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in training models larger than your
// accelerator memory then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Off-device AdamW: the optimiser state (`m`/`v`) lives in **host memory or
//! on disk**, and the update runs on the CPU. Only the weight and gradient stay
//! on the device, where the forward/backward need them.
//!
//! This is the streamed optimiser (`streamed`) behind the historical
//! `OffloadAdam` interface. The state format is the parameter's own
//! ([`ParamStore::set_precision`]): fp32 by default, blockwise int8 or bf16
//! where the policy says so, so the host state costs 8, 2 or 4 bytes per
//! element instead of the old fp32 master + m + v triple (12). The device
//! weight is the value; no second host copy of it is kept.
//!
//! Per step each tensor is read off the device once (one tensor at a time, so
//! the host staging is the largest tensor, not the model), updated in tiles,
//! and written back once. A ranged device read in `gpu-core` would reduce the
//! staging to a tile.

use std::io;
use std::path::{Path, PathBuf};

use gpu_core::Gpu;
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, ValueDtype};
use paramstore::ParamStore;

use crate::checkpoint::{CheckpointDir, RunState};
use crate::codec::Rounding;
use crate::gradnorm::{DeviceGradNorm, NormItem};
use crate::rule::Rule;
use crate::storage::{BlockStateStorage, FileBacking, MemBacking, OptimizerStateStorage};
use crate::streamed::{state_layout, ParamTiles, StreamedOptimizer, TensorSpec};

/// Tile size of the host update, in elements.
const OFFLOAD_TILE: usize = 1 << 26;

/// The device-resident values and gradients of the offloaded tensors, as tiles.
///
/// A device that copies a sub-range back ([`Gpu::reads_ranges`]) is read and written one tile at a time, so the host
/// holds the tile and nothing else. Any other device is read whole, once per tensor and pass, and cached while the
/// tensor's tiles are processed; written values are flushed to the device when the tensor's last tile lands.
struct DeviceTiles<'a> {
    gpu: &'a Gpu,
    ps: &'a ParamStore,
    names: Vec<(String, usize)>,
    ranged: bool,
    value: Option<(usize, Vec<f32>)>,
    grad: Option<(usize, Vec<f32>)>,
}

impl<'a> DeviceTiles<'a> {
    fn new(gpu: &'a Gpu, ps: &'a ParamStore, names: Vec<(String, usize)>) -> Self {
        DeviceTiles { gpu, ps, names, ranged: gpu.reads_ranges(), value: None, grad: None }
    }

    fn load_value(&mut self, ti: usize) -> &mut Vec<f32> {
        if self.value.as_ref().map(|v| v.0) != Some(ti) {
            let (name, numel) = &self.names[ti];
            self.value = Some((ti, self.gpu.read(self.ps.w(name), *numel)));
        }
        &mut self.value.as_mut().unwrap().1
    }
}

impl ParamTiles for DeviceTiles<'_> {
    fn read_value(&mut self, ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        if self.ranged {
            self.gpu.read_into(self.ps.w(&self.names[ti].0), start as u64, out);
            return Ok(());
        }
        out.copy_from_slice(&self.load_value(ti)[start..start + out.len()]);
        Ok(())
    }

    fn write_value(&mut self, ti: usize, start: usize, vals: &[f32], _rnd: Rounding) -> io::Result<()> {
        if self.ranged {
            self.gpu.write_f32_at(self.ps.w(&self.names[ti].0), start as u64, vals);
            return Ok(());
        }
        let n = vals.len();
        self.load_value(ti)[start..start + n].copy_from_slice(vals);
        let (name, numel) = &self.names[ti];
        if start + n == *numel {
            let w = &self.value.as_ref().unwrap().1;
            self.gpu.write(self.ps.w(name), bytemuck::cast_slice(w));
        }
        Ok(())
    }

    fn read_grad(&mut self, ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        if self.ranged {
            self.gpu.read_into(self.ps.g(&self.names[ti].0), start as u64, out);
            return Ok(());
        }
        if self.grad.as_ref().map(|g| g.0) != Some(ti) {
            let (name, numel) = &self.names[ti];
            self.grad = Some((ti, self.gpu.read(self.ps.g(name), *numel)));
        }
        out.copy_from_slice(&self.grad.as_ref().unwrap().1[start..start + out.len()]);
        Ok(())
    }
}

/// Host-resident AdamW over the store's [`Role::Offload`](paramstore::Role)
/// parameters, with state in `S` ([`MemBacking`] by default, or a file via
/// [`OffloadAdam::on_disk`]).
pub struct OffloadAdam<S: OptimizerStateStorage = BlockStateStorage<MemBacking>> {
    opt: StreamedOptimizer<S>,
    /// The device reduction of the global gradient norm, resolved on the first clipped step; `Some(None)` when the
    /// device lacks its kernels (the norm is then measured on the host, which moves every gradient there).
    norm: Option<Option<DeviceGradNorm>>,
}

fn specs_of(ps: &ParamStore) -> Vec<TensorSpec> {
    // LoRA (the only source of a non-1.0 lr_mult today) and Role::Offload
    // never coexist in this tree (a LoRA-training ParamStore assigns every
    // non-adapter tensor Role::Frozen, never Role::Offload) - guard the
    // untested combination loudly rather than silently applying the GPU
    // path's per-tensor lr but not this host path's, if that ever changes.
    ps.offload
        .iter()
        .map(|(name, numel)| {
            assert_eq!(ps.lr_mult_of(name), 1.0, "OffloadAdam: {name} has a non-1.0 lr_mult, which this host path does not yet apply");
            let precision: ParamPrecision = ps.precision_of(name);
            assert!(
                precision.value == ValueDtype::F32 && precision.master == MasterPolicy::None,
                "OffloadAdam: {name}: the device weight is the f32 value; a master copy or other value dtype is not applicable ({precision:?})"
            );
            TensorSpec { name: name.clone(), numel: *numel, precision, row_len: None }
        })
        .collect()
}

fn default_rule() -> Rule {
    Rule::AdamW { adam: crate::Adam::default(), wd: 0.0 }
}

fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new().build_hasher().finish()
}

impl OffloadAdam {
    /// Optimiser with its state in host memory.
    pub fn new(_gpu: &Gpu, ps: &ParamStore) -> OffloadAdam {
        let specs = specs_of(ps);
        let layout = state_layout(&default_rule(), &specs);
        let st = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), OFFLOAD_TILE)
            .expect("state layout and backing agree");
        OffloadAdam { opt: StreamedOptimizer::new(default_rule(), specs, st, random_seed(), OFFLOAD_TILE).expect("valid specs"), norm: None }
    }
}

impl OffloadAdam<BlockStateStorage<FileBacking>> {
    /// Optimiser with its state streamed from the file `path` (created; fails
    /// if it exists): host memory is the staging workspace, not the state.
    pub fn on_disk(_gpu: &Gpu, ps: &ParamStore, path: &Path) -> io::Result<Self> {
        let specs = specs_of(ps);
        let layout = state_layout(&default_rule(), &specs);
        let st = BlockStateStorage::new(layout.clone(), FileBacking::create(path, layout.total_bytes())?, OFFLOAD_TILE)?;
        Ok(OffloadAdam { opt: StreamedOptimizer::new(default_rule(), specs, st, random_seed(), OFFLOAD_TILE)?, norm: None })
    }
}

impl<S: OptimizerStateStorage> OffloadAdam<S> {
    /// One AdamW step over the offloaded params at 1-based step `t`. Every
    /// gradient is multiplied by [`crate::grad_multiplier`] (the `extra_scale`
    /// average and the global clip over exactly these params). Matches
    /// `adamw.wgsl` element-wise.
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, gpu: &Gpu, ps: &ParamStore, t: u32, lr: f32, wd: f32, adam: crate::Adam, clip: Option<f32>, extra_scale: f32) {
        self.opt.set_rule(Rule::AdamW { adam, wd });
        let seed = self.opt.seed();
        self.opt.set_position(t - 1, seed);
        let names = self.opt.specs().iter().map(|s| (s.name.clone(), s.numel)).collect();
        let mut tiles = DeviceTiles::new(gpu, ps, names);
        let norm = self.norm.get_or_insert_with(|| DeviceGradNorm::new(gpu).ok());
        let report = match (clip, norm) {
            // The gradients are on the device: reduce their norm there rather than reading all of them to the host first.
            (Some(_), Some(norm)) => {
                let items: Vec<NormItem<'_>> = self.opt.specs().iter().map(|s| NormItem { buf: ps.g(&s.name), numel: s.numel, dtype: GradDtype::F32 }).collect();
                let multiplier = crate::grad_multiplier(norm.sum_sq(gpu, &items), clip, extra_scale);
                self.opt.step_scaled(&mut tiles, lr, multiplier)
            }
            _ => self.opt.step(&mut tiles, lr, clip, extra_scale),
        };
        report.unwrap_or_else(|e| panic!("OffloadAdam step: {e}"));
        if gpu_core::profile::enabled() {
            eprintln!("stage host optimizer update phases: {:?}", self.opt.take_phase_times());
        }
    }

    /// Atomically commit the offloaded values (read off the device), the host state, the step and the rounding seed
    /// as one generation of `dir`, keyed by the optimizer step.
    pub fn commit(&mut self, gpu: &Gpu, ps: &ParamStore, dir: &CheckpointDir) -> io::Result<PathBuf> {
        let names = self.opt.specs().iter().map(|s| (s.name.clone(), s.numel)).collect();
        let mut tiles = DeviceTiles::new(gpu, ps, names);
        let run = RunState { step: self.opt.step_count(), seed: self.opt.seed(), rng_counter: 0, schedule: Vec::new() };
        dir.commit(&mut self.opt, &mut tiles, &run)
    }

    /// Restore exactly the generation committed at optimizer step `step`: the values go back to the device, the state
    /// and the position to this optimiser. `Ok(None)` when no such generation was committed.
    pub fn restore_at(&mut self, gpu: &Gpu, ps: &ParamStore, dir: &CheckpointDir, step: u32) -> io::Result<Option<RunState>> {
        let names = self.opt.specs().iter().map(|s| (s.name.clone(), s.numel)).collect();
        let mut tiles = DeviceTiles::new(gpu, ps, names);
        dir.restore_at(&mut self.opt, &mut tiles, step)
    }

    /// The streamed optimiser, for checkpointing its state and position.
    pub fn streamed(&mut self) -> &mut StreamedOptimizer<S> {
        &mut self.opt
    }

    /// Host bytes held by the optimiser state and staging.
    pub fn host_bytes(&self) -> (u64, usize) {
        (self.opt.state_physical_bytes(), self.opt.staging_bytes())
    }
}
