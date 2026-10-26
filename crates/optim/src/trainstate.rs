// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in wiring full fine-tuning of new
// architectures onto a bounded memory budget then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The one entry point a generic trainer wires to: a parameter manifest, a
//! [`PrecisionPolicy`] and an update [`Rule`] in; a stepping, checkpointing
//! training state out.
//!
//! A model crate contributes only the manifest (`(name, numel)` pairs). The
//! per-group precision is resolved by the policy and reported, the state lives
//! in host memory or in a file, and every byte of it is accounted physically.
//! The trainer owns value and gradient access through [`ParamTiles`] and the
//! schedule through [`RunState`].
//!
//! ```ignore
//! let mut ts = TrainState::create(TrainStateConfig { manifest, policy, rule, seed: None,
//!                                                  tile_elems: 1 << 16, state_dir: None, row_len: vec![] })?;
//! println!("{}", ts.report());
//! let _ = ts.step(&mut params, lr, Some(1.0), 1.0)?;
//! ts.commit(&ckpt_dir, &mut params, &run)?;
//! ```

use std::io;
use std::path::PathBuf;

use paramstore::precision::{ParamBytes, PrecisionPolicy};

use crate::checkpoint::{CheckpointDir, RunState};
use crate::rule::Rule;
use crate::storage::{BlockStateStorage, FileBacking, MemBacking, OptimizerStateStorage};
use crate::streamed::{effective_tile, state_layout, ParamTiles, StepReport, StreamedOptimizer, TensorSpec};

pub struct TrainStateConfig {
    /// `(name, numel)` of every optimised parameter, in a stable order.
    pub manifest: Vec<(String, usize)>,
    pub policy: PrecisionPolicy,
    pub rule: Rule,
    /// Stochastic-rounding seed; `None` draws a random one (read it back with
    /// [`TrainState::seed`] and record it).
    pub seed: Option<u64>,
    pub tile_elems: usize,
    /// Directory for the streamed on-disk state file; `None` keeps it in memory.
    pub state_dir: Option<PathBuf>,
    /// `(name, row length)` of the matrices a factored rule ([`Rule::Adafactor`]) keeps row and column statistics
    /// for; a tensor not listed is treated as a vector. Ignored by elementwise rules.
    pub row_len: Vec<(String, usize)>,
}

pub struct TrainState {
    opt: StreamedOptimizer<Box<dyn OptimizerStateStorage + Send>>,
    policy_report: String,
    bytes: ParamBytes,
}

impl OptimizerStateStorage for Box<dyn OptimizerStateStorage + Send> {
    fn addressable(&self) -> bool {
        (**self).addressable()
    }
    fn slot_views(&mut self, tensor: usize, slots: &[usize], start: usize, n: usize) -> Option<Vec<crate::storage::SlotView<'_>>> {
        (**self).slot_views(tensor, slots, start, n)
    }
    fn read_tile(&mut self, tensor: usize, slot: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        (**self).read_tile(tensor, slot, start, out)
    }
    fn write_tile(&mut self, tensor: usize, slot: usize, start: usize, vals: &[f32], rnd: crate::codec::Rounding) -> io::Result<()> {
        (**self).write_tile(tensor, slot, start, vals, rnd)
    }
    fn physical_bytes(&self) -> u64 {
        (**self).physical_bytes()
    }
    fn staging_bytes(&self) -> usize {
        (**self).staging_bytes()
    }
    fn export(&mut self, w: &mut dyn io::Write) -> io::Result<()> {
        (**self).export(w)
    }
    fn import(&mut self, r: &mut dyn io::Read) -> io::Result<()> {
        (**self).import(r)
    }
    fn sync(&mut self) -> io::Result<()> {
        (**self).sync()
    }
}

impl TrainState {
    pub fn create(cfg: TrainStateConfig) -> io::Result<TrainState> {
        let specs: Vec<TensorSpec> = cfg
            .manifest
            .iter()
            .map(|(name, numel)| TensorSpec {
                name: name.clone(),
                numel: *numel,
                precision: cfg.policy.resolve(name, *numel).precision,
                row_len: cfg.row_len.iter().find(|(n, _)| n == name).map(|(_, k)| *k).filter(|_| matches!(cfg.rule, Rule::Adafactor { .. })),
            })
            .collect();
        let (policy_report, bytes) = cfg.policy.report(&cfg.manifest, cfg.rule.slots().len());
        let layout = state_layout(&cfg.rule, &specs);
        let tile = effective_tile(&specs, cfg.tile_elems);
        let state: Box<dyn OptimizerStateStorage + Send> = match &cfg.state_dir {
            None => Box::new(BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), tile)?),
            Some(dir) => {
                std::fs::create_dir_all(dir)?;
                let backing = FileBacking::create(&dir.join("optimizer.state"), layout.total_bytes())?;
                Box::new(BlockStateStorage::new(layout.clone(), backing, tile)?)
            }
        };
        let seed = cfg.seed.unwrap_or_else(|| {
            use std::hash::{BuildHasher, Hasher};
            std::collections::hash_map::RandomState::new().build_hasher().finish()
        });
        Ok(TrainState { opt: StreamedOptimizer::new(cfg.rule, specs, state, seed, cfg.tile_elems)?, policy_report, bytes })
    }

    /// Per-parameter precision choices with the rule that made each, and the
    /// physical byte totals; staging and state sizes of the live optimiser.
    pub fn report(&self) -> String {
        format!(
            "{}total: value {} B, grad {} B, moments {} B, master {} B; state file/memory {} B, host staging {} B (tile {} elems)\n",
            self.policy_report,
            self.bytes.value,
            self.bytes.grad,
            self.bytes.moments,
            self.bytes.master,
            self.opt.state_physical_bytes(),
            self.opt.staging_bytes(),
            self.opt.tile_elems(),
        )
    }

    pub fn seed(&self) -> u64 {
        self.opt.seed()
    }

    pub fn step_count(&self) -> u32 {
        self.opt.step_count()
    }

    /// Seed master copies from the current values (fresh runs only).
    pub fn init_master(&mut self, params: &mut dyn ParamTiles) -> io::Result<()> {
        self.opt.init_master(params)
    }

    /// [`Self::init_master`] for one tile of tensor `ti` whose initial values the caller already holds.
    pub fn init_master_tile(&mut self, ti: usize, start: usize, vals: &[f32]) -> io::Result<()> {
        self.opt.init_master_tile(ti, start, vals)
    }

    /// Where the tile loops spent their time since the last call.
    pub fn take_phase_times(&mut self) -> crate::streamed::PhaseTimes {
        self.opt.take_phase_times()
    }

    pub fn step(&mut self, params: &mut dyn ParamTiles, lr: f32, clip: Option<f32>, extra_scale: f32) -> io::Result<StepReport> {
        self.opt.step(params, lr, clip, extra_scale)
    }

    /// Sum of squares of every gradient (f64); with [`crate::grad_multiplier`] the global clip of a step
    /// whose gradients are all complete.
    pub fn grad_sum_sq(&mut self, params: &mut dyn ParamTiles) -> io::Result<f64> {
        self.opt.grad_sum_sq(params)
    }

    /// Open the next step (1-based index): the first half of the layer-at-a-time form of [`Self::step`].
    pub fn begin_step(&mut self) -> u32 {
        self.opt.begin_step()
    }

    /// Update tensor `ti` of the step opened by [`Self::begin_step`], each tensor once, in any order.
    pub fn update_tensor(&mut self, params: &mut dyn ParamTiles, ti: usize, lr: f32, grad_multiplier: f32) -> io::Result<()> {
        self.opt.update_tensor(params, ti, lr, grad_multiplier)
    }

    /// Stream the highest-precision weights (the master where a tensor has one, else its value from
    /// `values`) into `sink` encoded as `dst[i]`, a tile at a time.
    pub fn export_masters(
        &mut self,
        values: &mut dyn ParamTiles,
        dst: &[checkpoint::packed::TensorDescriptor],
        registry: &checkpoint::packed::CodecRegistry,
        sink: &mut dyn checkpoint::packed::TensorSink,
        tile_elems: usize,
    ) -> io::Result<crate::encoded::ExportReport> {
        crate::encoded::export_values(&mut crate::streamed::MasterView { opt: &mut self.opt, values }, dst, registry, sink, tile_elems)
    }

    /// Atomically commit values, state, step, seed and the trainer's schedule.
    /// `run.step` and `run.seed` should be [`Self::step_count`] and [`Self::seed`].
    pub fn commit(&mut self, dir: &CheckpointDir, params: &mut dyn ParamTiles, run: &RunState) -> io::Result<PathBuf> {
        dir.commit(&mut self.opt, params, run)
    }

    /// The run state to commit now, given the trainer's own schedule scalars
    /// and RNG counter.
    pub fn run_state(&self, rng_counter: u64, schedule: Vec<(String, f64)>) -> RunState {
        RunState { step: self.opt.step_count(), seed: self.opt.seed(), rng_counter, schedule }
    }

    /// [`Self::restore`] of exactly the checkpoint committed at `step`.
    pub fn restore_at(&mut self, dir: &CheckpointDir, params: &mut dyn ParamTiles, step: u32) -> io::Result<Option<RunState>> {
        let run = dir.restore_at(&mut self.opt, params, step)?;
        if run.is_some() {
            self.opt.publish_masters(params)?;
        }
        Ok(run)
    }

    /// Use `rule` from the next step on; it must keep the same state slots.
    pub fn set_rule(&mut self, rule: Rule) {
        self.opt.set_rule(rule);
    }

    /// Overwrite the master copy of tensor `ti` from `start`; `false` when the tensor has none.
    pub fn set_master(&mut self, ti: usize, start: usize, vals: &[f32]) -> io::Result<bool> {
        self.opt.set_master(ti, start, vals)
    }

    /// Read `out.len()` elements of tensor `ti` from `start` as the highest-precision copy: the master where
    /// the tensor has one, else its value from `values`.
    pub fn read_master(&mut self, values: &mut dyn ParamTiles, ti: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        crate::streamed::MasterView { opt: &mut self.opt, values }.read_value(ti, start, out)
    }

    /// Restore the newest valid checkpoint; `None` if there is none. Tensors with a master copy have their
    /// stored value rewritten from it, so the master stays the one source of truth.
    pub fn restore(&mut self, dir: &CheckpointDir, params: &mut dyn ParamTiles) -> io::Result<Option<RunState>> {
        let run = dir.restore(&mut self.opt, params)?;
        if run.is_some() {
            self.opt.publish_masters(params)?;
        }
        Ok(run)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Rounding;
    use crate::streamed::{tensor_layout, HostParams};
    use crate::Adam;
    use paramstore::precision::{GradDtype, Matcher, MasterPolicy, ParamPrecision, StateFormat, ValueDtype};

    fn host(numels: &[usize]) -> HostParams<MemBacking, MemBacking> {
        let l = tensor_layout(numels, StateFormat::F32);
        HostParams {
            values: BlockStateStorage::new(l.clone(), MemBacking::zeroed(l.total_bytes()), 4096).unwrap(),
            grads: BlockStateStorage::new(l.clone(), MemBacking::zeroed(l.total_bytes()), 4096).unwrap(),
        }
    }

    /// Spec: one policy gives a model's norms fp32 state and its big matrices
    /// int8 state; the report names the rule per parameter and the live
    /// state is the sum of the physical per-group sizes; a step trains both.
    #[test]
    fn a_mixed_precision_policy_trains_every_group_and_reports_physical_bytes() {
        let int8 = ParamPrecision {
            value: ValueDtype::F32,
            grad: GradDtype::F32,
            state: StateFormat::Int8Block { block: 64 },
            master: MasterPolicy::None,
            stochastic_rounding: true,
        };
        let policy = PrecisionPolicy {
            default: ParamPrecision::FP32,
            rules: vec![(Matcher::All(vec![Matcher::MinNumel(1024), Matcher::NameSuffix(".weight".into())]), int8)],
        };
        let manifest = vec![("blk.attn.weight".to_string(), 2048usize), ("blk.norm.scale".to_string(), 8usize)];
        let mut ts = TrainState::create(TrainStateConfig {
            manifest: manifest.clone(),
            policy,
            rule: Rule::AdamW { adam: Adam::default(), wd: 0.0 },
            seed: None,
            tile_elems: 100,
            state_dir: None,
            row_len: Vec::new(),
        })
        .unwrap();
        let report = ts.report();
        assert!(report.contains("blk.attn.weight") && report.contains("rule 0") && report.contains("default"), "{report}");
        // matrix: 2 slots x (2048 + 4*32); norm: 2 slots x 8 x 4
        assert_eq!(ts.opt.state_physical_bytes(), 2 * (2048 + 4 * 32) + 2 * 8 * 4);

        let mut p = host(&[2048, 8]);
        for (ti, n) in [(0usize, 2048usize), (1, 8)] {
            for start in (0..n).step_by(1024) {
                let m = 1024.min(n - start);
                p.values.write_tile(ti, 0, start, &vec![1.0; m], Rounding::NEAREST).unwrap();
                p.grads.write_tile(ti, 0, start, &vec![0.5; m], Rounding::NEAREST).unwrap();
            }
        }
        ts.step(&mut p, 0.1, None, 1.0).unwrap();
        for (ti, n) in [(0usize, 2048usize), (1, 8)] {
            let mut w = vec![0f32; n.min(1024)];
            p.values.read_tile(ti, 0, 0, &mut w).unwrap();
            assert!(w.iter().all(|&v| (v - 0.9).abs() < 0.02), "tensor {ti} moved by one lr step: {:?}", &w[..3]);
        }
    }
}
