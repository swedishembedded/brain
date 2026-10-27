// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A compiled [`Plan`] resident on the NPU, ready to run.
//!
//! NPU memory is several modest buffers rather than one large block: the
//! activation region, the constant chunks and the programs. NPU memory is
//! cacheable, so the CPU and the NPU hand tensors over through
//! explicit cache maintenance: the input is written back before the first
//! segment, a CPU step invalidates what it reads and writes back what it
//! produces, and the outputs are invalidated after the last step. Every tensor
//! is 64-byte aligned, so no maintenance ever touches a neighbour's lines.

use std::time::{Duration, Instant};

use crate::driver::{Buffer, Device};
use crate::graph::TensorId;
use crate::job::{self, Job};
use crate::layout::LANES;
use crate::plan::{AddressMap, Placement, Plan, Step};
use crate::{Error, Result};

/// Where one run's time went.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub npu: Duration,
    /// CPU steps, by kind, and the cache maintenance around them.
    pub add: Duration,
    pub max_pool: Duration,
    pub upsample: Duration,
    pub cache: Duration,
    /// Writing the input (and making it visible to the NPU), and making the
    /// outputs visible to the CPU.
    pub input: Duration,
    pub output: Duration,
    pub total: Duration,
}

impl Timing {
    /// Everything the CPU did.
    pub fn cpu(&self) -> Duration {
        self.add + self.max_pool + self.upsample + self.cache
    }
}

pub struct Model {
    device: Device,
    job: Job,
    activations: Buffer,
    /// Held for the NPU's sake; the CPU never touches them after load.
    constants: Vec<Buffer>,
    plan: Plan,
}

impl Model {
    /// Allocate NPU memory for `plan`, quantize and write its constants (one
    /// layer's float weights in memory at a time) and its programs.
    pub fn load(device: Device, plan: Plan) -> Result<Model> {
        // Largest first: the allocations most likely to fail go while the
        // most contiguous memory is still free.
        let mut order: Vec<usize> = (0..plan.constant_chunks().len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(plan.constant_chunks()[i]));
        let activations = device.alloc(plan.activation_bytes(), false)?;
        let mut constants: Vec<Option<Buffer>> = (0..order.len()).map(|_| None).collect();
        for i in order {
            constants[i] = Some(device.alloc(plan.constant_chunks()[i], false)?);
        }
        let mut constants: Vec<Buffer> = constants.into_iter().map(|b| b.expect("every chunk allocated")).collect();
        let programs = device.alloc(Job::program_bytes(plan.task_count()), false)?;
        let base = job::base_of(std::iter::once(&activations).chain(&constants).chain(std::iter::once(&programs)));
        let at = AddressMap { activations: activations.dma_addr() - base, constants: constants.iter().map(|b| b.dma_addr() - base).collect() };
        let tasks = {
            let mut chunks: Vec<&mut [u8]> = constants.iter_mut().map(Buffer::as_mut_slice).collect();
            plan.emit(&at, &mut chunks)?
        };
        for c in &constants {
            c.sync_to_device(0..c.len())?;
        }
        let job = Job::new(&device, programs, &tasks, base)?;
        Ok(Model { device, job, activations, constants, plan })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Bytes of NPU memory the model holds.
    pub fn resident_bytes(&self) -> usize {
        self.job.resident_bytes() + self.activations.len() + self.constants.iter().map(Buffer::len).sum::<usize>()
    }

    pub fn input(&self) -> Placement {
        self.plan.placement(self.plan.input())
    }

    /// Run once on an RGB8 image (HWC, the input's height and width),
    /// quantized with the input's parameters.
    pub fn run_rgb8(&mut self, rgb: &[u8]) -> Result<Timing> {
        let p = self.input();
        if p.shape.c != 3 || rgb.len() != p.shape.h * p.shape.w * 3 {
            return Err(Error::Unsupported(format!("the model input is {:?}; got {} RGB bytes", p.shape, rgb.len())));
        }
        let lut: Vec<u8> = (0..=255u8).map(|v| p.q.quantize(f32::from(v) / 255.0) as u8).collect();
        self.run_with(|data, p| {
            for (pixel, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
                let atom = &mut data[p.offset + pixel * LANES..][..3];
                for (d, &v) in atom.iter_mut().zip(px) {
                    *d = lut[usize::from(v)];
                }
            }
        })
    }

    /// Run once, with `fill` writing the input into the data region at its
    /// placement (native16; see [`crate::layout`]). The input is taken per run
    /// because its bytes are reused by later tensors once nothing reads it.
    pub fn run_with(&mut self, fill: impl FnOnce(&mut [u8], Placement)) -> Result<Timing> {
        let start = Instant::now();
        let p = self.input();
        fill(self.activations.as_mut_slice(), p);
        let mut timing = Timing::default();
        self.activations.sync_to_device(p.range())?;
        timing.input = start.elapsed();
        for step in self.plan.steps() {
            let t = Instant::now();
            match step {
                Step::Npu { tasks } => {
                    self.job.run(&self.device, Some(tasks.clone()))?;
                    timing.npu += t.elapsed();
                }
                cpu => {
                    for r in cpu_reads(cpu) {
                        self.activations.sync_from_device(self.plan.placement(r).range())?;
                    }
                    let work = Instant::now();
                    self.plan.run_cpu_step(cpu, self.activations.as_mut_slice());
                    let spent = work.elapsed();
                    *match cpu {
                        Step::Add { .. } => &mut timing.add,
                        Step::MaxPool { .. } => &mut timing.max_pool,
                        _ => &mut timing.upsample,
                    } += spent;
                    self.activations.sync_to_device(self.plan.placement(cpu_writes(cpu)).range())?;
                    timing.cache += t.elapsed() - spent;
                }
            }
        }
        let t = Instant::now();
        for &o in self.plan.outputs() {
            self.activations.sync_from_device(self.plan.placement(o).range())?;
        }
        timing.output = t.elapsed();
        timing.total = start.elapsed();
        Ok(timing)
    }

    /// Output `index` in place, as the last run left it.
    pub fn output(&self, index: usize) -> TensorView<'_> {
        let p = self.plan.placement(self.plan.outputs()[index]);
        TensorView { data: &self.activations.as_slice()[p.offset..p.offset + p.shape.bytes()], placement: p }
    }

    /// The activation region (for byte-exact comparison with
    /// [`Plan::simulate`]).
    pub fn activations(&self) -> &[u8] {
        &self.activations.as_slice()[..self.plan.activation_bytes()]
    }
}

fn cpu_reads(step: &Step) -> Vec<TensorId> {
    match *step {
        Step::Npu { .. } => Vec::new(),
        Step::Add { a, b, .. } => vec![a, b],
        Step::MaxPool { input, .. } | Step::Upsample2x { input, .. } => vec![input],
    }
}

fn cpu_writes(step: &Step) -> TensorId {
    match *step {
        Step::Npu { .. } => unreachable!("NPU steps write through the device"),
        Step::Add { output, .. } | Step::MaxPool { output, .. } | Step::Upsample2x { output, .. } => output,
    }
}

/// A read-only view of a native16 tensor.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    data: &'a [u8],
    placement: Placement,
}

impl TensorView<'_> {
    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// The raw int8 code at `(y, x, c)`.
    pub fn get(&self, y: usize, x: usize, c: usize) -> i8 {
        let s = self.placement.shape;
        self.data[(c / LANES) * s.plane_bytes() + (y * s.w + x) * LANES + c % LANES] as i8
    }

    /// The real value at `(y, x, c)`.
    pub fn value(&self, y: usize, x: usize, c: usize) -> f32 {
        self.placement.q.dequantize(self.get(y, x, c))
    }
}
