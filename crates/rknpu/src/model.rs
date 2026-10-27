// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A compiled [`Plan`] resident on the NPU, ready to run.
//!
//! The arena is cacheable, so the CPU and the NPU hand tensors over through
//! explicit cache maintenance: the input is written back before the first
//! segment, a CPU step invalidates what it reads and writes back what it
//! produces, and the outputs are invalidated after the last step. Every tensor
//! is 64-byte aligned, so no maintenance ever touches a neighbour's lines.

use std::time::{Duration, Instant};

use crate::driver::Device;
use crate::graph::TensorId;
use crate::job::Job;
use crate::layout::LANES;
use crate::plan::{Placement, Plan, Step};
use crate::Result;

/// Where one run's time went.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub npu: Duration,
    pub cpu: Duration,
    pub total: Duration,
}

pub struct Model {
    device: Device,
    job: Job,
    plan: Plan,
}

impl Model {
    /// Allocate NPU memory for `plan`, write its constants and programs.
    pub fn load(device: Device, plan: Plan) -> Result<Model> {
        let mut job = Job::new(&device, plan.data_bytes(), plan.tasks())?;
        plan.write_constants(job.data_mut());
        job.flush_all()?;
        Ok(Model { device, job, plan })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Bytes of NPU memory the model holds.
    pub fn resident_bytes(&self) -> usize {
        self.job.resident_bytes()
    }

    pub fn input(&self) -> Placement {
        self.plan.placement(self.plan.input())
    }

    /// Write the input with `fill`, which gets the data region and the
    /// input's placement (native16; see [`crate::layout`]).
    pub fn write_input(&mut self, fill: impl FnOnce(&mut [u8], Placement)) {
        let p = self.input();
        fill(self.job.data_mut(), p);
    }

    /// Write an RGB8 image (HWC, the input's shape) as the input, quantized
    /// with the input's parameters.
    pub fn write_input_rgb8(&mut self, rgb: &[u8]) {
        let p = self.input();
        assert_eq!(rgb.len(), p.shape.h * p.shape.w * 3, "rgb8 input must be {}x{}x3", p.shape.h, p.shape.w);
        assert_eq!(p.shape.c, 3, "the model input has {} channels, not 3", p.shape.c);
        let lut: Vec<u8> = (0..=255u8).map(|v| p.q.quantize(f32::from(v) / 255.0) as u8).collect();
        let data = &mut self.job.data_mut()[p.offset..];
        for (pixel, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
            let atom = &mut data[pixel * LANES..pixel * LANES + 3];
            for (d, &v) in atom.iter_mut().zip(px) {
                *d = lut[usize::from(v)];
            }
        }
    }

    /// Run the whole plan once.
    pub fn run(&mut self) -> Result<Timing> {
        let start = Instant::now();
        let mut timing = Timing::default();
        self.job.flush(self.input().range())?;
        for step in self.plan.steps() {
            let t = Instant::now();
            match step {
                Step::Npu { tasks } => {
                    self.job.run(&self.device, Some(tasks.clone()))?;
                    timing.npu += t.elapsed();
                }
                cpu => {
                    for r in cpu_reads(cpu) {
                        self.job.invalidate(self.plan.placement(r).range())?;
                    }
                    self.plan.run_cpu_step(cpu, self.job.data_mut());
                    self.job.flush(self.plan.placement(cpu_writes(cpu)).range())?;
                    timing.cpu += t.elapsed();
                }
            }
        }
        for &o in self.plan.outputs() {
            self.job.invalidate(self.plan.placement(o).range())?;
        }
        timing.total = start.elapsed();
        Ok(timing)
    }

    /// Output `index` in place, as the last run left it.
    pub fn output(&self, index: usize) -> TensorView<'_> {
        let p = self.plan.placement(self.plan.outputs()[index]);
        TensorView { data: &self.job.data()[p.offset..p.offset + p.shape.bytes()], placement: p }
    }

    /// The whole data region (for byte-exact comparison with
    /// [`Plan::simulate`]).
    pub fn data(&self) -> &[u8] {
        self.job.data()
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
