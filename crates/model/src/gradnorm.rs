// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of large language models for its clients. If your
// team needs expertise in measuring a global gradient norm over gradients that never leave the device then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The sum of squares of device gradients, reduced on the device.
//!
//! A global gradient-norm clip needs `sum(g^2)` over every gradient before any update. The reduction runs as the
//! cooperative `gradnorm_part` kernel (f32 gradients) or its `gradnorm_part_bf16` twin (bf16 gradients, never widened):
//! each buffer is split over up to 512 workgroups, each writing one partial, and only those partials (a few hundred
//! floats a buffer) are read back and added in f64. The gradients themselves never cross to the host.

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::precision::GradDtype;

/// One gradient buffer to reduce: `numel` elements stored as `dtype` (bf16 packs two to a word).
#[derive(Clone, Copy)]
pub struct NormItem<'a> {
    pub buf: &'a DeviceBuffer,
    pub numel: usize,
    pub dtype: GradDtype,
}

/// The kernels and the partials buffer of the device reduction.
pub struct DeviceGradNorm {
    part_f32: usize,
    part_bf16: usize,
    parts: DeviceBuffer,
    capacity: u32,
}

/// Partials one reduction can hold before it is split into rounds.
const PARTS_CAPACITY: u32 = 4096;

impl DeviceGradNorm {
    /// Resolve the kernels on `gpu`; an error names the missing one.
    pub fn new(gpu: &Gpu) -> Result<DeviceGradNorm, String> {
        let find = |n: &str| gpu.kernel_index(n).ok_or_else(|| format!("device gradient norm: kernel '{n}' is not registered on this device"));
        Ok(DeviceGradNorm { part_f32: find("gradnorm_part")?, part_bf16: find("gradnorm_part_bf16")?, parts: gpu.storage(u64::from(PARTS_CAPACITY)), capacity: PARTS_CAPACITY })
    }

    /// The sum of squares of each item, in the order given (f64 sums of the device's f32 partials). Items are
    /// reduced in rounds that fit the partials buffer, one submission and one small readback per round.
    pub fn sum_sq_each(&self, gpu: &Gpu, items: &[NormItem<'_>]) -> Vec<f64> {
        let mut out = Vec::with_capacity(items.len());
        let mut round: Vec<(NormItem<'_>, u32, u32)> = Vec::new();
        let mut used = 0u32;
        for item in items {
            let nwg = paramstore::gradnorm_parts(item.numel);
            if used + nwg > self.capacity {
                self.reduce_round(gpu, &round, used, &mut out);
                round.clear();
                used = 0;
            }
            round.push((*item, used, nwg));
            used += nwg;
        }
        if !round.is_empty() {
            self.reduce_round(gpu, &round, used, &mut out);
        }
        out
    }

    /// The sum of squares of every item.
    pub fn sum_sq(&self, gpu: &Gpu, items: &[NormItem<'_>]) -> f64 {
        self.sum_sq_each(gpu, items).iter().sum()
    }

    fn reduce_round(&self, gpu: &Gpu, round: &[(NormItem<'_>, u32, u32)], used: u32, out: &mut Vec<f64>) {
        let mut uniforms = Vec::with_capacity(round.len());
        let mut steps = Vec::with_capacity(round.len());
        for (item, off, nwg) in round {
            assert!(item.dtype == GradDtype::F32 || item.numel.is_multiple_of(2), "a bf16 gradient has an even element count (got {})", item.numel);
            let kernel = if item.dtype == GradDtype::Bf16 { self.part_bf16 } else { self.part_f32 };
            let ub = gpu.uniform_dynamic(3);
            gpu.write(&ub, &[item.numel as u32, *off, *nwg]);
            steps.push(gpu.step_buf(kernel, &ub, &[item.buf, &self.parts], nwg * 64));
            uniforms.push(ub);
        }
        gpu.submit(&[], &steps);
        let parts = gpu.read(&self.parts, used as usize);
        out.extend(round.iter().map(|(_, off, nwg)| parts[*off as usize..(*off + *nwg) as usize].iter().map(|&p| f64::from(p)).sum::<f64>()));
    }
}
