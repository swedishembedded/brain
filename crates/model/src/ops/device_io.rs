// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of models larger than the accelerator for its clients. If
// your team needs expertise in streaming weights and gradients between host and device, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The device a resident training state ([`crate::resident_ft`]) moves values and gradients through: a tiled upload
//! into a [`super::Weight`] ([`super::DeviceWeightSink`]), an in-place tile rewrite ([`super::rewrite_tile`]) and a
//! ranged gradient read. None of it runs a product kernel, so a model whose own kernels compute its products drives
//! such a state on its own device handle through [`StoreDevice`], which needs three kernels instead of the whole
//! [`super::Ops`] set. Being the model's own handle, the state's reads and writes are ordered with the model's
//! recorded work without a flush between handles.

use gpu_core::{DeviceBuffer, Gpu};

/// What a resident training state needs of a device.
pub trait DeviceIo {
    /// The handle the state reads, writes and measures gradient norms on.
    fn gpu(&self) -> &Gpu;
    /// The handle tiled uploads go through (a transfer stream where one is set up).
    fn upload_gpu(&self) -> &Gpu {
        self.gpu()
    }
    /// `words` f32 of `buf` from word `start`, staging only that range.
    fn read_range(&self, buf: &DeviceBuffer, start: u64, words: usize) -> Vec<f32>;
}

impl DeviceIo for super::Ops {
    fn gpu(&self) -> &Gpu {
        super::Ops::gpu(self)
    }
    fn upload_gpu(&self) -> &Gpu {
        super::Ops::upload_gpu(self)
    }
    fn read_range(&self, buf: &DeviceBuffer, start: u64, words: usize) -> Vec<f32> {
        super::Ops::read_range(self, buf, start, words)
    }
}

/// The kernels [`StoreDevice`] needs on the model's handle: the ranged copy and the device gradient norm.
#[must_use]
pub fn store_device_kernels() -> &'static [(&'static str, &'static str)] {
    &[("copy_cols", kernels::COPY_COLS), ("gradnorm_part", kernels::GRADNORM_PART), ("gradnorm_part_bf16", kernels::GRADNORM_PART_BF16)]
}

/// A [`DeviceIo`] over a model's own handle (one that registered [`store_device_kernels`]), borrowed for a call.
pub struct StoreDevice<'a> {
    gpu: &'a Gpu,
    copy_cols: usize,
}

impl<'a> StoreDevice<'a> {
    /// The state's view of `gpu`; an error names a kernel the handle did not register.
    pub fn new(gpu: &'a Gpu) -> Result<StoreDevice<'a>, String> {
        for (name, _) in store_device_kernels() {
            gpu.kernel_index(name).ok_or_else(|| format!("resident state: kernel '{name}' is not registered on this device (append model::ops::store_device_kernels())"))?;
        }
        let copy_cols = gpu.kernel_index("copy_cols").expect("checked above");
        Ok(StoreDevice { gpu, copy_cols })
    }
}

impl DeviceIo for StoreDevice<'_> {
    fn gpu(&self) -> &Gpu {
        self.gpu
    }
    fn read_range(&self, buf: &DeviceBuffer, start: u64, words: usize) -> Vec<f32> {
        super::tiled::read_range_with(self.gpu, self.copy_cols, buf, start, words)
    }
}
