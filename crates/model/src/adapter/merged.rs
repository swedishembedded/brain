// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for low-rank adaptation of every learned tensor of a model, not only its
// linear layers, for its clients. If your team needs expertise in parameter-efficient fine-tuning of queries, gates and
// other small parameter matrices then you can procure our services by sending an email to info@swedishembedded.com.

//! A LoRA pair over a small f32 parameter matrix that is not read by a linear layer: a bank of learned queries, the
//! factors of a gate, a table. The operator that reads such a matrix takes it whole (a depth-attention query bank, a
//! message gate's weights), so the correction cannot ride beside a product the way [`super::exec::AdapterRuntime`]
//! corrects a linear's output. Instead the effective matrix is materialised once per step,
//!
//! `W_eff = W + scale * B A`,   `A [rank, inn]`, `B [out, rank]`,
//!
//! the operator reads `W_eff` and writes its gradient `dW_eff` as it would for a trainable matrix, and the pair's
//! gradients are projected from it: `dB += scale * dW_eff A^T`, `dA += scale * B^T dW_eff`. This is exact (it is the
//! chain rule through the sum), and cheap for the matrices it is meant for: one `[out, inn]` buffer and three products
//! of rank `rank` per step. A large linear keeps the runtime correction, which never forms `W_eff`.
//!
//! The products run on the fp32 kernels whatever the model's [`Ops::set_gemm_precision`] ([`crate::ops::fp32_kernels`]):
//! `W_eff` is a weight, which the served adapter bank forms exactly on the host
//! ([`crate::lora::device_adapter::fold_delta`]), and a TF32 tier would round the factors (about 5e-4 relative) into
//! every product that reads it; three rank-`rank` products a step gain nothing from tensor cores.
//!
//! The adapter's tensors are stored like any LoRA pair (`{name}.lora_a`, `{name}.lora_b`), so saving, reading back and
//! folding an adapter are unchanged.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

use crate::ops::{Ops, Weight};

/// Kernels a [`MergedLora`] dispatches beyond [`Ops`]'s products: append to the model's list.
pub const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY), ("splice_at", kernels::SPLICE_AT)];

/// One merged LoRA pair over an `[out, inn]` f32 matrix: the effective matrix and its gradient buffer.
pub struct MergedLora {
    out: u32,
    inn: u32,
    rank: u32,
    scale: f32,
    eff: DeviceBuffer,
    d_eff: DeviceBuffer,
    axpy: usize,
    copy: usize,
}

impl MergedLora {
    /// The buffers of a pair of `rank` over an `[out, inn]` matrix whose product is scaled by `scale`. The device must
    /// have registered [`KERNELS`].
    pub fn new(gpu: &Gpu, out: u32, inn: u32, rank: u32, scale: f32) -> Result<MergedLora, String> {
        if out == 0 || inn == 0 || rank == 0 {
            return Err(format!("merged LoRA: a rank-{rank} pair over [{out}, {inn}] is empty"));
        }
        if !scale.is_finite() {
            return Err(format!("merged LoRA: scale {scale} is not finite"));
        }
        let kernel = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("merged LoRA: kernel {name} is not registered (append model::adapter::merged::KERNELS)"));
        let n = u64::from(out) * u64::from(inn);
        Ok(MergedLora { out, inn, rank, scale, eff: gpu.storage(n), d_eff: gpu.storage(n), axpy: kernel("axpy")?, copy: kernel("splice_at")? })
    }

    /// `W_eff`, what the operator reads after [`Self::refresh`].
    pub fn eff(&self) -> &DeviceBuffer {
        &self.eff
    }

    /// `dW_eff`, where the operator ADDS its gradient; [`Self::project`] reads it. Zero it before each backward.
    pub fn d_eff(&self) -> &DeviceBuffer {
        &self.d_eff
    }

    fn numel(&self) -> u32 {
        self.out * self.inn
    }

    /// Record `W_eff = base + scale * B A` from the pair's current factors.
    pub fn refresh(&self, ops: &Ops, s: &mut Vec<Step>, base: &DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        let g = ops.gpu();
        let n = self.numel();
        let prod = g.storage(u64::from(n));
        // dX[out, inn] = dY[out, rank] W[rank, inn]: the product B A.
        crate::ops::fp32_kernels(|| ops.matmul_dx(s, &Weight::dense_f32(a.clone(), self.rank, self.inn), b, self.out, &prod, false));
        s.push(g.step(self.copy, &[base, &self.eff], &[n, 0, 0], n));
        s.push(g.step(self.axpy, &[&self.eff, &prod], &[n, f(self.scale)], n));
    }

    /// Record the pair's gradients from `dW_eff`, ADDED into `ga` (`[rank, inn]`, `None` for a frozen `A`) and `gb`
    /// (`[out, rank]`).
    pub fn project(&self, ops: &Ops, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, ga: Option<&DeviceBuffer>, gb: &DeviceBuffer) {
        let g = ops.gpu();
        let (out, inn, r) = (self.out, self.inn, self.rank);
        // dW_eff A^T: rows of dW_eff against the rows of A.
        let db = g.storage(u64::from(out * r));
        crate::ops::fp32_kernels(|| ops.matmul(s, &Weight::dense_f32(a.clone(), r, inn), &ops.act_f32(&self.d_eff, 0, out, inn), &db, 0));
        s.push(g.step(self.axpy, &[gb, &db], &[out * r, f(self.scale)], out * r));
        if let Some(ga) = ga {
            // B^T dW_eff, accumulated into a fresh (zeroed) buffer.
            let da = g.storage(u64::from(r * inn));
            crate::ops::fp32_kernels(|| ops.matmul_dw(s, &self.d_eff, b, out, r, inn, &da));
            s.push(g.step(self.axpy, &[ga, &da], &[r * inn, f(self.scale)], r * inn));
        }
    }
}
