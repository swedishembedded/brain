// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Mirage on brain's GPU op layer.
//!
//! [`Device`] owns an [`Ops`] façade over a [`Gpu`] chosen by the caller (the
//! CPU backend through the WGSL JIT, CUDA, Vulkan). Operators are built from
//! the shared kernels and the selector-aware GEMM; the host operators in
//! [`crate::vision`] and [`crate::decoder`] stay as the parity oracle.
//!
//! Swedish Embedded AB implements solutions for running perception models on
//! whatever accelerator a customer already owns. If your team needs expertise
//! in portable GPU inference then you can procure our services by sending an
//! email to info@swedishembedded.com.

pub mod decoder;
pub mod form;
pub mod hadamard;
pub mod hconv;
pub mod kron;
pub mod lookup;
pub mod batch_model;
pub use batch_model::{BatchControl, DecoderMemory, Described, Extracted, Image, MirageModel, Timings};
pub mod prims;
pub mod reader;
pub mod train;
pub mod vision;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::ops::{Ops, Weight};

use crate::vision::ops::{Linear, NORM_EPS};

/// Kernels Mirage dispatches by name that the shared list may not carry.
const OWN_KERNELS: &[(&str, &str)] = &[
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("mirage_rms_axpy", kernels::MIRAGE_RMS_AXPY),
    ("hconv_glu", kernels::HCONV_GLU),
    ("mirage_conv2d", kernels::MIRAGE_CONV2D),
    ("mirage_dwconv", kernels::MIRAGE_DWCONV),
    ("mirage_rows_mask", kernels::MIRAGE_ROWS_MASK),
    ("mirage_add_masked", kernels::MIRAGE_ADD_MASKED),
    ("mirage_gated_mix", kernels::MIRAGE_GATED_MIX),
    ("mirage_grn_energy", kernels::MIRAGE_GRN_ENERGY),
    ("mirage_grn_scale", kernels::MIRAGE_GRN_SCALE),
    ("mirage_grn_apply", kernels::MIRAGE_GRN_APPLY),
    ("mirage_chan_fma", kernels::MIRAGE_CHAN_FMA),
    ("mirage_axpy", kernels::MIRAGE_AXPY),
    ("mirage_swiglu_rows", kernels::MIRAGE_SWIGLU_ROWS),
    ("mirage_mean_h", kernels::MIRAGE_MEAN_H),
    ("mirage_rope2d", kernels::MIRAGE_ROPE2D),
    ("mirage_attn", kernels::MIRAGE_ATTN),
    ("mirage_attn_scores", kernels::MIRAGE_ATTN_SCORES),
    ("mirage_attn_softmax", kernels::MIRAGE_ATTN_SOFTMAX),
    ("mirage_attn_apply", kernels::MIRAGE_ATTN_APPLY),
    ("mirage_upsample2", kernels::MIRAGE_UPSAMPLE2),
    ("mirage_embed_lanes", kernels::MIRAGE_EMBED_LANES),
    ("mirage_gather_rows", kernels::MIRAGE_GATHER_ROWS),
    ("mirage_engram_gate", kernels::MIRAGE_ENGRAM_GATE),
    ("mirage_mhc_maps", kernels::MIRAGE_MHC_MAPS),
    ("mirage_mhc_mix", kernels::MIRAGE_MHC_MIX),
    ("mirage_mhc_post", kernels::MIRAGE_MHC_POST),
    ("mirage_taps", kernels::MIRAGE_TAPS),
    ("mirage_tok_at", kernels::MIRAGE_TOK_AT),
    ("mirage_pos_inc", kernels::MIRAGE_POS_INC),
    ("mirage_tok_put", kernels::MIRAGE_TOK_PUT),
    ("mirage_engram_rows", kernels::MIRAGE_ENGRAM_ROWS),
    ("mirage_sigmoid_mul", kernels::MIRAGE_SIGMOID_MUL),
    ("mirage_row_write", kernels::MIRAGE_ROW_WRITE),
    ("mirage_rope_pos", kernels::MIRAGE_ROPE_POS),
    ("mirage_hada_mid", kernels::MIRAGE_HADA_MID),
    ("mirage_lane_mean", kernels::MIRAGE_LANE_MEAN),
    ("mirage_band_cols", kernels::MIRAGE_BAND_COLS),
    ("mirage_hconv_band_glu", kernels::MIRAGE_HCONV_BAND_GLU),
    ("mirage_kron_rows", kernels::MIRAGE_KRON_ROWS),
    ("mirage_kron_hada", kernels::MIRAGE_KRON_HADA),
    ("mirage_cq_gather", kernels::MIRAGE_CQ_GATHER),
    ("argmax_row", kernels::ARGMAX_ROW),
    ("mirage_argmax_wg", kernels::MIRAGE_ARGMAX_WG),
    ("silu", kernels::SILU),
    ("add", kernels::ADD),
    ("mul", kernels::MUL),
    ("mirage_copy_rows", kernels::MIRAGE_COPY_ROWS),
    ("sigmoid", kernels::SIGMOID),
    ("bias_add", kernels::BIAS_ADD),
];

/// Kernels Mirage needs beyond what [`model::ops::kernel_list`] carries.
fn kernel_list() -> Vec<(&'static str, &'static str)> {
    let mut list = model::ops::kernel_list().to_vec();
    // The rotated-codebook products of a model loaded from the deployment archive's packed matrices.
    for &(name, src) in model::ops::cq_kernels().iter().chain(OWN_KERNELS).chain(train::KERNELS) {
        if !list.iter().any(|(n, _)| *n == name) {
            list.push((name, src));
        }
    }
    list
}

/// A device and the kernel indices Mirage dispatches by name.
pub struct Device {
    ops: Ops,
    kernel: std::collections::HashMap<&'static str, usize>,
    /// One word that position-reading kernels bind when their position is a parameter instead.
    pub(crate) pos_dummy: DeviceBuffer,
    /// Bytes of matrix weights uploaded through [`Device::weight`] (dense f32) and [`Device::linear_weight`] (packed).
    resident: ResidentWeights,
    form: WeightForm,
}

/// What the device holds for its matrix weights.
#[derive(Debug, Default)]
struct ResidentWeights {
    dense: std::sync::atomic::AtomicU64,
    packed: std::sync::atomic::AtomicU64,
    structured: std::sync::atomic::AtomicU64,
}

/// Resident bytes of a model's matrix weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightBytes {
    /// Dense f32 matrices (the folded convolution and Hadamard matrices, and every matrix of an FP32 model).
    pub dense: u64,
    /// Rotated-codebook matrices at their stored size (indices, fp16 norms, codebook).
    pub packed: u64,
    /// The parameters of structured operators at their original size (bands, Kronecker factors, permutations).
    pub structured: u64,
}

/// How the structured mixers (HConvGLU channel bands, Hadamard rotations) are resident and run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightForm {
    /// Kernels on the original parameters: Walsh and Kronecker factors, 65-tap bands.
    #[default]
    Structured,
    /// The transforms multiplied out into dense `n x n` matrices, run by the selector's GEMMs.
    Folded,
}

impl Device {
    /// Open the device the process selected (`BRAIN_DEVICE` / `BRAIN_BACKEND`).
    pub fn open() -> Result<Device, String> {
        Device::new(Gpu::new(&kernel_list()))
    }

    /// Build over an existing device; its kernel list must be [`Device::kernels`].
    pub fn new(gpu: Gpu) -> Result<Device, String> {
        Device::new_with(gpu, WeightForm::default())
    }

    /// [`Device::new`] with the residency form of the structured mixers chosen.
    pub fn new_with(gpu: Gpu, form: WeightForm) -> Result<Device, String> {
        let ops = Ops::new(gpu)?;
        let mut kernel = std::collections::HashMap::new();
        for &(name, _) in OWN_KERNELS.iter().chain(train::KERNELS) {
            let index = ops.gpu().kernel_index(name).ok_or_else(|| format!("the device lacks kernel `{name}`"))?;
            kernel.insert(name, index);
        }
        let pos_dummy = ops.gpu().storage(1);
        Ok(Device { ops, kernel, pos_dummy, resident: ResidentWeights::default(), form })
    }

    /// The kernel list a [`Gpu`] must be created with for [`Device::new`].
    pub fn kernels() -> Vec<(&'static str, &'static str)> {
        kernel_list()
    }

    /// Pipeline index of a kernel by name.
    pub(crate) fn k(&self, name: &str) -> usize {
        self.kernel[name]
    }

    /// The underlying device.
    pub fn gpu(&self) -> &Gpu {
        self.ops.gpu()
    }

    /// Choose how the packed (rotated-codebook) matrices read their activations; the default is f32, int8 is the
    /// opt-in A8 precision (`model::cq::CqActivations`).
    pub fn set_cq_activations(&mut self, activations: model::cq::CqActivations) {
        self.ops.set_cq_activations(activations);
    }

    /// The op façade.
    pub fn ops(&self) -> &Ops {
        &self.ops
    }

    /// Upload a `[out, in]` matrix as a resident GEMM weight.
    pub fn weight(&self, name: &str, wt: &[f32], out: usize, inn: usize) -> Weight {
        assert_eq!(wt.len(), out * inn);
        self.resident.dense.fetch_add(wt.len() as u64 * 4, std::sync::atomic::Ordering::Relaxed);
        Weight::dense_f32(self.gpu().storage_init(name, wt), out as u32, inn as u32)
    }

    /// The resident weight of a host linear: its stored rotated-codebook form when the model was loaded packed
    /// (index words, norms and codebook stay resident, 2-4 bits per weight), otherwise the dense f32 matrix.
    pub fn linear_weight(&self, name: &str, l: &Linear) -> Weight {
        match &l.cq {
            Some(cq) => self.packed_weight(name, cq),
            None => self.weight(name, &l.wt, l.out, l.inn),
        }
    }

    /// A rotated-codebook weight uploaded as stored.
    pub fn packed_weight(&self, name: &str, cq: &model::cq::Cq) -> Weight {
        self.resident.packed.fetch_add(cq.bytes() as u64, std::sync::atomic::Ordering::Relaxed);
        Weight::cq(&self.ops, name, cq)
    }

    /// The embedding table as a dense f32 gather source, counted with the dense weights.
    pub(crate) fn gather_table(&self, values: &[f32]) -> DeviceBuffer {
        self.resident.dense.fetch_add(values.len() as u64 * 4, std::sync::atomic::Ordering::Relaxed);
        self.buf(values)
    }

    /// A resident parameter of a structured operator (band taps, Kronecker factors, permutations), counted separately.
    pub fn param_buffer(&self, name: &str, values: &[f32]) -> DeviceBuffer {
        self.resident.structured.fetch_add(values.len() as u64 * 4, std::sync::atomic::Ordering::Relaxed);
        self.gpu().storage_init(name, values)
    }

    /// The residency form of the structured mixers.
    pub fn form(&self) -> WeightForm {
        self.form
    }

    /// Bytes of matrix weights uploaded so far.
    pub fn weight_bytes(&self) -> WeightBytes {
        WeightBytes { dense: self.resident.dense.load(std::sync::atomic::Ordering::Relaxed), packed: self.resident.packed.load(std::sync::atomic::Ordering::Relaxed), structured: self.resident.structured.load(std::sync::atomic::Ordering::Relaxed) }
    }

    /// The rows of several linears sharing one input, as one weight: stacked in packed form when every part has
    /// one, else concatenated dense.
    pub fn linear_weight_cat(&self, name: &str, parts: &[&Linear]) -> Weight {
        let inn = parts[0].inn;
        assert!(parts.iter().all(|p| p.inn == inn), "linears stacked into one weight share their input width");
        if let Some(packed) = parts.iter().map(|p| p.cq.as_deref()).collect::<Option<Vec<_>>>() {
            if let Ok(cq) = model::cq::Cq::concat_rows(&packed) {
                return self.packed_weight(name, &cq);
            }
        }
        let wt: Vec<f32> = parts.iter().flat_map(|p| p.wt.iter().copied()).collect();
        self.weight(name, &wt, parts.iter().map(|p| p.out).sum(), inn)
    }

    /// `y[rows, out] = x[rows, in] W^T` with the selected GEMM.
    pub fn linear(&self, s: &mut Vec<Step>, w: &Weight, x: &DeviceBuffer, rows: u32, y: &DeviceBuffer) {
        // Every Mirage weight (dense f32 or packed CQ) reads the f32 activations; `Ops::act` would also pack an int8
        // copy (two reduction dispatches per product) that nothing reads.
        debug_assert!(w.reads_f32_activations());
        let act = self.ops.act_f32(x, 0, rows, w.k());
        self.ops.matmul(s, w, &act, y, 0);
    }

    /// `(1 + scale)`-weighted RMS norm over rows of width `d`: `weight` is `1 + scale`.
    pub fn zc_norm(&self, s: &mut Vec<Step>, x: &DeviceBuffer, weight: &DeviceBuffer, rows: u32, d: u32, y: &DeviceBuffer) {
        s.push(self.gpu().dispatch(self.k("rmsnorm_rows"), &[x, weight, y], &[d, rows, NORM_EPS.to_bits()], gpu_core::Dispatch::Workgroups(rows)));
    }

    /// `y += scale * zc_norm(x)` row by row, one dispatch (the norm and the residual add of a gated branch).
    pub fn zc_norm_axpy(&self, s: &mut Vec<Step>, scale: f32, x: &DeviceBuffer, weight: &DeviceBuffer, rows: u32, d: u32, y: &DeviceBuffer) {
        s.push(self.gpu().dispatch(self.k("mirage_rms_axpy"), &[x, weight, y], &[d, rows, NORM_EPS.to_bits(), scale.to_bits()], gpu_core::Dispatch::Workgroups(rows)));
    }

    /// The gated product of HConvGLU from a fused `[rows, 2n]` projection.
    pub fn hconv_glu(&self, s: &mut Vec<Step>, ab: &DeviceBuffer, bias: &DeviceBuffer, rows: u32, n: u32, z: &DeviceBuffer) {
        s.push(self.gpu().step(self.k("hconv_glu"), &[ab, bias, z], &[rows, n], rows * n));
    }

    /// Read `len` values of `buf` starting at value `offset`.
    pub fn read_at(&self, buf: &DeviceBuffer, offset: usize, len: usize) -> Vec<f32> {
        let mut out = vec![0f32; len];
        self.gpu().read_into(buf, offset as u64, &mut out);
        out
    }

    /// Run `steps` and read `len` values of `buf`.
    pub fn run_read(&self, steps: &[Step], buf: &DeviceBuffer, len: usize) -> Vec<f32> {
        self.gpu().submit(&[], steps);
        self.gpu().read(buf, len)
    }
}
