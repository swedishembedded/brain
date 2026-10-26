// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of large models on a single GPU for its clients. If your
// team needs expertise in trading activation memory against recompute and precision then you can procure our
// services by sending an email to info@swedishembedded.com.

//! What a training forward keeps for its backward, as one model-agnostic policy.
//!
//! A layered decoder's backward needs, per layer, the layer's input and the intermediates computed from it. The
//! trainer can
//!
//! * [`SavedPolicy::Keep`] every layer's intermediates through the forward (fastest, most memory),
//! * [`SavedPolicy::Recompute`] them: keep only the layer-boundary residuals and rebuild one layer's intermediates
//!   at a time inside the backward (one extra forward), or
//! * [`SavedPolicy::Quantized`]: recompute, and hold the boundary residuals themselves in a narrower
//!   [`SavedFormat`] (bf16, or FP8 E4M3 with one scale per row).
//!
//! The policy is a value the planner chooses ([`crate::memplan`] prices every rung and [`crate::memplan::saved_policy`]
//! reads the choice back) and the trainers obey; no model decides it. A quantized boundary is part of the model: the
//! forward continues from the rounded value (store, then load, then the next layer), so the backward's recompute sees
//! exactly the input the forward used and the only deviation from fp32 is that one rounding per boundary.
//!
//! Recurrent state, norm statistics and softmax statistics are never stored through this module: they stay in
//! full precision wherever the trainer keeps them. Only the boundary residual stream is.
//!
//! [`BoundaryStore`] is the device store; [`SavedFormat::round_trip_rows`] is the host reference the kernels
//! (`pack_bf16`, `pack_fp8_rows`) are tested against.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::fp8::Fp8Format;
use crate::half::f32_to_bf16;

/// Kernels a trainer registers to hold boundaries in a narrow format; append to the model's kernel list.
#[must_use]
pub fn kernels() -> &'static [(&'static str, &'static str)] {
    &[
        ("pack_bf16", kernels::PACK_BF16),
        ("unpack_bf16", kernels::UNPACK_BF16),
        ("pack_fp8_rows", kernels::PACK_FP8_ROWS),
        ("unpack_fp8_rows", kernels::UNPACK_FP8_ROWS),
    ]
}

/// How one saved row-matrix (`rows x width`) is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SavedFormat {
    /// Full precision (4 B per element); the unquantized reference.
    F32,
    /// bf16, two per word (2 B per element): relative error 2^-8 per element.
    Bf16,
    /// FP8 E4M3 with one f32 scale per row (about 1 B per element): relative error 2^-4 per element against the
    /// row's own scale. `width` must be a multiple of 4.
    Fp8,
}

impl SavedFormat {
    pub const ALL: [SavedFormat; 3] = [SavedFormat::F32, SavedFormat::Bf16, SavedFormat::Fp8];

    pub fn label(self) -> &'static str {
        match self {
            SavedFormat::F32 => "f32",
            SavedFormat::Bf16 => "bf16",
            SavedFormat::Fp8 => "fp8",
        }
    }

    pub fn parse(s: &str) -> Option<SavedFormat> {
        SavedFormat::ALL.into_iter().find(|f| f.label() == s)
    }

    /// Bytes of `rows x width` elements in this format, counting the scales.
    pub fn bytes(self, rows: u64, width: u64) -> u64 {
        let elems = rows * width;
        match self {
            SavedFormat::F32 => 4 * elems,
            SavedFormat::Bf16 => 4 * elems.div_ceil(2),
            SavedFormat::Fp8 => 4 * elems.div_ceil(4) + 4 * rows,
        }
    }

    /// Whether the accuracy of holding a residual stream in this format has been measured well enough for the
    /// planner to choose it unasked. bf16 is (`qwen35` `lora_bf16_residuals`: the adapter gradients agree to
    /// bf16 precision); FP8 is available when a caller names it but is not picked by the planner until it is
    /// measured on a real model.
    pub fn validated(self) -> bool {
        !matches!(self, SavedFormat::Fp8)
    }

    /// The host reference: `x` (row-major, `width` per row) after one store and load in this format.
    pub fn round_trip_rows(self, x: &[f32], width: usize) -> Vec<f32> {
        assert!(width > 0 && x.len().is_multiple_of(width), "round_trip_rows: {} elements are not whole rows of {width}", x.len());
        match self {
            SavedFormat::F32 => x.to_vec(),
            SavedFormat::Bf16 => x.iter().map(|&v| f32::from_bits((f32_to_bf16(v) as u32) << 16)).collect(),
            SavedFormat::Fp8 => {
                assert!(width.is_multiple_of(4), "round_trip_rows: fp8 rows pack four codes to a word, width {width} is not a multiple of 4");
                x.chunks(width)
                    .flat_map(|row| {
                        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                        let scale = if amax == 0.0 { 1.0 } else { amax / Fp8Format::E4M3.max_finite() };
                        row.iter().map(move |&v| Fp8Format::E4M3.decode(Fp8Format::E4M3.encode(v / scale)) * scale)
                    })
                    .collect()
            }
        }
    }
}

/// What the forward keeps for the backward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SavedPolicy {
    /// Every layer's intermediates and boundaries, full precision.
    Keep,
    /// Layer-boundary residuals only (full precision); each layer is rebuilt in the backward.
    Recompute,
    /// As [`SavedPolicy::Recompute`], the boundary residuals held in the given narrower format.
    Quantized(SavedFormat),
}

impl SavedPolicy {
    /// The rungs the planner may choose among, widest first.
    pub const LADDER: [SavedPolicy; 4] = [SavedPolicy::Keep, SavedPolicy::Recompute, SavedPolicy::Quantized(SavedFormat::Bf16), SavedPolicy::Quantized(SavedFormat::Fp8)];

    /// Whether the backward rebuilds a layer's intermediates instead of reading kept ones.
    pub fn recomputes(self) -> bool {
        !matches!(self, SavedPolicy::Keep)
    }

    /// The format the boundary residuals are held in.
    pub fn boundary_format(self) -> SavedFormat {
        match self {
            SavedPolicy::Keep | SavedPolicy::Recompute => SavedFormat::F32,
            SavedPolicy::Quantized(f) => f,
        }
    }

    /// `Recompute` when `on`, else `Keep`: the policy of a caller that only knows the classic switch.
    pub fn recompute_if(on: bool) -> SavedPolicy {
        if on {
            SavedPolicy::Recompute
        } else {
            SavedPolicy::Keep
        }
    }

    /// `Recompute`, or `Quantized(f)` for a narrow `f`: the one constructor that cannot name an inconsistent pair.
    pub fn recompute_with(f: SavedFormat) -> SavedPolicy {
        if f == SavedFormat::F32 {
            SavedPolicy::Recompute
        } else {
            SavedPolicy::Quantized(f)
        }
    }

    /// The label the planner reports a choice under (and [`SavedPolicy::parse`] reads back).
    pub fn label(self) -> String {
        match self {
            SavedPolicy::Keep => "keep".into(),
            SavedPolicy::Recompute => "recompute".into(),
            SavedPolicy::Quantized(f) => format!("recompute+{}", f.label()),
        }
    }

    pub fn parse(label: &str) -> Option<SavedPolicy> {
        match label {
            "keep" => Some(SavedPolicy::Keep),
            "recompute" => Some(SavedPolicy::Recompute),
            other => other.strip_prefix("recompute+").and_then(SavedFormat::parse).filter(|f| *f != SavedFormat::F32).map(SavedPolicy::Quantized),
        }
    }
}

/// The boundary residual stream of one training pass: `boundaries` row-matrices of `rows x width`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SavedLayout {
    pub boundaries: u64,
    pub rows: u64,
    pub width: u64,
}

impl SavedLayout {
    /// Device bytes of the whole stream held in `format`.
    pub fn stream_bytes(&self, format: SavedFormat) -> u64 {
        self.boundaries * format.bytes(self.rows, self.width)
    }

    /// The two full-precision working buffers (a layer's input and output) a quantized stream decodes into.
    pub fn working_bytes(&self) -> u64 {
        2 * SavedFormat::F32.bytes(self.rows, self.width)
    }
}

/// The device store of a pass's boundary residuals in a narrow format.
pub struct BoundaryStore {
    format: SavedFormat,
    rows: u32,
    width: u32,
    slots: Vec<Slot>,
    pack: usize,
    unpack: usize,
}

enum Slot {
    Bf16(DeviceBuffer),
    Fp8 { codes: DeviceBuffer, scales: DeviceBuffer },
}

impl BoundaryStore {
    /// `boundaries` slots for `rows x width` residuals. `format` must be narrow; the model's kernel list must
    /// include [`kernels`].
    pub fn new(gpu: &Gpu, format: SavedFormat, boundaries: usize, rows: u32, width: u32) -> Result<BoundaryStore, String> {
        let total = rows.checked_mul(width).ok_or_else(|| format!("saved boundaries: {rows} x {width} elements overflow a u32 index"))?;
        let (pack, unpack, slot): (&str, &str, Box<dyn Fn() -> Slot>) = match format {
            SavedFormat::F32 => return Err("saved boundaries: a full-precision stream needs no store".into()),
            SavedFormat::Bf16 => ("pack_bf16", "unpack_bf16", Box::new(|| Slot::Bf16(gpu.storage(total.div_ceil(2) as u64)))),
            SavedFormat::Fp8 => {
                if !width.is_multiple_of(4) {
                    return Err(format!("saved boundaries: fp8 rows pack four codes to a word, width {width} is not a multiple of 4"));
                }
                ("pack_fp8_rows", "unpack_fp8_rows", Box::new(|| Slot::Fp8 { codes: gpu.storage((total / 4) as u64), scales: gpu.storage(rows as u64) }))
            }
        };
        let kernel = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("saved boundaries: kernel {name} is not registered (append model::saved::kernels() to the kernel list)"));
        Ok(BoundaryStore { format, rows, width, slots: (0..boundaries).map(|_| slot()).collect(), pack: kernel(pack)?, unpack: kernel(unpack)? })
    }

    pub fn format(&self) -> SavedFormat {
        self.format
    }

    /// Device bytes of the whole store.
    pub fn bytes(&self) -> u64 {
        self.slots.len() as u64 * self.format.bytes(self.rows as u64, self.width as u64)
    }

    /// The step that stores `src` (fp32 `[rows, width]`) as boundary `l`, for a caller recording a tape.
    pub fn store_step(&self, gpu: &Gpu, l: usize, src: &DeviceBuffer) -> Step {
        let total = self.rows * self.width;
        match &self.slots[l] {
            Slot::Bf16(packed) => gpu.step(self.pack, &[src, packed], &[total], total.div_ceil(2)),
            Slot::Fp8 { codes, scales } => gpu.step(self.pack, &[src, codes, scales], &[self.rows, self.width], self.rows),
        }
    }

    /// The step that decodes boundary `l` into `dst` (fp32 `[rows, width]`).
    pub fn load_step(&self, gpu: &Gpu, l: usize, dst: &DeviceBuffer) -> Step {
        let total = self.rows * self.width;
        match &self.slots[l] {
            Slot::Bf16(packed) => gpu.step(self.unpack, &[packed, dst], &[total], total),
            Slot::Fp8 { codes, scales } => gpu.step(self.unpack, &[codes, scales, dst], &[total, self.width], total),
        }
    }

    /// Store `src` as boundary `l` now.
    pub fn store(&self, gpu: &Gpu, l: usize, src: &DeviceBuffer) {
        gpu.submit(&[], &[self.store_step(gpu, l, src)]);
    }

    /// Decode boundary `l` into `dst` now.
    pub fn load(&self, gpu: &Gpu, l: usize, dst: &DeviceBuffer) {
        gpu.submit(&[], &[self.load_step(gpu, l, dst)]);
    }
}

/// The activation-recompute contract every model that offers [`SavedPolicy::Recompute`] honours, for its own test to
/// run: `build` makes a fresh trainable model (the same initial weights every call) and `step` runs one forward and
/// backward on a fixed batch (`zero_grads`, the batch, `forward`, `backward`), returning the loss.
///
/// * the recomputing build's loss and every parameter's gradient equal the build that keeps every layer, bit for bit
///   (each layer is replayed from its kept input exactly as the forward ran it);
/// * where the backend accounts for its allocations, the recomputing build holds fewer live device bytes: the
///   activation buffers of every layer but one per kind are released.
pub fn check_recompute<M: crate::Model>(name: &str, build: impl Fn() -> M, step: impl Fn(&M) -> f32) -> Result<(), String> {
    let live = |m: &M| m.optimizer_parts().and_then(|p| p.gpu.memory_usage()).map(|u| u.live);
    let grads = |m: &M| -> Vec<(String, Vec<u32>)> {
        let mut names = m.param_names();
        names.sort();
        names.into_iter().map(|n| (n.clone(), m.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect()
    };
    let keep = build();
    let keep_live = live(&keep);
    let want = step(&keep);
    let want_grads = grads(&keep);
    drop(keep);
    let mut rc = build();
    let before = live(&rc);
    rc.set_saved_policy(SavedPolicy::Recompute).map_err(|e| format!("{name}: {e}"))?;
    if let (Some(before), Some(after)) = (before.or(keep_live), live(&rc)) {
        if after >= before {
            return Err(format!("{name}: recomputing holds {after} live device bytes, keeping every layer {before}"));
        }
    }
    let got = step(&rc);
    if got.to_bits() != want.to_bits() {
        return Err(format!("{name}: recomputing gives loss {got}, keeping every layer {want}"));
    }
    if want_grads.is_empty() {
        return Err(format!("{name}: no trained parameter to compare"));
    }
    for ((n, a), (_, b)) in want_grads.iter().zip(grads(&rc)) {
        if let Some(i) = a.iter().zip(&b).position(|(x, y)| x != y) {
            return Err(format!("{name}: {n}[{i}] gradient {} recomputing, {} keeping every layer", f32::from_bits(b[i]), f32::from_bits(a[i])));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_policy_reads_back_from_its_label() {
        for p in SavedPolicy::LADDER {
            assert_eq!(SavedPolicy::parse(&p.label()), Some(p));
        }
        assert_eq!(SavedPolicy::parse("recompute+f32"), None, "a full-precision stream is plain recompute");
        assert_eq!(SavedPolicy::parse("offload"), None);
        assert_eq!(SavedPolicy::recompute_with(SavedFormat::F32), SavedPolicy::Recompute);
    }

    #[test]
    fn only_quantized_policies_narrow_the_boundary_and_keep_alone_does_not_recompute() {
        assert!(!SavedPolicy::Keep.recomputes());
        assert!(SavedPolicy::Recompute.recomputes());
        assert_eq!(SavedPolicy::Recompute.boundary_format(), SavedFormat::F32);
        assert_eq!(SavedPolicy::Quantized(SavedFormat::Fp8).boundary_format(), SavedFormat::Fp8);
    }

    #[test]
    fn narrower_formats_hold_fewer_bytes_counting_scales() {
        let (rows, width) = (1000, 64);
        assert_eq!(SavedFormat::F32.bytes(rows, width), 4 * 64_000);
        assert_eq!(SavedFormat::Bf16.bytes(rows, width), 2 * 64_000);
        assert_eq!(SavedFormat::Fp8.bytes(rows, width), 64_000 + 4 * 1000);
    }

    #[test]
    fn the_host_reference_stays_inside_each_formats_error_bound() {
        let width = 16;
        let x: Vec<f32> = (0..64 * width).map(|i| ((i as f32) * 0.37).sin() * (1.0 + (i % 7) as f32)).collect();
        for (format, rel) in [(SavedFormat::F32, 0.0), (SavedFormat::Bf16, 2f32.powi(-8)), (SavedFormat::Fp8, 2f32.powi(-4))] {
            let y = format.round_trip_rows(&x, width);
            for (row_x, row_y) in x.chunks(width).zip(y.chunks(width)) {
                let step = row_x.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 448.0;
                for (a, b) in row_x.iter().zip(row_y) {
                    // Relative to the element, plus the subnormal floor of the row's own scale for fp8.
                    let floor = if format == SavedFormat::Fp8 { step * 2f32.powi(-9) } else { 0.0 };
                    assert!((a - b).abs() <= rel * a.abs() + floor, "{format:?}: {a} stored as {b}");
                }
            }
        }
    }

    #[test]
    fn an_all_zero_fp8_row_decodes_to_zero() {
        assert_eq!(SavedFormat::Fp8.round_trip_rows(&[0.0; 8], 4), vec![0.0; 8]);
    }
}
