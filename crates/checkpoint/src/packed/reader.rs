// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Tile reads over checkpoints: [`PackedTensorReader`] hands out the packed
//! bytes of a flat element range and nothing else.
//!
//! There is deliberately no whole-tensor method and no fallback to one: a
//! source that cannot serve a range (a torch pickle, a computed checkpoint)
//! reports an error at [`PackedTensorReader::descriptor`] time instead of
//! quietly decoding the tensor whole. [`PackedReader`] serves memory-mapped
//! safetensors (single file or sharded HF directory) and GGUF (single or split).

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::Path;

use super::codec::TileBuf;
use super::encoding::{Element, FloatFormat, QuantEncoding, ScaleGranularity, TensorDescriptor, INT8_SCALE_SUFFIX};
use super::hfquant::{DeclaredWeight, HfQuant, Nvfp4Global, TensorDeclaration};
use crate::gguf::GgmlType;
use crate::weightio::WeightReader;

/// The `__metadata__` key holding the JSON map of tensor name to logical shape and encoding spec that a packed
/// file written by `convert_to_file` carries.
pub const PACKED_METADATA_KEY: &str = "brain.packed";

/// Read-only, tile-granular access to a checkpoint's quantized tensors.
pub trait PackedTensorReader: Send + Sync {
    /// Payload tensor names. Sidecar scale tensors that belong to a payload
    /// tensor are not listed; they are read through it.
    fn names(&self) -> Vec<String>;

    /// What `name` is, or why it cannot be read as packed tiles.
    fn descriptor(&self, name: &str) -> Result<TensorDescriptor, String>;

    /// Fill `dst` with the payload bytes and scale values covering the flat
    /// element range `elems` (aligned to `desc.tile_quantum()`), replacing its
    /// contents. Reads and copies only those bytes.
    fn read_blocks(&self, desc: &TensorDescriptor, elems: &Range<u64>, dst: &mut TileBuf) -> Result<(), String>;

    /// Hint that tile `elems` is consumed: drop its file-backed pages from
    /// this process's residency. Best effort; a later read re-faults them.
    fn release(&self, desc: &TensorDescriptor, elems: &Range<u64>);
}

/// [`PackedTensorReader`] over a safetensors file, HF directory or GGUF.
pub struct PackedReader {
    inner: WeightReader,
    overrides: HashMap<String, TensorDescriptor>,
    hidden: HashSet<String>,
    /// The FP8 tile size the checkpoint declares (`weight_block_size`), tried
    /// before the inference candidates.
    fp8_block: Option<u32>,
    /// What the release declares about its quantization (`config.json`, `hf_quant_config.json`).
    declared: Option<HfQuant>,
    /// Scale sidecars the file does not store but declares how to compute, computed once when the declaration is
    /// adopted and served like stored ones: the NVFP4 `<w>_scale_2` of a release storing `<w>_amax` and
    /// `<w>_outer_scale_denominator` (little-endian f32 `amax / denominator`, per slab or per tensor).
    derived: HashMap<String, Vec<u8>>,
}

/// The suffixes of an NVFP4 second level declared as `amax / outer_scale_denominator`.
const AMAX_SUFFIX: &str = "_amax";
const DENOMINATOR_SUFFIX: &str = "_outer_scale_denominator";
/// The suffix of the f32 second level of NVFP4, stored (ModelOpt) or derived from the amax sidecars.
const GLOBAL_SUFFIX: &str = "_scale_2";

/// The quantization sidecars of `reader`'s checkpoint: every `<w>_scale_inv`, `<w>_scale`, `<w>_scale_2`, `<w>_amax`
/// and `<w>_outer_scale_denominator` whose payload `<w>` (named `<module>.weight` or `<base>_weight`) is stored with a
/// scale, and the `input_scale` of every such module. They describe the payload they sit beside and are never
/// parameters of a model, so an importer that walks tensor names skips them (and reads the payload through
/// [`PackedReader`], which applies the scales).
pub fn quant_sidecars(reader: &WeightReader) -> HashSet<String> {
    let names: HashSet<&str> = reader.names().collect();
    let mut out = HashSet::new();
    for n in &names {
        if !n.ends_with(".weight") && !n.ends_with("_weight") {
            continue;
        }
        let has_scale = ["_scale_inv", "_scale"].iter().any(|s| names.contains(format!("{n}{s}").as_str()));
        if has_scale {
            out.extend(["_scale_inv", "_scale", GLOBAL_SUFFIX, AMAX_SUFFIX, DENOMINATOR_SUFFIX].iter().map(|s| format!("{n}{s}")));
            if let Some(stem) = n.strip_suffix(".weight") {
                out.insert(format!("{stem}.input_scale"));
            }
        }
    }
    out.retain(|n| names.contains(n.as_str()));
    out
}

/// FP8 `weight_block_size` candidates tried when inferring the tile from the
/// scale grid's shape, most common first. The first that reproduces the grid
/// exactly wins (a grid alone is ambiguous: 3 bands over 300 rows fits 100 and 128).
const FP8_BLOCKS: [u32; 5] = [128, 64, 32, 256, 16];

/// Elements per scale in brain's int8-native checkpoints (`weightio::PACKED_INT8_LAYOUT` 2, `model::int8::GROUP`).
const INT8_NATIVE_GROUP: u32 = 32;

impl PackedReader {
    /// Open a `.safetensors`/`.gguf` file or a HF checkpoint directory.
    pub fn open(path: &Path) -> Result<PackedReader, String> {
        let inner = WeightReader::open_hf_dir(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let declared = if path.is_dir() { HfQuant::from_dir(path)? } else { None };
        Ok(Self::new(inner).with_declaration(declared))
    }

    pub fn new(inner: WeightReader) -> PackedReader {
        let overrides = Self::declared_in_files(&inner);
        let mut r = PackedReader { inner, overrides, hidden: HashSet::new(), fp8_block: None, declared: None, derived: HashMap::new() };
        r.hide_sidecars();
        r
    }

    /// The encodings a packed file declares for itself (`brain.packed` in a safetensors file's metadata, as
    /// written by `convert_to_file`): they say what the dtype alone cannot (NF4 codes in bytes).
    fn declared_in_files(inner: &WeightReader) -> HashMap<String, TensorDescriptor> {
        let mut out = HashMap::new();
        let mut seen: Vec<*const crate::mmap::MmapSafetensors> = Vec::new();
        for name in inner.names() {
            let Some(file) = inner.st_for(name) else { continue };
            if seen.contains(&(file as *const _)) {
                continue;
            }
            seen.push(file as *const _);
            let Some(map) = file.metadata().get(PACKED_METADATA_KEY).and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok()) else { continue };
            for (tensor, v) in map.as_object().into_iter().flatten() {
                let shape: Option<Vec<u64>> = v["shape"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect());
                let encoding = v["encoding"].as_str().and_then(|e| QuantEncoding::from_spec(e).ok());
                if let (Some(shape), Some(encoding)) = (shape, encoding) {
                    out.insert(tensor.clone(), TensorDescriptor::new(tensor.clone(), shape, encoding));
                }
            }
        }
        out
    }

    /// Declare the encoding of a tensor whose dtype alone does not say (brain's
    /// packed-int8 / NF4 `U32` words). Validated against the stored size on use.
    pub fn with_descriptor(mut self, desc: TensorDescriptor) -> PackedReader {
        self.overrides.insert(desc.name.clone(), desc);
        self.hide_sidecars();
        self
    }

    /// Declare the checkpoint's FP8 `weight_block_size`. A scale grid alone can
    /// fit several block sizes (3 bands over 300 rows fits 100 and 128); the
    /// declared one is tried first, and a grid it does not fit is an error.
    pub fn with_fp8_block(mut self, block: u32) -> PackedReader {
        self.fp8_block = Some(block);
        self.hide_sidecars();
        self
    }

    /// What the release declares about its quantization (`config.json` / `hf_quant_config.json`), when it declares any.
    pub fn declaration(&self) -> Option<&HfQuant> {
        self.declared.as_ref()
    }

    /// Adopt the quantization the release declares: its FP8 block size, which tensors are quantized, and the scale
    /// levels it declares as computed from other sidecars (computed here, once).
    pub fn with_declaration(mut self, declared: Option<HfQuant>) -> PackedReader {
        if let Some(b) = declared.as_ref().and_then(HfQuant::fp8_block) {
            self.fp8_block.get_or_insert(b);
        }
        self.declared = declared;
        self.derive_globals();
        self.hide_sidecars();
        self
    }

    /// Fold every NVFP4 `amax / outer_scale_denominator` pair into the f32 `_scale_2` level it declares. A pair whose
    /// shapes do not fit is left underived: [`PackedTensorReader::descriptor`] names the problem.
    fn derive_globals(&mut self) {
        self.derived.clear();
        let Some(q) = self.declared.as_ref().filter(|q| q.explicit()) else { return };
        let mut derived = HashMap::new();
        for n in self.inner.names() {
            if !matches!(q.declared(n), DeclaredWeight::Nvfp4 { global: Nvfp4Global::AmaxOverDenominator, .. }) {
                continue;
            }
            let Some(shape) = self.shape_of(n) else { continue };
            if self.amax_granularity(n, &shape).is_err() {
                continue;
            }
            let (Some((amax, _)), Some((den, _))) = (self.stored(&format!("{n}{AMAX_SUFFIX}")), self.stored(&format!("{n}{DENOMINATOR_SUFFIX}"))) else { continue };
            let den = f32::from_le_bytes(den[..4].try_into().expect("a validated one-element f32 denominator"));
            let global: Vec<u8> = amax.as_chunks::<4>().0.iter().flat_map(|a| (f32::from_le_bytes(*a) / den).to_le_bytes()).collect();
            derived.insert(format!("{n}{GLOBAL_SUFFIX}"), global);
        }
        self.derived = derived;
    }

    /// The granularity of the NVFP4 second level of `name` (stored shape `shape`) declared as `amax / denominator`:
    /// one amax for the tensor, or one per index of its leading axis; one f32 denominator; no stored `_scale_2` beside
    /// them (two sources for one level would be a guess).
    fn amax_granularity(&self, name: &str, shape: &[u64]) -> Result<ScaleGranularity, String> {
        let (amax, den) = (format!("{name}{AMAX_SUFFIX}"), format!("{name}{DENOMINATOR_SUFFIX}"));
        for side in [&amax, &den] {
            match self.dtype_of(side).as_deref() {
                Some("F32") => {}
                Some(other) => return Err(format!("{name}: {side} is stored as {other}, the declaration needs F32")),
                None => return Err(format!("{name}: declared amax_denominator but {side} is missing")),
            }
        }
        if self.dtype_of(&format!("{name}{GLOBAL_SUFFIX}")).is_some() {
            return Err(format!("{name}: stores both {name}{GLOBAL_SUFFIX} and the {AMAX_SUFFIX}/{DENOMINATOR_SUFFIX} pair declared to replace it"));
        }
        let numel = |s: &str| self.shape_of(s).map(|v| v.iter().product::<u64>()).unwrap_or(0);
        if numel(&den) != 1 {
            return Err(format!("{name}: {den} has shape {:?}, expected one value", self.shape_of(&den).unwrap_or_default()));
        }
        let amax_shape = self.shape_of(&amax).unwrap_or_default();
        match amax_shape.iter().product::<u64>() {
            1 => Ok(ScaleGranularity::Tensor),
            _ if shape.len() >= 2 && amax_shape == shape[..1] => Ok(ScaleGranularity::Slab),
            _ => Err(format!("{name}: {amax} has shape {amax_shape:?}, expected one value or one per index of the leading axis of {shape:?}")),
        }
    }

    fn hide_sidecars(&mut self) {
        let mut hidden = HashSet::new();
        let names: Vec<String> = self.inner.names().map(str::to_string).collect();
        for n in &names {
            // The sidecars of a tensor declared by name belong to it, whether or not it reads cleanly: an error is the
            // tensor's, never a stray parameter's.
            match self.declared.as_ref().filter(|q| q.explicit()).map(|q| q.declared(n)) {
                Some(DeclaredWeight::Nvfp4 { .. }) => {
                    hidden.extend(["_scale", GLOBAL_SUFFIX, AMAX_SUFFIX, DENOMINATOR_SUFFIX].iter().map(|s| format!("{n}{s}")));
                    continue;
                }
                Some(DeclaredWeight::Mxfp8 { .. }) => {
                    hidden.insert(format!("{n}_scale"));
                    continue;
                }
                _ => {}
            }
            match self.descriptor_unchecked(n) {
                Ok(d) => {
                    hidden.extend(d.encoding.scales.iter().map(|l| format!("{n}{}", l.suffix)));
                    // The activation scale of a W8A8 / W4A4 release belongs to the weight it was calibrated with;
                    // brain runs float activations, so it is never a tensor of its own.
                    if let Some(module) = n.strip_suffix(".weight").filter(|_| !d.encoding.scales.is_empty()) {
                        hidden.insert(format!("{module}.input_scale"));
                    }
                }
                // An FP8 weight whose scale grid cannot be interpreted still owns
                // it: the error belongs to the weight, not to a stray tensor.
                Err(_) if self.dtype_of(n).as_deref() == Some("F8_E4M3") && self.dtype_of(&format!("{n}_scale_inv")).is_some() => {
                    hidden.insert(format!("{n}_scale_inv"));
                }
                Err(_) => {}
            }
        }
        self.hidden = hidden;
    }

    /// Dtype, shape and location queries all go to the reader that really
    /// holds the tensor, so a renamed view of a checkpoint is read packed.
    fn dtype_of(&self, name: &str) -> Option<String> {
        let (r, n) = self.inner.resolve(name);
        r.dtype(&n).map(str::to_string)
    }

    fn shape_of(&self, name: &str) -> Option<Vec<u64>> {
        let (r, n) = self.inner.resolve(name);
        r.shape(&n).map(<[u64]>::to_vec)
    }

    /// Infer the descriptor from dtype and sibling tensors, without checking sidecar sizes.
    fn descriptor_unchecked(&self, name: &str) -> Result<TensorDescriptor, String> {
        if let Some(d) = self.overrides.get(name) {
            return Ok(d.clone());
        }
        let shape = self.shape_of(name).ok_or_else(|| format!("{name}: not in checkpoint"))?;
        let (holder, held) = self.inner.resolve(name);
        if let Some(g) = holder.gguf() {
            let (_, ty) = g.raw_tensor_bytes(&held).ok_or_else(|| format!("{name}: not in checkpoint"))?;
            return Ok(TensorDescriptor::new(name, shape, ggml_encoding(ty).map_err(|e| format!("{name}: {e}"))?));
        }
        if holder.st_for(&held).is_none() {
            return Err(format!("{name}: torch and computed checkpoints can only be decoded whole; convert to safetensors first"));
        }
        let dtype = self.dtype_of(name).unwrap_or_default();
        let dtype = dtype.as_str();
        if let Some(q) = self.declared.as_ref().filter(|q| q.explicit()) {
            let decl = q.declaration(name).ok_or_else(|| format!("{name}: not declared by the release's hf_quant_config.json parameter_groups"))?;
            return self.declared_descriptor(name, shape, dtype, decl);
        }
        self.check_declaration(name, dtype)?;
        let has = |suffix: &str, want: &str| self.dtype_of(&format!("{name}{suffix}")).as_deref() == Some(want);
        let enc = match dtype {
            "F32" => QuantEncoding::f32(),
            "F16" => QuantEncoding::f16(),
            "BF16" => QuantEncoding::bf16(),
            "F8_E4M3" if has("_scale_inv", "BF16") || has("_scale_inv", "F32") => {
                let grid = self.shape_of(&format!("{name}_scale_inv")).unwrap_or_default();
                let block = infer_fp8_block(&shape, &grid, self.fp8_block).ok_or_else(|| format!("{name}: {name}_scale_inv has shape {grid:?}, which fits no known FP8 block size for shape {shape:?}"))?;
                let mut e = QuantEncoding::fp8_e4m3_tiled(block);
                if has("_scale_inv", "F32") {
                    e.scales[0].format = FloatFormat::F32;
                }
                e
            }
            "F8_E4M3" if self.has_weight_scale(name) => self.fp8_weight_scale_encoding(name, &shape)?,
            "F8_E4M3" => QuantEncoding::fp8(FloatFormat::E4M3Fn),
            "F8_E5M2" => QuantEncoding::fp8(FloatFormat::E5M2),
            "U8" if has("_scale", "F8_E4M3") && has("_scale_2", "F32") => {
                // NVFP4: two E2M1 codes per stored byte, so the logical row is twice as long. A stacked tensor may carry
                // one second-level scale per leading index (per expert).
                let mut logical = shape.clone();
                *logical.last_mut().ok_or_else(|| format!("{name}: scalar NVFP4 tensor"))? *= 2;
                let per_slab = logical.len() >= 2 && self.shape_of(&format!("{name}_scale_2")).is_some_and(|g| g == logical[..1] && g[0] > 1);
                return Ok(TensorDescriptor::new(name, logical, if per_slab { QuantEncoding::nvfp4_per_slab() } else { QuantEncoding::nvfp4() }));
            }
            "U8" => QuantEncoding { element: Element::Int { bits: 8, signed: false }, scales: Vec::new() },
            "U32" if has(INT8_SCALE_SUFFIX, "F32") => {
                // brain's int8-native layout: four codes per stored word, so the logical row is four times as long.
                let mut logical = shape.clone();
                *logical.last_mut().ok_or_else(|| format!("{name}: scalar packed-int8 tensor"))? *= 4;
                return Ok(TensorDescriptor::new(name, logical, QuantEncoding::int8_blocked(INT8_NATIVE_GROUP)));
            }
            "U32" => QuantEncoding { element: Element::Int { bits: 32, signed: false }, scales: Vec::new() },
            other => return Err(format!("{name}: dtype {other} has no packed encoding; declare one with PackedReader::with_descriptor")),
        };
        Ok(TensorDescriptor::new(name, shape, enc))
    }

    /// The release declares `name` quantized: refuse what brain has no codec for, and a dtype that contradicts the
    /// declaration (a stored tensor is never reinterpreted to fit a config).
    fn check_declaration(&self, name: &str, dtype: &str) -> Result<(), String> {
        let Some(q) = &self.declared else { return Ok(()) };
        match q.declared(name) {
            DeclaredWeight::Unsupported(why) if !dtype.starts_with('F') && !dtype.starts_with("BF") => Err(format!("{name}: {why} is not supported")),
            DeclaredWeight::Fp8 { .. } if dtype != "F8_E4M3" && dtype != "F8_E5M2" && self.dtype_of(&format!("{name}_scale")).is_some() => {
                Err(format!("{name}: declared FP8 but stored as {dtype}"))
            }
            DeclaredWeight::Nvfp4 { group, .. } if dtype == "U8" && group != 16 => Err(format!("{name}: NVFP4 group size {group} is not supported (16)")),
            _ => Ok(()),
        }
    }

    /// The descriptor of a tensor an explicit (`parameter_groups`) release declares: the declared encoding, checked
    /// against the stored dtype, the declared block shape and every scale sidecar's dtype and shape.
    fn declared_descriptor(&self, name: &str, shape: Vec<u64>, dtype: &str, decl: &TensorDeclaration) -> Result<TensorDescriptor, String> {
        if let DeclaredWeight::Unsupported(why) = &decl.weight {
            return Err(format!("{name}: {why} is not supported"));
        }
        if dtype != decl.stored_dtype {
            return Err(format!("{name}: declared {:?}, stored as {} but the file has {dtype}", decl.weight, decl.stored_dtype));
        }
        let mut logical = shape.clone();
        let enc = match decl.weight {
            DeclaredWeight::Cast(f) => QuantEncoding { element: Element::Float(f), scales: Vec::new() },
            DeclaredWeight::Nvfp4 { group, global } => {
                *logical.last_mut().ok_or_else(|| format!("{name}: scalar NVFP4 tensor"))? *= 2;
                self.check_block_scale(name, &logical, decl, group, &["F8_E4M3"])?;
                let mut enc = QuantEncoding::nvfp4();
                enc.scales[1].granularity = ScaleGranularity::Block(group);
                enc.scales[0].granularity = match global {
                    Nvfp4Global::AmaxOverDenominator => {
                        let g = self.amax_granularity(name, &shape)?;
                        if !self.derived.contains_key(&format!("{name}{GLOBAL_SUFFIX}")) {
                            return Err(format!("{name}: its amax / denominator second level was not derived when the declaration was adopted"));
                        }
                        g
                    }
                    Nvfp4Global::Stored => match self.shape_of(&format!("{name}{GLOBAL_SUFFIX}")) {
                        Some(g) if g.iter().product::<u64>() == 1 => ScaleGranularity::Tensor,
                        Some(g) if g == logical[..1] => ScaleGranularity::Slab,
                        g => return Err(format!("{name}: {name}{GLOBAL_SUFFIX} has shape {g:?}, expected one value or one per leading index")),
                    },
                };
                enc
            }
            DeclaredWeight::Mxfp8 { block } => {
                self.check_block_scale(name, &logical, decl, block, &["U8", "F8_E8M0"])?;
                let mut enc = QuantEncoding::mxfp8();
                enc.scales[0].granularity = ScaleGranularity::Block(block);
                enc
            }
            DeclaredWeight::Native | DeclaredWeight::Fp8 { .. } | DeclaredWeight::Undeclared | DeclaredWeight::Unsupported(_) => {
                return Err(format!("{name}: {:?} is not a parameter_groups declaration", decl.weight))
            }
        };
        Ok(TensorDescriptor::new(name, logical, enc))
    }

    /// The `<name>_scale` sidecar of a block-scaled declaration: one of `dtypes`, shaped like the logical tensor with
    /// its rows cut into `block`-wide blocks. The declared block shape's leading extents (the producer's tile) must
    /// divide the corresponding axes; the stored grid, one scale per row and block, is what is read.
    fn check_block_scale(&self, name: &str, logical: &[u64], decl: &TensorDeclaration, block: u32, dtypes: &[&str]) -> Result<(), String> {
        let side = format!("{name}_scale");
        let stored = self.dtype_of(&side).ok_or_else(|| format!("{name}: declared block scales but {side} is missing"))?;
        if !dtypes.contains(&stored.as_str()) {
            return Err(format!("{name}: {side} is stored as {stored}, the declaration needs one of {dtypes:?}"));
        }
        let inner = *logical.last().unwrap_or(&0);
        let mut want = logical.to_vec();
        if let Some(l) = want.last_mut() {
            *l = inner / block as u64;
        }
        let got = self.shape_of(&side).unwrap_or_default();
        if !inner.is_multiple_of(block as u64) || got != want {
            return Err(format!("{name}: {side} has shape {got:?}, but {block}-wide blocks of logical shape {logical:?} need {want:?}"));
        }
        let fits = decl.block_shape.iter().rev().zip(logical.iter().rev()).all(|(b, d)| d.is_multiple_of(*b));
        if decl.block_shape.len() > logical.len() || !fits {
            return Err(format!("{name}: declared weight_block_shape {:?} does not tile logical shape {logical:?}", decl.block_shape));
        }
        Ok(())
    }

    /// An FP8 weight with a `<name>_scale` sibling: the ModelOpt / compressed-tensors layout (`weight_scale`).
    fn has_weight_scale(&self, name: &str) -> bool {
        matches!(self.dtype_of(&format!("{name}_scale")).as_deref(), Some("F32" | "BF16" | "F16"))
    }

    /// The encoding of an FP8 weight whose scale sibling is `<name>_scale`: one scale for the tensor (static
    /// per-tensor quantization), one per output row (per-channel), or a 2-D tile grid (block scaled).
    fn fp8_weight_scale_encoding(&self, name: &str, shape: &[u64]) -> Result<QuantEncoding, String> {
        let side = format!("{name}_scale");
        let grid = self.shape_of(&side).unwrap_or_default();
        let numel: u64 = grid.iter().product();
        let format = match self.dtype_of(&side).as_deref() {
            Some("BF16") => FloatFormat::Bf16,
            Some("F16") => FloatFormat::F16,
            _ => FloatFormat::F32,
        };
        let mut enc = QuantEncoding::fp8(FloatFormat::E4M3Fn);
        let granularity = if numel == 1 {
            ScaleGranularity::Tensor
        } else if shape.len() == 2 && (grid == [shape[0]] || grid == [shape[0], 1]) {
            ScaleGranularity::Row
        } else {
            let block = infer_fp8_block(shape, &grid, self.fp8_block).ok_or_else(|| format!("{name}: {side} has shape {grid:?}, which fits no scale layout for shape {shape:?}"))?;
            ScaleGranularity::Tile2d { rows: block, cols: block }
        };
        enc.scales.push(super::encoding::ScaleLevel { granularity, format, suffix: "_scale".into() });
        Ok(enc)
    }

    fn sidecar_name(desc: &TensorDescriptor, level: usize) -> String {
        format!("{}{}", desc.name, desc.encoding.scales[level].suffix)
    }

    /// Whole payload bytes of `name` in storage, whichever file holds it (or as derived from the declaration).
    fn stored(&self, name: &str) -> Option<(&[u8], &str)> {
        if let Some(d) = self.derived.get(name) {
            return Some((d.as_slice(), "F32"));
        }
        let (holder, held) = self.inner.resolve(name);
        if let Some(g) = holder.gguf() {
            return g.raw_tensor_bytes(&held).map(|(b, _)| (b, "GGUF"));
        }
        let m = holder.st_for(&held)?;
        Some((m.raw_range(&held, 0..m.nbytes(&held)? as usize)?, m.dtype(&held)?))
    }
}

fn infer_fp8_block(shape: &[u64], grid: &[u64], declared: Option<u32>) -> Option<u32> {
    let ([rows, cols], [gr, gc]) = (<[u64; 2]>::try_from(shape).ok()?, <[u64; 2]>::try_from(grid).ok()?);
    let fits = |b: &u32| rows.div_ceil(*b as u64) == gr && cols.div_ceil(*b as u64) == gc;
    match declared {
        Some(b) => Some(b).filter(fits),
        None => FP8_BLOCKS.into_iter().find(fits),
    }
}

/// A GGUF tensor type as an encoding. Plain float types become float encodings
/// so they decode without a block decoder; types with no decoder are refused.
fn ggml_encoding(ty: u32) -> Result<QuantEncoding, String> {
    match GgmlType::from_id(ty) {
        Some(GgmlType::F32) => Ok(QuantEncoding::f32()),
        Some(GgmlType::F16) => Ok(QuantEncoding::f16()),
        Some(GgmlType::BF16) => Ok(QuantEncoding::bf16()),
        Some(t) if t.block_elems() > 1 => Ok(QuantEncoding::ggml(t)),
        Some(t) => Err(format!("ggml type {} has no packed encoding", t.name())),
        None => Err(format!("unknown ggml type id {ty}")),
    }
}

impl PackedTensorReader for PackedReader {
    fn names(&self) -> Vec<String> {
        self.inner.names().filter(|n| !self.hidden.contains(*n)).map(str::to_string).collect()
    }

    fn descriptor(&self, name: &str) -> Result<TensorDescriptor, String> {
        let d = self.descriptor_unchecked(name)?;
        d.validate()?;
        let (payload, _) = self.stored(name).ok_or_else(|| format!("{name}: not in checkpoint"))?;
        if payload.len() as u64 != d.payload_bytes() {
            return Err(format!("{name}: stored payload is {} bytes but {} of shape {:?} needs {}", payload.len(), d.encoding, d.shape, d.payload_bytes()));
        }
        for (l, level) in d.encoding.scales.iter().enumerate() {
            let side = Self::sidecar_name(&d, l);
            let (bytes, _) = self.stored(&side).ok_or_else(|| format!("{name}: scale tensor {side} is missing"))?;
            if bytes.len() as u64 != d.scale_bytes(l) {
                return Err(format!("{name}: scale tensor {side} is {} bytes, expected {} for {:?} {:?}", bytes.len(), d.scale_bytes(l), level.granularity, level.format));
            }
        }
        Ok(d)
    }

    fn read_blocks(&self, desc: &TensorDescriptor, elems: &Range<u64>, dst: &mut TileBuf) -> Result<(), String> {
        let name = &desc.name;
        if elems.start >= elems.end || elems.end > desc.numel() {
            return Err(format!("{name}: tile {elems:?} is outside {} elements", desc.numel()));
        }
        let (payload, _) = self.stored(name).ok_or_else(|| format!("{name}: not in checkpoint"))?;
        let pr = desc.payload_range(elems);
        let src = payload.get(pr.start as usize..pr.end as usize).ok_or_else(|| format!("{name}: payload range {pr:?} outside {} stored bytes", payload.len()))?;
        crate::copy_parallel(&mut dst.payload, src);
        dst.scales.resize_with(desc.encoding.scales.len(), Vec::new);
        for (l, level) in desc.encoding.scales.iter().enumerate() {
            let side = Self::sidecar_name(desc, l);
            let (bytes, _) = self.stored(&side).ok_or_else(|| format!("{name}: scale tensor {side} is missing"))?;
            let w = level.format.bits() as u64 / 8;
            let r = desc.scale_range(l, elems);
            let s = bytes.get((r.start * w) as usize..(r.end * w) as usize).ok_or_else(|| format!("{side}: scale range {r:?} outside stored scales"))?;
            dst.scales[l].clear();
            dst.scales[l].reserve_exact(s.len());
            dst.scales[l].extend_from_slice(s);
        }
        Ok(())
    }

    fn release(&self, desc: &TensorDescriptor, elems: &Range<u64>) {
        let pr = desc.payload_range(elems);
        let range = pr.start as usize..pr.end as usize;
        let (holder, held) = self.inner.resolve(&desc.name);
        if let Some(g) = holder.gguf() {
            g.advise_dontneed_range(&held, range);
        } else if let Some(m) = holder.st_for(&held) {
            m.advise_dontneed_range(&held, range);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_write::{write, TensorOut};
    use crate::packed::testfix::{bf16_bytes, bytes, e4m3_bytes, f32_bytes, write_safetensors, RawTensor, TmpDir};

    fn read_all(r: &PackedReader, d: &TensorDescriptor) -> TileBuf {
        let mut buf = TileBuf::default();
        r.read_blocks(d, &(0..d.numel()), &mut buf).unwrap();
        buf
    }

    #[test]
    fn a_scale_grid_that_fits_no_block_size_is_an_error_naming_it() {
        let dir = TmpDir::new("rd");
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "a.weight", dtype: "F8_E4M3", shape: vec![6, 10], bytes: e4m3_bytes(1, 60) },
                RawTensor { name: "a.weight_scale_inv", dtype: "BF16", shape: vec![3, 2], bytes: bf16_bytes(&[1.0; 6]) },
                RawTensor { name: "norm", dtype: "F32", shape: vec![4], bytes: f32_bytes(&[1.0; 4]) },
            ],
        );
        let r = PackedReader::open(&dir.join("m.safetensors")).unwrap();
        assert_eq!(r.descriptor("norm").unwrap().encoding, QuantEncoding::f32());
        assert!(r.descriptor("a.weight").unwrap_err().contains("fits no known FP8 block size"));
    }

    #[test]
    fn fp8_with_scale_grid_reads_tile_by_tile_across_shards() {
        let dir = TmpDir::new("shard");
        let (rows, cols) = (20usize, 40usize);
        let w = e4m3_bytes(5, rows * cols);
        let grid = bf16_bytes(&[1.0, 2.0, 4.0, 8.0, 0.5, 0.25]);
        write_safetensors(&dir.join("s1.safetensors"), &[RawTensor { name: "a.weight", dtype: "F8_E4M3", shape: vec![rows as u64, cols as u64], bytes: w.clone() }]);
        write_safetensors(&dir.join("s2.safetensors"), &[RawTensor { name: "a.weight_scale_inv", dtype: "BF16", shape: vec![2, 3], bytes: grid.clone() }]);
        let index = serde_json::json!({"weight_map": {"a.weight": "s1.safetensors", "a.weight_scale_inv": "s2.safetensors"}});
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
        let r = PackedReader::open(&dir.0).unwrap();
        assert_eq!(r.names(), ["a.weight"]);
        let d = r.descriptor("a.weight").unwrap();
        assert_eq!(d.encoding, QuantEncoding::fp8_e4m3_tiled(16));
        let mut buf = TileBuf::default();
        r.read_blocks(&d, &(10 * 40..20 * 40), &mut buf).unwrap();
        assert_eq!(buf.payload, &w[400..800]);
        assert_eq!(buf.scales[0], grid, "rows 10..20 straddle both scale bands");
        r.read_blocks(&d, &(16 * 40..20 * 40), &mut buf).unwrap();
        assert_eq!(buf.scales[0], &grid[6..12], "rows 16..20 sit in the second band only");
    }

    #[test]
    fn declared_packed_layouts_and_nvfp4_siblings_are_recognised() {
        let dir = TmpDir::new("decl");
        let words = bytes(7, 4 * 8 * 4); // [4, 32] int8 as [4, 8] u32 words
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "q.weight", dtype: "U32", shape: vec![4, 8], bytes: words.clone() },
                RawTensor { name: "q.weight_scale", dtype: "F32", shape: vec![4, 1], bytes: f32_bytes(&[1.0; 4]) },
                RawTensor { name: "f.weight", dtype: "U8", shape: vec![2, 16], bytes: bytes(8, 32) },
                RawTensor { name: "f.weight_scale", dtype: "F8_E4M3", shape: vec![2, 2], bytes: e4m3_bytes(9, 4) },
                RawTensor { name: "f.weight_scale_2", dtype: "F32", shape: vec![], bytes: f32_bytes(&[2.0]) },
            ],
        );
        let declared = TensorDescriptor::new("q.weight", vec![4, 32], QuantEncoding::int8_blocked(32));
        let mut declared_enc = declared.clone();
        declared_enc.encoding.scales[0].suffix = "_scale".into();
        let r = PackedReader::open(&dir.join("m.safetensors")).unwrap().with_descriptor(declared_enc);
        assert_eq!(r.names(), ["f.weight", "q.weight"]);
        let q = r.descriptor("q.weight").unwrap();
        assert_eq!(read_all(&r, &q).payload, words);
        let f = r.descriptor("f.weight").unwrap();
        assert_eq!((f.shape.clone(), f.encoding.clone()), (vec![2, 32], QuantEncoding::nvfp4()));
        let wrong = PackedReader::open(&dir.join("m.safetensors")).unwrap().with_descriptor(TensorDescriptor::new("q.weight", vec![4, 64], QuantEncoding::int8_blocked(32)));
        assert!(wrong.descriptor("q.weight").unwrap_err().contains("stored payload"), "a declared layout that disagrees with the stored size is refused");
    }

    /// Spec: brain's int8-native layout (`u32` words plus an f32 `<name>.scale` grid) is recognised from the file alone,
    /// as the int8 encoding whose payload and scales are the stored bytes; its sidecar is not a tensor of its own.
    #[test]
    fn an_int8_native_tensor_is_recognised_without_a_declaration() {
        let dir = TmpDir::new("int8native");
        let words = bytes(5, 4 * 8 * 4); // [4, 32] int8 codes as [4, 8] u32 words
        let scales = f32_bytes(&[0.5, 1.0, 1.5, 2.0]);
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "q.weight", dtype: "U32", shape: vec![4, 8], bytes: words.clone() },
                RawTensor { name: "q.weight.scale", dtype: "F32", shape: vec![4, 1], bytes: scales.clone() },
            ],
        );
        let r = PackedReader::open(&dir.join("m.safetensors")).unwrap();
        assert_eq!(r.names(), ["q.weight"]);
        let d = r.descriptor("q.weight").unwrap();
        assert_eq!((d.shape.clone(), d.encoding.clone()), (vec![4, 32], QuantEncoding::int8_blocked(32)));
        let got = read_all(&r, &d);
        assert_eq!((got.payload, got.scales), (words, vec![scales]));
    }

    #[test]
    fn gguf_block_types_are_served_as_raw_block_ranges() {
        let dir = TmpDir::new("gguf");
        let data: Vec<f32> = (0..4 * 64).map(|i| (i as f32 * 0.37).sin()).collect();
        let q8 = crate::quant::quantize(crate::gguf::TYPE_Q8_0, &data).unwrap();
        let t = |name: &str, shape: Vec<usize>, ty: u32, data: Vec<u8>| TensorOut { name: name.into(), shape, ty, data };
        write(
            dir.join("m.gguf").to_str().unwrap(),
            &[],
            &[t("w", vec![4, 64], crate::gguf::TYPE_Q8_0, q8.clone()), t("n", vec![8], 0, f32_bytes(&[1.0; 8]))],
            32,
        )
        .unwrap();
        let r = PackedReader::open(&dir.join("m.gguf")).unwrap();
        let d = r.descriptor("w").unwrap();
        assert_eq!((d.shape.clone(), d.encoding.clone()), (vec![4, 64], QuantEncoding::ggml(GgmlType::Q8_0)));
        assert_eq!(r.descriptor("n").unwrap().encoding, QuantEncoding::f32());
        let mut buf = TileBuf::default();
        r.read_blocks(&d, &(64..192), &mut buf).unwrap();
        assert_eq!(buf.payload, &q8[2 * 34..6 * 34]);
    }

    #[test]
    fn a_renamed_view_is_read_packed_from_its_source() {
        use crate::weightio::{Renamed, WeightReader};
        let dir = TmpDir::new("renamed");
        let w = e4m3_bytes(31, 20 * 40);
        let grid = bf16_bytes(&[1.0, 2.0, 4.0, 8.0, 0.5, 0.25]);
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "hf.w", dtype: "F8_E4M3", shape: vec![20, 40], bytes: w.clone() },
                RawTensor { name: "hf.w_scale_inv", dtype: "BF16", shape: vec![2, 3], bytes: grid.clone() },
            ],
        );
        let src = WeightReader::open(dir.join("m.safetensors").to_str().unwrap()).unwrap();
        let map = vec![("blocks.0.w".to_string(), "hf.w".to_string()), ("blocks.0.w_scale_inv".to_string(), "hf.w_scale_inv".to_string())];
        let view = WeightReader::derived(Box::new(Renamed::new(src, map, serde_json::Value::Null).unwrap()));
        let r = PackedReader::new(view);
        assert_eq!(r.names(), ["blocks.0.w"]);
        let d = r.descriptor("blocks.0.w").unwrap();
        assert_eq!(d.encoding, QuantEncoding::fp8_e4m3_tiled(16), "the renamed tensor keeps its stored encoding, not f32");
        let buf = read_all(&r, &d);
        assert_eq!((buf.payload, buf.scales[0].clone()), (w, grid));
    }

    /// Spec: a ModelOpt mixed-precision release (FP8 with a per-tensor `weight_scale`, NVFP4 with block and global
    /// scales, bf16 elsewhere) reads as each tensor's stored encoding; its `input_scale` sidecars are not tensors.
    #[test]
    fn a_modelopt_mixed_release_reads_each_tensor_in_its_stored_encoding() {
        let dir = TmpDir::new("modelopt");
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                RawTensor { name: "a.weight", dtype: "F8_E4M3", shape: vec![4, 32], bytes: e4m3_bytes(1, 128) },
                RawTensor { name: "a.weight_scale", dtype: "F32", shape: vec![], bytes: f32_bytes(&[0.25]) },
                RawTensor { name: "a.input_scale", dtype: "F32", shape: vec![], bytes: f32_bytes(&[0.5]) },
                RawTensor { name: "e.weight", dtype: "U8", shape: vec![2, 16], bytes: bytes(2, 32) },
                RawTensor { name: "e.weight_scale", dtype: "F8_E4M3", shape: vec![2, 2], bytes: e4m3_bytes(3, 4) },
                RawTensor { name: "e.weight_scale_2", dtype: "F32", shape: vec![], bytes: f32_bytes(&[0.01]) },
                RawTensor { name: "e.input_scale", dtype: "F32", shape: vec![], bytes: f32_bytes(&[1.0]) },
                RawTensor { name: "n.weight", dtype: "BF16", shape: vec![4], bytes: bf16_bytes(&[1.0; 4]) },
            ],
        );
        let hf = serde_json::json!({"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": {"a": {"quant_algo": "FP8"}, "e": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}}});
        std::fs::write(dir.join("hf_quant_config.json"), hf.to_string()).unwrap();
        let r = PackedReader::open(&dir.0).unwrap();
        assert_eq!(r.names(), ["a.weight", "e.weight", "n.weight"]);
        let a = r.descriptor("a.weight").unwrap();
        assert_eq!(a.encoding.scales.len(), 1);
        assert_eq!((a.encoding.scales[0].granularity, a.encoding.scales[0].suffix.as_str()), (ScaleGranularity::Tensor, "_scale"));
        let got = read_all(&r, &a);
        assert_eq!(got.scales[0], f32_bytes(&[0.25]));
        let e = r.descriptor("e.weight").unwrap();
        assert_eq!((e.shape.clone(), e.encoding.clone()), (vec![2, 32], QuantEncoding::nvfp4()));
        assert_eq!(r.descriptor("n.weight").unwrap().encoding, QuantEncoding::bf16());
    }

    /// Spec: per-channel and block-grid `weight_scale` layouts are told apart by the scale tensor's shape, and a
    /// release declaring a method brain has no codec for is refused by name on the tensors that need it.
    #[test]
    fn fp8_weight_scale_layouts_and_refused_methods() {
        let dir = TmpDir::new("fp8layouts");
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                RawTensor { name: "c.weight", dtype: "F8_E4M3", shape: vec![4, 32], bytes: e4m3_bytes(1, 128) },
                RawTensor { name: "c.weight_scale", dtype: "F32", shape: vec![4, 1], bytes: f32_bytes(&[1.0; 4]) },
                RawTensor { name: "b.weight", dtype: "F8_E4M3", shape: vec![32, 32], bytes: e4m3_bytes(2, 1024) },
                RawTensor { name: "b.weight_scale", dtype: "BF16", shape: vec![2, 2], bytes: bf16_bytes(&[1.0; 4]) },
                RawTensor { name: "q.qweight", dtype: "I32", shape: vec![4, 4], bytes: bytes(3, 64) },
            ],
        );
        std::fs::write(dir.join("config.json"), serde_json::json!({"quantization_config": {"quant_method": "gptq", "bits": 4}}).to_string()).unwrap();
        let r = PackedReader::open(&dir.0).unwrap().with_fp8_block(16);
        assert_eq!(r.descriptor("c.weight").unwrap().encoding.scales[0].granularity, ScaleGranularity::Row);
        assert_eq!(r.descriptor("b.weight").unwrap().encoding.scales[0].granularity, ScaleGranularity::Tile2d { rows: 16, cols: 16 });
        assert!(r.descriptor("q.qweight").unwrap_err().contains("gptq"));
    }

    fn parameter_groups(dir: &TmpDir, mx_scale: Vec<u64>, n_dtype: &str) {
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                RawTensor { name: "e.w13_weight", dtype: "U8", shape: vec![2, 4, 16], bytes: bytes(1, 128) },
                RawTensor { name: "e.w13_weight_scale", dtype: "F8_E4M3", shape: vec![2, 4, 2], bytes: e4m3_bytes(2, 16) },
                RawTensor { name: "e.w13_weight_amax", dtype: "F32", shape: vec![2], bytes: f32_bytes(&[3.0, 0.7]) },
                RawTensor { name: "e.w13_weight_outer_scale_denominator", dtype: "F32", shape: vec![1], bytes: f32_bytes(&[2688.0]) },
                RawTensor { name: "m.w13_weight", dtype: "F8_E4M3", shape: vec![2, 4, 32], bytes: e4m3_bytes(3, 256) },
                RawTensor { name: "m.w13_weight_scale", dtype: "U8", shape: mx_scale.clone(), bytes: vec![126; mx_scale.iter().product::<u64>() as usize] },
                RawTensor { name: "n", dtype: n_dtype, shape: vec![4], bytes: bf16_bytes(&[1.0; 4]) },
                RawTensor { name: "stray", dtype: "BF16", shape: vec![2], bytes: bf16_bytes(&[1.0; 2]) },
            ],
        );
        let g = |t: &str, c: &str, o: &str, block: serde_json::Value, extra: serde_json::Value, tensors: &[&str]| {
            let mut v = serde_json::json!({"compute_dtype": c, "output_dtype": o, "required_activation_quantization": null, "weight_block_shape": block, "weight_transform": t, "tensors": tensors});
            if let Some(f) = extra.as_str() {
                v["weight_scale_format"] = f.into();
            }
            v
        };
        let hf = serde_json::json!({"quant_method": "vendor", "model": {"parameter_groups": [
            g("cast", "bfloat16", "bfloat16", serde_json::json!([]), serde_json::Value::Null, &["n"]),
            g("nvfp4_nearest", "nvfp4", "uint8", serde_json::json!([1, 16]), "amax_denominator".into(), &["e.w13_weight"]),
            g("mxfp8_nearest", "mxfp8", "float8_e4m3fn", serde_json::json!([4, 32]), serde_json::Value::Null, &["m.w13_weight"])]}});
        std::fs::write(dir.join("hf_quant_config.json"), hf.to_string()).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}).to_string()).unwrap();
    }

    /// Spec: a `parameter_groups` release reads each tensor as declared: NVFP4 experts with their per-expert second
    /// level folded from `amax / denominator`, MXFP8 with a `U8`-stored E8M0 scale, casts in their dtype; sidecars are
    /// not listed, and a stored tensor the release does not declare is refused by name.
    #[test]
    fn a_parameter_groups_release_reads_each_tensor_as_declared() {
        use crate::packed::codec::{CodecRegistry, Tile};
        let dir = TmpDir::new("pgroups");
        parameter_groups(&dir, vec![2, 4, 1], "BF16");
        let r = PackedReader::open(&dir.0).unwrap();
        let mut names = r.names();
        names.sort();
        assert_eq!(names, ["e.w13_weight", "m.w13_weight", "n", "stray"]);
        assert!(r.descriptor("stray").unwrap_err().contains("not declared"));
        assert_eq!(r.descriptor("n").unwrap().encoding, QuantEncoding::bf16());
        let e = r.descriptor("e.w13_weight").unwrap();
        assert_eq!((e.shape.clone(), e.encoding.clone()), (vec![2, 4, 32], QuantEncoding::nvfp4_per_slab()));
        let mut buf = TileBuf::default();
        r.read_blocks(&e, &(128..256), &mut buf).unwrap();
        assert_eq!(buf.scales[0], f32_bytes(&[0.7f32 / 2688.0]), "expert 1's second level is its amax over the denominator");
        let m = r.descriptor("m.w13_weight").unwrap();
        assert_eq!(m.encoding, QuantEncoding::mxfp8());
        let all = read_all(&r, &m);
        let tile = Tile { elems: 0..m.numel(), payload: &all.payload, scales: vec![&all.scales[0]] };
        let mut out = Vec::new();
        CodecRegistry::reference().decoder(&m.encoding).unwrap().decode_tile(&m, &tile, &mut out).unwrap();
        let want: Vec<f32> = all.payload.iter().map(|&c| crate::safetensors::e4m3fn_to_f32(c) * 0.5).collect();
        assert_eq!(out, want, "E8M0 code 126 is a scale of 1/2");
    }

    /// Spec: a declaration contradicting what is stored is an error naming both, never a reinterpretation.
    #[test]
    fn a_stored_dtype_or_scale_shape_contradicting_the_declaration_is_an_error() {
        let dir = TmpDir::new("pgroups-scale");
        parameter_groups(&dir, vec![2, 4, 2], "BF16");
        let err = PackedReader::open(&dir.0).unwrap().descriptor("m.w13_weight").unwrap_err();
        assert!(err.contains("m.w13_weight_scale has shape [2, 4, 2]"), "{err}");
        let dir = TmpDir::new("pgroups-dtype");
        parameter_groups(&dir, vec![2, 4, 1], "F16");
        let err = PackedReader::open(&dir.0).unwrap().descriptor("n").unwrap_err();
        assert!(err.contains("BF16") && err.contains("F16"), "{err}");
    }
}
