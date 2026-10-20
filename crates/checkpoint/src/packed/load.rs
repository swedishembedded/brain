// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The tiled load executor. A load has exactly three kinds of step, and the
//! caller picks one explicitly per tensor ([`LoadPolicy`]):
//!
//! - **Preserve**: copy the packed bytes and scales unchanged.
//! - **Repack**: change the physical layout losslessly (e.g. GGUF Q8_0's
//!   interleaved f16 scales to int8 codes plus an f32 scale tensor). Only pairs
//!   with an exact conversion exist; anything else is an error.
//! - **Convert**: change the numeric representation (decode tile, encode tile),
//!   lossy in general, only on request, and recorded in the [`LoadPlan`].
//!
//! No step expands a quantized tensor to full precision on its own: the only
//! f32 data is one tile's decode scratch. Every tile's scratch is leased from the
//! shared [`BoundedWorkspace`] before it is allocated, so the staging footprint
//! is the configured budget, independent of the tensor's size.

use std::fmt;
use std::ops::Range;

use super::budget::{resident_file_bytes, BoundedWorkspace, Pool};
use super::codec::{CodecRegistry, Tile, TileBuf};
use super::encoding::{Element, QuantEncoding, TensorDescriptor};
use super::reader::PackedTensorReader;
use super::sink::TensorSink;
use crate::gguf::GgmlType;

/// What the caller wants done to a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadPolicy {
    /// Keep the stored encoding.
    Preserve,
    /// Convert to this encoding losslessly (layout change only).
    Repack(QuantEncoding),
    /// Re-quantize to this encoding. Explicit, bounded, recorded.
    Convert(QuantEncoding),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadOp {
    Preserve,
    Repack,
    Convert,
}

/// The plan for one tensor.
#[derive(Clone, Debug)]
pub struct LoadStep {
    pub src: TensorDescriptor,
    pub dst: TensorDescriptor,
    pub op: LoadOp,
    /// Elements per tile (a multiple of the combined tile quantum).
    pub tile_elems: u64,
    pub tiles: u64,
    /// Staging bytes leased per full tile: the plan's upper bound on this
    /// load's host staging.
    pub staging_bytes_per_tile: u64,
    /// Full-size copies of the tensor this step creates. Always 0: tiles are the
    /// only intermediate; the field exists so a report can state the invariant.
    pub intermediate_full_tensors: u32,
    /// The step decodes to f32 per tile (Convert, or Repack through f32).
    pub decodes_to_f32: bool,
}

/// What a load will do, before it does it.
#[derive(Clone, Debug, Default)]
pub struct LoadPlan {
    pub steps: Vec<LoadStep>,
}

impl fmt::Display for LoadPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in &self.steps {
            writeln!(
                f,
                "{:?} {} [{}] -> [{}] tiles={}x{} elems, staging<={} B/tile, intermediate full tensors: {}",
                s.op,
                s.src.name,
                s.src.encoding,
                s.dst.encoding,
                s.tiles,
                s.tile_elems,
                s.staging_bytes_per_tile,
                s.intermediate_full_tensors
            )?;
        }
        Ok(())
    }
}

/// What a load did.
#[derive(Clone, Debug)]
pub struct LoadReport {
    pub step: LoadStep,
    pub tiles_written: u64,
    pub bytes_read: u64,
    /// Most staging bytes this load held at once (see [`BoundedWorkspace::peak`]).
    pub peak_staging_bytes: u64,
    /// Highest process-wide file-backed resident bytes sampled after a tile was
    /// released, minus the level when the load began. `None` off Linux.
    pub file_resident_growth_bytes: Option<u64>,
}

/// Executor settings.
#[derive(Clone, Debug)]
pub struct LoadConfig {
    /// Target staging bytes for one tile. The executor shrinks tiles to fit and
    /// refuses a tensor whose smallest legal tile does not.
    pub tile_bytes: u64,
}

impl Default for LoadConfig {
    fn default() -> Self {
        LoadConfig { tile_bytes: 4 << 20 }
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    a / gcd(a, b) * b
}

/// Plans and runs loads against one reader, codec set and workspace.
pub struct Loader<'a> {
    pub reader: &'a dyn PackedTensorReader,
    pub codecs: &'a CodecRegistry,
    pub workspace: BoundedWorkspace,
    pub config: LoadConfig,
}

/// Is `(src, dst)` a lossless layout change the executor implements?
fn repack_supported(src: &QuantEncoding, dst: &QuantEncoding) -> bool {
    matches!(src.element, Element::Ggml(GgmlType::Q8_0)) && *dst == QuantEncoding::int8_blocked(32)
}

/// Lossless Q8_0 -> int8 codes + f32 block scales, one tile.
fn repack_q8_0(tile: &Tile<'_>, out: &mut TileBuf) {
    let (be, bb) = (GgmlType::Q8_0.block_elems(), GgmlType::Q8_0.block_bytes());
    out.payload.clear();
    out.scales.resize_with(1, Vec::new);
    out.scales[0].clear();
    for blk in tile.payload.chunks_exact(bb) {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        out.scales[0].extend_from_slice(&d.to_le_bytes());
        out.payload.extend_from_slice(&blk[2..2 + be]);
    }
}

impl Loader<'_> {
    /// Plan loading `name` under `policy`.
    pub fn plan(&self, name: &str, policy: &LoadPolicy) -> Result<LoadStep, String> {
        let src = self.reader.descriptor(name)?;
        let (op, dst) = match policy {
            LoadPolicy::Preserve => (LoadOp::Preserve, src.clone()),
            LoadPolicy::Repack(e) if *e == src.encoding => (LoadOp::Preserve, src.clone()),
            LoadPolicy::Repack(e) => {
                if !repack_supported(&src.encoding, e) {
                    return Err(format!("{name}: no lossless repack from {} to {e}; request Convert explicitly if a numeric change is intended", src.encoding));
                }
                (LoadOp::Repack, TensorDescriptor::new(name, src.shape.clone(), e.clone()))
            }
            LoadPolicy::Convert(e) if *e == src.encoding => (LoadOp::Preserve, src.clone()),
            LoadPolicy::Convert(e) => (LoadOp::Convert, TensorDescriptor::new(name, src.shape.clone(), e.clone())),
        };
        dst.validate()?;
        let decodes = op == LoadOp::Convert;
        if decodes {
            self.codecs.decoder(&src.encoding).map_err(|e| format!("{name}: {e}"))?;
            self.codecs.encoder(&dst.encoding).map_err(|e| format!("{name}: {e}"))?;
        }
        let mut quantum = src.tile_quantum();
        if op != LoadOp::Preserve {
            quantum = lcm(quantum, dst.tile_quantum());
        }
        let cost = |elems: u64| staging_bytes(&src, &dst, op, elems);
        if cost(quantum) > self.tile_cap() {
            return Err(format!(
                "{name}: the smallest legal tile ({quantum} elements, {} staging bytes) exceeds the {} byte tile budget; raise the staging budget",
                cost(quantum),
                self.tile_cap()
            ));
        }
        let numel = src.numel();
        let mut tile_elems = quantum;
        while tile_elems.saturating_add(quantum) <= numel && cost(tile_elems + quantum) <= self.tile_cap() {
            tile_elems += quantum;
        }
        let tile_elems = tile_elems.min(numel);
        Ok(LoadStep {
            tiles: numel.div_ceil(tile_elems),
            staging_bytes_per_tile: cost(tile_elems),
            tile_elems,
            intermediate_full_tensors: 0,
            decodes_to_f32: decodes,
            src,
            dst,
            op,
        })
    }

    /// Plan several tensors.
    pub fn plan_all(&self, requests: &[(String, LoadPolicy)]) -> Result<LoadPlan, String> {
        Ok(LoadPlan { steps: requests.iter().map(|(n, p)| self.plan(n, p)).collect::<Result<_, _>>()? })
    }

    fn tile_cap(&self) -> u64 {
        self.config.tile_bytes.min(self.workspace.budget().stats(Pool::HostStaging).capacity)
    }

    /// Run one planned step into `sink`. On any failure the sink is told to
    /// abort and every staging lease is released before returning.
    pub fn execute(&self, step: &LoadStep, sink: &mut dyn TensorSink) -> Result<LoadReport, String> {
        // One lease for the whole load: the tile buffers below are allocated once
        // and reused, so they are held (and accounted) from here until the load
        // ends. Concurrent loads therefore partition the shared budget.
        let _lease = self.workspace.lease(Pool::HostStaging, step.staging_bytes_per_tile).map_err(|e| format!("{}: {e}", step.src.name))?;
        let base_resident = resident_file_bytes();
        let mut resident_peak = base_resident;
        sink.begin(&step.dst)?;
        let mut report = LoadReport { step: step.clone(), tiles_written: 0, bytes_read: 0, peak_staging_bytes: 0, file_resident_growth_bytes: None };
        let result = self.run_tiles(step, sink, &mut report, &mut resident_peak);
        report.peak_staging_bytes = self.workspace.peak();
        match result {
            Ok(()) => {
                if let Err(e) = sink.finish(&step.dst) {
                    sink.abort(&step.dst);
                    return Err(format!("{}: finishing the destination failed: {e}", step.src.name));
                }
                report.file_resident_growth_bytes = base_resident.zip(resident_peak).map(|(b, p)| p.saturating_sub(b));
                Ok(report)
            }
            Err(e) => {
                sink.abort(&step.dst);
                Err(format!("{}: load aborted after {} tiles: {e}", step.src.name, report.tiles_written))
            }
        }
    }

    fn run_tiles(&self, step: &LoadStep, sink: &mut dyn TensorSink, report: &mut LoadReport, resident_peak: &mut Option<u64>) -> Result<(), String> {
        let numel = step.src.numel();
        let mut read = TileBuf::default();
        let mut encoded = TileBuf::default();
        let mut decoded: Vec<f32> = Vec::new();
        let mut start = 0;
        while start < numel {
            let elems: Range<u64> = start..(start + step.tile_elems).min(numel);
            self.reader.read_blocks(&step.src, &elems, &mut read)?;
            report.bytes_read += read.payload.len() as u64 + read.scales.iter().map(|s| s.len() as u64).sum::<u64>();
            let src_tile = Tile { elems: elems.clone(), payload: &read.payload, scales: read.scales.iter().map(Vec::as_slice).collect() };
            match step.op {
                LoadOp::Preserve => sink.write_tile(&step.dst, &src_tile)?,
                LoadOp::Repack => {
                    repack_q8_0(&src_tile, &mut encoded);
                    sink.write_tile(&step.dst, &Tile { elems: elems.clone(), payload: &encoded.payload, scales: encoded.scales.iter().map(Vec::as_slice).collect() })?;
                }
                LoadOp::Convert => {
                    decoded.clear();
                    decoded.reserve_exact((elems.end - elems.start) as usize);
                    self.codecs.decoder(&step.src.encoding)?.decode_tile(&step.src, &src_tile, &mut decoded)?;
                    self.codecs.encoder(&step.dst.encoding)?.encode_tile(&step.dst, &elems, &decoded, &mut encoded)?;
                    sink.write_tile(&step.dst, &Tile { elems: elems.clone(), payload: &encoded.payload, scales: encoded.scales.iter().map(Vec::as_slice).collect() })?;
                }
            }
            self.reader.release(&step.src, &elems);
            report.tiles_written += 1;
            if let Some(now) = resident_file_bytes() {
                *resident_peak = Some(resident_peak.map_or(now, |p| p.max(now)));
            }
            start = elems.end;
        }
        Ok(())
    }

    /// Plan and run one tensor.
    pub fn load(&self, name: &str, policy: &LoadPolicy, sink: &mut dyn TensorSink) -> Result<LoadReport, String> {
        let step = self.plan(name, policy)?;
        self.execute(&step, sink)
    }
}

/// Staging bytes one tile of `elems` elements needs: the bytes read, plus for
/// decoding steps the f32 scratch and the encoded tile (twice, covering the
/// word buffer the packers build before writing bytes).
fn staging_bytes(src: &TensorDescriptor, dst: &TensorDescriptor, op: LoadOp, elems: u64) -> u64 {
    let tile_bytes = |d: &TensorDescriptor| -> u64 {
        let e = 0..elems;
        let payload = d.payload_range(&e);
        let mut b = payload.end - payload.start;
        for (l, s) in d.encoding.scales.iter().enumerate() {
            let r = d.scale_range(l, &e);
            b += (r.end - r.start) * s.format.bits() as u64 / 8;
        }
        b
    };
    match op {
        LoadOp::Preserve => tile_bytes(src),
        LoadOp::Repack => tile_bytes(src) + tile_bytes(dst),
        LoadOp::Convert => tile_bytes(src) + elems * 4 + 2 * tile_bytes(dst),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::budget::MemoryBudget;
    use crate::packed::codec::rows::codebook4_quantize_row;
    use crate::packed::reader::PackedReader;
    use crate::packed::sink::MemorySink;
    use crate::packed::testfix::{bf16_bytes, e4m3_bytes, write_safetensors, RawTensor, TmpDir};
    use crate::safetensors::e4m3fn_to_f32;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// Discards tiles after checking they arrive in order and cover the tensor once.
    #[derive(Default)]
    struct CheckSink {
        next: u64,
        tiles: u64,
        fail_at_tile: Option<u64>,
        aborted: bool,
        finished: bool,
    }

    impl TensorSink for CheckSink {
        fn begin(&mut self, _: &TensorDescriptor) -> Result<(), String> {
            Ok(())
        }
        fn write_tile(&mut self, _: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
            assert_eq!(tile.elems.start, self.next, "tiles must arrive in order without gaps");
            self.next = tile.elems.end;
            self.tiles += 1;
            if self.fail_at_tile == Some(self.tiles) {
                return Err("sink full".into());
            }
            Ok(())
        }
        fn finish(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
            assert_eq!(self.next, dst.numel());
            self.finished = true;
            Ok(())
        }
        fn abort(&mut self, _: &TensorDescriptor) {
            self.aborted = true;
        }
    }

    fn loader<'a>(r: &'a dyn PackedTensorReader, codecs: &'a CodecRegistry, budget: &MemoryBudget, tile_bytes: u64) -> Loader<'a> {
        Loader { reader: r, codecs, workspace: BoundedWorkspace::new(budget.clone()), config: LoadConfig { tile_bytes } }
    }

    /// A bf16 `[rows, cols]` tensor of smooth pseudo values.
    fn bf16_file(dir: &TmpDir, rows: usize, cols: usize) -> PackedReader {
        let vals: Vec<f32> = (0..rows * cols).map(|i| ((i * 31 % 257) as f32 - 128.0) / 64.0).collect();
        write_safetensors(&dir.join("m.safetensors"), &[RawTensor { name: "w", dtype: "BF16", shape: vec![rows as u64, cols as u64], bytes: bf16_bytes(&vals) }]);
        PackedReader::open(&dir.join("m.safetensors")).unwrap()
    }

    #[test]
    fn a_tensor_far_larger_than_the_budget_converts_within_it() {
        let dir = TmpDir::new("big");
        let r = bf16_file(&dir, 256, 32768); // 16 MiB of bf16
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(512 << 10, 0));
        let l = loader(&r, &codecs, &budget, 128 << 10);
        let mut sink = CheckSink::default();
        let (report, heap_peak) = crate::testalloc::peak_live(|| l.load("w", &LoadPolicy::Convert(QuantEncoding::nf4_blocked(32)), &mut sink).unwrap());
        assert!(sink.finished && report.tiles_written > 100, "{} tiles", report.tiles_written);
        assert!(report.step.intermediate_full_tensors == 0 && report.step.decodes_to_f32);
        let cap = budget.stats(Pool::HostStaging).capacity;
        assert!(report.peak_staging_bytes <= cap, "leased {} > budget {cap}", report.peak_staging_bytes);
        assert!(heap_peak as u64 <= cap + (64 << 10), "heap peak {heap_peak} B for a 16 MiB tensor under a {cap} B budget");
        assert_eq!(budget.stats(Pool::HostStaging).in_use, 0);
        eprintln!("MEASURED big-convert: tensor 16 MiB bf16 -> nf4, budget {cap} B, leased peak {} B, heap peak {heap_peak} B, {} tiles", report.peak_staging_bytes, report.tiles_written);
    }

    /// Wraps a reader and records the largest range it was asked for.
    struct Spy<'a> {
        inner: &'a PackedReader,
        largest: AtomicU64,
    }

    impl PackedTensorReader for Spy<'_> {
        fn names(&self) -> Vec<String> {
            self.inner.names()
        }
        fn descriptor(&self, name: &str) -> Result<TensorDescriptor, String> {
            self.inner.descriptor(name)
        }
        fn read_blocks(&self, desc: &TensorDescriptor, elems: &Range<u64>, dst: &mut TileBuf) -> Result<(), String> {
            self.largest.fetch_max(elems.end - elems.start, Ordering::Relaxed);
            self.inner.read_blocks(desc, elems, dst)
        }
        fn release(&self, desc: &TensorDescriptor, elems: &Range<u64>) {
            self.inner.release(desc, elems)
        }
    }

    #[test]
    fn no_read_path_asks_for_a_whole_tensor() {
        let dir = TmpDir::new("spy");
        let inner = bf16_file(&dir, 64, 4096);
        let spy = Spy { inner: &inner, largest: AtomicU64::new(0) };
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(1 << 20, 0));
        let l = loader(&spy, &codecs, &budget, 64 << 10);
        for policy in [LoadPolicy::Preserve, LoadPolicy::Convert(QuantEncoding::int8_blocked(32))] {
            spy.largest.store(0, Ordering::Relaxed);
            l.load("w", &policy, &mut CheckSink::default()).unwrap();
            assert!(spy.largest.load(Ordering::Relaxed) <= 64 * 4096 / 8, "{policy:?} read a range of {} elements", spy.largest.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn fp8_to_nf4_tiled_convert_equals_the_whole_tensor_reference() {
        let dir = TmpDir::new("ref");
        let (rows, cols) = (32usize, 64usize);
        let w = e4m3_bytes(11, rows * cols);
        let grid = [0.5f32, 1.0, 2.0, 0.25, 4.0, 1.5, 0.75, 3.0];
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "w", dtype: "F8_E4M3", shape: vec![rows as u64, cols as u64], bytes: w.clone() },
                RawTensor { name: "w_scale_inv", dtype: "BF16", shape: vec![2, 4], bytes: bf16_bytes(&grid) },
            ],
        );
        // Independent reference: decode by hand, quantize row by row.
        let dequant: Vec<f32> = (0..rows * cols).map(|i| e4m3fn_to_f32(w[i]) * half::bf16::from_f32(grid[(i / cols) / 16 * 4 + (i % cols) / 16]).to_f32()).collect();
        let (mut ref_words, mut ref_scales) = (vec![0u32; rows * cols / 8], vec![0f32; rows * cols / 32]);
        for r in 0..rows {
            codebook4_quantize_row(&dequant[r * cols..(r + 1) * cols], 32, &crate::packed::codec::NF4_CODEBOOK, &mut ref_words[r * cols / 8..(r + 1) * cols / 8], &mut ref_scales[r * cols / 32..(r + 1) * cols / 32]);
        }
        let want_payload: Vec<u8> = ref_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let want_scales: Vec<u8> = ref_scales.iter().flat_map(|w| w.to_le_bytes()).collect();

        let r = PackedReader::open(&dir.join("m.safetensors")).unwrap();
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(1 << 20, 0));
        let l = loader(&r, &codecs, &budget, 1024); // two rows per tile
        let mut sink = MemorySink::default();
        let rep = l.load("w", &LoadPolicy::Convert(QuantEncoding::nf4_blocked(32)), &mut sink).unwrap();
        assert!(rep.tiles_written > 8, "{} tiles", rep.tiles_written);
        assert_eq!((sink.payload, sink.scales[0].clone()), (want_payload, want_scales));
    }

    #[test]
    fn q8_0_repack_to_int8_blocks_is_lossless_and_other_repacks_are_refused() {
        let dir = TmpDir::new("rp");
        let data: Vec<f32> = (0..8 * 64).map(|i| ((i * 13 % 71) as f32 - 35.0) / 9.0).collect();
        let q8 = crate::quant::quantize(crate::gguf::TYPE_Q8_0, &data).unwrap();
        crate::gguf_write::write(dir.join("m.gguf").to_str().unwrap(), &[], &[crate::gguf_write::TensorOut { name: "w".into(), shape: vec![8, 64], ty: crate::gguf::TYPE_Q8_0, data: q8.clone() }], 32).unwrap();
        let r = PackedReader::open(&dir.join("m.gguf")).unwrap();
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(1 << 20, 0));
        let l = loader(&r, &codecs, &budget, 2048);
        let (mut a, mut b) = (MemorySink::default(), MemorySink::default());
        let pre = l.load("w", &LoadPolicy::Preserve, &mut a).unwrap();
        assert_eq!(a.payload, q8, "Preserve copies the stored bytes");
        assert_eq!(pre.step.op, LoadOp::Preserve);
        let rep = l.load("w", &LoadPolicy::Repack(QuantEncoding::int8_blocked(32)), &mut b).unwrap();
        assert!(!rep.step.decodes_to_f32 && rep.step.intermediate_full_tensors == 0);
        let decode = |enc: QuantEncoding, s: &MemorySink| {
            let d = TensorDescriptor::new("w", vec![8, 64], enc);
            let tile = Tile { elems: 0..512, payload: &s.payload, scales: s.scales.iter().map(Vec::as_slice).collect() };
            let mut out = Vec::new();
            codecs.decoder(&d.encoding).unwrap().decode_tile(&d, &tile, &mut out).unwrap();
            out
        };
        assert_eq!(decode(QuantEncoding::int8_blocked(32), &b), decode(QuantEncoding::ggml(GgmlType::Q8_0), &a), "repack must not change a single decoded value");
        let err = l.plan("w", &LoadPolicy::Repack(QuantEncoding::nf4_blocked(32))).unwrap_err();
        assert!(err.contains("no lossless repack"), "{err}");
    }

    #[test]
    fn concurrent_loads_share_one_budget() {
        let dir = TmpDir::new("conc");
        let r = Arc::new(bf16_file(&dir, 64, 8192));
        let codecs = Arc::new(CodecRegistry::reference());
        let one = {
            let budget = MemoryBudget::new(1 << 20, 0);
            loader(r.as_ref(), &codecs, &budget, 32 << 10).plan("w", &LoadPolicy::Convert(QuantEncoding::int8_blocked(32))).unwrap().staging_bytes_per_tile
        };
        // Room for one and a half loads: two cannot hold their staging at once.
        let budget = MemoryBudget::new(one + one / 2, 0);
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let (r, codecs, budget) = (r.clone(), codecs.clone(), budget.clone());
                std::thread::spawn(move || {
                    let mut sink = MemorySink::default();
                    loader(r.as_ref(), &codecs, &budget, 32 << 10).load("w", &LoadPolicy::Convert(QuantEncoding::int8_blocked(32)), &mut sink).unwrap();
                    sink.payload
                })
            })
            .collect();
        let outs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(outs[0], outs[1]);
        let s = budget.stats(Pool::HostStaging);
        assert!(s.high_water <= s.capacity && s.high_water >= one, "high water {} cap {}", s.high_water, s.capacity);
        assert_eq!(s.in_use, 0);
    }

    #[test]
    fn a_failed_load_aborts_the_sink_and_returns_every_byte_to_the_budget() {
        let dir = TmpDir::new("fail");
        let r = bf16_file(&dir, 16, 4096);
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(256 << 10, 0));
        let l = loader(&r, &codecs, &budget, 16 << 10);
        let mut sink = CheckSink { fail_at_tile: Some(3), ..Default::default() };
        let err = l.load("w", &LoadPolicy::Convert(QuantEncoding::int8_blocked(32)), &mut sink).unwrap_err();
        assert!(err.contains("sink full") && sink.aborted && !sink.finished, "{err}");
        assert_eq!(budget.stats(Pool::HostStaging).in_use, 0);
        l.load("w", &LoadPolicy::Preserve, &mut CheckSink::default()).expect("the budget is usable after a failed load");
    }

    #[test]
    fn a_tensor_whose_smallest_tile_exceeds_the_budget_is_refused_up_front() {
        let dir = TmpDir::new("tiny");
        let r = bf16_file(&dir, 4, 4096);
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(64, 0));
        let err = loader(&r, &codecs, &budget, 1 << 20).plan("w", &LoadPolicy::Convert(QuantEncoding::int8_blocked(32))).unwrap_err();
        assert!(err.contains("exceeds"), "{err}");
    }

    /// RssFile is process-wide: the two residency tests must not overlap.
    #[cfg(target_os = "linux")]
    static RSS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(target_os = "linux")]
    #[test]
    fn consumed_file_pages_are_released_so_residency_stays_near_the_tile() {
        let _serial = RSS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = TmpDir::new("rss");
        let r = bf16_file(&dir, 64, 524288); // 64 MiB file
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(2 << 20, 0));
        let l = loader(&r, &codecs, &budget, 1 << 20);
        let rep = l.load("w", &LoadPolicy::Preserve, &mut CheckSink::default()).unwrap();
        let grew = rep.file_resident_growth_bytes.expect("RssFile is reported on Linux");
        assert!(grew < 16 << 20, "file-backed residency grew by {grew} B while streaming a 64 MiB tensor");
        eprintln!("MEASURED file-pages: 64 MiB tensor preserved in 1 MiB tiles, RssFile growth {grew} B, leased peak {} B", rep.peak_staging_bytes);
    }

    /// Forwards everything except `release`: the negative control showing the
    /// residency measurement can see pages pile up.
    struct NoRelease<'a>(&'a PackedReader);

    impl PackedTensorReader for NoRelease<'_> {
        fn names(&self) -> Vec<String> {
            self.0.names()
        }
        fn descriptor(&self, name: &str) -> Result<TensorDescriptor, String> {
            self.0.descriptor(name)
        }
        fn read_blocks(&self, desc: &TensorDescriptor, elems: &Range<u64>, dst: &mut TileBuf) -> Result<(), String> {
            self.0.read_blocks(desc, elems, dst)
        }
        fn release(&self, _: &TensorDescriptor, _: &Range<u64>) {}
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn without_release_the_same_load_leaves_the_whole_file_resident() {
        let _serial = RSS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = TmpDir::new("rss0");
        let inner = bf16_file(&dir, 64, 524288);
        let r = NoRelease(&inner);
        let (codecs, budget) = (CodecRegistry::reference(), MemoryBudget::new(2 << 20, 0));
        let rep = loader(&r, &codecs, &budget, 1 << 20).load("w", &LoadPolicy::Preserve, &mut CheckSink::default()).unwrap();
        let grew = rep.file_resident_growth_bytes.unwrap();
        assert!(grew > 48 << 20, "control: expected ~64 MiB resident without release, saw {grew} B");
        eprintln!("MEASURED file-pages control (no release): RssFile growth {grew} B");
    }
}
