// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! [`PackedSource`]: the compatibility bridge from the tile reader to
//! [`TensorSource`], the by-name f32 interface existing model builders pull
//! from.
//!
//! A model supplies only a declarative name map ([`Binding`]: canonical name to
//! stored name, plus an optional additive constant for checkpoints that store
//! a fold of the value). Everything else, discovering scale tensors, decoding
//! FP8/NF4/GGUF/bf16 bytes, bounding scratch, releasing file pages, is the
//! reader and the codec registry. Chunks are decoded tile by tile, so host
//! scratch is `O(max_elems)`, never `O(tensor)`; only [`TensorSource::with_tensor`]
//! (whose contract is a whole-tensor slice) assembles a whole tensor, in tiles.

use std::collections::HashMap;
use std::ops::Range;

use super::codec::{CodecRegistry, Tile, TileBuf};
use super::concat::ConcatView;
use super::load::{LoadPolicy, LoadReport, LoadRequest, Loader};
use super::sink::TensorSink;
use super::encoding::{ScaleGranularity, TensorDescriptor, INT8_SCALE_SUFFIX};
use super::reader::{PackedReader, PackedTensorReader};
use super::report::PrecisionReport;
use crate::weightio::WeightReader;
use crate::TensorSource;

/// Tile length used to assemble a whole tensor for `with_tensor`.
const WHOLE_TENSOR_TILE_ELEMS: u64 = 1 << 22;

/// Where one canonical tensor comes from.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// The tensor's name in the checkpoint.
    pub source: String,
    /// Added to every decoded value (a stored `w` that is applied as `1 + w`
    /// is declared with `add: 1.0`). `0.0` is the identity.
    pub add: f32,
}

impl Binding {
    pub fn new(source: impl Into<String>) -> Binding {
        Binding { source: source.into(), add: 0.0 }
    }
}

struct Bound {
    desc: TensorDescriptor,
    add: f32,
}

/// A [`TensorSource`] over a [`PackedTensorReader`].
pub struct PackedSource {
    reader: Box<dyn PackedTensorReader>,
    codecs: CodecRegistry,
    tensors: HashMap<String, Bound>,
}

impl PackedSource {
    /// Serve `bindings` from `reader`. Every binding is resolved and every
    /// encoding checked decodable now, from headers alone, so a mismatch is one
    /// named error at open rather than a failure deep into a load.
    pub fn new(reader: Box<dyn PackedTensorReader>, codecs: CodecRegistry, bindings: impl IntoIterator<Item = (String, Binding)>) -> Result<PackedSource, String> {
        let mut tensors = HashMap::new();
        for (canonical, b) in bindings {
            let desc = reader.descriptor(&b.source)?;
            codecs.decoder(&desc.encoding).map_err(|e| format!("{}: {e}", b.source))?;
            tensors.insert(canonical, Bound { desc, add: b.add });
        }
        Ok(PackedSource { reader, codecs, tensors })
    }

    /// Every tensor under its own stored name.
    pub fn identity(reader: Box<dyn PackedTensorReader>, codecs: CodecRegistry) -> Result<PackedSource, String> {
        let bindings: Vec<_> = reader.names().into_iter().map(|n| (n.clone(), Binding::new(n))).collect();
        PackedSource::new(reader, codecs, bindings)
    }

    /// [`Self::identity`] over every tensor the reader can serve packed; tensors
    /// with no packed encoding (an integer buffer, an unsupported GGUF type) are
    /// left out instead of failing the open. For a checkpoint that also holds
    /// tensors the caller never asks for.
    pub fn identity_lenient(reader: Box<dyn PackedTensorReader>, codecs: CodecRegistry) -> PackedSource {
        let mut tensors = HashMap::new();
        for name in reader.names() {
            let Ok(desc) = reader.descriptor(&name) else { continue };
            if codecs.decoder(&desc.encoding).is_ok() {
                tensors.insert(name, Bound { desc, add: 0.0 });
            }
        }
        PackedSource { reader, codecs, tensors }
    }

    /// Canonical names served, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name).map(|b| &b.desc)
    }

    /// Elements `[start, start + len)` of `name`'s flat layout as f32, decoding
    /// only the tiles that cover them (a handful of embedding rows out of a
    /// large table). `None` when the name is unknown or the range leaves the
    /// tensor. The decoded span is at most the range rounded out to the tensor's
    /// tile quantum.
    ///
    /// The pages read stay mapped: a point read is repeated on the same hot rows (every serving round looks the
    /// same tokens' embeddings up again), and dropping the mapping after each read costs a `madvise` and a page
    /// fault per row - 15 ms of host time in front of a 600-row round. A table's pages are bounded by the table.
    pub fn read_range(&self, name: &str, start: u64, len: u64) -> Result<Option<Vec<f32>>, String> {
        let Some(b) = self.tensors.get(name) else { return Ok(None) };
        let Some(end) = start.checked_add(len).filter(|&e| e <= b.desc.numel() && len > 0) else { return Ok(None) };
        let q = b.desc.tile_quantum();
        let span = start / q * q..(end.div_ceil(q) * q).min(b.desc.numel());
        let (mut read, mut out) = (TileBuf::default(), Vec::new());
        self.decode_range(b, &span, &mut read, &mut out, false)?;
        out.truncate((end - span.start) as usize);
        out.drain(..(start - span.start) as usize);
        Ok(Some(out))
    }

    /// Rows `rows` of the 2-D tensor `name` (`row_len` elements each) concatenated in the order asked, the rows decoded in
    /// parallel. `None` when the name is unknown or a row leaves the tensor. See [`Self::read_range`].
    pub fn read_rows(&self, name: &str, rows: &[usize], row_len: usize) -> Result<Option<Vec<f32>>, String> {
        let Some(b) = self.tensors.get(name) else { return Ok(None) };
        let numel = b.desc.numel();
        if row_len == 0 || rows.iter().any(|&r| r.checked_add(1).and_then(|e| e.checked_mul(row_len)).is_none_or(|e| e as u64 > numel)) {
            return Ok(None);
        }
        let decoded = backend_cpu::par::map(rows.len(), |i| self.read_range(name, (rows[i] * row_len) as u64, row_len as u64));
        let mut out = Vec::with_capacity(rows.len() * row_len);
        for row in decoded {
            out.extend_from_slice(&row?.expect("the row was bounds-checked above"));
        }
        Ok(Some(out))
    }

    /// The format and bytes of every tensor served, as stored, under the names the model asks for.
    pub fn precision_report(&self) -> PrecisionReport {
        let renamed: Vec<TensorDescriptor> = self.tensors.iter().map(|(canonical, b)| TensorDescriptor { name: canonical.clone(), ..b.desc.clone() }).collect();
        PrecisionReport::from_descriptors(&renamed)
    }

    /// Decode tile `elems` of `b` into `out` (replacing its contents), with
    /// `read` as the staging buffer.
    fn decode_range(&self, b: &Bound, elems: &Range<u64>, read: &mut TileBuf, out: &mut Vec<f32>, release: bool) -> Result<(), String> {
        let desc = &b.desc;
        self.reader.read_blocks(desc, elems, read)?;
        let t = Tile { elems: elems.clone(), payload: &read.payload, scales: read.scales.iter().map(Vec::as_slice).collect() };
        out.clear();
        out.reserve_exact((elems.end - elems.start) as usize);
        self.codecs.decoder(&desc.encoding)?.decode_tile(desc, &t, out)?;
        if b.add != 0.0 {
            out.iter_mut().for_each(|v| *v += b.add);
        }
        if release {
            self.reader.release(desc, elems);
        }
        Ok(())
    }

    /// Tile length for chunks of at most `max_elems`: rounded down to the
    /// tensor's tile quantum but never below one quantum; `0` means one tile.
    fn tile_len(desc: &TensorDescriptor, max_elems: u64) -> u64 {
        let quantum = desc.tile_quantum();
        if max_elems == 0 { desc.numel() } else { (max_elems / quantum).max(1) * quantum }
    }

    /// Decode `b` in tiles of at most `max_elems`, calling `f` with each tile's
    /// element offset and values, in order.
    fn for_each_tile(&self, b: &Bound, max_elems: u64, f: &mut dyn FnMut(u64, &[f32])) -> Result<(), String> {
        let (numel, tile) = (b.desc.numel(), Self::tile_len(&b.desc, max_elems));
        let (mut read, mut out) = (TileBuf::default(), Vec::new());
        let mut start = 0;
        while start < numel {
            let elems = start..(start + tile).min(numel);
            self.decode_range(b, &elems, &mut read, &mut out, true)?;
            f(start, &out);
            start = elems.end;
        }
        Ok(())
    }

    /// The whole tensor in one vector, decoded tile by tile straight into it
    /// (no second tensor-sized scratch).
    fn decode_whole(&self, b: &Bound) -> Result<Vec<f32>, String> {
        let (numel, tile) = (b.desc.numel(), Self::tile_len(&b.desc, WHOLE_TENSOR_TILE_ELEMS));
        let mut read = TileBuf::default();
        if tile >= numel {
            let mut out = Vec::new();
            self.decode_range(b, &(0..numel), &mut read, &mut out, true)?;
            return Ok(out);
        }
        let (mut whole, mut part) = (Vec::with_capacity(numel as usize), Vec::new());
        let mut start = 0;
        while start < numel {
            let elems = start..(start + tile).min(numel);
            self.decode_range(b, &elems, &mut read, &mut part, true)?;
            whole.extend_from_slice(&part);
            start = elems.end;
        }
        Ok(whole)
    }
}

impl PackedSource {
    /// A fused group whose parts are stored in different encodings (FP8 attention next to bf16, say) cannot
    /// be joined byte for byte. Under an explicit destination encoding (`Repack` or `Convert`, the caller's
    /// request to change the representation) each part is loaded on its own into its row range of one fused
    /// destination, tile by tile, so the group still never passes through a whole f32 tensor. Under `Preserve`
    /// there is no single encoding to join them in, and the group is declined: the caller loads the parts as
    /// separate weights.
    fn load_mixed_parts(&self, parts: &[TensorDescriptor], req: &LoadRequest<'_>, sink: &mut dyn TensorSink) -> Result<Option<LoadReport>, String> {
        let (LoadPolicy::Repack(dst_enc) | LoadPolicy::Convert(dst_enc)) = req.policy else { return Ok(None) };
        let k = parts[0].shape.last().copied().unwrap_or(0);
        if parts.iter().any(|p| p.shape.len() != 2 || p.shape[1] != k) || dst_enc.scales.iter().any(|s| s.granularity == ScaleGranularity::Tensor) {
            return Ok(None);
        }
        let rows: u64 = parts.iter().map(|p| p.shape[0]).sum();
        let names = parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("+");
        let fused = TensorDescriptor::new(format!("fused({names})"), vec![rows, k], dst_enc.clone());
        fused.validate()?;
        // Every part must start and end on a tile of the destination, so its tiles and scale bands are its own.
        let quantum = fused.encode_quantum();
        if parts.iter().any(|p| !p.numel().is_multiple_of(quantum)) {
            return Ok(None);
        }
        let loader = Loader { reader: &*self.reader, codecs: &self.codecs, workspace: req.workspace.clone(), config: req.config.clone() };
        let Ok(steps) = parts.iter().map(|p| loader.plan(&p.name, req.policy)).collect::<Result<Vec<_>, _>>() else { return Ok(None) };
        sink.begin(&fused)?;
        let mut report: Option<LoadReport> = None;
        let mut first_elem = 0;
        for step in &steps {
            let mut part_sink = RowRangeSink { inner: &mut *sink, fused: &fused, first_elem };
            match loader.execute(step, &mut part_sink) {
                Ok(r) => {
                    report = Some(match report.take() {
                        None => r,
                        Some(mut total) => {
                            total.tiles_written += r.tiles_written;
                            total.bytes_read += r.bytes_read;
                            total.peak_staging_bytes = total.peak_staging_bytes.max(r.peak_staging_bytes);
                            total
                        }
                    })
                }
                Err(e) => {
                    sink.abort(&fused);
                    return Err(e);
                }
            }
            first_elem += step.src.numel();
        }
        sink.finish(&fused)?;
        Ok(report.map(|mut r| {
            r.step.dst = fused;
            r
        }))
    }
}

/// One part of a fused destination: forwards the part's tiles to the shared sink at the part's element
/// offset, and leaves `begin`/`finish`/`abort` of the whole fused tensor to the caller.
struct RowRangeSink<'a> {
    inner: &'a mut dyn TensorSink,
    fused: &'a TensorDescriptor,
    first_elem: u64,
}

impl TensorSink for RowRangeSink<'_> {
    fn begin(&mut self, _: &TensorDescriptor) -> Result<(), String> {
        Ok(())
    }
    fn write_tile(&mut self, _: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
        let elems = tile.elems.start + self.first_elem..tile.elems.end + self.first_elem;
        self.inner.write_tile(self.fused, &Tile { elems, payload: tile.payload, scales: tile.scales.clone() })
    }
    fn finish(&mut self, _: &TensorDescriptor) -> Result<(), String> {
        Ok(())
    }
    fn abort(&mut self, _: &TensorDescriptor) {}
}

impl TensorSource for PackedSource {
    /// Panics on a read error after open validated the headers (a truncated or
    /// concurrently modified file): silently reporting "not found" would
    /// surface as a missing-tensor naming bug far from the cause.
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let Some(b) = self.tensors.get(name) else { return false };
        f(&self.decode_whole(b).unwrap_or_else(|e| panic!("PackedSource: {name}: {e}")));
        true
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        let Some(b) = self.tensors.get(name) else { return false };
        self.for_each_tile(b, max_elems as u64, f).unwrap_or_else(|e| panic!("PackedSource: {name}: {e}"));
        true
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.tensors.get(name).map(|b| b.desc.numel() as usize)
    }

    /// Only the tiles that cover the range are decoded.
    fn with_tensor_range(&self, name: &str, start: usize, len: usize, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.read_range(name, start as u64, len as u64).unwrap_or_else(|e| panic!("PackedSource: {name}: {e}")) {
            Some(v) => {
                f(&v);
                true
            }
            None => false,
        }
    }

    fn packed_descriptor(&self, name: &str) -> Option<TensorDescriptor> {
        self.tensors.get(name).map(|b| b.desc.clone())
    }

    /// Several names are fused row-wise (see `concat`); a binding with an
    /// additive fold cannot be loaded as bytes and is left to the f32 interface.
    fn load_packed(&self, names: &[&str], req: &LoadRequest<'_>, sink: &mut dyn TensorSink) -> Result<Option<LoadReport>, String> {
        let mut parts = Vec::with_capacity(names.len());
        for n in names {
            match self.tensors.get(*n) {
                Some(b) if b.add == 0.0 => parts.push(b.desc.clone()),
                _ => return Ok(None),
            }
        }
        let mut run = |reader: &dyn PackedTensorReader, name: &str| -> Result<Option<LoadReport>, String> {
            let loader = Loader { reader, codecs: &self.codecs, workspace: req.workspace.clone(), config: req.config.clone() };
            let Ok(step) = loader.plan(name, req.policy) else { return Ok(None) };
            loader.execute(&step, sink).map(Some)
        };
        match parts.as_slice() {
            [] => Ok(None),
            [one] => run(&*self.reader, &one.name),
            _ => match ConcatView::new(&*self.reader, parts.clone()) {
                Ok(view) => run(&view, &view.descriptor().name.clone()),
                Err(_) => self.load_mixed_parts(&parts, req, sink),
            },
        }
    }

    fn advise_drop(&self, name: &str) {
        if let Some(b) = self.tensors.get(name) {
            self.reader.release(&b.desc, &(0..b.desc.numel()));
        }
    }
}

/// Does the file hold tensors a plain f32 read cannot interpret: reduced floats, FP8 with scale sidecars, or brain's
/// packed int8 words? A file of f32 tensors alone has nothing packed to preserve.
fn holds_packed_tensors(reader: &WeightReader) -> bool {
    reader.names().any(|n| match reader.dtype(n) {
        Some("BF16" | "F16") => true,
        Some(d) if d.starts_with("F8_") => true,
        Some("U32") => reader.dtype(&format!("{n}{INT8_SCALE_SUFFIX}")).is_some(),
        _ => false,
    })
}

/// The source that serves a brain-native or Hugging Face safetensors checkpoint in the encodings it stores: a file
/// with reduced-precision, FP8 or packed-int8 tensors is read through its packed view (bf16, FP8 and int8 tensors
/// stream into a resident tier without a whole-tensor f32 copy, and a tier that is the stored encoding is a byte
/// copy), any other file is read directly. Tensors keep their stored names.
pub fn stored_source(reader: WeightReader) -> Box<dyn TensorSource + Send + Sync> {
    if holds_packed_tensors(&reader) {
        Box::new(PackedSource::identity_lenient(Box::new(PackedReader::new(reader)), CodecRegistry::reference()))
    } else {
        Box::new(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::reader::PackedReader;
    use crate::packed::testfix::{bf16_bytes, e4m3_bytes, write_safetensors, RawTensor, TmpDir};
    use crate::safetensors::e4m3fn_to_f32;

    #[test]
    fn fp8_and_plain_tensors_decode_through_a_name_map_with_bounded_chunks() {
        let dir = TmpDir::new("src");
        let (rows, cols) = (32usize, 48usize);
        let w = e4m3_bytes(21, rows * cols);
        let grid = [0.5f32, 1.0, 2.0, 4.0, 0.25, 8.0];
        let norm = [0.5f32, -0.25, 1.5];
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "a.weight", dtype: "F8_E4M3", shape: vec![rows as u64, cols as u64], bytes: w.clone() },
                RawTensor { name: "a.weight_scale_inv", dtype: "BF16", shape: vec![2, 3], bytes: bf16_bytes(&grid) },
                RawTensor { name: "n.weight", dtype: "BF16", shape: vec![3], bytes: bf16_bytes(&norm) },
            ],
        );
        let reader = PackedReader::open(&dir.join("m.safetensors")).unwrap().with_fp8_block(16);
        let bindings = [("blocks.0.a".to_string(), Binding::new("a.weight")), ("blocks.0.ln".to_string(), Binding { source: "n.weight".into(), add: 1.0 })];
        let src = PackedSource::new(Box::new(reader), CodecRegistry::reference(), bindings).unwrap();
        assert_eq!(src.numel("blocks.0.a"), Some(rows * cols));

        let want: Vec<f32> = (0..rows * cols).map(|i| e4m3fn_to_f32(w[i]) * half::bf16::from_f32(grid[(i / cols) / 16 * 3 + (i % cols) / 16]).to_f32()).collect();
        let (mut got, mut largest) = (Vec::new(), 0usize);
        assert!(src.with_tensor_chunks("blocks.0.a", 2 * cols, &mut |off, c| {
            assert_eq!(off as usize, got.len());
            largest = largest.max(c.len());
            got.extend_from_slice(c);
        }));
        assert_eq!(got, want);
        assert!(largest <= 2 * cols, "chunk of {largest} elements exceeds the requested 2 rows");
        let mut whole = Vec::new();
        assert!(src.with_tensor("blocks.0.a", &mut |d| whole = d.to_vec()));
        assert_eq!(whole, want);
        assert!(src.with_tensor("blocks.0.ln", &mut |d| assert_eq!(d, [1.5, 0.75, 2.5])));
        assert!(!src.with_tensor("missing", &mut |_| {}));
        // A range inside one tile row and one straddling the scale bands: the same values as the whole decode.
        for (start, len) in [(5u64, 7u64), (cols as u64 - 3, 9), (15 * cols as u64 + 40, 2 * cols as u64)] {
            assert_eq!(src.read_range("blocks.0.a", start, len).unwrap().unwrap(), want[start as usize..(start + len) as usize], "range {start}+{len}");
        }
        assert_eq!(src.read_range("blocks.0.a", (rows * cols) as u64 - 1, 2).unwrap(), None, "a range past the end is refused, not truncated");
    }

    #[test]
    fn fused_parts_load_as_the_parts_bytes_end_to_end_and_unfusable_parts_are_declined() {
        use crate::packed::budget::{BoundedWorkspace, MemoryBudget};
        use crate::packed::load::{LoadConfig, LoadPolicy};
        use crate::packed::sink::MemorySink;
        let dir = TmpDir::new("fuse");
        let cols = 32usize;
        let (a, b) = (e4m3_bytes(41, 16 * cols), e4m3_bytes(42, 32 * cols));
        let (ga, gb) = (bf16_bytes(&[1.0, 2.0]), bf16_bytes(&[4.0, 8.0, 0.5, 0.25]));
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "a", dtype: "F8_E4M3", shape: vec![16, cols as u64], bytes: a.clone() },
                RawTensor { name: "a_scale_inv", dtype: "BF16", shape: vec![1, 2], bytes: ga.clone() },
                RawTensor { name: "b", dtype: "F8_E4M3", shape: vec![32, cols as u64], bytes: b.clone() },
                RawTensor { name: "b_scale_inv", dtype: "BF16", shape: vec![2, 2], bytes: gb.clone() },
                RawTensor { name: "n", dtype: "BF16", shape: vec![3], bytes: bf16_bytes(&[1.0; 3]) },
            ],
        );
        let reader = PackedReader::open(&dir.join("m.safetensors")).unwrap().with_fp8_block(16);
        let src = PackedSource::identity(Box::new(reader), CodecRegistry::reference()).unwrap();
        let (ws, cfg) = (BoundedWorkspace::new(MemoryBudget::new(1 << 20, 0)), LoadConfig { tile_bytes: 700 });
        let req = LoadRequest { policy: &LoadPolicy::Preserve, workspace: &ws, config: &cfg };
        let mut sink = MemorySink::default();
        let rep = src.load_packed(&["a", "b"], &req, &mut sink).unwrap().expect("same-encoding tiled parts fuse");
        assert!(rep.tiles_written > 1, "{} tiles", rep.tiles_written);
        assert_eq!(sink.payload, [a, b].concat());
        assert_eq!(sink.scales[0], [ga, gb].concat());
        // Parts of different shape class or encoding are declined, nothing is written.
        let mut untouched = MemorySink::default();
        assert!(src.load_packed(&["a", "n"], &req, &mut untouched).unwrap().is_none());
        assert!(src.load_packed(&["a", "missing"], &req, &mut untouched).unwrap().is_none());
        assert_eq!(untouched.tiles, 0);
    }

    /// A fused group whose parts are stored differently (FP8 next to bf16) is never joined through f32: under
    /// `Preserve` it is declined (the caller loads the parts as separate weights), and under an explicit
    /// destination encoding each part is converted tile by tile into its rows of one fused tensor, byte for byte
    /// what converting the parts separately and joining them gives.
    #[test]
    fn a_fused_group_of_mixed_encodings_is_split_or_explicitly_converted() {
        use crate::packed::budget::{BoundedWorkspace, MemoryBudget};
        use crate::packed::encoding::QuantEncoding;
        use crate::packed::load::{LoadConfig, LoadPolicy};
        use crate::packed::sink::MemorySink;
        let dir = TmpDir::new("mixed-fuse");
        let cols = 32usize;
        let vals: Vec<f32> = (0..32 * cols).map(|i| ((i * 37 % 211) as f32 - 105.0) / 40.0).collect();
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "a", dtype: "F8_E4M3", shape: vec![16, cols as u64], bytes: e4m3_bytes(41, 16 * cols) },
                RawTensor { name: "a_scale_inv", dtype: "BF16", shape: vec![1, 2], bytes: bf16_bytes(&[1.0, 2.0]) },
                RawTensor { name: "b", dtype: "BF16", shape: vec![32, cols as u64], bytes: bf16_bytes(&vals) },
            ],
        );
        let reader = PackedReader::open(&dir.join("m.safetensors")).unwrap().with_fp8_block(16);
        let src = PackedSource::identity(Box::new(reader), CodecRegistry::reference()).unwrap();
        let (ws, cfg) = (BoundedWorkspace::new(MemoryBudget::new(1 << 20, 0)), LoadConfig { tile_bytes: 1024 });

        let mut declined = MemorySink::default();
        let preserve = LoadRequest { policy: &LoadPolicy::Preserve, workspace: &ws, config: &cfg };
        assert!(src.load_packed(&["a", "b"], &preserve, &mut declined).unwrap().is_none(), "no single encoding to preserve both in");
        assert_eq!(declined.tiles, 0);

        for dst in [QuantEncoding::nf4_blocked(32), QuantEncoding::int8_blocked(32), QuantEncoding::bf16()] {
            let policy = LoadPolicy::Convert(dst.clone());
            let req = LoadRequest { policy: &policy, workspace: &ws, config: &cfg };
            let mut fused = MemorySink::default();
            let report = src.load_packed(&["a", "b"], &req, &mut fused).unwrap().expect("an explicit target joins mixed parts");
            assert!(fused.finished && !fused.aborted && report.tiles_written > 2, "{dst}: {} tiles", report.tiles_written);
            assert_eq!(report.step.dst.shape, vec![48, cols as u64]);
            let (mut payload, mut scales) = (Vec::new(), vec![Vec::new(); dst.scales.len()]);
            for part in ["a", "b"] {
                let mut one = MemorySink::default();
                src.load_packed(&[part], &req, &mut one).unwrap().unwrap();
                payload.extend_from_slice(&one.payload);
                for (l, s) in one.scales.iter().enumerate() {
                    scales[l].extend_from_slice(s);
                }
            }
            assert_eq!(fused.payload, payload, "{dst}: payload");
            assert_eq!(fused.scales, scales, "{dst}: scales");
        }
    }
}
