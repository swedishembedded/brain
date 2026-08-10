// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming, memory-mapped weight I/O — the load/convert foundation.
//!
//! HARD invariant: loading **and** conversion mmap the file and stream one
//! tensor at a time — decode/dequant a single tensor into a small buffer, hand
//! it to the caller, drop it. Peak host memory ≈ one tensor's fp32 expansion,
//! never the whole model (a 4B model is ~16 GB as f32; a quantized GGUF blows
//! up ~8× when dequantized in bulk).
//!
//! [`WeightReader`] is a lazy reader over **both** safetensors and GGUF, chosen
//! by file content (GGUF magic) then extension. [`open`](WeightReader::open)
//! mmaps + parses the header only; [`tensor`](WeightReader::tensor) /
//! [`for_each`](WeightReader::for_each) materialize at most one tensor at a
//! time. [`StWriter`] is the mirror on the write side: an incremental
//! safetensors writer that plans offsets up front (sizes only) and appends each
//! tensor's bytes as the caller yields them, so a converter never builds the
//! whole output in RAM (unlike `safetensors::serialize` / [`crate::st`]).
//!
//! Native-only: mmap and fs do not exist on wasm, which keeps the eager
//! byte-slice paths in [`crate::st`] / [`crate::gguf`].
#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde_json::Value;

use crate::gguf::MmapGguf;
use crate::mmap::MmapSafetensors;
use crate::st::{ModelCard, CONFIG_KEY};

/// The on-disk format a [`WeightReader`] wraps.
enum Inner {
    St(MmapSafetensors),
    Gguf(MmapGguf),
    /// A sharded HF checkpoint (`model.safetensors.index.json` +
    /// `model-K-of-N.safetensors`): one mmap per shard file + which shard owns
    /// each tensor name. See [`WeightReader::open_hf_dir`].
    StSharded(Vec<MmapSafetensors>, HashMap<String, usize>),
}

/// A lazy, mmap-backed reader over a safetensors or GGUF weight file. Decodes
/// one tensor at a time from the mapping; `open` reads only the header.
pub struct WeightReader {
    inner: Inner,
    /// Names in the underlying file's iteration order (st: name-sorted; gguf: file order).
    order: Vec<String>,
    /// Unified per-tensor shape (u64), for the borrow-returning `shape` accessor.
    shapes: HashMap<String, Vec<u64>>,
}

impl WeightReader {
    /// mmap `path` and parse only its header (no tensor data is read). Format is
    /// sniffed from the leading `GGUF` magic, falling back to a `.gguf`
    /// extension, else treated as safetensors.
    pub fn open(path: &str) -> io::Result<WeightReader> {
        let inner = if is_gguf(path)? {
            Inner::Gguf(MmapGguf::open(path).map_err(inval)?)
        } else {
            Inner::St(MmapSafetensors::open(path).map_err(inval)?)
        };
        let (order, shapes) = match &inner {
            Inner::St(m) => shape_index(m.names(), |n| m.shape(n).map(usize_to_u64)),
            Inner::Gguf(m) => shape_index(m.names(), |n| m.shape(n).map(usize_to_u64)),
            Inner::StSharded(..) => unreachable!("open() constructs only St/Gguf; see open_hf_dir"),
        };
        Ok(WeightReader { inner, order, shapes })
    }

    /// Open a **foreign** (non-brain) HuggingFace checkpoint directory: a
    /// single `model.safetensors`, or a sharded set of
    /// `model.safetensors.index.json` plus `model-K-of-N.safetensors` files.
    /// Each shard is mmapped (header only) — `tensor`/`for_each` still decode
    /// exactly one tensor at a time, from whichever shard owns it, so an
    /// importer streaming through a sharded multi-GB checkpoint never holds
    /// more than one shard's mmap header plus one tensor's fp32 expansion.
    /// Unlike [`open`](Self::open), `config()`/`card()` return empty/`None`
    /// here — a foreign checkpoint has no `brain.config`/`brain.card`; read
    /// the source's own `config.json` separately (as every importer already
    /// does) and build a [`ModelCard`] for the *output*.
    pub fn open_hf_dir(dir: &Path) -> io::Result<WeightReader> {
        // HF-transformers and diffusers checkpoints name their shard index
        // differently (mirrors crate::safetensors::index_filename).
        let index_path = ["model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"]
            .into_iter()
            .map(|n| dir.join(n))
            .find(|p| p.exists());
        let Some(index_path) = index_path else {
            // No index: exactly one *.safetensors file (mirrors
            // crate::safetensors::read_model_dir's single-file fallback).
            let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            candidates.sort();
            let path = candidates
                .into_iter()
                .next()
                .ok_or_else(|| inval(format!("no .safetensors file in {}", dir.display())))?;
            let path = path.to_str().ok_or_else(|| inval("non-utf8 shard path".to_string()))?;
            return Self::open(path);
        };

        let idx_bytes = std::fs::read(&index_path)?;
        let idx: Value = serde_json::from_slice(&idx_bytes).map_err(|e| inval(format!("bad index.json: {e}")))?;
        let weight_map = idx["weight_map"]
            .as_object()
            .ok_or_else(|| inval("index.json: missing weight_map object".to_string()))?;

        // Unique shard filenames, sorted for a deterministic shard index
        // (matches read_model_dir's determinism).
        let shard_files: std::collections::BTreeSet<String> = weight_map
            .values()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| inval("index.json: weight_map value is not a string".to_string())))
            .collect::<io::Result<_>>()?;
        let shard_files: Vec<String> = shard_files.into_iter().collect();
        let shard_of: HashMap<&str, usize> = shard_files.iter().enumerate().map(|(i, f)| (f.as_str(), i)).collect();

        let mut readers = Vec::with_capacity(shard_files.len());
        for f in &shard_files {
            readers.push(MmapSafetensors::open(dir.join(f)).map_err(inval)?);
        }

        let mut owner: HashMap<String, usize> = HashMap::with_capacity(weight_map.len());
        let mut shapes = HashMap::with_capacity(weight_map.len());
        let mut order = Vec::with_capacity(weight_map.len());
        for (name, file_val) in weight_map {
            let file = file_val.as_str().unwrap(); // validated above
            let si = shard_of[file];
            owner.insert(name.clone(), si);
            if let Some(s) = readers[si].shape(name) {
                shapes.insert(name.clone(), usize_to_u64(s));
            }
            order.push(name.clone());
        }
        order.sort(); // deterministic regardless of the index JSON's own key order

        Ok(WeightReader { inner: Inner::StSharded(readers, owner), order, shapes })
    }

    /// Tensor names, in the underlying file's order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(|s| s.as_str())
    }

    /// A tensor's shape, if present.
    pub fn shape(&self, name: &str) -> Option<&[u64]> {
        self.shapes.get(name).map(|s| s.as_slice())
    }

    /// A tensor's declared safetensors dtype string (`"F32"`/`"U32"`/…), if
    /// present -- the safe way to check whether a tensor is packed (`U32`,
    /// [`Self::tensor_u32`]'s domain) before reaching for [`Self::tensor`]
    /// or [`Self::tensor_u32`], both of which panic on the wrong accessor
    /// rather than silently return an empty/reinterpreted value. `None` for
    /// GGUF -- brain's int8-native packed-weight convention is
    /// safetensors-only ([`Self::tensor_u32`]'s own doc).
    pub fn dtype(&self, name: &str) -> Option<&str> {
        match &self.inner {
            Inner::St(m) => m.dtype(name),
            Inner::Gguf(_) => None,
            Inner::StSharded(readers, owner) => readers[*owner.get(name)?].dtype(name),
        }
    }

    /// The model config: `brain.config` for safetensors, the KV map for GGUF.
    pub fn config(&self) -> Value {
        match &self.inner {
            Inner::St(m) => m.config(),
            Inner::Gguf(m) => m.config(),
            // A foreign checkpoint's config is its own config.json, read
            // separately by the caller -- see open_hf_dir's doc.
            Inner::StSharded(..) => Value::Null,
        }
    }

    /// The [`ModelCard`], if one can be derived (always `Some` for GGUF).
    pub fn card(&self) -> Option<ModelCard> {
        match &self.inner {
            Inner::St(m) => m.card(),
            Inner::Gguf(m) => Some(m.model_card()),
            Inner::StSharded(..) => None,
        }
    }

    /// The tokenizer embedded in a GGUF's `tokenizer.ggml.*` KV, if any.
    /// Always `None` for safetensors (whose tokenizer is a sibling file).
    pub fn tokenizer(&self) -> Option<crate::gguf::GgufTokenizer> {
        match &self.inner {
            Inner::St(_) | Inner::StSharded(..) => None,
            Inner::Gguf(m) => m.tokenizer(),
        }
    }

    /// Decode/dequant exactly one tensor to fp32, now, from the mmap. `None` if
    /// the name is unknown. Panics with a clear message only if a GGUF tensor
    /// exists but its quant type is unsupported (IQ/TQ/MXFP4 codebooks).
    pub fn tensor(&self, name: &str) -> Option<Vec<f32>> {
        match &self.inner {
            Inner::St(m) => m.tensor_f32(name),
            Inner::Gguf(m) => m.tensor(name).map(|r| r.expect("gguf dequant")),
            Inner::StSharded(readers, owner) => readers[*owner.get(name)?].tensor_f32(name),
        }
    }

    /// Decode exactly one tensor as packed `u32` words -- the read side of
    /// [`StWriter::write_u32`]'s int8-native packed layout
    /// (`model::int8::quantize_weight`'s 4-int8-per-u32 packing, stored
    /// as-is). `None` if the name is unknown. Panics if the tensor's declared
    /// dtype is not `U32` (safetensors: [`MmapSafetensors::tensor_u32`]'s own
    /// panic), or unconditionally for GGUF -- brain writes int8-native
    /// checkpoints only as safetensors, so a GGUF caller reaching for this
    /// has the wrong reader, not a value worth defaulting past.
    pub fn tensor_u32(&self, name: &str) -> Option<Vec<u32>> {
        match &self.inner {
            Inner::St(m) => m.tensor_u32(name),
            Inner::Gguf(_) => panic!("tensor_u32: '{name}': GGUF has no U32 packed-weight convention -- this is a safetensors-only accessor"),
            Inner::StSharded(readers, owner) => readers[*owner.get(name)?].tensor_u32(name),
        }
    }

    /// Stream every tensor, one at a time: for each, decode it, call `f`, then
    /// drop it before the next. Peak extra allocation is one tensor's fp32.
    pub fn for_each<F: FnMut(&str, &[u64], Vec<f32>)>(&self, mut f: F) {
        for name in &self.order {
            let data = self.tensor(name).expect("indexed tensor");
            let shape = self.shapes.get(name).map(|s| s.as_slice()).unwrap_or(&[]);
            f(name, shape, data);
        }
    }
}

impl crate::TensorSource for WeightReader {
    /// Streaming source: decode exactly one tensor from the mmap, lend it to
    /// `f`, drop it on return (peak host ≈ one tensor's fp32 expansion).
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.tensor(name) {
            Some(v) => {
                f(&v);
                true
            }
            None => false,
        }
    }

    /// Zero-copy for safetensors when the dtype already matches (`F32`/`U32`,
    /// 4-byte aligned); `None` for GGUF (every tensor is quantized on disk, so
    /// there is nothing to bind as-is without dequantizing).
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        match &self.inner {
            Inner::St(m) => m.raw_words(name),
            Inner::Gguf(_) => None,
            Inner::StSharded(readers, owner) => readers[*owner.get(name)?].raw_words(name),
        }
    }

    /// Bounded chunked decode for safetensors (the real win — a BF16 tensor
    /// converts one chunk at a time into a reused scratch instead of once as
    /// a whole `Vec<f32>`). GGUF inherits the trait default (materialize via
    /// `with_tensor`, one chunk) — its quant-block dequant does not yet have
    /// a chunked path; that is a named follow-up, not a silent regression,
    /// since the default's cost is identical to today's behaviour.
    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        match &self.inner {
            Inner::St(m) => m.with_tensor_chunks(name, max_elems, f),
            Inner::Gguf(_) => self.with_tensor(name, &mut |d| f(0, d)),
            Inner::StSharded(readers, owner) => match owner.get(name) {
                Some(&si) => readers[si].with_tensor_chunks(name, max_elems, f),
                None => false,
            },
        }
    }

    fn numel(&self, name: &str) -> Option<usize> {
        match &self.inner {
            Inner::St(m) => m.numel(name),
            Inner::Gguf(_) => self.shapes.get(name).map(|s| s.iter().product::<u64>() as usize),
            Inner::StSharded(readers, owner) => readers[*owner.get(name)?].numel(name),
        }
    }
}

fn usize_to_u64(s: &[usize]) -> Vec<u64> {
    s.iter().map(|&d| d as u64).collect()
}

/// Build the (order, name→u64-shape) index from a name list + shape lookup.
fn shape_index(
    names: &[String],
    shape_of: impl Fn(&str) -> Option<Vec<u64>>,
) -> (Vec<String>, HashMap<String, Vec<u64>>) {
    let mut shapes = HashMap::with_capacity(names.len());
    for n in names {
        if let Some(s) = shape_of(n) {
            shapes.insert(n.clone(), s);
        }
    }
    (names.to_vec(), shapes)
}

/// True if `path` is GGUF: `GGUF` magic in the first four bytes, else a `.gguf`
/// extension. Reads only four bytes.
fn is_gguf(path: &str) -> io::Result<bool> {
    use io::Read;
    let mut magic = [0u8; 4];
    match File::open(path).and_then(|mut f| f.read_exact(&mut magic)) {
        Ok(()) if &magic == b"GGUF" => return Ok(true),
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {}
        Err(e) => return Err(e),
    }
    Ok(Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")))
}

fn inval(e: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

// ---- incremental safetensors writer ----

/// A planned tensor's on-disk element type. Both variants are 4 bytes/element
/// (so byte-range planning is dtype-agnostic — only the header's declared
/// `dtype` string and which `write*` method may target the slot differ).
/// `U32` is for int8-native storage: `model::int8::quantize_weight`'s packed
/// layout (4 int8 packed per u32) stored as-is, no repacking at load time —
/// see [`StWriter::create_mixed`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    U32,
}

impl Dtype {
    fn tag(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::U32 => "U32",
        }
    }
}

/// One planned tensor: name, expected element count, its absolute byte
/// offset in the output file, and its declared dtype (byte width is 4 for
/// both variants today, so offset planning does not need to branch on it).
struct Planned {
    name: String,
    numel: usize,
    offset: u64,
    dtype: Dtype,
}

/// An incremental F32 safetensors writer. Offsets are planned up front from
/// (name, shape) — sizes only, no data — the header is written immediately, and
/// each tensor's little-endian f32 bytes are [`write`](Self::write)able **in
/// any order** (each exactly once) by seeking to its planned offset — a
/// converter that streams through its SOURCE in the source's own natural
/// order (rather than the output plan's order) doesn't need to buffer/reorder
/// anything to satisfy this writer. Never holds more than the caller's current
/// tensor in RAM. The output format matches [`crate::st`] (`brain.config`,
/// card via `to_metadata`).
pub struct StWriter {
    file: BufWriter<File>,
    tmp: String,
    path: String,
    plan: Vec<Planned>,
    index_by_name: HashMap<String, usize>,
    written: Vec<bool>,
    written_count: usize,
    blob_start: u64,
}

impl StWriter {
    /// Plan an output file: for each `(name, shape)`, reserve a contiguous F32
    /// byte range. Writes `[u64 header_len][json header]` immediately; tensor
    /// data must then be supplied by [`write`](StWriter::write) in plan order.
    pub fn create(
        path: &str,
        plan: &[(String, Vec<u64>)],
        config: &Value,
        card: Option<&ModelCard>,
    ) -> io::Result<StWriter> {
        let mixed: Vec<(String, Vec<u64>, Dtype)> = plan.iter().map(|(n, s)| (n.clone(), s.clone(), Dtype::F32)).collect();
        Self::create_mixed(path, &mixed, config, card)
    }

    /// [`create`](StWriter::create), but each planned tensor declares its own
    /// [`Dtype`] — the seam an int8-native checkpoint (packed weights stored
    /// as `Dtype::U32`, matching `model::int8::quantize_weight`'s layout
    /// exactly, plus an ordinary `Dtype::F32` per-channel scale tensor
    /// alongside it) needs. `create` is unchanged and still produces
    /// byte-identical output to before this existed — every current caller
    /// (`qwen`/`glm`/`lfm`'s importers) is unaffected.
    pub fn create_mixed(
        path: &str,
        plan: &[(String, Vec<u64>, Dtype)],
        config: &Value,
        card: Option<&ModelCard>,
    ) -> io::Result<StWriter> {
        // Plan contiguous byte ranges (every dtype here is 4 bytes/element)
        // and build the header.
        let mut header = serde_json::Map::new();
        let mut off: u64 = 0;
        let mut planned = Vec::with_capacity(plan.len());
        for (name, shape, dtype) in plan {
            let numel: u64 = shape.iter().product();
            let nbytes = numel * 4;
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype.tag(),
                    "shape": shape,
                    "data_offsets": [off, off + nbytes],
                }),
            );
            planned.push(Planned { name: name.clone(), numel: numel as usize, offset: off, dtype: *dtype });
            off += nbytes;
        }

        let mut meta = serde_json::Map::new();
        meta.insert(CONFIG_KEY.to_string(), Value::String(serde_json::to_string(config)?));
        if let Some(c) = card {
            for (k, v) in c.to_metadata() {
                meta.insert(k, Value::String(v));
            }
        }
        header.insert("__metadata__".to_string(), Value::Object(meta));
        let hbytes = serde_json::to_vec(&Value::Object(header))?;

        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let tmp = format!("{path}.tmp");
        let mut file = BufWriter::new(File::create(&tmp)?);
        file.write_all(&(hbytes.len() as u64).to_le_bytes())?;
        file.write_all(&hbytes)?;
        let blob_start = 8 + hbytes.len() as u64;

        let index_by_name: HashMap<String, usize> = planned.iter().enumerate().map(|(i, p)| (p.name.clone(), i)).collect();
        let written = vec![false; planned.len()];
        Ok(StWriter { file, tmp, path: path.to_string(), plan: planned, index_by_name, written, written_count: 0, blob_start })
    }

    /// Write one planned tensor's data. May be called in any order (each
    /// planned name exactly once) — seeks to that tensor's pre-planned offset,
    /// so a caller streaming through its source in the source's own order
    /// never needs to buffer or reorder its output.
    pub fn write(&mut self, name: &str, data: &[f32]) -> io::Result<()> {
        let i = self.check_slot(name, data.len(), Dtype::F32)?;
        self.seek_to(i)?;
        // Stream the f32 little-endian bytes without materializing the whole blob.
        for v in data {
            self.file.write_all(&v.to_le_bytes())?;
        }
        self.mark_written(i);
        Ok(())
    }

    /// [`write`](StWriter::write) for a `Dtype::U32`-planned tensor — the
    /// int8-native path: `data` is `model::int8::quantize_weight`'s packed
    /// output (4 int8 per u32), written as-is with no repacking.
    pub fn write_u32(&mut self, name: &str, data: &[u32]) -> io::Result<()> {
        let i = self.check_slot(name, data.len(), Dtype::U32)?;
        self.seek_to(i)?;
        for v in data {
            self.file.write_all(&v.to_le_bytes())?;
        }
        self.mark_written(i);
        Ok(())
    }

    fn check_slot(&self, name: &str, len: usize, want: Dtype) -> io::Result<usize> {
        let i = *self.index_by_name.get(name).ok_or_else(|| inval(format!("st: '{name}' is not in the plan")))?;
        if self.written[i] {
            return Err(inval(format!("st: '{name}' written twice")));
        }
        let p = &self.plan[i];
        if p.dtype != want {
            return Err(inval(format!(
                "st: '{name}' is planned as {:?} but write{} was called",
                p.dtype,
                if want == Dtype::U32 { "_u32" } else { "" }
            )));
        }
        if p.numel != len {
            return Err(inval(format!("st: '{name}' expects {} elems, got {len}", p.numel)));
        }
        Ok(i)
    }

    fn seek_to(&mut self, i: usize) -> io::Result<()> {
        use io::{Seek, SeekFrom};
        self.file.seek(SeekFrom::Start(self.blob_start + self.plan[i].offset)).map(|_| ())
    }

    fn mark_written(&mut self, i: usize) {
        self.written[i] = true;
        self.written_count += 1;
    }

    /// Flush and atomically rename tmp → path. Errors if the plan was not fully
    /// written (the file would have holes).
    pub fn finish(mut self) -> io::Result<()> {
        if self.written_count != self.plan.len() {
            let missing: Vec<&str> = self
                .plan
                .iter()
                .zip(&self.written)
                .filter(|(_, &w)| !w)
                .map(|(p, _)| p.name.as_str())
                .collect();
            return Err(inval(format!(
                "st: wrote {}/{} planned tensors (missing: {missing:?})",
                self.written_count,
                self.plan.len()
            )));
        }
        self.file.flush()?;
        drop(self.file);
        std::fs::rename(&self.tmp, &self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // ---- test-scoped counting allocator: tracks peak live bytes while armed ----

    struct Counting;
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static ARMED: AtomicBool = AtomicBool::new(false);

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = System.alloc(layout);
            if !p.is_null() && ARMED.load(Ordering::Relaxed) {
                let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            }
            p
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            if ARMED.load(Ordering::Relaxed) {
                // Saturating: frees of pre-arm allocations must not underflow.
                let mut cur = LIVE.load(Ordering::Relaxed);
                loop {
                    let next = cur.saturating_sub(layout.size());
                    match LIVE.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(c) => cur = c,
                    }
                }
            }
            System.dealloc(ptr, layout);
        }
    }

    #[global_allocator]
    static ALLOC: Counting = Counting;

    /// Run `f` with peak-live-byte tracking armed; returns the peak.
    fn peak_live<R>(f: impl FnOnce() -> R) -> (R, usize) {
        LIVE.store(0, Ordering::Relaxed);
        PEAK.store(0, Ordering::Relaxed);
        ARMED.store(true, Ordering::Relaxed);
        let r = f();
        ARMED.store(false, Ordering::Relaxed);
        (r, PEAK.load(Ordering::Relaxed))
    }

    fn scratch(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("brain-weightio-{}-{}.st", name, std::process::id()))
            .to_str()
            .unwrap()
            .to_string()
    }

    // ---- StWriter round-trips ----

    #[test]
    fn stwriter_roundtrips_eager_and_streaming() {
        let p = scratch("rt");
        let cfg = serde_json::json!({"d_model": 8, "n_layers": 1});
        let card = ModelCard::new("m", "fam");
        let plan =
            vec![("a".to_string(), vec![2u64, 2]), ("b".to_string(), vec![3u64])];
        let a = vec![1.0f32, -2.5, 3.25, 4.0];
        let b = vec![0.1f32, 0.2, 0.3];

        let mut w = StWriter::create(&p, &plan, &cfg, Some(&card)).unwrap();
        w.write("a", &a).unwrap();
        w.write("b", &b).unwrap();
        w.finish().unwrap();

        // Eager reload (crate::st) sees identical values + config + card.
        let m = crate::st::load_safetensors(&p).unwrap();
        assert_eq!(m.tensors["a"], a);
        assert_eq!(m.tensors["b"], b);
        assert_eq!(m.config(), cfg);
        assert_eq!(m.card().unwrap(), card);

        // Streaming reload matches the eager map exactly.
        let r = WeightReader::open(&p).unwrap();
        assert_eq!(r.config(), cfg);
        assert_eq!(r.card().unwrap(), card);
        let mut seen: HashMap<String, Vec<f32>> = HashMap::new();
        r.for_each(|n, shape, data| {
            assert_eq!(shape, r.shape(n).unwrap());
            seen.insert(n.to_string(), data);
        });
        assert_eq!(seen["a"], a);
        assert_eq!(seen["b"], b);
        assert_eq!(r.tensor("a").unwrap(), a);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn create_mixed_writes_packed_int8_alongside_f32_scale() {
        // The Qwen3-Omni motivation: a real DP4A packed weight
        // (model::int8::quantize_weight's exact output shape) plus its
        // per-channel scale, stored as two entries with different dtypes in
        // one file, and an ordinary F32 tensor alongside them.
        let p = scratch("i8-mixed");
        let cfg = serde_json::json!({"family": "omni"});
        let card = ModelCard::new("m", "omni");
        let plan = vec![
            ("expert.0.gate.weight".to_string(), vec![4u64, 2], Dtype::U32), // [n=4, k/4=2]
            ("expert.0.gate.scale".to_string(), vec![4u64], Dtype::F32),
            ("plain.weight".to_string(), vec![3u64], Dtype::F32),
        ];
        let packed: Vec<u32> = vec![0x0403_0201, 0xFFFE_FDFC, 0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444, 0x5555_5555, 0x6666_6666];
        let scale = vec![0.01f32, 0.02, 0.03, 0.04];
        let plain = vec![9.0f32, 8.0, 7.0];

        let mut w = StWriter::create_mixed(&p, &plan, &cfg, Some(&card)).unwrap();
        // dtype mismatch is rejected in both directions.
        assert!(w.write("expert.0.gate.weight", &plain).is_err());
        assert!(w.write_u32("plain.weight", &packed[..3]).is_err());
        w.write_u32("expert.0.gate.weight", &packed).unwrap();
        w.write("expert.0.gate.scale", &scale).unwrap();
        w.write("plain.weight", &plain).unwrap();
        w.finish().unwrap();

        // Independent verification via the raw `safetensors` crate (NOT
        // brain's own reader), so this test cannot pass merely because a
        // brain-side reader and writer share the same bug.
        let bytes = std::fs::read(&p).unwrap();
        let (_header_len, sts) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        assert_eq!(sts.tensors().iter().find(|(n, _)| *n == "expert.0.gate.weight").unwrap().1.dtype, safetensors::Dtype::U32);
        assert_eq!(sts.tensors().iter().find(|(n, _)| *n == "expert.0.gate.scale").unwrap().1.dtype, safetensors::Dtype::F32);
        assert_eq!(sts.tensors().iter().find(|(n, _)| *n == "plain.weight").unwrap().1.dtype, safetensors::Dtype::F32);

        let full = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let got_packed = full.tensor("expert.0.gate.weight").unwrap();
        assert_eq!(got_packed.shape(), &[4, 2]);
        let got_packed_u32: Vec<u32> = got_packed.data().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(got_packed_u32, packed);

        let got_scale = full.tensor("expert.0.gate.scale").unwrap();
        let got_scale_f32: Vec<f32> = got_scale.data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(got_scale_f32, scale);

        // Brain's own streaming reader reads both the F32 tensors AND the
        // packed U32 tensor -- bit-identical round trip through tensor_u32,
        // read back independently of the raw-safetensors check above.
        let r = WeightReader::open(&p).unwrap();
        assert_eq!(r.tensor("plain.weight").unwrap(), plain);
        assert_eq!(r.tensor("expert.0.gate.scale").unwrap(), scale);
        assert_eq!(r.tensor_u32("expert.0.gate.weight").unwrap(), packed);
        // tensor_f32 on a U32 tensor must error loudly, not silently return
        // an empty/reinterpreted vector -- reaching for the wrong accessor is
        // a caller bug, not a value worth papering over.
        let f32_on_u32 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.tensor("expert.0.gate.weight")));
        assert!(f32_on_u32.is_err(), "tensor() (f32 decode) on a U32 tensor should panic, not return a value");

        std::fs::remove_file(&p).ok();
    }

    /// The round-trip gate Phase 4 asks for, in isolation from the mixed-file
    /// test above: a packed tensor written via `write_u32` reads back
    /// bit-identical via `tensor_u32`, and an unknown/wrong-accessor dtype
    /// errors (panics) rather than silently yielding `[]` -- the defect
    /// `decode`'s old `_ => Vec::new()` catch-all had.
    #[test]
    fn u32_round_trips_and_wrong_accessor_errors_loudly() {
        let p = scratch("u32-roundtrip");
        let plan = vec![("packed".to_string(), vec![3u64], Dtype::U32)];
        let data: Vec<u32> = vec![0xDEAD_BEEF, 0x0000_0001, 0xFFFF_FFFF];

        let mut w = StWriter::create_mixed(&p, &plan, &Value::Null, None).unwrap();
        w.write_u32("packed", &data).unwrap();
        w.finish().unwrap();

        let r = WeightReader::open(&p).unwrap();
        assert_eq!(r.tensor_u32("packed").unwrap(), data);

        // Wrong accessor on a real U32 tensor: loud, not empty.
        let via_f32 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.tensor("packed")));
        assert!(via_f32.is_err(), "tensor_f32 on a U32 tensor must panic rather than return []");

        // Wrong accessor the other direction: tensor_u32 on a real F32
        // tensor must also refuse rather than reinterpret the bits.
        let plan2 = vec![("float".to_string(), vec![2u64])];
        let p2 = scratch("u32-roundtrip-f32");
        let mut w2 = StWriter::create(&p2, &plan2, &Value::Null, None).unwrap();
        w2.write("float", &[1.5, 2.5]).unwrap();
        w2.finish().unwrap();
        let r2 = WeightReader::open(&p2).unwrap();
        let u32_via_f32 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r2.tensor_u32("float")));
        assert!(u32_via_f32.is_err(), "tensor_u32 on an F32 tensor must panic rather than reinterpret its bits");

        std::fs::remove_file(&p).ok();
        std::fs::remove_file(&p2).ok();
    }

    #[test]
    fn stwriter_enforces_plan_order() {
        let p = scratch("order");
        let plan = vec![("a".to_string(), vec![1u64]), ("b".to_string(), vec![1u64])];
        let mut w = StWriter::create(&p, &plan, &Value::Null, None).unwrap();
        assert!(w.write("c", &[1.0]).is_err()); // not in the plan
        assert!(w.write("a", &[1.0, 2.0]).is_err()); // wrong count
        w.write("a", &[1.0]).unwrap();
        assert!(w.write("a", &[9.0]).is_err()); // written twice
        // finish before writing 'b' must fail (would leave a hole).
        let mut w2 = StWriter::create(&p, &plan, &Value::Null, None).unwrap();
        w2.write("a", &[1.0]).unwrap();
        let err = format!("{}", w2.finish().unwrap_err());
        assert!(err.contains('b'), "error should name the missing tensor: {err}");
        w.write("b", &[2.0]).unwrap();
        w.finish().unwrap();
        std::fs::remove_file(&p).ok();
    }

    /// A caller may write planned tensors in ANY order (e.g. streaming through
    /// its source in the source's own natural order, not the output plan's) --
    /// the result is byte-identical to writing in plan order.
    #[test]
    fn stwriter_out_of_order_write_matches_in_order() {
        let plan = vec![("a".to_string(), vec![2u64]), ("b".to_string(), vec![3u64]), ("c".to_string(), vec![1u64])];
        let cfg = serde_json::json!({"k": "v"});

        let p1 = scratch("order-fwd");
        let mut w1 = StWriter::create(&p1, &plan, &cfg, None).unwrap();
        w1.write("a", &[1.0, 2.0]).unwrap();
        w1.write("b", &[3.0, 4.0, 5.0]).unwrap();
        w1.write("c", &[6.0]).unwrap();
        w1.finish().unwrap();

        let p2 = scratch("order-rev");
        let mut w2 = StWriter::create(&p2, &plan, &cfg, None).unwrap();
        w2.write("c", &[6.0]).unwrap();
        w2.write("a", &[1.0, 2.0]).unwrap();
        w2.write("b", &[3.0, 4.0, 5.0]).unwrap();
        w2.finish().unwrap();

        let b1 = std::fs::read(&p1).unwrap();
        let b2 = std::fs::read(&p2).unwrap();
        assert_eq!(b1, b2, "write order must not affect the output bytes");
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }

    /// Streaming `for_each` equals eager `load_safetensors` for an F16 file.
    #[test]
    fn streaming_matches_eager_f16() {
        // Hand-build an F16 + F32 safetensors via the safetensors crate.
        let f16b: Vec<u8> = [half::f16::from_f32(1.5), half::f16::from_f32(-2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f32b: Vec<u8> = [3.0f32, -4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let a = safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2], &f16b).unwrap();
        let b = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &f32b).unwrap();
        let bytes =
            safetensors::serialize(vec![("a".to_string(), a), ("b".to_string(), b)], None).unwrap();
        let p = scratch("f16");
        std::fs::write(&p, &bytes).unwrap();

        let eager = crate::st::load_safetensors(&p).unwrap();
        let r = WeightReader::open(&p).unwrap();
        r.for_each(|n, _s, data| assert_eq!(&data, &eager.tensors[n], "{n}"));
        assert_eq!(r.tensor("a").unwrap(), vec![1.5, -2.0]);
        std::fs::remove_file(&p).ok();
    }

    /// GGUF streaming `tensor(name)` for a quantized tensor equals the P1c eager
    /// dequant (same hand-built Q4_0 block fixture style as gguf.rs).
    #[test]
    fn gguf_streaming_matches_eager_q4_0() {
        fn put_str(v: &mut Vec<u8>, s: &str) {
            v.extend((s.len() as u64).to_le_bytes());
            v.extend(s.as_bytes());
        }
        // One Q4_0 tensor of 32 elems: d=2.0, qs[0]=0x3A, rest 0x88.
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes());
        h.extend(1u64.to_le_bytes()); // 1 tensor
        h.extend(0u64.to_le_bytes()); // 0 kv
        put_str(&mut h, "w");
        h.extend(1u32.to_le_bytes()); // 1 dim
        h.extend(32u64.to_le_bytes()); // ne[0] = 32
        h.extend(2u32.to_le_bytes()); // type Q4_0
        h.extend(0u64.to_le_bytes()); // offset
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        h.extend(half::f16::from_f32(2.0).to_le_bytes());
        h.push(0x3A);
        h.extend([0x88u8; 15]);

        let p = std::env::temp_dir().join(format!("brain-weightio-q4-{}.gguf", std::process::id()));
        std::fs::write(&p, &h).unwrap();
        let ps = p.to_str().unwrap();

        let eager = crate::gguf::load_gguf(ps).unwrap();
        let r = WeightReader::open(ps).unwrap();
        assert_eq!(r.shape("w"), Some([32u64].as_slice()));
        assert_eq!(r.tensor("w").unwrap(), eager.tensors["w"]);
        assert_eq!(r.tensor("w").unwrap()[0], 4.0); // (0xA-8)*2
        assert_eq!(r.tensor("w").unwrap()[16], -10.0); // (0x3-8)*2
        std::fs::remove_file(&p).ok();
    }

    /// Memory bound: streaming a multi-tensor file via `for_each` keeps peak live
    /// allocation near ONE tensor, far below the file's total tensor bytes (what
    /// an eager load holds at once).
    #[test]
    fn for_each_is_memory_bounded() {
        let p = scratch("membound");
        let n = 100_000usize; // 400 KB per tensor as f32
        let ntensors = 8usize;
        let one_bytes = n * 4;
        let total_bytes = one_bytes * ntensors;

        // Write via the streaming writer (itself bounded — one tensor at a time).
        let plan: Vec<(String, Vec<u64>)> =
            (0..ntensors).map(|i| (format!("t{i}"), vec![n as u64])).collect();
        let mut w = StWriter::create(&p, &plan, &Value::Null, None).unwrap();
        for i in 0..ntensors {
            let data = vec![i as f32; n];
            w.write(&format!("t{i}"), &data).unwrap();
        }
        w.finish().unwrap();

        let r = WeightReader::open(&p).unwrap();

        // Stream: retain only a checksum per tensor, dropping each Vec immediately.
        let (sum, peak_stream) = peak_live(|| {
            let mut acc = 0.0f64;
            r.for_each(|_n, _s, data| {
                acc += data[0] as f64; // touch, do not retain
            });
            acc
        });
        assert_eq!(sum, (0..ntensors).map(|i| i as f64).sum::<f64>());

        // Peak while streaming must be well under the whole-file footprint and
        // within a small multiple of a single tensor.
        assert!(
            peak_stream < total_bytes / 2,
            "streaming peak {peak_stream} not << total {total_bytes}"
        );
        assert!(
            peak_stream < one_bytes * 3,
            "streaming peak {peak_stream} exceeds ~one tensor ({one_bytes})"
        );

        // For contrast, the eager load holds every tensor at once: peak ≥ total.
        let (_m, peak_eager) = peak_live(|| crate::st::load_safetensors(&p).unwrap());
        assert!(
            peak_eager >= total_bytes,
            "eager peak {peak_eager} should hold the whole file ({total_bytes})"
        );
        assert!(peak_stream < peak_eager / 2);

        std::fs::remove_file(&p).ok();
    }

    /// The streaming [`crate::TensorSource`] (a `WeightReader`) yields, for every
    /// tensor, byte-identical f32 data to the eager whole-model `by_role("")`
    /// map — the numeric-parity guarantee the streaming model-load path relies on
    /// (equal weights in ⇒ identical device weights ⇒ identical numerics). No GPU.
    #[test]
    fn tensor_source_streaming_matches_eager() {
        use crate::TensorSource;
        let p = scratch("srcparity");
        let cfg = serde_json::json!({"d_model": 4, "n_layers": 2});
        let tensors = vec![
            ("tok.weight".to_string(), vec![3u64, 4], (0..12).map(|i| i as f32 * 0.5 - 1.0).collect::<Vec<f32>>()),
            ("blocks.0.w".to_string(), vec![4u64, 4], (0..16).map(|i| (i as f32).sin()).collect::<Vec<f32>>()),
            ("blocks.1.b".to_string(), vec![4u64], vec![-2.0f32, 0.0, 3.5, 7.25]),
        ];
        crate::st::save_safetensors(&p, &tensors, &cfg, None).unwrap();

        // Eager whole-model host copy (what the old load path built).
        let eager = crate::load(&p).by_role("");
        // Streaming source over the same file.
        let reader = WeightReader::open(&p).unwrap();

        assert_eq!(reader.config(), cfg);
        for (name, _shape, data) in &tensors {
            // HashMap source == the raw data.
            let mut from_map: Option<Vec<f32>> = None;
            assert!(eager.with_tensor(name, &mut |v| from_map = Some(v.to_vec())));
            assert_eq!(from_map.as_ref().unwrap(), data, "{name} eager");
            // WeightReader source == the raw data, byte-for-byte.
            let mut from_reader: Option<Vec<f32>> = None;
            assert!(reader.with_tensor(name, &mut |v| from_reader = Some(v.to_vec())));
            assert_eq!(from_reader.as_ref().unwrap(), data, "{name} streamed");
        }
        // Absent name: both report not-found without invoking the callback.
        assert!(!reader.with_tensor("nope", &mut |_| panic!("must not call")));
        assert!(!eager.with_tensor("nope", &mut |_| panic!("must not call")));
        std::fs::remove_file(&p).ok();
    }

    /// `open()` reads only the header, not the blob: opening a file whose header
    /// declares a huge tensor is instant and allocates almost nothing.
    #[test]
    fn open_does_not_read_blob() {
        // Header claims a 256 MB tensor; we only write a stub blob, and never
        // touch the tensor, so mmap + header parse must not fault the data in.
        let p = scratch("lazyopen");
        let huge = 64_000_000u64; // 256 MB as f32
        let header = serde_json::json!({
            "big": {"dtype": "F32", "shape": [huge], "data_offsets": [0u64, huge * 4]},
        });
        let hb = serde_json::to_vec(&header).unwrap();
        let mut file = (hb.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&hb);
        // Only a few real bytes of the "blob" — enough to mmap, not the full size.
        file.extend_from_slice(&[0u8; 64]);
        std::fs::write(&p, &file).unwrap();

        let (r, peak) = peak_live(|| WeightReader::open(&p).unwrap());
        assert_eq!(r.shape("big"), Some([huge].as_slice()));
        // Opening allocated far less than the declared tensor (no blob copy).
        assert!(peak < (huge * 4) as usize / 100, "open allocated {peak}");
        std::fs::remove_file(&p).ok();
    }

    // ---- open_hf_dir: sharded + single-file foreign checkpoints ----

    /// A one-tensor F32 safetensors byte buffer (mirrors
    /// crate::safetensors::tests::one_tensor_bytes).
    fn one_tensor_bytes(name: &str, vals: &[f32]) -> Vec<u8> {
        let hdr = serde_json::json!({
            name: {"dtype": "F32", "shape": [vals.len()], "data_offsets": [0, vals.len() * 4]},
        });
        let hb = serde_json::to_vec(&hdr).unwrap();
        let mut buf = (hb.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(&hb);
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("brain-weightio-hfdir-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn open_hf_dir_sharded_matches_eager_read_model_dir() {
        let dir = scratch_dir("sharded");
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), one_tensor_bytes("a", &[1.0, 2.0])).unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), one_tensor_bytes("b", &[3.0, 4.0, 5.0])).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 20},
                "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
            }))
            .unwrap(),
        )
        .unwrap();

        let eager = crate::safetensors::read_model_dir(&dir).unwrap();
        let mut eager_map: HashMap<String, Vec<f32>> = HashMap::new();
        for t in eager {
            eager_map.insert(t.name, t.data);
        }

        let streamed = WeightReader::open_hf_dir(&dir).unwrap();
        assert_eq!(streamed.shape("a"), Some([2u64].as_slice()));
        assert_eq!(streamed.shape("b"), Some([3u64].as_slice()));
        assert_eq!(streamed.tensor("a").unwrap(), eager_map["a"]);
        assert_eq!(streamed.tensor("b").unwrap(), eager_map["b"]);
        // Config/card are intentionally empty for a foreign checkpoint.
        assert_eq!(streamed.config(), Value::Null);
        assert!(streamed.card().is_none());

        let mut seen: HashMap<String, Vec<f32>> = HashMap::new();
        streamed.for_each(|n, _, data| {
            seen.insert(n.to_string(), data);
        });
        assert_eq!(seen, eager_map);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_hf_dir_recognizes_the_diffusers_index_filename() {
        // Diffusers-style checkpoints (e.g. Z-Image's `transformer/` dir) name
        // their index `diffusion_pytorch_model.safetensors.index.json`, not
        // the HF-transformers `model.safetensors.index.json`. A dir with only
        // the diffusers name must still be read as fully sharded, not
        // silently fall back to "no index -> just open one file".
        let dir = scratch_dir("diffusers-sharded");
        std::fs::write(dir.join("diffusion_pytorch_model-00001-of-00002.safetensors"), one_tensor_bytes("a", &[1.0, 2.0])).unwrap();
        std::fs::write(dir.join("diffusion_pytorch_model-00002-of-00002.safetensors"), one_tensor_bytes("b", &[3.0, 4.0, 5.0])).unwrap();
        let index = serde_json::json!({
            "metadata": {"total_size": 20},
            "weight_map": {
                "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
            },
        });
        std::fs::write(dir.join("diffusion_pytorch_model.safetensors.index.json"), serde_json::to_vec(&index).unwrap()).unwrap();

        let streamed = WeightReader::open_hf_dir(&dir).unwrap();
        assert_eq!(streamed.tensor("a").unwrap(), vec![1.0, 2.0]);
        assert_eq!(streamed.tensor("b").unwrap(), vec![3.0, 4.0, 5.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_hf_dir_single_file_no_index() {
        let dir = scratch_dir("single");
        std::fs::write(dir.join("model.safetensors"), one_tensor_bytes("w", &[9.0, 8.0])).unwrap();

        let streamed = WeightReader::open_hf_dir(&dir).unwrap();
        assert_eq!(streamed.tensor("w").unwrap(), vec![9.0, 8.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_hf_dir_missing_reports_a_clean_error() {
        let dir = scratch_dir("empty");
        assert!(WeightReader::open_hf_dir(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
