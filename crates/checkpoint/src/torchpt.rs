// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure-Rust reader for PyTorch `torch.save` checkpoints (`.pt`), the
//! zip-container format used by torch >= 1.6.
//!
//! A `.pt` file is an uncompressed ZIP (see [`crate::zipread`]) holding a
//! pickle stream (`<root>/data.pkl`) describing the object tree plus one raw
//! little-endian storage blob per tensor (`<root>/data/<key>`, where `<root>`
//! is the single top-level directory, usually `archive` or the file stem).
//! We interpret exactly the pickle protocol-2+ opcode subset torch emits for
//! (possibly nested) state_dicts, resolve every `_rebuild_tensor_v2` node
//! against its storage blob, and materialize each tensor as *contiguous* f32
//! (applying `storage_offset` and strides, so non-contiguous views come out
//! correct). F64/F32/F16/BF16/I64 storages are converted to f32; any other dtype
//! is an error — the reader guarantees full coverage: every tensor in the
//! file is returned or the whole read fails. Non-tensor leaves (ints, floats,
//! strs, None, ...) are skipped silently but counted in
//! [`ReadReport::skipped_non_tensor`].
//!
//! I64 is supported because it is unavoidable, not because it is meaningful:
//! every `nn.BatchNorm2d` serializes an int64 `num_batches_tracked` scalar, so
//! *any* conv net with BatchNorm is unreadable without it. Widening i64 to f32
//! is exact below 2^24 and approximate above — the same lossy-by-design contract
//! F64 already carries. Importers drop these tensors by name.
//!
//! Nested dict keys are flattened with '.' joins: `{"denoiser": {"conv.weight":
//! t}}` yields the tensor name `denoiser.conv.weight`. Elements of lists and
//! tuples are flattened with their index as the name component.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::safetensors::{bf16_to_f32, f16_to_f32};
use crate::zipread;

/// One tensor from a checkpoint: flattened name, shape, contiguous f32 data.
#[derive(Debug)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Full result of reading a checkpoint: every tensor, plus a count of the
/// non-tensor leaves (hyperparameters, step counters, strings, ...) skipped.
#[derive(Debug)]
pub struct ReadReport {
    pub tensors: Vec<NamedTensor>,
    pub skipped_non_tensor: usize,
}

// ---------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum DType {
    F32,
    F64,
    F16,
    BF16,
    I64,
}

impl DType {
    fn elem_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::F16 | DType::BF16 => 2,
        }
    }
}

/// Map a torch storage class (from the persistent-id tuple) to its dtype.
/// Accepts both the legacy `torch.FloatStorage` and the `torch.storage.*`
/// module paths — the class *name* carries the dtype either way.
fn storage_dtype(module: &str, name: &str) -> Result<DType, String> {
    match name {
        "FloatStorage" => Ok(DType::F32),
        "DoubleStorage" => Ok(DType::F64),
        "HalfStorage" => Ok(DType::F16),
        "BFloat16Storage" => Ok(DType::BF16),
        // Every `nn.BatchNorm2d` serializes an int64 `num_batches_tracked` scalar,
        // so any conv net with BN is unreadable without this arm. Converted to f32
        // like the other non-f32 dtypes; see the note on `DType::I64` in
        // `storage_f32` for the precision caveat.
        "LongStorage" => Ok(DType::I64),
        _ => Err(format!(
            "torchpt: unsupported storage dtype {module}.{name} \
             (supported: FloatStorage, DoubleStorage, HalfStorage, BFloat16Storage, LongStorage)"
        )),
    }
}

// ---------------------------------------------------------------------------
// pickle value model
// ---------------------------------------------------------------------------

/// A storage reference resolved from a persistent id: dtype + archive key.
struct StorageRef {
    dtype: DType,
    key: String,
}

/// A `torch._utils._rebuild_tensor_v2` node: view of a storage.
struct TensorNode {
    storage: Rc<StorageRef>,
    offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

/// Pickle values. Containers are `Rc<RefCell<..>>` so that memoized objects
/// (BINPUT before SETITEMS/APPENDS is the standard pickle pattern) share
/// mutations with the copy left on the stack.
#[derive(Clone)]
enum Val {
    None,
    Bool(bool),
    Int(i64),
    /// Payload kept for value-model completeness; floats are non-tensor
    /// leaves the flattener only counts, so it is never read back.
    Float(#[allow(dead_code)] f64),
    Str(Rc<String>),
    Tuple(Rc<Vec<Val>>),
    List(Rc<RefCell<Vec<Val>>>),
    Dict(Rc<RefCell<Vec<(Val, Val)>>>),
    Global(Rc<(String, String)>),
    Storage(Rc<StorageRef>),
    Tensor(Rc<TensorNode>),
    /// Stack sentinel for the MARK opcode; never appears inside values.
    Mark,
}

impl Val {
    fn kind(&self) -> &'static str {
        match self {
            Val::None => "None",
            Val::Bool(_) => "bool",
            Val::Int(_) => "int",
            Val::Float(_) => "float",
            Val::Str(_) => "str",
            Val::Tuple(_) => "tuple",
            Val::List(_) => "list",
            Val::Dict(_) => "dict",
            Val::Global(_) => "global",
            Val::Storage(_) => "storage",
            Val::Tensor(_) => "tensor",
            Val::Mark => "mark",
        }
    }
}

fn as_usize(v: &Val, what: &str) -> Result<usize, String> {
    match v {
        Val::Int(i) if *i >= 0 => Ok(*i as usize),
        other => Err(format!("torchpt: expected non-negative int for {what}, got {}", other.kind())),
    }
}

/// A tuple (or list) of non-negative ints — tensor sizes and strides.
fn as_usize_seq(v: &Val, what: &str) -> Result<Vec<usize>, String> {
    let items: Vec<Val> = match v {
        Val::Tuple(t) => t.as_ref().clone(),
        Val::List(l) => l.borrow().clone(),
        other => return Err(format!("torchpt: expected tuple for {what}, got {}", other.kind())),
    };
    items.iter().map(|e| as_usize(e, what)).collect()
}

// ---------------------------------------------------------------------------
// pickle opcode interpreter
// ---------------------------------------------------------------------------

struct Unpickler<'a> {
    b: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    memo: HashMap<u32, Val>,
}

impl<'a> Unpickler<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self
            .b
            .get(self.pos..self.pos + n)
            .ok_or_else(|| format!("torchpt: pickle truncated at offset {}", self.pos))?;
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16le(&mut self) -> Result<u16, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32le(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    /// Read up to (and consuming) the next `\n`; used by GLOBAL.
    fn line(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.b.len() {
            return Err("torchpt: pickle truncated inside GLOBAL line".into());
        }
        let s = std::str::from_utf8(&self.b[start..self.pos])
            .map_err(|_| "torchpt: non-utf8 GLOBAL line".to_string())?
            .to_string();
        self.pos += 1; // consume '\n'
        Ok(s)
    }
    /// Bytes -> string: utf8, falling back to latin-1 (pickle STRING opcodes
    /// are raw bytes; torch only puts ascii in them).
    fn str_from(bytes: &[u8]) -> String {
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => bytes.iter().map(|&b| b as char).collect(),
        }
    }

    fn pop(&mut self, ctx: &str) -> Result<Val, String> {
        self.stack.pop().ok_or_else(|| format!("torchpt: pickle stack underflow in {ctx}"))
    }
    /// Pop all values above (and including) the most recent MARK.
    fn pop_mark(&mut self, ctx: &str) -> Result<Vec<Val>, String> {
        let at = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Val::Mark))
            .ok_or_else(|| format!("torchpt: no MARK on stack in {ctx}"))?;
        let items = self.stack.split_off(at + 1);
        self.stack.pop(); // the mark itself
        Ok(items)
    }

    fn memo_put(&mut self, key: u32) -> Result<(), String> {
        let v = self.stack.last().ok_or("torchpt: BINPUT on empty stack")?.clone();
        self.memo.insert(key, v);
        Ok(())
    }
    fn memo_get(&mut self, key: u32) -> Result<(), String> {
        let v = self.memo.get(&key).ok_or_else(|| format!("torchpt: BINGET of unset memo key {key}"))?;
        self.stack.push(v.clone());
        Ok(())
    }

    /// Insert into a dict, replacing an existing entry when the key is an
    /// equal primitive (state_dict keys are strings; duplicates are rare but
    /// pickle semantics are last-wins).
    fn dict_insert(d: &Rc<RefCell<Vec<(Val, Val)>>>, k: Val, v: Val) {
        let mut dm = d.borrow_mut();
        let same = |a: &Val, b: &Val| match (a, b) {
            (Val::Str(x), Val::Str(y)) => x == y,
            (Val::Int(x), Val::Int(y)) => x == y,
            (Val::Bool(x), Val::Bool(y)) => x == y,
            (Val::None, Val::None) => true,
            _ => false,
        };
        if let Some(slot) = dm.iter_mut().find(|(ek, _)| same(ek, &k)) {
            slot.1 = v;
        } else {
            dm.push((k, v));
        }
    }

    /// REDUCE / NEWOBJ: call `callable(*args)` for the torch subset.
    fn call(&mut self, callable: Val, args: Val, ctx: &str) -> Result<Val, String> {
        let g = match callable {
            Val::Global(g) => g,
            other => return Err(format!("torchpt: {ctx} on non-global callable ({})", other.kind())),
        };
        let args: Vec<Val> = match args {
            Val::Tuple(t) => t.as_ref().clone(),
            other => return Err(format!("torchpt: {ctx} args is not a tuple ({})", other.kind())),
        };
        let (module, name) = (g.0.as_str(), g.1.as_str());
        match (module, name) {
            ("collections", "OrderedDict") => {
                // OrderedDict() or OrderedDict([(k, v), ...]); the pairs form
                // shows up when hooks dicts carry initial items.
                let d = Rc::new(RefCell::new(Vec::new()));
                if let Some(first) = args.first() {
                    let pairs: Vec<Val> = match first {
                        Val::List(l) => l.borrow().clone(),
                        Val::Tuple(t) => t.as_ref().clone(),
                        other => {
                            return Err(format!(
                                "torchpt: OrderedDict arg is not a pair sequence ({})",
                                other.kind()
                            ))
                        }
                    };
                    for p in pairs {
                        let kv: Vec<Val> = match p {
                            Val::Tuple(t) => t.as_ref().clone(),
                            Val::List(l) => l.borrow().clone(),
                            other => {
                                return Err(format!(
                                    "torchpt: OrderedDict pair is not a tuple ({})",
                                    other.kind()
                                ))
                            }
                        };
                        if kv.len() != 2 {
                            return Err("torchpt: OrderedDict pair does not have 2 elements".into());
                        }
                        Self::dict_insert(&d, kv[0].clone(), kv[1].clone());
                    }
                }
                Ok(Val::Dict(d))
            }
            ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
                // (storage, storage_offset, size, stride[, requires_grad,
                //  backward_hooks[, metadata]])
                if args.len() < 4 {
                    return Err(format!("torchpt: {name} expects >= 4 args, got {}", args.len()));
                }
                let storage = match &args[0] {
                    Val::Storage(s) => s.clone(),
                    other => {
                        return Err(format!("torchpt: {name} arg 0 is not a storage ({})", other.kind()))
                    }
                };
                let offset = as_usize(&args[1], "storage_offset")?;
                let size = as_usize_seq(&args[2], "tensor size")?;
                let stride = as_usize_seq(&args[3], "tensor stride")?;
                if size.len() != stride.len() {
                    return Err(format!(
                        "torchpt: tensor size rank {} != stride rank {}",
                        size.len(),
                        stride.len()
                    ));
                }
                Ok(Val::Tensor(Rc::new(TensorNode { storage, offset, size, stride })))
            }
            ("torch._utils", "_rebuild_parameter") => {
                // (tensor, requires_grad, backward_hooks) -> the wrapped tensor.
                match args.first() {
                    Some(t @ Val::Tensor(_)) => Ok(t.clone()),
                    _ => Err("torchpt: _rebuild_parameter arg 0 is not a tensor".into()),
                }
            }
            _ => {
                // Any other class (an nn.Module subclass -- Ultralytics' YOLO
                // checkpoints pickle live module OBJECTS, not a plain
                // state_dict) has no importable definition here and its
                // constructor args are irrelevant: a plain Python object with
                // no custom `__reduce__` unpickles as `cls.__new__(cls)` (or
                // `REDUCE` with args we don't need) followed by
                // `obj.__setstate__(obj.__dict__)` via BUILD. Modeling it as
                // an empty dict lets the EXISTING BUILD/dict-merge logic
                // above populate it exactly like an OrderedDict: nn.Module's
                // `__dict__` holds `_parameters`/`_buffers`/`_modules` (all
                // OrderedDicts, so tensors nest inside them) alongside
                // `training`/hook dicts/other bookkeeping that `BUILD`
                // already drops (no tensor inside -> not merged). The
                // resulting flattened names carry PyTorch's internal
                // `_modules`/`_parameters` path segments, which is a
                // Torch-pickling-convention detail an importer normalizes
                // away (strip those exact segments) rather than something
                // this reader can know how to skip generically -- it doesn't
                // know which dict IS a `_modules`-shaped one.
                Ok(Val::Dict(Rc::new(RefCell::new(Vec::new()))))
            }
        }
    }

    /// Resolve a BINPERSID value: torch persistent ids are
    /// `('storage', StorageType, key, device, numel)`.
    fn persistent_load(&mut self, pid: Val) -> Result<Val, String> {
        let t = match pid {
            Val::Tuple(t) => t,
            other => return Err(format!("torchpt: persistent id is not a tuple ({})", other.kind())),
        };
        if t.len() < 3 {
            return Err(format!("torchpt: persistent id tuple too short ({} elements)", t.len()));
        }
        match &t[0] {
            Val::Str(s) if s.as_str() == "storage" => {}
            other => {
                return Err(format!("torchpt: unsupported persistent id kind ({})", other.kind()))
            }
        }
        let (module, name) = match &t[1] {
            Val::Global(g) => (g.0.clone(), g.1.clone()),
            other => {
                return Err(format!("torchpt: persistent id storage type is not a global ({})", other.kind()))
            }
        };
        let key = match &t[2] {
            Val::Str(s) => s.as_ref().clone(),
            other => return Err(format!("torchpt: persistent id key is not a str ({})", other.kind())),
        };
        let dtype = storage_dtype(&module, &name)?;
        Ok(Val::Storage(Rc::new(StorageRef { dtype, key })))
    }

    fn run(&mut self) -> Result<Val, String> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    // PROTO <byte>
                    let v = self.u8()?;
                    if v > 5 {
                        return Err(format!("torchpt: unsupported pickle protocol {v}"));
                    }
                }
                0x95 => {
                    // FRAME <8-byte len> — framing hint only, ignore.
                    self.take(8)?;
                }
                0x2e => {
                    // STOP
                    return self.pop("STOP");
                }

                // -- containers ------------------------------------------------
                0x7d => self.stack.push(Val::Dict(Rc::new(RefCell::new(Vec::new())))), // EMPTY_DICT
                0x64 => {
                    // DICT: build from mark-delimited k, v pairs
                    let items = self.pop_mark("DICT")?;
                    if items.len() % 2 != 0 {
                        return Err("torchpt: DICT with odd number of stack items".into());
                    }
                    let d = Rc::new(RefCell::new(Vec::new()));
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        Self::dict_insert(&d, k, v);
                    }
                    self.stack.push(Val::Dict(d));
                }
                0x73 => {
                    // SETITEM
                    let v = self.pop("SETITEM value")?;
                    let k = self.pop("SETITEM key")?;
                    match self.stack.last() {
                        Some(Val::Dict(d)) => Self::dict_insert(&d.clone(), k, v),
                        other => {
                            return Err(format!(
                                "torchpt: SETITEM on non-dict ({})",
                                other.map_or("empty stack", |v| v.kind())
                            ))
                        }
                    }
                }
                0x75 => {
                    // SETITEMS: mark-delimited k, v pairs into the dict below
                    let items = self.pop_mark("SETITEMS")?;
                    if items.len() % 2 != 0 {
                        return Err("torchpt: SETITEMS with odd number of stack items".into());
                    }
                    match self.stack.last() {
                        Some(Val::Dict(d)) => {
                            let d = d.clone();
                            let mut it = items.into_iter();
                            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                                Self::dict_insert(&d, k, v);
                            }
                        }
                        other => {
                            return Err(format!(
                                "torchpt: SETITEMS on non-dict ({})",
                                other.map_or("empty stack", |v| v.kind())
                            ))
                        }
                    }
                }
                0x28 => self.stack.push(Val::Mark), // MARK
                0x5d => self.stack.push(Val::List(Rc::new(RefCell::new(Vec::new())))), // EMPTY_LIST
                0x61 => {
                    // APPEND
                    let v = self.pop("APPEND value")?;
                    match self.stack.last() {
                        Some(Val::List(l)) => l.borrow_mut().push(v),
                        other => {
                            return Err(format!(
                                "torchpt: APPEND on non-list ({})",
                                other.map_or("empty stack", |v| v.kind())
                            ))
                        }
                    }
                }
                0x65 => {
                    // APPENDS: mark-delimited values into the list below
                    let items = self.pop_mark("APPENDS")?;
                    match self.stack.last() {
                        Some(Val::List(l)) => l.borrow_mut().extend(items),
                        other => {
                            return Err(format!(
                                "torchpt: APPENDS on non-list ({})",
                                other.map_or("empty stack", |v| v.kind())
                            ))
                        }
                    }
                }
                0x29 => self.stack.push(Val::Tuple(Rc::new(Vec::new()))), // EMPTY_TUPLE
                0x85 => {
                    // TUPLE1
                    let a = self.pop("TUPLE1")?;
                    self.stack.push(Val::Tuple(Rc::new(vec![a])));
                }
                0x86 => {
                    // TUPLE2
                    let b = self.pop("TUPLE2")?;
                    let a = self.pop("TUPLE2")?;
                    self.stack.push(Val::Tuple(Rc::new(vec![a, b])));
                }
                0x87 => {
                    // TUPLE3
                    let c = self.pop("TUPLE3")?;
                    let b = self.pop("TUPLE3")?;
                    let a = self.pop("TUPLE3")?;
                    self.stack.push(Val::Tuple(Rc::new(vec![a, b, c])));
                }
                0x74 => {
                    // TUPLE (from mark)
                    let items = self.pop_mark("TUPLE")?;
                    self.stack.push(Val::Tuple(Rc::new(items)));
                }

                // -- memo ------------------------------------------------------
                0x71 => {
                    // BINPUT <byte>
                    let k = self.u8()? as u32;
                    self.memo_put(k)?;
                }
                0x72 => {
                    // LONG_BINPUT <u32>
                    let k = self.u32le()?;
                    self.memo_put(k)?;
                }
                0x94 => {
                    // MEMOIZE: key = current memo size
                    let k = self.memo.len() as u32;
                    self.memo_put(k)?;
                }
                0x68 => {
                    // BINGET <byte>
                    let k = self.u8()? as u32;
                    self.memo_get(k)?;
                }
                0x6a => {
                    // LONG_BINGET <u32>
                    let k = self.u32le()?;
                    self.memo_get(k)?;
                }

                // -- scalars ---------------------------------------------------
                0x4e => self.stack.push(Val::None),        // NONE
                0x88 => self.stack.push(Val::Bool(true)),  // NEWTRUE
                0x89 => self.stack.push(Val::Bool(false)), // NEWFALSE
                0x4a => {
                    // BININT <i32 LE>
                    let b = self.take(4)?;
                    self.stack.push(Val::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64));
                }
                0x4b => {
                    // BININT1 <u8>
                    let v = self.u8()?;
                    self.stack.push(Val::Int(v as i64));
                }
                0x4d => {
                    // BININT2 <u16 LE>
                    let v = self.u16le()?;
                    self.stack.push(Val::Int(v as i64));
                }
                0x8a => {
                    // LONG1 <len byte> <two's-complement LE bytes>
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?;
                    if n > 8 {
                        return Err(format!("torchpt: LONG1 with {n} bytes exceeds i64"));
                    }
                    let mut buf = [0u8; 8];
                    buf[..n].copy_from_slice(raw);
                    // sign-extend
                    if n > 0 && raw[n - 1] & 0x80 != 0 {
                        for b in buf.iter_mut().skip(n) {
                            *b = 0xff;
                        }
                    }
                    self.stack.push(Val::Int(i64::from_le_bytes(buf)));
                }
                0x47 => {
                    // BINFLOAT <f64 BE>
                    let b = self.take(8)?;
                    self.stack.push(Val::Float(f64::from_be_bytes([
                        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                    ])));
                }

                // -- strings ---------------------------------------------------
                0x55 => {
                    // SHORT_BINSTRING <len byte> <bytes>
                    let n = self.u8()? as usize;
                    let s = Self::str_from(self.take(n)?);
                    self.stack.push(Val::Str(Rc::new(s)));
                }
                0x58 => {
                    // BINUNICODE <u32 len> <utf8>
                    let n = self.u32le()? as usize;
                    let raw = self.take(n)?;
                    let s = std::str::from_utf8(raw)
                        .map_err(|_| "torchpt: BINUNICODE with invalid utf8".to_string())?;
                    self.stack.push(Val::Str(Rc::new(s.to_string())));
                }
                0x8c => {
                    // SHORT_BINUNICODE <len byte> <utf8>
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?;
                    let s = std::str::from_utf8(raw)
                        .map_err(|_| "torchpt: SHORT_BINUNICODE with invalid utf8".to_string())?;
                    self.stack.push(Val::Str(Rc::new(s.to_string())));
                }

                // -- globals / object construction -----------------------------
                0x63 => {
                    // GLOBAL <module>\n<name>\n
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Val::Global(Rc::new((module, name))));
                }
                0x93 => {
                    // STACK_GLOBAL: name, module from the stack
                    let name = self.pop("STACK_GLOBAL name")?;
                    let module = self.pop("STACK_GLOBAL module")?;
                    match (module, name) {
                        (Val::Str(m), Val::Str(n)) => self
                            .stack
                            .push(Val::Global(Rc::new((m.as_ref().clone(), n.as_ref().clone())))),
                        _ => return Err("torchpt: STACK_GLOBAL with non-str operands".into()),
                    }
                }
                0x52 => {
                    // REDUCE
                    let args = self.pop("REDUCE args")?;
                    let callable = self.pop("REDUCE callable")?;
                    let v = self.call(callable, args, "REDUCE")?;
                    self.stack.push(v);
                }
                0x81 => {
                    // NEWOBJ: cls.__new__(cls, *args) — same subset as REDUCE
                    let args = self.pop("NEWOBJ args")?;
                    let cls = self.pop("NEWOBJ class")?;
                    let v = self.call(cls, args, "NEWOBJ")?;
                    self.stack.push(v);
                }
                0x62 => {
                    // BUILD: obj.__setstate__(state). For dicts the state sets
                    // *instance attributes* (e.g. a state_dict's `_metadata`
                    // version table), not mapping items — python's loader keeps
                    // them off the item view, so we drop attribute subtrees
                    // unless they contain tensors (those are merged as items so
                    // the full-coverage guarantee holds).
                    let state = self.pop("BUILD state")?;
                    let obj = self.pop("BUILD object")?;
                    match (&obj, &state) {
                        (_, Val::None) => {}
                        (Val::Dict(d), Val::Dict(s)) => {
                            let pairs: Vec<(Val, Val)> = s.borrow().clone();
                            for (k, v) in pairs {
                                if contains_tensor(&v) {
                                    Self::dict_insert(d, k, v);
                                }
                            }
                        }
                        (o, s) => {
                            return Err(format!(
                                "torchpt: BUILD on unsupported object ({} with {} state)",
                                o.kind(),
                                s.kind()
                            ))
                        }
                    }
                    self.stack.push(obj);
                }
                0x51 => {
                    // BINPERSID
                    let pid = self.pop("BINPERSID")?;
                    let v = self.persistent_load(pid)?;
                    self.stack.push(v);
                }

                other => {
                    return Err(format!(
                        "torchpt: unsupported pickle opcode 0x{other:02x} at offset {}",
                        self.pos - 1
                    ))
                }
            }
        }
    }
}

/// Does this subtree contain any tensor? Used by BUILD to decide whether a
/// dict instance attribute may be dropped (pure metadata) or must be kept.
fn contains_tensor(v: &Val) -> bool {
    match v {
        Val::Tensor(_) => true,
        Val::Dict(d) => d.borrow().iter().any(|(_, e)| contains_tensor(e)),
        Val::List(l) => l.borrow().iter().any(contains_tensor),
        Val::Tuple(t) => t.iter().any(contains_tensor),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// archive: storage decoding + tensor materialization
// ---------------------------------------------------------------------------

struct Archive<'a> {
    bytes: &'a [u8],
    /// entry name -> (offset, len)
    entries: HashMap<String, (usize, usize)>,
    /// the single top-level directory ("archive", file stem, ...; may be "")
    root: String,
    /// decoded storages by key (shared storages are decoded once)
    cache: HashMap<String, Rc<Vec<f32>>>,
}

impl<'a> Archive<'a> {
    /// Decode the storage blob `<root>/data/<key>` to f32.
    fn storage_f32(&mut self, r: &StorageRef) -> Result<Rc<Vec<f32>>, String> {
        if let Some(v) = self.cache.get(&r.key) {
            return Ok(v.clone());
        }
        let name = if self.root.is_empty() {
            format!("data/{}", r.key)
        } else {
            format!("{}/data/{}", self.root, r.key)
        };
        let &(off, len) = self
            .entries
            .get(&name)
            .ok_or_else(|| format!("torchpt: storage entry '{name}' missing from archive"))?;
        let raw = &self.bytes[off..off + len];
        let esz = r.dtype.elem_size();
        if len % esz != 0 {
            return Err(format!(
                "torchpt: storage '{name}' has {len} bytes, not a multiple of element size {esz}"
            ));
        }
        let data: Vec<f32> = match r.dtype {
            DType::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            DType::F64 => raw
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
            DType::F16 => raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            DType::BF16 => raw
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            // Integer storages exist in real checkpoints only as bookkeeping
            // (BatchNorm's `num_batches_tracked` step counter, index tensors).
            // Widening to f32 is exact below 2^24 and approximate above it —
            // the same lossy-by-design contract F64 already has here. Callers
            // that care about an int64 payload must not route it through this
            // reader; callers that don't (every importer, which drops
            // `num_batches_tracked`) are unaffected.
            DType::I64 => raw
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect(),
        };
        let rc = Rc::new(data);
        self.cache.insert(r.key.clone(), rc.clone());
        Ok(rc)
    }

    /// Materialize a (possibly non-contiguous) tensor node as contiguous f32.
    fn materialize(&mut self, name: &str, t: &TensorNode) -> Result<NamedTensor, String> {
        let storage = self.storage_f32(&t.storage)?;
        let numel: usize = t.size.iter().product();
        let mut data = Vec::with_capacity(numel);
        if numel > 0 {
            // Row-major walk over the index space, gathering via strides.
            let mut idx = vec![0usize; t.size.len()];
            'gather: loop {
                let off =
                    t.offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
                let v = *storage.get(off).ok_or_else(|| {
                    format!(
                        "torchpt: tensor '{name}' indexes element {off} past end of storage \
                         '{}' ({} elements)",
                        t.storage.key,
                        storage.len()
                    )
                })?;
                data.push(v);
                // odometer increment (last dim fastest); empty idx = 0-dim scalar
                let mut d = t.size.len();
                loop {
                    if d == 0 {
                        break 'gather;
                    }
                    d -= 1;
                    idx[d] += 1;
                    if idx[d] < t.size[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }
        Ok(NamedTensor { name: name.to_string(), shape: t.size.clone(), data })
    }
}

/// Render a dict key as a name component. State_dict keys are strings; int
/// keys occur in optimizer states and are rendered in decimal.
fn key_string(k: &Val) -> Result<String, String> {
    match k {
        Val::Str(s) => Ok(s.as_ref().clone()),
        Val::Int(i) => Ok(i.to_string()),
        other => Err(format!("torchpt: unsupported dict key type {}", other.kind())),
    }
}

fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// Walk the unpickled tree, flattening dict keys with '.' joins. Tensors are
/// materialized; every other leaf increments `skipped`.
fn flatten(
    v: &Val,
    prefix: &str,
    arch: &mut Archive,
    out: &mut Vec<NamedTensor>,
    skipped: &mut usize,
) -> Result<(), String> {
    match v {
        Val::Dict(d) => {
            let pairs: Vec<(Val, Val)> = d.borrow().clone();
            for (k, val) in &pairs {
                let name = join(prefix, &key_string(k)?);
                flatten(val, &name, arch, out, skipped)?;
            }
        }
        Val::List(l) => {
            let items: Vec<Val> = l.borrow().clone();
            for (i, e) in items.iter().enumerate() {
                flatten(e, &join(prefix, &i.to_string()), arch, out, skipped)?;
            }
        }
        Val::Tuple(t) => {
            for (i, e) in t.iter().enumerate() {
                flatten(e, &join(prefix, &i.to_string()), arch, out, skipped)?;
            }
        }
        Val::Tensor(t) => {
            let name = if prefix.is_empty() { "tensor" } else { prefix };
            out.push(arch.materialize(name, t)?);
        }
        Val::Mark => return Err("torchpt: internal error: MARK escaped the pickle stack".into()),
        // Non-tensor leaves (None, bool, int, float, str, stray globals or
        // bare storages) are skipped but counted.
        _ => *skipped += 1,
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

/// Parse a `torch.save` zip container from an in-memory byte buffer.
pub fn parse(bytes: &[u8]) -> Result<ReadReport, String> {
    let entries = zipread::parse(bytes)?;
    // Locate <root>/data.pkl; the root is the archive's single top-level dir.
    let mut pkl: Option<(String, usize, usize)> = None;
    for e in &entries {
        let root = if e.name == "data.pkl" {
            Some(String::new())
        } else {
            e.name.strip_suffix("/data.pkl").map(|r| r.to_string())
        };
        if let Some(root) = root {
            if pkl.is_some() {
                return Err("torchpt: multiple data.pkl entries in archive".into());
            }
            pkl = Some((root, e.offset, e.len));
        }
    }
    let (root, off, len) =
        pkl.ok_or("torchpt: no data.pkl entry — not a torch >= 1.6 zip checkpoint")?;

    let mut u = Unpickler { b: &bytes[off..off + len], pos: 0, stack: Vec::new(), memo: HashMap::new() };
    let tree = u.run()?;

    let mut arch = Archive {
        bytes,
        entries: entries.into_iter().map(|e| (e.name, (e.offset, e.len))).collect(),
        root,
        cache: HashMap::new(),
    };
    let mut tensors = Vec::new();
    let mut skipped = 0usize;
    flatten(&tree, "", &mut arch, &mut tensors, &mut skipped)?;
    Ok(ReadReport { tensors, skipped_non_tensor: skipped })
}

/// Read a `.pt` checkpoint from disk, returning every tensor plus the count
/// of skipped non-tensor leaves.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_report(path: &str) -> Result<ReadReport, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    parse(&bytes)
}

/// Read a `.pt` checkpoint from disk (tensors-only convenience).
#[cfg(not(target_arch = "wasm32"))]
pub fn read(path: &str) -> Result<Vec<NamedTensor>, String> {
    read_report(path).map(|r| r.tensors)
}
