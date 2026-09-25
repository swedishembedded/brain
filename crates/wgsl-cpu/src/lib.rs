// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compile brain's WGSL compute kernels to native CPU code.
//!
//! Swedish Embedded AB implements compiler and code-generation work of this
//! kind - shader or DSL front-end to native machine code, JIT or ahead of
//! time - for teams that need one source of truth to execute on targets it was
//! never written for. If your team needs expertise in IR translation, JIT
//! compilation or retargeting an existing kernel language, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! WGSL stays the single source of truth: we parse each kernel with the same
//! `naga` front-end the wgpu/vulkan paths use, then translate the resulting
//! naga IR straight to Cranelift IR and JIT it to a native function. The whole
//! kernel set is barrier-free / atomic-free and every invocation computes one
//! output element from a linear `global_invocation_id`, so a compiled kernel is
//! just a function that loops over a contiguous range of invocation ids:
//!
//! ```text
//! extern "C" fn(start, end, grid_x, grid_y, uniform: *const u32, bufs: *const *mut u8)
//! ```
//!
//! The CPU backend (in `gpu-core`) hands each rayon worker a disjoint
//! `[start, end)` sub-range of the `grid_x*grid_y*64` invocations; because each
//! invocation owns a disjoint output element there is no cross-invocation
//! synchronisation to model.
//!
//! The handled IR subset: f32/u32/i32/bool scalars and, by scalarization, the
//! vectors, matrices, structs and fixed-size arrays built from them - as
//! values, `var`s (register-backed, or stack-backed when they contain an
//! array), and places in the uniform, storage and work-group buffers laid out
//! per WGSL's host-shareable rules ([`shape`], [`aggregate`]); user functions,
//! inlined at their call sites ([`inline`]); the builtin input vectors;
//! `if`/`loop`/`break`/`continue`/`return`; and a closed set of math, vector
//! and matrix builtins. Anything outside that subset is a hard error at
//! compile time rather than silently miscompiled.

mod aggregate;
mod inline;
mod shape;

use cranelift_codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types, AbiParam, BlockArg, FuncRef, InstBuilder, MemFlags, Signature, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{Linkage, Module};
use cranelift_jit::{JITBuilder, JITModule};
use std::collections::{HashMap, HashSet};

use naga::{
    AddressSpace, BinaryOperator, Block, BuiltIn, Expression, Handle, Literal, MathFunction,
    Scalar, ScalarKind, Statement, TypeInner, UnaryOperator,
};

use inline::Frame;
use shape::Shape;

/// The ABI of every compiled kernel.
pub type KernelFn =
    unsafe extern "C" fn(u64, u64, u32, u32, *const u32, *const *mut u8);

/// A set of compiled kernels, indexed in the order they were passed to [`Jit::new`].
///
/// Owns the backing `JITModule` (keeps the executable code alive) and the raw
/// finalized function pointers. The compiled code is read-only after
/// construction, so calling the kernels from many threads concurrently is sound.
pub struct Jit {
    _module: JITModule,
    /// One entry per registered kernel (index-stable). `None` for a kernel the
    /// CPU JIT declined to compile — e.g. a multi-barrier tiled GEMM the
    /// work-group model can't express. Such a kernel must run via a native fast
    /// path (see `backend-cpu`) or the GPU backend; dispatching it through the JIT
    /// panics with a clear message rather than miscompiling.
    funcs: Vec<Option<*const u8>>,
    names: Vec<String>,
    /// Per-kernel work-group size: `None` for the ordinary one-output-per-
    /// invocation kernels, `Some(wgsize)` for kernels that use workgroup memory /
    /// `workgroupBarrier()` (compiled with the work-group execution model). The
    /// CPU dispatcher must hand the latter ranges aligned to `wgsize`.
    wg_size: Vec<Option<u32>>,
}

/// The compile error that means "this backend's work-group execution model
/// cannot express this kernel", as opposed to "this kernel or this compiler is
/// broken": more than one top-level barrier (`compile_one_wg`'s
/// split-at-barrier model). Only it is skippable; everything else is a hard
/// error, so a genuine port bug can never masquerade as a skipped kernel.
///
/// It is a structural fact a kernel DECLARES in its header (`@cpu no` /
/// `native-only`) and that `scripts/build/kernelmeta.py::cpu` derives from the
/// same code the compiler reads, so the two can never disagree silently. No
/// other error message may therefore contain the word "barrier".
fn is_unsupported_workgroup_structure(e: &str) -> bool {
    e.to_lowercase().contains("barrier")
}

// The JIT-compiled code is immutable after `new` returns; only `&Jit` is shared.
unsafe impl Send for Jit {}
unsafe impl Sync for Jit {}

impl Jit {
    /// Parse and JIT-compile every `(name, wgsl_src)` kernel. Returns an error
    /// string identifying the offending kernel on the first failure.
    pub fn new(kernels: &[(&str, &str)]) -> Result<Jit, String> {
        // Host-side construction cost, not GPU-kernel-dispatch time -- gated on
        // the same `BRAIN_PROFILE` convention `deepseek2ocr::stage_time` and
        // `backend-cpu`'s own per-kernel table already use, so a load-time
        // profile shows this bracket beside the streaming-upload one it
        // competes with for the same 20+ seconds. See
        // `deepseek-ocr`'s model-construction investigation for why this
        // needed a real number instead of a guess.
        let t0 = std::time::Instant::now();
        let profile = std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false);
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").unwrap();
        // We synthesise our own bounds via the kernels' early-return mask, so the
        // generated loads/stores are trusted; no need for spectre mitigations.
        flags.set("enable_verifier", "true").unwrap();
        let isa = cranelift_native::builder()
            .map_err(|e| format!("host ISA unavailable: {e}"))?
            .finish(settings::Flags::new(flags))
            .map_err(|e| format!("ISA finish failed: {e}"))?;

        let mut builder =
            JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        for (name, ptr) in math_symbols() {
            builder.symbol(name, ptr);
        }
        let mut module = JITModule::new(builder);

        // Declare the imported math intrinsics once; reused across kernels.
        let math = MathRefs::declare(&mut module)?;

        let mut ctx = module.make_context();
        let mut fctx = FunctionBuilderContext::new();
        let mut ids = Vec::with_capacity(kernels.len());
        let mut wg_size = Vec::with_capacity(kernels.len());

        for (name, src) in kernels {
            ctx.func.signature = kernel_signature(module.target_config().pointer_type());
            match compile_one(src, &mut module, &math, &mut ctx, &mut fctx) {
                Ok(wg) => {
                    wg_size.push(wg);
                    let id = module
                        .declare_function(name, Linkage::Export, &ctx.func.signature)
                        .map_err(|e| format!("declare {name:?}: {e}"))?;
                    module
                        .define_function(id, &mut ctx)
                        .map_err(|e| format!("define {name:?}: {e:?}"))?;
                    module.clear_context(&mut ctx);
                    ids.push(Some(id));
                }
                // A work-group kernel the CPU model can't express: a tiled GEMM
                // with a barrier inside the K-loop. Skip it - such a kernel
                // runs via a native fast path on CPU or on the GPU backend,
                // and every one of them declares `@cpu no`/`native-only`
                // (cross-checked by `scripts/build/kernelmeta.py::cpu`, which
                // derives that cell from the same structural fact). Genuine
                // compile errors still fail hard, so a real port bug cannot
                // hide as a skip.
                Err(e) if is_unsupported_workgroup_structure(&e) => {
                    eprintln!("wgsl-cpu: kernel {name:?} not JIT-compiled ({e}); must use a native fast path or the GPU");
                    wg_size.push(None);
                    ids.push(None);
                    module.clear_context(&mut ctx);
                }
                Err(e) => return Err(format!("kernel {name:?}: {e}")),
            }
        }

        module
            .finalize_definitions()
            .map_err(|e| format!("finalize: {e}"))?;

        let funcs = ids.iter().map(|id| id.map(|id| module.get_finalized_function(id))).collect();
        if profile {
            eprintln!("wgsl-cpu: Jit::new compiled {} kernels in {:.1} ms", kernels.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Jit {
            _module: module,
            funcs,
            names: kernels.iter().map(|(n, _)| n.to_string()).collect(),
            wg_size,
        })
    }

    /// Index of `name`, or `None` if no such kernel was compiled.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Work-group size of kernel `kind` if it is a work-group kernel (uses
    /// workgroup memory / barriers), else `None`. The CPU dispatcher uses this to
    /// hand work-group kernels ranges aligned to whole workgroups.
    pub fn workgroup_size(&self, kind: usize) -> Option<u32> {
        self.wg_size.get(kind).copied().flatten()
    }

    /// Run kernel `kind` over invocation ids `[start, end)`.
    ///
    /// # Safety
    /// `uniform` must point to the packed uniform u32 stream the kernel expects,
    /// and `bufs` to an array of base pointers (one per storage binding, in
    /// binding order) each at least as large as the kernel addresses. Concurrent
    /// calls must target disjoint output regions (the one-output-per-invocation
    /// invariant the dispatcher upholds).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &self,
        kind: usize,
        start: u64,
        end: u64,
        grid_x: u32,
        grid_y: u32,
        uniform: *const u32,
        bufs: *const *mut u8,
    ) {
        let ptr = self.funcs[kind].unwrap_or_else(|| {
            panic!(
                "wgsl-cpu: kernel {:?} was not JIT-compiled (unsupported work-group structure); \
                 dispatch it via the native fast path or the GPU backend",
                self.names.get(kind).map(String::as_str).unwrap_or("?")
            )
        });
        let f: KernelFn = std::mem::transmute(ptr);
        f(start, end, grid_x, grid_y, uniform, bufs);
    }
}

/// `(start, end, grid_x, grid_y, uniform_ptr, bufs_ptr)`.
fn kernel_signature(ptr: types::Type) -> Signature {
    let mut sig = Signature::new(cranelift_codegen::isa::CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64)); // start
    sig.params.push(AbiParam::new(types::I64)); // end
    sig.params.push(AbiParam::new(types::I32)); // grid_x
    sig.params.push(AbiParam::new(types::I32)); // grid_y
    sig.params.push(AbiParam::new(ptr)); // uniform
    sig.params.push(AbiParam::new(ptr)); // bufs
    sig
}

// ---------------------------------------------------------------------------
// Math intrinsics (libm-style f32 calls Cranelift can't lower natively).
// ---------------------------------------------------------------------------

extern "C" fn w_expf(x: f32) -> f32 { x.exp() }
extern "C" fn w_logf(x: f32) -> f32 { x.ln() }
extern "C" fn w_sinf(x: f32) -> f32 { x.sin() }
extern "C" fn w_cosf(x: f32) -> f32 { x.cos() }
extern "C" fn w_tanhf(x: f32) -> f32 { x.tanh() }
extern "C" fn w_powf(x: f32, y: f32) -> f32 { x.powf(y) }
extern "C" fn w_atan2f(y: f32, x: f32) -> f32 { y.atan2(x) }

fn math_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("brain_expf", w_expf as *const u8),
        ("brain_logf", w_logf as *const u8),
        ("brain_sinf", w_sinf as *const u8),
        ("brain_cosf", w_cosf as *const u8),
        ("brain_tanhf", w_tanhf as *const u8),
        ("brain_powf", w_powf as *const u8),
        ("brain_atan2f", w_atan2f as *const u8),
    ]
}

struct MathRefs {
    // FuncIds of imported intrinsics; turned into FuncRefs per function.
    unary: HashMap<&'static str, cranelift_module::FuncId>,
    powf: cranelift_module::FuncId,
    atan2: cranelift_module::FuncId,
}

/// The imported intrinsics as references usable from one function.
struct FnRefs {
    unary: HashMap<&'static str, FuncRef>,
    powf: FuncRef,
    atan2: FuncRef,
}

impl MathRefs {
    fn declare(module: &mut JITModule) -> Result<MathRefs, String> {
        let mut unary_sig = module.make_signature();
        unary_sig.params.push(AbiParam::new(types::F32));
        unary_sig.returns.push(AbiParam::new(types::F32));
        let mut unary = HashMap::new();
        for name in ["brain_expf", "brain_logf", "brain_sinf", "brain_cosf", "brain_tanhf"] {
            let id = module
                .declare_function(name, Linkage::Import, &unary_sig)
                .map_err(|e| format!("declare {name}: {e}"))?;
            unary.insert(name, id);
        }
        let mut bin_sig = module.make_signature();
        bin_sig.params.push(AbiParam::new(types::F32));
        bin_sig.params.push(AbiParam::new(types::F32));
        bin_sig.returns.push(AbiParam::new(types::F32));
        let powf = module
            .declare_function("brain_powf", Linkage::Import, &bin_sig)
            .map_err(|e| format!("declare brain_powf: {e}"))?;
        let atan2 = module
            .declare_function("brain_atan2f", Linkage::Import, &bin_sig)
            .map_err(|e| format!("declare brain_atan2f: {e}"))?;
        Ok(MathRefs { unary, powf, atan2 })
    }

    fn import(&self, module: &mut JITModule, func: &mut cranelift_codegen::ir::Function) -> FnRefs {
        FnRefs {
            unary: self.unary.iter().map(|(k, id)| (*k, module.declare_func_in_func(*id, func))).collect(),
            powf: module.declare_func_in_func(self.powf, func),
            atan2: module.declare_func_in_func(self.atan2, func),
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar type tags carried alongside every Cranelift value.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    F32,
    U32,
    I32,
    Bool,
}

impl Ty {
    /// Takes the WHOLE `naga::Scalar` (kind AND width), not just `ScalarKind`,
    /// because `ScalarKind` alone cannot distinguish f16 (width 2) from f32
    /// (width 4) - both are `ScalarKind::Float`. Matching on `(kind, width)`
    /// together is the fix for a real silent-miscompile bug this JIT had: an
    /// `enable f16;` kernel's `f16` arithmetic used to compile and run as
    /// plain fp32 with no error, so `60000.0 * 1.0 + 6000.0` came back as the
    /// un-saturated fp32 sum `66000.0` instead of the `+inf` a real f16 ALU
    /// must saturate to past its 65504 max. f16 is refused outright rather
    /// than silently executed wrong.
    fn from_scalar(scalar: Scalar) -> Result<Ty, String> {
        Ok(match (scalar.kind, scalar.width) {
            (ScalarKind::Float, 4) => Ty::F32,
            (ScalarKind::Float, 2) => {
                return Err(
                    "f16 (`enable f16;`, width-2 float) has no native execution path on this \
                     CPU JIT and is refused rather than silently executed as fp32: a real f16 \
                     ALU saturates `60000.0 * 1.0 + 6000.0` to +inf (past f16's 65504 max), but \
                     running that arithmetic in fp32 registers instead silently returns the \
                     un-saturated 66000.0 - a wrong answer with no error, not a rounding \
                     difference"
                        .to_string(),
                )
            }
            (ScalarKind::Uint, 4) => Ty::U32,
            (ScalarKind::Sint, 4) => Ty::I32,
            (ScalarKind::Bool, _) => Ty::Bool,
            _ => return Err(format!("unsupported scalar {scalar:?}")),
        })
    }
    fn is_float(self) -> bool {
        matches!(self, Ty::F32)
    }
}

/// The result of evaluating a naga expression: a materialised scalar, an
/// aggregate value, or a "place" (an addressable location) that a
/// `Load`/`Store` resolves.
#[derive(Clone, Copy)]
enum Eval {
    Scalar(Value, Ty),
    /// An aggregate value: its flattened scalars are `Tr::aggs[id]`.
    Agg(u32, Shape),
    Place(Place),
}

#[derive(Clone, Copy)]
enum Place {
    /// A mutable scalar function-local (`var`) backed by a Cranelift SSA variable.
    Local(Variable, Ty),
    /// A scalar in memory at `addr`; `readonly` for the uniform buffer.
    Mem { addr: Value, elem: Ty, readonly: bool },
    /// An aggregate in memory at `addr + off`, in WGSL's host-shareable layout.
    Ptr { addr: Value, off: u32, shape: Shape, readonly: bool },
    /// Scalars `[start, start + flat_len(shape))` of variable group `group`
    /// (a register-backed aggregate local).
    Vars { group: u32, start: u32, shape: Shape },
    /// A dynamically indexed part of variable group `group`: candidate `j`
    /// of `n` starts at scalar `lo + j * step`, and `idx` picks one.
    VarsDyn { group: u32, lo: u32, step: u32, n: u32, idx: Value, shape: Shape },
}

/// Backing storage for a naga function-local variable.
#[derive(Clone, Copy)]
enum LocalSlot {
    /// A scalar `var`: a Cranelift SSA variable.
    Scalar(Variable, Ty),
    /// A vector, matrix or struct `var` without arrays: one Cranelift
    /// variable per scalar, in variable group `group`.
    Vars { group: u32, shape: Shape },
    /// A `var` containing an array: a stack slot at `base`, computed where
    /// the local is declared so it dominates every use.
    Stack { base: Value, shape: Shape },
}

/// Load/store flags: the uniform buffer is immutable for a whole dispatch, so
/// its loads may be merged, hoisted or dropped when unused.
fn mem_flags(readonly: bool) -> MemFlags {
    if readonly {
        MemFlags::trusted().with_readonly().with_can_move()
    } else {
        MemFlags::trusted()
    }
}

/// `(element scalar type, element count)` of a `var<workgroup>` array.
fn workgroup_array_info(m: &naga::Module, ty: Handle<naga::Type>) -> Result<(Ty, u32), String> {
    match &m.types[ty].inner {
        TypeInner::Array { base, size, .. } => {
            let elem = match &m.types[*base].inner {
                TypeInner::Scalar(s) => Ty::from_scalar(*s)?,
                other => return Err(format!("workgroup array of non-scalar {other:?}")),
            };
            let count = match size {
                naga::ArraySize::Constant(n) => n.get(),
                other => return Err(format!("non-constant workgroup array size {other:?}")),
            };
            Ok((elem, count))
        }
        other => Err(format!("expected a workgroup array, got {other:?}")),
    }
}

// ---------------------------------------------------------------------------
// Per-kernel translation.
// ---------------------------------------------------------------------------

fn compile_one(
    src: &str,
    module: &mut JITModule,
    math: &MathRefs,
    ctx: &mut cranelift_codegen::Context,
    fctx: &mut FunctionBuilderContext,
) -> Result<Option<u32>, String> {
    let nmod = naga::front::wgsl::parse_str(src).map_err(|e| format!("WGSL parse: {e:?}"))?;
    let entry = nmod
        .entry_points
        .iter()
        .find(|e| e.name == "main")
        .ok_or("no `main` entry point")?;
    let func = &entry.function;

    // Work-group kernels (workgroup memory / `workgroupBarrier()`) use a separate
    // execution model; everything else is the one-output-per-invocation path.
    if is_workgroup_kernel(&nmod, func) {
        let wgsize = entry.workgroup_size[0];
        compile_one_wg(&nmod, entry, module, math, ctx, fctx)?;
        return Ok(Some(wgsize));
    }

    let ptr_ty = module.target_config().pointer_type();
    let mut builder = FunctionBuilder::new(&mut ctx.func, fctx);
    let fns = math.import(module, builder.func);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    let p = builder.block_params(entry_block).to_vec();
    let (start, end, gx, gy, uniform_ptr, bufs_ptr) =
        (p[0], p[1], p[2], p[3], p[4], p[5]);

    // Resolve which function argument carries which builtin vec3.
    let ba = builtin_args(func);
    let gid_arg = ba.gid.ok_or("kernel missing global_invocation_id arg")?;
    let nwg_arg = ba.nwg.ok_or("kernel missing num_workgroups arg")?;

    // Base pointer of each storage binding (binding 1.. -> bufs[binding-1]).
    let mut buf_base: HashMap<u32, Value> = HashMap::new();
    for (_h, gv) in nmod.global_variables.iter() {
        let binding = match &gv.binding {
            Some(b) => b.binding,
            None => continue,
        };
        match gv.space {
            AddressSpace::Uniform => {}
            AddressSpace::Storage { .. } => {
                let slot = (binding - 1) as i64;
                let addr = builder.ins().iadd_imm(bufs_ptr, slot * 8);
                let base = builder.ins().load(ptr_ty, MemFlags::trusted(), addr, 0);
                buf_base.insert(binding, base);
            }
            other => return Err(format!("unsupported address space {other:?}")),
        }
    }

    // Per-invocation loop: header(gidx) { if gidx>=end -> exit else body }.
    let header = builder.create_block();
    builder.append_block_param(header, types::I64);
    let body = builder.create_block();
    let latch = builder.create_block();
    let exit = builder.create_block();
    builder.ins().jump(header, &[BlockArg::from(start)]);

    builder.switch_to_block(header);
    let gidx = builder.block_params(header)[0];
    let done = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, gidx, end);
    builder.ins().brif(done, exit, &[], body, &[]);

    builder.switch_to_block(body);
    // Synthesise the builtin components so the shader's own index math reproduces
    // our linear `gidx`: nwg=(gx,gy,1); gid=(gidx % (gx*64), gidx / (gx*64), 0).
    let gx64 = builder.ins().imul_imm(gx, 64);
    let gx64_64 = builder.ins().uextend(types::I64, gx64);
    let gid_x64 = builder.ins().urem(gidx, gx64_64);
    let gid_y64 = builder.ins().udiv(gidx, gx64_64);
    let gid_x = builder.ins().ireduce(types::I32, gid_x64);
    let gid_y = builder.ins().ireduce(types::I32, gid_y64);
    let zero32 = builder.ins().iconst(types::I32, 0);
    let one32 = builder.ins().iconst(types::I32, 1);
    let gid = [gid_x, gid_y, zero32];
    let nwg = [gx, gy, one32];

    let empty_wg = HashMap::new();
    let env = Env {
        ptr_ty,
        buf_base: &buf_base,
        wg_mem: &empty_wg,
        uniform_ptr,
        gid,
        nwg,
        local_id: [zero32, zero32, zero32],
        wgid: [zero32, zero32, zero32],
        ba: BuiltinArgs { gid: Some(gid_arg), nwg: Some(nwg_arg), local_id: None, wgid: None },
        fns: &fns,
        latch,
        active_addr: None,
    };
    let mut tr = Tr::new(&nmod, &mut builder, func, env);
    tr.declare_locals()?;

    let fell_through = tr.block(&func.body)?;
    if fell_through {
        tr.b.ins().jump(latch, &[]);
    }

    builder.switch_to_block(latch);
    let next = builder.ins().iadd_imm(gidx, 1);
    builder.ins().jump(header, &[BlockArg::from(next)]);

    builder.switch_to_block(exit);
    builder.ins().return_(&[]);

    builder.seal_all_blocks();
    builder.finalize();
    Ok(None)
}

/// A kernel is a "work-group kernel" if it declares any `var<workgroup>` global
/// or contains a `workgroupBarrier()` — it then needs the cooperative execution
/// model (shared memory + barrier-split segments) rather than the independent
/// one-output-per-invocation model.
fn is_workgroup_kernel(m: &naga::Module, func: &naga::Function) -> bool {
    let has_wg_global = m
        .global_variables
        .iter()
        .any(|(_, gv)| matches!(gv.space, AddressSpace::WorkGroup));
    has_wg_global || block_has_barrier(&func.body)
}

/// Whether a statement block contains a barrier (searched recursively). A
/// `workgroupUniformLoad` is one: every invocation waits at it for the load.
fn block_has_barrier(block: &Block) -> bool {
    block.iter().any(|s| match s {
        Statement::ControlBarrier(_) | Statement::WorkGroupUniformLoad { .. } => true,
        Statement::Block(b) => block_has_barrier(b),
        Statement::If { accept, reject, .. } => block_has_barrier(accept) || block_has_barrier(reject),
        Statement::Loop { body, continuing, .. } => block_has_barrier(body) || block_has_barrier(continuing),
        _ => false,
    })
}

/// Split a work-group kernel body at its single top-level `workgroupBarrier()`
/// into the cooperative segment before it and the segment after. Restricted to
/// exactly one top-level barrier (no nesting) — sufficient for the tiled kernels.
fn split_at_barrier(body: &Block) -> Result<(Block, Block), String> {
    let stmts: Vec<&Statement> = body.iter().collect();
    let bpos = stmts
        .iter()
        .position(|s| matches!(s, Statement::ControlBarrier(_)))
        .ok_or("work-group kernel without a top-level barrier (unsupported structure)")?;
    let before: Vec<Statement> = stmts[..bpos].iter().map(|s| (*s).clone()).collect();
    let after: Vec<Statement> = stmts[bpos + 1..].iter().map(|s| (*s).clone()).collect();
    if before.iter().any(|s| block_has_barrier(&Block::from_vec(vec![s.clone()])))
        || after.iter().any(|s| block_has_barrier(&Block::from_vec(vec![s.clone()])))
    {
        return Err("only a single top-level workgroupBarrier() is supported".into());
    }
    Ok((Block::from_vec(before), Block::from_vec(after)))
}

/// Collect every per-invocation `LocalVariable` this block either stores into
/// (a `Store` whose pointer is that local or a component, member or element of
/// it) or loads from (a `Load` of the same, appearing anywhere within one of
/// the block's `Emit` ranges, at any nesting depth). Recurses into
/// `If`/`Loop`/`Block` bodies so a store or load guarded by a conditional is
/// still seen.
fn locals_touched(
    block: &Block,
    func: &naga::Function,
    touched: &mut HashSet<Handle<naga::LocalVariable>>,
) {
    for s in block.iter() {
        match s {
            Statement::Store { pointer, .. } => {
                touched.extend(root_local(func, *pointer));
            }
            Statement::Emit(range) => {
                for eh in range.clone() {
                    if let Expression::Load { pointer } = func.expressions[eh] {
                        touched.extend(root_local(func, pointer));
                    }
                }
            }
            Statement::Block(b) => locals_touched(b, func, touched),
            Statement::If { accept, reject, .. } => {
                locals_touched(accept, func, touched);
                locals_touched(reject, func, touched);
            }
            Statement::Loop { body, continuing, .. } => {
                locals_touched(body, func, touched);
                locals_touched(continuing, func, touched);
            }
            _ => {}
        }
    }
}

/// The local variable a pointer expression points into, through any chain of
/// `Access`/`AccessIndex`.
fn root_local(func: &naga::Function, mut pointer: Handle<Expression>) -> Option<Handle<naga::LocalVariable>> {
    loop {
        match func.expressions[pointer] {
            Expression::LocalVariable(h) => return Some(h),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => pointer = base,
            _ => return None,
        }
    }
}

/// Compile a work-group kernel: process whole workgroups, each running its
/// `wgsize` invocations cooperatively over per-workgroup scratch (`var<workgroup>`)
/// with the body split at the barrier into two per-invocation segment loops.
///
/// Layout of the generated function (`start`/`end` are workgroup-aligned
/// invocation ids supplied by the dispatcher):
/// ```text
/// for wg in start/wgsize .. end/wgsize:
///     for lid in 0..wgsize: <segment before barrier>   // cooperative load etc.
///     for lid in 0..wgsize: <segment after  barrier>   // per-invocation compute
/// ```
/// `var<workgroup>` arrays are a stack slot (allocated once, reused per wg);
/// per-invocation `var` locals are SSA values re-initialised each lid iteration,
/// so none may cross the barrier (the tiled kernels are written that way).
fn compile_one_wg(
    nmod: &naga::Module,
    entry: &naga::EntryPoint,
    module: &mut JITModule,
    math: &MathRefs,
    ctx: &mut cranelift_codegen::Context,
    fctx: &mut FunctionBuilderContext,
) -> Result<(), String> {
    let func = &entry.function;
    let ptr_ty = module.target_config().pointer_type();
    let wgsize = entry.workgroup_size[0] as i64;
    if wgsize == 0 {
        return Err("work-group kernel with zero workgroup_size".into());
    }

    // Validate the barrier structure BEFORE building any Cranelift IR. A tiled
    // GEMM (multiple/nested barriers) errors here — and if that error escaped
    // after a `FunctionBuilder` had started, the builder would drop without
    // `finalize()`, leaving the shared `FunctionBuilderContext` corrupt so the
    // NEXT kernel's compile fails with a bogus block-param mismatch (this bit
    // conv_epilogue when it followed matmul_reg2 in a model's kernel list).
    let (seg_before, seg_after) = split_at_barrier(&func.body)?;

    // Per-invocation `var` locals are SSA values re-initialised each `lid`
    // iteration by `emit_invocation_loop`, called once per segment with its own
    // fresh set of Cranelift variables — a local written in `seg_before` and
    // read in `seg_after` would silently read back its zero-initialised value,
    // the same class of bug `carried`'s ordering above exists to prevent for
    // pre-barrier expressions. No kernel in this tree does this today (checked
    // below, not assumed), so catch it before it can miscompile one silently.
    {
        let mut before_locals = HashSet::new();
        locals_touched(&seg_before, func, &mut before_locals);
        let mut after_locals = HashSet::new();
        locals_touched(&seg_after, func, &mut after_locals);
        if let Some(h) = before_locals.intersection(&after_locals).next() {
            let name = func.local_variables[*h].name.clone().unwrap_or_default();
            // Deliberately worded without the substring "barrier": `Jit::new`
            // treats any error containing it as an expected barrier-STRUCTURAL
            // limitation (a shape the JIT's one-barrier model can't express at
            // all, e.g. a tiled GEMM with a barrier inside the K-loop) and
            // falls back to a native/GPU path rather than failing the build.
            // This is a different, worse kind of error — the barrier structure
            // is fine, but a value would silently read as stale/zero across
            // the workgroup synchronisation point - so it must hard-fail, not
            // be swallowed into that fallback.
            return Err(format!(
                "local `{name}` is live across the workgroup synchronisation point; the JIT \
                 re-initialises per-invocation locals for each segment, so a value stored before \
                 it and read after would silently read back zero"
            ));
        }
    }

    let mut builder = FunctionBuilder::new(&mut ctx.func, fctx);
    let fns = math.import(module, builder.func);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    let p = builder.block_params(entry_block).to_vec();
    let (start, end, gx, gy, uniform_ptr, bufs_ptr) = (p[0], p[1], p[2], p[3], p[4], p[5]);

    let ba = builtin_args(func);

    // Storage base pointers + workgroup-memory stack slots.
    use cranelift_codegen::ir::{StackSlotData, StackSlotKind};
    let mut buf_base: HashMap<u32, Value> = HashMap::new();
    let mut wg_mem: HashMap<Handle<naga::GlobalVariable>, Value> = HashMap::new();
    // Every `var<workgroup>` array, with the element count that has to be
    // re-zeroed at the start of EVERY work-group. The stack slot is allocated
    // once and reused for all of them, so without this a slot no invocation
    // writes on this pass reads back the previous work-group's value - and
    // WGSL guarantees it reads zero. A reduction whose tail lanes are inactive
    // folds that stale value into its sum and reports a plausible wrong
    // number, which is why this is not left to "the kernels always write it".
    let mut wg_fill: Vec<(Value, Ty, i64)> = Vec::new();
    for (h, gv) in nmod.global_variables.iter() {
        match gv.space {
            AddressSpace::Uniform => {}
            AddressSpace::Storage { .. } => {
                let binding = gv.binding.as_ref().map(|b| b.binding).ok_or("storage without binding")?;
                let slot = (binding - 1) as i64;
                let addr = builder.ins().iadd_imm(bufs_ptr, slot * 8);
                let base = builder.ins().load(ptr_ty, MemFlags::trusted(), addr, 0);
                buf_base.insert(binding, base);
            }
            AddressSpace::WorkGroup => {
                let (elem, count) = workgroup_array_info(nmod, gv.ty)?;
                let slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    count * 4,
                    2,
                ));
                let base = builder.ins().stack_addr(ptr_ty, slot, 0);
                wg_mem.insert(h, base);
                wg_fill.push((base, elem, count as i64));
            }
            other => return Err(format!("unsupported address space {other:?}")),
        }
    }

    // Expressions materialised (Emit'd) before the barrier but used after it must
    // be re-materialised at the TOP of the AFTER-segment's invocation body —
    // otherwise the after-segment lazily evaluates them on first use (often deep
    // in a nested loop) and reuses them in a shallower block, breaking SSA
    // dominance. This must NOT also run before the BEFORE-segment's own
    // translation: `carried` sweeps every top-level `Emit` in `seg_before`,
    // including ones that syntactically follow a `Store` to a local variable
    // earlier in that same segment (e.g. `var a = x[row]; partial[t] = a;` emits
    // `Load(a)` as part of the second statement's top-level Emit). Pre-evaluating
    // that Emit before `seg_before` has even run its own `Store` reads the
    // variable's zero-initialised value and caches it, so the segment's real,
    // correctly-ordered evaluation later finds the (wrong) value already cached
    // and never re-evaluates it — silently dropping the local's assignment. Only
    // `seg_after`'s Tr instance needs the pre-materialised values (it starts with
    // an empty cache and no other way to see seg_before's results); `seg_before`
    // always sees its own expressions translated in true program order, so it
    // must not be pre-fed a stale carried list.
    let mut carried: Vec<Handle<Expression>> = Vec::new();
    for s in seg_before.iter() {
        if let Statement::Emit(range) = s {
            for h in range.clone() {
                carried.push(h);
            }
        }
    }
    let no_carried: Vec<Handle<Expression>> = Vec::new();

    // Workgroup loop bounds: [start/wgsize, end/wgsize).
    let wgsz = builder.ins().iconst(types::I64, wgsize);
    let wg_lo = builder.ins().udiv(start, wgsz);
    let wg_hi = builder.ins().udiv(end, wgsz);

    let wg_header = builder.create_block();
    builder.append_block_param(wg_header, types::I64);
    let wg_body = builder.create_block();
    let wg_latch = builder.create_block();
    let exit = builder.create_block();
    builder.ins().jump(wg_header, &[BlockArg::from(wg_lo)]);

    builder.switch_to_block(wg_header);
    let wg = builder.block_params(wg_header)[0];
    let wg_done = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, wg, wg_hi);
    builder.ins().brif(wg_done, exit, &[], wg_body, &[]);

    builder.switch_to_block(wg_body);
    // WGSL zero-initialises `var<workgroup>` at the start of every work-group;
    // a reused stack slot does not, so do it explicitly here, once per
    // work-group, before either segment runs.
    for (base, elem, count) in &wg_fill {
        let zero = match elem {
            Ty::F32 => builder.ins().f32const(0.0),
            t => builder.ins().iconst(cl_ty(*t), 0),
        };
        emit_fill(&mut builder, *base, *count, zero);
    }
    // The per-invocation "still running" mask. An invocation that returns
    // before the barrier has left the kernel for good - WGSL's uniformity rule
    // is what makes a pre-barrier return legal, and it means the invocation
    // does no further work AT ALL, not merely none before the barrier. The two
    // segments are separate loops here, so the fact has to be carried across
    // them in work-group scratch; a per-invocation SSA local could not survive
    // the split (that is exactly what `f6` refuses).
    let active_slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        (wgsize as u32) * 4,
        2,
    ));
    let active_base = builder.ins().stack_addr(ptr_ty, active_slot, 0);
    // The work-group / global ids are recomputed inside each invocation loop from
    // `wg` (the loop-header param) so every operand is block-local — avoids
    // cross-block dominance issues for values shared by both segment loops.
    let bx = WgBuiltins { wg, gx, gy, wgsize: wgsize as i32 };
    for (seg, c, first) in [(&seg_before, &no_carried, true), (&seg_after, &carried, false)] {
        emit_invocation_loop(
            &mut builder, nmod, func, seg, c, &ba, &wg_mem, &buf_base, uniform_ptr, &fns, ptr_ty,
            &bx, active_base, first,
        )?;
    }
    builder.ins().jump(wg_latch, &[]);

    builder.switch_to_block(wg_latch);
    let next = builder.ins().iadd_imm(wg, 1);
    builder.ins().jump(wg_header, &[BlockArg::from(next)]);

    builder.switch_to_block(exit);
    builder.ins().return_(&[]);
    builder.seal_all_blocks();
    builder.finalize();
    Ok(())
}

/// Emit `for i in 0..count { base[i] = value }` for a 4-byte-strided array,
/// leaving control at the loop-exit block. `value` must be defined in a block
/// that dominates this one.
fn emit_fill(builder: &mut FunctionBuilder, base: Value, count: i64, value: Value) {
    let header = builder.create_block();
    builder.append_block_param(header, types::I64);
    let body = builder.create_block();
    let exit = builder.create_block();
    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().jump(header, &[BlockArg::from(zero)]);

    builder.switch_to_block(header);
    let i = builder.block_params(header)[0];
    let n = builder.ins().iconst(types::I64, count);
    let done = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i, n);
    builder.ins().brif(done, exit, &[], body, &[]);

    builder.switch_to_block(body);
    let off = builder.ins().imul_imm(i, 4);
    let addr = builder.ins().iadd(base, off);
    builder.ins().store(MemFlags::trusted(), value, addr, 0);
    let next = builder.ins().iadd_imm(i, 1);
    builder.ins().jump(header, &[BlockArg::from(next)]);

    builder.switch_to_block(exit);
}

/// Per-workgroup builtin inputs shared by both segment loops. `wg` is the flat
/// work-group id (the wg-loop header param); `gx`/`gy` are num_workgroups.
struct WgBuiltins {
    wg: Value,
    gx: Value,
    gy: Value,
    wgsize: i32,
}

/// Emit `for lid in 0..wgsize { <translate seg for invocation (wg,lid)> }`,
/// leaving control at the loop-exit block so the caller continues after it.
#[allow(clippy::too_many_arguments)]
fn emit_invocation_loop(
    builder: &mut FunctionBuilder,
    nmod: &naga::Module,
    func: &naga::Function,
    seg: &Block,
    carried: &[Handle<Expression>],
    ba: &BuiltinArgs,
    wg_mem: &HashMap<Handle<naga::GlobalVariable>, Value>,
    buf_base: &HashMap<u32, Value>,
    uniform_ptr: Value,
    fns: &FnRefs,
    ptr_ty: types::Type,
    bx: &WgBuiltins,
    // Base of the per-invocation "still running" mask (one i32 per lane).
    active_base: Value,
    // True for the segment BEFORE the barrier, which arms the mask and may
    // clear it; false for the segment after, which honours it.
    first: bool,
) -> Result<(), String> {
    let lid_header = builder.create_block();
    builder.append_block_param(lid_header, types::I32);
    let lid_body = builder.create_block();
    let lid_latch = builder.create_block();
    let lid_exit = builder.create_block();
    let zero = builder.ins().iconst(types::I32, 0);
    builder.ins().jump(lid_header, &[BlockArg::from(zero)]);

    builder.switch_to_block(lid_header);
    let lid = builder.block_params(lid_header)[0];
    let n_wg = builder.ins().iconst(types::I32, bx.wgsize as i64);
    let lid_done = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, lid, n_wg);
    builder.ins().brif(lid_done, lid_exit, &[], lid_body, &[]);

    builder.switch_to_block(lid_body);
    let z = builder.ins().iconst(types::I32, 0);
    let one = builder.ins().iconst(types::I32, 1);
    // workgroup_id from the flat wg: x = wg % gx, y = wg / gx (computed here so
    // all operands are defined in this block).
    let gx64 = builder.ins().uextend(types::I64, bx.gx);
    let rem = builder.ins().urem(bx.wg, gx64);
    let wgid_x = builder.ins().ireduce(types::I32, rem);
    let div = builder.ins().udiv(bx.wg, gx64);
    let wgid_y = builder.ins().ireduce(types::I32, div);
    // global_invocation_id.x = workgroup_id.x * wgsize + local_id.x
    let wgsz = builder.ins().iconst(types::I32, bx.wgsize as i64);
    let prod = builder.ins().imul(wgid_x, wgsz);
    let gid_x = builder.ins().iadd(prod, lid);
    let local_id = [lid, z, z];
    let wgid = [wgid_x, wgid_y, z];
    let nwg = [bx.gx, bx.gy, one];
    let gid = [gid_x, wgid_y, z];

    // This lane's slot in the mask.
    let lid64 = builder.ins().uextend(types::I64, lid);
    let active_off = builder.ins().imul_imm(lid64, 4);
    let active_addr = builder.ins().iadd(active_base, active_off);
    if first {
        let alive = builder.ins().iconst(types::I32, 1);
        builder.ins().store(MemFlags::trusted(), alive, active_addr, 0);
    } else {
        // An invocation that returned before the barrier does not run this
        // segment either. Without this the surplus invocations of a padded
        // grid - exactly the ones every `if (w >= p.n_wg) { return; }` guard
        // exists to exclude - would write their output anyway.
        let a = builder.ins().load(types::I32, MemFlags::trusted(), active_addr, 0);
        let dead = builder.ins().icmp_imm(IntCC::Equal, a, 0);
        let run = builder.create_block();
        builder.ins().brif(dead, lid_latch, &[], run, &[]);
        builder.switch_to_block(run);
    }

    let fell_through = {
        let env = Env {
            ptr_ty,
            buf_base,
            wg_mem,
            uniform_ptr,
            gid,
            nwg,
            local_id,
            wgid,
            ba: *ba,
            fns,
            latch: lid_latch,
            active_addr: if first { Some(active_addr) } else { None },
        };
        let mut tr = Tr::new(nmod, &mut *builder, func, env);
        // Per-invocation locals, re-declared (and so re-initialised) each
        // iteration; none may be live across the barrier (checked above).
        tr.declare_locals()?;
        // Pre-materialise the carried pre-barrier lets in this (dominating) block.
        for &h in carried {
            tr.eval(h)?;
        }
        tr.block(seg)?
    };
    if fell_through {
        builder.ins().jump(lid_latch, &[]);
    }

    builder.switch_to_block(lid_latch);
    let nxt = builder.ins().iadd_imm(lid, 1);
    builder.ins().jump(lid_header, &[BlockArg::from(nxt)]);

    builder.switch_to_block(lid_exit);
    Ok(())
}

/// The function-argument indices of the builtin vec3 inputs the kernel declares.
#[derive(Default, Clone, Copy)]
struct BuiltinArgs {
    gid: Option<u32>,
    nwg: Option<u32>,
    local_id: Option<u32>,
    wgid: Option<u32>,
}

fn builtin_args(func: &naga::Function) -> BuiltinArgs {
    let mut b = BuiltinArgs::default();
    for (i, arg) in func.arguments.iter().enumerate() {
        if let Some(naga::Binding::BuiltIn(bi)) = &arg.binding {
            match bi {
                BuiltIn::GlobalInvocationId => b.gid = Some(i as u32),
                BuiltIn::NumWorkGroups => b.nwg = Some(i as u32),
                BuiltIn::LocalInvocationId => b.local_id = Some(i as u32),
                BuiltIn::WorkGroupId => b.wgid = Some(i as u32),
                _ => {}
            }
        }
    }
    b
}

fn cl_ty(ty: Ty) -> types::Type {
    match ty {
        Ty::F32 => types::F32,
        Ty::Bool => types::I8,
        _ => types::I32,
    }
}

/// What a translated function body needs from the kernel around it: buffer
/// base pointers, the builtin input values, the intrinsic references, and
/// where an entry-point `return` goes.
struct Env<'a> {
    ptr_ty: types::Type,
    buf_base: &'a HashMap<u32, Value>,
    /// Base address of each `var<workgroup>` global (per-workgroup scratch),
    /// keyed by global handle. Empty for one-output-per-invocation kernels.
    wg_mem: &'a HashMap<Handle<naga::GlobalVariable>, Value>,
    uniform_ptr: Value,
    gid: [Value; 3],
    nwg: [Value; 3],
    local_id: [Value; 3],
    wgid: [Value; 3],
    ba: BuiltinArgs,
    fns: &'a FnRefs,
    /// Where an entry-point `return` jumps: the next invocation.
    latch: cranelift_codegen::ir::Block,
    /// Address of this invocation's slot in the work-group "still running"
    /// mask, for the segment BEFORE the barrier. `None` everywhere else: the
    /// one-output-per-invocation path has no segment after a barrier to
    /// suppress, and the after-segment has already honoured the mask on entry.
    active_addr: Option<Value>,
}

struct Tr<'a, 'b> {
    module_ref: &'a naga::Module,
    b: &'a mut FunctionBuilder<'b>,
    env: Env<'a>,
    /// The function being translated: the entry point, or an inlined callee.
    frame: Frame<'a>,
    /// Flattened aggregate values, indexed by `Eval::Agg`.
    aggs: Vec<Vec<(Value, Ty)>>,
    /// The variables of every register-backed aggregate local, indexed by
    /// `Place::Vars::group`.
    var_groups: Vec<Vec<(Variable, Ty)>>,
    /// The user functions being inlined, innermost last.
    call_stack: Vec<Handle<naga::Function>>,
}

impl<'a, 'b> Tr<'a, 'b> {
    fn new(
        module_ref: &'a naga::Module,
        b: &'a mut FunctionBuilder<'b>,
        entry: &'a naga::Function,
        env: Env<'a>,
    ) -> Self {
        Tr {
            module_ref,
            b,
            env,
            frame: Frame::new(entry, true, Vec::new(), None),
            aggs: Vec::new(),
            var_groups: Vec::new(),
            call_stack: Vec::new(),
        }
    }

    /// Translate a block. Returns `true` if control falls through the end (so the
    /// caller must emit the continuation jump), `false` if it already terminated
    /// (via return/break/continue).
    fn block(&mut self, block: &Block) -> Result<bool, String> {
        for stmt in block.iter() {
            match stmt {
                Statement::Emit(range) => {
                    for h in range.clone() {
                        self.eval(h)?;
                    }
                }
                Statement::Block(inner) => {
                    if !self.block(inner)? {
                        return Ok(false);
                    }
                }
                Statement::Store { pointer, value } => {
                    let place = self.place(*pointer)?;
                    self.store(place, *value)?;
                }
                Statement::Call { function, arguments, result } => {
                    self.inline_call(*function, arguments, *result)?;
                }
                Statement::Return { value } => {
                    if !self.frame.entry {
                        self.inline_return(*value)?;
                        return Ok(false);
                    }
                    // Per-invocation early-out: skip to the next invocation -
                    // and, in a work-group kernel, record that this lane is
                    // done so the segment after the barrier skips it too.
                    if let Some(addr) = self.env.active_addr {
                        let dead = self.b.ins().iconst(types::I32, 0);
                        self.b.ins().store(MemFlags::trusted(), dead, addr, 0);
                    }
                    self.b.ins().jump(self.env.latch, &[]);
                    return Ok(false);
                }
                Statement::Break => {
                    let (_, brk) = *self.frame.loop_stack.last().ok_or("break outside loop")?;
                    self.b.ins().jump(brk, &[]);
                    return Ok(false);
                }
                Statement::Continue => {
                    let (cont, _) = *self.frame.loop_stack.last().ok_or("continue outside loop")?;
                    self.b.ins().jump(cont, &[]);
                    return Ok(false);
                }
                Statement::If { condition, accept, reject } => {
                    let cond = self.scalar(*condition)?;
                    let then_b = self.b.create_block();
                    let else_b = self.b.create_block();
                    let merge_b = self.b.create_block();
                    self.b.ins().brif(cond.0, then_b, &[], else_b, &[]);

                    self.b.switch_to_block(then_b);
                    if self.block(accept)? {
                        self.b.ins().jump(merge_b, &[]);
                    }
                    self.b.switch_to_block(else_b);
                    if self.block(reject)? {
                        self.b.ins().jump(merge_b, &[]);
                    }
                    self.b.switch_to_block(merge_b);
                }
                Statement::Loop { body, continuing, break_if } => {
                    let body_b = self.b.create_block();
                    let cont_b = self.b.create_block();
                    let exit_b = self.b.create_block();
                    self.b.ins().jump(body_b, &[]);

                    self.b.switch_to_block(body_b);
                    self.frame.loop_stack.push((cont_b, exit_b));
                    if self.block(body)? {
                        self.b.ins().jump(cont_b, &[]);
                    }
                    self.frame.loop_stack.pop();

                    self.b.switch_to_block(cont_b);
                    if self.block(continuing)? {
                        if let Some(bi) = break_if {
                            let c = self.scalar(*bi)?;
                            self.b.ins().brif(c.0, exit_b, &[], body_b, &[]);
                        } else {
                            self.b.ins().jump(body_b, &[]);
                        }
                    }
                    self.b.switch_to_block(exit_b);
                }
                // Barriers are handled structurally by the work-group compile path
                // (it splits the body at the control barrier into per-invocation
                // segment loops), so within a translated segment a barrier is a no-op.
                Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => {}
                other => return Err(format!("unsupported statement {other:?}")),
            }
        }
        Ok(true)
    }

    /// `*place = value`.
    fn store(&mut self, place: Place, value: Handle<Expression>) -> Result<(), String> {
        match place {
            Place::Local(var, ty) => {
                let v = self.scalar(value)?;
                let v = self.coerce(v, ty)?;
                self.b.def_var(var, v);
            }
            Place::Mem { addr, elem, .. } => {
                let v = self.scalar(value)?;
                let v = self.coerce(v, elem)?;
                self.b.ins().store(MemFlags::trusted(), v, addr, 0);
            }
            _ => {
                let (vals, _) = self.value(value)?;
                self.store_place(place, &vals)?;
            }
        }
        Ok(())
    }

    /// Evaluate an expression, memoising the result if it has a single
    /// program point (see [`Frame::emitted`]).
    fn eval(&mut self, h: Handle<Expression>) -> Result<Eval, String> {
        if let Some(e) = self.frame.cache.get(&h) {
            return Ok(*e);
        }
        let e = self.eval_uncached(h)?;
        if self.frame.emitted.contains(&h) {
            self.frame.cache.insert(h, e);
        }
        Ok(e)
    }

    /// Evaluate an expression that must be a scalar value.
    fn scalar(&mut self, h: Handle<Expression>) -> Result<(Value, Ty), String> {
        match self.eval(h)? {
            Eval::Scalar(v, t) => Ok((v, t)),
            Eval::Place(p) => match self.load_place(p)? {
                Eval::Scalar(v, t) => Ok((v, t)),
                _ => Err("expected a scalar, got an aggregate place".into()),
            },
            Eval::Agg(_, shape) => Err(format!("expected a scalar, got {shape:?}")),
        }
    }

    /// Evaluate an expression that must yield an addressable place.
    fn place(&mut self, h: Handle<Expression>) -> Result<Place, String> {
        match self.eval(h)? {
            Eval::Place(p) => Ok(p),
            _ => Err("expected place, got a value".into()),
        }
    }

    fn eval_uncached(&mut self, h: Handle<Expression>) -> Result<Eval, String> {
        // `func` is a copy of the frame's `&'a Function`, independent of the
        // `&mut self` borrow, so expression refs derived from it stay valid
        // across the `&mut self` recursion below.
        let func = self.frame.func;
        let expr = &func.expressions[h];
        match expr {
            Expression::Literal(lit) => self.literal(lit),
            Expression::ZeroValue(ty) => self.zero_value(*ty),
            Expression::Constant(c) => {
                let init = self.module_ref.constants[*c].init;
                self.eval_global_const(init)
            }
            Expression::FunctionArgument(i) => self.function_argument(*i),
            Expression::GlobalVariable(g) => Ok(Eval::Place(self.global_place(*g)?)),
            Expression::LocalVariable(l) => Ok(Eval::Place(self.local_place(*l))),
            Expression::Load { pointer } => {
                let p = self.place(*pointer)?;
                self.load_place(p)
            }
            Expression::Access { base, index } => self.access(*base, *index),
            Expression::AccessIndex { base, index } => {
                // A component of a builtin vector argument (gid.x, nwg.x, ...).
                if let (true, Expression::FunctionArgument(ai)) = (self.frame.entry, &func.expressions[*base]) {
                    let comp = *index as usize;
                    let ba = self.env.ba;
                    let v = if Some(*ai) == ba.gid {
                        self.env.gid[comp]
                    } else if Some(*ai) == ba.nwg {
                        self.env.nwg[comp]
                    } else if Some(*ai) == ba.local_id {
                        self.env.local_id[comp]
                    } else if Some(*ai) == ba.wgid {
                        self.env.wgid[comp]
                    } else {
                        return Err("AccessIndex on unknown builtin arg".into());
                    };
                    return Ok(Eval::Scalar(v, Ty::U32));
                }
                self.access_index(*base, *index)
            }
            Expression::Compose { ty, components } => self.compose(*ty, components),
            Expression::Splat { size, value } => {
                let v = self.scalar(*value)?;
                Ok(self.splat(*size, v))
            }
            Expression::Swizzle { size, vector, pattern } => self.swizzle(*size, *vector, pattern),
            Expression::Unary { op, expr } => match self.eval(*expr)? {
                Eval::Scalar(v, t) => {
                    let (r, t) = self.unary_scalar(*op, (v, t));
                    Ok(Eval::Scalar(r, t))
                }
                e => self.unary_agg(*op, e),
            },
            Expression::Binary { op, left, right } => {
                let l = self.eval(*left)?;
                let r = self.eval(*right)?;
                match (l, r) {
                    (Eval::Scalar(lv, lt), Eval::Scalar(rv, rt)) => {
                        let (v, t) = self.binary_scalar(*op, (lv, lt), (rv, rt))?;
                        Ok(Eval::Scalar(v, t))
                    }
                    _ => self.binary_agg(*op, l, r),
                }
            }
            Expression::Select { condition, accept, reject } => {
                let c = self.eval(*condition)?;
                let a = self.eval(*accept)?;
                let r = self.eval(*reject)?;
                match (c, a, r) {
                    (Eval::Scalar(c, _), Eval::Scalar(a, ta), Eval::Scalar(r, _)) => {
                        Ok(Eval::Scalar(self.b.ins().select(c, a, r), ta))
                    }
                    _ => self.select_agg(c, a, r),
                }
            }
            Expression::Math { fun, arg, arg1, arg2, .. } => {
                let a = self.eval(*arg)?;
                let b = arg1.map(|h| self.eval(h)).transpose()?;
                let c = arg2.map(|h| self.eval(h)).transpose()?;
                let scalar = |e: Option<Eval>| match e {
                    None => Some(None),
                    Some(Eval::Scalar(v, t)) => Some(Some((v, t))),
                    Some(_) => None,
                };
                match (a, scalar(b), scalar(c)) {
                    (Eval::Scalar(v, t), Some(b), Some(c)) => {
                        let (r, t) = self.math_scalar(*fun, (v, t), b, c)?;
                        Ok(Eval::Scalar(r, t))
                    }
                    _ => self.math_agg(*fun, a, b, c),
                }
            }
            Expression::As { expr, kind, convert } => match self.eval(*expr)? {
                Eval::Scalar(v, t) => {
                    let (r, t) = self.cast_scalar((v, t), *kind, *convert)?;
                    Ok(Eval::Scalar(r, t))
                }
                e => self.cast_agg(e, *kind, *convert),
            },
            Expression::Relational { fun, argument } => self.relational(*fun, *argument),
            Expression::CallResult(_) => Err("a call result used before its call".into()),
            other => Err(format!("unsupported expression {other:?}")),
        }
    }

    fn literal(&mut self, lit: &Literal) -> Result<Eval, String> {
        Ok(match lit {
            Literal::F32(x) => Eval::Scalar(self.b.ins().f32const(*x), Ty::F32),
            Literal::F64(x) => Eval::Scalar(self.b.ins().f32const(*x as f32), Ty::F32),
            Literal::AbstractFloat(x) => Eval::Scalar(self.b.ins().f32const(*x as f32), Ty::F32),
            Literal::U32(x) => Eval::Scalar(self.b.ins().iconst(types::I32, *x as i64), Ty::U32),
            Literal::I32(x) => {
                Eval::Scalar(self.b.ins().iconst(types::I32, *x as i64 & 0xffff_ffff), Ty::I32)
            }
            Literal::AbstractInt(x) => {
                Eval::Scalar(self.b.ins().iconst(types::I32, *x & 0xffff_ffff), Ty::I32)
            }
            Literal::Bool(x) => Eval::Scalar(self.b.ins().iconst(types::I8, *x as i64), Ty::Bool),
            other => return Err(format!("unsupported literal {other:?}")),
        })
    }

    fn unary_scalar(&mut self, op: UnaryOperator, (v, t): (Value, Ty)) -> (Value, Ty) {
        let r = match op {
            UnaryOperator::Negate => {
                if t.is_float() {
                    self.b.ins().fneg(v)
                } else {
                    self.b.ins().ineg(v)
                }
            }
            UnaryOperator::LogicalNot => {
                let z = self.b.ins().iconst(types::I8, 0);
                self.b.ins().icmp(IntCC::Equal, v, z)
            }
            UnaryOperator::BitwiseNot => self.b.ins().bnot(v),
        };
        (r, t)
    }

    fn binary_scalar(
        &mut self,
        op: BinaryOperator,
        (l, lt): (Value, Ty),
        (r, rt): (Value, Ty),
    ) -> Result<(Value, Ty), String> {
        let float = lt.is_float() || rt.is_float();
        use BinaryOperator::*;
        let ins = self.b.ins();
        Ok(match op {
            Add if float => (ins.fadd(l, r), Ty::F32),
            Add => (ins.iadd(l, r), lt),
            Subtract if float => (ins.fsub(l, r), Ty::F32),
            Subtract => (ins.isub(l, r), lt),
            Multiply if float => (ins.fmul(l, r), Ty::F32),
            Multiply => (ins.imul(l, r), lt),
            Divide if float => (ins.fdiv(l, r), Ty::F32),
            Divide if lt == Ty::I32 => (ins.sdiv(l, r), Ty::I32),
            Divide => (ins.udiv(l, r), Ty::U32),
            Modulo if float => return Err("float modulo unsupported".into()),
            Modulo if lt == Ty::I32 => (ins.srem(l, r), Ty::I32),
            Modulo => (ins.urem(l, r), Ty::U32),
            Equal if float => (ins.fcmp(FloatCC::Equal, l, r), Ty::Bool),
            Equal => (ins.icmp(IntCC::Equal, l, r), Ty::Bool),
            NotEqual if float => (ins.fcmp(FloatCC::NotEqual, l, r), Ty::Bool),
            NotEqual => (ins.icmp(IntCC::NotEqual, l, r), Ty::Bool),
            Less if float => (ins.fcmp(FloatCC::LessThan, l, r), Ty::Bool),
            Less if lt == Ty::I32 => (ins.icmp(IntCC::SignedLessThan, l, r), Ty::Bool),
            Less => (ins.icmp(IntCC::UnsignedLessThan, l, r), Ty::Bool),
            LessEqual if float => (ins.fcmp(FloatCC::LessThanOrEqual, l, r), Ty::Bool),
            LessEqual if lt == Ty::I32 => {
                (ins.icmp(IntCC::SignedLessThanOrEqual, l, r), Ty::Bool)
            }
            LessEqual => (ins.icmp(IntCC::UnsignedLessThanOrEqual, l, r), Ty::Bool),
            Greater if float => (ins.fcmp(FloatCC::GreaterThan, l, r), Ty::Bool),
            Greater if lt == Ty::I32 => (ins.icmp(IntCC::SignedGreaterThan, l, r), Ty::Bool),
            Greater => (ins.icmp(IntCC::UnsignedGreaterThan, l, r), Ty::Bool),
            GreaterEqual if float => (ins.fcmp(FloatCC::GreaterThanOrEqual, l, r), Ty::Bool),
            GreaterEqual if lt == Ty::I32 => {
                (ins.icmp(IntCC::SignedGreaterThanOrEqual, l, r), Ty::Bool)
            }
            GreaterEqual => (ins.icmp(IntCC::UnsignedGreaterThanOrEqual, l, r), Ty::Bool),
            And => (ins.band(l, r), lt),
            InclusiveOr => (ins.bor(l, r), lt),
            ExclusiveOr => (ins.bxor(l, r), lt),
            LogicalAnd => (ins.band(l, r), Ty::Bool),
            LogicalOr => (ins.bor(l, r), Ty::Bool),
            ShiftLeft => (ins.ishl(l, r), lt),
            ShiftRight if lt == Ty::I32 => (ins.sshr(l, r), Ty::I32),
            ShiftRight => (ins.ushr(l, r), Ty::U32),
        })
    }

    /// A math builtin on scalar arguments (applied per component to vectors
    /// by `math_agg`).
    fn math_scalar(
        &mut self,
        fun: MathFunction,
        (a, at): (Value, Ty),
        arg1: Option<(Value, Ty)>,
        arg2: Option<(Value, Ty)>,
    ) -> Result<(Value, Ty), String> {
        use MathFunction::*;
        let need = |v: Option<(Value, Ty)>| v.ok_or(format!("{fun:?} needs more arguments"));
        match fun {
            Sqrt => Ok((self.b.ins().sqrt(a), Ty::F32)),
            InverseSqrt => {
                let s = self.b.ins().sqrt(a);
                let one = self.b.ins().f32const(1.0);
                Ok((self.b.ins().fdiv(one, s), Ty::F32))
            }
            Abs if at.is_float() => Ok((self.b.ins().fabs(a), Ty::F32)),
            Abs => Ok((self.b.ins().iabs(a), at)),
            // length(x) = |x| and distance(x, y) = |x - y| for scalars.
            Length => Ok((self.b.ins().fabs(a), Ty::F32)),
            Distance => {
                let d = self.b.ins().fsub(a, need(arg1)?.0);
                Ok((self.b.ins().fabs(d), Ty::F32))
            }
            Min | Max => {
                let (b, bt) = need(arg1)?;
                let float = at.is_float() || bt.is_float();
                let v = match (fun, float) {
                    (Min, true) => self.b.ins().fmin(a, b),
                    (Max, true) => self.b.ins().fmax(a, b),
                    (Min, false) if at == Ty::I32 => self.b.ins().smin(a, b),
                    (Min, false) => self.b.ins().umin(a, b),
                    (Max, false) if at == Ty::I32 => self.b.ins().smax(a, b),
                    (Max, false) => self.b.ins().umax(a, b),
                    _ => unreachable!(),
                };
                Ok((v, if float { Ty::F32 } else { at }))
            }
            Fma => {
                let (b, _) = need(arg1)?;
                let (c, _) = need(arg2)?;
                Ok((self.b.ins().fma(a, b, c), Ty::F32))
            }
            Step => {
                // step(edge, x) = x < edge ? 0.0 : 1.0
                let (x, _) = need(arg1)?;
                let lt = self.b.ins().fcmp(FloatCC::LessThan, x, a);
                let zero = self.b.ins().f32const(0.0);
                let one = self.b.ins().f32const(1.0);
                Ok((self.b.ins().select(lt, zero, one), Ty::F32))
            }
            Pow | Atan2 => {
                let (b, _) = need(arg1)?;
                let f = if fun == Pow { self.env.fns.powf } else { self.env.fns.atan2 };
                let call = self.b.ins().call(f, &[a, b]);
                Ok((self.b.inst_results(call)[0], Ty::F32))
            }
            Exp | Log | Sin | Cos | Tanh => {
                let sym = match fun {
                    Exp => "brain_expf",
                    Log => "brain_logf",
                    Sin => "brain_sinf",
                    Cos => "brain_cosf",
                    Tanh => "brain_tanhf",
                    _ => unreachable!(),
                };
                let fref = self.env.fns.unary[sym];
                let call = self.b.ins().call(fref, &[a]);
                Ok((self.b.inst_results(call)[0], Ty::F32))
            }
            // --- rounding: Cranelift has these natively, and each matches the
            // WGSL builtin exactly. `Round` is roundToIntegralTiesToEven, which
            // is WGSL's "halfway cases round to even" -- NOT `round-half-away`,
            // so do not reach for a `floor(x+0.5)` shortcut here.
            Floor => Ok((self.b.ins().floor(a), Ty::F32)),
            Ceil => Ok((self.b.ins().ceil(a), Ty::F32)),
            Trunc => Ok((self.b.ins().trunc(a), Ty::F32)),
            Round => Ok((self.b.ins().nearest(a), Ty::F32)),
            // fract(e) = e - floor(e)
            Fract => {
                let f = self.b.ins().floor(a);
                Ok((self.b.ins().fsub(a, f), Ty::F32))
            }
            // clamp(e, low, high) = min(max(e, low), high)
            Clamp => {
                let (lo, _) = need(arg1)?;
                let (hi, _) = need(arg2)?;
                let v = if at.is_float() {
                    let m = self.b.ins().fmax(a, lo);
                    self.b.ins().fmin(m, hi)
                } else if at == Ty::I32 {
                    let m = self.b.ins().smax(a, lo);
                    self.b.ins().smin(m, hi)
                } else {
                    let m = self.b.ins().umax(a, lo);
                    self.b.ins().umin(m, hi)
                };
                Ok((v, at))
            }
            // saturate(e) = clamp(e, 0.0, 1.0)
            Saturate => {
                let zero = self.b.ins().f32const(0.0);
                let one = self.b.ins().f32const(1.0);
                let m = self.b.ins().fmax(a, zero);
                Ok((self.b.ins().fmin(m, one), Ty::F32))
            }
            // mix(e1, e2, e3) = e1*(1-e3) + e2*e3  (the spec form, not the
            // algebraically-equal `e1 + e3*(e2-e1)`: they differ in fp32 rounding
            // and the spec form is what the wgpu path computes).
            Mix => {
                let (b, _) = need(arg1)?;
                let (t, _) = need(arg2)?;
                let one = self.b.ins().f32const(1.0);
                let inv = self.b.ins().fsub(one, t);
                let l = self.b.ins().fmul(a, inv);
                let r = self.b.ins().fmul(b, t);
                Ok((self.b.ins().fadd(l, r), Ty::F32))
            }
            // sign(e) = e < 0 ? -1 : (e > 0 ? 1 : 0)
            Sign if at.is_float() => {
                let zero = self.b.ins().f32const(0.0);
                let pos1 = self.b.ins().f32const(1.0);
                let neg1 = self.b.ins().f32const(-1.0);
                let gt = self.b.ins().fcmp(FloatCC::GreaterThan, a, zero);
                let hi = self.b.ins().select(gt, pos1, zero);
                let lt = self.b.ins().fcmp(FloatCC::LessThan, a, zero);
                Ok((self.b.ins().select(lt, neg1, hi), Ty::F32))
            }
            Sign => {
                let zero = self.b.ins().iconst(types::I32, 0);
                let pos1 = self.b.ins().iconst(types::I32, 1);
                let neg1 = self.b.ins().iconst(types::I32, -1);
                let gt = self.b.ins().icmp(IntCC::SignedGreaterThan, a, zero);
                let hi = self.b.ins().select(gt, pos1, zero);
                let lt = self.b.ins().icmp(IntCC::SignedLessThan, a, zero);
                Ok((self.b.ins().select(lt, neg1, hi), Ty::I32))
            }
            // dot4I8Packed(a, b): four signed-int8 multiply-accumulates over
            // the bytes of two u32s (the DP4A the packed-int8 GEMMs use).
            // Byte i sits at bits [8i, 8i+8): shift it to the top byte, then
            // an arithmetic shift right 24 sign-extends it to i32.
            Dot4I8Packed => {
                let (b2, _) = need(arg1)?;
                let mut acc = self.b.ins().iconst(types::I32, 0);
                for i in 0..4 {
                    let up = 24 - 8 * i;
                    let ai = self.b.ins().ishl_imm(a, up);
                    let ai = self.b.ins().sshr_imm(ai, 24);
                    let bi = self.b.ins().ishl_imm(b2, up);
                    let bi = self.b.ins().sshr_imm(bi, 24);
                    let p = self.b.ins().imul(ai, bi);
                    acc = self.b.ins().iadd(acc, p);
                }
                Ok((acc, Ty::I32))
            }
            other => Err(format!("unsupported math fn {other:?}")),
        }
    }

    fn cast_scalar(
        &mut self,
        (v, t): (Value, Ty),
        kind: ScalarKind,
        convert: Option<u8>,
    ) -> Result<(Value, Ty), String> {
        let target = Ty::from_scalar(Scalar { kind, width: convert.unwrap_or(4) })?;
        if t == target {
            return Ok((v, t));
        }
        if convert.is_none() {
            // Bitcast (reinterpret), same width.
            let v = match (t, target) {
                (Ty::F32, _) => self.b.ins().bitcast(types::I32, MemFlags::new(), v),
                (_, Ty::F32) => self.b.ins().bitcast(types::F32, MemFlags::new(), v),
                _ => v, // int<->int reinterpret is a no-op
            };
            return Ok((v, target));
        }
        let v = match (t, target) {
            (Ty::U32, Ty::F32) => self.b.ins().fcvt_from_uint(types::F32, v),
            (Ty::I32, Ty::F32) => self.b.ins().fcvt_from_sint(types::F32, v),
            (Ty::F32, Ty::U32) => self.b.ins().fcvt_to_uint_sat(types::I32, v),
            (Ty::F32, Ty::I32) => self.b.ins().fcvt_to_sint_sat(types::I32, v),
            (Ty::U32, Ty::I32) | (Ty::I32, Ty::U32) => v,
            (Ty::Bool, _) => self.b.ins().uextend(types::I32, v),
            _ => return Err(format!("unsupported cast {t:?} -> {target:?}")),
        };
        Ok((v, target))
    }

    /// Coerce a scalar to the requested storage type (only int-width retags and
    /// the bool->int widening that arise from our type tracking).
    fn coerce(&mut self, (v, t): (Value, Ty), want: Ty) -> Result<Value, String> {
        if t == want {
            return Ok(v);
        }
        match (t, want) {
            (Ty::U32, Ty::I32) | (Ty::I32, Ty::U32) => Ok(v),
            (Ty::Bool, Ty::U32) | (Ty::Bool, Ty::I32) => Ok(self.b.ins().uextend(types::I32, v)),
            _ => Err(format!("cannot store {t:?} into {want:?}")),
        }
    }

    /// Emit a widening to i64. Named `emit_*` rather than `to_*`: this appends
    /// an instruction to the function under construction, it does not convert
    /// `self`.
    fn emit_i64(&mut self, v: Value) -> Value {
        let t = self.b.func.dfg.value_type(v);
        if t == types::I64 {
            v
        } else {
            self.b.ins().uextend(types::I64, v)
        }
    }

    fn zero(&mut self, t: Ty) -> Value {
        match t {
            Ty::F32 => self.b.ins().f32const(0.0),
            Ty::Bool => self.b.ins().iconst(types::I8, 0),
            _ => self.b.ins().iconst(types::I32, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptrs(bufs: &mut [&mut [f32]]) -> Vec<*mut u8> {
        bufs.iter_mut().map(|b| b.as_mut_ptr() as *mut u8).collect()
    }

    #[test]
    fn add2_runs_on_cpu() {
        let jit = Jit::new(&[("add2", kernels::ADD2)]).expect("compile add2");
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 4];
        let mut a2 = a.clone();
        let mut b2 = b.clone();
        let bufs = {
            let v: Vec<*mut u8> = vec![
                a2.as_mut_ptr() as *mut u8,
                b2.as_mut_ptr() as *mut u8,
                out.as_mut_ptr() as *mut u8,
            ];
            v
        };
        let uniform = [4u32];
        // grid_x = ceil(4/64)=1, so 64 invocations; the kernel masks idx>=4.
        unsafe {
            jit.run(0, 0, 64, 1, 1, uniform.as_ptr(), bufs.as_ptr());
        }
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// `dot4I8Packed` must lower correctly: the packed-int8 decode GEMV
    /// (single barrier, so JIT-able) against a host int8 reference. Signed
    /// bytes are the trap — a zero-extend instead of sign-extend passes on
    /// positive values and corrupts every negative weight.
    ///
    /// `k = 64` is TWO of `model::int8::GROUP`'s 32-element weight-scale
    /// groups, so `sw` is `[n, 2]` and a kernel that ignored the group axis
    /// (applying group 0's scale to the whole row) fails here rather than
    /// passing on a one-group shape.
    #[test]
    fn dot4i8packed_matches_host_reference() {
        let jit = Jit::new(&[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV)]).expect("compile i8 gemv");
        let (m, k, n) = (2usize, 64usize, 3usize);
        let (kg, ng) = (k / 4, k / 32);
        // Deliberately mixed-sign int8 values.
        let xq_i: Vec<i8> = (0..m * k).map(|i| ((i as i32 * 37 + 11) % 255 - 127) as i8).collect();
        let wq_i: Vec<i8> = (0..n * k).map(|i| ((i as i32 * 53 + 5) % 255 - 127) as i8).collect();
        let pack = |v: &[i8]| -> Vec<f32> {
            v.chunks(4)
                .map(|c| {
                    let w = (c[0] as u8 as u32)
                        | ((c[1] as u8 as u32) << 8)
                        | ((c[2] as u8 as u32) << 16)
                        | ((c[3] as u8 as u32) << 24);
                    f32::from_bits(w)
                })
                .collect()
        };
        let mut xq = pack(&xq_i);
        let mut wq = pack(&wq_i);
        let mut sx: Vec<f32> = vec![0.5, 0.25];
        // [n, ng], every entry distinct so a wrong group index shows.
        let mut sw: Vec<f32> = vec![1.0, 4.0, 2.0, 8.0, 0.125, 0.5];
        assert_eq!(sw.len(), n * ng);
        let mut out = vec![0.0f32; m * n];
        let bufs = [xq.as_mut_ptr() as *mut u8,
            wq.as_mut_ptr() as *mut u8,
            sx.as_mut_ptr() as *mut u8,
            sw.as_mut_ptr() as *mut u8,
            out.as_mut_ptr() as *mut u8];
        let uniform = [m as u32, kg as u32, n as u32];
        // One workgroup (64 threads) per output column.
        unsafe {
            jit.run(0, 0, (n * 64) as u64, n as u32, 1, uniform.as_ptr(), bufs.as_ptr());
        }
        for row in 0..m {
            for col in 0..n {
                // Group-wise dequant: each 32-element block carries its own
                // scale, so the sum is per group and then folded.
                let want: f32 = (0..ng)
                    .map(|g| {
                        let acc: i32 = (g * 32..g * 32 + 32)
                            .map(|kk| xq_i[row * k + kk] as i32 * wq_i[col * k + kk] as i32)
                            .sum();
                        acc as f32 * sw[col * ng + g]
                    })
                    .sum::<f32>()
                    * sx[row];
                let got = out[row * n + col];
                assert!(
                    (got - want).abs() <= want.abs() * 1e-5 + 1e-3,
                    "out[{row},{col}] = {got}, want {want}"
                );
            }
        }
    }

    /// The work-group tiled conv (workgroup memory + barrier) compiled by the JIT
    /// must match a direct scalar convolution. Exercises the whole B1 machinery:
    /// `var<workgroup>` staging, `workgroupBarrier()` segment split, and the
    /// local_invocation_id / workgroup_id builtins.
    #[test]
    fn conv2d_tiled_workgroup_matches_scalar() {
        let jit = Jit::new(&[("conv2d_tiled", kernels::CONV2D_TILED)]).expect("compile tiled conv");
        let idx = jit.index_of("conv2d_tiled").unwrap();
        assert_eq!(jit.workgroup_size(idx), Some(64), "tiled conv must be a work-group kernel");

        // Small conv: N=1, Cin=4, 7x5, Cout=6, 3x3 stride1 pad1.
        let (n, cin, h, w, cout, k, stride, pad) = (1usize, 4, 7, 5, 6, 3usize, 1usize, 1usize);
        let ho = (h + 2 * pad - k) / stride + 1;
        let wo = (w + 2 * pad - k) / stride + 1;
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let mut x: Vec<f32> = (0..n * cin * h * w).map(|_| rnd()).collect();
        let mut wt: Vec<f32> = (0..cout * cin * k * k).map(|_| rnd()).collect();
        let mut y = vec![0.0f32; n * cout * ho * wo];

        let params: [u32; 10] = [
            n as u32, cin as u32, h as u32, w as u32, cout as u32, k as u32,
            stride as u32, pad as u32, ho as u32, wo as u32,
        ];
        let bufs = [x.as_mut_ptr() as *mut u8,
            wt.as_mut_ptr() as *mut u8,
            y.as_mut_ptr() as *mut u8];
        // Dispatch: one workgroup per (n, co, 64-spatial-block).
        let psz = ho * wo;
        let blocks = psz.div_ceil(64);
        let num_wg = (n * cout * blocks) as u32;
        unsafe {
            jit.run(idx, 0, (num_wg as u64) * 64, num_wg, 1, params.as_ptr(), bufs.as_ptr());
        }

        // Scalar reference.
        let mut yref = vec![0.0f32; n * cout * ho * wo];
        for co in 0..cout {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = 0.0f32;
                    for ci in 0..cin {
                        for kh in 0..k {
                            let hi = oh * stride + kh;
                            if hi < pad || hi - pad >= h { continue; }
                            for kw in 0..k {
                                let wi = ow * stride + kw;
                                if wi < pad || wi - pad >= w { continue; }
                                let xi = ((ci) * h + (hi - pad)) * w + (wi - pad);
                                let wi2 = ((co * cin + ci) * k + kh) * k + kw;
                                acc += x[xi] * wt[wi2];
                            }
                        }
                    }
                    yref[(co * ho + oh) * wo + ow] = acc;
                }
            }
        }
        let mut maxerr = 0.0f32;
        for (a, b) in y.iter().zip(yref.iter()) {
            maxerr = maxerr.max((a - b).abs());
        }
        assert!(maxerr < 1e-4, "tiled conv max abs err {maxerr}");
    }

    #[test]
    fn matmul_runs_on_cpu() {
        // x:[M=2,K=3], w:[N=2,K=3], out:[M,N], out[m,n]=sum_k x[m,k]*w[n,k]
        let jit = Jit::new(&[("matmul", kernels::MATMUL)]).expect("compile matmul");
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut w = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0f32; 4];
        let bufs = [x.as_mut_ptr() as *mut u8,
            w.as_mut_ptr() as *mut u8,
            out.as_mut_ptr() as *mut u8];
        let uniform = [2u32, 3u32, 2u32]; // m,k,n
        unsafe {
            jit.run(0, 0, 64, 1, 1, uniform.as_ptr(), bufs.as_ptr());
        }
        // out[0,0]=x[0,0]=1, out[0,1]=x[0,1]=2, out[1,0]=x[1,0]=4, out[1,1]=x[1,1]=5
        assert_eq!(out, vec![1.0, 2.0, 4.0, 5.0]);
        let _ = ptrs;
    }
}
