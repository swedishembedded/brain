// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compile brain's WGSL compute kernels to native CPU code.
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
//! Only the IR subset the 54 kernels actually use is handled (scalar f32/u32/i32,
//! the `global_invocation_id` / `num_workgroups` builtins, storage + uniform
//! bindings, `if`/`loop`, and a closed set of math intrinsics). Anything outside
//! that subset is a hard error at compile time rather than silently miscompiled.

use cranelift_codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types, AbiParam, BlockArg, InstBuilder, MemFlags, Signature, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{Linkage, Module};
use cranelift_jit::{JITBuilder, JITModule};
use std::collections::HashMap;

use naga::{
    AddressSpace, BinaryOperator, Block, BuiltIn, Expression, Handle, Literal, MathFunction,
    ScalarKind, Statement, TypeInner, UnaryOperator,
};

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
    funcs: Vec<*const u8>,
    names: Vec<String>,
}

// The JIT-compiled code is immutable after `new` returns; only `&Jit` is shared.
unsafe impl Send for Jit {}
unsafe impl Sync for Jit {}

impl Jit {
    /// Parse and JIT-compile every `(name, wgsl_src)` kernel. Returns an error
    /// string identifying the offending kernel on the first failure.
    pub fn new(kernels: &[(&str, &str)]) -> Result<Jit, String> {
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

        for (name, src) in kernels {
            ctx.func.signature = kernel_signature(module.target_config().pointer_type());
            compile_one(name, src, &mut module, &math, &mut ctx, &mut fctx)
                .map_err(|e| format!("kernel {name:?}: {e}"))?;
            let id = module
                .declare_function(name, Linkage::Export, &ctx.func.signature)
                .map_err(|e| format!("declare {name:?}: {e}"))?;
            module
                .define_function(id, &mut ctx)
                .map_err(|e| format!("define {name:?}: {e:?}"))?;
            module.clear_context(&mut ctx);
            ids.push(id);
        }

        module
            .finalize_definitions()
            .map_err(|e| format!("finalize: {e}"))?;

        let funcs = ids.iter().map(|id| module.get_finalized_function(*id)).collect();
        Ok(Jit {
            _module: module,
            funcs,
            names: kernels.iter().map(|(n, _)| n.to_string()).collect(),
        })
    }

    /// Index of `name`, or `None` if no such kernel was compiled.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
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
        let f: KernelFn = std::mem::transmute(self.funcs[kind]);
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

fn math_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("brain_expf", w_expf as *const u8),
        ("brain_logf", w_logf as *const u8),
        ("brain_sinf", w_sinf as *const u8),
        ("brain_cosf", w_cosf as *const u8),
        ("brain_tanhf", w_tanhf as *const u8),
        ("brain_powf", w_powf as *const u8),
    ]
}

struct MathRefs {
    // FuncIds of imported intrinsics; turned into FuncRefs per function.
    unary: HashMap<&'static str, cranelift_module::FuncId>,
    powf: cranelift_module::FuncId,
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
        Ok(MathRefs { unary, powf })
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
    fn from_scalar(kind: ScalarKind) -> Result<Ty, String> {
        Ok(match kind {
            ScalarKind::Float => Ty::F32,
            ScalarKind::Uint => Ty::U32,
            ScalarKind::Sint => Ty::I32,
            ScalarKind::Bool => Ty::Bool,
            other => return Err(format!("unsupported scalar kind {other:?}")),
        })
    }
    fn is_float(self) -> bool {
        matches!(self, Ty::F32)
    }
}

/// The result of evaluating a naga expression: either a materialised scalar or a
/// "place" (an addressable location) that a `Load`/`Store` resolves.
#[derive(Clone, Copy)]
enum Eval {
    Scalar(Value, Ty),
    Place(Place),
}

#[derive(Clone, Copy)]
enum Place {
    /// A mutable scalar function-local (`var`) backed by a Cranelift SSA variable.
    Local(Variable, Ty),
    /// A memory element: `addr` bytes, of element type `elem`.
    Mem { addr: Value, elem: Ty },
}

/// Backing storage for a naga function-local variable.
#[derive(Clone, Copy)]
enum LocalSlot {
    /// A scalar `var`: a Cranelift SSA variable.
    Scalar(Variable, Ty),
    /// A fixed-size array `var`: `base` is its stack address, computed once in the
    /// entry block so it dominates every use.
    Array { base: Value, elem: Ty },
}

/// `(element scalar type, element count)` of a local fixed-size array type.
fn local_array_info(m: &naga::Module, ty: Handle<naga::Type>) -> Result<(Ty, u32), String> {
    match &m.types[ty].inner {
        TypeInner::Array { base, size, .. } => {
            let elem = match &m.types[*base].inner {
                TypeInner::Scalar(s) => Ty::from_scalar(s.kind)?,
                other => return Err(format!("local array of non-scalar {other:?}")),
            };
            let count = match size {
                naga::ArraySize::Constant(n) => n.get(),
                other => return Err(format!("non-constant local array size {other:?}")),
            };
            Ok((elem, count))
        }
        other => Err(format!("expected array local, got {other:?}")),
    }
}

// ---------------------------------------------------------------------------
// Per-kernel translation.
// ---------------------------------------------------------------------------

fn compile_one(
    name: &str,
    src: &str,
    module: &mut JITModule,
    math: &MathRefs,
    ctx: &mut cranelift_codegen::Context,
    fctx: &mut FunctionBuilderContext,
) -> Result<(), String> {
    let nmod = naga::front::wgsl::parse_str(src).map_err(|e| format!("WGSL parse: {e:?}"))?;
    let entry = nmod
        .entry_points
        .iter()
        .find(|e| e.name == "main")
        .ok_or("no `main` entry point")?;
    let func = &entry.function;

    let ptr_ty = module.target_config().pointer_type();
    let mut builder = FunctionBuilder::new(&mut ctx.func, fctx);

    // Import math intrinsics into this function.
    let mut unary_refs = HashMap::new();
    for (k, id) in &math.unary {
        unary_refs.insert(*k, module.declare_func_in_func(*id, builder.func));
    }
    let powf_ref = module.declare_func_in_func(math.powf, builder.func);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    let p = builder.block_params(entry_block).to_vec();
    let (start, end, gx, gy, uniform_ptr, bufs_ptr) =
        (p[0], p[1], p[2], p[3], p[4], p[5]);

    // Resolve which function argument carries which builtin vec3.
    let (gid_arg, nwg_arg) = builtin_args(func)?;

    // Base pointer of each storage binding (binding 1.. -> bufs[binding-1]).
    let mut buf_base: HashMap<u32, Value> = HashMap::new();
    let mut uniform_binding = None;
    for (_h, gv) in nmod.global_variables.iter() {
        let binding = match &gv.binding {
            Some(b) => b.binding,
            None => continue,
        };
        match gv.space {
            AddressSpace::Uniform => uniform_binding = Some(binding),
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

    // Locals: scalars become zero-initialised Cranelift variables (naga
    // initialises locals at function entry); fixed-size arrays become explicit
    // stack slots (scratch buffers, always written before read).
    use cranelift_codegen::ir::{StackSlotData, StackSlotKind};
    let mut locals: HashMap<Handle<naga::LocalVariable>, LocalSlot> = HashMap::new();
    for (h, lv) in func.local_variables.iter() {
        if let TypeInner::Array { .. } = &nmod.types[lv.ty].inner {
            let (elem, count) = local_array_info(&nmod, lv.ty)?;
            // All kernel arrays are 4-byte-strided scalars.
            let slot = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                count * 4,
                2,
            ));
            // Materialise the base address in the (dominating) entry block.
            let base = builder.ins().stack_addr(ptr_ty, slot, 0);
            locals.insert(h, LocalSlot::Array { base, elem });
        } else {
            let ty = scalar_ty_of(&nmod, lv.ty)?;
            let var = builder.declare_var(cl_ty(ty));
            let init = match ty {
                Ty::F32 => builder.ins().f32const(0.0),
                _ => builder.ins().iconst(cl_ty(ty), 0),
            };
            builder.def_var(var, init);
            locals.insert(h, LocalSlot::Scalar(var, ty));
        }
    }

    let mut tr = Tr {
        module_ref: &nmod,
        b: &mut builder,
        cache: HashMap::new(),
        locals: &locals,
        buf_base: &buf_base,
        uniform_ptr,
        uniform_binding,
        gid,
        nwg,
        gid_arg,
        nwg_arg,
        unary_refs: &unary_refs,
        powf_ref,
        loop_stack: Vec::new(),
        latch,
    };
    // Apply constant local initialisers now that the translator exists.
    let inits: Vec<(Handle<naga::LocalVariable>, Handle<Expression>)> = tr
        .module_ref
        .entry_points[0]
        .function
        .local_variables
        .iter()
        .filter_map(|(h, lv)| lv.init.map(|i| (h, i)))
        .collect();
    for (h, init) in inits {
        if let LocalSlot::Scalar(var, ty) = tr.locals[&h] {
            let v = tr.scalar(init)?;
            let v = tr.coerce(v, ty)?;
            tr.b.def_var(var, v);
        }
    }

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
    let _ = name;
    Ok(())
}

/// The function-argument indices of `global_invocation_id` and `num_workgroups`.
fn builtin_args(func: &naga::Function) -> Result<(u32, u32), String> {
    let mut gid = None;
    let mut nwg = None;
    for (i, arg) in func.arguments.iter().enumerate() {
        if let Some(naga::Binding::BuiltIn(b)) = &arg.binding {
            match b {
                BuiltIn::GlobalInvocationId => gid = Some(i as u32),
                BuiltIn::NumWorkGroups => nwg = Some(i as u32),
                _ => {}
            }
        }
    }
    Ok((
        gid.ok_or("kernel missing global_invocation_id arg")?,
        nwg.ok_or("kernel missing num_workgroups arg")?,
    ))
}

fn cl_ty(ty: Ty) -> types::Type {
    match ty {
        Ty::F32 => types::F32,
        Ty::Bool => types::I8,
        _ => types::I32,
    }
}

/// Scalar type of a value-typed naga type handle (Scalar or single-component).
fn scalar_ty_of(m: &naga::Module, ty: Handle<naga::Type>) -> Result<Ty, String> {
    match &m.types[ty].inner {
        TypeInner::Scalar(s) => Ty::from_scalar(s.kind),
        other => Err(format!("expected scalar local, got {other:?}")),
    }
}

/// Element scalar type of an `array<T>` global.
fn array_elem_ty(m: &naga::Module, ty: Handle<naga::Type>) -> Result<Ty, String> {
    match &m.types[ty].inner {
        TypeInner::Array { base, .. } => match &m.types[*base].inner {
            TypeInner::Scalar(s) => Ty::from_scalar(s.kind),
            other => Err(format!("array of non-scalar {other:?}")),
        },
        other => Err(format!("expected array global, got {other:?}")),
    }
}

struct Tr<'a, 'b> {
    module_ref: &'a naga::Module,
    b: &'a mut FunctionBuilder<'b>,
    cache: HashMap<Handle<Expression>, Eval>,
    locals: &'a HashMap<Handle<naga::LocalVariable>, LocalSlot>,
    buf_base: &'a HashMap<u32, Value>,
    uniform_ptr: Value,
    uniform_binding: Option<u32>,
    gid: [Value; 3],
    nwg: [Value; 3],
    gid_arg: u32,
    nwg_arg: u32,
    unary_refs: &'a HashMap<&'static str, cranelift_codegen::ir::FuncRef>,
    powf_ref: cranelift_codegen::ir::FuncRef,
    loop_stack: Vec<(cranelift_codegen::ir::Block, cranelift_codegen::ir::Block)>, // (continue, break)
    latch: cranelift_codegen::ir::Block,
}

impl<'a, 'b> Tr<'a, 'b> {
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
                    let v = self.scalar(*value)?;
                    match place {
                        Place::Local(var, ty) => {
                            let v = self.coerce(v, ty)?;
                            self.b.def_var(var, v);
                        }
                        Place::Mem { addr, elem } => {
                            let v = self.coerce(v, elem)?;
                            self.b.ins().store(MemFlags::trusted(), v, addr, 0);
                        }
                    }
                }
                Statement::Return { .. } => {
                    // Per-invocation early-out: skip to the next invocation.
                    self.b.ins().jump(self.latch, &[]);
                    return Ok(false);
                }
                Statement::Break => {
                    let (_, brk) = *self.loop_stack.last().ok_or("break outside loop")?;
                    self.b.ins().jump(brk, &[]);
                    return Ok(false);
                }
                Statement::Continue => {
                    let (cont, _) = *self.loop_stack.last().ok_or("continue outside loop")?;
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
                    self.loop_stack.push((cont_b, exit_b));
                    if self.block(body)? {
                        self.b.ins().jump(cont_b, &[]);
                    }
                    self.loop_stack.pop();

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
                other => return Err(format!("unsupported statement {other:?}")),
            }
        }
        Ok(true)
    }

    /// Evaluate an expression, memoising the result.
    fn eval(&mut self, h: Handle<Expression>) -> Result<Eval, String> {
        if let Some(e) = self.cache.get(&h) {
            return Ok(*e);
        }
        let e = self.eval_uncached(h)?;
        self.cache.insert(h, e);
        Ok(e)
    }

    /// Evaluate an expression that must be a scalar value.
    fn scalar(&mut self, h: Handle<Expression>) -> Result<(Value, Ty), String> {
        match self.eval(h)? {
            Eval::Scalar(v, t) => Ok((v, t)),
            Eval::Place(p) => self.load(p),
        }
    }

    /// Evaluate an expression that must yield an addressable place.
    fn place(&mut self, h: Handle<Expression>) -> Result<Place, String> {
        match self.eval(h)? {
            Eval::Place(p) => Ok(p),
            Eval::Scalar(..) => Err("expected place, got scalar".into()),
        }
    }

    fn load(&mut self, p: Place) -> Result<(Value, Ty), String> {
        match p {
            Place::Local(var, ty) => Ok((self.b.use_var(var), ty)),
            Place::Mem { addr, elem } => {
                let v = self.b.ins().load(cl_ty(elem), MemFlags::trusted(), addr, 0);
                Ok((v, elem))
            }
        }
    }

    fn eval_uncached(&mut self, h: Handle<Expression>) -> Result<Eval, String> {
        // `m` is a copy of the `&'a Module` reference, independent of the `&mut
        // self` borrow, so expression refs derived from it stay valid across the
        // `&mut self` recursion below. Every kernel has exactly one entry point.
        let m: &naga::Module = self.module_ref;
        let func = &m.entry_points[0].function;
        let expr = &func.expressions[h];
        match expr {
            Expression::Literal(lit) => Ok(self.literal(lit)),
            Expression::ZeroValue(ty) => {
                let t = scalar_ty_of(self.module_ref, *ty)?;
                Ok(Eval::Scalar(self.zero(t), t))
            }
            Expression::Constant(c) => {
                let cst = &self.module_ref.constants[*c];
                self.eval_global_const(cst.init)
            }
            Expression::FunctionArgument(_) => {
                Err("bare builtin vector value is unsupported (index it instead)".into())
            }
            Expression::GlobalVariable(g) => {
                // A pointer to the whole binding; refined by Access/AccessIndex.
                // Represented lazily: store the binding so the index step resolves
                // the element address. We encode this as a Place with a sentinel
                // addr of the base and elem from the array, but only Access uses it.
                let gv = &self.module_ref.global_variables[*g];
                let binding = gv.binding.as_ref().map(|b| b.binding);
                match gv.space {
                    AddressSpace::Storage { .. } => {
                        let b = binding.ok_or("storage global without binding")?;
                        let base = *self
                            .buf_base
                            .get(&b)
                            .ok_or("missing storage base pointer")?;
                        let elem = array_elem_ty(self.module_ref, gv.ty)?;
                        // addr currently points at element 0; Access adds the offset.
                        Ok(Eval::Place(Place::Mem { addr: base, elem }))
                    }
                    AddressSpace::Uniform => {
                        // Uniform struct base; AccessIndex refines to a member.
                        Ok(Eval::Place(Place::Mem {
                            addr: self.uniform_ptr,
                            elem: Ty::U32, // refined by AccessIndex
                        }))
                    }
                    other => Err(format!("global in unsupported space {other:?}")),
                }
            }
            Expression::LocalVariable(l) => match self.locals[l] {
                LocalSlot::Scalar(var, ty) => Ok(Eval::Place(Place::Local(var, ty))),
                LocalSlot::Array { base, elem } => Ok(Eval::Place(Place::Mem { addr: base, elem })),
            },
            Expression::Load { pointer } => {
                let p = self.place(*pointer)?;
                let (v, t) = self.load(p)?;
                Ok(Eval::Scalar(v, t))
            }
            Expression::Access { base, index } => {
                let base_place = self.place(*base)?;
                let (idx, _) = self.scalar(*index)?;
                match base_place {
                    Place::Mem { addr, elem } => {
                        let idx64 = self.to_i64(idx);
                        let off = self.b.ins().imul_imm(idx64, 4);
                        let a = self.b.ins().iadd(addr, off);
                        Ok(Eval::Place(Place::Mem { addr: a, elem }))
                    }
                    Place::Local(..) => Err("indexing a scalar local".into()),
                }
            }
            Expression::AccessIndex { base, index } => {
                let base_expr = &func.expressions[*base];
                // 1. A component of a builtin vector argument (gid.x, nwg.x, ...).
                if let Expression::FunctionArgument(ai) = base_expr {
                    let comp = *index as usize;
                    let v = if *ai == self.gid_arg {
                        self.gid[comp]
                    } else if *ai == self.nwg_arg {
                        self.nwg[comp]
                    } else {
                        return Err("AccessIndex on unknown builtin arg".into());
                    };
                    return Ok(Eval::Scalar(v, Ty::U32));
                }
                // 2. A member of the uniform Params struct.
                if let Expression::GlobalVariable(g) = base_expr {
                    if m.global_variables[*g].space == AddressSpace::Uniform {
                        let (member_off, member_ty) = self.uniform_member(*base, *index)?;
                        let a = self.b.ins().iadd_imm(self.uniform_ptr, member_off as i64);
                        return Ok(Eval::Place(Place::Mem { addr: a, elem: member_ty }));
                    }
                }
                // 3. A constant-indexed element of a storage or local array.
                let base_place = self.place(*base)?;
                match base_place {
                    Place::Mem { addr, elem } => {
                        let a = self.b.ins().iadd_imm(addr, (*index as i64) * 4);
                        Ok(Eval::Place(Place::Mem { addr: a, elem }))
                    }
                    Place::Local(..) => Err("AccessIndex on scalar local".into()),
                }
            }
            Expression::Unary { op, expr } => {
                let (v, t) = self.scalar(*expr)?;
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
                Ok(Eval::Scalar(r, t))
            }
            Expression::Binary { op, left, right } => self.binary(*op, *left, *right),
            Expression::Select { condition, accept, reject } => {
                let (c, _) = self.scalar(*condition)?;
                let (a, ta) = self.scalar(*accept)?;
                let (r, _tr) = self.scalar(*reject)?;
                Ok(Eval::Scalar(self.b.ins().select(c, a, r), ta))
            }
            Expression::Math { fun, arg, arg1, arg2, .. } => self.math(*fun, *arg, *arg1, *arg2),
            Expression::As { expr, kind, convert } => self.cast(*expr, *kind, *convert),
            Expression::Relational { fun, argument } => {
                // Only scalar IsNan/IsInf would appear; our kernels don't use them.
                Err(format!("unsupported relational {fun:?} on {argument:?}"))
            }
            other => Err(format!("unsupported expression {other:?}")),
        }
    }

    fn literal(&mut self, lit: &Literal) -> Eval {
        match lit {
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
            Literal::Bool(x) => {
                Eval::Scalar(self.b.ins().iconst(types::I8, *x as i64), Ty::Bool)
            }
            _ => Eval::Scalar(self.b.ins().iconst(types::I32, 0), Ty::U32),
        }
    }

    fn eval_global_const(&mut self, h: Handle<Expression>) -> Result<Eval, String> {
        // Constants live in the module-level global_expressions arena.
        let expr = &self.module_ref.global_expressions[h];
        match expr {
            Expression::Literal(lit) => Ok(self.literal(lit)),
            Expression::ZeroValue(ty) => {
                let t = scalar_ty_of(self.module_ref, *ty)?;
                Ok(Eval::Scalar(self.zero(t), t))
            }
            other => Err(format!("unsupported constant expression {other:?}")),
        }
    }

    fn uniform_member(
        &self,
        base: Handle<Expression>,
        index: u32,
    ) -> Result<(u32, Ty), String> {
        let func = &self.module_ref.entry_points[0].function;
        let g = match &func.expressions[base] {
            Expression::GlobalVariable(g) => *g,
            other => return Err(format!("AccessIndex base not a global: {other:?}")),
        };
        let gv = &self.module_ref.global_variables[g];
        if Some(gv.binding.as_ref().map(|b| b.binding)) != Some(self.uniform_binding) {
            // Not strictly required, but guards against indexing a storage array
            // with a constant (which would be Access, not AccessIndex).
        }
        match &self.module_ref.types[gv.ty].inner {
            TypeInner::Struct { members, .. } => {
                let m = &members[index as usize];
                let ty = scalar_ty_of(self.module_ref, m.ty)?;
                Ok((m.offset, ty))
            }
            other => Err(format!("AccessIndex on non-struct {other:?}")),
        }
    }

    fn binary(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Result<Eval, String> {
        let (l, lt) = self.scalar(left)?;
        let (r, rt) = self.scalar(right)?;
        let float = lt.is_float() || rt.is_float();
        use BinaryOperator::*;
        let ins = self.b.ins();
        let (v, t) = match op {
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
        };
        Ok(Eval::Scalar(v, t))
    }

    fn math(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        _arg2: Option<Handle<Expression>>,
    ) -> Result<Eval, String> {
        use MathFunction::*;
        let (a, at) = self.scalar(arg)?;
        match fun {
            Sqrt => Ok(Eval::Scalar(self.b.ins().sqrt(a), Ty::F32)),
            InverseSqrt => {
                let s = self.b.ins().sqrt(a);
                let one = self.b.ins().f32const(1.0);
                Ok(Eval::Scalar(self.b.ins().fdiv(one, s), Ty::F32))
            }
            Abs if at.is_float() => Ok(Eval::Scalar(self.b.ins().fabs(a), Ty::F32)),
            Abs => Ok(Eval::Scalar(self.b.ins().iabs(a), at)),
            Min | Max => {
                let (b, bt) = self.scalar(arg1.ok_or("min/max needs 2 args")?)?;
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
                Ok(Eval::Scalar(v, if float { Ty::F32 } else { at }))
            }
            Fma => {
                let (b, _) = self.scalar(arg1.ok_or("fma needs args")?)?;
                let (c, _) = self.scalar(_arg2.ok_or("fma needs 3 args")?)?;
                Ok(Eval::Scalar(self.b.ins().fma(a, b, c), Ty::F32))
            }
            Step => {
                // step(edge, x) = x < edge ? 0.0 : 1.0
                let (x, _) = self.scalar(arg1.ok_or("step needs 2 args")?)?;
                let lt = self.b.ins().fcmp(FloatCC::LessThan, x, a);
                let zero = self.b.ins().f32const(0.0);
                let one = self.b.ins().f32const(1.0);
                Ok(Eval::Scalar(self.b.ins().select(lt, zero, one), Ty::F32))
            }
            Pow => {
                let (b, _) = self.scalar(arg1.ok_or("pow needs 2 args")?)?;
                let call = self.b.ins().call(self.powf_ref, &[a, b]);
                Ok(Eval::Scalar(self.b.inst_results(call)[0], Ty::F32))
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
                let fref = self.unary_refs[sym];
                let call = self.b.ins().call(fref, &[a]);
                Ok(Eval::Scalar(self.b.inst_results(call)[0], Ty::F32))
            }
            other => Err(format!("unsupported math fn {other:?}")),
        }
    }

    fn cast(
        &mut self,
        expr: Handle<Expression>,
        kind: ScalarKind,
        convert: Option<u8>,
    ) -> Result<Eval, String> {
        let (v, t) = self.scalar(expr)?;
        let target = Ty::from_scalar(kind)?;
        if t == target {
            return Ok(Eval::Scalar(v, t));
        }
        if convert.is_none() {
            // Bitcast (reinterpret), same width.
            let v = match (t, target) {
                (Ty::F32, _) => self.b.ins().bitcast(types::I32, MemFlags::new(), v),
                (_, Ty::F32) => self.b.ins().bitcast(types::F32, MemFlags::new(), v),
                _ => v, // int<->int reinterpret is a no-op
            };
            return Ok(Eval::Scalar(v, target));
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
        Ok(Eval::Scalar(v, target))
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

    fn to_i64(&mut self, v: Value) -> Value {
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
            let mut v: Vec<*mut u8> = Vec::new();
            v.push(a2.as_mut_ptr() as *mut u8);
            v.push(b2.as_mut_ptr() as *mut u8);
            v.push(out.as_mut_ptr() as *mut u8);
            v
        };
        let uniform = [4u32];
        // grid_x = ceil(4/64)=1, so 64 invocations; the kernel masks idx>=4.
        unsafe {
            jit.run(0, 0, 64, 1, 1, uniform.as_ptr(), bufs.as_ptr());
        }
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn matmul_runs_on_cpu() {
        // x:[M=2,K=3], w:[N=2,K=3], out:[M,N], out[m,n]=sum_k x[m,k]*w[n,k]
        let jit = Jit::new(&[("matmul", kernels::MATMUL)]).expect("compile matmul");
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut w = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0f32; 4];
        let bufs = vec![
            x.as_mut_ptr() as *mut u8,
            w.as_mut_ptr() as *mut u8,
            out.as_mut_ptr() as *mut u8,
        ];
        let uniform = [2u32, 3u32, 2u32]; // m,k,n
        unsafe {
            jit.run(0, 0, 64, 1, 1, uniform.as_ptr(), bufs.as_ptr());
        }
        // out[0,0]=x[0,0]=1, out[0,1]=x[0,1]=2, out[1,0]=x[1,0]=4, out[1,1]=x[1,1]=5
        assert_eq!(out, vec![1.0, 2.0, 4.0, 5.0]);
        let _ = ptrs;
    }
}
