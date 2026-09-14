// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Translate brain's WGSL compute kernels to CUDA C++ text - the **generated**
//! implementation tier (`ImplSource::Generated`).
//!
//! Swedish Embedded AB implements source-to-source compilers and kernel
//! retargeting for its clients - taking one authoritative kernel description
//! and making it execute correctly on hardware it was never written for. If
//! your team needs expertise in IR translation or in proving a generated
//! backend numerically equal to its reference, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! WGSL stays the single source of truth. This crate parses a kernel with the
//! same `naga` front-end the wgpu/vulkan/CPU paths use and walks the resulting
//! IR, emitting CUDA C++ text; a caller compiles that text (NVRTC or an
//! ahead-of-time cubin) and launches it. Nothing here touches a GPU, a driver
//! or a toolkit, so this crate builds and its tests run anywhere.
//!
//! # Execution model
//!
//! One WGSL workgroup is one CUDA block, so the mapping is direct and needs no
//! index rewriting:
//!
//! ```text
//! blockDim  = (workgroup_size, 1, 1)      local_invocation_id = threadIdx
//! gridDim   = (grid_x, grid_y, 1)         workgroup_id        = blockIdx
//!                                         num_workgroups      = gridDim
//! ```
//!
//! The generated entry point is `extern "C"` (so no name mangling to resolve)
//! and takes the uniform stream first, then one pointer per storage binding in
//! ascending binding order.
//!
//! # What this is NOT
//!
//! It is not a performance tier. A mechanically translated kernel inherits the
//! WGSL execution model - no warp intrinsics, no register blocking, no
//! vectorised loads - and reports itself as `Generated` so a hand-written
//! kernel it should have lost to cannot be quietly replaced by it.
//!
//! # Correctness hazards this emitter exists to handle
//!
//! Every way a shader-to-CUDA translation goes wrong produces a plausible
//! number, not a compile error, so each is handled explicitly and each has a
//! test that fails if the handling is removed:
//!
//! 1. **An early `return` before a barrier.** WGSL allows it when the
//!    predicate is workgroup-uniform, and this repo's reduction kernels rely on
//!    that. A thread that has genuinely returned can never arrive at
//!    `__syncthreads()`. Returns in a barrier-using kernel are therefore
//!    emitted as a guard flag, never as a `return` - see [`Gen::block`].
//! 2. **Uniform layout.** WGSL's 16-byte struct/vector alignment is not the
//!    C++ default, so no uniform struct is ever transliterated: each member is
//!    read at the byte offset naga's layouter computed.
//! 3. **Shift semantics.** The PTX ISA clamps a shift amount above the word
//!    width where the CPU reference masks it, so the mask is written out.
//! 4. **Multiply-add contraction** is the compiler caller's job (`--fmad=false`),
//!    but this emitter never writes `a*b+c` in a form that hides the two
//!    roundings, and emits WGSL's explicit `fma()` as `fmaf`.
//! 5. **`__restrict__` is never emitted.** brain's device buffers alias by
//!    design: clones share an allocation and a sliced step binds overlapping
//!    ranges of one buffer.
//! 6. **`var<workgroup>` zero-init.** WGSL zero-initialises workgroup memory;
//!    `__shared__` does not, so the zeroing and the barrier that publishes it
//!    are emitted at entry.

use std::collections::HashMap;
use std::fmt::Write as _;

use naga::{
    AddressSpace, BinaryOperator, Block, BuiltIn, Expression, Handle, Literal, MathFunction,
    Scalar, ScalarKind, Statement, TypeInner, UnaryOperator,
};

/// One WGSL kernel translated to CUDA C++.
#[derive(Debug)]
pub struct Kernel {
    /// The `extern "C" __global__` entry point name in [`Kernel::source`].
    pub entry: String,
    /// Self-contained CUDA C++ - no `#include`, so NVRTC needs no header path.
    pub source: String,
    /// Threads per block the kernel must be launched with. This is the WGSL
    /// `@workgroup_size`, not a tuning knob: the kernel's own index arithmetic
    /// and its `__shared__` reductions are written against it.
    pub block_dim: u32,
    /// Storage binding indices, ascending - the order the entry point takes
    /// its pointer arguments in, after the uniform stream.
    pub bindings: Vec<u32>,
    /// Size of the uniform `Params` block in bytes, under **WGSL** layout
    /// rules. 0 when the kernel declares no uniform.
    pub uniform_bytes: usize,
}

/// The `extern "C"` entry point name a kernel called `name` is emitted under.
///
/// Prefixed because a cubin's symbol table is flat and a kernel called `main`
/// would be indistinguishable from anything else called `main`.
pub fn entry_name(name: &str) -> String {
    let safe: String =
        name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    format!("brain_{safe}")
}

/// Parse `wgsl` and emit the CUDA C++ for its `main` entry point.
///
/// Anything outside the supported IR subset is an error naming what was found,
/// never a silently approximated translation.
pub fn generate(name: &str, wgsl: &str) -> Result<Kernel, String> {
    let m = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("WGSL parse: {e:?}"))?;
    let entry_point = m
        .entry_points
        .iter()
        .find(|e| e.name == "main")
        .ok_or("no `main` entry point")?;
    let func = &entry_point.function;

    if entry_point.workgroup_size[1] != 1 || entry_point.workgroup_size[2] != 1 {
        return Err(format!(
            "only 1-D @workgroup_size is supported, this kernel declares {:?}",
            entry_point.workgroup_size
        ));
    }
    let block_dim = entry_point.workgroup_size[0];
    if block_dim == 0 {
        return Err("@workgroup_size(0)".into());
    }

    let entry = entry_name(name);
    let has_barrier = block_has_barrier(&func.body);
    if has_barrier {
        check_barrier_structure(&func.body)?;
    }

    let mut g = Gen {
        m: &m,
        func,
        block_dim,
        guarded: has_barrier,
        decls: Vec::new(),
        cache: HashMap::new(),
        locals: HashMap::new(),
        globals: HashMap::new(),
        n_tmp: 0,
        n_loop: 0,
        loop_labels: Vec::new(),
        gid_arg: None,
        nwg_arg: None,
        lid_arg: None,
        wgid_arg: None,
    };
    let kernel = g.emit(&entry)?;

    // Post-condition for hazard 1, checked on the text that will actually be
    // compiled rather than on the intent that produced it: a barrier-using
    // kernel may not contain a `return` statement at all, because any return
    // reachable before a `__syncthreads()` strands the threads that did arrive.
    if has_barrier && kernel.source.contains("return;") {
        return Err(
            "a barrier-using kernel was emitted with a `return`, which threads that reach the \
             barrier would then wait on forever"
                .into(),
        );
    }
    Ok(kernel)
}

// ---------------------------------------------------------------------------
// Scalar types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    F32,
    U32,
    I32,
    Bool,
}

impl Ty {
    /// Takes the whole `naga::Scalar` (kind AND width). `ScalarKind` alone
    /// cannot tell f16 from f32 - both are `Float` - and emitting f16
    /// arithmetic as `float` is a silent precision LIE, not a rounding
    /// difference: a real f16 ALU saturates past 65504 where fp32 registers
    /// carry the value on. f16 is refused rather than approximated.
    fn from_scalar(s: Scalar) -> Result<Ty, String> {
        Ok(match (s.kind, s.width) {
            (ScalarKind::Float, 4) => Ty::F32,
            (ScalarKind::Float, 2) => {
                return Err("f16 (`enable f16;`) has no generated CUDA path and is refused \
                            rather than silently executed as fp32"
                    .into())
            }
            (ScalarKind::Uint, 4) => Ty::U32,
            (ScalarKind::Sint, 4) => Ty::I32,
            (ScalarKind::Bool, _) => Ty::Bool,
            _ => return Err(format!("unsupported scalar {s:?}")),
        })
    }
    fn c(self) -> &'static str {
        match self {
            Ty::F32 => "float",
            Ty::U32 => "unsigned int",
            Ty::I32 => "int",
            Ty::Bool => "bool",
        }
    }
    fn zero(self) -> &'static str {
        match self {
            Ty::F32 => "0.0f",
            Ty::U32 => "0u",
            Ty::I32 => "0",
            Ty::Bool => "false",
        }
    }
    fn is_float(self) -> bool {
        self == Ty::F32
    }
}

/// The result of translating a naga expression: a C++ value expression, or a
/// place that a `Load`/`Store`/`Access` refines.
#[derive(Clone)]
enum Eval {
    Value(String, Ty),
    Place(Place),
}

#[derive(Clone)]
enum Place {
    /// A scalar lvalue: a local variable, or an already-indexed array element.
    Lvalue(String, Ty),
    /// An array base (storage binding, workgroup scratch, or a local array)
    /// awaiting an index.
    ArrayBase(String, Ty),
    /// The uniform block. Read-only, and only ever refined by `AccessIndex`.
    UniformBase,
}

/// What a global variable became in the emitted source.
#[derive(Clone)]
enum GlobalKind {
    /// A kernel pointer parameter for a storage binding.
    Storage { ident: String, elem: Ty },
    /// A `__shared__` array.
    WorkGroup { ident: String, elem: Ty },
    /// The uniform block, read through byte offsets.
    Uniform,
}

// ---------------------------------------------------------------------------
// Structural checks (done before a single line is emitted)
// ---------------------------------------------------------------------------

fn block_has_barrier(b: &Block) -> bool {
    b.iter().any(|s| match s {
        Statement::ControlBarrier(_) => true,
        Statement::Block(inner) => block_has_barrier(inner),
        Statement::If { accept, reject, .. } => block_has_barrier(accept) || block_has_barrier(reject),
        Statement::Loop { body, continuing, .. } => {
            block_has_barrier(body) || block_has_barrier(continuing)
        }
        _ => false,
    })
}

/// Whether this block continues the loop it belongs to. Stops at a nested
/// `Loop`, whose own `continue`s belong to that inner loop instead.
fn block_has_continue(b: &Block) -> bool {
    b.iter().any(|s| match s {
        Statement::Continue => true,
        Statement::Block(inner) => block_has_continue(inner),
        Statement::If { accept, reject, .. } => {
            block_has_continue(accept) || block_has_continue(reject)
        }
        _ => false,
    })
}

fn block_has_return(b: &Block) -> bool {
    b.iter().any(|s| match s {
        Statement::Return { .. } => true,
        Statement::Block(inner) => block_has_return(inner),
        Statement::If { accept, reject, .. } => block_has_return(accept) || block_has_return(reject),
        Statement::Loop { body, continuing, .. } => {
            block_has_return(body) || block_has_return(continuing)
        }
        _ => false,
    })
}

/// The two shapes the guard transform cannot express, refused up front rather
/// than mistranslated.
///
/// * A barrier under `if`/`loop`: the guard the transform wraps the body in
///   would still be open around it, so the deactivated threads would not
///   arrive. (A barrier under NON-uniform control flow is invalid WGSL to
///   begin with; this refuses the uniform case too, because proving uniformity
///   is a separate analysis and guessing at it is how a deadlock ships.)
/// * A `return` inside a loop: turning that return into "stop doing work" is
///   exactly what the guard does, but it also stops the loop from ever
///   exiting through it, so a loop whose only exit was the return would spin.
fn check_barrier_structure(body: &Block) -> Result<(), String> {
    for s in body.iter() {
        match s {
            Statement::ControlBarrier(_) => {}
            Statement::Loop { body, continuing, .. } => {
                if block_has_barrier(body) || block_has_barrier(continuing) {
                    return Err("a barrier inside a loop is not supported by the generated tier; \
                                the guarded-body transform cannot keep the deactivated threads \
                                arriving at it"
                        .into());
                }
                if block_has_return(body) || block_has_return(continuing) {
                    return Err("a `return` inside a loop in a barrier-using kernel is not \
                                supported: the guarded-body transform turns it into a flag, \
                                which would leave the loop with no way out"
                        .into());
                }
            }
            Statement::If { accept, reject, .. } => {
                if block_has_barrier(accept) || block_has_barrier(reject) {
                    return Err("a barrier under an `if` is not supported by the generated tier; \
                                the threads the condition excludes would never arrive at it"
                        .into());
                }
                // A loop nested under an `if` still has to be checked for a
                // `return` the guard transform would turn into an infinite one.
                check_barrier_structure(accept)?;
                check_barrier_structure(reject)?;
            }
            Statement::Block(inner) => check_barrier_structure(inner)?,
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The emitter
// ---------------------------------------------------------------------------

struct Gen<'a> {
    m: &'a naga::Module,
    func: &'a naga::Function,
    block_dim: u32,
    /// This kernel contains a barrier, so `return` becomes a guard flag.
    guarded: bool,
    /// Declarations hoisted to the top of the function. Every temporary lives
    /// at function scope, so no `goto` can ever jump across an initialisation
    /// and no value is out of scope at a later use.
    decls: Vec<String>,
    cache: HashMap<Handle<Expression>, Eval>,
    locals: HashMap<Handle<naga::LocalVariable>, Place>,
    globals: HashMap<Handle<naga::GlobalVariable>, GlobalKind>,
    n_tmp: usize,
    /// Monotonic loop counter. The `continue` label is named from it and NEVER
    /// reused: two sibling loops that both took label 0 would emit the same
    /// label twice in one function, which is a redefinition error, and a
    /// `goto` from the second would jump into the first.
    n_loop: usize,
    /// Labels of the loops currently open, innermost last.
    loop_labels: Vec<usize>,
    gid_arg: Option<u32>,
    nwg_arg: Option<u32>,
    lid_arg: Option<u32>,
    wgid_arg: Option<u32>,
}

impl<'a> Gen<'a> {
    fn emit(&mut self, entry: &str) -> Result<Kernel, String> {
        for (i, arg) in self.func.arguments.iter().enumerate() {
            if let Some(naga::Binding::BuiltIn(b)) = &arg.binding {
                match b {
                    BuiltIn::GlobalInvocationId => self.gid_arg = Some(i as u32),
                    BuiltIn::NumWorkGroups => self.nwg_arg = Some(i as u32),
                    BuiltIn::LocalInvocationId => self.lid_arg = Some(i as u32),
                    BuiltIn::WorkGroupId => self.wgid_arg = Some(i as u32),
                    other => return Err(format!("unsupported builtin input {other:?}")),
                }
            } else {
                return Err("a compute entry point may only take builtin inputs".into());
            }
        }

        // Bindings and workgroup scratch, in a stable order.
        let mut bindings: Vec<(u32, String, Ty)> = Vec::new();
        let mut shared: Vec<(String, Ty, u32)> = Vec::new();
        let mut uniform_bytes = 0usize;
        for (h, gv) in self.m.global_variables.iter() {
            match gv.space {
                AddressSpace::Uniform => {
                    uniform_bytes = self.struct_size(gv.ty)?;
                    self.globals.insert(h, GlobalKind::Uniform);
                }
                AddressSpace::Storage { .. } => {
                    let b = gv.binding.as_ref().map(|b| b.binding).ok_or("storage without binding")?;
                    let elem = array_elem_ty(self.m, gv.ty)?;
                    let ident = format!("__b{b}");
                    bindings.push((b, ident.clone(), elem));
                    self.globals.insert(h, GlobalKind::Storage { ident, elem });
                }
                AddressSpace::WorkGroup => {
                    let (elem, count) = array_info(self.m, gv.ty)?;
                    let name = gv.name.clone().unwrap_or_else(|| format!("wg{}", shared.len()));
                    let ident = format!("__wg_{}", ident_of(&name));
                    shared.push((ident.clone(), elem, count));
                    self.globals.insert(h, GlobalKind::WorkGroup { ident, elem });
                }
                other => return Err(format!("unsupported address space {other:?}")),
            }
        }
        bindings.sort_by_key(|(b, _, _)| *b);

        // Locals. WGSL zero-initialises both scalars and arrays, so the
        // declaration does too - a generated kernel may not inherit C++'s
        // "whatever was there" for a value the source language defined.
        // The index suffix is not decoration: WGSL scopes locals per block, so
        // two `var s` in sibling blocks are distinct variables with the same
        // name, and every local here is declared once at function scope.
        for (i, (h, lv)) in self.func.local_variables.iter().enumerate() {
            let name = lv.name.clone().unwrap_or_default();
            let ident = format!("__l{i}_{}", ident_of(&name));
            match &self.m.types[lv.ty].inner {
                TypeInner::Array { .. } => {
                    let (elem, count) = array_info(self.m, lv.ty)?;
                    self.decls.push(format!("{} {ident}[{count}] = {{}};", elem.c()));
                    self.locals.insert(h, Place::ArrayBase(ident, elem));
                }
                TypeInner::Scalar(s) => {
                    let ty = Ty::from_scalar(*s)?;
                    self.decls.push(format!("{} {ident} = {};", ty.c(), ty.zero()));
                    self.locals.insert(h, Place::Lvalue(ident, ty));
                }
                other => return Err(format!("unsupported local type {other:?}")),
            }
        }

        let mut body = String::new();

        // Builtins, named once so the kernel's own index arithmetic reproduces
        // exactly what the WGSL dispatch would have handed it.
        let bd = self.block_dim;
        if self.gid_arg.is_some() {
            let _ = writeln!(body, "  const unsigned int __gid_x = blockIdx.x * {bd}u + threadIdx.x;");
            let _ = writeln!(body, "  const unsigned int __gid_y = blockIdx.y + threadIdx.y;");
            let _ = writeln!(body, "  const unsigned int __gid_z = blockIdx.z + threadIdx.z;");
        }
        if self.nwg_arg.is_some() {
            let _ = writeln!(body, "  const unsigned int __nwg_x = gridDim.x;");
            let _ = writeln!(body, "  const unsigned int __nwg_y = gridDim.y;");
            let _ = writeln!(body, "  const unsigned int __nwg_z = gridDim.z;");
        }
        if self.lid_arg.is_some() {
            let _ = writeln!(body, "  const unsigned int __lid_x = threadIdx.x;");
            let _ = writeln!(body, "  const unsigned int __lid_y = threadIdx.y;");
            let _ = writeln!(body, "  const unsigned int __lid_z = threadIdx.z;");
        }
        if self.wgid_arg.is_some() {
            let _ = writeln!(body, "  const unsigned int __wgid_x = blockIdx.x;");
            let _ = writeln!(body, "  const unsigned int __wgid_y = blockIdx.y;");
            let _ = writeln!(body, "  const unsigned int __wgid_z = blockIdx.z;");
        }

        // Hazard 6: WGSL zero-initialises `var<workgroup>`; `__shared__` keeps
        // whatever the previously resident block left in it. Zero it with the
        // whole block, unconditionally (before any guard), and publish it with
        // a barrier - a kernel whose tail lanes never write their slot reads
        // zeros under WGSL and reads a stale partial sum without this.
        for (ident, elem, count) in &shared {
            let _ = writeln!(body, "  __shared__ {} {ident}[{count}];", elem.c());
        }
        if !shared.is_empty() {
            for (ident, elem, count) in &shared {
                let _ = writeln!(
                    body,
                    "  for (unsigned int __z = threadIdx.x; __z < {count}u; __z += blockDim.x) {ident}[__z] = {};",
                    elem.zero()
                );
            }
            let _ = writeln!(body, "  __syncthreads();");
        }

        if self.guarded {
            let _ = writeln!(
                body,
                "  bool __active = true; // hazard 1: a return before a barrier is a flag, never a return"
            );
        }

        // Local initialisers, after the builtins they may read.
        let inits: Vec<(Handle<naga::LocalVariable>, Handle<Expression>)> = self
            .func
            .local_variables
            .iter()
            .filter_map(|(h, lv)| lv.init.map(|i| (h, i)))
            .collect();
        for (h, init) in inits {
            let place = self.locals[&h].clone();
            let (v, vt) = self.value(init, &mut body)?;
            if let Place::Lvalue(ident, ty) = place {
                let v = coerce(&v, vt, ty)?;
                let _ = writeln!(body, "  {ident} = {v};");
            } else {
                return Err("an array local with an initialiser is unsupported".into());
            }
        }

        // `self.func` is a shared reference with its own lifetime, so copying
        // it out lets the body be walked while `self` is borrowed mutably -
        // no clone of the statement tree.
        let f: &naga::Function = self.func;
        self.block(&f.body, &mut body, 1)?;

        // Assemble.
        let mut params: Vec<String> = Vec::new();
        if uniform_bytes > 0 {
            params.push("const unsigned int* __params".to_string());
        }
        for (_, ident, elem) in &bindings {
            // Hazard 5: no `__restrict__`. brain's DeviceBuffer clones alias by
            // design and a sliced step binds overlapping ranges of one buffer,
            // so the promise would be false at exactly the call sites that
            // matter. Nor is any binding `const`: a read-only binding aliasing
            // a written one would otherwise be a type-level lie too.
            params.push(format!("{}* {ident}", elem.c()));
        }

        let mut source = String::new();
        source.push_str(PREAMBLE);
        let _ = writeln!(source, "extern \"C\" __global__ void {entry}({})", params.join(", "));
        let _ = writeln!(source, "{{");
        for d in &self.decls {
            let _ = writeln!(source, "  {d}");
        }
        source.push_str(&body);
        let _ = writeln!(source, "}}");

        Ok(Kernel {
            entry: entry.to_string(),
            source,
            block_dim: self.block_dim,
            bindings: bindings.iter().map(|(b, _, _)| *b).collect(),
            uniform_bytes,
        })
    }

    // -- statements ---------------------------------------------------------

    /// Translate a statement block at `indent`, returning whether control may
    /// have left it (a `return`/`break`/`continue` in the unguarded form).
    ///
    /// # Hazard 1: the guarded-body transform
    ///
    /// In a barrier-using kernel (`self.guarded`) a WGSL `return` becomes
    /// `__active = false` and every subsequent statement is wrapped in
    /// `if (__active)`, EXCEPT the barriers, which every thread must still
    /// reach. The wrapper is closed before a barrier and reopened after it, so
    /// the emitted shape is
    ///
    /// ```text
    /// if (cond) { __active = false; }
    /// if (__active) { ... }
    /// __syncthreads();
    /// if (__active) { ... }
    /// ```
    ///
    /// which is what WGSL's own rule ("the predicate is workgroup-uniform")
    /// means operationally, written down instead of assumed.
    fn block(&mut self, b: &Block, out: &mut String, depth: usize) -> Result<Flow, String> {
        let pad = "  ".repeat(depth);
        let mut open = false;
        let mut may_deactivate = false;
        let mut terminated = false;

        for s in b.iter() {
            if let Statement::ControlBarrier(_) = s {
                if open {
                    let _ = writeln!(out, "{pad}}}");
                    open = false;
                }
                let _ = writeln!(out, "{pad}__syncthreads();");
                continue;
            }
            if self.guarded && may_deactivate && !open {
                let _ = writeln!(out, "{pad}if (__active) {{");
                open = true;
            }
            let inner = if open { depth + 1 } else { depth };
            let pad = "  ".repeat(inner);
            // Whether THIS statement may have cleared the guard flag. If it
            // did, the wrapper it ran under (entered while the flag was still
            // set) has to be closed, so the statements after it re-test the
            // flag instead of inheriting a decision taken before it dropped.
            let mut deactivated_here = false;

            match s {
                Statement::Emit(range) => {
                    for h in range.clone() {
                        self.emit_expr(h, out, inner)?;
                    }
                }
                Statement::Block(nested) => {
                    let _ = writeln!(out, "{pad}{{");
                    let f = self.block(nested, out, inner + 1)?;
                    let _ = writeln!(out, "{pad}}}");
                    deactivated_here |= f.may_deactivate;
                    if f.terminated && !self.guarded {
                        terminated = true;
                    }
                }
                Statement::Store { pointer, value } => {
                    let place = self.place(*pointer, out, inner)?;
                    let (v, vt) = self.value(*value, out)?;
                    match place {
                        Place::Lvalue(lv, ty) => {
                            let v = coerce(&v, vt, ty)?;
                            let _ = writeln!(out, "{pad}{lv} = {v};");
                        }
                        _ => return Err("store to a non-scalar place".into()),
                    }
                }
                Statement::Return { value: Some(_) } => {
                    return Err("a compute entry point cannot return a value".into())
                }
                Statement::Return { .. } => {
                    if self.guarded {
                        let _ = writeln!(out, "{pad}__active = false;");
                        deactivated_here = true;
                    } else {
                        let _ = writeln!(out, "{pad}return;");
                        terminated = true;
                    }
                }
                Statement::Break => {
                    let _ = writeln!(out, "{pad}break;");
                    terminated = true;
                }
                Statement::Continue => {
                    let label = *self.loop_labels.last().ok_or("continue outside a loop")?;
                    // Not C's `continue`: WGSL's continuing block must still run.
                    let _ = writeln!(out, "{pad}goto __cont_{label};");
                    terminated = true;
                }
                Statement::If { condition, accept, reject } => {
                    let (c, _) = self.value(*condition, out)?;
                    let _ = writeln!(out, "{pad}if ({c}) {{");
                    let fa = self.block(accept, out, inner + 1)?;
                    let fr = if reject.is_empty() {
                        let _ = writeln!(out, "{pad}}}");
                        Flow::default()
                    } else {
                        let _ = writeln!(out, "{pad}}} else {{");
                        let f = self.block(reject, out, inner + 1)?;
                        let _ = writeln!(out, "{pad}}}");
                        f
                    };
                    deactivated_here |= fa.may_deactivate | fr.may_deactivate;
                }
                Statement::Loop { body, continuing, break_if } => {
                    let label = self.n_loop;
                    self.n_loop += 1;
                    self.loop_labels.push(label);
                    let _ = writeln!(out, "{pad}while (true) {{");
                    let f = self.block(body, out, inner + 1)?;
                    deactivated_here |= f.may_deactivate;
                    // The label exists only for a `continue`, which is NOT C's
                    // `continue`: WGSL's continuing block has to run on that
                    // path too, and C would skip it. Emitted only when
                    // something jumps to it, so no unused label is generated.
                    if block_has_continue(body) {
                        let _ = writeln!(out, "{pad}  __cont_{label}: ;");
                    }
                    let fc = self.block(continuing, out, inner + 1)?;
                    deactivated_here |= fc.may_deactivate;
                    if let Some(cond) = break_if {
                        let (c, _) = self.value(*cond, out)?;
                        let _ = writeln!(out, "{pad}  if ({c}) break;");
                    }
                    let _ = writeln!(out, "{pad}}}");
                    self.loop_labels.pop();
                }
                Statement::MemoryBarrier(_) => {
                    let _ = writeln!(out, "{pad}__threadfence_block();");
                }
                other => return Err(format!("unsupported statement {other:?}")),
            }

            if deactivated_here {
                may_deactivate = true;
                if open {
                    let _ = writeln!(out, "{}}}", "  ".repeat(depth));
                    open = false;
                }
            }
            if terminated && !self.guarded {
                break;
            }
        }
        if open {
            let _ = writeln!(out, "{}}}", "  ".repeat(depth));
        }
        Ok(Flow { terminated, may_deactivate })
    }

    // -- expressions --------------------------------------------------------

    /// Materialise an `Emit`ted expression into a function-scope temporary, so
    /// its evaluation ORDER is the source's. Inlining the text instead would
    /// let a load float past a store to the same location and read the wrong
    /// value - the same reason the Cranelift path materialises at `Emit`.
    fn emit_expr(&mut self, h: Handle<Expression>, out: &mut String, depth: usize) -> Result<(), String> {
        if self.cache.contains_key(&h) {
            return Ok(());
        }
        let e = self.eval(h, out, depth)?;
        match e {
            Eval::Value(text, ty) => {
                let name = format!("__e{}", self.n_tmp);
                self.n_tmp += 1;
                self.decls.push(format!("{} {name};", ty.c()));
                let _ = writeln!(out, "{}{name} = {text};", "  ".repeat(depth));
                self.cache.insert(h, Eval::Value(name, ty));
            }
            place => {
                self.cache.insert(h, place);
            }
        }
        Ok(())
    }

    fn eval(&mut self, h: Handle<Expression>, out: &mut String, depth: usize) -> Result<Eval, String> {
        if let Some(e) = self.cache.get(&h) {
            return Ok(e.clone());
        }
        let e = self.eval_uncached(h, out, depth)?;
        self.cache.insert(h, e.clone());
        Ok(e)
    }

    /// An expression that must be a value.
    fn value(&mut self, h: Handle<Expression>, out: &mut String) -> Result<(String, Ty), String> {
        match self.eval(h, out, 1)? {
            Eval::Value(v, t) => Ok((v, t)),
            Eval::Place(Place::Lvalue(lv, t)) => Ok((lv, t)),
            Eval::Place(_) => Err("expected a value, got an unindexed array or the uniform block".into()),
        }
    }

    fn place(&mut self, h: Handle<Expression>, out: &mut String, depth: usize) -> Result<Place, String> {
        match self.eval(h, out, depth)? {
            Eval::Place(p) => Ok(p),
            Eval::Value(..) => Err("expected a place, got a value".into()),
        }
    }

    fn eval_uncached(
        &mut self,
        h: Handle<Expression>,
        out: &mut String,
        depth: usize,
    ) -> Result<Eval, String> {
        let func = self.func;
        let expr = &func.expressions[h];
        match expr {
            Expression::Literal(lit) => Ok(literal(lit)?),
            Expression::ZeroValue(ty) => {
                let t = scalar_ty_of(self.m, *ty)?;
                Ok(Eval::Value(t.zero().to_string(), t))
            }
            Expression::Constant(c) => {
                let init = self.m.constants[*c].init;
                match &self.m.global_expressions[init] {
                    Expression::Literal(lit) => literal(lit),
                    Expression::ZeroValue(ty) => {
                        let t = scalar_ty_of(self.m, *ty)?;
                        Ok(Eval::Value(t.zero().to_string(), t))
                    }
                    other => Err(format!("unsupported constant expression {other:?}")),
                }
            }
            Expression::GlobalVariable(g) => match self.globals.get(g).cloned() {
                Some(GlobalKind::Storage { ident, elem }) => Ok(Eval::Place(Place::ArrayBase(ident, elem))),
                Some(GlobalKind::WorkGroup { ident, elem }) => {
                    Ok(Eval::Place(Place::ArrayBase(ident, elem)))
                }
                Some(GlobalKind::Uniform) => Ok(Eval::Place(Place::UniformBase)),
                None => Err("global variable in an unsupported address space".into()),
            },
            Expression::LocalVariable(l) => Ok(Eval::Place(self.locals[l].clone())),
            Expression::Load { pointer } => match self.eval(*pointer, out, depth)? {
                Eval::Place(Place::Lvalue(lv, t)) => Ok(Eval::Value(lv, t)),
                // A uniform member is not addressable in the emitted source -
                // it is read out of the byte stream - so its "pointer" already
                // evaluated to the loaded value.
                Eval::Value(v, t) => Ok(Eval::Value(v, t)),
                Eval::Place(_) => Err("load from an unindexed array or the uniform block".into()),
            },
            Expression::Access { base, index } => {
                let b = self.place(*base, out, depth)?;
                let (idx, _) = self.value(*index, out)?;
                match b {
                    // `size_t` rather than the shader's u32: the index itself
                    // stays u32 (so the source's own wrapping arithmetic is
                    // preserved) and only the final address computation widens,
                    // which is what a 64-bit device pointer needs.
                    Place::ArrayBase(base, elem) => {
                        Ok(Eval::Place(Place::Lvalue(format!("{base}[(size_t)({idx})]"), elem)))
                    }
                    _ => Err("indexing something that is not an array".into()),
                }
            }
            Expression::AccessIndex { base, index } => {
                let base_expr = &func.expressions[*base];
                // A component of a builtin vector input.
                if let Expression::FunctionArgument(ai) = base_expr {
                    let comp = ["x", "y", "z"]
                        .get(*index as usize)
                        .ok_or("builtin vector component out of range")?;
                    let name = if Some(*ai) == self.gid_arg {
                        "__gid"
                    } else if Some(*ai) == self.nwg_arg {
                        "__nwg"
                    } else if Some(*ai) == self.lid_arg {
                        "__lid"
                    } else if Some(*ai) == self.wgid_arg {
                        "__wgid"
                    } else {
                        return Err("component of an unknown builtin argument".into());
                    };
                    return Ok(Eval::Value(format!("{name}_{comp}"), Ty::U32));
                }
                // Hazard 2: a uniform member is read at the offset WGSL's
                // layout rules put it at, never at a C++ struct's.
                let b = self.place(*base, out, depth)?;
                match b {
                    Place::UniformBase => {
                        let g = match base_expr {
                            Expression::GlobalVariable(g) => *g,
                            other => return Err(format!("uniform access on {other:?}")),
                        };
                        let (off, ty) = self.uniform_member(g, *index)?;
                        let f = match ty {
                            Ty::F32 => "__brain_uf32",
                            Ty::I32 => "__brain_ui32",
                            Ty::U32 => "__brain_uu32",
                            Ty::Bool => return Err("a bool uniform member is unsupported".into()),
                        };
                        Ok(Eval::Value(format!("{f}(__params, {off}u)"), ty))
                    }
                    Place::ArrayBase(base, elem) => {
                        Ok(Eval::Place(Place::Lvalue(format!("{base}[{index}]"), elem)))
                    }
                    Place::Lvalue(..) => Err("AccessIndex on a scalar".into()),
                }
            }
            Expression::Unary { op, expr } => {
                let (v, t) = self.value(*expr, out)?;
                let r = match op {
                    UnaryOperator::Negate if t.is_float() => format!("(-({v}))"),
                    // Signed negation written through unsigned so the wrap WGSL
                    // defines is not C++'s signed-overflow UB.
                    UnaryOperator::Negate => format!("((int)(0u - (unsigned int)({v})))"),
                    UnaryOperator::LogicalNot => format!("(!({v}))"),
                    UnaryOperator::BitwiseNot => format!("(~({v}))"),
                };
                Ok(Eval::Value(r, t))
            }
            Expression::Binary { op, left, right } => {
                let (l, lt) = self.value(*left, out)?;
                let (r, rt) = self.value(*right, out)?;
                binary(*op, &l, lt, &r, rt)
            }
            Expression::Select { condition, accept, reject } => {
                let (c, _) = self.value(*condition, out)?;
                let (a, at) = self.value(*accept, out)?;
                let (r, _) = self.value(*reject, out)?;
                Ok(Eval::Value(format!("(({c}) ? ({a}) : ({r}))"), at))
            }
            Expression::Math { fun, arg, arg1, arg2, .. } => {
                let (a, at) = self.value(*arg, out)?;
                let mut rest = Vec::new();
                for h in [arg1, arg2].into_iter().flatten() {
                    rest.push(self.value(*h, out)?);
                }
                math(*fun, (&a, at), &rest)
            }
            Expression::As { expr, kind, convert } => {
                let (v, t) = self.value(*expr, out)?;
                cast(&v, t, *kind, *convert)
            }
            other => Err(format!("unsupported expression {other:?}")),
        }
    }

    /// `(byte offset, scalar type)` of uniform struct member `index`, taken
    /// from naga's own layout of the WGSL type - never from C++ packing.
    fn uniform_member(&self, g: Handle<naga::GlobalVariable>, index: u32) -> Result<(u32, Ty), String> {
        let gv = &self.m.global_variables[g];
        match &self.m.types[gv.ty].inner {
            TypeInner::Struct { members, .. } => {
                let mem = members
                    .get(index as usize)
                    .ok_or_else(|| format!("uniform member {index} out of range"))?;
                let ty = scalar_ty_of(self.m, mem.ty).map_err(|e| {
                    format!("uniform member {index} is not a scalar and cannot be read: {e}")
                })?;
                Ok((mem.offset, ty))
            }
            other => Err(format!("uniform block is not a struct: {other:?}")),
        }
    }

    fn struct_size(&self, ty: Handle<naga::Type>) -> Result<usize, String> {
        match &self.m.types[ty].inner {
            TypeInner::Struct { span, .. } => Ok(*span as usize),
            other => Err(format!("uniform block is not a struct: {other:?}")),
        }
    }
}

/// What leaving a statement block may have done.
#[derive(Default, Clone, Copy)]
struct Flow {
    /// Control left the block (`return`/`break`/`continue`), so the statements
    /// after it in the same block are unreachable.
    terminated: bool,
    /// A guarded `return` may have cleared `__active`, so what follows has to
    /// be re-guarded.
    may_deactivate: bool,
}

// ---------------------------------------------------------------------------
// Scalar-level translation
// ---------------------------------------------------------------------------

fn literal(lit: &Literal) -> Result<Eval, String> {
    Ok(match lit {
        Literal::F32(x) => Eval::Value(f32_text(*x)?, Ty::F32),
        Literal::F64(x) => Eval::Value(f32_text(*x as f32)?, Ty::F32),
        Literal::AbstractFloat(x) => Eval::Value(f32_text(*x as f32)?, Ty::F32),
        Literal::U32(x) => Eval::Value(format!("{x}u"), Ty::U32),
        // Through the unsigned bit pattern: `-2147483648` has no spelling as a
        // C++ int literal (it parses as a negated long).
        Literal::I32(x) => Eval::Value(format!("((int){}u)", *x as u32), Ty::I32),
        Literal::AbstractInt(x) => Eval::Value(format!("((int){}u)", *x as i32 as u32), Ty::I32),
        Literal::Bool(x) => Eval::Value(x.to_string(), Ty::Bool),
        other => return Err(format!("unsupported literal {other:?}")),
    })
}

/// A float literal that reads back as the same bits.
///
/// Rust's `{:?}` prints the shortest decimal that round-trips through f32, and
/// the result is re-parsed here to prove it: a constant that drifts by one ulp
/// on the way into the generated source is exactly the kind of difference a
/// parity assertion would later blame on the hardware.
fn f32_text(x: f32) -> Result<String, String> {
    if !x.is_finite() {
        return Ok(if x.is_nan() {
            "__int_as_float(0x7fc00000)".to_string()
        } else if x > 0.0 {
            "__int_as_float(0x7f800000)".to_string()
        } else {
            "__int_as_float(0xff800000)".to_string()
        });
    }
    let s = format!("{x:?}");
    if s.parse::<f32>() != Ok(x) {
        return Err(format!("float literal {x} does not round-trip as {s:?}"));
    }
    Ok(format!("{s}f"))
}

fn binary(op: BinaryOperator, l: &str, lt: Ty, r: &str, rt: Ty) -> Result<Eval, String> {
    use BinaryOperator::*;
    let float = lt.is_float() || rt.is_float();
    let int_wrap = |o: &str| format!("((int)((unsigned int)({l}) {o} (unsigned int)({r})))");
    let v = match op {
        // Signed arithmetic goes through unsigned: WGSL defines i32 overflow as
        // wrapping, C++ calls it undefined, and an optimiser is entitled to act
        // on the difference.
        Add if float || lt != Ty::I32 => format!("(({l}) + ({r}))"),
        Add => int_wrap("+"),
        Subtract if float || lt != Ty::I32 => format!("(({l}) - ({r}))"),
        Subtract => int_wrap("-"),
        Multiply if float || lt != Ty::I32 => format!("(({l}) * ({r}))"),
        Multiply => int_wrap("*"),
        Divide => format!("(({l}) / ({r}))"),
        Modulo if float => return Err("float modulo is unsupported".into()),
        Modulo => format!("(({l}) % ({r}))"),
        Equal => format!("(({l}) == ({r}))"),
        NotEqual => format!("(({l}) != ({r}))"),
        Less => format!("(({l}) < ({r}))"),
        LessEqual => format!("(({l}) <= ({r}))"),
        Greater => format!("(({l}) > ({r}))"),
        GreaterEqual => format!("(({l}) >= ({r}))"),
        And => format!("(({l}) & ({r}))"),
        InclusiveOr => format!("(({l}) | ({r}))"),
        ExclusiveOr => format!("(({l}) ^ ({r}))"),
        LogicalAnd => format!("(({l}) && ({r}))"),
        LogicalOr => format!("(({l}) || ({r}))"),
        // Hazard 3: the PTX ISA CLAMPS a shift amount above the word width
        // (x << 32 == 0); x86, and therefore the reference this tier is
        // compared against, MASKS it (x << 32 == x). WGSL calls the range
        // indeterminate, so neither target's habit may be inherited - the mask
        // is written out so both sides compute the same thing.
        ShiftLeft if lt == Ty::I32 => {
            format!("((int)((unsigned int)({l}) << (({r}) & 31u)))")
        }
        ShiftLeft => format!("(({l}) << (({r}) & 31u))"),
        ShiftRight => format!("(({l}) >> (({r}) & 31u))"),
    };
    let ty = match op {
        Equal | NotEqual | Less | LessEqual | Greater | GreaterEqual | LogicalAnd | LogicalOr => {
            Ty::Bool
        }
        _ if float => Ty::F32,
        _ => lt,
    };
    Ok(Eval::Value(v, ty))
}

fn math(fun: MathFunction, a: (&str, Ty), rest: &[(String, Ty)]) -> Result<Eval, String> {
    use MathFunction::*;
    let (x, xt) = a;
    let arg = |i: usize| -> Result<&str, String> {
        rest.get(i).map(|(s, _)| s.as_str()).ok_or_else(|| format!("{fun:?} needs more arguments"))
    };
    let v = match fun {
        Sqrt => (format!("sqrtf({x})"), Ty::F32),
        // 1/sqrt, not `rsqrtf`: the fast reciprocal square root is a different
        // (approximate) function, and this tier is held to the reference's
        // value, not to a faster one.
        InverseSqrt => (format!("(1.0f / sqrtf({x}))"), Ty::F32),
        Abs if xt.is_float() => (format!("fabsf({x})"), Ty::F32),
        Abs => (format!("abs({x})"), xt),
        Min if xt.is_float() => (format!("fminf({x}, {})", arg(0)?), Ty::F32),
        Min => (format!("min({x}, {})", arg(0)?), xt),
        Max if xt.is_float() => (format!("fmaxf({x}, {})", arg(0)?), Ty::F32),
        Max => (format!("max({x}, {})", arg(0)?), xt),
        // The spec's own composition order, which is what the reference
        // computes; `min(max(..))` and `max(min(..))` differ on NaN.
        Clamp if xt.is_float() => {
            (format!("fminf(fmaxf({x}, {}), {})", arg(0)?, arg(1)?), Ty::F32)
        }
        Clamp => (format!("min(max({x}, {}), {})", arg(0)?, arg(1)?), xt),
        Saturate => (format!("fminf(fmaxf({x}, 0.0f), 1.0f)"), Ty::F32),
        // WGSL's explicit fma() IS a fused multiply-add and stays one. Hazard 4
        // is about the CONTRACTION of a written-out `a*b+c`, which the caller
        // disables at the compiler; it is not a licence to drop a real fma.
        Fma => (format!("fmaf({x}, {}, {})", arg(0)?, arg(1)?), Ty::F32),
        Mix => {
            // e1*(1-e3) + e2*e3, the spec form - not the algebraically equal
            // `e1 + e3*(e2-e1)`, which rounds differently.
            let (b, t) = (arg(0)?, arg(1)?);
            (format!("(({x}) * (1.0f - ({t})) + ({b}) * ({t}))"), Ty::F32)
        }
        Step => {
            let e = arg(0)?;
            (format!("((({e}) < ({x})) ? 0.0f : 1.0f)"), Ty::F32)
        }
        Sign if xt.is_float() => {
            (format!("((({x}) < 0.0f) ? -1.0f : ((({x}) > 0.0f) ? 1.0f : 0.0f))"), Ty::F32)
        }
        Sign => (format!("((({x}) < 0) ? -1 : ((({x}) > 0) ? 1 : 0))"), Ty::I32),
        Pow => (format!("powf({x}, {})", arg(0)?), Ty::F32),
        Exp => (format!("expf({x})"), Ty::F32),
        Log => (format!("logf({x})"), Ty::F32),
        Exp2 => (format!("exp2f({x})"), Ty::F32),
        Log2 => (format!("log2f({x})"), Ty::F32),
        Sin => (format!("sinf({x})"), Ty::F32),
        Cos => (format!("cosf({x})"), Ty::F32),
        Tanh => (format!("tanhf({x})"), Ty::F32),
        Floor => (format!("floorf({x})"), Ty::F32),
        Ceil => (format!("ceilf({x})"), Ty::F32),
        Trunc => (format!("truncf({x})"), Ty::F32),
        // WGSL round() is round-half-to-EVEN. `rintf` is that under the
        // default rounding mode; `roundf` is round-half-away-from-zero and
        // would disagree on every exact .5.
        Round => (format!("rintf({x})"), Ty::F32),
        Fract => (format!("(({x}) - floorf({x}))"), Ty::F32),
        Dot4I8Packed => (format!("__brain_dot4i8({x}, {})", arg(0)?), Ty::I32),
        other => return Err(format!("unsupported math function {other:?}")),
    };
    Ok(Eval::Value(v.0, v.1))
}

fn cast(v: &str, from: Ty, kind: ScalarKind, convert: Option<u8>) -> Result<Eval, String> {
    let to = Ty::from_scalar(Scalar { kind, width: convert.unwrap_or(4) })?;
    if from == to {
        return Ok(Eval::Value(v.to_string(), to));
    }
    let text = match convert {
        // A reinterpretation, not a conversion: the bits stay put.
        None => match (from, to) {
            (Ty::F32, Ty::U32) => format!("__float_as_uint({v})"),
            (Ty::F32, Ty::I32) => format!("__float_as_int({v})"),
            (Ty::U32, Ty::F32) => format!("__uint_as_float({v})"),
            (Ty::I32, Ty::F32) => format!("__int_as_float({v})"),
            (Ty::U32, Ty::I32) => format!("((int)({v}))"),
            (Ty::I32, Ty::U32) => format!("((unsigned int)({v}))"),
            _ => return Err(format!("unsupported bitcast {from:?} -> {to:?}")),
        },
        Some(_) => match (from, to) {
            (Ty::U32, Ty::F32) | (Ty::I32, Ty::F32) | (Ty::Bool, Ty::F32) => format!("((float)({v}))"),
            (Ty::F32, Ty::U32) | (Ty::I32, Ty::U32) | (Ty::Bool, Ty::U32) => {
                format!("((unsigned int)({v}))")
            }
            (Ty::F32, Ty::I32) | (Ty::U32, Ty::I32) | (Ty::Bool, Ty::I32) => format!("((int)({v}))"),
            (_, Ty::Bool) => format!("(({v}) != {})", from.zero()),
            _ => return Err(format!("unsupported conversion {from:?} -> {to:?}")),
        },
    };
    Ok(Eval::Value(text, to))
}

/// Store-side coercion: the int retags and the bool widening that come out of
/// tracking WGSL's types through C++ expressions.
fn coerce(v: &str, from: Ty, want: Ty) -> Result<String, String> {
    if from == want {
        return Ok(v.to_string());
    }
    match (from, want) {
        (Ty::U32, Ty::I32) => Ok(format!("((int)({v}))")),
        (Ty::I32, Ty::U32) => Ok(format!("((unsigned int)({v}))")),
        (Ty::Bool, Ty::U32) => Ok(format!("((unsigned int)({v}))")),
        (Ty::Bool, Ty::I32) => Ok(format!("((int)({v}))")),
        _ => Err(format!("cannot store {from:?} into {want:?}")),
    }
}

// ---------------------------------------------------------------------------
// naga type helpers
// ---------------------------------------------------------------------------

fn scalar_ty_of(m: &naga::Module, ty: Handle<naga::Type>) -> Result<Ty, String> {
    match &m.types[ty].inner {
        TypeInner::Scalar(s) => Ty::from_scalar(*s),
        other => Err(format!("expected a scalar type, got {other:?}")),
    }
}

fn array_elem_ty(m: &naga::Module, ty: Handle<naga::Type>) -> Result<Ty, String> {
    match &m.types[ty].inner {
        TypeInner::Array { base, .. } => scalar_ty_of(m, *base),
        other => Err(format!("expected an array binding, got {other:?}")),
    }
}

fn array_info(m: &naga::Module, ty: Handle<naga::Type>) -> Result<(Ty, u32), String> {
    match &m.types[ty].inner {
        TypeInner::Array { base, size, .. } => {
            let elem = scalar_ty_of(m, *base)?;
            match size {
                naga::ArraySize::Constant(n) => Ok((elem, n.get())),
                other => Err(format!("a fixed size is required here, got {other:?}")),
            }
        }
        other => Err(format!("expected an array, got {other:?}")),
    }
}

fn ident_of(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

/// Emitted ahead of every kernel. Self-contained: NVRTC compiles with no
/// include path, and a generated kernel must not depend on one existing.
const PREAMBLE: &str = r#"// Generated from WGSL by brain's wgsl-cuda (the T0 tier). Do not edit.
//
// Hazard 2: the uniform block is read through explicit byte offsets taken from
// the WGSL layout, never as a transliterated C++ struct.
__device__ __forceinline__ unsigned int __brain_uu32(const unsigned int* p, unsigned int off) {
  return p[off >> 2u];
}
__device__ __forceinline__ int __brain_ui32(const unsigned int* p, unsigned int off) {
  return (int)p[off >> 2u];
}
__device__ __forceinline__ float __brain_uf32(const unsigned int* p, unsigned int off) {
  return __uint_as_float(p[off >> 2u]);
}

// WGSL dot4I8Packed: four SIGNED int8 lanes multiplied and accumulated. Written
// out rather than emitted as __dp4a because this tier must be valid on every
// compute capability, and because the sign extension is where a transliteration
// goes wrong: a zero-extended lane is correct for every positive byte and wrong
// for every negative one. The masking form below is exact and has no
// implementation-defined shift in it.
__device__ __forceinline__ int __brain_dot4i8(unsigned int a, unsigned int b) {
  int acc = 0;
  for (int i = 0; i < 4; ++i) {
    int x = (int)((a >> (unsigned int)(i * 8)) & 0xffu);
    int y = (int)((b >> (unsigned int)(i * 8)) & 0xffu);
    if (x > 127) x -= 256;
    if (y > 127) y -= 256;
    acc += x * y;
  }
  return acc;
}

"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole covered subset must translate. A generator that only handles
    /// the kernel a golden test happens to run is not a tier.
    #[test]
    fn the_covered_subset_translates() {
        for (name, src) in [
            ("add2", kernels::ADD2),
            ("mul", kernels::MUL),
            ("gelu", kernels::GELU),
            ("add_inplace", kernels::ADD_INPLACE),
            ("quant_group_sum", kernels::QUANT_GROUP_SUM),
            ("gradnorm_part", kernels::GRADNORM_PART),
        ] {
            let k = generate(name, src).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(k.block_dim, 64);
            assert!(k.source.contains(&k.entry), "{name}: entry point missing from its own source");
        }
    }

    /// Hazard 1, at the text level: a barrier-using kernel carries a guard flag
    /// and no `return`, and the barrier itself sits outside the guard.
    #[test]
    fn a_barrier_kernel_guards_instead_of_returning() {
        let k = generate("gradnorm_part", kernels::GRADNORM_PART).expect("generate");
        assert!(!k.source.contains("return;"), "a return survived into a barrier kernel:\n{}", k.source);
        assert!(k.source.contains("__active = false;"), "no guard flag was emitted:\n{}", k.source);
        let sync = k.source.find("__syncthreads();").expect("barrier");
        let line_start = k.source[..sync].rfind('\n').map(|i| i + 1).unwrap_or(0);
        assert_eq!(
            k.source[line_start..sync].trim(),
            "",
            "the barrier must be a statement of its own, outside any guard"
        );
    }

    /// Hazard 6, at the text level: `__shared__` is zeroed and published before
    /// the body runs.
    #[test]
    fn workgroup_memory_is_zeroed_before_use() {
        let k = generate("gradnorm_part", kernels::GRADNORM_PART).expect("generate");
        let decl = k.source.find("__shared__").expect("shared declaration");
        let zero = k.source.find("__z += blockDim.x").expect("zeroing loop");
        let sync = k.source.find("__syncthreads();").expect("barrier");
        assert!(decl < zero && zero < sync, "the zeroing must precede the first barrier");
    }

    /// Hazard 5: never, for any kernel.
    #[test]
    fn no_kernel_promises_its_pointers_do_not_alias() {
        for (name, src) in [("add2", kernels::ADD2), ("add_inplace", kernels::ADD_INPLACE)] {
            let k = generate(name, src).expect("generate");
            assert!(!k.source.contains("__restrict"), "{name} declared __restrict__");
        }
    }

    /// The structural refusals are refusals, not silent mistranslations.
    #[test]
    fn unsupported_barrier_structures_are_refused() {
        const BARRIER_IN_LOOP: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
var<workgroup> s: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) li: vec3<u32>) {
    for (var i = 0u; i < p.n; i = i + 1u) {
        s[li.x] = f32(i);
        workgroupBarrier();
        o[li.x] = s[0];
    }
}
"#;
        let e = generate("barrier_in_loop", BARRIER_IN_LOOP).expect_err("must be refused");
        assert!(e.contains("barrier inside a loop"), "{e}");
    }

    /// f16 is refused rather than executed as fp32.
    #[test]
    fn f16_is_refused() {
        const SRC: &str = r#"
enable f16;
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    o[gid.x] = f16(1.0);
}
"#;
        let e = generate("half", SRC).expect_err("f16 must be refused");
        assert!(e.contains("f16"), "{e}");
    }
}

