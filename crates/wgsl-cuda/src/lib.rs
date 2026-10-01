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
//! # Limitations
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

mod uniform;
mod vector;

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
    // A kernel is "guarded" (every `return` becomes a flag that skips the rest
    // of the body, so threads that finished still arrive at later barriers)
    // only when some `return` is reached by part of the workgroup. A return the
    // whole workgroup takes together strands nobody, and a barrier under
    // uniform control flow is reached by everyone, so neither needs the
    // transform.
    let uniformity = uniform::analyse(&m, func);
    if has_barrier && uniformity.has_barrier_in_non_uniform_flow && !uniformity.has_non_uniform_return {
        return Err("a barrier is reached under non-uniform control flow (a branch or loop exit \
                    that depends on the thread id or on per-thread data), so threads that skip \
                    it would leave the others waiting forever"
            .into());
    }
    let guarded = has_barrier && uniformity.has_non_uniform_return;
    if guarded {
        check_barrier_structure(&func.body)?;
    }

    let mut g = Gen {
        m: &m,
        func,
        block_dim,
        guarded,
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
        lidx_arg: None,
        frames: Vec::new(),
        n_call: 0,
    };
    let kernel = g.emit(&entry)?;

    // Post-condition for hazard 1, checked on the text that will actually be
    // compiled rather than on the intent that produced it: a guarded
    // barrier-using kernel may not contain a `return` statement at all, because
    // any return reachable before a `__syncthreads()` strands the threads that
    // did arrive.
    if guarded && kernel.source.contains("return;") {
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

/// Component `c` of a vector whose first component is the lvalue `base`.
fn vec_comp(base: &str, c: usize) -> String {
    format!("(&{base})[{c}u]")
}

/// Component names, for identifiers: `x y z w`.
const COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];

/// The result of translating a naga expression: a C++ value expression, or a
/// place that a `Load`/`Store`/`Access` refines.
#[derive(Clone)]
enum Eval {
    Value(String, Ty),
    /// A vector value, scalarised: one expression per component. The generator
    /// has no CUDA vector type and no operator overloads to rely on, so every
    /// vector operation is the scalar operation applied lane by lane, which is
    /// also exactly how WGSL defines them.
    Vector(Vec<String>, Ty),
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
    /// A vector lvalue: a reference to its first component, the others lying
    /// `ty`-sized words after it. A vector local is a small array and an
    /// element of an array of vectors is a window into the array, so in both
    /// cases a component can be chosen at run time (`v[i]` with `i` a variable)
    /// as well as at translation time.
    VecRef { base: String, n: u32, ty: Ty },
    /// An array whose elements are vectors, awaiting an index. `stride` is the
    /// element's size in 32-bit words, which WGSL rounds `vec3` up to four.
    VecArrayBase { ident: String, elem: Ty, n: u32, stride: u32 },
}

/// What a global variable became in the emitted source.
#[derive(Clone)]
enum GlobalKind {
    /// A kernel pointer parameter for a storage binding. `vec` is
    /// `(components, stride in words)` when the array's elements are vectors.
    Storage { ident: String, elem: Ty, vec: Option<(u32, u32)> },
    /// A `__shared__` array.
    WorkGroup { ident: String, elem: Ty, vec: Option<(u32, u32)> },
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
    /// `@builtin(local_invocation_index)`, a scalar: the flat thread index in
    /// the workgroup, which for the 1-D blocks this tier requires is `threadIdx.x`.
    lidx_arg: Option<u32>,
    /// Calls being inlined, innermost last. Empty while the entry point's own
    /// body is being translated.
    frames: Vec<CallFrame>,
    /// Monotonic call counter: every inlined call gets its own labels, locals
    /// and result slots.
    n_call: usize,
}

/// What translating the body of an inlined function needs to know.
struct CallFrame {
    /// The caller's argument values, by parameter index.
    args: Vec<Eval>,
    /// The label a `return` jumps to, after the inlined body.
    label: usize,
    /// Where a returned value goes: one slot for a scalar, one per component
    /// for a vector; empty for a function that returns nothing.
    slots: Vec<String>,
    /// The type of the returned scalar or vector's components.
    ret_ty: Ty,
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
                    BuiltIn::LocalInvocationIndex => self.lidx_arg = Some(i as u32),
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
                    let layout = array_layout(self.m, gv.ty)?;
                    let (elem, vec) = (layout.elem, layout.vec);
                    let ident = format!("__b{b}");
                    bindings.push((b, ident.clone(), elem));
                    self.globals.insert(h, GlobalKind::Storage { ident, elem, vec });
                }
                AddressSpace::WorkGroup => {
                    let (elem, count) = array_info(self.m, gv.ty)?;
                    let vec = array_layout(self.m, gv.ty)?.vec;
                    let name = gv.name.clone().unwrap_or_else(|| format!("wg{}", shared.len()));
                    let ident = format!("__wg_{}", ident_of(&name));
                    shared.push((ident.clone(), elem, count));
                    self.globals.insert(h, GlobalKind::WorkGroup { ident, elem, vec });
                }
                other => return Err(format!("unsupported address space {other:?}")),
            }
        }
        bindings.sort_by_key(|(b, _, _)| *b);

        self.locals = self.declare_locals(self.func, "")?;

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
        if self.lidx_arg.is_some() {
            let _ = writeln!(body, "  const unsigned int __lidx = threadIdx.x;");
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
        self.init_locals(self.func, &mut body)?;

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
                    match place {
                        Place::Lvalue(lv, ty) => {
                            let (v, vt) = self.value(*value, out)?;
                            let v = coerce(&v, vt, ty)?;
                            let _ = writeln!(out, "{pad}{lv} = {v};");
                        }
                        Place::VecRef { base, n, ty } => {
                            let v = self.lanes(*value, out)?;
                            if !v.vector || v.comps.len() != n as usize {
                                return Err("a vector store needs a vector of the same width".into());
                            }
                            for (c, text) in v.comps.iter().enumerate() {
                                let text = coerce(text, v.ty, ty)?;
                                let _ = writeln!(out, "{pad}{} = {text};", vec_comp(&base, c));
                            }
                        }
                        _ => return Err("store to a non-scalar place".into()),
                    }
                }
                Statement::Return { value } if !self.frames.is_empty() => {
                    let (label, slots, ret_ty) = {
                        let f = self.frames.last().expect("a frame is active");
                        (f.label, f.slots.clone(), f.ret_ty)
                    };
                    if let Some(v) = value {
                        let lanes = self.lanes(*v, out)?;
                        if lanes.comps.len() != slots.len() {
                            return Err("a returned value does not match the function's result type".into());
                        }
                        for (slot, text) in slots.iter().zip(&lanes.comps) {
                            let text = coerce(text, lanes.ty, ret_ty)?;
                            let _ = writeln!(out, "{pad}{slot} = {text};");
                        }
                    }
                    let _ = writeln!(out, "{pad}goto __ret_{label};");
                    terminated = true;
                }
                Statement::Return { value: Some(_) } => {
                    return Err("a compute entry point cannot return a value".into())
                }
                Statement::Call { function, arguments, result } => {
                    self.inline_call(*function, arguments, *result, out, inner)?;
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

    /// Declare `func`'s local variables at function scope and return where each
    /// one lives. WGSL zero-initialises both scalars and arrays, so the
    /// declaration does too - a generated kernel may not inherit C++'s
    /// "whatever was there" for a value the source language defined. The index
    /// is not decoration: WGSL scopes locals per block, so two `var s` in
    /// sibling blocks are distinct variables with the same name. `tag`
    /// separates an inlined function's locals from the entry point's.
    fn declare_locals(
        &mut self,
        func: &naga::Function,
        tag: &str,
    ) -> Result<HashMap<Handle<naga::LocalVariable>, Place>, String> {
        let mut locals = HashMap::new();
        for (i, (h, lv)) in func.local_variables.iter().enumerate() {
            let name = lv.name.clone().unwrap_or_default();
            let ident = format!("__l{tag}{i}_{}", ident_of(&name));
            match &self.m.types[lv.ty].inner {
                TypeInner::Array { .. } => {
                    let (elem, count) = array_info(self.m, lv.ty)?;
                    self.decls.push(format!("{} {ident}[{count}] = {{}};", elem.c()));
                    let place = match array_layout(self.m, lv.ty)?.vec {
                        Some((n, stride)) => Place::VecArrayBase { ident, elem, n, stride },
                        None => Place::ArrayBase(ident, elem),
                    };
                    locals.insert(h, place);
                }
                TypeInner::Scalar(s) => {
                    let ty = Ty::from_scalar(*s)?;
                    self.decls.push(format!("{} {ident} = {};", ty.c(), ty.zero()));
                    locals.insert(h, Place::Lvalue(ident, ty));
                }
                TypeInner::Vector { size, scalar } => {
                    let ty = Ty::from_scalar(*scalar)?;
                    let n = *size as u32;
                    self.decls.push(format!("{} {ident}[{n}] = {{}};", ty.c()));
                    locals.insert(h, Place::VecRef { base: format!("{ident}[0]"), n, ty });
                }
                other => return Err(format!("unsupported local type {other:?}")),
            }
        }
        Ok(locals)
    }

    /// Emit the initialisers of `func`'s locals, which read `self.func`'s
    /// expressions: the caller swaps `self.func` first when `func` is a callee.
    fn init_locals(&mut self, func: &naga::Function, out: &mut String) -> Result<(), String> {
        let inits: Vec<(Handle<naga::LocalVariable>, Handle<Expression>)> =
            func.local_variables.iter().filter_map(|(h, lv)| lv.init.map(|i| (h, i))).collect();
        for (h, init) in inits {
            let place = self.locals[&h].clone();
            match place {
                Place::Lvalue(ident, ty) => {
                    let (v, vt) = self.value(init, out)?;
                    let v = coerce(&v, vt, ty)?;
                    let _ = writeln!(out, "  {ident} = {v};");
                }
                Place::VecRef { base, n, ty } => {
                    let v = self.lanes(init, out)?;
                    if !v.vector || v.comps.len() != n as usize {
                        return Err("a vector local needs a vector initialiser of the same width".into());
                    }
                    for (c, text) in v.comps.iter().enumerate() {
                        let text = coerce(text, v.ty, ty)?;
                        let _ = writeln!(out, "  {} = {text};", vec_comp(&base, c));
                    }
                }
                _ => return Err("an array local with an initialiser is unsupported".into()),
            }
        }
        Ok(())
    }

    /// Translate a call to a user function by inlining its body.
    ///
    /// Inlining rather than a `__device__` function because these helpers read
    /// the kernel's own storage bindings and workgroup arrays, which a separate
    /// C++ function would have to be handed one by one. The body is wrapped in
    /// a block, each `return` becomes "store the value, jump to the label after
    /// the block", and every local, temporary and label gets a name no other
    /// call can reuse. A helper may not contain a barrier (it could be reached
    /// by a subset of the workgroup, which the uniformity analysis does not
    /// follow into).
    fn inline_call(
        &mut self,
        function: Handle<naga::Function>,
        arguments: &[Handle<Expression>],
        result: Option<Handle<Expression>>,
        out: &mut String,
        depth: usize,
    ) -> Result<(), String> {
        let callee: &'a naga::Function = &self.m.functions[function];
        if block_has_barrier(&callee.body) {
            return Err("a barrier inside a called function is not supported by the generated tier".into());
        }
        if self.frames.len() >= 16 {
            return Err("function calls nested more than 16 deep (recursion is not valid WGSL)".into());
        }
        // The arguments are the CALLER's expressions: evaluate them before the
        // callee's own expression arena replaces it.
        let mut args = Vec::with_capacity(arguments.len());
        for a in arguments {
            args.push(self.eval(*a, out, depth)?);
        }
        let id = self.n_call;
        self.n_call += 1;

        let (slots, ret_ty) = match &callee.result {
            None => (Vec::new(), Ty::U32),
            Some(r) => match vector_shape(self.m, r.ty) {
                Some(shape) => {
                    let (t, n, _) = shape?;
                    ((0..n as usize).map(|c| format!("__call{id}_r{c}")).collect(), t)
                }
                None => (vec![format!("__call{id}_r")], scalar_ty_of(self.m, r.ty)?),
            },
        };
        for slot in &slots {
            self.decls.push(format!("{} {slot};", ret_ty.c()));
        }

        let saved_func = std::mem::replace(&mut self.func, callee);
        let saved_cache = std::mem::take(&mut self.cache);
        let saved_locals = std::mem::take(&mut self.locals);
        self.frames.push(CallFrame { args, label: id, slots: slots.clone(), ret_ty });

        let body = (|| -> Result<(), String> {
            self.locals = self.declare_locals(callee, &format!("c{id}_"))?;
            let pad = "  ".repeat(depth);
            let _ = writeln!(out, "{pad}{{");
            self.init_locals(callee, out)?;
            self.block(&callee.body, out, depth + 1)?;
            let _ = writeln!(out, "{pad}}}");
            let _ = writeln!(out, "{pad}__ret_{id}: ;");
            Ok(())
        })();

        self.frames.pop();
        self.func = saved_func;
        self.cache = saved_cache;
        self.locals = saved_locals;
        body?;

        if let Some(re) = result {
            let eval = match slots.len() {
                0 => return Err("a call result was used but the function returns nothing".into()),
                1 if !callee.result.as_ref().is_some_and(|r| vector_shape(self.m, r.ty).is_some()) => {
                    Eval::Value(slots[0].clone(), ret_ty)
                }
                _ => Eval::Vector(slots, ret_ty),
            };
            self.cache.insert(re, eval);
        }
        Ok(())
    }

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
            Eval::Vector(texts, ty) => {
                let mut names = Vec::with_capacity(texts.len());
                for (c, text) in texts.iter().enumerate() {
                    let name = format!("__e{}_{}", self.n_tmp, COMPONENTS[c]);
                    self.decls.push(format!("{} {name};", ty.c()));
                    let _ = writeln!(out, "{}{name} = {text};", "  ".repeat(depth));
                    names.push(name);
                }
                self.n_tmp += 1;
                self.cache.insert(h, Eval::Vector(names, ty));
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
            Eval::Vector(..) | Eval::Place(Place::VecRef { .. }) => Err("expected a scalar, got a vector".into()),
            Eval::Place(_) => Err("expected a value, got an unindexed array or the uniform block".into()),
        }
    }

    /// An operand that may be a scalar or a vector, as its component
    /// expressions. The lane-wise operations in [`vector`] take these.
    fn lanes(&mut self, h: Handle<Expression>, out: &mut String) -> Result<vector::Lanes, String> {
        match self.eval(h, out, 1)? {
            Eval::Value(v, t) => Ok(vector::Lanes::scalar(v, t)),
            Eval::Place(Place::Lvalue(lv, t)) => Ok(vector::Lanes::scalar(lv, t)),
            Eval::Vector(c, t) => Ok(vector::Lanes::vector(c, t)),
            Eval::Place(Place::VecRef { base, n, ty }) => Ok(vector::Lanes::vector((0..n as usize).map(|c| vec_comp(&base, c)).collect(), ty)),
            Eval::Place(_) => Err("expected a value, got an unindexed array or the uniform block".into()),
        }
    }

    fn place(&mut self, h: Handle<Expression>, out: &mut String, depth: usize) -> Result<Place, String> {
        match self.eval(h, out, depth)? {
            Eval::Place(p) => Ok(p),
            Eval::Value(..) | Eval::Vector(..) => Err("expected a place, got a value".into()),
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
            Expression::ZeroValue(ty) => self.zero_value(*ty),
            Expression::Constant(c) => self.constant(self.m.constants[*c].init),
            Expression::GlobalVariable(g) => match self.globals.get(g).cloned() {
                Some(GlobalKind::Storage { ident, elem, vec }) | Some(GlobalKind::WorkGroup { ident, elem, vec }) => {
                    Ok(Eval::Place(match vec {
                        Some((n, stride)) => Place::VecArrayBase { ident, elem, n, stride },
                        None => Place::ArrayBase(ident, elem),
                    }))
                }
                Some(GlobalKind::Uniform) => Ok(Eval::Place(Place::UniformBase)),
                None => Err("global variable in an unsupported address space".into()),
            },
            Expression::LocalVariable(l) => Ok(Eval::Place(self.locals[l].clone())),
            // Inside an inlined function an argument is the caller's value.
            Expression::FunctionArgument(ai) if !self.frames.is_empty() => {
                let f = self.frames.last().expect("a frame is active");
                f.args.get(*ai as usize).cloned().ok_or_else(|| "a function argument out of range".to_string())
            }
            Expression::FunctionArgument(ai) if Some(*ai) == self.lidx_arg => Ok(Eval::Value("__lidx".to_string(), Ty::U32)),
            Expression::Load { pointer } => match self.eval(*pointer, out, depth)? {
                Eval::Place(Place::Lvalue(lv, t)) => Ok(Eval::Value(lv, t)),
                // A uniform member is not addressable in the emitted source -
                // it is read out of the byte stream - so its "pointer" already
                // evaluated to the loaded value.
                Eval::Value(v, t) => Ok(Eval::Value(v, t)),
                Eval::Place(Place::VecRef { base, n, ty }) => {
                    Ok(Eval::Vector((0..n as usize).map(|c| vec_comp(&base, c)).collect(), ty))
                }
                Eval::Vector(c, t) => Ok(Eval::Vector(c, t)),
                Eval::Place(_) => Err("load from an unindexed array or the uniform block".into()),
            },
            Expression::Access { base, index } => {
                // A component of a vector VALUE chosen at run time: a select
                // over the components, since a value has no address.
                if let Eval::Vector(comps, t) = self.eval(*base, out, depth)? {
                    let (idx, _) = self.value(*index, out)?;
                    let mut chain = comps[comps.len() - 1].clone();
                    for (c, text) in comps.iter().enumerate().take(comps.len() - 1).rev() {
                        chain = format!("((({idx}) == {c}u) ? ({text}) : ({chain}))");
                    }
                    return Ok(Eval::Value(chain, t));
                }
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
                    // An element of an array of vectors: its components are
                    // `stride` words apart, starting at element `idx`.
                    Place::VecArrayBase { ident, elem, n, stride } => Ok(Eval::Place(Place::VecRef {
                        base: format!("{ident}[(size_t)({idx}) * {stride}u]"),
                        n,
                        ty: elem,
                    })),
                    // A component chosen at run time: components are
                    // consecutive, so the index is a pointer offset.
                    Place::VecRef { base, ty, .. } => {
                        Ok(Eval::Place(Place::Lvalue(format!("(&{base})[(size_t)({idx})]"), ty)))
                    }
                    _ => Err("indexing something that is not an array".into()),
                }
            }
            Expression::AccessIndex { base, index } => {
                let base_expr = &func.expressions[*base];
                // A component of a builtin vector input.
                if let (Expression::FunctionArgument(ai), true) = (base_expr, self.frames.is_empty()) {
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
                let b = self.eval(*base, out, depth)?;
                match b {
                    // A component of a vector value or a vector lvalue.
                    Eval::Vector(comps, t) => Ok(Eval::Value(self.component(&comps, *index)?, t)),
                    Eval::Place(Place::VecRef { base, n, ty }) => {
                        if *index >= n {
                            return Err(format!("vector component {index} out of range"));
                        }
                        Ok(Eval::Place(Place::Lvalue(vec_comp(&base, *index as usize), ty)))
                    }
                    Eval::Place(Place::VecArrayBase { ident, elem, n, stride }) => Ok(Eval::Place(Place::VecRef {
                        base: format!("{ident}[(size_t)({index}u) * {stride}u]"),
                        n,
                        ty: elem,
                    })),
                    Eval::Place(Place::UniformBase) => {
                        let g = match base_expr {
                            Expression::GlobalVariable(g) => *g,
                            other => return Err(format!("uniform access on {other:?}")),
                        };
                        let (off, ty, lanes) = self.uniform_member(g, *index)?;
                        let read = |at: u32| -> Result<String, String> {
                            let f = match ty {
                                Ty::F32 => "__brain_uf32",
                                Ty::I32 => "__brain_ui32",
                                Ty::U32 => "__brain_uu32",
                                Ty::Bool => return Err("a bool uniform member is unsupported".into()),
                            };
                            Ok(format!("{f}(__params, {at}u)"))
                        };
                        if lanes == 1 {
                            Ok(Eval::Value(read(off)?, ty))
                        } else {
                            // A vector member: its components are consecutive
                            // 32-bit words from the member's own offset.
                            Ok(Eval::Vector((0..lanes).map(|c| read(off + 4 * c)).collect::<Result<_, _>>()?, ty))
                        }
                    }
                    Eval::Place(Place::ArrayBase(base, elem)) => {
                        Ok(Eval::Place(Place::Lvalue(format!("{base}[{index}]"), elem)))
                    }
                    Eval::Place(Place::Lvalue(..)) | Eval::Value(..) => Err("AccessIndex on a scalar".into()),
                }
            }
            Expression::Unary { op, expr } => {
                let x = self.lanes(*expr, out)?;
                let op = *op;
                Ok(vector::unary_lanes(
                    |v, t| match op {
                        UnaryOperator::Negate if t.is_float() => format!("(-({v}))"),
                        // Signed negation written through unsigned so the wrap WGSL
                        // defines is not C++'s signed-overflow UB.
                        UnaryOperator::Negate => format!("((int)(0u - (unsigned int)({v})))"),
                        UnaryOperator::LogicalNot => format!("(!({v}))"),
                        UnaryOperator::BitwiseNot => format!("(~({v}))"),
                    },
                    &x,
                ))
            }
            Expression::Binary { op, left, right } => {
                let l = self.lanes(*left, out)?;
                let r = self.lanes(*right, out)?;
                vector::binary_lanes(*op, &l, &r)
            }
            Expression::Select { condition, accept, reject } => {
                let c = self.lanes(*condition, out)?;
                let a = self.lanes(*accept, out)?;
                let r = self.lanes(*reject, out)?;
                vector::select_lanes(&c, &a, &r)
            }
            Expression::Compose { ty, components } => {
                let Some(shape) = vector_shape(self.m, *ty) else {
                    return Err("only vectors can be composed (structs and matrices are unsupported)".into());
                };
                let (t, n, _) = shape?;
                let mut comps = Vec::with_capacity(n as usize);
                for c in components {
                    comps.extend(self.lanes(*c, out)?.comps);
                }
                if comps.len() != n as usize {
                    return Err(format!("a vec{n} composed from {} components", comps.len()));
                }
                Ok(Eval::Vector(comps, t))
            }
            Expression::Splat { size, value } => {
                let v = self.lanes(*value, out)?;
                Ok(Eval::Vector(vec![v.comps[0].clone(); *size as usize], v.ty))
            }
            Expression::Swizzle { size, vector, pattern } => {
                let v = self.lanes(*vector, out)?;
                let pick = |c: &naga::SwizzleComponent| *c as usize;
                let comps = pattern[..*size as usize].iter().map(|c| self.component(&v.comps, pick(c) as u32)).collect::<Result<_, _>>()?;
                Ok(Eval::Vector(comps, v.ty))
            }
            Expression::Relational { fun, argument } => {
                let a = self.lanes(*argument, out)?;
                match fun {
                    naga::RelationalFunction::All => Ok(vector::reduce_bool(true, &a)),
                    naga::RelationalFunction::Any => Ok(vector::reduce_bool(false, &a)),
                    other => Err(format!("unsupported relational function {other:?}")),
                }
            }
            Expression::Math { fun, arg, arg1, arg2, .. } => {
                let mut args = vec![self.lanes(*arg, out)?];
                for h in [arg1, arg2].into_iter().flatten() {
                    args.push(self.lanes(*h, out)?);
                }
                vector::math_lanes(*fun, &args)
            }
            Expression::As { expr, kind, convert } => {
                let x = self.lanes(*expr, out)?;
                vector::cast_lanes(&x, *kind, *convert)
            }
            other => Err(format!("unsupported expression {other:?}")),
        }
    }

    /// `(byte offset, scalar type)` of uniform struct member `index`, taken
    /// from naga's own layout of the WGSL type - never from C++ packing.
    fn uniform_member(&self, g: Handle<naga::GlobalVariable>, index: u32) -> Result<(u32, Ty, u32), String> {
        let gv = &self.m.global_variables[g];
        match &self.m.types[gv.ty].inner {
            TypeInner::Struct { members, .. } => {
                let mem = members
                    .get(index as usize)
                    .ok_or_else(|| format!("uniform member {index} out of range"))?;
                if let Some(shape) = vector_shape(self.m, mem.ty) {
                    let (ty, n, _) = shape?;
                    return Ok((mem.offset, ty, n));
                }
                let ty = scalar_ty_of(self.m, mem.ty).map_err(|e| {
                    format!("uniform member {index} is neither a scalar nor a vector and cannot be read: {e}")
                })?;
                Ok((mem.offset, ty, 1))
            }
            other => Err(format!("uniform block is not a struct: {other:?}")),
        }
    }

    /// Component `index` of a vector's component list.
    fn component(&self, comps: &[String], index: u32) -> Result<String, String> {
        comps.get(index as usize).cloned().ok_or_else(|| format!("vector component {index} out of range"))
    }

    /// The zero value of a scalar or vector type.
    fn zero_value(&self, ty: Handle<naga::Type>) -> Result<Eval, String> {
        match vector_shape(self.m, ty) {
            Some(shape) => {
                let (t, n, _) = shape?;
                Ok(Eval::Vector(vec![t.zero().to_string(); n as usize], t))
            }
            None => {
                let t = scalar_ty_of(self.m, ty)?;
                Ok(Eval::Value(t.zero().to_string(), t))
            }
        }
    }

    /// A module-level constant: a literal, a zero value, or a vector built
    /// from them.
    fn constant(&self, init: Handle<Expression>) -> Result<Eval, String> {
        match &self.m.global_expressions[init] {
            Expression::Literal(lit) => literal(lit),
            Expression::ZeroValue(ty) => self.zero_value(*ty),
            Expression::Splat { size, value } => match self.constant(*value)? {
                Eval::Value(v, t) => Ok(Eval::Vector(vec![v; *size as usize], t)),
                _ => Err("a splat of something that is not a scalar constant".into()),
            },
            Expression::Compose { ty, components } => {
                let Some(shape) = vector_shape(self.m, *ty) else {
                    return Err("only vector constants are supported".into());
                };
                let (t, n, _) = shape?;
                let mut comps = Vec::with_capacity(n as usize);
                for c in components {
                    match self.constant(*c)? {
                        Eval::Value(v, _) => comps.push(v),
                        Eval::Vector(v, _) => comps.extend(v),
                        Eval::Place(_) => return Err("a constant that is a place".into()),
                    }
                }
                Ok(Eval::Vector(comps, t))
            }
            other => Err(format!("unsupported constant expression {other:?}")),
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

/// The shape of an array's elements.
struct ArrayLayout {
    /// The scalar every element is made of.
    elem: Ty,
    /// `(components, stride in words)` when an element is a vector. WGSL pads a
    /// `vec3` to the alignment of a `vec4`, so its stride is four words, not
    /// three; getting that wrong shifts every element after the first.
    vec: Option<(u32, u32)>,
    /// The element count, when the array is not runtime-sized.
    count: Option<u32>,
}

impl ArrayLayout {
    /// Words one element occupies.
    fn stride(&self) -> u32 {
        self.vec.map_or(1, |(_, stride)| stride)
    }
}

/// `(scalar, components, stride)` of a 4-byte-component vector type.
fn vector_shape(m: &naga::Module, ty: Handle<naga::Type>) -> Option<Result<(Ty, u32, u32), String>> {
    match &m.types[ty].inner {
        TypeInner::Vector { size, scalar } => {
            let n = *size as u32;
            Some(Ty::from_scalar(*scalar).map(|t| (t, n, if n == 3 { 4 } else { n })))
        }
        _ => None,
    }
}

fn array_layout(m: &naga::Module, ty: Handle<naga::Type>) -> Result<ArrayLayout, String> {
    match &m.types[ty].inner {
        TypeInner::Array { base, size, .. } => {
            let (elem, vec) = match vector_shape(m, *base) {
                Some(shape) => {
                    let (t, n, stride) = shape?;
                    (t, Some((n, stride)))
                }
                None => (scalar_ty_of(m, *base)?, None),
            };
            let count = match size {
                naga::ArraySize::Constant(n) => Some(n.get()),
                _ => None,
            };
            Ok(ArrayLayout { elem, vec, count })
        }
        other => Err(format!("expected an array, got {other:?}")),
    }
}

/// `(scalar, element count)` of a fixed-size array of scalars or vectors; for
/// vectors the count is in words, which is what the emitted C++ array holds.
fn array_info(m: &naga::Module, ty: Handle<naga::Type>) -> Result<(Ty, u32), String> {
    let l = array_layout(m, ty)?;
    match l.count {
        Some(n) => Ok((l.elem, n * l.stride())),
        None => Err("a fixed size is required here, got a runtime-sized array".to_string()),
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

    /// Hazard 1, at the text level: when a `return` is reached by only part of
    /// the workgroup, the kernel carries a guard flag and no `return`, and the
    /// barrier itself sits outside the guard so the finished threads still
    /// arrive at it.
    #[test]
    fn a_barrier_kernel_with_a_per_thread_return_guards_instead_of_returning() {
        const SRC: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
var<workgroup> s: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    if (gid.x >= p.n) { return; }
    s[li.x] = 1.0;
    workgroupBarrier();
    o[gid.x] = s[0];
}
"#;
        let k = generate("per_thread_return", SRC).expect("generate");
        assert!(!k.source.contains("return;"), "a return survived into a guarded barrier kernel:\n{}", k.source);
        assert!(k.source.contains("__active = false;"), "no guard flag was emitted:\n{}", k.source);
        let sync = k.source.find("__syncthreads();").expect("barrier");
        let line_start = k.source[..sync].rfind('\n').map(|i| i + 1).unwrap_or(0);
        assert_eq!(
            k.source[line_start..sync].trim(),
            "",
            "the barrier must be a statement of its own, outside any guard"
        );
    }

    /// `gradnorm_part` bounds-checks on the workgroup index, which every thread
    /// of the workgroup agrees on, so its early return needs no guard flag.
    #[test]
    fn a_barrier_kernel_whose_return_is_workgroup_uniform_returns_for_real() {
        let k = generate("gradnorm_part", kernels::GRADNORM_PART).expect("generate");
        assert!(k.source.contains("return;"), "{}", k.source);
        assert!(!k.source.contains("__active"), "a uniform return must not be guarded:\n{}", k.source);
        assert!(k.source.contains("__syncthreads();"), "{}", k.source);
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

    const BARRIER_HEAD: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
var<workgroup> s: array<f32, 64>;
"#;

    fn barrier_kernel(body: &str) -> String {
        format!(
            "{BARRIER_HEAD}@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) li: vec3<u32>, \
             @builtin(workgroup_id) wi: vec3<u32>) {{\n{body}\n}}"
        )
    }

    /// A barrier in a loop whose trip count is uniform is reached by every
    /// thread the same number of times, so it is emitted where it stands.
    #[test]
    fn a_barrier_in_a_uniform_loop_is_emitted_inside_the_loop() {
        let src = barrier_kernel("for (var i = 0u; i < p.n; i = i + 1u) { s[li.x] = f32(i); workgroupBarrier(); o[li.x] = s[0]; }");
        let k = generate("uniform_loop", &src).expect("a uniform barrier loop is supported");
        let body = k.source.split("while (true)").nth(1).expect("the loop is emitted");
        assert!(body.contains("__syncthreads();"), "the barrier must stay inside the loop:\n{}", k.source);
    }

    /// A barrier under a branch on a uniform value is likewise fine.
    #[test]
    fn a_barrier_under_a_uniform_branch_is_emitted() {
        let src = barrier_kernel("s[li.x] = 1.0;\nif (wi.x < p.n) { workgroupBarrier(); }\no[li.x] = s[0];");
        generate("uniform_if", &src).expect("a uniform branch around a barrier is supported");
    }

    /// An early `return` the whole workgroup takes together (a bounds check on
    /// the workgroup id) leaves nobody behind, so it is a real `return` and the
    /// barriers after it need no guard.
    #[test]
    fn a_workgroup_uniform_return_is_a_real_return_before_a_barrier_loop() {
        let src = barrier_kernel(
            "if (wi.x >= p.n) { return; }\nfor (var i = 0u; i < p.n; i = i + 1u) { s[li.x] = f32(i); workgroupBarrier(); o[li.x] = s[0]; }",
        );
        let k = generate("uniform_return", &src).expect("a uniform return before a barrier loop is supported");
        assert!(k.source.contains("return;"), "{}", k.source);
        assert!(!k.source.contains("__active"), "no guard flag should be needed:\n{}", k.source);
    }

    /// A barrier a subset of the workgroup skips would hang the rest: refused.
    #[test]
    fn a_barrier_under_a_per_thread_branch_is_refused() {
        let src = barrier_kernel("if (li.x < 32u) { workgroupBarrier(); }");
        let e = generate("thread_branch", &src).expect_err("must be refused");
        assert!(e.contains("non-uniform control flow"), "{e}");
    }

    /// A loop whose trip count depends on the thread would reach the barrier a
    /// different number of times on each thread: refused.
    #[test]
    fn a_barrier_in_a_per_thread_loop_is_refused() {
        let src = barrier_kernel("for (var i = 0u; i < li.x; i = i + 1u) { workgroupBarrier(); }");
        let e = generate("thread_loop", &src).expect_err("must be refused");
        assert!(e.contains("non-uniform control flow"), "{e}");
    }

    /// A per-thread `return` ahead of a barrier loop is still the combination
    /// the guarded transform cannot express (the flag would leave the loop with
    /// no way out): refused, never mistranslated.
    #[test]
    fn a_per_thread_return_with_a_barrier_in_a_loop_is_still_refused() {
        let src = barrier_kernel(
            "if (li.x >= p.n) { return; }\nfor (var i = 0u; i < p.n; i = i + 1u) { workgroupBarrier(); }",
        );
        let e = generate("thread_return_loop", &src).expect_err("must be refused");
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

