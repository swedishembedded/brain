// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Function-scope state, local variables, and user function calls.
//!
//! Swedish Embedded AB implements shader-to-native compilers for its clients.
//! If your team needs expertise in function inlining or SSA construction for
//! a JIT, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A user function call is inlined at translation time: the callee's body is
//! translated in place, in a fresh [`Frame`] whose arguments are the caller's
//! evaluated argument values, whose locals are declared (and zeroed) at the
//! call site, and whose `return`s store the flattened result into per-call
//! variables and jump to a per-call return block. The caller continues in
//! that block and reads the result from those variables, so a `return` from
//! any depth of the callee's control flow lands in the same place. WGSL
//! forbids recursion; a recursive call is reported, not followed.

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::{Block as ClBlock, InstBuilder, StackSlotData, StackSlotKind};
use cranelift_frontend::Variable;
use naga::{Block, Expression, Handle, Statement, TypeInner};

use crate::shape::{contains_array, mem_size, scalar_tys, shape_of, Shape};
use crate::{cl_ty, Eval, LocalSlot, Place, Tr, Ty};

/// Where a `return` in an inlined callee goes: the caller's continuation
/// block, and one variable per scalar of the flattened result.
pub(crate) struct Ret {
    pub block: ClBlock,
    pub vars: Vec<(Variable, Ty)>,
    pub shape: Option<Shape>,
}

/// The per-function translation state: one for the entry point, and one per
/// inlined call while its body is being translated.
pub(crate) struct Frame<'a> {
    pub func: &'a naga::Function,
    /// The entry point, whose `return` ends the invocation and whose
    /// arguments are the builtin inputs.
    pub entry: bool,
    pub cache: HashMap<Handle<Expression>, Eval>,
    /// Every expression some `Emit` statement of this function evaluates.
    /// Only these are cached: everything else (a literal, a constant, a `let`
    /// bound to one) is one naga expression shared by all its uses, which may
    /// sit in blocks that do not dominate each other, so it is
    /// re-materialised at each use instead.
    pub emitted: HashSet<Handle<Expression>>,
    pub locals: HashMap<Handle<naga::LocalVariable>, LocalSlot>,
    pub args: Vec<Eval>,
    pub ret: Option<Ret>,
    /// `(continue, break)` targets of the enclosing loops.
    pub loop_stack: Vec<(ClBlock, ClBlock)>,
}

impl<'a> Frame<'a> {
    pub fn new(func: &'a naga::Function, entry: bool, args: Vec<Eval>, ret: Option<Ret>) -> Self {
        let mut emitted = HashSet::new();
        emitted_exprs(&func.body, &mut emitted);
        Frame {
            func,
            entry,
            cache: HashMap::new(),
            emitted,
            locals: HashMap::new(),
            args,
            ret,
            loop_stack: Vec::new(),
        }
    }
}

fn emitted_exprs(block: &Block, out: &mut HashSet<Handle<Expression>>) {
    for s in block.iter() {
        match s {
            Statement::Emit(range) => out.extend(range.clone()),
            Statement::Block(b) => emitted_exprs(b, out),
            Statement::If { accept, reject, .. } => {
                emitted_exprs(accept, out);
                emitted_exprs(reject, out);
            }
            Statement::Loop { body, continuing, .. } => {
                emitted_exprs(body, out);
                emitted_exprs(continuing, out);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    emitted_exprs(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

impl<'a, 'b> Tr<'a, 'b> {
    /// Declare the current frame's locals at the current program point, then
    /// apply their initialisers.
    ///
    /// * a scalar is one Cranelift variable, zero-initialised;
    /// * a vector, matrix or struct without arrays is one variable per scalar
    ///   of its flattened layout, zero-initialised, so component writes and
    ///   reads stay in registers (a dynamic index selects among them);
    /// * anything containing an array lives in a stack slot laid out like
    ///   memory, so it can be indexed dynamically at any depth. Stack-backed
    ///   locals are scratch: like the scalar arrays before them they are not
    ///   zero-filled, and every in-tree kernel writes them before reading.
    pub(crate) fn declare_locals(&mut self) -> Result<(), String> {
        let m = self.module_ref;
        let func = self.frame.func;
        for (h, lv) in func.local_variables.iter() {
            let shape = shape_of(m, lv.ty)?;
            let slot = match shape {
                Shape::Scalar(ty) => {
                    let var = self.b.declare_var(cl_ty(ty));
                    let zero = self.zero(ty);
                    self.b.def_var(var, zero);
                    LocalSlot::Scalar(var, ty)
                }
                s if contains_array(m, s) => {
                    let slot = self.b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        mem_size(m, lv.ty),
                        4,
                    ));
                    let base = self.b.ins().stack_addr(self.env.ptr_ty, slot, 0);
                    LocalSlot::Stack { base, shape: s }
                }
                s => {
                    let group = self.declare_var_group(s)?;
                    LocalSlot::Vars { group, shape: s }
                }
            };
            self.frame.locals.insert(h, slot);
        }
        for (h, lv) in func.local_variables.iter() {
            if let Some(init) = lv.init {
                let place = self.local_place(h);
                self.store(place, init)?;
            }
        }
        Ok(())
    }

    /// One zero-initialised variable per scalar of `shape`; returns the group id.
    fn declare_var_group(&mut self, shape: Shape) -> Result<u32, String> {
        let vars = self.declare_zeroed_vars(shape)?;
        self.var_groups.push(vars);
        Ok(self.var_groups.len() as u32 - 1)
    }

    fn declare_zeroed_vars(&mut self, shape: Shape) -> Result<Vec<(Variable, Ty)>, String> {
        let mut tys = Vec::new();
        scalar_tys(self.module_ref, shape, &mut tys)?;
        Ok(tys
            .into_iter()
            .map(|ty| {
                let var = self.b.declare_var(cl_ty(ty));
                let zero = self.zero(ty);
                self.b.def_var(var, zero);
                (var, ty)
            })
            .collect())
    }

    pub(crate) fn local_place(&self, h: Handle<naga::LocalVariable>) -> Place {
        match self.frame.locals[&h] {
            LocalSlot::Scalar(var, ty) => Place::Local(var, ty),
            LocalSlot::Vars { group, shape } => Place::Vars { group, start: 0, shape },
            LocalSlot::Stack { base, shape } => Place::Ptr { addr: base, off: 0, shape, readonly: false },
        }
    }

    /// The value of function argument `i`: a builtin input vector in the
    /// entry point, the caller's argument inside an inlined call.
    pub(crate) fn function_argument(&mut self, i: u32) -> Result<Eval, String> {
        if !self.frame.entry {
            return self.frame.args.get(i as usize).copied().ok_or_else(|| format!("no argument {i}"));
        }
        let ba = self.env.ba;
        let comps = if Some(i) == ba.gid {
            self.env.gid
        } else if Some(i) == ba.nwg {
            self.env.nwg
        } else if Some(i) == ba.local_id {
            self.env.local_id
        } else if Some(i) == ba.wgid {
            self.env.wgid
        } else {
            return Err(format!("entry-point argument {i} is not a supported builtin"));
        };
        let vals = comps.iter().map(|v| (*v, Ty::U32)).collect();
        Ok(self.make(vals, Shape::Vector(naga::VectorSize::Tri, Ty::U32)))
    }

    /// Inline a call to `function`; its result, if any, becomes the value of
    /// the caller's `result` expression.
    pub(crate) fn inline_call(
        &mut self,
        function: Handle<naga::Function>,
        arguments: &[Handle<Expression>],
        result: Option<Handle<Expression>>,
    ) -> Result<(), String> {
        let m = self.module_ref;
        let callee = &m.functions[function];
        let name = callee.name.as_deref().unwrap_or("?");
        if self.call_stack.contains(&function) {
            return Err(format!("recursive call to `{name}` (not legal WGSL)"));
        }
        let mut args = Vec::with_capacity(arguments.len());
        for (a, formal) in arguments.iter().zip(&callee.arguments) {
            let e = self.eval(*a)?;
            // A pointer argument is passed as the place itself; a value is
            // materialised in the caller, where it was evaluated.
            let e = match m.types[formal.ty].inner {
                TypeInner::Pointer { .. } => e,
                _ => self.materialize(e)?,
            };
            args.push(e);
        }
        let block = self.b.create_block();
        let (vars, shape) = match &callee.result {
            Some(r) => {
                let shape = shape_of(m, r.ty)?;
                (self.declare_zeroed_vars(shape)?, Some(shape))
            }
            None => (Vec::new(), None),
        };
        let frame = Frame::new(callee, false, args, Some(Ret { block, vars, shape }));
        let caller = std::mem::replace(&mut self.frame, frame);
        self.call_stack.push(function);
        let outcome = self.translate_body(&callee.body);
        self.call_stack.pop();
        let callee_frame = std::mem::replace(&mut self.frame, caller);
        outcome.map_err(|e| format!("in `{name}`: {e}"))?;

        self.b.switch_to_block(block);
        if let (Some(res), Some(ret)) = (result, callee_frame.ret) {
            let shape = ret.shape.ok_or("a void call has no result")?;
            let vals = ret.vars.iter().map(|(v, t)| (self.b.use_var(*v), *t)).collect();
            let e = self.make(vals, shape);
            self.frame.cache.insert(res, e);
        }
        Ok(())
    }

    /// Declare the current (callee) frame's locals and translate `body`,
    /// falling through to the frame's return block.
    fn translate_body(&mut self, body: &Block) -> Result<(), String> {
        self.declare_locals()?;
        if self.block(body)? {
            let block = self.frame.ret.as_ref().ok_or("inlined frame without a return block")?.block;
            self.b.ins().jump(block, &[]);
        }
        Ok(())
    }

    /// `return value` from an inlined callee.
    pub(crate) fn inline_return(&mut self, value: Option<Handle<Expression>>) -> Result<(), String> {
        let (block, vars) = {
            let r = self.frame.ret.as_ref().ok_or("return outside a function")?;
            (r.block, r.vars.clone())
        };
        if let Some(v) = value {
            let (vals, _) = self.value(v)?;
            if vals.len() != vars.len() {
                return Err(format!("return of {} scalars into {}", vals.len(), vars.len()));
            }
            for ((var, ty), val) in vars.iter().zip(vals) {
                let x = self.coerce(val, *ty)?;
                self.b.def_var(*var, x);
            }
        }
        self.b.ins().jump(block, &[]);
        Ok(())
    }
}
