// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Workgroup-uniformity analysis of one WGSL entry point.
//!
//! Swedish Embedded AB implements portable GPU compute stacks, including the
//! compiler passes that make one kernel source run correctly on every backend,
//! for its clients. If your team needs expertise in translating compute kernels
//! between GPU programming models then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! A barrier is only well defined when every thread of the workgroup reaches
//! it, which WGSL states as a rule about *uniform control flow*: the branch
//! conditions and loop exits that lead to a barrier must evaluate the same for
//! every invocation of the workgroup. The generated CUDA tier needs the same
//! fact to decide two things:
//!
//! * a `return` that every thread takes together (`if (b >= p.batch) {
//!   return; }` with `b` derived from the workgroup id) needs no guard, because
//!   no thread is left behind to miss a later `__syncthreads()`;
//! * a barrier inside a loop or an `if` is correct to emit directly when the
//!   loop trip count and the branch condition are uniform.
//!
//! A value is workgroup-uniform when it is built only from constants, the
//! workgroup id and workgroup count, loads from the uniform parameter block,
//! local variables only ever assigned uniform values in uniform flow, and
//! arithmetic over those. Anything else - the local or global invocation id,
//! storage and workgroup memory (written per thread), call and atomic results -
//! is non-uniform. The analysis is deliberately conservative: it may call a
//! uniform thing non-uniform (the generator then refuses or guards it, as
//! before) and never the reverse.

use std::collections::HashSet;

use naga::{
    AddressSpace, Binding, BuiltIn, Block, Expression, Function, Handle, LocalVariable, Module, Statement, StorageAccess,
};

/// What the generator needs to know about one entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uniformity {
    /// Some `return` is reached by only part of the workgroup.
    pub has_non_uniform_return: bool,
    /// Some barrier is reached under non-uniform control flow, so threads
    /// that skip it would leave the others waiting forever.
    pub has_barrier_in_non_uniform_flow: bool,
}

/// Analyse `func`, an entry point of `module`.
pub fn analyse(module: &Module, func: &Function) -> Uniformity {
    let mut non_uniform_locals: HashSet<Handle<LocalVariable>> = HashSet::new();
    loop {
        let mut walk = Walk {
            func,
            uniform: expression_uniformity(module, func, &non_uniform_locals),
            non_uniform_locals: &mut non_uniform_locals,
            changed: false,
            return_diverged: false,
            loop_diverged: false,
            saw_non_uniform_exit: false,
            out: Uniformity { has_non_uniform_return: false, has_barrier_in_non_uniform_flow: false },
        };
        walk.block(&func.body, true, true);
        let (changed, out) = (walk.changed, walk.out);
        // A store into a local can make a value non-uniform that an earlier
        // expression already read as uniform; iterate to a fixed point. The
        // set only grows and is bounded by the number of locals.
        if !changed {
            return out;
        }
    }
}

/// Whether each expression's value (or, for a pointer, its address) is the
/// same for every invocation of a workgroup, in arena order.
fn expression_uniformity(module: &Module, func: &Function, non_uniform_locals: &HashSet<Handle<LocalVariable>>) -> Vec<bool> {
    let mut uniform = vec![false; func.expressions.len()];
    for (h, e) in func.expressions.iter() {
        let u = |x: &Handle<Expression>| uniform[x.index()];
        uniform[h.index()] = match e {
            Expression::Literal(_) | Expression::Constant(_) | Expression::Override(_) | Expression::ZeroValue(_) => true,
            // Uniform by definition: the value is broadcast from one invocation.
            Expression::WorkGroupUniformLoadResult { .. } => true,
            Expression::FunctionArgument(i) => {
                matches!(
                    func.arguments[*i as usize].binding,
                    Some(Binding::BuiltIn(BuiltIn::WorkGroupId | BuiltIn::NumWorkGroups))
                )
            }
            // A pointer's own address: a variable's is fixed, an indexed one is
            // as uniform as its base and index.
            Expression::GlobalVariable(_) | Expression::LocalVariable(_) => true,
            Expression::Access { base, index } => u(base) && u(index),
            Expression::AccessIndex { base, .. } => u(base),
            Expression::Splat { value, .. } => u(value),
            Expression::Swizzle { vector, .. } => u(vector),
            Expression::Compose { components, .. } => components.iter().all(u),
            Expression::Unary { expr, .. } => u(expr),
            Expression::Binary { left, right, .. } => u(left) && u(right),
            Expression::Select { condition, accept, reject } => u(condition) && u(accept) && u(reject),
            Expression::Relational { argument, .. } => u(argument),
            Expression::As { expr, .. } => u(expr),
            Expression::Math { arg, arg1, arg2, arg3, .. } => {
                u(arg) && arg1.iter().all(u) && arg2.iter().all(u) && arg3.iter().all(u)
            }
            Expression::ArrayLength(p) => u(p),
            Expression::Load { pointer } => u(pointer) && root_is_uniform(module, func, *pointer, non_uniform_locals),
            // Call and atomic results, images, derivatives, subgroup and ray
            // query results: not provably uniform.
            _ => false,
        };
    }
    uniform
}

/// Whether the variable a pointer chain is rooted in holds workgroup-uniform
/// data at every point the kernel reads it.
fn root_is_uniform(module: &Module, func: &Function, mut pointer: Handle<Expression>, non_uniform_locals: &HashSet<Handle<LocalVariable>>) -> bool {
    loop {
        match &func.expressions[pointer] {
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => pointer = *base,
            Expression::LocalVariable(v) => return !non_uniform_locals.contains(v),
            // The uniform parameter block, and a storage buffer the kernel can
            // only read (a page table, a sequence-length list), hold the same
            // value for every thread at a given address for the whole
            // dispatch. Writable storage and workgroup memory are written by
            // individual threads, so a read of them is not assumed uniform.
            Expression::GlobalVariable(g) => {
                return match module.global_variables[*g].space {
                    AddressSpace::Uniform => true,
                    AddressSpace::Storage { access } => !access.contains(StorageAccess::STORE),
                    _ => false,
                }
            }
            _ => return false,
        }
    }
}

/// The local variable a pointer chain is rooted in, if it is rooted in one.
fn local_root(func: &Function, mut pointer: Handle<Expression>) -> Option<Handle<LocalVariable>> {
    loop {
        match &func.expressions[pointer] {
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => pointer = *base,
            Expression::LocalVariable(v) => return Some(*v),
            _ => return None,
        }
    }
}

struct Walk<'a> {
    func: &'a Function,
    uniform: Vec<bool>,
    non_uniform_locals: &'a mut HashSet<Handle<LocalVariable>>,
    changed: bool,
    /// Part of the workgroup has already returned, for the rest of the kernel.
    return_diverged: bool,
    /// Part of the workgroup has left the current loop iteration through a
    /// `break` or `continue`, for the rest of that loop body.
    loop_diverged: bool,
    /// A `break` or `continue` was reached by only part of the workgroup
    /// inside the loop being walked.
    saw_non_uniform_exit: bool,
    out: Uniformity,
}

impl Walk<'_> {
    /// `tail`: nothing runs after this block in the function, so a `return`
    /// that ends it strands nobody. The front-end copies the function's
    /// closing `return` into both arms of a trailing `if`, which is why this
    /// has to be known rather than assumed.
    fn block(&mut self, b: &Block, in_uniform_flow: bool, tail: bool) {
        let last = b.len().saturating_sub(1);
        for (i, s) in b.iter().enumerate() {
            let flow = in_uniform_flow && !self.return_diverged && !self.loop_diverged;
            self.statement(s, flow, tail && i == last);
        }
    }

    fn statement(&mut self, s: &Statement, flow: bool, tail: bool) {
        match s {
            Statement::Block(inner) => self.block(inner, flow, tail),
            Statement::If { condition, accept, reject } => {
                let inner = flow && self.uniform[condition.index()];
                self.block(accept, inner, tail);
                self.block(reject, inner, tail);
            }
            Statement::Switch { selector, cases } => {
                let inner = flow && self.uniform[selector.index()];
                for case in cases {
                    self.block(&case.body, inner, tail);
                }
            }
            Statement::Loop { body, continuing, break_if } => self.loop_(body, continuing, *break_if, flow),
            Statement::Return { .. } => {
                if !flow && !tail {
                    self.out.has_non_uniform_return = true;
                    self.return_diverged = true;
                }
            }
            Statement::Break | Statement::Continue => {
                if !flow {
                    self.saw_non_uniform_exit = true;
                    self.loop_diverged = true;
                }
            }
            Statement::ControlBarrier(_) => {
                if !flow {
                    self.out.has_barrier_in_non_uniform_flow = true;
                }
            }
            Statement::Store { pointer, value } => {
                if let Some(v) = local_root(self.func, *pointer) {
                    // A local becomes non-uniform when anything non-uniform
                    // is stored into it, or when the store itself is reached
                    // by only part of the workgroup.
                    let uniform_store = self.uniform[value.index()] && self.uniform[pointer.index()] && flow;
                    if !uniform_store && self.non_uniform_locals.insert(v) {
                        self.changed = true;
                    }
                }
            }
            _ => {}
        }
    }

    fn loop_(&mut self, body: &Block, continuing: &Block, break_if: Option<Handle<Expression>>, flow: bool) {
        let (saved_loop, saved_exit) = (self.loop_diverged, self.saw_non_uniform_exit);
        self.loop_diverged = false;
        self.saw_non_uniform_exit = false;
        // Nothing is "the end of the function" inside a loop body: another
        // iteration, or the code after the loop, still follows.
        self.block(body, flow, false);
        self.block(continuing, flow, false);
        if break_if.is_some_and(|c| !self.uniform[c.index()]) {
            self.saw_non_uniform_exit = true;
        }
        if self.saw_non_uniform_exit {
            // Some threads leave the loop earlier than others, so from the
            // second iteration on the whole body runs on a subset: walk it
            // again as non-uniform flow. The first walk only ever recorded
            // facts that hold under the optimistic assumption, so the union is
            // sound.
            self.loop_diverged = false;
            self.block(body, false, false);
            self.block(continuing, false, false);
        }
        self.loop_diverged = saved_loop;
        self.saw_non_uniform_exit = saved_exit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyse_entry(wgsl: &str) -> Uniformity {
        let m = naga::front::wgsl::parse_str(wgsl).expect("test kernel parses");
        let f = &m.entry_points.iter().find(|e| e.name == "main").expect("main").function;
        analyse(&m, f)
    }

    const HEAD: &str = "struct P { n: u32, }\n@group(0) @binding(0) var<uniform> p: P;\n\
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;\n\
        var<workgroup> s: array<f32, 64>;\n";

    fn kernel(body: &str) -> String {
        format!(
            "{HEAD}@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, \
             @builtin(workgroup_id) wid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {{\n{body}\n}}"
        )
    }

    /// A loop whose trip count comes from the uniform block runs the same
    /// number of times on every thread, so a barrier in it is fine.
    #[test]
    fn a_loop_with_a_uniform_trip_count_keeps_its_barrier_uniform() {
        let u = analyse_entry(&kernel("for (var i = 0u; i < p.n; i = i + 1u) { s[lid.x] = f32(i); workgroupBarrier(); o[gid.x] = s[63u - lid.x]; workgroupBarrier(); }"));
        assert!(!u.has_barrier_in_non_uniform_flow && !u.has_non_uniform_return, "{u:?}");
    }

    /// A trip count derived from the local invocation id differs per thread:
    /// the barrier inside would be reached a different number of times.
    #[test]
    fn a_loop_bounded_by_the_thread_id_makes_its_barrier_non_uniform() {
        let u = analyse_entry(&kernel("for (var i = 0u; i < lid.x; i = i + 1u) { workgroupBarrier(); }"));
        assert!(u.has_barrier_in_non_uniform_flow, "{u:?}");
    }

    /// A `return` on a condition made of the workgroup id and the uniform
    /// block is taken by the whole workgroup or by none of it.
    #[test]
    fn a_return_on_the_workgroup_id_is_uniform() {
        let u = analyse_entry(&kernel("if (wid.x >= p.n) { return; }\nworkgroupBarrier();\no[gid.x] = 1.0;"));
        assert!(!u.has_non_uniform_return && !u.has_barrier_in_non_uniform_flow, "{u:?}");
    }

    /// A `return` on the thread's own index strands the rest of the workgroup
    /// at any later barrier.
    #[test]
    fn a_return_on_the_thread_id_is_non_uniform_and_poisons_a_later_barrier() {
        let u = analyse_entry(&kernel("if (gid.x >= p.n) { return; }\nworkgroupBarrier();\no[gid.x] = 1.0;"));
        assert!(u.has_non_uniform_return && u.has_barrier_in_non_uniform_flow, "{u:?}");
    }

    /// The function's closing `return` is copied into both arms of a trailing
    /// `if`; that is the end of the kernel, not an early exit, and must not be
    /// read as a thread-dependent return.
    #[test]
    fn the_closing_return_of_a_kernel_is_not_an_early_exit() {
        let u = analyse_entry(&kernel("if (lid.x < 32u) { o[gid.x] = 1.0; }"));
        assert!(!u.has_non_uniform_return, "{u:?}");
    }

    /// A barrier under a branch on the thread id is reached by a subset.
    #[test]
    fn a_barrier_under_a_thread_id_branch_is_non_uniform() {
        let u = analyse_entry(&kernel("if (lid.x < 32u) { workgroupBarrier(); }"));
        assert!(u.has_barrier_in_non_uniform_flow, "{u:?}");
    }

    /// A page table the kernel can only read holds the same value for every
    /// thread, so a loop bound read from it is uniform; the same read from a
    /// buffer the kernel also writes is not assumed to be.
    #[test]
    fn a_loop_bounded_by_a_read_only_storage_value_is_uniform() {
        let head = "@group(0) @binding(0) var<storage, read> table: array<u32>;\n\
            @group(0) @binding(1) var<storage, read_write> o: array<f32>;\n";
        let main = "@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n";
        let body = |bound: &str| format!("{head}{main}for (var i = 0u; i < {bound}; i = i + 1u) {{ workgroupBarrier(); }}\n}}");
        assert!(!analyse_entry(&body("table[wid.x]")).has_barrier_in_non_uniform_flow);
        assert!(analyse_entry(&body("table[lid.x]")).has_barrier_in_non_uniform_flow, "a per-thread address is not uniform");
        let rw = "@group(0) @binding(0) var<storage, read_write> table: array<u32>;\n\
            @group(0) @binding(1) var<storage, read_write> o: array<f32>;\n";
        let rw_body = format!("{rw}{main}for (var i = 0u; i < table[wid.x]; i = i + 1u) {{ workgroupBarrier(); }}\n}}");
        assert!(analyse_entry(&rw_body).has_barrier_in_non_uniform_flow, "writable storage is not assumed uniform");
    }

    /// A local assigned from the thread id is non-uniform for every later
    /// read, even where the read looks like a plain variable.
    #[test]
    fn a_local_assigned_a_thread_value_is_non_uniform_afterwards() {
        let u = analyse_entry(&kernel("var t = 0u;\nt = lid.x;\nfor (var i = 0u; i < t; i = i + 1u) { workgroupBarrier(); }"));
        assert!(u.has_barrier_in_non_uniform_flow, "{u:?}");
    }

    /// A local only ever assigned uniform values stays uniform.
    #[test]
    fn a_local_assigned_only_uniform_values_stays_uniform() {
        let u = analyse_entry(&kernel("var t = 0u;\nt = p.n + wid.x;\nfor (var i = 0u; i < t; i = i + 1u) { workgroupBarrier(); }"));
        assert!(!u.has_barrier_in_non_uniform_flow, "{u:?}");
    }

    /// A `break` on a thread-dependent condition makes the loop's later
    /// barrier non-uniform: threads leave on different iterations.
    #[test]
    fn a_thread_dependent_break_makes_the_loop_barrier_non_uniform() {
        let u = analyse_entry(&kernel("for (var i = 0u; i < p.n; i = i + 1u) { if (lid.x == i) { break; } workgroupBarrier(); }"));
        assert!(u.has_barrier_in_non_uniform_flow, "{u:?}");
    }
}
