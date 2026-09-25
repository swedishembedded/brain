// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Aggregate values and places: vectors, matrices, structs and arrays,
//! lowered to lists of scalar Cranelift values.
//!
//! Swedish Embedded AB implements shader-to-native compilers for its clients.
//! If your team needs expertise in scalarizing vector and matrix code for a
//! CPU target, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! An aggregate value is its flattened scalar list ([`crate::shape`]). An
//! aggregate place is one of: memory laid out per WGSL's host-shareable rules
//! ([`Place::Ptr`]: uniform, storage, work-group and stack-backed locals), a
//! run of a register-backed local's variables ([`Place::Vars`]), or a
//! dynamically indexed part of one ([`Place::VarsDyn`], which reads with a
//! select chain over the candidates and writes with one select per candidate
//! scalar, so a runtime index never leaves registers).
//!
//! Arithmetic follows WGSL: componentwise on vectors and matrices with scalar
//! broadcast, the three linear-algebra products for `*`, and sums of products
//! accumulated left to right (`m * v = m[0] v.x + m[1] v.y + ...`).

use cranelift_codegen::ir::{condcodes::FloatCC, condcodes::IntCC, InstBuilder, MemFlags, Value};
use naga::{
    AddressSpace, BinaryOperator, Expression, Handle, MathFunction, RelationalFunction, ScalarKind,
    SwizzleComponent, UnaryOperator, VectorSize,
};

use crate::shape::{element, flat_len, mem_offsets, member, scalar_tys, shape_of, Shape};
use crate::{cl_ty, mem_flags, Eval, Place, Tr, Ty};

type Vals = Vec<(Value, Ty)>;

impl<'a, 'b> Tr<'a, 'b> {
    /// Wrap a flattened value as an [`Eval`].
    pub(crate) fn make(&mut self, vals: Vals, shape: Shape) -> Eval {
        match shape {
            Shape::Scalar(_) => Eval::Scalar(vals[0].0, vals[0].1),
            _ => {
                self.aggs.push(vals);
                Eval::Agg(self.aggs.len() as u32 - 1, shape)
            }
        }
    }

    /// Evaluate `h` as a value (loading a place), flattened.
    pub(crate) fn value(&mut self, h: Handle<Expression>) -> Result<(Vals, Shape), String> {
        let e = self.eval(h)?;
        self.value_of(e)
    }

    pub(crate) fn value_of(&mut self, e: Eval) -> Result<(Vals, Shape), String> {
        match e {
            Eval::Scalar(v, t) => Ok((vec![(v, t)], Shape::Scalar(t))),
            Eval::Agg(id, shape) => Ok((self.aggs[id as usize].clone(), shape)),
            Eval::Place(p) => {
                let loaded = self.load_place(p)?;
                self.value_of(loaded)
            }
        }
    }

    /// A value form of `e`: places are loaded, values pass through.
    pub(crate) fn materialize(&mut self, e: Eval) -> Result<Eval, String> {
        match e {
            Eval::Place(p) => self.load_place(p),
            v => Ok(v),
        }
    }

    pub(crate) fn load_place(&mut self, p: Place) -> Result<Eval, String> {
        let m = self.module_ref;
        Ok(match p {
            Place::Local(var, ty) => Eval::Scalar(self.b.use_var(var), ty),
            Place::Mem { addr, elem, readonly } => {
                Eval::Scalar(self.b.ins().load(cl_ty(elem), mem_flags(readonly), addr, 0), elem)
            }
            Place::Ptr { addr, off, shape, readonly } => {
                let mut offs = Vec::new();
                mem_offsets(m, shape, off, &mut offs)?;
                let vals = offs
                    .into_iter()
                    .map(|(o, t)| (self.b.ins().load(cl_ty(t), mem_flags(readonly), addr, o as i32), t))
                    .collect();
                self.make(vals, shape)
            }
            Place::Vars { group, start, shape } => {
                let n = flat_len(m, shape)? as usize;
                let vars = self.var_groups[group as usize][start as usize..start as usize + n].to_vec();
                let vals = vars.into_iter().map(|(v, t)| (self.b.use_var(v), t)).collect();
                self.make(vals, shape)
            }
            Place::VarsDyn { group, lo, step, n, idx, shape } => {
                let len = flat_len(m, shape)?;
                let vars = self.var_groups[group as usize].clone();
                let mut vals = Vec::with_capacity(len as usize);
                for c in 0..len {
                    let pick: Vec<Value> =
                        (0..n).map(|j| self.b.use_var(vars[(lo + j * step + c) as usize].0)).collect();
                    let v = self.select_among(idx, &pick);
                    vals.push((v, vars[(lo + c) as usize].1));
                }
                self.make(vals, shape)
            }
        })
    }

    pub(crate) fn store_place(&mut self, p: Place, vals: &[(Value, Ty)]) -> Result<(), String> {
        let m = self.module_ref;
        let one = |vals: &[(Value, Ty)]| -> Result<(Value, Ty), String> {
            match vals {
                [v] => Ok(*v),
                _ => Err(format!("storing {} scalars into a scalar", vals.len())),
            }
        };
        match p {
            Place::Local(var, ty) => {
                let v = self.coerce(one(vals)?, ty)?;
                self.b.def_var(var, v);
            }
            Place::Mem { addr, elem, .. } => {
                let v = self.coerce(one(vals)?, elem)?;
                self.b.ins().store(MemFlags::trusted(), v, addr, 0);
            }
            Place::Ptr { addr, off, shape, .. } => {
                let mut offs = Vec::new();
                mem_offsets(m, shape, off, &mut offs)?;
                check_len(offs.len(), vals.len())?;
                for ((o, t), v) in offs.into_iter().zip(vals) {
                    let v = self.coerce(*v, t)?;
                    self.b.ins().store(MemFlags::trusted(), v, addr, o as i32);
                }
            }
            Place::Vars { group, start, shape } => {
                let n = flat_len(m, shape)? as usize;
                check_len(n, vals.len())?;
                let vars = self.var_groups[group as usize][start as usize..start as usize + n].to_vec();
                for ((var, t), v) in vars.into_iter().zip(vals) {
                    let v = self.coerce(*v, t)?;
                    self.b.def_var(var, v);
                }
            }
            Place::VarsDyn { group, lo, step, n, idx, shape } => {
                check_len(flat_len(m, shape)? as usize, vals.len())?;
                let vars = self.var_groups[group as usize].clone();
                for j in 0..n {
                    let hit = self.b.ins().icmp_imm(IntCC::Equal, idx, j as i64);
                    for (c, v) in vals.iter().enumerate() {
                        let (var, t) = vars[(lo + j * step) as usize + c];
                        let v = self.coerce(*v, t)?;
                        let old = self.b.use_var(var);
                        let new = self.b.ins().select(hit, v, old);
                        self.b.def_var(var, new);
                    }
                }
            }
        }
        Ok(())
    }

    /// `pick[idx]`, as a chain of selects (element 0 for an out-of-range
    /// index, which WGSL allows).
    fn select_among(&mut self, idx: Value, pick: &[Value]) -> Value {
        let mut acc = pick[0];
        for (j, v) in pick.iter().enumerate().skip(1) {
            let hit = self.b.ins().icmp_imm(IntCC::Equal, idx, j as i64);
            acc = self.b.ins().select(hit, *v, acc);
        }
        acc
    }

    /// `addr + off` without an instruction when `off` is zero.
    fn offset_addr(&mut self, addr: Value, off: u32) -> Value {
        if off == 0 {
            addr
        } else {
            self.b.ins().iadd_imm(addr, off as i64)
        }
    }

    /// A place in memory holding `shape`: scalars become [`Place::Mem`].
    fn mem_place(&mut self, addr: Value, off: u32, shape: Shape, readonly: bool) -> Place {
        match shape {
            Shape::Scalar(elem) => Place::Mem { addr: self.offset_addr(addr, off), elem, readonly },
            _ => Place::Ptr { addr, off, shape, readonly },
        }
    }

    /// The place a module-scope variable names.
    pub(crate) fn global_place(&mut self, g: Handle<naga::GlobalVariable>) -> Result<Place, String> {
        let m = self.module_ref;
        let gv = &m.global_variables[g];
        let shape = shape_of(m, gv.ty)?;
        let (addr, readonly) = match gv.space {
            AddressSpace::WorkGroup => {
                (*self.env.wg_mem.get(&g).ok_or("workgroup global without scratch base")?, false)
            }
            AddressSpace::Storage { .. } => {
                let b = gv.binding.as_ref().map(|b| b.binding).ok_or("storage global without binding")?;
                (*self.env.buf_base.get(&b).ok_or("missing storage base pointer")?, false)
            }
            AddressSpace::Uniform => (self.env.uniform_ptr, true),
            other => return Err(format!("global in unsupported space {other:?}")),
        };
        Ok(self.mem_place(addr, 0, shape, readonly))
    }

    /// `base.index` (`AccessIndex`) on a place or a value.
    pub(crate) fn access_index(&mut self, base: Handle<Expression>, index: u32) -> Result<Eval, String> {
        let m = self.module_ref;
        match self.eval(base)? {
            Eval::Place(p) => {
                let place = match p {
                    Place::Ptr { addr, off, shape, readonly } => {
                        let mb = member(m, shape, index)?;
                        self.mem_place(addr, off + mb.offset, mb.shape, readonly)
                    }
                    Place::Vars { group, start, shape } => {
                        let mb = member(m, shape, index)?;
                        Place::Vars { group, start: start + mb.flat, shape: mb.shape }
                    }
                    Place::VarsDyn { group, lo, step, n, idx, shape } => {
                        let mb = member(m, shape, index)?;
                        Place::VarsDyn { group, lo: lo + mb.flat, step, n, idx, shape: mb.shape }
                    }
                    Place::Local(..) | Place::Mem { .. } => {
                        return Err("component access on a scalar place".into())
                    }
                };
                Ok(Eval::Place(place))
            }
            v => {
                let (vals, shape) = self.value_of(v)?;
                let mb = member(m, shape, index)?;
                let n = flat_len(m, mb.shape)? as usize;
                let part = vals[mb.flat as usize..mb.flat as usize + n].to_vec();
                Ok(self.make(part, mb.shape))
            }
        }
    }

    /// `base[index]` (`Access`, a runtime index) on a place or a value.
    pub(crate) fn access(&mut self, base: Handle<Expression>, index: Handle<Expression>) -> Result<Eval, String> {
        let m = self.module_ref;
        let b = self.eval(base)?;
        let (idx, _) = self.scalar(index)?;
        match b {
            Eval::Place(p) => {
                let place = match p {
                    Place::Ptr { addr, off, shape, readonly } => {
                        let el = element(m, shape)?;
                        let idx64 = self.emit_i64(idx);
                        let scaled = self.b.ins().imul_imm(idx64, el.mem_stride as i64);
                        let a = self.b.ins().iadd(addr, scaled);
                        self.mem_place(a, off, el.shape, readonly)
                    }
                    Place::Vars { group, start, shape } => {
                        let el = element(m, shape)?;
                        let n = el.count.ok_or("runtime-sized array in a register local")?;
                        Place::VarsDyn { group, lo: start, step: el.flat_stride, n, idx, shape: el.shape }
                    }
                    Place::VarsDyn { group, lo, step, n, idx: outer, shape } => {
                        // Candidates of the outer index split into `count`
                        // contiguous elements each: flatten the two indices.
                        let el = element(m, shape)?;
                        let count = el.count.ok_or("runtime-sized array in a register local")?;
                        debug_assert_eq!(step, count * el.flat_stride);
                        let scaled = self.b.ins().imul_imm(outer, count as i64);
                        let combined = self.b.ins().iadd(scaled, idx);
                        Place::VarsDyn {
                            group,
                            lo,
                            step: el.flat_stride,
                            n: n * count,
                            idx: combined,
                            shape: el.shape,
                        }
                    }
                    Place::Local(..) | Place::Mem { .. } => return Err("indexing a scalar place".into()),
                };
                Ok(Eval::Place(place))
            }
            v => {
                let (vals, shape) = self.value_of(v)?;
                let el = element(m, shape)?;
                let count = el.count.ok_or("runtime-sized array value")?;
                let len = el.flat_stride;
                let mut out = Vec::with_capacity(len as usize);
                for c in 0..len {
                    let pick: Vec<Value> = (0..count).map(|j| vals[(j * len + c) as usize].0).collect();
                    let v = self.select_among(idx, &pick);
                    out.push((v, vals[c as usize].1));
                }
                Ok(self.make(out, el.shape))
            }
        }
    }

    pub(crate) fn compose(
        &mut self,
        ty: Handle<naga::Type>,
        components: &[Handle<Expression>],
    ) -> Result<Eval, String> {
        let shape = shape_of(self.module_ref, ty)?;
        let mut vals = Vec::new();
        for c in components {
            vals.extend(self.value(*c)?.0);
        }
        check_len(flat_len(self.module_ref, shape)? as usize, vals.len())?;
        Ok(self.make(vals, shape))
    }

    pub(crate) fn splat(&mut self, size: VectorSize, value: (Value, Ty)) -> Eval {
        self.make(vec![value; size as usize], Shape::Vector(size, value.1))
    }

    pub(crate) fn swizzle(
        &mut self,
        size: VectorSize,
        vector: Handle<Expression>,
        pattern: &[SwizzleComponent; 4],
    ) -> Result<Eval, String> {
        let (vals, _) = self.value(vector)?;
        let picked: Vals = pattern[..size as usize].iter().map(|c| vals[*c as usize]).collect();
        let ty = picked[0].1;
        Ok(self.make(picked, Shape::Vector(size, ty)))
    }

    pub(crate) fn zero_value(&mut self, ty: Handle<naga::Type>) -> Result<Eval, String> {
        let shape = shape_of(self.module_ref, ty)?;
        let mut tys = Vec::new();
        scalar_tys(self.module_ref, shape, &mut tys)?;
        let vals = tys.into_iter().map(|t| (self.zero(t), t)).collect();
        Ok(self.make(vals, shape))
    }

    /// A componentwise op over one aggregate: the result keeps the operand's
    /// shape, with the scalar type the op produced.
    fn componentwise(&mut self, shape: Shape, out: Vals) -> Result<Eval, String> {
        let shape = match shape {
            Shape::Vector(n, _) => Shape::Vector(n, out[0].1),
            m @ Shape::Matrix { .. } => m,
            other => return Err(format!("componentwise operation on {other:?}")),
        };
        Ok(self.make(out, shape))
    }

    pub(crate) fn unary_agg(&mut self, op: UnaryOperator, e: Eval) -> Result<Eval, String> {
        let (vals, shape) = self.value_of(e)?;
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            out.push(self.unary_scalar(op, v));
        }
        self.componentwise(shape, out)
    }

    pub(crate) fn binary_agg(&mut self, op: BinaryOperator, l: Eval, r: Eval) -> Result<Eval, String> {
        let (lv, ls) = self.value_of(l)?;
        let (rv, rs) = self.value_of(r)?;
        if op == BinaryOperator::Multiply {
            match (ls, rs) {
                (Shape::Matrix { cols, rows }, Shape::Vector(..)) => {
                    let out = self.mat_vec(&lv, cols as usize, rows as usize, &rv);
                    return Ok(self.make(out, Shape::Vector(rows, Ty::F32)));
                }
                (Shape::Vector(..), Shape::Matrix { cols, rows }) => {
                    let r = rows as usize;
                    let out = (0..cols as usize).map(|c| (self.dot(&lv, &rv[c * r..c * r + r]), Ty::F32)).collect();
                    return Ok(self.make(out, Shape::Vector(cols, Ty::F32)));
                }
                (Shape::Matrix { cols: inner, rows }, Shape::Matrix { cols, rows: inner2 }) => {
                    if inner != inner2 {
                        return Err("matrix product of mismatched dimensions".into());
                    }
                    let k = inner as usize;
                    let mut out = Vec::new();
                    for c in 0..cols as usize {
                        out.extend(self.mat_vec(&lv, k, rows as usize, &rv[c * k..c * k + k]));
                    }
                    return Ok(self.make(out, Shape::Matrix { cols, rows }));
                }
                _ => {}
            }
        }
        let n = lv.len().max(rv.len());
        if (lv.len() != n && lv.len() != 1) || (rv.len() != n && rv.len() != 1) {
            return Err(format!("{op:?} of {ls:?} and {rs:?}"));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = lv[if lv.len() == 1 { 0 } else { i }];
            let b = rv[if rv.len() == 1 { 0 } else { i }];
            out.push(self.binary_scalar(op, a, b)?);
        }
        self.componentwise(if lv.len() == n { ls } else { rs }, out)
    }

    /// `m * v` for a column-major `cols x rows` matrix: `sum_c m[c] v[c]`.
    fn mat_vec(&mut self, m: &[(Value, Ty)], cols: usize, rows: usize, v: &[(Value, Ty)]) -> Vals {
        (0..rows)
            .map(|r| {
                let mut acc = self.b.ins().fmul(m[r].0, v[0].0);
                for c in 1..cols {
                    let p = self.b.ins().fmul(m[c * rows + r].0, v[c].0);
                    acc = self.b.ins().fadd(acc, p);
                }
                (acc, Ty::F32)
            })
            .collect()
    }

    /// Dot product, accumulated left to right.
    fn dot(&mut self, a: &[(Value, Ty)], b: &[(Value, Ty)]) -> Value {
        let float = a[0].1.is_float();
        let mul = |s: &mut Self, x: Value, y: Value| if float { s.b.ins().fmul(x, y) } else { s.b.ins().imul(x, y) };
        let mut acc = mul(self, a[0].0, b[0].0);
        for i in 1..a.len() {
            let p = mul(self, a[i].0, b[i].0);
            acc = if float { self.b.ins().fadd(acc, p) } else { self.b.ins().iadd(acc, p) };
        }
        acc
    }

    fn length(&mut self, a: &[(Value, Ty)]) -> Value {
        let d = self.dot(a, a);
        self.b.ins().sqrt(d)
    }

    /// Determinant of the `rows x cols` minor of an `n x n` column-major
    /// matrix, by cofactor expansion down its first column.
    fn det(&mut self, m: &[(Value, Ty)], n: usize, rows: &[usize], cols: &[usize]) -> Value {
        let e = |r: usize, c: usize| m[c * n + r].0;
        if rows.len() == 2 {
            let a = self.b.ins().fmul(e(rows[0], cols[0]), e(rows[1], cols[1]));
            let b = self.b.ins().fmul(e(rows[0], cols[1]), e(rows[1], cols[0]));
            return self.b.ins().fsub(a, b);
        }
        let mut acc = None;
        for (i, &r) in rows.iter().enumerate() {
            let minor_rows: Vec<usize> = rows.iter().copied().filter(|&x| x != r).collect();
            let minor = self.det(m, n, &minor_rows, &cols[1..]);
            let term = self.b.ins().fmul(e(r, cols[0]), minor);
            acc = Some(match acc {
                None => term,
                Some(a) if i % 2 == 1 => self.b.ins().fsub(a, term),
                Some(a) => self.b.ins().fadd(a, term),
            });
        }
        acc.expect("a matrix has at least two rows")
    }

    pub(crate) fn math_agg(
        &mut self,
        fun: MathFunction,
        a: Eval,
        b: Option<Eval>,
        c: Option<Eval>,
    ) -> Result<Eval, String> {
        use MathFunction::*;
        let (av, ashape) = self.value_of(a)?;
        let bv = match b {
            Some(e) => Some(self.value_of(e)?.0),
            None => None,
        };
        let cv = match c {
            Some(e) => Some(self.value_of(e)?.0),
            None => None,
        };
        let need = |v: &Option<Vals>| v.clone().ok_or(format!("{fun:?} needs more arguments"));
        match fun {
            Dot => {
                let bv = need(&bv)?;
                let ty = av[0].1;
                Ok(Eval::Scalar(self.dot(&av, &bv), ty))
            }
            Cross => {
                let bv = need(&bv)?;
                if av.len() != 3 || bv.len() != 3 {
                    return Err("cross of non-3-vectors".into());
                }
                let mut out = Vec::with_capacity(3);
                for (i, j) in [(1, 2), (2, 0), (0, 1)] {
                    let p = self.b.ins().fmul(av[i].0, bv[j].0);
                    let q = self.b.ins().fmul(av[j].0, bv[i].0);
                    out.push((self.b.ins().fsub(p, q), Ty::F32));
                }
                Ok(self.make(out, ashape))
            }
            Length => Ok(Eval::Scalar(self.length(&av), Ty::F32)),
            Distance => {
                let bv = need(&bv)?;
                let mut d = Vec::with_capacity(av.len());
                for (x, y) in av.iter().zip(&bv) {
                    d.push((self.b.ins().fsub(x.0, y.0), Ty::F32));
                }
                Ok(Eval::Scalar(self.length(&d), Ty::F32))
            }
            Normalize => {
                let len = self.length(&av);
                let out = av.iter().map(|x| (self.b.ins().fdiv(x.0, len), Ty::F32)).collect();
                Ok(self.make(out, ashape))
            }
            Determinant => match ashape {
                Shape::Matrix { cols, rows } if cols == rows => {
                    let n = cols as usize;
                    let idx: Vec<usize> = (0..n).collect();
                    Ok(Eval::Scalar(self.det(&av, n, &idx, &idx), Ty::F32))
                }
                other => Err(format!("determinant of {other:?}")),
            },
            _ => {
                let bcast = |v: &Option<Vals>, i: usize| v.as_ref().map(|v| v[if v.len() == 1 { 0 } else { i }]);
                let mut out = Vec::with_capacity(av.len());
                for (i, x) in av.iter().enumerate() {
                    out.push(self.math_scalar(fun, *x, bcast(&bv, i), bcast(&cv, i))?);
                }
                self.componentwise(ashape, out)
            }
        }
    }

    pub(crate) fn select_agg(&mut self, cond: Eval, accept: Eval, reject: Eval) -> Result<Eval, String> {
        let (cv, _) = self.value_of(cond)?;
        let (av, shape) = self.value_of(accept)?;
        let (rv, _) = self.value_of(reject)?;
        check_len(av.len(), rv.len())?;
        let mut out = Vec::with_capacity(av.len());
        for i in 0..av.len() {
            let c = cv[if cv.len() == 1 { 0 } else { i }].0;
            out.push((self.b.ins().select(c, av[i].0, rv[i].0), av[i].1));
        }
        Ok(self.make(out, shape))
    }

    pub(crate) fn relational(&mut self, fun: RelationalFunction, argument: Handle<Expression>) -> Result<Eval, String> {
        let (vals, shape) = self.value(argument)?;
        match fun {
            RelationalFunction::All | RelationalFunction::Any => {
                let mut acc = vals[0].0;
                for v in &vals[1..] {
                    acc = if fun == RelationalFunction::All {
                        self.b.ins().band(acc, v.0)
                    } else {
                        self.b.ins().bor(acc, v.0)
                    };
                }
                Ok(Eval::Scalar(acc, Ty::Bool))
            }
            RelationalFunction::IsNan | RelationalFunction::IsInf => {
                let mut out = Vec::with_capacity(vals.len());
                for (v, _) in vals {
                    let r = if fun == RelationalFunction::IsNan {
                        self.b.ins().fcmp(FloatCC::Unordered, v, v)
                    } else {
                        let a = self.b.ins().fabs(v);
                        let inf = self.b.ins().f32const(f32::INFINITY);
                        self.b.ins().fcmp(FloatCC::Equal, a, inf)
                    };
                    out.push((r, Ty::Bool));
                }
                match shape {
                    Shape::Scalar(_) => Ok(Eval::Scalar(out[0].0, Ty::Bool)),
                    s => self.componentwise(s, out),
                }
            }
        }
    }

    pub(crate) fn cast_agg(&mut self, e: Eval, kind: ScalarKind, convert: Option<u8>) -> Result<Eval, String> {
        let (vals, shape) = self.value_of(e)?;
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            out.push(self.cast_scalar(v, kind, convert)?);
        }
        self.componentwise(shape, out)
    }

    /// A module-scope constant's initialiser (the global expression arena).
    pub(crate) fn eval_global_const(&mut self, h: Handle<Expression>) -> Result<Eval, String> {
        let m = self.module_ref;
        match &m.global_expressions[h] {
            Expression::Literal(lit) => self.literal(lit),
            Expression::ZeroValue(ty) => self.zero_value(*ty),
            Expression::Constant(c) => self.eval_global_const(m.constants[*c].init),
            Expression::Compose { ty, components } => {
                let shape = shape_of(m, *ty)?;
                let mut vals = Vec::new();
                for c in components {
                    let e = self.eval_global_const(*c)?;
                    vals.extend(self.value_of(e)?.0);
                }
                check_len(flat_len(m, shape)? as usize, vals.len())?;
                Ok(self.make(vals, shape))
            }
            Expression::Splat { size, value } => {
                let e = self.eval_global_const(*value)?;
                let (v, _) = self.value_of(e)?;
                Ok(self.splat(*size, v[0]))
            }
            other => Err(format!("unsupported constant expression {other:?}")),
        }
    }
}

fn check_len(want: usize, got: usize) -> Result<(), String> {
    if want == got {
        Ok(())
    } else {
        Err(format!("aggregate of {got} scalars where {want} are expected"))
    }
}
