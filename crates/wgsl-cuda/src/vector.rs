// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vector operations, scalarised.
//!
//! Swedish Embedded AB implements portable GPU compute stacks, including the
//! kernel translators that keep one source running on several backends, for its
//! clients. If your team needs expertise in translating compute kernels between
//! GPU programming models then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! The generated tier has no CUDA vector type and no operator overloads to
//! lean on. A WGSL vector value is instead a list of scalar component
//! expressions, and every operation is the scalar operation applied lane by
//! lane - which is how WGSL defines them, and which means every rounding,
//! overflow and NaN decision the scalar translator already makes (and has
//! hazard notes for) applies unchanged to each lane.
//!
//! Anything that is genuinely a vector operation rather than a lane-wise one
//! (`dot`, `length`, `normalize`, `cross`, `distance`) is spelled out below
//! in the evaluation order the WGSL spec gives, left to right, so the sum of
//! products rounds the way the reference rounds it.

use naga::{MathFunction, ScalarKind};

use super::{binary, cast, math, BinaryOperator, Eval, Ty};

/// One operand of a lane-wise operation: its component expressions, and whether
/// it was a vector (a scalar operand is splatted to the other operand's width).
#[derive(Clone)]
pub(super) struct Lanes {
    pub comps: Vec<String>,
    pub ty: Ty,
    pub vector: bool,
}

impl Lanes {
    pub(super) fn scalar(text: String, ty: Ty) -> Lanes {
        Lanes { comps: vec![text], ty, vector: false }
    }

    pub(super) fn vector(comps: Vec<String>, ty: Ty) -> Lanes {
        Lanes { comps, ty, vector: true }
    }

    /// Component `i`, repeating a scalar for every lane.
    fn at(&self, i: usize) -> &str {
        if self.vector {
            &self.comps[i]
        } else {
            &self.comps[0]
        }
    }
}

/// The widest operand decides the result width; two vectors must agree.
fn width(ops: &[&Lanes]) -> Result<usize, String> {
    let mut n = 1;
    for o in ops.iter().filter(|o| o.vector) {
        if n != 1 && o.comps.len() != n {
            return Err(format!("vector operands of width {n} and {} cannot be combined", o.comps.len()));
        }
        n = o.comps.len();
    }
    Ok(n)
}

/// Gather per-lane scalar results back into a vector value.
fn gather(parts: Vec<Eval>) -> Result<Eval, String> {
    let mut comps = Vec::with_capacity(parts.len());
    let mut ty = None;
    for p in parts {
        match p {
            Eval::Value(text, t) => {
                comps.push(text);
                ty = Some(t);
            }
            _ => return Err("a lane-wise operation produced something that is not a scalar".into()),
        }
    }
    Ok(Eval::Vector(comps, ty.ok_or("a vector with no components")?))
}

pub(super) fn binary_lanes(op: BinaryOperator, l: &Lanes, r: &Lanes) -> Result<Eval, String> {
    if !l.vector && !r.vector {
        return binary(op, l.at(0), l.ty, r.at(0), r.ty);
    }
    let n = width(&[l, r])?;
    gather((0..n).map(|i| binary(op, l.at(i), l.ty, r.at(i), r.ty)).collect::<Result<_, _>>()?)
}

pub(super) fn unary_lanes(f: impl Fn(&str, Ty) -> String, x: &Lanes) -> Eval {
    if !x.vector {
        return Eval::Value(f(x.at(0), x.ty), x.ty);
    }
    Eval::Vector(x.comps.iter().map(|c| f(c, x.ty)).collect(), x.ty)
}

pub(super) fn cast_lanes(x: &Lanes, kind: ScalarKind, convert: Option<u8>) -> Result<Eval, String> {
    if !x.vector {
        return cast(x.at(0), x.ty, kind, convert);
    }
    gather(x.comps.iter().map(|c| cast(c, x.ty, kind, convert)).collect::<Result<_, _>>()?)
}

/// `select(reject, accept, cond)`, where `cond` is one bool for the whole value
/// or one per lane.
pub(super) fn select_lanes(cond: &Lanes, accept: &Lanes, reject: &Lanes) -> Result<Eval, String> {
    if !accept.vector && !reject.vector && !cond.vector {
        let (c, a, r) = (cond.at(0), accept.at(0), reject.at(0));
        return Ok(Eval::Value(format!("(({c}) ? ({a}) : ({r}))"), accept.ty));
    }
    let n = width(&[cond, accept, reject])?;
    Ok(Eval::Vector(
        (0..n).map(|i| format!("(({}) ? ({}) : ({}))", cond.at(i), accept.at(i), reject.at(i))).collect(),
        accept.ty,
    ))
}

/// `a0*b0 + a1*b1 + ...`, summed left to right as the spec orders it.
fn dot(a: &Lanes, b: &Lanes) -> Result<String, String> {
    let n = width(&[a, b])?;
    let mut sum = format!("(({}) * ({}))", a.at(0), b.at(0));
    for i in 1..n {
        sum = format!("(({sum}) + (({}) * ({})))", a.at(i), b.at(i));
    }
    Ok(sum)
}

pub(super) fn math_lanes(fun: MathFunction, args: &[Lanes]) -> Result<Eval, String> {
    use MathFunction::*;
    let first = args.first().ok_or("a math function with no arguments")?;
    if args.iter().all(|a| !a.vector) {
        let rest: Vec<(String, Ty)> = args[1..].iter().map(|a| (a.at(0).to_string(), a.ty)).collect();
        return math(fun, (first.at(0), first.ty), &rest);
    }
    match fun {
        Dot => return Ok(Eval::Value(dot(first, &args[1])?, Ty::F32)),
        Length => return Ok(Eval::Value(format!("sqrtf({})", dot(first, first)?), Ty::F32)),
        Distance => {
            let d = difference(first, &args[1])?;
            return Ok(Eval::Value(format!("sqrtf({})", dot(&d, &d)?), Ty::F32));
        }
        Normalize => {
            let len = format!("sqrtf({})", dot(first, first)?);
            return Ok(Eval::Vector(first.comps.iter().map(|c| format!("(({c}) / ({len}))")).collect(), Ty::F32));
        }
        Cross => {
            let (a, b) = (first, &args[1]);
            if a.comps.len() != 3 || b.comps.len() != 3 {
                return Err("cross needs two vec3 operands".into());
            }
            let c = |i: usize, j: usize| format!("((({}) * ({})) - (({}) * ({})))", a.comps[i], b.comps[j], a.comps[j], b.comps[i]);
            return Ok(Eval::Vector(vec![c(1, 2), c(2, 0), c(0, 1)], Ty::F32));
        }
        _ => {}
    }
    // Everything else is lane-wise; a scalar argument (the `t` of `mix`, a
    // splatted clamp bound) repeats for every lane.
    let refs: Vec<&Lanes> = args.iter().collect();
    let n = width(&refs)?;
    gather(
        (0..n)
            .map(|i| {
                let rest: Vec<(String, Ty)> = args[1..].iter().map(|a| (a.at(i).to_string(), a.ty)).collect();
                math(fun, (first.at(i), first.ty), &rest)
            })
            .collect::<Result<_, _>>()?,
    )
}

/// `a - b`, lane-wise, as a vector operand.
fn difference(a: &Lanes, b: &Lanes) -> Result<Lanes, String> {
    match binary_lanes(BinaryOperator::Subtract, a, b)? {
        Eval::Vector(comps, ty) => Ok(Lanes::vector(comps, ty)),
        Eval::Value(text, ty) => Ok(Lanes::scalar(text, ty)),
        Eval::Place(_) | Eval::Agg(..) | Eval::Matrix(..) => Err("a difference of vectors is a vector".into()),
    }
}

/// `all(v)` / `any(v)` over a vector of bools.
pub(super) fn reduce_bool(all: bool, x: &Lanes) -> Eval {
    let op = if all { " && " } else { " || " };
    let body = x.comps.iter().map(|c| format!("({c})")).collect::<Vec<_>>().join(op);
    Eval::Value(format!("({body})"), Ty::Bool)
}
