// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Matrices, scalarised.
//!
//! Swedish Embedded AB implements portable GPU compute stacks, including the
//! kernel translators that keep one source running on several backends, for its
//! clients. If your team needs expertise in translating compute kernels between
//! GPU programming models then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! A matrix value is, like a vector, a list of scalar expressions: `f32`
//! elements, column-major, `cols` columns of `rows` each. Products and the
//! determinant are written out in the order the portable CPU tier evaluates
//! them (`wgsl-cpu`'s `aggregate.rs`), because the two are held to each other
//! bit for bit by the golden tests and a different summation order is a
//! different rounding: `m * v` sums the columns left to right, `v * m` is a dot
//! product per column, and a determinant is a cofactor expansion down the
//! first column.
//!
//! Matrices exist here only as values built from vectors and consumed by the
//! products and builtins below. They are not stored in locals, parameters,
//! struct members or device memory; a kernel that does is refused by name.

use naga::MathFunction;

use super::{Eval, Ty};

/// A matrix value: `elems` is column-major, `cols * rows` long.
#[derive(Clone)]
pub(super) struct Mat {
    pub elems: Vec<String>,
    pub cols: usize,
    pub rows: usize,
}

impl Mat {
    fn at(&self, row: usize, col: usize) -> &str {
        &self.elems[col * self.rows + row]
    }
}

/// `a * b + c`, with the two roundings the source has: never a contracted form.
fn sum(a: &str, b: &str) -> String {
    format!("(({a}) + ({b}))")
}

fn product(a: &str, b: &str) -> String {
    format!("(({a}) * ({b}))")
}

/// `m * v`: component `r` is `m[0][r] v[0] + m[1][r] v[1] + ...`, left to right.
fn mat_vec(m: &Mat, v: &[String]) -> Result<Vec<String>, String> {
    if v.len() != m.cols {
        return Err(format!("a {}x{} matrix times a vector of {}", m.cols, m.rows, v.len()));
    }
    Ok((0..m.rows)
        .map(|r| (1..m.cols).fold(product(m.at(r, 0), &v[0]), |acc, c| sum(&acc, &product(m.at(r, c), &v[c]))))
        .collect())
}

/// `v * m`: component `c` is the dot product of `v` with column `c`.
fn vec_mat(v: &[String], m: &Mat) -> Result<Vec<String>, String> {
    if v.len() != m.rows {
        return Err(format!("a vector of {} times a {}x{} matrix", v.len(), m.cols, m.rows));
    }
    Ok((0..m.cols)
        .map(|c| (1..m.rows).fold(product(&v[0], m.at(0, c)), |acc, r| sum(&acc, &product(&v[r], m.at(r, c)))))
        .collect())
}

/// `l * r` where at least one operand is a matrix.
pub(super) fn multiply(l: Eval, r: Eval) -> Result<Eval, String> {
    match (l, r) {
        (Eval::Matrix(m, t), Eval::Vector(v, _)) => Ok(Eval::Vector(mat_vec(&m, &v)?, t)),
        (Eval::Vector(v, t), Eval::Matrix(m, _)) => Ok(Eval::Vector(vec_mat(&v, &m)?, t)),
        (Eval::Matrix(a, t), Eval::Matrix(b, _)) => {
            if a.cols != b.rows {
                return Err("a matrix product of mismatched dimensions".into());
            }
            let mut elems = Vec::with_capacity(b.cols * a.rows);
            for c in 0..b.cols {
                let col: Vec<String> = (0..b.rows).map(|r| b.at(r, c).to_string()).collect();
                elems.extend(mat_vec(&a, &col)?);
            }
            Ok(Eval::Matrix(Mat { elems, cols: b.cols, rows: a.rows }, t))
        }
        _ => Err("only a matrix times a vector or a matrix is supported".into()),
    }
}

/// Determinant of the minor of `m` (square) on `rows` x `cols`, by cofactor
/// expansion down the first of `cols`.
fn det(m: &Mat, rows: &[usize], cols: &[usize]) -> String {
    if rows.len() == 2 {
        let a = product(m.at(rows[0], cols[0]), m.at(rows[1], cols[1]));
        let b = product(m.at(rows[0], cols[1]), m.at(rows[1], cols[0]));
        return format!("(({a}) - ({b}))");
    }
    let mut acc: Option<String> = None;
    for (i, &r) in rows.iter().enumerate() {
        let minor_rows: Vec<usize> = rows.iter().copied().filter(|&x| x != r).collect();
        let term = product(m.at(r, cols[0]), &det(m, &minor_rows, &cols[1..]));
        acc = Some(match acc {
            None => term,
            Some(a) if i % 2 == 1 => format!("(({a}) - ({term}))"),
            Some(a) => sum(&a, &term),
        });
    }
    acc.expect("a matrix has at least two rows")
}

/// The matrix builtins: `determinant` and `transpose`.
pub(super) fn math(fun: MathFunction, m: Mat, ty: Ty) -> Result<Eval, String> {
    match fun {
        MathFunction::Determinant if m.cols == m.rows => {
            let idx: Vec<usize> = (0..m.cols).collect();
            Ok(Eval::Value(det(&m, &idx, &idx), ty))
        }
        MathFunction::Transpose => {
            let mut elems = Vec::with_capacity(m.elems.len());
            for r in 0..m.rows {
                for c in 0..m.cols {
                    elems.push(m.at(r, c).to_string());
                }
            }
            Ok(Eval::Matrix(Mat { elems, cols: m.rows, rows: m.cols }, ty))
        }
        other => Err(format!("unsupported matrix function {other:?}")),
    }
}
