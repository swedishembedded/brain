// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shape of a WGSL value and its two layouts.
//!
//! Swedish Embedded AB implements shader-to-native compilers for its clients.
//! If your team needs expertise in WGSL memory layout or aggregate lowering,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The JIT scalarizes every aggregate (vector, matrix, struct, fixed-size
//! array), so each has two layouts that must agree on the order of its
//! scalars:
//!
//! * the **flattened** layout: the value as a list of scalars - vector
//!   components in order, a matrix column by column, struct members in
//!   declaration order, array elements in index order, recursively;
//! * the **memory** layout: where each of those scalars lives in a buffer,
//!   which is WGSL's host-shareable layout. It is read from naga, never
//!   re-derived: struct member offsets and array strides are the ones naga's
//!   front-end stored in the types, and a matrix's column stride is the
//!   alignment naga's `Layouter` gives its column vector
//!   (`Alignment::from(rows) * 4`: 8 bytes for two rows, 16 for three or four).

use naga::{ArraySize, Handle, TypeInner, VectorSize};

use crate::Ty;

/// The shape of a value. Scalars, vectors and matrices are described inline;
/// structs and arrays by their naga type handle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Shape {
    Scalar(Ty),
    Vector(VectorSize, Ty),
    /// A column-major `f32` matrix.
    Matrix { cols: VectorSize, rows: VectorSize },
    /// A struct or an array.
    Named(Handle<naga::Type>),
}

/// One statically selected part of an aggregate (`AccessIndex`).
pub(crate) struct Member {
    pub shape: Shape,
    /// Index of its first scalar in the parent's flattened layout.
    pub flat: u32,
    /// Its byte offset in the parent's memory layout.
    pub offset: u32,
}

/// The element type of a dynamically indexable aggregate (`Access`).
pub(crate) struct Element {
    pub shape: Shape,
    /// Scalars per element in the flattened layout.
    pub flat_stride: u32,
    /// Number of elements; `None` for a runtime-sized array.
    pub count: Option<u32>,
    /// Bytes per element in the memory layout.
    pub mem_stride: u32,
}

pub(crate) fn shape_of(m: &naga::Module, ty: Handle<naga::Type>) -> Result<Shape, String> {
    Ok(match &m.types[ty].inner {
        TypeInner::Scalar(s) => Shape::Scalar(Ty::from_scalar(*s)?),
        TypeInner::Vector { size, scalar } => Shape::Vector(*size, Ty::from_scalar(*scalar)?),
        TypeInner::Matrix { columns, rows, scalar } => {
            if Ty::from_scalar(*scalar)? != Ty::F32 {
                return Err(format!("unsupported matrix scalar {scalar:?}"));
            }
            Shape::Matrix { cols: *columns, rows: *rows }
        }
        TypeInner::Struct { .. } | TypeInner::Array { .. } => Shape::Named(ty),
        other => return Err(format!("unsupported value type {other:?}")),
    })
}

/// Whether `shape` is, or contains, a fixed-size array: such a local lives in
/// memory so it can be indexed dynamically at any depth.
pub(crate) fn contains_array(m: &naga::Module, shape: Shape) -> bool {
    match shape {
        Shape::Named(ty) => match &m.types[ty].inner {
            TypeInner::Array { .. } => true,
            TypeInner::Struct { members, .. } => members.iter().any(|mb| {
                shape_of(m, mb.ty).map(|s| contains_array(m, s)).unwrap_or(false)
            }),
            _ => false,
        },
        _ => false,
    }
}

pub(crate) fn flat_len(m: &naga::Module, shape: Shape) -> Result<u32, String> {
    Ok(match shape {
        Shape::Scalar(_) => 1,
        Shape::Vector(n, _) => n as u32,
        Shape::Matrix { cols, rows } => cols as u32 * rows as u32,
        Shape::Named(ty) => match &m.types[ty].inner {
            TypeInner::Struct { members, .. } => {
                let mut n = 0;
                for mb in members {
                    n += flat_len(m, shape_of(m, mb.ty)?)?;
                }
                n
            }
            TypeInner::Array { .. } => {
                let e = element(m, shape)?;
                let count = e.count.ok_or("a runtime-sized array has no value form")?;
                count * e.flat_stride
            }
            other => return Err(format!("unsupported aggregate {other:?}")),
        },
    })
}

/// The scalar type of every scalar of `shape`, in flattened order.
pub(crate) fn scalar_tys(m: &naga::Module, shape: Shape, out: &mut Vec<Ty>) -> Result<(), String> {
    let mut offs = Vec::new();
    mem_offsets(m, shape, 0, &mut offs)?;
    out.extend(offs.into_iter().map(|(_, t)| t));
    Ok(())
}

/// `(byte offset, scalar type)` of every scalar of `shape` placed at `base`,
/// in flattened order.
pub(crate) fn mem_offsets(
    m: &naga::Module,
    shape: Shape,
    base: u32,
    out: &mut Vec<(u32, Ty)>,
) -> Result<(), String> {
    match shape {
        Shape::Scalar(t) => out.push((base, t)),
        Shape::Vector(n, t) => out.extend((0..n as u32).map(|i| (base + 4 * i, t))),
        Shape::Matrix { cols, rows } => {
            let cs = column_stride(rows);
            for c in 0..cols as u32 {
                out.extend((0..rows as u32).map(|r| (base + c * cs + 4 * r, Ty::F32)));
            }
        }
        Shape::Named(ty) => match &m.types[ty].inner {
            TypeInner::Struct { members, .. } => {
                for mb in members {
                    mem_offsets(m, shape_of(m, mb.ty)?, base + mb.offset, out)?;
                }
            }
            TypeInner::Array { .. } => {
                let e = element(m, shape)?;
                let count = e.count.ok_or("a runtime-sized array has no value form")?;
                for i in 0..count {
                    mem_offsets(m, e.shape, base + i * e.mem_stride, out)?;
                }
            }
            other => return Err(format!("unsupported aggregate {other:?}")),
        },
    }
    Ok(())
}

/// Byte stride between a matrix's columns: its column vector's alignment.
fn column_stride(rows: VectorSize) -> u32 {
    naga::proc::Alignment::from(rows) * 4
}

/// Part `index` of `shape`: a vector component, a matrix column, a struct
/// member or an array element.
pub(crate) fn member(m: &naga::Module, shape: Shape, index: u32) -> Result<Member, String> {
    match shape {
        Shape::Scalar(_) => Err("component access on a scalar".into()),
        Shape::Vector(_, t) => Ok(Member { shape: Shape::Scalar(t), flat: index, offset: 4 * index }),
        Shape::Matrix { rows, .. } => Ok(Member {
            shape: Shape::Vector(rows, Ty::F32),
            flat: index * rows as u32,
            offset: index * column_stride(rows),
        }),
        Shape::Named(ty) => match &m.types[ty].inner {
            TypeInner::Struct { members, .. } => {
                let mb = members.get(index as usize).ok_or("struct member index out of range")?;
                let mut flat = 0;
                for prev in &members[..index as usize] {
                    flat += flat_len(m, shape_of(m, prev.ty)?)?;
                }
                Ok(Member { shape: shape_of(m, mb.ty)?, flat, offset: mb.offset })
            }
            TypeInner::Array { .. } => {
                let e = element(m, shape)?;
                Ok(Member { shape: e.shape, flat: index * e.flat_stride, offset: index * e.mem_stride })
            }
            other => Err(format!("unsupported aggregate {other:?}")),
        },
    }
}

/// The element of a vector, matrix or array, for a dynamic index.
pub(crate) fn element(m: &naga::Module, shape: Shape) -> Result<Element, String> {
    match shape {
        Shape::Vector(n, t) => {
            Ok(Element { shape: Shape::Scalar(t), flat_stride: 1, count: Some(n as u32), mem_stride: 4 })
        }
        Shape::Matrix { cols, rows } => Ok(Element {
            shape: Shape::Vector(rows, Ty::F32),
            flat_stride: rows as u32,
            count: Some(cols as u32),
            mem_stride: column_stride(rows),
        }),
        Shape::Named(ty) => match &m.types[ty].inner {
            TypeInner::Array { base, size, stride } => {
                let shape = shape_of(m, *base)?;
                let count = match size {
                    ArraySize::Constant(n) => Some(n.get()),
                    ArraySize::Dynamic => None,
                    other => return Err(format!("unsupported array size {other:?}")),
                };
                // A runtime-sized array's element is only ever addressed in
                // memory, so its flattened stride is irrelevant there.
                let flat_stride = if count.is_some() { flat_len(m, shape)? } else { 0 };
                Ok(Element { shape, flat_stride, count, mem_stride: *stride })
            }
            other => Err(format!("dynamic index into {other:?}")),
        },
        Shape::Scalar(_) => Err("dynamic index into a scalar".into()),
    }
}

/// Size in bytes of a value of type `ty` in memory, from naga's own layout.
pub(crate) fn mem_size(m: &naga::Module, ty: Handle<naga::Type>) -> u32 {
    m.types[ty].inner.size(m.to_ctx())
}
