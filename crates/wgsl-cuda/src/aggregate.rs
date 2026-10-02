// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structs and arrays of structs.
//!
//! Swedish Embedded AB implements portable GPU compute stacks, including the
//! kernel translators that keep one source running on several backends, for its
//! clients. If your team needs expertise in translating compute kernels between
//! GPU programming models then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! # Two representations, and why they never mix
//!
//! A struct has two homes in a generated kernel, and each gets the
//! representation that cannot be wrong for it:
//!
//! * **Kernel-private** (a `var`, a function argument or result, a temporary):
//!   a real C++ `struct`, `__S<n>`, whose members are `m0, m1, ...` in
//!   declaration order. Nothing outside the kernel sees its layout, so C++'s is
//!   as good as any, and copies, whole-struct assignment and zero-initialisation
//!   (`= {}`, which is what WGSL's zero-initialised `var` means) are the
//!   language's own. A vector member is `T m[n]`; an array of vectors is the
//!   flat `T m[count * stride]` a storage array of vectors already is, so one
//!   `VecArrayBase` place serves both.
//! * **Device memory** (the uniform block and storage bindings): never a C++
//!   struct - hazard 2. A position in memory is a pointer plus a *byte offset*
//!   taken from naga's own layout of the WGSL type, refined member by member
//!   and element by element, and a leaf is read or written through the 32-bit
//!   word at that offset. Reading a whole struct builds the private struct
//!   from its leaves; writing one takes it apart again.

use std::fmt::Write as _;

use naga::{ArraySize, Handle, TypeInner};

use super::{clamped, coerce, ident_of, vector_shape, Eval, Gen, Place, Ty};

/// The C++ member name of struct member `index`.
fn member(index: usize) -> String {
    format!("m{index}")
}

/// Byte offset `base + add`, as C++ text. Offsets are `size_t`: a storage
/// binding can be larger than a 32-bit byte offset reaches.
pub(super) fn offset_plus(base: &str, add: u32) -> String {
    if add == 0 {
        base.to_string()
    } else {
        format!("({base} + {add}u)")
    }
}

/// The read helper for a 32-bit scalar kind.
fn read_helper(ty: Ty) -> Result<&'static str, String> {
    match ty {
        Ty::F32 => Ok("__brain_uf32"),
        Ty::I32 => Ok("__brain_ui32"),
        Ty::U32 => Ok("__brain_uu32"),
        Ty::Bool => Err("a bool in device memory is unsupported".into()),
    }
}

/// The write helper for a 32-bit scalar kind.
pub(super) fn write_helper(ty: Ty) -> Result<&'static str, String> {
    match ty {
        Ty::F32 => Ok("__brain_sf32"),
        Ty::I32 => Ok("__brain_si32"),
        Ty::U32 => Ok("__brain_su32"),
        Ty::Bool => Err("a bool in device memory is unsupported".into()),
    }
}

impl Gen<'_> {
    /// The C++ name of struct type `ty`, defining it (and the structs it holds)
    /// on first use. Definitions are emitted ahead of the kernel, innermost
    /// first, which is the order they are registered in.
    pub(super) fn cpp_type(&mut self, ty: Handle<naga::Type>) -> Result<String, String> {
        if let Some(name) = self.struct_names.get(&ty) {
            return Ok(name.clone());
        }
        if let TypeInner::Array { size: ArraySize::Constant(n), base, .. } = &self.m.types[ty].inner {
            // An array VALUE (a parameter, a result, a struct member) is a
            // struct around the elements, so it copies and assigns like any
            // other value; `v` is the flat array a place indexes into.
            let (n, base) = (n.get(), *base);
            let (elem, count) = match &self.m.types[base].inner {
                TypeInner::Scalar(s) => (Ty::from_scalar(*s)?.c().to_string(), n),
                TypeInner::Vector { .. } => {
                    let (t, _, stride) = vector_shape(self.m, base).expect("a vector")?;
                    (t.c().to_string(), n * stride)
                }
                TypeInner::Struct { .. } => (self.cpp_type(base)?, n),
                other => return Err(format!("an array of {other:?} is unsupported")),
            };
            let name = format!("__A{}", self.struct_names.len());
            self.struct_defs.push(format!("struct {name} {{ {elem} v[{count}]; }};\n"));
            self.struct_names.insert(ty, name.clone());
            return Ok(name);
        }
        let TypeInner::Struct { members, .. } = &self.m.types[ty].inner else {
            return Err(format!("expected a struct, got {:?}", self.m.types[ty].inner));
        };
        let mut body = String::new();
        for (i, mem) in members.iter().enumerate() {
            let decl = self.member_decl(mem.ty, &member(i))?;
            let _ = writeln!(body, "  {decl};");
        }
        let label = self.m.types[ty].name.as_deref().map(ident_of).unwrap_or_default();
        let name = format!("__S{}_{label}", self.struct_names.len());
        self.struct_defs.push(format!("struct {name} {{\n{body}}};\n"));
        self.struct_names.insert(ty, name.clone());
        Ok(name)
    }

    /// `T name` / `T name[n]` / `S name` for a member (or array element) of type `ty`.
    fn member_decl(&mut self, ty: Handle<naga::Type>, name: &str) -> Result<String, String> {
        match &self.m.types[ty].inner {
            TypeInner::Scalar(s) => Ok(format!("{} {name}", Ty::from_scalar(*s)?.c())),
            TypeInner::Vector { size, scalar } => Ok(format!("{} {name}[{}]", Ty::from_scalar(*scalar)?.c(), *size as u32)),
            TypeInner::Struct { .. } => Ok(format!("{} {name}", self.cpp_type(ty)?)),
            TypeInner::Array { size: ArraySize::Constant(_), .. } => Ok(format!("{} {name}", self.cpp_type(ty)?)),
            other => Err(format!("unsupported member type {other:?}")),
        }
    }

    /// The place a kernel-private C++ lvalue `lv` of WGSL type `ty` is.
    pub(super) fn private_place(&mut self, lv: String, ty: Handle<naga::Type>) -> Result<Place, String> {
        match &self.m.types[ty].inner {
            TypeInner::Scalar(s) => Ok(Place::Lvalue(lv, Ty::from_scalar(*s)?)),
            TypeInner::Vector { size, scalar } => {
                Ok(Place::VecRef { base: format!("{lv}[0]"), n: *size as u32, ty: Ty::from_scalar(*scalar)? })
            }
            TypeInner::Struct { .. } => {
                self.cpp_type(ty)?;
                Ok(Place::Struct { lv, ty })
            }
            TypeInner::Array { size: ArraySize::Constant(_), .. } => {
                self.cpp_type(ty)?;
                Ok(Place::Array { lv, ty })
            }
            other => Err(format!("unsupported type {other:?}")),
        }
    }

    /// Member `index` of struct type `sty`: its type and byte offset.
    fn member_of(&self, sty: Handle<naga::Type>, index: u32) -> Result<(Handle<naga::Type>, u32), String> {
        match &self.m.types[sty].inner {
            TypeInner::Struct { members, .. } => members
                .get(index as usize)
                .map(|m| (m.ty, m.offset))
                .ok_or_else(|| format!("struct member {index} out of range")),
            other => Err(format!("member access on {other:?}")),
        }
    }

    /// Member `index` of the kernel-private struct lvalue `lv`.
    pub(super) fn struct_member_place(&mut self, lv: &str, sty: Handle<naga::Type>, index: u32) -> Result<Place, String> {
        let (mty, _) = self.member_of(sty, index)?;
        self.private_place(format!("{lv}.{}", member(index as usize)), mty)
    }

    /// Member `index` of a struct VALUE.
    pub(super) fn struct_member_value(&mut self, text: &str, sty: Handle<naga::Type>, index: u32) -> Result<Eval, String> {
        let place = self.struct_member_place(&format!("({text})"), sty, index)?;
        self.load_place(place)
    }

    /// What reading `place` produces, for the places that hold a value.
    pub(super) fn load_place(&mut self, place: Place) -> Result<Eval, String> {
        match place {
            Place::Lvalue(lv, t) => Ok(Eval::Value(lv, t)),
            Place::VecRef { base, n, ty } => Ok(Eval::Vector((0..n as usize).map(|c| super::vec_comp(&base, c)).collect(), ty)),
            Place::Struct { lv, ty } | Place::Array { lv, ty } => Ok(Eval::Agg(lv, ty)),
            Place::MemScalar { ptr, off, ty } => Ok(Eval::Value(format!("{}({ptr}, {off})", read_helper(ty)?), ty)),
            Place::Mem { ptr, off, ty } => match &self.m.types[ty].inner {
                TypeInner::Vector { size, scalar } => {
                    let t = Ty::from_scalar(*scalar)?;
                    let lanes = (0..*size as u32)
                        .map(|c| Ok(format!("{}({ptr}, {})", read_helper(t)?, offset_plus(&off, 4 * c))))
                        .collect::<Result<Vec<_>, String>>()?;
                    Ok(Eval::Vector(lanes, t))
                }
                TypeInner::Struct { .. } => {
                    let name = self.cpp_type(ty)?;
                    Ok(Eval::Agg(format!("{name}{}", self.mem_init(&ptr, &off, ty)?), ty))
                }
                other => Err(format!("cannot load {other:?} from device memory")),
            },
            _ => Err("load from an unindexed array".into()),
        }
    }

    /// The memory place at byte offset `off` of `ptr` holding a `ty`.
    pub(super) fn mem_place(&self, ptr: String, off: String, ty: Handle<naga::Type>) -> Result<Place, String> {
        match &self.m.types[ty].inner {
            TypeInner::Scalar(s) => Ok(Place::MemScalar { ptr, off, ty: Ty::from_scalar(*s)? }),
            TypeInner::Array { base, size: ArraySize::Constant(n), stride } => {
                Ok(Place::MemArray { ptr, off, elem: *base, stride: *stride, nel: n.get().to_string() })
            }
            TypeInner::Array { .. } => Err("a runtime-sized array is only valid as a storage binding".into()),
            _ => Ok(Place::Mem { ptr, off, ty }),
        }
    }

    /// Member `index` of the struct held in memory at `off`.
    pub(super) fn mem_member(&self, ptr: String, off: &str, sty: Handle<naga::Type>, index: u32) -> Result<Place, String> {
        let (mty, moff) = self.member_of(sty, index)?;
        self.mem_place(ptr, offset_plus(off, moff), mty)
    }

    /// Element `index` of the array held in memory, clamped to the array.
    pub(super) fn mem_element(
        &self,
        ptr: String,
        off: &str,
        elem: Handle<naga::Type>,
        stride: u32,
        nel: &str,
        index: &str,
    ) -> Result<Place, String> {
        let at = format!("({off} + {} * {stride}u)", clamped(&format!("(size_t)({index})"), nel));
        self.mem_place(ptr, at, elem)
    }

    /// Component `index` (a C++ expression) of the vector held in memory.
    pub(super) fn mem_component(&self, ptr: String, off: &str, ty: Handle<naga::Type>, index: &str) -> Result<Place, String> {
        let TypeInner::Vector { size, scalar } = &self.m.types[ty].inner else {
            return Err("component access on something that is not a vector".into());
        };
        let at = format!("({off} + {} * 4u)", clamped(&format!("(size_t)({index})"), &(*size as u32).to_string()));
        Ok(Place::MemScalar { ptr, off: at, ty: Ty::from_scalar(*scalar)? })
    }

    /// The C++ initialiser of a `ty` read out of memory at `off`: the brace
    /// list that builds the private struct, or a bare read for a scalar.
    fn mem_init(&mut self, ptr: &str, off: &str, ty: Handle<naga::Type>) -> Result<String, String> {
        match &self.m.types[ty].inner {
            TypeInner::Scalar(s) => {
                let t = Ty::from_scalar(*s)?;
                Ok(format!("{}({ptr}, {off})", read_helper(t)?))
            }
            TypeInner::Vector { size, scalar } => {
                let t = Ty::from_scalar(*scalar)?;
                let lanes = (0..*size as u32)
                    .map(|c| Ok(format!("{}({ptr}, {})", read_helper(t)?, offset_plus(off, 4 * c))))
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(format!("{{{}}}", lanes.join(", ")))
            }
            TypeInner::Struct { members, .. } => {
                let members: Vec<(Handle<naga::Type>, u32)> = members.iter().map(|m| (m.ty, m.offset)).collect();
                let mut parts = Vec::with_capacity(members.len());
                for (mty, moff) in members {
                    let at = offset_plus(off, moff);
                    parts.push(match &self.m.types[mty].inner {
                        TypeInner::Struct { .. } => format!("{}{}", self.cpp_type(mty)?, self.mem_init(ptr, &at, mty)?),
                        _ => self.mem_init(ptr, &at, mty)?,
                    });
                }
                Ok(format!("{{{}}}", parts.join(", ")))
            }
            TypeInner::Array { base, size: ArraySize::Constant(n), stride } => {
                let mut parts = Vec::new();
                for e in 0..n.get() {
                    let at = offset_plus(off, e * stride);
                    match &self.m.types[*base].inner {
                        TypeInner::Scalar(_) | TypeInner::Struct { .. } => {
                            let init = self.mem_init(ptr, &at, *base)?;
                            parts.push(match &self.m.types[*base].inner {
                                TypeInner::Struct { .. } => format!("{}{init}", self.cpp_type(*base)?),
                                _ => init,
                            });
                        }
                        // An array of vectors is the flat word array, a vec3
                        // padded to its four-word stride (the pad is zero: it
                        // is never read back out).
                        TypeInner::Vector { size, scalar } => {
                            let t = Ty::from_scalar(*scalar)?;
                            let (_, _, words) = vector_shape(self.m, *base).expect("a vector")?;
                            for c in 0..words {
                                parts.push(if c < *size as u32 {
                                    format!("{}({ptr}, {})", read_helper(t)?, offset_plus(&at, 4 * c))
                                } else {
                                    t.zero().to_string()
                                });
                            }
                        }
                        other => return Err(format!("an array of {other:?} is unsupported")),
                    }
                }
                // The array is the wrapper's one member.
                Ok(format!("{{{{{}}}}}", parts.join(", ")))
            }
            other => Err(format!("cannot read {other:?} from device memory")),
        }
    }

    /// Write `value` into memory at `off`, as statements.
    pub(super) fn mem_store(
        &mut self,
        ptr: &str,
        off: &str,
        ty: Handle<naga::Type>,
        value: Eval,
        out: &mut String,
        pad: &str,
    ) -> Result<(), String> {
        match (&self.m.types[ty].inner, value) {
            (TypeInner::Vector { size, scalar }, Eval::Vector(lanes, vt)) => {
                if lanes.len() != *size as usize {
                    return Err("a vector store needs a vector of the same width".into());
                }
                let t = Ty::from_scalar(*scalar)?;
                for (c, lane) in lanes.iter().enumerate() {
                    let lane = coerce(lane, vt, t)?;
                    let _ = writeln!(out, "{pad}{}({ptr}, {}, {lane});", write_helper(t)?, offset_plus(off, 4 * c as u32));
                }
                Ok(())
            }
            (TypeInner::Struct { .. }, Eval::Agg(text, _)) => {
                let name = self.cpp_type(ty)?;
                let tmp = format!("__st{}", self.n_tmp);
                self.n_tmp += 1;
                let _ = writeln!(out, "{pad}{{");
                let _ = writeln!(out, "{pad}  const {name} {tmp} = {text};");
                self.store_cpp(ptr, off, ty, &tmp, out, &format!("{pad}  "))?;
                let _ = writeln!(out, "{pad}}}");
                Ok(())
            }
            (other, _) => Err(format!("cannot store this value into {other:?} in device memory")),
        }
    }

    /// Store the C++ value `cpp` (the private representation of a `ty`) into
    /// memory at `off`, leaf by leaf.
    fn store_cpp(&mut self, ptr: &str, off: &str, ty: Handle<naga::Type>, cpp: &str, out: &mut String, pad: &str) -> Result<(), String> {
        match &self.m.types[ty].inner {
            TypeInner::Scalar(s) => {
                let t = Ty::from_scalar(*s)?;
                let _ = writeln!(out, "{pad}{}({ptr}, {off}, {cpp});", write_helper(t)?);
                Ok(())
            }
            TypeInner::Vector { size, scalar } => {
                let t = Ty::from_scalar(*scalar)?;
                for c in 0..*size as u32 {
                    let _ = writeln!(out, "{pad}{}({ptr}, {}, {cpp}[{c}]);", write_helper(t)?, offset_plus(off, 4 * c));
                }
                Ok(())
            }
            TypeInner::Struct { members, .. } => {
                let members: Vec<(Handle<naga::Type>, u32)> = members.iter().map(|m| (m.ty, m.offset)).collect();
                for (i, (mty, moff)) in members.into_iter().enumerate() {
                    self.store_cpp(ptr, &offset_plus(off, moff), mty, &format!("{cpp}.{}", member(i)), out, pad)?;
                }
                Ok(())
            }
            TypeInner::Array { base, size: ArraySize::Constant(n), stride } => {
                let (base, n, stride) = (*base, n.get(), *stride);
                for e in 0..n {
                    let at = offset_plus(off, e * stride);
                    match &self.m.types[base].inner {
                        TypeInner::Scalar(_) | TypeInner::Struct { .. } => {
                            self.store_cpp(ptr, &at, base, &format!("{cpp}.v[{e}]"), out, pad)?;
                        }
                        TypeInner::Vector { size, scalar } => {
                            let t = Ty::from_scalar(*scalar)?;
                            let (_, _, words) = vector_shape(self.m, base).expect("a vector")?;
                            for c in 0..*size as u32 {
                                let _ = writeln!(
                                    out,
                                    "{pad}{}({ptr}, {}, {cpp}.v[{}]);",
                                    write_helper(t)?,
                                    offset_plus(&at, 4 * c),
                                    e * words + c
                                );
                            }
                        }
                        other => return Err(format!("an array of {other:?} is unsupported")),
                    }
                }
                Ok(())
            }
            other => Err(format!("cannot store {other:?} into device memory")),
        }
    }

    /// The C++ initialiser for a member (or whole struct) of type `ty` from the
    /// translated component `e`.
    fn init_of(&mut self, ty: Handle<naga::Type>, e: &Eval) -> Result<String, String> {
        match (&self.m.types[ty].inner, e) {
            (TypeInner::Scalar(s), Eval::Value(v, vt)) => coerce(v, *vt, Ty::from_scalar(*s)?),
            (TypeInner::Scalar(s), Eval::Place(Place::Lvalue(v, vt))) => coerce(v, *vt, Ty::from_scalar(*s)?),
            (TypeInner::Vector { scalar, .. }, Eval::Vector(lanes, vt)) => {
                let t = Ty::from_scalar(*scalar)?;
                let lanes = lanes.iter().map(|l| coerce(l, *vt, t)).collect::<Result<Vec<_>, _>>()?;
                Ok(format!("{{{}}}", lanes.join(", ")))
            }
            (TypeInner::Struct { .. } | TypeInner::Array { .. }, Eval::Agg(text, _)) => Ok(text.clone()),
            (other, _) => Err(format!("a struct member of type {other:?} cannot be built from this component")),
        }
    }

    /// `Compose` of struct or array type `ty` from its member values.
    pub(super) fn compose_aggregate(&mut self, ty: Handle<naga::Type>, comps: Vec<Eval>) -> Result<Eval, String> {
        let name = self.cpp_type(ty)?;
        let (parts, wrap) = match &self.m.types[ty].inner {
            TypeInner::Struct { members, .. } => {
                let tys: Vec<Handle<naga::Type>> = members.iter().map(|m| m.ty).collect();
                if tys.len() != comps.len() {
                    return Err(format!("a struct of {} members composed from {} components", tys.len(), comps.len()));
                }
                let mut parts = Vec::with_capacity(comps.len());
                for (mty, c) in tys.into_iter().zip(&comps) {
                    parts.push(self.init_of(mty, c)?);
                }
                (parts, false)
            }
            TypeInner::Array { base, .. } => {
                let base = *base;
                let mut parts = Vec::with_capacity(comps.len());
                for c in &comps {
                    match (&self.m.types[base].inner, c) {
                        // An array of vectors is flat, a vec3 padded to its
                        // four-word stride.
                        (TypeInner::Vector { .. }, Eval::Vector(lanes, vt)) => {
                            let (t, _, stride) = vector_shape(self.m, base).expect("a vector")?;
                            for k in 0..stride as usize {
                                parts.push(match lanes.get(k) {
                                    Some(l) => coerce(l, *vt, t)?,
                                    None => t.zero().to_string(),
                                });
                            }
                        }
                        _ => parts.push(self.init_of(base, c)?),
                    }
                }
                (parts, true)
            }
            other => return Err(format!("compose of {other:?}")),
        };
        let braces = if wrap { format!("{name}{{{{{}}}}}", parts.join(", ")) } else { format!("{name}{{{}}}", parts.join(", ")) };
        Ok(Eval::Agg(braces, ty))
    }

    /// The zero value of struct or array type `ty`: C++'s `{}`, which is
    /// WGSL's too.
    pub(super) fn zero_aggregate(&mut self, ty: Handle<naga::Type>) -> Result<Eval, String> {
        let name = self.cpp_type(ty)?;
        Ok(Eval::Agg(format!("{name}{{}}"), ty))
    }

    /// The element count of the array type `ty`.
    pub(super) fn array_count(&self, ty: Handle<naga::Type>) -> Result<u32, String> {
        match &self.m.types[ty].inner {
            TypeInner::Array { size: ArraySize::Constant(n), .. } => Ok(n.get()),
            other => Err(format!("expected a fixed-size array, got {other:?}")),
        }
    }

    /// Element `idx` (a C++ index expression) of the array whose wrapper is the
    /// C++ lvalue `lv`, clamped to the array like every other index.
    pub(super) fn array_element(&mut self, lv: &str, ty: Handle<naga::Type>, idx: &str) -> Result<Place, String> {
        let TypeInner::Array { base, .. } = &self.m.types[ty].inner else {
            return Err("indexing something that is not an array".into());
        };
        let base = *base;
        let at = clamped(&format!("(size_t)({idx})"), &self.array_count(ty)?.to_string());
        match &self.m.types[base].inner {
            TypeInner::Scalar(s) => Ok(Place::Lvalue(format!("{lv}.v[{at}]"), Ty::from_scalar(*s)?)),
            TypeInner::Vector { .. } => {
                let (t, n, stride) = vector_shape(self.m, base).expect("a vector")?;
                Ok(Place::VecRef { base: format!("{lv}.v[{at} * {stride}u]"), n, ty: t })
            }
            TypeInner::Struct { .. } => Ok(Place::Struct { lv: format!("{lv}.v[{at}]"), ty: base }),
            other => Err(format!("an array of {other:?} is unsupported")),
        }
    }
}
