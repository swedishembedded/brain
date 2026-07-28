// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kernel specialisation — one WGSL source, tunable constants (S3).
//!
//! Every tile size, workgroup size and unroll factor in the kernel tree is a
//! literal, so tuning a tile per device historically meant writing another
//! file — which does not scale to (ops × shapes × devices). This module
//! rewrites the *declarations* of those literals before compilation:
//!
//! * `const NAME: u32 = <literal>u;` — the tunable-constant idiom the tiled
//!   kernels already use (`BM`/`BN`/`BKG`/`WG` in `matmul_i8_dyn`);
//! * the `@workgroup_size(<literal>)` attribute, under the pseudo-parameter
//!   name `"workgroup_size"` — kept explicit rather than inferred, because
//!   every backend lays out the dispatch grid from that literal
//!   (`backend_api::workgroup_size_of` scans it), so silently coupling it to
//!   some const would hide a load-bearing rewrite.
//!
//! Substitution happens on the source *text*, so a specialised kernel is just
//! another `(name, source)` pair and flows through every backend — wgpu, the
//! CPU JIT, native Vulkan — with zero backend changes. WGSL stays the single
//! source of truth: this parameterises a kernel, it does not fork it, and the
//! unspecialised source is byte-identical to today's (provably inert until a
//! selector asks for a variant).
//!
//! [`interned`] caches specialisations process-wide by `(source ptr, params)`
//! and returns `'static` strings, so variants compose with kernel-set consts
//! and the test-device pool exactly like hand-written kernels.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Rewrite `src`, setting each named tunable to a new value. Parameters are
/// `(name, value)` where `name` is either a `const NAME: u32` declared in the
/// source or the pseudo-name `"workgroup_size"`. A name the source does not
/// declare is an ERROR — a silently ignored parameter is how a tuner ends up
/// measuring the kernel it did not ask for.
pub fn specialize(src: &str, params: &[(&str, u32)]) -> Result<String, String> {
    let mut out = src.to_string();
    for &(name, value) in params {
        out = if name == "workgroup_size" {
            rewrite_workgroup_size(&out, value)?
        } else {
            rewrite_const(&out, name, value)?
        };
    }
    Ok(out)
}

/// The registered name a specialised variant gets: `base#k=v,k=v` — unique per
/// parameter set, and self-describing in profiles and error messages.
pub fn variant_name(base: &str, params: &[(&str, u32)]) -> String {
    let mut s = String::from(base);
    for (i, (k, v)) in params.iter().enumerate() {
        s.push(if i == 0 { '#' } else { ',' });
        s.push_str(k);
        s.push('=');
        s.push_str(&v.to_string());
    }
    s
}

/// [`specialize`], interned: the same `(source, params)` always returns the
/// same `'static` `(name, source)` pair. Specialisations are few (a tuner
/// probes a handful of tile sizes), so leaking them is the working set, not a
/// leak.
pub fn interned(
    base: &'static str,
    src: &'static str,
    params: &[(&str, u32)],
) -> Result<(&'static str, &'static str), String> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, String), (&'static str, &'static str)>>> =
        OnceLock::new();
    let name = variant_name(base, params);
    let key = (src.as_ptr() as usize, name.clone());
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(&hit) = cache.get(&key) {
        return Ok(hit);
    }
    let specialised = specialize(src, params)?;
    let entry: (&'static str, &'static str) =
        (Box::leak(name.into_boxed_str()), Box::leak(specialised.into_boxed_str()));
    cache.insert(key, entry);
    Ok(entry)
}

/// Replace the literal in `const NAME: u32 = <lit>u;`.
fn rewrite_const(src: &str, name: &str, value: u32) -> Result<String, String> {
    let decl = format!("const {name}: u32 = ");
    let at = src
        .find(&decl)
        .ok_or_else(|| format!("no tunable `const {name}: u32 = ...` in kernel source"))?;
    let rest = &src[at + decl.len()..];
    let lit_len = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if lit_len == 0 {
        return Err(format!("`const {name}` is not initialised with a literal"));
    }
    let mut tail = &rest[lit_len..];
    if tail.starts_with('u') {
        tail = &tail[1..];
    }
    Ok(format!("{}{}{}u{}", &src[..at], decl, value, tail))
}

/// Replace the literal in `@workgroup_size(<lit>)` (first extent; the kernels
/// in this tree are 1-D).
fn rewrite_workgroup_size(src: &str, value: u32) -> Result<String, String> {
    let attr = "@workgroup_size(";
    let at = src.find(attr).ok_or("no @workgroup_size attribute in kernel source")?;
    let rest = &src[at + attr.len()..];
    let lit_len = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if lit_len == 0 {
        return Err("@workgroup_size extent is not a literal".into());
    }
    Ok(format!("{}{}{}{}", &src[..at], attr, value, &rest[lit_len..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: &str = "const TILE: u32 = 8u;\n@compute @workgroup_size(64)\nfn main() { let x = TILE; }";

    #[test]
    fn rewrites_consts_and_workgroup_size() {
        let s = specialize(K, &[("TILE", 16), ("workgroup_size", 128)]).unwrap();
        assert!(s.contains("const TILE: u32 = 16u;"));
        assert!(s.contains("@workgroup_size(128)"));
        // Uses of the const are untouched — only the declaration changes.
        assert!(s.contains("let x = TILE;"));
    }

    /// No parameters = byte-identical source: the mechanism is provably inert
    /// until a selector asks for a variant.
    #[test]
    fn empty_params_are_identity() {
        assert_eq!(specialize(K, &[]).unwrap(), K);
    }

    /// A parameter the source does not declare is an error, not a no-op — a
    /// tuner must never measure a kernel it did not actually specialise.
    #[test]
    fn unknown_parameter_is_an_error() {
        assert!(specialize(K, &[("NOPE", 4)]).is_err());
    }

    #[test]
    fn variant_names_are_unique_and_self_describing() {
        assert_eq!(variant_name("matmul", &[]), "matmul");
        assert_eq!(variant_name("matmul", &[("TILE", 16), ("workgroup_size", 128)]), "matmul#TILE=16,workgroup_size=128");
    }

    #[test]
    fn interned_returns_the_same_statics() {
        static SRC: &str = "const TILE: u32 = 8u;\n@compute @workgroup_size(64)\nfn main() {}";
        let a = interned("k", SRC, &[("TILE", 32)]).unwrap();
        let b = interned("k", SRC, &[("TILE", 32)]).unwrap();
        assert!(std::ptr::eq(a.0, b.0) && std::ptr::eq(a.1, b.1));
        assert_eq!(a.0, "k#TILE=32");
        // A real in-tree kernel specialises too.
        let (n, s) = interned("matmul_i8_dyn", crate::MATMUL_I8_DYN, &[("BKG", 4)]).unwrap();
        assert_eq!(n, "matmul_i8_dyn#BKG=4");
        assert!(s.contains("const BKG: u32 = 4u;"));
    }
}
