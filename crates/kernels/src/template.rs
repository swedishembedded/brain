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

use backend_api::DType;

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

/// One specialised kernel: `(variant name, specialised WGSL source)`, both
/// leaked to `'static` so a pipeline can hold them for the process lifetime.
pub type Variant = (&'static str, &'static str);

/// Interning key: the base source's address (kernel sources are `'static`
/// consts, so the pointer is a stable identity) plus the variant name.
type VariantKey = (usize, String);

/// [`specialize`], interned: the same `(source, params)` always returns the
/// same `'static` `(name, source)` pair. Specialisations are few (a tuner
/// probes a handful of tile sizes), so leaking them is the working set, not a
/// leak.
pub fn interned(
    base: &'static str,
    src: &'static str,
    params: &[(&str, u32)],
) -> Result<Variant, String> {
    static CACHE: OnceLock<Mutex<HashMap<VariantKey, Variant>>> = OnceLock::new();
    let name = variant_name(base, params);
    let key = (src.as_ptr() as usize, name.clone());
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(&hit) = cache.get(&key) {
        return Ok(hit);
    }
    let specialised = specialize(src, params)?;
    let entry: Variant =
        (Box::leak(name.into_boxed_str()), Box::leak(specialised.into_boxed_str()));
    cache.insert(key, entry);
    Ok(entry)
}

/// One storage-tier rewrite of a kernel's f32 weight binding (B4 bf16, B5
/// f16): ONE kernel source produces an f32 variant (today's default,
/// byte-identical, returned unchanged) and a packed-2-per-`u32` bf16/f16
/// storage variant that decodes inline to f32 for compute - no device
/// feature required (pure integer/bitcast WGSL, so it runs on the CPU JIT, a
/// Pascal-class GPU with no fast bf16/f16 path, and in the browser), and no
/// second physical kernel file to keep in sync.
///
/// Rewrites, on the source TEXT (comment-stripped scan, same "scan the code,
/// not the comments" contract [`backend_api::workgroup_size_of`] already
/// established):
///
/// * the binding declaration - `var<storage, read> <binding>: array<f32>;`
///   becomes `array<u32>;`;
/// * every `<binding>[IDENT]` load, where `IDENT` must already be a BARE
///   identifier, into the inline decode expression for `dt` (see
///   [`bf16_decode_expr`]/[`f16_decode_expr`]'s construction below). A
///   compound index (`<binding>[expr]`) is an ERROR, not a silent rewrite:
///   `IDENT` appears MULTIPLE TIMES in the expansion (bf16 twice - once to
///   pick the packed word, once to pick which half; f16 several more times,
///   since the magic-multiply decode below references the extracted 16-bit
///   value repeatedly), so inlining an expression with a side effect or an
///   expensive sub-expression would double-evaluate it. A kernel with a
///   compound weight index needs a `let wi = <expr>;` hoist FIRST, as its own
///   tiny, behaviour-preserving source edit - this function deliberately does
///   not attempt to hoist automatically (see `matmul.wgsl`/`matmul_gemv.wgsl`/
///   `matmul_reg3.wgsl`'s own such hoists, added for exactly this reason - the
///   SAME hoists serve both the bf16 and f16 variants, since the hoist only
///   moves the index expression, not the decode).
///
/// bf16→f32 is *exact* (no rounding, no denormal/inf/NaN special-casing): a
/// bf16 word is literally the top 16 bits of an f32, so left-shifting it into
/// the high half of a u32 and bitcasting reproduces the value bit-for-bit.
/// f16→f32 is NOT exact bit reinterpretation - f16's 5-bit exponent (vs f32's
/// 8-bit) needs real re-biasing, done via the magic-multiply technique (see
/// [`f16_decode_expr`]'s doc comment) rather than a native `f16` WGSL type or
/// `unpack2x16float`/`extractBits` builtins, neither of which this repo's CPU
/// JIT (`crates/wgsl-cpu`) has a lowering for.
///
/// [`DType::BF16`] and [`DType::F16`] are implemented; every other tier is an
/// `Err` (loudly, not a silent no-op or a panic) - `F32`/`I8`/`Q4` have no
/// storage-tier decode concept at all (`F32` needs no rewrite, `I8`/`Q4`
/// already have their own dedicated packed kernels, `matmul_i8*`/
/// `matmul_q4*`, predating this templater - see `model::ops`).
///
/// Variant naming reuses this module's `#k=v` convention with the binding
/// name as the key: `dtype_variant("matmul", MATMUL, "w", DType::BF16)`
/// yields `"matmul#w=bf16"` (`"matmul#w=f16"` for `DType::F16`), so it flows
/// through the existing kernel-registry/selector/autotuner machinery
/// unchanged - a specialised bf16/f16 kernel is just another `(name, source)`
/// pair, exactly like [`interned`]'s tile-size variants.
pub fn dtype_variant(
    name: &str,
    src: &'static str,
    binding: &str,
    dt: DType,
) -> Result<Variant, String> {
    let tag = dtype_tag(dt)?;
    let vname = format!("{name}#{binding}={tag}");

    static CACHE: OnceLock<Mutex<HashMap<(usize, String), Variant>>> = OnceLock::new();
    let key = (src.as_ptr() as usize, vname.clone());
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(&hit) = cache.get(&key) {
        return Ok(hit);
    }

    let with_decl = rewrite_packed_declaration(src, binding)?;
    let rewritten = rewrite_packed_loads(&with_decl, binding, dt)?;

    let entry: Variant = (Box::leak(vname.into_boxed_str()), Box::leak(rewritten.into_boxed_str()));
    cache.insert(key, entry);
    Ok(entry)
}

/// The `#k=v` tag [`dtype_variant`] uses for `dt` - `BF16`/`F16` are
/// implemented; every other tier is a loud `Err` (see [`dtype_variant`]'s doc
/// comment for why).
fn dtype_tag(dt: DType) -> Result<&'static str, String> {
    match dt {
        DType::BF16 => Ok("bf16"),
        DType::F16 => Ok("f16"),
        other => Err(format!(
            "dtype_variant: no storage-tier decode expression for {other:?} yet -- only DType::BF16 \
             (B4) and DType::F16 (B5) are implemented; DType::F32/I8/Q4 have no storage-tier rewrite \
             concept (F32 needs none, I8/Q4 already have their own dedicated packed kernels)"
        )),
    }
}

/// Blank every `//`-to-end-of-line comment span with ASCII spaces, preserving
/// every other byte and every line's length/position exactly - so a byte
/// offset found in the blanked copy indexes identically into the original
/// source. Same "scan the code, not the comments" contract
/// [`backend_api::workgroup_size_of`] already established, generalised from a
/// per-line prefix scan to a whole-source byte scan so a match can be found
/// (and an error reported) anywhere, not just at a line's start.
fn blank_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                out[i] = b' ';
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    // WGSL kernel sources are ASCII; the loop above only ever replaces bytes
    // that were part of an ASCII `//`-comment with an ASCII space, so this
    // can never land mid-codepoint.
    String::from_utf8(out).expect("kernel source stays ASCII after comment-blanking")
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Every byte offset in `hay` where `needle` occurs as a whole word - i.e.
/// not immediately preceded by an identifier character (so binding `"w"`'s
/// `"w["` needle does not match inside `"raw["`).
fn find_word_all(hay: &str, needle: &str) -> Vec<usize> {
    let bytes = hay.as_bytes();
    let mut out = Vec::new();
    let mut start = 0;
    while let Some(rel) = hay.get(start..).and_then(|s| s.find(needle)) {
        let pos = start + rel;
        let before_ok = pos == 0 || !is_ident_byte(bytes[pos - 1]);
        if before_ok {
            out.push(pos);
        }
        start = pos + 1;
    }
    out
}

/// Rewrite `var<storage, read> <binding>: array<f32>;` (arbitrary whitespace)
/// to `array<u32>;`, leaving every other binding's declaration untouched.
/// Shared by both the bf16 and f16 storage tiers - both pack 2 elements per
/// `u32`, so the declaration rewrite is identical; only [`rewrite_packed_loads`]'s
/// decode expression differs per `dt`.
fn rewrite_packed_declaration(src: &str, binding: &str) -> Result<String, String> {
    let code = blank_comments(src);
    let marker = "var<storage, read>";
    let want = format!("{binding}:");
    let mut search_from = 0;
    while let Some(rel) = code.get(search_from..).and_then(|s| s.find(marker)) {
        let marker_pos = search_from + rel;
        let after = marker_pos + marker.len();
        // The declaration statement's window: up to its terminating `;` (or
        // end of source, defensively).
        let stmt_end = code.get(after..).and_then(|s| s.find(';')).map(|p| after + p).unwrap_or(code.len());
        let stmt = &code[after..stmt_end];
        if !find_word_all(stmt, &want).is_empty() {
            let Some(fpos) = stmt.find("array<f32>") else {
                return Err(format!(
                    "dtype_variant: `var<storage, read> {binding}: ...` declaration has no \
                     `array<f32>` element type (got `{}`) -- only a plain f32 storage array can be \
                     rewritten to a packed bf16/f16 one",
                    stmt.trim()
                ));
            };
            let at = after + fpos;
            let mut out = String::with_capacity(src.len());
            out.push_str(&src[..at]);
            out.push_str("array<u32>");
            out.push_str(&src[at + "array<f32>".len()..]);
            return Ok(out);
        }
        search_from = after;
    }
    Err(format!(
        "dtype_variant: no `var<storage, read> {binding}: array<f32>;` declaration found in kernel source"
    ))
}

/// Rewrite every `<binding>[IDENT]` load into the inline decode expression
/// for `dt` ([`bf16_decode_expr`]/[`f16_decode_expr`]). `IDENT` must be a bare
/// identifier (see [`dtype_variant`]'s doc comment for why) - any other index
/// expression is an `Err`.
fn rewrite_packed_loads(src: &str, binding: &str, dt: DType) -> Result<String, String> {
    let code = blank_comments(src);
    let bytes = code.as_bytes();
    let needle = format!("{binding}[");

    // Collect (start_of_`binding[`, index_of_closing_`]`, bare_ident_text) for
    // every occurrence, validating each before rewriting any of them - a
    // caller sees every offending compound index this function would refuse,
    // not just the first.
    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    for pos in find_word_all(&code, &needle) {
        let open = pos + binding.len();
        let mut depth = 1i32;
        let mut i = open + 1;
        let close = loop {
            if i >= bytes.len() {
                return Err(format!(
                    "dtype_variant: unbalanced `[` for `{binding}[` in kernel source (no matching `]`)"
                ));
            }
            match bytes[i] {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        break i;
                    }
                }
                _ => {}
            }
            i += 1;
        };
        let idx_text = code[open + 1..close].trim();
        if !is_bare_ident(idx_text) {
            return Err(format!(
                "dtype_variant: `{binding}[{idx_text}]` is not a bare identifier -- the decode \
                 expansion reads the index MULTIPLE TIMES (bf16: twice, to pick the packed word and \
                 which half; f16: several more, the magic-multiply decode references the extracted \
                 16-bit value repeatedly), so a compound index would be double-evaluated. Hoist it to \
                 `let wi = {idx_text};` in the kernel source first, then template `{binding}[wi]`."
            ));
        }
        spans.push((pos, close, idx_text.to_string()));
    }

    let mut out = String::with_capacity(src.len() + spans.len() * 96);
    let mut cursor = 0usize;
    for (pos, close, ident) in &spans {
        out.push_str(&src[cursor..*pos]);
        out.push_str(&decode_expr(binding, ident, dt));
        cursor = close + 1;
    }
    out.push_str(&src[cursor..]);
    Ok(out)
}

/// `<binding>[<ident>]`'s inline decode expression for `dt` - [`dtype_tag`]
/// already gated `dt` to `BF16`/`F16` before this is ever called (every other
/// tier errors out of [`dtype_variant`] before reaching the rewrite passes).
fn decode_expr(binding: &str, ident: &str, dt: DType) -> String {
    match dt {
        DType::BF16 => bf16_decode_expr(binding, ident),
        DType::F16 => f16_decode_expr(binding, ident),
        other => unreachable!(
            "decode_expr: dtype_tag already rejected {other:?} before rewrite_packed_loads was called"
        ),
    }
}

/// `true` iff `s` is a single WGSL identifier (`[A-Za-z_][A-Za-z0-9_]*`) with
/// no surrounding operators, indexing, or whitespace beyond what was already
/// trimmed.
fn is_bare_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The inline bf16→f32 decode for `<binding>[<ident>]`: the top 16 bits of an
/// f32, packed two-per-`u32` (low half = even index, high half = odd index -
/// the SAME convention `checkpoint::safetensors::bf16_to_f32`'s `(h as u32) <<
/// 16` decodes and `model::half::pack_bf16` packs). Exact - no rounding, no
/// denormal/inf/NaN special-casing needed (see [`dtype_variant`]'s doc
/// comment).
fn bf16_decode_expr(binding: &str, ident: &str) -> String {
    format!(
        "bitcast<f32>((({binding}[{ident} >> 1u] >> (16u * ({ident} & 1u))) & 0xFFFFu) << 16u)"
    )
}

/// The inline f16→f32 decode for `<binding>[<ident>]` (B5): the packed-2-
/// per-`u32` half extraction is identical to [`bf16_decode_expr`]'s (low half
/// = even index, high half = odd index), but unlike bf16 the resulting
/// 16-bit pattern `h` is NOT already an f32's top 16 bits - f16's 5-bit
/// exponent needs real re-biasing against f32's 8-bit exponent, done here with
/// a THREE-WAY split on `h`'s exponent field (`(h >> 10) & 0x1F`): normal
/// (`[1,30]`), subnormal (`0`), and `inf`/`NaN` (`31`).
///
/// **Normal** (`exponent in [1,30]`) uses the classic "magic multiply":
/// `(h & 0x7FFF) << 13` places f16's 10-bit mantissa into the TOP 10 bits of
/// f32's 23-bit mantissa field and f16's 5-bit exponent into the LOW 5 bits
/// of f32's 8-bit exponent field (top 3 bits zero, and for a NORMAL `h` this
/// shifted pattern's own exponent field is always `>= 1`, i.e. itself a
/// normal f32 - see the FTZ paragraph below for why that matters).
/// Multiplying by the constant `2^112` (`0x77800000`) re-biases it: the
/// shifted pattern's float value has exponent field `e_h`, i.e. actual power
/// `e_h - 127`; multiplying by `2^112` gives `e_h - 15`, f16's own bias -
/// exact, since the multiply only changes the exponent field, not the
/// mantissa bits.
///
/// **Subnormal** (`exponent == 0`, true value `mantissa * 2^-24`) does NOT
/// reuse the multiply above - a real, empirically-found bug this function's
/// first draft had: `(h & 0x7FFF) << 13` for SUBNORMAL `h` (mantissa in
/// `[0,1023]`, exponent bits already zero) always lands strictly below
/// `0x0080_0000` (f32's own normal/subnormal boundary), i.e. the shifted
/// bit pattern is ITSELF A SUBNORMAL f32. Some real GPU hardware flushes
/// subnormal float OPERANDS to zero in compute shaders (confirmed on this
/// repo's own Intel Arc iGPU, via `crates/model/tests/f16_roundtrip.rs`'s
/// dedicated subnormal-weight-row edge case: the naive multiply decoded a
/// real subnormal f16 weight to exactly `0.0` on real wgpu/Vulkan hardware,
/// while the CPU JIT - Rust float math, no FTZ - got it right, so this was
/// caught by the dual-backend test disagreeing, not by the host-side-only
/// bit-math check, which cannot see hardware FTZ behaviour at all). Fixed
/// with the standard FTZ-safe "magic bias" construction instead: build a
/// NORMAL float `a = bitcast<f32>(0x3880_0000 | (mantissa << 13))` (exponent
/// field fixed at `113`, i.e. value `(1 + mantissa/1024) * 2^-14` - always
/// normal, since the exponent field is nonzero by construction) and subtract
/// the matching normal constant `b = bitcast<f32>(0x3880_0000) == 2^-14`:
/// `a - b == mantissa/1024 * 2^-14 == mantissa * 2^-24`, the exact target
/// value, computed via a subtraction of two NORMAL floats whose exact
/// mathematical result (`<= 1023 * 2^-24 ≈ 6.1e-5`) is itself comfortably
/// within f32's normal range (f32's own subnormal threshold is `2^-126`,
/// many orders of magnitude smaller) - so neither operand nor the result is
/// ever a subnormal f32, and this branch is FTZ-safe on every backend.
/// `mantissa == 0` (true zero) falls out of the same formula for free:
/// `a == b`, so `a - b == 0.0` exactly.
///
/// **Inf/NaN** (`exponent == 31`) is its own branch, selected via the OUTER
/// `select(...)`: `0x7F800000 | ((h & 0x3FF) << 13)` builds f32's `inf`
/// pattern with the mantissa's top 10 bits set from `h`'s mantissa - a zero
/// mantissa (`inf`) stays `inf`, a nonzero one (`NaN`) stays some `NaN`
/// (exact payload not preserved, which is fine - IEEE-754 only requires SOME
/// NaN survive).
///
/// The sign bit (`h & 0x8000`) is reapplied exactly once, after the
/// value/exponent path, regardless of which of the three branches ran.
/// Verified bit-for-bit against an independent (non-bit-trick, direct
/// sign/exponent/mantissa reconstruction) reference across all 65536 possible
/// `h` patterns - see this module's
/// `f16_decode_matches_an_independent_reference_for_every_possible_bit_pattern`
/// test - though that host-side check alone did NOT catch the FTZ bug above
/// (Rust float arithmetic never flushes subnormals); the real gate for that
/// was the dual-backend roundtrip test actually running on real hardware.
///
/// Pure integer/bitcast WGSL (`select`/`bitcast`, both core WGSL) - no
/// `enable f16;`, no native half type, no `unpack2x16float`/`extractBits`
/// builtins (this repo's CPU JIT has no lowering for either, confirmed in
/// B4's and earlier phases' notes).
fn f16_decode_expr(binding: &str, ident: &str) -> String {
    let h = format!("(({binding}[{ident} >> 1u] >> (16u * ({ident} & 1u))) & 0xFFFFu)");
    // FTZ-safe: `a`/`b` are both normal f32s (exponent field fixed at 113),
    // and their exact difference stays in f32's normal range - see this
    // function's doc comment for the full derivation.
    let subnormal = format!(
        "(bitcast<f32>(0x38800000u | (({h} & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u))"
    );
    // The classic magic-multiply, safe here because a NORMAL h's shifted
    // pattern is never itself subnormal (see doc comment).
    let normal = format!("(bitcast<f32>(({h} & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u))");
    let inf_or_nan = format!("bitcast<f32>(0x7F800000u | (({h} & 0x3FFu) << 13u))");
    format!(
        "bitcast<f32>(bitcast<u32>(select(select({subnormal}, {normal}, (({h} >> 10u) & 0x1Fu) != 0u), \
         {inf_or_nan}, (({h} >> 10u) & 0x1Fu) == 31u)) | (({h} & 0x8000u) << 16u))"
    )
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

    /// A tiny kernel shaped like `matmul.wgsl` after its own bare-identifier
    /// hoist - `x` untouched, `w` storage-tier-templatable.
    const MM: &str = "\
struct Params { m: u32, k: u32, n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main() {
    let wi = 3u;
    out[0] = x[0] * w[wi];
}
";

    #[test]
    fn dtype_variant_rewrites_declaration_and_load() {
        let (name, src) = dtype_variant("mm", MM, "w", DType::BF16).unwrap();
        assert_eq!(name, "mm#w=bf16");
        // The `w` binding narrows to a packed u32 array...
        assert!(src.contains("var<storage, read>       w:   array<u32>;"), "{src}");
        // ...`x` is completely untouched (still f32, no decode expression).
        assert!(src.contains("var<storage, read>       x:   array<f32>;"), "{src}");
        assert!(src.contains("x[0]"));
        // The load site expands to the inline bf16 decode, reading `wi` twice.
        assert!(
            src.contains("bitcast<f32>(((w[wi >> 1u] >> (16u * (wi & 1u))) & 0xFFFFu) << 16u)"),
            "{src}"
        );
        // The original compact `w[wi]` load is gone.
        assert!(!src.contains("* w[wi]"), "{src}");
    }

    /// [`interned`]'s "same source, same params -> same statics" contract,
    /// proven directly against [`dtype_variant`] rather than reimplemented.
    #[test]
    fn dtype_variant_is_stable_across_calls() {
        let a = dtype_variant("mm", MM, "w", DType::BF16).unwrap();
        let b = dtype_variant("mm", MM, "w", DType::BF16).unwrap();
        assert!(std::ptr::eq(a.0, b.0) && std::ptr::eq(a.1, b.1));
    }

    /// A real in-tree kernel, after its own hoist, templates cleanly too -
    /// the same proof `interned_returns_the_same_statics` does for `specialize`.
    #[test]
    fn dtype_variant_handles_the_real_matmul_kernel() {
        let (name, src) = dtype_variant("matmul", crate::MATMUL, "w", DType::BF16).unwrap();
        assert_eq!(name, "matmul#w=bf16");
        assert!(src.contains("array<u32>"));
        assert!(src.contains("bitcast<f32>(((w[wi >> 1u]"), "{src}");
    }

    /// A compound weight index (no bare-identifier hoist) is a loud `Err`,
    /// never a silent double-evaluating rewrite - the exact failure mode this
    /// function exists to prevent (see its own doc comment).
    #[test]
    fn dtype_variant_rejects_a_compound_index() {
        const UNHOISTED: &str = "\
@group(0) @binding(2) var<storage, read> w: array<f32>;
fn main() { let v = w[base + i]; }
";
        let err = dtype_variant("u", UNHOISTED, "w", DType::BF16).unwrap_err();
        assert!(err.contains("bare identifier"), "{err}");
        assert!(err.contains("base + i"), "{err}");
    }

    /// `DType::BF16`/`DType::F16` are implemented (B4/B5) - every other tier
    /// is a loud `Err`, not a silent pass-through or a panic (`F32`/`I8`/`Q4`
    /// have no storage-tier decode concept at all).
    #[test]
    fn dtype_variant_rejects_unimplemented_tiers() {
        for dt in [DType::F32, DType::I8, DType::Q4] {
            let err = dtype_variant("mm", MM, "w", dt).unwrap_err();
            assert!(err.contains("BF16") && err.contains("F16"), "{dt:?}: {err}");
        }
    }

    /// A binding whose declaration is missing entirely fails loudly instead
    /// of silently producing an unrewritten f32 kernel under a bf16 name.
    #[test]
    fn dtype_variant_errors_when_the_declaration_is_missing() {
        const NO_W: &str = "@group(0) @binding(1) var<storage, read> x: array<f32>;\nfn main() {}";
        let err = dtype_variant("u", NO_W, "w", DType::BF16).unwrap_err();
        assert!(err.contains("no `var<storage, read> w"), "{err}");
    }

    // --- B5: f16 storage tier ---------------------------------------------

    #[test]
    fn dtype_variant_rewrites_declaration_and_load_for_f16() {
        let (name, src) = dtype_variant("mm", MM, "w", DType::F16).unwrap();
        assert_eq!(name, "mm#w=f16");
        // The `w` binding narrows to a packed u32 array (same packing as
        // bf16 - 2 elements per word)...
        assert!(src.contains("var<storage, read>       w:   array<u32>;"), "{src}");
        // ...`x` is completely untouched (still f32, no decode expression).
        assert!(src.contains("var<storage, read>       x:   array<f32>;"), "{src}");
        assert!(src.contains("x[0]"));
        // The load site expands to the three-way f16 decode (normal magic-
        // multiply, FTZ-safe subnormal bias-subtract, inf/nan), with the
        // packed-half extraction `(w[wi >> 1u] >> (16u * (wi & 1u))) & 0xFFFFu`
        // referenced multiple times (once per use of `h` in the formula).
        let extract = "(w[wi >> 1u] >> (16u * (wi & 1u))) & 0xFFFFu";
        assert!(src.contains(extract), "{src}");
        assert!(src.matches(extract).count() >= 6, "{src}");
        assert!(src.contains("0x77800000u"), "{src}"); // normal branch: magic constant, 2^112
        assert!(src.contains("0x38800000u"), "{src}"); // subnormal branch: FTZ-safe bias, 2^-14
        assert!(src.contains("0x7F800000u"), "{src}"); // inf/nan branch: f32 inf exponent field
        assert!(src.matches("select(").count() >= 2, "{src}"); // nested three-way select
        // The original compact `w[wi]` load is gone.
        assert!(!src.contains("* w[wi]"), "{src}");
    }

    #[test]
    fn dtype_variant_is_stable_across_calls_for_f16() {
        let a = dtype_variant("mm", MM, "w", DType::F16).unwrap();
        let b = dtype_variant("mm", MM, "w", DType::F16).unwrap();
        assert!(std::ptr::eq(a.0, b.0) && std::ptr::eq(a.1, b.1));
    }

    /// bf16 and f16 variants of the SAME source are independent cache
    /// entries under distinct names, not a collision.
    #[test]
    fn dtype_variant_bf16_and_f16_are_distinct_variants() {
        let (bf16_name, bf16_src) = dtype_variant("mm", MM, "w", DType::BF16).unwrap();
        let (f16_name, f16_src) = dtype_variant("mm", MM, "w", DType::F16).unwrap();
        assert_ne!(bf16_name, f16_name);
        assert_ne!(bf16_src, f16_src);
    }

    /// A real in-tree kernel, after its own hoist, templates cleanly for f16
    /// too - the same proof `dtype_variant_handles_the_real_matmul_kernel`
    /// gives for bf16.
    #[test]
    fn dtype_variant_handles_the_real_matmul_kernel_for_f16() {
        let (name, src) = dtype_variant("matmul", crate::MATMUL, "w", DType::F16).unwrap();
        assert_eq!(name, "matmul#w=f16");
        assert!(src.contains("array<u32>"));
        assert!(src.contains("(w[wi >> 1u] >> (16u * (wi & 1u))) & 0xFFFFu"), "{src}");
    }

    /// A compound weight index is a loud `Err` for f16 too, not just bf16.
    #[test]
    fn dtype_variant_rejects_a_compound_index_for_f16() {
        const UNHOISTED: &str = "\
@group(0) @binding(2) var<storage, read> w: array<f32>;
fn main() { let v = w[base + i]; }
";
        let err = dtype_variant("u", UNHOISTED, "w", DType::F16).unwrap_err();
        assert!(err.contains("bare identifier"), "{err}");
        assert!(err.contains("base + i"), "{err}");
    }

    /// Host-side mirror of the WGSL f16 decode expression [`f16_decode_expr`]
    /// generates - pure u32/f32 bitcast arithmetic, byte-for-byte the same
    /// three-way branch the templated shader performs (FTZ-safe magic-bias
    /// subtract for subnormals, magic-multiply for normals, a separate branch
    /// for inf/NaN, sign reapplied at the end). Kept here, by hand, in sync
    /// with the WGSL text this module emits - this is the "prototype in
    /// plain Rust, then port to WGSL" verification step, not a duplicate
    /// implementation the WGSL calls. Note this Rust mirror does NOT itself
    /// prove FTZ-safety (host Rust float arithmetic never flushes
    /// subnormals) - that property was verified by hand (see
    /// [`f16_decode_expr`]'s doc comment: every intermediate's exponent field
    /// is provably nonzero by construction) and confirmed empirically by
    /// `crates/model/tests/f16_roundtrip.rs` actually running on real GPU
    /// hardware.
    fn f16_decode_bits_wgsl_mirror(h: u16) -> u32 {
        let h = h as u32;
        let exp = (h >> 10) & 0x1F;
        let subnormal = f32::from_bits(0x3880_0000 | ((h & 0x3FF) << 13)) - f32::from_bits(0x3880_0000);
        let normal = f32::from_bits((h & 0x7FFF) << 13) * f32::from_bits(0x7780_0000);
        let inf_or_nan = f32::from_bits(0x7F80_0000 | ((h & 0x3FF) << 13));
        let unsigned = if exp == 31 {
            inf_or_nan
        } else if exp != 0 {
            normal
        } else {
            subnormal
        };
        unsigned.to_bits() | ((h & 0x8000) << 16)
    }

    /// An INDEPENDENT reference for f16→f32, deliberately NOT sharing the
    /// magic-multiply/magic-bias bit tricks: direct sign/exponent/mantissa
    /// field reconstruction, scaled by an exact power-of-two float multiply
    /// (exact because the integer mantissa values here are always small
    /// enough to represent exactly in f32, and multiplying by a power of two
    /// changes only the exponent field, never rounds). This is the "tiny
    /// host-side Rust equivalent" the phase brief asked for, checked BEFORE
    /// trusting [`f16_decode_bits_wgsl_mirror`] (and therefore
    /// [`f16_decode_expr`]'s WGSL text) blindly.
    fn f16_decode_bits_independent_reference(h: u16) -> u32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1F) as i32;
        let mant = (h & 0x3FF) as u32;
        let mag: f32 = if exp == 0 {
            if mant == 0 {
                0.0
            } else {
                // Subnormal: mant/1024 * 2^-14 == mant * 2^-24, exact (mant
                // fits in 10 bits, 2^-24 is an exact f32 power-of-two scale).
                (mant as f32) * f32::from_bits(((-24i32 + 127) as u32) << 23)
            }
        } else if exp == 31 {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            // Normal: (1 + mant/1024) * 2^(exp-15) == (1024+mant) * 2^(exp-25),
            // exact (1024+mant fits in 11 bits, 2^(exp-25) is an exact scale
            // for every exp in [1,30]).
            ((1024 + mant) as f32) * f32::from_bits(((exp - 25 + 127) as u32) << 23)
        };
        mag.to_bits() | (sign << 31)
    }

    /// The TDD gate this phase's brief specifically asked for: verify the
    /// magic-multiply decode against an independent reference for ALL 65536
    /// possible f16 bit patterns (f16's entire representable space is small
    /// enough to check exhaustively, not just at a handful of sample points)
    /// - normals, subnormals (including the smallest, `0x0001`), zero/negative
    /// zero, the largest normal, overflow/underflow boundaries, +-inf, and
    /// every NaN encoding. NaN pairs are accepted as equal iff BOTH sides are
    /// NaN (exact payload is not required to match - see [`f16_decode_expr`]'s
    /// doc comment for why); every other pattern must match bit-for-bit.
    #[test]
    fn f16_decode_matches_an_independent_reference_for_every_possible_bit_pattern() {
        let mut checked_subnormal = false;
        let mut checked_overflow = false;
        let mut checked_underflow = false;
        for h in 0u32..=0xFFFFu32 {
            let h = h as u16;
            let got = f32::from_bits(f16_decode_bits_wgsl_mirror(h));
            let want = f32::from_bits(f16_decode_bits_independent_reference(h));
            if want.is_nan() {
                assert!(got.is_nan(), "h=0x{h:04x}: want NaN, got {got}");
                continue;
            }
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "h=0x{h:04x}: magic-multiply {got} (0x{:08x}) != independent reference {want} \
                 (0x{:08x})",
                got.to_bits(),
                want.to_bits()
            );
            let exp = (h >> 10) & 0x1F;
            if exp == 0 && (h & 0x3FF) != 0 {
                checked_subnormal = true;
            }
        }
        checked_overflow |= f32::from_bits(f16_decode_bits_wgsl_mirror(0x7C00)).is_infinite();
        checked_underflow |= f32::from_bits(f16_decode_bits_wgsl_mirror(0x0000)) == 0.0;
        assert!(checked_subnormal, "the sweep must have covered at least one subnormal h");
        assert!(checked_overflow, "0x7C00 (f16 +inf) must decode to +inf");
        assert!(checked_underflow, "0x0000 (f16 +0) must decode to +0.0");
    }

    /// Spot-checks the phase brief's own edge-case table by name, pinned to
    /// literal f16 bit patterns (not `model::half::f32_to_f16` - this crate
    /// does not depend on `brain-model` - so each pattern is derived by hand
    /// from the IEEE-754 half-precision layout and double-checked against the
    /// independent reference above).
    #[test]
    fn f16_decode_matches_known_values() {
        let dec = |h: u16| f32::from_bits(f16_decode_bits_wgsl_mirror(h));
        assert_eq!(dec(0x0000), 0.0);
        assert!(dec(0x0000).is_sign_positive());
        assert_eq!(dec(0x8000), -0.0);
        assert!(dec(0x8000).is_sign_negative());
        assert_eq!(dec(0x3C00), 1.0);
        assert_eq!(dec(0xBC00), -1.0);
        assert_eq!(dec(0x0400), 2f32.powi(-14)); // smallest normal
        assert_eq!(dec(0x7BFF), 65504.0); // largest normal
        assert_eq!(dec(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(dec(0x0002), 2f32.powi(-23)); // another subnormal
        assert_eq!(dec(0x7C00), f32::INFINITY);
        assert_eq!(dec(0xFC00), f32::NEG_INFINITY);
        assert!(dec(0x7E00).is_nan());
        assert!(dec(0xFE00).is_nan());
    }
}
