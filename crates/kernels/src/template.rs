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

/// The WRITE-direction sibling of [`dtype_variant`] (B9): rewrites a storage
/// binding declared `array<f32>` into a packed-2-per-`u32` array whose
/// `<binding>[IDENT] = <expr>;` ASSIGNMENT statements become a genuine
/// read-modify-write PACK of `<expr>` into the half of the shared `u32` word
/// `IDENT` selects - leaving the OTHER half's bits untouched.
///
/// **Why this exists as a separate function, not a mode flag on
/// [`dtype_variant`].** Every kernel B4/B5/B8 templatized (`matmul`/`embed`/
/// `moe_linear_gated`/the plain convs) only ever READS a packed weight - the
/// host uploads it once (`Weight::upload`), the device never writes it back.
/// A paged-KV-cache pool is different: `paged_kv_append*` WRITES a freshly
/// computed K/V value into the pool on every decode step, for the life of a
/// sequence - there is no host-side pack step to reuse. Packing on WRITE needs
/// the opposite transform from decoding on READ (`bf16_pack_expr` rounds an
/// f32 DOWN to 16 bits; [`bf16_decode_expr`] widens 16 bits back to f32
/// exactly), and - the genuinely new part - it needs to PRESERVE the sibling
/// half's existing bits, since two logically unrelated cache slots can share
/// one packed word (see this function's own `rewrite_packed_stores` for
/// exactly when that happens and why the rewrite is safe even then).
///
/// Shares [`rewrite_packed_declaration`] with [`dtype_variant`] (the
/// `array<f32>` → `array<u32>` rewrite is identical either direction) but
/// calls [`rewrite_packed_stores`], not [`rewrite_packed_loads`], for the body
/// rewrite. Naming convention is IDENTICAL to [`dtype_variant`]'s
/// (`"{name}#{binding}={tag}"`) - a store-direction variant is still just
/// another `(name, source)` pair to the kernel registry/selector, so it needs
/// no separate namespace.
///
/// Only [`DType::BF16`] is implemented (matching this phase's own scope,
/// "bf16 KV-cache" - B9); `F16`'s real re-biasing PACK direction (as opposed
/// to its already-implemented DECODE direction) is a follow-up, not attempted
/// here. `F32`/`I8`/`Q4` have no storage-tier rewrite concept at all, same as
/// [`dtype_variant`].
pub fn dtype_variant_store(
    name: &str,
    src: &'static str,
    binding: &str,
    dt: DType,
) -> Result<Variant, String> {
    if dt != DType::BF16 {
        return Err(format!(
            "dtype_variant_store: no storage-tier PACK expression for {dt:?} yet -- only DType::BF16 \
             (B9) is implemented for the write direction; DType::F16's real re-biased pack is a \
             follow-up, and F32/I8/Q4 have no storage-tier rewrite concept at all"
        ));
    }
    let vname = format!("{name}#{binding}=bf16");

    static CACHE: OnceLock<Mutex<HashMap<(usize, String), Variant>>> = OnceLock::new();
    let key = (src.as_ptr() as usize, vname.clone());
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(&hit) = cache.get(&key) {
        return Ok(hit);
    }

    let with_decl = rewrite_packed_declaration(src, binding)?;
    let rewritten = rewrite_packed_stores(&with_decl, binding, dt)?;

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

/// Rewrite `var<storage, read> <binding>: array<f32>;` OR `var<storage,
/// read_write> <binding>: array<f32>;` (arbitrary whitespace) to `array<u32>;`,
/// leaving every other binding's declaration untouched. Shared by the bf16/f16
/// READ tiers ([`rewrite_packed_loads`]) and the bf16 WRITE tier (B9,
/// [`rewrite_packed_stores`]) - all three pack 2 elements per `u32`, so the
/// declaration rewrite is identical; only the load/store expansion differs.
/// `read_write` (not just `read`) matters for B9: a KV-cache pool binding a
/// PACK variant writes is declared `read_write` in its f32 source too (the
/// read-modify-write the pack needs reads the very same storage binding it
/// writes), so the marker must match both access modes, not just `read`.
fn rewrite_packed_declaration(src: &str, binding: &str) -> Result<String, String> {
    let code = blank_comments(src);
    let marker = "var<storage, read"; // matches both `read>` and `read_write>`
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

/// Rewrite every `<binding>[IDENT] = <expr>;` ASSIGNMENT statement (B9's write
/// direction) into a read-modify-write PACK of `<expr>` into the half of the
/// shared `u32` word `IDENT` selects, preserving the sibling half's bits.
/// `IDENT` must be a bare identifier, same hard precondition as
/// [`rewrite_packed_loads`] (the generated block references it twice - once
/// for the word index, once for the half selector).
///
/// **A `<binding>[IDENT]` occurrence that is NOT immediately followed by `=`
/// (skipping whitespace) is left untouched** - this function only rewrites
/// STORES; a kernel that both reads and writes the same binding (none in this
/// tree today) would need [`rewrite_packed_loads`] run first for its reads.
/// `==` is explicitly excluded from counting as an assignment (a comparison,
/// not a store).
///
/// **Why the read-modify-write is safe even when two DIFFERENT cache slots
/// share one packed word.** The pool's flat layout is `slot*kv_stride + c`;
/// packing 2-per-`u32` over that flat index means a word's two halves belong
/// to the SAME slot whenever `kv_stride` is even (every real head_dim in this
/// tree is even, so this is the common case and no cross-slot sharing ever
/// happens) - but when `kv_stride` is ODD, consecutive slots' packed words DO
/// straddle a slot boundary, so an append to slot `s+1` can share a word with
/// slot `s`'s already-committed last element. The generated code below reads
/// the word FRESH (`{binding}[_pw]`) immediately before writing it back with
/// only the target half's bits changed (`& ~mask` clears exactly the target
/// half, `| (bits << shift)` sets only that half) - so a call that runs AFTER
/// a previous append to the sibling half has fully completed (i.e. the two
/// dispatches are sequenced, not concurrent - the normal case for KV-cache
/// append, one token appended at a time) always preserves that committed
/// value exactly. Two THREADS writing the SAME word inside one dispatch
/// (e.g. two different sequences in one batched call landing on adjacent
/// slots of an odd-`kv_stride` pool) would race, same as any non-atomic
/// read-modify-write across concurrent GPU threads - not a defect this
/// rewrite introduces, but a real caveat for an odd-`kv_stride` config, worth
/// stating rather than leaving implicit.
fn rewrite_packed_stores(src: &str, binding: &str, dt: DType) -> Result<String, String> {
    let code = blank_comments(src);
    let bytes = code.as_bytes();
    let needle = format!("{binding}[");

    // (pos of `binding[`, bare ident text, byte offset of the value expr's
    // start, byte offset of the terminating `;`) for every ASSIGNMENT
    // occurrence, validated before any rewrite - same "report every offender,
    // not just the first" discipline as `rewrite_packed_loads`.
    let mut spans: Vec<(usize, String, usize, usize)> = Vec::new();
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
        let idx_text = code[open + 1..close].trim().to_string();

        // Is this occurrence an assignment target? Skip whitespace after `]`;
        // require a bare `=` (not `==`).
        let mut j = close + 1;
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'=' || bytes.get(j + 1) == Some(&b'=') {
            continue; // a READ occurrence -- rewrite_packed_loads's concern, not this function's
        }
        if !is_bare_ident(&idx_text) {
            return Err(format!(
                "dtype_variant: `{binding}[{idx_text}] = ...` store index is not a bare identifier -- \
                 the pack expansion reads the index twice (once for the packed word, once for the half \
                 selector), so a compound index would be double-evaluated. Hoist it to `let wi = \
                 {idx_text};` in the kernel source first, then write `{binding}[wi] = ...;`."
            ));
        }
        let value_start = j + 1;
        let semi = code.get(value_start..).and_then(|s| s.find(';')).map(|p| value_start + p).ok_or_else(
            || format!("dtype_variant: unterminated assignment to `{binding}[{idx_text}]` (no `;`)"),
        )?;
        spans.push((pos, idx_text, value_start, semi));
    }

    if spans.is_empty() {
        return Err(format!(
            "dtype_variant_store: no `{binding}[IDENT] = ...;` assignment statement found in kernel \
             source -- the write-direction templater needs at least one store to pack"
        ));
    }

    let mut out = String::with_capacity(src.len() + spans.len() * 256);
    let mut cursor = 0usize;
    for (pos, ident, value_start, semi) in &spans {
        out.push_str(&src[cursor..*pos]);
        let value_expr = src[*value_start..*semi].trim();
        out.push_str(&pack_stmt(binding, ident, value_expr, dt));
        cursor = semi + 1; // consume the trailing `;` too -- pack_stmt supplies its own
    }
    out.push_str(&src[cursor..]);
    Ok(out)
}

/// The compound statement replacing `<binding>[<ident>] = <value_expr>;` -
/// hoists `value_expr` and `ident` into named `let`s FIRST (so each is
/// evaluated exactly once, even though the pack/word/half math below
/// references them more than once), then does the actual read-modify-write:
/// read the word, clear exactly the target half's bits (`& ~(0xFFFFu <<
/// shift)`), OR in the new packed bits shifted into that half. The sibling
/// half's bits are never touched - they are simply not part of `mask`.
fn pack_stmt(binding: &str, ident: &str, value_expr: &str, dt: DType) -> String {
    let bits = match dt {
        DType::BF16 => bf16_pack_expr("_pf"),
        other => unreachable!(
            "pack_stmt: dtype_variant_store already rejected {other:?} before this point was reached"
        ),
    };
    format!(
        "{{ let _pf = {value_expr}; let _pi = {ident}; let _pw = _pi >> 1u; let _ps = 16u * (_pi & 1u); \
         let _pb = {bits}; {binding}[_pw] = ({binding}[_pw] & ~(0xFFFFu << _ps)) | (_pb << _ps); }}"
    )
}

/// The inline f32→bf16 PACK for the already-hoisted `let`-bound value named
/// `value_ident`: round-to-nearest-even on the low 16 bits (the standard
/// "add a rounding bias, then truncate" bf16 technique - matches
/// `model::half::f32_to_bf16`'s host algorithm, NOT plain truncation, so a
/// device-packed cache value and a host-packed weight round the SAME way).
/// `rounding_bias = 0x7FFF + ((bits >> 16) & 1)` - the extra `+1` when the
/// bit BELOW the rounding point is already 1 implements ties-to-even (a
/// value exactly halfway between two representable bf16s rounds toward the
/// one with an even low mantissa bit, not always up). Returns the packed
/// 16-bit pattern in the LOW 16 bits of a `u32` - [`pack_stmt`] shifts it
/// into position for whichever half `IDENT` selects.
fn bf16_pack_expr(value_ident: &str) -> String {
    format!(
        "(((bitcast<u32>({value_ident}) + (0x7FFFu + ((bitcast<u32>({value_ident}) >> 16u) & 1u))) >> 16u) & 0xFFFFu)"
    )
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

/// Native f16 COMPUTE tier (B11) - a DIFFERENT mechanism from
/// [`dtype_variant`]/[`dtype_variant_store`]'s storage tier, not a third mode
/// bolted onto them. The storage tier decodes packed bf16/f16 BYTES to f32
/// with plain integer/bitcast WGSL - no device feature, runs everywhere
/// including the CPU JIT, because the arithmetic stays fp32 the whole time.
/// This tier is the opposite: the WGSL body already does its arithmetic in
/// the `f16` TYPE, in registers - genuinely narrower ALU ops, not
/// decode-then-fp32-compute. That requires the WGSL `enable f16;` extension,
/// gated on `wgpu::Features::SHADER_F16` (requested by `backend-wgpu`
/// wherever the adapter reports it, a no-op everywhere else).
///
/// This function's whole job is the ONE textual step a hand-written native-f16
/// kernel body needs to become a registrable `Variant`: prepending the
/// `enable f16;` directive `wgpu`/naga require before any `f16`-typed
/// declaration is legal, and giving it the same `(name, source)` shape
/// [`interned`]/[`dtype_variant`] already use, so it flows through the same
/// kernel-registry/selector machinery as any other kernel. Deliberately NOT a
/// general rewrite engine (unlike `dtype_variant`, which rewrites an existing
/// f32 kernel's storage declaration and every load): a correct, general
/// f32-local-var-to-f16 source transform would have to reason about which
/// locals are safe to narrow, which is exactly the "third `Ops` dispatch
/// class" this phase deliberately did not attempt to build. `body` is
/// expected to already be a complete, self-contained kernel written with
/// real `f16`-typed locals; this function does not inspect or validate
/// that, only wraps it.
///
/// Deliberately wgpu-only. Before M8.0 (`kernel-performance.md`) the REAL
/// reason was worse than "cannot compile": this repo's CPU JIT (`wgsl-cpu`)
/// had no `f16` entry in its own `Ty` lattice (`{F32, U32, I32, Bool}`), and
/// its `Ty::from_scalar` took only `naga::ScalarKind` - not the whole
/// `naga::Scalar` - so it mapped EVERY `ScalarKind::Float` - f16 (2-byte) and
/// f32 (4-byte) alike - to the SAME `Ty::F32` arm, silently. Confirmed at the
/// time by actually compiling AND RUNNING [`native_f16_poc::ELEMENTWISE_FMA`]
/// through `wgsl_cpu::Jit::new`: it compiled WITHOUT ERROR and ran EVERY
/// `f16` operation as fp32 - e.g. `60000.0 * 1.0 + 6000.0` (which a real f16
/// ALU saturates to `+inf`, past f16's 65504 max) came back as the plain
/// fp32 sum `66000.0`. M8.0 fixed this structurally: `Ty::from_scalar` now
/// takes the whole `Scalar` (kind AND width) and refuses width-2 float
/// outright, naming f16 in the error, so `wgsl_cpu::Jit::new` now REJECTS
/// this source instead of silently miscompiling it. This function stays
/// wgpu-only regardless - the CPU JIT still has no `f16` entry in its `Ty`
/// lattice, it now just fails loudly instead of silently, and
/// `caps.numeric.f16` staying structurally `false` on `backend-cpu`
/// (unconditionally, independent of anything wgpu measures - see
/// `crates/backend-wgpu/tests/native_f16.rs`'s
/// `numeric_f16_never_entangles_across_backends`) remains the belt to that
/// fix's suspenders, not a lone line of defense any more. See
/// `crates/backend-wgpu/tests/native_f16.rs`'s
/// `native_f16_kernel_is_rejected_by_the_cpu_jit_rather_than_silently_
/// diverging` for the real compile-and-refuse proof.
pub fn native_f16_variant(name: &'static str, body: &'static str) -> Variant {
    let source = format!("enable f16;\n{body}");
    (name, Box::leak(source.into_boxed_str()))
}

/// The B11 proof-of-concept native-f16 kernel bodies. Neither is wired into
/// [`dtype_variant`]/`model::ops::Weight` - see [`native_f16_variant`]'s own
/// doc comment for why a real `Weight::F16Native` dispatch tier is explicitly
/// out of this phase's scope. Both need [`native_f16_variant`] to become a
/// compilable `Variant` (they are not `enable f16;`-prefixed themselves, so
/// pasting either directly into a kernel list without that wrapper fails to
/// compile - deliberately, so the `enable` directive can never be forgotten
/// at a call site).
pub mod native_f16_poc {
    /// Correctness PoC: ONE native f16 fused-multiply-add per output element
    /// (`out = a*b+c`, computed entirely in `f16` registers - the `f32`
    /// storage arrays are I/O only, converted at the boundary). Deliberately
    /// the SIMPLEST possible native-arithmetic kernel - an elementwise op,
    /// not a reduction - so a wrong result can only come from the `f16`
    /// arithmetic itself, never from a reduction's summation order.
    pub const ELEMENTWISE_FMA: &str = r#"
struct Params { n: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read>       c:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }

    let ah: f16 = f16(a[idx]);
    let bh: f16 = f16(b[idx]);
    let ch: f16 = f16(c[idx]);
    let rh: f16 = ah * bh + ch;
    out[idx] = f32(rh);
}
"#;

    /// Throughput PoC: byte-for-byte the same dependency-free, register-only
    /// FMA-chain SHAPE `kernels::ROOF_FMA` (`roof_fma.wgsl`) uses to measure
    /// the silicon's fp32 rate - eight independent accumulators so the
    /// pipeline is never dependency-stalled, `c`/`d` arriving through the
    /// uniform (bitcast from `u32`) so nothing constant-folds, `c=d=0.5h`
    /// (fixed point `1.0h`, no overflow/denormals in steady state) - with
    /// every accumulator declared `f16` instead of `f32`. Reusing the exact
    /// shape is deliberate: it is what makes a timing comparison against
    /// `ROOF_FMA` apples-to-apples (identical dispatch geometry, identical
    /// amount of arithmetic per thread-iteration; the ONLY difference is the
    /// register width the shader compiler emits ALU ops for), rather than
    /// comparing two kernels that also differ in memory traffic or thread
    /// occupancy for unrelated reasons.
    pub const ROOF_FMA: &str = r#"
struct Params {
    n: u32,      // active threads
    iters: u32,  // FMA-loop trip count (the caller calibrates this for duration)
    c: u32,      // bitcast<f32> multiplier, narrowed to f16 on load
    d: u32,      // bitcast<f32> addend, narrowed to f16 on load
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }

    let c: f16 = f16(bitcast<f32>(p.c));
    let d: f16 = f16(bitcast<f32>(p.d));
    let s: f16 = f16(inp[idx]);

    var a0 = s;
    var a1 = s + 1.0h;
    var a2 = s + 2.0h;
    var a3 = s + 3.0h;
    var a4 = s + 4.0h;
    var a5 = s + 5.0h;
    var a6 = s + 6.0h;
    var a7 = s + 7.0h;

    for (var i: u32 = 0u; i < p.iters; i = i + 1u) {
        a0 = a0 * c + d;
        a1 = a1 * c + d;
        a2 = a2 * c + d;
        a3 = a3 * c + d;
        a4 = a4 * c + d;
        a5 = a5 * c + d;
        a6 = a6 * c + d;
        a7 = a7 * c + d;
    }

    out[idx] = f32(((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)));
}
"#;
}

/// M8.7 - the ONE hand-written, production (not PoC) native-f16 dispatch
/// kernel: a native-f16 sibling of `matmul_reg3.wgsl`, registered via
/// [`native_f16_variant`] and dispatched through
/// `gpu_core::provider::native_f16::NativeF16Provider`. Deliberately NOT a
/// general f32-to-f16 source rewrite (see [`native_f16_variant`]'s own doc
/// comment for why that is out of scope) - this is a second, hand-written
/// kernel body, kept in lock-step with `matmul_reg3.wgsl` by construction:
/// identical `Params`, identical 128x128 tile / 8x8-per-thread register
/// block / 256-thread workgroup / software-pipelined K-chunk staging (every
/// global load, shared-memory store, barrier, and guarded output write is
/// BYTE-IDENTICAL to `matmul_reg3.wgsl` - the diff is exactly the inner FMA
/// block below), so a reviewer diffing the two sees only the one change
/// this milestone is actually about.
///
/// **What's actually narrowed, and what is not.** `x`/`w`/`out` stay plain
/// `array<f32>` storage bindings - no packed-f16 buffer layout, no
/// bandwidth change from `matmul_reg3`'s own. Only the MULTIPLY narrows:
/// each of the 64 per-thread products converts its two `f32` shared-memory
/// operands to `f16` (`f16(a)`/`f16(b)`), multiplies in `f16`, then widens
/// the product back to `f32` (`f32(ah * bh)`) before adding into the
/// existing `f32` accumulator - narrow-register multiply, wide accumulate,
/// exactly the mechanism `native_f16_poc::ROOF_FMA` measured (1.38x-3.76x on
/// this campaign's own Intel Arc iGPU, `kernel-performance.md`'s B11/M8.7
/// entries) rather than a bandwidth-driven win. A packed-f16 WEIGHT buffer
/// (halving bytes moved, the way `dtype_variant`'s storage tier already
/// does for `BF16`/`F16`) is a real, separate follow-up this milestone does
/// not attempt - see `NativeF16Provider`'s own module doc for the exact
/// reason (a plain-f32 buffer bit-pattern must never be reachable through a
/// path that expects packed f16 bytes, or vice versa).
///
/// **Subnormals flush to zero on this hardware, by design, not by
/// oversight.** `native_f16_poc::ELEMENTWISE_FMA`'s own real-hardware
/// measurement (`kernel-performance.md`'s B11 table) already established
/// that this campaign's Intel Arc iGPU's native-f16 ALU flushes a subnormal
/// `f16` RESULT to zero rather than preserving it - the identical
/// `f16(a) * f16(b)` primitive this kernel's inner loop uses, just inside a
/// GEMM instead of a single FMA. `crates/gpu-core/tests/
/// native_f16_provider.rs`'s own subnormal-boundary case pins this same,
/// already-documented behaviour for the GEMM path rather than silently
/// ignoring it.
pub mod native_f16_matmul {
    /// `enable f16;` is prepended by [`super::native_f16_variant`] - this
    /// constant, like [`super::native_f16_poc`]'s two bodies, is NOT itself
    /// prefixed with it (so pasting it anywhere without that wrapper fails to
    /// compile, deliberately - the same contract [`super::native_f16_poc`]'s
    /// own doc comment states).
    pub const MATMUL_REG3_F16N: &str = r#"
struct Params { m: u32, k: u32, n: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       w:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 8u;
const SP: u32 = 129u;  // padded shared stride (BM + 1)
const WG: u32 = 256u;
const LN: u32 = 16u;   // lane grid: 16 x 16 threads, stride-16 interleave

var<workgroup> As: array<f32, 1032>;  // BK*SP, k-major: As[kk*SP + r]
var<workgroup> Bs: array<f32, 1032>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let tid = lid.x;
    let ty = tid / LN;
    let tx = tid % LN;
    let wg = wgid.y * nwg.x + wgid.x;
    let tiles_n = (p.n + BN - 1u) / BN;
    let row0 = (wg / tiles_n) * BM;
    let col0 = (wg % tiles_n) * BN;

    // Each thread stages 4 A and 4 B elements; only the k-offset moves per chunk.
    var sr: array<u32, 4>;
    var skk: array<u32, 4>;
    var arow_g: array<u32, 4>;
    var brow_g: array<u32, 4>;
    for (var e = 0u; e < 4u; e = e + 1u) {
        let idx = tid + e * WG;   // 0..1023
        let r = idx / BK;         // 0..127
        let kk = idx % BK;        // 0..7
        sr[e] = r; skk[e] = kk;
        arow_g[e] = row0 + r;
        brow_g[e] = col0 + r;
    }

    // 64 scalar-register accumulators, f32 -- unchanged from matmul_reg3.
    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0; var c04 = 0.0; var c05 = 0.0; var c06 = 0.0; var c07 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0; var c14 = 0.0; var c15 = 0.0; var c16 = 0.0; var c17 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0; var c24 = 0.0; var c25 = 0.0; var c26 = 0.0; var c27 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0; var c34 = 0.0; var c35 = 0.0; var c36 = 0.0; var c37 = 0.0;
    var c40 = 0.0; var c41 = 0.0; var c42 = 0.0; var c43 = 0.0; var c44 = 0.0; var c45 = 0.0; var c46 = 0.0; var c47 = 0.0;
    var c50 = 0.0; var c51 = 0.0; var c52 = 0.0; var c53 = 0.0; var c54 = 0.0; var c55 = 0.0; var c56 = 0.0; var c57 = 0.0;
    var c60 = 0.0; var c61 = 0.0; var c62 = 0.0; var c63 = 0.0; var c64 = 0.0; var c65 = 0.0; var c66 = 0.0; var c67 = 0.0;
    var c70 = 0.0; var c71 = 0.0; var c72 = 0.0; var c73 = 0.0; var c74 = 0.0; var c75 = 0.0; var c76 = 0.0; var c77 = 0.0;

    var rA: array<f32, 4>;
    var rB: array<f32, 4>;

    let nchunks = (p.k + BK - 1u) / BK;

    // Prime: load chunk 0 into shared.
    for (var e = 0u; e < 4u; e = e + 1u) {
        let gk = skk[e];
        if (arow_g[e] < p.m && gk < p.k) { As[skk[e] * SP + sr[e]] = x[arow_g[e] * p.k + gk]; }
        else                             { As[skk[e] * SP + sr[e]] = 0.0; }
        if (brow_g[e] < p.n && gk < p.k) { let wi = brow_g[e] * p.k + gk; Bs[skk[e] * SP + sr[e]] = w[wi]; }
        else                             { Bs[skk[e] * SP + sr[e]] = 0.0; }
    }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        let has_next = c + 1u < nchunks;
        if (has_next) {
            let k1 = (c + 1u) * BK;
            for (var e = 0u; e < 4u; e = e + 1u) {
                let gk = k1 + skk[e];
                if (arow_g[e] < p.m && gk < p.k) { rA[e] = x[arow_g[e] * p.k + gk]; } else { rA[e] = 0.0; }
                if (brow_g[e] < p.n && gk < p.k) { let wi = brow_g[e] * p.k + gk; rB[e] = w[wi]; } else { rB[e] = 0.0; }
            }
        }
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let ao = kk * SP + ty;
            let bo = kk * SP + tx;
            let a0 = As[ao + 0u];
            let a1 = As[ao + 16u];
            let a2 = As[ao + 32u];
            let a3 = As[ao + 48u];
            let a4 = As[ao + 64u];
            let a5 = As[ao + 80u];
            let a6 = As[ao + 96u];
            let a7 = As[ao + 112u];
            let b0 = Bs[bo + 0u];
            let b1 = Bs[bo + 16u];
            let b2 = Bs[bo + 32u];
            let b3 = Bs[bo + 48u];
            let b4 = Bs[bo + 64u];
            let b5 = Bs[bo + 80u];
            let b6 = Bs[bo + 96u];
            let b7 = Bs[bo + 112u];
            // The one real change from matmul_reg3.wgsl: narrow both
            // operands to f16 registers, multiply IN f16, then widen the
            // product back to f32 before accumulating -- narrow-ALU
            // multiply, wide accumulate, matching native_f16_poc::ROOF_FMA's
            // own measured mechanism (see this module's doc comment).
            let ah0: f16 = f16(a0); let ah1: f16 = f16(a1); let ah2: f16 = f16(a2); let ah3: f16 = f16(a3);
            let ah4: f16 = f16(a4); let ah5: f16 = f16(a5); let ah6: f16 = f16(a6); let ah7: f16 = f16(a7);
            let bh0: f16 = f16(b0); let bh1: f16 = f16(b1); let bh2: f16 = f16(b2); let bh3: f16 = f16(b3);
            let bh4: f16 = f16(b4); let bh5: f16 = f16(b5); let bh6: f16 = f16(b6); let bh7: f16 = f16(b7);
            c00 += f32(ah0 * bh0); c01 += f32(ah0 * bh1); c02 += f32(ah0 * bh2); c03 += f32(ah0 * bh3); c04 += f32(ah0 * bh4); c05 += f32(ah0 * bh5); c06 += f32(ah0 * bh6); c07 += f32(ah0 * bh7);
            c10 += f32(ah1 * bh0); c11 += f32(ah1 * bh1); c12 += f32(ah1 * bh2); c13 += f32(ah1 * bh3); c14 += f32(ah1 * bh4); c15 += f32(ah1 * bh5); c16 += f32(ah1 * bh6); c17 += f32(ah1 * bh7);
            c20 += f32(ah2 * bh0); c21 += f32(ah2 * bh1); c22 += f32(ah2 * bh2); c23 += f32(ah2 * bh3); c24 += f32(ah2 * bh4); c25 += f32(ah2 * bh5); c26 += f32(ah2 * bh6); c27 += f32(ah2 * bh7);
            c30 += f32(ah3 * bh0); c31 += f32(ah3 * bh1); c32 += f32(ah3 * bh2); c33 += f32(ah3 * bh3); c34 += f32(ah3 * bh4); c35 += f32(ah3 * bh5); c36 += f32(ah3 * bh6); c37 += f32(ah3 * bh7);
            c40 += f32(ah4 * bh0); c41 += f32(ah4 * bh1); c42 += f32(ah4 * bh2); c43 += f32(ah4 * bh3); c44 += f32(ah4 * bh4); c45 += f32(ah4 * bh5); c46 += f32(ah4 * bh6); c47 += f32(ah4 * bh7);
            c50 += f32(ah5 * bh0); c51 += f32(ah5 * bh1); c52 += f32(ah5 * bh2); c53 += f32(ah5 * bh3); c54 += f32(ah5 * bh4); c55 += f32(ah5 * bh5); c56 += f32(ah5 * bh6); c57 += f32(ah5 * bh7);
            c60 += f32(ah6 * bh0); c61 += f32(ah6 * bh1); c62 += f32(ah6 * bh2); c63 += f32(ah6 * bh3); c64 += f32(ah6 * bh4); c65 += f32(ah6 * bh5); c66 += f32(ah6 * bh6); c67 += f32(ah6 * bh7);
            c70 += f32(ah7 * bh0); c71 += f32(ah7 * bh1); c72 += f32(ah7 * bh2); c73 += f32(ah7 * bh3); c74 += f32(ah7 * bh4); c75 += f32(ah7 * bh5); c76 += f32(ah7 * bh6); c77 += f32(ah7 * bh7);
        }
        workgroupBarrier();
        if (has_next) {
            for (var e = 0u; e < 4u; e = e + 1u) {
                As[skk[e] * SP + sr[e]] = rA[e];
                Bs[skk[e] * SP + sr[e]] = rB[e];
            }
        }
        workgroupBarrier();
    }

    // Guarded stores: thread (ty,tx) owns rows ty+16i and columns tx+16j.
    let m0 = row0 + ty + 0u;
    let m1 = row0 + ty + 16u;
    let m2 = row0 + ty + 32u;
    let m3 = row0 + ty + 48u;
    let m4 = row0 + ty + 64u;
    let m5 = row0 + ty + 80u;
    let m6 = row0 + ty + 96u;
    let m7 = row0 + ty + 112u;

    if (m0 < p.m) {
        let r0 = m0 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r0 + 0u] = c00; }
        if (col0 + tx + 16u < p.n) { out[r0 + 16u] = c01; }
        if (col0 + tx + 32u < p.n) { out[r0 + 32u] = c02; }
        if (col0 + tx + 48u < p.n) { out[r0 + 48u] = c03; }
        if (col0 + tx + 64u < p.n) { out[r0 + 64u] = c04; }
        if (col0 + tx + 80u < p.n) { out[r0 + 80u] = c05; }
        if (col0 + tx + 96u < p.n) { out[r0 + 96u] = c06; }
        if (col0 + tx + 112u < p.n) { out[r0 + 112u] = c07; }
    }
    if (m1 < p.m) {
        let r1 = m1 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r1 + 0u] = c10; }
        if (col0 + tx + 16u < p.n) { out[r1 + 16u] = c11; }
        if (col0 + tx + 32u < p.n) { out[r1 + 32u] = c12; }
        if (col0 + tx + 48u < p.n) { out[r1 + 48u] = c13; }
        if (col0 + tx + 64u < p.n) { out[r1 + 64u] = c14; }
        if (col0 + tx + 80u < p.n) { out[r1 + 80u] = c15; }
        if (col0 + tx + 96u < p.n) { out[r1 + 96u] = c16; }
        if (col0 + tx + 112u < p.n) { out[r1 + 112u] = c17; }
    }
    if (m2 < p.m) {
        let r2 = m2 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r2 + 0u] = c20; }
        if (col0 + tx + 16u < p.n) { out[r2 + 16u] = c21; }
        if (col0 + tx + 32u < p.n) { out[r2 + 32u] = c22; }
        if (col0 + tx + 48u < p.n) { out[r2 + 48u] = c23; }
        if (col0 + tx + 64u < p.n) { out[r2 + 64u] = c24; }
        if (col0 + tx + 80u < p.n) { out[r2 + 80u] = c25; }
        if (col0 + tx + 96u < p.n) { out[r2 + 96u] = c26; }
        if (col0 + tx + 112u < p.n) { out[r2 + 112u] = c27; }
    }
    if (m3 < p.m) {
        let r3 = m3 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r3 + 0u] = c30; }
        if (col0 + tx + 16u < p.n) { out[r3 + 16u] = c31; }
        if (col0 + tx + 32u < p.n) { out[r3 + 32u] = c32; }
        if (col0 + tx + 48u < p.n) { out[r3 + 48u] = c33; }
        if (col0 + tx + 64u < p.n) { out[r3 + 64u] = c34; }
        if (col0 + tx + 80u < p.n) { out[r3 + 80u] = c35; }
        if (col0 + tx + 96u < p.n) { out[r3 + 96u] = c36; }
        if (col0 + tx + 112u < p.n) { out[r3 + 112u] = c37; }
    }
    if (m4 < p.m) {
        let r4 = m4 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r4 + 0u] = c40; }
        if (col0 + tx + 16u < p.n) { out[r4 + 16u] = c41; }
        if (col0 + tx + 32u < p.n) { out[r4 + 32u] = c42; }
        if (col0 + tx + 48u < p.n) { out[r4 + 48u] = c43; }
        if (col0 + tx + 64u < p.n) { out[r4 + 64u] = c44; }
        if (col0 + tx + 80u < p.n) { out[r4 + 80u] = c45; }
        if (col0 + tx + 96u < p.n) { out[r4 + 96u] = c46; }
        if (col0 + tx + 112u < p.n) { out[r4 + 112u] = c47; }
    }
    if (m5 < p.m) {
        let r5 = m5 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r5 + 0u] = c50; }
        if (col0 + tx + 16u < p.n) { out[r5 + 16u] = c51; }
        if (col0 + tx + 32u < p.n) { out[r5 + 32u] = c52; }
        if (col0 + tx + 48u < p.n) { out[r5 + 48u] = c53; }
        if (col0 + tx + 64u < p.n) { out[r5 + 64u] = c54; }
        if (col0 + tx + 80u < p.n) { out[r5 + 80u] = c55; }
        if (col0 + tx + 96u < p.n) { out[r5 + 96u] = c56; }
        if (col0 + tx + 112u < p.n) { out[r5 + 112u] = c57; }
    }
    if (m6 < p.m) {
        let r6 = m6 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r6 + 0u] = c60; }
        if (col0 + tx + 16u < p.n) { out[r6 + 16u] = c61; }
        if (col0 + tx + 32u < p.n) { out[r6 + 32u] = c62; }
        if (col0 + tx + 48u < p.n) { out[r6 + 48u] = c63; }
        if (col0 + tx + 64u < p.n) { out[r6 + 64u] = c64; }
        if (col0 + tx + 80u < p.n) { out[r6 + 80u] = c65; }
        if (col0 + tx + 96u < p.n) { out[r6 + 96u] = c66; }
        if (col0 + tx + 112u < p.n) { out[r6 + 112u] = c67; }
    }
    if (m7 < p.m) {
        let r7 = m7 * p.n + col0 + tx;
        if (col0 + tx + 0u < p.n) { out[r7 + 0u] = c70; }
        if (col0 + tx + 16u < p.n) { out[r7 + 16u] = c71; }
        if (col0 + tx + 32u < p.n) { out[r7 + 32u] = c72; }
        if (col0 + tx + 48u < p.n) { out[r7 + 48u] = c73; }
        if (col0 + tx + 64u < p.n) { out[r7 + 64u] = c74; }
        if (col0 + tx + 80u < p.n) { out[r7 + 80u] = c75; }
        if (col0 + tx + 96u < p.n) { out[r7 + 96u] = c76; }
        if (col0 + tx + 112u < p.n) { out[r7 + 112u] = c77; }
    }
}
"#;
}

/// M8.6: the E4M3/E5M2 portable FP8
/// decode expressions - a DIFFERENT mechanism from [`dtype_variant`], not a
/// third tier bolted onto it. `dtype_variant`'s declaration/load rewrite is
/// hardcoded to 2-per-`u32` packing (bf16/f16) with no second (scale)
/// binding; FP8 here packs 4 bytes/`u32` AND needs a blockwise scale binding
/// indexed `(row/128, col/128)`, neither of which that generic rewrite
/// supports. So the FP8 matmul kernels (`matmul_gemv_f8e4m3.wgsl`/
/// `matmul_gemv_f8e5m2.wgsl`) are hand-written, not templated - these two
/// functions exist so the decode EXPRESSION itself (the fiddly, easy-to-get-
/// wrong part) has exactly one implementation, pasted verbatim into both the
/// hand-written kernel and this module's own exhaustive test, instead of a
/// second hand-copy that could silently drift. `byte_expr` must already
/// evaluate to a `u32` holding the raw byte in bits `[0,8)` (upper bits
/// zero). Unlike [`bf16_decode_expr`]/[`f16_decode_expr`] this does not
/// extract the byte from a packed word itself: the extraction differs per
/// kernel, since which word/lane a byte lives in is a property of the
/// CALLER's own layout, not of the decode.
///
/// **E4M3 (OCP/PyTorch `float8_e4m3fn`): 1 sign, 4 exponent (bias 7), 3
/// mantissa bits, NO infinities** - only byte `0x7F`/`0xFF` (exponent and
/// mantissa both all-ones) is NaN; every OTHER `exponent==0b1111` byte is a
/// regular (the largest, magnitude 448) finite value - see
/// `checkpoint::safetensors::e4m3fn_to_f32_scalar` for the field-
/// reconstruction reference this mirrors as a bit-trick (that function is
/// this format's existing HOST-side decoder; this is the same math, ported
/// to WGSL `select`/`bitcast`, not a new format).
///
/// Same magic-multiply/magic-bias shape [`f16_decode_expr`] uses (subnormal:
/// FTZ-safe magic-bias subtract of two NORMAL f32s; normal: magic-multiply by
/// `2^(127-7)`; the third branch here is a flat NaN override, not inf/NaN,
/// since this format has no infinity), scaled to a 3-bit mantissa (shift by
/// `23-3=20`, not f16's `23-10=13`) and a 4-bit exponent (bias 7, not 15).
pub fn f8e4m3_decode_expr(byte_expr: &str) -> String {
    let mag7 = format!("({byte_expr} & 0x7Fu)");
    let sign = format!("(({byte_expr} & 0x80u) << 24u)");
    // Subnormal (mag7 < 8, i.e. exponent field 0): magic-bias subtract of two
    // NORMAL f32s (exponent field fixed at 121, i.e. 2^-6) - see
    // `f16_decode_expr`'s doc for why this is FTZ-safe on every backend.
    let subnormal = format!("(bitcast<f32>(0x3C800000u | (({mag7}) << 20u)) - bitcast<f32>(0x3C800000u))");
    // Normal (mag7 >= 8): magic-multiply by 2^(127-7) = 2^120
    // (`bitcast<f32>(0x7B800000u)`) - placing the raw 4-bit exponent value
    // straight into f32's exponent field (bias 127) computes `(1+mant/8) *
    // 2^(exp-127)`; this constant rebias it to `2^(exp-7)`.
    let normal = format!("(bitcast<f32>(({mag7}) << 20u) * bitcast<f32>(0x7B800000u))");
    // The ONE reserved NaN pattern (mag7 == 0x7F, both signs) - not an
    // inf/NaN branch (this format has no infinity), a flat override: the
    // magic-multiply formula above would otherwise compute a perfectly
    // ordinary finite value (480.0) for this exact byte, which is wrong.
    format!(
        "bitcast<f32>(bitcast<u32>(select(select({subnormal}, {normal}, ({mag7}) >= 8u), \
         bitcast<f32>(0x7FC00000u), ({mag7}) == 0x7Fu)) | {sign})"
    )
}

/// **E5M2 (OCP FP8 E5M2): 1 sign, 5 exponent (bias 15), 2 mantissa bits, HAS
/// infinities** - structurally a narrower [`f16_decode_expr`] (same
/// exponent width AND bias as f16 - `65504`'s exponent range, just 10
/// mantissa bits instead of 2), so this is the IDENTICAL three-branch shape
/// (subnormal magic-bias / normal magic-multiply / inf-or-NaN), scaled to a
/// 2-bit mantissa (shift by `23-2=21`) - even the magic-bias/magic-multiply
/// CONSTANTS are byte-identical to f16's own (`0x38800000`/`0x77800000`),
/// since both share exponent bias 15 and range.
pub fn f8e5m2_decode_expr(byte_expr: &str) -> String {
    let exp = format!("(({byte_expr} >> 2u) & 0x1Fu)");
    let mag7 = format!("({byte_expr} & 0x7Fu)"); // exp(5) | mantissa(2), 7 bits
    let mant2 = format!("({byte_expr} & 0x3u)");
    let sign = format!("(({byte_expr} & 0x80u) << 24u)");
    let subnormal = format!("(bitcast<f32>(0x38800000u | (({mant2}) << 21u)) - bitcast<f32>(0x38800000u))");
    let normal = format!("(bitcast<f32>(({mag7}) << 21u) * bitcast<f32>(0x77800000u))");
    let inf_or_nan = format!("bitcast<f32>(0x7F800000u | (({mant2}) << 21u))");
    format!(
        "bitcast<f32>(bitcast<u32>(select(select({subnormal}, {normal}, ({exp}) != 0u), \
         {inf_or_nan}, ({exp}) == 31u)) | {sign})"
    )
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
    /// enough to check exhaustively, not just at a handful of sample points) -
    /// normals, subnormals (including the smallest, `0x0001`), zero/negative
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

    // --- M8.6: portable FP8 (E4M3/E5M2) decode expressions ------------------

    /// Host-side mirror of [`f8e4m3_decode_expr`]'s WGSL text - the same
    /// three-way branch (magic-bias subnormal / magic-multiply normal / a
    /// flat NaN override), byte-for-byte, kept in sync by hand exactly like
    /// [`f16_decode_bits_wgsl_mirror`].
    fn f8e4m3_decode_bits_wgsl_mirror(b: u8) -> u32 {
        let b = b as u32;
        let mag7 = b & 0x7F;
        let sign = (b & 0x80) << 24;
        let subnormal = f32::from_bits(0x3C80_0000 | (mag7 << 20)) - f32::from_bits(0x3C80_0000);
        let normal = f32::from_bits(mag7 << 20) * f32::from_bits(0x7B80_0000);
        let nan = f32::from_bits(0x7FC0_0000);
        let unsigned = if mag7 == 0x7F { nan } else if mag7 >= 8 { normal } else { subnormal };
        unsigned.to_bits() | sign
    }

    /// An INDEPENDENT E4M3 reference - direct sign/exponent/mantissa field
    /// reconstruction (not the magic-multiply/magic-bias bit tricks), scaled
    /// by an exact power-of-two float multiply, mirroring
    /// `checkpoint::safetensors::e4m3fn_to_f32_scalar`'s own formula (this
    /// crate cannot depend on `brain-checkpoint` - the dependency runs the
    /// other way - so this is re-derived independently, not imported).
    fn f8e4m3_decode_bits_independent_reference(b: u8) -> u32 {
        let sign = ((b >> 7) & 1) as u32;
        let exp = (b >> 3) & 0xF;
        let mant = (b & 0x7) as u32;
        if exp == 0xF && mant == 0x7 {
            return f32::NAN.to_bits() | (sign << 31);
        }
        let mag: f32 = if exp == 0 {
            // Subnormal: mant/8 * 2^-6 == mant * 2^-9, exact (mant fits in 3
            // bits, 2^-9 is an exact f32 power-of-two scale).
            (mant as f32) * f32::from_bits(((-9i32 + 127) as u32) << 23)
        } else {
            // Normal: (1 + mant/8) * 2^(exp-7) == (8+mant) * 2^(exp-10),
            // exact (8+mant fits in 4 bits, 2^(exp-10) exact for exp in [1,15]).
            ((8 + mant) as f32) * f32::from_bits(((exp as i32 - 10 + 127) as u32) << 23)
        };
        mag.to_bits() | (sign << 31)
    }

    /// Exhaustive: all 256 possible E4M3 byte patterns (the format's entire
    /// representable space), magic-multiply mirror vs independent reference.
    /// NaN pairs accepted as equal iff BOTH sides are NaN (only `0x7F`/`0xFF`
    /// should be, and the exhaustive sweep is what proves no OTHER byte
    /// accidentally decodes to one).
    #[test]
    fn f8e4m3_decode_matches_an_independent_reference_for_every_possible_byte() {
        let mut nan_count = 0;
        for b in 0u8..=255 {
            let got = f32::from_bits(f8e4m3_decode_bits_wgsl_mirror(b));
            let want = f32::from_bits(f8e4m3_decode_bits_independent_reference(b));
            if want.is_nan() {
                nan_count += 1;
                assert!(got.is_nan(), "b=0x{b:02x}: want NaN, got {got}");
                continue;
            }
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "b=0x{b:02x}: magic-multiply {got} (0x{:08x}) != independent reference {want} (0x{:08x})",
                got.to_bits(),
                want.to_bits()
            );
        }
        // E4M3 has EXACTLY two NaN encodings (0x7F, 0xFF) - not "no
        // infinities" alone (checked by the finite max-value assertion
        // below), the actual reserved-pattern count.
        assert_eq!(nan_count, 2, "E4M3 must have exactly 2 NaN byte patterns (0x7F, 0xFF)");
    }

    #[test]
    fn f8e4m3_decode_matches_known_values() {
        let dec = |b: u8| f32::from_bits(f8e4m3_decode_bits_wgsl_mirror(b));
        assert_eq!(dec(0x00), 0.0);
        assert!(dec(0x00).is_sign_positive());
        assert_eq!(dec(0x80), -0.0);
        assert!(dec(0x80).is_sign_negative());
        assert_eq!(dec(0x38), 1.0);
        assert_eq!(dec(0xB8), -1.0);
        assert_eq!(dec(0x08), 2f32.powi(-6)); // smallest normal
        assert_eq!(dec(0x01), 2f32.powi(-9)); // smallest subnormal
        assert_eq!(dec(0x7E), 448.0); // largest finite (E4M3FN has NO infinity)
        assert!(dec(0x7E).is_finite());
        assert!(dec(0x7F).is_nan()); // the ONE reserved positive NaN pattern
        assert!(dec(0xFF).is_nan()); // and its negative twin
        assert!(dec(0x7D).is_finite()); // exp=1111 mant!=111 is finite, NOT nan
    }

    /// Host-side mirror of [`f8e5m2_decode_expr`]'s WGSL text - structurally
    /// `f16_decode_bits_wgsl_mirror` narrowed to a 2-bit mantissa (same
    /// exponent width/bias, so the magic-bias/magic-multiply CONSTANTS are
    /// byte-identical to f16's own).
    fn f8e5m2_decode_bits_wgsl_mirror(b: u8) -> u32 {
        let b = b as u32;
        let exp = (b >> 2) & 0x1F;
        let mag7 = b & 0x7F;
        let mant2 = b & 0x3;
        let sign = (b & 0x80) << 24;
        let subnormal = f32::from_bits(0x3880_0000 | (mant2 << 21)) - f32::from_bits(0x3880_0000);
        let normal = f32::from_bits(mag7 << 21) * f32::from_bits(0x7780_0000);
        let inf_or_nan = f32::from_bits(0x7F80_0000 | (mant2 << 21));
        let unsigned = if exp == 31 { inf_or_nan } else if exp != 0 { normal } else { subnormal };
        unsigned.to_bits() | sign
    }

    /// An INDEPENDENT E5M2 reference, same shape as
    /// [`f16_decode_bits_independent_reference`] narrowed to 2 mantissa bits.
    fn f8e5m2_decode_bits_independent_reference(b: u8) -> u32 {
        let sign = ((b >> 7) & 1) as u32;
        let exp = (b >> 2) & 0x1F;
        let mant = (b & 0x3) as u32;
        let mag: f32 = if exp == 0 {
            // Subnormal: mant/4 * 2^-14 == mant * 2^-16, exact.
            (mant as f32) * f32::from_bits(((-16i32 + 127) as u32) << 23)
        } else if exp == 31 {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            // Normal: (1 + mant/4) * 2^(exp-15) == (4+mant) * 2^(exp-17), exact.
            ((4 + mant) as f32) * f32::from_bits(((exp as i32 - 17 + 127) as u32) << 23)
        };
        mag.to_bits() | (sign << 31)
    }

    /// Exhaustive: all 256 possible E5M2 byte patterns.
    #[test]
    fn f8e5m2_decode_matches_an_independent_reference_for_every_possible_byte() {
        let mut checked_inf = false;
        let mut checked_subnormal = false;
        for b in 0u8..=255 {
            let got = f32::from_bits(f8e5m2_decode_bits_wgsl_mirror(b));
            let want = f32::from_bits(f8e5m2_decode_bits_independent_reference(b));
            if want.is_nan() {
                assert!(got.is_nan(), "b=0x{b:02x}: want NaN, got {got}");
                continue;
            }
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "b=0x{b:02x}: magic-multiply {got} (0x{:08x}) != independent reference {want} (0x{:08x})",
                got.to_bits(),
                want.to_bits()
            );
            if want.is_infinite() {
                checked_inf = true;
            }
            let exp = (b >> 2) & 0x1F;
            if exp == 0 && (b & 0x3) != 0 {
                checked_subnormal = true;
            }
        }
        assert!(checked_inf, "the sweep must have covered E5M2's +-inf patterns (unlike E4M3, this format HAS infinities)");
        assert!(checked_subnormal, "the sweep must have covered at least one subnormal byte");
    }

    #[test]
    fn f8e5m2_decode_matches_known_values() {
        let dec = |b: u8| f32::from_bits(f8e5m2_decode_bits_wgsl_mirror(b));
        assert_eq!(dec(0x00), 0.0);
        assert_eq!(dec(0x80), -0.0);
        assert_eq!(dec(0x3C), 1.0);
        assert_eq!(dec(0xBC), -1.0);
        assert_eq!(dec(0x04), 2f32.powi(-14)); // smallest normal
        assert_eq!(dec(0x01), 2f32.powi(-16)); // smallest subnormal
        assert_eq!(dec(0x7B), 57344.0); // largest finite
        assert_eq!(dec(0x7C), f32::INFINITY);
        assert_eq!(dec(0xFC), f32::NEG_INFINITY);
        assert!(dec(0x7D).is_nan());
        assert!(dec(0x7F).is_nan());
        assert!(dec(0xFF).is_nan());
    }

    /// The two formats must disagree at at least one shared byte pattern -
    /// they are DIFFERENT layouts, not the same decode under two names. `0x7C`
    /// is E4M3's largest-but-one finite value but E5M2's `+inf`.
    #[test]
    fn f8e4m3_and_f8e5m2_are_not_the_same_decode() {
        let e4 = f32::from_bits(f8e4m3_decode_bits_wgsl_mirror(0x7C));
        let e5 = f32::from_bits(f8e5m2_decode_bits_wgsl_mirror(0x7C));
        assert!(e4.is_finite());
        assert!(e5.is_infinite());
        assert_ne!(e4.to_bits(), e5.to_bits());
    }

    /// `f8e4m3_decode_expr`/`f8e5m2_decode_expr` are hand-PASTED into
    /// `matmul_gemv_f8e4m3.wgsl`/`matmul_gemv_f8e5m2.wgsl` (see those
    /// kernels' own headers for why - the generic `dtype_variant` rewrite
    /// pipeline does not fit FP8's 4-per-`u32` packing plus its extra
    /// blockwise-scale binding). A hand-paste can silently drift from the
    /// function that generated it the moment either side is edited alone -
    /// this test is what makes that impossible: it asserts the exact
    /// substring these functions produce for the identifier `byte` is
    /// present, byte-for-byte, in each kernel's real source.
    #[test]
    fn the_kernels_pasted_decode_expressions_match_the_generator_functions_exactly() {
        let e4_expr = f8e4m3_decode_expr("byte");
        assert!(
            crate::MATMUL_GEMV_F8E4M3.contains(&e4_expr),
            "matmul_gemv_f8e4m3.wgsl's pasted decode expression has drifted from f8e4m3_decode_expr(\"byte\")"
        );
        let e5_expr = f8e5m2_decode_expr("byte");
        assert!(
            crate::MATMUL_GEMV_F8E5M2.contains(&e5_expr),
            "matmul_gemv_f8e5m2.wgsl's pasted decode expression has drifted from f8e5m2_decode_expr(\"byte\")"
        );
    }

    // --- B9: write-direction (pack) storage tier ----------------------------

    /// A tiny kernel shaped like `paged_kv_append_batched_word.wgsl` after
    /// its own bare-identifier hoist - `src` untouched (f32 activation),
    /// `pool` storage-tier-templatable on the WRITE side.
    const APPEND_SHAPED: &str = "\
@group(0) @binding(1) var<storage, read>       src:  array<f32>;
@group(0) @binding(2) var<storage, read_write> pool: array<f32>;
fn main() {
    let wi = 3u;
    pool[wi] = src[0];
}
";

    #[test]
    fn dtype_variant_store_rewrites_declaration_and_store() {
        let (name, src) = dtype_variant_store("append", APPEND_SHAPED, "pool", DType::BF16).unwrap();
        assert_eq!(name, "append#pool=bf16");
        // `pool` narrows to a packed u32 array, `read_write` preserved (the
        // pack needs to READ the word it is about to write)...
        assert!(src.contains("var<storage, read_write> pool: array<u32>;"), "{src}");
        // ...`src` is completely untouched (still f32, no rewrite at all).
        assert!(src.contains("var<storage, read>       src:  array<f32>;"), "{src}");
        // The value expression is hoisted exactly once (no double evaluation
        // even though the pack math below references it more than once).
        assert!(src.contains("let _pf = src[0];"), "{src}");
        assert!(src.contains("let _pi = wi;"), "{src}");
        // Read-modify-write: read the word, clear exactly the target half via
        // a mask, OR in the new bits shifted into that half.
        assert!(src.contains("pool[_pw]"), "{src}");
        assert!(src.contains("~(0xFFFFu << _ps)"), "{src}");
        // The original compact store is gone.
        assert!(!src.contains("pool[wi] = src[0];"), "{src}");
    }

    #[test]
    fn dtype_variant_store_is_stable_across_calls() {
        let a = dtype_variant_store("append", APPEND_SHAPED, "pool", DType::BF16).unwrap();
        let b = dtype_variant_store("append", APPEND_SHAPED, "pool", DType::BF16).unwrap();
        assert!(std::ptr::eq(a.0, b.0) && std::ptr::eq(a.1, b.1));
    }

    /// The real in-tree kernel (after its own B9 hoist) templates cleanly
    /// too, the same proof `dtype_variant_handles_the_real_matmul_kernel`
    /// gives for the read direction.
    #[test]
    fn dtype_variant_store_handles_the_real_paged_kv_append_batched_word_kernel() {
        let (name, src) = dtype_variant_store(
            "paged_kv_append_batched_word",
            crate::PAGED_KV_APPEND_BATCHED_WORD,
            "pool",
            DType::BF16,
        )
        .unwrap();
        assert_eq!(name, "paged_kv_append_batched_word#pool=bf16");
        assert!(src.contains("array<u32>"), "{src}");
        assert!(src.contains("let _pi = wi;"), "{src}");
        assert!(src.contains("pool[_pw]"), "{src}");
    }

    /// A compound store index (no bare-identifier hoist) is a loud `Err`,
    /// never a silent double-evaluating rewrite - same discipline as the read
    /// direction.
    #[test]
    fn dtype_variant_store_rejects_a_compound_index() {
        const UNHOISTED: &str = "\
@group(0) @binding(2) var<storage, read_write> pool: array<f32>;
fn main() { pool[base + i] = 1.0; }
";
        let err = dtype_variant_store("u", UNHOISTED, "pool", DType::BF16).unwrap_err();
        assert!(err.contains("bare identifier"), "{err}");
        assert!(err.contains("base + i"), "{err}");
    }

    /// A binding that is only ever READ (no `binding[IDENT] = ...;` anywhere)
    /// is a loud `Err`, not a silent no-op that returns an unrewritten kernel
    /// under a bf16 name.
    #[test]
    fn dtype_variant_store_errors_when_no_assignment_is_found() {
        const READ_ONLY: &str = "\
@group(0) @binding(1) var<storage, read> pool: array<f32>;
fn main() { let v = pool[wi]; }
";
        let err = dtype_variant_store("u", READ_ONLY, "pool", DType::BF16).unwrap_err();
        assert!(err.contains("no `pool[IDENT] = ...;`"), "{err}");
    }

    /// A plain READ occurrence (`let v = pool[wi];`) sitting alongside a
    /// genuine store in the SAME source is left untouched by the store
    /// rewriter - it only rewrites assignment targets, distinguishing `=`
    /// from `==` and from a bare load.
    #[test]
    fn dtype_variant_store_leaves_reads_of_the_same_binding_untouched() {
        const MIXED: &str = "\
@group(0) @binding(2) var<storage, read_write> pool: array<f32>;
fn main() {
    let old = pool[wi];
    pool[wi] = 2.0;
}
";
        let (_, src) = dtype_variant_store("u", MIXED, "pool", DType::BF16).unwrap();
        // The read is untouched - still a plain packed-u32 word load (no
        // decode expression was requested for it).
        assert!(src.contains("let old = pool[wi];"), "{src}");
        // The store is rewritten.
        assert!(src.contains("let _pf = 2.0;"), "{src}");
    }

    /// Only `DType::BF16` is implemented for the write direction (B9) -
    /// `F16`'s real re-biased pack is a follow-up, and `F32`/`I8`/`Q4` have no
    /// storage-tier rewrite concept at all, same as the read direction.
    #[test]
    fn dtype_variant_store_rejects_unimplemented_tiers() {
        for dt in [DType::F32, DType::F16, DType::I8, DType::Q4] {
            let err = dtype_variant_store("append", APPEND_SHAPED, "pool", dt).unwrap_err();
            assert!(err.contains("BF16"), "{dt:?}: {err}");
        }
    }

    /// Host mirror of [`bf16_pack_expr`]'s WGSL text - the exact same
    /// add-rounding-bias-then-truncate technique, checked directly in Rust
    /// rather than trusted from the generated text alone.
    fn bf16_pack_bits_wgsl_mirror(f: f32) -> u16 {
        let bits = f.to_bits();
        let bias = 0x7FFFu32 + ((bits >> 16) & 1);
        (bits.wrapping_add(bias) >> 16) as u16
    }

    /// Round-to-nearest-EVEN, not truncation - the same distinction B4's own
    /// `model::half::f32_to_bf16` ledger entry pinned for the host packer,
    /// checked here for the WGSL device packer's mirror: an exact tie rounds
    /// toward the EVEN low mantissa bit (down when the truncated value is
    /// already even, up when it is odd), and a value strictly past the
    /// halfway point always rounds up regardless of parity.
    #[test]
    fn bf16_pack_matches_known_values_and_rounds_to_nearest_even() {
        assert_eq!(bf16_pack_bits_wgsl_mirror(1.0), 0x3F80);
        assert_eq!(bf16_pack_bits_wgsl_mirror(-4.0), 0xC080);
        // Exact tie, truncated mantissa 0x3F80 is EVEN -> rounds DOWN (stays).
        assert_eq!(bf16_pack_bits_wgsl_mirror(f32::from_bits(0x3F80_8000)), 0x3F80);
        // Just past the halfway point -> rounds UP regardless of parity.
        assert_eq!(bf16_pack_bits_wgsl_mirror(f32::from_bits(0x3F80_8001)), 0x3F81);
        // Exact tie, truncated mantissa 0x3F81 is ODD -> rounds UP to EVEN (0x3F82).
        assert_eq!(bf16_pack_bits_wgsl_mirror(f32::from_bits(0x3F81_8000)), 0x3F82);
    }

    /// Pack-then-decode round-trips within bf16's own precision: the decoded
    /// value's top 16 bits equal the packed pattern (decode is exact bit
    /// widening - see [`bf16_decode_expr`]'s own doc comment), so packing then
    /// decoding must reproduce EXACTLY what the pack step chose, for a sweep
    /// of representative magnitudes/signs (not exhaustive - bf16's pack step
    /// is a plain rounding of a much larger f32 space, unlike f16's fully
    /// enumerable 65536-pattern decode space).
    #[test]
    fn bf16_pack_then_decode_reproduces_the_packed_pattern() {
        let samples: &[f32] =
            &[0.0, -0.0, 1.0, -1.0, 3.14285, -2.71993, 1e-30, -1e30, 65504.0, 123_456.79, -0.000123];
        for &f in samples {
            let packed = bf16_pack_bits_wgsl_mirror(f);
            // Decode: bf16 is exact bit widening (top 16 bits of an f32).
            let decoded = f32::from_bits((packed as u32) << 16);
            // Re-packing the decoded value must reproduce the SAME 16-bit
            // pattern (a bf16 value is already exactly representable, so
            // there is no further rounding on the second pack).
            assert_eq!(bf16_pack_bits_wgsl_mirror(decoded), packed, "f={f} decoded={decoded}");
        }
    }

    /// [`native_f16_variant`] does exactly one textual thing: prepend
    /// `enable f16;` as its own first line, leaving the body byte-identical
    /// otherwise - no rewrite of the kind [`dtype_variant`] performs.
    #[test]
    fn native_f16_variant_prepends_enable_f16_and_leaves_the_body_untouched() {
        let (name, src) = native_f16_variant("poc", native_f16_poc::ELEMENTWISE_FMA);
        assert_eq!(name, "poc");
        let mut lines = src.lines();
        assert_eq!(lines.next(), Some("enable f16;"));
        assert!(src.ends_with(native_f16_poc::ELEMENTWISE_FMA));
        assert!(!native_f16_poc::ELEMENTWISE_FMA.contains("enable f16;"));
    }

    #[test]
    fn native_f16_variant_wraps_the_roof_fma_body_too() {
        let (name, src) = native_f16_variant("roof_fma_f16", native_f16_poc::ROOF_FMA);
        assert_eq!(name, "roof_fma_f16");
        assert!(src.starts_with("enable f16;\n"));
        assert!(src.contains("var a0 = s;"));
    }
}
