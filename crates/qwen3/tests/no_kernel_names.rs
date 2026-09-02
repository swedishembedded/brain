// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! B7's own TDD gate: the migrated qwen3 forward-inference linear dispatch
//! (the 7 per-layer projections' fp32-vs-int8 fork, batched forward AND
//! KV-cache decode) must go through `model::ops::Ops`/`Weight`, never a
//! hand-picked kernel index or an `Option<crate::q8::Q8>` inspected at
//! dispatch time.
//!
//! Confirmed RED against the pre-B7 source (verified by hand before this
//! migration started, not re-derivable from the current tree - the four
//! fork sites this gate now polices were deleted, not merely refactored):
//! `model.rs`'s `forward_steps`/`decode_steps` each `if let Some((q8, ql)) =
//! q8l { q8.mm8(...) } else { ...MATMUL_I8_GEMV/MATMUL_GEMV/linear_kernel... }`
//! four times, and `self.q8: Option<crate::q8::Q8>` was inspected directly.
//! Now GREEN.
//!
//! **Scope - what this gate checks, and what it deliberately allow-lists.**
//!
//! 1. **`crate::q8::Q8`/`Lin8` INSTANCE inspection is banned outright**,
//!    everywhere in `crates/qwen3/src/*.rs`: `self.q8`, `self.w8`,
//!    `self.head8`, `Option<crate::q8::Q8>`, `crate::q8::Lin8`, a `.mm8(`/
//!    `.quant(` call on a `q8`-named receiver, or a direct `Q8::build(`
//!    call. `crate::q8::Q8::LINEARS`/`crate::q8::Q8::is_i8_linear(...)` -
//!    STATIC utility functions/consts on the type, never an instance - are
//!    explicitly NOT banned (`model.rs`/`serve.rs` both still call them, to
//!    avoid duplicating the 7-leaf-name list): the ban list below is
//!    written to match the exact `Option<...>`/`.method(` shapes an
//!    INSTANCE use takes, which a bare `Type::assoc_fn(...)` call never
//!    matches.
//! 2. **Hand-picked GEMM kernel names/`KernelVariant` matches are banned
//!    inside the migrated forward-inference FUNCTION BODIES specifically**:
//!    `qwen3::model::Qwen::{forward_steps,decode_steps}` and
//!    `qwen3::serve::Engine::{batched_tape,head_steps}`. These four
//!    functions are where the fp32-vs-int8 fork used to live; after B7 they
//!    call `Ops::act`/`Ops::matmul` (model.rs) or `Self::quant_once`/
//!    `Self::linear` (serve.rs) exclusively for the 7 per-layer linears.
//!    (M6.3 split `serve.rs`'s per-layer dispatch out of the old
//!    `run_batched_steps` into `batched_tape` - a pure tape builder the new
//!    decode tape cache in `run_batched_submit` can record once per `bsz`
//!    bucket and replay - so this check now targets `batched_tape`, the
//!    function that actually owns the per-layer linear dispatch;
//!    `run_batched_steps` itself is now a thin `write_batch_meta`/
//!    `write_batch_input`/`batched_tape` wrapper with no kernel choice of
//!    its own.)
//!    Attention/RoPE/RMSNorm/paged-KV-cache/embedding kernel names
//!    (`EMBED`, `ROPE_PAGED`, `ROPE_AT`, `KV_APPEND*`, `SCORES_*`,
//!    `APPLY_*`, `ADD2`, `BIAS_*`, `SPLICE*`, ...) still appear directly in
//!    these bodies and are NOT banned - they were never part of the
//!    fp32-vs-int8 fork this phase migrated (Ops covers `matmul` dispatch
//!    only, per B3's own scope), so they are legitimately still manual, the
//!    same category as every other model in this codebase's LM-
//!    head/embedding dispatch.
//! 3. **`serve.rs`'s OWN tuned int8/fp32 GEMM dispatch is a documented,
//!    deliberate exception, NOT migrated onto `Ops::matmul`.** `Ops::
//!    matmul` always resolves its kernel through a FIXED internal
//!    `CachedSelector<DefaultSelector>` with no way for a caller to inject a
//!    different one; `qwen3::serve::Engine` has a REAL, per-device MEASURED
//!    selector (`tuned_i8`/`AutoTuner`/`FileTuneStore`, built by
//!    `Engine::tune_i8` at construction) that the qwen-serving-perf-gate
//!    this phase must not regress directly exercises. Migrating `Engine`'s
//!    dispatch onto `Ops::matmul` would silently discard that measured
//!    selector. `Engine::{mm,mm_into,gemm_tier,quant_once,linear,
//!    mm8,tune_i8,measure_i8,rms}` (one contiguous block, bracketed by the
//!    `qwen3-serve-manual-gemm-dispatch` marker comments in `serve.rs`
//!    itself) are the allow-listed region check 2 exempts by construction
//!    (they are NOT `run_batched_steps`/`head_steps`) - this test also
//!    confirms the marker comments are still present and still bracket
//!    every remaining `KernelVariant`/int8-kernel-name reference in the
//!    file, so the allow-list cannot silently grow by accretion.

use std::fs;

const MODEL_RS: &str = "src/model.rs";
const SERVE_RS: &str = "src/serve.rs";

fn read(path: &str) -> String {
    let full = format!("{}/{path}", env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(&full).unwrap_or_else(|e| panic!("no_kernel_names: cannot read {full}: {e}"))
}

/// Every `.rs` file directly under `crates/qwen3/src/` (non-recursive - this
/// crate has no submodule directories), for the crate-wide instance-
/// inspection ban (check 1).
fn all_src_files() -> Vec<(String, String)> {
    let dir = format!("{}/src", env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).unwrap_or_else(|e| panic!("no_kernel_names: cannot list {dir}: {e}")) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let rel = format!("src/{}", path.file_name().unwrap().to_str().unwrap());
            let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("no_kernel_names: cannot read {path:?}: {e}"));
            out.push((rel, text));
        }
    }
    assert!(out.len() >= 10, "no_kernel_names: found suspiciously few qwen3/src/*.rs files ({}) - did the crate layout change?", out.len());
    out
}

/// Extract the body of `fn NAME(` (brace-balanced, starting at the `{` right
/// after the signature) from `src` - used to scope check 2 to exactly the
/// four migrated functions, not the whole file.
fn function_body<'a>(src: &'a str, name: &str) -> &'a str {
    let sig = format!("fn {name}(");
    let sig_at = src.find(&sig).unwrap_or_else(|| panic!("no_kernel_names: fn {name} not found - did it get renamed?"));
    let open = src[sig_at..].find('{').map(|i| sig_at + i).unwrap_or_else(|| panic!("no_kernel_names: fn {name} has no body"));
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[open..=i];
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("no_kernel_names: fn {name}'s body never balances its braces");
}

/// Check 1: no `crate::q8::Q8`/`Lin8` INSTANCE is ever inspected in this
/// crate's own source - the storage side of the fp32-vs-int8 fork this phase
/// retired. Static-utility calls (`Q8::LINEARS`, `Q8::is_i8_linear`) are
/// deliberately not in this list - see the module doc's point 1.
#[test]
fn q8_instances_are_never_inspected_anywhere_in_the_crate() {
    let banned = [
        "self.q8",
        "self.w8",
        "self.head8",
        "Option<crate::q8::Q8>",
        "crate::q8::Lin8",
        "q8.mm8(",
        "q8.quant(",
        "Q8::build(",
    ];
    for (path, text) in all_src_files() {
        for needle in banned {
            assert!(
                !text.contains(needle),
                "no_kernel_names: {path} still contains {needle:?} - B7 retired every `crate::q8` INSTANCE \
                 (weight storage/dispatch now goes through `model::ops::Weight`/a plain `HashMap`); a static \
                 utility call like `Q8::LINEARS`/`Q8::is_i8_linear` would NOT trip this check, so this is a \
                 real regression, not a false positive"
            );
        }
    }
}

/// Check 2: the four migrated forward-inference functions dispatch the 7
/// per-layer linears with no hand-picked GEMM kernel name and no
/// `KernelVariant` match of their own - only through `Ops::act`/`Ops::
/// matmul` (model.rs) or `Self::quant_once`/`Self::linear` (serve.rs).
#[test]
fn migrated_forward_paths_never_hand_pick_a_gemm_kernel() {
    // Names that used to be dispatched directly inside the fork sites this
    // phase deleted. Deliberately NOT `MATMUL`/`MATMUL_REG3`/`MATMUL_TILE`
    // (model.rs's LM head, or serve.rs's split-K fold, legitimately still
    // hand-dispatch those - out of B7's scope, see the module doc) nor any
    // attention/RoPE/norm/embed/KV-cache kernel name.
    let banned_idents = ["MATMUL_I8_GEMV", "MATMUL_I8_DYN", "MATMUL_I8", "MAX_ABS_ROW", "QUANT_PACK", "MATMUL_GEMV", "MATMUL_REG2"];
    let check = |func_src: &str, owner: &str| {
        assert!(
            !func_src.contains("KernelVariant"),
            "no_kernel_names: {owner} matches on `KernelVariant` directly - the migrated linear dispatch must \
             go through `Ops`/`Self::linear`, not its own selector logic"
        );
        // NOTE: deliberately not a bare `q8` substring ban here - both
        // functions carry explanatory comments about what they REPLACED
        // (e.g. "the pre-B7 `q8.quant` call this replaces"), which check 1
        // (`q8_instances_are_never_inspected_anywhere_in_the_crate`) already
        // covers precisely for real code shapes, everywhere in the crate.
        for ident in banned_idents {
            assert!(
                !func_src.contains(ident),
                "no_kernel_names: {owner} still references {ident:?} directly - this kernel choice must be made \
                 by `Ops`/`Self::linear`, not hand-picked in the forward path"
            );
        }
    };

    let model_src = read(MODEL_RS);
    check(function_body(&model_src, "forward_steps"), "qwen3::model::Qwen::forward_steps");
    check(function_body(&model_src, "decode_steps"), "qwen3::model::Qwen::decode_steps");
    // Both migrated model.rs functions must actually call the façade -
    // otherwise a body that dispatches NOTHING for the 7 linears would
    // vacuously pass the bans above.
    for name in ["forward_steps", "decode_steps"] {
        let body = function_body(&model_src, name);
        assert!(
            body.contains("self.ops.act(") || body.contains("self.ops_act("),
            "no_kernel_names: {name} never reaches `Ops` for its activation - did the migration get reverted?"
        );
        assert!(body.contains("self.ops_linear(") || body.contains("ops_linear("), "no_kernel_names: {name} never calls `ops_linear`/`Ops::matmul`");
    }

    // `ops_act` is allowed to stand in for `Ops::act` above only because it is
    // itself a thin choice BETWEEN two façade entry points. Check that here, so
    // the widened assertion cannot be satisfied by a helper that hand-rolls the
    // activation packing the façade exists to own.
    let helper = function_body(&model_src, "ops_act");
    check(helper, "qwen3::model::Qwen::ops_act");
    assert!(
        helper.contains("self.ops.act(") && helper.contains("self.ops.act_f32("),
        "no_kernel_names: `ops_act` must choose between the two `Ops` activation entry points, not build one itself"
    );

    let serve_src = read(SERVE_RS);
    check(function_body(&serve_src, "batched_tape"), "qwen3::serve::Engine::batched_tape");
    check(function_body(&serve_src, "head_steps"), "qwen3::serve::Engine::head_steps");
    for name in ["batched_tape", "head_steps"] {
        let body = function_body(&serve_src, name);
        assert!(body.contains("self.quant_once("), "no_kernel_names: {name} never calls `Self::quant_once` - did the migration get reverted?");
        assert!(body.contains("self.linear("), "no_kernel_names: {name} never calls `Self::linear`");
    }
}

/// Check 3: `serve.rs`'s allow-listed manual-tuned-dispatch region (point 3
/// of the module doc) is still marked and still contiguous - so the
/// allow-list this test's own doc comment describes cannot silently grow by
/// someone moving code around it without updating the markers (and, by
/// implication, this test's own doc comment).
///
/// Scoped to the two GEMM-relevant `KernelVariant` arms specifically
/// (`WorkgroupPerOutput`/`PackedInt8` - the GEMV-vs-tile choice this engine's
/// `tune_i8` measures), NOT `KernelVariant::SplitReduction` - that variant's
/// two OTHER call sites in this file (`submit_topk_head`/`greedy_from_hidden`
/// area) pick the on-device argmax/top-k reduction strategy for `Op::
/// ArgMaxRow`, a completely different, always-manual, out-of-B7's-scope
/// kernel category (never part of the fp32-vs-int8 linear fork this phase
/// migrated) - banning it here would be a false positive, not a real gap.
#[test]
fn serve_manual_gemm_dispatch_region_is_still_marked_and_contains_every_remaining_gemm_kernel_variant_reference() {
    let serve_src = read(SERVE_RS);
    const BEGIN: &str = "qwen3-serve-manual-gemm-dispatch BEGIN";
    const END: &str = "qwen3-serve-manual-gemm-dispatch END";
    let b = serve_src.find(BEGIN).unwrap_or_else(|| panic!("no_kernel_names: {SERVE_RS} is missing the {BEGIN:?} marker"));
    let e = serve_src.find(END).unwrap_or_else(|| panic!("no_kernel_names: {SERVE_RS} is missing the {END:?} marker"));
    assert!(b < e, "no_kernel_names: {SERVE_RS}'s BEGIN marker must precede its END marker");
    let inside = &serve_src[b..e];
    let before = &serve_src[..b];
    let after = &serve_src[e..];
    let gemm_variants = ["KernelVariant::WorkgroupPerOutput", "KernelVariant::PackedInt8"];
    for v in gemm_variants {
        let outside_hits = before.matches(v).count() + after.matches(v).count();
        assert_eq!(
            outside_hits, 0,
            "no_kernel_names: {SERVE_RS} references `{v}` outside the marked manual-dispatch region - either \
             move it inside the `qwen3-serve-manual-gemm-dispatch` markers (if it's part of that documented \
             exception) or route it through `Ops`/`Self::linear`"
        );
        assert!(inside.contains(v), "no_kernel_names: the marked region no longer contains `{v}` - is the marker still needed?");
    }
}
