// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! B8's negative test: proves the "leave it `@dtype f32`" decision for
//! norms/attention/most elementwise kernels was a deliberate judgment call,
//! not an oversight or an unenforced convention.
//!
//! Two DIFFERENT reasons show up in the program's own scoping, and this file
//! exercises both against the REAL `kernels::template::dtype_variant`
//! mechanism (not a reimplementation of `scripts/build/kernelmeta.py`'s
//! Python check - that script validates the SAME hard precondition this
//! module enforces at the Rust level, see `kernelmeta.dtype_errors`'s own
//! doc comment: "the templater's hard precondition... checked here in Python
//! ahead of any compile"):
//!
//! * **`rmsnorm`/`layernorm`** (norms): their gain/bias vector IS mechanically
//!   eligible - a bare-identifier-indexed `array<f32>` binding, exactly what
//!   `dtype_variant` requires - so leaving it `f32` is NOT "the templater
//!   refused it", it is a genuine, separate numerical/VRAM judgment call
//!   (a tiny `[d_model]` vector, VRAM-irrelevant, and a bf16/f16-narrowed
//!   gain read back into an f32 reduction is a correctness regression, not an
//!   optimisation).
//! * **`paged_decode_scores`** (attention/KV): its float storage bindings
//!   (`q`, `pool_k`) are ACTIVATIONS - the query row and the paged KV-cache
//!   pool - not a static per-model weight matrix at all, so a bf16/f16
//!   KV-cache is a real, separate, LATER program phase (B9), not something
//!   this phase's `@dtype` grammar (which asks "is there a large weight
//!   tensor worth a storage tier") was ever meant to answer for this kernel.
//!   As it happens `q`'s own index (`q[qb + d]`, a per-head/per-dim compound
//!   offset into the query row) is ALSO mechanically ineligible - `dtype_variant`
//!   refuses it outright, the same "not a bare identifier" error a real
//!   B4/B5 candidate would have needed a `let wi = ...;` hoist to clear. So
//!   this kernel is doubly not a candidate: wrong CATEGORY (activation, not
//!   weight) and, independently, not mechanically eligible today either - the
//!   test below checks the second fact directly (the templater's own hard
//!   precondition), since the category judgment is not something code can
//!   check on its own.

use backend_api::DType;

/// `@dtype` field text, verbatim, off a kernel's real embedded source - the
/// exact line `scripts/build/kernelmeta.py`'s own `_self_check` scans.
fn dtype_line(src: &str) -> &str {
    src.lines()
        .find(|l| l.trim_start().starts_with("// @dtype"))
        .unwrap_or_else(|| panic!("kernel source has no `// @dtype` header line"))
        .trim()
}

/// The three representative kernels named in this phase's own task brief -
/// one norm, one more norm (a different reduction shape), one attention/KV
/// kernel - all still declare `@dtype f32` after B8, unchanged from B6's
/// mechanical seeding.
#[test]
fn representative_norm_and_attention_kernels_still_declare_dtype_f32() {
    for (name, src) in [
        ("rmsnorm", kernels::RMSNORM),
        ("layernorm", kernels::LAYERNORM),
        ("paged_decode_scores", kernels::PAGED_DECODE_SCORES),
    ] {
        assert_eq!(dtype_line(src), "// @dtype f32", "{name}: expected @dtype f32");
    }
}

/// `rmsnorm.wgsl`'s `weight[c]` binding is genuinely, mechanically eligible
/// for `dtype_variant` (a real `array<f32>` binding, indexed only by the bare
/// identifier `c`) - proving the "stay f32" decision is a considered
/// numerical/VRAM judgment call about a gain vector, not a limitation of the
/// templater. If this assertion ever started failing (i.e. `dtype_variant`
/// started REFUSING `rmsnorm`'s `weight` binding), that would mean the
/// kernel's source changed shape under this test, not that the decision
/// documented above became wrong for a different reason.
#[test]
fn rmsnorm_gain_vector_is_mechanically_eligible_despite_staying_f32() {
    let (name, src) = kernels::template::dtype_variant("rmsnorm", kernels::RMSNORM, "weight", DType::BF16)
        .expect("rmsnorm's `weight[c]` binding must be mechanically templatable (bare-identifier index) - this is exactly why leaving it @dtype f32 is a deliberate judgment call about a tiny gain vector, not a mechanical refusal");
    assert_eq!(name, "rmsnorm#weight=bf16");
    assert!(src.contains("array<u32>"), "{src}");
}

/// `paged_decode_scores.wgsl`'s `q` binding is a per-request ACTIVATION (the
/// query row), not a static weight matrix that exists once per model - the
/// primary, CATEGORY reason this kernel is out of B8's scope (a real
/// per-request KV-cache dtype tier is the separate, deliberately deferred B9).
/// That category judgment is not something code can check on its own -
/// `dtype_variant` has no concept of "this f32 buffer is a KV-cache, not a
/// weight", it only sees a storage binding's declaration and load sites - but
/// as it happens `q[qb + d]`'s compound index is ALSO mechanically refused by
/// the templater's own hard precondition, exactly the "not a bare identifier"
/// error a real B4/B5 candidate needed a `let wi = ...;` hoist to clear. This
/// test pins that second, checkable fact: even setting the category judgment
/// aside, this kernel is not silently "one hoist away" from being a candidate
/// today.
#[test]
fn paged_decode_scores_q_binding_is_not_mechanically_eligible_either() {
    let err =
        kernels::template::dtype_variant("paged_decode_scores", kernels::PAGED_DECODE_SCORES, "q", DType::BF16)
            .unwrap_err();
    assert!(err.contains("bare identifier"), "{err}");
    assert!(err.contains("qb + d"), "{err}");
}
