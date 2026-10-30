// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Ops`/`Weight` - a model-facing façade over `backend_api::select`'s unified
//! kernel selector, `model::int8`/`model::int4`'s host quantizers, and
//! `model::dispatch`'s `I8Scratch`.
//!
//! **The problem this exists to fix.** Today a model like `flux1`/`flux2`
//! hand-numbers its own kernel-pipeline indices (`("matmul", kernels::MATMUL)`
//! at a fixed slot), hand-maintains a `LinW::{F32, I8}` enum per linear, and
//! forks `if let LinW::I8(wq, sw) = w { self.mm8(...) } else { self.mm_rows_at(...) }`
//! at every call site (`crates/flux1/src/model.rs`, `crates/flux2/src/model.rs`).
//! `Ops`/`Weight` collapse that into one call:
//! `ops.matmul(&mut s, &weight, &act, &y, yoff)` - the weight's own dtype
//! carries which kernel family it needs, and the kernel NAME is resolved
//! once, at construction, via `Gpu::kernel_index`, rather than re-derived (or
//! silently drifting) at every call site.
//!
//! **Scope of B3/B4/B5.** B3 built the façade and proved it reproduces
//! today's `dispatch.rs`/`int8.rs`/`int4.rs` numeric behaviour exactly
//! (`crates/model/tests/ops_facade_parity.rs`) for `F32`, `I8` (DP4A,
//! `model::int8`), `Q4` (W4A8, `model::int4`). B4 added the fourth tier,
//! `BF16` (`model::half::pack_bf16` + the `#w=bf16` kernel variants
//! `kernels::template::dtype_variant` produces - see `crates/model/tests/
//! bf16_roundtrip.rs`). B5 adds the fifth, `F16` (`model::half::pack_f16` +
//! the `#w=f16` kernel variants, magic-multiply decode - see
//! `crates/model/tests/f16_roundtrip.rs`). **No model crate's call sites were
//! migrated by any of these phases** - `crates/qwen3`, `crates/flux1`,
//! `crates/flux2` etc. are untouched; that migration is a later phase (B7).
//!
//! **B10 - bf16 training tier.** [`Ops::matmul_dx`]/[`Ops::matmul_dw`] extend
//! ONE kernel family (`matmul.wgsl`'s `Reference` variant + its `matmul_dx.
//! wgsl` backward) so the SAME `Weight::BF16` a caller already builds for
//! forward (B4) can also drive the backward-of-x. `matmul_dw` (gradient
//! w.r.t. the weight) is deliberately NOT extended - it has no `Weight`
//! parameter at all, since `matmul_dw.wgsl` never reads the weight buffer,
//! only `dy`/`x` (both always f32). This is the standard mixed-precision-
//! training split: the weight is READ at reduced precision in forward and in
//! the x-backward, but the WEIGHT'S OWN gradient is always computed/
//! accumulated in f32 and feeds an f32 AdamW step over an f32 master copy
//! (`crates/paramstore`/`crates/optim`, untouched by this phase). Gated
//! behind an explicit `Weight::BF16` exactly like every other bf16 tier in
//! this façade - a caller gets `Weight::F32` (today's existing behaviour)
//! unless it explicitly asks `Weight::upload` for `Dtype::BF16`, so nothing
//! existing changes without an opt-in. Gradient-checked by `gradcheck::
//! check_matmul_bf16_weight`. **Not wired into any model crate's training
//! loop** - same "build the façade, prove it, migrate later" precedent B3/
//! B4/B5/B8/B9 all followed.
//!
//! **Selection.** `Ops::matmul` is the one place a shape turns into a kernel
//! choice: it builds a `select::OpShape` from the weight's own `(n, k)` and
//! the call's `m`, asks `self.selector` (an `Arc<dyn KernelSelector>` -
//! `CachedSelector<DefaultSelector>` by default, the same static policy every
//! other selector consumer in this workspace starts from, but injectable via
//! [`Ops::with_selector`] - see that constructor's doc comment for why a
//! caller with a real per-device MEASURED policy, e.g. `qwen3::serve::
//! Engine`'s tuned int8 GEMV/tile choice, needs this seam instead of building
//! its own dispatch outside `Ops` entirely), and `Self::bind` maps
//! `(KernelVariant, Dtype)` to
//! the ONE kernel-name spelling this façade recognizes for that pair - the
//! only place a kernel-name string literal appears outside `kname`'s own
//! const definitions. Unlike `model::block::gemm_variant` (which lets each
//! model register whichever of `matmul_reg`/`matmul_reg2`/`matmul_reg3` it
//! likes, since the model owns dispatch), `Ops` owns kernel-name resolution
//! itself, so it fixes ONE canonical name per `KernelVariant` - a model with
//! a differently-named but bit-identical physical kernel simply registers it
//! under that canonical name when it builds its `Gpu`.
//!
//! **Offset arithmetic - the specific bug class this module must not
//! introduce.** `model::int8`'s own module doc already flags this: a byte/word
//! offset for a sliced dispatch is silently-wrong arithmetic when it drifts,
//! not a crash. The one subtlety worth spelling out here: **the packed
//! activation is ALWAYS int8**, never int4, even for a `Weight::Q4` linear -
//! q4 is W4A8 (`int4`'s own module doc), so only the WEIGHT narrows further;
//! the activation quantizer, its packing, and its offset math are identical
//! to the `I8` tier. `Ops::matmul`'s row-offset math for the packed activation
//! therefore always divides by `Dtype::I8.per_word()` (4), NEVER by
//! `w.dtype().per_word()` - using the weight's own `per_word()` there would
//! silently divide a `Q4` linear's activation offset by 8 instead of 4.
//!
//! **Activation quantization - `Ops::act`/`Act`.** A caller quantizes an
//! activation slice ONCE via `Ops::act`, then passes the resulting `Act` to
//! every `matmul` call that reads it - the "quantize `xn1` once, share across
//! q/k/v" invariant `qwen3::model` already relies on by hand. `Act` wraps a
//! fresh `model::dispatch::I8Scratch` (reused, not reimplemented) sized for
//! exactly this call's `[0, xr0+rows)` row range; every `matmul` call against
//! it - regardless of whether the weight it is paired with turns out to be
//! `F32` (which ignores the quantized form and reads `Act`'s raw buffer) or
//! `I8`/`Q4` (which reads the quantized form) - reuses the SAME quantization,
//! never re-dispatching `max_abs_row`/`quant_pack`. `Ops::act` always
//! quantizes eagerly (matching the `s: &mut Vec<Step>` parameter the phase
//! brief specifies): a call site that never pairs an activation with a
//! quantized weight pays for a quantization it does not use. A real model
//! call site (B7) already knows its own precision tier statically before it
//! ever calls `act`, so in practice this cost is never paid by a pure-fp32
//! forward; making it lazy/cached instead (only quantize on the first `I8`/
//! `Q4` `matmul` call against a given `Act`) is a reasonable follow-up this
//! phase deliberately leaves for whoever needs it.

use std::collections::HashMap;
use std::sync::Arc;

/// Concatenate `parts` end to end into one freshly allocated buffer, copying
/// each part in parallel chunks rather than one `extend_from_slice` call per
/// part.
///
/// A fresh multi-hundred-MB-to-GB allocation is virtual until touched, and a
/// serial `Vec::extend_from_slice` loop is exactly one thread doing that
/// first touch - measured as the dominant cost of building qwen3's fused
/// QKV/gate-up weights at real model scale (M4.1's row-concat-then-quantize
/// shape), independent of how little CPU work the copy itself needs.
/// Faulting (and copying) each part's range in parallel keeps that cost at
/// the pool's, not one core's - the same fix, and the same reason, as
/// `checkpoint::mmap`'s F32 tensor decode.
pub fn concat_rows_parallel(parts: &[&[f32]]) -> Vec<f32> {
    const CHUNK: usize = 1 << 16;
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = vec![0f32; total];
    let mut offset = 0usize;
    for p in parts {
        let len = p.len();
        backend_cpu::par::chunks_mut(&mut out[offset..offset + len], CHUNK, |c, dst| {
            let s = c * CHUNK;
            dst.copy_from_slice(&p[s..s + dst.len()]);
        });
        offset += len;
    }
    out
}

use gpu_core::provider::library::{self, GemmPrecision, LibraryGemmProvider};
use gpu_core::provider::{LowerCtx, Operand, OpRequest, Pass, ProviderRegistry, Role};
use gpu_core::select::{self, Dtype, KernelSelector, KernelVariant, Op, OpShape};
use gpu_core::{DeviceBuffer, DeviceCaps, Fp8Act, Gpu, Step};

use crate::dispatch::I8Scratch;

/// The [`Operand::range`] meaning "bind the whole buffer" - `Gpu::step_sliced`
/// reads `(0, 0)` as the entire binding, which is exactly what an unsliced
/// `Gpu::step` does.
///
/// Named, because the four façade methods that carry it were dispatching
/// through plain `Gpu::step` before they moved onto the `OperatorProvider`
/// seam, and a *computed* extent there would have been a new claim rather
/// than a move: a caller is free to hand any of them a buffer larger than
/// the logical tensor (a scratch arena, a slab shared with the next layer),
/// and a binding sized to `m * n` would have started refusing those.
const WHOLE: (u64, u64) = (0, 0);

/// The exact kernel-name spellings this façade dispatches by. Defined ONCE so
/// [`Ops::new`]'s registration check and [`Ops::bind`]'s (variant, dtype) →
/// name table can never spell one differently from the other.
mod kname {
    pub const MATMUL: &str = "matmul";
    pub const MATMUL_GEMV: &str = "matmul_gemv";
    pub const MATMUL_REG2: &str = "matmul_reg2";
    pub const MATMUL_I8_DYN: &str = "matmul_i8_dyn";
    pub const MATMUL_I8_GEMV: &str = "matmul_i8_gemv";
    pub const MATMUL_Q4_DYN: &str = "matmul_q4_dyn";
    /// M5.5's 128x128 register-tiled Q4 GEMM (`matmul_q4_dyn.wgsl`'s naive
    /// one-thread-per-output sibling, mirroring `matmul_i8_dyn`'s own tiled
    /// shape) - bound by [`Ops::bind`]'s `(PackedInt8, Dtype::Q4)` arm in
    /// place of [`MATMUL_Q4_DYN`] since the kernel-performance ledger's M5.5
    /// entry closed it as "a clean win at every measured shape" (the gain
    /// growing with m) and bit-identical to it.
    pub const MATMUL_Q4_DYN_REG: &str = "matmul_q4_dyn_reg";
    pub const MATMUL_Q4_GEMV: &str = "matmul_q4_gemv";
    pub const MAX_ABS_ROW: &str = "max_abs_row";
    pub const QUANT_PACK: &str = "quant_pack";
    /// M9's affine K-quant correction prepass (`S[m,g] = Σ_{k in g} xq[m,k]`,
    /// `crate::kquant`'s module doc) - the third, OPTIONAL step
    /// `crate::int8::quant_rows_steps`'s `xgs` field appends.
    pub const QUANT_GROUP_SUM: &str = "quant_group_sum";
    /// M11's affine K-quant (Q4_K/Q5_K) kernels, `CODE_BITS`-specialized via
    /// `kernels::template::interned` - see [`Ops::bind`]'s doc comment for
    /// why `Dtype::Q4K`/`Dtype::Q8K` each need their own physical kernel
    /// name rather than a runtime template parameter. Spelled by hand to
    /// match `kernels::template::variant_name`'s `"{base}#{k}={v}"`
    /// convention - `required_kernels_matches_kname_all` (this module's own
    /// test module) pins that these never drift from what `interned` really
    /// produces, the same discipline `MATMUL_BF16`'s doc comment describes
    /// for the B4/B5 storage tiers.
    pub const MATMUL_KQ_DYN_4: &str = "matmul_kq_dyn#CODE_BITS=4";
    pub const MATMUL_KQ_DYN_8: &str = "matmul_kq_dyn#CODE_BITS=8";
    pub const MATMUL_KQ_GEMV_4: &str = "matmul_kq_gemv#CODE_BITS=4";
    pub const MATMUL_KQ_GEMV_8: &str = "matmul_kq_gemv#CODE_BITS=8";
    /// M10's group=16 (Q6_K) reuse of the EXISTING symmetric int8 kernels via
    /// a template knob (`QPG`/`WPG`) - a DIFFERENT physical pipeline from the
    /// plain [`MATMUL_I8_DYN`]/[`MATMUL_I8_GEMV`] names above (which stay at
    /// their group=32 default), so it needs its own registered name; see
    /// [`Ops::bind`]'s `group` parameter for how a caller picks between them.
    pub const MATMUL_I8_DYN_G16: &str = "matmul_i8_dyn#QPG=1";
    pub const MATMUL_I8_GEMV_G16: &str = "matmul_i8_gemv#WPG=4";
    /// bf16-storage kernel names (B4) - `kernels::template::dtype_variant`'s
    /// own `variant_name` convention (`"{base}#{binding}={tag}"`), spelled
    /// here as plain literals (not computed via a call to `dtype_variant`
    /// itself) so this module stays a flat, `const`-evaluable name table like
    /// every other entry in this list. `ops.rs`'s own test module has a
    /// dedicated regression test (`bf16_kname_literals_match_dtype_variant_
    /// naming`) pinning that these three literals never drift from what
    /// `dtype_variant` actually produces for the real kernel sources - the
    /// ONLY place that link is checked, since this table cannot call the
    /// templater itself without becoming non-`const`.
    pub const MATMUL_BF16: &str = "matmul#w=bf16";
    pub const MATMUL_GEMV_BF16: &str = "matmul_gemv#w=bf16";
    /// Register-tiled bf16: `matmul_reg3.wgsl`, NOT `matmul_reg2.wgsl` -
    /// `MATMUL_REG2` stays F32's untouched existing default; `matmul_reg3` is
    /// the kernel this phase templatized, since its own header already
    /// describes it as `matmul_reg2`'s tiling with its two shared-memory
    /// bank-conflict patterns removed - the natural pick for a second
    /// physical kernel per dtype. Same `RegisterTiled` `KernelVariant`, two
    /// physically different kernel FILES per dtype - already precedented by
    /// `PackedInt8`'s `matmul_i8_dyn` vs `matmul_q4_dyn` split (see
    /// `Ops::threads`'s doc comment).
    pub const MATMUL_REG3_BF16: &str = "matmul_reg3#w=bf16";
    /// f16-storage kernel names (B5) - same naming convention as the bf16
    /// trio above, `#w=f16` instead of `#w=bf16`. `bf16_kname_literals_
    /// match_dtype_variant_naming` covers these too (renamed to `_and_f16`).
    pub const MATMUL_F16: &str = "matmul#w=f16";
    pub const MATMUL_GEMV_F16: &str = "matmul_gemv#w=f16";
    /// Register-tiled f16: `matmul_reg3.wgsl`, same physical file the bf16
    /// tier reuses - see [`MATMUL_REG3_BF16`]'s own doc comment for why a
    /// second physical file for a `RegisterTiled` tier is precedented.
    pub const MATMUL_REG3_F16: &str = "matmul_reg3#w=f16";

    // --- B8: embed / moe_linear_gated storage tiers -------------------------
    pub const EMBED: &str = "embed";
    pub const EMBED_BF16: &str = "embed#emb=bf16";
    pub const EMBED_F16: &str = "embed#emb=f16";
    /// The embedding gather over one vocab tile (`embed_tile.wgsl`), for a table larger than one binding.
    pub const EMBED_TILE: &str = "embed_tile";
    pub const EMBED_TILE_BF16: &str = "embed_tile#emb=bf16";
    pub const EMBED_TILE_F16: &str = "embed_tile#emb=f16";
    pub const MOE_LINEAR_GATED: &str = "moe_linear_gated";
    pub const MOE_LINEAR_GATED_BF16: &str = "moe_linear_gated#w=bf16";
    pub const MOE_LINEAR_GATED_F16: &str = "moe_linear_gated#w=f16";

    // --- B9: paged-KV-cache append/scores/apply bf16 storage tier -----------
    // `#pool=bf16` (append, WRITE direction - `kernels::template::
    // dtype_variant_store`) and `#pool_k=bf16`/`#pool_v=bf16` (scores/apply,
    // READ direction - `kernels::template::dtype_variant`, same naming
    // convention as every other tier in this table).
    pub const PAGED_KV_APPEND_BATCHED: &str = "paged_kv_append_batched";
    /// The BF16 append tier binds to a DIFFERENT physical kernel
    /// (`paged_kv_append_batched_word.wgsl`, one thread per even-sized chunk
    /// of a token with a serial inner loop), not a `dtype_variant_store` rewrite of the plain
    /// per-element `paged_kv_append_batched.wgsl` above - see that word-
    /// granularity kernel's own doc comment for the real race a per-element
    /// bf16 pack dispatch has (caught by this crate's own dual-backend test:
    /// green on the CPU JIT, red on real wgpu).
    pub const PAGED_KV_APPEND_BATCHED_WORD_BF16: &str = "paged_kv_append_batched_word#pool=bf16";
    pub const PAGED_DECODE_SCORES_BATCHED: &str = "paged_decode_scores_batched";
    pub const PAGED_DECODE_SCORES_BATCHED_BF16: &str = "paged_decode_scores_batched#pool_k=bf16";
    pub const PAGED_DECODE_APPLY_BATCHED: &str = "paged_decode_apply_batched";
    pub const PAGED_DECODE_APPLY_BATCHED_BF16: &str = "paged_decode_apply_batched#pool_v=bf16";

    // --- B10: bf16 training tier - matmul_dx (weight-READ, gated) / matmul_dw
    // (never touches the weight at all, so it has no BF16 name to bind) -----
    pub const MATMUL_DX: &str = "matmul_dx";
    /// `matmul_dx.wgsl`'s bf16-weight-read variant - the SAME `dtype_variant`
    /// mechanism `MATMUL_BF16` (forward) already uses, applied to the
    /// backward-of-x kernel this phase templatized. There is no
    /// `MATMUL_DX_F16`/`_I8`/`_Q4`: B10's own scope is deliberately narrowed
    /// to ONE kernel family, forward + dx, bf16 only (see this module's B10
    /// doc section).
    pub const MATMUL_DX_BF16: &str = "matmul_dx#w=bf16";
    /// `matmul_dw.wgsl` NEVER reads the weight buffer (its only inputs are
    /// `dy`/`x`, both always-f32 activations - see that kernel's own
    /// source) - so unlike every other name in this table, there is no
    /// per-dtype variant here AT ALL. This is B10's structural enforcement
    /// of "dW's accumulation is ALWAYS f32, regardless of the weight's
    /// storage tier": the ONE physical kernel this name binds to has no
    /// dtype knob to get wrong.
    pub const MATMUL_DW: &str = "matmul_dw";

    // --- frozen-base training tier: the tiled f32/bf16 input gradient ---
    //
    // NOT in `ALL`: a model that never trains over a bf16 base does not have to register them. They are looked
    // up when first dispatched and a missing one is a named panic - register `training_kernel_list()`.

    /// The plain f32 `matmul_dx_reg.wgsl`: the tiled input gradient of the f32
    /// tier (registered under this name by models that have it).
    pub const MATMUL_DX_REG: &str = "matmul_dx_reg";
    /// `matmul_dx_reg.wgsl` reading a bf16 weight: the tiled input gradient of
    /// the bf16 tier (`MATMUL_DX_BF16` is its naive twin).
    pub const MATMUL_DX_REG_BF16: &str = "matmul_dx_reg#w=bf16";

    pub const ALL: &[&str] = &[
        MATMUL,
        MATMUL_GEMV,
        MATMUL_REG2,
        MATMUL_I8_DYN,
        MATMUL_I8_GEMV,
        MATMUL_Q4_DYN,
        MATMUL_Q4_DYN_REG,
        MATMUL_Q4_GEMV,
        MAX_ABS_ROW,
        QUANT_PACK,
        QUANT_GROUP_SUM,
        MATMUL_KQ_DYN_4,
        MATMUL_KQ_DYN_8,
        MATMUL_KQ_GEMV_4,
        MATMUL_KQ_GEMV_8,
        MATMUL_I8_DYN_G16,
        MATMUL_I8_GEMV_G16,
        MATMUL_BF16,
        MATMUL_GEMV_BF16,
        MATMUL_REG3_BF16,
        MATMUL_F16,
        MATMUL_GEMV_F16,
        MATMUL_REG3_F16,
        EMBED,
        EMBED_BF16,
        EMBED_F16,
        EMBED_TILE,
        EMBED_TILE_BF16,
        EMBED_TILE_F16,
        MOE_LINEAR_GATED,
        MOE_LINEAR_GATED_BF16,
        MOE_LINEAR_GATED_F16,
        PAGED_KV_APPEND_BATCHED,
        PAGED_KV_APPEND_BATCHED_WORD_BF16,
        PAGED_DECODE_SCORES_BATCHED,
        PAGED_DECODE_SCORES_BATCHED_BF16,
        PAGED_DECODE_APPLY_BATCHED,
        PAGED_DECODE_APPLY_BATCHED_BF16,
        MATMUL_DX,
        MATMUL_DX_BF16,
        MATMUL_DW,
    ];
}

/// The kernel-name spellings [`Ops::new`] requires the `Gpu` it is built from
/// to have registered - exposed so a caller building that `Gpu`'s kernel list
/// (or a test) has one source for the exact spellings, instead of retyping
/// them and risking a silent mismatch against [`Ops::new`]'s own check.
pub const REQUIRED_KERNELS: &[&str] = kname::ALL;

/// The canonical `(name, wgsl_source)` list [`Ops::new`] requires - the ONE
/// ready-made kernel list an `Ops`-building caller feeds `Gpu::new` /
/// `gpu_core::testgpu::dev`, instead of hand-assembling its own.
///
/// # Why this could not simply be a `const`
///
/// Fifteen of these entries (the bf16/f16 storage tiers, B9's paged-KV bf16
/// write tier, and M12's `CODE_BITS`/`QPG`/`WPG`-specialised K-quant kernels)
/// exist only as values computed at runtime by
/// [`kernels::template::dtype_variant`]/`dtype_variant_store`/`interned` - a
/// specialised variant's WGSL source is generated, not an `include_str!`.
/// So the list is built once and leaked into a `OnceLock`, exactly the
/// "tiny working set, leaking it is fine" tradeoff `dtype_variant`'s own
/// interning cache already makes, which is also what lets this return the
/// `&'static [(&'static str, &'static str)]` slice `Gpu::new` wants.
///
/// # Why it is public, and what it replaced
///
/// Every `Ops`-building call site used to hand-maintain a byte-identical
/// copy of this list (`qwen3::serve::ops_kernel_list`,
/// `gradcheck::bf16_train::kernel_list`, and four `crates/model/tests/*`
/// files). That duplication is exactly how `qwen3::serve::ops_kernel_list`
/// silently drifted **15 kernels short** of [`REQUIRED_KERNELS`] - see
/// [`assert_kernel_list_complete`] for the full post-mortem. One list,
/// derived in one place, is the fix; [`assert_kernel_list_complete`] stays
/// as the gate for any caller that still builds its own (a model that
/// registers extra kernels of its own beside the façade set).
#[must_use]
pub fn kernel_list() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        // `dtype_variant(logical_name, source, binding, dtype)` yields the
        // `("<name>#<binding>=<dtype>", generated_source)` pair; the names it
        // produces are exactly the `kname::*_BF16`/`*_F16` spellings
        // `Ops::new` looks up, which `required_kernels_matches_kname_all`
        // pins.
        let var = |name, src, binding, dt| {
            kernels::template::dtype_variant(name, src, binding, dt)
                .expect("brain-kernels ships every dtype variant Ops::new requires")
        };
        let bf16_matmul = var("matmul", kernels::MATMUL, "w", Dtype::BF16);
        let bf16_gemv = var("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16);
        let bf16_reg3 = var("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16);
        let f16_matmul = var("matmul", kernels::MATMUL, "w", Dtype::F16);
        let f16_gemv = var("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::F16);
        let f16_reg3 = var("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::F16);
        let bf16_embed = var("embed", kernels::EMBED, "emb", Dtype::BF16);
        let f16_embed = var("embed", kernels::EMBED, "emb", Dtype::F16);
        let bf16_embed_tile = var("embed_tile", kernels::EMBED_TILE, "emb", Dtype::BF16);
        let f16_embed_tile = var("embed_tile", kernels::EMBED_TILE, "emb", Dtype::F16);
        let bf16_moe = var("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::BF16);
        let f16_moe = var("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::F16);
        // B9: paged-KV append is the WRITE direction, so it needs
        // `dtype_variant_store` over the word-granularity sibling kernel -
        // see `kname::PAGED_KV_APPEND_BATCHED_WORD_BF16`'s own doc comment.
        let bf16_kv_append = kernels::template::dtype_variant_store(
            "paged_kv_append_batched_word",
            kernels::PAGED_KV_APPEND_BATCHED_WORD,
            "pool",
            Dtype::BF16,
        )
        .expect("brain-kernels ships the paged-KV bf16 store variant");
        let bf16_decode_scores = var(
            "paged_decode_scores_batched",
            kernels::PAGED_DECODE_SCORES_BATCHED,
            "pool_k",
            Dtype::BF16,
        );
        let bf16_decode_apply = var(
            "paged_decode_apply_batched",
            kernels::PAGED_DECODE_APPLY_BATCHED,
            "pool_v",
            Dtype::BF16,
        );
        // B10: `matmul_dx`'s bf16-weight-read backward variant. `matmul_dw`
        // has no bf16 variant at all (it never reads the weight - see
        // `kname::MATMUL_DW`'s own doc comment).
        let bf16_matmul_dx = var("matmul_dx", kernels::MATMUL_DX, "w", Dtype::BF16);
        // M12: affine K-quant (Q4_K/Q5_K) kernels at each `CODE_BITS`
        // specialization, and the group=16 (Q6_K) reuse of the existing
        // symmetric kernels via M10's `QPG`/`WPG` knobs - all four/two are
        // `kernels::template::interned` specialisations, not separate
        // `.wgsl` files, exactly like the bf16/f16 storage tiers above.
        let spec = |base, src, params: &[(&str, u32)]| {
            kernels::template::interned(base, src, params)
                .unwrap_or_else(|e| panic!("brain-kernels: {base}{params:?} failed to specialize: {e}"))
        };
        let kq_dyn_4 = spec("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", 4)]);
        let kq_dyn_8 = spec("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", 8)]);
        let kq_gemv_4 = spec("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", 4)]);
        let kq_gemv_8 = spec("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", 8)]);
        let i8_dyn_g16 = spec("matmul_i8_dyn", kernels::MATMUL_I8_DYN, &[("QPG", 1)]);
        let i8_gemv_g16 = spec("matmul_i8_gemv", kernels::MATMUL_I8_GEMV, &[("WPG", 4)]);
        vec![
            ("matmul", kernels::MATMUL),
            ("matmul_gemv", kernels::MATMUL_GEMV),
            ("matmul_reg2", kernels::MATMUL_REG2),
            ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
            ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
            ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
            ("matmul_q4_dyn_reg", kernels::MATMUL_Q4_DYN_REG),
            ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
            ("max_abs_row", kernels::MAX_ABS_ROW),
            ("quant_pack", kernels::QUANT_PACK),
            ("quant_group_sum", kernels::QUANT_GROUP_SUM),
            kq_dyn_4,
            kq_dyn_8,
            kq_gemv_4,
            kq_gemv_8,
            i8_dyn_g16,
            i8_gemv_g16,
            bf16_matmul,
            bf16_gemv,
            bf16_reg3,
            f16_matmul,
            f16_gemv,
            f16_reg3,
            ("embed", kernels::EMBED),
            bf16_embed,
            f16_embed,
            ("embed_tile", kernels::EMBED_TILE),
            bf16_embed_tile,
            f16_embed_tile,
            ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
            bf16_moe,
            f16_moe,
            ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
            bf16_kv_append,
            ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
            bf16_decode_scores,
            ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
            bf16_decode_apply,
            ("matmul_dx", kernels::MATMUL_DX),
            ("matmul_dw", kernels::MATMUL_DW),
            bf16_matmul_dx,
        ]
    })
}

/// The kernels a frozen-base trainer needs beyond the [`REQUIRED_KERNELS`]
/// facade set: the tiled bf16 input gradient and the kernels of the quantized-execution contract (the
/// `weight_decode` variant of every stored tier, `copy_cols`, `accum_bf16`, the embedding gradient). A model that builds its own kernel
/// list appends this slice; [`training_kernel_list`] is the ready-made union
/// with [`kernel_list`]. Kept out of [`REQUIRED_KERNELS`] on purpose: an `Ops`
/// that never trains over a quantized or bf16 base must not have to register them.
#[must_use]
pub fn frozen_base_kernels() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut list = vec![
            kernels::template::dtype_variant("matmul_dx_reg", kernels::MATMUL_DX_REG, "w", Dtype::BF16)
                .expect("brain-kernels ships the bf16 tiled dx variant"),
            // The quantized-execution contract (`tiled`): the tile decode of every tier, the column copy
            // that gathers and scatters tiles, and the bf16 gradient accumulation.
            ("copy_cols", kernels::COPY_COLS),
            ("accum_bf16", kernels::ACCUM_BF16),
            // The embedding-table gradient over the distinct looked-up rows (`Ops::embedding_grad`).
            ("emb_bwd_uniq", kernels::EMB_BWD_UNIQ),
            ("emb_bwd_uniq_bf16", kernels::EMB_BWD_UNIQ_BF16),
            // The per-row reduction of `Ops::row_sq_norms`.
            ("row_dot", kernels::ROW_DOT),
            // The cooperative device gradient norm of a full fine-tune's weight gradients (`optim::gradnorm`).
            ("gradnorm_part", kernels::GRADNORM_PART),
            ("gradnorm_part_bf16", kernels::GRADNORM_PART_BF16),
        ];
        list.extend(kernels::template::weight_decode_variants());
        // The decode-regime GEMVs of the dense bf16 and the stored FP8 / NVFP4 tiers (`Ops::small_m_gemv`, `Ops::dense_small_m_gemv`); optional: a list without them
        // keeps the tile decode.
        list.extend(kernels::template::gemv_variants());
        list
    })
}

/// [`kernel_list`] plus [`frozen_base_kernels`]: the list an `Ops` that trains
/// over a quantized or bf16 frozen base feeds `Gpu::new`.
#[must_use]
pub fn training_kernel_list() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| kernel_list().iter().chain(frozen_base_kernels()).copied().collect())
}

/// Assert `list` (the `(name, wgsl_source)` pairs a caller is about to feed
/// `Gpu::new`/`gpu_core::testgpu::dev`, or has already registered on some
/// other `Gpu` it plans to build an [`Ops`] from) names every kernel in
/// [`REQUIRED_KERNELS`]. A caller that registers exactly the façade set
/// should call [`kernel_list`] rather than assembling its own list and
/// checking it here; this stays for the callers that register extra kernels
/// of their own alongside it. Hand-maintenance is exactly how
/// `qwen3::serve`'s
/// `ops_kernel_list` silently drifted 15 kernels short of `REQUIRED_KERNELS`
/// (missing `embed`, `moe_linear_gated`, every `paged_*_batched` bf16 tier,
/// and `matmul_dx`/`matmul_dw`) without `Ops::new`'s own fail-fast check ever
/// firing in a normal test run - `qwen3::serve::Engine` is only ever built
/// eagerly by the residency pool's lazy `activate()` path (GPU activation
/// on-demand, by design - many resident models share one GPU, so nothing is
/// uploaded until a request actually needs it), not at `brain serve`
/// startup, so the gap surfaced on a live server's first real request
/// instead of at `cargo test`/CI time. Calling this function from every
/// `Ops`-building call site's own test suite - a plain name-set comparison
/// against [`REQUIRED_KERNELS`], no `Gpu` required - closes that gap: drift
/// is now caught the moment `cargo test -p <crate>` runs, not three months
/// later on a paying customer's request. Panics (via `assert!`) naming every
/// missing kernel, rather than just the first one [`Ops::new`] would have
/// hit.
pub fn assert_kernel_list_complete(list: &[(&str, &str)]) {
    let have: std::collections::HashSet<&str> = list.iter().map(|(name, _)| *name).collect();
    let missing: Vec<&str> = REQUIRED_KERNELS.iter().filter(|name| !have.contains(*name)).copied().collect();
    assert!(
        missing.is_empty(),
        "kernel list is missing {} kernel(s) required by Ops::new: {missing:?} -- every model that \
         builds an `Ops` must register the full façade kernel set (REQUIRED_KERNELS), not just the \
         tiers it plans to use",
        missing.len()
    );
}

/// One linear layer's resident weight, at whichever tier
/// [`Weight::upload`]'s `want.promote(caps)` landed on.
#[derive(Clone)]
pub enum Weight {
    F32 { w: DeviceBuffer, n: u32, k: u32 },
    /// bf16 storage tier (B4): `w` packed two-per-`u32` over the FLAT `[n*k]`
    /// row-major weight (`model::half::pack_bf16`'s layout - no per-row
    /// repacking, since the templated kernels index `w` as one flat array
    /// exactly like the `F32` tier does). Decoded inline to f32 on load by
    /// the `#w=bf16` kernel variant (`kernels::template::dtype_variant`) -
    /// no device feature required, so this tier is available identically on
    /// the CPU JIT, any GPU backend, and in the browser.
    BF16 { w: DeviceBuffer, n: u32, k: u32 },
    /// f16 (real binary16, not bf16's truncated-f32 shortcut) storage tier
    /// (B5): same packed-two-per-`u32`, flat-`[n*k]` layout as `BF16`
    /// (`model::half::pack_f16`), decoded inline by the `#w=f16` kernel
    /// variant via `dtype_variant`'s magic-multiply expression - real 5-bit
    /// exponent re-biasing, still pure integer/bitcast WGSL, so this tier is
    /// available identically on the CPU JIT, any GPU backend, and in the
    /// browser, exactly like `BF16`.
    F16 { w: DeviceBuffer, n: u32, k: u32 },
    /// DP4A int8: `w` packed `[n, k/4]` u32 (`model::int8::quantize_weight`'s
    /// layout), `s` the GROUP-WISE scale `[n, k/32]`
    /// (`model::int8::GROUP`).
    I8 { w: DeviceBuffer, s: DeviceBuffer, n: u32, k: u32 },
    /// W4A8 int4: `w` packed `[n, k/8]` u32 (`model::int4::quantize_weight_q4`'s
    /// layout), `s` the same group-wise `[n, k/32]` scale the int8 tier uses.
    /// Activations stay int8 - see
    /// this module's doc comment on the offset-arithmetic rule this implies.
    Q4 { w: DeviceBuffer, s: DeviceBuffer, n: u32, k: u32 },
    /// NF4 frozen-weight tier (QLoRA): `w` packed `[n, k/8]` u32, code `b` of a
    /// word at bits `4b` (`model::lut4::quantize_weight_lut4`'s layout with
    /// `NF4_LUT`), `s` the group-wise `[n, k/32]` f32 scale. Unlike [`Weight::Q4`]
    /// this tier is FLOAT-read: its tile decode (`weight_decode`) yields
    /// `NF4_LUT[code] * scale` and the dense GEMM multiplies the tile with the
    /// f32 activations, so nothing but the stored weight is quantised and one
    /// copy serves both `x * W^T` and `dY * W`.
    NF4 { w: DeviceBuffer, s: DeviceBuffer, n: u32, k: u32 },
    /// The canonical brain K-quant device layout (M8-M11, byte-compressed in
    /// M14; see `crate::kquant`'s module doc for the full shape/correction-
    /// term derivation) - ONE variant for all three device instantiations,
    /// distinguished by `bits`/`group`/`affine`:
    ///
    /// | source   | bits | group | affine | [`Weight::dtype`] reports |
    /// |----------|------|-------|--------|----------------------------|
    /// | Q4_K     | 4    | 32    | true   | `Dtype::Q4K`               |
    /// | Q5_K     | 8    | 32    | true   | `Dtype::Q8K`               |
    /// | Q6_K     | 8    | 16    | false  | `Dtype::I8` (group=16)     |
    ///
    /// `w`: `[n, k*bits/32]` u32 - packed codes, K-contiguous, `32/bits`
    /// codes per word, low bits first (unsigned raw value when `affine`,
    /// signed low-bits-two's-complement bias-folded value otherwise -
    /// `gguf::kquant`'s host relayout already did this fold).
    ///
    /// `sz`: the weight-scale plane, `KqScale`-tagged (see that enum's own
    /// doc comment for the affine-vs-non-affine shapes it holds and why).
    KQuant { w: DeviceBuffer, sz: KqScale, n: u32, k: u32, group: u32, bits: u32, affine: bool },
    /// Block-scaled FP8 storage tier (OCP E4M3 or E5M2 bytes, the `weight_scale_inv` convention of
    /// `crate::fp8`): `w` the bytes four to a `u32` along `k` (`[n, k/4]`, `crate::fp8::quantize_weight_fp8`'s
    /// layout), `s` one f32 scale per `block x block` tile of the logical `[n, k]` weight, row-major over
    /// `(ceil(n/block), ceil(k/block))`; a value is `decode(byte) * scale`. Preserved storage: a checkpoint's
    /// bytes and scales go in as they are. Float-read like [`Weight::NF4`] (no activation quantisation):
    /// executed through the bounded tile decode of this module's `tiled` submodule.
    FP8 { w: DeviceBuffer, s: DeviceBuffer, n: u32, k: u32, fmt: crate::fp8::Fp8Format, block: u32 },
    /// The OCP microscaling family, preserved: NVFP4 (E2M1 codes, an E4M3 scale byte per 16 weights, `global`
    /// per tensor), MXFP4 (E2M1, an E8M0 byte per 32) and MXFP8 (E4M3/E5M2, an E8M0 byte per 32). `w` the
    /// element codes in the tier's packing (`[n, k/8]` words of nibbles, or `[n, k/4]` words of bytes), `s` the
    /// scale bytes in row-major block order, four to a word. A value is `element * global * scale`, in that
    /// order, as the checkpoint codec decodes it. Float-read through the tile decode like [`Weight::FP8`].
    Micro { w: DeviceBuffer, s: DeviceBuffer, n: u32, k: u32, kind: MicroScaled, global: f32 },
}

/// Which member of the microscaling family a [`Weight::Micro`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicroScaled {
    /// E2M1 codes, an E4M3 scale per 16 weights and a per-tensor f32 scale.
    Nvfp4,
    /// E2M1 codes, an E8M0 scale per 32 weights.
    Mxfp4,
    /// E4M3 elements, an E8M0 scale per 32 weights.
    Mxfp8E4m3,
    /// E5M2 elements, an E8M0 scale per 32 weights.
    Mxfp8E5m2,
}

/// [`Weight::KQuant`]'s weight-scale plane - the shape differs by `affine`,
/// so this is an enum rather than one fixed buffer field.
#[derive(Clone)]
pub enum KqScale {
    /// Non-affine (Q6_K, group=16): a PLAIN `[n, k/group]` `ds`-only f32
    /// plane (`dm` is always `0.0` for a symmetric type, so there is
    /// nothing to interleave) - the layout the EXISTING `matmul_i8_dyn#
    /// QPG=1`/`matmul_i8_gemv#WPG=4` kernels (M10's group=16 template
    /// knobs) bind directly. M14's byte compression (packed `(sc,m)` +
    /// f16 `(d,dmin)`, see [`Packed`](KqScale::Packed)'s own doc comment)
    /// is deliberately OUT of scope for this arm - those two kernels are
    /// the EXISTING symmetric-family kernels every other int8 model already
    /// depends on, not new kernels this milestone owns; see the M14 ledger
    /// entry for the scoping reasoning.
    PlainF32(DeviceBuffer),
    /// Affine (Q4_K/Q5_K, group=32): M14's packed scale/min plane -
    /// `wsm: [n, ceil(k/group/2)]` u32 (per-group `(sc, m)` sub-scale byte
    /// pair, two groups/word) + `wd: [n, k/spb]` u32 (per-super-block
    /// `(d, dmin)` f16 bit-pattern pair, `spb=256` = `GPS=8` groups) -
    /// `matmul_kq_dyn`/`matmul_kq_gemv`/`matmul_kq_gemv_reg`/
    /// `moe_linear_gated_kq` bind these two directly (replacing the old M8-
    /// M13 flat `[n, 2*k/group]` f32 interleaved `(ds, dm)` plane those same
    /// four kernels used to bind - a full replacement, not a parallel
    /// variant, since nothing outside the `kernels`/`gguf`/`model` crates
    /// depended on the old layout's exact bytes: `Weight::KQuant` has no
    /// `Weight::upload` path at all, and the only two callers of this
    /// struct in the whole workspace are this crate's own test files).
    Packed { wsm: DeviceBuffer, wd: DeviceBuffer },
}

/// The square block edge of the scale grid [`Weight::upload`] gives an FP8 weight: the 128x128 tiles
/// of the `weight_scale_inv` checkpoint convention.
pub const FP8_BLOCK: u32 = 128;

impl Weight {
    /// A dense fp32 weight `[n, k]` over a parameter buffer the model already holds (shared, not copied): how
    /// a model hands an fp32 tensor to the shared ops (the chunked LM head, per-leaf matmuls) without naming
    /// the variant.
    #[must_use]
    pub fn dense_f32(w: DeviceBuffer, n: u32, k: u32) -> Weight {
        Weight::F32 { w, n, k }
    }

    /// Whether this is a dense f32 weight (one f32 buffer, no scales).
    pub fn is_dense_f32(&self) -> bool {
        matches!(self, Weight::F32 { .. })
    }

    /// This weight read from `buf` instead, for a dense f32 weight (whose values a caller moved, such as into a
    /// layer window's slot); `None` for any other tier, whose storage is not one f32 buffer.
    pub fn rebound_f32(&self, buf: DeviceBuffer) -> Option<Weight> {
        match self {
            Weight::F32 { n, k, .. } => Some(Weight::F32 { w: buf, n: *n, k: *k }),
            _ => None,
        }
    }

    /// Whether a product against this tier reads the activations as f32 rows (every float-read tier). The
    /// activation-quantizing tiers (int8 DP4A, int4, K-quants) need [`Ops::act`] to quantize them first.
    pub fn reads_f32_activations(&self) -> bool {
        matches!(self, Weight::F32 { .. } | Weight::BF16 { .. } | Weight::F16 { .. } | Weight::FP8 { .. } | Weight::NF4 { .. } | Weight::Micro { .. })
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            Weight::F32 { .. } => Dtype::F32,
            Weight::BF16 { .. } => Dtype::BF16,
            Weight::F16 { .. } => Dtype::F16,
            Weight::I8 { .. } => Dtype::I8,
            Weight::Q4 { .. } => Dtype::Q4,
            Weight::NF4 { .. } => Dtype::NF4,
            Weight::FP8 { fmt: crate::fp8::Fp8Format::E4M3, .. } => Dtype::F8E4M3,
            Weight::FP8 { fmt: crate::fp8::Fp8Format::E5M2, .. } => Dtype::F8E5M2,
            Weight::Micro { kind: MicroScaled::Nvfp4 | MicroScaled::Mxfp4, .. } => Dtype::F4E2M1,
            Weight::Micro { kind: MicroScaled::Mxfp8E4m3, .. } => Dtype::F8E4M3,
            Weight::Micro { kind: MicroScaled::Mxfp8E5m2, .. } => Dtype::F8E5M2,
            // `affine` picks Q4_K vs Q5_K by `bits` (both group=32); a
            // non-affine (Q6_K) weight reuses `Dtype::I8` - it is, from the
            // selector's perspective, exactly `I8` with a different
            // weight-scale group (see `Weight::group`/`Ops::bind`'s own
            // `group` parameter for how that distinction actually reaches a
            // physical kernel name).
            Weight::KQuant { affine: true, bits: 4, .. } => Dtype::Q4K,
            Weight::KQuant { affine: true, bits: 8, .. } => Dtype::Q8K,
            Weight::KQuant { affine: true, bits, .. } => {
                unreachable!("Weight::KQuant: affine K-quant only ever has bits=4 (Q4_K) or bits=8 (Q5_K), got bits={bits}")
            }
            Weight::KQuant { affine: false, .. } => Dtype::I8,
        }
    }

    /// The weight-scale group width along `k` this weight was packed with -
    /// 32 for every tier except a non-affine (Q6_K) [`Weight::KQuant`],
    /// which is 16. [`Ops::bind`]'s own `group` parameter exists ONLY to
    /// carry this to the physical-kernel-name choice (see that method's doc
    /// comment); every dtype this crate had before M12 was implicitly
    /// group=32 everywhere, which is exactly why this returns a plain `u32`
    /// rather than an `Option` - there is always a real answer.
    pub fn group(&self) -> u32 {
        match self {
            Weight::KQuant { group, .. } => *group,
            _ => 32,
        }
    }

    pub fn n(&self) -> u32 {
        match self {
            Weight::F32 { n, .. }
            | Weight::BF16 { n, .. }
            | Weight::F16 { n, .. }
            | Weight::I8 { n, .. }
            | Weight::Q4 { n, .. }
            | Weight::NF4 { n, .. }
            | Weight::FP8 { n, .. }
            | Weight::Micro { n, .. }
            | Weight::KQuant { n, .. } => *n,
        }
    }

    pub fn k(&self) -> u32 {
        match self {
            Weight::F32 { k, .. }
            | Weight::BF16 { k, .. }
            | Weight::F16 { k, .. }
            | Weight::I8 { k, .. }
            | Weight::Q4 { k, .. }
            | Weight::NF4 { k, .. }
            | Weight::FP8 { k, .. }
            | Weight::Micro { k, .. }
            | Weight::KQuant { k, .. } => *k,
        }
    }

    /// Device bytes an `[n, k]` weight occupies once uploaded at `dtype`: the packed
    /// elements (`Dtype::per_word` to a `u32` word) plus the scales the tier carries - one
    /// f32 per group of 32 along `k` for int8, 4-bit and NF4, one per [`FP8_BLOCK`] x
    /// [`FP8_BLOCK`] block for FP8, none for the 16- and 32-bit tiers. The one place a tier's
    /// footprint is stated, so a planner's estimate and a built model's report agree.
    #[must_use]
    pub fn bytes_for(dtype: Dtype, n: usize, k: usize) -> u64 {
        let packed = (n as u64 * k as u64).div_ceil(dtype.per_word() as u64) * 4;
        let scales = match dtype {
            Dtype::I8 | Dtype::Q4 | Dtype::NF4 => crate::int8::scale_len(n, k) as u64 * 4,
            Dtype::F8E4M3 | Dtype::F8E5M2 => (n as u64).div_ceil(FP8_BLOCK as u64) * (k as u64).div_ceil(FP8_BLOCK as u64) * 4,
            _ => 0,
        };
        packed + scales
    }

    /// Device bytes this weight occupies ([`Self::bytes_for`] at its own tier).
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        Weight::bytes_for(self.dtype(), self.n() as usize, self.k() as usize)
    }

    /// The ONE upload path: quantizes/packs `raw` (`[n, k]`, row-major) per
    /// `want.promote(ops.caps().numeric)` - never narrower than what `want`
    /// asked for, never wider than what the device can execute - then
    /// uploads. A caller never hand-picks a buffer layout itself.
    ///
    /// `want` must be `F32`, `BF16`, `F16`, `I8`, `Q4`, `NF4`, `F8E4M3` or `F8E5M2`. The FP8 tiers
    /// are block-scaled at [`FP8_BLOCK`] and are never promoted: their decode is portable integer
    /// arithmetic that needs no device capability.
    pub fn upload(ops: &Ops, raw: &[f32], n: usize, k: usize, want: Dtype) -> Weight {
        assert_eq!(raw.len(), n * k, "Weight::upload: raw len {} != n*k ({n}*{k})", raw.len());
        assert!(
            matches!(want, Dtype::F32 | Dtype::BF16 | Dtype::F16 | Dtype::I8 | Dtype::Q4 | Dtype::NF4 | Dtype::F8E4M3 | Dtype::F8E5M2),
            "Weight::upload: {want:?} weights are not implemented yet"
        );
        if let Some(fmt) = match want {
            Dtype::F8E4M3 => Some(crate::fp8::Fp8Format::E4M3),
            Dtype::F8E5M2 => Some(crate::fp8::Fp8Format::E5M2),
            _ => None,
        } {
            let (packed, scales) = crate::fp8::quantize_weight_fp8(raw, n, k, fmt, FP8_BLOCK as usize);
            let w = ops.gpu.storage(packed.len() as u64);
            ops.gpu.write(&w, &packed);
            let s = ops.gpu.storage_init("weight_scale_fp8", &scales);
            ops.gpu.prepare_fp8_linear();
            return Weight::FP8 { w, s, n: n as u32, k: k as u32, fmt, block: FP8_BLOCK };
        }
        match want.promote(&ops.caps.numeric) {
            Dtype::F32 => {
                let w = ops.gpu.storage_init("weight_f32", raw);
                Weight::F32 { w, n: n as u32, k: k as u32 }
            }
            Dtype::BF16 => {
                // Flat pack over the whole `[n*k]` row-major weight - the
                // templated kernels index `w` as one flat array (row*k+col
                // arithmetic happens in WGSL, not via any implicit per-row
                // buffer stride), so no reshaping is needed here, unlike the
                // I8/Q4 tiers below (which pack per output row).
                let packed = crate::half::pack_bf16(raw);
                let w = ops.gpu.storage(packed.len() as u64);
                ops.gpu.write(&w, &packed);
                Weight::BF16 { w, n: n as u32, k: k as u32 }
            }
            Dtype::F16 => {
                // Same flat-pack shape as BF16 above, `model::half::pack_f16`
                // instead of `pack_bf16` - see `Weight::F16`'s own doc
                // comment for the layout/decode contract.
                let packed = crate::half::pack_f16(raw);
                let w = ops.gpu.storage(packed.len() as u64);
                ops.gpu.write(&w, &packed);
                Weight::F16 { w, n: n as u32, k: k as u32 }
            }
            Dtype::I8 => {
                let (packed, scales) = crate::int8::quantize_weight(raw, n, k);
                let w = ops.gpu.storage(packed.len() as u64);
                ops.gpu.write(&w, &packed);
                let s = ops.gpu.storage_init("weight_scale_i8", &scales);
                Weight::I8 { w, s, n: n as u32, k: k as u32 }
            }
            Dtype::Q4 => {
                let (packed, scales) = crate::int4::quantize_weight_q4(raw, n, k);
                let w = ops.gpu.storage(packed.len() as u64);
                ops.gpu.write(&w, &packed);
                let s = ops.gpu.storage_init("weight_scale_q4", &scales);
                Weight::Q4 { w, s, n: n as u32, k: k as u32 }
            }
            // `Weight::KQuant` is deliberately NOT reachable through this
            // generic path: it needs the codes' own affine/symmetric bit
            // layout and a (ds, dm) or ds-only scale plane straight from
            // `gguf::kquant`'s host relayout, not a re-quantization of a
            // flat `[n, k]` f32 tensor this function could produce by
            // calling some `quantize_weight_kq(raw, ..)` - K-quant's whole
            // point is to reach the device WITHOUT ever materializing fp32.
            // The `assert!` above already refuses `Q4K`/`Q8K` before this
            // match runs, so this arm is unreachable in practice; it stays
            // an explicit `unreachable!` (not a silent fallthrough) so a
            // future loosening of that assert fails loudly here instead of
            // constructing a bogus `Weight` from fp32 data.
            Dtype::Q4K | Dtype::Q8K => {
                unreachable!("Weight::upload: {want:?} is refused by the assert above -- build a Weight::KQuant directly from gguf::kquant's own packed output instead")
            }
            Dtype::NF4 => {
                let (packed, scales) = crate::lut4::quantize_weight_lut4(raw, n, k, &crate::lut4::NF4_LUT);
                let w = ops.gpu.storage(packed.len() as u64);
                ops.gpu.write(&w, &packed);
                let s = ops.gpu.storage_init("weight_scale_nf4", &scales);
                Weight::NF4 { w, s, n: n as u32, k: k as u32 }
            }
            // `F4E2M1` has its host quantize/dequantize pair and a W4A8 decode
            // kernel (`matmul_q4_gemv_f4e2m1`) but no `Weight` arm: nothing
            // trains over it. The `assert!` above refuses it before this match.
            Dtype::F4E2M1 => {
                unreachable!("Weight::upload: {want:?} is refused by the assert above -- no Weight::F4E2M1 facade arm has been built")
            }
            // M8.6 added `F8E4M3`/`F8E5M2` as `DType` tiers, `kernels::
            // template::f8e4m3_decode_expr`/`f8e5m2_decode_expr`, and a
            // native blockwise-scale device kernel
            // (`matmul_gemv_f8e4m3`/`matmul_gemv_f8e5m2`), but deliberately
            // did NOT wire a `Weight::F8E4M3`/`Weight::F8E5M2` arm into this
            // façade this session either - same clearly-scoped follow-up
            // shape as `NF4`/`F4E2M1` just above, and for the same reason:
            // the `assert!` above already refuses both before this match
            // runs.
            Dtype::F8E4M3 | Dtype::F8E5M2 => unreachable!("Weight::upload: the FP8 tiers return before this match"),
        }
    }
}

impl Weight {
    /// [`Weight::upload`] for the dense tiers (`F32`, `BF16`, `F16`; `I8` through
    /// [`crate::int8::quantize_from`]'s own bounded reader) straight from a checkpoint, without ever
    /// holding the tensor on the host in full: the source is read in bounded chunks (at most
    /// `paramstore::UPLOAD_CHUNK_WORDS` values) and each is packed and written at its place in the
    /// device buffer. The result equals `Weight::upload` of the whole tensor, element for element.
    pub fn upload_from(ops: &Ops, source: &dyn checkpoint::TensorSource, name: &str, n: usize, k: usize, want: Dtype) -> Result<Weight, String> {
        let numel = n * k;
        let tier = want.promote(&ops.caps.numeric);
        // A half tier packs two values to a word, so a chunk is cut on an even element count and lands on a word boundary.
        let chunk = paramstore::UPLOAD_CHUNK_WORDS & !1;
        let gpu = ops.upload_gpu();
        /// Writes one chunk (the `off`-th element on) of the tensor into the buffer being built.
        type WriteChunk<'a> = &'a dyn Fn(&Gpu, &DeviceBuffer, u64, &[f32]);
        let stream = |words: usize, write: WriteChunk<'_>| -> Result<DeviceBuffer, String> {
            let buf = gpu.storage(words as u64);
            let mut seen = 0usize;
            let found = source.with_tensor_chunks(name, chunk, &mut |off, part| {
                write(gpu, &buf, off, part);
                seen += part.len();
            });
            if !found {
                return Err(format!("Weight::upload_from: the checkpoint has no tensor '{name}'"));
            }
            if seen != numel {
                return Err(format!("Weight::upload_from: tensor '{name}' holds {seen} values, expected {numel} ({n}x{k})"));
            }
            Ok(buf)
        };
        match tier {
            Dtype::F32 => Ok(Weight::F32 { w: stream(numel, &|g, b, off, part| g.write_f32_at(b, off, part))?, n: n as u32, k: k as u32 }),
            Dtype::BF16 => {
                assert!(numel.is_multiple_of(2), "Weight::upload_from: a half-precision tensor needs an even element count");
                Ok(Weight::BF16 { w: stream(numel / 2, &|g, b, off, part| g.write_at(b, off / 2, &crate::half::pack_bf16(part)))?, n: n as u32, k: k as u32 })
            }
            Dtype::F16 => {
                assert!(numel.is_multiple_of(2), "Weight::upload_from: a half-precision tensor needs an even element count");
                Ok(Weight::F16 { w: stream(numel / 2, &|g, b, off, part| g.write_at(b, off / 2, &crate::half::pack_f16(part)))?, n: n as u32, k: k as u32 })
            }
            other => Err(format!("Weight::upload_from: {other:?} has no bounded loader (F32, BF16 and F16 do; int8 banks go through model::int8::quantize_from)")),
        }
    }
}

/// Which storage tier each quantizable leaf uploads at - a per-leaf
/// generalization of the single [`Dtype`] `qwen3::Qwen::new_shard_dt` takes.
/// `Qwen35::new_impl_on`'s old `i8: bool` could not express "Q4 on the MLP,
/// F32 on the GDN decay/beta gates"; this is the value that replaces it
/// (M24). Deliberately NOT a fourth `QTier` - `wan::block::QTier` and
/// `ltxv::block::QTier` are two private re-spellings of the same
/// two-variant subset of [`Dtype`] this type is built on; [`Dtype`] is
/// already the currency.
///
/// Matching is by SUBSTRING of the leaf's full name, not anchored to a path
/// segment - a rule `"up"` also matches a hypothetical `"group.weight"`.
/// Bounded in practice by the small, fixed set of leaf names a caller ever
/// consults this against, but real: document it at the call site, do not
/// assume it away.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierPolicy {
    default: Dtype,
    rules: Vec<(String, Dtype)>,
}

impl TierPolicy {
    /// Every leaf at the same tier - what a bare `Dtype` used to mean.
    pub fn uniform(default: Dtype) -> TierPolicy {
        TierPolicy { default, rules: Vec::new() }
    }

    /// Add a rule: any leaf name containing one of `patterns` wants `dt`.
    /// Declaration order matters - [`Self::want`] scans rules in REVERSE, so
    /// a later `with` call narrows an earlier, broader one for a name both
    /// match, without needing to reorder the earlier call.
    pub fn with(mut self, patterns: &[&str], dt: Dtype) -> TierPolicy {
        for p in patterns {
            self.rules.push(((*p).to_string(), dt));
        }
        self
    }

    /// Each named leaf at its own tier (`(name, tier, weights)`): the tier
    /// holding the most weights is the default, and every other leaf a rule
    /// naming it in full - how a checkpoint quantized tensor by tensor is held
    /// as it is stored. Full names never match one another by substring when
    /// they differ only in a layer number (`blocks.1.` is not inside
    /// `blocks.11.`). No leaves: uniform int8.
    pub fn per_leaf(leaves: impl IntoIterator<Item = (String, Dtype, u64)>) -> TierPolicy {
        let leaves: Vec<(String, Dtype, u64)> = leaves.into_iter().collect();
        let mut weights: Vec<(Dtype, u64)> = Vec::new();
        for &(_, tier, w) in &leaves {
            match weights.iter_mut().find(|(t, _)| *t == tier) {
                Some((_, total)) => *total += w,
                None => weights.push((tier, w)),
            }
        }
        let default = weights.iter().max_by_key(|(_, w)| *w).map_or(Dtype::I8, |(t, _)| *t);
        leaves.iter().filter(|(_, tier, _)| *tier != default).fold(TierPolicy::uniform(default), |p, (name, tier, _)| p.with(&[name], *tier))
    }

    /// The tier leaf `name` should upload at: the LAST rule whose pattern is
    /// a substring of `name`, or [`Self::uniform`]'s default if none match.
    pub fn want(&self, name: &str) -> Dtype {
        self.rule_for(name).unwrap_or(self.default)
    }

    /// The default tier and every rule's tier.
    pub fn dtypes(&self) -> impl Iterator<Item = Dtype> + '_ {
        std::iter::once(self.default).chain(self.rules.iter().map(|(_, dt)| *dt))
    }

    /// The tier a rule names for `name`, `None` when no rule matches (so the default would apply).
    pub fn rule_for(&self, name: &str) -> Option<Dtype> {
        self.rules.iter().rev().find(|(pat, _)| name.contains(pat.as_str())).map(|(_, dt)| *dt)
    }

    /// The tier every leaf without a rule takes.
    pub fn default_dtype(&self) -> Dtype {
        self.default
    }

    /// How many `pattern=tier` rules were declared.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// `true` if any leaf can land at a non-F32 tier - the train/quantize
    /// mutual-exclusion assert the old `i8: bool` gated reads this instead.
    pub fn quantizes_anything(&self) -> bool {
        self.default != Dtype::F32 || self.rules.iter().any(|(_, dt)| *dt != Dtype::F32)
    }

    /// A short label for logs and result fingerprints: the default tier, and
    /// how many leaves rules put at each other tier (`"q4k, i8 x114"`) -
    /// [`Self::describe`] spells out every rule, which a per-leaf policy has
    /// by the hundred.
    pub fn summary(&self) -> String {
        let mut others: Vec<(Dtype, usize)> = Vec::new();
        for (_, dt) in &self.rules {
            match others.iter_mut().find(|(t, _)| t == dt) {
                Some((_, n)) => *n += 1,
                None => others.push((*dt, 1)),
            }
        }
        others.iter().fold(self.default.name().to_string(), |s, (dt, n)| format!("{s}, {} x{n}", dt.name()))
    }

    /// Round-trippable string form: `"q4"` (uniform), or
    /// `"q4,in_proj_a.weight=f32,in_proj_b.weight=f32"` (default, then
    /// `pattern=tier` rules in declaration order) - what the serving CLI's
    /// `--weight-format` flag parses.
    pub fn describe(&self) -> String {
        let mut s = self.default.name().to_string();
        for (pat, dt) in &self.rules {
            s.push(',');
            s.push_str(pat);
            s.push('=');
            s.push_str(dt.name());
        }
        s
    }

    /// [`Self::describe`]'s inverse. Rejects an unknown tier name loudly -
    /// never silently defaults to F32 - so a typo in an env var fails fast
    /// instead of quietly serving the wrong precision.
    pub fn parse(s: &str) -> Result<TierPolicy, String> {
        let mut parts = s.split(',');
        let default_s = parts.next().unwrap_or("");
        let default = Dtype::from_name(default_s).ok_or_else(|| format!("TierPolicy::parse: unknown tier {default_s:?}"))?;
        let mut policy = TierPolicy::uniform(default);
        for part in parts {
            let (pat, dt_s) =
                part.split_once('=').ok_or_else(|| format!("TierPolicy::parse: rule {part:?} is not `pattern=tier`"))?;
            let dt = Dtype::from_name(dt_s).ok_or_else(|| format!("TierPolicy::parse: unknown tier {dt_s:?} in rule {part:?}"))?;
            policy = policy.with(&[pat], dt);
        }
        Ok(policy)
    }
}

/// One paged-KV-cache pool buffer (B9) - `Weight`'s sibling for CACHE PAGES,
/// deliberately NOT a `Weight` variant. A KV page has no `(n, k)` GEMM shape
/// and no "load once from a checkpoint" story: `Weight::upload` packs a value
/// known entirely on the host, once, before the device ever sees it; a
/// paged-KV pool starts EMPTY and is grown one token at a time BY the device,
/// through [`Ops::kv_append_batched`] - the write direction `Weight` never
/// needed at all. Forcing it into `Weight`'s enum would mean every `Weight`
/// match arm elsewhere in this crate (`Ops::matmul`, `Ops::embed`, `Ops::
/// moe_linear`) grows an unreachable case for a variant with a completely
/// different shape and lifecycle - a poor fit, not a missing convenience.
///
/// Addressed the SAME way `pool_k`/`pool_v`/`pool` are addressed in the WGSL
/// kernels this wraps: one FLAT `[num_blocks * block_size * kv_stride]`
/// array, `slot*kv_stride + c` indexing done entirely in WGSL (no per-block
/// reshaping on the Rust side, exactly like `Weight`'s own flat `[n*k]`
/// layout leaves row/col arithmetic to the kernel).
pub enum KvPage {
    F32 { buf: DeviceBuffer },
    /// bf16 storage tier (B9): packed two-per-`u32` over the flat pool, same
    /// low-half-even/high-half-odd convention `Weight::BF16`/`model::half::
    /// pack_bf16` already established. Decoded inline on READ by
    /// `paged_decode_scores_batched#pool_k=bf16`/`paged_decode_apply_batched
    /// #pool_v=bf16` (`kernels::template::dtype_variant` - exact bit
    /// widening, same as every other bf16 READ tier). PACKED inline on WRITE
    /// by `paged_kv_append_batched_word#pool=bf16`
    /// (`kernels::template::dtype_variant_store`, B9's new capability, over a
    /// SEPARATE word-granularity physical kernel - see `kname::
    /// PAGED_KV_APPEND_BATCHED_WORD_BF16`'s own doc comment for why) - a
    /// genuine read-modify-write of the shared `u32` word that preserves the
    /// sibling half untouched; see that function's own doc comment for the
    /// exact correctness argument, including the case where two DIFFERENT
    /// cache slots share one word (only possible when `kv_stride` is odd -
    /// every real head_dim in this tree is even, so this is a deliberately
    /// exercised edge case in the tests, not the common path).
    BF16 { buf: DeviceBuffer },
}

impl KvPage {
    pub fn dtype(&self) -> Dtype {
        match self {
            KvPage::F32 { .. } => Dtype::F32,
            KvPage::BF16 { .. } => Dtype::BF16,
        }
    }

    pub fn buf(&self) -> &DeviceBuffer {
        match self {
            KvPage::F32 { buf } | KvPage::BF16 { buf } => buf,
        }
    }

    /// A zero-initialized pool for `num_blocks` blocks of `block_size` tokens
    /// each, `kv_stride` elements per token (`n_kv_heads * head_dim`) - the
    /// SAME total logical element count `num_blocks * block_size *
    /// kv_stride` either dtype addresses, but `BF16` allocates HALF as many
    /// `u32` words (packed 2-per-word, `div_ceil` so an odd total element
    /// count - only possible with an odd `kv_stride`, deliberately exercised
    /// by this phase's read-modify-write stress test - still gets a whole
    /// trailing word rather than truncating). `want` must be `F32` or `BF16`;
    /// every other `Dtype` is a loud panic, matching `Weight::upload`'s own
    /// "never a silent wrong tier" discipline. Unlike `Weight::upload`, this
    /// does NOT consult `DeviceCaps.numeric` to promote/demote: a caller that
    /// wants the capability-aware policy checks `ops.caps().numeric.
    /// bf16_storage` itself before choosing `want`, since (unlike a GEMM
    /// weight) there is no host-side f32 source to fall back to packing from
    /// here - the pool starts empty.
    pub fn zeros(ops: &Ops, num_blocks: u32, block_size: u32, kv_stride: u32, want: Dtype) -> KvPage {
        let words = Self::word_count(num_blocks, block_size, kv_stride, want);
        match want {
            Dtype::F32 => KvPage::F32 { buf: ops.gpu.storage(words) },
            Dtype::BF16 => KvPage::BF16 { buf: ops.gpu.storage(words) },
            other => panic!(
                "KvPage::zeros: {other:?} is not an implemented KV-cache storage tier (only F32/BF16 \
                 exist - see KvPage's own doc comment for why this is a distinct type from Weight)"
            ),
        }
    }

    /// The `u32`-word allocation [`Self::zeros`] requests from the device for
    /// `num_blocks` blocks of `block_size` tokens, `kv_stride` elements each -
    /// factored out of `zeros` so the VRAM-halving claim (`BF16` allocates
    /// HALF the words `F32` does, for the SAME logical pool shape) is a
    /// pure, directly testable function rather than something only provable
    /// by re-deriving the same arithmetic in a test. `F32` is one word
    /// (4 bytes) per element; `BF16` packs 2 elements per word, `div_ceil` so
    /// an ODD total element count (only possible with an odd `kv_stride` -
    /// every real head_dim in this tree is even, but this phase's own
    /// read-modify-write stress test deliberately uses one) still gets a
    /// whole trailing word rather than truncating and losing the last
    /// element. `want` other than `F32`/`BF16` returns 0 - `zeros`'s own
    /// match is what actually panics for those, this function is a pure size
    /// calculator with no side effects to gate.
    pub fn word_count(num_blocks: u32, block_size: u32, kv_stride: u32, want: Dtype) -> u64 {
        let total = num_blocks as u64 * block_size as u64 * kv_stride as u64;
        match want {
            Dtype::F32 => total,
            Dtype::BF16 => total.div_ceil(2),
            _ => 0,
        }
    }
}

/// Shape shared by [`Ops::decode_scores_batched`]/[`Ops::decode_apply_batched`]:
/// the batched paged-decode-attention `Params` both WGSL kernels declare,
/// grouped so a call site passes one value instead of nine positional
/// arguments. `scale` is meaningful only to `decode_scores_batched`;
/// `decode_apply_batched`'s own `Params` has no `scale` field at all, so its
/// kernel ignores this struct's `scale`.
#[derive(Clone, Copy, Debug)]
pub struct PagedDecodeShape {
    pub batch: u32,
    pub n_heads: u32,
    pub group: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub kv_stride: u32,
    pub cap: u32,
    pub max_bt: u32,
    pub scale: f32,
}

/// A quantized-once activation slice: rows `[xr0, xr0+rows)` of a `[.., k]`
/// f32 buffer, ready for every [`Ops::matmul`] call that reads it (`F32`
/// weights read the raw buffer directly; `I8`/`Q4` weights read the packed
/// form). See this module's doc comment for why quantization is eager and
/// unconditional rather than lazy.
pub struct Act {
    x: DeviceBuffer,
    xr0: u32,
    m: u32,
    k: u32,
    /// The packed int8 view of these rows. `None` when the caller declared,
    /// via [`Ops::act_f32`], that no weight reading this activation is
    /// quantized - which is what makes the two packing dispatches skippable
    /// rather than merely unused.
    quant: std::cell::RefCell<Option<Arc<I8Scratch>>>,
    /// Whether [`Ops::act`] left `quant` unmade because this device's tuned table has every linear of this width and
    /// row count read bf16 rows ([`Gpu::int8_packed_unneeded`]): a product that does need the packed form (a weight
    /// window or alignment no bf16 candidate serves) makes it then, at its own position in the step order
    /// ([`Ops::packed`]).
    quant_deferred: bool,
    /// `crate::kquant`'s `S[m,g] = Σ_{k in g} xq[m,k]` affine correction
    /// prepass (M9's `quant_group_sum.wgsl`), `[xr0+rows, k/GROUP]` f32. Made
    /// by the first [`Weight::KQuant`] product that reads these rows and
    /// shared by the rest (or up front by [`Ops::act_kq`]), so an activation
    /// that only `F32`/`BF16`/`F16`/`I8`/`Q4` weights read never pays for the
    /// dispatch, and a model holding K-quant weights needs no activation of
    /// its own. Ordering is the `bf16` cast's below.
    xgs: std::cell::RefCell<Option<DeviceBuffer>>,
    /// These rows rounded to bf16, made by the first product on the vendor
    /// library that reads them and shared by the rest: the linears of a group
    /// that read one activation (a GDN layer's four projections, attention's
    /// three, the MLP's two) pay for one cast instead of one each. The cast
    /// step precedes the first such product in the step order, which is the
    /// order every later product is submitted in.
    bf16: std::cell::RefCell<Option<DeviceBuffer>>,
    /// These rows quantised to E4M3 with a scale per row per 128 of K (the bytes and the scales), made by the first FP8-weight
    /// product whose chosen candidate reads them and shared by the rest, like `bf16`.
    e4m3: std::cell::RefCell<Option<(DeviceBuffer, DeviceBuffer)>>,
}

/// A model-facing façade over `backend_api::select`'s kernel selector,
/// `model::int8`/`model::int4`'s quantizers, and `model::dispatch::I8Scratch`.
/// See this module's doc comment.
pub struct Ops {
    gpu: Gpu,
    caps: DeviceCaps,
    idx: HashMap<&'static str, usize>,
    /// Kept for [`Ops::matmul_kernel`]'s own diagnostic report ONLY -
    /// [`Ops::matmul`] itself dispatches through `providers`, never this
    /// field directly, since M8.3 moved that dispatch onto the
    /// `OperatorProvider` seam. See [`Ops::with_providers`]'s doc comment for
    /// the one narrow case (a caller-built registry with its own selector,
    /// bypassing both [`Ops::new`] and [`Ops::with_selector`]) where this
    /// field's policy can differ from what `providers` actually runs.
    selector: Arc<dyn KernelSelector>,
    /// What [`Ops::matmul`], [`Ops::embed`], [`Ops::moe_linear`],
    /// [`Ops::matmul_dx`] and [`Ops::matmul_dw`] dispatch through - see
    /// `gpu_core::provider`'s module doc for this seam's scope, and for
    /// which of this façade's other methods are deliberately outside it.
    providers: ProviderRegistry,
    /// bf16-weight GEMMs on the vendor library's tensor cores, or the decline
    /// that sends them to the fp32 kernels. Starts at [`GemmPrecision::Auto`]
    /// (tensor cores wherever the device has them, by default); gradient
    /// checks and parity gates pin [`GemmPrecision::Fp32Reference`] through
    /// [`Ops::set_gemm_precision`].
    gemm: LibraryGemmProvider,
    /// Bytes of decoded f32 scratch one tile of a quantized product may use (see [`tiled`]).
    tile_budget_bytes: u64,
    /// A tiny buffer bound to the `weight_decode` bindings a tier does not use.
    decode_dummy: std::sync::OnceLock<DeviceBuffer>,
    /// The transfer handle tiled weight loads upload through, once [`Ops::enable_streamed_uploads`] set it.
    uploads: std::sync::OnceLock<StreamedUploads>,
}

/// A transfer [`Gpu`] handle (own stream, bounded pinned staging ring) and the pinned-pool lease that pays for
/// the ring; the lease goes back to the budget when the [`Ops`] drops.
struct StreamedUploads {
    gpu: Gpu,
    _lease: checkpoint::packed::Lease,
}

mod sink;
mod upload_source;
mod tiled;
mod decoded;
pub use sink::{rewrite_tile, upload_micro_bank, DeviceWeightSink, MicroBank};
pub use device_io::{store_device_kernels, DeviceIo, StoreDevice};
mod device_io;
pub use upload_source::{stored_tier, stored_tier_policy, upload_fused_from_source, upload_from_source, upload_group_preserved, upload_preserved, FusedGroup};
pub use upload_source::{conversion_policy, encoding_for};
pub use tiled::{plan_linear, tile_rows, uniq_padded, EmbeddingGrad, GradDtype, LinearPlan, DEFAULT_TILE_BUDGET_BYTES, GEMV_MAX_ROWS, UNIQ_PAD};

impl Ops {
    /// [`Ops::with_selector`] with the default policy
    /// (`CachedSelector<DefaultSelector>`) - what every model crate except
    /// `qwen3::serve` uses today.
    ///
    /// Resolves every kernel name in [`REQUIRED_KERNELS`] via
    /// `Gpu::kernel_index` ONCE, here - never at dispatch time. A missing
    /// name is an `Err` immediately, not a panic three linears deep into a
    /// forward pass.
    pub fn new(gpu: Gpu) -> Result<Ops, String> {
        Self::with_selector(gpu, Arc::new(select::CachedSelector::new(select::DefaultSelector)))
    }

    /// [`Ops::new`], but the kernel selector is the CALLER's, not a fixed
    /// internal default.
    ///
    /// **Why this exists.** Before this constructor, `Ops::matmul` always
    /// resolved through a hard-wired `CachedSelector<DefaultSelector>` with
    /// no injection point - which is exactly why `qwen3::serve::Engine` kept
    /// its own ad-hoc int8 GEMM dispatch (`Self::mm8`/`Self::tune_i8`)
    /// entirely OUTSIDE this façade instead of calling `Ops::matmul`: its
    /// `tuned_i8` selector is a REAL, per-device MEASURED policy
    /// (`AutoTuner`+`FileTuneStore`), and routing through the old fixed
    /// `Ops::new` would have silently discarded it in favour of the static
    /// default. A caller with such a policy now hands it to `Ops` directly
    /// (`Arc<dyn KernelSelector>` - `Arc`, not `Box`, so the same instance
    /// can also be kept by the caller and reused for its own dispatch, the
    /// way `qwen3::serve::Engine` does).
    ///
    /// Internally this is [`ProviderRegistry::for_gpu`] - **the production
    /// provider chain**, not a bare reference registry.
    ///
    /// That is the one behavioural difference this constructor has ever had.
    /// Which providers `for_gpu` puts ahead of the WGSL reference is decided
    /// entirely by what `gpu`'s own device reports about itself, so on every
    /// device that reports nothing extra (the CPU JIT, wgpu, Vulkan) the
    /// chain still contains exactly the reference provider and this
    /// constructor's contract is unchanged bit-for-bit. On a device that
    /// does - today, one reporting a compute capability - a specialised
    /// provider is asked FIRST, may decline, and whatever happens is
    /// recorded in that dispatch's `ImplChoice`. See `ProviderRegistry::
    /// for_gpu`'s own doc comment for the gating rule and for which
    /// providers are deliberately still absent from it.
    pub fn with_selector(gpu: Gpu, selector: Arc<dyn KernelSelector>) -> Result<Ops, String> {
        let providers = ProviderRegistry::for_gpu(&gpu, selector.clone());
        Self::build(gpu, selector, providers)
    }

    /// [`Ops::new`], but the DISPATCH is the caller's whole
    /// [`ProviderRegistry`], not just a [`KernelSelector`] the fixed WGSL
    /// path consults. `Ops::matmul` pushes an `OpRequest` at `registry.
    /// dispatch(...)` instead of resolving a kernel by hand - see
    /// `gpu_core::provider`'s module doc for the ABI and this seam's current
    /// `Op::MatMul`-only scope (M8.3, `kernel-performance.md` Phase 8).
    ///
    /// [`Ops::matmul_kernel`]'s diagnostic report is sourced from a fresh
    /// `CachedSelector<DefaultSelector>`, NOT from any selector `providers`'
    /// own reference provider may have been built with - a `ProviderRegistry`
    /// does not expose one generically (a non-WGSL provider need not have a
    /// `KernelSelector` at all). This cannot change what `Ops::matmul` itself
    /// dispatches (that always goes through `providers`, never this field);
    /// it only means a caller of THIS constructor with a custom-selector
    /// registry should not rely on `matmul_kernel`'s reported name matching
    /// its own policy. No existing caller uses this constructor with a
    /// non-default selector (it did not exist before this milestone), so
    /// this is a documented limitation of new surface, not a behaviour
    /// change to anything that already worked.
    pub fn with_providers(gpu: Gpu, providers: ProviderRegistry) -> Result<Ops, String> {
        let selector: Arc<dyn KernelSelector> = Arc::new(select::CachedSelector::new(select::DefaultSelector));
        Self::build(gpu, selector, providers)
    }

    fn build(gpu: Gpu, selector: Arc<dyn KernelSelector>, providers: ProviderRegistry) -> Result<Ops, String> {
        let caps = gpu.caps();
        let mut idx = HashMap::with_capacity(kname::ALL.len());
        for &name in kname::ALL {
            let i = gpu.kernel_index(name).ok_or_else(|| {
                format!(
                    "Ops::new: kernel '{name}' is not registered on this Gpu -- every model that \
                     builds an `Ops` must register the full façade kernel set ({:?}), not just the \
                     tiers it plans to use",
                    kname::ALL
                )
            })?;
            idx.insert(name, i);
        }
        let gemm = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
        Ok(Ops { gpu, caps, idx, selector, providers, gemm, tile_budget_bytes: DEFAULT_TILE_BUDGET_BYTES, decode_dummy: std::sync::OnceLock::new(), uploads: std::sync::OnceLock::new() })
    }

    /// Choose the arithmetic of the GEMMs against bf16 weights: `Auto` takes
    /// the vendor library's bf16 tensor cores where the device has them and
    /// falls back to the fp32 kernels everywhere else, without a flag or an
    /// error; `Fp32Reference` pins the fp32 kernels (what gradient checks and
    /// cross-backend parity use). See [`gpu_core::provider::library`].
    pub fn set_gemm_precision(&mut self, precision: GemmPrecision) {
        self.gemm = LibraryGemmProvider::for_gpu(&self.gpu, precision);
    }

    /// The vendor-library GEMM provider (inactive on a device without the
    /// library, or when the fp32 reference tier is pinned): what a model asks
    /// for the products that are not a frozen weight times an activation, such
    /// as a LoRA adapter's, before falling back to its own kernels.
    pub fn tensor_gemm(&self) -> &LibraryGemmProvider {
        &self.gemm
    }

    /// Which arithmetic path the bf16-weight GEMMs of this `Ops` take.
    pub fn gemm_path(&self) -> &library::GemmPath {
        self.gemm.path()
    }

    /// `y[yoff ..] = act @ W[first.., :]^T` on the library path, for an f32 or bf16 `w` (every other tier
    /// reaches the library as a decoded bf16 tile, see [`tiled`]);
    /// `false` (nothing pushed) when the fp32 kernels must run instead.
    fn library_forward(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), act: &Act, y: &DeviceBuffer, yoff: u64) -> bool {
        if let Weight::F32 { w: wb, n, k } = w {
            // The TF32 tier reads the fp32 weight and activation as they are stored.
            let window = if rows.0 == 0 && rows.1 == *n { WHOLE } else { Self::row_window(rows.0, rows.1, *n, *k, 1) };
            let m = act.m;
            let x_op = library::Operand { buf: &act.x, range: (act.xr0 as u64 * *k as u64, m as u64 * *k as u64) };
            let y_op = library::Operand { buf: y, range: (yoff, m as u64 * rows.1 as u64) };
            return self.gemm.forward_tf32(&self.gpu, s, &x_op, &library::Operand { buf: wb, range: window }, &y_op, m, rows.1, *k);
        }
        let Some((wb, window)) = self.library_weight(w, rows) else { return false };
        let mut prep = Vec::new();
        let k = w.k();
        let m = act.m;
        let (w_op, y_op) = (library::Operand { buf: &wb, range: window }, library::Operand { buf: y, range: (yoff, m as u64 * rows.1 as u64) });
        // The activation rounded to bf16, from an earlier product on these rows or made now.
        let shared = act.bf16.borrow().clone();
        let (xb, fresh) = match shared {
            Some(xb) => (xb, false),
            None => {
                let x_op = library::Operand { buf: &act.x, range: (act.xr0 as u64 * k as u64, m as u64 * k as u64) };
                match self.gemm.cast_bf16(&self.gpu, &mut prep, &x_op, m as u64 * k as u64) {
                    Some(xb) => (xb, true),
                    None => return false,
                }
            }
        };
        let ok = self.gemm.forward_bf16(&self.gpu, &mut prep, &xb, &w_op, &y_op, m, rows.1, k);
        if ok {
            s.append(&mut prep);
            if fresh {
                *act.bf16.borrow_mut() = Some(xb);
            }
        }
        ok
    }

    /// An int8-weight product on the dequantize-in-register candidate this device's tuned table names for its shape
    /// class, if it names one and the dispatch can run it: the weight codes are multiplied by their group scale in
    /// registers and the activations are the bf16 rows of `act` (rounded once, shared by every product that reads
    /// them), so no activation quantisation is read at all. `false` leaves the packed int8 form to the caller.
    /// Only whole matrices of group-32 weights.
    fn dequant_forward(&self, s: &mut Vec<Step>, w: &Weight, act: &Act, y: &DeviceBuffer, yoff: u64) -> bool {
        let Weight::I8 { w: wb, s: sw, n, k } = w else { return false };
        let (m, n, k) = (act.m, *n, *k);
        let params = [m, k / 4, n];
        // The bf16 candidates are the CUDA provider's native tier: a registry without it (or with it disabled) stays on
        // the reference kernels.
        if w.group() != 32 || !self.providers.runs("cuda") || !self.gpu.int8_linear_bf16_input(&params) {
            return false;
        }
        let mut prep = Vec::new();
        let Some((xb, fresh)) = self.bf16_rows(act, &mut prep) else { return false };
        let Some((step, _)) = self.gpu.int8_dequant_step(&[&xb, wb, sw, y], &[(0, 0), (0, 0), (0, 0), (yoff, m as u64 * n as u64)], &params) else { return false };
        s.append(&mut prep);
        s.push(step);
        if fresh {
            *act.bf16.borrow_mut() = Some(xb);
        }
        true
    }

    /// An FP8-weight product (E4M3 bytes with the checkpoint's 128 x 128 block scales, held as stored) on the native
    /// candidate this device's tuned table names for its shape class: the weight bytes are decoded in registers or shared
    /// memory against the bf16 rows of `act` (rounded once, shared by every product that reads them) and the sum is f32
    /// across all of K, with one scale per 128 of K. `false` leaves the product to the tile decode. Only whole matrices.
    fn fp8_forward(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), act: &Act, y: &DeviceBuffer, yoff: u64) -> bool {
        let Weight::FP8 { w: wb, s: sb, n, k, fmt: crate::fp8::Fp8Format::E4M3, block } = w else { return false };
        let (m, n, k) = (act.m, *n, *k);
        if *block != FP8_BLOCK || rows != (0, n) || !k.is_multiple_of(FP8_BLOCK) || !self.providers.runs("cuda") || !self.gpu.has_fp8_linear() {
            return false;
        }
        let params = [m, k / 4, n];
        let Some(form) = self.gpu.fp8_linear_act(&params) else { return false };
        let out = (yoff, m as u64 * n as u64);
        let mut prep = Vec::new();
        let step = match form {
            Fp8Act::Bf16 => {
                let Some((xb, fresh)) = self.bf16_rows(act, &mut prep) else { return false };
                let Some((step, _)) = self.gpu.fp8_linear_step(form, &[&xb, wb, sb, y], &[(0, 0), (0, 0), (0, 0), out], &params) else { return false };
                if fresh {
                    *act.bf16.borrow_mut() = Some(xb);
                }
                step
            }
            Fp8Act::E4m3 => {
                let shared = act.e4m3.borrow().clone();
                let (xq, sx, fresh) = match shared {
                    Some((xq, sx)) => (xq, sx, false),
                    None => {
                        let x_op = library::Operand { buf: &act.x, range: (act.xr0 as u64 * k as u64, m as u64 * k as u64) };
                        let Some((xq, sx)) = library::quantize_e4m3_rows(&self.gpu, &mut prep, &x_op, m, k) else { return false };
                        (xq, sx, true)
                    }
                };
                let Some((step, _)) = self.gpu.fp8_linear_step(form, &[&xq, wb, &sx, sb, y], &[(0, 0), (0, 0), (0, 0), (0, 0), out], &params) else { return false };
                if fresh {
                    *act.e4m3.borrow_mut() = Some((xq, sx));
                }
                step
            }
        };
        s.append(&mut prep);
        s.push(step);
        true
    }

    /// The rows of `act` rounded to bf16, from an earlier product on these rows or cast now (the cast step is pushed into
    /// `prep`; the second value is whether it is new, to be remembered on `act` once the product that needs it is kept).
    /// `None` when the device declines the cast.
    fn bf16_rows(&self, act: &Act, prep: &mut Vec<Step>) -> Option<(DeviceBuffer, bool)> {
        if let Some(xb) = act.bf16.borrow().clone() {
            return Some((xb, false));
        }
        let k = act.k;
        let x_op = library::Operand { buf: &act.x, range: (act.xr0 as u64 * k as u64, act.m as u64 * k as u64) };
        library::cast_bf16(&self.gpu, prep, &x_op, act.m as u64 * k as u64).map(|xb| (xb, true))
    }

    /// The bf16 weight rows `rows` of `w` as the library path reads them: the buffer and the range to bind.
    /// `None` for a tier or a device the library path does not serve (only bf16 storage is read in place).
    fn library_weight(&self, w: &Weight, rows: (u32, u32)) -> Option<(DeviceBuffer, (u64, u64))> {
        if !self.gemm.is_active() {
            return None;
        }
        let (first, count) = rows;
        let (n, k) = (w.n(), w.k());
        match w {
            Weight::BF16 { w: wb, .. } => {
                let window = if first == 0 && count == n { WHOLE } else { (first as u64 * k as u64 / 2, count as u64 * k as u64 / 2) };
                Some((wb.clone(), window))
            }
            _ => None,
        }
    }

    /// `dx[dx_row0.., :] (+)= dy[m, count] @ W[first.., :]` on the library path
    /// for a bf16 `w`; `false` when the fp32 kernels must run instead.
    #[allow(clippy::too_many_arguments)]
    fn library_dx(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), dy: &DeviceBuffer, m: u32, dx: &DeviceBuffer, dx_row0: u32, accumulate: bool) -> bool {
        if let Weight::F32 { w: wb, n, k } = w {
            let window = if rows.0 == 0 && rows.1 == *n { WHOLE } else { Self::row_window(rows.0, rows.1, *n, *k, 1) };
            let dx_range = if dx_row0 == 0 { WHOLE } else { (dx_row0 as u64 * *k as u64, m as u64 * *k as u64) };
            let dy_op = library::Operand { buf: dy, range: WHOLE };
            return self.gemm.dx_tf32(&self.gpu, s, &dy_op, &library::Operand { buf: wb, range: window }, &library::Operand { buf: dx, range: dx_range }, m, rows.1, *k, accumulate);
        }
        let Some((wb, window)) = self.library_weight(w, rows) else { return false };
        let mut prep = Vec::new();
        let k = w.k();
        let dx_range = if dx_row0 == 0 { WHOLE } else { (dx_row0 as u64 * k as u64, m as u64 * k as u64) };
        let ok = self.gemm.dx(
            &self.gpu,
            &mut prep,
            &library::Operand { buf: dy, range: WHOLE },
            &library::Operand { buf: &wb, range: window },
            &library::Operand { buf: dx, range: dx_range },
            m,
            rows.1,
            k,
            accumulate,
        );
        if ok {
            s.append(&mut prep);
        }
        ok
    }

    pub fn caps(&self) -> &DeviceCaps {
        &self.caps
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The handle weight bytes are written through: the transfer handle once streamed uploads are enabled,
    /// else the compute handle.
    pub(crate) fn upload_gpu(&self) -> &Gpu {
        self.uploads.get().map_or(&self.gpu, |u| &u.gpu)
    }

    /// Upload the weights of every tiled load ([`DeviceWeightSink`], so every `upload_*` call) through a transfer
    /// handle of this device: its own stream, so a layer's bytes move while this handle's kernels run, and a
    /// pinned staging ring of `staging_bytes` the bytes are copied through. The ring is leased from `budget`'s
    /// [`Pool::Pinned`](checkpoint::packed::Pool::Pinned) for as long as this `Ops` lives, so the host memory a
    /// streamed load adds is a configured number, not a function of the layer size. Results are bit-identical to
    /// the plain path; on a device that cannot stage writes the handle uploads synchronously. Call it once,
    /// before the first load; a second call is refused.
    pub fn enable_streamed_uploads(&self, budget: &checkpoint::packed::MemoryBudget, staging_bytes: u64) -> Result<(), String> {
        let lease = budget.lease(checkpoint::packed::Pool::Pinned, staging_bytes).map_err(|e| e.to_string())?;
        let gpu = self.gpu.upload_handle(staging_bytes)?;
        self.uploads.set(StreamedUploads { gpu, _lease: lease }).map_err(|_| "Ops::enable_streamed_uploads: uploads are already streamed".to_string())
    }

    /// Quantize rows `[xr0, xr0+rows)` of `x` (`[.., k]`) for every
    /// subsequent [`Ops::matmul`] call against the returned [`Act`]. Wraps a
    /// fresh [`I8Scratch`] sized for exactly `[0, xr0+rows)` - reusing its
    /// existing offset arithmetic (`I8Scratch::quant_rows` /
    /// `dispatch::quant_rows_steps`), not reimplementing it. `xr0` must clear
    /// the 64-row (256B) storage-binding alignment `quant_rows_steps` itself
    /// asserts.
    pub fn act(&self, s: &mut Vec<Step>, x: &DeviceBuffer, xr0: u32, rows: u32, k: u32) -> Act {
        assert!(rows > 0, "Ops::act: rows must be > 0 (got 0)");
        // A width the packed path cannot hold (not a whole number of
        // `crate::int8::GROUP`s) simply is not packed: [`Self::act_f32`] is
        // what a dense product reads, and a product that REQUIRES the packed
        // rows (`Weight::I8`, `Q4`) refuses by name when it finds none. This
        // is what lets a narrow model - a toy config, a small test tower -
        // build at all.
        if !k.is_multiple_of(crate::int8::GROUP as u32) {
            return self.act_f32(x, xr0, rows, k);
        }
        // The packed form is made only where some linear of this width reads it: when this device's tuned table has
        // every int8 linear of these rows and this width on a candidate that reads bf16 rows, the two quantisation
        // dispatches (and the scratch) are skipped and a stray product that does need them makes them then.
        let deferred = self.providers.runs("cuda") && self.gpu.int8_packed_unneeded(rows, k / 4);
        let quant = (!deferred).then(|| self.quantize_rows(s, x, xr0, rows, k));
        Act { x: x.clone(), xr0, m: rows, k, quant: std::cell::RefCell::new(quant), quant_deferred: deferred, xgs: Default::default(), bf16: Default::default(), e4m3: Default::default() }
    }

    /// The packed int8 form of rows `[xr0, xr0 + rows)` of `x`: scratch sized for exactly `[0, xr0 + rows)` and the
    /// `max_abs_row` / `quant_pack` dispatches that fill it.
    fn quantize_rows(&self, s: &mut Vec<Step>, x: &DeviceBuffer, xr0: u32, rows: u32, k: u32) -> Arc<I8Scratch> {
        let total = (xr0 + rows) as u64;
        let quant = I8Scratch::new(&self.gpu, total, total, &[k]);
        quant.quant_rows(&self.gpu, [self.idx[kname::MAX_ABS_ROW], self.idx[kname::QUANT_PACK]], s, x, xr0, xr0 + rows, k);
        Arc::new(quant)
    }

    /// The packed int8 form of `act` for a product that reads it, made now when [`Ops::act`] deferred it. `None`
    /// for an activation built by [`Ops::act_f32`], which promises no quantised weight reads it.
    fn packed(&self, s: &mut Vec<Step>, act: &Act) -> Option<Arc<I8Scratch>> {
        if let Some(q) = act.quant.borrow().as_ref() {
            return Some(q.clone());
        }
        if !act.quant_deferred {
            return None;
        }
        let q = self.quantize_rows(s, &act.x, act.xr0, act.m, act.k);
        *act.quant.borrow_mut() = Some(q.clone());
        Some(q)
    }

    /// The [`Act`] for rows `[0, rows)` of `x` (`[rows, k]`) that something
    /// else has already quantised: `sx` holds the `[rows, k/32]` group scales and
    /// `xq` the `[rows, k/4]` packed int8 rows, exactly what [`Ops::act`]'s two
    /// dispatches would have written. This is the seam a fused kernel uses to
    /// hand a linear its activation without the `max_abs_row` and `quant_pack`
    /// dispatches (or the scratch allocation) [`Ops::act`] would add.
    pub fn act_prequantized(&self, x: &DeviceBuffer, sx: &DeviceBuffer, xq: &DeviceBuffer, rows: u32, k: u32) -> Act {
        assert!(rows > 0, "Ops::act_prequantized: rows must be > 0 (got 0)");
        assert_eq!(k % 4, 0, "int8 K must be a multiple of 4 (got {k})");
        Act { x: x.clone(), xr0: 0, m: rows, k, quant: std::cell::RefCell::new(Some(Arc::new(I8Scratch::from_parts(sx.clone(), k, xq.clone())))), quant_deferred: false, xgs: Default::default(), bf16: Default::default(), e4m3: Default::default() }
    }

    /// [`Ops::act`], plus the affine K-quant group-sum prepass
    /// (`quant_group_sum.wgsl`, M9) every [`Weight::KQuant`] matmul needs.
    /// Builds the SAME `I8Scratch` `act` does (identical `sx`/`xq`, so the
    /// packed activation this produces is byte-identical to what `act` would
    /// have produced for the same call), plus one more `[xr0+rows, k/GROUP]`
    /// f32 buffer that `crate::int8::quant_rows_steps`'s `xgs` seam fills as
    /// a THIRD dispatch alongside `max_abs_row`/`quant_pack` - see that
    /// function's own doc comment for why this is a separate step rather
    /// than folded into `quant_pack` itself (the group sum is read by every
    /// linear pairing this activation with an affine weight; computing it
    /// once here, rather than once per `n`-column inside the GEMM's k-loop,
    /// is the entire point of `quant_group_sum` existing as a prepass).
    ///
    /// `k` must be a multiple of `crate::int8::GROUP` (32) - the same
    /// contract `quant_rows_steps`'s own `xgs` branch asserts.
    pub fn act_kq(&self, s: &mut Vec<Step>, x: &DeviceBuffer, xr0: u32, rows: u32, k: u32) -> Act {
        assert!(rows > 0, "Ops::act_kq: rows must be > 0 (got 0)");
        let total = (xr0 + rows) as u64;
        let quant = I8Scratch::new(&self.gpu, total, total, &[k]);
        let xgs = self.gpu.storage(total * (k as u64 / crate::int8::GROUP as u64));
        s.extend(crate::int8::quant_rows_steps(
            &self.gpu,
            crate::int8::QuantRows {
                kernels: [self.idx[kname::MAX_ABS_ROW], self.idx[kname::QUANT_PACK]],
                x,
                sx: &quant.sx,
                xq: quant.xq_for(k),
                xgs: Some((self.idx[kname::QUANT_GROUP_SUM], &xgs)),
            },
            xr0,
            xr0 + rows,
            k,
        ));
        Act { x: x.clone(), xr0, m: rows, k, quant: std::cell::RefCell::new(Some(Arc::new(quant))), quant_deferred: false, xgs: std::cell::RefCell::new(Some(xgs)), bf16: Default::default(), e4m3: Default::default() }
    }

    /// [`Ops::act`] for a caller whose weights are ALL fp32-family, so the
    /// packed int8 view would be built and never read.
    ///
    /// The packing is not free: it is a `max_abs_row` and a `quant_pack`
    /// dispatch plus a fresh `I8Scratch` allocation per activation, and a
    /// decode tape builds one activation per linear group per layer per token.
    /// Measured on a Qwen3-VL-4B caption (fp32 weights throughout), the two
    /// kernels were a measurable share of all device time and the allocations ran to tens of
    /// thousands per image, for a buffer no dispatch ever bound.
    ///
    /// The declaration is the CALLER's because only the caller knows its
    /// weight tier; passing an `act_f32` activation to a quantized weight is
    /// refused by name in [`Ops::matmul`] rather than silently reading
    /// whatever a missing buffer would have held.
    pub fn act_f32(&self, x: &DeviceBuffer, xr0: u32, rows: u32, k: u32) -> Act {
        assert!(rows > 0, "Ops::act_f32: rows must be > 0 (got 0)");
        Act { x: x.clone(), xr0, m: rows, k, quant: Default::default(), quant_deferred: false, xgs: Default::default(), bf16: Default::default(), e4m3: Default::default() }
    }

    /// [`Self::act_f32`] for all `rows` rows of `x` whose bf16 rounding a fused kernel has already produced
    /// (`bf16`, as [`library::LibraryGemmProvider::silu_mul_bf16`] returns it): the library products that
    /// read this activation take it instead of casting it again.
    pub fn act_f32_with_bf16(&self, x: &DeviceBuffer, rows: u32, k: u32, bf16: DeviceBuffer) -> Act {
        assert!(rows > 0, "Ops::act_f32_with_bf16: rows must be > 0 (got 0)");
        Act { x: x.clone(), xr0: 0, m: rows, k, quant: Default::default(), quant_deferred: false, xgs: Default::default(), bf16: std::cell::RefCell::new(Some(bf16)), e4m3: Default::default() }
    }

    /// [`Self::act_f32`] for `rows` rows of `x` whose E4M3 quantisation (`xq` bytes, `sx` one scale per row per 128 of K,
    /// as `quant_e4m3_128` makes them) a fused kernel has already produced: the FP8 products that read this activation as
    /// E4M3 take it instead of quantising it again.
    pub fn act_f32_with_e4m3(&self, x: &DeviceBuffer, rows: u32, k: u32, xq: DeviceBuffer, sx: DeviceBuffer) -> Act {
        assert!(rows > 0 && k.is_multiple_of(FP8_BLOCK), "Ops::act_f32_with_e4m3: {rows} rows of {k} (K a multiple of {FP8_BLOCK})");
        Act { x: x.clone(), xr0: 0, m: rows, k, quant: Default::default(), quant_deferred: false, xgs: Default::default(), bf16: Default::default(), e4m3: std::cell::RefCell::new(Some((xq, sx))) }
    }

    /// Whether an FP8-weight product of `rows` rows against `n` outputs of width `k` reads E4M3 activations on this device
    /// (the form its shape class is tuned to): a producer that can write them in its own pass then spares the product a
    /// separate quantisation launch.
    pub fn fp8_reads_e4m3(&self, rows: u32, k: u32, n: u32) -> bool {
        k.is_multiple_of(FP8_BLOCK) && self.providers.runs("cuda") && self.gpu.has_fp8_linear() && self.gpu.fp8_linear_act(&[rows, k / 4, n]) == Some(Fp8Act::E4m3)
    }

    /// The f32 rows of an activation built by [`Self::act_f32`]: the buffer, its
    /// first row, row count and width, for a caller that dispatches a kernel
    /// of its own over them.
    pub fn f32_rows<'a>(&self, act: &'a Act) -> (&'a DeviceBuffer, u32, u32, u32) {
        (&act.x, act.xr0, act.m, act.k)
    }

    /// `(variant, dtype, group) -> kernel name`. The ONLY place in this crate
    /// a kernel-name string literal is chosen by a match arm (`kname`'s own
    /// consts are the only place one is spelled at all). Any pair not listed
    /// here is unreachable given [`select::candidates`]'s own contract
    /// (`F32`-family dtypes only ever offer `{Reference, WorkgroupPerOutput,
    /// RegisterTiled}`; `I8`/`Q4`/`Q4K`/`Q8K` only ever offer
    /// `{WorkgroupPerOutput, PackedInt8}` - see `backend_api::select::
    /// candidates`) crossed with [`Weight::upload`]'s own `DType::promote`
    /// gate (an `I8`/`Q4`/`BF16` `Weight` exists at all only on a device
    /// whose caps already promoted it there), so a panic here means one of
    /// those two contracts broke, not a normal runtime condition.
    ///
    /// `BF16` (B4) and `F16` (B5) each get their OWN kernel names, distinct
    /// from `F32`'s: a real `Weight::BF16`/`Weight::F16` buffer holds PACKED
    /// u32 words, not raw f32 values, so dispatching it through the plain f32
    /// kernel would silently reinterpret those bit patterns as garbage f32s.
    /// `RegisterTiled`'s bf16/f16 kernel is `matmul_reg3`, NOT the
    /// `matmul_reg2` the `F32` arm uses - see `kname::MATMUL_REG3_BF16`'s own
    /// doc comment for why picking a different physical file per dtype for
    /// the same `KernelVariant` is already precedented (`PackedInt8`'s
    /// i8-vs-q4 split, see [`Ops::threads`]).
    ///
    /// # Why `group` (M12)
    ///
    /// `group` is the weight-scale group width [`Weight::group`] reports -
    /// 32 for every tier that predates M12, always passed as a literal `32`
    /// by every one of THOSE dtypes' arms below (never actually read for
    /// them; `Weight::group()` already defaults to 32 for anything that
    /// isn't a `Weight::KQuant`, so this parameter existing at all only
    /// matters for `Dtype::I8` at `group == 16` - Q6_K, M10's group=16 reuse
    /// of the existing symmetric kernels via a template knob). `Dtype::I8`
    /// therefore branches on it (`32` -> the untouched default
    /// `matmul_i8_{dyn,gemv}`, `16` -> `matmul_i8_{dyn,gemv}#{QPG=1,WPG=4}`,
    /// M10's own specialised pair); `Dtype::Q4K`/`Dtype::Q8K` never branch on
    /// it at all, because `matmul_kq_dyn`/`matmul_kq_gemv`'s own weight-scale
    /// group is FIXED at 32 by construction (see those kernels' own header:
    /// "this kernel serves only Q4_K/Q5_K, both group=32; Q6_K's group=16
    /// reaches the device through the EXISTING symmetric kernels' own QPG
    /// knob instead, not this one") - `CODE_BITS` (folded into `dt` itself:
    /// `Q4K` vs `Q8K`), not `group`, is what selects between their two
    /// specialisations.
    fn bind(v: KernelVariant, dt: Dtype, group: u32) -> &'static str {
        use KernelVariant::*;
        match (v, dt, group) {
            (Reference, Dtype::F32, 32) => kname::MATMUL,
            (Reference, Dtype::BF16, 32) => kname::MATMUL_BF16,
            (Reference, Dtype::F16, 32) => kname::MATMUL_F16,
            (WorkgroupPerOutput, Dtype::F32, 32) => kname::MATMUL_GEMV,
            (WorkgroupPerOutput, Dtype::BF16, 32) => kname::MATMUL_GEMV_BF16,
            (WorkgroupPerOutput, Dtype::F16, 32) => kname::MATMUL_GEMV_F16,
            (RegisterTiled, Dtype::F32, 32) => kname::MATMUL_REG2,
            (RegisterTiled, Dtype::BF16, 32) => kname::MATMUL_REG3_BF16,
            (RegisterTiled, Dtype::F16, 32) => kname::MATMUL_REG3_F16,
            (WorkgroupPerOutput, Dtype::I8, 32) => kname::MATMUL_I8_GEMV,
            (PackedInt8, Dtype::I8, 32) => kname::MATMUL_I8_DYN,
            (WorkgroupPerOutput, Dtype::I8, 16) => kname::MATMUL_I8_GEMV_G16,
            (PackedInt8, Dtype::I8, 16) => kname::MATMUL_I8_DYN_G16,
            (WorkgroupPerOutput, Dtype::Q4, 32) => kname::MATMUL_Q4_GEMV,
            // M5.5's register-tiled `matmul_q4_dyn_reg` (128x128 tile,
            // mirroring `matmul_i8_dyn`'s own shape) replaces the naive
            // `matmul_q4_dyn` here - see `kname::MATMUL_Q4_DYN_REG`'s own
            // doc comment and `Ops::threads`'s doc comment for the dispatch-
            // geometry consequence of this switch.
            (PackedInt8, Dtype::Q4, 32) => kname::MATMUL_Q4_DYN_REG,
            (WorkgroupPerOutput, Dtype::Q4K, 32) => kname::MATMUL_KQ_GEMV_4,
            (PackedInt8, Dtype::Q4K, 32) => kname::MATMUL_KQ_DYN_4,
            (WorkgroupPerOutput, Dtype::Q8K, 32) => kname::MATMUL_KQ_GEMV_8,
            (PackedInt8, Dtype::Q8K, 32) => kname::MATMUL_KQ_DYN_8,
            (v, dt, group) => panic!(
                "Ops::matmul: select::candidates offered {v:?} for dtype {dt:?} at group={group} \
                 which this façade has no kernel bound for -- see `Ops::bind`'s doc comment for why \
                 this should be unreachable"
            ),
        }
    }

    /// Dispatch invocation count for `(variant, dtype, m, n)` - the same
    /// formulas `model::dispatch::mm_rows_off`/`mm8_rows_off` and
    /// `model::block::pick_gemm`/`gemm_variant` already use per variant
    /// family, EXCEPT `PackedInt8`, which - unlike every other variant here -
    /// is not one fixed dispatch geometry: `matmul_i8_dyn.wgsl` and (since
    /// M5.5 was wired into [`Ops::bind`]) `matmul_q4_dyn_reg.wgsl` are BOTH
    /// register-tiled (128×128 tile, 256-thread workgroup, `workgroup_id`-
    /// indexed - `matmul_q4_dyn_reg.wgsl`'s own header: "mirroring
    /// `matmul_i8_dyn`'s shape"), and need the SAME tile dispatch count; only
    /// `matmul_q4_gemv`'s `WorkgroupPerOutput` sibling and the two K-quant
    /// `WorkgroupPerOutput` kernels stay outside this arm. The dispatch count
    /// has to follow the REAL kernel [`Ops::bind`] chose, not the logical
    /// variant name alone - using `m*n` for a register-tiled kernel
    /// under-dispatches it (leaves real output elements never written);
    /// using the tile formula for a one-thread-per-output kernel merely
    /// over-dispatches (extra threads whose output index falls outside the
    /// real grid are harmless) - caught in the direction that actually
    /// corrupts output by this façade's own parity test.
    ///
    /// `Dtype::Q4K`/`Dtype::Q8K` MUST land in the `tile()` arm alongside
    /// `Dtype::I8`/`Dtype::Q4`, not a `m*n` fallback: `matmul_kq_dyn.wgsl` is
    /// `matmul_i8_dyn`'s own 128×128 register-tiled sibling (see that
    /// kernel's own header - "Register-block ownership, bank padding,
    /// software pipelining and the epilogue are copied from `matmul_i8_dyn`
    /// unchanged"), so it needs the SAME dispatch geometry `Dtype::I8`
    /// already gets - getting this wrong under-dispatches the tile grid and
    /// leaves real output elements never written (silent output corruption,
    /// not a crash).
    ///
    /// **M8.3**: this formula itself (and its regression test,
    /// `kq_dtypes_dispatch_the_tiled_formula_not_m_times_n`) moved verbatim
    /// to `gpu_core::provider::wgsl::WgslProvider::threads` - it has zero
    /// dependency on this crate's kernel-name table, so it now lives where
    /// [`Ops::matmul`] actually calls it (inside `WgslProvider::lower`), not
    /// duplicated here as dead code. **M8.5** added `NF4`/`F4E2M1` and
    /// **M8.6** added `F8E4M3`/`F8E5M2` to that moved formula directly (this
    /// doc comment's home moved before either landed, so the integrating
    /// cherry-picks applied both dtype additions there instead of
    /// reintroducing this function).
    ///
    /// `y[yoff .. yoff + m*n)] = act[xr0..xr0+m, :] @ wᵀ`, where `m` is
    /// however many rows `act` was built for ([`Ops::act`]'s `rows`). The
    /// selection/kernel-name/thread-count/dispatch logic that used to live
    /// directly in this method now lives in `gpu_core::provider::wgsl::
    /// WgslProvider::lower` (moved verbatim - same selector call, same
    /// [`Self::bind`] name table, same thread-count formula, same
    /// `step_sliced` call); this method's own job is now only building the
    /// per-dtype [`Operand`] bundle (identical buffers, identical offsets,
    /// identical params to what it pushed by hand before) and handing it to
    /// `self.providers.dispatch`. Zero behaviour change: `crates/model/tests/
    /// ops_facade_parity.rs` (unchanged) still asserts the output is
    /// bit-identical to the pre-existing `dispatch.rs`-driven oracle.
    pub fn matmul(&self, s: &mut Vec<Step>, w: &Weight, act: &Act, y: &DeviceBuffer, yoff: u64) {
        let (n, k) = (w.n(), w.k());
        assert_eq!(act.k, k, "Ops::matmul: activation width {} does not match weight K {k}", act.k);
        let m = act.m;
        if self.dense_small_m_gemv(s, w, (0, n), act, y, yoff) || self.library_forward(s, w, (0, n), act, y, yoff) {
            return;
        }
        let shape = OpShape { m, n, k, dtype: w.dtype() };
        let group = w.group();
        let dt = w.dtype();
        let bind = |variant: KernelVariant| -> (usize, &'static str) {
            let name = Self::bind(variant, dt, group);
            (self.idx[name], name)
        };

        let oo = (yoff, m as u64 * n as u64);

        match w {
            // `BF16`/`F16` read the SAME `act.x` (raw f32, never quantized -
            // only the WEIGHT narrows for these tiers) with the SAME
            // `[m, k, n]` params `F32` uses; the weight buffer's own offset
            // is always `(0, 0)` for all three (whole-buffer, independent of
            // which activation rows `m` covers), so only which physical
            // kernel `Self::bind` chose differs - the packed-vs-plain
            // layout is entirely the kernel's own concern, not this dispatch
            // site's.
            Weight::F32 { w: wb, .. } | Weight::BF16 { w: wb, .. } | Weight::F16 { w: wb, .. } => {
                let xo = (act.xr0 as u64 * k as u64, m as u64 * k as u64);
                let attrs = [m, k, n];
                let operands = [
                    Operand { role: Role::Act, buf: &act.x, range: xo, dtype: Dtype::F32 },
                    Operand { role: Role::Weight, buf: wb, range: (0, 0), dtype: dt },
                    Operand { role: Role::Out, buf: y, range: oo, dtype: Dtype::F32 },
                ];
                let req = OpRequest { op: Op::MatMul, shape, pass: Pass::Forward, operands: &operands, attrs: &attrs, group: 32, bind: &bind };
                let mut ctx = LowerCtx { gpu: &self.gpu, caps: &self.caps, steps: s, capture: false };
                self.providers.dispatch(&mut ctx, &req);
            }
            Weight::I8 { w: wb, s: sw, .. } | Weight::Q4 { w: wb, s: sw, .. } => {
                // The packed activation BUFFER OFFSET is ALWAYS int8-word
                // sized (W4A8 - see this module's doc comment), so the
                // offset divisor is `Dtype::I8.per_word()`, NEVER
                // `w.dtype().per_word()` - a `Q4` weight's own per_word()
                // (8) would silently halve this offset instead of
                // quartering it.
                let per_word = Dtype::I8.per_word() as u64;
                let kg = k as u64 / per_word;
                let xo = (act.xr0 as u64 * kg, m as u64 * kg);
                let so = crate::int8::act_scale_range(act.xr0, m, k);
                // The kernel's own `k` PARAM (not a buffer offset) is a
                // SEPARATE contract per dtype, and the two disagree:
                // `matmul_i8_{dyn,gemv}.wgsl` take the packed word count
                // (`kg = k/4`) directly, since x and w share one word
                // density there - but `matmul_q4_{dyn,gemv}.wgsl` take the
                // RAW logical `k`, un-divided, because x (int8, 4/word) and
                // w (int4, 8/word) have DIFFERENT word densities for the
                // same K (see `matmul_q4_dyn.wgsl`'s own header: "a single
                // shared `kg` the way the int8 family uses would be
                // ambiguous about which operand it counts"). Passing `kg`
                // to the q4 kernels (or raw `k` to the i8 kernels) is
                // exactly the silently-wrong-arithmetic class this module's
                // doc comment warns about - caught by this façade's own
                // parity test before this comment was written.
                //
                // `Dtype::Q4K`/`Dtype::Q8K` never reach this arm (they have
                // their own `Weight::KQuant` arm below), so the `_ => k`
                // fallback here is exercised ONLY by `Dtype::Q4` today - but
                // it IS the right answer for the affine kernels too, by the
                // same reasoning `matmul_q4_{dyn,gemv}` already established:
                // `matmul_kq_{dyn,gemv}.wgsl`'s own header states their `k`
                // param is "the RAW LOGICAL reduction length, NOT a packed-
                // word count" for the identical reason (`xq` and `wq` have
                // DIFFERENT word densities - 4 codes/word for `xq` always,
                // `32/CODE_BITS` for `wq`), so a real sixth dtype landing in
                // this same `I8`/`Q4` arm by mistake would be the only way
                // this fallback silently mis-derives `param_k` - it does not
                // today.
                let param_k = match dt {
                    Dtype::I8 => kg as u32,
                    _ => k,
                };
                if self.dequant_forward(s, w, act, y, yoff) {
                    return;
                }
                let quant = self.packed(s, act).expect(
                    "Ops::matmul: this activation was built with Ops::act_f32, which promises no \
                     quantized weight reads it - build it with Ops::act instead",
                );
                let attrs = [m, param_k, n];
                let operands = [
                    Operand { role: Role::Act, buf: quant.xq_for(k), range: xo, dtype: Dtype::I8 },
                    Operand { role: Role::Weight, buf: wb, range: (0, 0), dtype: dt },
                    Operand { role: Role::ActScale, buf: &quant.sx, range: so, dtype: Dtype::F32 },
                    Operand { role: Role::WeightScale, buf: sw, range: (0, 0), dtype: Dtype::F32 },
                    Operand { role: Role::Out, buf: y, range: oo, dtype: Dtype::F32 },
                ];
                let req = OpRequest { op: Op::MatMul, shape, pass: Pass::Forward, operands: &operands, attrs: &attrs, group, bind: &bind };
                let mut ctx = LowerCtx { gpu: &self.gpu, caps: &self.caps, steps: s, capture: false };
                self.providers.dispatch(&mut ctx, &req);
            }
            // NF4 is float-read (QLoRA): the kernel decodes each weight load
            // and multiplies it with the raw f32 activation, so this arm never
            // touches `act.quant` and bypasses the W4A8 selector arms. It binds
            // the variant itself, like the int8 frozen-base GEMM of `qwen3`.
            Weight::NF4 { .. } | Weight::FP8 { .. } | Weight::Micro { .. } => self.matmul_rows(s, w, (0, n), act, y, yoff),
            // M12: affine K-quant (`Weight::KQuant { affine: true, .. }` -
            // Q4_K/Q5_K, `matmul_kq_dyn`/`matmul_kq_gemv`). The non-affine
            // (Q6_K, group=16) instantiation never constructs THIS variant's
            // buffer shape - it is a `Weight::I8`-shaped weight (plain `s`,
            // no `sz`/`xgs`) with `group() == 16`, so it already took the
            // `Weight::I8 { .. }` arm above via `Self::bind`'s `group`
            // parameter; only the affine pair needs the extra `wsm`/`wd`/
            // `xgs` buffers this arm binds.
            Weight::KQuant { w: wb, sz, .. } => {
                assert!(
                    dt == Dtype::Q4K || dt == Dtype::Q8K,
                    "Ops::matmul: Weight::KQuant reached the affine dispatch arm with dtype {dt:?} -- \
                     a non-affine (Q6_K) KQuant weight must be built with a plain ds-only `sz` and \
                     dispatched as Dtype::I8 at group=16, not through this arm"
                );
                let KqScale::Packed { wsm, wd } = sz else {
                    unreachable!(
                        "Ops::matmul: an affine (Q4K/Q8K) Weight::KQuant must carry KqScale::Packed -- \
                         the assert above already refused a non-affine dtype, so a KqScale::PlainF32 \
                         reaching here is a construction bug, not a normal input"
                    )
                };
                // Same packed-word-per-int8-word offset rule the `I8`/`Q4`
                // arm above uses (`xq` is always 4 int8/word, regardless of
                // the WEIGHT's own code density) - `matmul_kq_{dyn,gemv}`
                // read `xq` through the identical `model::int8` packing.
                let per_word = Dtype::I8.per_word() as u64;
                let kg = k as u64 / per_word;
                let xo = (act.xr0 as u64 * kg, m as u64 * kg);
                let so = crate::int8::act_scale_range(act.xr0, m, k);
                // `xgs` is `[.., k/GROUP]` f32 - one group sum per row, the
                // SAME "own width" offset convention `so`/`xo` already
                // follow (see `crate::int8::quant_rows_steps`'s own doc
                // comment on `xgs`'s offset units).
                let go = crate::int8::act_scale_range(act.xr0, m, k);
                let quant = self.packed(s, act).expect(
                    "Ops::matmul: this activation was built with Ops::act_f32, which promises no \
                     quantized weight reads it - build it with Ops::act or Ops::act_kq instead",
                );
                // A native kernel serving this product sums the activation
                // groups itself, so the prepass is made only for a kernel that
                // reads it; the native one gets the scales as a placeholder.
                let native = self.gpu.native_kernel_for(self.idx[self.matmul_kernel(w, m)], &[m, k, n]).is_some();
                let (xgs, go) = if native {
                    (quant.sx.clone(), so)
                } else {
                    let xgs = act
                        .xgs
                        .borrow_mut()
                        .get_or_insert_with(|| {
                            let xgs = self.gpu.storage((act.xr0 + m) as u64 * (k as u64 / crate::int8::GROUP as u64));
                            s.push(self.gpu.step_sliced(self.idx[kname::QUANT_GROUP_SUM], &[quant.xq_for(k), &xgs], &[xo, go], &[m, k], m * k / crate::int8::GROUP as u32));
                            xgs
                        })
                        .clone();
                    (xgs, go)
                };
                let xgs = &xgs;
                // `k` (RAW LOGICAL K, not a packed word count) - see the
                // long comment on the `I8`/`Q4` arm above for why this is
                // the correct param here too.
                let attrs = [m, k, n];
                let operands = [
                    Operand { role: Role::Act, buf: quant.xq_for(k), range: xo, dtype: Dtype::I8 },
                    Operand { role: Role::Weight, buf: wb, range: (0, 0), dtype: dt },
                    Operand { role: Role::ActScale, buf: &quant.sx, range: so, dtype: Dtype::F32 },
                    Operand { role: Role::WeightMin, buf: wsm, range: (0, 0), dtype: Dtype::F32 },
                    Operand { role: Role::WeightScale, buf: wd, range: (0, 0), dtype: Dtype::F32 },
                    Operand { role: Role::ActGroupSum, buf: xgs, range: go, dtype: Dtype::F32 },
                    Operand { role: Role::Out, buf: y, range: oo, dtype: Dtype::F32 },
                ];
                let req = OpRequest { op: Op::MatMul, shape, pass: Pass::Forward, operands: &operands, attrs: &attrs, group: 32, bind: &bind };
                let mut ctx = LowerCtx { gpu: &self.gpu, caps: &self.caps, steps: s, capture: false };
                self.providers.dispatch(&mut ctx, &req);
            }
        }
    }

    /// Whether a `[m, n]` GEMM output fills enough 128x128 tiles for the
    /// register-tiled kernel to beat the naive one: the selector's own shape
    /// rule (`select::candidates`, skipping the GEMV arm no frozen-base kernel
    /// has), and only on a device that executes workgroup barriers - the
    /// tiled variants of the frozen-base kernels have no CPU fast path, so the
    /// CPU JIT must keep the naive kernel.
    fn tiled_gemm(&self, m: u32, n: u32) -> bool {
        self.caps.workgroup_reductions
            && select::candidates(Op::MatMul, OpShape { m, n, k: 0, dtype: Dtype::F32 }, &self.caps)
                .into_iter()
                .find(|v| *v != KernelVariant::WorkgroupPerOutput)
                == Some(KernelVariant::RegisterTiled)
    }

    /// The registered index of a [`kname`] kernel outside [`kname::ALL`] (see
    /// [`frozen_base_kernels`]), or a panic naming the missing registration.
    fn frozen_base_kernel(&self, name: &str) -> usize {
        self.gpu.kernel_index(name).unwrap_or_else(|| {
            panic!(
                "Ops: kernel '{name}' is not registered on this Gpu -- a frozen-base bf16/quantized-training \
                 build must register model::ops::training_kernel_list()"
            )
        })
    }

    /// `(dtype) -> embed kernel name`. Unlike [`Self::bind`], there is no
    /// `KernelVariant` choice to make here - `embed.wgsl` has exactly one
    /// dispatch shape per dtype (a plain gather, no GEMV/tiled alternative),
    /// so this façade method bypasses `select::candidates` entirely and binds
    /// directly. `I8`/`Q4` have no embed kernel at all (no quantized
    /// embedding table exists in this tree) - a panic here means a caller
    /// asked [`Weight::upload`] for a tier this method cannot serve, which
    /// should never happen given `Weight::upload`'s own tier assertion.
    /// The vocab-tile twin of [`Self::bind_embed`] (`embed_tile.wgsl`), for a table larger than one binding.
    fn bind_embed_tile(dt: Dtype) -> &'static str {
        match dt {
            Dtype::BF16 => kname::EMBED_TILE_BF16,
            Dtype::F16 => kname::EMBED_TILE_F16,
            _ => kname::EMBED_TILE,
        }
    }

    fn bind_embed(dt: Dtype) -> &'static str {
        match dt {
            Dtype::F32 => kname::EMBED,
            Dtype::BF16 => kname::EMBED_BF16,
            Dtype::F16 => kname::EMBED_F16,
            other => panic!(
                "Ops::embed: no embed kernel for dtype {other:?} -- only F32/BF16/F16 embedding \
                 tables are implemented (no embed_i8/embed_q4 kernel exists in this tree)"
            ),
        }
    }

    /// `x[t, c] = table[token[t], c]` for `seq_len` tokens - the same
    /// contract every model's own hand-dispatched `EMBED` kernel index
    /// already uses (e.g. `qwen3::model::Qwen`'s
    /// `g.step(EMBED, &[&self.tokens, self.w("tok.weight"), &self.res[0]],
    /// &[d, 1], d)`), but `table` can be `Weight::{F32,BF16,F16}` - a bf16/f16
    /// embedding table is a genuine VRAM win (`[vocab, d_model]` scales with
    /// vocabulary size, easily hundreds of MB to low GB at fp32 for a modern
    /// tokenizer). `table`'s own `n`/`k` are `(vocab_rows, d_model)` - the
    /// SAME `[n, k]` row-major shape [`Weight::upload`] already builds for a
    /// GEMM weight, since an embedding table IS just a `[vocab, d_model]`
    /// matrix; only the OPERATION (gather, not a reduction) differs, which is
    /// exactly why this needs its own method rather than reusing
    /// [`Self::matmul`]. No model crate's call sites are migrated by this
    /// method's addition - see this crate's `ops` module doc for why that is
    /// deliberate (matching B3's own "build the façade, prove it, migrate
    /// later" precedent).
    pub fn embed(&self, s: &mut Vec<Step>, table: &Weight, tokens: &DeviceBuffer, seq_len: u32, out: &DeviceBuffer) {
        let d_model = table.k();
        let dt = table.dtype();
        // `bind_embed` already panics loudly for `I8`/`Q4` before the match
        // below is ever reached, so those two arms there are unreachable by
        // construction, not merely by convention.
        // Resolved HERE, not inside the closure: `bind_embed` is what panics
        // by name for a dtype with no embed kernel, and the provider calls
        // the closure only after this method's own match has already
        // committed to an arm - so a lazy resolution would reach the
        // `unreachable!` below instead, with a message saying a panic that
        // never happened already had.
        let kname = Self::bind_embed(dt);
        let bind = |_: KernelVariant| -> (usize, &'static str) { (self.idx[kname], kname) };
        match table {
            Weight::F32 { w, .. } | Weight::BF16 { w, .. } | Weight::F16 { w, .. } => {
                // A table larger than one binding (a 248k-row vocabulary is 2.5 GB at bf16, past a Vulkan device's
                // 2 GB storage-binding ceiling) is gathered tile by tile: `embed_tile` binds rows `[v0, v0 + count)`
                // and writes only the tokens that fall there, so every output element is written by exactly one
                // tile. Tiles are sized in 32-bit words, two bf16/f16 elements to a word.
                let row_words = if dt == Dtype::F32 { d_model } else { d_model / 2 };
                assert!(dt == Dtype::F32 || d_model.is_multiple_of(2), "Ops::embed: a {dt:?} table row of {d_model} elements is not whole words");
                let tiles = crate::block::vocab_tiles_on(&self.gpu, table.n() as u64, row_words as u64);
                if tiles.len() > 1 {
                    let kind = self.idx[Self::bind_embed_tile(dt)];
                    let rw = row_words as u64;
                    for (v0, count) in tiles {
                        s.push(self.gpu.step_sliced(kind, &[tokens, w, out], &[WHOLE, (v0 as u64 * rw, count as u64 * rw), WHOLE], &[d_model, seq_len, v0, count], seq_len * d_model));
                    }
                    return;
                }
                let operands = [
                    // The token buffer holds u32 row INDICES, not numbers the
                    // kernel does arithmetic on. `Role::Aux` says so (it is
                    // neither an activation nor a weight), and the `dtype`
                    // field - informational, read by no provider for this
                    // operand - records the 32-bit storage width rather than
                    // claiming an arithmetic tier `DType` has no spelling for.
                    Operand { role: Role::Aux(0), buf: tokens, range: WHOLE, dtype: Dtype::F32 },
                    Operand { role: Role::Weight, buf: w, range: WHOLE, dtype: dt },
                    Operand { role: Role::Out, buf: out, range: WHOLE, dtype: Dtype::F32 },
                ];
                let attrs = [d_model, seq_len];
                let req = OpRequest {
                    op: Op::Embed,
                    shape: OpShape { m: seq_len, n: d_model, k: table.n(), dtype: dt },
                    pass: Pass::Forward,
                    operands: &operands,
                    attrs: &attrs,
                    group: 32,
                    bind: &bind,
                };
                let mut ctx = LowerCtx { gpu: &self.gpu, caps: &self.caps, steps: s, capture: false };
                self.providers.dispatch(&mut ctx, &req);
            }
            Weight::I8 { .. } | Weight::Q4 { .. } | Weight::NF4 { .. } | Weight::KQuant { .. } | Weight::FP8 { .. } | Weight::Micro { .. } => {
                unreachable!("Ops::embed: bind_embed already panicked for this dtype above")
            }
        }
    }

    /// `(dtype) -> moe_linear_gated kernel name`, mirroring [`Self::bind_embed`]:
    /// no `KernelVariant` choice (`moe_linear_gated.wgsl` is one fixed
    /// one-thread-per-output-element dispatch shape per dtype - the
    /// register-tiled alternative every OTHER GEMM variant offers cannot
    /// safely early-exit per row under a workgroup barrier, see that kernel's
    /// own header). `I8`/`Q4` route through `model::moe::MoeIds8`'s own
    /// `moe_linear_gated_i8.wgsl`/`_q4.wgsl` directly - DIFFERENT buffer/param
    /// shapes (packed activation + two scales, not a plain `[x, w, gate,
    /// out]`), so this façade does not attempt to unify them here.
    fn bind_moe_linear(dt: Dtype) -> &'static str {
        match dt {
            Dtype::F32 => kname::MOE_LINEAR_GATED,
            Dtype::BF16 => kname::MOE_LINEAR_GATED_BF16,
            Dtype::F16 => kname::MOE_LINEAR_GATED_F16,
            other => panic!(
                "Ops::moe_linear: {other:?} experts are dispatched via model::moe::MoeIds8/MoeIds4 \
                 directly (moe_linear_gated_i8.wgsl/_q4.wgsl have a different buffer/param shape), \
                 not through this method"
            ),
        }
    }

    /// `out[row, :] = 0` for a row not routed to this expert (`gate[row,
    /// e_idx] <= 0`), else `out = x @ Wᵀ` - the same contract
    /// `model::moe::expert_fwd`'s own `g.step(ids.linear_gated, &[x, w, gate,
    /// out], &[m, k, n, e, e_idx], m * n)` dispatches by hand, but `w` can be
    /// `Weight::{F32,BF16,F16}`. Sparse-MoE expert weights are one of this
    /// program's genuinely large weight-storage candidates: a 256-expert MoE
    /// (e.g. `crates/qwen35moe`) holds 256 independent gate/up/down
    /// projections, so a bf16/f16 storage tier on the expert bank is a real,
    /// multiplicative VRAM win, not a per-layer rounding error.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_linear(
        &self,
        s: &mut Vec<Step>,
        w: &Weight,
        x: &DeviceBuffer,
        gate: &DeviceBuffer,
        n_experts: u32,
        e_idx: u32,
        m: u32,
        out: &DeviceBuffer,
    ) {
        let (n, k) = (w.n(), w.k());
        let dt = w.dtype();
        // `bind_moe_linear` already panics loudly for `I8`/`Q4` before the
        // match below is ever reached, so that arm there is unreachable by
        // construction, not merely by convention.
        // Eagerly resolved, for the reason `Ops::embed` states: this is the
        // call that panics by name for an unsupported dtype, and it has to
        // happen before the match rather than inside the provider's callback.
        let kname = Self::bind_moe_linear(dt);
        let bind = |_: KernelVariant| -> (usize, &'static str) { (self.idx[kname], kname) };
        match w {
            Weight::F32 { w: wb, .. } | Weight::BF16 { w: wb, .. } | Weight::F16 { w: wb, .. } => {
                let operands = [
                    Operand { role: Role::Act, buf: x, range: WHOLE, dtype: Dtype::F32 },
                    Operand { role: Role::Weight, buf: wb, range: WHOLE, dtype: dt },
                    // The routing gate: `[rows, n_experts]`, read to decide
                    // whether this expert owns a row at all. Neither an
                    // activation the kernel multiplies nor a weight.
                    Operand { role: Role::Aux(0), buf: gate, range: WHOLE, dtype: Dtype::F32 },
                    Operand { role: Role::Out, buf: out, range: WHOLE, dtype: Dtype::F32 },
                ];
                // `w_off = 0`: `w` is this expert's own `[n, k]` matrix, so
                // its rows start at the buffer's first element.
                let attrs = [m, k, n, n_experts, e_idx, 0];
                let req = OpRequest {
                    op: Op::MoeExpertLinear,
                    shape: OpShape { m, n, k, dtype: dt },
                    pass: Pass::Forward,
                    operands: &operands,
                    attrs: &attrs,
                    group: 32,
                    bind: &bind,
                };
                let mut ctx = LowerCtx { gpu: &self.gpu, caps: &self.caps, steps: s, capture: false };
                self.providers.dispatch(&mut ctx, &req);
            }
            Weight::I8 { .. } | Weight::Q4 { .. } | Weight::NF4 { .. } | Weight::KQuant { .. } | Weight::FP8 { .. } | Weight::Micro { .. } => {
                unreachable!("Ops::moe_linear: bind_moe_linear already panicked for this dtype above")
            }
        }
    }

    // --- B9: paged-KV-cache append/scores/apply, batched family -------------
    //
    // `qwen3::serve::Engine`'s existing int8 KV-cache tier
    // (`paged_kv_append_i8_clipped_batched`/`paged_decode_scores_i8_batched`/
    // `paged_decode_apply_i8_batched`) is this façade's closest precedent -
    // same three-operation family, same batched dispatch shape - but that
    // engine dispatches its own tuned kernel selection by hand (`Ops::matmul`'s
    // own doc comment explains why `serve.rs` was deliberately NOT migrated
    // onto `Ops::matmul` in B7: a real, measured per-device selector `Ops`
    // cannot express). These three methods are the `Ops`-level, kernel-proven
    // bf16 KV-cache tier - NOT wired into `qwen3::serve::Engine` by this
    // phase: adding a third hand-dispatched tier to that engine and
    // re-running its own full test suite was judged a materially larger,
    // riskier change than proving the tier at the kernel/`Ops` level alone,
    // matching the "build the façade, prove it, migrate later" precedent
    // this whole program has followed since B3.

    /// `(dtype) -> (kernel name, threads)` for [`Ops::kv_append_batched`].
    /// UNLIKE every other `bind_*` method in this façade, the two dtypes here
    /// bind to genuinely DIFFERENT PHYSICAL KERNELS with different dispatch
    /// GEOMETRIES, not just different kernel names at the SAME geometry - see
    /// `kname::PAGED_KV_APPEND_BATCHED_WORD_BF16`'s own doc comment for why
    /// (a per-element bf16 pack dispatch races; the word-granularity sibling
    /// kernel does not). `F32` stays one thread per (token, element)
    /// (`batch * kv_stride`, this kernel's own established parallelism);
    /// `BF16` is one thread per even-sized chunk of a token
    /// ([`crate::kv_tier::append_word_threads`]), each looping over its chunk
    /// serially so a packed word is private to one thread.
    fn bind_kv_append(dt: Dtype, batch: u32, kv_stride: u32) -> (&'static str, u32) {
        match dt {
            Dtype::F32 => (kname::PAGED_KV_APPEND_BATCHED, batch * kv_stride),
            Dtype::BF16 => (kname::PAGED_KV_APPEND_BATCHED_WORD_BF16, crate::kv_tier::append_word_threads(batch, kv_stride)),
            other => panic!(
                "Ops::kv_append_batched: no paged-KV append kernel for dtype {other:?} -- only F32/BF16 \
                 KV-cache pages are implemented (see KvPage's own doc comment)"
            ),
        }
    }

    fn bind_decode_scores(dt: Dtype) -> &'static str {
        match dt {
            Dtype::F32 => kname::PAGED_DECODE_SCORES_BATCHED,
            Dtype::BF16 => kname::PAGED_DECODE_SCORES_BATCHED_BF16,
            other => panic!(
                "Ops::decode_scores_batched: no paged-decode-scores kernel for dtype {other:?} -- only \
                 F32/BF16 KV-cache pages are implemented"
            ),
        }
    }

    fn bind_decode_apply(dt: Dtype) -> &'static str {
        match dt {
            Dtype::F32 => kname::PAGED_DECODE_APPLY_BATCHED,
            Dtype::BF16 => kname::PAGED_DECODE_APPLY_BATCHED_BF16,
            other => panic!(
                "Ops::decode_apply_batched: no paged-decode-apply kernel for dtype {other:?} -- only \
                 F32/BF16 KV-cache pages are implemented"
            ),
        }
    }

    /// Append one token's projected K (or V) per sequence in the batch into
    /// `pool` at that sequence's `(blocks[b], offsets[b])` -
    /// `paged_kv_append_batched`'s contract exactly, but `pool` is now a
    /// [`KvPage`] instead of a raw [`DeviceBuffer`], so the SAME call works
    /// whether `pool` is `F32` (a plain write) or `BF16` (B9: a genuine
    /// read-modify-write pack of the shared `u32` word - see `KvPage`'s own
    /// doc comment). `src` is always f32 (the projected activation this
    /// token's K/V came from) - only the CACHE narrows, exactly like the
    /// weight-tier methods above leave the ACTIVATION at f32 and narrow only
    /// the weight/table/expert-bank side.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_batched(
        &self,
        s: &mut Vec<Step>,
        pool: &KvPage,
        src: &DeviceBuffer,
        blocks: &DeviceBuffer,
        offsets: &DeviceBuffer,
        batch: u32,
        kv_stride: u32,
        block_size: u32,
    ) {
        let (name, threads) = Self::bind_kv_append(pool.dtype(), batch, kv_stride);
        let kind = self.idx[name];
        s.push(self.gpu.step(kind, &[src, blocks, offsets, pool.buf()], &[batch, kv_stride, block_size], threads));
    }

    /// `scores[b,h,j] = (q[b,h,:] . pool_k[...]) * scale` for every sequence
    /// `b` in the batch - `paged_decode_scores_batched`'s contract, `pool_k` a
    /// [`KvPage`] instead of a raw buffer. `q` stays f32 always (an
    /// ACTIVATION, never narrowed - matching every other `Ops` method's own
    /// asymmetry between the narrowed STORAGE side and the exact activation
    /// side).
    pub fn decode_scores_batched(
        &self,
        s: &mut Vec<Step>,
        q: &DeviceBuffer,
        pool_k: &KvPage,
        block_tables: &DeviceBuffer,
        seq_lens: &DeviceBuffer,
        scores: &DeviceBuffer,
        shape: PagedDecodeShape,
    ) {
        let kind = self.idx[Self::bind_decode_scores(pool_k.dtype())];
        let threads = shape.batch * shape.n_heads * shape.cap;
        s.push(self.gpu.step(
            kind,
            &[q, pool_k.buf(), block_tables, seq_lens, scores],
            &[
                shape.batch,
                shape.n_heads,
                shape.group,
                shape.head_dim,
                shape.block_size,
                shape.kv_stride,
                shape.cap,
                shape.max_bt,
                shape.scale.to_bits(),
            ],
            threads,
        ));
    }

    /// `ctx[b,h,d] = sum_j probs[b,h,j] * pool_v[...]` for every sequence `b`
    /// in the batch - `paged_decode_apply_batched`'s contract, `pool_v` a
    /// [`KvPage`] instead of a raw buffer. `probs` stays f32 always (an
    /// activation). `shape.scale` is unused here - `decode_apply_batched`'s
    /// own kernel `Params` has no `scale` field at all (see
    /// [`PagedDecodeShape`]'s own doc comment).
    pub fn decode_apply_batched(
        &self,
        s: &mut Vec<Step>,
        probs: &DeviceBuffer,
        pool_v: &KvPage,
        block_tables: &DeviceBuffer,
        seq_lens: &DeviceBuffer,
        ctx: &DeviceBuffer,
        shape: PagedDecodeShape,
    ) {
        let kind = self.idx[Self::bind_decode_apply(pool_v.dtype())];
        let threads = shape.batch * shape.n_heads * shape.head_dim;
        s.push(self.gpu.step(
            kind,
            &[probs, pool_v.buf(), block_tables, seq_lens, ctx],
            &[
                shape.batch,
                shape.n_heads,
                shape.group,
                shape.head_dim,
                shape.block_size,
                shape.kv_stride,
                shape.cap,
                shape.max_bt,
            ],
            threads,
        ));
    }

    // --- B10: bf16 training tier - ONE kernel family (matmul's `Reference`
    // variant), forward + backward, gated behind an explicit `Weight::BF16`,
    // default OFF -----------------------------------------------------------
    //
    // Standard mixed-precision-training pattern: the WEIGHT is stored/read at
    // bf16 (a real VRAM/bandwidth win on the READ side, both directions this
    // phase touches), but the master copy that AdamW actually updates stays
    // f32 (`crates/paramstore`/`crates/optim`, untouched by this phase) and
    // the weight GRADIENT (`dW`) is always computed and accumulated in f32 -
    // never narrowed, never read back through a packed buffer. `Self::
    // matmul_dw` below enforces that second half structurally: it has no
    // `Weight`/`Dtype` parameter at all, so there is no argument a caller
    // could pass to make it read a packed buffer even by mistake.
    //
    // B10 extended `matmul.wgsl`'s `Reference` variant and its `matmul_dx.wgsl`
    // backward to bf16; the frozen-base trainer (NF4 tier, tiled bf16 dx) later
    // added the register-tiled `matmul_reg3`/`matmul_dx_reg` siblings of both
    // tiers - see `frozen_base_kernels`. `crates/gradcheck::
    // check_matmul_bf16_weight` is the bf16 correctness gate, `crates/model/
    // tests/ops_nf4_weight.rs` the NF4 one.

    /// [`Self::matmul`] for several weights that read the SAME activation, each
    /// into its own output buffer (whole, `yoff = 0`) - a decode layer's qkv/b/a/z,
    /// q/k/v or gate/up projections.
    ///
    /// Where the device is offered `matmul_i8_gemv_multi` (`matmul_q4_gemv_multi`)
    /// and every weight is a standard int8 (q4) one that [`Self::matmul`] would
    /// run as the decode GEMV, the
    /// whole group is ONE launch whose blocks fill the card together instead of
    /// one launch per matrix, each with an empty last wave (the 48-row b and a
    /// projections run six blocks). Each output element is the one a separate
    /// launch produces, bit for bit - the kernel runs the single GEMV's own block
    /// function. Anything else - another tier, a batch past the kernel's tile,
    /// more than four weights - is a [`Self::matmul`] per weight, exactly as
    /// before.
    pub fn matmul_group(&self, s: &mut Vec<Step>, group: &[(&Weight, &DeviceBuffer)], act: &Act) {
        if let Some(step) = self.fused_group(group, act) {
            s.push(step);
            return;
        }
        for (w, y) in group {
            self.matmul(s, w, act, y, 0);
        }
    }

    /// The one-launch form of [`Self::matmul_group`], or `None` where the group
    /// is not eligible for it.
    fn fused_group(&self, group: &[(&Weight, &DeviceBuffer)], act: &Act) -> Option<Step> {
        if !(2..=4).contains(&group.len()) || act.xr0 != 0 {
            return None;
        }
        if matches!(group[0].0, Weight::BF16 { .. }) {
            return self.fused_bf16_group(group, act);
        }
        let quant = act.quant.borrow().clone()?;
        let (m, k) = (act.m, act.k);
        // Every weight of the group must be the same tier, run as that tier's
        // decode GEMV; the multi kernel takes the single GEMV's own `k` word.
        let (fused, gemv, kparam) = match group[0].0 {
            Weight::I8 { .. } => (gpu_core::Fused::I8GemvMulti, kname::MATMUL_I8_GEMV, k / 4),
            Weight::Q4 { .. } => (gpu_core::Fused::Q4GemvMulti, kname::MATMUL_Q4_GEMV, k),
            _ => return None,
        };
        let mut params = vec![m, kparam];
        for (w, _) in group {
            let (Weight::I8 { n, k: wk, .. } | Weight::Q4 { n, k: wk, .. }) = w else { return None };
            if *wk != k || w.group() != 32 || self.matmul_kernel(w, m) != gemv {
                return None;
            }
            // A member the tuned table puts on a dequantize-in-register candidate runs as its own product.
            if matches!(w, Weight::I8 { .. }) && self.providers.runs("cuda") && self.gpu.int8_linear_bf16_input(&[m, k / 4, *n]) {
                return None;
            }
            params.push(*n);
        }
        // Unused sets are bound to buffers the kernel never touches (they own no
        // blocks); each is its own one-word allocation because the facade refuses
        // a dispatch whose last binding appears twice.
        let unused: Vec<DeviceBuffer> = (group.len()..4).map(|_| self.gpu.storage(1)).collect();
        params.resize(6, 0);
        let (xq, sx) = (quant.xq_for(k), &quant.sx);
        let mut bufs: Vec<&DeviceBuffer> = vec![xq, sx];
        for i in 0..4 {
            match group.get(i) {
                Some((Weight::I8 { w, s, .. } | Weight::Q4 { w, s, .. }, y)) => bufs.extend([w, s, *y]),
                _ => bufs.extend([xq, sx, &unused[i - group.len()]]),
            }
        }
        self.gpu.fused_step(fused, &bufs, &params)
    }

    /// [`Self::fused_group`] for dense bf16 weights: the decode GEMV's own conditions (rows, `k`), every member the same `k`.
    fn fused_bf16_group(&self, group: &[(&Weight, &DeviceBuffer)], act: &Act) -> Option<Step> {
        let (m, k) = (act.m, act.k);
        if m == 0 || m > self.dense_gemv_max_rows() || !k.is_multiple_of(8) {
            return None;
        }
        let mut params = vec![m, k];
        for (w, _) in group {
            let Weight::BF16 { n, k: wk, .. } = w else { return None };
            if *wk != k {
                return None;
            }
            params.push(*n);
        }
        // Unused sets are bound to buffers the kernel never touches (they own no blocks); each is its own one-word
        // allocation because the facade refuses a dispatch whose last binding appears twice.
        let unused: Vec<DeviceBuffer> = (group.len()..4).map(|_| self.gpu.storage(1)).collect();
        params.resize(6, 0);
        let mut bufs: Vec<&DeviceBuffer> = vec![&act.x];
        for i in 0..4 {
            match group.get(i) {
                Some((Weight::BF16 { w, .. }, y)) => bufs.extend([w, *y]),
                _ => bufs.extend([&act.x, &unused[i - group.len()]]),
            }
        }
        self.gpu.fused_step(gpu_core::Fused::Bf16GemvMulti, &bufs, &params)
    }

    /// `dX[m, k] = sum_n dY[m, n] * W[n, k]` (`accumulate` selects overwrite
    /// vs add, matching `matmul_dx.wgsl`'s own `Params.accumulate` field) -
    /// the gradient w.r.t. the ACTIVATION input of a linear whose weight is
    /// `w`: the frozen-base training path, which reads the SAME weight copy
    /// the forward multiplied by, so `dX` stays consistent with it.
    ///
    /// `w` may be `Weight::F32`, `Weight::BF16` (B10: inline bitcast decode,
    /// `dtype_variant`) or any quantized tier (decoded in bounded row tiles, see
    /// the `tiled` module; no transposed copy exists). A
    /// result that fills 128x128 tiles runs the register-tiled `matmul_dx_reg`
    /// kernel of the tier on a device that executes workgroup barriers, the
    /// naive `matmul_dx` otherwise; the bf16 tiled kernel is optional (see
    /// [`frozen_base_kernels`]) and a Gpu without it keeps the naive one. `w` is
    /// never written. Other tiers panic: their backward-through-the-weight is
    /// not implemented.
    pub fn matmul_dx(&self, s: &mut Vec<Step>, w: &Weight, dy: &DeviceBuffer, m: u32, dx: &DeviceBuffer, accumulate: bool) {
        let (n, k) = (w.n(), w.k());
        if matches!(self.linear_plan(w, n), LinearPlan::TileDecode { .. }) {
            return self.dx_tiled(s, w, (0, n), dy, m, dx, 0, accumulate);
        }
        if self.library_dx(s, w, (0, n), dy, m, dx, 0, accumulate) {
            return;
        }
        let dt = w.dtype();
        let params = [m, k, n, accumulate as u32];
        // Tiled when the `[m, k]` result fills tiles (see `Self::tiled_gemm`);
        // the bf16 tiled kernel is optional so a model that registered only
        // the naive one keeps working.
        let tiled = self.tiled_gemm(m, k);
        match w {
            Weight::BF16 { .. } if tiled && self.gpu.kernel_index(kname::MATMUL_DX_REG_BF16).is_some() => {
                self.matmul_dx_rows(s, w, (0, n), dy, m, dx, 0, accumulate)
            }
            Weight::F32 { w: wb, .. } | Weight::BF16 { w: wb, .. } => {
                let kname = match dt {
                    Dtype::F32 => kname::MATMUL_DX,
                    _ => kname::MATMUL_DX_BF16,
                };
                let bind = |_: KernelVariant| -> (usize, &'static str) { (self.idx[kname], kname) };
                let operands = [
                    // `dY`, the incoming gradient: the per-sample operand of
                    // this dispatch, in the position the forward pass's
                    // activation occupies. `Role` has no gradient spelling,
                    // and inventing one would only rename this.
                    Operand { role: Role::Act, buf: dy, range: WHOLE, dtype: Dtype::F32 },
                    Operand { role: Role::Weight, buf: wb, range: WHOLE, dtype: dt },
                    Operand { role: Role::Out, buf: dx, range: WHOLE, dtype: Dtype::F32 },
                ];
                let req = OpRequest {
                    op: Op::MatMulDx,
                    shape: OpShape { m, n, k, dtype: dt },
                    pass: Pass::Backward,
                    operands: &operands,
                    attrs: &params,
                    group: 32,
                    bind: &bind,
                };
                let mut ctx = LowerCtx { gpu: &self.gpu, caps: &self.caps, steps: s, capture: false };
                self.providers.dispatch(&mut ctx, &req);
            }
            Weight::F16 { .. } | Weight::I8 { .. } | Weight::NF4 { .. } | Weight::Q4 { .. } | Weight::KQuant { .. } | Weight::FP8 { .. } | Weight::Micro { .. } => {
                unreachable!("{dt:?} runs through the tile decode above")
            }
        }
    }

    /// Whether a `[m, n]` GEMM output fills enough 128x128 tiles for the
    /// register-tiled kernels to be the right choice on this device (see
    /// [`Self::matmul_rows`]); exposed so a caller dispatching a sibling kernel
    /// of its own (the weight gradient) follows the same rule.
    #[must_use]
    pub fn fills_tile(&self, m: u32, n: u32) -> bool {
        self.tiled_gemm(m, n)
    }

    /// [`Self::matmul`] against the row window `rows = (first, count)` of `w`
    /// (`[n, k]`): `y[m, count] = act @ W[first..first+count, :]^T`. This is the
    /// vocab-tile primitive of a head whose weight exceeds one storage binding.
    /// Reads the f32 rows of `act` (never quantised), so only the float-read
    /// tiers exist: `F32`, `BF16`, `NF4`. The window start must be a legal
    /// 256-byte binding offset of every buffer it slices (rows of
    /// `lcm(64, ...)` words: 2048 elements of the first row for NF4's scale
    /// plane), which is asserted. Dispatches the register-tiled kernel where
    /// [`Self::fills_tile`] says so, else the naive one.
    pub fn matmul_rows(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), act: &Act, y: &DeviceBuffer, yoff: u64) {
        let (first, count) = rows;
        let (n, k, m) = (w.n(), w.k(), act.m);
        assert_eq!(act.k, k, "Ops::matmul_rows: activation width {} does not match weight K {k}", act.k);
        assert!(count > 0 && first as u64 + count as u64 <= n as u64, "Ops::matmul_rows: rows {first}+{count} outside the {n} rows of the weight");
        if matches!(self.linear_plan(w, count), LinearPlan::TileDecode { .. }) {
            if self.fp8_forward(s, w, rows, act, y, yoff) || self.small_m_gemv(s, w, rows, act, y, yoff) {
                return;
            }
            return self.forward_tiled(s, w, rows, act, y, yoff);
        }
        if self.dense_small_m_gemv(s, w, rows, act, y, yoff) || self.library_forward(s, w, rows, act, y, yoff) {
            return;
        }
        let tiled = self.tiled_gemm(m, count);
        let grid = if tiled { gpu_core::Dispatch::Workgroups(m.div_ceil(128) * count.div_ceil(128)) } else { gpu_core::Dispatch::Threads(m * count) };
        let xo = (act.xr0 as u64 * k as u64, m as u64 * k as u64);
        let oo = (yoff, m as u64 * count as u64);
        let params = [m, k, count];
        let (name, bufs, ranges): (&str, Vec<&DeviceBuffer>, Vec<(u64, u64)>) = match w {
            Weight::F32 { w: wb, .. } => {
                (if tiled { kname::MATMUL_REG2 } else { kname::MATMUL }, vec![&act.x, wb, y], vec![xo, Self::row_window(first, count, n, k, 1), oo])
            }
            Weight::BF16 { w: wb, .. } => {
                (if tiled { kname::MATMUL_REG3_BF16 } else { kname::MATMUL_BF16 }, vec![&act.x, wb, y], vec![xo, Self::row_window(first, count, n, k, 2), oo])
            }
            other => unreachable!("Ops::matmul_rows: {:?} runs through the tile decode above", other.dtype()),
        };
        s.push(self.gpu.dispatch_sliced(self.idx[name], &bufs, &ranges, &params, grid));
    }

    /// [`Self::matmul_dx`] against the row window `rows = (first, count)` of
    /// `w`: `dx[dx_row0 .. dx_row0 + m, :] (+)= dy[m, count] @ W[first.., :]`,
    /// `dy` dense `[m, count]`. With `accumulate` set the window's contribution
    /// is added, which is how the passes over the vocab tiles of a head sum into
    /// one `dX`. Same tiers, window rules and kernel choice as
    /// [`Self::matmul_rows`]; the tiled `matmul_dx_reg` kernels of the F32 and
    /// bf16 tiers are optional registrations (see [`frozen_base_kernels`]) and
    /// their absence keeps the naive kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dx_rows(&self, s: &mut Vec<Step>, w: &Weight, rows: (u32, u32), dy: &DeviceBuffer, m: u32, dx: &DeviceBuffer, dx_row0: u32, accumulate: bool) {
        let (first, count) = rows;
        let (n, k) = (w.n(), w.k());
        assert!(count > 0 && first as u64 + count as u64 <= n as u64, "Ops::matmul_dx_rows: rows {first}+{count} outside the {n} rows of the weight");
        if matches!(self.linear_plan(w, count), LinearPlan::TileDecode { .. }) {
            return self.dx_tiled(s, w, rows, dy, m, dx, dx_row0, accumulate);
        }
        if self.library_dx(s, w, rows, dy, m, dx, dx_row0, accumulate) {
            return;
        }
        let tiled_wanted = self.tiled_gemm(m, k);
        let dxo = if dx_row0 == 0 { WHOLE } else { (dx_row0 as u64 * k as u64, m as u64 * k as u64) };
        let params = [m, k, count, accumulate as u32];
        let (naive, tiled, bufs, ranges): (&str, &str, Vec<&DeviceBuffer>, Vec<(u64, u64)>) = match w {
            Weight::F32 { w: wb, .. } => {
                (kname::MATMUL_DX, kname::MATMUL_DX_REG, vec![dy, wb, dx], vec![WHOLE, Self::row_window(first, count, n, k, 1), dxo])
            }
            Weight::BF16 { w: wb, .. } => {
                (kname::MATMUL_DX_BF16, kname::MATMUL_DX_REG_BF16, vec![dy, wb, dx], vec![WHOLE, Self::row_window(first, count, n, k, 2), dxo])
            }
            other => unreachable!("Ops::matmul_dx_rows: {:?} runs through the tile decode above", other.dtype()),
        };
        let registered = |name: &str| self.gpu.kernel_index(name);
        let (kind, tile) = match (tiled_wanted, registered(tiled)) {
            (true, Some(kind)) => (kind, true),
            _ => (registered(naive).unwrap_or_else(|| self.frozen_base_kernel(naive)), false),
        };
        let grid = if tile { gpu_core::Dispatch::Workgroups(m.div_ceil(128) * k.div_ceil(128)) } else { gpu_core::Dispatch::Threads(m * k) };
        s.push(self.gpu.dispatch_sliced(kind, &bufs, &ranges, &params, grid));
    }

    /// The binding range of rows `first..first+count` of an `[n, k]` weight
    /// stored `per_word` elements to a 4-byte word (or, for NF4's scale plane,
    /// one word per 32): the whole buffer for the full window, else a sub-range
    /// that must start on a 256-byte (64-word) boundary.
    fn row_window(first: u32, count: u32, n: u32, k: u32, per_word: u64) -> (u64, u64) {
        if first == 0 && count == n {
            return WHOLE;
        }
        let (start, len) = (first as u64 * k as u64, count as u64 * k as u64);
        assert!(start % per_word == 0 && len % per_word == 0, "weight row window {first}+{count} of width {k} is not a whole number of words");
        assert!((start / per_word).is_multiple_of(64), "weight row window starting at row {first} (width {k}) is not a 256-byte binding offset");
        (start / per_word, len / per_word)
    }

    /// `dW[n, k] += sum_m dY[m, n] * X[m, k]` - the gradient w.r.t. the
    /// WEIGHT of a linear, dispatched through the plain f32 `matmul_dw.wgsl`
    /// kernel UNCONDITIONALLY. There is no `Weight`/`Dtype` parameter on this
    /// method at all, by design: `matmul_dw.wgsl` never reads `w` (its only
    /// inputs are `dy` and `x`, both always-f32 activations), so a weight's
    /// storage tier has NO EFFECT on this computation - narrowing it further
    /// would be a change to a kernel that structurally cannot express one.
    /// This is B10's "dW's accumulation is ALWAYS f32" invariant enforced by
    /// the type signature, not by a caller's discipline. Accumulates
    /// (matching `matmul_dw.wgsl`'s own always-add contract) - a caller
    /// zeroes `dw` once before a backward pass, exactly like every other `dw`
    /// kernel in this codebase already does (e.g. via `Gpu::submit`'s
    /// `clears` list).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dw(&self, s: &mut Vec<Step>, x: &DeviceBuffer, dy: &DeviceBuffer, m: u32, n: u32, k: u32, dw: &DeviceBuffer) {
        let bind = |_: KernelVariant| -> (usize, &'static str) { (self.idx[kname::MATMUL_DW], kname::MATMUL_DW) };
        let operands = [
            Operand { role: Role::Act, buf: dy, range: WHOLE, dtype: Dtype::F32 },
            // `X`, the forward activation. A second always-f32 input rather
            // than a weight - `matmul_dw` never reads a weight at all, which
            // is what makes this Op's accumulation f32 by construction.
            Operand { role: Role::Aux(0), buf: x, range: WHOLE, dtype: Dtype::F32 },
            // `dW` is ACCUMULATED into, never overwritten (`matmul_dw.wgsl`'s
            // own always-add contract) - the one operand in this façade whose
            // role is `Accum` rather than `Out`... except that
            // `WgslProvider::lower` requires the LAST operand to be the
            // output, since that is the bind position `Gpu::step_sliced`
            // gives it. So `Out` it is, and the accumulate semantics stay
            // where they already were: in the kernel.
            Operand { role: Role::Out, buf: dw, range: WHOLE, dtype: Dtype::F32 },
        ];
        let attrs = [m, k, n];
        let req = OpRequest {
            op: Op::MatMulDw,
            // `dtype` is `F32` unconditionally and is NOT a weight tier here:
            // this Op has no weight (see `Op::MatMulDw`'s own doc), so there
            // is no storage tier for a shape to carry.
            shape: OpShape { m, n, k, dtype: Dtype::F32 },
            pass: Pass::Backward,
            operands: &operands,
            attrs: &attrs,
            group: 32,
            bind: &bind,
        };
        let mut ctx = LowerCtx { gpu: &self.gpu, caps: &self.caps, steps: s, capture: false };
        self.providers.dispatch(&mut ctx, &req);
    }

    /// Which kernel [`Self::matmul`] would dispatch for this weight at `m`
    /// rows - the selector's REAL choice, resolved through the same
    /// [`Self::bind`] the dispatch itself uses.
    ///
    /// Exists so a caller that has to *report* what ran (a device benchmark,
    /// a `--explain` flag) reads it off the one selection path instead of
    /// re-deriving the policy and drifting from it. Returning the kernel name
    /// rather than the [`KernelVariant`] is deliberate: the variant alone is
    /// ambiguous (`PackedInt8` is `matmul_i8_dyn` for `I8` but
    /// `matmul_q4_dyn_reg` for `Q4` - two different physical kernel files
    /// behind the same variant, see [`Ops::threads`]'s doc comment), and the
    /// name is what a reader can grep for.
    #[must_use]
    pub fn matmul_kernel(&self, w: &Weight, m: u32) -> &'static str {
        let shape = OpShape { m, n: w.n(), k: w.k(), dtype: w.dtype() };
        Self::bind(self.selector.select(Op::MatMul, shape, &self.caps), w.dtype(), w.group())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matches plain sequential concatenation for an arbitrary set of part
    /// lengths, including ones that straddle `concat_rows_parallel`'s own
    /// chunk boundary (`CHUNK = 1 << 16`) - a part shorter than, equal to,
    /// and spanning several chunks all in the same call.
    #[test]
    fn concat_rows_parallel_matches_sequential_concat_across_chunk_boundaries() {
        let a: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..(1 << 16) + 10).map(|i| i as f32 * 0.5).collect();
        let c: Vec<f32> = (0..3).map(|i| 100.0 + i as f32).collect();
        let got = concat_rows_parallel(&[&a, &b, &c]);
        let mut want = Vec::new();
        want.extend_from_slice(&a);
        want.extend_from_slice(&b);
        want.extend_from_slice(&c);
        assert_eq!(got, want);
    }

    #[test]
    fn concat_rows_parallel_of_nothing_is_empty() {
        assert_eq!(concat_rows_parallel(&[]), Vec::<f32>::new());
    }

    /// `Ops::bind`'s `(PackedInt8, Dtype::Q4)` arm must resolve to the M5.5
    /// register-tiled kernel, `matmul_q4_dyn_reg` - NOT the naive
    /// one-thread-per-output `matmul_q4_dyn` this arm used to bind (see this
    /// module's own kernel-performance ledger entry: faster at every
    /// measured shape, the gain growing with m, and bit-identical to `matmul_q4_dyn`,
    /// `crates/model/tests/matmul_q4_gemm.rs::
    /// matmul_q4_dyn_reg_matches_fp32_oracle_and_is_bit_identical_to_dyn`).
    /// The sibling `(PackedInt8, Dtype::I8)` arm already binds its own tiled
    /// kernel (`matmul_i8_dyn`); this pins that `Q4` now gets the same
    /// treatment instead of staying the one dtype still on the naive tier.
    #[test]
    fn bind_packed_int8_q4_dispatches_the_register_tiled_kernel_not_the_naive_one() {
        assert_eq!(
            Ops::bind(KernelVariant::PackedInt8, Dtype::Q4, 32),
            kname::MATMUL_Q4_DYN_REG,
            "Ops::bind(PackedInt8, Q4) must dispatch matmul_q4_dyn_reg (M5.5's register-tiled kernel), \
             not the naive matmul_q4_dyn"
        );
    }

    /// [`Ops::with_selector`]'s whole reason to exist: `Ops::matmul` must
    /// consult the INJECTED selector, not the fixed internal
    /// `CachedSelector<DefaultSelector>` [`Ops::new`] builds. `Flip` is
    /// constructed to disagree with `DefaultSelector` at this exact
    /// `(op, shape, caps)` WHATEVER `testgpu::dev`'s real capabilities are
    /// (deliberately not hard-coding a shape assumed to cross some
    /// capability threshold - that would make the test's own correctness
    /// depend on facts about the test backend it never asserts), so a kernel
    /// index difference between the two `Ops` here can only come from the
    /// selector actually being asked. Compares `Step::meta().kernel`
    /// (queued, never submitted) rather than executing the dispatch - this
    /// only needs to prove which kernel was CHOSEN, not that every
    /// `KernelVariant` runs correctly on this backend.
    #[test]
    fn matmul_dispatches_via_the_injected_selector_not_the_fixed_default() {
        struct Flip;
        impl KernelSelector for Flip {
            fn select(&self, op: Op, shape: OpShape, caps: &DeviceCaps) -> KernelVariant {
                match select::DefaultSelector.select(op, shape, caps) {
                    KernelVariant::Reference => KernelVariant::WorkgroupPerOutput,
                    _ => KernelVariant::Reference,
                }
            }
        }

        let (m, n, k) = (4u32, 8u32, 8u32);
        let w_h = vec![0f32; (n * k) as usize];

        // The selector governs the WGSL kernels; the vendor-library GEMM
        // (the default tensor-core tier on CUDA) answers the F32 product
        // before any selector is asked, so pin the fp32 reference tier here.
        let dispatch = |ops: &mut Ops| -> usize {
            ops.set_gemm_precision(GemmPrecision::Fp32Reference);
            let x = ops.gpu().storage((m * k) as u64);
            let weight = Weight::upload(ops, &w_h, n as usize, k as usize, Dtype::F32);
            let act = ops.act_f32(&x, 0, m, k);
            let y = ops.gpu().storage((m * n) as u64);
            let mut steps = Vec::new();
            ops.matmul(&mut steps, &weight, &act, &y, 0);
            steps[0].meta().expect("Ops::matmul's step carries facade meta").kernel
        };

        // Reference provider only: this is about the selector reaching the
        // dispatch, and a native provider (CUDA's tiled GEMM) answers for the
        // plain F32 shape regardless of what a selector says.
        let mut ops_default = Ops::with_providers(
            gpu_core::testgpu::dev(kernel_list()),
            ProviderRegistry::reference(Arc::new(select::CachedSelector::new(select::DefaultSelector))),
        )
        .expect("Ops::with_providers");
        let mut ops_flip = Ops::with_providers(gpu_core::testgpu::dev(kernel_list()), ProviderRegistry::reference(Arc::new(Flip)))
            .expect("Ops::with_providers");

        assert_ne!(
            dispatch(&mut ops_default),
            dispatch(&mut ops_flip),
            "Ops::with_selector's injected selector must be consulted by Ops::matmul, not ignored \
             in favour of Ops::new's fixed CachedSelector<DefaultSelector>"
        );
    }

    #[test]
    fn required_kernels_matches_kname_all() {
        // `REQUIRED_KERNELS` and the canonical [`kernel_list`] must name the
        // exact same set -- if `Ops::new`'s check and what a real caller
        // registers on its `Gpu` ever disagree, this is where it would first
        // show up.
        let mut want: Vec<&str> = kernel_list().iter().map(|(n, _)| *n).collect();
        let mut got: Vec<&str> = REQUIRED_KERNELS.to_vec();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(want, got);
    }

    /// [`assert_kernel_list_complete`] itself, exercised against this
    /// module's own `kernel_list()` - the shared helper every OTHER
    /// `Ops`-building call site's test suite also calls (`qwen3::model`,
    /// `qwen3::serve`, `gradcheck::bf16_train`) must at minimum accept the
    /// canonical, known-complete list this module maintains.
    #[test]
    fn assert_kernel_list_complete_accepts_the_canonical_list() {
        assert_kernel_list_complete(kernel_list());
    }

    /// [`assert_kernel_list_complete`] must reject a list missing even one
    /// required kernel, and name it in the panic message - the exact
    /// property that would have caught `qwen3::serve::ops_kernel_list`'s
    /// 15-kernel gap at `cargo test` time instead of on a live server's
    /// first request.
    #[test]
    #[should_panic(expected = "embed")]
    fn assert_kernel_list_complete_rejects_a_list_missing_embed() {
        let without_embed: Vec<(&str, &str)> =
            kernel_list().iter().filter(|(n, _)| *n != kname::EMBED).copied().collect();
        assert_kernel_list_complete(&without_embed);
    }

    /// `kname`'s bf16/f16 name literals are plain consts (see that module's
    /// doc comment for why), not computed via `dtype_variant` -- this is the
    /// one place their spelling is checked against what `dtype_variant`
    /// actually produces for the REAL kernel sources, so the two can never
    /// silently drift apart.
    #[test]
    fn bf16_and_f16_kname_literals_match_dtype_variant_naming() {
        let (n, _) = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
        assert_eq!(n, kname::MATMUL_BF16);
        let (n, _) =
            kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
        assert_eq!(n, kname::MATMUL_GEMV_BF16);
        let (n, _) =
            kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
        assert_eq!(n, kname::MATMUL_REG3_BF16);

        let (n, _) = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::F16).unwrap();
        assert_eq!(n, kname::MATMUL_F16);
        let (n, _) =
            kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::F16).unwrap();
        assert_eq!(n, kname::MATMUL_GEMV_F16);
        let (n, _) =
            kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::F16).unwrap();
        assert_eq!(n, kname::MATMUL_REG3_F16);
    }

    /// The B8 kname literals (`embed`/`moe_linear_gated` storage tiers), same
    /// contract as [`bf16_and_f16_kname_literals_match_dtype_variant_naming`]
    /// above - pinned against the real kernel sources so `kname`'s plain
    /// string literals can never silently drift from what `dtype_variant`
    /// actually produces.
    #[test]
    fn b8_kname_literals_match_dtype_variant_naming() {
        let (n, _) = kernels::template::dtype_variant("embed", kernels::EMBED, "emb", Dtype::BF16).unwrap();
        assert_eq!(n, kname::EMBED_BF16);
        let (n, _) = kernels::template::dtype_variant("embed", kernels::EMBED, "emb", Dtype::F16).unwrap();
        assert_eq!(n, kname::EMBED_F16);
        let (n, _) =
            kernels::template::dtype_variant("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::BF16)
                .unwrap();
        assert_eq!(n, kname::MOE_LINEAR_GATED_BF16);
        let (n, _) =
            kernels::template::dtype_variant("moe_linear_gated", kernels::MOE_LINEAR_GATED, "w", Dtype::F16)
                .unwrap();
        assert_eq!(n, kname::MOE_LINEAR_GATED_F16);
    }

    /// The frozen-base kname literals are exactly the names `frozen_base_kernels`
    /// registers, so a dispatch by `kname` can never miss its kernel.
    #[test]
    fn frozen_base_kname_literals_match_the_registered_variants() {
        let registered: Vec<&str> = frozen_base_kernels().iter().map(|(n, _)| *n).collect();
        let want = [kname::MATMUL_DX_REG_BF16];
        for n in want {
            assert!(registered.contains(&n), "{n} is not registered by frozen_base_kernels()");
        }
    }

    /// The B9 kname literals (paged-KV append/scores/apply bf16 tiers), same
    /// contract as [`b8_kname_literals_match_dtype_variant_naming`] above -
    /// pinned against the real kernel sources, including the WRITE-direction
    /// `dtype_variant_store` naming (append), not just the READ-direction
    /// `dtype_variant` naming (scores/apply).
    #[test]
    fn b9_kname_literals_match_dtype_variant_naming() {
        let (n, _) = kernels::template::dtype_variant_store(
            "paged_kv_append_batched_word",
            kernels::PAGED_KV_APPEND_BATCHED_WORD,
            "pool",
            Dtype::BF16,
        )
        .unwrap();
        assert_eq!(n, kname::PAGED_KV_APPEND_BATCHED_WORD_BF16);
        let (n, _) = kernels::template::dtype_variant(
            "paged_decode_scores_batched",
            kernels::PAGED_DECODE_SCORES_BATCHED,
            "pool_k",
            Dtype::BF16,
        )
        .unwrap();
        assert_eq!(n, kname::PAGED_DECODE_SCORES_BATCHED_BF16);
        let (n, _) = kernels::template::dtype_variant(
            "paged_decode_apply_batched",
            kernels::PAGED_DECODE_APPLY_BATCHED,
            "pool_v",
            Dtype::BF16,
        )
        .unwrap();
        assert_eq!(n, kname::PAGED_DECODE_APPLY_BATCHED_BF16);
    }

    /// The B10 kname literal (`matmul_dx`'s bf16-weight-read backward tier),
    /// same contract as [`b9_kname_literals_match_dtype_variant_naming`]
    /// above. `matmul_dw` has no literal to pin here - it has no bf16 name at
    /// all (see `kname::MATMUL_DW`'s own doc comment).
    #[test]
    fn b10_kname_literal_matches_dtype_variant_naming() {
        let (n, _) = kernels::template::dtype_variant("matmul_dx", kernels::MATMUL_DX, "w", Dtype::BF16).unwrap();
        assert_eq!(n, kname::MATMUL_DX_BF16);
    }

    /// The M12 kname literals (affine K-quant `CODE_BITS` specialisations,
    /// group=16 `QPG`/`WPG` specialisations) - same contract as the B8/B9/B10
    /// tests above, but pinned against `kernels::template::interned` (a
    /// plain tunable-const specialisation) rather than `dtype_variant` (a
    /// storage-tier decode rewrite): `interned`'s `variant_name` is `"{base}
    /// #{k}={v}"`, the same `#k=v` convention `dtype_variant` also follows,
    /// so the two naming schemes agree even though the underlying rewrite
    /// differs.
    #[test]
    fn m12_kname_literals_match_interned_naming() {
        let (n, _) = kernels::template::interned("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", 4)]).unwrap();
        assert_eq!(n, kname::MATMUL_KQ_DYN_4);
        let (n, _) = kernels::template::interned("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", 8)]).unwrap();
        assert_eq!(n, kname::MATMUL_KQ_DYN_8);
        let (n, _) = kernels::template::interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", 4)]).unwrap();
        assert_eq!(n, kname::MATMUL_KQ_GEMV_4);
        let (n, _) = kernels::template::interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", 8)]).unwrap();
        assert_eq!(n, kname::MATMUL_KQ_GEMV_8);
        let (n, _) = kernels::template::interned("matmul_i8_dyn", kernels::MATMUL_I8_DYN, &[("QPG", 1)]).unwrap();
        assert_eq!(n, kname::MATMUL_I8_DYN_G16);
        let (n, _) = kernels::template::interned("matmul_i8_gemv", kernels::MATMUL_I8_GEMV, &[("WPG", 4)]).unwrap();
        assert_eq!(n, kname::MATMUL_I8_GEMV_G16);
    }

    /// `Ops::bind`'s `(variant, dtype, group)` table, exercised directly
    /// (pure function, no `Gpu` needed) - the M12 arms specifically, since
    /// this is the ONE place a `Q4K`/`Q8K`/`I8`-at-group=16 request turns
    /// into a physical kernel name.
    #[test]
    fn bind_routes_the_m12_dtypes_to_the_right_physical_kernel() {
        use KernelVariant::*;
        assert_eq!(Ops::bind(PackedInt8, Dtype::Q4K, 32), kname::MATMUL_KQ_DYN_4);
        assert_eq!(Ops::bind(WorkgroupPerOutput, Dtype::Q4K, 32), kname::MATMUL_KQ_GEMV_4);
        assert_eq!(Ops::bind(PackedInt8, Dtype::Q8K, 32), kname::MATMUL_KQ_DYN_8);
        assert_eq!(Ops::bind(WorkgroupPerOutput, Dtype::Q8K, 32), kname::MATMUL_KQ_GEMV_8);
        // Q6_K's group=16 stays tagged `Dtype::I8` but must resolve to the
        // SEPARATE group=16-specialised physical kernel, not the plain
        // group=32 default `matmul_i8_dyn`/`matmul_i8_gemv` names.
        assert_eq!(Ops::bind(PackedInt8, Dtype::I8, 16), kname::MATMUL_I8_DYN_G16);
        assert_eq!(Ops::bind(WorkgroupPerOutput, Dtype::I8, 16), kname::MATMUL_I8_GEMV_G16);
        // group=32 (every dtype that predates M12) is untouched.
        assert_eq!(Ops::bind(PackedInt8, Dtype::I8, 32), kname::MATMUL_I8_DYN);
        assert_eq!(Ops::bind(WorkgroupPerOutput, Dtype::I8, 32), kname::MATMUL_I8_GEMV);
    }

    /// [`KvPage::word_count`] - the actual allocation-size logic
    /// [`KvPage::zeros`] calls - is genuinely half for `BF16` at an EVEN
    /// total element count (the realistic case: every real head_dim in this
    /// tree is even), and still strictly less than `F32`'s word count
    /// (rounded up, never down) at an ODD one (this phase's own
    /// read-modify-write stress test's shape).
    #[test]
    fn bf16_kv_page_word_count_is_half_the_f32_word_count() {
        // Even total (16 blocks * 16 tokens * 128 kv_stride = 32768, even).
        let f32_words = KvPage::word_count(16, 16, 128, Dtype::F32);
        let bf16_words = KvPage::word_count(16, 16, 128, Dtype::BF16);
        assert_eq!(f32_words, 32768);
        assert_eq!(bf16_words, 16384, "bf16 must be EXACTLY half the f32 word count at an even total");
        assert_eq!(bf16_words * 4, f32_words * 4 / 2, "byte size: bf16 must be exactly half of f32's");

        // Odd total (1 block * 2 tokens * 3 kv_stride = 6, even actually --
        // use an odd kv_stride directly: 1 block * 1 token * 3 = 3, odd).
        let f32_odd = KvPage::word_count(1, 1, 3, Dtype::F32);
        let bf16_odd = KvPage::word_count(1, 1, 3, Dtype::BF16);
        assert_eq!(f32_odd, 3);
        assert_eq!(bf16_odd, 2, "div_ceil(3, 2) = 2 -- rounds UP so the trailing element still gets a word");
        assert!(bf16_odd * 4 < f32_odd * 4, "bf16 must still be strictly smaller even when not exactly halvable");
    }

    #[test]
    fn new_fails_loudly_on_a_missing_kernel() {
        // A `Gpu` missing even one required kernel must fail at `Ops::new`,
        // never silently at the first `matmul`/`act` call.
        let gpu = gpu_core::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        let err = match Ops::new(gpu) {
            Ok(_) => panic!("Ops::new should fail when required kernels are missing"),
            Err(e) => e,
        };
        assert!(err.contains("matmul_gemv") || err.contains("not registered"), "unexpected error: {err}");
    }

    #[test]
    fn new_succeeds_when_every_required_kernel_is_present() {
        let gpu = gpu_core::testgpu::dev(kernel_list());
        Ops::new(gpu).expect("Ops::new should succeed with every required kernel registered");
    }
}
