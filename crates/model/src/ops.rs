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
//! `Ops`/`Weight` collapse that into one call: `ops.matmul(&mut s, &weight, &act, &y, yoff)`
//! - the weight's own dtype carries which kernel family it needs, and the
//! kernel NAME is resolved once, at construction, via `Gpu::kernel_index`,
//! rather than re-derived (or silently drifting) at every call site.
//!
//! **Scope of B3/B4.** B3 built the façade and proved it reproduces today's
//! `dispatch.rs`/`int8.rs`/`int4.rs` numeric behaviour exactly
//! (`crates/model/tests/ops_facade_parity.rs`) for `F32`, `I8` (DP4A,
//! `model::int8`), `Q4` (W4A8, `model::int4`). B4 adds the fourth tier,
//! `BF16` (`model::half::pack_bf16` + the `#w=bf16` kernel variants
//! `kernels::template::dtype_variant` produces - see `crates/model/tests/
//! bf16_roundtrip.rs`). **No model crate's call sites were migrated by
//! either phase** - `crates/qwen3`, `crates/flux1`, `crates/flux2` etc. are
//! untouched; that migration is a later phase (B7). **`F16` is still
//! deliberately absent** - unlike bf16 (whose 16 bits are exactly an f32's
//! top 16 bits, so the decode is a bitcast with no rounding), f16 needs real
//! 5-bit exponent re-biasing in the kernel decode expression, which is B5's
//! job, not this phase's. `Weight::upload` asserts loudly rather than
//! silently miscompiling if ever asked for one.
//!
//! **Selection.** `Ops::matmul` is the one place a shape turns into a kernel
//! choice: it builds a `select::OpShape` from the weight's own `(n, k)` and
//! the call's `m`, asks `self.selector` (a `CachedSelector<DefaultSelector>`
//! by default - the same static policy every other selector consumer in this
//! workspace starts from), and `Self::bind` maps `(KernelVariant, Dtype)` to
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

use gpu_core::select::{self, Dtype, KernelSelector, KernelVariant, Op, OpShape};
use gpu_core::{DeviceBuffer, DeviceCaps, Gpu, Step};

use crate::dispatch::I8Scratch;

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
    pub const MATMUL_Q4_GEMV: &str = "matmul_q4_gemv";
    pub const MAX_ABS_ROW: &str = "max_abs_row";
    pub const QUANT_PACK: &str = "quant_pack";
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
    pub const ALL: &[&str] = &[
        MATMUL,
        MATMUL_GEMV,
        MATMUL_REG2,
        MATMUL_I8_DYN,
        MATMUL_I8_GEMV,
        MATMUL_Q4_DYN,
        MATMUL_Q4_GEMV,
        MAX_ABS_ROW,
        QUANT_PACK,
        MATMUL_BF16,
        MATMUL_GEMV_BF16,
        MATMUL_REG3_BF16,
    ];
}

/// The kernel-name spellings [`Ops::new`] requires the `Gpu` it is built from
/// to have registered - exposed so a caller building that `Gpu`'s kernel list
/// (or a test) has one source for the exact spellings, instead of retyping
/// them and risking a silent mismatch against [`Ops::new`]'s own check.
pub const REQUIRED_KERNELS: &[&str] = kname::ALL;

/// One linear layer's resident weight, at whichever tier
/// [`Weight::upload`]'s `want.promote(caps)` landed on.
///
/// `F16` is deliberately absent - it needs real exponent re-biasing (B5), not
/// just a bit-exact reinterpretation - see this module's doc comment.
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
    /// DP4A int8: `w` packed `[n, k/4]` u32 (`model::int8::quantize_weight`'s
    /// layout), `s` the per-channel (per-row) scale `[n]`.
    I8 { w: DeviceBuffer, s: DeviceBuffer, n: u32, k: u32 },
    /// W4A8 int4: `w` packed `[n, k/8]` u32 (`model::int4::quantize_weight_q4`'s
    /// layout), `s` the per-channel scale `[n]`. Activations stay int8 - see
    /// this module's doc comment on the offset-arithmetic rule this implies.
    Q4 { w: DeviceBuffer, s: DeviceBuffer, n: u32, k: u32 },
}

impl Weight {
    pub fn dtype(&self) -> Dtype {
        match self {
            Weight::F32 { .. } => Dtype::F32,
            Weight::BF16 { .. } => Dtype::BF16,
            Weight::I8 { .. } => Dtype::I8,
            Weight::Q4 { .. } => Dtype::Q4,
        }
    }

    pub fn n(&self) -> u32 {
        match self {
            Weight::F32 { n, .. } | Weight::BF16 { n, .. } | Weight::I8 { n, .. } | Weight::Q4 { n, .. } => *n,
        }
    }

    pub fn k(&self) -> u32 {
        match self {
            Weight::F32 { k, .. } | Weight::BF16 { k, .. } | Weight::I8 { k, .. } | Weight::Q4 { k, .. } => *k,
        }
    }

    /// The ONE upload path: quantizes/packs `raw` (`[n, k]`, row-major) per
    /// `want.promote(ops.caps().numeric)` - never narrower than what `want`
    /// asked for, never wider than what the device can execute - then
    /// uploads. A caller never hand-picks a buffer layout itself.
    ///
    /// `want` must be `F32`, `BF16`, `I8`, or `Q4` - asserted loudly, since
    /// `F16` has no `Weight` arm yet (see this module's doc comment).
    pub fn upload(ops: &Ops, raw: &[f32], n: usize, k: usize, want: Dtype) -> Weight {
        assert_eq!(raw.len(), n * k, "Weight::upload: raw len {} != n*k ({n}*{k})", raw.len());
        assert!(
            matches!(want, Dtype::F32 | Dtype::BF16 | Dtype::I8 | Dtype::Q4),
            "Weight::upload: {want:?} weights are not implemented yet -- F16 needs the kernel \
             templater's real exponent re-biasing (B5); see `model::ops`'s module doc comment"
        );
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
            other => unreachable!(
                "DType::promote({want:?}) returned {other:?} -- promote() only ever returns the \
                 requested tier or F32 (never invents a third dtype), and both are handled above"
            ),
        }
    }
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
    quant: I8Scratch,
}

/// A model-facing façade over `backend_api::select`'s kernel selector,
/// `model::int8`/`model::int4`'s quantizers, and `model::dispatch::I8Scratch`.
/// See this module's doc comment.
pub struct Ops {
    gpu: Gpu,
    caps: DeviceCaps,
    idx: HashMap<&'static str, usize>,
    selector: Box<dyn KernelSelector>,
}

impl Ops {
    /// Resolves every kernel name in [`REQUIRED_KERNELS`] via
    /// `Gpu::kernel_index` ONCE, here - never at dispatch time. A missing
    /// name is an `Err` immediately, not a panic three linears deep into a
    /// forward pass.
    pub fn new(gpu: Gpu) -> Result<Ops, String> {
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
        let selector: Box<dyn KernelSelector> = Box::new(select::CachedSelector::new(select::DefaultSelector));
        Ok(Ops { gpu, caps, idx, selector })
    }

    pub fn caps(&self) -> &DeviceCaps {
        &self.caps
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
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
        let total = (xr0 + rows) as u64;
        let quant = I8Scratch::new(&self.gpu, total, total, &[k]);
        quant.quant_rows(&self.gpu, [self.idx[kname::MAX_ABS_ROW], self.idx[kname::QUANT_PACK]], s, x, xr0, xr0 + rows, k);
        Act { x: x.clone(), xr0, m: rows, k, quant }
    }

    /// `(variant, dtype) -> kernel name`. The ONLY place in this crate a
    /// kernel-name string literal is chosen by a match arm (`kname`'s own
    /// consts are the only place one is spelled at all). Any pair not listed
    /// here is unreachable given [`select::candidates`]'s own contract
    /// (`F32`-family dtypes only ever offer `{Reference, WorkgroupPerOutput,
    /// RegisterTiled}`; `I8`/`Q4` only ever offer `{WorkgroupPerOutput,
    /// PackedInt8}` - see `backend_api::select::candidates`) crossed with
    /// [`Weight::upload`]'s own `DType::promote` gate (an `I8`/`Q4`/`BF16`
    /// `Weight` exists at all only on a device whose caps already promoted it
    /// there), so a panic here means one of those two contracts broke, not a
    /// normal runtime condition.
    ///
    /// `BF16` (B4) gets its OWN kernel names, distinct from `F32`'s - unlike
    /// `F16` (still grouped with `F32`, since no `Weight::F16` exists yet to
    /// ever reach this dtype in practice), a real `Weight::BF16` buffer holds
    /// PACKED u32 words, not raw f32 values, so dispatching it through the
    /// plain f32 kernel would silently reinterpret those bit patterns as
    /// garbage f32s. `RegisterTiled`'s bf16 kernel is `matmul_reg3`, NOT the
    /// `matmul_reg2` the `F32`/`F16` arm uses - see `kname::MATMUL_REG3_BF16`'s
    /// own doc comment for why picking a different physical file per dtype
    /// for the same `KernelVariant` is already precedented (`PackedInt8`'s
    /// i8-vs-q4 split, see [`Ops::threads`]).
    fn bind(v: KernelVariant, dt: Dtype) -> &'static str {
        use KernelVariant::*;
        match (v, dt) {
            (Reference, Dtype::F32 | Dtype::F16) => kname::MATMUL,
            (Reference, Dtype::BF16) => kname::MATMUL_BF16,
            (WorkgroupPerOutput, Dtype::F32 | Dtype::F16) => kname::MATMUL_GEMV,
            (WorkgroupPerOutput, Dtype::BF16) => kname::MATMUL_GEMV_BF16,
            (RegisterTiled, Dtype::F32 | Dtype::F16) => kname::MATMUL_REG2,
            (RegisterTiled, Dtype::BF16) => kname::MATMUL_REG3_BF16,
            (WorkgroupPerOutput, Dtype::I8) => kname::MATMUL_I8_GEMV,
            (PackedInt8, Dtype::I8) => kname::MATMUL_I8_DYN,
            (WorkgroupPerOutput, Dtype::Q4) => kname::MATMUL_Q4_GEMV,
            (PackedInt8, Dtype::Q4) => kname::MATMUL_Q4_DYN,
            (v, dt) => panic!(
                "Ops::matmul: select::candidates offered {v:?} for dtype {dt:?}, which this façade \
                 has no kernel bound for -- see `Ops::bind`'s doc comment for why this should be \
                 unreachable"
            ),
        }
    }

    /// Dispatch invocation count for `(variant, dtype, m, n)` - the same
    /// formulas `model::dispatch::mm_rows_off`/`mm8_rows_off` and
    /// `model::block::pick_gemm`/`gemm_variant` already use per variant
    /// family, EXCEPT `PackedInt8`, which - unlike every other variant here -
    /// is not one fixed dispatch geometry: `matmul_i8_dyn.wgsl` is
    /// register-tiled (128×128 tile, 256-thread workgroup, `workgroup_id`-
    /// indexed - its own header: "Layout mirrors matmul_reg3"), but
    /// `matmul_q4_dyn.wgsl` is deliberately the NAIVE one-thread-per-output
    /// tier (its own header: "the correct-first, non-tiled q4 GEMM... A
    /// register-tiled `matmul_q4_dyn` ... is the documented follow-on
    /// optimization... not attempted here"). Same `KernelVariant`, two
    /// physically different kernels - the dispatch count has to follow the
    /// REAL kernel [`Ops::bind`] chose, not the logical variant name alone.
    /// Using the tile formula for `matmul_q4_dyn` under-dispatches it
    /// (leaves real output elements never written); using `m*n` for
    /// `matmul_i8_dyn` merely over-dispatches (extra workgroups whose tile
    /// index falls outside the real grid are harmless) - caught in the
    /// direction that actually corrupts output by this façade's own parity
    /// test.
    fn threads(v: KernelVariant, dt: Dtype, m: u32, n: u32) -> u32 {
        let tile = || m.div_ceil(128) * n.div_ceil(128) * 256;
        match v {
            KernelVariant::Reference => m * n,
            KernelVariant::WorkgroupPerOutput => n * 64,
            KernelVariant::RegisterTiled => tile(),
            KernelVariant::PackedInt8 => match dt {
                Dtype::I8 => tile(),
                _ => m * n,
            },
            KernelVariant::SplitReduction => {
                unreachable!("Op::MatMul's candidates() never returns SplitReduction")
            }
        }
    }

    /// `y[yoff .. yoff + m*n)] = act[xr0..xr0+m, :] @ wᵀ`, where `m` is
    /// however many rows `act` was built for ([`Ops::act`]'s `rows`). The
    /// ENTIRE selection policy lives here: build an [`OpShape`] from `w`'s
    /// own `(n, k)` and `act`'s `m`, ask the selector, [`Ops::bind`] the
    /// choice to a kernel name, look up its index, push one [`Step`].
    pub fn matmul(&self, s: &mut Vec<Step>, w: &Weight, act: &Act, y: &DeviceBuffer, yoff: u64) {
        let (n, k) = (w.n(), w.k());
        assert_eq!(act.k, k, "Ops::matmul: activation width {} does not match weight K {k}", act.k);
        let m = act.m;
        let shape = OpShape { m, n, k, dtype: w.dtype() };
        let variant = self.selector.select(Op::MatMul, shape, &self.caps);
        let kind = self.idx[Self::bind(variant, w.dtype())];
        let threads = Self::threads(variant, w.dtype(), m, n);
        match w {
            // `BF16` reads the SAME `act.x` (raw f32, never quantized - only
            // the WEIGHT narrows for this tier) with the SAME `[m, k, n]`
            // params `F32` uses; the weight buffer's own offset is always
            // `(0, 0)` for both (whole-buffer, independent of which
            // activation rows `m` covers), so only which physical kernel
            // `kind`/`Self::bind` chose differs - the packed-vs-plain layout
            // is entirely the kernel's own concern, not this dispatch site's.
            Weight::F32 { w: wb, .. } | Weight::BF16 { w: wb, .. } => {
                let xo = (act.xr0 as u64 * k as u64, m as u64 * k as u64);
                let oo = (yoff, m as u64 * n as u64);
                s.push(self.gpu.step_sliced(kind, &[&act.x, wb, y], &[xo, (0, 0), oo], &[m, k, n], threads));
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
                let so = (act.xr0 as u64, m as u64);
                let oo = (yoff, m as u64 * n as u64);
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
                let param_k = match w.dtype() {
                    Dtype::I8 => kg as u32,
                    _ => k,
                };
                s.push(self.gpu.step_sliced(
                    kind,
                    &[act.quant.xq_for(k), wb, &act.quant.sx, sw, y],
                    &[xo, (0, 0), so, (0, 0), oo],
                    &[m, param_k, n],
                    threads,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full façade kernel set, including the three bf16-storage variants
    /// (B4) - built via [`kernels::template::dtype_variant`] rather than a
    /// `const` list, since a specialised variant's source is computed, not a
    /// plain `include_str!`. `gpu_core::testgpu::dev` wants a `'static`
    /// slice, so this leaks the `Vec` once (via `OnceLock`) rather than
    /// reallocating it per call - the same "tiny working set, leaking it is
    /// fine" tradeoff `dtype_variant`'s own interning cache already makes.
    fn kernel_list() -> &'static [(&'static str, &'static str)] {
        static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
        LIST.get_or_init(|| {
            let bf16_matmul = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
            let bf16_gemv =
                kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
            let bf16_reg3 =
                kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
            vec![
                ("matmul", kernels::MATMUL),
                ("matmul_gemv", kernels::MATMUL_GEMV),
                ("matmul_reg2", kernels::MATMUL_REG2),
                ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
                ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
                ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
                ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
                ("max_abs_row", kernels::MAX_ABS_ROW),
                ("quant_pack", kernels::QUANT_PACK),
                bf16_matmul,
                bf16_gemv,
                bf16_reg3,
            ]
        })
    }

    #[test]
    fn required_kernels_matches_kname_all() {
        // `REQUIRED_KERNELS` and this test module's own `kernel_list()` must
        // name the exact same set -- if `Ops::new`'s check and a real
        // caller's `Gpu::new` kernel list ever disagree, this is where it
        // would first show up.
        let mut want: Vec<&str> = kernel_list().iter().map(|(n, _)| *n).collect();
        let mut got: Vec<&str> = REQUIRED_KERNELS.to_vec();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(want, got);
    }

    /// `kname`'s bf16 name literals are plain consts (see that module's doc
    /// comment for why), not computed via `dtype_variant` -- this is the one
    /// place their spelling is checked against what `dtype_variant` actually
    /// produces for the REAL kernel sources, so the two can never silently
    /// drift apart.
    #[test]
    fn bf16_kname_literals_match_dtype_variant_naming() {
        let (n, _) = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
        assert_eq!(n, kname::MATMUL_BF16);
        let (n, _) =
            kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
        assert_eq!(n, kname::MATMUL_GEMV_BF16);
        let (n, _) =
            kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
        assert_eq!(n, kname::MATMUL_REG3_BF16);
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
