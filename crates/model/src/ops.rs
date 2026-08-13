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
//! **Scope of this phase (B3).** This module builds the façade and proves it
//! reproduces today's `dispatch.rs`/`int8.rs`/`int4.rs` numeric behaviour
//! exactly (`crates/model/tests/ops_facade_parity.rs`). It does **not**
//! migrate any model crate's call sites onto it - `crates/qwen3`,
//! `crates/flux1`, `crates/flux2` etc. are untouched; that migration is a
//! later phase (B7). `Weight` supports exactly the three tiers this repo can
//! actually dispatch today: `F32`, `I8` (DP4A, `model::int8`), `Q4` (W4A8,
//! `model::int4`). **`BF16`/`F16` `Weight` arms are deliberately NOT added
//! yet** - `DType::promote` can already report a device supports them, but no
//! kernel varies its *load* by dtype (the "kernel templater", B4/B5's job);
//! adding enum arms with no way to construct or dispatch them would be dead
//! code. `Weight::upload` asserts loudly rather than silently miscompiling if
//! ever asked for one.
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
/// `BF16`/`F16` arms are deliberately absent - see this module's doc comment.
pub enum Weight {
    F32 { w: DeviceBuffer, n: u32, k: u32 },
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
            Weight::I8 { .. } => Dtype::I8,
            Weight::Q4 { .. } => Dtype::Q4,
        }
    }

    pub fn n(&self) -> u32 {
        match self {
            Weight::F32 { n, .. } | Weight::I8 { n, .. } | Weight::Q4 { n, .. } => *n,
        }
    }

    pub fn k(&self) -> u32 {
        match self {
            Weight::F32 { k, .. } | Weight::I8 { k, .. } | Weight::Q4 { k, .. } => *k,
        }
    }

    /// The ONE upload path: quantizes/packs `raw` (`[n, k]`, row-major) per
    /// `want.promote(ops.caps().numeric)` - never narrower than what `want`
    /// asked for, never wider than what the device can execute - then
    /// uploads. A caller never hand-picks a buffer layout itself.
    ///
    /// `want` must be `F32`, `I8`, or `Q4` - asserted loudly, since `BF16`/
    /// `F16` have no `Weight` arm yet (see this module's doc comment).
    pub fn upload(ops: &Ops, raw: &[f32], n: usize, k: usize, want: Dtype) -> Weight {
        assert_eq!(raw.len(), n * k, "Weight::upload: raw len {} != n*k ({n}*{k})", raw.len());
        assert!(
            matches!(want, Dtype::F32 | Dtype::I8 | Dtype::Q4),
            "Weight::upload: {want:?} weights are not implemented yet -- BF16/F16 need the kernel \
             templater (B4/B5); see `model::ops`'s module doc comment"
        );
        match want.promote(&ops.caps.numeric) {
            Dtype::F32 => {
                let w = ops.gpu.storage_init("weight_f32", raw);
                Weight::F32 { w, n: n as u32, k: k as u32 }
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
    /// [`Weight::upload`]'s own `DType::promote` gate (an `I8`/`Q4` `Weight`
    /// exists at all only on a device whose caps already promoted it there),
    /// so a panic here means one of those two contracts broke, not a normal
    /// runtime condition.
    fn bind(v: KernelVariant, dt: Dtype) -> &'static str {
        use KernelVariant::*;
        match (v, dt) {
            (Reference, Dtype::F32 | Dtype::BF16 | Dtype::F16) => kname::MATMUL,
            (WorkgroupPerOutput, Dtype::F32 | Dtype::BF16 | Dtype::F16) => kname::MATMUL_GEMV,
            (RegisterTiled, Dtype::F32 | Dtype::BF16 | Dtype::F16) => kname::MATMUL_REG2,
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
            Weight::F32 { w: wb, .. } => {
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

    const KERNELS: &[(&str, &str)] = &[
        ("matmul", kernels::MATMUL),
        ("matmul_gemv", kernels::MATMUL_GEMV),
        ("matmul_reg2", kernels::MATMUL_REG2),
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
        ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
        ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
        ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
        ("max_abs_row", kernels::MAX_ABS_ROW),
        ("quant_pack", kernels::QUANT_PACK),
    ];

    #[test]
    fn required_kernels_matches_kname_all() {
        // `REQUIRED_KERNELS` and this test module's own `KERNELS` list must
        // name the exact same set -- if `Ops::new`'s check and a real
        // caller's `Gpu::new` kernel list ever disagree, this is where it
        // would first show up.
        let mut want: Vec<&str> = KERNELS.iter().map(|(n, _)| *n).collect();
        let mut got: Vec<&str> = REQUIRED_KERNELS.to_vec();
        want.sort_unstable();
        got.sort_unstable();
        assert_eq!(want, got);
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
        let gpu = gpu_core::testgpu::dev(KERNELS);
        Ops::new(gpu).expect("Ops::new should succeed with every required kernel registered");
    }
}
