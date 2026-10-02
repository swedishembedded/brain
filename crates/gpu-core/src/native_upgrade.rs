// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Transparent NATIVE kernel upgrades - a hand-written kernel a model inherits
//! without knowing it exists, on the devices whose backend can compile it.
//!
//! Swedish Embedded AB implements memory-bandwidth-bound inference kernels
//! for its clients. If your team needs expertise in getting quantised LLM
//! decode onto the roofline of the memory system, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! [`crate::upgrade`] redirects a registered WGSL kernel to a faster WGSL
//! sibling. This module is its second tier: where `upgrade` has already
//! resolved a row for a kernel on this device, a row here may redirect the
//! same dispatch to a registry kernel from `kernels_cuda` instead, through
//! `Backend::register_native`/`step_native`. The bar is `upgrade`'s bar -
//! identical contract (same uniform, same bindings, same output layout) and
//! identical results - so no call site sees the substitution, and the
//! results are gated on the raw bits against the WGSL tier.
//!
//! Three conditions, each a queried fact and none a backend name:
//!
//! 1. the WGSL upgrade for the same kernel is ACTIVE on this device, so the
//!    capability policy in `backend_api::select` has already said this
//!    regime wants a workgroup-per-output GEMV at all;
//! 2. the device reports a compute capability, and the registry has a kernel
//!    for this operator and weight tier at or below it;
//! 3. the backend accepts the kernel's source (`register_native` is `Some`).
//!    Every backend that cannot compile CUDA C++ answers `None`, so the
//!    redirect declines there by the same mechanism the CUDA provider does.
//!
//! A dispatch whose shape is outside what the native kernel serves keeps the
//! WGSL tier: the redirect is per dispatch, not per handle.
//!
//! `BRAIN_NO_NATIVE_KERNELS=1` pins every dispatch to the WGSL tier - the A/B
//! switch the native kernel's own measurements are taken with.

use backend_api::select::{Dtype, Op};
use backend_api::{BindKind, Backend, CudaLaunch, NativeId, NativeSpec};

/// One drop-in native replacement for a registered kernel.
pub(crate) struct Row {
    /// The kernel name models register and dispatch by index.
    pub slow: &'static str,
    /// The `kernels_cuda` registry entry's weight tier, which with
    /// [`Self::op`] names the entry.
    pub weight: Dtype,
    pub op: Op,
    /// The storage/uniform bindings of `slow`, in order - the native kernel
    /// takes the identical list.
    pub bindings: &'static [BindKind],
    /// Whether the native kernel serves a dispatch of these caller params.
    pub serves: fn(&[u32]) -> bool,
    /// Whether `slow` must also have an ACTIVE [`crate::upgrade`] row on this
    /// device (condition 1 of the module doc). A kernel with no WGSL sibling to
    /// upgrade to - there is no regime choice to defer to - says `false`.
    pub requires_wgsl_upgrade: bool,
}

/// `matmul_i8_gemv.wgsl`'s bindings: params, `xq`, `wq`, `sx`, `sw`, `out`.
const I8_GEMV_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `Params { m, kg, n }`. The kernel needs `K` to be a whole number of 32
/// element weight-scale groups (`kg % 8 == 0`, the WGSL kernel's own
/// contract). It serves the whole decode regime the WGSL GEMV does: a block
/// covers one tile of rows, and more rows take more blocks of the grid, each
/// re-streaming the weights - four passes at 32 rows is still far less than the
/// register-tiled prefill kernel's mostly idle 128x128 tiles cost there.
fn serves_i8_gemv(p: &[u32]) -> bool {
    let (m, kg, n) = match p {
        [m, kg, n, ..] => (*m, *kg, *n),
        _ => return false,
    };
    m >= 1 && m <= backend_api::select::DECODE_REGIME_MAX_ROWS && n >= 1 && kg >= 8 && kg % 8 == 0
}

/// `moe_i8_grouped.wgsl`'s bindings: params, `xq`, `sx`, `tab`, `perm`, `wq`, `sw`,
/// `out`.
const MOE_I8_GROUPED_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `Params { tiles, kg, n, xdiv, ne }`. The kernel needs `K` to be a whole number
/// of 32-element weight-scale groups (`kg % 8 == 0`, the WGSL kernel's own
/// contract) and the routing tables `ne` experts wide.
fn serves_moe_i8_grouped(p: &[u32]) -> bool {
    match p {
        [tiles, kg, n, xdiv, ne, ..] => *tiles >= 1 && *n >= 1 && *kg >= 8 && kg % 8 == 0 && *xdiv >= 1 && *ne >= 1,
        _ => false,
    }
}

pub(crate) const ROWS: &[Row] = &[
    Row { slow: "matmul_i8_gemv", weight: Dtype::I8, op: Op::MatMul, bindings: I8_GEMV_BINDINGS, serves: serves_i8_gemv, requires_wgsl_upgrade: true },
    Row {
        slow: "moe_i8_grouped",
        weight: Dtype::I8,
        op: Op::MoeExpertLinear,
        bindings: MOE_I8_GROUPED_BINDINGS,
        serves: serves_moe_i8_grouped,
        requires_wgsl_upgrade: false,
    },
];

/// A native kernel with no WGSL twin: it fuses a chain of dispatches the MODEL
/// builds, so there is nothing to redirect and a model asks for it by name -
/// [`crate::Gpu::fused_step`] - and keeps its own chain of WGSL dispatches for
/// every device that is not offered it.
///
/// The conditions are [`resolve`]'s, minus the first (there is no WGSL upgrade
/// to be active): a queried compute capability at or above the kernel's own
/// floor, a backend that accepts the source, and `BRAIN_NO_NATIVE_KERNELS`
/// unset. A shape outside [`Fused::serves`] is the caller's to route to its
/// WGSL chain, exactly like a redirected dispatch the native kernel declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fused {
    /// `add_rms_quant`: residual add + RMSNorm + per-row int8 scale + pack.
    /// Params `[d, rows, eps (f32 bits), flags]` (flags bit 0: add `b` first);
    /// bindings `a, b, w` read, `sum, xn, xq, sx` written.
    AddRmsQuant,
    /// `quant_epilogue`: an elementwise producer (`mode` 0 plain, 1 `silu(a) *
    /// b`, 2 `a * sigmoid(b)`) + per-row int8 scale + pack. Params `[k, rows,
    /// mode, 0]`; bindings `a, b` read, `y, xq, sx` written.
    QuantEpilogue,
    /// `gdn_decode`: `rows` consecutive Gated DeltaNet decode steps (conv, SiLU,
    /// L2 norm, gates, delta-rule state update, gated RMSNorm) of ONE sequence
    /// with 128-wide key and value heads and a 4-tap conv, taken in order in one
    /// launch. Params `[nkh, nvh, group, l2_eps, rms_eps, q_scale, rows, 0]`
    /// (floats as bits); `mixed`, `bproj`, `aproj`, `z` and `gated` carry one
    /// row per step. Bindings `mixed,
    /// conv_w, hist (RW), bproj, aproj, a_log, dt_bias, state (RW), z, norm_w`
    /// read, `gated` written. The head and conv shapes are the CALLER's to
    /// check - the params cannot say them.
    GdnDecode,
    /// `gdn_decode_pool`: [`Fused::GdnDecode`] for a BATCH of sequences whose
    /// recurrent state and conv window are rows of two pools: block (key head,
    /// batch row) runs the single-sequence body for one step on that row's
    /// inputs and pool row `rows[bi]`, updating the pools in place - no state is
    /// staged in or out. Params `[nkh, nvh, group, l2_eps, rms_eps, q_scale, b,
    /// 0]` (index 6 is the batch, not a step count); bindings as
    /// [`Fused::GdnDecode`] (`mixed`, `bproj`, `aproj`, `z`, `gated` carry a row
    /// per sequence; `hist` and `state` are the pools) plus `rows` read, last.
    GdnDecodePool,
    /// `gqa_decode_prep`: for each of `rows` tokens of a gated-attention layer,
    /// the `[value|gate]` split, the per-head QK RMSNorm, the partial rotary
    /// rotation and the K/V append into the paged pools. Params `[nh, nkv,
    /// head_dim, half, eps (f32 bits), block_size, rows, 0]`; bindings `q_full,
    /// k, v, q_norm, k_norm, cos, sin, blocks, offsets` read (all but the two
    /// norm gains carry one row per token), `q_out, q_gate, pool_k, pool_v`
    /// written.
    GqaDecodePrep,
    /// `matmul_i8_gemv_multi`: up to four int8 matrices that read ONE packed
    /// activation, multiplied in a single launch - each output bit-identical to
    /// the single `matmul_i8_gemv` on that matrix. Params `[m, kg, n0, n1, n2,
    /// n3]` (`n_i == 0` leaves set `i` unused); bindings `xq`, `sx`, then for
    /// each set its `wq`, `sw` (read) and `out` (written).
    I8GemvMulti,
}

/// `add_rms_quant`'s bindings: params, `a`, `b`, `w`, `sum`, `xn`, `xq`, `sx`.
const ADD_RMS_QUANT_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `quant_epilogue`'s bindings: params, `a`, `b`, `y`, `xq`, `sx`.
const QUANT_EPILOGUE_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `gdn_decode`'s bindings: params, `mixed`, `conv_w`, `hist`, `bproj`,
/// `aproj`, `a_log`, `dt_bias`, `state`, `z`, `norm_w`, `gated`.
const GDN_DECODE_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `gdn_decode_pool`'s bindings: `gdn_decode`'s, then `rows`.
const GDN_DECODE_POOL_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
];

/// `gqa_decode_prep`'s bindings: params, `q_full`, `k`, `v`, `q_norm`,
/// `k_norm`, `cos`, `sin`, `blocks`, `offsets`, `q_out`, `q_gate`, `pool_k`,
/// `pool_v`.
const GQA_DECODE_PREP_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `matmul_i8_gemv_multi`'s bindings: params, `xq`, `sx`, then `wq`, `sw`, `out`
/// for each of four sets.
const I8_GEMV_MULTI_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

impl Fused {

    /// The `kernels_cuda` registry entry's name.
    pub fn registry_name(self) -> &'static str {
        match self {
            Fused::AddRmsQuant => "add_rms_quant",
            Fused::QuantEpilogue => "quant_epilogue",
            Fused::GdnDecode => "gdn_decode",
            Fused::GdnDecodePool => "gdn_decode_pool",
            Fused::GqaDecodePrep => "gqa_decode_prep",
            Fused::I8GemvMulti => "matmul_i8_gemv_multi",
        }
    }

    fn bindings(self) -> &'static [BindKind] {
        match self {
            Fused::AddRmsQuant => ADD_RMS_QUANT_BINDINGS,
            Fused::QuantEpilogue => QUANT_EPILOGUE_BINDINGS,
            Fused::GdnDecode => GDN_DECODE_BINDINGS,
            Fused::GdnDecodePool => GDN_DECODE_POOL_BINDINGS,
            Fused::GqaDecodePrep => GQA_DECODE_PREP_BINDINGS,
            Fused::I8GemvMulti => I8_GEMV_MULTI_BINDINGS,
        }
    }

    /// Whether the kernel serves a dispatch with these `params`. A `false`
    /// means "use the WGSL chain", never an error.
    pub fn serves(self, params: &[u32]) -> bool {
        match self {
            // One 64-thread block per row keeps `d / 64` elements per thread in
            // registers: up to 80 of them (5120 wide), and a whole number of
            // int8 words.
            Fused::AddRmsQuant => matches!(params, [d, rows, _, _] if *rows >= 1 && *d >= 4 && d % 4 == 0 && *d <= 64 * 80),
            // 1024 threads per row keep `k / 1024` elements each in registers:
            // up to 18 of them.
            Fused::QuantEpilogue => matches!(params, [k, rows, mode, _] if *rows >= 1 && *k >= 4 && k % 4 == 0 && *k <= 1024 * 18 && *mode <= 2),
            // A block is one key head's `group` value heads, 128 threads each
            // (3 is what one SM's registers hold at 128 live state words).
            Fused::GdnDecode => {
                matches!(params, [nkh, nvh, group, _, _, _, rows, _] if *nkh >= 1 && (1..=3).contains(group) && *nvh == nkh * group && (1..=8).contains(rows))
            }
            Fused::GdnDecodePool => matches!(params, [nkh, nvh, group, _, _, _, b, _] if *nkh >= 1 && (1..=3).contains(group) && *nvh == nkh * group && *b >= 1),
            // A head is one 256-thread block holding up to 512 values; the
            // rotated span `2 * half` has to fit inside it.
            Fused::GqaDecodePrep => {
                matches!(params, [nh, nkv, hd, half, _, _, rows, _] if *nh >= 1 && *nkv >= 1 && nh % nkv == 0 && (2..=512).contains(hd) && *half >= 1 && 2 * half <= *hd && *rows >= 1)
            }
            // The single GEMV's own contract for every matrix: one tile of x
            // rows per weight pass and K a whole number of 32-element groups.
            Fused::I8GemvMulti => {
                let rows = kernels_cuda::get("matmul_i8_gemv").map_or(0, |k| k.tile.0);
                matches!(params, [m, kg, n0, ..] if params.len() == 6 && *m >= 1 && *m <= rows && *kg >= 8 && kg % 8 == 0 && *n0 >= 1)
            }
        }
    }

    /// Blocks a dispatch with these `params` launches.
    pub(crate) fn blocks(self, params: &[u32]) -> u32 {
        match self {
            Fused::AddRmsQuant | Fused::QuantEpilogue => params[1],
            Fused::GdnDecode => params[0],
            Fused::GdnDecodePool => params[0] * params[6],
            Fused::GqaDecodePrep => (params[0] + params[1]) * params[6],
            Fused::I8GemvMulti => {
                let (rows, cols) = kernels_cuda::get("matmul_i8_gemv").map_or((1, 1), |k| k.tile);
                params[0].div_ceil(rows) * params[2..6].iter().map(|n| n.div_ceil(cols)).sum::<u32>()
            }
        }
    }
}

/// The native id of `which` on this backend, or `None` where it is not
/// offered. Compiles on first use per backend, so a handle caches the answer.
pub(crate) fn resolve_fused(backend: &dyn Backend, which: Fused) -> Option<NativeId> {
    if disabled() {
        return None;
    }
    let cc = backend.caps().arch.compute_capability?;
    let k = kernels_cuda::get(which.registry_name())?;
    if k.min_cc > cc {
        return None;
    }
    backend.register_native(&NativeSpec::Cuda {
        src: k.src,
        entry: k.entry,
        block_dim: k.block_dim,
        bindings: which.bindings(),
        shared_bytes: k.shared_bytes,
        launch: CudaLaunch::NONE,
    })
}

/// `BRAIN_NO_NATIVE_KERNELS=1` disables the tier. Read once: the policy must
/// stay fixed for a given process.
fn disabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_NO_NATIVE_KERNELS").map(|v| v != "0").unwrap_or(false))
}

/// One resolved redirect for a handle.
#[derive(Clone, Debug)]
pub(crate) struct Active {
    /// The pipeline slot the model registered and dispatches by index.
    pub slow: usize,
    pub id: NativeId,
    /// The registry entry's name, for diagnostics.
    pub kernel: &'static str,
    /// Output tile a block covers, `(rows of x, weight rows)`.
    pub tile: (u32, u32),
    serves: fn(&[u32]) -> bool,
}

/// The active native redirects for a handle. Empty on every backend that
/// cannot compile the registry's source and on a device below the kernel's
/// capability floor. `wgsl_upgrades` is [`crate::upgrade::resolve`]'s answer
/// for the same handle (condition 1 in the module doc).
pub(crate) fn resolve(
    names: &[String],
    backend: &dyn Backend,
    wgsl_upgrades: &[crate::upgrade::Active],
) -> Vec<Active> {
    if disabled() {
        return Vec::new();
    }
    let Some(cc) = backend.caps().arch.compute_capability else {
        return Vec::new();
    };
    ROWS.iter()
        .filter_map(|row| {
            let slow = names.iter().position(|n| n == row.slow)?;
            if row.requires_wgsl_upgrade {
                wgsl_upgrades.iter().find(|a| a.slow == slow)?;
            }
            let k = kernels_cuda::find(row.op, row.weight, cc)?;
            let id = backend.register_native(&NativeSpec::Cuda {
                src: k.src,
                entry: k.entry,
                block_dim: k.block_dim,
                bindings: row.bindings,
                shared_bytes: k.shared_bytes,
                launch: CudaLaunch::NONE,
            })?;
            Some(Active { slow, id, kernel: k.name, tile: k.tile, serves: row.serves })
        })
        .collect()
}

/// The native pipeline and BLOCK count to dispatch instead of `kind`, or
/// `None` to keep the WGSL tier. `params` is the caller's own uniform; a call
/// without one (`Gpu::step_buf`, whose uniform lives in a caller-owned
/// buffer) is never redirected, exactly as a shape-specialised
/// [`crate::upgrade`] row is not.
#[inline]
pub(crate) fn apply(active: &[Active], kind: usize, params: &[u32]) -> Option<(NativeId, u32)> {
    let a = active.iter().find(|a| a.slow == kind)?;
    if !(a.serves)(params) {
        return None;
    }
    // `Params { m, .., n }`: one block covers `tile.0` rows of x by `tile.1`
    // weight rows, the count `CudaKernel::blocks_for` states.
    let blocks = params[0].div_ceil(a.tile.0) * params[2].div_ceil(a.tile.1);
    Some((a.id, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_int8_gemv_row_names_a_registry_entry_with_its_own_bindings() {
        let row = &ROWS[0];
        let k = kernels_cuda::get("matmul_i8_gemv").expect("registry entry");
        assert_eq!((k.op, k.weight), (row.op, row.weight));
        // Same binding list as the WGSL kernel's own declaration: one uniform
        // and five storage buffers.
        let wgsl = kernels::MATMUL_I8_GEMV_REG;
        let declared = wgsl.lines().filter(|l| l.trim_start().starts_with("@group(")).count();
        assert_eq!(declared, row.bindings.len());
    }

    #[test]
    fn the_grouped_moe_row_names_a_registry_entry_with_its_own_bindings() {
        let row = ROWS.iter().find(|r| r.slow == "moe_i8_grouped").expect("the row");
        let k = kernels_cuda::get("moe_i8_grouped_mma").expect("registry entry");
        assert_eq!((k.op, k.weight), (row.op, row.weight));
        let declared = kernels::MOE_I8_GROUPED.lines().filter(|l| l.trim_start().starts_with("@group(")).count();
        assert_eq!(declared, row.bindings.len());
        // No WGSL sibling exists to upgrade to, so the row must not wait for one.
        assert!(!row.requires_wgsl_upgrade);
        assert!(serves_moe_i8_grouped(&[40, 512, 2048, 9, 257]));
        assert!(!serves_moe_i8_grouped(&[40, 516, 2048, 9, 257]), "K not a whole number of scale groups");
        assert!(!serves_moe_i8_grouped(&[40, 512, 2048]), "short params");
    }

    #[test]
    fn only_shapes_the_kernel_serves_are_redirected() {
        assert!(serves_i8_gemv(&[1, 1280, 17408]));
        assert!(serves_i8_gemv(&[8, 8, 1]));
        assert!(!serves_i8_gemv(&[0, 1280, 64]), "no rows");
        assert!(serves_i8_gemv(&[9, 1280, 64]), "a second tile of x rows is more blocks, not another kernel");
        assert!(serves_i8_gemv(&[32, 1280, 64]));
        assert!(!serves_i8_gemv(&[33, 1280, 64]), "past the decode regime");
        assert!(!serves_i8_gemv(&[1, 1284, 64]), "K not a whole number of scale groups");
        assert!(!serves_i8_gemv(&[1, 1280]), "short params");
    }
}
