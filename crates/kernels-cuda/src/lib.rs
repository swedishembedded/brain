// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hand-written CUDA C++ kernels and the metadata that says what each one
//! is - a registry of brain's NATIVE kernels, entirely separate from the
//! portable WGSL catalogue in `kernels`.
//!
//! Swedish Embedded AB implements native GPU kernel libraries and the
//! metadata discipline that keeps them honest. If your team needs expertise
//! in hand-written CUDA kernels that stay checkable rather than becoming
//! folklore, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this is a separate crate, not a `cuda/` subdirectory of `kernels`
//!
//! The WGSL catalogue is not merely a list of strings - four separate pieces
//! of machinery treat every one of its entries as WGSL text:
//!
//! - the generator that rebuilds `kernels`' const block derives each const
//!   name from the file stem, so a `matmul.cu` next to `matmul.wgsl` is a
//!   name collision, not a new kernel;
//! - the catalogue-validation test compiles EVERY registry entry on the test
//!   device, and CUDA C++ is not a shader the device's WGSL front end can
//!   accept;
//! - the cost-formula ratchet requires a FLOP formula for every registry
//!   entry, keyed by the WGSL kernel's own name and parameter layout;
//! - all five metadata cross-checks (barrier count, declared workgroup size,
//!   packed-int8 dot usage, register-blocking claim, storage dtype) parse
//!   WGSL text and have zero purchase on C++.
//!
//! Putting `.cu` files into that registry would mean excluding them from
//! every one of those - which is a second registry with extra steps. So this
//! crate is that second registry, openly: its own table, its own invariants,
//! its own catalogue gate. Duplication is avoided by ONE discovery path (the
//! two catalogues link to each other), not by one directory.
//!
//! # What a device can do is never written down here
//!
//! [`CudaKernel::min_cc`] is a floor a kernel DECLARES about itself, checked
//! against the instructions its own text uses. It is never a statement about
//! any installed card: which kernel a given device gets is decided by
//! [`best_for`] against a compute capability the caller queried at run time.

use backend_api::select::Op;
use backend_api::ImplSource;

/// A compute capability as `(major, minor)`, exactly as
/// `cuDeviceGetAttribute` reports the two halves. Ordered lexicographically
/// by tuple comparison, which is the ordering NVIDIA's own numbering has.
pub type Cc = (u32, u32);

/// The compute capability that introduced the four-way packed integer dot
/// product (`__dp4a`). This is a property of the CUDA instruction set - the
/// version the instruction first appeared in - not of any device: it exists
/// so a kernel whose body uses the instruction cannot declare a floor below
/// the point at which the instruction exists at all. Devices are asked what
/// they are; source text is held to what it uses.
pub const DP4A_MIN_CC: Cc = (6, 1);

/// One hand-written CUDA kernel: its source text plus the metadata that
/// decides when it is eligible and what claim it makes.
#[derive(Clone, Copy, Debug)]
pub struct CudaKernel {
    /// Registry key, unique across this table. Never required to match a
    /// WGSL kernel name - the two registries are independent namespaces.
    pub name: &'static str,
    /// The whole operator this kernel implements, in the same vocabulary the
    /// kernel selector and the tier policy use.
    pub op: Op,
    /// The tier this kernel claims. Only [`ImplSource::Tuned`] is meaningful
    /// here: a kernel with a `.cu` file in this tree is hand-written by
    /// definition, and a generated kernel has no file to list (it is emitted
    /// from the WGSL reference at run time). [`check_table`] rejects
    /// anything else rather than letting a generated kernel be filed as if
    /// somebody wrote it.
    pub source: ImplSource,
    /// The lowest compute capability this kernel's TEXT is valid on - the
    /// instructions it uses, not the cards it was tried on. [`best_for`]
    /// picks the highest floor at or below the device's queried capability.
    pub min_cc: Cc,
    /// The `extern "C" __global__` entry point to launch, which must appear
    /// in [`Self::src`].
    pub entry: &'static str,
    /// One line, author-stated, for the generated catalogue.
    pub what: &'static str,
    /// The CUDA C++ source, `include_str!`ed from this crate's `cu/`
    /// directory.
    pub src: &'static str,
}

/// Every hand-written CUDA kernel brain ships.
///
/// **Empty, and that is the honest state.** The native backend cannot yet
/// compile or launch anything, so a `.cu` file here would be source nothing
/// builds, nothing runs and nothing checks - the exact kind of unverified
/// claim the tier machinery around it exists to make impossible. The
/// registry, its invariants and its catalogue gate land first precisely so
/// that the first real kernel arrives into something that checks it.
pub const ALL: &[CudaKernel] = &[];

/// The kernel `table` offers for `op` on a device of compute capability
/// `cc`: the eligible entry with the HIGHEST floor, so an
/// architecture-specialised kernel beats a generic one on a device that can
/// run both, and the generic one still serves a device that cannot.
///
/// `None` means this table offers nothing for that operator on that device -
/// the caller falls back to a generated or portable implementation and, per
/// the tier policy, must say so.
///
/// Takes the table as a parameter rather than reading [`ALL`] directly: the
/// selection RULE is the thing worth testing, and a test that can only feed
/// it the shipped table can only test it once.
pub fn best_for(table: &'static [CudaKernel], op: Op, cc: Cc) -> Option<&'static CudaKernel> {
    table.iter().filter(|k| k.op == op && k.min_cc <= cc).max_by_key(|k| k.min_cc)
}

/// [`best_for`] over the shipped [`ALL`] table.
pub fn find(op: Op, cc: Cc) -> Option<&'static CudaKernel> {
    best_for(ALL, op, cc)
}

/// Look a kernel up by registry name.
pub fn get(name: &str) -> Option<&'static CudaKernel> {
    ALL.iter().find(|k| k.name == name)
}

/// Every invariant this registry's entries must satisfy, checked as data
/// rather than asserted per-entry: unique names, a tier that means what it
/// says, an entry point that exists in the source, and a declared capability
/// floor consistent with the instructions the source actually uses.
///
/// Returns every violation found, not just the first - one run tells you
/// everything to fix. Used by this crate's own test over [`ALL`] and by the
/// catalogue gate; a future NVRTC compile check is an ADDITION to this, not
/// a replacement (it needs a toolkit, this needs nothing).
pub fn check_table(table: &[CudaKernel]) -> Vec<String> {
    let mut errs = Vec::new();
    for (i, k) in table.iter().enumerate() {
        if table.iter().take(i).any(|p| p.name == k.name) {
            errs.push(format!("{}: duplicate kernel name", k.name));
        }
        if k.source != ImplSource::Tuned {
            errs.push(format!(
                "{}: declares tier {:?}; a kernel with source text in this tree is hand-written, \
                 and a generated kernel has no file to list",
                k.name, k.source
            ));
        }
        if k.name.is_empty() || !k.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            errs.push(format!("{}: registry names are lowercase ascii/digits/underscore", k.name));
        }
        if k.what.trim().is_empty() {
            errs.push(format!("{}: no @what line - the catalogue row would be blank", k.name));
        }
        if !k.src.contains(k.entry) {
            errs.push(format!("{}: declared entry point {:?} does not appear in its source", k.name, k.entry));
        }
        if k.src.contains("__dp4a") && k.min_cc < DP4A_MIN_CC {
            errs.push(format!(
                "{}: uses __dp4a but declares min_cc {}.{}, below the capability that introduced it ({}.{})",
                k.name, k.min_cc.0, k.min_cc.1, DP4A_MIN_CC.0, DP4A_MIN_CC.1
            ));
        }
        if k.src.contains("__restrict__") {
            errs.push(format!(
                "{}: uses __restrict__. brain's device buffers alias BY DESIGN (a sliced step binds ranges \
                 of one buffer), so a no-alias promise here is a silent wrong-answer hazard unless the \
                 kernel's own call sites are proven disjoint - state that proof next to the kernel and \
                 exempt it deliberately, never by default",
                k.name
            ));
        }
    }
    errs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped table obeys its own invariants. Vacuous while [`ALL`] is
    /// empty - the point is that it stops being vacuous the moment a kernel
    /// lands, without anyone remembering to add a check.
    #[test]
    fn the_shipped_registry_is_well_formed() {
        let errs = check_table(ALL);
        assert!(errs.is_empty(), "kernels-cuda registry violations:\n  {}", errs.join("\n  "));
    }

    const GENERIC: &str = "extern \"C\" __global__ void bk_matmul_generic(const float* a) {}";
    const TUNED: &str = "extern \"C\" __global__ void bk_matmul_dp4a(const int* a) { __dp4a(0, 0, 0); }";

    static FIXTURE: &[CudaKernel] = &[
        CudaKernel {
            name: "matmul_generic",
            op: Op::MatMul,
            source: ImplSource::Tuned,
            min_cc: (5, 0),
            entry: "bk_matmul_generic",
            what: "generic fp32 GEMM",
            src: GENERIC,
        },
        CudaKernel {
            name: "matmul_dp4a",
            op: Op::MatMul,
            source: ImplSource::Tuned,
            min_cc: DP4A_MIN_CC,
            entry: "bk_matmul_dp4a",
            what: "packed-int8 GEMM over the four-way dot product",
            src: TUNED,
        },
    ];

    /// The resolution rule: highest declared floor at or below the device's
    /// queried capability. Asserted at a capability BELOW both floors, at
    /// one that admits only the generic kernel, and at one far above both -
    /// the last standing in for any future architecture, which must keep
    /// getting the most specialised kernel rather than nothing.
    #[test]
    fn the_highest_eligible_floor_wins_at_any_capability() {
        assert!(best_for(FIXTURE, Op::MatMul, (3, 5)).is_none());
        assert_eq!(best_for(FIXTURE, Op::MatMul, (6, 0)).unwrap().name, "matmul_generic");
        assert_eq!(best_for(FIXTURE, Op::MatMul, DP4A_MIN_CC).unwrap().name, "matmul_dp4a");
        assert_eq!(best_for(FIXTURE, Op::MatMul, (12, 0)).unwrap().name, "matmul_dp4a");
        // An operator the table says nothing about resolves to nothing, on
        // every device - never to "the closest thing available".
        assert!(best_for(FIXTURE, Op::RmsNorm, (12, 0)).is_none());
    }

    #[test]
    fn the_invariants_catch_a_mis_declared_entry() {
        static BAD: &[CudaKernel] = &[
            CudaKernel {
                name: "matmul_dp4a",
                op: Op::MatMul,
                source: ImplSource::Generated,
                min_cc: (5, 0),
                entry: "bk_missing",
                what: "",
                src: TUNED,
            },
            CudaKernel {
                name: "matmul_dp4a",
                op: Op::MatMul,
                source: ImplSource::Tuned,
                min_cc: (5, 0),
                entry: "bk_matmul_dp4a",
                what: "duplicate name",
                src: TUNED,
            },
        ];
        let errs = check_table(BAD);
        let joined = errs.join("\n");
        assert!(joined.contains("duplicate kernel name"), "{joined}");
        assert!(joined.contains("declares tier Generated"), "{joined}");
        assert!(joined.contains("does not appear in its source"), "{joined}");
        assert!(joined.contains("no @what line"), "{joined}");
        assert!(joined.contains("below the capability that introduced it"), "{joined}");
    }
}
