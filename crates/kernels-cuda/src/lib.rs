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

/// The lowest compute capability a CUDA 12.x toolchain will emit code for at
/// all (`--gpu-architecture=sm_50`). A kernel whose text uses nothing beyond
/// the always-available core - shared memory, `__syncthreads`, fp32
/// arithmetic, 64-bit address arithmetic - declares this floor, which says
/// "there is no capability this toolchain can target where this source is
/// invalid", not "this kernel was written for Maxwell".
///
/// Like [`DP4A_MIN_CC`] this is a property of the TOOLCHAIN and the
/// instruction set, never of a card: the floor a kernel declares is checked
/// against the capability a device was *asked* for, and a device below every
/// floor simply gets no native kernel.
pub const BASELINE_MIN_CC: Cc = (5, 0);

/// Shared memory per block every CUDA compute capability guarantees. A
/// kernel's declared `__shared__` must fit inside this, so that "does this
/// device have room" is a question only about cards that grant MORE - which
/// is asked of the driver, per device, never written down. A card granting
/// more is an opportunity a future kernel may query for; it is never a floor
/// this file may assume.
pub const PORTABLE_SHARED_BYTES: u32 = 48 * 1024;

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
    /// This kernel's name as a DISPATCH RECORD reports it: [`Self::name`]
    /// under a `native:` qualifier.
    ///
    /// Written out rather than formatted at use because a dispatch record
    /// holds `&'static str` and formatting one per lowered request would
    /// have to leak it. [`check_table`] pins the two spellings together, so
    /// the duplication cannot drift.
    ///
    /// The qualifier is not decoration: a bare lowercase name in a dispatch
    /// record is indistinguishable from a WGSL catalogue kernel, and this
    /// registry is a different namespace whose names are under no obligation
    /// to be absent from that one.
    pub reported: &'static str,
    /// Threads per block the kernel's own index arithmetic is written
    /// against - CUDA C++ has no `@workgroup_size` attribute for a backend to
    /// read, so a launcher would otherwise have to guess.
    pub block_dim: u32,
    /// Output elements one block covers, as `(rows, cols)` of the `(m, n)`
    /// output. The launcher turns a shape into a block count with it
    /// (`ceil(m/rows) * ceil(n/cols)`); the kernel reconstructs its own tile
    /// from the flat block index the same way.
    pub tile: (u32, u32),
    /// Static `__shared__` bytes the kernel declares. Checked against the
    /// device's QUERIED shared-memory-per-block limit before dispatch, so a
    /// kernel a card cannot host is declined rather than failing at launch.
    pub shared_bytes: u32,
    /// The CUDA C++ source, `include_str!`ed from this crate's `cu/`
    /// directory.
    pub src: &'static str,
}

impl CudaKernel {
    /// How many blocks cover an `(m, n)` output with this kernel's tile.
    pub fn blocks_for(&self, m: u32, n: u32) -> u32 {
        m.div_ceil(self.tile.0.max(1)) * n.div_ceil(self.tile.1.max(1))
    }
}

/// Every hand-written CUDA kernel brain ships.
///
/// One entry today, and the table says only what is true of it. Its floor is
/// the toolchain baseline rather than any card's capability, because that is
/// what its text actually needs; the capability a device reports is asked of
/// the driver and met against this table by [`best_for`], which is where the
/// architecture-specific decision lives. A second, higher-floor entry for
/// the same operator is what makes that resolution visible in production
/// rather than only in [`best_for`]'s own test, and none is written yet.
pub const ALL: &[CudaKernel] = &[CudaKernel {
    name: "matmul_f32_tiled",
    op: Op::MatMul,
    source: ImplSource::Tuned,
    min_cc: BASELINE_MIN_CC,
    entry: "brain_matmul_f32_tiled",
    what: "fp32 out = x @ W^T; 64x64 shared tile, 4x4 register block, reference reduction order",
    reported: "native:matmul_f32_tiled",
    block_dim: 256,
    tile: (64, 64),
    // 2 tiles x 16 staged k x (64 + 1 pad) floats. Stated here because the
    // provider checks it against the device's own queried limit before it
    // ever asks the driver to launch.
    shared_bytes: 2 * 16 * (64 + 1) * 4,
    src: include_str!("../cu/matmul_f32_tiled.cu"),
}];

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

/// `src` with `//` comments removed - what the instruction scans in
/// [`check_table`] must look at.
///
/// A kernel's header PROSE is the natural place to say which instructions it
/// uses and which it deliberately avoids, and a scan over raw text reads
/// those sentences as if they were code: a kernel whose header explains that
/// it carries no aliasing promise gets failed for the word appearing in the
/// explanation. That is the identical mistake `backend_api::
/// workgroup_size_of` documents having made against `@workgroup_size` in
/// WGSL headers, with the same fix - scan the code, not the comments.
///
/// Line comments only. Every kernel in this tree writes its header as `//`
/// lines, and stripping `/* */` correctly needs a real lexer (string
/// literals, nesting) for no gain against source nobody writes that way; a
/// block comment that mentions an instruction is therefore still read as
/// code, which fails LOUDLY and is fixed by rewording, never silently.
fn code_of(src: &str) -> String {
    src.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n")
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
        // Instruction scans read the CODE, never the header prose that
        // explains which instructions the kernel uses - see `code_of`.
        let code = code_of(k.src);
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
        if k.reported != format!("native:{}", k.name) {
            errs.push(format!(
                "{}: reports itself as {:?}; a dispatch record must name it \"native:{}\" so it cannot \
                 be read as a WGSL catalogue kernel",
                k.name, k.reported, k.name
            ));
        }
        if k.what.trim().is_empty() {
            errs.push(format!("{}: no @what line - the catalogue row would be blank", k.name));
        }
        if !k.src.contains(k.entry) {
            errs.push(format!("{}: declared entry point {:?} does not appear in its source", k.name, k.entry));
        }
        if code.contains("__dp4a") && k.min_cc < DP4A_MIN_CC {
            errs.push(format!(
                "{}: uses __dp4a but declares min_cc {}.{}, below the capability that introduced it ({}.{})",
                k.name, k.min_cc.0, k.min_cc.1, DP4A_MIN_CC.0, DP4A_MIN_CC.1
            ));
        }
        if k.block_dim == 0 || k.block_dim % 32 != 0 {
            errs.push(format!(
                "{}: block_dim {} is not a non-zero multiple of the warp granularity every CUDA \
                 capability schedules in - a partial warp wastes lanes at every launch",
                k.name, k.block_dim
            ));
        }
        if k.tile.0 == 0 || k.tile.1 == 0 {
            errs.push(format!("{}: a tile of {:?} covers no output, so no block count can be derived", k.name, k.tile));
        }
        if k.shared_bytes > PORTABLE_SHARED_BYTES {
            errs.push(format!(
                "{}: declares {} bytes of __shared__, above the {PORTABLE_SHARED_BYTES} every CUDA \
                 capability guarantees per block - a card that grants more must be QUERIED, never assumed",
                k.name, k.shared_bytes
            ));
        }
        if code.contains("__restrict__") {
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

    /// The shipped table obeys its own invariants.
    #[test]
    fn the_shipped_registry_is_well_formed() {
        let errs = check_table(ALL);
        assert!(errs.is_empty(), "kernels-cuda registry violations:\n  {}", errs.join("\n  "));
        assert!(!ALL.is_empty(), "the registry ships at least one hand-written kernel");
    }

    /// A block count must COVER the output, at shapes that divide the tile
    /// and at shapes that do not - an under-count leaves real output
    /// elements never written, which is silent corruption rather than a
    /// crash (the same failure mode the WGSL thread-count formula's own
    /// regression test exists for).
    #[test]
    fn the_block_count_covers_every_output_element() {
        for k in ALL {
            for (m, n) in [(1u32, 1u32), (64, 64), (65, 64), (64, 65), (300, 260), (37, 53), (513, 257)] {
                let blocks = k.blocks_for(m, n);
                assert!(
                    (blocks as u64) * (k.tile.0 as u64) * (k.tile.1 as u64) >= (m as u64) * (n as u64),
                    "{}: {blocks} blocks of {:?} do not cover a {m}x{n} output",
                    k.name,
                    k.tile
                );
            }
        }
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
            reported: "native:matmul_generic",
            block_dim: 64,
            tile: (1, 64),
            shared_bytes: 0,
            src: GENERIC,
        },
        CudaKernel {
            name: "matmul_dp4a",
            op: Op::MatMul,
            source: ImplSource::Tuned,
            min_cc: DP4A_MIN_CC,
            entry: "bk_matmul_dp4a",
            what: "packed-int8 GEMM over the four-way dot product",
            reported: "native:matmul_dp4a",
            block_dim: 128,
            tile: (32, 32),
            shared_bytes: 1024,
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

    /// A kernel's header explaining which instructions it avoids is PROSE,
    /// not a use of them. Before this was fixed, the shipped kernel failed
    /// its own registry check for saying in its header that it carries no
    /// aliasing promise - the scan could not tell a sentence from a
    /// declaration, exactly as the WGSL work-group-size scan once could not.
    #[test]
    fn an_instruction_named_only_in_a_comment_is_not_a_use_of_it() {
        static PROSE: &[CudaKernel] = &[CudaKernel {
            name: "prose_only",
            op: Op::MatMul,
            source: ImplSource::Tuned,
            // Below DP4A's floor on purpose: the header mentions the
            // instruction, the body does not use it, and only the body counts.
            min_cc: BASELINE_MIN_CC,
            entry: "bk_prose",
            what: "mentions __dp4a and __restrict__ in its header and uses neither",
            reported: "native:prose_only",
            block_dim: 64,
            tile: (1, 64),
            shared_bytes: 0,
            src: "// no __restrict__ here, and no __dp4a either\n\
                  extern \"C\" __global__ void bk_prose(const float* a) {}\n",
        }];
        assert!(check_table(PROSE).is_empty(), "{:?}", check_table(PROSE));
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
                reported: "matmul_dp4a",
                block_dim: 33,
                tile: (0, 8),
                shared_bytes: PORTABLE_SHARED_BYTES + 1,
                src: TUNED,
            },
            CudaKernel {
                name: "matmul_dp4a",
                op: Op::MatMul,
                source: ImplSource::Tuned,
                min_cc: (5, 0),
                entry: "bk_matmul_dp4a",
                what: "duplicate name",
                reported: "native:matmul_dp4a",
                block_dim: 64,
                tile: (16, 16),
                shared_bytes: 0,
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
        assert!(joined.contains("is not a non-zero multiple of the warp granularity"), "{joined}");
        assert!(joined.contains("covers no output"), "{joined}");
        assert!(joined.contains("bytes of __shared__, above the"), "{joined}");
        assert!(joined.contains("reports itself as"), "{joined}");
    }
}
