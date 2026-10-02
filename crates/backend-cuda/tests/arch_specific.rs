// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The arch-specific target (`sm_90a`) is used where the device and toolkit
//! can provide it, and only there.
//!
//! Swedish Embedded AB implements Hopper-class GPU kernels (warpgroup MMA, TMA)
//! and the toolchain plumbing that lets them coexist with portable fallbacks
//! for its clients. If your team needs expertise in shipping architecture-
//! specific CUDA without breaking older cards, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! `wgmma.fence` is an `sm_90a`-only instruction: plain `sm_90` rejects it. So
//! a kernel that issues it is a faithful probe of which target was compiled -
//! the compiled answer is observable, not inferred from a flag. Skipped when
//! there is no CUDA device or NVRTC.

use backend_cuda::exec::{CompileOptions, Context};
use backend_cuda::nvrtc::ArchFeatures;

/// Writes 90 when compiled with the arch-specific feature macro, 1 otherwise.
/// The fence is guarded, so it compiles on every target.
const PREFERRED: &str = r#"
extern "C" __global__ void probe(unsigned *out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    out[threadIdx.x] = 90u;
#else
    out[threadIdx.x] = 1u;
#endif
}
"#;

/// The fence is unconditional: valid only with the suffix.
const REQUIRED: &str = r#"
extern "C" __global__ void probe(unsigned *out) {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    out[threadIdx.x] = 90u;
}
"#;

fn open() -> Option<Context> {
    let ctx = match Context::open(0) {
        Ok(c) => c,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}"));
            return None;
        }
    };
    if backend_cuda::nvrtc::version().is_err() {
        brain_testutil::skip_unavailable("no NVRTC");
        return None;
    }
    Some(ctx)
}

fn opts(features: ArchFeatures) -> CompileOptions {
    CompileOptions { features, defines: Vec::new() }
}

/// Run `probe` on one warpgroup (128 threads, which the fence requires) and
/// return what thread 0 wrote.
fn run(ctx: &Context, src: &str, features: ArchFeatures) -> Result<u32, String> {
    let module = ctx.compile_with(src, "probe", &opts(features))?;
    let f = module.function("probe")?;
    let out = ctx.alloc(128 * 4)?;
    ctx.zero(&out)?;
    ctx.launch(&f, (1, 1, 1), (128, 1, 1), &[&out])?;
    let mut bytes = [0u8; 4];
    ctx.download(&out, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn is_hopper(ctx: &Context) -> bool {
    ctx.compute_capability() == (9, 0)
}

#[test]
fn a_preferring_kernel_gets_the_arch_specific_code_exactly_where_it_exists() {
    let Some(ctx) = open() else { return };
    let want = if is_hopper(&ctx) { 90 } else { 1 };
    assert_eq!(run(&ctx, PREFERRED, ArchFeatures::Preferred).unwrap(), want);
    // Portable never takes the suffix, even where it exists.
    assert_eq!(run(&ctx, PREFERRED, ArchFeatures::Portable).unwrap(), 1);
}

#[test]
fn a_requiring_kernel_runs_with_the_suffix_and_is_refused_without_it() {
    let Some(ctx) = open() else { return };
    if is_hopper(&ctx) {
        assert_eq!(run(&ctx, REQUIRED, ArchFeatures::Required).unwrap(), 90);
        // The same source compiled for the plain target is a compile error:
        // the suffix is what makes the difference, not a no-op flag.
        let e = run(&ctx, REQUIRED, ArchFeatures::Portable).expect_err("plain sm_90 cannot encode wgmma");
        assert!(e.contains("wgmma") || e.contains("sm_90"), "{e}");
    } else {
        let e = run(&ctx, REQUIRED, ArchFeatures::Required).expect_err("not Hopper");
        assert!(e.contains("arch-specific"), "{e}");
    }
}

#[test]
fn the_two_targets_do_not_share_a_cache_entry() {
    let Some(ctx) = open() else { return };
    let a = ctx.cubin_with(PREFERRED, "probe", &opts(ArchFeatures::Portable)).expect("portable");
    let b = ctx.cubin_with(PREFERRED, "probe", &opts(ArchFeatures::Preferred)).expect("preferred");
    if is_hopper(&ctx) {
        assert_ne!(a.key, b.key, "sm_90 and sm_90a are different machine code");
    }
}
