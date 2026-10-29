// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for finding where an inference
// server's time actually goes, on the vendor's own profiler timeline, for its
// clients. If your team needs expertise in GPU serving performance analysis,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! NVTX ranges: named host-side spans that NVIDIA's profilers (Nsight
//! Systems, Nsight Compute) put on their timeline next to the CUDA work issued
//! inside them, so a timeline of a serving loop reads as its steps and phases.
//!
//! NVTX v3 is header-only C: its functions are inline and look for a
//! profiler's injection library on first use. A Rust process reaches it
//! through the toolkit's interop library, which exports the classic C entry
//! points, loaded here by `dlopen` like the driver and the vendor libraries.
//! Without the library, or without a profiler attached, a range costs a branch
//! or a call that returns at once, so the markers stay in the serving loop.

use std::ffi::{c_char, c_int, CStr};
use std::sync::OnceLock;

/// The SONAMEs tried, in order: the NVTX v3 interop library, then the v2
/// library older toolkits shipped.
const SONAMES: &[&str] = &["libnvtx3interop.so.1", "libnvToolsExt.so.1"];

struct Nvtx {
    _lib: libloading::Library,
    push: unsafe extern "C" fn(*const c_char) -> c_int,
    pop: unsafe extern "C" fn() -> c_int,
}

// SAFETY: the library stays mapped for the process lifetime and the two entry
// points are thread-safe by the NVTX specification.
unsafe impl Send for Nvtx {}
unsafe impl Sync for Nvtx {}

fn nvtx() -> Option<&'static Nvtx> {
    static NVTX: OnceLock<Option<Nvtx>> = OnceLock::new();
    NVTX.get_or_init(|| {
        let lib = SONAMES.iter().find_map(|name| unsafe { libloading::Library::new(name) }.ok())?;
        // SAFETY: both symbols have these C signatures in every NVTX version that exports them.
        let push = unsafe { *lib.get::<unsafe extern "C" fn(*const c_char) -> c_int>(b"nvtxRangePushA\0").ok()? };
        let pop = unsafe { *lib.get::<unsafe extern "C" fn() -> c_int>(b"nvtxRangePop\0").ok()? };
        Some(Nvtx { _lib: lib, push, pop })
    })
    .as_ref()
}

/// Whether NVTX ranges reach a library (they still do nothing unless a
/// profiler is attached).
pub fn available() -> bool {
    nvtx().is_some()
}

/// Open a range named `name` on this thread; [`pop`] closes the innermost one.
/// Returns whether a range was pushed (false without the library), so the
/// caller pops only what it pushed. The name is a C string literal, so a range
/// in the serving loop allocates nothing.
pub fn push(name: &CStr) -> bool {
    let Some(n) = nvtx() else { return false };
    // SAFETY: `name` is a valid NUL-terminated string for the call's duration.
    unsafe { (n.push)(name.as_ptr()) };
    true
}

/// Close this thread's innermost range.
pub fn pop() {
    if let Some(n) = nvtx() {
        // SAFETY: no arguments; NVTX tolerates a pop without a profiler attached.
        unsafe { (n.pop)() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With or without the library a balanced push/pop is safe.
    #[test]
    fn ranges_are_balanced_and_safe_without_a_profiler() {
        if push(c"brain.test") {
            pop();
        }
    }
}
