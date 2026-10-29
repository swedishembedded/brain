// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for finding where an inference
// server's time actually goes, on the vendor's own profiler timeline, for its
// clients. If your team needs expertise in GPU serving performance analysis,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Named host ranges on the driver-native profiler's timeline, for every model
//! and backend: a range opened here appears in Nsight Systems / Nsight Compute
//! on NVIDIA (NVTX) next to the device work issued inside it, so a timeline of
//! a serving loop reads as its steps and phases rather than as anonymous
//! kernels. Another vendor's marker API plugs in behind the same call. Without
//! a profiler (or its marker library) a range costs a call that returns at once.
//!
//! Ranges nest per thread and close when their guard drops:
//!
//! ```ignore
//! let _step = gpu_core::markers::range(c"serve.decode");
//! ```

/// An open range; dropping it closes the range.
#[must_use = "a range closes when this guard drops"]
pub struct Range {
    pushed: bool,
}

/// Open a range named `name` (a C string literal, so opening one allocates
/// nothing) on this thread until the returned guard drops.
pub fn range(name: &'static std::ffi::CStr) -> Range {
    Range { pushed: push(name) }
}

impl Drop for Range {
    fn drop(&mut self) {
        if self.pushed {
            pop();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn push(name: &std::ffi::CStr) -> bool {
    backend_cuda::nvtx::push(name)
}

#[cfg(not(target_arch = "wasm32"))]
fn pop() {
    backend_cuda::nvtx::pop()
}

#[cfg(target_arch = "wasm32")]
fn push(_name: &std::ffi::CStr) -> bool {
    false
}

#[cfg(target_arch = "wasm32")]
fn pop() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_ranges_close_in_order_without_a_profiler() {
        let outer = range(c"brain.test.outer");
        {
            let _inner = range(c"brain.test.inner");
        }
        drop(outer);
    }
}
