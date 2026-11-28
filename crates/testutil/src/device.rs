// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device a test binary runs on, as the test harness asks for it: the one seam between a harness (a gate script,
//! a Makefile backend suite, a developer's shell) and the device the library is told to use.
//!
//! The library reads no device variable. A process nothing configured runs on the CPU, and a brain binary takes its
//! device from `--device`/`--backend`. A test binary takes it from here: [`device`] reads `BRAIN_TEST_DEVICE` (the
//! one place any code reads it, `make check/scripts` holds it so) and passes it to the library through
//! `gpu_core::select_compute`, exactly as a tool passes its own flags. Its grammar is `--device` optionally followed
//! by `:` and a `--backend`: `cpu`, `vulkan`, `gpu0`, `gpu0:cuda`, `gpu:wgpu`. Unset or empty means the CPU. A device
//! this machine cannot give stops the binary, naming the variable and its value: a run on the CPU in its place would
//! be a silent substitution.
//!
//! A test binary states that it takes the harness's device with [`configure_test_binary!`] at its root, so the
//! selection is made before any test builds anything; a test that needs a GPU says so with [`gpu`].
//!
//! Swedish Embedded AB implements training and inference stacks whose tests run on exactly the hardware they are
//! asked to. If your team needs expertise in backend-matrix test harnesses for GPU code, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::OnceLock;

/// The variable the harness names the device in.
pub const VARIABLE: &str = "BRAIN_TEST_DEVICE";

/// The device the harness asked for, as the library resolved it.
#[derive(Debug)]
pub struct TestDevice {
    /// The `--device` part: `cpu` when the harness named none.
    pub device: String,
    /// The `--backend` part, when named.
    pub backend: Option<String>,
    /// The compute set the library now runs on.
    pub set: gpu_core::ComputeSet,
}

impl TestDevice {
    /// Whether the harness asked for a GPU.
    pub fn is_gpu(&self) -> bool {
        self.set.gpu_enabled() || self.set.software
    }
}

/// The parts of a `BRAIN_TEST_DEVICE` value: `(device, backend)`. Empty is the CPU.
///
/// # Errors
/// More than one `:`, or an empty part beside one.
pub fn parse(text: &str) -> Result<(String, Option<String>), String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(("cpu".into(), None));
    }
    match text.split(':').collect::<Vec<_>>().as_slice() {
        [device] => Ok(((*device).to_string(), None)),
        [device, backend] if !device.is_empty() && !backend.is_empty() => Ok(((*device).to_string(), Some((*backend).to_string()))),
        _ => Err(format!("{VARIABLE}={text:?}: expected <device>[:<backend>], e.g. cpu, vulkan, gpu0:cuda")),
    }
}

/// The device the harness asked this test binary for, selected in the library on first use.
///
/// # Panics
/// The value does not parse, or names a device this machine cannot give.
pub fn device() -> &'static TestDevice {
    static DEVICE: OnceLock<TestDevice> = OnceLock::new();
    DEVICE.get_or_init(|| {
        let text = std::env::var("BRAIN_TEST_DEVICE").unwrap_or_default();
        let (device, backend) = parse(&text).unwrap_or_else(|e| panic!("{e}"));
        let set = gpu_core::select_compute(&device, backend.as_deref())
            .unwrap_or_else(|e| panic!("{VARIABLE}={text:?} asks for a device this machine cannot give: {e}"));
        TestDevice { device, backend, set }
    })
}

/// [`device`] for a test that needs a GPU: the harness must have asked for one.
///
/// # Panics
/// The harness asked for the CPU, or for a GPU this machine cannot give.
#[track_caller]
pub fn gpu() -> &'static TestDevice {
    let d = device();
    if !d.is_gpu() {
        panic!(
            "{} needs a GPU: run it with {VARIABLE} naming one (e.g. {VARIABLE}=gpu0:cuda); it asked for {:?}",
            std::panic::Location::caller(),
            d.device
        );
    }
    d
}

/// Makes this test binary take the harness's device ([`device`]) before any test runs: one line at the root of the
/// binary (an integration test's `main.rs`, a crate's `#[cfg(test)]` module root).
///
/// It runs [`device`] when the binary is loaded (an ELF initialiser on Linux), so every test of the binary, and
/// everything the library builds for it, runs on the one selected device.
#[macro_export]
macro_rules! configure_test_binary {
    () => {
        #[cfg(target_os = "linux")]
        #[used]
        #[link_section = ".init_array"]
        static BRAIN_TEST_DEVICE_INIT: extern "C" fn() = {
            extern "C" fn init() {
                let _ = $crate::device::device();
            }
            init
        };
    };
}
