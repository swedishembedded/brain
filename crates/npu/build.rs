// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Best-effort OpenVINO probe. Mirrors `crates/vulkan/build.rs`: it NEVER fails
//! the build. OpenVINO is a default dependency on x86_64 linux/windows, but with
//! `runtime-linking` it is `dlopen`'d at run time (not linked at build time), so
//! even a machine without OpenVINO installed builds fine — a missing runtime only
//! surfaces when a `NpuSession` is opened. This probe just emits a friendly note
//! and a `cfg(have_openvino_runtime)` flag when the runtime dir is discoverable.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_openvino_runtime)");

    // The probe is only meaningful on the platforms that link the OpenVINO crate.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let supported = target_arch == "x86_64" && (target_os == "linux" || target_os == "windows");
    if !supported {
        return;
    }

    let found = ["INTEL_OPENVINO_DIR", "OPENVINO_INSTALL_DIR"]
        .iter()
        .any(|k| std::env::var_os(k).is_some());

    if found {
        println!("cargo:rustc-cfg=have_openvino_runtime");
    } else {
        println!(
            "cargo:warning=brain-npu: no INTEL_OPENVINO_DIR / OPENVINO_INSTALL_DIR in env. \
             The OpenVINO runtime is loaded at run time (runtime-linking); the build still \
             succeeds. Source setupvars.sh before `brain --device npu` / `brain npu run`. \
             See docs/models/yolo/npu.md."
        );
    }
}
