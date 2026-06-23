// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build script: best-effort compilation of the GLSL cooperative-matrix
//! compute shader to SPIR-V.
//!
//! This only does anything when the `vulkan-coopmat` feature is enabled. It
//! tries `glslc` (shaderc) first, then `glslangValidator`. If neither is on
//! PATH it prints a clear note and SKIPS -- it never fails the build. The
//! runtime (`src/vulkan/matmul.rs`) loads the compiled `coopmat.spv` from
//! `OUT_DIR` when present and otherwise falls back to the naga-compiled scalar
//! matmul. This keeps the default build, and machines without a GLSL compiler,
//! fully buildable.

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Declare the cfg we may set so `cargo check` doesn't warn about it.
    println!("cargo:rustc-check-cfg=cfg(have_coopmat_spv)");

    // Only relevant to the optional Vulkan path.
    if env::var_os("CARGO_FEATURE_VULKAN_COOPMAT").is_none() {
        return;
    }

    let src = Path::new("src/shaders_vk/matmul_coopmat.comp");
    println!("cargo:rerun-if-changed=src/shaders_vk/matmul_coopmat.comp");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let out_spv = Path::new(&out_dir).join("matmul_coopmat.spv");

    // Try glslc (shaderc) first; it has the cleanest target-env flags.
    if try_glslc(src, &out_spv) {
        println!("cargo:warning=vulkan-coopmat: compiled matmul_coopmat.comp via glslc");
        emit_have_spv();
        return;
    }
    // Fall back to glslangValidator.
    if try_glslang(src, &out_spv) {
        println!("cargo:warning=vulkan-coopmat: compiled matmul_coopmat.comp via glslangValidator");
        emit_have_spv();
        return;
    }

    // Neither compiler present: skip, do NOT fail. The runtime will use the
    // naga-compiled scalar matmul fallback.
    println!(
        "cargo:warning=vulkan-coopmat: no glslc/glslangValidator on PATH; \
         skipping cooperative-matrix SPIR-V build. The tensor-core kernel will \
         be unavailable at runtime (scalar fallback used). See README_VULKAN.md \
         to install a GLSL compiler and recompile."
    );
}

/// Set a cfg so runtime code can `include_bytes!` the SPIR-V only when it exists.
fn emit_have_spv() {
    println!("cargo:rustc-cfg=have_coopmat_spv");
}

fn try_glslc(src: &Path, out: &Path) -> bool {
    // --target-env=vulkan1.3 enables the cooperative-matrix extension path.
    run(Command::new("glslc")
        .arg("-fshader-stage=compute")
        .arg("--target-env=vulkan1.3")
        .arg("-O")
        .arg(src)
        .arg("-o")
        .arg(out))
}

fn try_glslang(src: &Path, out: &Path) -> bool {
    run(Command::new("glslangValidator")
        .arg("-V")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg("-S")
        .arg("comp")
        .arg(src)
        .arg("-o")
        .arg(out))
}

fn run(cmd: &mut Command) -> bool {
    match cmd.status() {
        Ok(status) => status.success(),
        Err(_) => false, // binary not found / not executable
    }
}
