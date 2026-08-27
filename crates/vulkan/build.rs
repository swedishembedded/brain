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

    // This crate exists to provide the cooperative-matrix path, so always
    // attempt the (best-effort, never-failing) GLSL->SPIR-V compile.
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
    //
    // Say exactly what to install and exactly what is lost. The previous text
    // pointed at a README that does not exist in this tree, and "the
    // tensor-core kernel will be unavailable" reads as though a served model
    // just got slower - it does not. The cooperative-matrix kernel is reached
    // only from the `moe pid vk-info` / `moe pid vk-matmul` demo entries; the
    // real native-Vulkan backend (`--device vulkan`, brain-backend-vulkan)
    // compiles the ordinary WGSL kernels through naga and never touches it. So
    // this is a missing DEMO, not a silently degraded inference path.
    println!(
        "cargo:warning=vulkan-coopmat: no glslc/glslangValidator on PATH \
         (Debian/Ubuntu: apt-get install glslc glslang-tools); skipping the \
         cooperative-matrix SPIR-V build. Only the `moe pid vk-matmul` demo \
         uses it, and it falls back to the scalar kernel -- no model or \
         benchmark path is affected."
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
