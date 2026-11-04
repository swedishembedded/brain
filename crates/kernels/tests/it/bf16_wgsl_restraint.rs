// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M8.8 - a permanent, grep-level restraint, not a kernel: WGSL has NO
//! `enable bf16;` and no bf16 scalar type at all (unlike `f16`, which the
//! spec exposes as a real, narrow arithmetic type - see M8.7's
//! `NativeF16Provider`) - there is no rewrite or polyfill that gets bf16
//! ARITHMETIC into WGSL. Native bf16 compute can only ever come from a
//! genuinely non-WGSL provider (a SPIR-V bf16 extension via
//! `backend_api::Backend::register_native`/`step_native`, or AVX512-BF16/
//! AMX-BF16 on the CPU backend) - neither exists in this tree today. These
//! two tests are the tripwire: simple, fast, tree-wide, textual - not a
//! semantic analysis - so they cost nothing to run on every `cargo test` and
//! catch the day someone pastes `enable bf16;` into a `.wgsl` file expecting
//! it to work the way `enable f16;` does.

use std::path::PathBuf;

fn wgsl_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wgsl")
}

/// No on-disk WGSL kernel source may enable or declare bf16 - checked
/// against every real `.wgsl` file in `crates/kernels/wgsl/`, the exact
/// directory `scripts/build/gen-kernel-table.py`'s `kernelmeta.WGSL` walks
/// for the kernel catalogue.
#[test]
fn no_on_disk_wgsl_kernel_enables_or_declares_bf16() {
    let dir = wgsl_dir();
    let mut checked = 0usize;
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("reading {}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        assert!(
            !src.contains("enable bf16"),
            "{}: contains `enable bf16;` -- WGSL has no bf16 extension at all (unlike f16), so \
             this can never have been meant to compile; see this file's own module doc for where \
             native bf16 compute can actually come from",
            path.display()
        );
        // No standalone `bf16` CODE token either (a bf16 scalar/vector type
        // name, had one ever been proposed) - `//` comments are stripped
        // first, since this header convention legitimately names `bf16` as a
        // SUPPORTED STORAGE dtype (`// @dtype f32|bf16|f16`, or prose like
        // `moe_linear_gated_kq.wgsl`'s own "the bf16/f16 WEIGHT STORAGE
        // tier" line) - that tier stays fp32 arithmetic throughout
        // (`dtype_variant`'s existing storage rewrite) and is exactly what
        // this restraint does NOT restrain; only bf16 as an actual WGSL
        // TYPE/directive token in code is checked.
        for line in src.lines() {
            let code = line.split("//").next().unwrap_or("");
            for tok in code.split(|c: char| !c.is_alphanumeric() && c != '_') {
                assert_ne!(tok, "bf16", "{}: bare `bf16` code token found -- WGSL has no bf16 type", path.display());
            }
        }
        checked += 1;
    }
    assert!(checked > 50, "expected to check a real, populated wgsl/ directory, only found {checked} files");
}

/// No `OperatorProvider` may claim a bf16 COMPUTE tier
/// (`select::Requirement::bf16_compute: true`) for a WGSL-backed provider -
/// checked textually across the two places a `Requirement` gets built for a
/// WGSL dispatch: `backend_api::select`'s own kernel-variant requirement
/// table, and every provider under `gpu_core::provider`. `bf16_storage`
/// (bytes merely held/decoded, `dtype_variant`'s existing storage tier,
/// arithmetic still fp32 throughout) is unaffected - only the COMPUTE flag
/// is restrained, since that is the one WGSL structurally cannot satisfy.
#[test]
fn no_wgsl_backed_requirement_claims_bf16_compute() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let mut checked = 0usize;
    for rel in [
        "crates/backend-api/src/select.rs",
        "crates/gpu-core/src/provider/mod.rs",
        "crates/gpu-core/src/provider/wgsl.rs",
        "crates/gpu-core/src/provider/native_f16.rs",
        "crates/gpu-core/src/provider/parity.rs",
    ] {
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        assert!(
            !src.contains("bf16_compute: true"),
            "{}: sets Requirement::bf16_compute: true -- no provider dispatching through WGSL can \
             ever satisfy that (WGSL has no bf16 arithmetic at all), see this file's own module doc",
            path.display()
        );
        checked += 1;
    }
    assert_eq!(checked, 5);
}
