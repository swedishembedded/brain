// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `FlashGqaIds::supported` gates the flash training trio on the device's
//! workgroup memory. The gate must cover what the backward kernels actually
//! declare: when it asked for 32 KiB while the dkv kernel declared 64 bytes
//! more, a device with exactly 32 KiB passed the gate and then failed to build
//! the kernel. This reads the declarations from the WGSL itself.

use model::block::FlashGqaIds;

/// Bytes of every `var<workgroup> name: array<f32, N>;` in a kernel source.
fn workgroup_bytes(src: &str) -> u32 {
    src.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("var<workgroup>")?;
            let inner = rest.split("array<f32,").nth(1)?;
            let n: u32 = inner.split('>').next()?.trim().parse().ok()?;
            Some(n * 4)
        })
        .sum()
}

#[test]
fn backward_gate_covers_the_workgroup_memory_the_kernels_declare() {
    let dq = workgroup_bytes(kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DQ);
    let dkv = workgroup_bytes(kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DKV);
    assert!(dq > 0 && dkv > 0, "no workgroup arrays parsed (dq {dq}, dkv {dkv})");
    assert_eq!(
        FlashGqaIds::BWD_WORKGROUP_BYTES,
        dq.max(dkv),
        "the gate must ask for exactly the larger kernel's workgroup memory"
    );
}
