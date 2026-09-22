// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `adamw.wgsl` reads a per-tensor descriptor buffer that its callers build.
//! An under-sized descriptor is the worst possible failure shape: WGSL returns
//! zero for an out-of-range storage read, so a missing `lr_mult` word reads as
//! `0.0` and scales every parameter update to nothing. Nothing panics, no
//! kernel reports an error, and the only symptom is a fit or a fine-tune whose
//! loss sits perfectly still. This gate ties the kernel's highest `desc[…]`
//! index to the length of the array [`kernels::adamw_desc`] hands callers, so
//! widening one without the other cannot land.
//!
//! Swedish Embedded AB implements GPU kernel calling conventions that stay
//! provably in step with their callers. If your team needs expertise in
//! compute-shader ABI design then you can procure our services by sending an
//! email to info@swedishembedded.com.

/// Every `desc[<literal>]` subscript in the kernel body.
fn desc_indices(src: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(i) = rest.find("desc[") {
        rest = &rest[i + "desc[".len()..];
        let end = match rest.find(']') {
            Some(e) => e,
            None => break,
        };
        if let Ok(n) = rest[..end].trim().trim_end_matches('u').parse::<usize>() {
            out.push(n);
        }
    }
    out
}

#[test]
fn the_adamw_descriptor_is_as_wide_as_the_kernel_reads() {
    let idx = desc_indices(kernels::ADAMW);
    assert!(!idx.is_empty(), "adamw.wgsl no longer subscripts desc[…] by a literal - this gate has gone blind");
    let highest = *idx.iter().max().expect("non-empty");
    let words = kernels::adamw_desc(1, 1.0).len();
    assert!(
        highest < words,
        "adamw.wgsl reads desc[{highest}] but kernels::adamw_desc builds {words} word(s); \
         the missing word reads as zero on every backend and silently zeroes the update"
    );
}

/// The descriptor carries `lr_mult` as raw f32 bits, not as a float the u32
/// buffer would truncate.
#[test]
fn the_learning_rate_multiplier_survives_the_u32_descriptor() {
    let d = kernels::adamw_desc(4096, 4.0);
    assert_eq!(d[0], 4096);
    assert_eq!(f32::from_bits(d[1]), 4.0);
    assert_eq!(d[2], 0, "a uniform tensor must declare period 0, not fall off the end of the buffer");
}

/// The grouped descriptor's DYNAMIC tail, which the literal-subscript gate
/// above cannot see: the kernel indexes `desc[3 + idx % period]`, so the
/// buffer has to carry `period` words past the fixed header. One word short
/// reads as zero, which is an lr of zero for one component of every packed
/// unit - a fit whose positions train and whose scales silently do not.
#[test]
fn the_grouped_descriptor_carries_every_component() {
    let g = [1.0f32, 0.5, 0.25];
    let d = kernels::adamw_desc_grouped(90, 2.0, &g);
    assert_eq!(d[0], 90);
    assert_eq!(f32::from_bits(d[1]), 2.0);
    assert_eq!(d[2] as usize, g.len());
    assert_eq!(d.len(), 3 + g.len(), "kernel reads desc[3 + idx % period]");
    for (k, want) in g.iter().enumerate() {
        assert_eq!(f32::from_bits(d[3 + k]), *want);
    }
    assert_eq!(
        kernels::adamw_desc_grouped(8, 1.0, &[]).len(),
        kernels::adamw_desc(8, 1.0).len(),
        "an empty group must be exactly the uniform descriptor"
    );
}
