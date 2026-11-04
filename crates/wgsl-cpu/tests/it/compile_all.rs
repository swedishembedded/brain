// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use backend_api::DType;

#[test]
fn all_kernels_compile() {
    let mut failed = Vec::new();
    for (name, src) in kernels::ALL {
        match wgsl_cpu::Jit::new(&[(name, src)]) {
            Ok(_) => {}
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }
    if !failed.is_empty() {
        panic!(
            "{} of {} kernels failed:\n{}",
            failed.len(),
            kernels::ALL.len(),
            failed.join("\n")
        );
    }
}

/// B6's portability gate: every kernel whose header `@dtype` field declares a
/// storage tier beyond the `f32`/`n/a` default (today: `matmul`,
/// `matmul_gemv`, `matmul_reg3`, all `f32|bf16|f16`) must have EVERY declared
/// tier actually reachable on the CPU JIT - the lowest-common-denominator
/// backend - with exactly one documented exception: a kernel with more than
/// one top-level `workgroupBarrier()` (`matmul_reg3` today), which the JIT's
/// one-barrier execution model can never express regardless of dtype - this
/// is already proven harmless elsewhere (a dedicated model-crate test asserts
/// `select::candidates` filters `RegisterTiled` out on CPU caps before
/// dispatch ever reaches it, substituting `Reference` instead). This test
/// asserts that EXACT, narrow exception - a kernel
/// cannot claim a tier it cannot execute for some OTHER, undocumented
/// reason - rather than silently accepting any failure.
///
/// A base (non-`@dtype`-tiered) kernel needs no cross-product here - it is
/// already covered, unchanged, by `all_kernels_compile` above.
#[test]
fn dtype_tiers_compile_or_fail_only_for_the_documented_barrier_reason() {
    let mut exercised = 0usize;
    let mut unexpected = Vec::new();

    for (name, src) in kernels::ALL {
        let tiers = declared_dtype_tiers(src);
        if tiers.is_empty() {
            continue;
        }
        let bindings = tpl_bindings(src);
        if bindings.is_empty() {
            panic!(
                "{name}: @dtype declares a storage tier ({tiers:?}) but has no \
                 `@tpl <binding>[, <binding>] -> ...` header line naming which bindings to templatize"
            );
        }
        // A comma-separated `@tpl` (`paged_flash_decode`: `pool_k,pool_v`;
        // `paged_flash_prefill_hd256`: `pool_k, pool_v`) names MULTIPLE
        // independent bindings, each needing its own `dtype_variant` call
        // chained over the previous call's own output - exactly what those
        // kernels' headers document ("chained twice... dtype_variant(..,
        // "pool_k", BF16) then dtype_variant again over ITS OWN output with
        // "pool_v""). `tpl_bindings` already split the list.
        // `dtype_variant` only rewrites a binding's declaration/loads, never
        // control flow (see its own doc comment), so a variant's barrier
        // count is always identical to its base kernel's - the ONE
        // documented exception below is known ahead of the call, not
        // discovered by trial and error.
        let base_barriers = src.matches("workgroupBarrier").count();
        let expect_compiles = base_barriers <= 1;

        for dt in tiers {
            exercised += 1;
            let (mut vname, mut vsrc): (&str, &'static str) = (*name, *src);
            for binding in bindings.iter().copied() {
                let variant = kernels::template::dtype_variant(vname, vsrc, binding, dt)
                    .unwrap_or_else(|e| {
                        panic!("{name}: dtype_variant({dt:?}, {binding:?}) build failed: {e}")
                    });
                vname = variant.0;
                vsrc = variant.1;
            }
            let jit = wgsl_cpu::Jit::new(&[(vname, vsrc)]).unwrap_or_else(|e| {
                panic!("{vname}: Jit::new failed (not a barrier-structural issue): {e}")
            });
            let compiled = kernel_is_compiled(&jit, vname);
            if compiled != expect_compiles {
                unexpected.push(format!(
                    "{vname}: base kernel has {base_barriers} barrier(s), expected {} but the \
                     JIT compiled = {compiled}",
                    if expect_compiles {
                        "it to compile"
                    } else {
                        "the known >1-barrier refusal"
                    }
                ));
            }
        }
    }

    assert!(
        exercised > 0,
        "no kernel declared a bf16/f16 @dtype tier - the cross-product exercised nothing"
    );
    assert!(
        unexpected.is_empty(),
        "{} unexpected dtype-variant compile result(s):\n{}",
        unexpected.len(),
        unexpected.join("\n")
    );
}

/// Whether kernel `name` in `jit` actually produced runnable native code.
/// `Jit::new`'s own success/failure cannot say this on its own - a >1-barrier
/// kernel is a documented SOFT skip (an `eprintln`, that kernel's slot left
/// `None`), not a hard `Err`, exactly so a mixed kernel set can still build
/// (see `crates/wgsl-cpu/src/lib.rs`'s own module/struct doc comments). So
/// this calls the compiled function over an EMPTY invocation range
/// (`start == end == 0`, `catch_unwind`ing the panic `Jit::run` raises for an
/// uncompiled kernel - see its own doc comment) instead of dereferencing
/// anything: the entry block for both the plain and work-group execution
/// paths unconditionally loads each storage binding's BASE POINTER before
/// the per-invocation loop runs, so that needs real backing memory, but
/// nothing beyond that base pointer is ever read or written when the loop
/// body itself never executes.
fn kernel_is_compiled(jit: &wgsl_cpu::Jit, name: &str) -> bool {
    let idx = jit
        .index_of(name)
        .unwrap_or_else(|| panic!("{name}: not registered in this Jit"));
    // Every kernel in this tree binds <=8 storage buffers
    // (`crates/kernels/src/lib.rs`'s own module doc) - 16 slots is margin,
    // not a measured bound.
    let bufs: [*mut u8; 16] = [std::ptr::null_mut(); 16];
    let uniform = [0u32; 64];
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // the expected-panic path is asserted, not printed
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        jit.run(idx, 0, 0, 1, 1, uniform.as_ptr(), bufs.as_ptr());
    }));
    std::panic::set_hook(hook);
    result.is_ok()
}

/// The `bf16`/`f16` tiers a kernel's own `// @dtype ...` header line declares
/// beyond the `f32` default - `f32`/`n/a` need no cross-product (the base
/// kernel is already covered by `all_kernels_compile`).
fn declared_dtype_tiers(src: &str) -> Vec<DType> {
    let line = src.lines().find(|l| l.trim_start().starts_with("// @dtype")).unwrap_or_else(|| {
        panic!(
            "kernel source has no `// @dtype` header line (run \
             `scripts/build/seed-kernel-meta.py`)"
        )
    });
    let value = line.trim_start().trim_start_matches("// @dtype").trim();
    value
        .split('|')
        .filter_map(|t| match t {
            "bf16" => Some(DType::BF16),
            "f16" => Some(DType::F16),
            _ => None, // "f32" and "n/a" need no templated variant
        })
        .collect()
}

/// The binding names a kernel's `// @tpl <binding>[, <binding>] -> ... storage
/// variant` header line names - the field B4 added and explicitly deferred
/// parsing of to "B6" in its own comment (see `matmul.wgsl`'s header: "header
/// field, parsing deferred to B6"); this is that parse. The list is the text
/// before `->`, split on commas with the optional space after each trimmed;
/// empty when the kernel has no `@tpl` line.
fn tpl_bindings(src: &str) -> Vec<&str> {
    let Some(line) = src.lines().find(|l| l.trim_start().starts_with("// @tpl")) else {
        return Vec::new();
    };
    let names = line.trim_start().trim_start_matches("// @tpl");
    let names = names.split("->").next().unwrap_or(names);
    names.split(',').map(str::trim).filter(|b| !b.is_empty()).collect()
}
