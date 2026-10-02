// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which of the WGSL catalogue the CUDA backend can actually run.
//!
//! Swedish Embedded AB implements portable GPU compute stacks and the coverage
//! gates that keep a second backend honest for its clients. If your team needs
//! expertise in bringing an existing kernel catalogue to CUDA without a
//! hand-written copy of every kernel, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! The generated tier turns each WGSL kernel into CUDA C++ and compiles it with
//! NVRTC on first dispatch, so an unsupported kernel is discovered at the model
//! step that needs it. This test moves that discovery to one place: every
//! kernel in `kernels::ALL` is generated, and where a CUDA device and NVRTC are
//! present, every generated source is compiled for the device's own
//! architecture. A kernel falls into exactly one of:
//!
//! - **runnable**: generated and compiled;
//! - **refused**: the generator names a construct it does not support (a real,
//!   tracked gap, never an approximation);
//! - **broken**: generated but NVRTC rejects it, which is a generator defect.
//!
//! Set `BRAIN_CUDA_COVERAGE_OUT` to a path to write the report as JSON. The
//! assertions are a ratchet: no kernel may be broken, and none may be refused -
//! the whole catalogue is generated, so a new kernel that uses a construct the
//! generator lacks fails here, by name, instead of at the model step that needs
//! it.

use std::collections::BTreeMap;

use kernels::ALL;

/// A coarse, stable category for a generator refusal, so a report can be
/// grouped and a fix can be sized by how many kernels it unlocks.
fn category(reason: &str) -> &'static str {
    if reason.contains("barrier") {
        "barrier-in-loop"
    } else if reason.contains("f16") {
        "f16"
    } else if reason.contains("workgroup") {
        "workgroup-shape"
    } else if reason.contains("Atomic") || reason.contains("atomic") {
        "atomics"
    } else if reason.contains("Call") || reason.contains("function call") {
        "function-call"
    } else if reason.contains("WGSL parse") {
        "parse"
    } else {
        "other"
    }
}

struct Outcome {
    name: &'static str,
    state: &'static str,
    detail: String,
}

fn classify(compile: &dyn Fn(&str, &str) -> Result<(), String>, compiling: bool) -> Vec<Outcome> {
    let mut out = Vec::with_capacity(ALL.len());
    for &(name, src) in ALL {
        match wgsl_cuda::generate(name, src) {
            Err(reason) => out.push(Outcome { name, state: "refused", detail: format!("{}: {}", category(&reason), reason.lines().next().unwrap_or("")) }),
            Ok(k) if !compiling => out.push(Outcome { name, state: "generated", detail: k.entry }),
            Ok(k) => match compile(&k.source, &k.entry) {
                Ok(()) => out.push(Outcome { name, state: "runnable", detail: k.entry }),
                Err(e) => out.push(Outcome { name, state: "broken", detail: e.lines().take(3).collect::<Vec<_>>().join(" | ") }),
            },
        }
    }
    out
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ")
}

#[test]
fn every_catalogue_kernel_is_runnable_or_refused_by_name() {
    // The device and its toolkit, when present. Without them the generator
    // half still runs: it needs no driver.
    let ctx = backend_cuda::exec::Context::open(0).ok();
    let compiling = ctx.is_some() && backend_cuda::nvrtc::version().is_ok();
    if !compiling {
        brain_testutil::skip_unavailable("no CUDA device or NVRTC: checking generation only, not compilation");
    }
    let compile = |src: &str, entry: &str| -> Result<(), String> {
        ctx.as_ref().expect("compiling implies a context").cubin(src, entry).map(|_| ())
    };
    let outcomes = classify(&compile, compiling);

    let mut by_state: BTreeMap<&str, usize> = BTreeMap::new();
    let mut refused_by_category: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for o in &outcomes {
        *by_state.entry(o.state).or_default() += 1;
        if o.state == "refused" {
            let cat = o.detail.split(':').next().unwrap_or("other").to_string();
            refused_by_category.entry(cat).or_default().push(o.name);
        }
    }
    println!("catalogue coverage over {} kernels: {by_state:?}", outcomes.len());
    for (cat, names) in &refused_by_category {
        println!("  refused for {cat}: {}", names.len());
    }

    if let Ok(path) = std::env::var("BRAIN_CUDA_COVERAGE_OUT") {
        let rows: Vec<String> = outcomes
            .iter()
            .map(|o| format!("    {{\"kernel\": \"{}\", \"state\": \"{}\", \"detail\": \"{}\"}}", o.name, o.state, json_escape(&o.detail)))
            .collect();
        let cc = ctx.as_ref().map(|c| format!("\"{}.{}\"", c.compute_capability().0, c.compute_capability().1)).unwrap_or_else(|| "null".into());
        let body = format!("{{\n  \"compute_capability\": {cc},\n  \"kernels\": [\n{}\n  ]\n}}\n", rows.join(",\n"));
        std::fs::write(&path, body).unwrap_or_else(|e| panic!("writing {path}: {e}"));
    }

    let broken: Vec<_> = outcomes.iter().filter(|o| o.state == "broken").map(|o| format!("{}: {}", o.name, o.detail)).collect();
    assert!(broken.is_empty(), "{} generated kernel(s) NVRTC rejects, a generator defect:\n{}", broken.len(), broken.join("\n"));
    let refused: Vec<_> = outcomes.iter().filter(|o| o.state == "refused").map(|o| format!("{}: {}", o.name, o.detail)).collect();
    assert!(refused.is_empty(), "{} catalogue kernel(s) the generator refuses:\n{}", refused.len(), refused.join("\n"));
}
