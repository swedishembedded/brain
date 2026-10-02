// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain-cuda-aot`: compile brain's CUDA kernels offline.
//!
//! Swedish Embedded AB implements deployment pipelines for GPU software for its
//! clients, including shipping compiled kernels to machines that carry a
//! driver and no toolchain. If your team needs expertise in making a CUDA
//! application run on a locked-down box without a compiler, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! Builds every kernel the CUDA backend can be asked to run - the whole WGSL
//! catalogue through `wgsl-cuda`, and the hand-written kernels of
//! `kernels-cuda` - for each target of the toolkit lane in use, plus portable
//! PTX, and writes them with a manifest into the AOT directory. The CUDA
//! backend loads that directory at run time when it exists, so a machine with
//! a driver and no NVRTC runs the kernels built here.
//!
//! The sources are produced by the same code the backend runs, so the manifest
//! keys match what the backend asks for. Kernels the backend only creates at
//! run time (template variants specialised for a dtype or a KV tier) are not in
//! the catalogue and still need NVRTC.
//!
//! ```text
//! brain-cuda-aot [--out DIR] [--targets sm_61,sm_70,...] [--ptx compute_61|none] [--only SUBSTRING]
//! ```

use backend_cuda::aot::{self, Job, Plan};
use backend_cuda::nvrtc::{self, Target};
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    out: Option<PathBuf>,
    targets: Option<Vec<Target>>,
    ptx: Option<Option<(u32, u32)>>,
    only: Option<String>,
}

/// `sm_90` or `sm_90a`.
fn parse_target(s: &str) -> Result<Target, String> {
    let body = s.strip_prefix("sm_").ok_or_else(|| format!("`{s}`: a target looks like sm_90 or sm_90a"))?;
    let (digits, arch_specific) = match body.strip_suffix('a') {
        Some(d) => (d, true),
        None => (body, false),
    };
    Ok(Target { cc: split_cc(digits).ok_or_else(|| format!("`{s}`: not a compute capability"))?, arch_specific })
}

/// `61` as `(6, 1)`, `100` as `(10, 0)`: the minor is the last digit.
fn split_cc(digits: &str) -> Option<(u32, u32)> {
    let (major, minor) = digits.split_at(digits.len().checked_sub(1)?);
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { out: None, targets: None, ptx: None, only: None };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--out" => a.out = Some(value()?.into()),
            "--targets" => a.targets = Some(value()?.split(',').map(parse_target).collect::<Result<_, _>>()?),
            "--ptx" => {
                let v = value()?;
                a.ptx = Some(if v == "none" {
                    None
                } else {
                    let d = v.strip_prefix("compute_").ok_or_else(|| format!("--ptx {v}: expected compute_61 or none"))?;
                    Some(split_cc(d).ok_or_else(|| format!("--ptx {v}: not a compute capability"))?)
                })
            }
            "--only" => a.only = Some(value()?),
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(a)
}

const USAGE: &str = "usage: brain-cuda-aot [--out DIR] [--targets sm_61,sm_70,...] [--ptx compute_61|none] [--only SUBSTRING]";

/// Every kernel the backend can be asked for, as build jobs.
fn jobs() -> Result<Vec<Job>, String> {
    let mut jobs = Vec::new();
    let mut refused = Vec::new();
    for &(name, wgsl) in kernels::ALL {
        match wgsl_cuda::generate(name, wgsl) {
            Ok(g) => jobs.push(Job { kernel: name.into(), src: g.source, entry: g.entry, defines: vec![], min_cc: kernels_cuda::BASELINE_MIN_CC }),
            Err(e) => refused.push(format!("{name}: {}", e.lines().next().unwrap_or(""))),
        }
    }
    if !refused.is_empty() {
        // The catalogue is whole by contract (the backend's coverage test
        // asserts it); a refusal here is a generator regression to fix, and a
        // directory missing those kernels would hide it until a model needs one.
        return Err(format!("the generator refuses {} catalogue kernel(s):\n{}", refused.len(), refused.join("\n")));
    }
    for k in kernels_cuda::ALL {
        jobs.push(Job { kernel: k.reported.into(), src: k.src.into(), entry: k.entry.into(), defines: vec![], min_cc: k.min_cc });
    }
    Ok(jobs)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let out = args
        .out
        .or_else(aot::default_dir)
        .ok_or("no output directory: pass --out, or set BRAIN_CUDA_AOT_DIR")?;
    let toolkit = nvrtc::version()?;
    let mut plan = Plan::for_toolkit(toolkit);
    if let Some(t) = args.targets {
        plan.cubins = t;
    }
    if let Some(p) = args.ptx {
        plan.ptx = p;
    }
    let mut jobs = jobs()?;
    if let Some(only) = &args.only {
        jobs.retain(|j| j.kernel.contains(only.as_str()));
    }
    eprintln!(
        "brain-cuda-aot: {} kernels x [{}]{} with NVRTC {}.{} into {}",
        jobs.len(),
        plan.cubins.iter().map(Target::name).collect::<Vec<_>>().join(", "),
        plan.ptx.map(|(a, b)| format!(" + PTX compute_{a}{b}")).unwrap_or_default(),
        toolkit.0,
        toolkit.1,
        out.display()
    );
    let report = aot::build(&out, &jobs, &plan)?;
    for s in &report.skipped {
        eprintln!("skipped {s}");
    }
    eprintln!("brain-cuda-aot: wrote {} images and {}", report.built, out.join(aot::MANIFEST_FILE).display());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.is_empty() => {
            eprintln!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("brain-cuda-aot: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_parse_with_and_without_the_arch_suffix() {
        assert_eq!(parse_target("sm_61").unwrap(), Target::plain((6, 1)));
        assert_eq!(parse_target("sm_90a").unwrap(), Target { cc: (9, 0), arch_specific: true });
        assert_eq!(parse_target("sm_100").unwrap(), Target::plain((10, 0)));
        assert!(parse_target("compute_61").is_err());
        assert!(parse_target("sm_x").is_err());
    }

    /// The directory is useless if a refused kernel is silently absent from it.
    #[test]
    fn every_catalogue_and_native_kernel_is_a_job() {
        let jobs = jobs().expect("the whole catalogue translates");
        assert_eq!(jobs.len(), kernels::ALL.len() + kernels_cuda::ALL.len());
        let mut names: Vec<&str> = jobs.iter().map(|j| j.kernel.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), jobs.len(), "a kernel name is the manifest's replacement slot, so it must be unique");
    }
}
