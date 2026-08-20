// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain quantize` - the export sibling of `brain import`.
//!
//! Swedish Embedded AB implements checkpoint conversion and on-device
//! quantization pipelines for its clients. If your team needs expertise in
//! shrinking a model to fit the accelerator it has to run on, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! `brain import` reads a quantized GGUF and produces brain's internal
//! format, picking a hand-written per-architecture importer from the file's
//! own `general.architecture`. This goes the other way and needs no
//! per-architecture code at all: `checkpoint::quantize` decides each tensor's
//! fate from its shape plus a name list supplied on the command line, so a
//! checkpoint brain has never seen converts without a new importer, a new
//! subcommand, or an edit to any registry.
//!
//! What it deliberately does NOT do is guess a model's never-quantize list.
//! Which named tensors a given architecture must keep at full precision is
//! knowledge about that architecture (modulation tables, conditioning
//! projections, anything whose numeric scale the rest of the graph rides on)
//! and no shape implies it. `--keep` takes it explicitly; the default is
//! structural rules only, and the printed plan says what that decided for
//! every tensor so the answer is auditable rather than assumed.

use std::time::Instant;

use checkpoint::gguf::GgufValue;
use checkpoint::quantize::{convert, plan, Decision, Policy, Report, Tier};

const USAGE: &str = "\
brain quantize SRC --out PATH [options]

  SRC              a .safetensors file, a HuggingFace-style directory of
                   them, or an existing .gguf - anything implementing
                   checkpoint::TensorSource with a manifest.

  --out PATH       destination .gguf (written via a .tmp + rename).
  --tier T         target tier. Q8_0 (default) is the only tier with a
                   consumer in this tree today.
  --arch NAME      value for the output's `general.architecture` KV. Defaults
                   to the source's own when the source is a GGUF, else the
                   destination file's stem.
  --name NAME      value for `general.name`. Defaults to the source's stem.
  --keep SUBSTR    never quantize a tensor whose name contains SUBSTR.
                   Repeatable, or comma-separated.
  --min-elems N    never quantize a tensor with fewer than N elements
                   (default 0: structural rules only).
  --plan           print the per-tensor plan and the totals, write nothing.
  --quiet          only print the summary.

Structural rules are not options: a tensor is quantizable only if it is
rank 2 and its fastest-varying dimension is a whole number of blocks. Every
other tensor is written through as F32, and the plan says which rule kept it.
";

/// Parsed `brain quantize` arguments.
struct Args {
    src: String,
    out: Option<String>,
    tier: Tier,
    arch: Option<String>,
    name: Option<String>,
    keep: Vec<String>,
    min_elems: usize,
    plan_only: bool,
    quiet: bool,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        src: String::new(),
        out: None,
        tier: Tier::Q8_0,
        arch: None,
        name: None,
        keep: Vec::new(),
        min_elems: 0,
        plan_only: false,
        quiet: false,
    };
    let mut i = 0;
    let need = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("brain quantize: {flag} needs a value"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--out" => a.out = Some(need(&mut i, "--out")?),
            "--tier" => {
                let t = need(&mut i, "--tier")?;
                a.tier = match t.to_ascii_uppercase().as_str() {
                    "Q8_0" => Tier::Q8_0,
                    other => return Err(format!("brain quantize: unknown tier '{other}' (supported: Q8_0)")),
                };
            }
            "--arch" => a.arch = Some(need(&mut i, "--arch")?),
            "--name" => a.name = Some(need(&mut i, "--name")?),
            "--keep" => a.keep.extend(need(&mut i, "--keep")?.split(',').filter(|s| !s.is_empty()).map(str::to_string)),
            "--min-elems" => {
                let v = need(&mut i, "--min-elems")?;
                a.min_elems = v.parse().map_err(|_| format!("brain quantize: --min-elems wants a number, got '{v}'"))?;
            }
            "--plan" => a.plan_only = true,
            "--quiet" => a.quiet = true,
            "-h" | "--help" => return Err(String::new()),
            other if other.starts_with('-') => return Err(format!("brain quantize: unknown option '{other}'")),
            other if a.src.is_empty() => a.src = other.to_string(),
            other => return Err(format!("brain quantize: unexpected argument '{other}'")),
        }
        i += 1;
    }
    if a.src.is_empty() {
        return Err("brain quantize: no source given".to_string());
    }
    Ok(a)
}

/// The source, kept alive as a concrete type: `TensorManifest` is used
/// through a reference, and both variants are memory-mapped, so opening one
/// touches the header only.
enum Source {
    Gguf(checkpoint::gguf::MmapGguf),
    Safetensors(checkpoint::weightio::WeightReader),
}

impl Source {
    fn open(path: &str) -> Result<Source, String> {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            return checkpoint::weightio::WeightReader::open_hf_dir(p).map(Source::Safetensors).map_err(|e| format!("opening {path}: {e}"));
        }
        if path.ends_with(".gguf") {
            return checkpoint::gguf::MmapGguf::open(path).map(Source::Gguf).map_err(|e| format!("opening {path}: {e}"));
        }
        checkpoint::weightio::WeightReader::open(path).map(Source::Safetensors).map_err(|e| format!("opening {path}: {e}"))
    }

    fn manifest(&self) -> &dyn checkpoint::quantize::TensorManifest {
        match self {
            Source::Gguf(g) => g,
            Source::Safetensors(w) => w,
        }
    }

    /// The source's own declared architecture, when it has one.
    fn architecture(&self) -> Option<String> {
        match self {
            Source::Gguf(g) => g.kv().get("general.architecture").and_then(|v| v.as_str()).map(str::to_string),
            Source::Safetensors(_) => None,
        }
    }
}

fn stem(path: &str) -> String {
    std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
}

fn human(bytes: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}

/// Print the totals every run ends with, whether or not it wrote anything.
fn print_summary(report: &Report, elapsed_s: Option<f64>) {
    let (q, k) = (report.quantized(), report.kept());
    println!(
        "  {} tensors: {q} quantized to {}, {k} written through as F32",
        report.rows.len(),
        report.tier.name()
    );
    println!("  {} parameters quantized", report.quantized_params());
    println!(
        "  {} of tensor data (against {} at f32: {:.2}x smaller)",
        human(report.output_bytes()),
        human(report.f32_bytes()),
        report.f32_bytes() as f64 / report.output_bytes().max(1) as f64
    );
    // Why each kept tensor was kept, aggregated: a count with no reason is
    // exactly what lets a real weight matrix silently miss the fast path.
    let mut reasons: std::collections::BTreeMap<String, (usize, u64)> = std::collections::BTreeMap::new();
    for r in report.rows.iter().filter(|r| !r.quantized()) {
        let key = match &r.decision {
            Decision::Keep(k) => match k {
                checkpoint::quantize::Kept::NotRank2 { rank } => format!("not rank 2 (rank {rank})"),
                checkpoint::quantize::Kept::RowNotBlockAligned { row, block } => format!("row {row} is not a multiple of {block}"),
                checkpoint::quantize::Kept::TooSmall { min, .. } => format!("fewer than {min} elements"),
                checkpoint::quantize::Kept::NeverQuantize { pattern } => format!("name contains '{pattern}'"),
            },
            Decision::Quantize => unreachable!("filtered to kept rows"),
        };
        let e = reasons.entry(key).or_insert((0, 0));
        e.0 += 1;
        e.1 += r.nbytes as u64;
    }
    for (why, (n, bytes)) in reasons {
        println!("    kept {n:>5} ({:>10}): {why}", human(bytes));
    }
    if let Some(s) = elapsed_s {
        println!("  {s:.1}s");
    }
}

/// `brain quantize` entry point. Exits the process.
pub fn run_quantize(argv: &[String]) -> ! {
    let args = match parse(argv) {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("{msg}\n");
            }
            print!("{USAGE}");
            std::process::exit(if msg.is_empty() { 0 } else { 2 });
        }
    };

    let src = match Source::open(&args.src) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("brain quantize: {e}");
            std::process::exit(1);
        }
    };
    let manifest = src.manifest();

    let mut policy = Policy::new().min_elems(args.min_elems);
    if !args.keep.is_empty() {
        policy = policy.never_quantize(&args.keep.iter().map(String::as_str).collect::<Vec<_>>());
    }

    if args.plan_only {
        let rows = match plan(manifest, args.tier, &policy) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("brain quantize: {e}");
                std::process::exit(1);
            }
        };
        if !args.quiet {
            for r in &rows {
                let what = if r.quantized() { args.tier.name().to_string() } else { format!("F32  ({:?})", r.decision) };
                println!("{:<70} {:>18?} {:>12} {what}", r.name, r.shape, human(r.nbytes as u64));
            }
        }
        println!("plan for {} -> (nothing written)", args.src);
        print_summary(&Report { tier: args.tier, rows }, None);
        std::process::exit(0);
    }

    let Some(out) = args.out.clone() else {
        eprintln!("brain quantize: --out is required (or pass --plan to see what would happen)\n");
        print!("{USAGE}");
        std::process::exit(2);
    };

    let arch = args.arch.clone().or_else(|| src.architecture()).unwrap_or_else(|| stem(&out));
    let name = args.name.clone().unwrap_or_else(|| stem(&args.src));
    let kv = vec![
        ("general.architecture".to_string(), GgufValue::String(arch.clone())),
        ("general.name".to_string(), GgufValue::String(name)),
        // ggml's file-type tag for an all-Q8_0 file. Informational: every
        // tensor also carries its own type, which is what the reader uses.
        ("general.file_type".to_string(), GgufValue::U32(7)),
        ("general.quantization_version".to_string(), GgufValue::U32(2)),
        ("general.source.name".to_string(), GgufValue::String(stem(&args.src))),
    ];

    println!("brain quantize: {} -> {out} ({}, architecture '{arch}')", args.src, args.tier.name());
    let started = Instant::now();
    let total = manifest.tensor_names().len();
    let mut done_bytes: u64 = 0;
    let quiet = args.quiet;
    let report = convert(manifest, args.tier, &policy, &kv, &out, &mut |i, row| {
        done_bytes += row.nbytes as u64;
        // Progress at a fixed cadence rather than per tensor: a real
        // conversion is hundreds of tensors over tens of minutes, and a
        // converter that prints nothing is indistinguishable from a hung one.
        if !quiet && (i + 1) % 32 == 0 || i + 1 == total {
            let s = started.elapsed().as_secs_f64();
            println!(
                "  [{:>4}/{total}] {} written, {:.0} MiB/s, {s:.0}s elapsed  ({})",
                i + 1,
                human(done_bytes),
                done_bytes as f64 / s.max(1e-9) / (1024.0 * 1024.0),
                row.name
            );
        }
    });

    match report {
        Ok(report) => {
            let on_disk = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
            println!("wrote {out} ({} on disk)", human(on_disk));
            print_summary(&report, Some(started.elapsed().as_secs_f64()));
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("brain quantize: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_the_documented_options() {
        let a = parse(&s(&["src.safetensors", "--out", "o.gguf", "--keep", "embed,norm", "--min-elems", "4096", "--arch", "gemma4"])).unwrap();
        assert_eq!(a.src, "src.safetensors");
        assert_eq!(a.out.as_deref(), Some("o.gguf"));
        assert_eq!(a.keep, vec!["embed".to_string(), "norm".to_string()]);
        assert_eq!(a.min_elems, 4096);
        assert_eq!(a.arch.as_deref(), Some("gemma4"));
        assert_eq!(a.tier, Tier::Q8_0);
    }

    #[test]
    fn an_unknown_tier_is_refused_by_name_rather_than_silently_defaulted() {
        let err = parse(&s(&["src", "--tier", "Q4_K_M"])).unwrap_err();
        assert!(err.contains("Q4_K_M"), "the error must name the tier asked for: {err}");
    }

    #[test]
    fn a_missing_source_is_an_error_not_an_empty_conversion() {
        assert!(parse(&s(&["--out", "o.gguf"])).unwrap_err().contains("no source"));
    }
}
