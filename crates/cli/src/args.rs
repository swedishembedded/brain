// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared CLI argument grammar for the model commands (`gpt`/`qwen`/`glm`).
//!
//! The grammar itself now lives in `crates/appopts`, because the samples need
//! the same one and a second ad-hoc parser per application is how
//! `--flag` at the end of a line comes to mean two different things. It is
//! re-exported here so every `crate::args::Args` call site is unchanged.
//!
//! What stays is the part that is genuinely about THIS binary: how `--out`
//! spells a blob name, and how a verb is canonicalised.

use std::path::Path;

pub use appopts::Args;

/// The seed a distributed rank trains with when `--seed` was not given: the
/// digest of the plan file, which every rank of the job holds byte for byte.
/// Same plan, same starting weights and batch order on every rank; ranks that
/// drew different seeds would diverge before the first reduction. `what` names
/// the verb in the notice.
pub(crate) fn plan_seed(plan: &Path, what: &str) -> std::io::Result<u64> {
    use sha2::{Digest as _, Sha256};
    let digest = Sha256::digest(std::fs::read(plan)?);
    let seed = u64::from_le_bytes(digest[..8].try_into().expect("a SHA-256 digest holds eight bytes"));
    eprintln!("{what}: no --seed given, using seed {seed} from the job plan {}", plan.display());
    Ok(seed)
}

/// Resolve `--seed` for a verb that walks its own flag loop: the value given,
/// or a freshly drawn random one printed to stderr (see [`Args::seed`]), plus
/// the tokens the seed did not claim for the verb's own loop. A `--seed` that
/// is not a seed exits 2 before any other flag is looked at. A help request
/// draws silently: nobody is running anything, so there is nothing to
/// reproduce.
pub fn split_seed(args: &[String], what: &str) -> (u64, Vec<String>) {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return (data::rng::random_seed(), args.to_vec());
    }
    let mut parsed = Args::new(args);
    let seed = parsed.seed(what);
    (seed, parsed.leftovers())
}

/// The flag every full fine-tune verb takes for how the run stores what it trains: `--train-precision SPEC`, one
/// model-agnostic value parsed by [`model::TrainPrecision::parse`] (`state=int8,master=f32,sr=on,narrow=on`, and
/// `weights=fp8,window=2,clip=two-pass` for the linears resident in a low-precision encoding, optionally streamed
/// through a window of layers). A model that does not run a choice refuses it by name when the run starts
/// ([`model::TrainPrecisionSupport::check`]); no model has a flag of its own for this.
pub const TRAIN_PRECISION_FLAG: &str = "--train-precision";

/// The usage line of [`TRAIN_PRECISION_FLAG`].
pub const TRAIN_PRECISION_USAGE: &str = "[--train-precision state=f32|bf16|int8[:B]|log8[:B],master=none|bf16|f32,grad=f32|bf16,sr=on|off,narrow=on|off,weights=bf16|fp8[:B]|int8|nf4,window=N,clip=two-pass|running]";

/// Take `--train-precision` off `args` for a verb that walks its own flag loop: the parsed value (the all-fp32
/// default when the flag is absent) and the tokens it did not claim.
pub fn split_train_precision(args: &[String]) -> Result<(model::TrainPrecision, Vec<String>), String> {
    let mut parsed = Args::new(args);
    let precision = match parsed.take_str(TRAIN_PRECISION_FLAG) {
        Some(spec) => model::TrainPrecision::parse(&spec).map_err(|e| format!("{TRAIN_PRECISION_FLAG} {spec:?}: {e}"))?,
        None => model::TrainPrecision::default(),
    };
    Ok((precision, parsed.leftovers()))
}

/// The value of `--activations`: what a trainer's backward keeps of its forward, `keep` every block's activations or
/// `recompute` them from each block's input. Any other value is refused, naming the flag.
pub fn parse_activations(value: &str) -> Result<model::saved::SavedPolicy, String> {
    use model::saved::SavedPolicy;
    match SavedPolicy::parse(value) {
        Some(policy @ (SavedPolicy::Keep | SavedPolicy::Recompute)) => Ok(policy),
        _ => Err(format!("--activations: {value} is not one of keep, recompute")),
    }
}

/// `--out`'s value, accepting either a dedicated command's own bare-path
/// spelling (`--out out.png`) or the generic capability-manifest convention
/// `brain caps <arch>` documents (`--out <blob_name>=<path>`, what `brain
/// do`/D-Bus callers actually type) - the manifest line is the only place a
/// caller learns an action's `--out` syntax from, and typing it against a
/// dedicated command whose own ad-hoc parser only knew the bare form
/// used to silently write a file literally named `image=out.png` (or worse,
/// turn a path VALUE into a nested `image=/...` directory tree), with no
/// error at all - found live on `brain flux2 generate`. Strips `blob_name=`
/// when present; anything else (including a value that merely starts with
/// the name as part of its own filename, not followed by `=`) passes
/// through unchanged.
///
/// A malformed value exits the process here, naming the flag, exactly as a
/// value-less `--flag` does in [`Args::take_str`] above - the mistake is the
/// caller's typing and this is the last point at which it is still
/// attributable to it.
pub fn strip_out_name_prefix<'a>(raw: &'a str, blob_name: &str) -> &'a str {
    parse_out_name_prefix(raw, blob_name).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    })
}

/// [`strip_out_name_prefix`]'s fallible core.
///
/// `--out image=` names the blob and then nothing at all. Stripping the prefix
/// leaves an EMPTY path, which fails unpredictably deep inside whatever opens
/// the output file - a "No such file or directory" on a path the caller never
/// typed, or, before the prefix form was understood at all, a file literally
/// named `image=`. Neither is something a caller can act on, so the empty
/// value is refused here instead, quoting what was typed.
pub fn parse_out_name_prefix<'a>(raw: &'a str, blob_name: &str) -> Result<&'a str, String> {
    let path = raw.strip_prefix(blob_name).and_then(|r| r.strip_prefix('=')).unwrap_or(raw);
    if path.is_empty() {
        return Err(format!("--out {raw:?} names no output path; write it as '--out <path>' or '--out {blob_name}=<path>'"));
    }
    Ok(path)
}

/// Canonicalise a verb, accepting back-compat aliases so old invocations keep
/// working across the unified model CLIs (`gen`→`infer`, `fine-tune`→`finetune`,
/// `sample`→`infer`).
pub fn canon_verb(v: &str) -> &str {
    match v {
        "gen" | "sample" | "generate" => "infer",
        "fine-tune" => "finetune",
        other => other,
    }
}

#[cfg(test)]
mod out_name_prefix_tests {
    use super::{parse_out_name_prefix, strip_out_name_prefix};

    #[test]
    fn accepts_either_the_bare_path_or_the_documented_name_equals_path_form() {
        assert_eq!(strip_out_name_prefix("out.png", "image"), "out.png");
        assert_eq!(strip_out_name_prefix("image=out.png", "image"), "out.png");
        assert_eq!(strip_out_name_prefix("image=out/a/b.png", "image"), "out/a/b.png");
        assert_eq!(strip_out_name_prefix("imageX=out.png", "image"), "imageX=out.png");
        assert_eq!(strip_out_name_prefix("adapter=my.brain", "adapter"), "my.brain");
        assert_eq!(strip_out_name_prefix("video=clip.mp4", "video"), "clip.mp4");
    }

    /// `--out image=` names the blob and then no path at all. Stripping the
    /// prefix leaves an EMPTY path, which fails unpredictably deep inside
    /// whatever opens the file (and on some platforms creates nothing at all)
    /// rather than telling the caller what they typed wrong. Refuse it here,
    /// where the mistake is still attributable to the flag.
    #[test]
    fn a_name_equals_with_no_path_is_refused_rather_than_becoming_an_empty_path() {
        let e = parse_out_name_prefix("image=", "image").unwrap_err();
        assert!(e.contains("--out"), "the refusal must name the flag: {e}");
        assert!(e.contains("image="), "and quote what was typed: {e}");
        // A bare empty value is the same mistake without the prefix.
        assert!(parse_out_name_prefix("", "image").is_err());
        // Everything the accepted forms above allow still parses.
        assert_eq!(parse_out_name_prefix("image=out.png", "image").unwrap(), "out.png");
        assert_eq!(parse_out_name_prefix("out.png", "image").unwrap(), "out.png");
        assert_eq!(parse_out_name_prefix("imageX=", "image").unwrap(), "imageX=");
    }
}

#[cfg(test)]
mod activations_tests {
    use super::parse_activations;
    use model::saved::SavedPolicy;

    #[test]
    fn activations_accept_keep_and_recompute_and_refuse_anything_else_naming_the_flag() {
        assert_eq!(parse_activations("keep"), Ok(SavedPolicy::Keep));
        assert_eq!(parse_activations("recompute"), Ok(SavedPolicy::Recompute));
        for bad in ["recompute+bf16", "on", ""] {
            let e = parse_activations(bad).unwrap_err();
            assert!(e.contains("--activations"), "{e}");
        }
    }
}
