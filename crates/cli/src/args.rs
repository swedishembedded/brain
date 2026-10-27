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
    model::saved::SavedPolicy::parse_activations(value).map_err(|e| format!("--{e}"))
}

/// The flag every training verb takes for what the backward keeps of the forward.
pub const ACTIVATIONS_FLAG: &str = "--activations";

/// The usage line of [`ACTIVATIONS_FLAG`].
pub const ACTIVATIONS_USAGE: &str = "[--activations keep|recompute]";

/// Take `--activations` off `args` for a verb that walks its own flag loop: the parsed policy (`None` when the flag
/// is absent, so the trainer keeps its own default) and the tokens it did not claim. A value other than `keep` or
/// `recompute` is refused naming the flag.
pub fn split_activations(args: &[String]) -> Result<(Option<model::saved::SavedPolicy>, Vec<String>), String> {
    let mut parsed = Args::new(args);
    let policy = parsed.take_str(ACTIVATIONS_FLAG).map(|value| parse_activations(&value)).transpose()?;
    Ok((policy, parsed.leftovers()))
}

/// [`split_activations`] for a verb that reports a usage mistake by exiting: a refused value is printed under the
/// verb's name (`what`) and the process exits 2 before any other work.
pub fn split_activations_or_exit(args: &[String], what: &str) -> (Option<model::saved::SavedPolicy>, Vec<String>) {
    split_activations(args).unwrap_or_else(|e| {
        eprintln!("{what}: {e}");
        std::process::exit(2);
    })
}

/// Fold `--activations` into the `activations=` key of `--train-precision`, for a verb that takes both. The two
/// spell the same choice, so naming different values is a conflict and is refused by name; naming the same value
/// twice, or only one of them, is accepted.
pub fn merge_activations(flag: Option<model::saved::SavedPolicy>, precision: &mut model::TrainPrecision) -> Result<(), String> {
    match (flag, precision.activations) {
        (Some(flag), Some(spec)) if flag != spec => {
            Err(format!("{ACTIVATIONS_FLAG} {flag:?} conflicts with activations={spec:?} in {TRAIN_PRECISION_FLAG}; name the choice once"))
        }
        (Some(flag), _) => {
            precision.activations = Some(flag);
            Ok(())
        }
        (None, _) => Ok(()),
    }
}

/// Give `model` the saved-activation policy a verb was asked for: `None` leaves the model's own default, and a
/// model that does not run the choice refuses it by name, so the verb exits before training instead of ignoring
/// the flag.
pub fn apply_activations<M: model::Model>(model: &mut M, policy: Option<model::saved::SavedPolicy>, name: &str) -> Result<(), String> {
    match policy {
        Some(policy) => model.set_saved_policy(policy).map_err(|e| format!("{name}: {ACTIVATIONS_FLAG} {policy:?}: {e}")),
        None => Ok(()),
    }
}

/// [`apply_activations`] for a verb that reports a refusal by exiting 1 before any training step.
pub fn apply_activations_or_exit<M: model::Model>(model: &mut M, policy: Option<model::saved::SavedPolicy>, name: &str) {
    if let Err(e) = apply_activations(model, policy, name) {
        eprintln!("{e}");
        std::process::exit(1);
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

#[cfg(test)]
mod split_activations_tests {
    use super::{merge_activations, split_activations};
    use model::saved::SavedPolicy;

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn the_flag_is_taken_off_and_the_rest_is_left_in_order() {
        let (keep, rest) = split_activations(&argv(&["data", "--activations", "keep", "--steps", "3"])).unwrap();
        assert_eq!(keep, Some(SavedPolicy::Keep));
        assert_eq!(rest, argv(&["data", "--steps", "3"]));
        let (recompute, _) = split_activations(&argv(&["--activations", "recompute"])).unwrap();
        assert_eq!(recompute, Some(SavedPolicy::Recompute));
    }

    #[test]
    fn an_absent_flag_leaves_the_trainers_own_default() {
        let (none, rest) = split_activations(&argv(&["data", "--steps", "3"])).unwrap();
        assert_eq!(none, None);
        assert_eq!(rest, argv(&["data", "--steps", "3"]));
    }

    #[test]
    fn a_bogus_value_is_refused_naming_the_flag() {
        let e = split_activations(&argv(&["--activations", "bogus"])).unwrap_err();
        assert!(e.contains("--activations") && e.contains("bogus"), "{e}");
    }

    #[test]
    fn the_flag_and_train_precision_may_agree_but_not_conflict() {
        let mut precision = model::TrainPrecision::parse("activations=recompute").unwrap();
        assert_eq!(merge_activations(None, &mut precision), Ok(()));
        assert_eq!(merge_activations(Some(SavedPolicy::Recompute), &mut precision), Ok(()));
        let e = merge_activations(Some(SavedPolicy::Keep), &mut precision).unwrap_err();
        assert!(e.contains("--activations") && e.contains("--train-precision"), "{e}");
        let mut unset = model::TrainPrecision::default();
        merge_activations(Some(SavedPolicy::Keep), &mut unset).unwrap();
        assert_eq!(unset.activations, Some(SavedPolicy::Keep));
    }
}
