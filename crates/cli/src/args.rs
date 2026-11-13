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

pub use appopts::Args;

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
