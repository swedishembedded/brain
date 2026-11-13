// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a model's weights are: resolved from the model store, with a
//! `--<role> PATH` flag per checkpoint role (`--dit`, `--text-encoder`, ...)
//! to name one outright.

use std::collections::BTreeMap;

use brain_modelstore::resolve::ArchSpec;
use capability::Assembly;

/// Split `args` into the `--<role> VALUE` overrides `spec` declares (keyed by
/// role, underscores as in the spec) and everything else, in order. A role
/// flag with no value after it is left for the caller's parser to refuse.
pub fn role_overrides(spec: &dyn ArchSpec, args: &[String]) -> (BTreeMap<String, String>, Vec<String>) {
    let mut overrides = BTreeMap::new();
    let mut remaining = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let role = spec.roles().iter().find(|r| args[i] == format!("--{}", r.replace('_', "-")) || args[i] == format!("--{r}"));
        match (role, args.get(i + 1)) {
            (Some(role), Some(value)) => {
                overrides.insert((*role).to_string(), value.clone());
                i += 2;
            }
            _ => {
                remaining.push(args[i].clone());
                i += 1;
            }
        }
    }
    (overrides, remaining)
}

/// The assembly of a model that reads no weights through the resolver.
pub fn empty_assembly() -> Assembly {
    Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() }
}

/// The help lines for `spec`'s role flags.
pub fn role_help(spec: &dyn ArchSpec) -> String {
    spec.roles().iter().map(|r| format!("  --{} <PATH>   this checkpoint role, instead of the model store's\n", r.replace('_', "-"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::ArtifactRecord;
    use brain_modelstore::resolve::{AssembleOutcome, AssembledVariant, Confidence};
    use std::path::Path;

    struct TwoRoleSpec;
    impl ArchSpec for TwoRoleSpec {
        fn arch(&self) -> &'static str {
            "two"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["dit", "text_encoder"]
        }
        fn classify(&self, _: &[ArtifactRecord], _: &Path) -> Vec<(usize, String, Confidence)> {
            Vec::new()
        }
        fn assemble(&self, _: &BTreeMap<String, usize>, _: &[ArtifactRecord], _: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/two".to_string(), variant: None }))
        }
        fn validate(&self, _: &Assembly) -> Result<(), String> {
            Ok(())
        }
    }

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// Only declared roles are taken out, in either spelling; a role flag
    /// with no value stays for the parser to refuse.
    #[test]
    fn only_declared_role_flags_are_taken() {
        let (o, rest) = role_overrides(&TwoRoleSpec, &s(&["--prompt", "a dog", "--text-encoder", "Qwen/Qwen3-8B", "--steps", "4", "--dit", "d.gguf", "--variant", "x"]));
        assert_eq!(o.get("text_encoder").map(String::as_str), Some("Qwen/Qwen3-8B"));
        assert_eq!(o.get("dit").map(String::as_str), Some("d.gguf"));
        assert_eq!(rest, s(&["--prompt", "a dog", "--steps", "4", "--variant", "x"]));
        let (o, rest) = role_overrides(&TwoRoleSpec, &s(&["--dit"]));
        assert!(o.is_empty());
        assert_eq!(rest, s(&["--dit"]));
    }
}
