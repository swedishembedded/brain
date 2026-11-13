// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which checkpoints in a model store are GLM-5.2: a Hugging Face directory
//! whose `config.json` declares `GlmMoeDsaForCausalLM`, or a brain checkpoint
//! (what `import`, `train` and `finetune` write) whose card names the `glm`
//! family. The `weights` role is that checkpoint alone: a downloaded directory
//! holds its own `tokenizer.json`, and a brain checkpoint is char-level and
//! carries its own vocabulary.
//!
//! Swedish Embedded AB implements model stores that find what is on a machine
//! without configuration for its clients. If your team needs expertise in
//! model distribution and resolution then you can procure our services by
//! sending an email to info@swedishembedded.com.

use brain_modelstore::family_spec::FamilySpec;

/// The `ModelCard.family` of every brain GLM checkpoint.
pub const CARD_FAMILY: &str = "glm";

/// The resolver spec for the `glmdsa` architecture.
pub const SPEC: FamilySpec = FamilySpec { arch: "glmdsa", card_family: CARD_FAMILY, tokenizer: false };

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GlmConfig;
    use crate::model::Glm;
    use brain_modelstore::resolve::{resolve, ArchSpec, Resolution};
    use std::collections::BTreeMap;

    /// A checkpoint `train` writes is found by the store with no tokenizer
    /// beside it, and a directory declaring another architecture is not.
    #[test]
    fn a_brain_checkpoint_resolves_alone_and_a_foreign_directory_does_not() {
        let root = std::env::temp_dir().join(format!("glmdsa-spec-{}", std::process::id()));
        let ours = root.join("brain").join("glm");
        let foreign = root.join("vendor").join("llama");
        std::fs::create_dir_all(&ours).unwrap();
        std::fs::create_dir_all(&foreign).unwrap();
        std::fs::write(foreign.join("config.json"), r#"{"architectures":["LlamaForCausalLM"]}"#).unwrap();
        let cfg = GlmConfig::tiny();
        let init = crate::init::init_weights(&cfg, 1);
        let path = ours.join("glm.safetensors");
        Glm::new(cfg.clone(), 1, cfg.block_size, &init).save(path.to_str().unwrap());

        let records = brain_modelstore::inventory::scan(&root);
        let specs: Vec<&dyn ArchSpec> = vec![&SPEC];
        match resolve("glmdsa", &records, &specs, &BTreeMap::new()) {
            Resolution::Resolved(assembly) => assert_eq!(assembly.roles["weights"], path),
            other => panic!("expected Resolved, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
