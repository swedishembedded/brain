// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A Qwen3 embedding checkpoint behind the residency executor.
//!
//! A Qwen3-Embedding checkpoint has the same architecture and config as a chat
//! checkpoint of the family; what differs is how it is used. It is run through
//! the decoder's prefill, the hidden state of the last token (an end-of-text
//! token is appended) is taken, and it is L2-normalised. That is the
//! [`qwen3::caps`] `embed` action, and this resident serves exactly that
//! action under the checkpoint's own catalog id, so `/v1/embeddings` resolves
//! it and a chat request for it is refused as an unknown capability.
//!
//! The checkpoint loads on the first request and stays resident inside the
//! provider; the residency budget sees no cost for it until it is loaded, so
//! this suits the small embedding checkpoints it is made for.

use std::sync::Arc;

use capability::Manifest;
use checkpoint::st::ModelCard;
use qwen3::caps::{manifest, QwenProvider};
use residency::bridge::ProviderResident;
use residency::ResidentModel;

/// Whether `id` names an embedding checkpoint rather than a chat one. The
/// checkpoints share config and architecture, so the repository name is the
/// only thing that tells them apart.
#[must_use]
pub fn is_embedding_id(id: &str) -> bool {
    id.rsplit('/')
        .next()
        .is_some_and(|repo| repo.to_ascii_lowercase().contains("embedding"))
}

/// The manifest served for an embedding checkpoint: the `embed` action alone,
/// under the checkpoint's id.
#[must_use]
pub fn embedding_manifest(id: &str) -> Manifest {
    let mut m = manifest();
    m.model = id.to_string();
    m.actions.retain(|a| a.name == "embed");
    m.for_serving()
}

/// The resident for the embedding checkpoint at `weights` (a file or a
/// downloaded directory) with its tokenizer.
#[must_use]
pub fn embedding_resident(
    weights: &str,
    card: &ModelCard,
    tokenizer: Option<&str>,
) -> Arc<dyn ResidentModel> {
    let provider =
        QwenProvider::new().with_defaults(Some(weights.to_string()), tokenizer.map(str::to_string));
    Arc::new(ProviderResident::stateless_with_manifest(
        Arc::new(provider),
        embedding_manifest(&card.id),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_repository_named_embedding_is_an_embedding_checkpoint() {
        assert!(is_embedding_id("Qwen/Qwen3-Embedding-0.6B"));
        assert!(!is_embedding_id("Qwen/Qwen3-0.6B"));
        assert!(!is_embedding_id("Embedding-Labs/chat-model"));
    }

    #[test]
    fn the_served_manifest_is_the_embed_action_alone_under_the_checkpoints_id() {
        let m = embedding_manifest("Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(m.model, "Qwen/Qwen3-Embedding-0.6B");
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["embed"]);
        assert!(m.actions[0].params.iter().any(|p| p.name == "text"));
        assert!(
            m.actions[0].params.iter().all(|p| p.name != "weights"),
            "the checkpoint path is the host's, not the caller's"
        );
    }
}
