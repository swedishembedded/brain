// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which checkpoints in a model store are Qwen3-Omni, in its two forms:
//!
//! * [`SPEC`] - the published Hugging Face directory (its `config.json`
//!   declares `Qwen3OmniMoeForConditionalGeneration`), read as downloaded by
//!   the full Thinker/Talker/Code2Wav model. It holds its own tokenizer.
//! * [`INT8_SPEC`] - the brain-native W8A16 checkpoint `import` writes (card
//!   family `omni`), served by the GPU-sharded int8 Thinker, with the
//!   `tokenizer.json` beside it when there is one.
//!
//! Swedish Embedded AB implements model stores that find what is on a machine
//! without configuration for its clients. If your team needs expertise in
//! model distribution and resolution then you can procure our services by
//! sending an email to info@swedishembedded.com.

use brain_modelstore::family_spec::{FamilySpec, TokenizerRole};

/// The architecture both forms resolve under (`brain serve --pin
/// qwen3omnimoe.ROLE=PATH`).
pub const ARCH: &str = "qwen3omnimoe";
/// The resolver name of the int8 form (`--pin qwen3omnimoe-int8.ROLE=PATH`).
pub const INT8_ARCH: &str = "qwen3omnimoe-int8";
/// The family `crate::import::import_as` stamps on the checkpoint it writes.
pub const INT8_CARD_FAMILY: &str = "omni";

/// The published directory. No brain checkpoint carries this card family: the
/// only one brain writes is the int8 form below.
pub const SPEC: FamilySpec = FamilySpec { arch: ARCH, card_family: ARCH, tokenizer: TokenizerRole::None };

/// The W8A16 checkpoint `import` writes.
pub const INT8_SPEC: FamilySpec = FamilySpec { arch: INT8_ARCH, card_family: INT8_CARD_FAMILY, tokenizer: TokenizerRole::Optional };
