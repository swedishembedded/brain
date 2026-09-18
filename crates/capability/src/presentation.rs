// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Optional, additive descriptive metadata a model may publish about itself:
//! what it is CALLED, who made it, what kind of task it performs, and how a
//! caller should present it.
//!
//! # Why this belongs here and not in the caller
//!
//! Nothing in [`crate::Manifest`] previously said what a model is called.
//! The only names on the wire are the dispatch id (a short slug like
//! `brain/flux2-klein`) and a prose `summary` written to describe an
//! architecture, not to label a button. A caller building a user-facing
//! surface therefore had to INVENT a display name by mechanically cleaning
//! up the slug, which is a heuristic that cannot distinguish `sdxl` (an
//! initialism, upper-case it) from `sam2` (a word plus a number, title-case
//! it) without a hand-maintained exception list that drifts from this
//! repo's own catalogue.
//!
//! The fix is for the model to say. Every field here is optional and every
//! one defaults to absent, so a manifest that declares nothing is unchanged
//! on the wire and a caller keeps whatever fallback it already had. A
//! manifest that DOES declare something replaces a guess with a fact.
//!
//! # Vocabularies are documented strings, not enums
//!
//! [`ActionPresentation::ui_kind`] and friends are `Option<String>` with the
//! accepted values listed on each field and exposed as constants below. An
//! enum would mean every new surface a downstream product invents requires a
//! release of this crate before it can be described - the exact coupling
//! this module exists to remove. A caller that meets a value it does not
//! recognise falls back to its own default, the same as for an absent one.
//!
//! Swedish Embedded AB builds self-describing model-serving interfaces where
//! the runtime, not the UI, owns the truth about what a model is and how it
//! should be presented. If your team needs expertise in capability
//! manifests, model registries or serving contracts, you can procure our
//! services by sending an email to info@swedishembedded.com.

use serde_json::{json, Value};

/// Accepted [`ModelPresentation::maturity`] values.
pub mod maturity {
    /// Validated against a reference and safe to build a product on.
    pub const STABLE: &str = "stable";
    /// Works, but some documented part of the contract is unproven.
    pub const EXPERIMENTAL: &str = "experimental";
    /// Loads and runs, with a known-red correctness gate. Never present this
    /// as ready.
    pub const INCOMPLETE: &str = "incomplete";
}

/// Accepted [`ActionPresentation::interaction_kind`] values.
pub mod interaction {
    /// A person waits for this and looks at the result.
    pub const INTERACTIVE: &str = "interactive";
    /// Run over many inputs, results collected later.
    pub const BATCH: &str = "batch";
    /// Produces or updates weights rather than an inference result.
    pub const TRAINING: &str = "training";
}

/// Accepted [`ActionPresentation::ui_kind`] values: the shape of surface
/// that presents this action well.
pub mod ui {
    pub const CHAT: &str = "chat";
    pub const IMAGE: &str = "image";
    pub const AUDIO: &str = "audio";
    pub const VIDEO: &str = "video";
    pub const TEXT: &str = "text";
    pub const TIMESERIES: &str = "timeseries";
    pub const SCENE_3D: &str = "scene3d";
    pub const TRAINING: &str = "training";
}

/// Accepted [`ActionPresentation::duration_class`] values.
pub mod duration {
    /// Sub-second to a few seconds. A caller may block on it.
    pub const INTERACTIVE: &str = "interactive";
    /// Seconds to minutes. A caller should show progress.
    pub const SLOW: &str = "slow";
    /// Minutes to hours. A caller should treat it as a job, not a request.
    pub const LONG_RUNNING: &str = "long_running";
}

/// What a model is called and what kind of thing it is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelPresentation {
    /// The model's real product name, e.g. `"FLUX.2 Klein"` - never derived
    /// from the dispatch id.
    pub display_name: Option<String>,
    /// Who published the weights, e.g. `"Black Forest Labs"`.
    pub vendor: Option<String>,
    /// The family several variants share, e.g. `"FLUX.2"`, so a caller can
    /// group `klein-4b` and `klein-9b` without parsing their ids.
    pub family: Option<String>,
    /// The released version or variant, e.g. `"Klein 9B"`.
    pub version: Option<String>,
    /// What tasks this model performs, in a caller's own task vocabulary,
    /// e.g. `["image-generation", "image-editing"]`. Free-form on purpose:
    /// this is how a product groups models into a menu, and that menu is
    /// not this crate's to define.
    pub task_tags: Vec<String>,
    /// How far this model's own validation has got - see [`maturity`].
    pub maturity: Option<String>,
    /// Which action a caller should offer first, when the model has several
    /// and one of them is the obvious "Run" - the name of an
    /// [`crate::ActionSpec`] this same manifest declares.
    pub default_action: Option<String>,
    /// The weights' licence, e.g. `"apache-2.0"` or `"flux-2-nc"`. A caller
    /// deploying commercially needs this BEFORE it runs anything, which is
    /// why it travels on the manifest rather than in a README.
    pub license: Option<String>,
}

impl ModelPresentation {
    /// Whether anything at all was declared. An all-absent presentation is
    /// serialized as nothing, so a manifest that opts out is byte-identical
    /// to one from before this existed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == ModelPresentation::default()
    }

    pub fn display_name(mut self, v: &str) -> ModelPresentation {
        self.display_name = Some(v.into());
        self
    }
    pub fn vendor(mut self, v: &str) -> ModelPresentation {
        self.vendor = Some(v.into());
        self
    }
    pub fn family(mut self, v: &str) -> ModelPresentation {
        self.family = Some(v.into());
        self
    }
    pub fn version(mut self, v: &str) -> ModelPresentation {
        self.version = Some(v.into());
        self
    }
    pub fn task_tags(mut self, v: &[&str]) -> ModelPresentation {
        self.task_tags = v.iter().map(|s| (*s).to_string()).collect();
        self
    }
    pub fn maturity(mut self, v: &str) -> ModelPresentation {
        self.maturity = Some(v.into());
        self
    }
    pub fn default_action(mut self, v: &str) -> ModelPresentation {
        self.default_action = Some(v.into());
        self
    }
    pub fn license(mut self, v: &str) -> ModelPresentation {
        self.license = Some(v.into());
        self
    }

    /// This presentation as JSON, or [`Value::Null`] when nothing was
    /// declared.
    #[must_use]
    pub fn to_json(&self) -> Value {
        if self.is_empty() {
            return Value::Null;
        }
        let mut v = json!({});
        insert_opt(&mut v, "display_name", &self.display_name);
        insert_opt(&mut v, "vendor", &self.vendor);
        insert_opt(&mut v, "family", &self.family);
        insert_opt(&mut v, "version", &self.version);
        insert_opt(&mut v, "maturity", &self.maturity);
        insert_opt(&mut v, "default_action", &self.default_action);
        insert_opt(&mut v, "license", &self.license);
        if !self.task_tags.is_empty() {
            v["task_tags"] = json!(self.task_tags);
        }
        v
    }
}

/// How one action should be presented and what it accepts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActionPresentation {
    /// A human label for this action, e.g. `"Generate image"` - never
    /// derived from the action name.
    pub title: Option<String>,
    /// Whether a person waits for this - see [`interaction`].
    pub interaction_kind: Option<String>,
    /// The shape of surface that presents it well - see [`ui`].
    pub ui_kind: Option<String>,
    /// Container formats this action's blob inputs accept, e.g.
    /// `["png", "jpeg", "webp"]`. [`crate::BlobSpec::media`] already says
    /// WHAT KIND of data an input is; this says which encodings of it a
    /// caller may hand over without converting first.
    pub accepted_formats: Vec<String>,
    /// Roughly how long this takes - see [`duration`]. A class, never a
    /// number: the real figure depends on the device, the parameters and
    /// what else is resident, none of which a manifest knows.
    pub duration_class: Option<String>,
}

impl ActionPresentation {
    /// Whether anything at all was declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == ActionPresentation::default()
    }

    pub fn title(mut self, v: &str) -> ActionPresentation {
        self.title = Some(v.into());
        self
    }
    pub fn interaction_kind(mut self, v: &str) -> ActionPresentation {
        self.interaction_kind = Some(v.into());
        self
    }
    pub fn ui_kind(mut self, v: &str) -> ActionPresentation {
        self.ui_kind = Some(v.into());
        self
    }
    pub fn accepted_formats(mut self, v: &[&str]) -> ActionPresentation {
        self.accepted_formats = v.iter().map(|s| (*s).to_string()).collect();
        self
    }
    pub fn duration_class(mut self, v: &str) -> ActionPresentation {
        self.duration_class = Some(v.into());
        self
    }

    /// This presentation as JSON, or [`Value::Null`] when nothing was
    /// declared.
    #[must_use]
    pub fn to_json(&self) -> Value {
        if self.is_empty() {
            return Value::Null;
        }
        let mut v = json!({});
        insert_opt(&mut v, "title", &self.title);
        insert_opt(&mut v, "interaction_kind", &self.interaction_kind);
        insert_opt(&mut v, "ui_kind", &self.ui_kind);
        insert_opt(&mut v, "duration_class", &self.duration_class);
        if !self.accepted_formats.is_empty() {
            v["accepted_formats"] = json!(self.accepted_formats);
        }
        v
    }
}

fn insert_opt(v: &mut Value, key: &str, value: &Option<String>) {
    if let Some(s) = value {
        v[key] = json!(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_undeclared_presentation_serializes_to_nothing() {
        assert_eq!(ModelPresentation::default().to_json(), Value::Null);
        assert_eq!(ActionPresentation::default().to_json(), Value::Null);
    }

    #[test]
    fn only_declared_fields_appear() {
        let v = ModelPresentation::default().display_name("FLUX.2 Klein").to_json();
        assert_eq!(v["display_name"], json!("FLUX.2 Klein"));
        assert!(v.get("vendor").is_none(), "an undeclared field is absent, not null: {v}");
    }

    #[test]
    fn task_tags_survive_in_order() {
        let v = ModelPresentation::default().task_tags(&["image-generation", "image-editing"]).to_json();
        assert_eq!(v["task_tags"], json!(["image-generation", "image-editing"]));
    }
}
