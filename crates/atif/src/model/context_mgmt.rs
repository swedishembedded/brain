// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Section VII "Context Management Convention": a typed helper for the
//! `extra.context_management` object producers place on a system step that
//! transforms the agent's context window (compaction, pruning, injection).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `context_management` convention object.
///
/// Both fields are plain `String`s rather than closed enums because the RFC
/// explicitly calls them "Extensible by producers" - a consumer must be able
/// to round-trip a producer-defined `type`/`boundary` value it doesn't
/// recognize rather than fail to parse it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextManagement {
    /// Kind of context transformation: "compaction" | "pruning" | "injection"
    /// or a producer-defined value.
    #[serde(rename = "type")]
    pub kind: String,
    /// How the transformation affects context for subsequent steps:
    /// "replace" | "append" | "truncate" or a producer-defined value.
    pub boundary: String,
}

impl ContextManagement {
    /// Construct a new context-management descriptor.
    pub fn new(kind: impl Into<String>, boundary: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            boundary: boundary.into(),
        }
    }

    /// The JSON key this convention nests under inside `step.extra`.
    pub const EXTRA_KEY: &'static str = "context_management";

    /// Serialize this descriptor to the `serde_json::Value` form expected at
    /// `extra.context_management`.
    pub fn to_extra_value(&self) -> Value {
        serde_json::to_value(self).expect("ContextManagement always serializes")
    }

    /// Extract a `ContextManagement` from a step's `extra` object, if present
    /// and well-formed.
    pub fn from_extra(extra: &Value) -> Option<Self> {
        extra
            .get(Self::EXTRA_KEY)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Insert (or replace) this descriptor at `extra.context_management`,
    /// creating `extra` as a JSON object if it was absent or not an object.
    pub fn insert_into_extra(&self, extra: &mut Option<Value>) {
        let obj = extra.get_or_insert_with(|| Value::Object(serde_json::Map::new()));
        if !obj.is_object() {
            *obj = Value::Object(serde_json::Map::new());
        }
        obj.as_object_mut()
            .expect("just ensured object")
            .insert(Self::EXTRA_KEY.to_string(), self.to_extra_value());
    }
}
