// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain/demo`: the worked example of the capability interface, always
//! available with no weights. Every transport smoke test exercises it.

use std::sync::Arc;

use capability::{Action, ActionSpec, Blob, Invocation, Manifest, Media, ParamType, Progress, Provider};
use serde_json::json;

/// The demo model's catalog id.
pub const MODEL: &str = "brain/demo";

/// A trivial always-available model so the generic dispatch path (and the
/// tests) work with no weights - and as a worked example of the
/// [`Provider`]/[`Action`] pattern.
pub struct DemoModel;
struct EchoAction;

impl Action for EchoAction {
    fn spec(&self) -> ActionSpec {
        use capability::{BlobSpec, ParamSpec};
        ActionSpec::new("echo", "repeat text, optionally upper/lower-cased")
            .param(ParamSpec::new("text", ParamType::Str, "the text").required())
            .param(ParamSpec::new("times", ParamType::Int, "repeat count").default(json!(1)))
            .param(ParamSpec::new("mode", ParamType::Enum(vec!["as-is".into(), "upper".into(), "lower".into()]), "casing").default(json!("as-is")))
            .output(BlobSpec::new("result", Media::Text, "the echoed text"))
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
        use capability::Outcome;
        let text = inv.get_str("text").unwrap_or_default();
        let n = inv.get_i64("times").unwrap_or(1).max(0) as usize;
        let s = match inv.get_str("mode").as_deref() {
            Some("upper") => text.to_uppercase(),
            Some("lower") => text.to_lowercase(),
            _ => text,
        };
        progress(Progress::step(1, 1, "echoing"));
        let out = s.repeat(n);
        Ok(Outcome::new().set("chars", json!(out.len())).blob("result", Blob::new(Media::Text, out.into_bytes())))
    }
}

impl Provider for DemoModel {
    fn manifest(&self) -> Manifest {
        Manifest::new(MODEL, "a trivial always-available model (no weights) - a worked example of the capability interface", vec![EchoAction.spec()])
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "echo").then(|| Arc::new(EchoAction) as Arc<dyn Action>)
    }
}
