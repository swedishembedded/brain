// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain plan <model> <action>` - **will this run here, and what will it
//! cost me?**
//!
//! # Why this asks a daemon instead of working it out itself
//!
//! A plan is a statement about a host as it is RIGHT NOW: what is already
//! resident, how full each device is, what would have to be evicted. A fresh
//! CLI process has nothing resident and nothing scheduled, so it would
//! cheerfully answer "plenty of room, nothing to evict" for every model on a
//! box that is in fact full. That answer is not merely incomplete, it is
//! wrong in the exact case a caller asks the question. So this is a client of
//! the running `brain serve --dbus` daemon, which is the process that owns the
//! residency, and it says so plainly when no daemon is there rather than
//! inventing a local answer.
//!
//! Swedish Embedded AB implements capacity and placement tooling for teams
//! running many models on finite hardware. If your team needs expertise in
//! predicting what a workload will cost before committing to it, you can
//! procure our services by sending an email to info@swedishembedded.com.

use serde_json::Value;

const USAGE: &str = "\
usage: brain plan <model> <action> [--params JSON] [--json]
                  [--bus session|system|<address>] [--name <bus name>]

Asks a running `brain serve --dbus` daemon what would happen if this action
ran on it now: memory required, whether it is already resident, which device
it would be placed on, what would be evicted to make room, and every device
it could run on at all. Reserves nothing and loads nothing.
";

/// `brain plan ...`. Returns the process exit code.
pub fn run_plan(argv: &[String]) -> i32 {
    let mut args = crate::args::Args::new(argv);
    if args.take_flag("--help") || args.take_flag("-h") {
        print!("{USAGE}");
        return 0;
    }
    let as_json = args.take_flag("--json");
    let params = args
        .take_str("--params")
        .unwrap_or_else(|| "{}".to_string());
    let bus = args.take_str("--bus");
    let name = args.take_str("--name");
    let (Some(model), Some(action)) = (args.positional(), args.positional()) else {
        eprint!("brain plan: a model and an action are both required\n\n{USAGE}");
        return 2;
    };
    args.finish();

    let opts = brain_dbus::DbusOpts {
        bus: parse_bus(bus.as_deref()),
        name: name.unwrap_or_else(|| "com.swedishembedded.Brain1".to_string()),
    };

    let json = match brain_dbus::client::plan_blocking(&opts, &model, &action, &params) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("brain plan: {e}");
            eprintln!("brain plan: a plan describes a LIVE host, so it needs a running daemon -- start one with `brain serve --dbus`");
            return 1;
        }
    };
    let plan: Value = match serde_json::from_str(&json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("brain plan: the daemon's reply is not valid JSON: {e}");
            return 1;
        }
    };
    if as_json {
        println!("{json}");
    } else {
        print!("{}", render(&plan));
    }
    // A plan that says "this will not run" is a successful ANSWER, not a
    // failed command -- but a script asking "can I run this?" needs to branch
    // on it without parsing prose, so it reaches the shell as an exit code.
    i32::from(!plan["runnable"].as_bool().unwrap_or(false))
}

fn parse_bus(spec: Option<&str>) -> brain_dbus::BusKind {
    match spec {
        None | Some("session") => brain_dbus::BusKind::Session,
        Some("system") => brain_dbus::BusKind::System,
        Some(address) => brain_dbus::BusKind::Address(address.to_string()),
    }
}

/// Renders a `residency::RunPlan`'s JSON as the aligned block a person reads.
///
/// Reads the document field by field rather than through a typed parse: the
/// producing side (`residency::RunPlan::to_json`) is hand-written too, and a
/// hand-written reader beside it is symmetric. A field this build does not
/// know about is simply not printed, which is the same tolerance every other
/// brain wire reader has.
fn render(plan: &Value) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{} {}\n",
        plan["model"].as_str().unwrap_or("?"),
        plan["action"].as_str().unwrap_or("?")
    ));
    let req = &plan["required"];
    out.push_str(&format!(
        "  required      vram {}  ram {}\n",
        bytes(req["vram"].as_u64()),
        bytes(req["ram"].as_u64())
    ));
    match plan["resident_on"].as_object() {
        Some(p) => out.push_str(&format!(
            "  status        already resident on {} ({})\n",
            p.get("device").and_then(Value::as_str).unwrap_or("?"),
            p.get("tier").and_then(Value::as_str).unwrap_or("?")
        )),
        None => out.push_str("  status        not resident\n"),
    }
    if let Some(device) = plan["device"].as_str() {
        out.push_str(&format!("  device        {device}\n"));
    }
    let evict = plan["evict"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    if evict.is_empty() {
        out.push_str("  evict         nothing\n");
    } else {
        for e in evict {
            out.push_str(&format!(
                "  evict         {} from {} (frees {})\n",
                e["model"].as_str().unwrap_or("?"),
                e["device"].as_str().unwrap_or("?"),
                bytes(e["frees"].as_u64())
            ));
        }
    }
    if let Some(transfer) = plan["estimated_transfer"].as_u64() {
        out.push_str(&format!(
            "  transfer      {} into device memory\n",
            bytes(Some(transfer))
        ));
    }
    if let Some(devices) = plan["supported_devices"].as_array() {
        let names: Vec<&str> = devices.iter().filter_map(Value::as_str).collect();
        out.push_str(&format!(
            "  could run on  {}\n",
            if names.is_empty() {
                "nothing on this host".to_string()
            } else {
                names.join(", ")
            }
        ));
    }
    match (plan["runnable"].as_bool(), plan["refusal"].as_str()) {
        (Some(true), _) => out.push_str("  verdict       runnable\n"),
        (_, Some(why)) => out.push_str(&format!("  verdict       will not run: {why}\n")),
        _ => out.push_str("  verdict       will not run\n"),
    }
    out
}

fn bytes(n: Option<u64>) -> String {
    n.map(loader::progress::human_bytes)
        .unwrap_or_else(|| "?".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// **Non-vacuity for every fixture below.** The three cases above are
    /// hand-written JSON, which can only prove the renderer is consistent
    /// with itself. This one renders a plan produced by the REAL
    /// `residency::RunPlan::to_json`, through a real `Executor`, so a field
    /// renamed on the producing side fails here rather than silently
    /// printing `?` on a user's terminal.
    #[test]
    fn the_renderer_reads_the_field_names_a_real_run_plan_actually_publishes() {
        use std::sync::Arc;

        use capability::{
            Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress, Provider,
        };

        struct Noop;
        impl Action for Noop {
            fn spec(&self) -> ActionSpec {
                ActionSpec::new("go", "a weightless action")
            }
            fn run(&self, _inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
                Ok(Outcome::new())
            }
        }
        struct Tiny;
        impl Provider for Tiny {
            fn manifest(&self) -> Manifest {
                Manifest::new("tiny", "a weightless test provider", vec![Noop.spec()])
            }
            fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
                (name == "go").then(|| Arc::new(Noop) as Arc<dyn Action>)
            }
        }

        let model: Arc<dyn residency::ResidentModel> = Arc::new(
            residency::bridge::ProviderResident::stateless(Arc::new(Tiny)),
        );
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(residency::Device::Gpu(0), 8 << 30, 0);
        let exec = residency::Executor::start(vec![model], budgets, residency::Policy::default());
        let plan = exec
            .plan("tiny", "go", Invocation::default())
            .expect("a registered model and action must plan");

        let out = render(&plan.to_json());
        assert!(out.starts_with("tiny go\n"), "{out}");
        assert!(out.contains("status        not resident"), "{out}");
        assert!(out.contains("verdict       runnable"), "{out}");
        assert!(
            !out.contains('?'),
            "every field the renderer prints must have been found: {out}"
        );
    }

    /// The rendering is what a person actually reads, so the shape a real
    /// `RunPlan::to_json` produces has to survive it -- including the two
    /// answers that change a decision: what gets evicted, and whether it runs.
    #[test]
    fn a_plan_that_evicts_to_fit_says_what_it_would_evict_and_that_it_runs() {
        let out = render(&json!({
            "model": "brain/sam2", "action": "segment", "instance_key": "sam2:f16",
            "required": {"vram": 2147483648u64, "ram": 0, "npu": 0, "mapped": 0},
            "resident_on": null,
            "device": "gpu0",
            "evict": [{"model": "brain/qwen35", "instance_key": "q:1", "device": "gpu0", "frees": 9020000000u64}],
            "supported_devices": ["cpu", "gpu0"],
            "estimated_transfer": 2147483648u64,
            "runnable": true,
            "refusal": null
        }));
        assert!(out.contains("not resident"), "{out}");
        assert!(out.contains("device        gpu0"), "{out}");
        assert!(
            out.contains("evict         brain/qwen35 from gpu0"),
            "{out}"
        );
        assert!(out.contains("verdict       runnable"), "{out}");
    }

    /// A model that fits nowhere must say WHY in the same breath, rather than
    /// printing an empty device list and leaving the reader to infer it.
    #[test]
    fn a_plan_that_cannot_run_prints_the_refusal_it_was_given() {
        let out = render(&json!({
            "model": "brain/flux2", "action": "generate", "instance_key": "f:1",
            "required": {"vram": 99000000000u64, "ram": 0, "npu": 0, "mapped": 0},
            "resident_on": null,
            "device": null,
            "evict": [],
            "supported_devices": [],
            "estimated_transfer": 0,
            "runnable": false,
            "refusal": "no device on this host is large enough, even empty"
        }));
        assert!(out.contains("could run on  nothing on this host"), "{out}");
        assert!(
            out.contains("will not run: no device on this host is large enough"),
            "{out}"
        );
    }

    #[test]
    fn an_already_resident_model_says_where_it_is_rather_than_where_it_would_go() {
        let out = render(&json!({
            "model": "brain/sam2", "action": "segment", "instance_key": "sam2:f16",
            "required": {"vram": 1024, "ram": 0, "npu": 0, "mapped": 0},
            "resident_on": {"device": "gpu1", "tier": "hot"},
            "device": "gpu1",
            "evict": [],
            "supported_devices": ["gpu0", "gpu1"],
            "estimated_transfer": 0,
            "runnable": true,
            "refusal": null
        }));
        assert!(out.contains("already resident on gpu1 (hot)"), "{out}");
        assert!(out.contains("evict         nothing"), "{out}");
    }

    #[test]
    fn an_explicit_bus_address_is_passed_through_rather_than_treated_as_a_keyword() {
        assert_eq!(parse_bus(None), brain_dbus::BusKind::Session);
        assert_eq!(parse_bus(Some("system")), brain_dbus::BusKind::System);
        assert_eq!(
            parse_bus(Some("unix:path=/run/brain/bus")),
            brain_dbus::BusKind::Address("unix:path=/run/brain/bus".to_string())
        );
    }
}
