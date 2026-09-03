// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain gguf inspect PATH [--json]` - a local, pre-import look at a real
//! `.gguf` file on disk: its full KV metadata plus a per-tensor name/dtype/
//! shape/size listing.
//!
//! Deliberately NOT `brain models info` (which resolves a model-store
//! reference through `brain_modelstore::Store` and reads tensors via
//! `checkpoint::weightio::WeightReader`): this command's whole point is to
//! look at a file BEFORE it is pulled or imported into the store, so `PATH`
//! is a real filesystem path opened directly with
//! `checkpoint::gguf::MmapGguf::open`, never a model reference.
//!
//! Both render modes read the SAME `MmapGguf`, following `crate::tree`'s own
//! convention (`brain models {list,list-adapters,info}`): the plain tree is
//! for a human at a terminal, `--json` is the same information as data.
//! `MmapGguf::kv()`'s `tokenizer.ggml.tokens` array can carry 100k+ entries
//! (a real vocabulary) - the plain tree elides ANY large KV array to its
//! length plus a few entries rather than printing the whole thing; `--json`
//! is unaffected, since `MmapGguf::config()` already returns the KV
//! losslessly and a consumer parsing JSON is not scrolling a terminal.
//!
//! Swedish Embedded AB implements checkpoint tooling for its clients. If your
//! team needs to inspect a downloaded GGUF checkpoint before committing to
//! importing or serving it, you can procure our services by sending an email
//! to info@swedishembedded.com.

use checkpoint::gguf::{GgufValue, MmapGguf};

use crate::args::Args;
use crate::tree::{self, Node};

const USAGE: &str = "\
usage: brain gguf inspect PATH [--json]

  inspect   render PATH's KV metadata and tensor list as a tree (or, with
            --json, as data). PATH is a real filesystem path to a .gguf
            file, opened directly - NOT a model-store reference; use this
            to look at a checkpoint before `brain pull`/`brain import` ever
            put it in the store.

  --json    emit the same information as JSON instead of the plain tree.
";

/// Dispatch for `brain gguf <subcommand>`. Only `inspect` is implemented;
/// `kv`/`tensors` were mentioned as possible future sub-verbs but are not
/// needed yet - `inspect` alone covers both (metadata tree + tensor tree).
pub fn run_gguf(argv: &[String]) -> i32 {
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return 0;
    }
    match argv.first().map(String::as_str) {
        Some("inspect") => run_inspect(&argv[1..]),
        Some(other) => {
            eprintln!("brain gguf: unknown subcommand {other:?}\n{USAGE}");
            2
        }
        None => {
            eprintln!("{USAGE}");
            2
        }
    }
}

fn run_inspect(args: &[String]) -> i32 {
    let mut a = Args::new(args);
    let json = a.take_flag("--json");
    let Some(path) = a.positional() else {
        eprintln!("{USAGE}");
        return 2;
    };
    a.finish();

    let mg = match MmapGguf::open(&path) {
        Ok(mg) => mg,
        Err(e) => {
            eprintln!("brain gguf inspect: {e}");
            return 1;
        }
    };

    if json {
        let out = serde_json::json!({
            "path": path,
            "kv": mg.config(),
            "tensors": tree::node_to_json(&tensor_nodes(&mg)),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        return 0;
    }

    let header = format!("{path}   {} tensors   {:.2}B params", mg.names().len(), mg.param_count() as f64 / 1e9);
    let mut nodes = vec![Node::leaf(header)];
    nodes.push(Node::branch("metadata".to_string(), kv_nodes(&mg)));
    nodes.push(Node::branch("tensors".to_string(), tensor_nodes(&mg)));
    for line in tree::render_plain(&nodes) {
        println!("{line}");
    }
    0
}

/// One leaf per KV key, `key = value` - see the module doc for why a large
/// array (`tokenizer.ggml.tokens` on a real vocabulary, but the rule is
/// generic, not name-specific) is elided rather than printed in full.
fn kv_nodes(mg: &MmapGguf) -> Vec<Node> {
    mg.kv().iter().map(|(k, v)| Node::leaf(format!("{k} = {}", fmt_value(v)))).collect()
}

/// Render one KV scalar/array for the plain tree. An array over
/// `ELIDE_ABOVE` entries prints its length plus the first few values instead
/// of every entry - the difference between a readable tree and a terminal
/// buried under a 100k-entry vocabulary.
const ELIDE_ABOVE: usize = 8;
const ELIDE_SHOW: usize = 3;

fn fmt_value(v: &GgufValue) -> String {
    match v {
        GgufValue::U8(x) => x.to_string(),
        GgufValue::I8(x) => x.to_string(),
        GgufValue::U16(x) => x.to_string(),
        GgufValue::I16(x) => x.to_string(),
        GgufValue::U32(x) => x.to_string(),
        GgufValue::I32(x) => x.to_string(),
        GgufValue::U64(x) => x.to_string(),
        GgufValue::I64(x) => x.to_string(),
        GgufValue::F32(x) => x.to_string(),
        GgufValue::F64(x) => x.to_string(),
        GgufValue::Bool(x) => x.to_string(),
        GgufValue::String(s) => format!("{s:?}"),
        GgufValue::Array(items) => {
            if items.len() > ELIDE_ABOVE {
                let head: Vec<String> = items.iter().take(ELIDE_SHOW).map(fmt_value).collect();
                format!("[{} items] {}, ...", items.len(), head.join(", "))
            } else {
                format!("[{}]", items.iter().map(fmt_value).collect::<Vec<_>>().join(", "))
            }
        }
    }
}

/// Tensor name/dtype/shape/size, grouped by `.`-separated segment into a
/// tree - the same column convention and grouping shape as `models_cli`'s
/// own `build_tensor_tree`, sourced directly from `MmapGguf` (name/shape/
/// dtype/`raw_tensor_bytes`) instead of `WeightReader`, since this command
/// never opens the store.
fn tensor_nodes(mg: &MmapGguf) -> Vec<Node> {
    #[derive(Default)]
    struct Branch {
        children: std::collections::BTreeMap<String, Branch>,
        leaves: Vec<String>,
    }
    let mut root = Branch::default();
    for name in mg.names() {
        let shape = mg.shape(name).map(|s| format!("{s:?}")).unwrap_or_else(|| "?".to_string());
        let dtype = mg.dtype(name).unwrap_or("?");
        let size = mg.raw_tensor_bytes(name).map(|(bytes, _ty)| crate::pull_cli::human_bytes(bytes.len() as u64)).unwrap_or_else(|| "?".to_string());
        let line = format!("{name}  {dtype}  {shape}  {size}");
        let segs: Vec<&str> = name.split('.').collect();
        let mut cur = &mut root;
        for seg in &segs[..segs.len().saturating_sub(1)] {
            cur = cur.children.entry((*seg).to_string()).or_default();
        }
        cur.leaves.push(line);
    }

    fn to_nodes(name: &str, b: Branch) -> Node {
        let mut children: Vec<Node> = b.children.into_iter().map(|(k, v)| to_nodes(&k, v)).collect();
        children.extend(b.leaves.into_iter().map(Node::leaf));
        Node::branch(name.to_string(), children)
    }
    root.children.into_iter().map(|(k, v)| to_nodes(&k, v)).chain(root.leaves.into_iter().map(Node::leaf)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::gguf_write::{write, TensorOut};

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// A path is required: no positional argument is a usage error (exit 2),
    /// not an attempt to open an empty path.
    #[test]
    fn a_missing_path_is_a_usage_error() {
        assert_eq!(run_gguf(&s(&["inspect"])), 2);
    }

    /// `--json` is recognized and does not get mistaken for the positional
    /// PATH - a nonexistent path still fails to OPEN (exit 1), proving the
    /// flag itself was consumed rather than treated as the file name.
    #[test]
    fn json_flag_is_recognized_and_does_not_consume_the_path() {
        let code = run_gguf(&s(&["inspect", "--json", "/nonexistent/does-not-exist.gguf"]));
        assert_eq!(code, 1);
    }

    /// Same as above with the flag given before the path - order must not
    /// matter, matching `Args`'s own take-and-remove contract.
    #[test]
    fn json_flag_before_the_path_is_also_recognized() {
        let code = run_gguf(&s(&["inspect", "/nonexistent/does-not-exist.gguf", "--json"]));
        assert_eq!(code, 1);
    }

    /// An unrecognized `gguf` sub-verb is a usage error, not a silent no-op.
    #[test]
    fn unknown_subcommand_is_a_usage_error() {
        assert_eq!(run_gguf(&s(&["bogus"])), 2);
    }

    /// A nonexistent path is a clean error (exit 1), never a panic.
    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        assert_eq!(run_gguf(&s(&["inspect", "/nonexistent/does-not-exist.gguf"])), 1);
    }

    fn fixture(dir: &std::path::Path) -> String {
        let path = dir.join("m.gguf").to_string_lossy().into_owned();
        let kv = vec![
            ("general.architecture".to_string(), GgufValue::String("qwen3".to_string())),
            ("general.name".to_string(), GgufValue::String("m22-fixture".to_string())),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array((0..20).map(|i| GgufValue::String(format!("tok{i}"))).collect()),
            ),
        ];
        let tensors = vec![
            TensorOut { name: "tok_embeddings.weight".into(), shape: vec![4, 2], ty: 0, data: (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect() },
            TensorOut { name: "layers.0.attn.q.weight".into(), shape: vec![2, 2], ty: 0, data: (0..4u32).flat_map(|i| (i as f32).to_le_bytes()).collect() },
        ];
        write(&path, &kv, &tensors, 32).unwrap();
        path
    }

    /// End-to-end against a real (synthetic) GGUF: both render modes run
    /// without panicking and produce non-empty output naming a real tensor
    /// and the elided tokenizer array.
    #[test]
    fn inspect_runs_against_a_real_gguf_in_both_modes() {
        let dir = std::env::temp_dir().join(format!("brain-gguf-cli-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = fixture(&dir);

        let mg = MmapGguf::open(&path).unwrap();
        let plain: Vec<String> = tree::render_plain(&[Node::leaf("h".to_string()), Node::branch("metadata", kv_nodes(&mg)), Node::branch("tensors", tensor_nodes(&mg))]);
        assert!(!plain.is_empty());
        let joined = plain.join("\n");
        assert!(joined.contains("layers"), "tensor tree must show a real tensor: {joined}");
        assert!(joined.contains("20 items"), "a large KV array must be elided with its length: {joined}");
        assert!(!joined.contains("tok19"), "an elided array must not print every entry: {joined}");

        assert_eq!(run_gguf(&s(&["inspect", &path])), 0);
        assert_eq!(run_gguf(&s(&["inspect", &path, "--json"])), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
