// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The streaming GGUF → brain-native import driver, written once.
//!
//! Every GGUF-sourced model needs the same loop: walk the source tensors,
//! decide what each one becomes, dequantize it one tensor at a time (peak host
//! memory stays one tensor's fp32 expansion, never the checkpoint), write it
//! under brain's own name, and then prove - in **both** directions - that
//! nothing was lost:
//!
//! - every planned output tensor was written exactly **once** (a partial
//!   checkpoint is a bug, never a warning), and
//! - every source tensor was **classified**: either mapped, or dropped with a
//!   stated reason that is counted and printed. An unrecognized name is an
//!   error, not a silent skip - a converter that renames a leaf must break the
//!   import loudly rather than produce a checkpoint missing a projection.
//!
//! Only the *decisions* are per-model ([`Mapped`], the classifier closure and
//! the caller's parameter list); the loop and the bookkeeping are not. The
//! first importer in this tree hand-wrote both, twice (once streaming to a
//! file, once collecting into a map for a truncated load) - those are [`to_st`]
//! and [`to_map`] here, sharing one implementation with [`dry_run`], which runs
//! the identical checks off the header alone for checkpoints too large to
//! expand.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use checkpoint::gguf::MmapGguf;
use checkpoint::st::ModelCard;
use checkpoint::weightio::StWriter;
use serde_json::Value;

/// One GGUF source tensor's disposition at import.
pub enum Mapped {
    /// A plain 1:1 rename; the tensor is copied verbatim.
    Simple(String),
    /// A stacked tensor split into `into.len()` **equal, contiguous** chunks,
    /// chunk `i` written as `into[i]`.
    ///
    /// This is the expert-stack case - GGUF stores a MoE layer's experts as
    /// one `[n_experts, …]` tensor while brain's MoE dispatch reads one 2-D
    /// expert weight per call - generalized to any leading-axis split, since
    /// "contiguous slices along the slowest axis" is the only thing the driver
    /// actually needs to know. Build the expert case with
    /// [`Mapped::expert_stack`]; a fused shared-expert pair is the same shape.
    ///
    /// Only valid when the split axis is the **outermost** one: GGUF's
    /// dequantized output is row-major in torch order, so chunk `i` is exactly
    /// `data[i * chunk .. (i + 1) * chunk]`. Splitting an inner axis would
    /// need strides and is deliberately not offered.
    Split { into: Vec<String> },
    /// Not imported. The `&'static str` is the reason, counted and reported by
    /// [`ImportStats`] - "dropped" must always be a decision on the record.
    Dropped(&'static str),
}

impl Mapped {
    /// The MoE expert stack, under brain's shared expert naming
    /// (`blocks.{layer}.mlp.experts.{e}.{leaf}.weight`).
    pub fn expert_stack(layer: usize, leaf: &str, n_experts: usize) -> Mapped {
        Mapped::Split {
            into: (0..n_experts).map(|e| format!("blocks.{layer}.mlp.experts.{e}.{leaf}.weight")).collect(),
        }
    }
}

/// What an import actually did - the receipt the two-way coverage check hands
/// back, so the caller can print it (or assert on it in a test) rather than
/// trusting that "no error" meant "everything landed".
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ImportStats {
    /// Source tensors seen in the GGUF.
    pub source_tensors: usize,
    /// Output tensors written (≥ `source_tensors` when expert stacks fan out).
    pub written: usize,
    /// Dropped source tensors, counted per stated reason.
    pub dropped: BTreeMap<&'static str, usize>,
}

impl fmt::Display for ImportStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} source tensors -> {} written", self.source_tensors, self.written)?;
        for (reason, n) in &self.dropped {
            write!(f, "; dropped {n} ({reason})")?;
        }
        Ok(())
    }
}

/// Where the driver puts a finished tensor. Private: the shapes brain needs
/// are all provided here, and a fourth would be a new entry point rather than
/// a public trait.
trait Sink {
    /// Whether tensor **data** is needed. A dry run answers `false`, and the
    /// driver then never calls `mg.tensor()` - element counts come from the
    /// header's shapes instead, so the entire mapping of a checkpoint far too
    /// large to expand can still be proven, for the cost of a header parse.
    const NEEDS_DATA: bool;
    fn put(&mut self, name: &str, data: &[f32]) -> Result<(), String>;
}

impl Sink for StWriter {
    const NEEDS_DATA: bool = true;
    fn put(&mut self, name: &str, data: &[f32]) -> Result<(), String> {
        self.write(name, data).map_err(|e| format!("{name}: {e}"))
    }
}

impl Sink for HashMap<String, Vec<f32>> {
    const NEEDS_DATA: bool = true;
    fn put(&mut self, name: &str, data: &[f32]) -> Result<(), String> {
        self.insert(name.to_string(), data.to_vec());
        Ok(())
    }
}

/// The sink of [`dry_run`]: verifies, keeps nothing.
struct DryRun;

impl Sink for DryRun {
    const NEEDS_DATA: bool = false;
    fn put(&mut self, _name: &str, _data: &[f32]) -> Result<(), String> {
        Ok(())
    }
}

/// The loop itself. `expected` is the plan (output name → element count);
/// `label` prefixes every error so a failure names the model, not just the
/// tensor.
fn run<S: Sink>(
    mg: &MmapGguf,
    expected: &HashMap<&str, usize>,
    classify: &dyn Fn(&str) -> Result<Mapped, String>,
    sink: &mut S,
    label: &str,
) -> Result<ImportStats, String> {
    let mut stats = ImportStats { source_tensors: mg.names().len(), ..ImportStats::default() };
    let mut written: HashSet<String> = HashSet::with_capacity(expected.len());

    // Verify one output slot's element count against the plan, refuse a second
    // write to the same slot, then hand it to the sink.
    let put = |sink: &mut S, written: &mut HashSet<String>, name: &str, numel: usize, data: &[f32]| -> Result<(), String> {
        let want = *expected
            .get(name)
            .ok_or_else(|| format!("{label} import: mapped an unexpected tensor {name:?} (not in the parameter list)"))?;
        if numel != want {
            return Err(format!("{label} import: {name}: element count {numel} != expected {want}"));
        }
        if !written.insert(name.to_string()) {
            return Err(format!("{label} import: {name} written twice"));
        }
        if S::NEEDS_DATA {
            sink.put(name, data)?;
        }
        Ok(())
    };

    for name in mg.names() {
        let mapped = classify(name).map_err(|e| format!("{label} import: {e}"))?;
        if let Mapped::Dropped(reason) = mapped {
            *stats.dropped.entry(reason).or_insert(0) += 1;
            continue;
        }
        // One source tensor's fp32 expansion is the driver's peak host cost -
        // and a dry run does not pay even that.
        let data = if S::NEEDS_DATA { read(mg, name, label)? } else { Vec::new() };
        let numel = if S::NEEDS_DATA { data.len() } else { numel_of(mg, name, label)? };

        match mapped {
            Mapped::Simple(brain_name) => {
                put(sink, &mut written, &brain_name, numel, &data)?;
                stats.written += 1;
            }
            Mapped::Split { into } => {
                if into.is_empty() {
                    return Err(format!("{label} import: {name} classified as a split into zero tensors"));
                }
                if numel % into.len() != 0 {
                    return Err(format!(
                        "{label} import: {name}: {numel} elements not divisible into {} contiguous chunks",
                        into.len()
                    ));
                }
                let chunk = numel / into.len();
                for (i, brain_name) in into.iter().enumerate() {
                    let slice = if S::NEEDS_DATA { &data[i * chunk..(i + 1) * chunk] } else { &data[..] };
                    put(sink, &mut written, brain_name, chunk, slice)?;
                    stats.written += 1;
                }
            }
            Mapped::Dropped(_) => unreachable!("dropped tensors are handled before any read"),
        }
    }

    // Two-way coverage, second direction: nothing planned may be missing.
    let missing: Vec<&str> = expected.keys().copied().filter(|n| !written.contains(*n)).collect();
    if !missing.is_empty() {
        let mut sample: Vec<&str> = missing.iter().copied().take(5).collect();
        sample.sort_unstable();
        return Err(format!("{label} import: {} planned tensors never written, e.g. {sample:?}", missing.len()));
    }

    Ok(stats)
}

fn read(mg: &MmapGguf, name: &str, label: &str) -> Result<Vec<f32>, String> {
    mg.tensor(name).ok_or_else(|| format!("{label} import: {name} vanished between names() and tensor()"))?
}

fn numel_of(mg: &MmapGguf, name: &str, label: &str) -> Result<usize, String> {
    let shape = mg.shape(name).ok_or_else(|| format!("{label} import: {name} has no shape"))?;
    Ok(shape.iter().product())
}

/// Import into a brain-native safetensors file, streaming one source tensor at
/// a time.
///
/// `param_list` is the canonical output manifest derived from the model's own
/// config (name → element count); it becomes both the writer's plan and the
/// coverage contract. Shapes are written flat (one dimension, the element
/// count) - brain's parameter stores are flat buffers and every existing
/// importer in this tree plans the same way.
pub fn to_st(
    mg: &MmapGguf,
    param_list: &[(String, usize)],
    classify: &dyn Fn(&str) -> Result<Mapped, String>,
    out_path: &str,
    config: &Value,
    card: Option<&ModelCard>,
    label: &str,
) -> Result<ImportStats, String> {
    let plan: Vec<(String, Vec<u64>)> = param_list.iter().map(|(n, numel)| (n.clone(), vec![*numel as u64])).collect();
    let expected: HashMap<&str, usize> = param_list.iter().map(|(n, numel)| (n.as_str(), *numel)).collect();
    if expected.len() != param_list.len() {
        return Err(format!("{label} import: parameter list contains duplicate names"));
    }

    let mut writer = StWriter::create(out_path, &plan, config, card).map_err(|e| format!("create {out_path}: {e}"))?;
    let stats = run(mg, &expected, classify, &mut writer, label)?;
    // `finish` re-checks the plan itself; `run`'s check has already reported
    // any gap by name, which is the message worth reading.
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!("{label}: {stats} -> {out_path}");
    Ok(stats)
}

/// Import into an in-memory map instead of a file.
///
/// The **truncated-load** path: a classifier that drops every block past a cut
/// (the layer-count in `param_list`'s owning config) never reaches
/// `mg.tensor()` for those names, so neither the dequantization nor the disk
/// cost of the discarded layers is paid. That matters for checkpoints whose
/// fp32 re-encoding does not fit on disk at all.
pub fn to_map(
    mg: &MmapGguf,
    param_list: &[(String, usize)],
    classify: &dyn Fn(&str) -> Result<Mapped, String>,
    label: &str,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let expected: HashMap<&str, usize> = param_list.iter().map(|(n, numel)| (n.as_str(), *numel)).collect();
    if expected.len() != param_list.len() {
        return Err(format!("{label} import: parameter list contains duplicate names"));
    }
    let mut out: HashMap<String, Vec<f32>> = HashMap::with_capacity(param_list.len());
    run(mg, &expected, classify, &mut out, label)?;
    Ok(out)
}

/// Run the identical classification and two-way coverage check **from the
/// header alone** - no tensor bytes are read and nothing is written.
///
/// Element counts come from the declared shapes, so this catches every failure
/// a real import would (unclassified source tensor, mapping onto a name
/// outside the plan, a shape disagreement, a double write, a planned tensor
/// never produced) except a corrupt/unsupported quantization block. It runs in
/// milliseconds regardless of checkpoint size, which is what makes the mapping
/// of a model whose fp32 expansion is tens of gigabytes testable at all.
pub fn dry_run(
    mg: &MmapGguf,
    param_list: &[(String, usize)],
    classify: &dyn Fn(&str) -> Result<Mapped, String>,
    label: &str,
) -> Result<ImportStats, String> {
    let expected: HashMap<&str, usize> = param_list.iter().map(|(n, numel)| (n.as_str(), *numel)).collect();
    if expected.len() != param_list.len() {
        return Err(format!("{label} import: parameter list contains duplicate names"));
    }
    run(mg, &expected, classify, &mut DryRun, label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::gguf::GgufValue;
    use checkpoint::gguf_write::{write, TensorOut};

    /// Two source tensors: a plain one and a 3-way stack, plus one to drop.
    fn synthetic(path: &str) {
        let f32t = |name: &str, shape: Vec<usize>| {
            let numel: usize = shape.iter().product();
            TensorOut {
                name: name.to_string(),
                shape,
                ty: 0,
                data: (0..numel).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            }
        };
        let kvs = vec![("general.architecture".to_string(), GgufValue::String("toy".to_string()))];
        let tensors = vec![
            f32t("norm.weight", vec![4]),
            f32t("stack.weight", vec![3, 2]), // 3 chunks of 2
            f32t("junk.weight", vec![2]),
        ];
        write(path, &kvs, &tensors, 32).unwrap();
    }

    fn classify_ok(name: &str) -> Result<Mapped, String> {
        match name {
            "norm.weight" => Ok(Mapped::Simple("norm.weight".to_string())),
            "stack.weight" => Ok(Mapped::Split {
                into: (0..3).map(|e| format!("experts.{e}.weight")).collect(),
            }),
            "junk.weight" => Ok(Mapped::Dropped("test junk")),
            other => Err(format!("unrecognized tensor {other:?}")),
        }
    }

    fn plan() -> Vec<(String, usize)> {
        let mut p = vec![("norm.weight".to_string(), 4)];
        p.extend((0..3).map(|e| (format!("experts.{e}.weight"), 2)));
        p
    }

    fn temp(tag: &str) -> String {
        std::env::temp_dir().join(format!("gguf-import-{tag}-{}.gguf", std::process::id())).to_string_lossy().into_owned()
    }

    #[test]
    fn split_chunks_are_contiguous_and_in_order() {
        let path = temp("split");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let got = to_map(&mg, &plan(), &classify_ok, "toy").unwrap();

        assert_eq!(got.len(), 4, "one plain tensor + a 3-way fan-out");
        assert_eq!(got["norm.weight"], vec![0.0, 1.0, 2.0, 3.0]);
        // The whole point of Split: chunk e is data[e*2..(e+1)*2], not a
        // transpose or an interleave.
        assert_eq!(got["experts.0.weight"], vec![0.0, 1.0]);
        assert_eq!(got["experts.1.weight"], vec![2.0, 3.0]);
        assert_eq!(got["experts.2.weight"], vec![4.0, 5.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_dry_run_checks_the_same_things_without_reading_data() {
        let path = temp("dry");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();

        let real = to_map(&mg, &plan(), &classify_ok, "toy").unwrap();
        let dry = dry_run(&mg, &plan(), &classify_ok, "toy").unwrap();
        assert_eq!(dry.written, real.len());
        assert_eq!(dry.source_tensors, 3);
        assert_eq!(dry.dropped.get("test junk"), Some(&1));

        // A shape lie must still be caught from the header alone.
        let mut bad = plan();
        bad[0].1 = 5;
        let err = dry_run(&mg, &bad, &classify_ok, "toy").unwrap_err();
        assert!(err.contains("element count 4 != expected 5"), "got {err:?}");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stats_count_drops_by_reason() {
        let path = temp("stats");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let out = std::env::temp_dir().join(format!("gguf-import-stats-{}.st", std::process::id()));
        let out = out.to_string_lossy().into_owned();

        let stats =
            to_st(&mg, &plan(), &classify_ok, &out, &serde_json::json!({}), None, "toy").unwrap();
        assert_eq!(stats.source_tensors, 3);
        assert_eq!(stats.written, 4);
        assert_eq!(stats.dropped.get("test junk"), Some(&1));

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn an_unclassified_source_tensor_is_an_error() {
        let path = temp("unknown");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let strict = |name: &str| -> Result<Mapped, String> {
            if name == "junk.weight" {
                Err("unrecognized tensor \"junk.weight\"".to_string())
            } else {
                classify_ok(name)
            }
        };
        let err = to_map(&mg, &plan(), &strict, "toy").unwrap_err();
        assert!(err.contains("junk.weight"), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_planned_tensor_that_is_never_written_fails_by_name() {
        let path = temp("missing");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let mut p = plan();
        p.push(("absent.weight".to_string(), 1));
        let err = to_map(&mg, &p, &classify_ok, "toy").unwrap_err();
        assert!(err.contains("absent.weight"), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_mapped_name_outside_the_plan_fails() {
        let path = temp("extra");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let stray = |name: &str| -> Result<Mapped, String> {
            if name == "junk.weight" {
                Ok(Mapped::Simple("not_planned.weight".to_string()))
            } else {
                classify_ok(name)
            }
        };
        let err = to_map(&mg, &plan(), &stray, "toy").unwrap_err();
        assert!(err.contains("not_planned.weight"), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_shape_disagreement_between_plan_and_source_fails() {
        let path = temp("shape");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let mut p = plan();
        p[0].1 = 5; // norm.weight really has 4 elements
        let err = to_map(&mg, &p, &classify_ok, "toy").unwrap_err();
        assert!(err.contains("element count 4 != expected 5"), "got {err:?}");
        std::fs::remove_file(&path).ok();
    }
}
