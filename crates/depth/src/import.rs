// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Load a released ZipDepth `.pth` into a brain init map.
//!
//! This is a 1:1 NAME COPY, and it is allowed to be that simple only because
//! [`crate::config::ZipConfig::param_list`] was built from the reference's own
//! `state_dict` keys and is checked against the real file
//! (`tests/p1_param_layout.rs`, `strict=True`, zero missing/extra/mismatched). So
//! there is no translation table to maintain and drift — the importer's job is to
//! read the file, drop the int64 counters, verify the shapes, and hand back the
//! `HashMap<String, Vec<f32>>` that `ParamStore::new` already consumes.
//!
//! A silent-drop importer ships a half-loaded model; a hard-erroring one cannot
//! import at all. This one is strict by default and says exactly what diverged:
//! every expected tensor must be present with the right element count, and every
//! file tensor must be either expected or a known-skippable counter — anything
//! else is an error naming the tensor.

use std::collections::HashMap;

use crate::config::ZipConfig;

/// int64 BatchNorm step counters (`num_batches_tracked`): bookkeeping no path here
/// reads, and BN is folded into the conv for export. `param_list` omits them by
/// design, so the importer skips them by name rather than treating them as extra.
fn is_counter(name: &str) -> bool {
    name.ends_with("num_batches_tracked")
}

/// Read `path` and return the init map for a model of shape `cfg`.
///
/// Errors — never silently — if the file and the config disagree: a missing
/// expected tensor, an unexpected file tensor, or a shape/element mismatch. The
/// message names the offending tensor so a wrong `cfg` (e.g. the unfold vs blend
/// variant) is a one-line diagnosis, not a mystery depth map.
pub fn load(path: &str, cfg: &ZipConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let tensors = checkpoint::torchpt::read(path)?;
    let expected: HashMap<String, usize> =
        cfg.param_list().into_iter().map(|(n, s)| (n, s.iter().product())).collect();

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    let mut unexpected = Vec::new();
    for t in tensors {
        if is_counter(&t.name) {
            continue;
        }
        match expected.get(&t.name) {
            None => unexpected.push(t.name),
            Some(&numel) => {
                if t.data.len() != numel {
                    return Err(format!(
                        "tensor `{}`: file has {} elements, the model expects {} \
                         (shape {:?}) — wrong variant or config?",
                        t.name,
                        t.data.len(),
                        numel,
                        t.shape
                    ));
                }
                out.insert(t.name, t.data);
            }
        }
    }

    if !unexpected.is_empty() {
        unexpected.sort();
        return Err(format!(
            "the checkpoint has {} tensor(s) the model does not declare — likely the \
             wrong variant (unfold vs blend upsampler). First few: {:?}",
            unexpected.len(),
            &unexpected[..unexpected.len().min(5)]
        ));
    }

    let missing: Vec<&String> = expected.keys().filter(|k| !out.contains_key(*k)).collect();
    if !missing.is_empty() {
        let mut m: Vec<&String> = missing;
        m.sort();
        return Err(format!(
            "the checkpoint is missing {} tensor(s) the model needs. First few: {:?}",
            m.len(),
            &m[..m.len().min(5)]
        ));
    }

    Ok(out)
}

/// Load a checkpoint into a fresh [`ParamStore`] on `gpu`.
///
/// Convenience over [`load`] + [`paramstore::ParamStore::new`]: the param list and
/// the init map are both derived from the same `cfg`, so they cannot disagree on
/// which tensors exist.
pub fn load_into(
    gpu: &gpu_core::Gpu,
    path: &str,
    cfg: &ZipConfig,
) -> Result<paramstore::ParamStore, String> {
    let init = load(path, cfg)?;
    let params: Vec<(String, usize)> =
        cfg.param_list().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
    Ok(paramstore::ParamStore::new(gpu, params, &init))
}

/// The tensor names in a checkpoint (for variant auto-detection), without loading
/// the data into a model. Skips the int64 counters.
pub fn tensor_names(path: &str) -> Result<Vec<String>, String> {
    let tensors = checkpoint::torchpt::read(path)?;
    Ok(tensors.into_iter().map(|t| t.name).filter(|n| !is_counter(n)).collect())
}

/// Pick the [`ZipConfig`] by inspecting the checkpoint's own tensor names, so a
/// caller never has to match a `--variant` flag to the file: `where_conv.*` →
/// blend (NPU) upsampler, otherwise the unfold (base) variant. A wrong variant
/// was the classic footgun ("11 tensors the model does not declare").
pub fn cfg_for_checkpoint(path: &str) -> Result<ZipConfig, String> {
    let names = tensor_names(path)?;
    let blend = names.iter().any(|n| n.contains("where_conv"));
    Ok(ZipConfig { upsample_unfold: !blend, ..ZipConfig::base() })
}
