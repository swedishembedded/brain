// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import an official Ultralytics `yolov8n.pt` checkpoint into brain's native
//! format -- a pure Rust port of `tools/yolo_export/export_yolov8.py`'s
//! already-validated, auditable 1:1 string remap (`ULTRA_INDEX_TO_BRAIN`/
//! `BN_SUFFIX`/`HEAD_BRANCH`), reading tensors via
//! `checkpoint::torchpt::read` instead of shelling out to Python/torch.
//!
//! Ultralytics' `.pt` checkpoints pickle a live `nn.Module` object graph
//! (custom classes like `Conv`/`C2f`/`Detect`), not a plain `state_dict()`
//! dict -- `torchpt`'s generic-class fallback (any unrecognized `NEWOBJ`/
//! `REDUCE` global becomes an empty dict that `BUILD` then populates just
//! like an `OrderedDict`) reads straight through that, but the resulting
//! flattened names carry PyTorch's own pickling bookkeeping segments
//! (`_modules`/`_parameters`/`_buffers`) interleaved with the real path, e.g.
//! `model._modules.model._modules.0._modules.bn._buffers.running_mean`
//! instead of `state_dict()`'s `model.0.bn.running_mean`. [`normalize_name`]
//! undoes exactly that (verified against a real `yolov8n.pt`, confirmed live
//! this session), before [`ultra_to_brain`] does the real remap.

use std::collections::BTreeMap;

use crate::config::YoloConfig;

/// A brain-named tensor ready for `checkpoint::st::save_safetensors`.
type BrainTensor = (String, Vec<usize>, Vec<f32>);

/// Ultralytics module-index -> brain prefix, yolov8n's `DetectionModel`
/// layout: backbone 0..9, neck 12/15/16/18/19/21, head 22 (`Detect`).
/// Indices 10/13 are `Upsample`, 11/14/17/20 are `Concat` -- no weights.
const ULTRA_INDEX_TO_BRAIN: &[(u32, &str)] = &[
    (0, "backbone.0"),
    (1, "backbone.1"),
    (2, "backbone.2"),
    (3, "backbone.3"),
    (4, "backbone.4"),
    (5, "backbone.5"),
    (6, "backbone.6"),
    (7, "backbone.7"),
    (8, "backbone.8"),
    (9, "backbone.9"),
    (12, "neck.0"),
    (15, "neck.1"),
    (16, "neck.2"),
    (18, "neck.3"),
    (19, "neck.4"),
    (21, "neck.5"),
];

/// Ultralytics `Detect` head: `cv2` = box(reg), `cv3` = cls.
const HEAD_BRANCH: &[(&str, &str)] = &[("cv2", "reg"), ("cv3", "cls")];

/// Strip PyTorch's own object-pickling bookkeeping segments
/// (`_modules`/`_parameters`/`_buffers`) plus the outer checkpoint dict's
/// `"model"` wrapper key, recovering exactly the name `m.model.state_dict()`
/// would have produced (the shape `ultra_to_brain` below is written against).
/// `None` for anything not under that wrapper (optimizer state, `train_args`,
/// `date`, ... -- whatever else an Ultralytics `ckpt` dict carries).
fn normalize_name(raw: &str) -> Option<String> {
    let mut parts = raw.split('.');
    if parts.next() != Some("model") {
        return None;
    }
    // `kept[0]` is already `DetectionModel`'s own `.model` (nn.Sequential)
    // attribute name -- the leading "model" consumed above was only the
    // OUTER checkpoint dict's wrapper key, a distinct segment.
    let kept: Vec<&str> = parts.filter(|s| !matches!(*s, "_modules" | "_parameters" | "_buffers")).collect();
    if kept.is_empty() {
        return None;
    }
    Some(kept.join("."))
}

fn remap_conv_tail(tail: &str) -> Option<&'static str> {
    match tail {
        "conv.weight" => Some("conv.weight"),
        "bn.weight" => Some("bn.gamma"),
        "bn.bias" => Some("bn.beta"),
        "bn.running_mean" => Some("bn.run_mean"),
        "bn.running_var" => Some("bn.run_var"),
        // `bn.num_batches_tracked` (BN bookkeeping) and anything else
        // (an unexpected bias, ...) has no brain counterpart.
        _ => None,
    }
}

/// Map one (already state_dict-shaped, i.e. post-[`normalize_name`])
/// Ultralytics tensor name to its brain name, or `None` if it has no brain
/// counterpart and must be dropped. Pure string remap, no arithmetic on
/// values -- a line-by-line port of `export_yolov8.py::ultra_to_brain`.
fn ultra_to_brain(name: &str) -> Option<String> {
    let rest = name.strip_prefix("model.")?;
    let (idx_s, rest) = rest.split_once('.').unwrap_or((rest, ""));
    let idx: u32 = idx_s.parse().ok()?;

    // --- Detect head (model.22): cv2/cv3 . {scale} . {0,1,2} . ... ---------
    if idx == 22 {
        let toks: Vec<&str> = rest.split('.').collect();
        let (branch_key, scale, sub) = (*toks.first()?, *toks.get(1)?, *toks.get(2)?);
        let branch = HEAD_BRANCH.iter().find(|(k, _)| *k == branch_key)?.1;
        let tail = toks[3..].join(".");
        let base = format!("head.{scale}.{branch}.{sub}");
        return match sub {
            "0" | "1" => remap_conv_tail(&tail).map(|bt| format!("{base}.{bt}")),
            // The final layer: Ultralytics' biased Conv2d (weight + bias);
            // brain's head is also biased, so both map 1:1.
            "2" => match tail.as_str() {
                "weight" => Some(format!("{base}.weight")),
                "bias" => Some(format!("{base}.bias")),
                _ => None,
            },
            _ => None,
        };
        // model.22.dfl.conv.weight (the fixed DFL projection) falls through
        // every arm above to None -- brain computes the DFL box expectation
        // analytically, so it has no DFL conv.
    }

    // --- backbone / neck modules --------------------------------------------
    let base = ULTRA_INDEX_TO_BRAIN.iter().find(|(i, _)| *i == idx)?.1;

    // Plain Conv module (model.0/1/3/5/7/16/19): rest IS a conv/bn tail.
    if let Some(bt) = remap_conv_tail(rest) {
        return Some(format!("{base}.{bt}"));
    }

    // Nested (C2f / SPPF): peel sub-prefixes (cv1/cv2/m.<i>) until the
    // remaining tail is a conv/bn tail. brain uses the SAME cv1/cv2/m.<i> names.
    let toks: Vec<&str> = rest.split('.').collect();
    let mut sub_prefix: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match toks[i] {
            t @ ("cv1" | "cv2") => {
                sub_prefix.push(t);
                i += 1;
            }
            "m" => {
                sub_prefix.push(toks[i]);
                sub_prefix.push(*toks.get(i + 1)?);
                i += 2;
            }
            _ => break,
        }
    }
    let conv_tail = toks[i..].join(".");
    let bt = remap_conv_tail(&conv_tail)?;
    if sub_prefix.is_empty() {
        Some(format!("{base}.{bt}"))
    } else {
        Some(format!("{base}.{}.{bt}", sub_prefix.join(".")))
    }
}

/// Read an Ultralytics `yolov8n.pt` checkpoint and remap every tensor to its
/// brain name, validated against [`YoloConfig::yolov8n`]'s exact expected
/// (name, element-count) list -- fails loudly (naming every mismatch/missing/
/// extra tensor) rather than writing a checkpoint that would silently load
/// wrong. Returns brain-named `(name, shape, data)` tensors, ready for
/// `checkpoint::st::save_safetensors`.
pub fn import_yolov8n(path: &str) -> Result<Vec<BrainTensor>, String> {
    let raw = checkpoint::torchpt::read(path)?;

    let mut mapped: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    for t in raw {
        let Some(norm) = normalize_name(&t.name) else { continue };
        let Some(brain_name) = ultra_to_brain(&norm) else { continue };
        if let Some((prev_shape, _)) = mapped.insert(brain_name.clone(), (t.shape, t.data)) {
            return Err(format!("{path}: duplicate mapped tensor {brain_name:?} (previously shape {prev_shape:?})"));
        }
    }

    let expected = YoloConfig::yolov8n().full_param_list();
    let mut problems = Vec::new();
    for (name, expected_numel) in &expected {
        match mapped.get(name) {
            None => problems.push(format!("missing: {name} (expected {expected_numel} elements)")),
            Some((_, data)) if data.len() != *expected_numel => {
                problems.push(format!("{name}: got {} elements, expected {expected_numel}", data.len()))
            }
            Some(_) => {}
        }
    }
    let expected_names: std::collections::BTreeSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    for name in mapped.keys() {
        if !expected_names.contains(name.as_str()) {
            problems.push(format!("unexpected: {name} (mapped from the checkpoint but not in YoloConfig::yolov8n's param list)"));
        }
    }
    if !problems.is_empty() {
        return Err(format!("{path}: {} tensor mismatch(es) against YoloConfig::yolov8n:\n  {}", problems.len(), problems.join("\n  ")));
    }

    Ok(expected.into_iter().map(|(name, _)| { let (shape, data) = mapped.remove(&name).unwrap(); (name, shape, data) }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_name_strips_pickling_bookkeeping_and_the_outer_wrapper() {
        assert_eq!(
            normalize_name("model._modules.model._modules.0._modules.bn._buffers.num_batches_tracked").as_deref(),
            Some("model.0.bn.num_batches_tracked")
        );
        assert_eq!(normalize_name("model._modules.model._modules.22._modules.cv2._modules.0._modules.0._modules.conv._parameters.weight").as_deref(), Some("model.22.cv2.0.0.conv.weight"));
        // Not under the "model" wrapper (e.g. optimizer state, train_args) -> None.
        assert_eq!(normalize_name("epoch"), None);
        assert_eq!(normalize_name("train_args.lr0"), None);
    }

    #[test]
    fn ultra_to_brain_maps_plain_conv_c2f_and_head_units() {
        // Plain Conv module (model.0 = backbone.0).
        assert_eq!(ultra_to_brain("model.0.conv.weight").as_deref(), Some("backbone.0.conv.weight"));
        assert_eq!(ultra_to_brain("model.0.bn.weight").as_deref(), Some("backbone.0.bn.gamma"));
        assert_eq!(ultra_to_brain("model.0.bn.running_var").as_deref(), Some("backbone.0.bn.run_var"));
        // C2f nested (model.2 = backbone.2): cv1 direct, m.<i>.cv<j> nested.
        assert_eq!(ultra_to_brain("model.2.cv1.conv.weight").as_deref(), Some("backbone.2.cv1.conv.weight"));
        assert_eq!(ultra_to_brain("model.2.m.0.cv1.bn.bias").as_deref(), Some("backbone.2.m.0.cv1.bn.beta"));
        // SPPF (model.9 = backbone.9).
        assert_eq!(ultra_to_brain("model.9.cv2.conv.weight").as_deref(), Some("backbone.9.cv2.conv.weight"));
        // Neck (model.12 = neck.0).
        assert_eq!(ultra_to_brain("model.12.cv1.conv.weight").as_deref(), Some("neck.0.cv1.conv.weight"));
        // Detect head (model.22): cv2=reg, cv3=cls; sub "2" is the biased final Conv2d.
        assert_eq!(ultra_to_brain("model.22.cv2.0.0.conv.weight").as_deref(), Some("head.0.reg.0.conv.weight"));
        assert_eq!(ultra_to_brain("model.22.cv3.1.2.weight").as_deref(), Some("head.1.cls.2.weight"));
        assert_eq!(ultra_to_brain("model.22.cv3.2.2.bias").as_deref(), Some("head.2.cls.2.bias"));
    }

    #[test]
    fn ultra_to_brain_drops_batchnorm_bookkeeping_and_the_fixed_dfl_projection() {
        assert_eq!(ultra_to_brain("model.0.bn.num_batches_tracked"), None);
        assert_eq!(ultra_to_brain("model.22.dfl.conv.weight"), None);
        // Upsample/Concat modules (10/11/13/14/17/20) hold no weights at all.
        assert_eq!(ultra_to_brain("model.11.something"), None);
    }

    #[test]
    fn the_remap_covers_yoloconfig_yolov8n_exactly_one_to_one() {
        // Torch-free completeness proof, mirroring export_yolov8.py's
        // --check-map-only: every brain name YoloConfig::yolov8n().
        // full_param_list() expects is reachable from exactly one
        // (synthetic) Ultralytics-shaped name via normalize_name + ultra_to_brain.
        fn u_conv(mod_idx: u32, sub: &str, out: &mut Vec<String>) {
            let p = if sub.is_empty() { format!("model.{mod_idx}") } else { format!("model.{mod_idx}.{sub}") };
            for suffix in ["conv.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"] {
                out.push(format!("{p}.{suffix}"));
            }
        }
        fn u_c2f(mod_idx: u32, n: u32, out: &mut Vec<String>) {
            u_conv(mod_idx, "cv1", out);
            for i in 0..n {
                u_conv(mod_idx, &format!("m.{i}.cv1"), out);
                u_conv(mod_idx, &format!("m.{i}.cv2"), out);
            }
            u_conv(mod_idx, "cv2", out);
        }
        fn u_sppf(mod_idx: u32, out: &mut Vec<String>) {
            u_conv(mod_idx, "cv1", out);
            u_conv(mod_idx, "cv2", out);
        }

        let mut ultra = Vec::new();
        u_conv(0, "", &mut ultra);
        u_conv(1, "", &mut ultra);
        u_c2f(2, 1, &mut ultra);
        u_conv(3, "", &mut ultra);
        u_c2f(4, 2, &mut ultra);
        u_conv(5, "", &mut ultra);
        u_c2f(6, 2, &mut ultra);
        u_conv(7, "", &mut ultra);
        u_c2f(8, 1, &mut ultra);
        u_sppf(9, &mut ultra);
        for m in [12, 15] {
            u_c2f(m, 1, &mut ultra);
        }
        u_conv(16, "", &mut ultra);
        u_c2f(18, 1, &mut ultra);
        u_conv(19, "", &mut ultra);
        u_c2f(21, 1, &mut ultra);
        for s in 0..3 {
            for branch in ["cv2", "cv3"] {
                u_conv(22, &format!("{branch}.{s}.0"), &mut ultra);
                u_conv(22, &format!("{branch}.{s}.1"), &mut ultra);
                ultra.push(format!("model.22.{branch}.{s}.2.weight"));
                ultra.push(format!("model.22.{branch}.{s}.2.bias"));
            }
        }
        ultra.push("model.22.dfl.conv.weight".to_string());

        let mapped: std::collections::BTreeSet<String> = ultra.iter().filter_map(|n| ultra_to_brain(n)).collect();
        let expected: std::collections::BTreeSet<String> = YoloConfig::yolov8n().full_param_list().into_iter().map(|(n, _)| n).collect();
        let missing: Vec<&String> = expected.difference(&mapped).collect();
        let extra: Vec<&String> = mapped.difference(&expected).collect();
        assert!(missing.is_empty(), "brain names with no source: {missing:?}");
        assert!(extra.is_empty(), "mapped names not in YoloConfig::yolov8n: {extra:?}");
    }
}
