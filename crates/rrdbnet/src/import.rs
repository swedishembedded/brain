// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Load a released Real-ESRGAN checkpoint and check it against the config.
//!
//! The releases are torch archives whose weights sit under `params_ema` (the
//! EMA copy, which is what the reference's `load_network` reads) or `params`.
//! Picking the wrong one is not an error — both are complete, correctly-shaped
//! state dicts — it just runs the non-EMA weights and produces a slightly worse
//! image, so the choice is made explicitly here and reported.

use std::collections::HashMap;

use vae::blocks::Tensors;

use crate::config::RrdbConfig;

/// Which top-level entry of the archive the weights came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `params_ema` — the exponential moving average. What the reference loads.
    Ema,
    /// `params` — the raw training weights. Used only when there is no EMA copy.
    Raw,
}

/// Read `path` and return `(tensors, shapes, source)`.
///
/// Tensor names are stripped of the `params_ema.`/`params.` prefix, so they
/// match [`RrdbConfig::param_list`] and the reference's own module paths.
#[allow(clippy::type_complexity)]
pub fn read(
    path: &str,
) -> Result<(Tensors, HashMap<String, Vec<usize>>, Source), String> {
    let raw = checkpoint::torchpt::read(path).map_err(|e| format!("read {path}: {e}"))?;

    // Prefer the EMA copy, exactly as the reference does.
    let src = if raw.iter().any(|t| t.name.starts_with("params_ema.")) {
        Source::Ema
    } else if raw.iter().any(|t| t.name.starts_with("params.")) {
        Source::Raw
    } else {
        return Err(format!("{path}: no `params_ema.*` or `params.*` tensors"));
    };
    let prefix = if src == Source::Ema { "params_ema." } else { "params." };

    let mut tensors: Tensors = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for t in raw {
        let Some(name) = t.name.strip_prefix(prefix) else { continue };
        shapes.insert(name.to_string(), t.shape.clone());
        tensors.insert(name.to_string(), (t.shape, t.data));
    }
    Ok((tensors, shapes, src))
}

/// Check `tensors` covers exactly what `cfg`'s forward reads, with the right
/// shapes, and return them.
///
/// Both directions matter. A MISSING tensor is caught at first dispatch anyway,
/// but with a message about a buffer rather than a name; an EXTRA tensor means
/// the config was derived wrong (too few blocks, one upsample stage short) and
/// would otherwise run happily on a truncated network.
pub fn validate(tensors: Tensors, cfg: &RrdbConfig) -> Result<Tensors, String> {
    let want = cfg.param_list();
    let mut missing = Vec::new();
    for (name, shape) in &want {
        match tensors.get(name) {
            None => missing.push(name.clone()),
            Some((got, data)) => {
                if got != shape {
                    return Err(format!("{name}: checkpoint has {got:?}, config wants {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if data.len() != n {
                    return Err(format!("{name}: {} floats for shape {shape:?}", data.len()));
                }
            }
        }
    }
    if !missing.is_empty() {
        missing.truncate(8);
        return Err(format!("{} tensor(s) missing, e.g. {missing:?}", missing.len()));
    }

    let wanted: std::collections::HashSet<&str> = want.iter().map(|(n, _)| n.as_str()).collect();
    let extra: Vec<&str> =
        tensors.keys().map(|k| k.as_str()).filter(|k| !wanted.contains(k)).take(8).collect();
    if !extra.is_empty() {
        return Err(format!(
            "checkpoint carries {} tensor(s) the forward never reads, e.g. {extra:?} — \
             the derived config is probably too small",
            tensors.len() - want.len()
        ));
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RrdbConfig {
        RrdbConfig {
            in_channels: 3,
            out_channels: 3,
            num_feat: 8,
            num_grow_ch: 4,
            num_block: 2,
            scale: 4,
        }
    }

    fn full(cfg: &RrdbConfig) -> Tensors {
        cfg.param_list()
            .into_iter()
            .map(|(n, s)| {
                let len = s.iter().product();
                (n, (s, vec![0.0f32; len]))
            })
            .collect()
    }

    #[test]
    fn a_complete_checkpoint_validates() {
        let c = cfg();
        assert!(validate(full(&c), &c).is_ok());
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let c = cfg();
        let mut t = full(&c);
        t.remove("body.1.rdb2.conv3.weight");
        let e = validate(t, &c).unwrap_err();
        assert!(e.contains("body.1.rdb2.conv3.weight"), "{e}");
    }

    #[test]
    fn a_wrong_shape_is_named_with_both_shapes() {
        let c = cfg();
        let mut t = full(&c);
        t.insert("conv_first.weight".into(), (vec![8, 3, 5, 5], vec![0.0; 600]));
        let e = validate(t, &c).unwrap_err();
        assert!(e.contains("conv_first.weight") && e.contains("[8, 3, 5, 5]"), "{e}");
    }

    /// The direction that catches a mis-derived config: a 3-block checkpoint
    /// read as 2 blocks leaves body.2.* unread, and the net would quietly run
    /// one block short.
    #[test]
    fn an_undersized_config_is_caught_by_the_leftovers() {
        let small = cfg();
        let big = RrdbConfig { num_block: 3, ..small.clone() };
        let e = validate(full(&big), &small).unwrap_err();
        assert!(e.contains("never reads"), "{e}");
    }
}
