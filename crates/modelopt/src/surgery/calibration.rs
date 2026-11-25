// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A calibration on disk: one safetensors file whose tensors are its large
//! vectors (stream second moments, channel statistics, curvature) as f32 and
//! whose config holds the rest.

use std::path::Path;

use checkpoint::weightio::{StWriter, WeightReader};
use model::surgery::Calibration;
use serde_json::Value;

/// Every large vector of `cal` under its tensor name, taken out of it.
fn vectors(cal: &mut Calibration) -> Vec<(String, Vec<f64>)> {
    let mut out = Vec::new();
    for (name, s) in &mut cal.streams {
        out.push((format!("streams/{name}/covariance"), std::mem::take(&mut s.covariance)));
        for (norm, c) in &mut s.shared {
            out.push((format!("streams/{name}/shared/{norm}"), std::mem::take(c)));
        }
    }
    for (name, c) in &mut cal.channels {
        out.push((format!("channels/{name}/sum_sq"), std::mem::take(&mut c.sum_sq)));
    }
    for (name, c) in &mut cal.curvature {
        out.push((format!("curvature/{name}/fisher"), std::mem::take(&mut c.fisher)));
        out.push((format!("curvature/{name}/taylor"), std::mem::take(&mut c.taylor)));
    }
    out
}

/// Write `cal` to `path`.
pub fn save(path: &Path, cal: &Calibration) -> Result<(), String> {
    let mut rest = cal.clone();
    let tensors = vectors(&mut rest);
    let meta = serde_json::to_value(&rest).map_err(|e| e.to_string())?;
    let plan: Vec<(String, Vec<u64>)> = tensors.iter().map(|(n, v)| (n.clone(), vec![v.len() as u64])).collect();
    let file = path.to_str().ok_or_else(|| format!("{}: not UTF-8", path.display()))?;
    let mut w = StWriter::create(file, &plan, &meta, None).map_err(|e| format!("{file}: {e}"))?;
    for (n, v) in &tensors {
        w.write(n, &v.iter().map(|&x| x as f32).collect::<Vec<_>>()).map_err(|e| format!("{file}: {e}"))?;
    }
    w.finish().map_err(|e| format!("{file}: {e}"))
}

/// Read a calibration [`save`] wrote.
pub fn load(path: &Path) -> Result<Calibration, String> {
    let file = path.to_str().ok_or_else(|| format!("{}: not UTF-8", path.display()))?;
    let r = WeightReader::open(file).map_err(|e| format!("{file}: {e}"))?;
    let meta: Value = r.config();
    let mut cal: Calibration = serde_json::from_value(meta).map_err(|e| format!("{file}: not a calibration: {e}"))?;
    let read = |name: String| -> Result<Vec<f64>, String> { Ok(r.tensor(&name).ok_or_else(|| format!("{file}: no {name}"))?.into_iter().map(f64::from).collect()) };
    for (name, s) in &mut cal.streams {
        s.covariance = read(format!("streams/{name}/covariance"))?;
        let norms: Vec<String> = s.shared.keys().cloned().collect();
        for norm in norms {
            s.shared.insert(norm.clone(), read(format!("streams/{name}/shared/{norm}"))?);
        }
    }
    for (name, c) in &mut cal.channels {
        c.sum_sq = read(format!("channels/{name}/sum_sq"))?;
    }
    for (name, c) in &mut cal.curvature {
        c.fisher = read(format!("curvature/{name}/fisher"))?;
        c.taylor = read(format!("curvature/{name}/taylor"))?;
    }
    Ok(cal)
}
