// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wire (JSON) serialization for the forecasting domain types.
//!
//! Kept out of `lib.rs` because it is verbose and mechanical. Bulk numeric
//! fields (context values, quantile matrices, sample trajectories) are carried
//! as base64 LE-f32 via [`crate::bytes`] with an explicit `shape`; small scalars
//! and metadata stay plain JSON so a request is human-readable at the edges.
//!
//! Every `*_from_value` is tolerant of missing optional fields and never
//! panics — a malformed payload yields a defaulted/empty value, and structural
//! validation is the model's job (see `forecast::ForecastModel::validate`).

use crate::bytes;
use forecast::{
    BacktestReport, BacktestRow, BacktestSpec, Block, Capabilities, CovariateSupport, Forecast,
    ForecastSpec, Item, Kind, Panel, Representation, Role, TargetForecast, Variate,
};
use serde_json::{json, Value};

// ---- Panel (input) ---------------------------------------------------------

pub fn panel_to_value(p: &Panel) -> Value {
    json!({
        "freq": p.freq,
        "start": p.start,
        "items": p.items.iter().map(item_to_value).collect::<Vec<_>>(),
    })
}

fn item_to_value(it: &Item) -> Value {
    json!({
        "item_id": it.item_id,
        "variates": it.variates.iter().map(variate_to_value).collect::<Vec<_>>(),
    })
}

fn variate_to_value(v: &Variate) -> Value {
    let mut o = json!({
        "name": v.name,
        "role": v.role.as_str(),
        "kind": v.kind.as_str(),
        "data": bytes::encode_f32(&v.data),
    });
    let m = o.as_object_mut().unwrap();
    if let Some(f) = &v.future {
        m.insert("future".into(), json!(bytes::encode_f32(f)));
    }
    if let Some(obs) = &v.observed {
        m.insert("observed".into(), json!(bytes::encode_f32(obs)));
    }
    if let Some(c) = v.cardinality {
        m.insert("cardinality".into(), json!(c));
    }
    o
}

pub fn panel_from_value(v: &Value) -> Panel {
    Panel {
        freq: v["freq"].as_str().unwrap_or_default().to_string(),
        start: v["start"].as_str().map(|s| s.to_string()),
        items: arr(&v["items"]).iter().map(item_from_value).collect(),
    }
}

fn item_from_value(v: &Value) -> Item {
    Item {
        item_id: v["item_id"].as_str().unwrap_or_default().to_string(),
        variates: arr(&v["variates"]).iter().map(variate_from_value).collect(),
    }
}

fn variate_from_value(v: &Value) -> Variate {
    Variate {
        name: v["name"].as_str().unwrap_or_default().to_string(),
        role: Role::parse(v["role"].as_str().unwrap_or("target")).unwrap_or(Role::Target),
        kind: Kind::parse(v["kind"].as_str().unwrap_or("continuous")).unwrap_or(Kind::Continuous),
        data: f32s(&v["data"]),
        future: v["future"].as_str().map(|s| bytes::decode_f32(s).unwrap_or_default()),
        observed: v["observed"].as_str().map(|s| bytes::decode_f32(s).unwrap_or_default()),
        cardinality: v["cardinality"].as_u64().map(|c| c as u32),
    }
}

// ---- ForecastSpec ----------------------------------------------------------

pub fn spec_to_value(s: &ForecastSpec) -> Value {
    json!({
        "horizon": s.horizon,
        "representations": s.representations.iter().map(|r| r.as_str()).collect::<Vec<_>>(),
        "quantile_levels": s.quantile_levels,
        "num_samples": s.num_samples,
        "seed": s.seed,
    })
}

pub fn spec_from_value(v: &Value) -> ForecastSpec {
    let reps: Vec<Representation> = arr(&v["representations"])
        .iter()
        .filter_map(|r| r.as_str().and_then(Representation::parse))
        .collect();
    ForecastSpec {
        horizon: v["horizon"].as_u64().unwrap_or(1) as usize,
        representations: if reps.is_empty() { vec![Representation::Quantiles] } else { reps },
        quantile_levels: f32s(&v["quantile_levels"]),
        num_samples: v["num_samples"].as_u64().unwrap_or(0) as usize,
        seed: v["seed"].as_u64().unwrap_or(0),
    }
}

// ---- Forecast (output) -----------------------------------------------------

pub fn forecast_to_value(f: &Forecast) -> Value {
    json!({
        "model": f.model,
        "model_version": f.model_version,
        "native_representation": f.native_representation.as_str(),
        "horizon": f.horizon,
        "freq": f.freq,
        "targets": f.targets.iter().map(target_to_value).collect::<Vec<_>>(),
    })
}

fn block_to_value(b: &Block) -> Value {
    json!({
        "shape": b.shape,
        "data": bytes::encode_f32(&b.data),
        "derived": b.derived,
        "method": b.method,
    })
}

fn target_to_value(t: &TargetForecast) -> Value {
    let mut o = json!({ "item_id": t.item_id, "name": t.name });
    let m = o.as_object_mut().unwrap();
    if let Some(b) = &t.quantiles {
        m.insert("quantiles".into(), block_to_value(b));
        m.insert("levels".into(), json!(t.levels));
    }
    if let Some(b) = &t.samples {
        m.insert("samples".into(), block_to_value(b));
    }
    if let Some(b) = &t.mean {
        m.insert("mean".into(), block_to_value(b));
    }
    if let Some(b) = &t.distribution {
        m.insert("distribution".into(), block_to_value(b));
        m.insert("dist_family".into(), json!(t.dist_family));
    }
    if let Some(b) = &t.classes {
        m.insert("classes".into(), block_to_value(b));
        m.insert("class_labels".into(), json!(t.class_labels));
    }
    o
}

pub fn forecast_from_value(v: &Value) -> Forecast {
    Forecast {
        model: v["model"].as_str().unwrap_or_default().to_string(),
        model_version: v["model_version"].as_str().unwrap_or_default().to_string(),
        native_representation: Representation::parse(
            v["native_representation"].as_str().unwrap_or("point"),
        )
        .unwrap_or(Representation::Point),
        horizon: v["horizon"].as_u64().unwrap_or(0) as usize,
        freq: v["freq"].as_str().unwrap_or_default().to_string(),
        targets: arr(&v["targets"]).iter().map(target_from_value).collect(),
    }
}

fn block_from_value(v: &Value) -> Option<Block> {
    if v.is_null() || !v.is_object() {
        return None;
    }
    let shape: Vec<usize> =
        arr(&v["shape"]).iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
    let data = f32s(&v["data"]);
    if shape.iter().product::<usize>() != data.len() {
        return None;
    }
    Some(Block {
        shape,
        data,
        derived: v["derived"].as_bool().unwrap_or(false),
        method: v["method"].as_str().unwrap_or_default().to_string(),
    })
}

fn target_from_value(v: &Value) -> TargetForecast {
    TargetForecast {
        item_id: v["item_id"].as_str().unwrap_or_default().to_string(),
        name: v["name"].as_str().unwrap_or_default().to_string(),
        quantiles: block_from_value(&v["quantiles"]),
        levels: f32s(&v["levels"]),
        samples: block_from_value(&v["samples"]),
        mean: block_from_value(&v["mean"]),
        distribution: block_from_value(&v["distribution"]),
        dist_family: v["dist_family"].as_str().unwrap_or_default().to_string(),
        classes: block_from_value(&v["classes"]),
        class_labels: arr(&v["class_labels"])
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect(),
    }
}

// ---- Capabilities ----------------------------------------------------------

pub fn capabilities_from_value(v: &Value) -> Capabilities {
    let cov = match v["covariates"].as_str().unwrap_or("none") {
        "full" => CovariateSupport::Full,
        "calendar_only" => CovariateSupport::CalendarOnly,
        _ => CovariateSupport::None,
    };
    Capabilities {
        name: v["name"].as_str().unwrap_or_default().to_string(),
        max_context: v["max_context"].as_u64().unwrap_or(0) as usize,
        max_horizon: v["max_horizon"].as_u64().map(|x| x as usize),
        native_representation: Representation::parse(
            v["native_representation"].as_str().unwrap_or("point"),
        )
        .unwrap_or(Representation::Point),
        covariates: cov,
        supports_known_future: v["supports_known_future"].as_bool().unwrap_or(false),
        multivariate: v["multivariate"].as_bool().unwrap_or(false),
        arbitrary_quantile_levels: v["arbitrary_quantile_levels"].as_bool().unwrap_or(false),
        stochastic: v["stochastic"].as_bool().unwrap_or(false),
        requires_variates: arr(&v["requires_variates"])
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect(),
    }
}

// ---- Backtest --------------------------------------------------------------

pub fn backtest_spec_to_value(s: &BacktestSpec) -> Value {
    json!({
        "models": s.models,
        "horizon": s.horizon,
        "origins": s.origins,
        "stride": s.stride,
        "metrics": s.metrics,
        "quantile_levels": s.quantile_levels,
        "seed": s.seed,
    })
}

pub fn backtest_spec_from_value(v: &Value) -> BacktestSpec {
    BacktestSpec {
        models: strs(&v["models"]),
        horizon: v["horizon"].as_u64().unwrap_or(1) as usize,
        origins: v["origins"].as_u64().unwrap_or(30) as usize,
        stride: v["stride"].as_u64().unwrap_or(1) as usize,
        metrics: strs(&v["metrics"]),
        quantile_levels: f32s(&v["quantile_levels"]),
        seed: v["seed"].as_u64().unwrap_or(0),
    }
}

pub fn backtest_report_to_value(r: &BacktestReport) -> Value {
    json!({
        "rows": r.rows.iter().map(|row| json!({
            "model": row.model,
            "metric": row.metric,
            "value": row.value,
            "n_origins": row.n_origins,
        })).collect::<Vec<_>>(),
    })
}

pub fn backtest_report_from_value(v: &Value) -> BacktestReport {
    BacktestReport {
        rows: arr(&v["rows"])
            .iter()
            .map(|row| BacktestRow {
                model: row["model"].as_str().unwrap_or_default().to_string(),
                metric: row["metric"].as_str().unwrap_or_default().to_string(),
                value: row["value"].as_f64().unwrap_or(0.0) as f32,
                n_origins: row["n_origins"].as_u64().unwrap_or(0) as usize,
            })
            .collect(),
    }
}

// ---- helpers ---------------------------------------------------------------

fn arr(v: &Value) -> &[Value] {
    v.as_array().map(|a| a.as_slice()).unwrap_or(&[])
}

fn f32s(v: &Value) -> Vec<f32> {
    // Accept either a base64 f32 string or a plain JSON number array.
    if let Some(s) = v.as_str() {
        return bytes::decode_f32(s).unwrap_or_default();
    }
    arr(v).iter().filter_map(|x| x.as_f64().map(|n| n as f32)).collect()
}

fn strs(v: &Value) -> Vec<String> {
    arr(v).iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect()
}
