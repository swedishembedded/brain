// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibrated KV-cache clip ceilings for the scaled KV tiers (int8, fp8) -
//! architecture-agnostic, like the rest of
//! this crate (`paged`'s `BlockAllocator`/`BlockTable`, `serve`'s
//! `PagedDecoder` trait + `Scheduler<D>`, `actstats`'s percentile reservoir).
//! [`KvCalib`] itself names no model: it is a pure `(layer, kv_head, K|V) ->
//! f32` table plus JSON I/O, built from an `actstats::Collector` any paged-KV
//! model can populate its own way. `crates/qwen` is the first (and today
//! only) consumer — see `crates/qwen3/src/serve.rs`'s `Engine::calibrate_kv`/
//! `set_kv_calib` for how a concrete `PagedDecoder` implementation wires this
//! in — but nothing here is Qwen-specific; a second paged-attention model
//! adopting the same INT8-KV-with-calibration pattern reuses this file and
//! its `kv_calib.json` format as-is.
//!
//! A per-`(layer, kv_head, K|V)` clip ceiling, derived offline (`brain qwen
//! calib` today) from a chosen percentile of a representative prompt set's
//! K / V magnitude distribution (`actstats::Collector`).
//!
//! The mechanism that makes this a win is percentile CLIPPING, not
//! staticness: a static per-(layer,head) scale sized to the worst token in
//! the calibration set would waste resolution on a typical token relative to
//! an online per-token absmax. [`KvCalib`] is instead used as a CEILING on
//! that online scale (`paged_kv_append_i8_clipped_batched.wgsl`:
//! `scale = min(this_token's_absmax, clip[head]) / 127`) — keeping the
//! per-token adaptivity while denying one rare outlier token the ability to
//! crush every other token's resolution in that head. A `KvCalib` whose
//! every entry is [`f32::MAX`] (the uncalibrated sentinel, see
//! [`KvCalib::disabled`]) degrades the clipped kernel to bit-identical
//! behaviour vs the plain online-absmax one, which is what makes the A/B
//! (and the "no calibration file present" default) clean.
//!
//! This was built against a real-checkpoint measurement, on the one
//! consumer that exists today.
//!
//! # Calibrating any architecture
//!
//! A model joins by implementing [`KvCalibrationSource`]: it runs its own prefill
//! over a prompt set and reports each KV layer's K and V rows to a [`KvObserver`]
//! (a plane's dequantized rows go through [`observe_rows`], which splits them per
//! kv-head). [`calibrate`] turns those observations into a [`KvCalib`] at a
//! percentile; [`KvCalib::apply_to_layer`] installs one layer's ceilings on a
//! [`KvLayer`](crate::kv_tier::KvLayer) of any scaled tier. Layers without KV (a
//! hybrid decoder's recurrent layers) are simply never observed and keep the
//! uncalibrated sentinel, so a table is indexed by the model's own layer index.
//! The ceiling applies unchanged to int8 (`scale = min(absmax, clip) / 127`) and
//! fp8 E4M3 (`/ 448`): both are per-row absmax scalings, so the same outlier
//! argument holds.
//!
//! Swedish Embedded AB implements solutions for serving long contexts in the
//! least GPU memory that keeps accuracy. If your team needs expertise in KV
//! cache quantization and calibration you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::Path;

/// Which half of a layer's KV cache a row belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvSide {
    K,
    V,
}

/// The collector stream a `(layer, side, kv_head)` row set is accumulated under.
/// The one place the naming convention lives: [`KvCalib::from_collector`] reads
/// exactly these names.
pub fn stream_name(layer: usize, side: KvSide, head: usize) -> String {
    let side = match side {
        KvSide::K => 'k',
        KvSide::V => 'v',
    };
    format!("layer{layer:02}.{side}.head{head}")
}

/// Receives the K and V rows a model computes while it prefills calibration prompts.
pub trait KvObserver {
    /// `values` are the elements of one kv-head's rows (any number of tokens
    /// concatenated), K post-RoPE exactly as the cache would store them.
    fn observe(&mut self, layer: usize, side: KvSide, head: usize, values: &[f32]);
}

impl KvObserver for crate::actstats::Collector {
    fn observe(&mut self, layer: usize, side: KvSide, head: usize, values: &[f32]) {
        crate::actstats::Collector::observe(self, &stream_name(layer, side, head), values);
    }
}

/// Feed `rows` (`[tokens, n_kv * head_dim]`, f32) of one layer's plane to `obs`, one
/// call per (token, kv-head) so every head's stream sees its own rows.
pub fn observe_rows(obs: &mut dyn KvObserver, layer: usize, side: KvSide, rows: &[f32], n_kv: usize, head_dim: usize) {
    let stride = n_kv * head_dim;
    assert!(stride > 0 && rows.len().is_multiple_of(stride), "observe_rows: {} values are not whole rows of {n_kv} x {head_dim}", rows.len());
    for row in rows.chunks_exact(stride) {
        for (head, values) in row.chunks_exact(head_dim).enumerate() {
            obs.observe(layer, side, head, values);
        }
    }
}

/// What a model offers to be calibrated: nothing about its attention beyond running prompts.
pub trait KvCalibrationSource {
    /// `(layers, kv heads, head_dim)`: `layers` counts ALL decoder layers, so
    /// tables are indexed by the model's own layer index.
    fn kv_geometry(&self) -> (usize, usize, usize);

    /// Run every prompt through the model's real prefill - in a representation
    /// that keeps the pre-quantization values (an f32 cache) - and report each
    /// KV layer's rows to `obs`.
    fn observe_prompts(&mut self, prompts: &[Vec<u32>], obs: &mut dyn KvObserver) -> Result<(), String>;
}

/// Calibrate `model_id`'s KV clip ceilings at percentile `q` from `prompts`.
/// Refuses an empty prompt set: a table built from nothing would be all
/// sentinels, which is "uncalibrated" and must not be saved as a calibration.
pub fn calibrate(source: &mut dyn KvCalibrationSource, model_id: &str, prompts: &[Vec<u32>], q: f32) -> Result<KvCalib, String> {
    if prompts.is_empty() {
        return Err("calibrate: no prompts".into());
    }
    let (n_layers, n_kv, head_dim) = source.kv_geometry();
    let mut collector = crate::actstats::Collector::new();
    source.observe_prompts(prompts, &mut collector)?;
    Ok(KvCalib::from_collector(model_id, n_layers, n_kv, head_dim, q, &collector))
}

/// One checkpoint's calibrated KV clip ceilings, `[layer][kv_head]` for K and
/// V independently (K is quantized post-RoPE, a different distribution than
/// raw V, so they are calibrated and stored separately).
#[derive(Clone, Debug)]
pub struct KvCalib {
    pub model_id: String,
    pub n_layers: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// The percentile this table was built at (e.g. `0.999`). `1.0` marks
    /// [`KvCalib::disabled`] (no real calibration — every ceiling is
    /// [`f32::MAX`]).
    pub percentile: f32,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvCalib {
    /// The uncalibrated sentinel: every ceiling is [`f32::MAX`], so
    /// `min(absmax, clip)` never binds and the clipped kernel matches the
    /// plain online-absmax one bit-for-bit.
    pub fn disabled(n_layers: usize, n_kv: usize, head_dim: usize) -> KvCalib {
        KvCalib {
            model_id: String::new(),
            n_layers,
            n_kv,
            head_dim,
            percentile: 1.0,
            k: vec![vec![f32::MAX; n_kv]; n_layers],
            v: vec![vec![f32::MAX; n_kv]; n_layers],
        }
    }

    /// The per-kv-head ceilings of `layer`'s K or V plane.
    pub fn clip(&self, layer: usize, side: KvSide) -> &[f32] {
        match side {
            KvSide::K => &self.k[layer],
            KvSide::V => &self.v[layer],
        }
    }

    /// Install `layer`'s ceilings on a layer's planes. A plane of an unscaled tier
    /// (f32, bf16) has no scale to cap, so it is left alone and `false` is returned.
    pub fn apply_to_layer(&self, gpu: &gpu_core::Gpu, layer: usize, planes: &mut crate::kv_tier::KvLayer) -> bool {
        let k = planes.k.set_clip(gpu, self.clip(layer, KvSide::K));
        let v = planes.v.set_clip(gpu, self.clip(layer, KvSide::V));
        k && v
    }

    pub fn is_disabled(&self) -> bool {
        self.k.iter().flatten().all(|&x| x == f32::MAX) && self.v.iter().flatten().all(|&x| x == f32::MAX)
    }

    /// Build from a [`crate::actstats::Collector`] populated by a model's own
    /// calibration pass (`crates/qwen3/src/serve.rs`'s `Engine::calibrate_kv`
    /// is the one that exists today; stream names `layer{L:02}.{k|v}.head{H}`
    /// is that caller's convention, not one this function enforces), at
    /// percentile `q` (e.g. `0.999`). A `(layer, head)` the collector never
    /// observed — shouldn't happen with a non-empty, representative prompt
    /// set, but defensive rather than panicking — falls back to `f32::MAX`
    /// (no clip for that stream specifically, not a failure of the whole table).
    pub fn from_collector(model_id: &str, n_layers: usize, n_kv: usize, head_dim: usize, q: f32, collector: &crate::actstats::Collector) -> KvCalib {
        assert!((0.0..=1.0).contains(&q), "percentile must be in [0,1], got {q}");
        let mut k = vec![vec![f32::MAX; n_kv]; n_layers];
        let mut v = vec![vec![f32::MAX; n_kv]; n_layers];
        for l in 0..n_layers {
            for h in 0..n_kv {
                if let Some(val) = collector.quantile(&format!("layer{l:02}.k.head{h}"), q) {
                    k[l][h] = val;
                }
                if let Some(val) = collector.quantile(&format!("layer{l:02}.v.head{h}"), q) {
                    v[l][h] = val;
                }
            }
        }
        KvCalib { model_id: model_id.to_string(), n_layers, n_kv, head_dim, percentile: q, k, v }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "model_id": self.model_id,
            "n_layers": self.n_layers,
            "n_kv": self.n_kv,
            "head_dim": self.head_dim,
            "percentile": self.percentile,
            "k": self.k,
            "v": self.v,
        })
    }

    /// Parse from the JSON produced by [`KvCalib::to_json`]. Rejects a
    /// malformed or shape-inconsistent document with a clear message rather
    /// than silently defaulting fields — a calibration table that quietly
    /// loaded wrong would misclip in a way nothing downstream can detect.
    pub fn from_json(v: &serde_json::Value) -> Result<KvCalib, String> {
        let model_id = v.get("model_id").and_then(|x| x.as_str()).unwrap_or_default().to_string();
        let n_layers = v.get("n_layers").and_then(|x| x.as_u64()).ok_or("missing/invalid n_layers")? as usize;
        let n_kv = v.get("n_kv").and_then(|x| x.as_u64()).ok_or("missing/invalid n_kv")? as usize;
        let head_dim = v.get("head_dim").and_then(|x| x.as_u64()).ok_or("missing/invalid head_dim")? as usize;
        let percentile = v.get("percentile").and_then(|x| x.as_f64()).ok_or("missing/invalid percentile")? as f32;
        let parse_table = |field: &str| -> Result<Vec<Vec<f32>>, String> {
            let arr = v.get(field).and_then(|x| x.as_array()).ok_or_else(|| format!("missing/invalid {field}"))?;
            if arr.len() != n_layers {
                return Err(format!("{field}: {} layers, expected n_layers={n_layers}", arr.len()));
            }
            arr.iter()
                .enumerate()
                .map(|(l, row)| {
                    let row = row.as_array().ok_or_else(|| format!("{field}[{l}]: not an array"))?;
                    if row.len() != n_kv {
                        return Err(format!("{field}[{l}]: {} heads, expected n_kv={n_kv}", row.len()));
                    }
                    row.iter().map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| format!("{field}[{l}]: non-numeric entry"))).collect()
                })
                .collect()
        };
        let k = parse_table("k")?;
        let v_table = parse_table("v")?;
        Ok(KvCalib { model_id, n_layers, n_kv, head_dim, percentile, k, v: v_table })
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(&self.to_json()).expect("KvCalib JSON is always serializable");
        std::fs::write(path, text)
    }

    pub fn load(path: &Path) -> Result<KvCalib, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&v)
    }

    /// Discover `kv_calib.json` next to a checkpoint's directory, mirroring
    /// `data::chat_template::ChatTemplate::from_model_dir`. `None` — not an
    /// error — when the file is absent, malformed, or shaped for a
    /// different model: "no calibration" means "serve uncalibrated", not
    /// "fail to load". Every rejection reason is still printed to stderr so
    /// a genuinely stale/wrong file doesn't fail silently.
    pub fn from_model_dir(dir: &Path, n_layers: usize, n_kv: usize, head_dim: usize) -> Option<KvCalib> {
        let path = dir.join("kv_calib.json");
        if !path.exists() {
            return None;
        }
        match Self::load(&path) {
            Ok(c) if c.n_layers == n_layers && c.n_kv == n_kv && c.head_dim == head_dim => Some(c),
            Ok(c) => {
                eprintln!(
                    "{}: shape mismatch (L{} kv{} hd{}, model is L{n_layers} kv{n_kv} hd{head_dim}); serving uncalibrated",
                    path.display(),
                    c.n_layers,
                    c.n_kv,
                    c.head_dim
                );
                None
            }
            Err(e) => {
                eprintln!("{e}; serving uncalibrated");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_collector() -> crate::actstats::Collector {
        let c = crate::actstats::Collector::new();
        // layer00: K tight, V has a real tail; layer01: nothing observed for K.
        c.observe("layer00.k.head0", &(1..=6000).map(|v| v as f32).collect::<Vec<_>>());
        let mut v_tail: Vec<f32> = (1..=6000).map(|v| v as f32).collect();
        v_tail.push(100_000.0);
        c.observe("layer00.v.head0", &v_tail);
        c
    }

    /// A stand-in decoder: `kv_layers` of its `n_layers` keep KV; layer l's K rows
    /// are `l + 1` times a ramp, V rows carry one huge outlier token.
    struct FakeModel {
        n_layers: usize,
        kv_layers: Vec<usize>,
    }

    impl KvCalibrationSource for FakeModel {
        fn kv_geometry(&self) -> (usize, usize, usize) {
            (self.n_layers, 2, 4)
        }

        fn observe_prompts(&mut self, prompts: &[Vec<u32>], obs: &mut dyn KvObserver) -> Result<(), String> {
            for &l in &self.kv_layers {
                for prompt in prompts {
                    let k: Vec<f32> = (0..prompt.len() * 8).map(|i| (l + 1) as f32 * (1 + i % 1000) as f32).collect();
                    let mut v = vec![1.0f32; prompt.len() * 8];
                    v[0] = 1e6;
                    observe_rows(obs, l, KvSide::K, &k, 2, 4);
                    observe_rows(obs, l, KvSide::V, &v, 2, 4);
                }
            }
            Ok(())
        }
    }

    #[test]
    fn any_model_is_calibrated_through_the_source_trait_and_non_kv_layers_stay_uncalibrated() {
        let mut model = FakeModel { n_layers: 4, kv_layers: vec![1, 3] };
        let prompts = vec![vec![0u32; 600]; 2];
        let calib = calibrate(&mut model, "fake", &prompts, 0.99).unwrap();
        assert_eq!((calib.n_layers, calib.n_kv, calib.head_dim), (4, 2, 4));
        assert_eq!(calib.clip(0, KvSide::K), &[f32::MAX; 2], "layer 0 keeps no KV");
        assert_eq!(calib.clip(2, KvSide::V), &[f32::MAX; 2]);
        for l in [1usize, 3] {
            assert!(calib.clip(l, KvSide::K).iter().all(|&c| c > 0.0 && c <= (l + 1) as f32 * 1000.0), "layer {l} K ceiling is bounded by its own range");
            assert!(calib.clip(l, KvSide::V).iter().all(|&c| c < 1e5), "layer {l} V ceiling must ignore the 1e6 outlier");
        }
        assert!(calib.clip(3, KvSide::K)[0] > calib.clip(1, KvSide::K)[0], "ceilings follow each layer's own scale");
    }

    #[test]
    fn calibrating_from_no_prompts_is_refused() {
        let mut model = FakeModel { n_layers: 1, kv_layers: vec![0] };
        assert!(calibrate(&mut model, "fake", &[], 0.99).unwrap_err().contains("no prompts"));
    }

    #[test]
    fn observe_rows_gives_each_head_its_own_stream() {
        let mut obs = crate::actstats::Collector::new();
        // two tokens x (2 heads x 2 dims): head 0 holds 1,2 / 5,6; head 1 holds 3,4 / 7,8.
        observe_rows(&mut obs, 5, KvSide::K, &[1., 2., 3., 4., 5., 6., 7., 8.], 2, 2);
        assert_eq!(obs.quantile(&stream_name(5, KvSide::K, 0), 1.0), Some(6.0));
        assert_eq!(obs.quantile(&stream_name(5, KvSide::K, 1), 1.0), Some(8.0));
        assert_eq!(obs.quantile(&stream_name(5, KvSide::V, 0), 1.0), None);
    }

    #[test]
    fn disabled_is_the_bit_identical_sentinel() {
        let c = KvCalib::disabled(2, 3, 8);
        assert!(c.is_disabled());
        assert!(c.k.iter().flatten().all(|&x| x == f32::MAX));
        assert!(c.v.iter().flatten().all(|&x| x == f32::MAX));
    }

    #[test]
    fn from_collector_clips_below_absmax_for_a_tailed_stream_and_leaves_missing_streams_at_max() {
        let collector = tiny_collector();
        let calib = KvCalib::from_collector("test-model", 2, 1, 8, 0.999, &collector);
        assert!(!calib.is_disabled());
        // K (tight distribution): clip should sit near the bulk's own max.
        assert!(calib.k[0][0] < 6000.0 * 1.01 && calib.k[0][0] > 5000.0, "k[0][0]={}", calib.k[0][0]);
        // V (tailed): the p99.9 clip must be well below the 100000 outlier.
        assert!(calib.v[0][0] < 10_000.0, "v[0][0]={} should clip well below the 1e5 outlier", calib.v[0][0]);
        // layer01 was never observed -> falls back to the uncalibrated sentinel.
        assert_eq!(calib.k[1][0], f32::MAX);
        assert_eq!(calib.v[1][0], f32::MAX);
    }

    #[test]
    fn json_roundtrips_exactly() {
        let collector = tiny_collector();
        let calib = KvCalib::from_collector("roundtrip-model", 2, 1, 8, 0.999, &collector);
        let parsed = KvCalib::from_json(&calib.to_json()).unwrap();
        assert_eq!(parsed.model_id, calib.model_id);
        assert_eq!(parsed.n_layers, calib.n_layers);
        assert_eq!(parsed.n_kv, calib.n_kv);
        assert_eq!(parsed.head_dim, calib.head_dim);
        assert_eq!(parsed.percentile, calib.percentile);
        assert_eq!(parsed.k, calib.k);
        assert_eq!(parsed.v, calib.v);
    }

    #[test]
    fn from_json_rejects_a_shape_mismatch_instead_of_silently_truncating() {
        let mut doc = KvCalib::disabled(2, 3, 8).to_json();
        // Corrupt one row to have the wrong number of heads.
        doc["k"][0] = serde_json::json!([1.0, 2.0]); // 2 entries, but n_kv=3
        let err = KvCalib::from_json(&doc).unwrap_err();
        assert!(err.contains("k[0]"), "error should name the bad field: {err}");
    }

    #[test]
    fn save_and_load_roundtrip_via_a_real_file() {
        let calib = KvCalib::from_collector("file-model", 1, 1, 8, 0.99, &tiny_collector());
        let path = std::env::temp_dir().join(format!("kvcalib-test-{}.json", std::process::id()));
        calib.save(&path).unwrap();
        let loaded = KvCalib::load(&path).unwrap();
        assert_eq!(loaded.model_id, calib.model_id);
        assert_eq!(loaded.k, calib.k);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn from_model_dir_returns_none_without_a_file_and_without_erroring() {
        let dir = std::env::temp_dir().join(format!("kvcalib-empty-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(KvCalib::from_model_dir(&dir, 2, 3, 8, ).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_model_dir_returns_none_and_warns_on_a_shape_mismatch() {
        let dir = std::env::temp_dir().join(format!("kvcalib-mismatch-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        KvCalib::disabled(2, 3, 8).save(&dir.join("kv_calib.json")).unwrap();
        // Ask for a DIFFERENT shape than what was saved.
        assert!(KvCalib::from_model_dir(&dir, 4, 3, 8).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
