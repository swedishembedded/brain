// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Read the official `Qwen3-TTS-Tokenizer-12Hz` safetensors checkpoint as
//! brain's codec, as it is downloaded ([`view`]), or write it out as a brain
//! `.safetensors` container ([`import`]) — decode path **and** (additively) the
//! encode path.
//!
//! The decoder lives under the `decoder.*` prefix (271 tensors); the encoder
//! lives under `encoder.*` (225 tensors, a HuggingFace `MimiModel`). We do a
//! near 1:1 name remap (the decoder strips its `decoder.` prefix; the encoder
//! keeps its `encoder.` prefix so the two never collide) with these transforms:
//!   * each Euclidean codebook is collapsed, when it is read, from its two stored
//!     tensors `embedding_sum/embed_sum [bins,dim]` + `cluster_usage [bins]` into
//!     the usable embedding table `table = embed_sum / clamp(cluster_usage, eps)`
//!     (matches `EuclideanCodebook.decode`/`MimiEuclideanCodebook.embed`, eps =
//!     1e-5) — applied to both the decoder's codebooks and the encoder's;
//!   * the **decoder** quantizers' `input_proj` is dropped (decode never uses it);
//!     the **encoder** quantizers' `input_proj` is KEPT (encode projects
//!     `hidden_size -> codebook_dim` before the nearest-codebook search), while
//!     the encoder quantizers' `output_proj` (decode-side) is dropped;
//!   * the encoder only keeps the first `encoder_valid_num_quantizers` codebooks
//!     (1 semantic + 15 acoustic); the rest of the 32-deep RVQ and the codebooks'
//!     `initialized` flags are dropped (`encode` never reads past code 16);
//!
//! No tensor is transposed: brain `matmul` is `x @ Wᵀ` with `W:[out,in]`, exactly
//! `nn.Linear.weight`, and conv weights keep PyTorch `[Cout,Cin/G,K]` /
//! `[Cin,Cout/G,K]` layout that `conv1d`/`convtr1d` already expect.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use checkpoint::weightio::{DerivedCheckpoint, WeightReader};

/// Clamp epsilon for `EuclideanCodebook` (the reference's default `epsilon`).
const CODEBOOK_EPS: f32 = 1e-5;

/// Whether an encoder RVQ codebook layer (e.g.
/// `quantizer.acoustic_residual_vector_quantizer.layers.7.codebook.embed_sum`)
/// falls within the kept range: all semantic layers, but only the first
/// `n_aco_keep` acoustic layers (`encoder_valid_num_quantizers - num_semantic`).
fn quant_layer_in_range(name: &str, n_aco_keep: usize) -> bool {
    let idx = name
        .split("layers.")
        .nth(1)
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    if name.contains("semantic_residual_vector_quantizer") {
        true
    } else {
        idx < n_aco_keep
    }
}

/// Where one HF source tensor ends up: a plain 1:1 passthrough (with its final
/// output name), one half of a codebook pair (namespaced `d:`/`e:` parent key —
/// decoder and encoder codebooks never collide), or dropped.
enum Slot {
    Out(String),
    EmbSum(String),
    Cluster(String),
    Drop,
}

fn classify(full_name: &str, n_aco_keep: usize) -> Slot {
    if let Some(name) = full_name.strip_prefix("decoder.") {
        if let Some(parent) = name.strip_suffix("._codebook.embedding_sum") {
            return Slot::EmbSum(format!("d:{parent}"));
        }
        if let Some(parent) = name.strip_suffix("._codebook.cluster_usage") {
            return Slot::Cluster(format!("d:{parent}"));
        }
        if name.starts_with("quantizer.") && name.ends_with("input_proj.weight") {
            return Slot::Drop; // encode-side projection, unused on decode
        }
        return Slot::Out(name.to_string());
    }
    // ---- encoder.* (HuggingFace MimiModel) — the encode path ----
    let Some(name) = full_name.strip_prefix("encoder.") else {
        return Slot::Drop; // neither decoder nor encoder — ignore
    };
    // Keep only the first `valid_q` RVQ codebooks; drop the rest of the 32-deep
    // stack, the `initialized` flags, and the decode-side output_proj.
    if name.contains("residual_vector_quantizer.layers.") {
        let keep = quant_layer_in_range(name, n_aco_keep);
        if let Some(parent) = name.strip_suffix(".codebook.embed_sum") {
            return if keep { Slot::EmbSum(format!("e:encoder.{parent}")) } else { Slot::Drop };
        }
        if let Some(parent) = name.strip_suffix(".codebook.cluster_usage") {
            return if keep { Slot::Cluster(format!("e:encoder.{parent}")) } else { Slot::Drop };
        }
        return Slot::Drop; // `.codebook.initialized` and any other per-layer buffer
    }
    if name.ends_with("output_proj.weight") {
        return Slot::Drop; // encoder RVQ decode-side projection, unused on encode
    }
    Slot::Out(format!("encoder.{name}"))
}

/// How one codec tensor is read from the source checkpoint.
enum Derive {
    /// Verbatim, from this source tensor.
    Direct(String),
    /// `embed_sum / clamp(cluster_usage, eps)` of a codebook's two halves.
    Table { sum: String, usage: String },
}

/// The codec of a `Qwen3-TTS-Tokenizer-12Hz` checkpoint under brain's names,
/// each codebook collapsed into its table when it is read.
struct CodecView {
    src: WeightReader,
    index: Vec<(String, Vec<u64>)>,
    how: HashMap<String, Derive>,
    config: serde_json::Value,
}

impl DerivedCheckpoint for CodecView {
    fn index(&self) -> Vec<(String, Vec<u64>, &'static str)> {
        self.index.iter().map(|(n, s)| (n.clone(), s.clone(), "F32")).collect()
    }
    fn tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        match self.how.get(name)? {
            Derive::Direct(hf) => self.src.tensor(hf),
            Derive::Table { sum, usage } => {
                let (sum, usage) = (self.src.tensor(sum)?, self.src.tensor(usage)?);
                let dim = sum.len() / usage.len();
                Some(sum.iter().enumerate().map(|(i, s)| s / usage[i / dim].max(CODEBOOK_EPS)).collect())
            }
        }
    }
    fn tensor_u32(&self, _name: &str) -> Option<Vec<u32>> {
        None
    }
    fn config(&self) -> serde_json::Value {
        self.config.clone()
    }
}

/// The codec of the checkpoint at `ckpt_dir` (`config.json` +
/// `model.safetensors`), read under brain's names as it is downloaded. Fails
/// when there is no `decoder.*` tensor, or a codebook lacks one of its halves
/// or their shapes disagree.
pub fn view(ckpt_dir: &Path) -> Result<WeightReader, String> {
    let cfg_json = std::fs::read_to_string(ckpt_dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&cfg_json).map_err(|e| format!("parse config.json: {e}"))?;

    let st_path = ckpt_dir.join("model.safetensors");
    let src = WeightReader::open(st_path.to_str().ok_or("non-utf8 checkpoint path")?)
        .map_err(|e| format!("import: opening checkpoint: {e}"))?;

    // How many encoder quantizers to keep (1 semantic + 15 acoustic by default).
    let valid_q = config["encoder_valid_num_quantizers"].as_u64().unwrap_or(16) as usize;
    let n_sem = config["encoder_config"]["num_semantic_quantizers"].as_u64().unwrap_or(1) as usize;
    let n_aco_keep = valid_q.saturating_sub(n_sem);

    let mut index: Vec<(String, Vec<u64>)> = Vec::new();
    let mut how: HashMap<String, Derive> = HashMap::new();
    let mut sums: BTreeMap<String, String> = BTreeMap::new();
    let mut usages: BTreeMap<String, String> = BTreeMap::new();
    let mut decoder_seen = 0usize;
    for full_name in src.names() {
        if full_name.starts_with("decoder.") {
            decoder_seen += 1;
        }
        match classify(full_name, n_aco_keep) {
            Slot::Out(out_name) => {
                let shape = src.shape(full_name).unwrap_or_default().to_vec();
                if how.insert(out_name.clone(), Derive::Direct(full_name.to_string())).is_some() {
                    return Err(format!("duplicate tensor {out_name}"));
                }
                index.push((out_name, shape));
            }
            Slot::EmbSum(key) => {
                sums.insert(key, full_name.to_string());
            }
            Slot::Cluster(key) => {
                usages.insert(key, full_name.to_string());
            }
            Slot::Drop => {}
        }
    }
    if decoder_seen == 0 {
        return Err("no decoder.* tensors found in checkpoint".to_string());
    }
    if sums.len() != usages.len() {
        return Err(format!("codebook pairing mismatch: {} embedding_sum vs {} cluster_usage", sums.len(), usages.len()));
    }
    for (key, sum) in sums {
        let usage = usages.remove(&key).ok_or_else(|| format!("codebook {key}: missing cluster_usage"))?;
        let shape = src.shape(&sum).unwrap_or_default().to_vec();
        let bins = src.shape(&usage).unwrap_or_default().iter().product::<u64>();
        if shape.len() != 2 || shape[0] != bins {
            return Err(format!("codebook {key}: usage has {bins} bins, embedding sum is {shape:?}"));
        }
        let bare = key.split_once(':').map(|(_, r)| r).unwrap_or(key.as_str());
        let out_name = format!("{bare}.table");
        if how.insert(out_name.clone(), Derive::Table { sum, usage }).is_some() {
            return Err(format!("duplicate tensor {out_name}"));
        }
        index.push((out_name, shape));
    }
    Ok(WeightReader::derived(Box::new(CodecView { src, index, how, config })))
}

/// A codec checkpoint as the loaders take it: a brain file (from [`import`])
/// as it is, or the tokenizer checkpoint dir through [`view`].
pub fn open(path: &str) -> Result<WeightReader, String> {
    let r = if Path::new(path).is_dir() { view(Path::new(path)) } else { WeightReader::open(path).map_err(|e| e.to_string()) };
    r.map_err(|e| format!("{path}: {e}"))
}

/// [`open`], every tensor read as f32, for the eager loaders. Panics naming
/// `path` when it cannot be read, as `checkpoint::load` does.
pub fn load(path: &str) -> checkpoint::Container {
    open(path).and_then(|r| checkpoint::load_reader(&r)).unwrap_or_else(|e| panic!("{e}"))
}

/// Write the codec of `<ckpt_dir>` to the brain checkpoint `out_path` -
/// [`view`], one tensor at a time.
pub fn import(ckpt_dir: &str, out_path: &str) -> Result<(), String> {
    let v = view(Path::new(ckpt_dir))?;
    v.save(out_path, None).map_err(|e| format!("import: {e}"))?;
    eprintln!("codec import: {} params in {out_path} (codebooks collapsed)", v.names().count());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streaming `import` with TWO decoder codebooks + TWO encoder RVQ
    /// codebooks (one semantic, one acoustic, plus one out-of-range acoustic
    /// layer that must be dropped), each with distinct bins/dim/values — proves
    /// the fan-in collapse pairs `embed_sum`/`cluster_usage` correctly and never
    /// aliases one codebook's values onto another's table, alongside plain
    /// passthrough tensors and every documented drop case.
    #[test]
    fn streaming_import_collapses_codebooks_without_cross_aliasing() {
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("codec-import-src-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        let config = serde_json::json!({
            "encoder_valid_num_quantizers": 2,
            "encoder_config": {"num_semantic_quantizers": 1},
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();

        let plan: Vec<(String, Vec<u64>)> = vec![
            ("decoder.foo.weight".to_string(), vec![3]),
            ("decoder.quantizer.input_proj.weight".to_string(), vec![2]),
            ("decoder.quantizer.rvq.layers.0._codebook.embedding_sum".to_string(), vec![3, 2]),
            ("decoder.quantizer.rvq.layers.0._codebook.cluster_usage".to_string(), vec![3]),
            ("decoder.quantizer.rvq.layers.1._codebook.embedding_sum".to_string(), vec![2, 3]),
            ("decoder.quantizer.rvq.layers.1._codebook.cluster_usage".to_string(), vec![2]),
            ("encoder.bar.weight".to_string(), vec![2]),
            (
                "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum".to_string(),
                vec![2, 4],
            ),
            (
                "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage".to_string(),
                vec![2],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.embed_sum".to_string(),
                vec![3, 1],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.cluster_usage".to_string(),
                vec![3],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized".to_string(),
                vec![1],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.embed_sum".to_string(),
                vec![3, 1],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.cluster_usage".to_string(),
                vec![3],
            ),
            ("encoder.quantizer.some.output_proj.weight".to_string(), vec![2]),
        ];
        let src = dir.join("model.safetensors");
        let mut w =
            checkpoint::weightio::StWriter::create(src.to_str().unwrap(), &plan, &serde_json::Value::Null, None)
                .unwrap();
        w.write("decoder.foo.weight", &[7.0, 8.0, 9.0]).unwrap();
        w.write("decoder.quantizer.input_proj.weight", &[0.0, 0.0]).unwrap();
        w.write("decoder.quantizer.rvq.layers.0._codebook.embedding_sum", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        w.write("decoder.quantizer.rvq.layers.0._codebook.cluster_usage", &[1.0, 2.0, 5.0]).unwrap();
        w.write(
            "decoder.quantizer.rvq.layers.1._codebook.embedding_sum",
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .unwrap();
        w.write("decoder.quantizer.rvq.layers.1._codebook.cluster_usage", &[2.0, 4.0]).unwrap();
        w.write("encoder.bar.weight", &[10.0, 11.0]).unwrap();
        w.write(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
            &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        )
        .unwrap();
        w.write(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage",
            &[1.0, 4.0],
        )
        .unwrap();
        w.write(
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.embed_sum",
            &[9.0, 18.0, 27.0],
        )
        .unwrap();
        w.write("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.cluster_usage", &[3.0, 6.0, 9.0])
            .unwrap();
        w.write("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized", &[1.0]).unwrap();
        w.write(
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.embed_sum",
            &[100.0, 200.0, 300.0],
        )
        .unwrap();
        w.write("encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.cluster_usage", &[1.0, 1.0, 1.0])
            .unwrap();
        w.write("encoder.quantizer.some.output_proj.weight", &[0.0, 0.0]).unwrap();
        w.finish().unwrap();

        let out = std::env::temp_dir().join(format!("codec-import-out-{pid}.safetensors"));
        import(dir.to_str().unwrap(), out.to_str().unwrap()).unwrap();

        let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).unwrap();
        // passthrough tensors, prefix handling preserved.
        assert_eq!(reader.tensor("foo.weight").unwrap(), vec![7.0, 8.0, 9.0]);
        assert_eq!(reader.tensor("encoder.bar.weight").unwrap(), vec![10.0, 11.0]);

        // codebook collapse: embed_sum / clamp(cluster_usage, eps), hand-computed.
        assert_eq!(
            reader.tensor("quantizer.rvq.layers.0.table").unwrap(),
            vec![1.0, 2.0, 1.5, 2.0, 1.0, 1.2]
        );
        assert_eq!(
            reader.tensor("quantizer.rvq.layers.1.table").unwrap(),
            vec![5.0, 10.0, 15.0, 10.0, 12.5, 15.0]
        );
        assert_eq!(
            reader.tensor("encoder.quantizer.semantic_residual_vector_quantizer.layers.0.table").unwrap(),
            vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
        );
        assert_eq!(
            reader.tensor("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.table").unwrap(),
            vec![3.0, 3.0, 3.0]
        );

        // dropped: the out-of-range acoustic layer never produces a table (no
        // aliasing with layer 0's very different values), nor do input_proj /
        // output_proj / `initialized`.
        assert!(reader.tensor("encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.table").is_none());
        assert!(reader.tensor("quantizer.input_proj.weight").is_none());
        assert!(reader.tensor("encoder.quantizer.some.output_proj.weight").is_none());
        assert!(reader
            .tensor("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized")
            .is_none());

        assert_eq!(reader.names().count(), 6, "2 passthrough + 4 collapsed tables");

        // Served as downloaded: the view reads what the import wrote, and
        // writes nothing beside the checkpoint.
        let (viewed, written) = (load(dir.to_str().unwrap()), checkpoint::load(out.to_str().unwrap()));
        assert_eq!(viewed.header, written.header);
        assert_eq!(viewed.by_role(""), written.by_role(""));
        let mut listed: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        listed.sort();
        assert_eq!(listed, ["config.json", "model.safetensors"]);

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }
}
