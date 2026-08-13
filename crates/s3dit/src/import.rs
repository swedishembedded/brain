// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the original / Comfy Z-Image checkpoint layout into brain's
//! (diffusers-named) tensor map.
//!
//! The shipped weights (`z_image_turbo_bf16.safetensors`) use the original
//! naming: a fused `attention.qkv.weight`, `attention.out`, `q_norm`/`k_norm`,
//! top-level `x_embedder`/`final_layer`. brain's model uses the diffusers names
//! ([`crate::ZImageModel`]). This reverses the official
//! `z_image_convert_original_to_comfy.py` map (verified against it):
//!   - `attention.qkv.weight` → split (chunk-3, q|k|v row-blocks) into
//!     `attention.{to_q,to_k,to_v}.weight`;
//!   - `attention.out` → `attention.to_out.0`;
//!   - `attention.{q,k}_norm.weight` → `attention.norm_{q,k}.weight`;
//!   - `x_embedder.` → `all_x_embedder.{ps}-{pf}.`;
//!   - `final_layer.` → `all_final_layer.{ps}-{pf}.`.

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

use crate::block::Tensors;
use crate::grad::WeightsF32;
use crate::modelgrad::ModelWeightsF32;
use crate::model::ZImageConfig;

/// Remap original/Comfy Z-Image tensors → brain's diffusers-named map. Splits
/// the fused qkv (Z-Image is full MHA, so q/k/v are equal `dim`-row thirds).
pub fn import_comfy(tensors: Vec<StTensor>, cfg: &ZImageConfig) -> Tensors {
    let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let dim = cfg.dim as usize;
    let mut out = Tensors::new();
    for t in tensors {
        if let Some(base) = t.name.strip_suffix("qkv.weight") {
            // base = "…attention." ; split [3·dim, dim] into q|k|v.
            let dd = dim * dim;
            assert_eq!(t.data.len(), 3 * dd, "qkv {} has {} != 3·dim² elems", t.name, t.data.len());
            out.insert(format!("{base}to_q.weight"), (vec![dim, dim], t.data[0..dd].to_vec()));
            out.insert(format!("{base}to_k.weight"), (vec![dim, dim], t.data[dd..2 * dd].to_vec()));
            out.insert(format!("{base}to_v.weight"), (vec![dim, dim], t.data[2 * dd..3 * dd].to_vec()));
            continue;
        }
        let mut k = t.name;
        k = k.replace(".attention.out.", ".attention.to_out.0.");
        k = k.replace(".attention.k_norm.weight", ".attention.norm_k.weight");
        k = k.replace(".attention.q_norm.weight", ".attention.norm_q.weight");
        if let Some(rest) = k.strip_prefix("x_embedder.") {
            k = format!("{xk}{rest}");
        } else if let Some(rest) = k.strip_prefix("final_layer.") {
            k = format!("{fk}{rest}");
        }
        out.insert(k, (t.shape, t.data));
    }
    out
}

/// The streaming sibling of [`import_comfy`]: a `checkpoint::remap::RemapSource`
/// over `r` resolving every brain (diffusers-named) tensor to its Comfy source
/// via the SAME rename/qkv-split rules — reading no tensor data up front. A
/// `qkv.weight` still resolves to three zero-copy [`checkpoint::remap::Fetch::Slice`]s
/// (slicing a borrow is still a borrow); every renamed tensor is a
/// [`checkpoint::remap::Fetch::Whole`] pass-through. `ZImageDit{,I8,Shard}::
/// build_from_source` accept the result directly, so a build from this never
/// materializes the whole DiT checkpoint on the host — peak allocation is one
/// tensor (up to ~157 MB for `feed_forward.w1`, once converted from BF16), not
/// the whole ~24 GB model.
pub fn comfy_source<'a>(r: &'a checkpoint::weightio::WeightReader, cfg: &ZImageConfig) -> checkpoint::remap::RemapSource<'a> {
    let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let dim = cfg.dim as usize;
    let mut plan: HashMap<String, checkpoint::remap::Fetch> = HashMap::new();
    for name in r.names() {
        if let Some(base) = name.strip_suffix("qkv.weight") {
            // base = "…attention."; split [3·dim, dim] into q|k|v row-blocks —
            // three Slice fetches over the SAME source tensor, zero-copy.
            let dd = dim * dim;
            plan.insert(format!("{base}to_q.weight"), checkpoint::remap::Fetch::Slice { name: name.to_string(), start: 0, len: dd });
            plan.insert(format!("{base}to_k.weight"), checkpoint::remap::Fetch::Slice { name: name.to_string(), start: dd, len: dd });
            plan.insert(format!("{base}to_v.weight"), checkpoint::remap::Fetch::Slice { name: name.to_string(), start: 2 * dd, len: dd });
            continue;
        }
        let mut k = name.to_string();
        k = k.replace(".attention.out.", ".attention.to_out.0.");
        k = k.replace(".attention.k_norm.weight", ".attention.norm_k.weight");
        k = k.replace(".attention.q_norm.weight", ".attention.norm_q.weight");
        if let Some(rest) = k.strip_prefix("x_embedder.") {
            k = format!("{xk}{rest}");
        } else if let Some(rest) = k.strip_prefix("final_layer.") {
            k = format!("{fk}{rest}");
        }
        plan.insert(k, checkpoint::remap::Fetch::Whole(name.to_string()));
    }
    checkpoint::remap::RemapSource::new(r, plan)
}

/// Bridge the (already-`import_comfy`'d) inference tensor map into the **training**
/// weight format [`ModelWeightsF32`] — the piece that lets a real shipped Z-Image
/// checkpoint be fine-tuned (LoRA or full). The inference path and this share one
/// source of truth: the exact same tensor keys the inference model reads
/// ([`crate::block::BlockWeights`]/[`NormBufs`], `model.rs` embedders/final) are
/// read here. Blocks stay f32 (the 24 GB runtime type); the ~100 MB wrapper linears
/// widen to f64 (where the host reference math runs). Errors name the first missing
/// tensor so a layout mismatch fails loudly instead of silently zero-filling.
pub fn model_weights_from_comfy(t: &Tensors, cfg: &ZImageConfig) -> Result<ModelWeightsF32, String> {
    let f32v = |k: &str| -> Result<Vec<f32>, String> {
        t.get(k).map(|(_, d)| d.clone()).ok_or_else(|| format!("import: missing tensor {k}"))
    };
    let f64v = |k: &str| -> Result<Vec<f64>, String> { Ok(f32v(k)?.iter().map(|&x| x as f64).collect()) };

    // One transformer block (15 tensors; adaLN present only when modulated — the
    // context_refiner blocks are UNmodulated, matching the inference model).
    let block = |prefix: &str, modulated: bool| -> Result<WeightsF32, String> {
        let g = |leaf: &str| f32v(&format!("{prefix}.{leaf}"));
        Ok(WeightsF32 {
            wq: g("attention.to_q.weight")?,
            wk: g("attention.to_k.weight")?,
            wv: g("attention.to_v.weight")?,
            wo: g("attention.to_out.0.weight")?,
            w1: g("feed_forward.w1.weight")?,
            w2: g("feed_forward.w2.weight")?,
            w3: g("feed_forward.w3.weight")?,
            nq: g("attention.norm_q.weight")?,
            nk: g("attention.norm_k.weight")?,
            an1: g("attention_norm1.weight")?,
            an2: g("attention_norm2.weight")?,
            fn1: g("ffn_norm1.weight")?,
            fn2: g("ffn_norm2.weight")?,
            adaln_w: if modulated { g("adaLN_modulation.0.weight")? } else { Vec::new() },
            adaln_b: if modulated { g("adaLN_modulation.0.bias")? } else { Vec::new() },
        })
    };

    let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
    let xk = format!("all_x_embedder.{ps}-{pf}");
    let fk = format!("all_final_layer.{ps}-{pf}");
    let mut noise_ref = Vec::with_capacity(cfg.n_refiner_layers as usize);
    let mut ctx_ref = Vec::with_capacity(cfg.n_refiner_layers as usize);
    for l in 0..cfg.n_refiner_layers {
        noise_ref.push(block(&format!("noise_refiner.{l}"), true)?);
        ctx_ref.push(block(&format!("context_refiner.{l}"), false)?);
    }
    let mut main = Vec::with_capacity(cfg.n_layers as usize);
    for l in 0..cfg.n_layers {
        main.push(block(&format!("layers.{l}"), true)?);
    }

    Ok(ModelWeightsF32 {
        t0_w: f64v("t_embedder.mlp.0.weight")?, t0_b: f64v("t_embedder.mlp.0.bias")?,
        t2_w: f64v("t_embedder.mlp.2.weight")?, t2_b: f64v("t_embedder.mlp.2.bias")?,
        xemb_w: f64v(&format!("{xk}.weight"))?, xemb_b: f64v(&format!("{xk}.bias"))?,
        capn_w: f64v("cap_embedder.0.weight")?,
        cap1_w: f64v("cap_embedder.1.weight")?, cap1_b: f64v("cap_embedder.1.bias")?,
        noise_ref, ctx_ref, main,
        fadaln_w: f64v(&format!("{fk}.adaLN_modulation.1.weight"))?, fadaln_b: f64v(&format!("{fk}.adaLN_modulation.1.bias"))?,
        flin_w: f64v(&format!("{fk}.linear.weight"))?, flin_b: f64v(&format!("{fk}.linear.bias"))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(name: &str, data: Vec<f32>) -> StTensor {
        StTensor { name: name.to_string(), shape: vec![data.len()], data }
    }

    #[test]
    fn remap_and_qkv_split() {
        let mut cfg = ZImageConfig::turbo();
        cfg.dim = 2; // tiny: qkv = [6, 2] = 12 elems, split into 3× [2,2]=4.
        let qkv: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensors = vec![
            st("layers.0.attention.qkv.weight", qkv),
            st("layers.0.attention.out.weight", vec![9.0; 4]),
            st("layers.0.attention.q_norm.weight", vec![1.0; 2]),
            st("layers.0.attention.k_norm.weight", vec![1.0; 2]),
            st("x_embedder.weight", vec![2.0; 8]),
            st("final_layer.linear.bias", vec![3.0; 4]),
            st("cap_embedder.0.weight", vec![4.0; 2]),
        ];
        let m = import_comfy(tensors, &cfg);
        // qkv split: q=[0,1,2,3], k=[4,5,6,7], v=[8,9,10,11].
        assert_eq!(m["layers.0.attention.to_q.weight"].1, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(m["layers.0.attention.to_k.weight"].1, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(m["layers.0.attention.to_v.weight"].1, vec![8.0, 9.0, 10.0, 11.0]);
        // renames
        assert!(m.contains_key("layers.0.attention.to_out.0.weight"));
        assert!(m.contains_key("layers.0.attention.norm_q.weight"));
        assert!(m.contains_key("layers.0.attention.norm_k.weight"));
        assert!(m.contains_key("all_x_embedder.2-1.weight"));
        assert!(m.contains_key("all_final_layer.2-1.linear.bias"));
        // untouched
        assert!(m.contains_key("cap_embedder.0.weight"));
    }

    /// Insert the per-block tensor keys (post-import names) for `prefix`.
    fn ins_block(m: &mut Tensors, prefix: &str, modulated: bool) {
        for leaf in [
            "attention.to_q.weight", "attention.to_k.weight", "attention.to_v.weight",
            "attention.to_out.0.weight", "feed_forward.w1.weight", "feed_forward.w2.weight",
            "feed_forward.w3.weight", "attention.norm_q.weight", "attention.norm_k.weight",
            "attention_norm1.weight", "attention_norm2.weight", "ffn_norm1.weight", "ffn_norm2.weight",
        ] {
            m.insert(format!("{prefix}.{leaf}"), (vec![1], vec![1.0]));
        }
        if modulated {
            m.insert(format!("{prefix}.adaLN_modulation.0.weight"), (vec![1], vec![2.0]));
            m.insert(format!("{prefix}.adaLN_modulation.0.bias"), (vec![1], vec![3.0]));
        }
    }

    #[test]
    fn bridge_to_training_weights_covers_blocks_and_wrapper() {
        let mut cfg = ZImageConfig::turbo();
        cfg.dim = 2;
        cfg.n_layers = 3;
        cfg.n_refiner_layers = 2;
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let mut m = Tensors::new();
        for l in 0..cfg.n_layers {
            ins_block(&mut m, &format!("layers.{l}"), true);
        }
        for l in 0..cfg.n_refiner_layers {
            ins_block(&mut m, &format!("noise_refiner.{l}"), true);
            ins_block(&mut m, &format!("context_refiner.{l}"), false); // UNmodulated
        }
        for k in [
            "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", "t_embedder.mlp.2.weight",
            "t_embedder.mlp.2.bias", "cap_embedder.0.weight", "cap_embedder.1.weight",
            "cap_embedder.1.bias",
            &format!("all_x_embedder.{ps}-{pf}.weight"), &format!("all_x_embedder.{ps}-{pf}.bias"),
            &format!("all_final_layer.{ps}-{pf}.adaLN_modulation.1.weight"),
            &format!("all_final_layer.{ps}-{pf}.adaLN_modulation.1.bias"),
            &format!("all_final_layer.{ps}-{pf}.linear.weight"), &format!("all_final_layer.{ps}-{pf}.linear.bias"),
        ] {
            m.insert(k.to_string(), (vec![1], vec![7.0]));
        }

        let w = model_weights_from_comfy(&m, &cfg).expect("bridge");
        assert_eq!(w.main.len(), 3);
        assert_eq!(w.noise_ref.len(), 2);
        assert_eq!(w.ctx_ref.len(), 2);
        // context_refiner is unmodulated → empty adaLN; noise_refiner/main have it.
        assert!(w.ctx_ref[0].adaln_w.is_empty() && w.ctx_ref[0].adaln_b.is_empty());
        assert!(!w.noise_ref[0].adaln_w.is_empty());
        assert!(!w.main[0].adaln_w.is_empty());
        assert!(!w.main[0].wq.is_empty() && !w.xemb_w.is_empty() && !w.flin_w.is_empty());

        // A missing tensor must error (loudly, named) — not silently zero-fill.
        m.remove("layers.1.feed_forward.w2.weight");
        let err = match model_weights_from_comfy(&m, &cfg) {
            Ok(_) => panic!("expected a missing-tensor error"),
            Err(e) => e,
        };
        assert!(err.contains("layers.1.feed_forward.w2.weight"), "unhelpful error: {err}");
    }

    /// [`comfy_source`] must be byte-for-byte identical to the eager
    /// [`import_comfy`] for every renamed AND every qkv-split tensor — the
    /// same tiny fixture `remap_and_qkv_split` above uses, round-tripped
    /// through a real safetensors file so `comfy_source` reads it via a
    /// genuine `WeightReader`, not an in-memory shortcut.
    #[test]
    fn comfy_source_streaming_matches_eager_import_comfy() {
        use checkpoint::TensorSource;

        let mut cfg = ZImageConfig::turbo();
        cfg.dim = 2; // tiny: qkv = [6, 2] = 12 elems, split into 3x [2,2]=4.
        let qkv: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let named: Vec<(&str, Vec<f32>)> = vec![
            ("layers.0.attention.qkv.weight", qkv),
            ("layers.0.attention.out.weight", vec![9.0; 4]),
            ("layers.0.attention.q_norm.weight", vec![1.0; 2]),
            ("layers.0.attention.k_norm.weight", vec![1.5; 2]),
            ("x_embedder.weight", vec![2.0; 8]),
            ("final_layer.linear.bias", vec![3.0; 4]),
            ("cap_embedder.0.weight", vec![4.0; 2]),
        ];
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = named.iter().map(|(n, d)| (n.to_string(), vec![d.len() as u64], d.clone())).collect();
        let path = std::env::temp_dir().join(format!("brain-zimage-comfy-streaming-{}.safetensors", std::process::id()));
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();

        // Eager reference, over the exact same source tensors.
        let eager = import_comfy(named.into_iter().map(|(n, d)| st(n, d)).collect(), &cfg);

        let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap();
        let streamed = comfy_source(&reader, &cfg);

        // Same tensors `remap_and_qkv_split` checks explicitly, so a rename or
        // a qkv-split slice-boundary regression fails here identically.
        for name in eager.keys() {
            let mut got = None;
            assert!(streamed.with_tensor(name, &mut |d| got = Some(d.to_vec())), "missing {name}");
            assert_eq!(got.unwrap(), eager[name].1, "{name}");
        }

        std::fs::remove_file(&path).ok();
    }
}
