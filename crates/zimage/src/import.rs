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

use checkpoint::safetensors::StTensor;

use crate::block::Tensors;
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
}
