// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M23.1 follow-up: `gguf_vs_fp8_weights_real.rs` found `A_log`/`dt_bias`/
//! `conv1d.weight` with an EXACT digest match (same multiset of values) but
//! low cosine at layers 0 and 1 - a pure reordering, no quantization noise
//! (these leaves are small enough that llama.cpp very likely stores them
//! unquantized). This searches the small candidate space M21 already named
//! (`repeat_interleave` vs `tile` over `linear_num_value_heads`/
//! `linear_num_key_heads`) for the exact permutation, on the 48-element
//! `A_log`/`dt_bias` vectors where it is cheapest to see.
//!
//! ```text
//! BRAIN_QWEN35_DIR=/path/to/Qwen3.8-27B-FP8 \
//! BRAIN_QWEN35_GGUF=/path/to/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --release --test gguf_vs_fp8_permutation_search -- --nocapture
//! ```

use std::path::PathBuf;

use checkpoint::TensorSource;
use checkpoint::gguf::MmapGguf;
use checkpoint::mmap::MmapSafetensors;
use model::Shard;
use qwen35::config::Qwen35Config;
use qwen35::import::import_layer;
use qwen35::int8_gguf_resident::shard_source;

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_QWEN35_DIR").map(PathBuf::from)
}

fn gguf_path() -> Option<String> {
    std::env::var("BRAIN_QWEN35_GGUF").ok().filter(|p| !p.is_empty())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

/// group-major (g*sub_len+s) -> sub-major (s*n_groups+g), the "tile vs
/// repeat_interleave" transpose over a `[n_groups, sub_len]` reshape.
fn transpose(v: &[f32], n_groups: usize, sub_len: usize) -> Vec<f32> {
    assert_eq!(v.len(), n_groups * sub_len);
    let mut out = vec![0f32; v.len()];
    for g in 0..n_groups {
        for s in 0..sub_len {
            // v is [n_groups, sub_len] (group-major); out is [sub_len, n_groups] (sub-major), flattened.
            out[s * n_groups + g] = v[g * sub_len + s];
        }
    }
    out
}

#[test]
fn search_the_head_permutation_on_a_log_and_dt_bias() {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset");
        return;
    };
    let Some(gguf_p) = gguf_path() else {
        brain_testutil::skip("BRAIN_QWEN35_GGUF unset");
        return;
    };

    let cfg = Qwen35Config::qwen38_27b();
    let gguf = MmapGguf::open(&gguf_p).unwrap_or_else(|e| panic!("open {gguf_p}: {e}"));
    let nvh = cfg.linear_num_value_heads as usize; // 48
    let nkh = cfg.linear_num_key_heads as usize; // 16
    let group = cfg.linear_group() as usize; // 3
    assert_eq!(nvh, nkh * group);

    for l in [0usize, 1] {
        let shard_path = dir.join(format!("layers-{l}.safetensors"));
        let Ok(fp8) = MmapSafetensors::open(&shard_path) else {
            println!("layer {l}: not present, skipping");
            continue;
        };
        let fp8_layer = import_layer(&fp8, &cfg, l, 128).unwrap_or_else(|e| panic!("import_layer({l}): {e}"));
        let shard = Shard { start: l, end: l + 1, embed: false, head: false, gpu_index: Shard::ANY_GPU };
        let src = shard_source(&gguf, &cfg, &shard).unwrap_or_else(|e| panic!("shard_source: {e}"));

        for leaf in ["linear_attn.A_log", "linear_attn.dt_bias"] {
            let name = format!("blocks.{l}.{leaf}");
            let fp8_v = fp8_layer.get(&name).unwrap_or_else(|| panic!("fp8 missing {name}"));
            let mut gguf_v: Option<Vec<f32>> = None;
            src.with_tensor(&name, &mut |raw| gguf_v = Some(raw.to_vec()));
            let gguf_v = gguf_v.unwrap_or_else(|| panic!("gguf missing {name}"));
            assert_eq!(fp8_v.len(), nvh);
            assert_eq!(gguf_v.len(), nvh);

            let identity_cos = cosine(&gguf_v, fp8_v);
            // Candidate A: gguf is group-major [nkh, group] -> transpose to sub-major.
            let cand_a = transpose(&gguf_v, nkh, group);
            let cos_a = cosine(&cand_a, fp8_v);
            // Candidate B: gguf is group-major [group, nkh] -> transpose to sub-major.
            let cand_b = transpose(&gguf_v, group, nkh);
            let cos_b = cosine(&cand_b, fp8_v);
            // Candidate C: plain reverse.
            let cand_c: Vec<f32> = gguf_v.iter().rev().copied().collect();
            let cos_c = cosine(&cand_c, fp8_v);

            println!(
                "layer {l} {leaf}: identity cos={identity_cos:.4}  transpose[{nkh},{group}]->sub-major cos={cos_a:.4}  transpose[{group},{nkh}]->sub-major cos={cos_b:.4}  reverse cos={cos_c:.4}"
            );
            if identity_cos < 0.9999 {
                println!("  fp8  = {fp8_v:?}");
                println!("  gguf = {gguf_v:?}");
            }
        }

        // Per-head ROW permutation on [nvh, d_model] matrices: apply the same
        // group-major -> sub-major transpose to the ROW INDEX (not the flat
        // buffer), since these are Q8_0-quantized (real rounding noise on
        // top, so an exact cos=1.0 is not expected - only a large recovery).
        let d = cfg.d_model as usize;
        for leaf in ["linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"] {
            let name = format!("blocks.{l}.{leaf}");
            let fp8_v = fp8_layer.get(&name).unwrap_or_else(|| panic!("fp8 missing {name}"));
            let mut gguf_v: Option<Vec<f32>> = None;
            src.with_tensor(&name, &mut |raw| gguf_v = Some(raw.to_vec()));
            let gguf_v = gguf_v.unwrap_or_else(|| panic!("gguf missing {name}"));
            assert_eq!(fp8_v.len(), nvh * d);

            let identity_cos = cosine(&gguf_v, fp8_v);
            // dst row r is SUB-MAJOR (r = s*group+g, s outer/slow, g
            // inner/fast) - s = r/group, g = r%group - reading from the
            // GROUP-MAJOR source row g*nkh+s. (Was backwards - r/nkh,r%nkh -
            // in an earlier pass; fixed to match the exact transpose found
            // above.)
            let mut permuted = vec![0f32; gguf_v.len()];
            for r in 0..nvh {
                let (s, g) = (r / group, r % group);
                let src_row = g * nkh + s;
                permuted[r * d..(r + 1) * d].copy_from_slice(&gguf_v[src_row * d..(src_row + 1) * d]);
            }
            let permuted_cos = cosine(&permuted, fp8_v);
            println!("layer {l} {leaf}: identity cos={identity_cos:.4}  row-permuted cos={permuted_cos:.6}");
        }

        // conv1d.weight: [conv_dim, kernel] = [10240, 4], conv_dim = q(2048) |
        // k(2048) | v(6144=48*128) concatenated. Test the SAME per-head-block
        // permutation on just the v-portion's 48 128-channel blocks (q/k
        // portions have no per-value-head structure, left untouched).
        {
            let name = format!("blocks.{l}.linear_attn.conv1d.weight");
            let fp8_v = fp8_layer.get(&name).unwrap_or_else(|| panic!("fp8 missing {name}"));
            let mut gguf_v: Option<Vec<f32>> = None;
            src.with_tensor(&name, &mut |raw| gguf_v = Some(raw.to_vec()));
            let gguf_v = gguf_v.unwrap_or_else(|| panic!("gguf missing {name}"));
            let kernel = 4usize;
            let key_dim = cfg.linear_key_dim() as usize; // 2048
            let v_off = 2 * key_dim; // 4096
            let head_dim = cfg.linear_value_head_dim as usize; // 128
            assert_eq!(gguf_v.len(), (v_off + nvh * head_dim) * kernel);

            let identity_cos = cosine(&gguf_v, fp8_v);
            let mut permuted = gguf_v.clone();
            for h in 0..nvh {
                let (s, g) = (h / group, h % group);
                let src_head = g * nkh + s;
                for d in 0..head_dim {
                    let dst_ch = v_off + h * head_dim + d;
                    let src_ch = v_off + src_head * head_dim + d;
                    permuted[dst_ch * kernel..(dst_ch + 1) * kernel].copy_from_slice(&gguf_v[src_ch * kernel..(src_ch + 1) * kernel]);
                }
            }
            let permuted_cos = cosine(&permuted, fp8_v);
            println!("layer {l} linear_attn.conv1d.weight: identity cos={identity_cos:.4}  v-block-permuted cos={permuted_cos:.6}");
        }

        // in_proj_qkv: [conv_dim, d_model] = [10240, 5120], same q|k|v(48
        // heads x 128) row layout as conv1d's channel axis - permute only
        // the v-portion's 48 row-blocks.
        {
            let name = format!("blocks.{l}.linear_attn.in_proj_qkv.weight");
            let fp8_v = fp8_layer.get(&name).unwrap_or_else(|| panic!("fp8 missing {name}"));
            let mut gguf_v: Option<Vec<f32>> = None;
            src.with_tensor(&name, &mut |raw| gguf_v = Some(raw.to_vec()));
            let gguf_v = gguf_v.unwrap_or_else(|| panic!("gguf missing {name}"));
            let key_dim = cfg.linear_key_dim() as usize;
            let v_off = 2 * key_dim;
            let head_dim = cfg.linear_value_head_dim as usize;
            assert_eq!(gguf_v.len(), (v_off + nvh * head_dim) * d);

            let identity_cos = cosine(&gguf_v, fp8_v);
            let mut permuted = gguf_v.clone();
            for h in 0..nvh {
                let (s, g) = (h / group, h % group);
                let src_head = g * nkh + s;
                for row_in_head in 0..head_dim {
                    let dst_row = v_off + h * head_dim + row_in_head;
                    let src_row = v_off + src_head * head_dim + row_in_head;
                    permuted[dst_row * d..(dst_row + 1) * d].copy_from_slice(&gguf_v[src_row * d..(src_row + 1) * d]);
                }
            }
            let permuted_cos = cosine(&permuted, fp8_v);
            println!("layer {l} linear_attn.in_proj_qkv.weight: identity cos={identity_cos:.4}  v-block-permuted cos={permuted_cos:.6}");
        }

        // in_proj_z: [value_dim, d_model] = [6144, 5120], PURE v (48 heads x
        // 128) - permute all 48 row-blocks.
        {
            let name = format!("blocks.{l}.linear_attn.in_proj_z.weight");
            let fp8_v = fp8_layer.get(&name).unwrap_or_else(|| panic!("fp8 missing {name}"));
            let mut gguf_v: Option<Vec<f32>> = None;
            src.with_tensor(&name, &mut |raw| gguf_v = Some(raw.to_vec()));
            let gguf_v = gguf_v.unwrap_or_else(|| panic!("gguf missing {name}"));
            let head_dim = cfg.linear_value_head_dim as usize;
            assert_eq!(gguf_v.len(), nvh * head_dim * d);

            let identity_cos = cosine(&gguf_v, fp8_v);
            let mut permuted = gguf_v.clone();
            for h in 0..nvh {
                let (s, g) = (h / group, h % group);
                let src_head = g * nkh + s;
                for row_in_head in 0..head_dim {
                    let dst_row = h * head_dim + row_in_head;
                    let src_row = src_head * head_dim + row_in_head;
                    permuted[dst_row * d..(dst_row + 1) * d].copy_from_slice(&gguf_v[src_row * d..(src_row + 1) * d]);
                }
            }
            let permuted_cos = cosine(&permuted, fp8_v);
            println!("layer {l} linear_attn.in_proj_z.weight: identity cos={identity_cos:.4}  v-block-permuted cos={permuted_cos:.6}");
        }

        // out_proj: [d_model, value_dim] = [5120, 6144] - value_dim is the
        // INPUT (column) axis here, so permute COLUMN blocks, not rows.
        {
            let name = format!("blocks.{l}.linear_attn.out_proj.weight");
            let fp8_v = fp8_layer.get(&name).unwrap_or_else(|| panic!("fp8 missing {name}"));
            let mut gguf_v: Option<Vec<f32>> = None;
            src.with_tensor(&name, &mut |raw| gguf_v = Some(raw.to_vec()));
            let gguf_v = gguf_v.unwrap_or_else(|| panic!("gguf missing {name}"));
            let head_dim = cfg.linear_value_head_dim as usize;
            let value_dim = nvh * head_dim;
            assert_eq!(gguf_v.len(), d * value_dim);

            let identity_cos = cosine(&gguf_v, fp8_v);
            let mut permuted = gguf_v.clone();
            for row in 0..d {
                for h in 0..nvh {
                    let (s, g) = (h / group, h % group);
                    let src_head = g * nkh + s;
                    for col_in_head in 0..head_dim {
                        let dst_col = h * head_dim + col_in_head;
                        let src_col = src_head * head_dim + col_in_head;
                        permuted[row * value_dim + dst_col] = gguf_v[row * value_dim + src_col];
                    }
                }
            }
            let permuted_cos = cosine(&permuted, fp8_v);
            println!("layer {l} linear_attn.out_proj.weight: identity cos={identity_cos:.4}  col-block-permuted cos={permuted_cos:.6}");
        }
    }
}
