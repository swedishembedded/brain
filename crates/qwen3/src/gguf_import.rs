// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF import for the dense Qwen3 decoder - the second source format for the
//! same model [`crate::import`] reads from HuggingFace safetensors.
//!
//! Two things make this worth having rather than telling users to fetch the
//! bf16 checkpoint. A quantized GGUF is a substantially smaller download for
//! the same model, so it is a bytes win as well as a host-memory one; and
//! FLUX.2's text encoder **is** a Qwen3, so `BRAIN_FLUX2_TE` gets to point at
//! a GGUF for free (see [`crate::import::shard_source`], which sniffs the
//! naming convention rather than being told which one it is looking at).
//!
//! ## Where the name map comes from
//!
//! Transcribed from **llama.cpp** at revision
//! `d7a2074112d27649303fa107eb8c94db1ee435f3`, from the two files that are
//! authoritative for it:
//!
//! - `gguf-py/gguf/constants.py` - `MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3] =
//!   "qwen3"`, the `MODEL_TENSORS[MODEL_ARCH.QWEN3]` list (exactly the 15
//!   entries below), and `TENSOR_NAMES`, which spells each entry's GGUF name.
//! - `gguf-py/gguf/tensor_mapping.py` - the HF-name → `MODEL_TENSOR` table.
//!
//! plus `conversion/qwen.py`, where `Qwen3Model(Qwen2Model)` is registered for
//! `Qwen3ForCausalLM` with `model_arch = MODEL_ARCH.QWEN3` and inherits the
//! base `modify_tensors` - i.e. for a plain (non-rerank, non-MoE) Qwen3 the
//! conversion is a pure 1:1 rename with no reshaping, splitting or permuting
//! anywhere in it.
//!
//! It is transcribed rather than inferred from tensor shapes on purpose. At
//! Qwen3-8B's shape `q_proj` is `[4096, 4096]` and, on any GQA layer, `k_proj`
//! and `v_proj` are the same `[1024, 4096]` as each other - so a map guessed
//! from shapes would swap `k` and `v` silently, and on a square-`q` model
//! could swap `q` with `o`. The `qk_swap_is_caught_by_the_parity_gate` test
//! below is that failure mode, made to fail on purpose.
//!
//! | GGUF (llama.cpp) | HF (`Qwen3ForCausalLM`) | brain |
//! |---|---|---|
//! | `token_embd.weight` | `model.embed_tokens.weight` | `tok.weight` |
//! | `output_norm.weight` | `model.norm.weight` | `norm.weight` |
//! | `output.weight` | `lm_head.weight` | `lm_head.weight` (absent when tied) |
//! | `blk.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` | `blocks.N.ln1.weight` |
//! | `blk.N.attn_q.weight` | `…self_attn.q_proj.weight` | `blocks.N.attn.wq.weight` |
//! | `blk.N.attn_k.weight` | `…self_attn.k_proj.weight` | `blocks.N.attn.wk.weight` |
//! | `blk.N.attn_v.weight` | `…self_attn.v_proj.weight` | `blocks.N.attn.wv.weight` |
//! | `blk.N.attn_output.weight` | `…self_attn.o_proj.weight` | `blocks.N.attn.wo.weight` |
//! | `blk.N.attn_q_norm.weight` | `…self_attn.q_norm.weight` | `blocks.N.attn.q_norm.weight` |
//! | `blk.N.attn_k_norm.weight` | `…self_attn.k_norm.weight` | `blocks.N.attn.k_norm.weight` |
//! | `blk.N.ffn_norm.weight` | `…post_attention_layernorm.weight` | `blocks.N.ln2.weight` |
//! | `blk.N.ffn_gate.weight` | `…mlp.gate_proj.weight` | `blocks.N.mlp.gate.weight` |
//! | `blk.N.ffn_up.weight` | `…mlp.up_proj.weight` | `blocks.N.mlp.up.weight` |
//! | `blk.N.ffn_down.weight` | `…mlp.down_proj.weight` | `blocks.N.mlp.down.weight` |
//! | `rope_freqs.weight` | (rope-scaling factors) | dropped, by name |
//!
//! `blk.N.attn_{q,k,v}.bias` are accepted too. Qwen3 has no attention bias, so
//! a Qwen3 GGUF never carries them - but [`crate::QwenConfig`] describes Qwen2
//! as well (`attn_bias: true`), the GGUF spelling is the same, and a `qwen2`
//! file reaching this map should map or be refused, never be dropped quietly.
//!
//! No tensor is transposed and none is split: brain's `matmul` is
//! `out = x @ Wᵀ` with `W:[out,in]` row-major, which is what both HF
//! `nn.Linear.weight` and llama.cpp's dequantized row-major output already
//! are. The conversion is a rename plus a dequant, exactly as the HF route is
//! a rename plus a bf16→f32 widen.

use checkpoint::gguf::MmapGguf;
use checkpoint::st::ModelCard;
use gguf::import::{self, ImportStats, Leaf, Mapped};
use gguf::leaf::Role;
use gguf::ArchKv;

use crate::config::QwenConfig;

/// llama.cpp's `general.architecture` value for the dense Qwen3 family
/// (`MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3]`).
pub const GGUF_ARCHITECTURE: &str = "qwen3";

/// Not imported, by reason - [`Mapped::Dropped`]'s payload, counted and
/// printed by the shared driver so a drop is always on the record.
const DROP_TIED_HEAD: &str = "output.weight on a tied-embedding checkpoint (the head reuses tok.weight)";
const DROP_ROPE_FREQS: &str = "rope_freqs.weight (RoPE scaling factors, recomputed from rope_theta)";

/// Map one GGUF tensor name to its brain parameter name.
///
/// `None` means "deliberately not a brain parameter": the two named drops
/// above, and anything this map does not recognize. Callers that need the
/// distinction - the importer, which must refuse an unrecognized leaf rather
/// than lose a projection - use [`classify`] instead.
///
/// Shared with [`crate::import::hf_shard_source`]'s GGUF arm, so the streaming
/// text-encoder path and the whole-checkpoint importer cannot disagree about
/// which tensor is which.
pub fn gguf_to_brain(name: &str, tie: bool) -> Option<String> {
    // n_layers is deliberately u32::MAX here: this is the *name* map, and it
    // has no opinion about depth. A block index beyond the config's depth
    // still maps, and is then rejected by the caller's coverage check as a
    // brain parameter outside `param_list()` - which is how a 36-layer
    // checkpoint against a 28-layer config is caught.
    match import::split_name(name, u32::MAX) {
        Leaf::TokenEmbd => Some("tok.weight".to_string()),
        Leaf::OutputNorm => Some("norm.weight".to_string()),
        Leaf::Output => (!tie).then(|| "lm_head.weight".to_string()),
        Leaf::Block { layer, leaf } => {
            // The leaf VOCABULARY (which spellings exist and what they mean)
            // is `gguf::leaf`'s, shared with `qwen35moe`/`qwen35`; only this
            // model's own brain-parameter suffix is qwen3-specific.
            let leaf = match gguf::leaf::role(leaf)? {
                Role::AttnNorm => "ln1.weight",
                Role::FfnNorm => "ln2.weight",
                Role::AttnQ => "attn.wq.weight",
                Role::AttnK => "attn.wk.weight",
                Role::AttnV => "attn.wv.weight",
                Role::AttnOutput => "attn.wo.weight",
                Role::AttnQNorm => "attn.q_norm.weight",
                Role::AttnKNorm => "attn.k_norm.weight",
                Role::AttnQBias => "attn.wq.bias",
                Role::AttnKBias => "attn.wk.bias",
                Role::AttnVBias => "attn.wv.bias",
                Role::FfnGate => "mlp.gate.weight",
                Role::FfnUp => "mlp.up.weight",
                Role::FfnDown => "mlp.down.weight",
                _ => return None,
            };
            Some(format!("blocks.{layer}.{leaf}"))
        }
        Leaf::PastDepth { .. } | Leaf::Other => None,
    }
}

/// [`gguf_to_brain`] with the importer's strictness: an unrecognized tensor is
/// an **error**, not a silent skip, and each drop states its reason.
///
/// That asymmetry is the point. A converter that renames a leaf must break the
/// import loudly rather than quietly produce a checkpoint missing a
/// projection, and "we didn't recognize it" is exactly the case where a
/// missing projection would otherwise look like a clean run.
fn classify(name: &str, tie: bool) -> Result<Mapped, String> {
    if let Some(brain) = gguf_to_brain(name, tie) {
        return Ok(Mapped::Simple(brain));
    }
    match import::split_name(name, u32::MAX) {
        Leaf::Output => Ok(Mapped::Dropped(DROP_TIED_HEAD)),
        _ if name == "rope_freqs.weight" => Ok(Mapped::Dropped(DROP_ROPE_FREQS)),
        _ => Err(format!("unrecognized tensor {name:?} - the qwen3 name map has no entry for it")),
    }
}

/// Derive a [`QwenConfig`] from a GGUF's KV metadata.
///
/// Every field comes from llama.cpp's standardized `{arch}.…` keys except
/// `vocab`, which is read from `token_embd.weight`'s own shape: the tensor is
/// ground truth, and a `vocab_size` KV that disagrees with the embedding table
/// would produce a config that cannot load its own checkpoint.
///
/// `block_size` is 2048, matching [`crate::import::config_from_hf`] - it sizes
/// buffers, and the trained RoPE extent (`context_length`, carried through as
/// `max_position_embeddings`) would size them absurdly.
pub fn config_from_gguf(mg: &MmapGguf) -> Result<QwenConfig, String> {
    let kv = ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;
    config_from_kv(&kv, mg)
}

/// [`config_from_gguf`]'s core, against an already-scoped KV view.
///
/// Split out because a Qwen3 decoder is not always the whole checkpoint: a
/// Qwen3-VL GGUF carries the same dense decoder under its OWN architecture
/// prefix (`qwen3vl.*`), and reading it there must not mean a second
/// transcription of llama.cpp's key names that can drift from this one.
pub fn config_from_kv(kv: &ArchKv, mg: &MmapGguf) -> Result<QwenConfig, String> {
    let block_size = 2048;
    let vocab = mg
        .shape("token_embd.weight")
        .and_then(|s| s.first().copied())
        .ok_or("qwen3: missing token_embd.weight (cannot determine vocab)")? as u32;
    let head_dim = kv.req_u32("attention.key_length")?;
    let value_len = kv.u32_or("attention.value_length", head_dim);
    if value_len != head_dim {
        return Err(format!("qwen3: attention.key_length {head_dim} != value_length {value_len} (asymmetric head_dim is unsupported)"));
    }
    Ok(QwenConfig {
        vocab,
        block_size,
        n_layers: kv.req_u32("block_count")?,
        d_model: kv.req_u32("embedding_length")?,
        n_heads: kv.req_u32("attention.head_count")?,
        n_kv_heads: kv.req_u32("attention.head_count_kv")?,
        head_dim,
        d_ff: kv.req_u32("feed_forward_length")?,
        rope_theta: kv.f32_or("rope.freq_base", 1.0e6),
        rms_eps: kv.f32_or("attention.layer_norm_rms_epsilon", 1e-6),
        max_position_embeddings: kv.u32_or("context_length", block_size),
        // A GGUF states tying by OMITTING `output.weight` - there is no
        // `tie_word_embeddings` key. That is also what makes the tied drop in
        // `classify` unreachable for a real file: a tied checkpoint has no
        // head tensor to drop. It stays because a converter is free to ship
        // one anyway (HF Qwen3 checkpoints sometimes do), and dropping it with
        // a stated reason beats failing on it.
        tie_embeddings: !mg.names().iter().any(|n| n == "output.weight"),
        qk_norm: true,
        attn_bias: mg.names().iter().any(|n| n.ends_with("attn_q.bias")),
        lora: None,
    }
    .with_defaults())
}

/// Import a Qwen3 GGUF into a brain-native safetensors checkpoint.
///
/// The streaming loop, the one-tensor-at-a-time dequant and the two-way
/// coverage check are `gguf::import`'s, shared with every other GGUF-sourced
/// model in the tree. What is Qwen3-specific, and stays here, is
/// [`config_from_gguf`]'s manifest and [`classify`]'s name map.
pub fn import_gguf(gguf_path: &str, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let mg = MmapGguf::open(gguf_path)?;
    import_mmap(&mg, out_path, id_override)
}

/// [`import_gguf`] over an ALREADY-OPEN checkpoint - the shape the generic
/// architecture-dispatch registry needs, since it must read
/// `general.architecture` before it can know which importer to call.
pub fn import_mmap(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let cfg = config_from_gguf(mg)?;
    let params = cfg.param_list();

    let mut card = ModelCard::new(id_override.unwrap_or("qwen3"), "qwen");
    card.context_length = Some(cfg.block_size as u64);
    card.param_count = Some(params.iter().map(|(_, n)| *n as u64).sum());

    let tie = cfg.tie_embeddings;
    import::to_st(mg, &params, &|n| classify(n, tie), out_path, &cfg.to_json(), Some(&card), "qwen3")
}

/// Test fixtures for this importer, shared across crates.
///
/// `pub` (not `#[cfg(test)]`) so `brain-cli`'s GGUF-import-registry tests can
/// drive a REAL conversion through the generic architecture dispatch without a
/// second, drifting copy of this checkpoint builder. Not part of the model's
/// runtime surface.
#[doc(hidden)]
pub mod testing {
    use super::*;
    use checkpoint::gguf::GgufValue;
    use checkpoint::gguf_write::{write, TensorOut};

    /// The tiny shape both synthetic checkpoints below are built at. GQA (2 q
    /// heads over 1 kv head) and a decoupled `head_dim` are deliberate: with
    /// `n_heads == n_kv_heads` a q/k swap is a shape-compatible no-op and no
    /// coverage check could see it.
    pub const N_LAYERS: usize = 2;
    pub const VOCAB: usize = 5;
    pub const D_MODEL: usize = 6;
    pub const N_HEADS: usize = 2;
    pub const N_KV_HEADS: usize = 1;
    pub const HEAD_DIM: usize = 4;
    pub const D_FF: usize = 8;

    /// A distinct, exactly-representable value per (tensor, element), so any
    /// two tensors that got swapped show up as a value mismatch rather than
    /// passing a shape check.
    fn seq(base: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    /// The one description of the fixture checkpoint's contents, in HF names.
    /// Both writers below render THIS list, so the safetensors and the GGUF
    /// are the same logical checkpoint by construction rather than by two
    /// hand-kept copies that could drift.
    fn contents(tied: bool) -> Vec<(&'static str, String, usize, f32)> {
        let (hq, hkv) = (N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM);
        let mut out: Vec<(&'static str, String, usize, f32)> = vec![
            ("token_embd.weight", "model.embed_tokens.weight".into(), VOCAB * D_MODEL, 1_000_000.0),
            ("output_norm.weight", "model.norm.weight".into(), D_MODEL, 2_000_000.0),
        ];
        if !tied {
            out.push(("output.weight", "lm_head.weight".into(), VOCAB * D_MODEL, 3_000_000.0));
        }
        for l in 0..N_LAYERS {
            let b = 100_000.0 * (l + 1) as f32;
            let hf = |s: &str| format!("model.layers.{l}.{s}");
            // The GGUF name is `blk.{l}.{leaf}`; `leak` gives the &'static str
            // the table wants, and this runs once per test.
            let g = |leaf: &str| -> &'static str { format!("blk.{l}.{leaf}").leak() };
            out.extend([
                (g("attn_norm.weight"), hf("input_layernorm.weight"), D_MODEL, b + 10.0),
                (g("attn_q.weight"), hf("self_attn.q_proj.weight"), hq * D_MODEL, b + 1000.0),
                (g("attn_k.weight"), hf("self_attn.k_proj.weight"), hkv * D_MODEL, b + 2000.0),
                (g("attn_v.weight"), hf("self_attn.v_proj.weight"), hkv * D_MODEL, b + 3000.0),
                (g("attn_q_norm.weight"), hf("self_attn.q_norm.weight"), HEAD_DIM, b + 4000.0),
                (g("attn_k_norm.weight"), hf("self_attn.k_norm.weight"), HEAD_DIM, b + 5000.0),
                (g("attn_output.weight"), hf("self_attn.o_proj.weight"), D_MODEL * hq, b + 6000.0),
                (g("ffn_norm.weight"), hf("post_attention_layernorm.weight"), D_MODEL, b + 7000.0),
                (g("ffn_gate.weight"), hf("mlp.gate_proj.weight"), D_FF * D_MODEL, b + 8000.0),
                (g("ffn_up.weight"), hf("mlp.up_proj.weight"), D_FF * D_MODEL, b + 9000.0),
                (g("ffn_down.weight"), hf("mlp.down_proj.weight"), D_MODEL * D_FF, b + 10000.0),
            ]);
        }
        out
    }

    /// Write the fixture as a **GGUF**, every tensor F32 (ggml type 0) so the
    /// comparison against the safetensors route is about the NAME MAP and
    /// nothing else - a quantized fixture would fold a lossy dequant into the
    /// same assertion and make bit-identity unavailable for no gain.
    ///
    /// Ships a `rope_freqs.weight` too: a real llama.cpp Qwen3 conversion may,
    /// and the drop must be a counted decision rather than a silent skip.
    pub fn write_synthetic_gguf(path: &str, tied: bool) {
        let tensors: Vec<TensorOut> = contents(tied)
            .into_iter()
            .map(|(gname, _, numel, base)| TensorOut {
                name: gname.to_string(),
                shape: vec![numel], // flat: only the element count is load-bearing here
                ty: 0,
                data: seq(base, numel).iter().flat_map(|v| v.to_le_bytes()).collect(),
            })
            .chain(std::iter::once(TensorOut {
                name: "rope_freqs.weight".to_string(),
                shape: vec![HEAD_DIM / 2],
                ty: 0,
                data: seq(7.0, HEAD_DIM / 2).iter().flat_map(|v| v.to_le_bytes()).collect(),
            }))
            .collect();

        // `token_embd.weight` is written flat above, but `config_from_gguf`
        // reads `vocab` off its leading dim - so give that one its real 2-D
        // shape. Every other tensor's rank is irrelevant to this importer.
        let tensors: Vec<TensorOut> = tensors
            .into_iter()
            .map(|mut t| {
                if t.name == "token_embd.weight" || t.name == "output.weight" {
                    t.shape = vec![VOCAB, D_MODEL];
                }
                t
            })
            .collect();

        let kv = |k: &str, v: GgufValue| (k.to_string(), v);
        let kvs = vec![
            kv("general.architecture", GgufValue::String(GGUF_ARCHITECTURE.to_string())),
            kv("qwen3.block_count", GgufValue::U32(N_LAYERS as u32)),
            kv("qwen3.embedding_length", GgufValue::U32(D_MODEL as u32)),
            kv("qwen3.feed_forward_length", GgufValue::U32(D_FF as u32)),
            kv("qwen3.attention.head_count", GgufValue::U32(N_HEADS as u32)),
            kv("qwen3.attention.head_count_kv", GgufValue::U32(N_KV_HEADS as u32)),
            kv("qwen3.attention.key_length", GgufValue::U32(HEAD_DIM as u32)),
            kv("qwen3.attention.value_length", GgufValue::U32(HEAD_DIM as u32)),
            kv("qwen3.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6)),
            kv("qwen3.rope.freq_base", GgufValue::F32(1_000_000.0)),
            kv("qwen3.context_length", GgufValue::U32(40960)),
        ];
        write(path, &kvs, &tensors, 32).unwrap();
    }

    /// Write the SAME fixture as an HF checkpoint directory (`config.json` +
    /// `model.safetensors`), so the two import routes can be compared on
    /// identical logical content.
    pub fn write_synthetic_hf_dir(dir: &std::path::Path, tied: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let json = format!(
            r#"{{"vocab_size":{VOCAB},"hidden_size":{D_MODEL},"num_hidden_layers":{N_LAYERS},
            "num_attention_heads":{N_HEADS},"num_key_value_heads":{N_KV_HEADS},"head_dim":{HEAD_DIM},
            "intermediate_size":{D_FF},"rope_theta":1000000,"rms_norm_eps":1e-6,
            "max_position_embeddings":40960,"tie_word_embeddings":{tied}}}"#
        );
        std::fs::write(dir.join("config.json"), json).unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = contents(tied)
            .into_iter()
            .map(|(_, hname, numel, base)| (hname, vec![numel as u64], seq(base, numel)))
            .collect();
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{write_synthetic_gguf, write_synthetic_hf_dir};
    use super::*;
    use std::collections::HashMap;

    fn scratch(tag: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("brain-qwen3-gguf-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Every brain parameter of a checkpoint, read back from an imported
    /// safetensors file.
    fn read_back(path: &str, cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
        let r = checkpoint::weightio::WeightReader::open(path).unwrap();
        cfg.param_list().into_iter().map(|(n, _)| {
            let v = r.tensor(&n).unwrap_or_else(|| panic!("missing {n}"));
            (n, v)
        }).collect()
    }

    /// **The headline gate.** The GGUF route and the safetensors route must
    /// produce the *same brain parameters, bit for bit*, from the same logical
    /// checkpoint - not merely both "work".
    ///
    /// Bit-identity is available here because the fixture GGUF is F32: both
    /// routes copy the same f32 values under different source names, so
    /// anything but equality means the two disagree about which tensor is
    /// which. `assert_eq!` on the values, never a tolerance - a tolerance
    /// would pass a `k`/`v` swap on any layer whose two projections happened
    /// to be similar.
    #[test]
    fn the_gguf_route_and_the_safetensors_route_agree_bit_for_bit() {
        for tied in [false, true] {
            let dir = scratch(if tied { "parity-tied" } else { "parity-untied" });
            let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
            let hf = dir.join("hf");
            write_synthetic_gguf(&gguf, tied);
            write_synthetic_hf_dir(&hf, tied);

            // Both importers derive their own config from their own source;
            // they must agree about the model before they can agree about its
            // weights.
            let g_cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
            let h_cfg = crate::import::config_from_hf(&std::fs::read_to_string(hf.join("config.json")).unwrap()).unwrap();
            assert_eq!(g_cfg.to_json(), h_cfg.to_json(), "the two routes derive different configs (tied={tied})");

            let g_out = dir.join("from-gguf.safetensors").to_string_lossy().into_owned();
            let h_out = dir.join("from-hf.safetensors").to_string_lossy().into_owned();
            import_gguf(&gguf, &g_out, Some("test/qwen3-tiny")).expect("gguf import");
            crate::import::import(hf.to_str().unwrap(), &h_out).expect("hf import");

            let from_gguf = read_back(&g_out, &g_cfg);
            let from_hf = read_back(&h_out, &h_cfg);
            assert_eq!(from_gguf.len(), h_cfg.param_list().len());
            for (name, want) in &from_hf {
                assert_eq!(&from_gguf[name], want, "{name}: the two routes disagree (tied={tied})");
            }
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// The failure mode the map is transcribed rather than inferred to avoid,
    /// demonstrated: perturb the name map and the bit-for-bit parity gate
    /// above must go red. This runs the same comparison that gate does against
    /// deliberately swapped maps and asserts the result DIFFERS from the
    /// safetensors route.
    ///
    /// Both swaps matter, for different reasons:
    ///
    /// - **q ↔ k** is caught by the element count alone, because Qwen3 is GQA
    ///   (`q_dim != kv_dim`). Real, but the easy case.
    /// - **k ↔ v** is the one that motivates this whole approach: `k_proj` and
    ///   `v_proj` have *identical* shapes on every layer of every GQA model,
    ///   so no coverage check, no shape check and no dry run can see it. The
    ///   import succeeds, the model runs, and the output is quietly wrong.
    ///   Only comparing VALUES against the other route catches it - which is
    ///   why the gate asserts on bytes rather than on a tolerance.
    ///
    /// A gate nobody has watched fail is a gate nobody knows is connected.
    #[test]
    fn a_swapped_name_map_is_caught_by_the_parity_gate() {
        let dir = scratch("swap");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        let hf = dir.join("hf");
        write_synthetic_gguf(&gguf, false);
        write_synthetic_hf_dir(&hf, false);

        let cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
        let h_out = dir.join("from-hf.safetensors").to_string_lossy().into_owned();
        crate::import::import(hf.to_str().unwrap(), &h_out).unwrap();
        let from_hf = read_back(&h_out, &cfg);
        let mg = MmapGguf::open(&gguf).unwrap();

        let swap = |a: &'static str, b: &'static str| {
            move |n: &str| -> Result<Mapped, String> {
                let n = if n.ends_with(a) {
                    n.replace(a, b)
                } else if n.ends_with(b) {
                    n.replace(b, a)
                } else {
                    n.to_string()
                };
                classify(&n, false)
            }
        };

        // q ↔ k: shapes disagree under GQA, so the import itself refuses.
        let err = gguf::import::to_map(&mg, &cfg.param_list(), &swap("attn_q.weight", "attn_k.weight"), "qwen3-mutant")
            .expect_err("a q/k swap must not import cleanly at a GQA shape");
        assert!(err.contains("element count"), "{err}");

        // k ↔ v: shapes AGREE, so it imports cleanly - and only the value
        // comparison can tell. This is the gate earning its keep.
        let mutant = gguf::import::to_map(&mg, &cfg.param_list(), &swap("attn_k.weight", "attn_v.weight"), "qwen3-mutant")
            .expect("a k/v swap is shape-compatible and WILL import cleanly - that is the whole problem");
        assert_eq!(mutant.len(), cfg.param_list().len());
        let differing: Vec<String> = cfg
            .param_list()
            .into_iter()
            .filter(|(n, _)| mutant[n] != from_hf[n])
            .map(|(n, _)| n)
            .collect();
        assert!(
            !differing.is_empty(),
            "a k/v swap must change the imported weights - if it does not, the parity gate proves nothing"
        );
        // ...and it must be exactly the k and v projections that moved.
        assert!(differing.iter().all(|n| n.ends_with("attn.wk.weight") || n.ends_with("attn.wv.weight")), "{differing:?}");
        assert_eq!(differing.len(), 2 * super::testing::N_LAYERS, "every layer's k and v must have moved: {differing:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two-way coverage, first direction: an unrecognized source tensor is an
    /// error, not a silent skip. A converter that renames a leaf must break
    /// the import loudly rather than write a checkpoint missing a projection.
    #[test]
    fn an_unrecognized_tensor_is_refused_by_name() {
        let err = classify("blk.0.attn_wibble.weight", false).unwrap_err();
        assert!(err.contains("attn_wibble"), "{err}");
        // ...while the two real drops are decisions with stated reasons.
        assert!(matches!(classify("rope_freqs.weight", false), Ok(Mapped::Dropped(_))));
        assert!(matches!(classify("output.weight", true), Ok(Mapped::Dropped(_))));
        assert!(matches!(classify("output.weight", false), Ok(Mapped::Simple(_))));
    }

    /// Two-way coverage, second direction, and the drop accounting: every
    /// planned brain parameter is written exactly once, and the one dropped
    /// source tensor is COUNTED rather than lost.
    #[test]
    fn import_covers_every_planned_tensor_and_counts_the_rope_freqs_drop() {
        let dir = scratch("coverage");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf(&gguf, false);
        let out = dir.join("out.safetensors").to_string_lossy().into_owned();

        let stats = import_gguf(&gguf, &out, None).unwrap();
        let cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
        assert_eq!(stats.written, cfg.param_list().len());
        assert_eq!(stats.dropped.get(DROP_ROPE_FREQS), Some(&1), "the dropped tensor must be on the record: {stats}");
        assert_eq!(stats.source_tensors, stats.written + 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint deeper than the config claims is the classic
    /// wrong-checkpoint mistake, and must be refused rather than silently
    /// truncated. The name map has no depth opinion (`u32::MAX`), so this is
    /// the coverage check doing its job.
    #[test]
    fn a_checkpoint_deeper_than_the_config_is_refused() {
        let dir = scratch("deeper");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf(&gguf, false);
        let mg = MmapGguf::open(&gguf).unwrap();
        let mut cfg = config_from_gguf(&mg).unwrap();
        cfg.n_layers = 1; // the file has 2

        let err = gguf::import::dry_run(&mg, &cfg.param_list(), &|n| classify(n, false), "qwen3").unwrap_err();
        assert!(err.contains("blocks.1."), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_from_gguf_matches_the_synthetic_header() {
        let dir = scratch("cfg");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf(&gguf, false);
        let cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
        assert_eq!(cfg.vocab, 5);
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.d_model, 6);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.n_kv_heads, 1);
        assert_eq!(cfg.head_dim, 4);
        assert_eq!(cfg.d_ff, 8);
        assert_eq!(cfg.max_position_embeddings, 40960);
        assert_eq!(cfg.block_size, 2048, "buffers are sized by block_size, never by the trained RoPE extent");
        assert!(!cfg.tie_embeddings, "output.weight is present -> untied");
        assert!(cfg.qk_norm);
        assert!(!cfg.attn_bias);

        // ...and tying is stated by OMITTING output.weight.
        let tied_path = dir.join("tied.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf(&tied_path, true);
        assert!(config_from_gguf(&MmapGguf::open(&tied_path).unwrap()).unwrap().tie_embeddings);

        // A GGUF that is not a qwen3 is refused by name, not silently defaulted.
        std::fs::remove_dir_all(&dir).ok();
    }
}
