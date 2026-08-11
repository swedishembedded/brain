// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A resident int8 weight store for the Thinker's routed MoE experts — the
//! dominant memory cost of the whole checkpoint (see [`expert_bytes`]'s doc:
//! ~27.6 GiB of the ~36 GB total at the real Thinker shape), and the reason
//! a single 24 GB P40 cannot hold the Thinker at all without quantization
//! and cannot hold it ALONE even quantized, motivating the two-card split
//! this module's [`crate::int8_thinker_resident::Int8ThinkerResident`] (see
//! that module) places across.
//!
//! **Scope, deliberately narrower than "the whole Thinker in int8"**: only
//! the routed-expert linears (`gate`/`up`/`down`, 128 experts x 3 x 48
//! layers) are stored here. Attention (`wq`/`wk`/`wv`/`wo`), the router,
//! norms, `embed_tokens`, and `lm_head` stay fp32, read and held resident
//! the ordinary way. `omni::import::should_quantize` DID quantize those too
//! at import time (every 2-D weight with `k % 4 == 0`, not just MoE
//! experts) — so this is a real, deliberate scope cut, not a checkpoint
//! limitation: the non-expert weights total only ~5-6 GiB combined (attention
//! ~3 GiB + embed/lm_head ~2.4 GiB across 48 layers + vocab 152064, computed
//! from `MoeTextConfig::thinker_defaults`'s real dims), small enough next to
//! the ~27.6 GiB of expert weights that quantizing them too would buy little
//! additional headroom for real added risk (a second int8 dispatch path
//! through `layer_fwd`/`layer_decode_step`'s attention projections, on top
//! of the MoE one this module + [`crate::thinker`]'s int8 branch already
//! add). Extending to full-model int8 is real, separate follow-up work if
//! the ~5-6 GiB saved ever turns out to matter.

use std::collections::HashMap;

use checkpoint::weightio::WeightReader;
use gpu_core::{DeviceBuffer, Gpu};

use crate::config::MoeTextConfig;

/// One int8-quantized expert linear: packed weight + per-channel scale, read
/// straight from the checkpoint's ALREADY-quantized `U32`/`F32` `.scale`
/// pair (`omni::import` quantized at IMPORT time — unlike `qwen3::q8::Q8`,
/// which quantizes an fp32 SOURCE checkpoint on load, there is no
/// `model::int8::quantize_weight` call here at all, just a read + upload).
pub struct ExpertLin8 {
    pub packed: DeviceBuffer,
    pub scale: DeviceBuffer,
}

/// One layer's routed experts, indexed `0..n_experts`.
pub struct ThinkerLayerExperts8 {
    pub experts: Vec<(ExpertLin8, ExpertLin8, ExpertLin8)>,
}

impl ThinkerLayerExperts8 {
    /// Expert `e`'s `(gate, up, down)` weights, as [`model::moe::Lin8`]
    /// views — the exact shape [`model::moe::expert_fwd_i8`] takes. This is
    /// the per-LAYER accessor `crate::thinker::moe_sublayer`'s `int8_experts`
    /// parameter calls (it already knows which layer it is; only the expert
    /// index varies within one call) — see [`ThinkerInt8Store::lin8_at`] for
    /// the multi-layer (by absolute layer index) sibling.
    pub fn lin8_at(&self, e: usize) -> (model::moe::Lin8<'_>, model::moe::Lin8<'_>, model::moe::Lin8<'_>) {
        let (gate, up, down) = &self.experts[e];
        (
            model::moe::Lin8 { wq: &gate.packed, sw: &gate.scale },
            model::moe::Lin8 { wq: &up.packed, sw: &up.scale },
            model::moe::Lin8 { wq: &down.packed, sw: &down.scale },
        )
    }
}

/// A resident (GPU-uploaded) subset of the Thinker's routed-expert weights —
/// only the ABSOLUTE layer indices named at construction (a caller building
/// a two-card shard builds two of these, one per device, with disjoint
/// layer ranges — see [`Self::build`]'s doc).
pub struct ThinkerInt8Store {
    pub layers: HashMap<usize, ThinkerLayerExperts8>,
}

/// The brain-native tensor name for expert `e`'s `leaf` weight in layer `l`
/// (`gate.weight` | `up.weight` | `down.weight`) — the exact convention
/// `omni::import::map_thinker` writes (`thinker.blocks.{l}.mlp.experts.{e}.
/// {leaf}`), confirmed against that module's own test
/// (`map_thinker("thinker.model.layers.0.mlp.experts.127.down_proj.weight")
/// == "thinker.blocks.0.mlp.experts.127.down.weight"`).
fn expert_name(l: usize, e: usize, leaf: &str) -> String {
    format!("thinker.blocks.{l}.mlp.experts.{e}.{leaf}.weight")
}

impl ThinkerInt8Store {
    /// Stream + upload `owned_layers`' routed-expert weights from `reader`
    /// onto `gpu`. Panics loudly (never silently skips) if an expected
    /// tensor is missing or has the wrong dtype — `WeightReader::tensor_u32`/
    /// `tensor` already panic on a dtype mismatch (this session's earlier
    /// `checkpoint` work), so a checkpoint that was not actually quantized
    /// the way `omni::import` promises fails HERE, at load time, not with a
    /// silently-wrong forward pass later.
    pub fn build(gpu: &Gpu, reader: &WeightReader, owned_layers: impl Iterator<Item = usize>, cfg: &MoeTextConfig) -> ThinkerInt8Store {
        let mut layers = HashMap::new();
        for l in owned_layers {
            let mut experts = Vec::with_capacity(cfg.n_experts as usize);
            for e in 0..cfg.n_experts as usize {
                let gate = load_lin8(gpu, reader, &expert_name(l, e, "gate"));
                let up = load_lin8(gpu, reader, &expert_name(l, e, "up"));
                let down = load_lin8(gpu, reader, &expert_name(l, e, "down"));
                experts.push((gate, up, down));
            }
            layers.insert(l, ThinkerLayerExperts8 { experts });
        }
        ThinkerInt8Store { layers }
    }

    /// Layer `l`'s expert store, or panics if this shard does not own it (a
    /// caller bug: dispatching a layer this shard does not hold, never a
    /// value worth defaulting past). Call [`ThinkerLayerExperts8::lin8_at`]
    /// on the result for one expert's weights.
    pub fn layer(&self, l: usize) -> &ThinkerLayerExperts8 {
        self.layers.get(&l).unwrap_or_else(|| panic!("ThinkerInt8Store: layer {l} not owned by this shard"))
    }

    /// Expert `e`'s `(gate, up, down)` weights in layer `l` — shorthand for
    /// `self.layer(l).lin8_at(e)`.
    pub fn lin8_at(&self, l: usize, e: usize) -> (model::moe::Lin8<'_>, model::moe::Lin8<'_>, model::moe::Lin8<'_>) {
        self.layer(l).lin8_at(e)
    }
}

/// Read one packed int8 weight + its `.scale` sibling and upload both,
/// UNQUANTIZED (no dequantize-then-f32 round trip) -- the primitive every
/// resident int8 weight (expert or otherwise) is built from. `pub` so both
/// in-crate non-expert callers (e.g. `int8_thinker_resident`'s `lm_head_w`
/// loader) and integration tests that need an independently-assembled int8
/// reference can reuse it without re-deriving this read+upload sequence.
pub fn load_lin8(gpu: &Gpu, reader: &WeightReader, name: &str) -> ExpertLin8 {
    let packed = reader.tensor_u32(name).unwrap_or_else(|| panic!("ThinkerInt8Store: missing packed weight '{name}'"));
    let scale_name = format!("{name}.scale");
    let scale = reader.tensor(&scale_name).unwrap_or_else(|| panic!("ThinkerInt8Store: missing scale sibling '{scale_name}'"));
    let pb = gpu.storage(packed.len() as u64);
    gpu.write(&pb, &packed);
    gpu.poll_wait(); // reclaim the write's staging buffer before the next weight (qwen3::q8's own discipline)
    let sb = gpu.storage_init(&scale_name, &scale);
    ExpertLin8 { packed: pb, scale: sb }
}

/// One layer's routed experts PLUS the always-active shared expert's own
/// quantized weights — Talker's one real architectural difference from
/// Thinker (`crate::talker`'s module doc). `shared_expert_gate` (the
/// `[1, hidden]` sigmoid-gate projection) is loaded fp32-resident via
/// [`crate::int8_thinker_resident::load_mat`], which dequantizes on read if
/// the checkpoint happened to quantize it (`omni::import::should_quantize`'s
/// literal rank-2/k%4==0 rule does not special-case a singleton output
/// channel) — matching `model::moe::shared_expert_fwd_i8`'s own "not worth
/// quantizing a rank-1 output" scope cut regardless of how the checkpoint
/// stored it.
pub struct TalkerLayerExperts8 {
    pub experts: Vec<(ExpertLin8, ExpertLin8, ExpertLin8)>,
    pub shared_expert: (ExpertLin8, ExpertLin8, ExpertLin8),
    pub shared_expert_gate: DeviceBuffer,
}

impl TalkerLayerExperts8 {
    /// Expert `e`'s `(gate, up, down)` weights — same contract as
    /// [`ThinkerLayerExperts8::lin8_at`].
    pub fn lin8_at(&self, e: usize) -> (model::moe::Lin8<'_>, model::moe::Lin8<'_>, model::moe::Lin8<'_>) {
        let (gate, up, down) = &self.experts[e];
        (
            model::moe::Lin8 { wq: &gate.packed, sw: &gate.scale },
            model::moe::Lin8 { wq: &up.packed, sw: &up.scale },
            model::moe::Lin8 { wq: &down.packed, sw: &down.scale },
        )
    }
    /// The shared expert's own `(gate, up, down)` weights as [`model::moe::
    /// Lin8`] views — the exact shape [`model::moe::shared_expert_fwd_i8`] takes.
    pub fn shared_lin8(&self) -> (model::moe::Lin8<'_>, model::moe::Lin8<'_>, model::moe::Lin8<'_>) {
        let (gate, up, down) = &self.shared_expert;
        (
            model::moe::Lin8 { wq: &gate.packed, sw: &gate.scale },
            model::moe::Lin8 { wq: &up.packed, sw: &up.scale },
            model::moe::Lin8 { wq: &down.packed, sw: &down.scale },
        )
    }
}

/// A resident (GPU-uploaded) subset of the Talker's routed+shared expert
/// weights — same per-device layer-range shape as [`ThinkerInt8Store`].
pub struct TalkerInt8Store {
    pub layers: HashMap<usize, TalkerLayerExperts8>,
}

/// The brain-native tensor name for routed expert `e`'s `leaf` weight in
/// Talker layer `l` — `omni::import::map_talker`'s convention
/// (`talker.blocks.{l}.mlp.experts.{e}.{leaf}.weight`), confirmed against
/// that module's own test (`map_talker("talker.model.layers.19.mlp.experts.
/// 5.up_proj.weight") == "talker.blocks.19.mlp.experts.5.up.weight"`).
fn talker_expert_name(l: usize, e: usize, leaf: &str) -> String {
    format!("talker.blocks.{l}.mlp.experts.{e}.{leaf}.weight")
}

/// The brain-native tensor name for the shared expert's `leaf` weight in
/// Talker layer `l` (`talker.blocks.{l}.mlp.shared_expert.{leaf}.weight`) —
/// `map_talker`'s convention, confirmed against its own test
/// (`map_talker("talker.model.layers.0.mlp.shared_expert.down_proj.weight")
/// == "talker.blocks.0.mlp.shared_expert.down.weight"`).
fn talker_shared_expert_name(l: usize, leaf: &str) -> String {
    format!("talker.blocks.{l}.mlp.shared_expert.{leaf}.weight")
}

impl TalkerInt8Store {
    /// [`ThinkerInt8Store::build`]'s Talker twin: routed experts PLUS the
    /// shared expert's three linears, plus its gate (dequantized on read if
    /// needed — see [`TalkerLayerExperts8`]'s doc).
    pub fn build(gpu: &Gpu, reader: &WeightReader, owned_layers: impl Iterator<Item = usize>, cfg: &MoeTextConfig) -> TalkerInt8Store {
        let mut layers = HashMap::new();
        for l in owned_layers {
            let mut experts = Vec::with_capacity(cfg.n_experts as usize);
            for e in 0..cfg.n_experts as usize {
                let gate = load_lin8(gpu, reader, &talker_expert_name(l, e, "gate"));
                let up = load_lin8(gpu, reader, &talker_expert_name(l, e, "up"));
                let down = load_lin8(gpu, reader, &talker_expert_name(l, e, "down"));
                experts.push((gate, up, down));
            }
            let shared_expert = (
                load_lin8(gpu, reader, &talker_shared_expert_name(l, "gate")),
                load_lin8(gpu, reader, &talker_shared_expert_name(l, "up")),
                load_lin8(gpu, reader, &talker_shared_expert_name(l, "down")),
            );
            let gate_name = format!("talker.blocks.{l}.mlp.shared_expert_gate.weight");
            let shared_expert_gate = crate::int8_thinker_resident::load_mat(reader, gpu, &gate_name, 1, cfg.hidden);
            layers.insert(l, TalkerLayerExperts8 { experts, shared_expert, shared_expert_gate });
        }
        TalkerInt8Store { layers }
    }

    /// Layer `l`'s expert store — same "panic on an unowned layer, never
    /// default past it" contract as [`ThinkerInt8Store::layer`].
    pub fn layer(&self, l: usize) -> &TalkerLayerExperts8 {
        self.layers.get(&l).unwrap_or_else(|| panic!("TalkerInt8Store: layer {l} not owned by this shard"))
    }

    /// Routed expert `e`'s `(gate, up, down)` weights in layer `l` —
    /// shorthand for `self.layer(l).lin8_at(e)`, same convenience
    /// [`ThinkerInt8Store::lin8_at`] provides.
    pub fn lin8_at(&self, l: usize, e: usize) -> (model::moe::Lin8<'_>, model::moe::Lin8<'_>, model::moe::Lin8<'_>) {
        self.layer(l).lin8_at(e)
    }

    /// Layer `l`'s shared expert `(gate, up, down)` weights — shorthand for
    /// `self.layer(l).shared_lin8()`.
    pub fn shared_lin8_at(&self, l: usize) -> (model::moe::Lin8<'_>, model::moe::Lin8<'_>, model::moe::Lin8<'_>) {
        self.layer(l).shared_lin8()
    }
}

/// Real per-device byte total for `layers`' routed-expert weights, computed
/// from the checkpoint's DECLARED shapes (`WeightReader::shape`) — no GPU,
/// no upload, so a caller can call this for [`crate::int8_thinker_resident::
/// Int8ThinkerResident`]'s `estimate_multi` BEFORE deciding placement,
/// exactly the "know the cost before building" contract
/// `residency::MultiDeviceResidentModel::estimate_multi` requires.
///
/// At the real Thinker shape (`d_model=2048, moe_ff=768, n_experts=128`):
/// each expert's 3 linears are `[768,2048]` (gate/up) and `[2048,768]`
/// (down), 1,572,864 params each, packed 4-per-`u32` = 1,572,864 bytes
/// (~1.5 MiB) + a negligible per-channel scale. ~4.5 MiB/expert x 128 x 48
/// layers ~= 27.6 GiB total (a 24/24 layer split puts ~13.8 GiB/card).
pub fn expert_bytes(reader: &WeightReader, layers: impl Iterator<Item = usize>, cfg: &MoeTextConfig) -> u64 {
    let mut total = 0u64;
    for l in layers {
        for e in 0..cfg.n_experts as usize {
            for leaf in ["gate", "up", "down"] {
                let name = expert_name(l, e, leaf);
                if let Some(shape) = reader.shape(&name) {
                    let (n, kg) = (shape[0], shape[1]); // shape is [n, k/4] (packed) per omni::import's own plan
                    total += n * kg * 4; // packed bytes: u32 * 4
                    total += n * 4; // scale: f32 per output channel
                } else {
                    panic!("ThinkerInt8Store::expert_bytes: missing '{name}' in this checkpoint");
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::weightio::{Dtype, StWriter};
    use data::rng::Lcg;
    use model::int8::quantize_weight;
    use model::moe::{expert_fwd_i8, ExpertScratch8, MoeIds8, MoeShape};

    const PIPES: &[(&str, &str)] = &[
        ("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8),
        ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
        ("silu_mul", kernels::SILU_MUL),
        ("scale_add", kernels::SCALE_ADD),
        ("max_abs_row", kernels::MAX_ABS_ROW),
        ("quant_pack", kernels::QUANT_PACK),
        ("router_gate", kernels::ROUTER_GATE),
    ];

    fn idx(g: &gpu_core::Gpu, name: &str) -> usize {
        g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
    }

    /// A tiny synthetic int8 checkpoint, written via the SAME `StWriter`
    /// M18's real import used, with the EXACT brain-native names/dtypes
    /// `omni::import` produces (`thinker.blocks.{l}.mlp.experts.{e}.{leaf}
    /// .weight` as `U32`, `.scale` as `F32`) — the honest structural
    /// alternative to a real 36 GB checkpoint (not present in this
    /// environment), proving the STREAMING/NAMING/UPLOAD machinery is
    /// correct rather than re-proving `quantize_weight`'s own math (already
    /// covered by `crates/model/tests/moe_sparse_i8_parity.rs`).
    fn write_synthetic_checkpoint(path: &str, cfg: &MoeTextConfig, layers: &[usize], seed: u64) -> HashMap<(usize, usize, &'static str), Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let (d, ff) = (cfg.hidden as usize, cfg.moe_intermediate as usize);
        let mut host_weights = HashMap::new();
        let mut plan: Vec<(String, Vec<u64>, Dtype)> = Vec::new();
        let mut packed_by_name: HashMap<String, (Vec<u32>, Vec<f32>)> = HashMap::new();
        for &l in layers {
            for e in 0..cfg.n_experts as usize {
                for (leaf, n, k) in [("gate", ff, d), ("up", ff, d), ("down", d, ff)] {
                    let w = rng.vec_scaled(n * k, 0.5);
                    let (packed, scale) = quantize_weight(&w, n, k);
                    let name = expert_name(l, e, leaf);
                    plan.push((name.clone(), vec![n as u64, (k / 4) as u64], Dtype::U32));
                    plan.push((format!("{name}.scale"), vec![n as u64], Dtype::F32));
                    packed_by_name.insert(name.clone(), (packed, scale));
                    host_weights.insert((l, e, leaf), w);
                }
            }
        }
        let mut writer = StWriter::create_mixed(path, &plan, &serde_json::Value::Null, None).expect("create synthetic checkpoint");
        for (name, _, _) in &plan {
            if let Some(base) = name.strip_suffix(".scale") {
                let (_, scale) = &packed_by_name[base];
                writer.write(name, scale).expect("write scale");
            } else {
                let (packed, _) = &packed_by_name[name];
                writer.write_u32(name, packed).expect("write packed");
            }
        }
        writer.finish().expect("finish synthetic checkpoint");
        host_weights
    }

    fn tiny_cfg() -> MoeTextConfig {
        let mut cfg = MoeTextConfig::thinker_defaults();
        // Shrink to a size real hardware/CI can build+validate in
        // milliseconds -- d_model/moe_ff stay multiples of 4 (int8 packing).
        cfg.n_layers = 3;
        cfg.hidden = 16;
        cfg.moe_intermediate = 12;
        cfg.n_experts = 5;
        cfg.top_k = 2;
        cfg
    }

    #[test]
    fn build_streams_the_real_brain_native_names_and_round_trips() {
        let cfg = tiny_cfg();
        let dir = std::env::temp_dir().join(format!("int8_resident_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("thinker.safetensors");
        let host = write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, &[0, 1, 2], 777);

        let reader = WeightReader::open(path.to_str().unwrap()).expect("open synthetic checkpoint");

        // expert_bytes (host-only, no GPU) must match what build() actually uploads.
        let expected_bytes = expert_bytes(&reader, [0usize, 1, 2].into_iter(), &cfg);

        let g = gpu_core::testgpu::dev(PIPES);
        // Shard: this "device" owns layers 0 and 2 only (simulating a
        // two-card layer-range split missing the middle layer, to prove
        // `lin8_at` really scopes to OWNED layers, not "any layer exists").
        let store = ThinkerInt8Store::build(&g, &reader, [0usize, 2].into_iter(), &cfg);
        assert_eq!(store.layers.len(), 2);
        assert!(store.layers.contains_key(&0) && store.layers.contains_key(&2) && !store.layers.contains_key(&1));

        // Round-trip: read back layer 0, expert 0's gate weight and compare
        // against the quantize_weight() output that was written (bit-exact:
        // this is a straight upload, no re-quantization).
        let (gate8, _, _) = store.lin8_at(0, 0);
        let (expected_packed, expected_scale) = quantize_weight(&host[&(0, 0, "gate")], cfg.moe_intermediate as usize, cfg.hidden as usize);
        let got_packed = g.read(gate8.wq, expected_packed.len()).iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        // read() returns f32; reinterpret back to u32 words for comparison
        // against the packed int8 layout (bit-for-bit, not numeric).
        assert_eq!(got_packed, expected_packed, "packed weight must round-trip bit-exact");
        let got_scale = g.read(gate8.sw, expected_scale.len());
        assert_eq!(got_scale, expected_scale, "scale must round-trip exactly");

        let expected_bytes_for_02: u64 = expert_bytes(&reader, [0usize, 2].into_iter(), &cfg);
        assert!(expected_bytes_for_02 < expected_bytes, "shard of 2 layers must cost less than all 3");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "not owned by this shard")]
    fn lin8_at_panics_on_an_unowned_layer_rather_than_silently_reading_garbage() {
        let cfg = tiny_cfg();
        let dir = std::env::temp_dir().join(format!("int8_resident_test2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("thinker.safetensors");
        write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, &[0], 111);
        let reader = WeightReader::open(path.to_str().unwrap()).expect("open synthetic checkpoint");
        let g = gpu_core::testgpu::dev(PIPES);
        let store = ThinkerInt8Store::build(&g, &reader, [0usize].into_iter(), &cfg);
        let _ = store.lin8_at(1, 0); // layer 1 was never built
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The store's `Lin8` views must feed `expert_fwd_i8` and produce the
    /// SAME result as the fp32 source weights through the fp32 sparse path
    /// (within `moe_sparse_i8_parity.rs`'s own quantization tolerance) —
    /// proving `ThinkerInt8Store` is not just structurally plausible but
    /// numerically USABLE by the real kernel it exists to feed.
    #[test]
    fn store_weights_produce_correct_expert_fwd_i8_output() {
        let cfg = tiny_cfg();
        let dir = std::env::temp_dir().join(format!("int8_resident_test3_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("thinker.safetensors");
        let host = write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, &[0], 2024);
        let reader = WeightReader::open(path.to_str().unwrap()).expect("open synthetic checkpoint");

        let g = gpu_core::testgpu::dev(PIPES);
        let store = ThinkerInt8Store::build(&g, &reader, [0usize].into_iter(), &cfg);

        let ids8 = MoeIds8 {
            linear_gated_i8: idx(&g, "moe_linear_gated_i8"),
            silu_mul: idx(&g, "silu_mul"),
            scale_add: idx(&g, "scale_add"),
            quant: [idx(&g, "max_abs_row"), idx(&g, "quant_pack")],
        };
        let router_ids = model::moe::MoeIds { router_gate: idx(&g, "router_gate"), linear_gated: 0, silu_mul: 0, scale_add: 0 };

        let (m, d, ff, e) = (4u32, cfg.hidden, cfg.moe_intermediate, cfg.n_experts);
        let shape = MoeShape { rows: m, d_model: d, moe_ff: ff, n_experts: e, top_k: cfg.top_k };
        let mut rng = Lcg::new(55);
        let logits = g.storage_init("logits", &rng.vec_scaled((m * e) as usize, 2.0));
        let x_host = rng.vec_scaled((m * d) as usize, 1.0);
        let x = g.storage_init("x", &x_host);
        let gate = g.storage((m * e) as u64);
        g.submit(&[], &[model::moe::router_fwd(&g, &router_ids, &shape, &logits, &gate)]);

        // int8 path via the store.
        let xq = g.storage((m * d / 4) as u64);
        let sx = g.storage(m as u64);
        g.submit(&[], &model::int8::quant_rows_steps(&g, model::int8::QuantRows { kernels: ids8.quant, x: &x, sx: &sx, xq: &xq }, 0, m, d));
        let scratch8 = ExpertScratch8 {
            gate_pre: &g.storage((m * ff) as u64),
            up: &g.storage((m * ff) as u64),
            h: &g.storage((m * ff) as u64),
            hq: &g.storage((m * ff / 4) as u64),
            sh: &g.storage(m as u64),
            expert_out: &g.storage((m * d) as u64),
        };
        let acc_i8 = g.storage((m * d) as u64);
        for ei in 0..e as usize {
            let (gw, uw, dw) = store.lin8_at(0, ei);
            let steps = expert_fwd_i8(&g, &ids8, &shape, &xq, &sx, &gate, gw, uw, dw, &scratch8, &acc_i8, ei as u32, ei != 0);
            g.submit(&[], &steps);
        }

        // fp32 reference: the SAME host weights the checkpoint was quantized
        // from, through model::moe::expert_fwd -- the exact oracle
        // `moe_sparse_i8_parity.rs` already trusts.
        let gate_w: Vec<DeviceBuffer> = (0..e as usize).map(|ei| g.storage_init(&format!("gw{ei}"), &host[&(0, ei, "gate")])).collect();
        let up_w: Vec<DeviceBuffer> = (0..e as usize).map(|ei| g.storage_init(&format!("uw{ei}"), &host[&(0, ei, "up")])).collect();
        let down_w: Vec<DeviceBuffer> = (0..e as usize).map(|ei| g.storage_init(&format!("dw{ei}"), &host[&(0, ei, "down")])).collect();
        let fp32_ids = model::moe::MoeIds { router_gate: idx(&g, "router_gate"), linear_gated: idx(&g, "moe_linear_gated"), silu_mul: idx(&g, "silu_mul"), scale_add: idx(&g, "scale_add") };
        let scratch = model::moe::ExpertScratch {
            gate_pre: &g.storage((m * ff) as u64),
            up: &g.storage((m * ff) as u64),
            h: &g.storage((m * ff) as u64),
            expert_out: &g.storage((m * d) as u64),
        };
        let acc_fp32 = g.storage((m * d) as u64);
        for ei in 0..e as usize {
            let steps = model::moe::expert_fwd(&g, &fp32_ids, &shape, &x, &gate, &gate_w[ei], &up_w[ei], &down_w[ei], &scratch, &acc_fp32, ei as u32, ei != 0);
            g.submit(&[], &steps);
        }

        g.poll_wait();
        let i8out = g.read(&acc_i8, (m * d) as usize);
        let fp32out = g.read(&acc_fp32, (m * d) as usize);
        assert!(fp32out.iter().any(|&v| v.abs() > 1e-9), "fp32 oracle is all-zero -- the test shape routes nothing");

        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in i8out.iter().zip(fp32out.iter()) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        let rel_l2 = (num / den.max(1e-12)).sqrt();
        // Same tolerance moe_sparse_i8_parity.rs uses (0.0084 measured
        // there; 0.02 leaves headroom without hiding a real regression).
        assert!(rel_l2 < 0.02, "ThinkerInt8Store-fed expert_fwd_i8 diverged from the fp32 oracle: rel_l2={rel_l2:.4}");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn talker_tiny_cfg() -> MoeTextConfig {
        let mut cfg = MoeTextConfig::talker_defaults();
        cfg.n_layers = 3;
        cfg.hidden = 16;
        cfg.moe_intermediate = 12;
        cfg.shared_expert_intermediate = 8;
        cfg.n_experts = 5;
        cfg.top_k = 2;
        cfg
    }

    /// [`write_synthetic_checkpoint`]'s Talker twin: routed experts PLUS the
    /// shared expert's three linears (all int8-quantized, `map_talker`'s
    /// naming) plus `shared_expert_gate.weight` written PLAIN F32 (never
    /// quantized, matching a real checkpoint's `should_quantize` outcome for
    /// a `[1, hidden]` tensor being irrelevant here -- `TalkerInt8Store::
    /// build` reads it through [`crate::int8_thinker_resident::load_mat`],
    /// which handles either dtype; this test only needs to prove the F32
    /// leg).
    fn write_synthetic_talker_checkpoint(path: &str, cfg: &MoeTextConfig, layers: &[usize], seed: u64) -> HashMap<(usize, usize, &'static str), Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let (d, ff, se_ff) = (cfg.hidden as usize, cfg.moe_intermediate as usize, cfg.shared_expert_intermediate as usize);
        let mut host_weights = HashMap::new();
        let mut plan: Vec<(String, Vec<u64>, Dtype)> = Vec::new();
        let mut packed_by_name: HashMap<String, (Vec<u32>, Vec<f32>)> = HashMap::new();
        let mut gate_by_name: HashMap<String, Vec<f32>> = HashMap::new();
        for &l in layers {
            for e in 0..cfg.n_experts as usize {
                for (leaf, n, k) in [("gate", ff, d), ("up", ff, d), ("down", d, ff)] {
                    let w = rng.vec_scaled(n * k, 0.5);
                    let (packed, scale) = quantize_weight(&w, n, k);
                    let name = talker_expert_name(l, e, leaf);
                    plan.push((name.clone(), vec![n as u64, (k / 4) as u64], Dtype::U32));
                    plan.push((format!("{name}.scale"), vec![n as u64], Dtype::F32));
                    packed_by_name.insert(name.clone(), (packed, scale));
                    host_weights.insert((l, e, leaf), w);
                }
            }
            for (leaf, n, k) in [("gate", se_ff, d), ("up", se_ff, d), ("down", d, se_ff)] {
                let w = rng.vec_scaled(n * k, 0.5);
                let (packed, scale) = quantize_weight(&w, n, k);
                let name = talker_shared_expert_name(l, leaf);
                plan.push((name.clone(), vec![n as u64, (k / 4) as u64], Dtype::U32));
                plan.push((format!("{name}.scale"), vec![n as u64], Dtype::F32));
                packed_by_name.insert(name.clone(), (packed, scale));
                host_weights.insert((l, usize::MAX, leaf), w); // usize::MAX marks "shared expert", not a routed index
            }
            let gate_w = rng.vec_scaled(d, 0.5); // [1, hidden]
            let gate_name = format!("talker.blocks.{l}.mlp.shared_expert_gate.weight");
            plan.push((gate_name.clone(), vec![1, d as u64], Dtype::F32));
            gate_by_name.insert(gate_name.clone(), gate_w.clone());
            host_weights.insert((l, usize::MAX, "gate_proj"), gate_w);
        }
        let mut writer = StWriter::create_mixed(path, &plan, &serde_json::Value::Null, None).expect("create synthetic checkpoint");
        for (name, _, _) in &plan {
            if let Some(base) = name.strip_suffix(".scale") {
                let (_, scale) = &packed_by_name[base];
                writer.write(name, scale).expect("write scale");
            } else if let Some(w) = gate_by_name.get(name) {
                writer.write(name, w).expect("write shared_expert_gate");
            } else {
                let (packed, _) = &packed_by_name[name];
                writer.write_u32(name, packed).expect("write packed");
            }
        }
        writer.finish().expect("finish synthetic checkpoint");
        host_weights
    }

    #[test]
    fn talker_build_streams_the_real_brain_native_names_and_round_trips() {
        let cfg = talker_tiny_cfg();
        let dir = std::env::temp_dir().join(format!("talker_int8_resident_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("talker.safetensors");
        let host = write_synthetic_talker_checkpoint(path.to_str().unwrap(), &cfg, &[0, 1, 2], 888);
        let reader = WeightReader::open(path.to_str().unwrap()).expect("open synthetic checkpoint");

        let g = gpu_core::testgpu::dev(PIPES);
        let store = TalkerInt8Store::build(&g, &reader, [0usize, 2].into_iter(), &cfg);
        assert_eq!(store.layers.len(), 2);
        assert!(store.layers.contains_key(&0) && store.layers.contains_key(&2) && !store.layers.contains_key(&1));

        // Round-trip: routed expert 0's gate weight, bit-exact.
        let (gate8, _, _) = store.lin8_at(0, 0);
        let (expected_packed, expected_scale) = quantize_weight(&host[&(0, 0, "gate")], cfg.moe_intermediate as usize, cfg.hidden as usize);
        let got_packed = g.read(gate8.wq, expected_packed.len()).iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        assert_eq!(got_packed, expected_packed, "routed expert packed weight must round-trip bit-exact");
        assert_eq!(g.read(gate8.sw, expected_scale.len()), expected_scale, "routed expert scale must round-trip exactly");

        // Round-trip: the shared expert's own gate linear, bit-exact.
        let (sgate8, _, _) = store.shared_lin8_at(0);
        let (expected_spacked, expected_sscale) = quantize_weight(&host[&(0, usize::MAX, "gate")], cfg.shared_expert_intermediate as usize, cfg.hidden as usize);
        let got_spacked = g.read(sgate8.wq, expected_spacked.len()).iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        assert_eq!(got_spacked, expected_spacked, "shared expert packed weight must round-trip bit-exact");
        assert_eq!(g.read(sgate8.sw, expected_sscale.len()), expected_sscale, "shared expert scale must round-trip exactly");

        // Round-trip: shared_expert_gate.weight (plain F32, never quantized).
        let got_gate = g.read(&store.layer(0).shared_expert_gate, cfg.hidden as usize);
        assert_eq!(got_gate, host[&(0, usize::MAX, "gate_proj")], "shared_expert_gate must round-trip exactly (plain f32, no requantization)");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "not owned by this shard")]
    fn talker_lin8_at_panics_on_an_unowned_layer_rather_than_silently_reading_garbage() {
        let cfg = talker_tiny_cfg();
        let dir = std::env::temp_dir().join(format!("talker_int8_resident_test2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("talker.safetensors");
        write_synthetic_talker_checkpoint(path.to_str().unwrap(), &cfg, &[0], 222);
        let reader = WeightReader::open(path.to_str().unwrap()).expect("open synthetic checkpoint");
        let g = gpu_core::testgpu::dev(PIPES);
        let store = TalkerInt8Store::build(&g, &reader, [0usize].into_iter(), &cfg);
        let _ = store.lin8_at(1, 0); // layer 1 was never built
        std::fs::remove_dir_all(&dir).ok();
    }
}
