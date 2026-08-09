// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A layer-sharded, int8-MoE-expert resident Thinker spanning TWO real GPUs —
//! the real wiring `.todo/omni-int8-dual-gpu-residency.md` asked for, built
//! on `crates/residency/src/multi.rs`'s `MultiDeviceResidentModel`/
//! `claim_multi` (that session's own addition) and `crate::int8_resident::
//! ThinkerInt8Store` (this one's).
//!
//! **Scope, honestly bounded**: this validates the CROSS-DEVICE MECHANISM
//! for real — real per-device `MultiDeviceCost` accounting from a real
//! checkpoint's declared shapes, a real `claim_multi`/`activate_multi`
//! round trip, two REAL `Gpu` instances (one per physical card, via
//! `Gpu::new_on_index`), each holding a REAL streamed `ThinkerInt8Store`
//! shard AND (as of this pass) real streamed attention/norm/router weights
//! via [`load_layer_bufs`] — dequantized on load if the checkpoint quantized
//! them (`omni::import::should_quantize` applies to the attention/router
//! projections too, not just the MoE experts, since they meet the same
//! rank-2/`k%4==0` shape test; `thinker::layer_fwd` has no int8 dispatch
//! path for them, only for the experts, so they always come back out as
//! plain f32 regardless of how the checkpoint stored them) — and a REAL
//! forward pass that hands the residual stream between shards via a host
//! round-trip (`gpu_a.read` → `gpu_b.write`, negligible bytes: `n * d_model`
//! floats per hop, `<10 KiB` at any realistic decode batch). What it still
//! does NOT do: `embed_tokens`/`lm_head` are out of scope (this is the
//! MoE-bearing-layers validation action, not a full `generate()` — see
//! [`Int8ThinkerInstance::forward`]'s own doc), and `synth_layer_bufs` /
//! `seed_for_layer` remain available (unused by `activate_multi` now) for
//! tests that want deterministic weights without a checkpoint on disk.

use std::collections::HashMap;
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, Invocation, Manifest, Media, Outcome, Progress, Provider};
use checkpoint::weightio::WeightReader;
use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::config::MoeTextConfig;
use crate::int8_resident::{expert_bytes, ThinkerInt8Store};
use crate::thinker::{final_norm, layer_fwd, lm_head_fwd, thinker_pipelines, ThinkerLayerWeights};

pub const MODEL: &str = "brain/omni-int8-thinker-multi";

/// A contiguous layer range `[start, end)` assigned to one device.
type LayerRange = std::ops::Range<usize>;

/// A layer's synthetic (deterministic, seeded by absolute layer index)
/// non-expert weights — kept for tests that want deterministic weights
/// without a checkpoint on disk; [`activate_multi`](Int8ThinkerResident::
/// activate_multi) now uses [`load_layer_bufs`], the real loader, instead
/// (see that function's doc for why this existed in the first place).
/// `seed_for_layer` is the ONLY thing that needs to agree between two
/// independent builders for them to reconstruct bit-identical weights
/// without sharing any state.
pub fn seed_for_layer(l: usize) -> u64 {
    100_000 + l as u64
}

pub struct LayerBufs {
    ln1: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    ln2: DeviceBuffer,
    router: DeviceBuffer,
}

/// Build layer `l`'s synthetic non-expert weights on `gpu`, deterministically
/// from `seed_for_layer(l)` — see this module's own doc.
pub fn synth_layer_bufs(gpu: &Gpu, cfg: &MoeTextConfig, l: usize) -> LayerBufs {
    let mut rng = Lcg::new(seed_for_layer(l));
    let (d, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let init = |rng: &mut Lcg, n: usize| gpu.storage_init("w", &rng.vec_scaled(n, 0.3));
    LayerBufs {
        ln1: init(&mut rng, d as usize),
        wq: init(&mut rng, (hq * d) as usize),
        wk: init(&mut rng, (hkv * d) as usize),
        wv: init(&mut rng, (hkv * d) as usize),
        wo: init(&mut rng, (d * hq) as usize),
        q_norm: init(&mut rng, hd as usize),
        k_norm: init(&mut rng, hd as usize),
        ln2: init(&mut rng, d as usize),
        router: init(&mut rng, (cfg.n_experts * d) as usize),
    }
}

/// One tensor, real, host-resident: plain f32 if the checkpoint stored it
/// that way, or unpacked via [`model::int8::dequantize_weight`] if
/// `omni::import` quantized it (`should_quantize`: rank-2, last dim a
/// multiple of 4 — attention/router/embed/lm_head projections all meet this
/// exactly like the MoE experts do, but unlike the experts have no int8
/// dispatch path of their own in `thinker::layer_fwd`, so they come back
/// out as plain f32 either way). `n`/`k` are only consulted on the
/// quantized branch (the packed shape is `[n, k/4]`, so `k` cannot be
/// recovered from the tensor's own shape alone).
pub fn load_mat_host(reader: &WeightReader, name: &str, n: u32, k: u32) -> Vec<f32> {
    match reader.dtype(name) {
        Some("U32") => {
            let packed = reader.tensor_u32(name).unwrap_or_else(|| panic!("missing tensor {name}"));
            let scale_name = format!("{name}.scale");
            let scale = reader.tensor(&scale_name).unwrap_or_else(|| panic!("missing tensor {scale_name} (scale sibling of quantized {name})"));
            model::int8::dequantize_weight(&packed, &scale, n as usize, k as usize)
        }
        _ => reader.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}")),
    }
}

/// [`load_mat_host`], uploaded to `gpu` — the GPU-resident variant every
/// per-layer weight (attention/router) wants.
pub fn load_mat(reader: &WeightReader, gpu: &Gpu, name: &str, n: u32, k: u32) -> DeviceBuffer {
    gpu.storage_init("w", &load_mat_host(reader, name, n, k))
}

/// A 1-D tensor (norm gains) — `omni::import::should_quantize` never
/// quantizes rank-1 tensors, so these are always plain f32.
pub fn load_vec(reader: &WeightReader, gpu: &Gpu, name: &str) -> DeviceBuffer {
    gpu.storage_init("w", &reader.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}")))
}

/// Build layer `l`'s REAL non-expert weights on `gpu`, streamed from
/// `reader` — the real counterpart to [`synth_layer_bufs`]. Brain-native
/// names (`omni::import::map_thinker`'s output, prefix `thinker.blocks.{l}.`
/// already applied by [`ThinkerLayerWeights`]'s own convention), matching
/// exactly what `crate::int8_resident::ThinkerInt8Store::build` reads for
/// the same layer's expert weights from the same checkpoint.
pub fn load_layer_bufs(gpu: &Gpu, reader: &WeightReader, cfg: &MoeTextConfig, l: usize) -> LayerBufs {
    let (d, hd, nh, nkv, ne) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.n_experts);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let p = |leaf: &str| format!("thinker.blocks.{l}.{leaf}");
    LayerBufs {
        ln1: load_vec(reader, gpu, &p("ln1.weight")),
        wq: load_mat(reader, gpu, &p("attn.wq.weight"), hq, d),
        wk: load_mat(reader, gpu, &p("attn.wk.weight"), hkv, d),
        wv: load_mat(reader, gpu, &p("attn.wv.weight"), hkv, d),
        wo: load_mat(reader, gpu, &p("attn.wo.weight"), d, hq),
        q_norm: load_vec(reader, gpu, &p("attn.q_norm.weight")),
        k_norm: load_vec(reader, gpu, &p("attn.k_norm.weight")),
        ln2: load_vec(reader, gpu, &p("ln2.weight")),
        router: load_mat(reader, gpu, &p("mlp.router.weight"), ne, d),
    }
}

pub fn weights(b: &LayerBufs) -> ThinkerLayerWeights<'_> {
    ThinkerLayerWeights { ln1: &b.ln1, wq: &b.wq, wk: &b.wk, wv: &b.wv, wo: &b.wo, q_norm: &b.q_norm, k_norm: &b.k_norm, ln2: &b.ln2, router: &b.router, experts: &[] }
}

/// One device's shard: its own `Gpu`, the absolute layer range it owns, the
/// resident int8 expert store for that range, and each owned layer's
/// synthetic non-expert weights (see this module's own doc).
struct DeviceShard {
    gpu: Gpu,
    range: LayerRange,
    store: ThinkerInt8Store,
    layer_bufs: HashMap<usize, LayerBufs>,
}

pub struct Int8ThinkerInstance {
    cfg: MoeTextConfig,
    shards: Vec<DeviceShard>,
    /// `thinker.embed_tokens.weight` `[vocab, hidden]`, dequantized (real
    /// checkpoints quantize it too — rank-2, last dim a multiple of 4, same
    /// as any other real weight) and kept HOST-resident: [`Self::generate`]
    /// only ever needs a per-token row gather, never a GEMM, so there is no
    /// reason to spend GPU memory or an upload on the ~1.2 GB dequantized
    /// table.
    embed_table_host: Vec<f32>,
    /// `thinker.norm.weight` and `thinker.lm_head.weight`, resident on the
    /// LAST shard's `Gpu` — that is where [`Self::forward`]'s final hidden
    /// state already ends up (its own host round-trip loop), so applying
    /// the head there avoids one extra cross-device hop.
    final_norm_w: DeviceBuffer,
    lm_head_w: DeviceBuffer,
}

impl Int8ThinkerInstance {
    /// Run every owned layer, in absolute layer order, across however many
    /// device shards this instance holds — a host round-trip hands the
    /// residual stream to the next shard's `Gpu` at each boundary. `x` is
    /// the initial hidden state `[n, d]`, host-resident (the caller's own
    /// embedding/splice step, out of scope here — see this module's doc).
    /// Returns the final hidden state `[n, d]`, host-resident (no final
    /// norm/lm_head — this is the MoE-bearing-layers validation action, not
    /// a full generate()).
    pub fn forward(&self, x_host: &[f32], n: u32) -> Vec<f32> {
        let d = self.cfg.hidden;
        assert_eq!(x_host.len(), (n * d) as usize, "x must be [n, d]");

        // M-RoPE table: diagonal (plain text), same construction
        // thinker_decode.rs/thinker_int8_parity.rs use.
        let tokens: Vec<u32> = (0..n).collect();
        let positions = qwenvl::mrope::get_rope_index(&tokens, u32::MAX, &[]);
        let section: [u32; 3] = [self.cfg.mrope_section[0], self.cfg.mrope_section[1], self.cfg.mrope_section[2]];
        let (cos_tab, sin_tab) = qwenvl::mrope::mrope_tables(&positions, section, self.cfg.head_dim, self.cfg.rope_theta);

        let mut h_host = x_host.to_vec();
        for shard in &self.shards {
            let gpu = &shard.gpu;
            let cos = gpu.storage_init("cos", &cos_tab);
            let sin = gpu.storage_init("sin", &sin_tab);
            let mut h = gpu.storage_init("h", &h_host);
            for l in shard.range.clone() {
                let lb = &shard.layer_bufs[&l];
                let w = weights(lb);
                let experts8 = shard.store.layer(l);
                let (out, ..) = layer_fwd(gpu, &self.cfg, &w, &h, &cos, &sin, n, None, Some(experts8));
                h = out;
            }
            // Host-mediated handoff to the next shard's device (a no-op
            // read+carry on the LAST shard -- still correct, just an extra
            // round trip nothing further consumes).
            h_host = gpu.read(&h, (n * d) as usize);
        }
        h_host
    }

    /// Greedy (argmax) text generation over the sharded int8 Thinker — the
    /// validation-tier `generate()` this module's own doc named as the real
    /// remainder: real tokens in, real sampled tokens out, EOS handling, no
    /// tokenization (matches `qwenvl::Qwen3Vl::generate`'s own contract:
    /// `prompt_ids`/`eos_ids` are already token ids, and the return value is
    /// the GENERATED tokens only, prompt excluded).
    ///
    /// Deliberately the simple O(T²) shape, not a KV-cache decode loop: each
    /// step re-embeds the WHOLE ids-so-far window and calls [`Self::forward`]
    /// again (a full recompute, same as `crate::generate`'s own
    /// non-resident precedent before its KV cache was added) rather than
    /// extending `thinker::layer_decode_step`'s per-layer cache across two
    /// device shards. That is real, separable follow-up (`layer_decode_step`
    /// needs its own `ThinkerLayerCache` per owned layer, and per this
    /// module's own layer-RANGE-never-expert-split sharding, each layer's
    /// cache would live entirely on the ONE shard that owns that layer —
    /// not a cross-device complication, just unbuilt), not attempted here:
    /// this validates the sharded MECHANISM end-to-end with real sampling,
    /// which is the actual gap `.todo/omni-int8-dual-gpu-residency.md`
    /// named, not a performance claim.
    pub fn generate(&self, prompt_ids: &[u32], max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
        let d = self.cfg.hidden as usize;
        let last = self.shards.last().expect("Int8ThinkerInstance has no shards");
        let embed_row = |t: u32| self.embed_table_host[t as usize * d..(t as usize + 1) * d].to_vec();

        let mut ids: Vec<u32> = prompt_ids.to_vec();
        let mut out = Vec::with_capacity(max_new_tokens as usize);
        if prompt_ids.is_empty() || max_new_tokens == 0 {
            return out;
        }

        for _ in 0..max_new_tokens {
            let n = ids.len() as u32;
            let mut x_host = Vec::with_capacity(ids.len() * d);
            for &t in &ids {
                x_host.extend_from_slice(&embed_row(t));
            }
            let hidden = self.forward(&x_host, n);
            let last_row = &hidden[(n as usize - 1) * d..n as usize * d];

            let gpu = &last.gpu;
            let h1 = gpu.storage_init("h1", last_row);
            let normed = final_norm(gpu, &self.cfg, &self.final_norm_w, &h1, 1);
            let logits = lm_head_fwd(gpu, &self.lm_head_w, &normed, 1, self.cfg.hidden, self.cfg.vocab);
            let logits_host = gpu.read(&logits, self.cfg.vocab as usize);
            let next = argmax(&logits_host);

            if eos_ids.contains(&next) {
                break;
            }
            ids.push(next);
            out.push(next);
        }
        out
    }
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("non-empty vocab")
}

impl Instance for Int8ThinkerInstance {
    /// `forward`: input blob `x` (raw LE f32 `[n, d]`, meta `{"n": n}`),
    /// output blob `hidden` (same shape) — internal/validation action, not
    /// real generation, see this module's own doc.
    ///
    /// `generate`: input blob `ids` (raw LE `u32` token ids), meta
    /// `{"max_new_tokens": u32, "eos_ids": [u32]}`; output blob `ids` (raw
    /// LE `u32`, the GENERATED tokens only, prompt excluded — matches
    /// [`Self::generate`]'s own contract).
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "forward" => {
                let blob = inv.get_blob("x").ok_or_else(|| format!("{MODEL}: missing 'x' blob"))?;
                let n = blob.meta.get("n").and_then(|v| v.as_u64()).ok_or_else(|| format!("{MODEL}: 'x' blob missing meta.n"))? as u32;
                let x_host: Vec<f32> = blob.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
                let out = self.forward(&x_host, n);
                let bytes: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok(Outcome::new().blob("hidden", Blob::new(Media::Bytes, bytes).with_meta(serde_json::json!({"n": n}))))
            }
            "generate" => {
                let blob = inv.get_blob("ids").ok_or_else(|| format!("{MODEL}: missing 'ids' blob"))?;
                let prompt_ids: Vec<u32> = blob.bytes.chunks_exact(4).map(|q| u32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
                let max_new_tokens = blob.meta.get("max_new_tokens").and_then(|v| v.as_u64()).ok_or_else(|| format!("{MODEL}: 'ids' blob missing meta.max_new_tokens"))? as u32;
                let eos_ids: Vec<u32> = blob
                    .meta
                    .get("eos_ids")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
                    .unwrap_or_default();
                let out = self.generate(&prompt_ids, max_new_tokens, &eos_ids);
                let bytes: Vec<u8> = out.iter().flat_map(|t| t.to_le_bytes()).collect();
                Ok(Outcome::new().blob("ids", Blob::new(Media::Bytes, bytes).with_meta(serde_json::json!({"n": out.len()}))))
            }
            other => Err(format!("{MODEL}: unknown action '{other}' (only 'forward'/'generate' exist)")),
        }
    }
}

/// The [`ResidentModel`]/[`MultiDeviceResidentModel`] adapter: real
/// `estimate_multi` (from the checkpoint's DECLARED shapes, no GPU), real
/// `activate_multi` (streams + uploads each shard's real int8 expert
/// weights via [`ThinkerInt8Store::build`]).
pub struct Int8ThinkerResident {
    pub checkpoint_path: String,
    pub cfg: MoeTextConfig,
}

impl Int8ThinkerResident {
    /// Contiguous layer ranges for `n_devices` shards — layer-RANGE
    /// (never expert-split — see `.todo/omni-int8-dual-gpu-residency.md`'s
    /// own reasoning: an expert-split cut would move a per-expert partial
    /// output every layer and break `expert_fwd_i8`'s "quantize the shared
    /// input once" contract), as even as `n_layers` allows.
    fn layer_ranges(&self, n_devices: usize) -> Vec<LayerRange> {
        let n = self.cfg.n_layers as usize;
        let base = n / n_devices;
        let extra = n % n_devices;
        let mut ranges = Vec::with_capacity(n_devices);
        let mut start = 0;
        for i in 0..n_devices {
            let len = base + usize::from(i < extra);
            ranges.push(start..start + len);
            start += len;
        }
        ranges
    }
}

impl ResidentModel for Int8ThinkerResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            MODEL,
            "A layer-sharded, int8-MoE-expert Thinker spanning two real GPUs via residency::multi, with real streamed weights (attention/norm/router/experts/embed/lm_head) and real greedy generation -- no tokenization (token ids in, token ids out) and no KV cache (each generate() step is a full recompute; see this module's own doc for why and what a follow-up KV-cache decode loop would need).",
            vec![
                ActionSpec::new("forward", "internal: run the sharded MoE-bearing layers on a raw hidden-state blob"),
                ActionSpec::new("generate", "greedy generation: token ids in (meta max_new_tokens/eos_ids), generated token ids out"),
            ],
        )
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{MODEL}: single-device activate is not supported -- this model is multi-device only, claim it via ResidencyManager::claim_multi"))
    }
}

impl MultiDeviceResidentModel for Int8ThinkerResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        let reader = WeightReader::open(&self.checkpoint_path).unwrap_or_else(|e| panic!("{MODEL}: cannot open '{}': {e}", self.checkpoint_path));
        let ranges = self.layer_ranges(2);
        let per_device: Vec<(Device, u64)> = ranges
            .iter()
            .enumerate()
            .map(|(i, r)| (Device::Gpu(i as u32), expert_bytes(&reader, r.clone(), &self.cfg)))
            .collect();
        MultiDeviceCost::new(per_device, 0)
    }

    fn activate_multi(&self, _key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let reader = WeightReader::open(&self.checkpoint_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.checkpoint_path))?;
        let ranges = self.layer_ranges(devices.len());
        let mut shards = Vec::with_capacity(devices.len());
        for (dev, range) in devices.iter().zip(ranges.iter()) {
            let idx = match dev {
                Device::Gpu(i) => *i,
                other => return Err(format!("{MODEL}: activate_multi got a non-GPU device {other:?}")),
            };
            let gpu = Gpu::new_on_index(idx, thinker_pipelines())?;
            let store = ThinkerInt8Store::build(&gpu, &reader, range.clone(), &self.cfg);
            let layer_bufs = range.clone().map(|l| (l, load_layer_bufs(&gpu, &reader, &self.cfg, l))).collect();
            shards.push(DeviceShard { gpu, range: range.clone(), store, layer_bufs });
        }
        // `Self::generate`'s needs: the embedding table stays host-side (a
        // per-token row gather, never a GEMM -- no reason to spend GPU
        // memory on it), final norm + lm_head live on the LAST shard's
        // `Gpu` (where `forward`'s own host round-trip already lands the
        // final hidden state).
        let (vocab, hidden) = (self.cfg.vocab, self.cfg.hidden);
        let embed_table_host = load_mat_host(&reader, "thinker.embed_tokens.weight", vocab, hidden);
        let last = shards.last().ok_or_else(|| format!("{MODEL}: activate_multi got zero devices"))?;
        let final_norm_w = load_vec(&reader, &last.gpu, "thinker.norm.weight");
        let lm_head_w = load_mat(&reader, &last.gpu, "thinker.lm_head.weight", vocab, hidden);
        Ok(Box::new(Int8ThinkerInstance { cfg: self.cfg.clone(), shards, embed_table_host, final_norm_w, lm_head_w }))
    }
}

/// A weights-free `Provider` wrapper so `Int8ThinkerResident` can also be
/// listed/dispatched the ordinary `capability` way (`brain caps`/`brain do`)
/// in addition to `residency::claim_multi` — mirrors every other model in
/// this codebase carrying both a `Provider` (discovery/direct dispatch) and
/// a `ResidentModel` (automatic scheduling) face over the SAME logic.
pub struct Int8ThinkerProvider {
    resident: Arc<Int8ThinkerResident>,
}

impl Int8ThinkerProvider {
    pub fn new(checkpoint_path: String, cfg: MoeTextConfig) -> Int8ThinkerProvider {
        Int8ThinkerProvider { resident: Arc::new(Int8ThinkerResident { checkpoint_path, cfg }) }
    }
}

impl Provider for Int8ThinkerProvider {
    fn manifest(&self) -> Manifest {
        self.resident.manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        matches!(name, "forward" | "generate").then(|| Arc::new(NotDirectlyDispatchable(name.to_string())) as Arc<dyn Action>)
    }
}

/// `Provider::action` must return something, but this model can only run
/// via `residency::claim_multi` (it needs two live GPU handles the plain
/// `capability::Action::run(&self, ...)` contract has no way to hand in) —
/// a clear, honest error instead of a `None` that would look like the
/// action does not exist at all. Carries the requested action's name so
/// [`Action::spec`] reports the SAME name it was looked up under (both
/// `forward` and `generate` share this one stub).
struct NotDirectlyDispatchable(String);
impl Action for NotDirectlyDispatchable {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new(&self.0, "multi-device only -- see Int8ThinkerResident's own doc")
    }
    fn run(&self, _inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        Err(format!("{MODEL}: this model is multi-device only -- dispatch it via residency::ResidencyManager::claim_multi, not a direct Action::run"))
    }
}
