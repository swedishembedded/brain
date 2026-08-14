// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Greedy text generation over the Thinker decoder, across however many GPUs
//! the box has, from a **raw HF checkpoint of any per-tensor dtype**.
//!
//! # What runs where, and what it costs
//!
//! [`ThinkerStack`] holds the decoder's weights according to
//! `crate::thinker_plan`'s placement: the longest run of layers that genuinely
//! fits is uploaded once, spread across the devices by
//! `model::shard::plan_by_capacity` (the one capacity-aware partitioner in
//! this engine - nothing here re-derives placement), and whatever is left over
//! is streamed per use onto the last stage. A checkpoint small enough to fit
//! streams nothing and never re-reads a weight; a 30B one at f32 (~120 GiB of
//! decoder weights) holds what it can and streams the rest, which is slow but
//! correct and, crucially, BOUNDED.
//!
//! # Two bugs this shape exists to prevent
//!
//! Both were live on the real 30B checkpoint, and both are about the streamed
//! layers rather than the math:
//!
//! 1. **Unbounded host materialization.** Each weight used to be decoded whole
//!    into a `Vec<f32>` and handed to `create_buffer_init`. Every upload here
//!    goes through [`paramstore::upload::Uploader`] instead, which decodes a
//!    bounded chunk at a time straight from the mapping - so a `BF16` tensor
//!    never exists as a whole f32 copy on the host, and no upload takes
//!    wgpu's mapped-at-creation path (which forces weights into an inefficient
//!    memory type on a non-ReBAR card).
//! 2. **Unbounded DEVICE accumulation** - the one that actually crashed. A
//!    dropped `DeviceBuffer` is not reclaimed until the commands referencing
//!    it have retired, and this loop records dispatches without submitting
//!    (the first real submit was the terminal readback). So every streamed
//!    layer's ~2.4 GiB of expert buffers stayed resident even though the live
//!    set was one layer, and a single request walked a 24 GB card to
//!    `wgpu error: Out of Memory` about nine layers in. [`Uploader::drain`]
//!    after each streamed layer forces the submit + wait that reclaims it.
//!
//! # KV cache
//!
//! The attention math is not the validation-tier O(T²) recompute:
//! [`generate_greedy`] prefills the prompt once (bulk-filling each layer's
//! persistent cache) and then decodes one token at a time against it. Each
//! layer's cache lives on the SAME device as that layer, so the cache never
//! crosses a card. What still costs per token is re-streaming the layers that
//! did not fit - the cache changes the attention complexity, not the weight
//! I/O of a model bigger than the box.

use std::collections::HashMap;

use checkpoint::weightio::WeightReader;
use gpu_core::{DeviceBuffer, Gpu};
use paramstore::upload::Uploader;
use qwen3vl::mrope::{get_rope_index, mrope_tables};

use crate::config::MoeTextConfig;
use crate::thinker::{final_norm, layer_decode_step, layer_fwd, lm_head_fwd, ThinkerLayerCache, ThinkerLayerWeights};
use crate::thinker_plan::{layer_cost, place_fewest_devices, ThinkerPlacement, EMBED_TENSOR};

/// How much of a generation went into RE-READING streamed layer weights, as
/// opposed to running kernels on them.
///
/// `BRAIN_PROFILE`'s per-kernel tables are device timestamps, so they see only
/// the dispatches; on this path most of the wall clock is spent with no kernel
/// running at all, re-reading the layers that did not fit. Without this counter
/// the kernel table reports a small total and silently omits the term that
/// actually dominates - which is how "the GPU is barely busy" gets misread as
/// "the kernels are slow". Aggregated for the process, printed by
/// [`dump_stream_profile`], same as the kernel tables it sits above.
mod stream_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    static LOAD_NS: AtomicU64 = AtomicU64::new(0);
    static LOADS: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);

    pub fn record(ns: u64, bytes: u64) {
        LOAD_NS.fetch_add(ns, Ordering::Relaxed);
        LOADS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    /// `(total load seconds, layer loads, bytes re-read)`.
    pub fn read() -> (f64, u64, u64) {
        (LOAD_NS.load(Ordering::Relaxed) as f64 / 1e9, LOADS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
    }
}

/// Print what streaming cost, then each device's own `BRAIN_PROFILE` kernel
/// table. A resident model's `Gpu` never drops, so the backend's drop-time dump
/// never fires and the tables are unreadable by construction without this - the
/// same escape hatch `fastvlm::caps` uses (`Gpu::dump_profile`).
pub fn dump_stream_profile(gpus: &[Gpu]) {
    if !gpu_core::profile::enabled() {
        return;
    }
    let (secs, loads, bytes) = stream_stats::read();
    if loads > 0 {
        eprintln!(
            "=== omni streamed weights (BRAIN_PROFILE) === {loads} layer load(s), {:.2} GiB re-read, {:.1} s ({:.2} GiB/s)",
            bytes as f64 / (1u64 << 30) as f64,
            secs,
            bytes as f64 / (1u64 << 30) as f64 / secs.max(f64::MIN_POSITIVE)
        );
    }
    for g in gpus {
        g.dump_profile();
    }
}

/// One decoder layer's weights on one device. Held for the whole generation
/// when the layer is resident; built and dropped per use when it is streamed.
pub struct OwnedLayer {
    ln1: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    ln2: DeviceBuffer,
    router: DeviceBuffer,
    experts: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
}

impl OwnedLayer {
    fn as_weights(&self) -> ThinkerLayerWeights<'_> {
        ThinkerLayerWeights {
            ln1: &self.ln1,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            ln2: &self.ln2,
            router: &self.router,
            experts: &self.experts,
        }
    }
}

/// Stream one rank-2 (or rank-1) tensor onto `up`'s device with bounded host
/// use, whatever dtype the checkpoint declares for it: the mapping's bytes are
/// lent zero-copy when they already match, else decoded a chunk at a time.
/// Shape comes from the checkpoint header, so no caller passes it in.
fn load_tensor(up: &mut Uploader, reader: &WeightReader, name: &str) -> Result<DeviceBuffer, String> {
    let numel: u64 = reader.shape(name).ok_or_else(|| format!("missing tensor {name}"))?.iter().product();
    let words = paramstore::dtype::device_f32_words(reader.dtype(name), numel) as usize;
    up.tensor(reader, name, words)
}

/// Build layer `l`'s weights on `up`'s device. Every tensor goes through the
/// bounded uploader - see this module's doc for why nothing here may
/// materialize a whole tensor host-side or use `storage_init`.
fn load_thinker_layer(up: &mut Uploader, reader: &WeightReader, l: u32, n_experts: u32) -> Result<OwnedLayer, String> {
    let p = |leaf: &str| format!("thinker.model.layers.{l}.{leaf}");
    let mut get = |name: String| load_tensor(up, reader, &name);
    let layer = OwnedLayer {
        ln1: get(p("input_layernorm.weight"))?,
        wq: get(p("self_attn.q_proj.weight"))?,
        wk: get(p("self_attn.k_proj.weight"))?,
        wv: get(p("self_attn.v_proj.weight"))?,
        wo: get(p("self_attn.o_proj.weight"))?,
        q_norm: get(p("self_attn.q_norm.weight"))?,
        k_norm: get(p("self_attn.k_norm.weight"))?,
        ln2: get(p("post_attention_layernorm.weight"))?,
        router: get(p("mlp.gate.weight"))?,
        experts: {
            let mut v = Vec::with_capacity(n_experts as usize);
            for e in 0..n_experts {
                v.push((
                    get(p(&format!("mlp.experts.{e}.gate_proj.weight")))?,
                    get(p(&format!("mlp.experts.{e}.up_proj.weight")))?,
                    get(p(&format!("mlp.experts.{e}.down_proj.weight")))?,
                ));
            }
            v
        },
    };
    Ok(layer)
}

/// The placed decoder: which device holds which layers, the resident ones'
/// weights, and the head tensors on the last stage.
///
/// Device identity is an INDEX into the caller's own `&[Gpu]`, never a
/// physical card number this type invented - the same discipline
/// `crate::thinker_plan` follows, and what lets a 1-GPU box, a 2xP40 box and a
/// mixed-capacity box all run this unchanged.
pub struct ThinkerStack {
    placement: ThinkerPlacement,
    resident: HashMap<usize, OwnedLayer>,
    final_norm_w: DeviceBuffer,
    lm_head_w: DeviceBuffer,
}

impl ThinkerStack {
    /// Plan across `gpus` (each with its USABLE byte capacity - a real number
    /// the caller queried, not an assumption) and upload everything the plan
    /// says is resident.
    ///
    /// `gpus` and `caps` are parallel; `caps[i]` is what device `i` may use.
    pub fn build(reader: &WeightReader, cfg: &MoeTextConfig, gpus: &[Gpu], caps: &[u64]) -> Result<ThinkerStack, String> {
        if gpus.is_empty() {
            return Err("omni: no device to place the Thinker on".to_string());
        }
        if gpus.len() != caps.len() {
            return Err(format!("omni: {} device(s) but {} capacities", gpus.len(), caps.len()));
        }
        let cost = layer_cost(reader, cfg).ok_or("omni: checkpoint is missing Thinker tensors this model loads")?;
        let devices: Vec<(usize, u64)> = caps.iter().copied().enumerate().collect();
        let placement = place_fewest_devices(&cost, &devices).ok_or_else(|| {
            format!(
                "omni: the Thinker does not fit the {} budgeted device(s) even streamed ({} bytes/layer at most, {} bytes available)",
                devices.len(),
                cost.per_layer.iter().max().copied().unwrap_or(0),
                devices.iter().map(|&(_, c)| c).sum::<u64>()
            )
        })?;

        let mut resident = HashMap::new();
        for stage in &placement.stages {
            let gpu = &gpus[stage.device];
            // ONE uploader per card for the whole stage: the staging-reclaim
            // accounting has to span every tensor, which is what keeps a
            // multi-GB load from accruing a shadow copy of itself.
            let mut up = Uploader::new(gpu);
            for l in stage.layers.clone() {
                let layer = load_thinker_layer(&mut up, reader, l as u32, cfg.n_experts)?;
                up.drain(&layer.ln1);
                resident.insert(l, layer);
            }
        }
        let last = &gpus[placement.last_device()];
        let mut up = Uploader::new(last);
        let final_norm_w = load_tensor(&mut up, reader, "thinker.model.norm.weight")?;
        let lm_head_w = load_tensor(&mut up, reader, "thinker.lm_head.weight")?;
        up.drain(&final_norm_w);
        eprintln!(
            "omni: thinker placed on {} device(s): {} resident layer(s) {:?}, {} streamed",
            placement.stages.len(),
            placement.stages.iter().map(|s| s.layers.len()).sum::<usize>(),
            placement.stages.iter().map(|s| (s.device, s.layers.clone(), s.bytes >> 20)).collect::<Vec<_>>(),
            placement.streamed.len()
        );
        Ok(ThinkerStack { placement, resident, final_norm_w, lm_head_w })
    }

    /// The placement this stack committed to.
    pub fn placement(&self) -> &ThinkerPlacement {
        &self.placement
    }

    /// Device index running layer `l`.
    fn device_of(&self, l: usize) -> usize {
        self.placement.device_of(l)
    }
}

/// One layer's persistent incremental-decode KV cache, allocated on the SAME
/// device as the layer it belongs to.
struct ThinkerKvCache {
    layers: Vec<(DeviceBuffer, DeviceBuffer)>,
    cap: u32,
}

impl ThinkerKvCache {
    fn new(stack: &ThinkerStack, gpus: &[Gpu], cfg: &MoeTextConfig, cap: u32) -> Self {
        let hkv = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let layers = (0..cfg.n_layers as usize)
            .map(|l| {
                let g = &gpus[stack.device_of(l)];
                (g.storage(cap as u64 * hkv), g.storage(cap as u64 * hkv))
            })
            .collect();
        Self { layers, cap }
    }
    fn layer(&self, l: usize) -> ThinkerLayerCache<'_> {
        ThinkerLayerCache { kcache: &self.layers[l].0, vcache: &self.layers[l].1 }
    }
}

/// What one pass through the decoder stack does at each layer.
enum Pass<'a> {
    /// Batched causal forward over `n` positions, optionally bulk-filling the
    /// KV cache (the prefill half of the decode loop).
    Prefill { cache: Option<&'a ThinkerKvCache>, n: u32 },
    /// A single new token attending against `cache` at row `pos`.
    Decode { cache: &'a ThinkerKvCache, pos: u32 },
}

/// Run layers `0..stop` of the stack over `x_host`, hopping the residual
/// stream host-side wherever the plan changes device, and return the final
/// hidden state as a device buffer ON THE LAST DEVICE TOUCHED (plus that
/// device's index).
///
/// Streamed layers are loaded through the bounded uploader and dropped
/// immediately, with a forced drain so the card actually reclaims them before
/// the next layer allocates - see this module's doc.
fn run_layers(
    stack: &ThinkerStack,
    gpus: &[Gpu],
    reader: &WeightReader,
    cfg: &MoeTextConfig,
    x_host: &[f32],
    cos_tab: &[f32],
    sin_tab: &[f32],
    stop: usize,
    pass: Pass,
) -> (DeviceBuffer, usize) {
    let d = cfg.hidden as usize;
    let rows = x_host.len() / d.max(1);
    let mut h_host: Vec<f32> = x_host.to_vec();
    let mut cur: Option<(usize, DeviceBuffer, DeviceBuffer, DeviceBuffer)> = None; // (device, h, cos, sin)

    for l in 0..stop {
        let dev = stack.device_of(l);
        let need_move = cur.as_ref().is_none_or(|(cd, ..)| *cd != dev);
        if need_move {
            if let Some((cd, h, ..)) = cur.take() {
                h_host = gpus[cd].read(&h, rows * d);
            }
            let g = &gpus[dev];
            let mut up = Uploader::new(g);
            let h = up.host_f32(&h_host);
            let cos = up.host_f32(cos_tab);
            let sin = up.host_f32(sin_tab);
            cur = Some((dev, h, cos, sin));
        }
        let (_, h, cos, sin) = cur.as_ref().expect("a device is selected by now");
        let g = &gpus[dev];

        let mut streamed = None;
        let w = match stack.resident.get(&l) {
            Some(layer) => layer,
            None => {
                let t0 = std::time::Instant::now();
                let mut up = Uploader::new(g);
                let layer = load_thinker_layer(&mut up, reader, l as u32, cfg.n_experts).unwrap_or_else(|e| panic!("omni: {e}"));
                // Charged BEFORE the dispatches that consume it, so this is the
                // re-read cost alone and never overlaps the kernel table. Only
                // under BRAIN_PROFILE: the byte figure re-walks this layer's
                // ~384 tensor headers, which is nothing against a 2.4 GiB read
                // but is not worth paying when nobody reads the number.
                if gpu_core::profile::enabled() {
                    stream_stats::record(t0.elapsed().as_nanos() as u64, crate::thinker_plan::layer_device_bytes(reader, cfg, l).unwrap_or(0));
                }
                streamed = Some((layer, up));
                &streamed.as_ref().expect("just set").0
            }
        };
        let weights = w.as_weights();
        let out = match &pass {
            Pass::Prefill { cache, n } => {
                let lc = cache.map(|c| c.layer(l));
                let (out, ..) = layer_fwd(g, cfg, &weights, h, cos, sin, *n, lc.as_ref(), None);
                out
            }
            Pass::Decode { cache, pos } => layer_decode_step(g, cfg, &weights, &cache.layer(l), h, cos, sin, *pos, cache.cap, None),
        };
        // Replace the residual BEFORE the streamed weights are dropped, then
        // force the submit+wait that lets the card actually reclaim them. Both
        // halves are load-bearing: dropping without draining is exactly the
        // unbounded accumulation that OOM'd a 24 GB card.
        let (cd, _, cos, sin) = cur.take().expect("a device is selected by now");
        cur = Some((cd, out, cos, sin));
        if let Some((layer, mut up)) = streamed {
            drop(layer);
            let (_, h, ..) = cur.as_ref().expect("just set");
            up.drain(h);
        }
    }

    match cur {
        Some((dev, h, ..)) => (h, dev),
        // Zero layers is degenerate but not a panic: upload the input as-is on
        // the last device so the caller's head still has something to apply.
        None => {
            let dev = stack.placement.last_device();
            let mut up = Uploader::new(&gpus[dev]);
            (up.host_f32(&h_host), dev)
        }
    }
}

/// Prefill: the whole prompt through every layer once, bulk-filling `cache`.
/// Returns the final-normed hidden state `[n, hidden]` on the head device.
fn prefill(stack: &ThinkerStack, gpus: &[Gpu], reader: &WeightReader, cfg: &MoeTextConfig, x_host: &[f32], positions: &[[u32; 3]], n: u32, cache: &ThinkerKvCache) -> DeviceBuffer {
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(positions, section, cfg.head_dim, cfg.rope_theta);
    let (h, dev) = run_layers(stack, gpus, reader, cfg, x_host, &cos_tab, &sin_tab, cfg.n_layers as usize, Pass::Prefill { cache: Some(cache), n });
    debug_assert_eq!(dev, stack.placement.last_device());
    final_norm(&gpus[dev], cfg, &stack.final_norm_w, &h, n)
}

/// One incremental decode step: a single new token's embedding row through
/// every layer. `cache_row` is the plain append index; `mrope_pos` is the real
/// 3-axis position, which can be non-monotonic per axis when a media block
/// appeared earlier in the prompt (`qwen3vl::mrope::get_rope_index_multi`).
fn decode_step(stack: &ThinkerStack, gpus: &[Gpu], reader: &WeightReader, cfg: &MoeTextConfig, x_host: &[f32], mrope_pos: [u32; 3], cache_row: u32, cache: &ThinkerKvCache) -> DeviceBuffer {
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&[mrope_pos], section, cfg.head_dim, cfg.rope_theta);
    let (h, dev) = run_layers(stack, gpus, reader, cfg, x_host, &cos_tab, &sin_tab, cfg.n_layers as usize, Pass::Decode { cache, pos: cache_row });
    final_norm(&gpus[dev], cfg, &stack.final_norm_w, &h, 1)
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("non-empty vocab")
}

/// Opt-in (`BRAIN_QWEN3OMNIMOE_DEBUG_LOGITS=1`) top-3 logit dump for one decode step
/// — diagnoses exactly the failure mode `crates/omni/tests/generate_e2e.rs`
/// found on the real checkpoint: a reference comparison (HF's bf16 compute
/// vs. this engine's fp32) diverging at a token whose top candidates are
/// closely spaced, distinguishing "a near-tied logit flipped by accumulated
/// rounding" (small margin between the top few candidates, the wanted token
/// still nearby) from "an actual bug" (a wildly wrong, confidently-argmaxed
/// token). Costs nothing when unset.
fn debug_log_top_candidates(cache_row: u32, logits: &[f32]) {
    if std::env::var("BRAIN_QWEN3OMNIMOE_DEBUG_LOGITS").is_err() {
        return;
    }
    let mut sorted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("decode step cache_row={cache_row}: top3 (token_id, logit) = {:?}", &sorted[..3.min(sorted.len())]);
}

/// Greedy (argmax) text generation. Prefills `prompt_ids` once (populating a
/// KV cache sized `prompt_ids.len() + max_new_tokens`), samples the first new
/// token from the prefill's last logit row, then decodes one token at a time
/// until `max_new_tokens` or an id in `eos_ids`. Returns the FULL sequence
/// (prompt + generated).
///
/// Positions are plain-sequential - the pure-text M-RoPE-collapse case (see
/// `crate::thinker`'s module doc); a caller with an image/audio/video span
/// goes through [`generate_greedy_multimodal`], which carries real per-axis
/// positions.
pub fn generate_greedy(stack: &ThinkerStack, gpus: &[Gpu], reader: &WeightReader, cfg: &MoeTextConfig, embed_table: &EmbedTable, prompt_ids: &[u32], max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
    if prompt_ids.is_empty() {
        return prompt_ids.to_vec();
    }
    let positions = get_rope_index(prompt_ids, u32::MAX, &[]); // plain-sequential: no placeholder id ever matches u32::MAX
    generate_greedy_with_embeds(stack, gpus, reader, cfg, embed_table, prompt_ids, None, &positions, max_new_tokens, eos_ids)
}

/// The shared implementation behind [`generate_greedy`] and
/// [`generate_greedy_multimodal`]: `x_host_override`, when `Some`, is the
/// prompt's embedding buffer verbatim (already spliced with media); when
/// `None` it is built by a plain per-token gather.
#[allow(clippy::too_many_arguments)]
fn generate_greedy_with_embeds(
    stack: &ThinkerStack,
    gpus: &[Gpu],
    reader: &WeightReader,
    cfg: &MoeTextConfig,
    embed_table: &EmbedTable,
    prompt_ids: &[u32],
    x_host_override: Option<Vec<f32>>,
    positions: &[[u32; 3]],
    max_new_tokens: u32,
    eos_ids: &[u32],
) -> Vec<u32> {
    let mut ids: Vec<u32> = prompt_ids.to_vec();
    assert_eq!(positions.len(), prompt_ids.len(), "generate_greedy: positions/prompt_ids length mismatch");
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return ids;
    }
    let head_gpu = &gpus[stack.placement.last_device()];

    let n0 = prompt_ids.len() as u32;
    let cap = n0 + max_new_tokens;
    let cache = ThinkerKvCache::new(stack, gpus, cfg, cap);

    let x_host = x_host_override.unwrap_or_else(|| {
        let mut x = Vec::with_capacity(prompt_ids.len() * cfg.hidden as usize);
        for &t in prompt_ids {
            x.extend_from_slice(&embed_table.row(reader, t));
        }
        x
    });
    let hidden = prefill(stack, gpus, reader, cfg, &x_host, positions, n0, &cache);
    let logits = lm_head_fwd(head_gpu, &stack.lm_head_w, &hidden, n0, cfg.hidden, cfg.vocab);
    let last_row = head_gpu.read(&logits, (n0 * cfg.vocab) as usize);
    let mut next = argmax(&last_row[((n0 - 1) * cfg.vocab) as usize..(n0 * cfg.vocab) as usize]);
    ids.push(next);
    let mut cache_row = n0;
    // New tokens are always plain text: continue the diagonal from the
    // prompt's last position + 1 on every axis.
    let mut mrope_pos = positions[positions.len() - 1].map(|p| p + 1);

    if !eos_ids.contains(&next) {
        for _ in 1..max_new_tokens {
            let x_row = embed_table.row(reader, next);
            let hidden = decode_step(stack, gpus, reader, cfg, &x_row, mrope_pos, cache_row, &cache);
            let logits = lm_head_fwd(head_gpu, &stack.lm_head_w, &hidden, 1, cfg.hidden, cfg.vocab);
            let row = head_gpu.read(&logits, cfg.vocab as usize);
            next = argmax(&row);
            debug_log_top_candidates(cache_row, &row);
            ids.push(next);
            cache_row += 1;
            mrope_pos = mrope_pos.map(|p| p + 1);
            if eos_ids.contains(&next) {
                break;
            }
        }
    }
    ids
}

/// Re-run a full KNOWN sequence (prompt + already-generated ids, teacher-
/// forced -- no sampling) through the Thinker's first `capture_layer + 1`
/// layers, returning that layer's raw (pre-final-norm) hidden state `[n, d]`
/// -- what `crate::talker_prompt`'s Thinker->Talker prefill assembly needs
/// (`TalkerConfig::accept_hidden_layer`). A second pass rather than capturing
/// during the incremental decode loop: the capture is needed for EVERY
/// position (prompt and generated alike), while decode only ever computes one
/// NEW position per step. Early-exits after `capture_layer`.
pub fn thinker_hidden_at_layer(stack: &ThinkerStack, gpus: &[Gpu], reader: &WeightReader, cfg: &MoeTextConfig, x_host: &[f32], positions: &[[u32; 3]], n: u32, capture_layer: u32) -> Vec<f32> {
    assert!(capture_layer < cfg.n_layers, "capture_layer {capture_layer} out of range ({} layers)", cfg.n_layers);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(positions, section, cfg.head_dim, cfg.rope_theta);
    let stop = capture_layer as usize + 1;
    let (h, dev) = run_layers(stack, gpus, reader, cfg, x_host, &cos_tab, &sin_tab, stop, Pass::Prefill { cache: None, n });
    gpus[dev].read(&h, (n * cfg.hidden) as usize)
}

/// The multimodal entry: `prompt.x_host` already has media spliced in
/// (`crate::mm::build_multimodal_prompt`), so no embedding gather is needed
/// for the prompt itself - only for tokens generated after it (always plain
/// text).
pub fn generate_greedy_multimodal(stack: &ThinkerStack, gpus: &[Gpu], reader: &WeightReader, cfg: &MoeTextConfig, embed_table: &EmbedTable, prompt: &crate::mm::MultimodalPrompt, max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
    generate_greedy_with_embeds(stack, gpus, reader, cfg, embed_table, &prompt.token_ids, Some(prompt.x_host.clone()), &prompt.positions, max_new_tokens, eos_ids)
}

/// How this model reads one embedding row.
///
/// The table is `[vocab, hidden]` - 1.2 GB as f32 at the real Thinker shape -
/// and generation only ever needs a per-token ROW gather, never a GEMM. So it
/// is neither uploaded nor expanded whenever the mapping can lend its bytes:
/// rows are read straight from the mapping and converted one row at a time.
/// Mirrors `crate::int8_thinker_resident::EmbedTable`, for the raw-HF dtypes.
pub enum EmbedTable {
    /// The mapping lends `F32` words directly: a row is a slice of the mmap.
    Mapped { k: usize },
    /// The mapping cannot lend this dtype as words (`BF16`/`F16`/a GGUF
    /// quant): one host copy, decoded once at load. Correct, just not free.
    Host { table: Vec<f32>, k: usize },
}

impl EmbedTable {
    /// Open the checkpoint's token embedding, without expanding it when the
    /// mapping allows a borrow.
    pub fn open(reader: &WeightReader) -> Result<EmbedTable, String> {
        let shape = reader.shape(EMBED_TENSOR).ok_or_else(|| format!("omni: missing {EMBED_TENSOR}"))?;
        if shape.len() != 2 {
            return Err(format!("omni: {EMBED_TENSOR} must be rank-2, got {shape:?}"));
        }
        let k = shape[1] as usize;
        if reader.dtype(EMBED_TENSOR) == Some("F32") && checkpoint::TensorSource::raw_words(reader, EMBED_TENSOR).is_some() {
            return Ok(EmbedTable::Mapped { k });
        }
        let table = reader.tensor(EMBED_TENSOR).ok_or_else(|| format!("omni: missing {EMBED_TENSOR}"))?;
        Ok(EmbedTable::Host { table, k })
    }

    /// The row width (`hidden`).
    pub fn hidden(&self) -> usize {
        match self {
            EmbedTable::Mapped { k } | EmbedTable::Host { k, .. } => *k,
        }
    }

    /// Token `t`'s embedding row, `[hidden]`.
    pub fn row(&self, reader: &WeightReader, t: u32) -> Vec<f32> {
        let t = t as usize;
        match self {
            EmbedTable::Mapped { k } => {
                let words = checkpoint::TensorSource::raw_words(reader, EMBED_TENSOR).expect("EmbedTable::Mapped implies a lendable mapping");
                words[t * k..(t + 1) * k].iter().map(|w| f32::from_bits(*w)).collect()
            }
            EmbedTable::Host { table, k } => table[t * k..(t + 1) * k].to_vec(),
        }
    }

    /// The already-materialized host copy, when there is one - so a caller
    /// that needs the whole table (`crate::mm`'s splice assembly) borrows it
    /// instead of decoding a second 1.2 GB copy.
    pub fn as_host_slice(&self) -> Option<&[f32]> {
        match self {
            EmbedTable::Host { table, .. } => Some(table),
            EmbedTable::Mapped { .. } => None,
        }
    }

    /// The whole table as f32 - for the callers that genuinely need a full
    /// host copy (`crate::mm`'s splice assembly, `crate::talker_prompt`).
    /// Decodes on demand from the mapping when the table is not already held.
    pub fn to_host(&self, reader: &WeightReader) -> Vec<f32> {
        match self {
            EmbedTable::Host { table, .. } => table.clone(),
            EmbedTable::Mapped { .. } => reader.tensor(EMBED_TENSOR).expect("EmbedTable::Mapped implies the tensor exists"),
        }
    }
}
