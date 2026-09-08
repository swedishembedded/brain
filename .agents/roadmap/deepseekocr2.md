# deepseekocr2 - roadmap

DeepSeek-OCR-2: the successor to DeepSeek-OCR (`crates/deepseek2ocr`,
`.agents/roadmap/deepseek2ocr.md`), released 2026-01-27, Apache 2.0. Same
document-image-in, text/markdown-out task. The decoder is unchanged from v1;
the vision front end ("DeepEncoder V2" / "Visual Causal Flow") is new: SAM
ViT-B feeding a 24-layer Qwen2 encoder run as a learned-query resampler under
a prefix-LM attention mask, then a single linear projector - replacing v1's
SAM -> 16x conv compressor -> CLIP-L/14 -> concat arrangement.

Plan: see the session plan this ledger was opened from (12 milestones,
M0-M12). This file tracks facts and decisions as milestones land; it does not
duplicate the plan's implementation detail.

## Facts pinned from the real GGUF headers (not assumed)

Checkpoint used for fact-pinning: the community Q8_0 conversion pair (LM +
mmproj) built against llama.cpp PR #20975 (`deepseekocr2` mtmd support, not
yet merged upstream as of this writing). No official `deepseek-ai`-published
GGUF exists; the reference safetensors checkpoint is `deepseek-ai/DeepSeek-OCR-2`.

### LM (`deepseek2-ocr` architecture)

- **`general.architecture = "deepseek2-ocr"` - IDENTICAL STRING to v1's LM
  GGUF.** This is not a naming accident: the decoder is genuinely the same
  architecture family, byte-for-byte the same config
  (`deepseek2-ocr.{attention.head_count,head_count_kv}=10/10` - plain MHA,
  `block_count=12`, `embedding_length=1280`, `expert_count=64`,
  `expert_used_count=6`, `expert_shared_count=2`,
  `expert_feed_forward_length=896`, `feed_forward_length=6848`,
  `leading_dense_block_count=1`, `vocab_size=129280`,
  `rope.dimension_count=0` - same full-head_dim NEOX rope quirk v1 already
  documents). Tensor names and shapes match v1's decoder exactly: dense
  `blk.0.{attn_k,attn_norm,attn_output,attn_q,attn_v,ffn_down,ffn_gate,ffn_norm,ffn_up}`,
  MoE `blk.1..11.{..., ffn_down_exps[896,1280,64], ffn_down_shexp[1792,1280],
  ffn_gate_exps[1280,896,64], ffn_gate_inp[1280,64], ffn_gate_shexp[1280,1792],
  ffn_up_exps[1280,896,64], ffn_up_shexp[1280,1792]}`. 155 tensors total
  (`3 + 9 + 11*13`), confirming no hidden extra tensor. Same tokenizer
  (`tokenizer.ggml.pre = "deepseek-v3"`, 129280 tokens, same reserved-token
  scheme) and no `scoring_func`/`topk_method`/`norm_topk_prob` KV keys, same
  as v1 (llama.cpp's compiled-in defaults for this arch apply unchanged).
- **Consequence: `crates/gguf/src/deepseek_ocr.rs` reads this file
  unmodified.** v2's decoder needs no new importer, no new gradcheck, no new
  LoRA wiring - `crates/deepseek2` carries over as-is.
- **The collision is real but narrow: it is a `brain_arch::by_gguf` routing
  question, not a decoder-import question.** `crates/arch/src/lib.rs:778-779`
  asserts `by_gguf` is a one-to-one map (one arch string -> one registry id).
  Since v2's LM GGUF's `general.architecture` string is indistinguishable
  from v1's, the GENERIC single-file path (`brain import <file.gguf>`, and
  any `arch!()` row's `gguf:` field keyed on this string) cannot tell the two
  models' LM files apart from the string alone - it would resolve to
  whichever id claims `"deepseek2-ocr"` first. **Decision: `deepseekocr2`'s
  `arch!()` row leaves its LM-side `gguf:` discriminator unclaimed (only
  `deepseek2ocr` claims `"deepseek2-ocr"`); `deepseekocr2` is reached through
  its own composite import (`crates/deepseekocr2/src/import.rs`, reading
  `LM`/`MMPROJ` by role from a directory, exactly like v1's `import.rs` -
  never through `by_gguf`) and through its vision-side discriminator
  instead** (below). A generic `brain import` of a bare v2 LM GGUF with no
  mmproj alongside it is expected to resolve as v1's decoder, correctly - the
  two are the same tensors under the same name, and importing the LM alone
  discards nothing meaningful either model needs from it.

### Vision (`clip` architecture, `clip.projector_type=deepseekocr2`, mmproj file, 473 tensors)

- **`clip.projector_type = "deepseekocr2"` - the unambiguous discriminator**
  (distinct from v1's `"deepseekocr"`). This is what `deepseekocr2`'s
  `arch!()` row's vision-side `gguf:` field should key on, mirroring how v1's
  own vision importer already discriminates on `clip.projector_type` rather
  than `general.architecture="clip"` (shared by every mmproj in the repo).
- SAM tower: **byte-identical to v1** - `clip.vision.sam.{block_count=12,
  embedding_length=768, head_count=12}`, `clip.vision.window_size=14`. Real
  tensor names/shapes confirm the same per-block layout v1 already imports:
  `v.sam.blk.N.{pre_ln,post_ln}.{weight,bias}` (both are LayerNorm, weight
  AND bias present - not RMSNorm), fused `attn.qkv.{weight[768,2304],bias}`,
  decomposed relative position `attn.pos_h.weight[64,27]` /
  `attn.pos_w.weight[64,27]`, `mlp.lin1.weight[768,3072]` /
  `mlp.lin2.weight[3072,768]`. **`crates/sam1` is reused with zero change.**
- **`downsample_channels` in `config.json` (`[512,1024]`) is CONFIRMED STALE.**
  Real tensors: `v.sam.net_2.weight[3,3,256,512]` (256->512, stride 2),
  `v.sam.net_3.weight[3,3,512,896]` (512->**896**, stride 2). 896 is the
  Qwen2 hybrid encoder's hidden size, not 1024 - the neck's final width feeds
  the encoder directly, with no intermediate projection. A 1024x1024 view
  (64x64 SAM patches, `image_size=1024`, `patch_size=16`) downsamples 4x
  spatially to a 16x16=**256**-token grid at width 896; a 768x768 tile
  (48x48 patches) downsamples to 12x12=**144** tokens at width 896.
- **`clip.use_gelu = true` is an inert converter artifact**, same shape as
  v1's `feed_forward_length` bug: the Qwen2 hybrid encoder uses SiLU
  (SwiGLU, `ffn_gate`/`ffn_up`/`ffn_down` triples per block, confirmed by
  the real tensor names `v.blk.N.ffn_{gate,down,up}.weight`), and llama.cpp's
  mtmd graph builder for this projector type does not read this key. Do not
  propagate it into the config.
- **The new Qwen2 hybrid encoder** - real tensors confirm, per block:
  `ln1.weight`/`ln2.weight` (RMSNorm - weight ONLY, no bias, unlike SAM's
  LayerNorm blocks), `attn_q.{weight[896,896],bias[896]}`,
  `attn_k.{weight[896,128],bias[128]}`, `attn_v.{weight[896,128],bias[128]}`
  (128 = 2 KV heads x 64 head_dim - **GQA 14/2 confirmed**, matching
  `clip.vision.attention.{head_count=14,head_count_kv=2}`; **qkv bias
  present, no q/k norm tensors anywhere - matches `QwenConfig::qwen2`'s
  shape exactly, not `qwen3`'s**), `attn_out.weight[896,896]`,
  `ffn_gate/up.weight[896,4864]`, `ffn_down.weight[4864,896]`
  (`clip.vision.feed_forward_length=4864`, matching the reference's
  `Qwen2Config(intermediate_size=4864)`). `clip.vision.block_count=24`
  confirmed. RMSNorm eps `clip.vision.attention.layer_norm_epsilon ~= 1e-6`.
  No `rope_theta`/`max_position_embeddings` KV present - as with v1's
  `rope.dimension_count=0` quirk, llama.cpp's compiled-in Qwen2 default
  applies unread from this file. Cross-checked against the merged mtmd
  graph builder for this projector type (`tools/mtmd/models/
  deepseekocr2.cpp`, landed on llama.cpp master 2026-05-29, PR #20975): its
  rope dispatch for the resampler pins **theta 1e6, NEOX (half-split)
  layout**, matching `Qwen2Config(rope_theta=1e6)`. The same source confirms
  the mask is built as a separate `[seq_len, seq_len]` f32 buffer supplied
  to the graph rather than computed inside the block itself - consistent
  with the earlier finding that the hybrid mask is assembled host-side
  before the encoder runs - and that the sequence order is **image tokens
  first, learned queries appended after**, with the view separator appended
  once more, conditionally, at the end of each view's sequence.
- **Learned query embeddings, real shapes**: `v.resample_query_768.weight
  [896,144]` (the 144-token/768-tile bank) and `v.resample_query_1024.weight
  [896,256]` (the 256-token/1024-global bank) - both present, both named
  after the reference's `query_768`/`query_1024`, GGUF's dim order reversed
  from the reference's `nn.Embedding(n_query, 896)`.
- **`v.view_seperator` is `[1280, 1]` - 1280-wide, the DECODER's hidden size,
  not the encoder's 896.** This settles an ordering question the plan left
  open: the per-view projector (`mm.model.fc`, `[896,1280]`) must run BEFORE
  the separator is appended, i.e. concatenation happens in 1280-dim
  projected space, not before projection. `mm.model.fc.{weight,bias}`
  confirmed as a single Linear(896->1280) - matches the reference's `linear`
  `MlpProjector` exactly, no MLP.
- **A final tower-wide norm exists and was not in the original architecture
  sketch: `v.post_ln.weight [896]`** - a weight-only (RMSNorm) tensor applied
  once, after all 24 encoder blocks and before the query slice/projector.
  Brings the real tensor count to exactly 473 (`24*12 + 1 + 12*14 + 11 + 5`);
  M1's classifier maps it to `vision.encoder.norm.weight`.
- **Still open, deliberately not guessed:** how MULTIPLE local tiles order
  themselves relative to each other and to the global view. The graph
  builder confirms one view's internal order (image tokens, then queries,
  then an optional separator) and the HF reference confirms one tile's
  placeholder layout, but neither source pinned here settles the sequence
  across several tiles in a >1x1 grid. Settle this at M5 against a real
  forward or a real dump, the same way v1's own row layout was settled -
  not from reading either source's code in isolation.

## Status

M0 (resources + fact-pinning) done. M2 (checkpoint-free tiny golden for the
new vision tower) done: `tools/goldens/deepseekocr2_dump_reference.py`
dumps `testdata/deepseekocr2/tiny/{ckpt/model.safetensors,golden.safetensors}`
+ `manifest-tiny.json` - the query-concat, the shared Qwen2 GQA prefix-LM
encoder (taps per view: concat input, per-layer pre/post-mask scores,
softmax probs, layer output, query slice, projector output), and the final
row-gather (local tiles row-major, then the global view, then the
separator), consumed by `crates/deepseekocr2/tests/tiny_ref.rs` (M3).

M1 (GGUF import for the mmproj) done: `crates/gguf/src/deepseekocr2_vision.rs`
classifies all 473 real mmproj tensors - SAM reuses `deepseek_ocr_vision::
SamConfig` verbatim under the same `vision.sam.*` names `crates/sam1` already
reads, the new Qwen2 resampler lands under `vision.encoder.*` plus
`vision.query_{local,global}` and `vision.projector.fc.*`/
`vision.view_separator` - with full two-way coverage proven against the real
checkpoint (`crates/gguf/tests/deepseekocr2.rs`). The LM half needed no new
code (see above). Registry wiring (`crates/arch`, `crates/cli/src/
gguf_import.rs`'s `IMPORTERS` table, the modelstore recipe) is deferred to
M7, once `crates/deepseekocr2` exists for `check-arch-names.sh` to point at.

M3 (the resampler's forward pass) done: new crate `crates/deepseekocr2`
(`config::{Qwen2EncoderConfig, DeepseekOcr2VisionConfig}`,
`encoder::Resampler`). Composes SAM's already-produced per-view token grid
(taken as a plain host slice - `sam1::SamEncoder` itself is not invoked here,
matching M2's golden's own scope) with the matching learned query bank, runs
the shared 24-Qwen2-block GQA tower under the prefix-LM mask, applies the
final norm, slices the query half, projects. The mask composes with the
mask-agnostic MHA `attn_scores_bidir`/`attn_softmax_bidir`/`attn_apply_bidir`
family with `attn_prefix_mask` dispatched in between (the
`crates/moondream3` precedent), fed by `model::block::kv_expand_fwd`
widening GQA's 2 KV heads to 14 (the `crates/lfm2` precedent) - zero new
kernels, as the plan's decisive finding predicted.

Found and fixed on the spot: M2's golden was missing the real model's final
tower-wide norm (`v.post_ln.weight` / `vision.encoder.norm.weight`, the
tensor M1 discovered mid-flight while M2 was already running in parallel and
had no way to see) - added to the dumper and regenerated, rather than
building M3's encoder to match an incomplete fixture.

`tests/tiny_ref.rs` passes on the real detected GPU (Intel Arc, Vulkan/wgpu
backend, not a CPU-backend stub): every tap - concat input, per-layer
pre/post-mask scores, softmax probs, layer output, query slice, projected
output, for all 6 local tiles and the global view, plus the host-side
row-gather - lands at cosine >= 0.999999 against the checkpoint-free golden.
`scores_post_mask` is checked by allow/disallow PATTERN rather than raw
value (the reference's `-1e9` sentinel and `attn_prefix_mask.wgsl`'s `-1e30`
both underflow probs to an exact 0.0 in fp32, but are not the same number,
so a plain cosine check on that one tap would not be a trustworthy signal
either way). The mutation check (inverting the mask's two arms) fails the
pattern check as required, proving the gate is not hollow.

One real wgpu constraint surfaced and got fixed during this milestone, worth
recording for the next one: `add2` (three distinct buffer args) cannot serve
a residual accumulation with the SAME buffer passed as both an input and the
output - wgpu's storage-binding usage-scope validation rejects a buffer
bound read-write in one slot and read in another within the same dispatch.
`add_inplace` (`out[i] += a[i]`, one read_write buffer) is the correct
kernel for every `x = x + delta` residual add in this tower, and probably in
any future one built the same way.

Registry wiring, the real `SamEncoder` composition, the device-side
row-splice into the decoder, and CLI/serving are still M5-M7's job.

M4 (backward + gradcheck) done: every dispatch in `encoder.rs`'s forward now
has a mirrored backward (`layer_bwd`, plus the final-norm/projector backward
in `Resampler::backward_train`), reached via `forward_train`/`backward_train`
- a training-shaped pair distinct from `resample_view`'s inference/tap
shape, sharing the same per-layer dispatch (`layer_fwd` now also returns the
retained device buffers backward needs, snapshotted by host round-trip at
both residual points since `x` is mutated in place twice per layer).
`attn_prefix_mask` needed no backward of its own, as the plan predicted -
the softmax jacobian over the already-masked `probs` carries the right
(~0) gradient into every masked entry, the same reasoning `crates/moondream3`
already established. `gradcheck::check_deepseekocr2`
(`crates/gradcheck/src/deepseekocr2.rs`) is a bespoke `CheckModel` harness,
not the blanket `model::Model` impl - the resampler has no natural batch or
loss of its own, so the harness supplies a fixed random linear probe against
the projected output and bridges `loss()`/`backward()`'s two-call contract
with a `RefCell<Option<TrainState>>`. Covers every trainable tensor AND the
SAM-token-grid input (needed for a real SAM+encoder joint fine-tune later).

**One real bug found and fixed by the gradcheck itself**, worth recording:
the first cut wired `kv_expand_bwd`'s dispatch through `ids.kv_expand` (the
FORWARD copy kernel) instead of the real `kv_expand_bwd.wgsl`/`KV_EXPAND_BWD`
pipeline - two kernels with adjacent names in this crate's own `PIPELINES`
list, easy to conflate. Symptom was exactly what a wrong-kernel dispatch
should look like: `attn.k.weight`/`attn.v.weight`/`attn.v.bias` came back
with an analytic gradient of EXACTLY `0.0` against a clearly nonzero
numeric one (`report.dead_gradients()`'s signature), while `norm1.weight` -
downstream of the same broken `d_xn1` fan-in - showed a real but
four-orders-of-magnitude-too-small analytic value. Registering
`kv_expand_bwd` as its own `Ids` field and pipeline entry fixed every
parameter at once; no other kernel wiring needed a change. Recorded here
because the failure signature (a specific parameter family reading exactly
zero) is the fast way to recognize this class of bug again.

M5 (row layout + composite splice) done: `crates/deepseekocr2/src/rows.rs`
(`Src`/`TileGrid`/`RowPlan`/`row_plan`, pure host index math, unit-tested in
isolation) formalizes the row order M2/M3 already assumed - local tiles
row-major width-first, then the global view, then one separator row.
**Still a design assumption, not an empirical fact**: which order MULTIPLE
local tiles take relative to each other and to the global view is pinned
nowhere in the real header or the merged graph builder for more than one
tile, so `row_plan`'s formula is what M6 must check against a real forward
or a real dump, not the other way around.

The device-side splice `deepseek2ocr2::model`'s brief called for turned out
to need no device dispatch at all: v2's row layout has no interleaving (no
`image_newline`), so every view's projected rows are already one contiguous
run, and the assembly `gather_rows` (already built in M3) does is exactly
what `Vec::extend_from_slice` gives for free - the adjoint added here,
`encoder::scatter_rows`, is a plain split for the same reason. This is a
real simplification versus v1's `RowGather` (`crates/deepseek2ocr/src/
layout.rs`), which needs device buffers and an index-gather kernel
specifically because MANY rows there read the same `image_newline` vector;
v2's separator is read by exactly ONE row, so its own gradient is that row
directly (`Resampler::write_view_separator_grad`), never a sum.

New composite `crates/deepseekocr2/src/model.rs` (`DeepseekOcr2`) splices
the resampler's gathered block into `deepseek2::DeepseekV2` via that
crate's existing, UNMODIFIED `enable_mm_splice`/`write_img_embeds`/
`read_d_img_embeds` seam - the same one v1's composite uses, crossing the
two towers as a host `Vec<f32>` so they need not share a device or backend.
`tests/composite.rs` gates: `gather_rows`/`scatter_rows` are exact host
inverses; the spliced separator row is bit-identical to the checkpoint
tensor at the row plan's last position (exact equality, not cosine, since
it is a direct copy); a projector row lands exactly where its run says it
should; a full end-to-end directional finite-difference check through
`forward`+`backward` (tile 0's SAM input perturbed, decoder loss compared)
- analytic vs numeric agree at rel 1.6e-4; and a mutation check that
swapping two tiles' order changes the assembled block, so the row order
itself is under test, not assumed. SAM's real `SamEncoder` is still not
invoked (M6/M7's job, per M3's scope note).

Confirmed pre-existing, unrelated to this milestone (matches M1's own
finding almost exactly): `make check/scripts` fails on `check-env-docs.sh`
- 9 undocumented `BRAIN_*` vars across `flux2`/`ltxv`/`qwen35`/the hub
endpoint, none read anywhere in this crate. `make clippy` fails on 6
pre-existing warnings in `crates/model/src/adapter/*`,
`crates/model/tests/adapter_lora_fa.rs`, `crates/cli/src/resolver_cli.rs`,
and `crates/flux2/tests/resolve_layout.rs` - `cargo clippy -p
brain-deepseekocr2 --all-targets` is itself warning-free. `make gradcheck`
(the full workspace suite, not just this crate) is green.

Remaining milestones (M6-M12) not started.
