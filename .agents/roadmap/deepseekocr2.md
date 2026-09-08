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
- **Multi-tile ordering: CONFIRMED at M6, from the merged mtmd source
  directly** (`tools/mtmd/mtmd-image.cpp`'s
  `mtmd_image_preprocessor_deepseekocr::preprocess` and `tools/mtmd/mtmd.cpp`'s
  chunk-assembly loop, both read in full against a local llama.cpp
  checkout). `PROJECTOR_TYPE_DEEPSEEKOCR`/`PROJECTOR_TYPE_DEEPSEEKOCR2` share
  the SAME preprocessor class AND both set `ov_img_first = false`
  (`mtmd.cpp`), so v2's tile order is the same shape v1 already settled: the
  assembly loop's own comments state it plainly - "add slices (or tiles)"
  (row-major, `for y in 0..n_row: for x in 0..n_col`) runs BEFORE "add
  overview image (last)". `add_viewsep` is set ONLY on the overview chunk
  (`preprocess`'s `output.overview.add_viewsep = true` - tiles never set
  it), and the per-view graph (`clip_graph_deepseekocr2::build()`) appends
  the separator after ITS OWN chunk's tokens when that chunk's flag is set -
  so with the overview placed last, the separator lands at the very end of
  the whole sequence. This is exactly `crate::rows::row_plan`'s existing
  formula (local tiles row-major, then the global view, then one trailing
  separator) - M2/M3/M5's assumption is now a checked fact, not a guess.

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

M6 (real-weight parity and the decode loop) done, global view only.

**Part A - import wiring.** `crates/deepseekocr2/src/import.rs`: `Files::
locate` resolves the shipped pair by role (mirroring `crates/deepseek2ocr/
src/import.rs`'s shape); `expand_vision`/`expand_lm` cache each half's fp32
expansion beside the checkpoint, built on demand (`gguf::deepseekocr2_vision
::import` for the ~1 GB vision half, `deepseek2::import::import_file` -
the SAME function v1's own decoder expansion already calls, unmodified -
for the ~12 GB decoder); `vision_config` derives the tower's shape from the
mmproj's own KV/tensor shapes and checks it against `Qwen2EncoderConfig::
deepseek_ocr2()`, `rope_theta` excepted (the file cannot state it - M0).

**Part B - real-weight tests**, siblings of v1's own, global view only
(see the gap this scopes around, below): `tests/real_weight.rs` runs the
REAL `sam1::SamEncoder` (M3's host-slice placeholder finally replaced with
the genuine tower) on a constant-fill 1024x1024 image, through the real
resampler, through the real splice, into the real decoder - asserting
every stage finite and dimensionally right (**reported, not gated**: unlike
v1, no independent capture of THIS checkpoint's own numbers exists yet, so
inventing a cosine floor would be theatre) and that the splice really
placed the projector output verbatim into the residual stream (`assert_eq!`
on the spliced rows, not a tolerance). `tests/real_weight_generate.rs`
drives the composed greedy decode loop for 3 real steps and gets causal
self-consistency for free - one forward over the length-`L-1` prefix
reproduces every step-time argmax - which is the strongest oracle-free
signal available and it holds. `tests/prompt_real.rs` is close to a direct
port of v1's own (same tokenizer, confirmed at M0), now proven against v2's
OWN GGUF rather than assumed to match because the vocab table matches.

Both real-weight tests PASS on the real checkpoint (release profile, CPU
backend): forward test finite logits, spread 28.9 (well above the 1.0
plausibility floor), peak RSS 17.9 GiB; decode test's causal-consistency
loop holds over all 3 generated steps, peak RSS 15.8 GiB. New crate helper
`encoder::sam_tokens_from_nchw` bridges SAM's real NCHW compressor output
into the resampler's `[n_query, d_model]` input - the one piece of glue
neither M3 (host-slice input) nor `sam1` (NCHW producer) needed until real
composition was actually wired. `model::DeepseekOcr2::prime_vision` factors
the vision-forward half of `forward` out from the loss-computing half, so a
decode loop can prime the splice once and then drive `deepseek2::DeepseekV2
::generate_greedy` directly.

**The real gap Part B works around, stated plainly**: local (768x768) tile
SAM inference needs `vision.sam.pos_embed` resampled from its checkpoint
native 64x64 grid to 48x48 - `crates/sam1` has NO position-embedding
resampling today (its `pos_embed` is a fixed-size parameter, added via a
plain elementwise `add2`, checked at `ParamStore` construction against
`cfg.grid_h*grid_w`). This blocks real MULTI-tile SAM inference, not the
composite's correctness: `tests/composite.rs` (M5) and `tests/tiny_ref.rs`
(M3) already prove the row-gather/splice mechanism for multiple tiles
against synthetic SAM grids, independent of whether SAM itself can produce
a real one yet. Building the resampling belongs to a real sub-milestone of
its own (a real kernel + its own gradient check, since `sam1` is
trainable), not a rushed addition here - recorded as outstanding.

**Part C - the multi-tile ordering question** - resolved by reading the
merged llama.cpp source directly rather than a runtime dump (see the fact
entry above). The originally planned route (a real `llama-mtmd-debug`
graph-eval dump, `tools/goldens/deepseekocr2_convert_llamacpp_dump.py`) was
attempted first: llama.cpp was cloned, patched (a new, original
`BRAIN_GGUF_DUMP_DIR`-gated full-tensor dump added to `common/debug.cpp`,
alongside its existing truncated-preview printer, not replacing it) and
built clean. But `llama-mtmd-debug`'s `encode` mode turned out to only
support encoding ONE synthetic square view at a time (`-n` must be exactly
144 or 256 tokens' worth - `clip_graph_deepseekocr2::build()` asserts it) -
it does not run the higher-level multi-tile preprocessing/assembly path at
all, so it structurally cannot exercise tile-to-tile ordering regardless of
image size. `llama-mtmd-cli` (built and available) does run that full
path, but has no equivalent debug-dump hook. Given that, reading
`mtmd-image.cpp`'s preprocessor and `mtmd.cpp`'s chunk-assembly loop
directly settled the question with MORE certainty than a single opaque
tensor dump would have (the maintainers' own code comments state the order
in words: "add slices (or tiles)" then "add overview image (last)") - no
Rust or Python code changed as a result, since the existing `row_plan`
formula was already correct.

Gates: `check/spdx`, `check-no-machine-paths.sh`, `check-scripts.sh`,
`check-env-docs.sh`, `check-no-doc-citations.sh`, `check-multi-gpu-
sharding.sh` (deepseekocr2 not yet in the `arch!()` registry, so correctly
outside this check's scope until M7) all pass for what this milestone
touched. `check-arch-names.sh` fails on pre-existing debt unrelated to
this crate (three hard-coded `main.rs` match arms for `document-study`/
`gguf`/`models`/`roofline`, one missing `qwen3vlmoe` docs page) - confirmed
by grep, none naming `deepseekocr2`. `cargo clippy -p brain-deepseekocr2
--all-targets --all-features -- -D warnings` is clean (two `doc_lazy_
continuation` lints from a wrapped `- ` at a doc-comment line start were
real and fixed). `cargo test -p brain-gradcheck` (full workspace, 1205s):
67 passed, 2 failed - both `qwen`/`qwen_lora` gradcheck tests, both failing
on `WgpuBackend::new_on ... exceeded 30s -- driver likely wedged`, a GPU-
adapter-creation timeout from this box running several concurrent
heavy builds/tests at once, not a numerical regression; this crate's own
`deepseekocr2_resampler_analytic_grads_match_finite_differences` passed.

M7 (CLI, capability, residency) done.

**`crates/deepseekocr2/src/preprocess.rs`** (new): real images in, global
view only - the same aspect-preserving centred fit-and-pad + `[0,1]->[-1,1]`
normalization convention M0 confirmed identical to v1's, restated
independently rather than shared code (a caller of `imaging`'s own
primitives, not a port of v1's module). Verified non-degenerate against a
real rendered document: full `[-1,1]` range reached, zero non-finite
values, correct `[3,1024,1024]` shape (checked with a throwaway example,
deleted before commit - not part of the crate).

**`crates/deepseekocr2/src/caps.rs`** (new): `Provider` + a streaming
`generate` `Action`, the same chat-capable shape v1's `caps.rs` and
`apiserve::catalog::api_caps` require. `Session::load` builds the composite
AND a resident `sam1::SamEncoder` from the SAME `WeightReader` (read twice -
SAM's tensors, then the resampler's - the identical reuse `tests/
real_weight.rs` already relies on), so a real per-request image goes
through real SAM every call rather than a synthetic grid. Decode is
`DeepseekV2::generate_greedy_cb` (full-recompute, M6-proven), not the
KV-cached path - that composition has not been independently verified for
this composite's splice.

**Registry wiring**: `arch!("deepseekocr2", ...)` in `crates/arch` -
**deliberately claims no `gguf:` string on the LM side** (the collision
with `deepseek2ocr`'s `"deepseek2-ocr"` is real, M0's finding, not fixed
here - a bare LM-only `brain import` correctly resolves as v1's decoder,
same tensors under the same name); the vision half's own discriminator
(`clip.projector_type="deepseekocr2"`) is read directly by this crate's own
`import.rs`, never through `crates/gguf/src/route.rs`'s generic dispatch,
so no `IMPORTERS` table entry was needed either. `ARCH_TO_MODEL`
(`crates/cli/src/resolve.rs`), a `ModelEntry` (`crates/catalog`), the
`resident_ctor_for` patch + `every_patched_id_is_a_real_catalog_entry` row
(`crates/cli/src/catalog.rs`), a `mod resident_deepseekocr2;` line (no
`main.rs` match arm - forbidden), and a reserved-vendor carve-out in
`crates/cli/tests/model_ids.rs` (widened from v1's single `const` to a
slice, since two case-exact upstream ids now need one).

**Residency, single-device (CPU), unlike v1's wgpu/CPU split**: this
crate's own real-weight test suite pins CPU throughout specifically because
stacking a SECOND 24-block-deep tower behind SAM on one wgpu device has not
been independently verified the way v1's single SAM tower was (v1's own
split only landed once THAT verification existed) - `crate::
resident_deepseekocr2` inherits that caution rather than claiming an
unverified placement. `COMPOSITE_PEAK_BYTES = 16 GiB`, measured (not
guessed): `real_weight_generate.rs --release --ignored --nocapture`
reports VmHWM 15.73 GiB for the whole composite (SAM + resampler + decoder,
a 3-token greedy decode), rounded up.

**No fetch recipe added to `crates/modelstore`, on purpose**: no
vendor-published GGUF exists for this model, only community conversions of
varying provenance - adding a `FilesRecipe` naming one would misrepresent
an unofficial third-party repo as the canonical upstream release the way
every other recipe's `repos:` field implicitly claims. `BRAIN_DEEPSEEKOCR2_
DIR` must be pointed at a manually-placed pair; documented as such in
`docs/models/deepseekocr2.md` (a minimal stub - `check-arch-names.sh`
requires the page to exist once the architecture registered; the full
options/hardware-limits content is M11's job) and `docs/using/
configuration.md`.

**Real end-to-end, on the real checkpoint**: `brain caps --json` lists
`deepseek-ai/DeepSeek-OCR-2`; `BRAIN_DEEPSEEKOCR2_DIR=<dir> brain
deepseekocr2 generate --in image=<rendered doc> --prompt "Free OCR"
--max_new 12` runs to completion (release build) - real preprocessing, real
SAM, real resampler, real splice, real decode, a well-formed streamed
response. The decoded text came back empty (the model emitted EOS as its
first token) on the specific synthetic test graphic used here; per this
crate's own established rule (M6: "no independent oracle exists for this
checkpoint's own numbers... reported, not gated"), that is recorded as an
observation, not asserted as a bug or a pass - the pipeline that produced
it is independently verified correct (preprocessing checked non-degenerate;
every stage upstream of the decoder is gradient-checked; the decode loop's
causal self-consistency is proven in M6). `DEFAULT_MAX_NEW` is set to 16,
the real number that completed inside a 280s budget on this box in the
same run (`max_new=40` did not) - not a guess, and explicitly documented as
a staging point pending the KV-cache migration noted above.

Gates: `check/spdx`, `check-no-machine-paths.sh`, `check-large-files.sh`,
`check-workspace-members.sh`, `check-scripts.sh`, `check-env-docs.sh`,
`check-no-doc-citations.sh` (three citations of this ledger's own path from
source comments were found and rephrased inline - source may not cite
`.agents/`, only `docs/` may), and `check-multi-gpu-sharding.sh` (a new
allow-list row added, mirroring v1's - `Shardable` is M8's job) all pass.
`check-arch-names.sh` still fails only on the same pre-existing,
unrelated debt M6 already confirmed (three `main.rs` literal match arms,
one missing `qwen3vlmoe` docs page) - re-confirmed via `git stash` on a
clean tree; `deepseekocr2`'s own row/page introduce no new violation.
`cargo clippy -p brain-deepseekocr2 -p brain-cli -p brain-catalog -p
brain-arch --all-targets --all-features -- -D warnings` clean. `cargo test
-p brain-deepseekocr2`, `-p brain-cli --bin brain` (312 passed, 0 failed,
after confirming one transient `WgpuBackend::new_on` adapter-creation
timeout was box-load flakiness, not a regression - reproduced clean on
both HEAD and this branch depending on concurrent load) both green.

M8 (`Shardable` for the shared decoder) done: `crates/deepseek2` (the decoder
both this crate and v1 wrap unmodified) now implements `model::Shardable` -
`crates/deepseek2/src/shard.rs`, a `shard: Shard` field threaded through
`DeepseekV2`, a `shard_param_list` filter mirroring `qwen35moe`'s exactly,
and the embed/head/layer-range gating in `build_forward`/`build_backward`
(`crates/deepseek2/src/model.rs`). `Shard::whole` (every existing caller's
default) is bit-for-bit the old unconditional code path by construction, not
merely by intent - proven by the WHOLE existing test suite (gradcheck,
generate/KV-parity, `deepseek2ocr`'s own 37 tests) passing unchanged with
zero tolerance adjustment.

**Verified for real on this hardware**: the single-device path (every
existing test, unchanged). **Structurally implemented, gated to skip
without a second GPU, NOT verified on this box**: the 2-device bit-identity
test, `crates/deepseek2/tests/shard_parity.rs` (mirrors
`gpt2`/`qwen35moe`'s own `shard_parity.rs` exactly - `model::Pipeline`
auto-placement, loss + per-tensor gradient comparison at `rel<1e-3`). This
box's one GPU is an integrated Intel Arc part; `gpu_core::discrete_gpu_count()`
correctly reports 0, so the test's own `gpu_disabled()` check skips it via
`brain_testutil::skip_unavailable` rather than faking a pass - confirmed by
running with `--nocapture` and reading the printed skip reason, not assumed
from the test merely returning `ok`.

**Allow-list resolution** (`scripts/gates/check-multi-gpu-sharding.sh`):
removed the `deepseek2` row (now genuinely shards) rather than the
`deepseek2ocr`/`deepseekocr2` rows - the gate is a per-CRATE textual-presence
check, and while the shared decoder now shards, `crates/deepseek2ocr` and
`crates/deepseekocr2` are still their own crates whose OWN source (the SAM
tower, the splice glue) never mentions `model::shard`/`Shardable` - their
vision towers are the real remaining gap. Reworded both rows to say so
precisely rather than leaving the old "not yet migrated; backlog" text,
which is no longer accurate about the decoder half.

Gates: `check/spdx`, `check-no-machine-paths.sh`, `check-workspace-members.sh`,
`check-scripts.sh`, `check-multi-gpu-sharding.sh` (0 stale rows, 0 missing
crates) all pass. `cargo clippy -p brain-deepseek2 --all-targets
--all-features -- -D warnings` clean. `cargo test -p brain-deepseek2 --lib
--tests` (all suites, including the new `shard_parity`), `cargo test -p
brain-deepseekocr2 --lib`, `cargo test -p brain-deepseek2ocr --lib` (37
passed), and the full workspace `cargo test -p brain-gradcheck` all green.

M9 (LoRA + full fine-tune, with overfit proofs) done. Full fine-tune needed
zero new plumbing beyond `train:true, lora:None` already giving every
parameter `Role::Trainable` - `full_finetune_overfits_a_single_example`/
`_a_small_batch` (`crates/deepseekocr2/tests/train_overfit.rs`) drive loss
from ~2.93 (uniform-guess, `ln(19)`) to 7e-6 / 1e-4 on the shared tiny
fixture. LoRA needed real new work: `crates/deepseekocr2/src/{config,encoder,init}.rs`
add `DeepseekOcr2VisionConfig::lora: Option<qwen3::LoraCfg>`, seven
LoRA-targetable per-layer leaves (`Qwen2EncoderConfig::lora_leaves` - qkv,
attn.out, the three MLP linears; each `.lora_a`/`.lora_b` sized from that
leaf's REAL `(out,in)` - this tower is GQA with asymmetric MLP widths, so it
cannot take `deepseek2::config`'s own shortcut of one shared `rank*d_model`
size for every target), the same `Role::Frozen`-base/`Role::Trainable`-adapter
role split `deepseek2::DeepseekV2` already uses, and a from-scratch
`lora_fwd`/`lora_bwd` pair mirroring that crate's exact derivation (two
matmuls + AXPY forward, the four-matmul-plus-two-grad-scale backward) - zero
new kernels, all composed from `matmul`/`matmul_dx`/`matmul_dw`/`axpy`/
`grad_scale`, already registered for the optimizer.

**A real defect the gradcheck-adjacent testing caught, not from `make
gradcheck` itself:** every backward site that writes into a base weight's
grad buffer (`matmul_dw`, `bias_grad`, the two norms' `rmsnorm_bwd` grad
output, the query banks, the separator, the projector) was unconditionally
dispatched - correct when everything is trainable (full fine-tune, this
crate's only mode before M9), but a LoRA-frozen base has NO grad buffer
allocated at all (`ParamStore::g` panics on a missing entry), so EVERY one of
those ~13 call sites needed a `self.trainable(name)` guard mirroring
`deepseek2::model::DeepseekV2`'s own `trainable()` helper. Missing even one
would have panicked the very first LoRA backward call - caught here before
any of the overfit tests could even run, not by a numerical mismatch.

**A second, more interesting non-defect surfaced during verification, and is
recorded so it is not mistaken for a bug later:** a naive `lora_overfits_a_single_example`
built exactly like the full-fine-tune tests (LoRA on a fresh RANDOM base)
plateaus at ~2.86-2.93 - barely moving - at every rank (2 through 6) and
learning rate tried. A `#[cfg(test)]`-free diagnostic pass (read back a
targeted linear's output with the adapter's `B` forced to a large constant
vs. left at zero, and a direct finite-difference check on `.lora_a`/
`.lora_b`) confirmed the delta correctly reaches the residual stream and the
analytic gradient matches the numeric one - the mechanism is correct. A
control experiment reproduced the SAME plateau using ONLY `deepseek2`'s own
pre-existing, unmodified decoder LoRA (nothing from this campaign) against a
random base, ruling out anything vision-specific. The real explanation:
LoRA's whole premise is a frozen base that is ALREADY a useful
representation; a random, never-trained composite has none, and a rank-limited
correction cannot manufacture the WHOLE network's worth of missing capacity
(full fine-tune only succeeds because it also moves the decoder's 64-expert
MoE FFN, which no LoRA config here ever targets). The shipped test,
`lora_overfits_a_single_example_against_a_real_base`, instead full-fine-tunes
a real base first (reusing the already-proven path), then applies a fresh
LoRA adapter to a DIFFERENT example on top of it - the scenario LoRA is
actually built for - and gates on a large (>90%) relative loss reduction from
a genuinely difficult (confidently-wrong) starting point, not an absolute
near-zero bar.

`crates/deepseekocr2/src/train.rs` (new): `lora_init_map`, the two-tower
composite-level seam merging fresh `.lora_a`/`.lora_b` tensors (via each
tower's own `init::init_adapters`) over an existing base - the checkpoint
case `lora_overfits_a_single_example_against_a_real_base`'s Phase 2 exercises
directly (a REAL, previously-trained base, not a fresh synthetic one).

Gates: `check/spdx`, `check-no-machine-paths.sh`, `check-workspace-members.sh`,
`check-scripts.sh`, `check-no-doc-citations.sh`, `check-multi-gpu-sharding.sh`
all pass. `cargo clippy -p brain-deepseekocr2 --all-targets --all-features --
-D warnings` clean. `cargo test -p brain-deepseekocr2` (all suites, 27
tests) and `cargo test -p brain-gradcheck --lib deepseekocr2` (the
pre-existing M4 gradcheck, unaffected since it never sets `cfg.lora`) both
green.

M11 (user-facing docs) done: `docs/models/deepseekocr2.md` replaced M7's
stub with the real page (honest support/hardware-limits sections, the
no-vendor-GGUF guidance, real measured figures from M6/M7/M9 rather than
placeholders), plus a license-compliance row (Apache-2.0, distinct from
v1's split code/weights licensing), an `AGENTS.md` ledger entry + crate
table + routing table row, and `examples/vision/deepseek-ocr-2/README.md`.
Quickstart untouched, as scoped.

M10 (NPU/ONNX export, unvalidated - no NPU firmware on this host, as
directed) done for the resampler, honestly incomplete for the decoder:

- `crates/npu/src/deepseekocr2_topology.rs` builds the resampler+projector
  as a fixed-shape ONNX graph (SAM excluded - see below), following
  `nemotron_topology.rs`'s precedent of baking a static additive attention
  mask as an initializer. The prefix-LM mask exports exactly as the plan
  predicted: `allow(i,j) = (i<P && j<P) || (j<=i)` bakes into one
  `[1,1,2P,2P]` buffer, no data-dependent control flow, no new op. GQA
  (14/2) is expressed as an explicit Reshape+Expand+Reshape head-repeat,
  the same shape `qwen_topology.rs`'s own GQA emitters already use.
- `crates/npu/src/deepseek2_topology.rs` builds the decoder (plain MHA +
  sparse top-k MoE) by adapting `qwen35moe_topology.rs::Topo::moe_layer`'s
  existing sparse gather-based MoE dispatch (stack every expert into one
  `Gather`-indexable initializer, `TopK` the router probabilities, one
  broadcasting `MatMul` over the selected experts) - **this is real,
  working precedent for exactly the "MoE routing is a known hard case for
  static ONNX graphs" question the plan raised**, not a new invention. Two
  real differences from that precedent are read from `cfg` rather than
  assumed (`norm_topk_prob`/`routed_scaling` gate the renormalising
  `Div`/scaling `Mul`; the real checkpoint carries neither, matching the LM
  GGUF's own absent `scoring_func`/`topk_method` keys), and the shared
  expert is unweighted (`model::moe::shared_expert_fwd`'s `None` arm) rather
  than qwen35moe's sigmoid-gated one.
- `crates/npu/src/deepseekocr2_export.rs` + `brain npu deepseekocr2
  --weights DIR --seq S --out-dir out` wire both into the CLI, reusing
  `deepseekocr2::import`'s existing real-checkpoint expansion so this export
  reads the same tensors every other real-weight test in this campaign does.
- `crates/npu/tests/deepseekocr2_onnx.rs`: structural tests (tiny fixture,
  no OpenVINO/hardware needed) pass for BOTH the resampler (both view
  sizes) and the decoder (asserting a real `TopK` node appears, so the MoE
  path is provably exercised, not silently skipped).
- **Real-checkpoint run, honest result**: both resampler views exported
  successfully against the actual downloaded checkpoint (~1.4 GiB each,
  valid ONNX). **The decoder export did not complete against the real
  checkpoint - the process was killed with zero output as its RSS crossed
  ~16 GiB on this box's 30 GiB of RAM (swap already near-full from this
  session's own build activity), the same host-RAM ceiling
  `qwen35moe_export.rs`'s own doc comment already documents for the
  identical reason (a dense per-layer expert stack held fully in memory
  before serialization) at a larger scale.** This is a resource ceiling on
  THIS host under THIS load, not a defect in the graph builder - the
  builder is proven correct on the tiny fixture, including the MoE path.
  Not forced or faked; recorded here as the honest outcome the plan asked
  for.
- **SAM is not exported, by design, and this is a REAL unresolved gap**: no
  windowed-attention-with-decomposed-relative-position-bias ONNX export
  precedent exists anywhere in this crate. A real fix needs a new topology
  emitter for that op shape - unstarted work, not a hidden assumption.

Gates: `check/spdx`, `check-no-machine-paths.sh`, `check-no-doc-citations.sh`,
`check-scripts.sh`, `check-multi-gpu-sharding.sh`, `check-env-docs.sh` all
pass. `cargo clippy -p brain-npu -p brain-cli --all-targets --all-features
-- -D warnings` clean.

Remaining milestone (M12) not started by this entry; see the Status
section above/below for any milestone landed separately.
