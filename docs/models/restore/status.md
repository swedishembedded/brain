# CodeFormer face restoration (`crates/restore`) — status

**Scope reached: goldens → import → FORWARD parity, per stage, at five `w`.**
Backward/gradcheck is deferred; the **serving contract is met** (`restore::caps`
`restore_face`, `crates/cli/src/resident_restore.rs`, D-Bus `Run`,
`examples/restore/`) apart from a batched `run_batch`. Both are at the bottom.

`crates/vqgan` already owns the VQ autoencoder (encoder / 1024×256 codebook /
generator) and is parity-gated on its own goldens — see
`docs/models/vqgan/status.md`. **None of it is re-implemented here.** This crate
adds exactly the three things that turn that autoencoder into a
degradation-robust restorer:

1. the 9-layer **code-prediction Transformer** that predicts codebook *indices*
   instead of looking up the nearest neighbour;
2. the **controllable feature transformation** (`Fuse_sft_block`) at four
   resolutions;
3. the **fidelity dial `w`**.

---

## Files

| File | What |
|---|---|
| `tools/codeformer_restore_dump_reference.py` | reference goldens, forward hooks, CPU/fp32/seed 1234, per-`w` |
| `crates/restore/src/config.rs` | `CodeFormerConfig`, the CFT tap table, the 515-tensor checkpoint manifest and the 533-tensor runtime manifest |
| `crates/restore/src/import.rs` | two-way coverage validation + the one boundary split (`in_proj_weight` → `qk` \| `v`) |
| `crates/restore/src/model.rs` | the forward graph (`KERNELS`, `CodeFormer`, `record_transformer`, `record_fuse`) |
| `crates/restore/tests/parity.rs` | the parity ladder |
| `testdata/restore/codeformer/` | goldens (gitignored, 2.1 GB) |

Goldens are regenerated with:

```bash
python3 tools/codeformer_restore_dump_reference.py \
  --code    <CodeFormer repo> \
  --weights <dir with codeformer.pth> \
  --out     testdata/restore/codeformer \
  --face    <CodeFormer>/inputs/cropped_faces/0143.png
```

Tests: `BRAIN_RESTORE_WEIGHTS=<dir with codeformer.pth> cargo test --release -p
brain-restore`. Every test skips itself when its fixture or the checkpoint is
absent. `BRAIN_RESTORE_DEVICE=cpu` runs the CPU JIT.

---

## Architecture notes (settled against the reference source, not guessed)

* `CodeFormer` **subclasses** `VQAutoEncoder`, so the encoder/generator/codebook
  tensor names and block schedule are `crates/vqgan`'s, index for index. The
  released `codeformer.pth` is 515 tensors: 329 VQGAN + 186 CodeFormer.
* **The Transformer layer is pre-norm and the position embedding is asymmetric**:
  `tgt2 = norm1(tgt); q = k = tgt2 + position_emb; v = tgt2`. q/k and v therefore
  read *different* inputs, which is why the fused `in_proj_weight[3E, E]` cannot
  be dispatched whole and is split at import into `qk[2E, E]` and `v[E, E]`.
  That split is also what the attention kernels want:
  `attn_scores_bidir` reads q and k out of ONE buffer at `qkv_stride = 2E`
  (`q_off = 0`, `k_off = E`), and `attn_apply_bidir` reads v out of its own at
  stride `E` — no strided sub-region writes, no offset views in the hot path.
* Attention is **bidirectional**, 8 heads, `head_dim = 64`, `1/√64` scaling —
  `nn.MultiheadAttention` with no mask.
* The MLP activation is `F.gelu`'s **default (erf) form**, not the tanh
  approximation: `gelu_erf`, not `gelu`.
* `idx_pred_layer = Sequential(LayerNorm(512), Linear(512, 1024, bias=False))` —
  the head linear has **no bias** (asserted in the manifest test).
* `topk(softmax(logits), 1)` == `argmax(logits)` because softmax is monotone;
  the port dispatches `argmax_row` and the dumper asserts the two agree.
* `quant_feat` is `get_codebook_feat` — the **raw** codebook gather, not the
  straight-through `z + (z_q − z)` form `quantize.forward` returns.
* CFT taps: encoder blocks `{5,8,11,14}` → generator blocks `{9,12,15,18}` for
  the released `connect_list = ['32','64','128','256']`. The table is
  transcribed from `codeformer_arch.py`, and a config test cross-checks every
  one of the six declared taps against the block schedule's own channel and
  resolution bookkeeping, so a transcription slip fails loudly rather than
  fusing the wrong pair of features.

### The dial `w` — direction, and why it is gated three ways

`Fuse_sft_block.forward` is `out = dec + w · (dec · scale(e) + shift(e))`, and
the reference **skips the block entirely when `w == 0`**.

* **`w = 0`** — no encoder feature reaches the generator; the image is the
  reconstruction of the *predicted codes* alone → maximum **quality**.
* **`w = 1`** — the full CFT residual → maximum **fidelity** to the input face.

The upstream README says the same in words ("smaller *w* tends to produce a
higher-quality result, while larger *w* yields a higher-fidelity result"), and
`inference_codeformer.py` defaults to `-w 0.5`.

An inverted dial is invisible in any single-`w` comparison and visible only to a
human looking at faces, so it is gated three ways:

1. **per-`w` stage parity** against goldens dumped at `w ∈ {0, 0.25, 0.5, 0.75, 1}`;
2. **`w = 0` is exactly the identity** — the port evaluates the fuse and scales
   by zero where the reference branches around it, and the test asserts
   `max|pre_fuse − post_fuse| == 0` (not "small") at all four taps;
3. **monotone drift** — `max|out(w) − out(0)|` must be non-decreasing in `w`, on
   the port's own outputs. The dumper asserts the same on the reference's, and
   the two agree to the printed digits (see the table below).

`w` lives in a **one-element device buffer** read by `scale_add`, not baked into
a recorded step, so changing it is a buffer write, not a graph rebuild.

---

## Forward parity — measured

Hardware: one **Tesla P40** (`backend-wgpu`, Vulkan), release build. Gates:
cosine ≥ 0.9999 **and** relative L2 ≤ 3e-3 (cosine alone is scale-invariant, so
a uniformly-2× stage would still report 1.000000000). Every number below is the
**worst stage in its group**.

### Encoder + code-prediction Transformer

| case | predicted-index mismatches | worst stage | cosine | 1 − cos | rel L2 |
|---|---|---|---|---|---|
| `face` (real aligned 512² face, 201/1024 distinct codes) | **0 / 256** | `logits_norm` | 1.000000000 | 2.99e-12 | 2.42e-6 |
| `synth` (deterministic pattern, 125/1024 distinct codes) | **0 / 256** | `ft.08` | 1.000000000 | 3.79e-12 | 2.73e-6 |

22 comparisons per case: the four CFT encoder taps, `lq_feat`, `feat_emb`, five
sub-taps inside layer 0 (`norm1`, `attn_out`, `norm2`, `linear1`, `linear2`),
all nine layer outputs, `logits_norm` and `logits`.

Selected `face` rows:

| stage | cosine | 1 − cos | max\|Δ\| | rel L2 |
|---|---|---|---|---|
| `enc.05` (256², 128 ch) | 1.000000000 | 1.19e-12 | 8.11e-6 | 1.21e-6 |
| `enc.14` (32², 256 ch) | 1.000000000 | 1.69e-12 | 3.53e-5 | 1.76e-6 |
| `lq_feat` | 1.000000000 | 6.13e-13 | 7.87e-6 | 1.06e-6 |
| `feat_emb` | 1.000000000 | 3.59e-13 | 3.19e-5 | 8.28e-7 |
| `ft.00.attn_out` | 1.000000000 | 3.16e-13 | 7.87e-6 | 7.21e-7 |
| `ft.08` (last layer) | 1.000000000 | 2.49e-12 | 2.02e-4 | 2.19e-6 |
| `logits` | 1.000000000 | 1.44e-12 | 5.63e-5 | 1.63e-6 |

### Generator + CFT, per `w`

`face`, worst stage in each group (23 comparisons at `w > 0`, 7 at `w = 0`
because the reference dumps no fuse internals where it branches around them):

| `w` | worst stage | cosine | 1 − cos | rel L2 | `output` cosine | `output` max\|Δ\| |
|---|---|---|---|---|---|---|
| 0.00 | `gen.18` | 1.000000000 | 4.84e-12 | 2.88e-6 | 1.000000000 | 1.57e-5 |
| 0.25 | `fuse.256.scale` | 1.000000000 | 5.45e-11 | 1.04e-5 | 1.000000000 | 1.79e-5 |
| 0.50 | `fuse.256.scale` | 1.000000000 | 5.62e-11 | 1.05e-5 | 1.000000000 | 3.30e-5 |
| 0.75 | `fuse.256.scale` | 1.000000000 | 5.73e-11 | 1.07e-5 | 1.000000000 | 2.00e-5 |
| 1.00 | `fuse.256.scale` | 1.000000000 | 6.07e-11 | 1.10e-5 | 1.000000000 | 1.98e-5 |

`synth` is the same shape; its worst is `fuse.256.scale` at 1 − cos 7.38e-11,
rel L2 1.21e-5 (at `w = 1`).

`quant_feat` is **bit-exact** (max\|Δ\| 0.0) at every `w` in both cases — the
codebook gather is a pure indexed copy and the indices match exactly.

### The dial, measured

`max|out(w) − out(0)|` — reference (dumper) and port agree to every printed
digit:

| case | w=0 | w=0.25 | w=0.5 | w=0.75 | w=1 |
|---|---|---|---|---|---|
| `face` | 0.0000 | 0.9444 | 1.4420 | 1.5193 | 1.6321 |
| `synth` | 0.0000 | 2.1283 | 2.8225 | 3.4448 | 3.7374 |

At `w = 0` the four fuse blocks are **bit-identical** to the identity
(`max|pre − post| == 0.0`), so the reference's `if w > 0` branch and the port's
"evaluate and scale by zero" are the same function, not merely close.

### Cross-backend: the CPU JIT

`BRAIN_RESTORE_DEVICE=cpu` (`backend-cpu`, WGSL → Cranelift, 48 threads), `face`:

| group | worst stage | cosine | 1 − cos | rel L2 |
|---|---|---|---|---|
| encoder + transformer | `enc.05` | 0.999999994 | 5.70e-9 | 1.08e-4 |
| generator + CFT, `w = 0.5` | `gen.24` | 0.999999876 | 1.24e-7 | 4.98e-4 |

> These CPU figures are **not reproducible digit-for-digit across runs**: the
> `backend-cpu` reduction order depends on how the rayon pool splits the work,
> so a re-run varies within the band. An independent re-run of the same test
> measured `enc.05` 0.999999994 / 1.075e-4 (identical) but `gen.24`
> 0.999999913 / 4.167e-4 rather than 0.999999876 / 4.98e-4. Treat the CPU
> column as an order of magnitude with a floor, not as a fingerprint — the GPU
> numbers above ARE reproducible and were re-measured identical.

Still **0 / 256** predicted-index mismatches, and the dial's drift matches the
GPU/reference to four digits (0.9442 / 1.4421 / 1.5195 / 1.6323 vs
0.9444 / 1.4420 / 1.5193 / 1.6321). The ~100× wider error band is the *same*
documented cause as `crates/vqgan`'s 512² CPU numbers — the CPU JIT cannot run
`gn_stats_wg` (two workgroup barriers), so every GroupNorm falls back to
`gn_stats`, which sums a group's up-to-16 M elements as one serial ascending run
instead of a 256-way tree. Summation order, not a port defect. The 3e-3 relative
floor keeps ~6× headroom over the worst backend.

### Production path

`pooled_matches_tapped`: the `taps = false` graph (activation pool ON, buffers
aliased) is **bit-identical** to the `taps = true` graph at `w = 0.5` —
identical indices, `max|Δ| == 0` on the image — and a second `generate()` at the
same `w` reproduces it exactly, which is what proves the four encoder features
that live *across* the two submits are never handed out as generator scratch.

---

## Reuse

Everything below was reused, not rewritten. **This crate adds no kernel and no
block.**

| Needed | Reused |
|---|---|
| encoder / generator block walk | `vqgan::model::run_blocks` (now segment-capable, see "shared changes") |
| conv / GroupNorm / SiLU / ResBlock / AttnBlock / upsample / asymmetric downsample | `vae::blocks::Builder` with `BlockNames::vqgan` |
| the `Fuse_sft_block.encode_enc` ResBlock | the same `Builder::resnet` — it *is* a VQGAN `ResBlock(2C→C)` |
| codebook gather | `vqgan::model::record_lookup` (`embed`) |
| checkpoint wrapper strip | `vqgan::import::strip_state_prefix` |
| ResBlock tensor naming in the manifest | `vqgan::config::block_tensors` |
| LayerNorm (+ the coalesced `layernorm_rows`, resolved by name) | `model::block::layernorm_fwd` / `LayerNormIds::resolve_fwd` |
| GEMM kernel choice | `model::block::pick_gemm` |
| bidirectional attention | `attn_scores_bidir` / `attn_softmax_bidir` / `attn_apply_bidir` — already in `vae::blocks::KERNELS`, resolved BY NAME |
| erf-GELU, LeakyReLU, Hadamard, channel concat, row argmax, strided copy, scalar-scaled accumulate | `gelu_erf`, `leaky_relu`, `mul`, `concat2`, `argmax_row`, `region_copy`, `scale_add` |

`restore::KERNELS` is `vqgan::KERNELS` (which is `vae::blocks::KERNELS` plus the
VQ trio) copied verbatim by a `const fn`, plus twelve appended slots. Two unit
tests assert the prefix is byte-identical and that every appended slot holds the
kernel its constant names.

`scale_add` is the MoE expert-combine kernel; at
`[seq_len=1, d_model=N, n_experts=1, e_idx=0, accumulate=1]` it degenerates to
`acc[i] += gate[0]·src[i]`, i.e. exactly "add `w` times the CFT residual", with
`w` read from a device buffer. That is why the dial needs no new kernel and no
graph rebuild.

---

## Shared files touched — REVIEW THESE

Three **additive** changes to `crates/vqgan`, all to avoid a second copy of code
that already exists:

1. `crates/vqgan/src/model.rs` — `run_blocks` went from `pub(crate)` to `pub`
   **and gained a `start: usize` parameter** (the global index of `blocks[0]`),
   so a caller can walk the same flat `nn.ModuleList` in *segments* and keep the
   checkpoint's positional tensor names correct. Both in-crate call sites
   (`model.rs`, `train.rs`) pass `0`. This is the only signature change.
2. `crates/vqgan/src/config.rs` — `block_tensors` is now `pub`.
3. `crates/vqgan/src/import.rs` — `strip_state_prefix` is now `pub`.

Nothing else outside `crates/restore/**`, `tools/`, `testdata/` and this file
was modified. `crates/vqgan` and `crates/restore` are both clippy-clean.

`AGENTS.md` — **done in adversarial review**: §12c no longer claims the
CodeFormer transformer and `w` are unimplemented, a §12c-bis entry describes
this crate, and the routing table gained a `restore` row.

---

## Found in adversarial review

The parity ladder was re-run independently and reproduces **every GPU number
above to the printed digit** (worst `1 − cos` 6.07e-11 at `fuse.256.scale`,
`w = 1`, `face`; drift `0 / 0.9444 / 1.4420 / 1.5193 / 1.6321`), 0/256 index
mismatches in both cases, on GPU and on the CPU JIT. Two real defects that the
parity gate structurally could not see were found and fixed; neither moved a
parity digit (re-measured identical after the fix).

1. **`generate()` / `code_logits()` read never-written buffers without
   complaining.** Both read device buffers that only `predict_codes()` writes
   (the four encoder features and the logits). On a freshly constructed
   `CodeFormer` they had never been written, so `generate()` returned a
   finite, plausible image computed from unwritten memory instead of failing —
   the "silently wrong, not a crash" class. The contract was documented on the
   method but not enforced. Fixed with an `AtomicBool` (kept `Sync` for the
   serving contract, which has since landed) set by `predict_codes` and asserted by both
   readers, plus a regression test in `pooled_matches_tapped` that asserts both
   entry points panic on a fresh model.
2. **~47 MB of dead VRAM pinned on the production path.**
   `vqgan::model::run_blocks` documents that a segment's *input* buffer stays
   the caller's, and the encoder deliberately holds its segment outputs (the
   CFT reads them much later). The **generator** loop was holding them too even
   though nothing reads a generator segment input again, so all four post-fuse
   tensors (32²·256 + 64²·256 + 128²·128 + 256²·128 floats) plus `lq_feat`
   stayed allocated for the life of the graph on the `taps = false` path.
   Fixed by returning them to the activation pool once the segment that reads
   them is recorded, guarded by an assert that a segment can never be empty
   (an empty segment returns its own input, which would make the free a
   use-after-free). `free` is a no-op in the tapped build, so the taps the
   parity ladder reads are unaffected — and `pooled_matches_tapped` still
   reports `max|Δ| == 0`, which is the gate on exactly this change.

Not defects, but recorded because they were checked: the fused `in_proj` split
matches what `torch.nn.functional._in_projection_packed` does when `q is k` but
`k is not v` (it takes the `w.chunk(3)` branch, so rows `0..2E` really are q|k);
the 1/√64 scale, the head-major `h·head_dim` layout, `dim_mlp = 2·dim_embd`,
LayerNorm eps 1e-5, `LeakyReLU(0.2)`, erf-GELU, both fuse-tap dictionaries and
the `w` direction were each re-derived from
`basicsr/archs/codeformer_arch.py` and agree; the golden manifest's checkpoint
sha256 matches the file on disk; and `crates/vqgan`'s own 20 tests still pass
after the `run_blocks` signature change.

---

## Stubbed / not implemented

* **`adain=True`.** `inference_codeformer.py` calls the model with
  `adain=True`, which adaptively instance-normalises `quant_feat` toward
  `lq_feat`'s per-channel statistics before the generator. The port implements
  the `adain=False` path (the architecture's own default) and the goldens were
  dumped with it. The reference's `calc_mean_std` uses torch's **unbiased**
  variance (`/(N−1)`) plus `eps = 1e-5` *inside* the sqrt, which no existing
  brain reduction reproduces exactly — `gn_stats` is population variance. This
  is the one place the CLI path diverges from what is gated, and it needs
  either a small dedicated kernel or a host reduction over a 256×256 tensor.
  A re-dump with `--adain` is required either way.
* **Face detection / alignment.** Not wired. `crates/facenet`'s SCRFD +
  5-point alignment is the right source, but its similarity transform targets
  the **ArcFace 112² template**, whereas CodeFormer crops to the **FFHQ 512²
  template** (`facexlib`'s `face_template`, a different 5-point target and a
  different scale). Wiring facenet's existing template would silently mis-frame
  every face — worse than not wiring it — so the pipeline needs
  `facenet::align` parameterised by the target template, which is a change to
  another component's crate. `CodeFormer::restore` takes an already-aligned
  512×512 face in `[-1, 1]`.
* **Background upsampling / paste-back** (Real-ESRGAN, `--bg_upsampler`): out of
  scope, a separate model.
* **Batch > 1.** The graph is built for one image; `bsz` is a literal 1 in every
  attention dispatch.
* **Sizes other than 512².** `position_emb` is a `[256, 512]` parameter with no
  interpolation, so the model is fixed to a 16×16 latent. `CodeFormer::new`
  asserts it rather than silently mis-indexing.
* **CLI / capability / residency / D-Bus**: none. `brain restore …` does not
  exist yet.

---

## What the deferred work needs

### Backward + gradcheck

* `vae::blocks::grad` already has the adjoints for the whole convolutional half
  (conv, GroupNorm, SiLU, add, upsample, the bidir attention quartet), and
  `vqgan::train` already stitches the VQ straight-through seam. The generator /
  encoder / `encode_enc` backward is therefore **available today**: build both
  `Builder`s with `set_train(true)` and append `vae::blocks::BWD_KERNELS`.
  Note `set_train(true)` also forces the *direct* conv and per-element attention
  lowerings, so the training forward is NOT the graph gated above — it needs its
  own cosine-1.0 check against the inference forward, as `flux2` does.
* The Transformer's backward needs the kernels `crates/clip` already registers
  and dispatches for the same shape of layer: `ln_stats(_rows)`,
  `layernorm_dx(_rows)`, `layernorm_dgamma/dbeta`, `matmul_dx(_reg)`,
  `matmul_dw(_reg)`, `bias_grad`, `gelu_erf_bwd`, and the
  `attn_bwd_{dscores,dv,dq,dk}_bidir` quartet. All exist. The only piece with no
  adjoint in the tree is the **`qk` / `v` split**: `d_in_proj` is the
  concatenation of `d_qk` and `d_v`, and `d(tgt2)` accumulates the q/k branch
  *and* the v branch while `d(position_emb)` takes only the q/k branch — a
  `region_copy` + `axpy` stitch, not a kernel.
* The CFT backward needs `leaky_relu_bwd` (exists), `mul`'s own backward (it
  composes from `mul`, per its header), a `concat2` adjoint (two `region_copy`s
  — `concat2` has no `_dx` sibling in the tree), and the `scale_add` adjoint,
  which for a scalar `gate` needs a **reduction over the whole tensor** to get
  `dw`. That reduction has no existing kernel (`gradnorm_part`/`_sq` is the
  closest shape). If `w` stays a hyperparameter rather than a learned one — it
  is a user dial, so it should — `dw` is not needed and `scale_add`'s adjoint is
  just `axpy` with `s = w`.
* `argmax_row` is not differentiable. Stage-II training (`code_only=True`)
  supervises the **logits** with cross-entropy against the teacher's
  nearest-neighbour indices and never runs the gather; stage III freezes
  `quantize`/`generator` and trains only the CFT. Both are the reference's own
  schedule and neither needs an argmax gradient — `ce_grad` / `ce_value` already
  exist.
* Gate it as `gradcheck::check_codeformer` with the FD floors the playbook sets
  (block < 1e-4, model < 1e-3).

### Serving contract — **MET** (2026-08-04), except for batching

* **`capability::Provider`**: `restore::caps`, one action `restore_face`, taking
  an (ideally aligned) RGB image — resized to 512² on the device — plus a typed
  `w ∈ [0, 1]` float param, returning an image. `w` is a device buffer, so a
  sweep is a buffer write and not a rebuild; measured over D-Bus, a 0 → 0.5 → 1
  sweep on one resident instance costs 29097 / 951 / 1050 ms (debug build).
* **Residency adapter**: `crates/cli/src/resident_restore.rs`
  (`RestoreResident`), registered in `resident::build_executor` behind
  `BRAIN_RESTORE_WEIGHTS`; `instance_key` is the fixed 512² graph, which is what
  makes the `w` sweep share one build. `estimate` budgets 1.2× the checkpoint
  plus 3 GiB of activation slack — still a bound, not a measurement.
* **D-Bus** `Run` with the image as a memfd blob, plus
  `examples/restore/restore_face.py`. Measured against the
  `gen_face_w{0p00,0p50,1p00}` goldens through a u8 PPM round trip on both
  sides: cosine **0.999998** at every `w`, `mean|out−ref|` 0.00099, and
  `mean|out−in|` 0.0359 / 0.0294 / 0.0274 — the dial moves in the right
  direction. Detection + alignment are still NOT chained in (see above);
  the action takes the face it is given.
* **Still owed — `run_batch`**: the architecture batches cleanly — every
  attention dispatch already takes `bsz` and every conv takes `N` — but the
  current graph hardcodes `bsz = 1` as a recorded step list over fixed buffers.
  Batching means parameterising `CodeFormer::new` on batch size and widening the
  `img_in` / `idx_in` / `out` buffers; the host round-trip for the argmax
  indices is per-batch, not per-image, so it does not serialise. Until then
  `run_batch` is the serial default, said so in-file.
* **Perf**: not profiled. The two P40 submits at 512² run in well under a second
  end to end in the test, but no per-kernel table has been published, so no
  optimisation claim is made here. Follow `docs/porting-playbook.md` §10 and
  copy `crates/flux2/src/bin/flux2_bench.rs`.
