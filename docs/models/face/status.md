# Face recognition (`crates/facenet`) — ledger

Model family: insightface **antelopev2** — SCRFD-10GF detection + 5-point
similarity alignment + the ArcFace **IResNet-100** (`glintr100`) embedding.
The visual analogue of `crates/speaker`: an embedding model consumed by a
generative one (PuLID / InstantID — `docs/imaging/plan.md`).

## Scope of this pass

**Goldens → import → FORWARD parity only.** Backward/gradcheck and the serving
contract are deliberately deferred; see "Deferred" below for what they will need.

## Deliverables

| Piece | Where |
|---|---|
| Reference dumper | `tools/goldens/arcface_dump_reference.py` |
| Goldens (gitignored) | `testdata/face/antelopev2/{arcface,arcface_blocks,scrfd,align,e2e}.safetensors` + `manifest.json` |
| Weights (gitignored) | `testdata/face/antelopev2/{glintr100,scrfd_10g_bnkps}.onnx` |
| Config | `crates/facenet/src/config.rs` |
| Import (two-way coverage) | `crates/facenet/src/import.rs` |
| Forward graphs | `crates/facenet/src/model.rs` |
| Alignment | `crates/facenet/src/align.rs` |
| Decode + NMS | `crates/facenet/src/detect.rs` |
| Parity test | `crates/facenet/tests/parity.rs` |
| ONNX **reader** (new, shared) | `crates/onnx/src/read.rs` |
| PReLU / AvgPool blocks (new, shared) | `vision::blocks::{PReLU, AvgPool}` |
| Similarity solve (new, shared) | `model::hostmath::{similarity_transform_2d, invert_affine_2x3}` |
| Embedding host math (new, shared) | `model::hostmath::{l2_normalize, cosine}` |
| Weights fetch | `scripts/data/fetch-testdata.sh` → `make fetch/testdata` (goldens are regenerated, not mirrored) |

```bash
CARGO_HOME=… cargo test --release -p brain-facenet --test parity -- --nocapture
BRAIN_DEVICE=cpu   # same numbers, no GPU
```

## Measured (GPU, Tesla P40; CPU backend identical to 7 digits of cosine)

7/7 parity tests pass on both backends. Worst per family:

| family | worst cos | worst max_abs |
|---|---|---|
| ArcFace 33 taps (blob → stem → `s{n}b0_*` → layer1..4 → 49 blocks → bn2 → fc → embedding) | 1.0000000 | 9.42e-6 (`embedding`) |
| ArcFace e2e, 4 photos (M / blob / embedding / normed) | 1.0000000 | 1.31e-5; cosine matrix max \|Δ\| 2.38e-7 |
| Alignment (`M`, `grid`, `grid_sample` warp) | 1.0000000 | 1.16e-2 vs torch `grid_sample`; 0.500/255 vs cv2 (expected) |
| SCRFD 41 taps incl. the 9 graph outputs | 1.0000000 | 1.05e-5 (`head8_cls_raw`) |
| SCRFD decode + NMS, 4 photos | — | box/kps ≤ 1.22e-4 px, scores to 6 digits |

## Findings

1. **Both released graphs are BatchNorm-FOLDED, and the folded tensors lost
   their names.** 256 of `glintr100.onnx`'s 462 initializers are bare SSA value
   numbers (`1335`, `1643`, …). A name remap — every other brain importer's
   strategy — is therefore impossible; import walks the graph **topologically**
   and binds weights positionally against an op sequence it asserts. That is a
   stronger check than a name map: a name map cannot notice a missing residual.

2. **`crates/onnx` was export-only; the import direction now exists in part.**
   `onnx::read` decodes initializers (raw_data f32/i32/i64/i8/u8 + the typed
   repeated fields), errors by name on external-data tensors, and refuses an
   int64 that is not exact in f32. It stops short of `Graph::from_proto` — a
   port rewrites topology as brain blocks, and that walk belongs to the model
   crate. A private reader inside `facenet` would have been a second decoder.

3. **`scrfd_10g_bnkps.onnx` shares two bias initializers between different
   convolutions** (`neck.fpn_convs.1` reads `neck.downsample_convs.0.conv.bias`,
   likewise `.2`/`.1`). Whether that is exporter deduplication or a release
   quirk, the goldens were dumped from this file, so import reproduces it: a
   source tensor counts as covered when used **one or more** times.

4. **The two preprocesses differ in `std` — 127.5 (ArcFace) vs 128.0 (SCRFD).**
   They look like the same "[-1, 1] map". Each lives in its own
   `config::Preprocess` and is never defaulted.

5. **The IResNet shortcut reads the block INPUT, not `bn1(x)`** (pre-activation
   residual). Feeding it `bn1(x)` runs, produces plausible features, and is
   wrong; only the `s{n}b0_branch` golden sees it.

6. **SCRFD's strided shortcut is `AveragePool(2,2) → conv1x1`**, not a strided
   1×1. Both give the right shape.

7. **cv2 warp parity is 0.5/255 and that is the correct expectation.**
   `cv2.warpAffine(INTER_LINEAR)` uses 5-bit fixed-point weights; `grid_sample`
   is fp32 bilinear. The alignment test gates tightly against the reference
   `grid_sample` output and loosely (≤ 1.0/255) against cv2 — tightening the cv2
   gate would mean the grid was wrong in a cancelling way.

8. **The 5-point → template fit has an irreducible ~1.4 px residual** (4 DOF,
   5 points). Parity compares `M`, never the landmark reprojection.

9. **NMS must run in SOURCE-image pixels, not detector-canvas pixels.** The
   Fast R-CNN `+1` area convention (`(x2-x1+1)*(y2-y1+1)`) is *not*
   scale-invariant, and insightface divides by `det_scale` **before** calling
   `nms`. Suppressing first and scaling afterwards is a different detection set
   from code that reads correct — it only shows up when two faces overlap near
   `nms_thresh`, which no single-face fixture can see. `decode` now scales first;
   `detect::tests::nms_runs_in_source_pixels_not_detector_pixels` pins a pair at
   IoU 0.3333 / 0.375 either side of a 0.35 threshold. Ties are broken the way
   `numpy.argsort()[::-1]` does (higher index first), for the same reason.

10. **The import checks geometry, not just shapes.** Both walks bind weights
    positionally, so a release with identical tensor shapes and a different
    stride/pad/dilation/group would import cleanly and run a different network.
    Every `Conv` is now asserted against the geometry the model will dispatch
    (derived from the config, so a config typo also fails here), every
    `BatchNormalization` against `bn_eval`'s hardcoded `eps = 1e-5`, the
    `Flatten` against `axis = 1`, and SCRFD's weightless `MaxPool` /
    `AveragePool` / `Resize` against 2x2/2 and `nearest`/`asymmetric`/`floor`.
    Those three carry no initializer, so the coverage ledger never looked at
    them.

## Deferred — what the follow-ups will need

**Backward / gradcheck (`check_arcface`)**
* The goldens are forward-only.
* The ArcFace **additive-angular-margin head is not in the ONNX** (the release is
  the backbone up to the `features` BN). It must come from the paper
  (`…/identity/papers/1801.07698-arcface.pdf`) or `arcface_torch`, which is **not
  on this box** — an acquisition item.
* `vision::blocks::PReLU::backward` already selects `prelu_bwd_wg` vs
  `prelu_bwd` on the **queried** `DeviceCaps::workgroup_reductions`. That is a
  correctness gate: `backend-cpu` reports it false and the cooperative kernel
  returns `da` all zeros there, silently.
* Every backward kernel the graph needs (`conv2d_dx/_dw`, `bias_grad`, `bn_d*`,
  `leaky_relu_bwd`, `maxpool2d_dx`, `avgpool2d_dx`, `resize_nearest_dx`,
  `prelu_bwd*`) is already registered in `facenet::PIPELINES`.
* `Conv` is built with `train = false` throughout; a trainer must flip
  `set_eval(false)` / `set_update_running(true)` per unit.

**Serving contract** (`docs/serving-contract.md`)
* `capability::Provider` + `crates/cli/src/resident_facenet.rs` registered in
  `resident::build_executor`.
* `Instance::run_batch`: ArcFace batches trivially (the ONNX input is
  `[N,3,112,112]`), but the current `ArcFace` pre-allocates `N = 1` activations —
  batching means threading `n` through the block constructors. SCRFD's input is
  `[1,3,H,W]`, so the detector batches only by looping or re-export.
* D-Bus `Run` with an image fd + `examples/face/`. `capability::Media::Image`
  already exists, so no surface change is expected.

**Not implemented in this pass**
* **The detector's input resize.** insightface's `SCRFD.detect` does
  `cv2.resize` to fit the long side, zero-pads to 640×640, and derives
  `det_scale = nh / h0` — which `detect::decode` needs. Neither the resize nor
  `det_scale` exists in the crate: `scrfd_decode_and_nms_reproduce_the_reference_detections`
  feeds the golden `det_blob` and passes the four `det_scale` values recorded in
  `manifest.json` as literals. Reproducing `cv2.INTER_LINEAR` is its own
  resampler-parity question (`imaging::Filter` names an exact reference per
  kernel and cv2's is not among them). Until it lands there is **no
  image-in → faces-out entry point**, only `Scrfd::forward(blob)`.
* **Batch is pinned to `N = 1`** — every block pre-allocates at `n = 1`.

**Performance** — not measured. The forward is correct and unoptimised:
* the `Conv` block allocates five full-size maps per unit even for a
  `Norm::None` inference unit, and `PReLU` a full `d_in`, which dominates
  SCRFD's 640×640 and ArcFace's 50-PReLU footprint;
* `Scrfd::forward` does a device→host `read_weight` of each head's scalar bbox
  scale on every frame — three GPU syncs per image for 3 floats that never
  change;
* `ArcFace::forward` / `Scrfd::forward` read back **every** tap. `embed_blob`
  throws them away rather than skipping them, so the production path pays the
  whole parity ladder's readbacks.
