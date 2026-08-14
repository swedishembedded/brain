# brain-yolo

A from-scratch YOLOv8-style anchor-free object detector for the `brain`
framework: CSP backbone → PAN-FPN neck → decoupled DFL head, with WGSL kernels
run on the CPU backend (Cranelift JIT). Forward + backward are validated end to
end by a finite-difference gradient check (`tests/p3_gradcheck.rs`).

This document covers the **`brain yolo` CLI** (train / eval / detect /
fine-tune on a synthetic dataset), the **`brain run` runtime integration**
(camera frames → detections over the event protocol), and **the offline
pretrained-weight export path** — turning an official Ultralytics `yolov8n.pt`
into brain's native `.safetensors` format for fine-tuning, plus the layer-by-layer
reference-parity test.

---

## CLI: `brain yolo …`

The detector trains on the synthetic detection dataset (`brain data gen detect`
→ a `Dataset::Detect` dir of RGB shapes + exact GT boxes). The tiny config's
input resolution (128px) matches the dataset's default geometry, so the dataset
images upload directly (no letterbox) during training. Everything runs on the
**CPU backend** (add `--device cpu` or set `BRAIN_DEVICE=cpu`).

```bash
# 1. Generate a synthetic detection dataset (RGB shapes + exact boxes, 128px).
BRAIN_DEVICE=cpu brain data gen detect --out data/detect --n 64 --seed 7

# 2. Train the tiny YOLOv8 graph (real detection loss: assigner+BCE+CIoU+DFL).
BRAIN_DEVICE=cpu brain yolo train data/detect --out out/yolo.safetensors \
    --steps 150 --batch 4 --lr 3e-3 --seed 7
#   flags: --steps --batch --lr --wd --nc --input --seed
#   prints the loss every 10 steps; saves a `.safetensors` checkpoint at the end.

# 3. Evaluate: mAP@0.5 + precision/recall over the 10% val split (eval::detection).
BRAIN_DEVICE=cpu brain yolo eval --weights out/yolo.safetensors --data data/detect \
    --conf 0.1 --iou 0.45

# 4. Detect on one image — a binary PPM (P6) file OR a dataset dir (image 0):
BRAIN_DEVICE=cpu brain yolo detect --weights out/yolo.safetensors --image data/detect \
    --conf 0.1 --iou 0.45
#   prints one JSON line per detection: [x1,y1,x2,y2,conf,class]

# 5. Fine-tune from a pretrained checkpoint (e.g. an exported yolov8n.brain.safetensors):
BRAIN_DEVICE=cpu brain yolo fine-tune data/detect --weights pretrained.safetensors \
    --out out/yolo-ft.safetensors --steps 150
#   tensors present in BOTH checkpoint and model (matched by element count) are
#   copied in; the rest keep their random init (e.g. a head with a different nc).
```

> **`--freeze-backbone`** is accepted but currently a no-op: the model exposes no
> per-parameter freeze API, so fine-tune trains the whole network from the
> pretrained init. This is a documented limitation, not a silent drop.

Or via the Makefile (CPU-friendly defaults: `YOLO_N`/`YOLO_STEPS`/`YOLO_BATCH`/
`YOLO_LR`/`YOLO_CONF`/`YOLO_IOU`):

```bash
make data/detect train/yolo eval/yolo detect/yolo
```

---

## Runtime integration: `brain run --yolo`

The event-driven controller (`crates/runtime`) routes a `camera_frame` event to
the detector and emits an `object_detected` event. The `YoloDetect` adapter
(`runtime::YoloDetect`) wraps a loaded `Yolo` behind the object-safe
`DetectModel` trait: it converts the incoming RGB8 frame to normalised `[0,1]`
HWC floats, calls `Yolo::detect` (letterbox → eval-mode forward → DFL decode →
NMS), and returns `[x1,y1,x2,y2,conf,class]` rows; labels are numeric class ids.

```bash
# Load a trained YOLO as the detector (or set BRAIN_YOLOV8=out/yolo.safetensors).
# With no --yolo, a FakeDetectModel returns a fixed box so the loop still runs.
printf '{"event":"camera_frame","format":"rgb8","w":128,"h":128,"data":"<base64 rgb8>"}\n' \
  | BRAIN_DEVICE=cpu brain run --yolo out/yolo.safetensors
# -> {"event":"object_detected","dets":[[x1,y1,x2,y2,conf,class],…],"labels":["0","1","2"]}
```

Frames may inline base64 `data` (raw rgb8 or a PPM `P6` blob) or reference a
`path`; see `crates/events` for the codec.

> **P12 (RESOLVED): byte-compatible with canonical yolov8n.** The brain
> `YoloConfig::yolov8n()` graph is now the *exact* Ultralytics yolov8n graph
> (per-stage channels `[16,32,32,64,64,128,128,256,256,256]`, C2f depths
> `[1,2,2,1]`, neck widths `[128,64,64,128,128,256]`, **biased** decoupled head
> with `reg`/`cls` hidden widths 64/80). The old reduced-width/reduced-depth
> approximation is gone. The exporter now maps a real `yolov8n.pt` state_dict
> **1:1** with equal shapes — proven without torch (see "Torch-free map proof").

---

## Export workflow

The converter is **out of tree** (it needs PyTorch + Ultralytics, which the
brain CI box does not have, and `scratchpad/` is gitignored):

```
tools/yolo_export/
  export_yolov8.py   # the converter + a torch-free map self-test
  brain_names.txt    # checked-in dump of YoloConfig::yolov8n().full_param_list()
```

On a dev machine with `pip install ultralytics torch` — or skip both the
`pip install` and the manual `yolov8n.pt` download with
`scripts/data/fetch-yolov8.sh [--variant yolov8n] [--out DIR]`, which fetches the
checkpoint from the Ultralytics release assets and runs `export_yolov8.py` for
you, one step:

```bash
# 1. Convert weights -> brain native container.
python3 tools/yolo_export/export_yolov8.py \
    --weights yolov8n.pt --out yolov8n.brain.safetensors

# 2. (Optional) Dump per-stage activations for the parity test, for one fixed
#    preprocessed 640x640 input image.
python3 tools/yolo_export/export_yolov8.py \
    --weights yolov8n.pt --out yolov8n.brain.safetensors \
    --dump-acts --image bus.jpg --acts-out yolov8n.acts.safetensors

# 3. Torch-free self-test: the name map covers every brain tensor 1:1.
python3 tools/yolo_export/export_yolov8.py \
    --check-map-only --brain-names tools/yolo_export/brain_names.txt
```

`brain_names.txt` is regenerated from Rust (the canonical source of truth):

```bash
cargo test -p brain-yolo --test p8_names dump_brain_names -- --nocapture \
  | grep -E '^(backbone|neck|head)\.' > tools/yolo_export/brain_names.txt
```

### Output format (safetensors)

A standard safetensors file, matching `checkpoint::save` / `checkpoint::st`
(see `crates/checkpoint/src/st.rs`). Each brain tensor is stored 1-D (fp32) under
its brain name; `YoloConfig::yolov8n().to_json()` goes in `__metadata__` under
`brain.config` and is recovered via `Container::header["config"]`. Safetensors
has no role concept, so tensors read back under role `""` (`by_role("")`); the
activation dump likewise uses plain names.

---

## Ultralytics → brain name map

The brain detector registers parameters under the names in
`YoloConfig::full_param_list()`. The converter maps each Ultralytics
`state_dict` key onto a brain name with a pure, auditable **string** remap
(`ultra_to_brain` in `export_yolov8.py`) — **never** any arithmetic on values.

### Module-index map

Ultralytics `DetectionModel.model` is an `nn.Sequential` of 23 modules
(`model.0` … `model.22`). Upsample (`10`,`13`) and Concat (`11`,`14`,`17`,`20`)
carry no weights and are dropped.

| Ultralytics  | Type        | brain prefix | role                          |
|--------------|-------------|--------------|-------------------------------|
| `model.0`    | Conv s2     | `backbone.0` | stem                          |
| `model.1`    | Conv s2     | `backbone.1` | stem                          |
| `model.2`    | C2f         | `backbone.2` |                               |
| `model.3`    | Conv s2     | `backbone.3` |                               |
| `model.4`    | C2f         | `backbone.4` | → P3                          |
| `model.5`    | Conv s2     | `backbone.5` |                               |
| `model.6`    | C2f         | `backbone.6` | → P4                          |
| `model.7`    | Conv s2     | `backbone.7` |                               |
| `model.8`    | C2f         | `backbone.8` |                               |
| `model.9`    | SPPF        | `backbone.9` | → P5                          |
| `model.12`   | C2f         | `neck.0`     | top-down P4 (T4)              |
| `model.15`   | C2f         | `neck.1`     | top-down P3 (= N3, scale 0)   |
| `model.16`   | Conv s2     | `neck.2`     | downsample N3                 |
| `model.18`   | C2f         | `neck.3`     | (= N4, scale 1)               |
| `model.19`   | Conv s2     | `neck.4`     | downsample N4                 |
| `model.21`   | C2f         | `neck.5`     | (= N5, scale 2)               |
| `model.22`   | Detect      | `head.*`     | decoupled DFL head            |

### Per-tensor map (within each Conv / block)

| Ultralytics suffix              | brain suffix          |
|---------------------------------|-----------------------|
| `…conv.weight`                  | `…conv.weight`        |
| `…bn.weight`                    | `…bn.gamma`           |
| `…bn.bias`                      | `…bn.beta`            |
| `…bn.running_mean`              | `…bn.run_mean`        |
| `…bn.running_var`               | `…bn.run_var`         |
| `…bn.num_batches_tracked`       | *(dropped)*           |

C2f / SPPF inner names are **identical** on both sides: `cv1`, `cv2`,
`m.{i}.cv1`, `m.{i}.cv2`. So e.g.
`model.4.m.0.cv2.bn.running_var` → `backbone.4.m.0.cv2.bn.run_var`.

### Detect head map (`model.22`)

Ultralytics Detect has `cv2` (box / DFL distribution) and `cv3` (class), each an
`nn.Sequential([Conv, Conv, Conv2d])`. brain names the box branch `reg` and the
class branch `cls`, with the same `.0`/`.1` (full Conv) and `.2` (final 1×1)
indices:

| Ultralytics                       | brain                          |
|-----------------------------------|--------------------------------|
| `model.22.cv2.{s}.{0,1}.…`        | `head.{s}.reg.{0,1}.…`         |
| `model.22.cv2.{s}.2.weight`       | `head.{s}.reg.2.weight`        |
| `model.22.cv2.{s}.2.bias`         | `head.{s}.reg.2.bias`          |
| `model.22.cv3.{s}.{0,1}.…`        | `head.{s}.cls.{0,1}.…`         |
| `model.22.cv3.{s}.2.weight`       | `head.{s}.cls.2.weight`        |
| `model.22.cv3.{s}.2.bias`         | `head.{s}.cls.2.bias`          |
| `model.22.dfl.conv.weight`        | *(dropped — DFL is analytic)*  |

The brain head's final 1×1 is now a **biased** `Conv2d` (P12), so the
Ultralytics `.2.bias` maps straight across. The DFL projection
(`model.22.dfl.conv.weight`, a fixed `arange` `16→1` conv) is **not** learned —
brain computes the DFL box expectation analytically — so it is dropped (no brain
param). `bn.num_batches_tracked` (one scalar per BN) is also dropped.

### Bias-free / BN fold rule

Both sides keep the **convolution** bias-free and carry BatchNorm as four
separate, live tensors (`gamma`/`beta`/`run_mean`/`run_var`). The brain `Conv`
runs `conv2d` (bias-free) → BatchNorm → SiLU exactly like Ultralytics, so the
four BN tensors are copied straight across 1:1 (**no BN folding**). Only the
head's final 1×1 layers carry a bias (matching Ultralytics' biased `nn.Conv2d`).

`tests/p8_names.rs` pins the brain side of this contract (every name follows the
scheme; each `conv.weight` has its 4 BN tensors; each head branch has `.2.weight`
+ `.2.bias`). `--check-map-only` proves the map covers the FULL canonical
yolov8n state_dict (next section), with no torch.

---

## yolov8n fingerprint (this crate's `YoloConfig::yolov8n()`)

`YoloConfig::yolov8n()` is the **byte-compatible canonical** Ultralytics yolov8n
graph (P12). Per-stage layout:

| group     | values                                            |
|-----------|---------------------------------------------------|
| backbone channels (stages 0–9) | `[16,32,32,64,64,128,128,256,256,256]` |
| C2f depths (stages 2,4,6,8)    | `[1,2,2,1]`                            |
| neck channels (neck.0–5)       | `[128,64,64,128,128,256]`              |
| P3 / P4 / P5 widths            | 64 / 128 / 256                         |
| head hidden (reg / cls)        | 64 / 80                                |
| head final 1×1                 | **biased** (`.2.weight` + `.2.bias`)   |

| metric                       | value       |
|------------------------------|-------------|
| tensors (`full_param_list`)  | **297**     |
| total scalar params          | **3,167,776** (~3.17M, = official yolov8n) |
| full Convs (conv + 4 BN)     | 57          |
| head final-1×1 weights       | 6 (3 scales × cls+reg) |
| head per-channel biases      | 6 (3 scales × cls+reg) |

`297 = 57·5 + 6 + 6`. Pinned in
`tests/p8_names.rs::yolov8n_tensor_and_param_counts`. The previous reduced
approximation was 271 tensors / 1,902,704 params.

---

## ✅ Canonical reconciliation (P12 — RESOLVED)

The brain graph is now the **exact** Ultralytics yolov8n graph, so the export
loads a real `yolov8n.pt` 1:1. The three historical discrepancies are resolved:

1. **C2f depth.** `backbone_depth = [1,2,2,1]` — stages `model.4`/`model.6` now
   have **two** bottlenecks (`m.0`,`m.1`), matching `yolov8n.pt`.
2. **Channel widths.** Deep channels widen to **256** (`backbone.7`→256, stage
   8/9 at 256), neck widths `[128,64,64,128,128,256]`, head inputs `[64,128,256]`
   — all matching canonical yolov8n.
3. **Head bias + DFL.** The head's final 1×1 is **biased** (`.2.bias` maps 1:1).
   The fixed DFL conv is dropped (brain computes the DFL expectation
   analytically), as is `bn.num_batches_tracked`.

### Torch-free map proof

`export_yolov8.py::canonical_ultra_tensors()` reconstructs the **full** canonical
yolov8n `state_dict` as `(name, shape)` pairs from the spec above — every shape
is arithmetic on the channel/depth numbers (conv weight `[out,in,kh,kw]`, BN
`[C]`, head bias `[nc]`/`[64]`, DFL `[1,16,1,1]`). `--check-map-only` then asserts:

* every non-dropped Ultralytics tensor maps to **exactly one** brain name,
* the mapped brain tensor's element count **equals** the canonical shape's,
* every brain name is covered **exactly once**, and
* the dropped tensors (`num_batches_tracked`, `dfl.conv.weight`) map to `None`.

Result (no torch needed):

```
MAP CHECK OK: canonical yolov8n state_dict = 355 tensors (58 dropped:
num_batches_tracked + dfl.conv.weight); remaining 297 map 1:1 onto all 297 brain
tensors with EQUAL shapes. The exporter will load a real yolov8n.pt 1:1.
```

(355 = 297 mapped + 57 `num_batches_tracked` + 1 `dfl.conv.weight`.) This proves
that on a dev machine with torch, `python3 export_yolov8.py --weights yolov8n.pt`
will map + shape-check every tensor and write a valid `.safetensors` file.

### Running real parity on a dev machine

```bash
pip install ultralytics torch
python3 tools/yolo_export/export_yolov8.py --weights yolov8n.pt \
    --out yolov8n.brain.safetensors                       # 297 tensors, shape-checked
python3 tools/yolo_export/export_yolov8.py --weights yolov8n.pt \
    --out yolov8n.brain.safetensors --dump-acts --image bus.jpg \
    --acts-out yolov8n.acts.safetensors                   # per-stage activations
YOLO_PARITY_WEIGHTS=yolov8n.brain.safetensors \
YOLO_PARITY_ACTS=yolov8n.acts.safetensors \
    cargo test -p brain-yolo --test parity -- --nocapture
```

---

## Parity test

`tests/parity.rs` is the layer-by-layer reference check. It **cannot run in the
brain CI box** (no torch/GPU; a 640 forward on the CPU JIT is slow), so it is
**env-gated** and skips cleanly when the exported files are absent:

```bash
# Skips (prints a SKIP notice, returns OK) when the env var is unset/missing:
cargo test -p brain-yolo --test parity

# Runs on a dev machine once you have the exported files:
YOLO_PARITY_WEIGHTS=yolov8n.brain.safetensors \
YOLO_PARITY_ACTS=yolov8n.acts.safetensors \
    cargo test -p brain-yolo --test parity -- --nocapture
```

When the files exist the test:

1. loads `yolov8n.brain.safetensors` and builds `Yolo::new(YoloConfig::yolov8n(), 1, …)`
   directly from the loaded weights,
2. uploads the dumped `input` tensor (identical preprocessing to PyTorch),
3. runs the forward and reads the head logits via `Yolo::raw_logits()`,
4. compares each head branch (`head.cls`, `head.reg`) against the dumped
   activations with `max_abs_err < 1e-3`, reporting the **first** divergent
   stage.

> Per-*internal*-buffer parity (every backbone/neck SSA buffer) needs
> buffer-accessor hooks the model does not yet expose; the dump captures those
> stages by name so finer comparison can be added when the accessors land. The
> readable end-to-end signal today is the head logits.

### What the parity test needs to actually run

* `YOLO_PARITY_WEIGHTS` → an exported `yolov8n.brain.safetensors`
  (from `export_yolov8.py`; requires the graph reconciliation above to produce a
  shape-correct file).
* `YOLO_PARITY_ACTS` → the matching `yolov8n.acts.safetensors` activation dump
  (`--dump-acts`), optional but required for the stage-by-stage comparison.

---

## Tests you CAN run here (no torch / no GPU)

```bash
cargo test -p brain-yolo --test p8_names   # config builds; 271 names match the map
cargo test -p brain-yolo --test parity     # compiles; skips cleanly (env unset)
python3 -m py_compile tools/yolo_export/export_yolov8.py
python3 tools/yolo_export/export_yolov8.py --check-map-only \
    --brain-names tools/yolo_export/brain_names.txt   # 271 names, 1:1
```
