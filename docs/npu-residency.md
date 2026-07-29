# NPU as a first-class schedulable compute target

**Goal.** Make the Intel NPU a compute the residency scheduler places models on
*automatically* — the way it already picks a GPU — so `brain serve --dbus` (and
every other execution path) transparently runs a model on the NPU when it has an
NPU path, with **one generic seam** per model instead of bespoke per-model plumbing.
This closes the gap the code already names (`gpu-core/src/devices.rs`: *"transparent
NPU scheduling needs the per-model export path first"*).

## Why it isn't automatic today

The NPU is **not** a `gpu-core` per-op backend — OpenVINO is a *whole-graph*
compiler, so a model reaches the NPU by exporting an ONNX graph
(`crates/npu/*_topology.rs`), compiling it with OpenVINO, and running named-tensor
inference (`crates/npu/openvino/*Session`). That machinery is mature (≈10 models
have topologies), but it is wired **ad-hoc**:

- Each model has a bespoke `*Session` with a shape-specific `run*`; only YOLO/depth
  go through the `backend_api::GraphBackend` trait. There is **no generic
  `NpuModel`** abstraction.
- The residency layer has no `Device::Npu`. `--device npu` resolves a
  `ComputeSet.npus`, but the serve path collapses it to a process-global
  `NPU_REQUESTED` bool that individual CLI subcommands read to *bypass* the
  scheduler. So `brain serve` never schedules onto the NPU.

## Architecture — three reusable layers

```
              ┌─────────────────────────── per model: ~a topology + a thin adapter
              ▼
   npu::NpuModel  (trait)         build(&mut GraphBuilder) + inputs()/outputs() + cache_key()
              │  (each model implements once; reuses the TopoBase block library)
              ▼
   npu::NpuGraph  (generic)       compile(onnx, NpuConfig) → named-tensor infer  (ONE runner, all models)
              │
              ▼
   residency::Device::Npu         a real schedulable device with a memory budget + its own lane
     + place.rs NPU preference     pick_device tries NPU first when the model supports it (MemCost.npu > 0)
     + Executor (unchanged)        auto-spawns an NPU lane once the NPU has a budget
```

### Layer 1 — `npu::NpuModel` + `npu::NpuGraph` (the reuse seam)

`NpuModel` is the *only* per-model NPU contract:

```rust
pub trait NpuModel: Send + Sync {
    fn build(&self, g: &mut onnx::GraphBuilder) -> Result<(), String>; // the device-heavy forward, via TopoBase blocks
    fn inputs(&self)  -> Vec<IoSpec>;   // named (name, dtype, shape)
    fn outputs(&self) -> Vec<IoSpec>;
    fn cache_key(&self) -> String;      // stable key → OpenVINO CacheDir warm-start
}
```

`NpuGraph` generalises the bespoke sessions into **one** named-tensor runner
(compile ONNX bytes/path with `NpuConfig`; feed `f32`/`i64` tensors by name; read
outputs by name). Existing sessions already do this internally with
`set_tensor`/`get_tensor` — `NpuGraph` just exposes it uniformly, so YOLO, depth,
the LLM decoders, the ASR encoders, forecasting cores, etc. all share the compile /
cache / infer / evict machinery. fp16 is the default NPU path (no calibration);
INT8/INT4 stay opt-in and orthogonal.

Adding a model to the NPU is then: **write its topology (reuse the block library) +
implement `NpuModel` + a thin residency adapter.** No new session, no new runtime
code.

### Layer 2 — `residency::Device::Npu(u32)`

A real device, budgeted and scheduled like a GPU. `MemCost` gains an `npu` field
(`MemCost::new(vram, ram)` unchanged → `npu = 0`; a model with an NPU path reports
`npu > 0`). `MemCost::on(Device::Npu) = self.npu`. The Executor spawns an NPU lane
automatically once the NPU has a budget, and `activate(key, Device::Npu(i))` lets an
adapter build its `NpuGraph` for that device.

### Layer 3 — auto-preference in `place.rs`

Placement is per-instance and separate from the (device-blind) scheduler, so
"prefer NPU" lives in `pick_device`: **if `cost.npu > 0` and an NPU has room, place
there first**, else fall back to GPU, else CPU. A model without an NPU path reports
`npu = 0` and is simply never placed on the NPU. `build_executor` gains an NPU
budget (from `ComputeSet.npus`); the serve path (`run_cli.rs`) propagates
`set.npus` instead of collapsing it to a bool.

## Dual-path residency adapters

Each resident adapter's `activate` branches on the assigned device:

```rust
fn activate(&self, key, device) -> Box<dyn Instance> {
    match device {
        Device::Npu(_) => Box::new(self.build_npu_instance()?),   // compile the NpuModel graph, host glue reused
        _              => Box::new(self.build_native_instance()?), // the existing per-op (CPU/GPU) instance
    }
}
```

The host glue (front end, RNN-T decode, tokenisation) is identical across paths —
only the compute core swaps between the WGSL Step-graph and the compiled NPU graph.

## Applying it (order)

1. **Foundation** — `Device::Npu`, `MemCost.npu`, `place.rs` preference,
   `build_executor`/`run_cli` wiring, `npu::NpuModel` + `NpuGraph`. Verified against
   an existing topology (depth: a clean 4-D graph) so the *scheduling* path is proven
   before the hard ASR export.
2. **Nemotron encoder** (the streaming ASR model) — FastConformer encoder topology
   (subsampling convs, macaron FFs, rel-pos attention with `rel_shift` + banded mask,
   GLU conv module, projectors), parity-gated against the golden pooler. RNN-T greedy
   decode stays on host (m=1). Wire `NemotronResident` NPU branch.
3. **Qwen3-ASR** — new audio-encoder topology; **reuse the existing Qwen3 decoder
   NPU path** (`qwen_topology`/`DecoderSession`) for the 1.7B decoder + splice.
4. **Keep going** — the other topology'd models (yolo, depth, glm, chronos2, kronos,
   fincast, codec, mirror, wm) slot into the same `NpuModel` seam + residency branch.

## The payoff

`brain serve --dbus` with an NPU present schedules the ASR encoder onto the NPU
automatically; the `examples/asr` streaming demo runs the same but **faster**
(the FastConformer encoder is the compute, and the NPU is ≈5× the CPU on that
matmul-heavy workload — see `docs/models/asr/status.md`). Every future model that
ships an `NpuModel` gets the same treatment for free.
