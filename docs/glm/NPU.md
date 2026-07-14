# GLM on the Intel NPU (dense-expert export) — design + status

The NPU is **not** a `gpu-core` backend: OpenVINO is a *whole-graph* compiler, so
`--device npu` is a separate export → quantize → compile → run path
(`crates/npu`), exactly like YOLO and Qwen. This document specifies the GLM
dense-expert export and its current status.

## Status

- **cpu / gpu**: fully supported today — the GLM model is written once against the
  `Gpu`/`Step` seam (`crates/glm`), so `--device cpu|gpu` runs it unchanged
  (validated: gradcheck + learnability + end-to-end `brain glm train/eval/infer`).
- **npu**: designed here, **not yet wired**. The export builder + parity test are
  the remaining work; they require an OpenVINO install + NPU hardware to validate
  (this repo's CI/build stays green without OpenVINO via `runtime-linking`, but
  numerical parity can only be confirmed on a machine with the runtime — the same
  gate the Qwen decoder export uses, `BRAIN_OV_PROBE`). Unverified graph code is
  intentionally **not** committed.

## Why "dense-expert"

The brain MoE forward already **evaluates every expert densely** and combines them
with a gate-weighted sum where non-top-k experts have gate 0 (`crates/glm`
`scale_add`). That means MoE maps to a static-shape ONNX graph **without any
data-dependent gather/scatter**: emit `E` dense expert FFNs + a router, and mask
the gate. This pays FLOPs for all `E` experts (no sparsity compute win) but is the
only NPU-friendly option — true sparse token routing needs `GatherND`/`ScatterND`
and dynamic shapes, which the static-shape NPU plugin is hostile to. The DSA
indexer likewise runs in **dense mode** on NPU (`index_topk >= seq`); sparse
selection is cpu/gpu-only.

## Graph (fixed sequence length `T`, cache-free prefill)

Mirror `crates/npu/src/qwen_topology.rs`'s `Topo` builder (Gather / MatMul / Mul /
Add / Sigmoid / Softmax / ReduceMean / Sqrt / Div / Reshape / Transpose / Slice /
Concat / Neg — all mapped by OpenVINO's ONNX frontend). Input `input_ids:[1,T]`
(i64) → output `logits:[1,T,vocab]` (f32). Per layer:

1. **RMSNorm** (`input_ln`) via `Mul`/`ReduceMean`/`Add`/`Sqrt`/`Div`/`Mul`
   (`Topo::rmsnorm`, already implemented for Qwen).
2. **MLA attention** (dense): `q_a → RMSNorm → q_b_{nope,rope}`; `kv_a_c →
   RMSNorm → kv_b_{nope,v}`, `kv_a_rope` for the shared key. RoPE on the rope
   slices via precomputed cos/sin (`Topo::rope*`). Scores = `nope·nope +
   rope·rope` (two `MatMul` + `Add`), `1/sqrt(qk_head_dim)` scale, causal mask add,
   `Softmax`, `MatMul` with v, `o_proj`.
3. **MLP**: dense SwiGLU (`Topo::linear` + `Sigmoid`/`Mul`) for the first
   `first_k_dense_replace` layers; else **MoE**:
   - router logits `MatMul` → `Sigmoid` → add `router.bias` → **TopK** (emit the
     `TopK` op) → threshold/`Where` mask → renorm → `Mul` by `routed_scaling`.
     (Alternative that avoids `TopK`: compute the gate on the host and feed it as
     an extra graph input — keeps the graph to already-mapped ops.)
   - `E` expert FFNs (each `gate`/`up`/`SiLU`/`down`), each scaled by its gate
     column and summed (`Mul` + `Add`); plus the shared expert FFN.
4. Residual `Add`.

Final `RMSNorm` (`norm`) → `lm_head` `MatMul` → `logits`. Linear weights are
`[out,in]` in brain and transposed once at export to ONNX `[in,out]` (as in
`qwen_topology`).

## Files to add (implementation plan)

- `crates/npu/src/glm_topology.rs` — `build_glm_graph(cfg, w, t, g)` mirroring
  `build_qwen_graph`; factor `Topo` out of `qwen_topology` into a shared module
  (or duplicate the ~150-line helper). Dense-expert MoE block as above.
- `crates/npu/src/glm_export.rs` — `build_glm_fp32_bytes` / `export_glm_fp32`
  (+ INT8 weight-only reusing `qwen_export`'s per-output-channel Q/DQ).
- `crates/cli/src/glm_cli.rs` — an `export` verb (`brain glm export --weights F
  --out model.onnx --seq T`), and NPU inference via the `qwen_decode`-style
  session.
- `crates/npu/tests/glm_onnx.rs` — parity test gated on `BRAIN_OV_PROBE`: train a
  tiny GLM → `model.logits_all(ids)` reference → `build_glm_fp32_bytes` → run via
  OpenVINO `DecoderSession` → assert per-position argmax agree + `max_abs < 1e-2`.
  Plus a **non-gated structural test**: build the graph, serialize, assert it
  decodes to a valid `ModelProto` with the expected inputs/outputs (this validates
  wiring/shapes without OpenVINO).

## Honest caveats

- MoE-on-NPU computes all `E` experts — bandwidth-bound INT8 decode makes this
  viable for small `E`, but there is no sparsity compute win.
- The DSA indexer and MTP speculative draft are **not** exported (dense attention;
  MTP is a host-side draft loop). These are inference-shape optimisations, not
  correctness features.
- Numerical parity is only assertable with OpenVINO installed + NPU hardware.
