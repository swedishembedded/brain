# mirage - roadmap

Scope: Mirage visual perception model (`crates/mirage`, arch id `mirage`).

Mirage (`Cactus-Compute/mirage`, Apache-2.0 weights and source) is a
51.3M-parameter perception model: a convolutional stem, 16 gated-mixing +
HConvGLU body blocks, 12 attention blocks, a CenterNet-style dense head with
mask prototypes, a text reader, and an 8-layer decoder with hyper-connection
lanes and engram tables. It ships as F32 safetensors (Flax names) and as a
2-bit Cactus-Quant `.cact`. The goal is the whole product in brain: import,
inference, training, serving, quantised kernels and a profiled throughput
story. Every lane below is open unless marked done.

Implementation rule: brain's Rust is written from the checkpoint's own facts
(tensor names, shapes, published formulas) and proven against numerical
goldens; nothing is transcribed from the vendor's source or engine.

## Lane L1 - foundation

| Item | State |
|---|---|
| Arch row `mirage` (Vision, `brain-mirage`, default ref `Cactus-Compute/mirage`); no capabilities declared | done |
| Crate in the workspace, layer map and default members | done |
| `config`: typed `MirageConfig` / `CheckpointConfig`, `deny_unknown_fields`, invariants name the field | done |
| `spec`: canonical manifest, 583 tensors / 54,078,804 parameters, training-only tensors marked | done |

## Open lanes

| Lane | Scope | State |
|---|---|---|
| Vision encoder forward | stem, body, attention top, level memory, dense head peaks | open |
| Decoder forward | lane stack, engrams, cross-attention, Hadamard MLP, logits | open |
| Heads | boxes, masks, slots, enum scoring | open |
| OCR | the reader and greedy collapse | open |
| Training | full / head / LoRA, multi-task, gradient checks | open |
| Serving | capability, CLI, SDK, HTTP, D-Bus, batching | open |
| Quantised kernels | native low-bit CQ matmul from the packed weights | open |
| Performance | profile-driven batched throughput | open |
