# Training in Rust + WGSL

The `moe` binary can **train the sparse-MoE Transformer from scratch on the
GPU** — the full forward + backpropagation + AdamW step are hand-written WGSL
compute kernels (no autograd framework, no CPU math in the hot path).

This is the same model as `tiny_sparse_moe.py`; the training step is a faithful
reimplementation, validated gradient-by-gradient against PyTorch autograd.

## Quick start

```bash
cd moe-rs
CARGO_HOME=/tmp/cargo-moe cargo build --release

# train from scratch; saves weights in the inference engine's format
./target/release/moe train --steps 2000 --batch-size 16 --block-size 64 \
    --lr 6e-4 --out moe_rs.safetensors

# generate from the model you just trained
./target/release/moe generate --weights moe_rs.safetensors --prompt 1,2,3,4 --max-new 64
```

Train flags: `--steps`, `--batch-size`, `--block-size` (sequence length, ≤64),
`--lr`, `--weight-decay`, `--seed`, `--out`. The toy corpus is generated on the
Rust side with the same bijective recurrence as the Python version.

## How it works

The forward pass is written **SSA-style**: every stage writes a *fresh* buffer,
which doubles as the activation cache backprop needs — so there are no mid-pass
copies. Attention is materialised (`scores`/`probs` kept) rather than flash,
which makes the backward pass straightforward and exact.

Backward is a hand-written reverse pass over the graph:

```
ce_grad → lm_head(dW tied, dX) → final RMSNorm
  per layer (reversed):
    MoE:  scale_add backward → router backward (combine + aux + z-loss grads)
          → per-expert (down/silu/up/gate matmul backward) → RMSNorm2
    Attn: out-proj backward → attention(dscores,dv,dq,dk) → RoPE backward
          → qkv backward → RMSNorm1
  embedding scatter (accumulates onto the tied lm_head grad)
→ AdamW update (decoupled weight decay, bias-corrected)
```

Weight-grad buffers are zeroed once per step (`clear_buffer`), then every
`matmul_dw`/`rmsnorm_dw`/`emb_bwd` **accumulates** into them; `wgpu` inserts the
inter-dispatch barriers automatically.

### Kernels added for training

Forward: `rope_train`, `attn_scores`, `attn_softmax`, `attn_apply`,
`router_gate_train`, `add2`, `ce_value`.
Backward: `ce_grad`, `matmul_dx`, `matmul_dw`, `rms_inv`, `rmsnorm_dx`,
`rmsnorm_dw`, `silu_bwd_da`, `silu_bwd_db`, `scale_add_dexp`, `scale_add_dgate`,
`expert_counts`, `router_bwd`, `rope_train_bwd`, `attn_bwd_d{scores,v,q,k}`,
`emb_bwd`, `adamw`.

## Correctness: finite-difference gradient check

Backprop correctness is gated by `make gradcheck` (the `brain-gradcheck` crate):
it perturbs each parameter and compares the analytic gradient against a central
finite-difference estimate of the loss, with no external reference needed.

```bash
make gradcheck
```

Every gradient — cross-entropy, all matmuls, RMSNorm, SwiGLU, the top-k router
*including* the load-balancing aux loss and the router z-loss, attention, RoPE,
and the tied embedding — is checked to finite-difference precision. (The earlier
PyTorch-parity `moe validate` / `train_ref.py` path has been retired in favour of
this self-contained check.)

## Notes / limits

- fp32 throughout; ≤5 storage buffers per kernel; one bind group. Requests
  `max_storage_buffers_per_shader_stage = 8`, well within Pascal/sm_61. Runs on
  any Vulkan/Metal/DX12 adapter (`PowerPreference::HighPerformance` picks a real
  GPU; falls back to `llvmpipe` software Vulkan on headless boxes).
- The kernels are correctness-first (naive matmul, materialised attention,
  per-token router loops). For this tiny model that's plenty; the throughput
  wins for larger models would be matmul tiling and fusing the per-expert
  matmuls — intentionally left out to keep the kernels readable and portable.
- Training uses the dense top-k MoE (no capacity dropping) — exact for
  inference/training quality; capacity limits only ever existed to bound memory.
