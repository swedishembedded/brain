# qwen35 - roadmap

Qwen3.8-27B (HF `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type:
"qwen3_5"`; llama.cpp `general.architecture = "qwen35"`) - the **dense** sibling of
`crates/qwen35moe` (`LLM_ARCH_QWEN35MOE`). Same hybrid 3:1 Gated-DeltaNet : gated-GQA
mixer split (`full_attention_interval=4`), same partial/M-RoPE, same
per-head-interleaved sigmoid attention-output gate - but a plain dense SwiGLU MLP
instead of the 256-expert MoE, plus a single-layer MTP head sharing
`embed_tokens`/`lm_head`, and a spliced Qwen3-VL-style vision tower (27 blocks,
hidden 1152, no DeepStack) reusing `crates/qwen3vl` unchanged. Weights ship as
DeepSeek-V3-style blockwise FP8 (E4M3, 128x128 `weight_scale_inv` in BF16), one
safetensors shard per decoder layer (`layers-{0..63}.safetensors`) plus
`outside.safetensors` (embed/lm_head/final-norm/whole vision tower, all BF16) and
`mtp.safetensors`. `reasoning_effort` (xhigh/medium/low) is a chat-template
system-prompt injection only - no architectural surface.

Unlike qwen35moe's port, `transformers.models.qwen3_5` (torch 2.13 / transformers
5.14.1) is installed on the development machine, so this port is gated by a real
`porting.md` §5 parity ladder against real reference goldens, not structural
correctness alone. The one part with no available reference is the MTP head:
`transformers`' own loader discards `mtp.*` on load
(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so it is implemented
structurally and never parity-claimed here.

## Done

- [x] M1: hoist the zero-coupling GDN scratch allocators (`gdn_chunk_size`,
  `GdnScratchBufs`/`GdnScratchTrainBufs`/`GdnBwdScratchBufs`) out of
  `crates/qwen35moe/src/model.rs` into `crates/model/src/gdn.rs`, migrate qwen35moe.
- [x] M2: reference goldens (`tools/goldens/qwen35_dump_reference.py`) - tiny
  all-dims-distinct text config, plus vision at real dims (depth 27/hidden 1152).
- [x] M3: `crates/qwen35` config/param-manifest/init.
- [x] M4: FP8 blockwise import (`crates/model/src/fp8.rs`, safetensors `F8_E4M3`
  support, two-way tensor coverage, the `(1+w)` RMSNorm fold applied on direct
  HF-safetensors import - unlike qwen35moe, which imports only from GGUF where
  llama.cpp's conversion already bakes the fold in).
- [x] M5: text-only forward at tiny dims, climbing parity rungs 1-3 against the M2
  goldens (cosine + rel_l2 + max_abs at every stage).
- [x] M6: backward + `gradcheck::check_qwen35` (both mixer types; wide init on
  `in_proj_qkv`/`conv1d` from day one per lessons.md #40, plus a no-zero-FD
  assertion so the same hollow-gradcheck failure mode cannot recur unnoticed).
- [x] M7: MTP head (`Qwen35::run_mtp_forward`/`mtp_backward` in
  `crates/qwen35/src/model.rs`; `mtp.layers.0.*` is one full Gated-Attention
  decoder layer reusing `layer_gqa_fwd`/`mlp_fwd` unchanged via weight-name-prefix
  parameterization, unlike glmdsa's position-wise-only MTP block). Gated by
  `gradcheck::check_qwen35_mtp` (every `mtp.*` tensor, seed 8, zero tolerance
  failures) plus `crates/qwen35/tests/mtp_convergence.rs` (a real training loop
  still reduces the combined loss; zeroing `mtp.fc_e`/`mtp.fc_h` moves the loss,
  proving the head is load-bearing rather than merely wired). Structural only -
  no reference oracle available here, see below.
- [x] M8: LoRA (rank/alpha adapters on all 12 targetable leaves -
  `crate::config::lora_targets()`: qwen35moe's 9 GDN/GQA projections plus this
  model's own dense-MLP `gate`/`up`/`down`, which qwen35moe never targets since
  its MLP is MoE) + full finetune (`crates/qwen35/src/finetune.rs`, mirrors
  `qwen3::finetune` - `Mode::FullOffload`/`Mode::Lora`, genuinely new surface
  for this family, qwen35moe has none). `Qwen35::lora_fwd`/`proj_bwd`'s LoRA
  branch mirror `qwen35moe::model::Qwen35`'s exactly (two-matmul + `AXPY`
  forward fusion, frozen-base dX-only backward + `A`/`B` adapter grads).
  Gated by `gradcheck::check_qwen35_lora` (every `.lora_a`/`.lora_b` tensor
  across both mixer types, zero tolerance failures across 6 probed seeds);
  `tests/lora_freezes_base.rs` (a real training loop changes only the
  adapters, every frozen base weight bit-identical); `tests/lora_roundtrip.rs`
  (adapters survive a save+reload cycle with real optimizer steps, reloaded
  logits reproduce the trained model and differ from the untrained base by a
  real margin, plus a `lora`-key-absent checkpoint loads as a plain model);
  `tests/convergence.rs` (full finetune and LoRA both drive a fixed batch's
  loss down, plus a cyclic-sequence memorization floor).

## Not yet done

- [ ] M9: vision tower splice (`crates/qwen35/src/vl.rs`, copied from
  `crates/qwen35moe/src/vl.rs`; real-dims parity vs M2's vision golden).
- [ ] M10: real-weight streaming parity (fetch the 30.9 GB FP8 checkpoint;
  per-layer streaming forward parity for layers {0, 3, 63}; full real-weight
  parity of the vision tower; embed/lm_head spot checks).
- [ ] M11: CLI (`brain qwen35 ...`), `caps.rs`/`serve.rs`/`shard.rs`, residency
  (`resident_qwen35.rs` + `catalog.rs`), docs.
- [ ] M12: finish the shared-code hoist - migrate qwen35moe off its private
  `crate::q8::Qwen35Q8` onto `model::ops::{Ops,Act,Weight}`, then hoist the GDN
  and gated-GQA mixer orchestration into `crates/model`, gated by a new
  `crates/model/tests/gdn_mixer_equivalence.rs` proving bit-identity between both
  crates' mixers on the same weights.
- [ ] M13: performance pass (profile-first; native device-side FP8 GEMM only if
  the profile says arithmetic is the limiter, not before).

## Recorded gaps (this development machine has no discrete GPU and 18 GiB usable RAM)

- No whole-model 27B forward, no whole-model torch reference, no e2e generation or
  perplexity number on real weights - unreachable at 27B vs 18 GiB with no
  discrete GPU. Rungs 4-5 of the parity ladder are out of reach here.
- No multi-GPU shard parity (`discrete_gpu_count() == 0` self-skips it) - and note
  qwen35moe's own `shard_parity.rs` does not run on this machine either, so any
  claim it protects a refactor here is a claim about a different machine.
- No int8 device-path validation - whether the Intel iGPU exposes a usable DP4A
  path is a measurement, not an assumption; unmeasured until run there.
- No serving throughput/latency or residency measurement on real weights.
- MTP head: structurally implemented, **no reference oracle** (see above) -
  gradchecked and overfit-tested, never parity-claimed.
- Vision + decoder fused end-to-end on real weights is not runnable (needs both
  towers resident simultaneously).
- No NPU (`NpuModel`) implementation this port - the firmware blocker on this
  exact host is diagnosed separately, not re-run here.

Never write an intermediate full-precision whole-model file (~108 GB) - quantized
device buffers must be built directly from the compressed FP8 checkpoint, same
constraint as qwen35moe.
