# Training and serving capability matrix

Which architecture can be fine-tuned how, over which checkpoint precision,
and served with adapters and which KV tiers. Every cell names the test that
backs it; a cell with no test is written `-` and means "not offered", never
"untested but probably works". Per-model pages hold the commands. This page
does not record speed or memory figures.

Tests are named `crate/tests/file.rs::test_name` (or `crate/src/file.rs` for a
unit test). A test that needs a real checkpoint or a CUDA card is marked
`(real)` or `(cuda)`; the others run on the CPU backend with random weights.

## Language models

| Architecture | Full fine-tune | LoRA, fp32 base | LoRA, bf16 base | LoRA, int8 base | LoRA, NF4 base (QLoRA) |
|---|---|---|---|---|---|
| `qwen3` | fp32 only, optimizer moments in host RAM: `qwen3/tests/finetune_full_hf.rs::a_full_fine_tune_of_a_transformers_directory_trains_every_weight_and_saves_one_checkpoint` | `qwen3/tests/lora_learning_gate.rs::lora_adapters_trained_on_different_targets_diverge_and_match_their_own_data_after_a_reload` | `qwen3/tests/lora_bf16_base.rs::the_bf16_base_trains_the_adapters_the_fp32_base_does` | `qwen3/tests/lora_int8_base.rs::the_int8_base_trains_the_adapters_the_fp32_base_does` | - |
| `qwen35` | fp32 resident only: `qwen35/tests/convergence.rs::qwen35_full_finetune_overfits_fixed_batch` | `qwen35/tests/convergence.rs::qwen35_lora_overfits_fixed_batch`; streamed per layer: `qwen35/tests/stream_train_equivalence.rs::streaming_lora_trainer_matches_the_resident_trainer_exactly` | `qwen35/tests/lora_bf16_base.rs::the_bf16_base_trains_the_adapters_the_fp32_base_does` | - | `qwen35/tests/lora_nf4_base.rs::the_nf4_base_trains_the_adapters_a_fp32_build_over_its_dequantised_values_does`, `::a_4_bit_base_still_overfits_a_fixed_batch`; real 27B (real, cuda): `qwen35/tests/stream_train_real.rs::short_real_streaming_lora_finetune_reduces_loss_and_shifts_generation` |
| `qwen35moe` | - | `qwen35moe/tests/lora_freezes_base.rs::lora_training_only_updates_adapters_cpu` (mixer projections only, never experts) | - | - | - |
| `glmdsa` | `glmdsa/tests/convergence.rs::glm_overfits_fixed_batch` | - | - | - | - |
| `gpt2` | `gpt2` trains from scratch (see its page) | - | - | - | - |

A qwen35 bf16 or NF4 base is frozen: combining it with full fine-tuning is
refused (`qwen35/tests/lora_bf16_base.rs::a_narrow_base_without_lora_is_refused`).
No architecture offers a full fine-tune over a quantized or fp8 base, and
none stores trainable parameters or optimizer moments below fp32.

## Image and video generators

| Architecture | LoRA, fp32 base | LoRA, int8 base | Other |
|---|---|---|---|
| `flux2` | host reference and device trainer: `flux2/tests/dev_grad.rs`, `flux2/tests/device_train.rs` | device trainer, CUDA: `flux2/tests/int8_trainer.rs`, `flux2/tests/int8_base_grads.rs` | adapters fold or apply at run time, including over an int8 base: `flux2/tests/lora_stack.rs::two_adapters_stack_each_at_its_own_scale`, `flux2/tests/lora_runtime_int8.rs::an_int8_runtime_adapted_forward_tracks_its_fp32_reference` |
| `s3dit` | `s3dit/tests/lora_train.rs::lora_only_overfits_with_base_frozen`, `s3dit/tests/device_train.rs::device_grads_match_host` | - | full model trains: `s3dit/tests/model_train.rs` |
| `wan` | `wan/tests/dev_lora_train.rs::device_lora_is_a_no_op_at_init_then_descends_with_the_base_frozen` | - | full device trainer: `wan/tests/device_train.rs::the_device_trainer_overfits_one_batch` |

## Serving adapters and KV tiers

| Architecture | Adapters while serving | Quantized weights while serving | KV tiers |
|---|---|---|---|
| `qwen3` | many adapters in one batch beside the base, base rows unchanged: `qwen3/tests/engine_adapter_bank.rs::mixed_batch_rows_equal_the_same_rows_run_alone_fp32`, `::mixed_batch_rows_equal_the_same_rows_run_alone_int8_weights`, `::base_rows_in_a_mixed_batch_are_bit_identical_int8`; one adapter attached or folded: `qwen3/tests/lora_runtime_adapter.rs::attached_adapter_matches_the_fold_and_detach_restores_the_base_exactly`; catalog level: `catalog/tests/multi_adapter.rs::base_and_adapters_share_one_instance_and_one_batch` | int8 and bf16 linears: `qwen3/tests/engine_half_weights.rs::half_precision_linears_compute_what_fp32_does_on_the_rounded_weights`, `qwen3/tests/engine_adapter_bank.rs::mixed_batch_rows_equal_the_same_rows_run_alone_int8_weights` | fp32 and int8: `qwen3/tests/engine_adapter_bank.rs::mixed_batch_rows_equal_the_same_rows_run_alone_fp32_weights_int8_kv`; host offload (real): `qwen3/tests/kv_offload_real.rs::host_kv_offload_swap_cost_and_concurrency_on_a_real_checkpoint` |
| `qwen35` | bank of adapters in a batch: `qwen35/tests/adapter_bank.rs::mixed_batch_matches_training_forward_and_base_rows_are_untouched_cpu`; folded: `qwen35/tests/lora_serve_fold.rs::folded_serve_matches_live_training_and_diverges_from_the_unadapted_base` | int8 and int4 linears: `qwen35/tests/model_i8_smoke.rs::int8_forward_tracks_fp32_within_quant_tolerance_default_backend`, `qwen35/tests/model_q4_smoke.rs::q4_forward_tracks_fp32_within_quant_tolerance_default_backend`; real GGUF (real): `qwen35/tests/gguf_resident_real.rs` | fp32, bf16, int8: `qwen35/tests/kv_tier.rs::bf16_kv_tracks_f32_logits_on_the_tiny_hybrid`, `::int8_kv_tracks_f32_logits_at_the_real_attention_dims`; real GGUF (real): `qwen35/tests/gguf_kv_tier_real.rs::compact_kv_tiers_agree_with_the_f32_cache_on_a_real_prompt` |
| `qwen35moe` | - | int8 mixer linears and head: `qwen35moe/tests/model_i8_smoke.rs::int8_forward_tracks_fp32_within_quant_tolerance_default_backend` | compact pool: `qwen35moe/tests/engine_tiers.rs::a_compact_kv_pool_answers_like_the_f32_pool` |

## Checkpoint precision on load

| Source | What stays packed | Test |
|---|---|---|
| Q8_0 GGUF into int8 weights | the blocks are repacked, never expanded to fp32 | `flux2/tests/gguf_direct_int8.rs::gguf_int8_is_bit_identical_to_the_fp32_round_trip`, `model/src/int8.rs` unit tests on `quantize_from` |
| brain packed-int8 safetensors | words copied as stored | `checkpoint/src/weightio.rs::with_tensor_u32_chunks_is_memory_bounded` |
| FP8 HF directory into bf16 or NF4 base | nothing: each tensor is dequantized to fp32 on the host, one tensor at a time, then packed | `qwen35/tests/fp8_source.rs::streaming_the_whole_model_keeps_host_allocation_near_one_tensor` |
| Other GGUF quant types (Q4_K and the rest) | nothing: decoded to fp32 and requantized to the requested tier | `gguf/tests/kquant.rs` covers the packed relayout only; no loader uses it yet |

Training over an FP8, NVFP4 or MX checkpoint without expanding it is not
available: there is no such `Weight` variant. The contributor roadmap for these gaps is
`.agents/roadmap/quantized-training.md`.
