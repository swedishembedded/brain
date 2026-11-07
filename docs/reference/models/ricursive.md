# Ripple-65B

Ripple-65B (`RicursiveForCausalLM`, `model_type: ricursive`) is an 80-layer
dense decoder that mixes two kinds of attention layer, four of one then one of
the other: sliding-window layers with RoPE, and global layers with full causal
attention and no positional rotation. Each block also has a per-layer value
residual, a sigmoid-gated attention output, per-head QK RMSNorm, four norms
(`input`, `post_attention`, `pre_feedforward`, `post_feedforward`) and a
clipped SwiGLU feed-forward. It is served like Qwen3: through the same
scheduler, tokenizer, chat-template, reply-parsing and policy paths.

## Support

| Capability | Supported |
|---|---|
| Config and two-way tensor import from the released safetensors | [x] |
| f32 prefill forward (every position's logits) on every backend | [x] |
| Parity with the reference implementation (random weights, real layers) | [x] |
| Paged serving engine under the generic scheduler | [x] |
| `brain serve` over the OpenAI, Anthropic and OpenRouter APIs and D-Bus; `brain do ricursive generate` | [x] |
| Chat with the checkpoint's own template, reasoning split out, OpenAI tool calls, streaming | [x] |
| Resolves from `ricursiveAI/Ripple-65B-Preview`, `ricursiveAI/Ripple-65B-Preview-NVFP4` or a local directory | [x] |
| Weights held as the checkpoint stores them (bf16; the official ModelOpt NVFP4 release with the MLP in NVFP4) | [x] |
| Weight formats `--weight-format` f32, bf16, f16, int8, fp8 | [x] |
| KV cache `--kv-cache` f32, bf16, int8, fp8 (global pools and sliding rings) | [x] |
| Latent sparse attention over the global layers (`--sparse-attention FILE`, `brain lsa`) | [x] |
| Layers staged over several cards, chosen by the memory plan from the device selection | [x] |
| Speculative decoding: batched verify with prompt-lookup drafting (`--speculative ngram`) | [x] |
| Speculative decoding with the `Ripple-65B-Preview-DFlash2` drafter | [ ] |
| Full fine-tuning (f32 weights, AdamW on the device) and LoRA/QLoRA through the shared trainers (`brain finetune`, `ChatFineTune`, `ChatFullFineTune`, `PreferenceFineTune`) | [x] |
| LoRA over a bf16, fp8, nf4 or int8 base, or the ModelOpt NVFP4 release as published; the adapter exports as a PEFT directory | [x] |
| Pipeline-parallel training over several devices (`model::Pipeline`) | [x] |
| Native sparse attention training (N-LSA) on the global layers | [x] |
| `brain serve --adapter FILE`: a trained adapter as a model of its own beside the base (its own copy of the weights) | [x] |
| Training stages laid out over the selected cards from the memory plan (`--stages N` for an exact count) | [x] |
| Loss over the vocabulary a chunk of rows at a time (no `[rows, vocab]` logits) | [x] |
| Layer-streamed or resident low-precision full fine-tuning of the 65B; a bank of adapter slots in `brain serve`; norms and `value_res_lambda` trained beside an adapter | [ ] |

## Serving it

```bash
brain pull ricursiveAI/Ripple-65B-Preview-NVFP4        # or ricursiveAI/Ripple-65B-Preview (bf16)
brain serve --openai --device gpu0 --kv-cache int8     # one 96 GiB card: the NVFP4 release
brain serve --openai --device gpu0,gpu1                # the bf16 release over two cards
```

`brain serve` finds the directory by the architecture its `config.json`
declares (`RicursiveForCausalLM`), so a local copy under any name works too:
put it at `<models dir>/<vendor>/<repo>`. A pull fetches `config.json`, the
tokenizer files, `chat_template.jinja` and the safetensors shards, nothing
else (the remote code in the repository is never fetched or run).

What the startup log tells you: the values the defaults chose
(`default kv-cache bf16`, `default ctx 131072`) and the placement
(`gpu0 layers 0..80: 52.3 GiB + 1.0 GiB fixed`). Anything the plan cannot
place is left out with the arithmetic: what the deployment needs (weights,
global KV, rings, index, fixed bytes per stage) and what the cards offer.

| Option | Ripple |
|---|---|
| `--weight-format SPEC` | `f32`, `bf16`, `f16`, `int8`, `fp8`, or a per-leaf list (`bf16,mlp=fp8`). Default: as the checkpoint stores them. NVFP4 cannot be asked for, because nothing quantizes into it; the NVFP4 release is simply served as published. |
| `--kv-cache TIER` | `f32`, `bf16`, `int8`, `fp8`. Default `bf16`, or `int8` with `--sparse-attention`. |
| `--sparse-attention FILE` | The indexer `brain lsa distill --arch ricursive` writes. It must be built for this model (geometry, global layers only, no tree, cluster block a whole number of KV blocks) and an f32, int8 or fp8 cache; anything else is refused at startup, naming the flag. |
| `--ctx N` | Per-sequence tokens. Default: the longest context that places on the selected cards, at most the model's own limit (2 Mi). An explicit value that does not place is refused, never shortened. On the host tier the default is 4096. |
| `--max-batch N` | Sequences decoded together (default 4); each costs a ring of the sliding layers. |
| `--speculative MODE` | `off` (default, and `auto`) or `ngram`. `mtp` and `dflash` are refused by name; `ngram` cannot be combined with `--sparse-attention`. |
| `--device SPEC` | Decides how many cards the layers are cut over: the fewest of the selected cards that hold the deployment, one pipeline stage each. `--device cpu` runs one stage on the host. `--device software` runs it on the software Vulkan adapter, where the sparse-attention kernels run without a GPU (validation only). |

`--exact-decode`, `--gemm`, `--gdn-state`, `--act-precision`,
`--linear-activations` and `--kv-calibration` are refused by name: Ripple has
no such path.

## Chat, reasoning and tool calls

The tokenizer is the checkpoint's `tokenizer.json` through brain's shared
byte-level BPE (roughly 100k tokens), and the prompt is the checkpoint's own
`chat_template.jinja` rendered by brain's shared template engine, not a copy
of it: when a release ships that file beside an older template inline in
`tokenizer_config.json` (Ripple does), the file wins, as in `transformers`.
`chat_template_kwargs` reach the template, so `{"clear_thinking": true}`
drops earlier turns' reasoning from the prompt (the default keeps it).

A reply is scanned into `reasoning_content` (between `<think>` and
`</think>`), `content`, and `tool_calls` (`<tool_call>` JSON objects, any
number). The newlines the template writes between those parts are framing and
are removed, so an assistant message sent back unchanged renders to exactly the
tokens the model generated. A tool schema reaches the template in the order
the client wrote it, and a tool call's arguments as the object they encode.
Streaming carries the same split as `reasoning_content`, `content` and
`tool_calls` deltas (a call's name first, then argument fragments).

The template has no thinking switch: `enable_thinking` and
`thinking_budget` (which needs `<think>` as a token) do not apply, and the
model decides whether to reason. Text after a tool call is returned as content
rather than dropped.

## Memory

By default every linear weight is held as the checkpoint stores it: the bf16
release stays bf16 on the device, and
`ricursiveAI/Ripple-65B-Preview-NVFP4` is loaded as published, with no
conversion step, its MLP in NVFP4 (E2M1 codes, an E4M3 scale per 16 weights,
an f32 per-tensor scale) and its attention, embedding and `lm_head` in bf16.
A format the device cannot hold is refused naming the tensor and the format it
would have used (int8 needs a packed-int8 dot, which the CPU backend does not
have).

The NVFP4 release is W4A4: it stores a static input scale per quantized
module. Brain keeps activations in float (W4A16, at least as precise as the
reference), so the scales are checked at open (one finite positive scalar per
module, none missing) and not applied.

The sliding layers keep a fixed ring of recent keys per sequence and the
global layers keep a growing paged cache, so the memory a sequence costs is a
fixed ring plus a per-token share of the global layers
(`ricursive::ring_bytes_per_slot`, `ricursive::global_kv_bytes_per_token`).
`ricursive::placement::footprint` states every term from the same functions
the engine allocates with, and the resident registers exactly that with the
scheduler. A cached prompt is reused at the length it was filed at.

Attention runs the fused paged flash kernels (no score slab, so no cap on the
context a dense read spans, at every KV tier) on a device with workgroup
barriers, and the three-kernel reference on the CPU backend.

## Speculative decoding

The engine verifies `1 + proposals` consecutive rows of a sequence in one
forward and the scheduler keeps the longest prefix the model agrees with, so
greedy output is unchanged whatever the drafter proposes (the shared
conformance suite and `a_verify_scores_every_row_and_a_rolled_back_one_leaves_no_trace`
check it, across ring wraps and block edges). `--speculative ngram` drafts by
prompt lookup; the acceptance figures are reported with the model's metrics
(`spec_proposed`, `spec_accepted`, `spec_accept_rate`, `spec_tokens_per_round`).
Verification is off beside `--sparse-attention`: a rejected row would leave
index pages built over rolled-back keys.

`ricursiveAI/Ripple-65B-Preview-DFlash2` (4.04 B parameters, a
`DFlash2DraftModel`) is not run yet. It drafts blocks of 16 tokens
conditioned on the target's hidden states at layers 1, 20, 39, 58 and 77
(five 5120-wide taps, fused by a `25600 -> 5120` projection), through four
sliding-window layers (window 4096) and one global NoPE layer with dynamic
four-tap convolutions and a rank-256 candidate selector over predecessor and
successor codebooks; the engine would have to retain those hidden states per
context token and the drafter's block forward would have to be ported (the
Qwen3.8 implementation in `qwen35::dflash2` is bound to its GGUF
single-stream path, not to the scheduler). `--speculative dflash` says so and
refuses.

## Training it

`brain finetune BASE --dataset chat.jsonl --out DIR` trains every weight (the checkpoint, with the tokenizer files,
is written under DIR); with `--lora RANK` it trains a LoRA adapter over the frozen base instead and writes
`DIR/adapter.safetensors`. `BASE` is a local file or directory in the release layout, a brain-format checkpoint,
or a model id. `brain ricursive finetune ...` is the same command. The adapter targets `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `g_proj`, `gate_proj`, `up_proj` and `down_proj` of every layer; `--base-format`
(`f32`, `bf16`, `int8`, `nf4`, `fp8`, or `nvfp4` for the NVFP4 release) says how the frozen base is held,
`--activations keep|recompute|bf16|fp8` what the backward keeps (Ripple recomputes every layer from its input
boundary, in f32 or narrowed, and refuses `keep` by name), `--continue FILE` continues an adapter at its own rank.
`--stages N` cuts the model into N pipeline stages (default: one when a device holds it, else the fewest that fit the
selected cards, printed). `--native-sparse` (with `--lsa-window`, `--lsa-sink`, `--lsa-top-k`, `--lsa-top-p`,
`--lsa-cluster-block`, `--lsa-window-only`) trains the global layers with native sparse attention on a device that runs
workgroup-barrier kernels, and is refused by name elsewhere. The trained adapter is served with
`brain serve --adapter DIR/adapter.safetensors`: it must have been trained on this checkpoint (its card records the
base) and fit the model, and is offered under its card id (`--adapter-id`). A variant of the adapter Ripple does not train (rank-stabilised, LoRA+, frozen A) is refused by name.

Every operation has a backward, checked against finite differences and against PyTorch autograd of the
released `modeling_ricursive.py` on random weights and on the real layers 0 and 4. The value residual reads the
embedding in every layer, so a pipeline stage holds the embedding table and recomputes it from the tokens. With
a library, `ricursive::Ripple` is a `model::Model` and `model::Shardable` (`model::Pipeline::<Ripple>`), and
`Ripple::enable_native_lsa` trains the global layers with native sparse attention (a device that runs
workgroup-barrier kernels).

## Using it as a library

```rust
let cfg = ricursive::RicursiveConfig::from_file("config.json")?;
let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
let weights = ricursive::open_safetensors("model.safetensors", &cfg, &layers)?;
let out = ricursive::RicursiveModel::new(cfg, &weights, None).forward(&token_ids)?;
// out.logits is [tokens, vocab]; out.layer_outputs holds every layer's output.
```

To serve it in a program, open a source with
`ricursive::open_shared_checkpoint`, build a `ricursive::RicursiveEngine` from
it and a `ricursive::ServeConfig` (block size, block count, sequence limit,
live sequences, cached prompts, prefill chunk) and hand it to
`model::serve::Scheduler::new`; the layers can be cut into pipeline stages
(`RicursiveEngine::from_source_staged`, with the plan from
`ricursive::placement`). Its global layers read through latent sparse
attention with `RicursiveEngine::set_lsa` (an indexer covering only global
layers, over an f32, int8 or fp8 cache, also across stages), which needs a
device that runs workgroup-barrier kernels.

The importer refuses a checkpoint with a missing tensor, an unknown tensor, a
wrong shape or a non-float dtype, naming the tensor, before it reads any
weights. A config that sets a RoPE scaling, an attention or MLP bias, a
non-SiLU activation or tied embeddings is refused, because this port does not
implement them.

Package: `brain-ricursive` (engine), `brain-catalog` (`resident_ricursive`).
