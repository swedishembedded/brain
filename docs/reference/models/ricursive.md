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
| Speculative decoding with the `Ripple-65B-Preview-DFlash2` drafter (`--speculative dflash`, `--draft-model`), also beside `--sparse-attention` | [x] |
| Measured acceptance and speedup of the drafter against the real 65B target | [ ] |
| Full fine-tuning (f32 weights, AdamW on the device) and LoRA/QLoRA through the shared trainers (`brain finetune`, `ChatFineTune`, `ChatFullFineTune`, `PreferenceFineTune`) | [x] |
| LoRA over a bf16, fp8, nf4 or int8 base, or the ModelOpt NVFP4 release as published; the adapter exports as a PEFT directory | [x] |
| Pipeline-parallel training over several devices (`model::Pipeline`) | [x] |
| Native sparse attention training (N-LSA) on the global layers | [x] |
| Trained adapters served from a bank over ONE resident base (`--adapter`, `--adapter-manifest`, `--watch-adapters`, adapters in the model store, `--adapter-slots`, `--adapter-rank-cap`, `--adapter-targets`): one batch mixes the base and several adapters, selected per request by model id; `TextGenerationPipeline::adapter` | [x] |
| Training stages laid out over the selected cards from the memory plan (`--stages N` for an exact count) | [x] |
| Loss over the vocabulary a chunk of rows at a time (no `[rows, vocab]` logits) | [x] |
| Resident or layer-streamed low-precision full fine-tuning (`--train-precision`); norms and `value_res_lambda` trained beside an adapter (`--modules-to-save`) | [x] |
| Run and measured on a GPU (CUDA or Vulkan): parity, tiers, serving, long context, training | [ ] |

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
| `--speculative MODE` | `off` (default, and `auto`), `ngram` (prompt lookup) or `dflash` (the DFlash2 draft model). `mtp` is refused by name; `ngram` takes no `--draft-model`. Both work beside `--sparse-attention`. |
| `--draft-model DIR` | The DFlash2 draft model: a directory, or the `<vendor>/<repo>` of a pulled one (looked up beside the target in the model store); implies `--speculative dflash`. Without it `--speculative dflash` reads `<repo>-DFlash2` (the target's repo without its quantization suffix) and says what to pull when it is not there. |
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
module. Brain keeps activations in float (W4A16: it does not quantize them to FP4 as the
release was calibrated for), so the scales are checked at open (one finite positive
scalar per module, none missing) and not applied. How far the release stays from the bf16
model is measured on two layers only.

The byte counts on this page are the memory plan's arithmetic; no card has measured
them. The sliding layers keep a fixed ring of recent keys per sequence and the
global layers keep a growing paged cache, so the memory a sequence costs is a
fixed ring plus a per-token share of the global layers
(`ricursive::ring_bytes_per_slot`, `ricursive::global_kv_bytes_per_token`).
`ricursive::placement::footprint` states every term from the same functions
the engine allocates with, and the resident registers exactly that with the
scheduler. A cached prompt is reused at the length it was filed at, and at the point where a later prompt left it: a prefill that leaves a cached prompt (or that the scheduler hints) files a prefix of its own there, rounded down to a KV block, so prompts that share a system prompt or document and then diverge reuse it. A re-sent conversation resumes from its previous prompt; the reply is prefilled again.

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
Verification works beside `--sparse-attention`: an index page is built when a forward
starts, over the positions cached before it, so a verify that crossed a cluster block of the index
would read a state a plain decode step builds one step later; the scheduler keeps each verify inside one
block (`PagedDecoder::spec_rows_limit`), and the output equals plain greedy decoding with the same index.

`ricursiveAI/Ripple-65B-Preview-DFlash2` (4.04 B parameters, a `DFlash2DraftModel`, see [DFlash2](dflash2.md))
drafts blocks of 16 tokens conditioned on the target's hidden states at layers 1, 20, 39, 58 and 77 (the output of
each layer, five 5120-wide taps, fused by a `25600 -> 5120` projection and an RMSNorm), through four sliding-window
layers (window 4096, RoPE theta 10000) and one full NoPE layer, each wrapped in dynamic four-tap convolutions
(groups of 16), and a rank-256 candidate selector over predecessor and successor codebooks walks the target head's
top 32 candidates into one path. The draft shares the target's embedding (the rows as the checkpoint stores them,
without `embed_scale`) and `lm_head`. Two conventions the published files do not settle,
the tap being the output of layer `i` and the unscaled noise embedding, are taken as
stated until the acceptance against the real target confirms them.

```bash
brain pull ricursiveAI/Ripple-65B-Preview-NVFP4
brain pull ricursiveAI/Ripple-65B-Preview-DFlash2
brain serve --openai --device gpu0 --kv-cache int8 --speculative dflash
```

The engine retains the residual at the five layers for every position it consumes (prefill, decode and verify rows,
bf16 on the host: 50 KiB a position, in blocks shared with the KV so prefix reuse and rollback need nothing more), and
`PagedDecoder::draft` runs the draft for every sequence of the step in one forward: each sequence has a cache slot
keyed by its request that is fed the taps of the positions it has not seen. The memory plan charges the draft's
weights (7.42 GiB), its K/V caches (held at the `--kv-cache` tier; a ring of window plus one block for each sliding layer per live
sequence, and the full layer paged in one pool as wide as the target's own block pool, handed back when a request
ends: 0.29 GiB at 128k tokens with an int8 cache) and one block of activations per sequence to the
last stage's card, and the taps to the host: on one 96 GiB card with the NVFP4 release and an int8 cache the longest
dense context falls from about 962k to about 684k tokens with the draft attached.

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
workgroup-barrier kernels, and is refused by name elsewhere. A trained adapter is served with
`brain serve --adapter DIR/adapter.safetensors` (or from a watched directory or the model store): it must have been
trained on this checkpoint (its card records the base) and fit the model, and is offered as a model under its card id
(`--adapter-id`, a reference of the form `vendor/repo:owner:name:tag`). All adapters and the base share one copy of the
weights; the adapters' slots on each device are part of the memory plan. A variant of the adapter Ripple does not train (rank-stabilised, LoRA+, frozen A) is refused by name.

A full fine-tune (no `--lora`) trains f32 weights with f32 AdamW on the device by default. `--train-precision
weights=bf16` (or `fp8`) instead holds the linears and the two tables on the device in that encoding with an f32
master copy and the optimizer state in host memory (`state=int8` shrinks it), and `window=N` keeps only N layers on
the device at a time, streaming the rest from the host: a bf16 65B fine-tune then needs one card of about 42 GiB at
one row of 4096 tokens and 370 to 490 GiB of host memory. A resident run works on one card or, with `--stages N` (no `window=`), as a pipeline of stages that each hold their share
of the weights, gradients and optimizer state: the 65B at 4096 tokens needs four cards. It writes the masters as a base the
trainer opens again. `--modules-to-save norms,gates` (a comma list of tensor names, leaves such as `input_layernorm`, or the groups `norms`
and `gates`) also trains those base tensors in full and carries them in the adapter file (PEFT's `modules_to_save`):
the card lists them, and `brain serve` applies each row's own copy in a batch that mixes the base and several
adapters. A full linear (`g_proj`, the embedding, the head) trains and is saved, but an adapter carrying one is
refused when served, by name; `export_peft` writes the norms as PEFT `modules_to_save` and refuses the bare
`value_res_lambda`. `--held-out FILE` scores a chat JSONL set on the model in the training process before the first step and after the
last (the loss is printed). `--policy-gradient` (with `--ratio-clip` and `--kl-beta`) trains the clipped policy gradient on
rollout records, each carrying the `advantage` of its rollout, instead of the supervised loss: the actions of
well-rewarded rollouts become likelier and those of badly rewarded ones less likely. The log-probabilities the
objective reads are the serving engine's for the same tokens, so a policy trained on the trainer's forward is the one
the engine samples from.

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

Package: `brain-ricursive` (engine), `brain-dflash2` (the draft model), `brain-catalog` (`resident_ricursive`).
