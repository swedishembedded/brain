# DFlash2 draft models

DFlash2 (`DFlash2DraftModel`) is a block-diffusion **draft model** for speculative
decoding. It is not a language model that serves on its own: a target's engine
loads it, and it proposes the next 15 tokens of every sequence in one forward
instead of one token per step. The scheduler verifies the proposals in one batched
target step and keeps the longest prefix the target agrees with, so greedy output is
unchanged and a sampled sequence keeps the target's distribution (rejection
sampling).

What makes it more than a small decoder:

* It reads the target's **hidden states**, not its tokens: the residual stream at a
  handful of the target's layers (`dflash_config.target_layer_ids`), fused by a
  `taps x width -> width` projection and normed. The target engine retains those
  rows for every position it consumes (`PagedDecoder::enable_taps`).
* The block `[anchor, MASK, ..., MASK]` is denoised **jointly** (the attention is not
  causal), through dynamic grouped convolutions around every attention and MLP
  sublayer whose taps are projected from the activations.
* Per-layer attention from the config: `layer_types` (a sliding window, or full
  attention) and `no_rope_layers` (a NoPE layer rotates nothing).
* A **candidate selector** walks the target head's top-k candidates at each position
  into one coherent path (greedy, at temperature 0).
* It stores neither the token embedding nor the output head: the target's are shared.

## Support

| Capability | Supported |
|---|---|
| Config from a Hugging Face `config.json` (`dflash_config`, `layer_types`, `no_rope_layers`, `rope_parameters`) or from GGUF metadata | [x] |
| Weights from the published bf16 safetensors as stored, or from a Q8_0 GGUF (int8) | [x] |
| Per-layer sliding window (a ring per sequence), full and NoPE layers | [x] |
| One batched forward over the blocks of several sequences | [x] |
| Parity with the host reference (`tools/goldens/dflash2_reference_forward.py`): a tiny random draft, the published `Ripple-65B-Preview-DFlash2` on the CPU backend | [x] |
| Ripple-65B through the generic scheduler: `brain serve --speculative dflash` | [x] |
| Qwen3.8-27B single-stream path (`qwen35::dflash2::Dflash2Drafter`) | [x] |
| Qwen3.8-27B behind the batched scheduler | [ ] |
| Measured acceptance and speedup against the real 65B target | [ ] |

## Using it with Ripple

```bash
brain pull ricursiveAI/Ripple-65B-Preview-NVFP4
brain pull ricursiveAI/Ripple-65B-Preview-DFlash2
brain serve --openai --device gpu0 --kv-cache int8 --speculative dflash
```

`--speculative dflash` reads the draft pulled beside the target (the target's repo
without a quantization suffix, plus `-DFlash2`); `--draft-model DIR` names another
directory or pulled `<vendor>/<repo>` and implies the mode. The memory plan charges the
draft's weights (7.4 GiB at bf16), its K/V caches and its
activations to the last stage's card (the caches at the target's `--kv-cache` tier, the full layers paged in one pool as wide as the target's block pool, handed back when a request ends), and the retained hidden states (5 x 5120 bf16
values a position) to the host, so the default context shrinks to leave room. A draft
built for another target (width, vocabulary, the layers it reads, the mask token) is
refused by name.

Beside `--sparse-attention` the verify is kept inside one cluster block of the
index, so every row reads the index a plain decode step at its position would.

## Using it as a library

```rust
let cfg = dflash2::Dflash2Config::from_json(&std::fs::read_to_string("config.json")?)?;
let source = dflash2::HfDraft::open(std::path::Path::new("draft-dir"))?;
let gpu = gpu_core::Gpu::open(None, &dflash2::pipelines(&cfg));
let model = dflash2::Dflash2::load(gpu, cfg, Box::new(source), /* slots */ 4, /* positions */ 8192)?;
let drafter = dflash2::Dflash2Drafter::new(model); // propose(&host, &taps, &requests)
```

A target engine provides `dflash2::DraftHost` (embedding in, head out) and
`dflash2::TapRows` (its retained hidden states);
`ricursive::RicursiveEngine::set_drafter` is the wiring for Ripple.

Package: `brain-dflash2` (the model), `brain-ricursive` (the engine side).
