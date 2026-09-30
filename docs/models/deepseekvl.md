<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# DeepSeek-VL

A Llama decoder reading image features from a hybrid vision tower: SAM-B
over the image at 1024 pixels and SigLIP-L over it at 384, joined by a split
MLP aligner.

| Checkpoint | Parameters | Licence |
|---|---|---|
| `deepseek-ai/deepseek-vl-7b-chat` | 7.3B | DeepSeek License |

**Status: runs as a library, not yet served.** `brain-deepseekvl` loads the
checkpoint as downloaded and runs the whole model: preprocessing, both
towers, the aligner, the conversation format with images spliced at their
placeholders, and greedy decoding. The decoder runs at the checkpoint's own
fp16 weights (about 14 GB of device memory). On the real checkpoint, every stage
matches the reference implementation, and 16 greedy tokens are identical to
its fp32 output. `brain serve` does not list the model yet.

```rust
use deepseekvl::prompt::{Role, Turn};

let m = deepseekvl::load(dir, qwen3::Dtype::F16, 2048)?;
let turns = [Turn { role: Role::User, content: "<image_placeholder>Describe this image.".into() }];
let ids = m.prompt_ids(&turns)?;
let embeds = m.image_embeds(&[image])?;
let reply = m.generate_greedy(&ids, &embeds, 256, &mut |_| {})?;
```

The checkpoint is recognized by its HF class `MultiModalityCausalLM`, which
Janus-Pro shares. The generation heads that only Janus-Pro configures tell the
two apart.

Package: `brain-deepseekvl`.
