# Wan (video generation)

Wan turns a text prompt - or a single still image - into a short video: about
five seconds, 81 frames at 16 fps. <!-- perf-number: the output frame rate is a property of the video the model generates, not a throughput claim -->
It is the first video model in brain, and the smallest variant is deliberately
modest: Wan2.1-T2V-1.3B generates 480p video in roughly 8 GB of VRAM, which puts
it on consumer hardware rather than on a datacentre card.

Reach for it when you want motion. For a single still frame, the image models
(`s3dit`, `flux2`) are far cheaper - Wan spends most of its work on making
frames agree with each other over time, which is wasted if you only need one.

`wan` names the whole upstream family rather than one release, the same way
`qwen3` covers every Qwen3 size. Wan2.1 and Wan2.2 share one architecture, one
Hugging Face class and one GGUF architecture tag (`wan`); which release you have
is a matter of configuration, not a different model.

## Support

| Capability | Supported |
|---|---|
| Inference | [ ] |
| LoRA fine-tune | [ ] |
| INT8 | [ ] |
| CLI (`brain wan <action>`) | [ ] |
| HTTP API | [ ] |
| D-Bus | [ ] |
| Batched serving | [ ] |

**The port is in progress.** The architecture id, the variant configuration and
the naming contract are registered; the model, importer, VAE and pipeline are
being built on top of them. `.agents/roadmap/wan.md` tracks what remains and
carries the bug ledger. This page becomes a user-facing guide once the
inference path lands - until then, treat the table above as the honest status.

## Variants

| Variant | Parameters | Sizes | Steps | Shift |
|---|---|---|---|---|
| `t2v-1.3B` | 1.3 B | 832x480, 480x832 | 50 | 5.0 |
| `t2v-14B` | 14 B | + 1280x720, 720x1280 | 50 | 5.0 |
| `i2v-14B` (480p) | 14 B | 832x480, 480x832 | 40 | 3.0 |
| `i2v-14B` (720p) | 14 B | + 1280x720, 720x1280 | 40 | 5.0 |

Note that 720p exists only on the 14 B tier; the 1.3 B variant is 480p-only
upstream. The shift value is a task-and-size rule rather than a resolution rule:
only image-to-video at 480p departs from 5.0.

## Getting the weights

The default checkpoint is `Wan-AI/Wan2.1-T2V-1.3B` (17.6 GB), which is
self-contained - transformer, VAE, umT5-XXL text encoder and tokenizer all live
in the one repository. The `-Diffusers` variant of the same model is 28.9 GB
because it stores the text encoder in fp32 rather than bf16; brain reads the
native repository by default and uses the diffusers naming only for importing.

```
export BRAIN_WAN_DIT=.../diffusion_pytorch_model.safetensors
export BRAIN_WAN_VAE=.../Wan2.1_VAE.pth
export BRAIN_WAN_T5=.../models_t5_umt5-xxl-enc-bf16.pth
export BRAIN_WAN_TOKENIZER=.../google/umt5-xxl
```

The tokenizer variable names the *directory*, not a file: brain reads that
directory's `tokenizer.json` rather than the `spiece.model` protobuf beside it.
Both describe the same 256k pieces, but the JSON is what
`AutoTokenizer.from_pretrained` loads and therefore what upstream Wan actually
tokenizes with, and it spells out its own normalizer and pre-tokenizer instead
of leaving them to sentencepiece's defaults. See [t5encoder](t5encoder.md).

Image-to-video additionally needs a CLIP ViT-H/14 vision tower in
`BRAIN_WAN_CLIP`. That one is not auto-fetched, because it ships with the I2V
checkpoints rather than the text-to-video one.

## Hardware and limits

81 frames at 480p is 32,760 transformer tokens; at 720p it is 75,600. A dense
attention score matrix at those lengths would be 4.3 GB and 22.9 GB per head
respectively, so Wan always runs through chunked or flash attention - that is a
requirement here, not a tuning option.

<!-- perf-number: 4x/8x are the VAE's fixed compression strides, an architectural constant, not a speedup -->
The video autoencoder compresses time 4x and space 8x, and gives the first frame
a latent frame of its own. Frame counts therefore have to take the form
`1 + 4k`: 81 frames become 21 latent frames. Asking for 80 is rejected rather
than quietly rounded.
