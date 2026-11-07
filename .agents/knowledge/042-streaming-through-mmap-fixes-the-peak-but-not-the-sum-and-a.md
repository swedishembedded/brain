<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 42. Streaming through mmap fixes the peak but not the sum, and a checkpoint's index filename is a naming convention, not a guarantee

**Lesson:** Streaming through mmap fixes the peak but not the sum, and a checkpoint's index
filename is a naming convention, not a guarantee.

## What happened
Fixing Z-Image's OOM on a memory-constrained box took three independent bugs, each invisible until
a real large checkpoint ran on a real machine.

1. **The index filename was hardcoded.** `checkpoint::weightio::WeightReader::open_hf_dir` and
   `safetensors::read_model_dir` both hardcoded the HF-transformers index filename
   (`model.safetensors.index.json`).
   - A diffusers-style checkpoint (`diffusion_pytorch_model.safetensors.index.json`, what
     Z-Image's `transformer/` dir actually ships) didn't fail to parse. It silently took the "no
     index found → there must be exactly one shard" fallback and opened only the
     alphabetically-first file.
   - Two-thirds of the checkpoint was never even mapped. The failure surfaced as a late,
     unrelated-looking panic ("missing tensor X") deep in a totally different layer, not as an
     error at open time.
2. **mmap streaming bounded the wrong resource.** Streaming a tensor from an mmap and dropping the
   returned `Vec<f32>` bounds *heap* allocation to one tensor, but says nothing about the mapped
   file's *page-cache* footprint.
   - Every page touched during a whole-checkpoint scan stays resident (counted in RSS) until the
     kernel feels enough pressure to reclaim it, and under swap exhaustion that reclaim can lose
     the race against the OOM killer.
   - `MADV_DONTNEED` existed in this repo for exactly this and had zero callers.
3. **A per-model default was topology-specific.** Keeping the encoder on CPU so it doesn't compete
   with the DiT for a card's VRAM was correct reasoning for a multi-GPU box. It was pure
   pessimization on a single-GPU box, because there the CPU alternative isn't cheaper: it's a
   permanently-resident fp32 copy instead of a smaller, on-demand, auto-dropped int8 one.

## Why
None of the three had a unit test written ahead of time that would have caught it. Each needed the
real checkpoint's actual shard layout, the real machine's actual memory pressure, or the real
machine's actual GPU count to manifest.

## What to do
- **A wrong-but-plausible fallback path masquerading as the single-shard case** is the general
  shape of bug 1. It will recur for any reader that treats "recognized index absent" as "no index
  exists" rather than "check for the other convention, then fail loudly."
- **A "streaming" read path is only actually bounded in RSS if something explicitly evicts what it
  just consumed.** mmap alone bounds the wrong resource.
- **A safe default on one topology is not a safe default on every topology.** Branch the default on
  what's actually true of the machine (here: GPU count), not on what's convenient to assume.

The fix in every case was cheap once found; the finding required actually running the thing end to
end, not reasoning about it in the abstract.
