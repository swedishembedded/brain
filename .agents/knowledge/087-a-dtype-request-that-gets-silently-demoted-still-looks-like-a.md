<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 87. A dtype request that gets silently demoted still LOOKS like a fix - `/proc/<pid>/smaps_rollup`'s `Pss_File` vs `Anonymous` split is what actually tells "reclaimable cache" from "real growth"

Loading MiniMax-H3's 63GB Qwen3-VL text encoder (`minimaxh3::caps::
build_text_encoder`) climbed to within a few GB of a 150GB container cap,
survived, but stayed suspicious: a 50-of-64-layer TRUNCATED shard at int8
should need ~25GB, not ~140GB. The instinct to blame page cache (`Qwen3Vl::
from_hf`'s mmap-backed streaming source, still touching every shard file it
reads) was wrong and nearly went unchallenged - a real run's own memory
curve LOOKED like reclaimable cache (climbed, plateaued near the cap,
eventually finished without being killed), which is exactly what page-cache
pressure-driven reclaim looks like from the outside. The user's insistence
that this was "obviously a bug, not a feature" forced checking rather than
assuming: `cat /proc/<pid>/smaps_rollup` showed `Pss_File: ~132MB` (page
cache was NEVER the dominant cost - the `advise_dontneed_tensor` streaming
fix was already working) against `Anonymous`/`Private_Dirty: ~90GB` - real,
non-reclaimable heap growth. The actual cause: `gpu_core::select::Dtype::
promote` (`backend_api::DType::promote`, called inside `Weight::upload`)
SILENTLY demotes an unsupported tier back to fp32, and the CPU JIT
backend's `NumericSupport.int8_dot` is `false` (`backend-cpu`'s own
`Caps`: "the multi-barrier packed-int8 GEMMs are outside the JIT's
single-barrier model, and there is no VNNI fast path yet") - so
`build_text_encoder`'s `Dtype::I8` request had been silently landing as
`Dtype::F32` (4x the intended bytes) through EVERY earlier attempt in this
port's history, including the ones that appeared to "test int8 vs fp32 and
found no difference" (there never was a difference to find - both requests
built the identical fp32 model). `Weight::upload`'s own doc already said
"read `Qwen::linear_dtype` for what actually landed, never this request" -
advice that existed and was not followed.

**Rules going forward**:
- When a real run's memory trace is ambiguous between "reclaimable cache,
  self-corrects under pressure" and "real growth, will eventually OOM",
  check `/proc/<pid>/smaps_rollup`'s `Pss_File`/`Pss_Anon` split before
  building a theory on top of the ambiguous top-level number. A climb that
  plateaus near a cap is not evidence either way on its own.
- A dtype request has TWO separate places it can silently fail to apply:
  the SOURCE side (a loader that always decodes to fp32 regardless of the
  destination dtype - lesson still applies from this port's OWN earlier
  `read_model_dir` finding) and the DESTINATION side (`promote` demoting
  an unsupported tier). Fixing one does not prove the other was never a
  problem - both must be checked, and checked on the ACTUAL device the
  real run used, not assumed from the tier name in the source code.
- A caller requesting a non-default numeric tier for its memory/speed
  properties, not just its accuracy properties, must query
  `Gpu::caps().numeric` (or read back the tier that actually landed) rather
  than hardcode the request - the request name is a wish, not a
  guarantee, and the gap between them is invisible until someone measures
  actual resident bytes against the arithmetic the tier name promises.
