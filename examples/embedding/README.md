# Long-context embeddings over D-Bus (LFM2.5-Encoder)

`embed_document.py` drives brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, see `examples/dbus/README.md` for the protocol):
the document travels **as a file descriptor** (sealed memfd) and the per-token
hidden states come back the same way — no bytes marshalled through D-Bus.
Concurrency goes through brain's residency executor: equal-length requests are
batched into one true batched forward on a device lane; the tokenizer runs on
the dispatcher thread so lane time is pure forward.

## Run

```bash
# deps: jeepney (pip install -e brain-py)
dbus-run-session -- bash -c '
  ./target/release/brain serve --dbus & sleep 2
  python3 examples/embedding/embed_document.py --input README.md --concurrent 4'
```

Nothing to pre-fetch: `--model` defaults to `LiquidAI/LFM2.5-350M`, a
fully-qualified `<vendor>/<repo>` reference — brain's transparent auto-fetch
downloads and converts it on the first request that
names it (that first call is as slow as the cold fetch; every one after is
instant). Point `--model` at `LiquidAI/LFM2.5-230M` for the smaller encoder,
or `brain/mock` for a weight-free smoke test.

Prefer an already-converted local checkpoint instead? Set `BRAIN_LFM2`/
`BRAIN_LFM2_TOKENIZER` before `brain serve --dbus` and pass `--model brain/lfm`
— the env-loaded-checkpoint fallback (the `brain/`
table), unchanged from before auto-fetch existed.

Environment knobs: `BRAIN_LFM2_BATCH` (batched-forward slots per instance,
default 2), `BRAIN_DEVICE` (which compute is schedulable).

Expected output shape (numbers are hardware-dependent):

```
document: doc.txt (11 KiB)
warm-up:
  [warm] 3547 tokens x 1024 dim over memfd (14188 KiB) in …s; mean[0]=-0.7685
4 concurrent request(s):
  [req0] … [req3]   ← equal completion times = they ran in batched groups
wall …s for 4 requests (batching/lanes = wall < sum)
```

## Notes

- The instance key is the **exact token length**: identical-length documents
  share one built graph and batch together; a new length builds (weight upload
  + graph) once and stays resident under the LRU/budget machinery.
- Bidirectional attention makes unmasked token-padding unsound, so requests are
  never padded with pad tokens — batch tails repeat a real sequence instead
  (exact results, some redundant compute).
- The benchmark twin of this example is `make perf/lfm` — the same executor
  measured by the brain perf suite (`brain perf run sweep --target
  lfm:<weights>:<tokenizer> --input 8192 --ladder 1,2,4,8`).

---

## CLIP text embeddings (`brain clip embed_text`)

The same generic surface serves CLIP. Both SDXL text towers are available; the
action returns the projected `text_embeds` when the tower projects and the
pooled EOS row when it does not, so a caller does not have to know which is
which.

```bash
BRAIN_CLIP_DIR=/path/to/stable-diffusion-xl-base-1.0 \
  ./target/release/brain clip embed_text \
    --text "a photo of a cat" --tower clip_l \
    --out embedding=/tmp/cat.f32     # 768 f32 LE (openclip_bigg gives 1280)
```

Verified end to end against HuggingFace on the released checkpoint — string in,
BPE, tower, pooling — at **cosine 1.0000000000 / max_abs 1.5e-5** (CLIP-L) and
**0.9999998212 / 1.0e-5** (OpenCLIP-bigG). `crates/clip/tests/serving.rs` is the
standing gate.

Unlike the face stack, CLIP's `run_batch` is a **genuine batched forward**: the
residency adapter groups a batch by tower and runs one forward per group at
`b = N`, because every row is the same fixed 77-token context. See
`crates/cli/src/resident_clip.rs`.
