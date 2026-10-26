# modelstore - roadmap

Scope: Model store resolution ladder: plan, download, convert (`crates/modelstore`).

The model store's resolution ladder (`crates/modelstore`): plan → download →
convert, driving every `default_ref`/`weights_env` auto-fetch (`crates/cli/
src/supply.rs`). GGUF v2/v3 reader, HF fetch over a host allowlist, and now
`HF_TOKEN`/`hf auth login` support (`crates/modelstore/src/hub.rs::hf_token`).

## Not yet done

- [x] **Resumable / idempotent fetch.** `execute` now skips a
      `Step::Download` whose `dest` already exists, so a killed multi-shard
      fetch resumes instead of re-downloading what already landed.

      The fix is simpler than the shape originally proposed here (stat `dest`
      and compare against a `Content-Length` from a HEAD): bare existence is
      the *correct* test, because `fetch::stream_to_file` writes to a `.part`
      sibling and renames into place only on full success. A file at `dest`
      is therefore by construction a completed download, and a partial
      transfer leaves a `.part` that the check never mistakes for the real
      one. No new `Hub` method, no extra network round-trip, and no change to
      the redirect/host-allowlist code. That atomic-rename invariant is now
      load-bearing for the skip and says so in `execute`'s doc comment: if
      `stream_to_file` ever writes `dest` incrementally, the check must become
      a size/digest comparison in the same change.

      Gated by two tests in `plan.rs`: one asserts an already-present file is
      not re-fetched (behaviourally - the local copy is overwritten with a
      sentinel between two `execute` calls) and that a skipped download
      reports no transfer progress; the other asserts the skip is per-FILE,
      so a repo missing one of its files still fetches that one. The second
      test exists because an all-or-nothing early-out would satisfy the first
      one while never completing a partially-fetched repo.
- [ ] HF_TOKEN is read fresh per-request (`hf_token()`) but there is still no
      config for a **non-default** token file path or a `HF_HOME` override -
      only `HF_TOKEN` and the fixed `~/.cache/huggingface/token` location the
      `hf`/`huggingface-cli` login flow writes.

## Selection by the store alone

`brain serve` serves the OpenAI API with no flags, publishes a connection
record (address and key) for local clients, and selects Qwen3 by the store
or `--checkpoint`/`--tokenizer`; `BRAIN_QWEN_WEIGHTS`, `BRAIN_QWEN_TOKENIZER`,
`BRAIN_API_KEY` and `BRAIN_RUNTIME_DIR` are no longer read. A family with
nothing on the machine is one counted startup line. Still selected by a
variable instead of the store, each to move onto a resolver spec and the
`model_dir` scan, its variable deleted outright:

- [ ] TimesFM-3 serving (`BRAIN_TIMESFM3`; `brain pull` fetches it but serving
      cannot find it).
- [ ] Kronos serving (`BRAIN_KRONOS_TOKENIZER` + `_DECODER`; `brain forecast
      predict` already resolves both from the store).
- [x] DeepSeek-OCR-2: pulled and served from the vendor repo through the resolver (no variable).
- [ ] Qwen3-VL serving (`BRAIN_QWEN3VL_WEIGHTS`; `brain caps`/`brain do`
      already resolve it from the store).
- [ ] The scheduler families the startup line counts with a variable as the
      remedy: Z-Image (`BRAIN_S3DIT_*`), FLUX.2 overrides, LFM2, Wan, LTX-Video,
      YOLO, the Qwen3-Omni int8 checkpoint.
- [ ] `BRAIN_MODELS_DIR` and `BRAIN_AUTO_FETCH`: a flag exists for each
      (`--models-dir`/`--brain-data-dir`, `--autofetch`), but `main` still
      hands them on through the environment; pass the value down instead.
- [ ] The config file's `BRAIN_*` keys shrink with the allowlist: every
      variable that becomes a flag moves to its subcommand section.
