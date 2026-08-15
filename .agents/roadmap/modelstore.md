# modelstore - roadmap

The model store's resolution ladder (`crates/modelstore`): plan → download →
convert, driving every `default_ref`/`weights_env` auto-fetch (`crates/cli/
src/supply.rs`). GGUF v2/v3 reader, HF fetch over a host allowlist, and now
`HF_TOKEN`/`hf auth login` support (`crates/modelstore/src/hub.rs::hf_token`).

## Not yet done

- [ ] **Resumable / idempotent fetch.** `plan_base` always returns the FULL
      artifact list for every family, and `execute`'s `Step::Download` always
      calls `hub.download(...)` unconditionally - there is no check for "does
      `dest` already exist with the expected size" before re-fetching. A
      killed-and-restarted fetch (a real scenario for the multi-GB/multi-shard
      checkpoints `default_ref` now reaches - `Tongyi-MAI/Z-Image-Turbo`,
      `Qwen/Qwen3-VL-4B-Instruct`) re-downloads shards that already landed
      correctly on the previous attempt, purely because the repo has not
      finished ALL its files yet (`store.local(reference)` - the one early-out
      `plan()` checks - is only `Some` once every role/file is present, for a
      compound/passthrough family). Confirmed live: a killed `qwen3vl`
      fetch's already-complete 4.97 GB shard 1 was re-downloaded from scratch
      on restart. Not a correctness bug (the re-fetch produces the same
      bytes), but a real time/bandwidth cost on an unauthenticated or
      bandwidth-constrained link, which is exactly when a multi-hour fetch is
      most likely to need restarting in the first place.
      Fix shape: before each `Step::Download`, stat `dest`; skip the download
      if it exists and matches an expected size (`Content-Length` from a HEAD,
      or a size already known from the repo listing) - no HTTP Range/partial-
      resume needed for a first pass, just "don't redo work already done".
- [ ] HF_TOKEN is read fresh per-request (`hf_token()`) but there is still no
      config for a **non-default** token file path or a `HF_HOME` override -
      only `HF_TOKEN` and the fixed `~/.cache/huggingface/token` location the
      `hf`/`huggingface-cli` login flow writes.
