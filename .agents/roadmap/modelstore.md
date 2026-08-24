# modelstore - roadmap

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
