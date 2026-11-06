<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# service-security - roadmap

Scope: Every socket brain listens on, hardened for a network where other hosts are hostile (HTTP, D-Bus, the event-protocol socket, the training collective, the serve daemon's state).

Threat model: brain is bound on a private network. Any other host on it is
an attacker without a key; a key-holder may also be hostile. A private network
is not a trust boundary: nothing may rely on "only friendly hosts can reach it".

Audit of 2026-11-06 (source-verified). Each finding below is a milestone; a
milestone is done when its test is red before the fix and green after.

## Milestones

| # | Finding | Severity | Status |
|---|---|---|---|
| S1 | `brain forecast serve --listen` is an unauthenticated plaintext TCP surface, and a `camera_frame` `path` makes the server read any file of its machine (`/dev/zero` exhausts memory, a FIFO hangs the thread, the error text is a file-existence oracle). The event socket also reads unbounded lines. | high | done: the TCP listener is removed (its one consumer uses the Unix socket; the network surface is the key-gated `forecast` action of `brain serve`), the socket is `0600`, a line over 64 MiB is refused, and a frame is decoded only when a detector is loaded (`crates/server/tests/it/socket_hardening.rs`, `crates/runtime/tests/it/controller.rs`) |
| S2 | The HTTP server sets no header-read, body-read or idle timeout and no connection cap: an unauthenticated peer holds sockets open until the process runs out of descriptors (slowloris). A key-holder holds every edge permit with slow uploads. | high | done: `crates/apiserve/src/listener.rs` replaces `axum::serve` (10 s header and idle deadline, 5 min whole-body deadline, 1024 connections per surface, HTTP/1.1 only; `crates/apiserve/tests/it/connections.rs`). A slow upload now holds its edge permit at most the body deadline; a key-holder can still fill the edge with long legitimate requests, which is S8's missing per-caller quota |
| S3 | The HTTP API is plaintext: the key and every prompt and output cross the network readable. Decision: native TLS (`--tls-cert`/`--tls-key`), and a non-loopback bind without TLS is refused unless the operator passes `--insecure-plaintext`. | high | open |
| S4 | The training collective authenticates nothing: any host that reaches the coordinator joins as a rank and injects gradients, and gradients and weights cross the network readable. Decision: a shared job secret, a mutual challenge-response at join, then every frame sealed with a per-session AEAD key. | high | done: `model::sealedwire` (mutual HMAC proof over fresh nonces, per-direction ChaCha20-Poly1305 keys from HKDF, sequence nonces), `--job-secret FILE` beside `--job-plan`/`--job-rank`, no unauthenticated constructor left; a stranger is dropped without disturbing the forming group. Tests and the measured cost: `distributed-execution.md` |
| S5 | D-Bus `Run`/`Subscribe` mmap a caller's fd without checking it cannot shrink: truncating it during the copy kills the server with SIGBUS. The number of fds a call carries is unbounded. | medium | open |
| S6 | The serve daemon's fallback state directory (`/tmp/brain-<uid>`) and `serve.log` are created with the default umask, and the log receives the `APIKEY` line: any local user reads the key, or pre-creates the directory. | medium | open |
| S7 | `max_tokens` has no upper bound of its own. Re-checked: the serving model refuses a prompt plus `max_tokens` over its context before admission, so the bound is the context; what is missing is a per-caller budget (S13). | low | merged into S13 |
| S8 | With the shipped key authenticator every key-holder shares one job scope (lists, follows and cancels every job); D-Bus `Cancel` takes sequential ids. The docs say a job belongs to its caller. | high | open: fixed by S13's identities |
| S9 | `GET /v1/videos/{id}/content` streams the file from a blocking thread that waits on a 4-slot channel: a client that stops reading holds that thread for as long as its connection lives. | medium | open |
| S10 | Media is decoded on the async runtime's worker threads, before admission, with no bound on how many decode at once: images (`media.rs` `decode_image`), chat `input_audio` and `/audio/transcriptions` (`decode_speech`). The request-wide pixel budget is checked after each image is decoded, so one 8 MiB request of flat PNGs decodes about 100 Mpx (about 1.2 GB of f32) and a handful of them stall every surface of the process. | high | open |
| S11 | The shipped system-bus policy lets only root own `com.swedishembedded.Brain1`, so `--dbus-system` makes the whole server (media decoders, model parsers, GPU drivers) run as root. | high | open |
| S12 | The video spool is `<tmp>/brain-videos-<pid>`, created with `DirBuilder::recursive`, which accepts a directory another local user created first: that user then reads every clip and can replace or redirect them. | medium | open |
| S13 | One key per dialect is the whole security identity: every key-holder may run every served action through `/v1/run`/`/v1/jobs` (training only behind `--training`, fetch only behind `--auto-fetch`), shares one job scope (S8), and has no budget of its own, so one compromised client can occupy every lane. Fix: named callers, each with its own key file, a role (`inference`, `training`, `admin`) checked at the one dispatch boundary every surface shares, its own job scope, and its own concurrent-request budget. | high | open |
