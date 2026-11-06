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
| S2 | The HTTP server sets no header-read, body-read or idle timeout and no connection cap: an unauthenticated peer holds sockets open until the process runs out of descriptors (slowloris). A key-holder holds every edge permit with slow uploads. | high | open |
| S3 | The HTTP API is plaintext: the key and every prompt and output cross the network readable. Decision: native TLS (`--tls-cert`/`--tls-key`), and a non-loopback bind without TLS is refused unless the operator passes `--insecure-plaintext`. | high | open |
| S4 | The training collective authenticates nothing: any host that reaches the coordinator joins as a rank and injects gradients, and gradients and weights cross the network readable. Decision: a shared job secret, a mutual challenge-response at join, then every frame sealed with a per-session AEAD key. | high | open |
| S5 | D-Bus `Run`/`Subscribe` mmap a caller's fd without checking it cannot shrink: truncating it during the copy kills the server with SIGBUS. The number of fds a call carries is unbounded. | medium | open |
| S6 | The serve daemon's fallback state directory (`/tmp/brain-<uid>`) and `serve.log` are created with the default umask, and the log receives the `APIKEY` line: any local user reads the key, or pre-creates the directory. | medium | open |
| S7 | `max_tokens` has no upper bound. | low | open |
| S8 | With the shipped key authenticator every key-holder shares one job scope (lists, follows and cancels every job); D-Bus `Cancel` takes sequential ids. The docs say a job belongs to its caller. | low | open |
| S9 | `GET /v1/videos/{id}/content` streams the file from a blocking thread that waits on a 4-slot channel: a client that stops reading holds that thread for as long as its connection lives. | medium | open |
