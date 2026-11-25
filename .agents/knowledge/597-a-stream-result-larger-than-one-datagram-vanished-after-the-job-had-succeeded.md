<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 597. A stream result larger than one datagram vanished after the job had succeeded

**Lesson:** A transport that sends each frame as one datagram has a size limit per frame, and the terminal frame is
the one that must never be lost. Hand a large result over out of band (an fd) and treat any other refused terminal
send as a closed stream, never as success.

## What happened
`Subscribe` streams send each frame as one `SOCK_SEQPACKET` datagram; brain-py reads 64 KiB per frame. A weight-surgery
job (Sten-TTS rung 1: every kept text id, MLP channel and layer in its manifest) finished, published and installed its
checkpoint, and then its `done` frame of several hundred KiB failed in `sendmsg` with `EMSGSIZE`. `send_terminal`
retried only on `EAGAIN` and returned on every other error as if it had sent, so the client saw the stream end without
a terminal frame (`brain_py.stream_disconnected`) after a 20-minute job, and the job's report was lost.

## What to do
`StreamTx::done` sends a result over 48 KiB as `{"type": "done", "result_fd": true}` with its JSON in a sealed memfd;
brain-py reads it (`_done_result`). An error message is cut to fit. Any other terminal send the socket refuses closes
the stream with a log line. A job that publishes before it reports should keep its report recoverable from the
artifact store as well.
