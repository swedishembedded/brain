# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for driving reasoning models through standard APIs for its clients. If your
# team needs expertise in client libraries for self-hosted LLM servers then you can procure our services by sending an
# email to info@swedishembedded.com.

"""The HTTP transports carry a request's reasoning settings to the server and bring the reasoning back.

A local stand-in server records the body each client posts and answers in the dialect's shape, reasoning included:
`reasoning_effort` must reach the wire (OpenAI: top level; Anthropic: `output_config.effort`) and the reply's
reasoning must come back as `reasoning_content`, never dropped silently.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from brain_py.anthropic import BrainAnthropic
from brain_py.openai import BrainOpenAI


class _Recorder(BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):  # noqa: N802 - the stdlib's handler name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        if self.path.endswith("/messages"):
            reply = {"content": [{"type": "thinking", "thinking": "think"}, {"type": "text", "text": "answer"}], "usage": {}}
        else:
            reply = {"choices": [{"message": {"content": "answer", "reasoning_content": "think"}}], "usage": {}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


def _serve():
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"127.0.0.1:{server.server_address[1]}"


def test_openai_carries_reasoning_effort_and_returns_reasoning():
    server, url = _serve()
    try:
        _Recorder.bodies.clear()
        out = BrainOpenAI(url, api_key="k").run("m", "generate", {"prompt": "q", "reasoning_effort": "low", "enable_thinking": True})
        assert _Recorder.bodies[-1]["reasoning_effort"] == "low"
        assert _Recorder.bodies[-1]["enable_thinking"] is True
        assert out.outputs["reasoning_content"] == "think" and out.text() == "answer"
    finally:
        server.shutdown()


def test_anthropic_carries_reasoning_effort_and_returns_thinking():
    server, url = _serve()
    try:
        _Recorder.bodies.clear()
        out = BrainAnthropic(url, api_key="k").run("m", "generate", {"prompt": "q", "reasoning_effort": "low"})
        assert _Recorder.bodies[-1]["output_config"] == {"effort": "low"}
        assert out.outputs["reasoning_content"] == "think" and out.text() == "answer"
    finally:
        server.shutdown()
