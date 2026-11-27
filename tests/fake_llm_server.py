"""A local HTTP server that records what model adapters actually send.

It answers OpenAI `/chat/completions` and Anthropic `/v1/messages` with a fixed
one-word reply, so the real pydantic-ai adapters (and the real provider SDKs)
can be driven offline and their request bodies asserted on.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
ANTHROPIC_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}



def _sse(events: list[tuple[str | None, object]]) -> bytes:
    out = []
    for name, data in events:
        if name:
            out.append(f"event: {name}")
        out.append("data: " + (data if isinstance(data, str) else json.dumps(data)))
        out.append("")
    return ("\n".join(out) + "\n").encode()


def openai_stream(text: str) -> bytes:
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    return _sse(
        [
            (None, {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}),
            (None, {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
            (None, "[DONE]"),
        ]
    )


def anthropic_stream(text: str) -> bytes:
    message = {**ANTHROPIC_REPLY, "content": [], "stop_reason": None,
               "usage": {"input_tokens": 1, "output_tokens": 0}}
    return _sse(
        [
            ("message_start", {"type": "message_start", "message": message}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 1}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


class FakeLLM:
    def __init__(self, status: int = 200):
        self.requests: list[dict] = []
        self.status = status
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length", 0)))
                recorder.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": json.loads(body or b"{}")}
                )
                parsed = recorder.requests[-1]["body"]
                anthropic = self.path.endswith("/messages")
                content_type = "application/json"
                if recorder.status != 200:
                    data = json.dumps({"error": {"message": "boom", "type": "server_error"}}).encode()
                elif parsed.get("stream"):
                    data = anthropic_stream("ok") if anthropic else openai_stream("ok")
                    content_type = "text/event-stream"
                else:
                    data = json.dumps(ANTHROPIC_REPLY if anthropic else OPENAI_REPLY).encode()
                self.send_response(recorder.status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()

    @property
    def last(self) -> dict:
        return self.requests[-1]
